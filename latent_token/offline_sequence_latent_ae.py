"""Continuous latent-token AE aligned with generate/ agent-centric dumps.

Per-timestep token order (OFFLINE_SEQUENCE + generate fields):

  [ obs_static, obs_dyn_1..K, action, reward_team, (reward_ind), done ]

- ``reward_team`` is required; ``reward_individual`` is optional (``use_individual_reward``).
- ``done`` token encodes termination: done / truncation(timeout) / is_win /
  combat_end (done & ~truncation).
- Visible dyn objects are packed (length K=max_visible); dump may still store M slots.
- No Transformer backbone: encode tokens → ResidualMLPDecoder heads.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

# Token type ids
TYPE_STATIC = 0
TYPE_DYN = 1
TYPE_ACT = 2
TYPE_REW_TEAM = 3
TYPE_REW_IND = 4
TYPE_DONE = 5
NUM_TOKEN_TYPES = 6

# done.npy / truncation.npy / is_win.npy → 4-d done feature
# [done, truncation, is_win, combat_end] where combat_end = done * (1 - truncation)
DONE_FEAT_DIM = 4


def pack_visible_dyn(
    dyn: jax.Array,
    mask: jax.Array,
    max_visible: int,
) -> tuple[jax.Array, jax.Array]:
    """Compact visible dyn objects to the front; keep at most ``max_visible``."""
    if max_visible < 1:
        raise ValueError("max_visible must be >= 1")
    order = jnp.argsort(-mask, axis=-1)
    idx = order[:, :max_visible]
    dyn_p = jnp.take_along_axis(dyn, idx[..., None], axis=1)
    mask_p = jnp.take_along_axis(mask, idx, axis=1)
    dyn_p = dyn_p * mask_p[..., None]
    return dyn_p, mask_p


def build_done_features(
    done: np.ndarray | jax.Array,
    truncation: np.ndarray | jax.Array,
    is_win: np.ndarray | jax.Array,
) -> np.ndarray | jax.Array:
    """Stack termination flags used by the done token."""
    done = jnp.asarray(done, dtype=jnp.float32).reshape(-1)
    truncation = jnp.asarray(truncation, dtype=jnp.float32).reshape(-1)
    is_win = jnp.asarray(is_win, dtype=jnp.float32).reshape(-1)
    combat_end = done * (1.0 - truncation)
    return jnp.stack([done, truncation, is_win, combat_end], axis=-1)


def build_done_features_np(
    done: np.ndarray, truncation: np.ndarray, is_win: np.ndarray
) -> np.ndarray:
    done = np.asarray(done, dtype=np.float32).reshape(-1)
    truncation = np.asarray(truncation, dtype=np.float32).reshape(-1)
    is_win = np.asarray(is_win, dtype=np.float32).reshape(-1)
    combat_end = done * (1.0 - truncation)
    return np.stack([done, truncation, is_win, combat_end], axis=-1).astype(np.float32)


class MLPEncoderContinuous(nn.Module):
    """Continuous encoder: Linear / (Linear→GELU→Dropout)* → Linear (no LN)."""

    hidden_sizes: Sequence[int]
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, *, train: bool = True):
        sizes = list(self.hidden_sizes)
        if not sizes:
            raise ValueError("hidden_sizes must be non-empty")
        for h in sizes[:-1]:
            x = nn.Dense(h)(x)
            x = nn.gelu(x)
            x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
        return nn.Dense(sizes[-1])(x)


class ResidualMLPDecoder(nn.Module):
    """LN(input) → MLP → optional residual → out."""

    out_dim: int
    hidden_sizes: Sequence[int]
    dropout: float = 0.0
    residual_connect: bool = False

    @nn.compact
    def __call__(self, z, *, train: bool = True):
        src = nn.LayerNorm()(z)
        sizes = list(self.hidden_sizes)
        if self.residual_connect and len(sizes) < 2:
            raise ValueError("residual_connect requires at least two hidden sizes")

        if self.residual_connect:
            h = src
            for h_dim in sizes[:-1]:
                h = nn.Dense(h_dim)(h)
                h = nn.gelu(h)
                h = nn.Dropout(rate=self.dropout, deterministic=not train)(h)
            h = nn.Dense(src.shape[-1])(h)
            h = h + src
            return nn.Dense(self.out_dim)(h)

        h = src
        for h_dim in sizes[:-1]:
            h = nn.Dense(h_dim)(h)
            h = nn.gelu(h)
            h = nn.Dropout(rate=self.dropout, deterministic=not train)(h)
        return nn.Dense(self.out_dim)(h)


class OfflineSequenceLatentAE(nn.Module):
    """Agent-centric step tokens: obs → action → reward_team → [reward_ind] → done."""

    static_dim: int
    dyn_dim: int
    max_visible: int
    action_dim: int = 8
    latent_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.0
    residual_decode: bool = False
    use_individual_reward: bool = False

    @nn.compact
    def __call__(
        self,
        static,
        dyn,
        mask,
        action,
        reward_team,
        done_feat,
        reward_individual=None,
        noise_std: float = 0.0,
        *,
        train: bool = True,
    ):
        """
        static: [B, S]
        dyn:    [B, M, F]
        mask:   [B, M]
        action: [B] int
        reward_team: [B] float (required)
        done_feat: [B, 4] = [done, truncation, is_win, combat_end]
        reward_individual: [B] float if use_individual_reward else ignored
        """
        bsz = static.shape[0]
        enc_sizes = (self.hidden_dim, self.latent_dim)
        dec_sizes = (self.hidden_dim, self.hidden_dim)
        k = self.max_visible

        dyn_p, mask_p = pack_visible_dyn(dyn, mask, self.max_visible)

        # --- obs group ---
        z_static = MLPEncoderContinuous(enc_sizes, self.dropout)(static, train=train)
        z_dyn = MLPEncoderContinuous(enc_sizes, self.dropout)(dyn_p, train=train)
        z_dyn = z_dyn * mask_p[..., None]

        # --- action / rewards / done ---
        z_act = nn.Embed(self.action_dim, self.latent_dim)(action)
        z_rew_team = MLPEncoderContinuous(enc_sizes, self.dropout)(
            reward_team[:, None], train=train
        )
        z_done = MLPEncoderContinuous(enc_sizes, self.dropout)(done_feat, train=train)

        z_rew_ind = None
        if self.use_individual_reward:
            if reward_individual is None:
                raise ValueError("use_individual_reward=True requires reward_individual")
            z_rew_ind = MLPEncoderContinuous(enc_sizes, self.dropout)(
                reward_individual[:, None], train=train
            )

        if noise_std > 0:
            ks = jax.random.split(self.make_rng("noise"), 6)
            z_static = z_static + noise_std * jax.random.normal(ks[0], z_static.shape)
            z_dyn = z_dyn + noise_std * jax.random.normal(ks[1], z_dyn.shape)
            z_act = z_act + noise_std * jax.random.normal(ks[2], z_act.shape)
            z_rew_team = z_rew_team + noise_std * jax.random.normal(ks[3], z_rew_team.shape)
            z_done = z_done + noise_std * jax.random.normal(ks[4], z_done.shape)
            z_dyn = z_dyn * mask_p[..., None]
            if z_rew_ind is not None:
                z_rew_ind = z_rew_ind + noise_std * jax.random.normal(
                    ks[5], z_rew_ind.shape
                )

        type_embed = nn.Embed(NUM_TOKEN_TYPES, self.latent_dim)
        z_static = z_static + type_embed(jnp.full((bsz,), TYPE_STATIC, jnp.int32))
        dyn_type = type_embed(jnp.full((bsz, k), TYPE_DYN, jnp.int32))
        z_dyn = z_dyn + dyn_type * mask_p[..., None]
        z_act = z_act + type_embed(jnp.full((bsz,), TYPE_ACT, jnp.int32))
        z_rew_team = z_rew_team + type_embed(
            jnp.full((bsz,), TYPE_REW_TEAM, jnp.int32)
        )
        z_done = z_done + type_embed(jnp.full((bsz,), TYPE_DONE, jnp.int32))
        if z_rew_ind is not None:
            z_rew_ind = z_rew_ind + type_embed(
                jnp.full((bsz,), TYPE_REW_IND, jnp.int32)
            )

        # Token order: obs group, action, reward_team, [reward_ind], done
        parts = [z_static[:, None, :], z_dyn, z_act[:, None, :], z_rew_team[:, None, :]]
        mask_parts = [
            jnp.ones((bsz, 1), dtype=mask_p.dtype),
            mask_p,
            jnp.ones((bsz, 1), dtype=mask_p.dtype),
            jnp.ones((bsz, 1), dtype=mask_p.dtype),
        ]
        if z_rew_ind is not None:
            parts.append(z_rew_ind[:, None, :])
            mask_parts.append(jnp.ones((bsz, 1), dtype=mask_p.dtype))
        parts.append(z_done[:, None, :])
        mask_parts.append(jnp.ones((bsz, 1), dtype=mask_p.dtype))

        tokens = jnp.concatenate(parts, axis=1)
        tok_mask = jnp.concatenate(mask_parts, axis=1)

        # Action CE from obs tokens only (no action / reward / done leak).
        obs_mask = jnp.concatenate(
            [jnp.ones((bsz, 1), dtype=mask_p.dtype), mask_p], axis=1
        )
        z_obs = tokens[:, : 1 + k, :]
        obs_denom = jnp.maximum(jnp.sum(obs_mask, axis=1, keepdims=True), 1.0)
        z_obs_ctx = jnp.sum(z_obs * obs_mask[..., None], axis=1) / obs_denom

        pred_static = ResidualMLPDecoder(
            self.static_dim, dec_sizes, self.dropout, self.residual_decode
        )(z_static, train=train)
        pred_dyn = ResidualMLPDecoder(
            self.dyn_dim, dec_sizes, self.dropout, self.residual_decode
        )(z_dyn, train=train)
        action_logits = ResidualMLPDecoder(
            self.action_dim, dec_sizes, self.dropout, self.residual_decode
        )(z_obs_ctx, train=train)
        pred_reward_team = ResidualMLPDecoder(
            1, dec_sizes, self.dropout, False
        )(z_rew_team, train=train).squeeze(-1)
        pred_done = ResidualMLPDecoder(
            DONE_FEAT_DIM, dec_sizes, self.dropout, False
        )(z_done, train=train)

        pred_reward_ind = None
        if z_rew_ind is not None:
            pred_reward_ind = ResidualMLPDecoder(
                1, dec_sizes, self.dropout, False
            )(z_rew_ind, train=train).squeeze(-1)

        return {
            "pred_static": pred_static,
            "pred_dyn": pred_dyn,
            "dyn_target": dyn_p,
            "dyn_mask": mask_p,
            "action_logits": action_logits,
            "pred_reward_team": pred_reward_team,
            "pred_reward_individual": pred_reward_ind,
            "pred_done": pred_done,
            "tokens": tokens,
            "token_mask": tok_mask,
            "max_visible": self.max_visible,
            "use_individual_reward": self.use_individual_reward,
        }


class TrainState(train_state.TrainState):
    pass


def load_agent_data(records_root: Path, *, use_individual_reward: bool = False):
    """Load generate/ agent-centric arrays. ``reward_team`` is required."""

    agent_dirs = sorted(records_root.glob("**/agent_centric/unit_*"))
    if not agent_dirs:
        agent_dirs = sorted(records_root.glob("**/agent_centric/ally_*"))
    if not agent_dirs:
        raise FileNotFoundError(f"No agent-centric dirs found under {records_root}")

    buckets: dict[str, list] = {
        "static": [],
        "dyn": [],
        "mask": [],
        "action": [],
        "reward_team": [],
        "reward_individual": [],
        "done": [],
        "truncation": [],
        "is_win": [],
    }

    for ad in agent_dirs:
        s = np.load(ad / "obs_static.npy").astype(np.float32)
        d = np.load(ad / "obs_dynamic.npy").astype(np.float32)
        m = np.load(ad / "obs_dynamic_mask.npy").astype(np.float32)
        a = np.load(ad / "behavior_action.npy").astype(np.int32)

        team_path = ad / "reward_team.npy"
        if not team_path.exists():
            raise FileNotFoundError(f"required reward_team.npy missing under {ad}")
        r_team = np.load(team_path).astype(np.float32)

        for name in ("done.npy", "truncation.npy", "is_win.npy"):
            if not (ad / name).exists():
                raise FileNotFoundError(f"required {name} missing under {ad}")
        done = np.load(ad / "done.npy").astype(np.float32)
        trunc = np.load(ad / "truncation.npy").astype(np.float32)
        is_win = np.load(ad / "is_win.npy").astype(np.float32)

        if use_individual_reward:
            ind_path = ad / "reward_individual.npy"
            if not ind_path.exists():
                raise FileNotFoundError(
                    f"--use_individual_reward set but missing {ind_path}"
                )
            r_ind = np.load(ind_path).astype(np.float32)
        else:
            r_ind = np.zeros_like(r_team, dtype=np.float32)

        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d = d * m[..., None]
        a = np.clip(a, 0, 7)

        buckets["static"].append(s)
        buckets["dyn"].append(d)
        buckets["mask"].append(m)
        buckets["action"].append(a)
        buckets["reward_team"].append(r_team)
        buckets["reward_individual"].append(r_ind)
        buckets["done"].append(done)
        buckets["truncation"].append(trunc)
        buckets["is_win"].append(is_win)

    out = {k: np.concatenate(v, 0) for k, v in buckets.items()}
    out["done_feat"] = build_done_features_np(
        out["done"], out["truncation"], out["is_win"]
    )
    return out


def normalize_arrays(data: dict, train_idx, out_stats: Path | None = None):
    static = data["static"]
    dyn = data["dyn"]
    mask = data["mask"]
    r_team = data["reward_team"]
    r_ind = data["reward_individual"]

    s_train = static[train_idx]
    d_train = dyn[train_idx]
    m_train = mask[train_idx]
    rt_train = r_team[train_idx]
    ri_train = r_ind[train_idx]

    s_mean = s_train.mean(axis=0, keepdims=True)
    s_std = np.maximum(s_train.std(axis=0, keepdims=True), 0.05)
    valid_dyn = d_train[m_train > 0.5]
    if len(valid_dyn) == 0:
        valid_dyn = d_train.reshape((-1, d_train.shape[-1]))
    d_mean = valid_dyn.mean(axis=0, keepdims=True)
    d_std = np.maximum(valid_dyn.std(axis=0, keepdims=True), 0.05)
    rt_mean = np.array([rt_train.mean()], dtype=np.float32)
    rt_std = np.array([max(float(rt_train.std()), 0.05)], dtype=np.float32)
    ri_mean = np.array([ri_train.mean()], dtype=np.float32)
    ri_std = np.array([max(float(ri_train.std()), 0.05)], dtype=np.float32)

    static_n = ((static - s_mean) / s_std).astype(np.float32)
    dyn_n = (dyn - d_mean.reshape((1, 1, -1))) / d_std.reshape((1, 1, -1))
    dyn_n = (dyn_n * mask[..., None]).astype(np.float32)
    r_team_n = ((r_team - rt_mean[0]) / rt_std[0]).astype(np.float32)
    r_ind_n = ((r_ind - ri_mean[0]) / ri_std[0]).astype(np.float32)

    if out_stats is not None:
        out_stats.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_stats,
            static_mean=s_mean,
            static_std=s_std,
            dyn_mean=d_mean,
            dyn_std=d_std,
            reward_team_mean=rt_mean,
            reward_team_std=rt_std,
            reward_individual_mean=ri_mean,
            reward_individual_std=ri_std,
        )

    return {
        "static": static_n,
        "dyn": dyn_n,
        "mask": mask.astype(np.float32),
        "action": data["action"].astype(np.int32),
        "reward_team": r_team_n,
        "reward_individual": r_ind_n,
        "done_feat": data["done_feat"].astype(np.float32),
    }


def _metrics_from_out(out, batch, *, reward_team_coef: float, reward_ind_coef: float):
    pred_s = out["pred_static"]
    pred_d = out["pred_dyn"]
    dyn_t = out["dyn_target"]
    dyn_m = out["dyn_mask"]
    logits = out["action_logits"]

    static_mse = jnp.mean((pred_s - batch["static"]) ** 2)
    dyn_sq = ((pred_d - dyn_t) ** 2) * dyn_m[..., None]
    dyn_denom = jnp.maximum(jnp.sum(dyn_m) * dyn_t.shape[-1], 1.0)
    dyn_mse = jnp.sum(dyn_sq) / dyn_denom
    action_ce = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["action"]
    ).mean()
    action_acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch["action"])
    reward_team_mse = jnp.mean(
        (out["pred_reward_team"] - batch["reward_team"]) ** 2
    )
    done_mse = jnp.mean((out["pred_done"] - batch["done_feat"]) ** 2)

    loss = (
        static_mse
        + dyn_mse
        + action_ce
        + reward_team_coef * reward_team_mse
        + done_mse
    )
    metrics = {
        "loss": loss,
        "static_mse": static_mse,
        "dynamic_masked_mse": dyn_mse,
        "action_ce": action_ce,
        "action_acc": action_acc,
        "reward_team_mse": reward_team_mse,
        "done_mse": done_mse,
        "reward_individual_mse": jnp.asarray(0.0),
    }
    if out["pred_reward_individual"] is not None:
        ri_mse = jnp.mean(
            (out["pred_reward_individual"] - batch["reward_individual"]) ** 2
        )
        loss = loss + reward_ind_coef * ri_mse
        metrics["loss"] = loss
        metrics["reward_individual_mse"] = ri_mse
    return loss, metrics


def _apply_model(params, model, batch, key, noise_std: float, *, train: bool):
    rngs = {"noise": key}
    if train:
        rngs["dropout"] = key
    kwargs = dict(
        static=batch["static"],
        dyn=batch["dyn"],
        mask=batch["mask"],
        action=batch["action"],
        reward_team=batch["reward_team"],
        done_feat=batch["done_feat"],
        noise_std=noise_std,
        train=train,
    )
    if model.use_individual_reward:
        kwargs["reward_individual"] = batch["reward_individual"]
    return model.apply({"params": params}, **kwargs, rngs=rngs)


def loss_metrics(
    params,
    model,
    batch,
    key,
    noise_std: float,
    reward_team_coef: float = 0.2,
    reward_ind_coef: float = 0.2,
):
    out = _apply_model(params, model, batch, key, noise_std, train=True)
    return _metrics_from_out(
        out, batch, reward_team_coef=reward_team_coef, reward_ind_coef=reward_ind_coef
    )


def eval_metrics(
    params,
    model,
    batch,
    key,
    noise_std: float,
    reward_team_coef: float = 0.2,
    reward_ind_coef: float = 0.2,
):
    out = _apply_model(params, model, batch, key, noise_std, train=False)
    _, metrics = _metrics_from_out(
        out, batch, reward_team_coef=reward_team_coef, reward_ind_coef=reward_ind_coef
    )
    return metrics


def make_train_step(reward_team_coef: float, reward_ind_coef: float):
    @jax.jit
    def train_step(state, batch, key):
        def fn(params):
            return loss_metrics(
                params,
                state.apply_fn,
                batch,
                key,
                0.0,
                reward_team_coef=reward_team_coef,
                reward_ind_coef=reward_ind_coef,
            )

        (loss, metrics), grads = jax.value_and_grad(fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return train_step


def _batch_from_idx(arrays, idx):
    batch = {
        "static": jnp.asarray(arrays["static"][idx]),
        "dyn": jnp.asarray(arrays["dyn"][idx]),
        "mask": jnp.asarray(arrays["mask"][idx]),
        "action": jnp.asarray(arrays["action"][idx]),
        "reward_team": jnp.asarray(arrays["reward_team"][idx]),
        "done_feat": jnp.asarray(arrays["done_feat"][idx]),
    }
    if "reward_individual" in arrays:
        batch["reward_individual"] = jnp.asarray(arrays["reward_individual"][idx])
    return batch


def batch_iter(arrays, idx, batch_size, rng):
    perm = rng.permutation(idx)
    for st in range(0, len(perm), batch_size):
        yield _batch_from_idx(arrays, perm[st : st + batch_size])


def evaluate(
    state,
    arrays,
    idx,
    batch_size,
    noise_stds,
    reward_team_coef: float = 0.2,
    reward_ind_coef: float = 0.2,
):
    metric_keys = [
        "loss",
        "static_mse",
        "dynamic_masked_mse",
        "action_ce",
        "action_acc",
        "reward_team_mse",
        "reward_individual_mse",
        "done_mse",
    ]
    rows = []
    for noise_std in noise_stds:
        sums = {k: 0.0 for k in metric_keys}
        count = 0
        for st in range(0, len(idx), batch_size):
            b = idx[st : st + batch_size]
            batch = _batch_from_idx(arrays, b)
            key = jax.random.PRNGKey(1000 + st + int(noise_std * 10000))
            metrics = eval_metrics(
                state.params,
                state.apply_fn,
                batch,
                key,
                float(noise_std),
                reward_team_coef=reward_team_coef,
                reward_ind_coef=reward_ind_coef,
            )
            bs = len(b)
            for k in sums:
                sums[k] += float(metrics[k]) * bs
            count += bs
        row = {k: v / max(count, 1) for k, v in sums.items()}
        row["noise_std"] = noise_std
        rows.append(row)
    return rows


def infer_max_visible(mask: np.ndarray, override: int | None = None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError("--max_visible must be >= 1")
        return int(override)
    counted = int(mask.sum(axis=1).max()) if len(mask) else 1
    return max(counted, 1)


def train_doc_model(arrays, train_idx, test_idx, args):
    max_visible = infer_max_visible(arrays["mask"], getattr(args, "max_visible", None))
    use_ind = bool(getattr(args, "use_individual_reward", False))
    extra = 1 if use_ind else 0
    token_len = 1 + max_visible + 1 + 1 + extra + 1  # obsS + K dyn + act + r_team + [rind] + done
    print(
        f"packing visibles: dump_M={arrays['dyn'].shape[1]}, K={max_visible}, "
        f"token_order=obs(1+K)+action+reward_team"
        f"{'+reward_ind' if use_ind else ''}+done, token_len={token_len}"
    )
    model = OfflineSequenceLatentAE(
        static_dim=arrays["static"].shape[-1],
        dyn_dim=arrays["dyn"].shape[-1],
        max_visible=max_visible,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual_decode=args.residual_decode,
        use_individual_reward=use_ind,
    )
    key = jax.random.PRNGKey(args.seed)
    init_batch = _batch_from_idx(arrays, train_idx[: min(4, len(train_idx))])
    init_kwargs = dict(
        static=init_batch["static"],
        dyn=init_batch["dyn"],
        mask=init_batch["mask"],
        action=init_batch["action"],
        reward_team=init_batch["reward_team"],
        done_feat=init_batch["done_feat"],
        noise_std=0.0,
        train=True,
    )
    if use_ind:
        init_kwargs["reward_individual"] = init_batch["reward_individual"]
    variables = model.init(
        {"params": key, "dropout": key, "noise": key},
        **init_kwargs,
    )
    state = TrainState.create(
        apply_fn=model, params=variables["params"], tx=optax.adam(args.lr)
    )
    step_fn = make_train_step(args.reward_team_coef, args.reward_ind_coef)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed + 1)

    for epoch in range(1, args.epochs + 1):
        for batch in batch_iter(arrays, train_idx, args.batch_size, rng):
            key, sub = jax.random.split(key)
            state, metrics = step_fn(state, batch, sub)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[doc_offline_sequence] epoch={epoch:04d} "
                f"loss={float(metrics['loss']):.6f} "
                f"static={float(metrics['static_mse']):.6f} "
                f"dyn={float(metrics['dynamic_masked_mse']):.6f} "
                f"act_acc={float(metrics['action_acc']):.4f} "
                f"r_team={float(metrics['reward_team_mse']):.6f} "
                f"done={float(metrics['done_mse']):.6f}",
                flush=True,
            )

    noise_stds = [float(x) for x in args.noise_stds.split(",")]
    rows = evaluate(
        state,
        arrays,
        test_idx,
        args.batch_size,
        noise_stds,
        args.reward_team_coef,
        args.reward_ind_coef,
    )
    for r in rows:
        r["model"] = "doc_offline_sequence"
    return state, rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train agent-centric latent AE (obs→action→reward_team→[rind]→done)"
    )
    ap.add_argument("--records_root", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--residual_decode", action="store_true")
    ap.add_argument("--max_visible", type=int, default=None)
    ap.add_argument(
        "--use_individual_reward",
        action="store_true",
        help="Include optional reward_individual token (requires reward_individual.npy)",
    )
    ap.add_argument("--reward_team_coef", type=float, default=0.2)
    ap.add_argument("--reward_ind_coef", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--noise_stds", default="0,0.01,0.05,0.1")
    ap.add_argument("--out_csv", default="outputs/doc_offline_sequence_reconstruct.csv")
    ap.add_argument("--stats_out", default="outputs/doc_offline_sequence_norm_stats.npz")
    args = ap.parse_args(argv)

    raw = load_agent_data(
        Path(args.records_root), use_individual_reward=args.use_individual_reward
    )
    print(
        "loaded",
        raw["static"].shape,
        raw["dyn"].shape,
        "reward_team",
        raw["reward_team"].shape,
        "done_feat",
        raw["done_feat"].shape,
        "use_individual_reward",
        args.use_individual_reward,
    )

    n = len(raw["action"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    split = max(1, int(0.8 * n))
    train_idx = perm[:split]
    test_idx = perm[split:] if split < n else perm[:split]

    arrays = normalize_arrays(raw, train_idx, Path(args.stats_out))
    k = infer_max_visible(arrays["mask"], args.max_visible)
    extra = 1 if args.use_individual_reward else 0
    print(
        f"token budget: packed obs(1+{k})+action+reward_team"
        f"{'+reward_ind' if extra else ''}+done "
        f"=> len={1 + k + 1 + 1 + extra + 1}"
    )

    _, rows = train_doc_model(arrays, train_idx, test_idx, args)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "noise_std",
        "loss",
        "static_mse",
        "dynamic_masked_mse",
        "action_ce",
        "action_acc",
        "reward_team_mse",
        "reward_individual_mse",
        "done_mse",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    print("\n=== doc_offline_sequence reconstruct under latent noise ===")
    for r in rows:
        print(
            f"{r['model']}, noise={r['noise_std']:.3f}, "
            f"static_mse={r['static_mse']:.6f}, "
            f"dynamic_mse={r['dynamic_masked_mse']:.6f}, "
            f"action_acc={r['action_acc']:.4f}, "
            f"r_team={r['reward_team_mse']:.6f}, "
            f"done={r['done_mse']:.6f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
