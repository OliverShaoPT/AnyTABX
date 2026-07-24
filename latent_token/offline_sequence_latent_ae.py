"""Doc-faithful continuous latent-token AE (OFFLINE_SEQUENCE.md §6).

Implements:
  - Static / shared Dynamic / Action encoders (encoder: no LayerNorm)
  - Visible dyn only: pack mask=1 objects; sequence length 1+K+1 (K=max_visible)
  - Token type embeddings: STATIC / DYN / ACT (DYN only on real visible slots)
  - Token sequence [z_static, z_dyn_1..K, z_act] (no Transformer backbone)
  - ResidualMLPDecoder: LayerNorm -> MLP (+ optional residual)
  - Losses: reconstruct static/dyn (masked), action CE from obs tokens,
    optional reward regression (label only, not in token sequence by default)
  - Latent noise reconstruct evaluation

Training data: agent-centric dumps under generate/ (obs_static / obs_dynamic / mask).
Dump may still store fixed M slots; this module packs visibles before encoding.
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

# Token type ids (doc §6.5)
TYPE_STATIC = 0
TYPE_DYN = 1
TYPE_ACT = 2
NUM_TOKEN_TYPES = 3


def pack_visible_dyn(
    dyn: jax.Array,
    mask: jax.Array,
    max_visible: int,
) -> tuple[jax.Array, jax.Array]:
    """Compact visible dyn objects to the front; keep at most ``max_visible``.

    Invisible / leftover slots are dropped (not kept as zero-padded M-length
    tokens). Within a batch, length is ``max_visible`` with trailing mask=0
    only when a sample has fewer visibles than ``max_visible``.
    """
    if max_visible < 1:
        raise ValueError("max_visible must be >= 1")
    # Visible (1) sorts before invisible (0).
    order = jnp.argsort(-mask, axis=-1)
    idx = order[:, :max_visible]
    dyn_p = jnp.take_along_axis(dyn, idx[..., None], axis=1)
    mask_p = jnp.take_along_axis(mask, idx, axis=1)
    dyn_p = dyn_p * mask_p[..., None]
    return dyn_p, mask_p


class MLPEncoderContinuous(nn.Module):
    """airsoul-style continuous encoder: Linear / (Linear→GELU→Dropout)* → Linear."""

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
    """airsoul ResidualMLPDecoder: LN(input) → MLP → optional residual → out."""

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
            # airsoul: pre maps to input dim, residual add, then post to out_dim
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
    """§6 layout: pack visible dyn → encode tokens → decode (no Transformer)."""

    static_dim: int
    dyn_dim: int
    max_visible: int  # K: max visible dyn tokens kept (not full dump M)
    action_dim: int = 8
    latent_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.0
    residual_decode: bool = False
    # Reward is a training label by default (doc §6.5); not inserted as z_rew token.

    @nn.compact
    def __call__(
        self,
        static,
        dyn,
        mask,
        action,
        noise_std: float = 0.0,
        *,
        train: bool = True,
    ):
        """
        static: [B, S]
        dyn:    [B, M, F]  dump slots (may include invisible); packed inside
        mask:   [B, M]
        action: [B]

        Token length is always ``1 + max_visible + 1``, not ``1 + M + 1``.
        """
        bsz = static.shape[0]
        enc_sizes = (self.hidden_dim, self.latent_dim)
        dec_sizes = (self.hidden_dim, self.hidden_dim)

        # Drop invisible dump slots: keep at most K visibles (doc §6.2).
        dyn_p, mask_p = pack_visible_dyn(dyn, mask, self.max_visible)
        k = self.max_visible

        # --- encoders (no LN); only packed K slots ---
        z_static = MLPEncoderContinuous(enc_sizes, self.dropout)(static, train=train)
        z_dyn = MLPEncoderContinuous(enc_sizes, self.dropout)(dyn_p, train=train)
        z_dyn = z_dyn * mask_p[..., None]
        z_act = nn.Embed(self.action_dim, self.latent_dim)(action)

        if noise_std > 0:
            k_s = self.make_rng("noise_s")
            k_d = self.make_rng("noise_d")
            k_a = self.make_rng("noise_a")
            z_static = z_static + noise_std * jax.random.normal(k_s, z_static.shape)
            z_dyn = z_dyn + noise_std * jax.random.normal(k_d, z_dyn.shape)
            z_act = z_act + noise_std * jax.random.normal(k_a, z_act.shape)
            z_dyn = z_dyn * mask_p[..., None]

        # Type embeds: DYN only on real visible packed slots (mask_p).
        type_embed = nn.Embed(NUM_TOKEN_TYPES, self.latent_dim)
        z_static = z_static + type_embed(jnp.full((bsz,), TYPE_STATIC, dtype=jnp.int32))
        dyn_type = type_embed(jnp.full((bsz, k), TYPE_DYN, dtype=jnp.int32))
        z_dyn = z_dyn + dyn_type * mask_p[..., None]
        z_act = z_act + type_embed(jnp.full((bsz,), TYPE_ACT, dtype=jnp.int32))

        tokens = jnp.concatenate(
            [z_static[:, None, :], z_dyn, z_act[:, None, :]],
            axis=1,
        )  # [B, 1+K+1, D]

        tok_mask = jnp.concatenate(
            [
                jnp.ones((bsz, 1), dtype=mask_p.dtype),
                mask_p,
                jnp.ones((bsz, 1), dtype=mask_p.dtype),
            ],
            axis=1,
        )

        # Action CE from obs tokens only (static + visible dyn).
        obs_mask = jnp.concatenate(
            [jnp.ones((bsz, 1), dtype=mask_p.dtype), mask_p],
            axis=1,
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
        pred_reward = ResidualMLPDecoder(
            1, dec_sizes, self.dropout, False
        )(z_obs_ctx, train=train).squeeze(-1)

        return {
            "pred_static": pred_static,
            "pred_dyn": pred_dyn,  # packed [B, K, F]
            "dyn_target": dyn_p,
            "dyn_mask": mask_p,
            "action_logits": action_logits,
            "pred_reward": pred_reward,
            "tokens": tokens,
            "token_mask": tok_mask,
            "max_visible": self.max_visible,
        }


class TrainState(train_state.TrainState):
    pass


def load_agent_data(records_root: Path):
    agent_dirs = sorted(records_root.glob("**/agent_centric/unit_*"))
    if not agent_dirs:
        agent_dirs = sorted(records_root.glob("**/agent_centric/ally_*"))
    if not agent_dirs:
        raise FileNotFoundError(f"No agent-centric dirs found under {records_root}")

    statics, dyns, masks, actions, rewards = [], [], [], [], []
    for ad in agent_dirs:
        s = np.load(ad / "obs_static.npy").astype(np.float32)
        d = np.load(ad / "obs_dynamic.npy").astype(np.float32)
        m = np.load(ad / "obs_dynamic_mask.npy").astype(np.float32)
        a = np.load(ad / "behavior_action.npy").astype(np.int32)
        if (ad / "reward_team.npy").exists():
            r = np.load(ad / "reward_team.npy").astype(np.float32)
        else:
            r = np.zeros_like(a, dtype=np.float32)

        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d = d * m[..., None]
        a = np.clip(a, 0, 7)
        statics.append(s)
        dyns.append(d)
        masks.append(m)
        actions.append(a)
        rewards.append(r)

    return (
        np.concatenate(statics, 0),
        np.concatenate(dyns, 0),
        np.concatenate(masks, 0),
        np.concatenate(actions, 0),
        np.concatenate(rewards, 0),
    )


def normalize(static, dyn, mask, reward, train_idx, out_stats: Path | None = None):
    s_train = static[train_idx]
    d_train = dyn[train_idx]
    m_train = mask[train_idx]
    r_train = reward[train_idx]

    s_mean = s_train.mean(axis=0, keepdims=True)
    s_std = np.maximum(s_train.std(axis=0, keepdims=True), 0.05)
    valid_dyn = d_train[m_train > 0.5]
    if len(valid_dyn) == 0:
        valid_dyn = d_train.reshape((-1, d_train.shape[-1]))
    d_mean = valid_dyn.mean(axis=0, keepdims=True)
    d_std = np.maximum(valid_dyn.std(axis=0, keepdims=True), 0.05)
    r_mean = np.array([r_train.mean()], dtype=np.float32)
    r_std = np.array([max(float(r_train.std()), 0.05)], dtype=np.float32)

    static_n = (static - s_mean) / s_std
    dyn_n = (dyn - d_mean.reshape((1, 1, -1))) / d_std.reshape((1, 1, -1))
    dyn_n = dyn_n * mask[..., None]
    reward_n = (reward - r_mean[0]) / r_std[0]

    if out_stats is not None:
        out_stats.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_stats,
            static_mean=s_mean,
            static_std=s_std,
            dyn_mean=d_mean,
            dyn_std=d_std,
            reward_mean=r_mean,
            reward_std=r_std,
        )
    return static_n.astype(np.float32), dyn_n.astype(np.float32), reward_n.astype(np.float32)


def _metrics_from_out(out, batch, reward_coef: float):
    pred_s = out["pred_static"]
    pred_d = out["pred_dyn"]
    dyn_t = out["dyn_target"]
    dyn_m = out["dyn_mask"]
    logits = out["action_logits"]
    pred_r = out["pred_reward"]

    static_mse = jnp.mean((pred_s - batch["static"]) ** 2)
    # Targets are packed visibles; trailing batch pads have mask 0.
    dyn_sq = ((pred_d - dyn_t) ** 2) * dyn_m[..., None]
    dyn_denom = jnp.maximum(jnp.sum(dyn_m) * dyn_t.shape[-1], 1.0)
    dyn_mse = jnp.sum(dyn_sq) / dyn_denom
    action_ce = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["action"]
    ).mean()
    action_acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch["action"])
    reward_mse = jnp.mean((pred_r - batch["reward"]) ** 2)
    loss = static_mse + dyn_mse + action_ce + reward_coef * reward_mse
    return loss, {
        "loss": loss,
        "static_mse": static_mse,
        "dynamic_masked_mse": dyn_mse,
        "action_ce": action_ce,
        "action_acc": action_acc,
        "reward_mse": reward_mse,
    }


def loss_metrics(params, model, batch, key, noise_std: float, reward_coef: float = 0.2):
    dropout_key, noise_key = jax.random.split(key)
    out = model.apply(
        {"params": params},
        batch["static"],
        batch["dyn"],
        batch["mask"],
        batch["action"],
        noise_std,
        train=True,
        rngs={
            "dropout": dropout_key,
            "noise_s": noise_key,
            "noise_d": noise_key,
            "noise_a": noise_key,
        },
    )
    return _metrics_from_out(out, batch, reward_coef)


def eval_metrics(params, model, batch, key, noise_std: float, reward_coef: float = 0.2):
    out = model.apply(
        {"params": params},
        batch["static"],
        batch["dyn"],
        batch["mask"],
        batch["action"],
        noise_std,
        train=False,
        rngs={
            "noise_s": key,
            "noise_d": key,
            "noise_a": key,
        },
    )
    _, metrics = _metrics_from_out(out, batch, reward_coef)
    return metrics


def make_train_step(reward_coef: float):
    @jax.jit
    def train_step(state, batch, key):
        def fn(params):
            return loss_metrics(
                params, state.apply_fn, batch, key, 0.0, reward_coef=reward_coef
            )

        (loss, metrics), grads = jax.value_and_grad(fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return train_step


def batch_iter(arrays, idx, batch_size, rng):
    perm = rng.permutation(idx)
    for st in range(0, len(perm), batch_size):
        b = perm[st : st + batch_size]
        yield {
            "static": jnp.asarray(arrays["static"][b]),
            "dyn": jnp.asarray(arrays["dyn"][b]),
            "mask": jnp.asarray(arrays["mask"][b]),
            "action": jnp.asarray(arrays["action"][b]),
            "reward": jnp.asarray(arrays["reward"][b]),
        }


def evaluate(state, arrays, idx, batch_size, noise_stds, reward_coef: float = 0.2):
    rows = []
    for noise_std in noise_stds:
        sums = {
            "loss": 0.0,
            "static_mse": 0.0,
            "dynamic_masked_mse": 0.0,
            "action_ce": 0.0,
            "action_acc": 0.0,
            "reward_mse": 0.0,
        }
        count = 0
        for st in range(0, len(idx), batch_size):
            b = idx[st : st + batch_size]
            batch = {
                "static": jnp.asarray(arrays["static"][b]),
                "dyn": jnp.asarray(arrays["dyn"][b]),
                "mask": jnp.asarray(arrays["mask"][b]),
                "action": jnp.asarray(arrays["action"][b]),
                "reward": jnp.asarray(arrays["reward"][b]),
            }
            key = jax.random.PRNGKey(1000 + st + int(noise_std * 10000))
            metrics = eval_metrics(
                state.params, state.apply_fn, batch, key, float(noise_std), reward_coef
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
    """K = max visibles in data (or CLI override); at least 1 for shape stability."""
    if override is not None:
        if override < 1:
            raise ValueError("--max_visible must be >= 1")
        return int(override)
    counted = int(mask.sum(axis=1).max()) if len(mask) else 1
    return max(counted, 1)


def train_doc_model(arrays, train_idx, test_idx, args):
    max_visible = infer_max_visible(
        arrays["mask"], getattr(args, "max_visible", None)
    )
    print(
        f"packing visibles: dump_M={arrays['dyn'].shape[1]}, "
        f"max_visible_K={max_visible}, "
        f"token_len=1+{max_visible}+1={1 + max_visible + 1}"
    )
    model = OfflineSequenceLatentAE(
        static_dim=arrays["static"].shape[-1],
        dyn_dim=arrays["dyn"].shape[-1],
        max_visible=max_visible,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual_decode=args.residual_decode,
    )
    key = jax.random.PRNGKey(args.seed)
    init_idx = train_idx[: min(4, len(train_idx))]
    init_batch = {
        "static": jnp.asarray(arrays["static"][init_idx]),
        "dyn": jnp.asarray(arrays["dyn"][init_idx]),
        "mask": jnp.asarray(arrays["mask"][init_idx]),
        "action": jnp.asarray(arrays["action"][init_idx]),
    }
    variables = model.init(
        {"params": key, "dropout": key, "noise_s": key, "noise_d": key, "noise_a": key},
        init_batch["static"],
        init_batch["dyn"],
        init_batch["mask"],
        init_batch["action"],
        0.0,
        train=True,
    )
    state = TrainState.create(
        apply_fn=model,
        params=variables["params"],
        tx=optax.adam(args.lr),
    )
    step_fn = make_train_step(args.reward_coef)
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
                f"reward={float(metrics['reward_mse']):.6f}",
                flush=True,
            )

    noise_stds = [float(x) for x in args.noise_stds.split(",")]
    rows = evaluate(
        state, arrays, test_idx, args.batch_size, noise_stds, args.reward_coef
    )
    for r in rows:
        r["model"] = "doc_offline_sequence"
    return state, rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train doc §6 OfflineSequenceLatentAE on agent-centric records"
    )
    ap.add_argument("--records_root", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--residual_decode", action="store_true")
    ap.add_argument(
        "--max_visible",
        type=int,
        default=None,
        help="Max packed visible dyn tokens K (default: max over dataset)",
    )
    ap.add_argument("--reward_coef", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--noise_stds", default="0,0.01,0.05,0.1")
    ap.add_argument("--out_csv", default="outputs/doc_offline_sequence_reconstruct.csv")
    ap.add_argument("--stats_out", default="outputs/doc_offline_sequence_norm_stats.npz")
    args = ap.parse_args(argv)

    static, dyn, mask, action, reward = load_agent_data(Path(args.records_root))
    print("loaded", static.shape, dyn.shape, mask.shape, action.shape)

    n = len(action)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    split = max(1, int(0.8 * n))
    train_idx, test_idx = perm[:split], perm[split:] if split < n else perm[:split]

    static_n, dyn_n, reward_n = normalize(
        static, dyn, mask, reward, train_idx, Path(args.stats_out)
    )
    arrays = {
        "static": static_n,
        "dyn": dyn_n,
        "mask": mask.astype(np.float32),
        "action": action.astype(np.int32),
        "reward": reward_n.astype(np.float32),
    }
    avg_vis = float(mask.sum(axis=1).mean())
    k = infer_max_visible(mask, args.max_visible)
    print(
        f"token budget: dump_fixed={1 + dyn.shape[1] + 1}, "
        f"packed=1+K+1 with K={k}, avg_visible≈{avg_vis:.2f}"
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
        "reward_mse",
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
            f"reward_mse={r['reward_mse']:.6f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
