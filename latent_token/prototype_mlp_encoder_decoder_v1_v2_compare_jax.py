from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

from latent_token.offline_sequence_latent_ae import (
    load_agent_data as load_doc_agent_data,
    normalize_arrays as normalize_doc_arrays,
    train_doc_model,
)


class MLP(nn.Module):
    dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, d in enumerate(self.dims):
            x = nn.Dense(d)(x)
            if i < len(self.dims) - 1:
                x = nn.gelu(x)
        return x


class VisibleTokenAE(nn.Module):
    static_dim: int
    dyn_dim: int
    action_dim: int = 8
    latent_dim: int = 64
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, static, dyn, mask, action, noise_std: float, train: bool = True):
        # static: [B, S]
        # dyn:    [B, M, F]
        # mask:   [B, M], 1 means visible / valid object
        # action: [B]

        z_static = MLP([self.hidden_dim, self.latent_dim])(static)
        z_dyn = MLP([self.hidden_dim, self.latent_dim])(dyn)
        z_action = nn.Embed(self.action_dim, self.latent_dim)(action)

        k_s = self.make_rng("noise_s")
        k_d = self.make_rng("noise_d")
        k_a = self.make_rng("noise_a")

        z_static = z_static + noise_std * jax.random.normal(k_s, z_static.shape)
        z_dyn = z_dyn + noise_std * jax.random.normal(k_d, z_dyn.shape)
        z_action = z_action + noise_std * jax.random.normal(k_a, z_action.shape)

        # invisible/padding dynamic tokens are not valid tokens
        z_dyn = z_dyn * mask[..., None]

        valid_count = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
        z_dyn_pool = jnp.sum(z_dyn, axis=1) / valid_count

        # decoder input LayerNorm, following doc/OFFLINE_SEQUENCE.md
        pred_static = MLP([self.hidden_dim, self.static_dim])(nn.LayerNorm()(z_static))
        pred_dyn = MLP([self.hidden_dim, self.dyn_dim])(nn.LayerNorm()(z_dyn))

        context = jnp.concatenate([z_static, z_dyn_pool, z_action], axis=-1)
        context = nn.LayerNorm()(context)

        action_logits = MLP([self.hidden_dim, self.action_dim])(context)
        pred_reward = MLP([self.hidden_dim, 1])(context).squeeze(-1)

        return pred_static, pred_dyn, action_logits, pred_reward



class FixedSlotV1AE(nn.Module):
    static_dim: int
    dyn_dim: int
    action_dim: int = 8
    latent_dim: int = 64
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, static, dyn, mask, action, noise_std: float, train: bool = True):
        # v1 normalized fixed-slot scheme:
        # [token_static, token_obj1, ..., token_objM, token_action]
        # All dynamic slots are encoded. Invisible/padding slots have been zeroed by preprocessing.
        # Loss is still computed only on mask=1 visible/valid dynamic objects.
        z_static = MLP([self.hidden_dim, self.latent_dim])(static)
        z_dyn = MLP([self.hidden_dim, self.latent_dim])(dyn)
        z_action = nn.Embed(self.action_dim, self.latent_dim)(action)

        k_s = self.make_rng("noise_s")
        k_d = self.make_rng("noise_d")
        k_a = self.make_rng("noise_a")

        z_static = z_static + noise_std * jax.random.normal(k_s, z_static.shape)
        z_dyn = z_dyn + noise_std * jax.random.normal(k_d, z_dyn.shape)
        z_action = z_action + noise_std * jax.random.normal(k_a, z_action.shape)

        pred_static = MLP([self.hidden_dim, self.static_dim])(nn.LayerNorm()(z_static))
        pred_dyn = MLP([self.hidden_dim, self.dyn_dim])(nn.LayerNorm()(z_dyn))

        # v1 fixed-slot context pools all slots, including zeroed invisible/padding slots.
        z_dyn_pool = jnp.mean(z_dyn, axis=1)
        context = jnp.concatenate([z_static, z_dyn_pool, z_action], axis=-1)
        context = nn.LayerNorm()(context)

        action_logits = MLP([self.hidden_dim, self.action_dim])(context)
        pred_reward = MLP([self.hidden_dim, 1])(context).squeeze(-1)

        return pred_static, pred_dyn, action_logits, pred_reward


class FlatBaselineAE(nn.Module):
    static_dim: int
    dyn_slots: int
    dyn_dim: int
    action_dim: int = 8
    latent_dim: int = 64
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, static, dyn, mask, action, noise_std: float, train: bool = True):
        dyn_masked = dyn * mask[..., None]
        action_onehot = jax.nn.one_hot(action, self.action_dim)

        x = jnp.concatenate(
            [static, dyn_masked.reshape((dyn.shape[0], -1)), action_onehot],
            axis=-1,
        )

        z = MLP([self.hidden_dim, self.latent_dim])(x)
        k = self.make_rng("noise")
        z = z + noise_std * jax.random.normal(k, z.shape)
        z = nn.LayerNorm()(z)

        total_out = self.static_dim + self.dyn_slots * self.dyn_dim + self.action_dim + 1
        y = MLP([self.hidden_dim, total_out])(z)

        s_end = self.static_dim
        d_end = s_end + self.dyn_slots * self.dyn_dim
        a_end = d_end + self.action_dim

        pred_static = y[:, :s_end]
        pred_dyn = y[:, s_end:d_end].reshape((-1, self.dyn_slots, self.dyn_dim))
        action_logits = y[:, d_end:a_end]
        pred_reward = y[:, a_end]

        return pred_static, pred_dyn, action_logits, pred_reward


class TrainState(train_state.TrainState):
    pass


def load_agent_data(records_root: Path):
    agent_dirs = sorted(records_root.glob("**/agent_centric/unit_*"))
    if not agent_dirs:
        agent_dirs = sorted(records_root.glob("**/agent_centric/ally_*"))
    if not agent_dirs:
        raise FileNotFoundError(f"No agent-centric dirs found under {records_root}")

    statics, dyns, masks, actions, rewards = [], [], [], [], []
    source_nan_static = 0
    source_nan_dyn = 0

    for ad in agent_dirs:
        s = np.load(ad / "obs_static.npy").astype(np.float32)
        d = np.load(ad / "obs_dynamic.npy").astype(np.float32)
        m = np.load(ad / "obs_dynamic_mask.npy").astype(np.float32)
        a = np.load(ad / "behavior_action.npy").astype(np.int32)

        if (ad / "reward_team.npy").exists():
            r = np.load(ad / "reward_team.npy").astype(np.float32)
        else:
            r = np.zeros_like(a, dtype=np.float32)

        source_nan_static += int(np.isnan(s).sum())
        source_nan_dyn += int(np.isnan(d).sum())

        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d = d * m[..., None]
        a = np.clip(a, 0, 7)

        statics.append(s)
        dyns.append(d)
        masks.append(m)
        actions.append(a)
        rewards.append(r)

    static = np.concatenate(statics, axis=0)
    dyn = np.concatenate(dyns, axis=0)
    mask = np.concatenate(masks, axis=0)
    action = np.concatenate(actions, axis=0)
    reward = np.concatenate(rewards, axis=0)

    return static, dyn, mask, action, reward, source_nan_static, source_nan_dyn


def normalize(static, dyn, mask, reward, train_idx, out_stats: Path):
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
    r_std = np.array([max(r_train.std(), 0.05)], dtype=np.float32)

    static_n = (static - s_mean) / s_std
    dyn_n = (dyn - d_mean.reshape((1, 1, -1))) / d_std.reshape((1, 1, -1))
    dyn_n = dyn_n * mask[..., None]
    reward_n = (reward - r_mean[0]) / r_std[0]

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


def loss_metrics(params, model, batch, key, noise_std: float):
    pred_s, pred_d, logits, pred_r = model.apply(
        {"params": params},
        batch["static"],
        batch["dyn"],
        batch["mask"],
        batch["action"],
        noise_std,
        rngs={
            "noise": key,
            "noise_s": key,
            "noise_d": key,
            "noise_a": key,
        },
    )

    static_mse = jnp.mean((pred_s - batch["static"]) ** 2)

    dyn_sq = ((pred_d - batch["dyn"]) ** 2) * batch["mask"][..., None]
    dyn_denom = jnp.maximum(jnp.sum(batch["mask"]) * batch["dyn"].shape[-1], 1.0)
    dyn_mse = jnp.sum(dyn_sq) / dyn_denom

    action_ce = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["action"]
    ).mean()
    action_acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch["action"])

    reward_mse = jnp.mean((pred_r - batch["reward"]) ** 2)

    loss = static_mse + dyn_mse + action_ce + 0.2 * reward_mse

    return loss, {
        "loss": loss,
        "static_mse": static_mse,
        "dynamic_masked_mse": dyn_mse,
        "action_ce": action_ce,
        "action_acc": action_acc,
        "reward_mse": reward_mse,
    }


@jax.jit
def train_step(state, batch, key):
    def fn(params):
        return loss_metrics(params, state.apply_fn, batch, key, 0.0)

    (loss, metrics), grads = jax.value_and_grad(fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def batch_iter(arrays, idx, batch_size, rng):
    perm = rng.permutation(idx)
    for st in range(0, len(perm), batch_size):
        b = perm[st: st + batch_size]
        yield {
            "static": jnp.asarray(arrays["static"][b]),
            "dyn": jnp.asarray(arrays["dyn"][b]),
            "mask": jnp.asarray(arrays["mask"][b]),
            "action": jnp.asarray(arrays["action"][b]),
            "reward": jnp.asarray(arrays["reward"][b]),
        }


def eval_model(state, arrays, idx, batch_size, noise_stds):
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
            b = idx[st: st + batch_size]
            batch = {
                "static": jnp.asarray(arrays["static"][b]),
                "dyn": jnp.asarray(arrays["dyn"][b]),
                "mask": jnp.asarray(arrays["mask"][b]),
                "action": jnp.asarray(arrays["action"][b]),
                "reward": jnp.asarray(arrays["reward"][b]),
            }
            key = jax.random.PRNGKey(1000 + st + int(noise_std * 10000))
            _, metrics = loss_metrics(state.params, state.apply_fn, batch, key, float(noise_std))

            bs = len(b)
            for k in sums:
                sums[k] += float(metrics[k]) * bs
            count += bs

        row = {k: v / max(count, 1) for k, v in sums.items()}
        row["noise_std"] = noise_std
        rows.append(row)

    return rows


def train_model(name, model, arrays, train_idx, test_idx, args):
    key = jax.random.PRNGKey(args.seed)

    init_idx = train_idx[: min(4, len(train_idx))]
    init_batch = {
        "static": jnp.asarray(arrays["static"][init_idx]),
        "dyn": jnp.asarray(arrays["dyn"][init_idx]),
        "mask": jnp.asarray(arrays["mask"][init_idx]),
        "action": jnp.asarray(arrays["action"][init_idx]),
        "reward": jnp.asarray(arrays["reward"][init_idx]),
    }

    params = model.init(
        key,
        init_batch["static"],
        init_batch["dyn"],
        init_batch["mask"],
        init_batch["action"],
        0.0,
    )["params"]

    state = TrainState.create(
        apply_fn=model,
        params=params,
        tx=optax.adam(args.lr),
    )

    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        for batch in batch_iter(arrays, train_idx, args.batch_size, rng):
            key, subkey = jax.random.split(key)
            state, metrics = train_step(state, batch, subkey)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[{name}] epoch={epoch:04d} "
                f"loss={float(metrics['loss']):.6f} "
                f"static={float(metrics['static_mse']):.6f} "
                f"dyn={float(metrics['dynamic_masked_mse']):.6f} "
                f"act_acc={float(metrics['action_acc']):.4f} "
                f"reward={float(metrics['reward_mse']):.6f}",
                flush=True,
            )

    noise_stds = [float(x) for x in args.noise_stds.split(",")]
    rows = eval_model(state, arrays, test_idx, args.batch_size, noise_stds)
    for r in rows:
        r["model"] = name
    return rows


def train_doc_offline_sequence(records_root: Path, args):
    """Doc model with token order obs→action→reward_team→[rind]→done."""

    raw = load_doc_agent_data(
        records_root, use_individual_reward=bool(args.use_individual_reward)
    )
    n = len(raw["action"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    split = max(1, int(0.8 * n))
    train_idx = perm[:split]
    test_idx = perm[split:] if split < n else perm[:split]
    arrays = normalize_doc_arrays(raw, train_idx, Path(args.stats_out).with_name(
        Path(args.stats_out).stem + "_doc.npz"
    ))
    # Ensure CLI fields expected by train_doc_model
    if not hasattr(args, "reward_team_coef"):
        args.reward_team_coef = 0.2
    if not hasattr(args, "reward_ind_coef"):
        args.reward_ind_coef = 0.2
    if not hasattr(args, "dropout"):
        args.dropout = 0.0
    if not hasattr(args, "residual_decode"):
        args.residual_decode = False
    _, rows = train_doc_model(arrays, train_idx, test_idx, args)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_root", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--noise_stds", default="0,0.01,0.05,0.1")
    ap.add_argument("--out_csv", default="outputs/v1_v2_doc_reconstruct_compare.csv")
    ap.add_argument("--stats_out", default="outputs/v1_v2_doc_norm_stats.npz")
    ap.add_argument(
        "--models",
        default="v1,v2,doc,flat",
        help="Comma list among: v1,v2,doc,flat",
    )
    ap.add_argument(
        "--max_visible",
        type=int,
        default=None,
        help="Doc model: max packed visible dyn tokens (default: data max)",
    )
    ap.add_argument(
        "--use_individual_reward",
        action="store_true",
        help="Doc model: include optional reward_individual token",
    )
    ap.add_argument("--reward_team_coef", type=float, default=0.2)
    ap.add_argument("--reward_ind_coef", type=float, default=0.2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--residual_decode", action="store_true")
    args = ap.parse_args()

    records_root = Path(args.records_root)
    static, dyn, mask, action, reward, nan_s, nan_d = load_agent_data(records_root)

    print("loaded")
    print("  static:", static.shape)
    print("  dyn:", dyn.shape)
    print("  mask:", mask.shape)
    print("  action:", action.shape)
    print("  reward:", reward.shape)
    print("  source obs_static NaN:", nan_s)
    print("  source obs_dynamic NaN:", nan_d)
    print("  note: source NaN is replaced with 0 before normalization in this prototype")

    n = len(action)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    split = max(1, int(0.8 * n))
    train_idx = perm[:split]
    test_idx = perm[split:] if split < n else perm[:split]

    static_n, dyn_n, reward_n = normalize(
        static,
        dyn,
        mask,
        reward,
        train_idx,
        Path(args.stats_out),
    )

    arrays = {
        "static": static_n,
        "dyn": dyn_n,
        "mask": mask.astype(np.float32),
        "action": action.astype(np.int32),
        "reward": reward_n.astype(np.float32),
    }

    avg_visible = float(mask.sum(axis=1).mean())
    fixed_tokens = 1 + dyn.shape[1] + 1
    visible_tokens = 1 + avg_visible + 1

    print("token budget")
    print("  fixed slot tokens per step:", fixed_tokens)
    print("  avg visible-token per step:", visible_tokens)

    static_dim = static.shape[-1]
    dyn_slots = dyn.shape[1]
    dyn_dim = dyn.shape[-1]

    fixed_v1_model = FixedSlotV1AE(
        static_dim=static_dim,
        dyn_dim=dyn_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )

    visible_model = VisibleTokenAE(
        static_dim=static_dim,
        dyn_dim=dyn_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )

    flat_model = FlatBaselineAE(
        static_dim=static_dim,
        dyn_slots=dyn_slots,
        dyn_dim=dyn_dim,
        latent_dim=args.latent_dim,
        hidden_dim=max(args.hidden_dim, 256),
    )

    wanted = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    all_rows = []
    if "v1" in wanted:
        all_rows.extend(
            train_model(
                "fixed_slot_v1_normalized",
                fixed_v1_model,
                arrays,
                train_idx,
                test_idx,
                args,
            )
        )
    if "v2" in wanted:
        all_rows.extend(
            train_model("visible_token_v2", visible_model, arrays, train_idx, test_idx, args)
        )
    if "doc" in wanted:
        all_rows.extend(train_doc_offline_sequence(records_root, args))
    if "flat" in wanted:
        all_rows.extend(
            train_model(
                "flat_latent_baseline", flat_model, arrays, train_idx, test_idx, args
            )
        )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fieldnames})

    print("\n=== v1 / v2 / doc_offline_sequence reconstruct under latent noise ===")
    for r in all_rows:
        print(
            f"{r['model']}, noise={r['noise_std']:.3f}, "
            f"static_mse={r['static_mse']:.6f}, "
            f"dynamic_mse={r['dynamic_masked_mse']:.6f}, "
            f"action_acc={r['action_acc']:.4f}, "
            f"reward_mse={r['reward_mse']:.6f}"
        )

    print(f"\nwrote {out}")
    print(f"wrote {args.stats_out}")


if __name__ == "__main__":
    main()
