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


class MLP(nn.Module):
    dims: Sequence[int]
    activate_last: bool = False

    @nn.compact
    def __call__(self, x):
        for i, d in enumerate(self.dims):
            x = nn.Dense(d)(x)
            if i < len(self.dims) - 1 or self.activate_last:
                x = nn.gelu(x)
        return x


class SlotLatentAE(nn.Module):
    static_dim: int
    dyn_dim: int
    action_dim: int = 8
    latent_dim: int = 64
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, static, dyn, mask, action, noise_std: float):
        # static: [B, S]
        # dyn:    [B, M, F]
        # mask:   [B, M]
        # action: [B]
        z_s = MLP([self.hidden_dim, self.latent_dim])(static)
        z_d = MLP([self.hidden_dim, self.latent_dim])(dyn)
        z_a = nn.Embed(self.action_dim, self.latent_dim)(action)

        k1 = self.make_rng("noise_s")
        k2 = self.make_rng("noise_d")
        k3 = self.make_rng("noise_a")
        z_s = z_s + noise_std * jax.random.normal(k1, z_s.shape)
        z_d = z_d + noise_std * jax.random.normal(k2, z_d.shape)
        z_a = z_a + noise_std * jax.random.normal(k3, z_a.shape)

        # Decoder input uses LayerNorm, matching doc suggestion.
        pred_static = MLP([self.hidden_dim, self.static_dim])(
            nn.LayerNorm()(z_s)
        )
        pred_dyn = MLP([self.hidden_dim, self.dyn_dim])(
            nn.LayerNorm()(z_d)
        )
        action_logits = MLP([self.hidden_dim, self.action_dim])(
            nn.LayerNorm()(z_a)
        )

        return pred_static, pred_dyn, action_logits


class FlatLatentAE(nn.Module):
    static_dim: int
    dyn_slots: int
    dyn_dim: int
    action_dim: int = 8
    latent_dim: int = 64
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, static, dyn, mask, action, noise_std: float):
        onehot = jax.nn.one_hot(action, self.action_dim)
        dyn_masked = dyn * mask[..., None]
        x = jnp.concatenate(
            [
                static,
                dyn_masked.reshape((dyn_masked.shape[0], -1)),
                onehot,
            ],
            axis=-1,
        )

        z = MLP([self.hidden_dim, self.latent_dim])(x)
        k = self.make_rng("noise")
        z = z + noise_std * jax.random.normal(k, z.shape)

        h = nn.LayerNorm()(z)
        out_dim = self.static_dim + self.dyn_slots * self.dyn_dim + self.action_dim
        y = MLP([self.hidden_dim, out_dim])(h)

        s_end = self.static_dim
        d_end = s_end + self.dyn_slots * self.dyn_dim
        pred_static = y[:, :s_end]
        pred_dyn = y[:, s_end:d_end].reshape((-1, self.dyn_slots, self.dyn_dim))
        action_logits = y[:, d_end:]
        return pred_static, pred_dyn, action_logits


class TrainState(train_state.TrainState):
    pass


def load_data(records_root: Path):
    agent_dirs = sorted(records_root.glob("**/agent_centric/unit_*"))
    if not agent_dirs:
        agent_dirs = sorted(records_root.glob("**/agent_centric/ally_*"))
    if not agent_dirs:
        raise FileNotFoundError(f"No agent_centric/unit_* or ally_* under {records_root}")

    statics, dyns, masks, actions = [], [], [], []
    source_nan = {}

    for ad in agent_dirs:
        s = np.load(ad / "obs_static.npy").astype(np.float32)
        d = np.load(ad / "obs_dynamic.npy").astype(np.float32)
        m = np.load(ad / "obs_dynamic_mask.npy").astype(np.float32)
        a = np.load(ad / "behavior_action.npy").astype(np.int32)

        source_nan[str(ad / "obs_static.npy")] = int(np.isnan(s).sum())
        source_nan[str(ad / "obs_dynamic.npy")] = int(np.isnan(d).sum())

        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d = d * m[..., None]

        statics.append(s)
        dyns.append(d)
        masks.append(m)
        actions.append(a)

    static = np.concatenate(statics, axis=0)
    dyn = np.concatenate(dyns, axis=0)
    mask = np.concatenate(masks, axis=0)
    action = np.concatenate(actions, axis=0)

    # Safety: action should be 0..7.
    action = np.clip(action, 0, 7)

    return static, dyn, mask, action, source_nan


def normalize_train_test(static, dyn, mask, train_idx, test_idx):
    s_train = static[train_idx]
    d_train = dyn[train_idx]
    m_train = mask[train_idx]

    s_mean = s_train.mean(axis=0, keepdims=True)
    s_std = s_train.std(axis=0, keepdims=True)
    s_std = np.maximum(s_std, 0.05)

    valid_dyn = d_train[m_train > 0.5]
    if len(valid_dyn) == 0:
        valid_dyn = d_train.reshape((-1, d_train.shape[-1]))
    d_mean = valid_dyn.mean(axis=0, keepdims=True)
    d_std = valid_dyn.std(axis=0, keepdims=True)
    d_std = np.maximum(d_std, 0.05)

    static_n = (static - s_mean) / s_std
    dyn_n = (dyn - d_mean.reshape((1, 1, -1))) / d_std.reshape((1, 1, -1))
    dyn_n = dyn_n * mask[..., None]

    data = {
        "train": {
            "static": static_n[train_idx].astype(np.float32),
            "dyn": dyn_n[train_idx].astype(np.float32),
            "mask": mask[train_idx].astype(np.float32),
        },
        "test": {
            "static": static_n[test_idx].astype(np.float32),
            "dyn": dyn_n[test_idx].astype(np.float32),
            "mask": mask[test_idx].astype(np.float32),
        },
    }
    return data


def make_batches(data, action, idx, batch_size, rng):
    perm = rng.permutation(len(idx))
    idx = idx[perm]
    for st in range(0, len(idx), batch_size):
        b = idx[st:st + batch_size]
        yield {
            "static": data["static"][b],
            "dyn": data["dyn"][b],
            "mask": data["mask"][b],
            "action": action[b],
        }


def loss_metrics(params, model, batch, key, noise_std):
    pred_s, pred_d, logits = model.apply(
        {"params": params},
        batch["static"],
        batch["dyn"],
        batch["mask"],
        batch["action"],
        noise_std,
        rngs={"noise": key, "noise_s": key, "noise_d": key, "noise_a": key},
    )

    static_mse = jnp.mean((pred_s - batch["static"]) ** 2)

    dyn_sq = ((pred_d - batch["dyn"]) ** 2) * batch["mask"][..., None]
    denom = jnp.maximum(jnp.sum(batch["mask"]) * batch["dyn"].shape[-1], 1.0)
    dyn_mse = jnp.sum(dyn_sq) / denom

    act_ce = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["action"]
    ).mean()
    act_acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch["action"])

    loss = static_mse + dyn_mse + act_ce
    return loss, {
        "loss": loss,
        "static_mse": static_mse,
        "dynamic_masked_mse": dyn_mse,
        "action_ce": act_ce,
        "action_acc": act_acc,
    }


@jax.jit
def train_step(state, batch, key):
    def _loss_fn(params):
        loss, metrics = loss_metrics(params, state.apply_fn, batch, key, 0.0)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def eval_model(state, model, arrays, action, batch_size, noise_stds):
    results = []
    n = len(action)
    full_idx = np.arange(n)

    for noise_std in noise_stds:
        sums = {
            "loss": 0.0,
            "static_mse": 0.0,
            "dynamic_masked_mse": 0.0,
            "action_ce": 0.0,
            "action_acc": 0.0,
        }
        count = 0

        for st in range(0, n, batch_size):
            b = full_idx[st:st + batch_size]
            batch = {
                "static": jnp.asarray(arrays["static"][b]),
                "dyn": jnp.asarray(arrays["dyn"][b]),
                "mask": jnp.asarray(arrays["mask"][b]),
                "action": jnp.asarray(action[b]),
            }
            key = jax.random.PRNGKey(1234 + int(noise_std * 10000) + st)
            _, metrics = loss_metrics(state.params, model, batch, key, float(noise_std))
            bs = len(b)
            for k in sums:
                sums[k] += float(metrics[k]) * bs
            count += bs

        row = {k: v / count for k, v in sums.items()}
        row["noise_std"] = noise_std
        results.append(row)

    return results


def train_one_model(name, model, train_arrays, test_arrays, action_train, action_test, args):
    key = jax.random.PRNGKey(args.seed)

    init_batch = {
        "static": jnp.asarray(train_arrays["static"][: min(4, len(action_train))]),
        "dyn": jnp.asarray(train_arrays["dyn"][: min(4, len(action_train))]),
        "mask": jnp.asarray(train_arrays["mask"][: min(4, len(action_train))]),
        "action": jnp.asarray(action_train[: min(4, len(action_train))]),
    }

    params = model.init(
        key,
        init_batch["static"],
        init_batch["dyn"],
        init_batch["mask"],
        init_batch["action"],
        0.0,
    )["params"]

    tx = optax.adam(args.lr)
    state = TrainState.create(apply_fn=model, params=params, tx=tx)

    rng_np = np.random.default_rng(args.seed)
    train_idx = np.arange(len(action_train))

    for step in range(1, args.steps + 1):
        for batch_np in make_batches(train_arrays, action_train, train_idx, args.batch_size, rng_np):
            key, subkey = jax.random.split(key)
            batch = {k: jnp.asarray(v) for k, v in batch_np.items()}
            state, metrics = train_step(state, batch, subkey)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"[{name}] epoch={step:04d} "
                f"loss={float(metrics['loss']):.6f} "
                f"static={float(metrics['static_mse']):.6f} "
                f"dyn={float(metrics['dynamic_masked_mse']):.6f} "
                f"act_acc={float(metrics['action_acc']):.4f}",
                flush=True,
            )

    noise_stds = [float(x) for x in args.noise_stds.split(",")]
    rows = eval_model(state, model, test_arrays, action_test, args.batch_size, noise_stds)
    for r in rows:
        r["model"] = name
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_root", type=str, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--noise_stds", type=str, default="0,0.01,0.05,0.1")
    ap.add_argument("--out_csv", type=str, default="outputs/latent_reconstruct_results.csv")
    args = ap.parse_args()

    records_root = Path(args.records_root)
    static, dyn, mask, action, source_nan = load_data(records_root)

    print("loaded:")
    print("  static:", static.shape)
    print("  dyn:", dyn.shape)
    print("  mask:", mask.shape)
    print("  action:", action.shape)
    print("  source obs_static nan:", sum(v for k, v in source_nan.items() if "obs_static" in k))
    print("  source obs_dynamic nan:", sum(v for k, v in source_nan.items() if "obs_dynamic" in k))
    print("  note: NaN is replaced with 0 for this smoke benchmark.")

    n = len(action)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    split = max(1, int(0.8 * n))
    train_idx, test_idx = perm[:split], perm[split:]
    if len(test_idx) == 0:
        test_idx = train_idx

    normed = normalize_train_test(static, dyn, mask, train_idx, test_idx)

    action_train = action[train_idx]
    action_test = action[test_idx]

    train_arrays = normed["train"]
    test_arrays = normed["test"]

    static_dim = static.shape[-1]
    dyn_slots = dyn.shape[1]
    dyn_dim = dyn.shape[-1]

    slot_model = SlotLatentAE(
        static_dim=static_dim,
        dyn_dim=dyn_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )
    flat_model = FlatLatentAE(
        static_dim=static_dim,
        dyn_slots=dyn_slots,
        dyn_dim=dyn_dim,
        latent_dim=args.latent_dim,
        hidden_dim=max(args.hidden_dim, 256),
    )

    all_rows = []
    all_rows.extend(
        train_one_model(
            "slot_latent_static_dynamic_action",
            slot_model,
            train_arrays,
            test_arrays,
            action_train,
            action_test,
            args,
        )
    )
    all_rows.extend(
        train_one_model(
            "flat_latent_baseline",
            flat_model,
            train_arrays,
            test_arrays,
            action_train,
            action_test,
            args,
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
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fieldnames})

    print("\n=== test reconstruction under latent noise ===")
    for r in all_rows:
        print(
            f"{r['model']}, noise={r['noise_std']:.3f}, "
            f"static_mse={r['static_mse']:.6f}, "
            f"dynamic_mse={r['dynamic_masked_mse']:.6f}, "
            f"action_acc={r['action_acc']:.4f}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
