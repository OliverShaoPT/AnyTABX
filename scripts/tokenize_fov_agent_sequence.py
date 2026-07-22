from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


UNIT_FEATURE_NAMES = [
    "is_self",
    "is_ally",
    "is_enemy",
    "rel_x",
    "rel_y",
    "dist",
    "health_ratio",
    "is_alive",
    "speed",
    "attack_range",
    "attack_damage",
    "cooldown",
    "body_radius",
    "is_visible",
    "is_attackable",
]

ZONE_FEATURE_NAMES = [
    "zone_type",
    "rel_x",
    "rel_y",
    "dist",
    "axis_x",
    "axis_y",
    "effect_value",
]


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def build_vocab(config):
    offsets = {}
    vocab_size = max(config["special_tokens"].values()) + 1
    start = vocab_size

    for name, size in config["unit_categorical"].items():
        offsets[f"unit:{name}"] = {"offset": start, "size": int(size), "type": "categorical"}
        start += int(size)

    for name, spec in config["unit_continuous"].items():
        offsets[f"unit:{name}"] = {
            "offset": start,
            "size": int(spec["bins"]),
            "type": "continuous",
            **spec,
        }
        start += int(spec["bins"])

    for name, size in config["zone_categorical"].items():
        offsets[f"zone:{name}"] = {"offset": start, "size": int(size), "type": "categorical"}
        start += int(size)

    for name, spec in config["zone_continuous"].items():
        offsets[f"zone:{name}"] = {
            "offset": start,
            "size": int(spec["bins"]),
            "type": "continuous",
            **spec,
        }
        start += int(spec["bins"])

    offsets["action"] = {
        "offset": start,
        "size": int(config["action"]["num_actions"]),
        "type": "categorical",
    }
    start += int(config["action"]["num_actions"])

    offsets["reward"] = {
        "offset": start,
        "size": int(config["reward"]["bins"]),
        "type": "continuous",
        **config["reward"],
    }
    start += int(config["reward"]["bins"])

    return offsets, start


def quantize_continuous(x, spec):
    x = np.nan_to_num(x, nan=0.0, posinf=spec["max"], neginf=spec["min"])
    x = np.clip(x, spec["min"], spec["max"])
    bins = int(spec["bins"])
    denom = max(float(spec["max"] - spec["min"]), 1e-8)
    q = np.floor((x - spec["min"]) / denom * bins).astype(np.int32)
    q = np.clip(q, 0, bins - 1)
    return q


def tokenize_unit_features(features, mask, config, offsets):
    # features: [S,T,U,F]
    S, T, U, F = features.shape
    token_ids = np.full((S, T, U, F), config["special_tokens"]["PAD"], dtype=np.int32)

    name_to_idx = {name: i for i, name in enumerate(UNIT_FEATURE_NAMES)}

    for name in UNIT_FEATURE_NAMES:
        idx = name_to_idx[name]
        key = f"unit:{name}"

        if key not in offsets:
            # Unknown / not tokenized field.
            token_ids[..., idx] = config["special_tokens"]["MASK"]
            continue

        spec = offsets[key]
        x = features[..., idx]

        if spec["type"] == "categorical":
            q = np.rint(np.nan_to_num(x, nan=0.0)).astype(np.int32)
            q = np.clip(q, 0, spec["size"] - 1)
        else:
            q = quantize_continuous(x, spec)

        token_ids[..., idx] = spec["offset"] + q

    # Invisible slots use UNSEEN token for all fields.
    token_ids = np.where(mask[..., None], token_ids, config["special_tokens"]["UNSEEN"])

    # Visible but dead units: keep features, but alive field itself records 0.
    return token_ids.astype(np.int32)


def tokenize_zone_features(features, mask, config, offsets):
    # features: [S,T,Z,Fz]
    S, T, Z, F = features.shape
    token_ids = np.full((S, T, Z, F), config["special_tokens"]["PAD"], dtype=np.int32)

    name_to_idx = {name: i for i, name in enumerate(ZONE_FEATURE_NAMES)}

    for name in ZONE_FEATURE_NAMES:
        idx = name_to_idx[name]
        key = f"zone:{name}"

        if key not in offsets:
            token_ids[..., idx] = config["special_tokens"]["MASK"]
            continue

        spec = offsets[key]
        x = features[..., idx]

        if spec["type"] == "categorical":
            q = np.rint(np.nan_to_num(x, nan=0.0)).astype(np.int32)
            q = np.clip(q, 0, spec["size"] - 1)
        else:
            q = quantize_continuous(x, spec)

        token_ids[..., idx] = spec["offset"] + q

    token_ids = np.where(mask[..., None], token_ids, config["special_tokens"]["EMPTY_ZONE"])
    return token_ids.astype(np.int32)


def tokenize_actions(actions, offsets):
    spec = offsets["action"]
    q = np.clip(actions.astype(np.int32), 0, spec["size"] - 1)
    return (spec["offset"] + q).astype(np.int32)


def tokenize_rewards(rewards, offsets):
    spec = offsets["reward"]
    q = quantize_continuous(rewards, spec)
    return (spec["offset"] + q).astype(np.int32)


def load_metadata(d):
    if "metadata" not in d.files:
        return {}
    try:
        return json.loads(str(d["metadata"]))
    except Exception:
        return {}


def convert_one(src: Path, dst: Path, config_path: Path):
    config = load_json(config_path)
    offsets, vocab_size = build_vocab(config)

    d = np.load(src, allow_pickle=True)

    fov_unit_features = d["fov_unit_features"]
    fov_unit_mask = d["fov_unit_mask"]
    fov_zone_features = d["fov_zone_features"]
    fov_zone_mask = d["fov_zone_mask"]
    ego_actions = d["ego_actions"]
    ego_rewards = d["ego_rewards"]

    unit_token_ids = tokenize_unit_features(
        fov_unit_features,
        fov_unit_mask,
        config,
        offsets,
    )
    zone_token_ids = tokenize_zone_features(
        fov_zone_features,
        fov_zone_mask,
        config,
        offsets,
    )
    action_token_ids = tokenize_actions(ego_actions, offsets)
    reward_token_ids = tokenize_rewards(ego_rewards, offsets)

    metadata = load_metadata(d)
    metadata.update({
        "source_file": str(src),
        "format": "tokenized-fov-agent-centered-sequence-v0",
        "tokenizer_config": str(config_path),
        "vocab_size": vocab_size,
        "vocab_offsets": offsets,
        "unit_token_ids_shape": list(unit_token_ids.shape),
        "zone_token_ids_shape": list(zone_token_ids.shape),
        "action_token_ids_shape": list(action_token_ids.shape),
        "reward_token_ids_shape": list(reward_token_ids.shape),
        "note": "This is field-wise tokenization. It has not yet flattened fields into final OmniRL sequence format.",
    })

    dst.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        dst,
        unit_token_ids=unit_token_ids,
        zone_token_ids=zone_token_ids,
        action_token_ids=action_token_ids,
        reward_token_ids=reward_token_ids,
        fov_unit_mask=fov_unit_mask,
        fov_zone_mask=fov_zone_mask,
        ego_actions=ego_actions,
        ego_rewards=ego_rewards,
        ego_done_agents=d["ego_done_agents"],
        ego_done_all=d["ego_done_all"],
        slot_unit_indices=d["slot_unit_indices"],
        ego_unit_indices=d["ego_unit_indices"],
        metadata=json.dumps(metadata),
    )

    print("saved:", dst)
    print("  unit_token_ids:", unit_token_ids.shape)
    print("  zone_token_ids:", zone_token_ids.shape)
    print("  action_token_ids:", action_token_ids.shape)
    print("  reward_token_ids:", reward_token_ids.shape)
    print("  vocab_size:", vocab_size)
    print("  token range:", int(unit_token_ids.min()), int(max(
        unit_token_ids.max(),
        zone_token_ids.max(),
        action_token_ids.max(),
        reward_token_ids.max(),
    )))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/tabx_fov_tokenizer_v0.json")
    args = ap.parse_args()

    convert_one(
        src=Path(args.input),
        dst=Path(args.output),
        config_path=Path(args.config),
    )


if __name__ == "__main__":
    main()
