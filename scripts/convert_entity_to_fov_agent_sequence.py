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


def load_metadata(d):
    if "metadata" not in d.files:
        return {}
    try:
        return json.loads(str(d["metadata"]))
    except Exception:
        try:
            return json.loads(d["metadata"].item())
        except Exception:
            return {"raw_metadata": str(d["metadata"])}


def build_slot_order(n_agents: int, n_units: int) -> np.ndarray:
    """
    For each ego agent, use fixed FOV slots:
      slot 0: self
      slot 1..N-1: all other units in global unit order, excluding self

    Shape:
      [A, N]
    """
    orders = []
    for ego in range(n_agents):
        orders.append([ego] + [j for j in range(n_units) if j != ego])
    return np.asarray(orders, dtype=np.int32)


def convert_units(d):
    pos = d["unit_positions"]                 # [T,E,N,2]
    health = d["unit_healths"]                # [T,E,N]
    max_health = d["unit_max_healths"]        # [T,E,N]
    alive = d["unit_is_alives"].astype(bool)  # [T,E,N]
    team = d["unit_teams"]                    # [T,E,N]

    speed = d["unit_speeds"]
    attack_range = d["unit_attack_ranges"]
    attack_damage = d["unit_attack_damages"]
    cooldown = d["unit_cooldowns"]
    body_radius = d["unit_body_radiuss"]

    visible = d["visible_matrix"].astype(bool)          # [T,E,N,N]
    attackable = d["attackable_matrix"].astype(bool)    # [T,E,N,N]

    actions = d["actions"]                  # [T,E,A]
    T, E, N, _ = pos.shape
    A = actions.shape[2]

    ego_pos = pos[:, :, :A, :]              # [T,E,A,2]
    rel = pos[:, :, None, :, :] - ego_pos[:, :, :, None, :]  # [T,E,A,N,2]
    dist = np.linalg.norm(rel, axis=-1)     # [T,E,A,N]

    unit_idx = np.arange(N)[None, None, None, :]
    ego_idx = np.arange(A)[None, None, :, None]

    is_self = (unit_idx == ego_idx)
    is_ally = (unit_idx < A) & (~is_self)
    is_enemy = unit_idx >= A

    ego_visible = visible[:, :, :A, :]      # [T,E,A,N]
    ego_attackable = attackable[:, :, :A, :]

    # Self should always be visible to itself.
    ego_visible = ego_visible | np.broadcast_to(is_self, (T, E, A, N))

    health_ratio = health / np.maximum(max_health, 1e-6)

    def expand_unit(x):
        return np.broadcast_to(x[:, :, None, :], (T, E, A, N))

    all_feat = np.stack(
        [
            np.broadcast_to(is_self, (T, E, A, N)).astype(np.float32),
            np.broadcast_to(is_ally, (T, E, A, N)).astype(np.float32),
            np.broadcast_to(is_enemy, (T, E, A, N)).astype(np.float32),
            rel[..., 0],
            rel[..., 1],
            dist,
            expand_unit(health_ratio),
            expand_unit(alive.astype(np.float32)),
            expand_unit(speed),
            expand_unit(attack_range),
            expand_unit(attack_damage),
            expand_unit(cooldown),
            expand_unit(body_radius),
            ego_visible.astype(np.float32),
            ego_attackable.astype(np.float32),
        ],
        axis=-1,
    )  # [T,E,A,N,F]

    # Mask means this slot is observable under FOV.
    # Invisible other units keep their fixed slot but features are zeroed.
    all_mask = ego_visible

    slot_order = build_slot_order(A, N)   # [A,N], self + all others
    S = slot_order.shape[1]
    F = all_feat.shape[-1]

    idx = np.broadcast_to(slot_order[None, None, :, :], (T, E, A, S))

    feat_idx = np.broadcast_to(idx[..., None], (T, E, A, S, F))
    selected_feat = np.take_along_axis(all_feat, feat_idx, axis=3)
    selected_mask = np.take_along_axis(all_mask, idx, axis=3)

    selected_feat = np.where(selected_mask[..., None], selected_feat, 0.0)
    selected_feat = np.nan_to_num(selected_feat, nan=0.0, posinf=0.0, neginf=0.0)

    # [T,E,A,S,F] -> [E*A,T,S,F]
    fov_unit_features = selected_feat.transpose(1, 2, 0, 3, 4).reshape(E * A, T, S, F)
    fov_unit_mask = selected_mask.transpose(1, 2, 0, 3).reshape(E * A, T, S)

    # [E*A,S], sequence order is episode-major then agent.
    slot_unit_indices = np.tile(slot_order[None, :, :], (E, 1, 1)).reshape(E * A, S)
    ego_unit_indices = np.tile(np.arange(A, dtype=np.int32), E)

    return (
        fov_unit_features.astype(np.float32),
        fov_unit_mask.astype(bool),
        slot_unit_indices.astype(np.int32),
        ego_unit_indices.astype(np.int32),
    )


def convert_zones(d):
    pos = d["unit_positions"]          # [T,E,N,2]
    zone_pos = d["zone_positions"]     # [T,E,Z,2]
    zone_types = d["zone_types"]       # [T,E,Z]
    zone_axes = d["zone_axes"]         # [T,E,Z,2]
    zone_effect = d["zone_effect_values"]

    actions = d["actions"]
    T, E, N, _ = pos.shape
    A = actions.shape[2]
    Z = zone_pos.shape[2]

    ego_pos = pos[:, :, :A, :]         # [T,E,A,2]

    rel = zone_pos[:, :, None, :, :] - ego_pos[:, :, :, None, :]  # [T,E,A,Z,2]
    dist = np.linalg.norm(rel, axis=-1)

    zone_mask = zone_types[:, :, None, :] != 0
    zone_mask = np.broadcast_to(zone_mask, (T, E, A, Z))

    f = np.stack(
        [
            np.broadcast_to(zone_types[:, :, None, :], (T, E, A, Z)).astype(np.float32),
            rel[..., 0],
            rel[..., 1],
            dist,
            np.broadcast_to(zone_axes[:, :, None, :, 0], (T, E, A, Z)),
            np.broadcast_to(zone_axes[:, :, None, :, 1], (T, E, A, Z)),
            np.broadcast_to(zone_effect[:, :, None, :], (T, E, A, Z)),
        ],
        axis=-1,
    )  # [T,E,A,Z,Fz]

    f = np.where(zone_mask[..., None], f, 0.0)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    fov_zone_features = f.transpose(1, 2, 0, 3, 4).reshape(E * A, T, Z, f.shape[-1])
    fov_zone_mask = zone_mask.transpose(1, 2, 0, 3).reshape(E * A, T, Z)

    return fov_zone_features.astype(np.float32), fov_zone_mask.astype(bool)


def convert_one(src: Path, dst: Path):
    d = np.load(src, allow_pickle=True)

    agent_obs = d["agent_obs"]                 # [T,E,A,obs_dim]
    available_actions = d["available_actions"] # [T,E,A,action_dim]
    actions = d["actions"]                     # [T,E,A]
    rewards = d["rewards"]                     # [T,E,A]
    done_agents = d["done_agents"]             # [T,E,A]
    done_all = d["done_all"]                   # [T,E]

    T, E, A, obs_dim = agent_obs.shape
    num_seq = E * A

    (
        fov_unit_features,
        fov_unit_mask,
        slot_unit_indices,
        ego_unit_indices,
    ) = convert_units(d)

    fov_zone_features, fov_zone_mask = convert_zones(d)

    ego_obs = agent_obs.transpose(1, 2, 0, 3).reshape(num_seq, T, obs_dim)
    ego_available_actions = available_actions.transpose(1, 2, 0, 3).reshape(
        num_seq, T, available_actions.shape[-1]
    )
    ego_actions = actions.transpose(1, 2, 0).reshape(num_seq, T)
    ego_rewards = rewards.transpose(1, 2, 0).reshape(num_seq, T)
    ego_done_agents = done_agents.transpose(1, 2, 0).reshape(num_seq, T)

    # done_all is episode-level, repeat it for each agent sequence.
    ego_done_all = np.repeat(done_all.transpose(1, 0)[:, None, :], A, axis=1).reshape(
        num_seq, T
    )

    metadata = load_metadata(d)
    metadata.update(
        {
            "source_file": str(src),
            "format": "fov-agent-centered sequence",
            "num_sequences": num_seq,
            "T": T,
            "episodes": E,
            "agents": A,
            "units_total": d["unit_positions"].shape[2],
            "zones_total": d["zone_positions"].shape[2],
            "unit_slot_rule": "slot0=self, remaining slots=all other units in global order excluding self",
            "visibility_rule": "features are kept only when visible_matrix[ego, unit] is true; invisible slots are zeroed",
            "unit_feature_names": UNIT_FEATURE_NAMES,
            "zone_feature_names": ZONE_FEATURE_NAMES,
            "note": "This converter follows TABX FOV visibility masks, not KNN.",
        }
    )

    dst.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        dst,
        fov_unit_features=fov_unit_features,
        fov_unit_mask=fov_unit_mask,
        fov_zone_features=fov_zone_features,
        fov_zone_mask=fov_zone_mask,
        slot_unit_indices=slot_unit_indices,
        ego_unit_indices=ego_unit_indices,
        ego_obs=ego_obs,
        ego_available_actions=ego_available_actions,
        ego_actions=ego_actions,
        ego_rewards=ego_rewards,
        ego_done_agents=ego_done_agents,
        ego_done_all=ego_done_all,
        metadata=json.dumps(metadata),
    )

    print("saved:", dst)
    print("  fov_unit_features:", fov_unit_features.shape)
    print("  fov_unit_mask:", fov_unit_mask.shape)
    print("  fov_zone_features:", fov_zone_features.shape)
    print("  fov_zone_mask:", fov_zone_mask.shape)
    print("  ego_obs:", ego_obs.shape)
    print("  ego_actions:", ego_actions.shape)
    print("  ego_rewards:", ego_rewards.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    convert_one(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
