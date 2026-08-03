"""Weighted behavior-policy mixture with switch cooldown."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class BehaviorSpec:
    """One entry in the behavior mixture."""

    policy_id: int
    name: str
    kind: str  # heuristic | oracle | oracle_eps
    weight: float
    heuristic: str | None = None
    epsilon: float | None = None


# Ordered by quality so default ``policy_id == policy_tag`` (0…7; 8=mask at convert).
DEFAULT_BEHAVIOR_MIX: tuple[BehaviorSpec, ...] = (
    BehaviorSpec(0, "heuristic_random", "heuristic", 0.12, heuristic="random"),
    BehaviorSpec(1, "heuristic_novice", "heuristic", 0.14, heuristic="novice"),
    BehaviorSpec(2, "heuristic_medium_eps0.5", "heuristic", 0.10, heuristic="medium", epsilon=0.5),
    BehaviorSpec(3, "heuristic_medium", "heuristic", 0.16, heuristic="medium"),
    BehaviorSpec(4, "heuristic_advanced", "heuristic", 0.12, heuristic="advanced"),
    BehaviorSpec(5, "oracle_eps0.3", "oracle_eps", 0.14, epsilon=0.3),
    BehaviorSpec(6, "oracle_eps0.1", "oracle_eps", 0.12, epsilon=0.1),
    BehaviorSpec(7, "oracle_pure", "oracle", 0.10, epsilon=0.0),
)

# Quality / mask index written to ``policy_tag.npy``.
POLICY_TAG_MASK = 8

# tag → short name (includes conversion-time mask).
POLICY_TAG_LEGEND: dict[int, str] = {
    0: "heuristic_random",
    1: "heuristic_novice",
    2: "heuristic_medium_noisy",
    3: "heuristic_medium",
    4: "heuristic_advanced",
    5: "oracle_eps_high",
    6: "oracle_eps_low",
    7: "oracle_pure",
    8: "mask",
}

# Default mix: policy_id → (name, policy_tag). Tag 8 is only set at agent-centric convert.
DEFAULT_POLICY_ID_MAPPING: tuple[tuple[int, str, int], ...] = (
    (0, "heuristic_random", 0),
    (1, "heuristic_novice", 1),
    (2, "heuristic_medium_eps0.5", 2),
    (3, "heuristic_medium", 3),
    (4, "heuristic_advanced", 4),
    (5, "oracle_eps0.3", 5),
    (6, "oracle_eps0.1", 6),
    (7, "oracle_pure", 7),
)


def policy_tag_legend_json() -> dict[str, str]:
    """String-keyed legend for ``meta.json``."""

    return {str(k): v for k, v in sorted(POLICY_TAG_LEGEND.items())}


def policy_id_mapping_json(mix: Sequence[BehaviorSpec] | None = None) -> list[dict[str, Any]]:
    """Full ``policy_id → name → tag`` rows for meta / docs."""

    if mix is None:
        mix = DEFAULT_BEHAVIOR_MIX
    rows = [
        {
            "policy_id": int(spec.policy_id),
            "name": str(spec.name),
            "policy_tag": int(policy_tag_for_spec(spec)),
        }
        for spec in mix
    ]
    rows.append(
        {
            "policy_id": None,
            "name": "mask",
            "policy_tag": int(POLICY_TAG_MASK),
            "note": "set on agent-centric convert when policy_mask==1",
        }
    )
    return rows


def policy_tag_for_spec(spec: BehaviorSpec) -> int:
    """Quality rank for dump/training: ``random=0`` … pure ``oracle=7``; ``8=mask``.

    Ladder:
      0 random, 1 novice, 2 medium+ε, 3 medium, 4 advanced,
      5 oracle_eps(~0.3), 6 oracle_eps(~0.1), 7 oracle, 8 mask
      (mask is applied in agent-centric conversion, not at env dump).

    Default mix is quality-ordered so ``policy_id == policy_tag``::

      0 heuristic_random
      1 heuristic_novice
      2 heuristic_medium_eps0.5
      3 heuristic_medium
      4 heuristic_advanced
      5 oracle_eps0.3
      6 oracle_eps0.1
      7 oracle_pure
      (convert) mask → 8
    """

    if spec.kind == "oracle":
        return 7
    if spec.kind == "oracle_eps":
        eps = float(spec.epsilon or 0.0)
        if eps <= 0.15:
            return 6
        return 5
    heuristic = str(spec.heuristic or "medium")
    eps = float(spec.epsilon or 0.0)
    if heuristic == "random":
        return 0
    if heuristic == "novice":
        return 1
    if heuristic == "medium":
        return 2 if eps >= 0.25 else 3
    if heuristic == "advanced":
        return 4
    if heuristic == "expert":
        return 5
    return 3


def policy_tag_table(mix: Sequence[BehaviorSpec]) -> list[int]:
    """Per-mix-index tags (same order as ``mix`` / ``lax.switch``)."""

    return [policy_tag_for_spec(spec) for spec in mix]


def policy_id_to_tag(mix: Sequence[BehaviorSpec]) -> dict[int, int]:
    """Map ``BehaviorSpec.policy_id`` → quality tag (last wins on duplicates)."""

    return {int(spec.policy_id): policy_tag_for_spec(spec) for spec in mix}


def behavior_mix_from_config(entries: Sequence[dict[str, Any]] | None) -> tuple[BehaviorSpec, ...]:
    """Build a behavior mix from YAML/JSON dict entries."""

    if not entries:
        return DEFAULT_BEHAVIOR_MIX
    mix: list[BehaviorSpec] = []
    for index, entry in enumerate(entries):
        kind = str(entry["kind"])
        if kind not in {"heuristic", "oracle", "oracle_eps"}:
            raise ValueError(f"Unknown behavior kind: {kind!r}")
        mix.append(
            BehaviorSpec(
                policy_id=int(entry.get("policy_id", index)),
                name=str(entry.get("name", f"policy_{index}")),
                kind=kind,
                weight=float(entry["weight"]),
                heuristic=entry.get("heuristic"),
                epsilon=None if entry.get("epsilon") is None else float(entry["epsilon"]),
            )
        )
    if not mix:
        raise ValueError("behavior_mix must not be empty")
    if sum(spec.weight for spec in mix) <= 0:
        raise ValueError("behavior mix weights must be positive")
    return tuple(mix)


@dataclass
class BehaviorSwitcher:
    """Sample a shared behavior policy for all allies; switch with cooldown."""

    mix: tuple[BehaviorSpec, ...] = DEFAULT_BEHAVIOR_MIX
    min_behavior_steps: int = 64
    behavior_switch_prob: float = 0.2
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self) -> None:
        weights = np.asarray([spec.weight for spec in self.mix], dtype=np.float64)
        if weights.sum() <= 0:
            raise ValueError("behavior mix weights must be positive")
        self._cdf = np.cumsum(weights / weights.sum())
        self.current: BehaviorSpec = self.sample()
        self.steps_since_switch: int = 0

    def sample(self) -> BehaviorSpec:
        u = float(self.rng.random())
        index = int(np.searchsorted(self._cdf, u, side="right"))
        index = min(index, len(self.mix) - 1)
        return self.mix[index]

    def maybe_switch(self) -> BehaviorSpec:
        self.steps_since_switch += 1
        if self.steps_since_switch < self.min_behavior_steps:
            return self.current
        if float(self.rng.random()) < self.behavior_switch_prob:
            self.current = self.sample()
            self.steps_since_switch = 0
        return self.current

    def force_sample(self) -> BehaviorSpec:
        self.current = self.sample()
        self.steps_since_switch = 0
        return self.current


def mix_cdf_jax(mix: Sequence[BehaviorSpec]) -> jax.Array:
    """Normalized CDF over mix weights (same order as ``mix``)."""

    weights = jnp.asarray([float(spec.weight) for spec in mix], dtype=jnp.float32)
    weights = weights / jnp.maximum(weights.sum(), jnp.float32(1e-12))
    return jnp.cumsum(weights)


def sample_policy_index(key: jax.Array, cdf: jax.Array) -> jax.Array:
    """Sample a mix index with the same rule as ``BehaviorSwitcher.sample``."""

    u = jax.random.uniform(key, (), dtype=jnp.float32)
    index = jnp.searchsorted(cdf, u, side="right")
    return jnp.minimum(index, cdf.shape[0] - 1).astype(jnp.int32)


def maybe_switch_policy_id(
    key: jax.Array,
    *,
    policy_id: jax.Array,
    steps_since_switch: jax.Array,
    cdf: jax.Array,
    min_behavior_steps: int,
    behavior_switch_prob: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JAX mid-episode switcher.

    Returns ``(key, policy_id, steps_since_switch)`` after one step of cooldown logic.
    """

    key, gate_key, sample_key = jax.random.split(key, 3)
    steps = steps_since_switch + jnp.int32(1)
    can_switch = steps >= jnp.int32(min_behavior_steps)
    do_switch = can_switch & (
        jax.random.uniform(gate_key, ()) < jnp.float32(behavior_switch_prob)
    )
    sampled = sample_policy_index(sample_key, cdf)
    new_id = jnp.where(do_switch, sampled, policy_id).astype(jnp.int32)
    new_steps = jnp.where(do_switch, jnp.int32(0), steps).astype(jnp.int32)
    return key, new_id, new_steps
