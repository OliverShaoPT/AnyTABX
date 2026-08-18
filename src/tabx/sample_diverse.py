"""CPU-parallel sampler for balanced tasks in two sparse t-SNE directions.

The two proposal profiles come from the current environment-descriptor t-SNE
audit:

``bottom_left``
    Large teams on large maps, with two to four relational zones and mostly
    far, line/dispersed engagements.

``top_right``
    Small teams on ordinary/small maps, with zero or one zone and mostly
    ambush/face-off/encircle engagements.

These names describe proposal directions in the current fixed-seed plot; they
are not treated as invertible t-SNE coordinates.  Every accepted task passes
the normal TABX validator, an inexpensive expert-vs-expert screening stage,
and a final expert-vs-expert evaluation whose observed ally win rate is in the
hard-coded interval [0.4, 0.6].

The collector uses CPU worker processes with asynchronous refill.  Each worker
is restricted to one CPU thread so JAX/XLA instances do not oversubscribe the
host.

Example::

    python -m src.tabx.sample_diverse \
        --n-tasks 40 \
        --cpu-cores 8 \
        --output outputs/diverse_tasks.json
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import random
import secrets
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Configure the dedicated CPU sampler before importing helpers that import JAX.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")

import numpy as np
import tyro

from src.tabx import sample_task as base
from src.tabx.task_generators import generate_programmatic_task


CORNER_PROFILES = ("bottom_left", "top_right")

BOTTOM_LEFT_COMPOSITIONS = (
    "ranged_fort",
    "frontline_backline",
    "elite_swarm",
)
BOTTOM_LEFT_LAYOUTS = (
    "crossfire",
    "narrow_depth",
    "split_force",
)
BOTTOM_LEFT_DISTANCES = ("far", "far", "medium")
BOTTOM_LEFT_SPREADS = ("line", "line", "dispersed")
BOTTOM_LEFT_ZONES = (
    "flank_bush",
    "retreat_swamps",
    "channel_split",
    "mixed_overlap",
)

TOP_RIGHT_COMPOSITIONS = (
    "assassin_fragile",
    "elite_swarm",
    "mobility_range",
)
TOP_RIGHT_LAYOUTS = (
    "ambush",
    "face_off",
    "encircle",
)
TOP_RIGHT_DISTANCES = ("close", "medium")
TOP_RIGHT_SPREADS = ("compact", "dispersed")
TOP_RIGHT_ZONES = (
    "void",
    "void",
    "void",
    "engagement_lava",
    "center_swamp",
    "asymmetric_cover",
)

MATCH_MODES = ("mirror", "mirror", "unit_swap", "cost_match")
ZONE_INTENSITIES = ("low", "medium", "high")
ZONE_RELATIONS = ("separated", "tangent", "overlap")

_CPU_WORKER_ID = -1


@dataclass(frozen=True)
class DiverseConfig(base.SampleConfig):
    """Configuration for balanced, corner-directed CPU collection."""

    output: str = "outputs/diverse_tasks.json"
    n_tasks: int = 40
    cpu_cores: int = 4
    max_n_ally: int = 10
    max_n_enemy: int = 10
    max_n_zone: int = 4
    programmatic_ratio: float = 1.0
    include_challenges: bool = False

    # The final gate is deliberately immutable through validation.
    filter_by_win_rate: bool = True
    win_rate_min: float = 0.4
    win_rate_max: float = 0.6
    filter_num_seeds: int = 64
    filter_epsilon: float = 0.05
    filter_max_episode_steps: int = 512
    filter_reject_all_truncated: bool = True
    max_truncation_rate: float = 0.20
    max_no_interaction_rate: float = 0.0

    # A permissive first stage avoids spending 64 rollouts on obvious misses.
    screening_num_seeds: int = 16
    screening_win_rate_min: float = 0.25
    screening_win_rate_max: float = 0.75

    # Only symmetric global stat scaling is permitted.  Diversity comes from
    # roster, geometry, terrain, and map size rather than an A/B stat handicap.
    stat_scales: tuple[float, ...] = (1.0,)
    couple_stat_scales: bool = True
    zone_strength_scales: tuple[float, ...] = (0.90, 1.00, 1.10)

    corner_profiles: tuple[str, ...] = CORNER_PROFILES
    generation_attempts: int = 64

    # Directional gates inferred from the fixed-seed t-SNE audit.
    bottom_left_min_total_units: int = 12
    bottom_left_min_units_per_team: int = 5
    bottom_left_min_zones: int = 2
    bottom_left_max_zones: int = 4
    bottom_left_map_scale_min: float = 1.05
    bottom_left_map_scale_max: float = 1.20

    top_right_max_total_units: int = 10
    top_right_max_units_per_team: int = 5
    top_right_min_zones: int = 0
    top_right_max_zones: int = 1
    top_right_map_scale_min: float = 0.85
    top_right_map_scale_max: float = 1.10

    # CPU scheduling and online diversity caps.
    generation_seed: int | None = None
    candidate_budget: int = 0
    candidate_budget_multiplier: int = 80
    max_in_flight: int = 0
    collection_time_budget_seconds: float = 0.0
    max_corner_cell_tasks: int = 3
    max_exact_roster_tasks: int = 2
    save_partial: bool = True


@dataclass(frozen=True)
class _CornerJob:
    job_id: int
    profile: str
    generation_seed: int
    config: DiverseConfig


@dataclass
class _CornerResult:
    job_id: int
    worker_id: int
    worker_pid: int
    profile: str
    task: dict[str, Any] | None
    digest: str | None
    reason: str


def _derive_seed(global_seed: int, job_id: int, phase: str) -> int:
    """Derive stable, disjoint generation/evaluation seeds."""

    import hashlib

    payload = f"sample-diverse:{global_seed}:{job_id}:{phase}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _continuous_scale(
    rng: random.Random,
    lower: float,
    upper: float,
) -> float:
    return float(rng.uniform(lower, upper))


def _team_sizes(task: dict[str, Any]) -> tuple[int, int]:
    teams = np.asarray(task["scenario"]["teams"], dtype=int).reshape(-1)
    return (
        int(np.count_nonzero(teams == 0)),
        int(np.count_nonzero(teams == 1)),
    )


def _corner_actuals(task: dict[str, Any]) -> dict[str, Any]:
    """Return interpretable values used by proposal gates and reports."""

    scenario = task["scenario"]
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    unit_ids = np.asarray(scenario["unit_ids"], dtype=int).reshape(-1)
    ally_count, enemy_count = _team_sizes(task)
    map_scale = min(
        float(task["grid_info"]["max_field_width"]) / 121.0,
        float(task["grid_info"]["max_field_height"]) / 78.0,
    )
    return {
        "team_sizes": [ally_count, enemy_count],
        "total_units": ally_count + enemy_count,
        "n_zone": int(task["zone_scenario"]["n_zone"]),
        "map_scale": map_scale,
        "ranged_ratio": float(np.mean(np.isin(unit_ids, (4, 5, 6)))),
        "healer_ratio": float(np.mean(np.isin(unit_ids, (7, 8)))),
        "ally_ranged_ratio": float(
            np.mean(np.isin(unit_ids[teams == 0], (4, 5, 6)))
        ),
        "enemy_ranged_ratio": float(
            np.mean(np.isin(unit_ids[teams == 1], (4, 5, 6)))
        ),
    }


def _matches_corner(
    actual: dict[str, Any],
    profile: str,
    config: DiverseConfig,
) -> bool:
    """Apply hard structural gates for one target direction."""

    ally_count, enemy_count = map(int, actual["team_sizes"])
    total_units = int(actual["total_units"])
    n_zone = int(actual["n_zone"])
    map_scale = float(actual["map_scale"])
    if profile == "bottom_left":
        return (
            total_units >= config.bottom_left_min_total_units
            and min(ally_count, enemy_count)
            >= config.bottom_left_min_units_per_team
            and config.bottom_left_min_zones
            <= n_zone
            <= config.bottom_left_max_zones
            and config.bottom_left_map_scale_min
            <= map_scale
            <= config.bottom_left_map_scale_max
        )
    if profile == "top_right":
        return (
            total_units <= config.top_right_max_total_units
            and max(ally_count, enemy_count)
            <= config.top_right_max_units_per_team
            and config.top_right_min_zones
            <= n_zone
            <= config.top_right_max_zones
            and config.top_right_map_scale_min
            <= map_scale
            <= config.top_right_map_scale_max
        )
    raise ValueError(f"Unknown corner profile {profile!r}.")


def _draw_recipe(
    rng: random.Random,
    profile: str,
    config: DiverseConfig,
) -> dict[str, Any]:
    """Draw one profile-specific programmatic generation recipe."""

    if profile == "bottom_left":
        return {
            "composition": rng.choice(BOTTOM_LEFT_COMPOSITIONS),
            "layout": rng.choice(BOTTOM_LEFT_LAYOUTS),
            "distance": rng.choice(BOTTOM_LEFT_DISTANCES),
            "spread": rng.choice(BOTTOM_LEFT_SPREADS),
            "zone": rng.choice(BOTTOM_LEFT_ZONES),
            "zone_intensity": rng.choice(ZONE_INTENSITIES),
            "zone_relation": rng.choice(ZONE_RELATIONS),
            "match_mode": rng.choice(MATCH_MODES),
            "map_scale": _continuous_scale(
                rng,
                config.bottom_left_map_scale_min,
                config.bottom_left_map_scale_max,
            ),
        }
    if profile == "top_right":
        return {
            "composition": rng.choice(TOP_RIGHT_COMPOSITIONS),
            "layout": rng.choice(TOP_RIGHT_LAYOUTS),
            "distance": rng.choice(TOP_RIGHT_DISTANCES),
            "spread": rng.choice(TOP_RIGHT_SPREADS),
            "zone": rng.choice(TOP_RIGHT_ZONES),
            "zone_intensity": rng.choice(ZONE_INTENSITIES),
            "zone_relation": rng.choice(ZONE_RELATIONS),
            "match_mode": rng.choice(MATCH_MODES),
            "map_scale": _continuous_scale(
                rng,
                config.top_right_map_scale_min,
                config.top_right_map_scale_max,
            ),
        }
    raise ValueError(f"Unknown corner profile {profile!r}.")


def _generate_corner_task(job: _CornerJob) -> dict[str, Any]:
    """Generate and statically validate one task in the requested direction."""

    config = job.config
    generation_seed = _derive_seed(
        job.generation_seed,
        job.job_id,
        "generate",
    )
    rng = random.Random(generation_seed)
    last_reason = "no_attempt"
    for attempt in range(config.generation_attempts):
        recipe = _draw_recipe(rng, job.profile, config)
        max_n_ally = config.max_n_ally
        max_n_enemy = config.max_n_enemy
        if job.profile == "top_right":
            max_n_ally = min(
                max_n_ally,
                config.top_right_max_units_per_team,
            )
            max_n_enemy = min(
                max_n_enemy,
                config.top_right_max_units_per_team,
            )
        try:
            task = generate_programmatic_task(
                rng,
                composition=recipe["composition"],
                layout=recipe["layout"],
                distance=recipe["distance"],
                spread=recipe["spread"],
                zone=recipe["zone"],
                zone_intensity=recipe["zone_intensity"],
                zone_relation=recipe["zone_relation"],
                map_scale=recipe["map_scale"],
                max_n_ally=max_n_ally,
                max_n_enemy=max_n_enemy,
                match_mode=recipe["match_mode"],
                price_rel_tol=config.composition_price_rel_tol,
                effective_rel_tol=config.composition_effective_rel_tol,
                health_frac_buckets=config.health_frac_buckets,
                health_frac_min=config.health_frac_min,
                health_frac_max=config.health_frac_max,
            )
            transform = rng.choice(config.transforms)
            symmetric_stat_scale = rng.choice(config.stat_scales)
            zone_strength_scale = rng.choice(config.zone_strength_scales)
            base._transform_task(task, transform)
            base._scale_team_stats(
                task,
                symmetric_stat_scale,
                symmetric_stat_scale,
            )
            base._scale_zone_strength(task, zone_strength_scale)
            base.validate_task(task)
        except ValueError as exc:
            last_reason = f"generation_or_validation:{exc}"
            continue

        actual = _corner_actuals(task)
        if not _matches_corner(actual, job.profile, config):
            last_reason = "directional_gate"
            continue
        metadata = task.setdefault("metadata", {})
        metadata.update(
            {
                "task_profile": "balanced_corner_diverse",
                "source": {
                    "kind": "programmatic",
                    "subtype": "tsne_corner_diverse",
                },
                "transform": transform,
                "symmetric_stat_scale": symmetric_stat_scale,
                "zone_strength_scale": zone_strength_scale,
                "diverse_sampling": {
                    "profile": job.profile,
                    "job_id": job.job_id,
                    "generation_seed": generation_seed,
                    "generation_attempt": attempt,
                    "recipe": recipe,
                    "actual": actual,
                },
            }
        )
        return task
    raise ValueError(
        f"no_candidate_matched_{job.profile}_after_"
        f"{config.generation_attempts}_attempts:{last_reason}"
    )


def _compact_evaluation(result: dict[str, Any], *, accepted: bool) -> dict[str, Any]:
    """Keep the task bank informative without copying unused rollout details."""

    return {
        "win_rate": result["win_rate"],
        "win_rate_ci95": result["win_rate_ci95"],
        "episode_length": result["episode_length"],
        "truncation_rate": result["truncation_rate"],
        "hp_margin": result["hp_margin"],
        "quality_flags": result["quality_flags"],
        "n_rollouts": result["n_rollouts"],
        "accepted": accepted,
        "reject_reason": result.get("reject_reason"),
    }


def _quality_reason(
    result: dict[str, Any],
    config: DiverseConfig,
) -> str | None:
    truncation_rate = float(result["truncation_rate"])
    if truncation_rate > config.max_truncation_rate:
        return "truncation_rate"
    no_interaction_rate = float(
        result["quality_flags"].get("no_interaction_rate", 0.0)
    )
    if no_interaction_rate > config.max_no_interaction_rate:
        return "no_interaction_rate"
    return None


def _evaluate_corner_task(
    task: dict[str, Any],
    job: _CornerJob,
) -> tuple[bool, str]:
    """Run permissive screening and strict [0.4, 0.6] confirmation."""

    config = job.config
    screening_seed = _derive_seed(config.seed, job.job_id, "screen")
    screening_accepted, screening = base._passes_win_rate_filter(
        task,
        physics=config.physics,
        heuristic=config.heuristic,
        num_seeds=config.screening_num_seeds,
        seed=screening_seed,
        epsilon=config.filter_epsilon,
        max_episode_steps=config.filter_max_episode_steps,
        win_rate_min=config.screening_win_rate_min,
        win_rate_max=config.screening_win_rate_max,
        reject_all_truncated=config.filter_reject_all_truncated,
    )
    screening_quality_reason = _quality_reason(screening, config)
    if screening_quality_reason is not None:
        screening_accepted = False
        screening["reject_reason"] = screening_quality_reason
    task["metadata"]["screening_eval"] = _compact_evaluation(
        screening,
        accepted=screening_accepted,
    )
    if not screening_accepted:
        return False, f"screening:{screening.get('reject_reason') or 'win_rate'}"

    confirmation_seed = _derive_seed(config.seed, job.job_id, "confirm")
    accepted, confirmation = base._passes_win_rate_filter(
        task,
        physics=config.physics,
        heuristic=config.heuristic,
        num_seeds=config.filter_num_seeds,
        seed=confirmation_seed,
        epsilon=config.filter_epsilon,
        max_episode_steps=config.filter_max_episode_steps,
        win_rate_min=0.4,
        win_rate_max=0.6,
        reject_all_truncated=config.filter_reject_all_truncated,
    )
    quality_reason = _quality_reason(confirmation, config)
    if quality_reason is not None:
        accepted = False
        confirmation["reject_reason"] = quality_reason
    task["metadata"]["filter_eval"] = _compact_evaluation(
        confirmation,
        accepted=accepted,
    )
    task["metadata"]["evaluation_protocol"] = {
        "profile": "balanced_only",
        "policy": "expert_vs_expert_win_rate_0.4_0.6",
        "win_rate_band": [0.4, 0.6],
        "num_seeds": config.filter_num_seeds,
        "seed": confirmation_seed,
        "epsilon": config.filter_epsilon,
        "max_episode_steps": config.filter_max_episode_steps,
        "max_truncation_rate": config.max_truncation_rate,
        "max_no_interaction_rate": config.max_no_interaction_rate,
    }
    task["metadata"]["validation_status"] = (
        "accepted" if accepted else "rejected"
    )
    task["metadata"]["validation_gate"] = {
        "selected": accepted,
        "reason": (
            "accepted"
            if accepted
            else confirmation.get("reject_reason") or "win_rate_out_of_band"
        ),
        "policy": "expert_vs_expert_win_rate_0.4_0.6",
    }
    if not accepted:
        return False, (
            f"confirmation:"
            f"{confirmation.get('reject_reason') or 'win_rate_out_of_band'}"
        )
    return True, "accepted"


def _initialize_cpu_worker(counter: Any) -> None:
    global _CPU_WORKER_ID
    with counter.get_lock():
        _CPU_WORKER_ID = int(counter.value)
        counter.value += 1
    base.jax.config.update("jax_platform_name", "cpu")


def _sample_cpu_worker(job: _CornerJob) -> _CornerResult:
    try:
        task = _generate_corner_task(job)
        accepted, reason = _evaluate_corner_task(task, job)
        if not accepted:
            return _CornerResult(
                job_id=job.job_id,
                worker_id=max(_CPU_WORKER_ID, 0),
                worker_pid=os.getpid(),
                profile=job.profile,
                task=None,
                digest=None,
                reason=reason,
            )
        digest = base._task_hash(task)
        return _CornerResult(
            job_id=job.job_id,
            worker_id=max(_CPU_WORKER_ID, 0),
            worker_pid=os.getpid(),
            profile=job.profile,
            task=task,
            digest=digest,
            reason="accepted",
        )
    except (RuntimeError, ValueError) as exc:
        return _CornerResult(
            job_id=job.job_id,
            worker_id=max(_CPU_WORKER_ID, 0),
            worker_pid=os.getpid(),
            profile=job.profile,
            task=None,
            digest=None,
            reason=f"{type(exc).__name__}:{exc}",
        )


def _multiprocessing_context() -> multiprocessing.context.BaseContext:
    method = "spawn" if os.name == "nt" else "forkserver"
    try:
        return multiprocessing.get_context(method)
    except ValueError:
        return multiprocessing.get_context("spawn")


def _worker_count(config: DiverseConfig, candidate_budget: int) -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(config.cpu_cores, available, candidate_budget))


def _profile_targets(
    n_tasks: int,
    profiles: Sequence[str],
) -> dict[str, int]:
    base_count, remainder = divmod(n_tasks, len(profiles))
    return {
        profile: base_count + int(index < remainder)
        for index, profile in enumerate(profiles)
    }


def _roster_key(task: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    teams = np.asarray(task["scenario"]["teams"], dtype=int).reshape(-1)
    unit_ids = np.asarray(task["scenario"]["unit_ids"], dtype=int).reshape(-1)
    rosters = [
        tuple(sorted(unit_ids[teams == team].tolist()))
        for team in (0, 1)
    ]
    return tuple(sorted(rosters))  # type: ignore[return-value]


def _corner_cell(task: dict[str, Any]) -> tuple[Any, ...]:
    metadata = task["metadata"]
    sampling = metadata["diverse_sampling"]
    actual = sampling["actual"]
    return (
        sampling["profile"],
        tuple(sorted(map(int, actual["team_sizes"]))),
        int(actual["n_zone"]),
        metadata.get("composition_archetype"),
        metadata.get("layout_archetype"),
        metadata.get("distance_bucket"),
        metadata.get("spread_bucket"),
        metadata.get("zone_archetype"),
    )


def _validate_config(config: DiverseConfig) -> None:
    if config.n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")
    if config.cpu_cores <= 0:
        raise ValueError("cpu_cores must be positive.")
    if (
        not config.filter_by_win_rate
        or not math.isclose(config.win_rate_min, 0.4)
        or not math.isclose(config.win_rate_max, 0.6)
    ):
        raise ValueError("The final win-rate gate is fixed to [0.4, 0.6].")
    if config.programmatic_ratio != 1.0 or config.include_challenges:
        raise ValueError(
            "sample_diverse requires programmatic_ratio=1 and "
            "include_challenges=False."
        )
    if config.max_n_ally <= 0 or config.max_n_enemy <= 0:
        raise ValueError("max_n_ally and max_n_enemy must be positive.")
    if config.max_n_zone < 0:
        raise ValueError("max_n_zone must be non-negative.")
    if not config.transforms:
        raise ValueError("transforms must not be empty.")
    if not config.zone_strength_scales:
        raise ValueError("zone_strength_scales must not be empty.")
    if any(value <= 0.0 for value in config.zone_strength_scales):
        raise ValueError("zone_strength_scales must be positive.")
    if config.filter_num_seeds <= 0 or config.screening_num_seeds <= 0:
        raise ValueError("Evaluation seed counts must be positive.")
    if not 0.0 < config.filter_epsilon <= 1.0:
        raise ValueError("filter_epsilon must lie in (0, 1].")
    if not (
        0.0
        <= config.screening_win_rate_min
        <= 0.4
        <= 0.6
        <= config.screening_win_rate_max
        <= 1.0
    ):
        raise ValueError("The screening band must contain [0.4, 0.6].")
    if config.max_truncation_rate < 0.0:
        raise ValueError("max_truncation_rate must be non-negative.")
    if config.max_no_interaction_rate < 0.0:
        raise ValueError("max_no_interaction_rate must be non-negative.")
    if tuple(config.stat_scales) != (1.0,) or not config.couple_stat_scales:
        raise ValueError(
            "sample_diverse only permits coupled symmetric stat scale 1.0."
        )
    if not config.corner_profiles:
        raise ValueError("corner_profiles must not be empty.")
    unknown_profiles = set(config.corner_profiles).difference(CORNER_PROFILES)
    if unknown_profiles:
        raise ValueError(
            f"Unknown corner profiles: {sorted(unknown_profiles)}."
        )
    if len(set(config.corner_profiles)) != len(config.corner_profiles):
        raise ValueError("corner_profiles must not contain duplicates.")
    if config.generation_attempts <= 0:
        raise ValueError("generation_attempts must be positive.")
    if config.max_n_ally < config.bottom_left_min_units_per_team:
        raise ValueError("max_n_ally is too small for the bottom-left profile.")
    if config.max_n_enemy < config.bottom_left_min_units_per_team:
        raise ValueError("max_n_enemy is too small for the bottom-left profile.")
    if config.top_right_max_units_per_team <= 0:
        raise ValueError("top_right_max_units_per_team must be positive.")
    if config.top_right_max_units_per_team > min(
        config.max_n_ally,
        config.max_n_enemy,
    ):
        raise ValueError(
            "top_right_max_units_per_team exceeds a team capacity."
        )
    if config.max_n_zone < config.bottom_left_max_zones:
        raise ValueError("max_n_zone is too small for the bottom-left profile.")
    if not (
        2
        <= config.bottom_left_min_total_units
        <= config.max_n_ally + config.max_n_enemy
    ):
        raise ValueError("Invalid bottom-left total-unit threshold.")
    if not (
        2
        <= config.top_right_max_total_units
        <= config.max_n_ally + config.max_n_enemy
    ):
        raise ValueError("Invalid top-right total-unit threshold.")
    if not (
        0
        <= config.bottom_left_min_zones
        <= config.bottom_left_max_zones
        <= config.max_n_zone
    ):
        raise ValueError("Invalid bottom-left zone range.")
    if not (
        0
        <= config.top_right_min_zones
        <= config.top_right_max_zones
        <= config.max_n_zone
    ):
        raise ValueError("Invalid top-right zone range.")
    if not (
        0.0
        < config.bottom_left_map_scale_min
        <= config.bottom_left_map_scale_max
    ):
        raise ValueError("Invalid bottom-left map scale range.")
    if not (
        0.0
        < config.top_right_map_scale_min
        <= config.top_right_map_scale_max
    ):
        raise ValueError("Invalid top-right map scale range.")
    if config.candidate_budget_multiplier <= 0:
        raise ValueError("candidate_budget_multiplier must be positive.")
    if 0 < config.candidate_budget < config.n_tasks:
        raise ValueError("candidate_budget must be zero or at least n_tasks.")
    if config.max_in_flight < 0:
        raise ValueError("max_in_flight must be non-negative.")
    if config.collection_time_budget_seconds < 0.0:
        raise ValueError(
            "collection_time_budget_seconds must be non-negative."
        )
    if config.max_corner_cell_tasks <= 0:
        raise ValueError("max_corner_cell_tasks must be positive.")
    if config.max_exact_roster_tasks <= 0:
        raise ValueError("max_exact_roster_tasks must be positive.")


def _collection_report(
    tasks: Sequence[dict[str, Any]],
    *,
    config: DiverseConfig,
    generation_seed: int,
    candidate_budget: int,
    submitted_jobs: int,
    reject_counts: Counter[str],
    elapsed_seconds: float,
    worker_count: int,
    max_in_flight: int,
    start_method: str,
) -> dict[str, Any]:
    profile_counts = Counter(
        str(task["metadata"]["diverse_sampling"]["profile"])
        for task in tasks
    )
    cells = Counter(_corner_cell(task) for task in tasks)
    rosters = Counter(_roster_key(task) for task in tasks)
    actuals = [
        task["metadata"]["diverse_sampling"]["actual"]
        for task in tasks
    ]
    win_rates = [
        float(task["metadata"]["filter_eval"]["win_rate"])
        for task in tasks
    ]
    return {
        "strategy": "balanced_tsne_corner_direction_sampling",
        "task_profile": "balanced_corner_diverse",
        "win_rate_band": [0.4, 0.6],
        "complete": len(tasks) == config.n_tasks,
        "partial_result": len(tasks) != config.n_tasks,
        "requested_tasks": config.n_tasks,
        "saved_tasks": len(tasks),
        "task_shortfall": max(0, config.n_tasks - len(tasks)),
        "candidate_budget": candidate_budget,
        "submitted_jobs": submitted_jobs,
        "generation_seed": generation_seed,
        "evaluation_seed": config.seed,
        "elapsed_seconds": elapsed_seconds,
        "reject_counts": dict(reject_counts),
        "profile_targets": _profile_targets(
            config.n_tasks,
            config.corner_profiles,
        ),
        "profile_distribution": dict(profile_counts),
        "occupied_corner_cells": len(cells),
        "max_corner_cell_count": max(cells.values(), default=0),
        "unique_rosters": len(rosters),
        "max_exact_roster_count": max(rosters.values(), default=0),
        "actual_distribution": {
            "total_units": dict(
                Counter(int(item["total_units"]) for item in actuals)
            ),
            "zone_count": dict(
                Counter(int(item["n_zone"]) for item in actuals)
            ),
            "map_scale": {
                "min": min(
                    (float(item["map_scale"]) for item in actuals),
                    default=None,
                ),
                "mean": (
                    float(np.mean([item["map_scale"] for item in actuals]))
                    if actuals
                    else None
                ),
                "max": max(
                    (float(item["map_scale"]) for item in actuals),
                    default=None,
                ),
            },
        },
        "accepted_win_rate": {
            "min": min(win_rates, default=None),
            "mean": float(np.mean(win_rates)) if win_rates else None,
            "max": max(win_rates, default=None),
        },
        "parallel_sampling": {
            "execution_backend": "cpu",
            "workers": worker_count,
            "threads_per_worker": 1,
            "max_in_flight": max_in_flight,
            "scheduling": "asynchronous_refill",
            "multiprocessing_start_method": start_method,
            "commit_policy": "asynchronous_completion_order",
        },
    }


def sample_tasks_with_report(
    config: DiverseConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect balanced tasks from both target directions in CPU workers."""

    _validate_config(config)
    generation_seed = (
        secrets.randbits(32)
        if config.generation_seed is None
        else int(config.generation_seed)
    )
    candidate_budget = (
        config.candidate_budget
        if config.candidate_budget > 0
        else config.n_tasks * config.candidate_budget_multiplier
    )
    worker_count = _worker_count(config, candidate_budget)
    max_in_flight = max(
        worker_count,
        config.max_in_flight or 2 * worker_count,
    )
    profile_targets = _profile_targets(
        config.n_tasks,
        config.corner_profiles,
    )
    profile_counts: Counter[str] = Counter()
    cell_counts: Counter[tuple[Any, ...]] = Counter()
    roster_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    reject_counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    tasks: list[dict[str, Any]] = []
    next_job_id = 0
    started = time.monotonic()

    managed_environment = {
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "XLA_FLAGS": (
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1"
        ),
    }
    previous_environment = {
        name: os.environ.get(name)
        for name in managed_environment
    }
    os.environ.update(managed_environment)

    print(
        "Diverse corner CPU sampling: "
        f"workers={worker_count}, max_in_flight={max_in_flight}, "
        f"target={config.n_tasks}, candidate_budget={candidate_budget}, "
        f"profiles={profile_targets}, win_rate=[0.4, 0.6], "
        f"generation_seed={generation_seed}, evaluation_seed={config.seed}",
        flush=True,
    )

    def time_exhausted() -> bool:
        return bool(
            config.collection_time_budget_seconds > 0.0
            and time.monotonic() - started
            >= config.collection_time_budget_seconds
        )

    def make_job() -> _CornerJob:
        nonlocal next_job_id
        job_id = next_job_id
        next_job_id += 1
        profiles_with_remaining_quota = tuple(
            profile
            for profile in config.corner_profiles
            if profile_counts[profile] < profile_targets[profile]
        )
        if not profiles_with_remaining_quota:
            raise RuntimeError("No corner profile has remaining quota.")
        profile = profiles_with_remaining_quota[
            job_id % len(profiles_with_remaining_quota)
        ]
        return _CornerJob(
            job_id=job_id,
            profile=profile,
            generation_seed=generation_seed,
            config=config,
        )

    def commit(result: _CornerResult) -> None:
        if result.task is None or result.digest is None:
            reject_counts[result.reason] += 1
            return
        if profile_counts[result.profile] >= profile_targets[result.profile]:
            reject_counts["profile_quota_filled"] += 1
            return
        if result.digest in seen_hashes:
            reject_counts["exact_duplicate"] += 1
            return
        cell = _corner_cell(result.task)
        if cell_counts[cell] >= config.max_corner_cell_tasks:
            reject_counts["corner_cell_cap"] += 1
            return
        roster = _roster_key(result.task)
        if roster_counts[roster] >= config.max_exact_roster_tasks:
            reject_counts["roster_cap"] += 1
            return

        result.task["metadata"]["parallel_sampling"] = {
            "job_id": result.job_id,
            "worker_id": result.worker_id,
            "worker_pid": result.worker_pid,
            "commit_policy": "asynchronous_completion_order",
        }
        tasks.append(result.task)
        seen_hashes.add(result.digest)
        profile_counts[result.profile] += 1
        cell_counts[cell] += 1
        roster_counts[roster] += 1
        print(
            f"job={result.job_id + 1}/{candidate_budget} ACCEPT "
            f"tasks={len(tasks)}/{config.n_tasks} "
            f"profile={result.profile} worker={result.worker_id} "
            f"win_rate="
            f"{result.task['metadata']['filter_eval']['win_rate']:.3f}",
            flush=True,
        )

    executor: ProcessPoolExecutor | None = None
    start_method = "local"
    try:
        if worker_count == 1:
            while (
                next_job_id < candidate_budget
                and len(tasks) < config.n_tasks
                and not time_exhausted()
            ):
                commit(_sample_cpu_worker(make_job()))
        else:
            context = _multiprocessing_context()
            start_method = context.get_start_method()
            worker_counter = context.Value("i", 0)
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_initialize_cpu_worker,
                initargs=(worker_counter,),
            )
            pending: dict[Any, _CornerJob] = {}

            def refill() -> None:
                while (
                    len(pending) < max_in_flight
                    and next_job_id < candidate_budget
                    and len(tasks) < config.n_tasks
                    and not time_exhausted()
                ):
                    job = make_job()
                    pending[executor.submit(_sample_cpu_worker, job)] = job

            refill()
            while (
                pending
                and len(tasks) < config.n_tasks
                and not time_exhausted()
            ):
                completed, _ = wait(
                    tuple(pending),
                    return_when=FIRST_COMPLETED,
                )
                for future in sorted(
                    completed,
                    key=lambda item: pending[item].job_id,
                ):
                    job = pending.pop(future)
                    result = future.result()
                    if result.job_id != job.job_id:
                        raise RuntimeError(
                            "Worker returned a mismatched job_id."
                        )
                    commit(result)
                    if len(tasks) >= config.n_tasks:
                        break
                refill()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    if not tasks:
        raise RuntimeError(
            "No balanced corner-directed task was collected. "
            f"candidate_budget={candidate_budget}, "
            f"rejections={dict(reject_counts)}"
        )
    if len(tasks) < config.n_tasks and not config.save_partial:
        raise RuntimeError(
            f"Only collected {len(tasks)}/{config.n_tasks} tasks; "
            f"rejections={dict(reject_counts)}"
        )
    if len(tasks) < config.n_tasks:
        print(
            f"WARNING: saving partial result {len(tasks)}/{config.n_tasks}; "
            f"rejections={dict(reject_counts)}",
            flush=True,
        )

    for index, task in enumerate(tasks):
        digest = base._task_hash(task)
        task["task_id"] = f"task_{index:06d}_{digest[:10]}"
        task["metadata"]["canonical_hash"] = digest

    report = _collection_report(
        tasks,
        config=config,
        generation_seed=generation_seed,
        candidate_budget=candidate_budget,
        submitted_jobs=next_job_id,
        reject_counts=reject_counts,
        elapsed_seconds=time.monotonic() - started,
        worker_count=worker_count,
        max_in_flight=max_in_flight,
        start_method=start_method,
    )
    return tasks, report


def save_task_bank(
    config: DiverseConfig,
    tasks: Sequence[dict[str, Any]],
    report: dict[str, Any],
) -> Path:
    """Write a standard TABX task bank with the sampling protocol attached."""

    filter_protocol = {
        "enabled": True,
        "task_profile": "balanced_corner_diverse",
        "policy": "expert_vs_expert_win_rate_0.4_0.6",
        "win_rate_min": 0.4,
        "win_rate_max": 0.6,
        "num_seeds": config.filter_num_seeds,
        "epsilon": config.filter_epsilon,
        "max_episode_steps": config.filter_max_episode_steps,
        "reject_all_truncated": config.filter_reject_all_truncated,
        "max_truncation_rate": config.max_truncation_rate,
        "max_no_interaction_rate": config.max_no_interaction_rate,
        "heuristic": config.heuristic,
        "physics": config.physics,
        "screening": {
            "num_seeds": config.screening_num_seeds,
            "win_rate_min": config.screening_win_rate_min,
            "win_rate_max": config.screening_win_rate_max,
        },
    }
    output = base.save_task_bank(
        config.output,
        tasks,
        seed=config.seed,
        physics=config.physics,
        heuristic=config.heuristic,
        max_n_ally=config.max_n_ally,
        max_n_enemy=config.max_n_enemy,
        max_n_zone=config.max_n_zone,
        filter_protocol=filter_protocol,
    )
    bank = json.loads(output.read_text(encoding="utf-8"))
    bank["manifest"]["generator"] = "src.tabx.sample_diverse"
    bank["manifest"]["generator_protocol"] = {
        "strategy": "balanced_tsne_corner_direction_sampling",
        "note": (
            "bottom_left/top_right are proposal directions from the audited "
            "fixed-seed t-SNE, not invertible t-SNE coordinates"
        ),
        "corner_recipes": {
            "bottom_left": {
                "compositions": list(BOTTOM_LEFT_COMPOSITIONS),
                "layouts": list(BOTTOM_LEFT_LAYOUTS),
                "distances": sorted(set(BOTTOM_LEFT_DISTANCES)),
                "spreads": sorted(set(BOTTOM_LEFT_SPREADS)),
                "zones": list(BOTTOM_LEFT_ZONES),
                "minimum_total_units": config.bottom_left_min_total_units,
                "minimum_units_per_team": (
                    config.bottom_left_min_units_per_team
                ),
                "zone_count": [
                    config.bottom_left_min_zones,
                    config.bottom_left_max_zones,
                ],
                "map_scale": [
                    config.bottom_left_map_scale_min,
                    config.bottom_left_map_scale_max,
                ],
            },
            "top_right": {
                "compositions": list(TOP_RIGHT_COMPOSITIONS),
                "layouts": list(TOP_RIGHT_LAYOUTS),
                "distances": sorted(set(TOP_RIGHT_DISTANCES)),
                "spreads": sorted(set(TOP_RIGHT_SPREADS)),
                "zones": sorted(set(TOP_RIGHT_ZONES)),
                "maximum_total_units": config.top_right_max_total_units,
                "maximum_units_per_team": (
                    config.top_right_max_units_per_team
                ),
                "zone_count": [
                    config.top_right_min_zones,
                    config.top_right_max_zones,
                ],
                "map_scale": [
                    config.top_right_map_scale_min,
                    config.top_right_map_scale_max,
                ],
            },
        },
        "collection_report": report,
    }
    output.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    config = tyro.cli(DiverseConfig)
    tasks, report = sample_tasks_with_report(config)
    output = save_task_bank(config, tasks, report)
    status = (
        "complete"
        if report["complete"]
        else f"partial={len(tasks)}/{config.n_tasks}"
    )
    print(
        f"Saved {len(tasks)} balanced diverse tasks to {output}; "
        f"collection={status}, "
        f"profiles={report['profile_distribution']}, "
        f"occupied_corner_cells={report['occupied_corner_cells']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
