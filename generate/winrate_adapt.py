"""Per-task ally behavior-mix reweighting to steer episode win rate."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from generate.behavior_mix import BehaviorSpec


def is_strong_spec(spec: BehaviorSpec) -> bool:
    if spec.kind in {"oracle", "oracle_eps"}:
        return True
    if spec.kind == "heuristic" and str(spec.heuristic or "") == "advanced":
        return True
    return False


def _group_weights(mix: Sequence[BehaviorSpec], *, strong: bool) -> np.ndarray:
    raw = np.asarray(
        [float(s.weight) if is_strong_spec(s) == strong else 0.0 for s in mix],
        dtype=np.float64,
    )
    total = float(raw.sum())
    if total <= 1e-12:
        # Degenerate: fall back to uniform over the requested group if any members exist.
        mask = np.asarray([is_strong_spec(s) == strong for s in mix], dtype=np.float64)
        if mask.sum() <= 0:
            return np.zeros(len(mix), dtype=np.float64)
        return mask / mask.sum()
    return raw / total


def reweight_mix(mix: Sequence[BehaviorSpec], strength: float) -> tuple[BehaviorSpec, ...]:
    """Interpolate mix weights between weak-heavy (0) and strong-heavy (1)."""

    s = float(np.clip(strength, 0.0, 1.0))
    weak = _group_weights(mix, strong=False)
    strong = _group_weights(mix, strong=True)
    if weak.sum() <= 0 and strong.sum() <= 0:
        raise ValueError("behavior mix has no usable weights")
    if weak.sum() <= 0:
        weights = strong
    elif strong.sum() <= 0:
        weights = weak
    else:
        weights = (1.0 - s) * weak + s * strong
    weights = weights / max(float(weights.sum()), 1e-12)
    return tuple(replace(spec, weight=float(w)) for spec, w in zip(mix, weights))


def team_hp_totals(
    health: np.ndarray,
    team: np.ndarray,
    *,
    is_alive: np.ndarray | None = None,
) -> tuple[float, float]:
    health = np.asarray(health, dtype=np.float64).reshape(-1)
    team = np.asarray(team, dtype=np.int32).reshape(-1)
    if is_alive is not None:
        alive = np.asarray(is_alive).reshape(-1).astype(bool)
        health = np.where(alive, health, 0.0)
    hp0 = float(health[team == 0].sum()) if np.any(team == 0) else 0.0
    hp1 = float(health[team == 1].sum()) if np.any(team == 1) else 0.0
    return hp0, hp1


def classify_episode(
    *,
    is_win: int,
    health: np.ndarray | None,
    team: np.ndarray | None,
    is_alive: np.ndarray | None,
) -> str:
    if health is not None and team is not None:
        hp0, hp1 = team_hp_totals(health, team, is_alive=is_alive)
        total = hp0 + hp1
        if total <= 1e-12:
            return "draw"
        r0 = round(hp0 / total, 4)
        r1 = round(hp1 / total, 4)
        if r0 == r1:
            return "draw"
        if r0 > r1:
            return "win"
        return "loss"
    return "win" if int(is_win) else "loss"


def summarize_arrays(
    *,
    done: np.ndarray,
    is_win: np.ndarray,
    unit_health: np.ndarray | None = None,
    unit_team: np.ndarray | None = None,
    unit_is_alive: np.ndarray | None = None,
) -> dict[str, int]:
    done = np.asarray(done, dtype=np.uint8).reshape(-1)
    is_win = np.asarray(is_win, dtype=np.uint8).reshape(-1)
    terminals = np.flatnonzero(done == 1)
    counts = {"win": 0, "draw": 0, "loss": 0, "episodes": 0}
    for t in terminals:
        h = unit_health[t] if unit_health is not None else None
        tm = unit_team[t] if unit_team is not None else None
        al = unit_is_alive[t] if unit_is_alive is not None else None
        outcome = classify_episode(is_win=int(is_win[t]), health=h, team=tm, is_alive=al)
        counts[outcome] += 1
        counts["episodes"] += 1
    return counts


def summarize_record_dir(record_dir: Path) -> dict[str, int]:
    record_dir = Path(record_dir)
    done = np.load(record_dir / "done.npy")
    is_win = np.load(record_dir / "is_win.npy")
    health = team = alive = None
    if (record_dir / "unit_health.npy").is_file() and (record_dir / "unit_team.npy").is_file():
        health = np.load(record_dir / "unit_health.npy")
        team = np.load(record_dir / "unit_team.npy")
        if (record_dir / "unit_is_alive.npy").is_file():
            alive = np.load(record_dir / "unit_is_alive.npy")
    return summarize_arrays(
        done=done,
        is_win=is_win,
        unit_health=health,
        unit_team=team,
        unit_is_alive=alive,
    )


def win_rate_from_counts(counts: dict[str, int]) -> float:
    """Ally win rate among decisive episodes (exclude draws)."""

    wins = int(counts.get("win", 0))
    losses = int(counts.get("loss", 0))
    decisive = wins + losses
    if decisive <= 0:
        return 0.0
    return wins / decisive


def summarize_record_paths(paths: Sequence[str | Path]) -> tuple[dict[str, int], float]:
    total = {"win": 0, "draw": 0, "loss": 0, "episodes": 0}
    for path in paths:
        c = summarize_record_dir(Path(path))
        for k in total:
            total[k] += int(c.get(k, 0))
    return total, win_rate_from_counts(total)


def in_win_rate_band(
    wr: float,
    *,
    win_rate_min: float,
    win_rate_max: float | None,
) -> bool:
    if wr < float(win_rate_min):
        return False
    if win_rate_max is not None and wr > float(win_rate_max):
        return False
    return True


def choose_strength(
    measured_wr: float,
    *,
    win_rate_min: float,
    win_rate_max: float | None,
    current_strength: float = 0.5,
) -> float:
    """One-step strength update toward the win-rate band."""

    s = float(np.clip(current_strength, 0.0, 1.0))
    if measured_wr < float(win_rate_min):
        # Move toward strong; jump harder when far below.
        gap = float(win_rate_min) - measured_wr
        return float(np.clip(s + max(0.25, gap), 0.0, 1.0))
    if win_rate_max is not None and measured_wr > float(win_rate_max):
        gap = measured_wr - float(win_rate_max)
        return float(np.clip(s - max(0.25, gap), 0.0, 1.0))
    return s


def adapt_meta_dict(
    *,
    strength: float | None,
    win_rate_min: float,
    win_rate_max: float | None,
    measured_win_rate: float | None,
    status: str,
    pilot: bool,
    mix: Sequence[BehaviorSpec],
) -> dict[str, Any]:
    return {
        "enabled": True,
        "pilot": bool(pilot),
        "strength": None if strength is None else float(strength),
        "win_rate_min": float(win_rate_min),
        "win_rate_max": None if win_rate_max is None else float(win_rate_max),
        "measured_win_rate": None
        if measured_win_rate is None
        else float(measured_win_rate),
        "status": str(status),
        "behavior_mix": [asdict(s) for s in mix],
    }
