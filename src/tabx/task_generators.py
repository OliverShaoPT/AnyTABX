"""Programmatic composition, layout, and relational-zone generators for TABX."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np

from src.tabx.constants import ALL_UNIT_NAMES
from src.tabx.units import get_all_unit_spec

COMPOSITION_ARCHETYPES = (
    "frontline_backline",
    "frontline_healer",
    "assassin_fragile",
    "ranged_fort",
    "elite_swarm",
    "mobility_range",
)
# How the second team is derived from the first (balance-first diversity).
COMPOSITION_MATCH_MODES = (
    "mirror",  # exact copy (trimmed to capacity)
    "unit_swap",  # copy then replace a few units in-role
    "cost_match",  # different roster with similar total unit price
)
LAYOUT_ARCHETYPES = (
    "face_off",
    "crossfire",
    "encircle",
    "ambush",
    "breakout",
    "narrow_depth",
    "split_force",
    "protect_core",
)
# Empirically significant layout biases from 60-task expert-vs-expert analysis
# (ε=0.05, 16 seeds). Values multiply the *enemy* budget targets relative to ally:
#   >1: layout favors ally → strengthen enemy (buff)
#   <1: layout favors enemy → weaken enemy (debuff)
# Only layouts with Wilson CI excluding 0.5, |pooled_wr-0.5|>=0.10, and n_tasks>=4.
LAYOUT_ENEMY_BUDGET_MULTIPLIER = {
    "ambush": 1.2,  # ally pooled wr ≈ 0.76
    "encircle": 1.2,  # ally pooled wr ≈ 0.64
    "breakout": 0.8,  # ally pooled wr ≈ 0.28
    "crossfire": 0.8,  # ally pooled wr ≈ 0.38
    "narrow_depth": 0.8,  # ally pooled wr ≈ 0.31
}
LAYOUT_BUDGET_BUFF = 1.2
LAYOUT_BUDGET_DEBUFF = 0.8
DISTANCE_BUCKETS = ("close", "medium", "far")
SPREAD_BUCKETS = ("compact", "line", "dispersed")
ZONE_ARCHETYPES = (
    "void",
    "engagement_lava",
    "flank_bush",
    "center_swamp",
    "retreat_swamps",
    "asymmetric_cover",
    "lava_corridor",
    "channel_split",
    "mixed_overlap",
)
ZONE_INTENSITY_BUCKETS = ("low", "medium", "high")
ZONE_RELATION_BUCKETS = ("separated", "tangent", "overlap")

# JSON/runtime unit IDs are zero-based and follow get_all_unit_spec().
FARMER, ASSASSIN, KING, MAMMOTH, ARCHER, CANNON, DEADEYE, HEALER, PALADIN = range(9)
FRONTLINE = (FARMER, KING, MAMMOTH)
BACKLINE = (ARCHER, CANNON, DEADEYE)
HEALERS = (HEALER, PALADIN)
FRAGILE_BACKLINE = (ARCHER, DEADEYE, HEALER)
ELITE = (KING, MAMMOTH, CANNON)
MOBILE = (FARMER, ASSASSIN, DEADEYE)
RANGED = (ARCHER, CANNON, DEADEYE)

DEFAULT_GRID_INFO = {
    "grid_width": 28,
    "grid_height": 18,
    "max_field_width": 121.0,
    "max_field_height": 78.0,
    "margin_width": 0.0,
    "margin_height": 0.0,
}


def _bounded_count(rng: random.Random, minimum: int, maximum: int, capacity: int) -> int:
    upper = min(maximum, capacity)
    if upper < minimum:
        return upper
    return rng.randint(minimum, upper)


def _sample_units(rng: random.Random, pool: Sequence[int], count: int) -> list[int]:
    return [rng.choice(pool) for _ in range(count)]


def _cap_units(units: list[int], capacity: int) -> list[int]:
    if capacity < 1:
        raise ValueError("Each team capacity must be at least one.")
    return units[:capacity]


def _unit_specs() -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in get_all_unit_spec().items()}


def _unit_prices() -> np.ndarray:
    return _unit_specs()["prices"]


def team_price(units: Sequence[int]) -> float:
    """Total unit price (environment cost) for a roster."""

    if not units:
        return 0.0
    prices = _unit_prices()
    return float(prices[np.asarray(units, dtype=int)].sum())


def unit_prices_list(units: Sequence[int]) -> list[float]:
    prices = _unit_prices()
    return [float(prices[int(unit_id)]) for unit_id in units]


def team_effective_value(units: Sequence[int], health_fracs: Sequence[float]) -> float:
    """Weighted strength: sum_i price_i * health_frac_i."""

    if len(units) != len(health_fracs):
        raise ValueError("units and health_fracs must have the same length.")
    return float(
        sum(price * float(frac) for price, frac in zip(unit_prices_list(units), health_fracs))
    )


def sample_health_fractions(
    rng: random.Random,
    count: int,
    buckets: Sequence[float],
) -> list[float]:
    """Sample per-unit HP fractions relative to each unit's template max HP."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    if not buckets:
        raise ValueError("health fraction buckets must not be empty.")
    if any(not 0.0 < float(value) <= 1.0 for value in buckets):
        raise ValueError("health fraction buckets must lie in (0, 1].")
    return [float(rng.choice(tuple(buckets))) for _ in range(count)]


def _fit_health_fractions(fracs: Sequence[float], count: int) -> list[float]:
    """Trim/pad health fractions to match a resized roster."""

    if count < 1:
        raise ValueError("count must be at least one.")
    values = [float(value) for value in fracs[:count]]
    if not values:
        raise ValueError("Cannot fit empty health fractions.")
    while len(values) < count:
        values.append(values[len(values) % len(fracs)])
    return values


def project_health_fractions(
    rng: random.Random,
    prices: Sequence[float],
    target_effective: float,
    *,
    f_min: float = 0.5,
    f_max: float = 1.0,
) -> list[float]:
    """Match sum(price * frac) to a target via scale + residual repair.

    This is O(n) and avoids nested roster search for the HP constraint.
    """

    if not 0.0 < f_min <= f_max <= 1.0:
        raise ValueError("Require 0 < f_min <= f_max <= 1.")
    price_values = [float(price) for price in prices]
    n_units = len(price_values)
    if n_units == 0:
        return []
    total_price = sum(price_values)
    target = min(f_max * total_price, max(f_min * total_price, float(target_effective)))

    # Random shape for diversity, then project onto the feasible effective budget.
    base = [rng.uniform(f_min, f_max) for _ in range(n_units)]
    current = sum(price * frac for price, frac in zip(price_values, base))
    if current < 1e-8:
        base = [(f_min + f_max) * 0.5] * n_units
        current = sum(price * frac for price, frac in zip(price_values, base))
    scale = target / current
    fracs = [min(f_max, max(f_min, frac * scale)) for frac in base]

    for _ in range(8):
        current = sum(price * frac for price, frac in zip(price_values, fracs))
        residual = target - current
        if abs(residual) / max(target, 1.0) < 1e-4:
            break
        if residual > 0:
            slots = [
                (index, (f_max - fracs[index]) * price_values[index])
                for index in range(n_units)
                if fracs[index] < f_max - 1e-9
            ]
        else:
            slots = [
                (index, (fracs[index] - f_min) * price_values[index])
                for index in range(n_units)
                if fracs[index] > f_min + 1e-9
            ]
        capacity = sum(room for _, room in slots)
        if capacity < 1e-9:
            break
        for index, room in slots:
            delta = residual * (room / capacity) / price_values[index]
            fracs[index] = min(f_max, max(f_min, fracs[index] + delta))
    return fracs


def _role_pool_for(unit_id: int) -> tuple[int, ...]:
    if unit_id in ELITE:
        return ELITE
    if unit_id in FRONTLINE:
        return FRONTLINE
    if unit_id in BACKLINE or unit_id in RANGED:
        return RANGED
    if unit_id in HEALERS:
        return HEALERS
    if unit_id == ASSASSIN:
        return MOBILE
    return tuple(range(9))


def _similar_price_alternatives(unit_id: int, *, rel_tol: float = 0.45) -> list[int]:
    """Prefer replacements with similar unit price, then same role pool."""

    prices = _unit_prices()
    target = float(prices[unit_id])

    def within(tol: float) -> list[int]:
        return [
            other
            for other in range(9)
            if other != unit_id
            and abs(float(prices[other]) - target) / max(target, 1.0) <= tol
        ]

    similar = within(rel_tol)
    role = set(_role_pool_for(unit_id))
    role_similar = [other for other in similar if other in role]
    if role_similar:
        return role_similar
    if similar:
        return similar
    loose = within(max(rel_tol, 0.75))
    if loose:
        return loose
    ranked = sorted(
        (other for other in range(9) if other != unit_id),
        key=lambda other: abs(float(prices[other]) - target),
    )
    return ranked[:3]


def sample_seed_team(rng: random.Random, archetype: str, capacity: int) -> list[int]:
    """Sample one role-structured team for a composition archetype."""

    if archetype not in COMPOSITION_ARCHETYPES:
        raise ValueError(f"Unknown composition archetype: {archetype!r}.")
    if capacity < 1:
        raise ValueError("Team capacity must be at least one.")

    if archetype == "frontline_backline":
        front = _bounded_count(rng, 2, 4, max(1, capacity - 1))
        back = _bounded_count(rng, 1, 3, max(1, capacity - front))
        team = _sample_units(rng, FRONTLINE, front) + _sample_units(rng, BACKLINE, back)
    elif archetype == "frontline_healer":
        healer_count = 1 if capacity < 5 else rng.randint(1, 2)
        combat_count = _bounded_count(rng, 2, 5, capacity - healer_count)
        combat_pool = FRONTLINE + (ASSASSIN,)
        team = _sample_units(rng, combat_pool, combat_count) + _sample_units(
            rng, HEALERS, healer_count
        )
    elif archetype == "assassin_fragile":
        # Seed is either assassin pack or fragile backline; opponent is matched later.
        if rng.random() < 0.5:
            team = [ASSASSIN] * _bounded_count(rng, 2, 4, capacity)
            if len(team) < capacity and rng.random() < 0.6:
                team.append(rng.choice((FARMER, PALADIN)))
        else:
            team = _sample_units(
                rng, FRAGILE_BACKLINE, _bounded_count(rng, 2, 5, capacity)
            )
    elif archetype == "ranged_fort":
        n_total = _bounded_count(rng, 3, 7, capacity)
        n_front = 1 if n_total > 2 else 0
        team = _sample_units(rng, FRONTLINE, n_front) + _sample_units(
            rng, RANGED, n_total - n_front
        )
    elif archetype == "elite_swarm":
        if rng.random() < 0.5:
            team = _sample_units(rng, ELITE, _bounded_count(rng, 1, 3, capacity))
        else:
            team = [FARMER] * _bounded_count(rng, 4, 9, capacity)
    else:  # mobility_range
        if rng.random() < 0.5:
            team = _sample_units(rng, MOBILE, _bounded_count(rng, 3, 7, capacity))
            if team:
                team[0] = ASSASSIN
        else:
            team = _sample_units(rng, RANGED, _bounded_count(rng, 3, 7, capacity))

    team = _cap_units(team, capacity)
    if not team:
        raise ValueError("Seed team generation produced an empty roster.")
    return team


def _fit_team_size(seed: Sequence[int], capacity: int) -> list[int]:
    """Trim or lightly pad a seed roster to fit the opposing capacity."""

    if capacity < 1:
        raise ValueError("Team capacity must be at least one.")
    team = list(seed[:capacity])
    if not team:
        raise ValueError("Cannot fit an empty seed team.")
    # Prefer keeping similar size; only pad when seed is shorter than capacity and tiny.
    while len(team) < min(capacity, max(len(seed), 1)) and len(team) < capacity:
        team.append(team[len(team) % len(seed)])
    return _cap_units(team, capacity)


def mirror_team(seed: Sequence[int], capacity: int) -> list[int]:
    """Copy the seed roster into the opposing capacity."""

    return _fit_team_size(seed, capacity)


def unit_swap_team(
    rng: random.Random,
    seed: Sequence[int],
    capacity: int,
    *,
    n_replace: int | None = None,
) -> list[int]:
    """Copy the seed roster, then replace a few units within their role pools."""

    team = _fit_team_size(seed, capacity)
    max_replace = min(3, len(team))
    if n_replace is None:
        n_replace = rng.randint(1, max_replace) if max_replace else 0
    n_replace = max(0, min(n_replace, len(team)))
    if n_replace == 0:
        return team
    for index in rng.sample(range(len(team)), k=n_replace):
        alternatives = _similar_price_alternatives(team[index])
        team[index] = rng.choice(alternatives)
    return team


def layout_enemy_budget_multiplier(layout: str | None) -> float:
    """Return enemy/ally budget multiplier for a layout, or 1.0 if neutral/unknown."""

    if layout is None:
        return 1.0
    return float(LAYOUT_ENEMY_BUDGET_MULTIPLIER.get(layout, 1.0))


def cost_matched_team(
    rng: random.Random,
    seed: Sequence[int],
    capacity: int,
    archetype: str,
    *,
    rel_tol: float = 0.15,
    n_trials: int = 64,
    target_price: float | None = None,
) -> list[int]:
    """Sample a roster whose total price is close to ``target_price`` (or the seed)."""

    target = float(team_price(seed) if target_price is None else target_price)
    seed_key = tuple(sorted(seed))
    best: list[int] | None = None
    best_diff = float("inf")
    for _ in range(max(1, n_trials)):
        candidate = sample_seed_team(rng, archetype, capacity)
        if tuple(sorted(candidate)) == seed_key and abs(target - team_price(seed)) < 1e-6:
            continue
        diff = abs(team_price(candidate) - target) / max(target, 1.0)
        if diff < best_diff:
            best_diff = diff
            best = candidate
        if diff <= rel_tol:
            return candidate
    if best is not None:
        return best
    # Fallback: keep balance via light unit swaps when no distinct roster is found.
    return unit_swap_team(rng, seed, capacity)


def derive_opponent_team(
    rng: random.Random,
    seed: Sequence[int],
    seed_health_fracs: Sequence[float],
    capacity: int,
    archetype: str,
    match_mode: str,
    *,
    price_rel_tol: float = 0.15,
    effective_rel_tol: float = 0.15,
    health_frac_min: float = 0.5,
    health_frac_max: float = 1.0,
    n_trials: int = 64,
    budget_multiplier: float = 1.0,
) -> tuple[list[int], list[float]]:
    """Build the second team + HP fractions under price and effective-value constraints.

    ``budget_multiplier`` scales both target price and target effective value for the
    derived team (used for layout-bias compensation).
    """

    if match_mode not in COMPOSITION_MATCH_MODES:
        raise ValueError(f"Unknown composition match mode: {match_mode!r}.")
    if budget_multiplier <= 0:
        raise ValueError("budget_multiplier must be positive.")
    base_price = team_price(seed)
    base_effective = team_effective_value(seed, seed_health_fracs)
    target_price = base_price * budget_multiplier
    target_effective = base_effective * budget_multiplier
    seed_key = tuple(sorted(seed))
    needs_budget_shift = abs(budget_multiplier - 1.0) > 1e-6

    def finalize(units: list[int]) -> tuple[list[int], list[float]]:
        fracs = project_health_fractions(
            rng,
            unit_prices_list(units),
            target_effective,
            f_min=health_frac_min,
            f_max=health_frac_max,
        )
        return units, fracs

    def cost_match_pair() -> tuple[list[int], list[float]]:
        best: tuple[list[int], list[float], float] | None = None
        for _ in range(max(1, n_trials)):
            candidate = sample_seed_team(rng, archetype, capacity)
            if tuple(sorted(candidate)) == seed_key and not needs_budget_shift:
                continue
            price_diff = abs(team_price(candidate) - target_price) / max(target_price, 1.0)
            if price_diff > price_rel_tol:
                continue
            units, fracs = finalize(candidate)
            effective_diff = abs(
                team_effective_value(units, fracs) - target_effective
            ) / max(target_effective, 1.0)
            score = price_diff + effective_diff
            if best is None or score < best[2]:
                best = (units, fracs, score)
            if effective_diff <= effective_rel_tol:
                return units, fracs
        if best is not None:
            return best[0], best[1]
        return finalize(
            cost_matched_team(
                rng,
                seed,
                capacity,
                archetype,
                rel_tol=price_rel_tol,
                n_trials=n_trials,
                target_price=target_price,
            )
        )

    # Layout compensation changes the budget; prefer cost matching over pure mirror.
    if needs_budget_shift or match_mode == "cost_match":
        return cost_match_pair()

    if match_mode == "mirror":
        mirrored = mirror_team(seed, capacity)
        if len(mirrored) == len(seed):
            return mirrored, _fit_health_fractions(seed_health_fracs, len(mirrored))
        return finalize(mirrored)

    # unit_swap
    swapped = unit_swap_team(rng, seed, capacity)
    price_diff = abs(team_price(swapped) - target_price) / max(target_price, 1.0)
    if price_diff > max(2.0 * price_rel_tol, 0.25):
        swapped = cost_matched_team(
            rng,
            seed,
            capacity,
            archetype,
            rel_tol=price_rel_tol,
            n_trials=n_trials,
            target_price=target_price,
        )
    return finalize(swapped)


def sample_compositions(
    rng: random.Random,
    archetype: str,
    max_n_ally: int,
    max_n_enemy: int,
    *,
    match_mode: str = "unit_swap",
    price_rel_tol: float = 0.15,
    effective_rel_tol: float = 0.15,
    health_frac_buckets: Sequence[float] = (0.6, 0.7, 0.8, 0.9, 1.0),
    health_frac_min: float = 0.5,
    health_frac_max: float = 1.0,
    layout: str | None = None,
) -> tuple[list[int], list[int], list[float], list[float], dict[str, Any]]:
    """Sample one seed team, then match the other on price and effective value.

    When ``layout`` has a significant empirical bias, the *enemy* budget targets are
    scaled by ``LAYOUT_ENEMY_BUDGET_MULTIPLIER`` (1.2 buff / 0.8 debuff) relative to ally.
    """

    if archetype not in COMPOSITION_ARCHETYPES:
        raise ValueError(f"Unknown composition archetype: {archetype!r}.")
    if match_mode not in COMPOSITION_MATCH_MODES:
        raise ValueError(f"Unknown composition match mode: {match_mode!r}.")
    if not 0.0 < health_frac_min <= health_frac_max <= 1.0:
        raise ValueError("Require 0 < health_frac_min <= health_frac_max <= 1.")

    enemy_budget_mult = layout_enemy_budget_multiplier(layout)
    ally_is_seed = rng.random() < 0.5
    seed_capacity, other_capacity = (
        (max_n_ally, max_n_enemy) if ally_is_seed else (max_n_enemy, max_n_ally)
    )
    # Multiplier is defined as enemy/ally. If we derive ally from an enemy seed,
    # invert it so the final enemy/ally ratio still matches the layout table.
    derive_mult = enemy_budget_mult if ally_is_seed else (1.0 / enemy_budget_mult)

    seed = sample_seed_team(rng, archetype, seed_capacity)
    seed_fracs = sample_health_fractions(rng, len(seed), health_frac_buckets)
    other, other_fracs = derive_opponent_team(
        rng,
        seed,
        seed_fracs,
        other_capacity,
        archetype,
        match_mode,
        price_rel_tol=price_rel_tol,
        effective_rel_tol=effective_rel_tol,
        health_frac_min=health_frac_min,
        health_frac_max=health_frac_max,
        budget_multiplier=derive_mult,
    )
    if ally_is_seed:
        ally, enemy = seed, other
        ally_fracs, enemy_fracs = seed_fracs, other_fracs
    else:
        ally, enemy = other, seed
        ally_fracs, enemy_fracs = other_fracs, seed_fracs
    ally = _cap_units(ally, max_n_ally)
    enemy = _cap_units(enemy, max_n_enemy)
    ally_fracs = ally_fracs[: len(ally)]
    enemy_fracs = enemy_fracs[: len(enemy)]
    if not ally or not enemy:
        raise ValueError("Composition generation produced an empty team.")

    ally_price = team_price(ally)
    enemy_price = team_price(enemy)
    ally_effective = team_effective_value(ally, ally_fracs)
    enemy_effective = team_effective_value(enemy, enemy_fracs)
    match_info = {
        "match_mode": match_mode,
        "seed_side": "ally" if ally_is_seed else "enemy",
        "layout": layout,
        "layout_enemy_budget_multiplier": enemy_budget_mult,
        "ally_price": ally_price,
        "enemy_price": enemy_price,
        "price_rel_diff": abs(ally_price - enemy_price)
        / max(max(ally_price, enemy_price), 1.0),
        "price_rel_tol": price_rel_tol,
        "ally_effective": ally_effective,
        "enemy_effective": enemy_effective,
        "effective_rel_diff": abs(ally_effective - enemy_effective)
        / max(max(ally_effective, enemy_effective), 1.0),
        "effective_rel_tol": effective_rel_tol,
        "enemy_over_ally_price": enemy_price / max(ally_price, 1.0),
        "enemy_over_ally_effective": enemy_effective / max(ally_effective, 1.0),
        "ally_health_fracs": [float(value) for value in ally_fracs],
        "enemy_health_fracs": [float(value) for value in enemy_fracs],
    }
    return ally, enemy, ally_fracs, enemy_fracs, match_info


def composition_features(
    ally: Sequence[int],
    enemy: Sequence[int],
    ally_health_fracs: Sequence[float] | None = None,
    enemy_health_fracs: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return interpretable composition metadata."""

    prices = _unit_prices()
    ally_fracs = (
        [1.0] * len(ally)
        if ally_health_fracs is None
        else [float(value) for value in ally_health_fracs]
    )
    enemy_fracs = (
        [1.0] * len(enemy)
        if enemy_health_fracs is None
        else [float(value) for value in enemy_health_fracs]
    )

    def summarize(units: Sequence[int], fracs: Sequence[float]) -> dict[str, Any]:
        counts = {name: 0 for name in ALL_UNIT_NAMES}
        for unit_id in units:
            counts[ALL_UNIT_NAMES[unit_id]] += 1
        total_price = float(prices[np.asarray(units)].sum()) if units else 0.0
        effective = team_effective_value(units, fracs) if units else 0.0
        return {
            "count": len(units),
            "unit_counts": {name: count for name, count in counts.items() if count},
            "total_price": total_price,
            "effective_value": effective,
            "mean_health_frac": float(sum(fracs) / len(fracs)) if fracs else 1.0,
            "health_fracs": [float(value) for value in fracs],
            "melee_ratio": float(
                sum(unit in FRONTLINE + (ASSASSIN,) for unit in units) / len(units)
            )
            if units
            else 0.0,
            "ranged_ratio": float(sum(unit in RANGED for unit in units) / len(units))
            if units
            else 0.0,
            "healer_ratio": float(sum(unit in HEALERS for unit in units) / len(units))
            if units
            else 0.0,
        }

    features = {
        "ally": summarize(ally, ally_fracs),
        "enemy": summarize(enemy, enemy_fracs),
    }
    ally_price = features["ally"]["total_price"]
    enemy_price = features["enemy"]["total_price"]
    ally_effective = features["ally"]["effective_value"]
    enemy_effective = features["enemy"]["effective_value"]
    features["price_rel_diff"] = abs(ally_price - enemy_price) / max(
        max(ally_price, enemy_price), 1.0
    )
    features["effective_rel_diff"] = abs(ally_effective - enemy_effective) / max(
        max(ally_effective, enemy_effective), 1.0
    )
    return features


def _formation_offsets(count: int, spread: str, axis: str = "vertical") -> np.ndarray:
    """Generate deterministic centered offsets for one team formation."""

    if count <= 0:
        return np.zeros((0, 2))
    spacing = {"compact": 5.5, "line": 7.5, "dispersed": 10.0}[spread]
    offsets = []
    if spread == "compact":
        columns = max(1, math.ceil(math.sqrt(count)))
        rows = math.ceil(count / columns)
        for index in range(count):
            row, column = divmod(index, columns)
            offsets.append(
                ((column - (columns - 1) / 2) * spacing, (row - (rows - 1) / 2) * spacing)
            )
    elif spread == "line":
        for index in range(count):
            coordinate = (index - (count - 1) / 2) * spacing
            offsets.append((0.0, coordinate) if axis == "vertical" else (coordinate, 0.0))
    else:
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for index in range(count):
            radius = spacing * math.sqrt(index)
            angle = index * golden_angle
            offsets.append((radius * math.cos(angle), radius * math.sin(angle)))
    return np.asarray(offsets, dtype=float)


def _point_towards(position: np.ndarray, target: np.ndarray) -> float:
    delta = target - position
    return float(math.atan2(delta[1], delta[0]))


def _layout_targets(
    rng: random.Random,
    layout: str,
    distance: str,
    spread: str,
    n_ally: int,
    n_enemy: int,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate desired centers before collision-safe placement."""

    separation = {
        "close": width * 0.20,
        "medium": width * 0.36,
        "far": width * 0.52,
    }[distance]
    vertical_span = min(height * 0.30, 18.0 * width / 121.0)
    ally_offsets = _formation_offsets(n_ally, spread)
    enemy_offsets = _formation_offsets(n_enemy, spread)

    if layout == "face_off":
        ally = ally_offsets + np.array([-separation / 2, 0.0])
        enemy = enemy_offsets + np.array([separation / 2, 0.0])
    elif layout == "crossfire":
        ally = ally_offsets + np.array([-separation / 2, vertical_span * 0.55])
        enemy = enemy_offsets + np.array([separation / 2, -vertical_span * 0.55])
    elif layout == "encircle":
        radius = max(12.0, separation * 0.60)
        ally_angles = np.linspace(0, 2 * math.pi, n_ally, endpoint=False)
        ally = np.column_stack((np.cos(ally_angles), np.sin(ally_angles))) * radius
        enemy = _formation_offsets(n_enemy, "compact")
    elif layout == "ambush":
        ally = _formation_offsets(n_ally, "line", axis="horizontal")
        ally[:, 1] += np.where(np.arange(n_ally) % 2 == 0, vertical_span, -vertical_span)
        ally[:, 0] -= separation * 0.15
        enemy = enemy_offsets + np.array([separation * 0.25, 0.0])
    elif layout == "breakout":
        ally = ally_offsets + np.array([-separation * 0.20, 0.0])
        angles = np.linspace(-math.pi / 2, math.pi / 2, n_enemy)
        enemy = np.column_stack(
            (
                np.cos(angles) * separation * 0.55,
                np.sin(angles) * min(vertical_span * 1.25, height * 0.35),
            )
        )
    elif layout == "narrow_depth":
        ally = _formation_offsets(n_ally, "line") + np.array([-separation / 2, 0.0])
        enemy = _formation_offsets(n_enemy, "line") + np.array([separation / 2, 0.0])
    elif layout == "split_force":
        ally = ally_offsets + np.array([-separation / 2, 0.0])
        enemy = enemy_offsets + np.array([separation / 2, 0.0])
        ally[:, 1] += np.where(np.arange(n_ally) % 2 == 0, vertical_span, -vertical_span)
        enemy[:, 1] += np.where(np.arange(n_enemy) % 2 == 0, vertical_span, -vertical_span)
    elif layout == "protect_core":
        ally = ally_offsets + np.array([-separation / 2, 0.0])
        enemy = enemy_offsets + np.array([separation / 2, 0.0])
        ally[:, 0] += np.linspace(3.0, -3.0, n_ally)
        enemy[:, 0] += np.linspace(-3.0, 3.0, n_enemy)
    else:
        raise ValueError(f"Unknown layout archetype: {layout!r}.")

    jitter = max(0.15, min(width, height) * 0.003)
    ally += np.asarray([[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in ally])
    enemy += np.asarray(
        [[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in enemy]
    )
    return ally, enemy


def _place_without_overlap(
    rng: random.Random,
    desired: np.ndarray,
    radii: np.ndarray,
    width: float,
    height: float,
    occupied: list[tuple[np.ndarray, float]],
) -> np.ndarray:
    """Move desired points to nearby collision-free locations."""

    result = np.zeros_like(desired)
    order = sorted(range(len(desired)), key=lambda index: -radii[index])
    for index in order:
        radius = float(radii[index])
        margin = 0.75 if radius >= 4.0 else 0.3
        candidates = [desired[index]]
        for ring in range(1, 18):
            step = 1.7 + radius * 0.55
            for angle_index in range(16):
                angle = 2 * math.pi * angle_index / 16 + rng.uniform(-0.04, 0.04)
                candidates.append(
                    desired[index] + ring * step * np.array([math.cos(angle), math.sin(angle)])
                )
        placed = None
        for candidate in candidates:
            candidate = np.clip(
                candidate,
                np.array([-width / 2 + radius + 0.5, -height / 2 + radius + 0.5]),
                np.array([width / 2 - radius - 0.5, height / 2 - radius - 0.5]),
            )
            if all(
                np.linalg.norm(candidate - other) >= radius + other_radius + margin
                for other, other_radius in occupied
            ):
                placed = candidate
                break
        if placed is None:
            raise ValueError("Could not place all units without overlap.")
        result[index] = placed
        occupied.append((placed, radius))
    return result


def build_scenario(
    rng: random.Random,
    ally_units: Sequence[int],
    enemy_units: Sequence[int],
    layout: str,
    distance: str,
    spread: str,
    map_scale: float,
    ally_health_fracs: Sequence[float] | None = None,
    enemy_health_fracs: Sequence[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a complete unpadded VectorizedScenario JSON dictionary."""

    if layout not in LAYOUT_ARCHETYPES:
        raise ValueError(f"Unknown layout archetype: {layout!r}.")
    if distance not in DISTANCE_BUCKETS or spread not in SPREAD_BUCKETS:
        raise ValueError("Unknown layout distance or spread bucket.")

    specs = _unit_specs()
    units = list(ally_units) + list(enemy_units)
    teams = np.asarray([0] * len(ally_units) + [1] * len(enemy_units), dtype=int)
    unit_ids = np.asarray(units, dtype=int)
    radii = specs["body_radiuses"][unit_ids].astype(float)
    width = DEFAULT_GRID_INFO["max_field_width"] * map_scale
    height = DEFAULT_GRID_INFO["max_field_height"] * map_scale
    ally_targets, enemy_targets = _layout_targets(
        rng, layout, distance, spread, len(ally_units), len(enemy_units), width, height
    )
    occupied: list[tuple[np.ndarray, float]] = []
    # Large units are placed collision-safely across both teams.
    ally_positions = _place_without_overlap(
        rng, ally_targets, radii[: len(ally_units)], width, height, occupied
    )
    enemy_positions = _place_without_overlap(
        rng, enemy_targets, radii[len(ally_units) :], width, height, occupied
    )
    positions = np.concatenate([ally_positions, enemy_positions])
    ally_centroid = ally_positions.mean(axis=0)
    enemy_centroid = enemy_positions.mean(axis=0)
    rotations = np.asarray(
        [
            _point_towards(position, enemy_centroid if team == 0 else ally_centroid)
            for position, team in zip(positions, teams)
        ]
    )
    pos_max = np.column_stack((width / 2 - radii, height / 2 - radii))
    damage = specs["attack_damages"][unit_ids].astype(float)
    ally_fracs = (
        np.ones(len(ally_units), dtype=float)
        if ally_health_fracs is None
        else np.asarray(ally_health_fracs, dtype=float)
    )
    enemy_fracs = (
        np.ones(len(enemy_units), dtype=float)
        if enemy_health_fracs is None
        else np.asarray(enemy_health_fracs, dtype=float)
    )
    if len(ally_fracs) != len(ally_units) or len(enemy_fracs) != len(enemy_units):
        raise ValueError("Health fractions must match each team's unit count.")
    if np.any(ally_fracs <= 0) or np.any(enemy_fracs <= 0):
        raise ValueError("Health fractions must be positive.")
    health_fracs = np.concatenate([ally_fracs, enemy_fracs])
    healths = specs["healths"][unit_ids].astype(float) * health_fracs

    def column(values: np.ndarray) -> list[list[Any]]:
        return np.asarray(values).reshape(-1, 1).tolist()

    scenario = {
        "positions": positions.tolist(),
        "rotations": column(rotations),
        "body_weights": column(specs["body_weights"][unit_ids].astype(float)),
        "body_radiuss": column(radii),
        "teams": column(teams),
        "pos_min": (-pos_max).tolist(),
        "pos_max": pos_max.tolist(),
        "unit_ids": column(unit_ids),
        "healths": column(healths),
        "attack_damages": column(damage),
        "attack_ranges": column(specs["attack_ranges"][unit_ids].astype(float)),
        "attack_cooldowns": column(specs["attack_cooldown"][unit_ids].astype(float)),
        "sight_angles": column(specs["sight_angles"][unit_ids].astype(float)),
        "is_alive": column(np.ones(len(units), dtype=bool)),
        "attack_types": column((damage < 0).astype(int)),
        "is_disabled": column(np.zeros(len(units), dtype=bool)),
        "speeds": column(specs["speeds"][unit_ids].astype(float)),
    }
    grid_info = {
        **DEFAULT_GRID_INFO,
        "max_field_width": width,
        "max_field_height": height,
    }
    return scenario, grid_info


def _geometry(scenario: dict[str, Any]) -> dict[str, np.ndarray]:
    positions = np.asarray(scenario["positions"], dtype=float)
    teams = np.asarray(scenario["teams"]).reshape(-1)
    unit_ids = np.asarray(scenario["unit_ids"]).reshape(-1)
    ally = positions[teams == 0]
    enemy = positions[teams == 1]
    ally_center = ally.mean(axis=0)
    enemy_center = enemy.mean(axis=0)
    direction = enemy_center - ally_center
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
    perpendicular = np.array([-direction[1], direction[0]])
    return {
        "positions": positions,
        "teams": teams,
        "unit_ids": unit_ids,
        "ally_center": ally_center,
        "enemy_center": enemy_center,
        "midpoint": (ally_center + enemy_center) / 2,
        "direction": direction,
        "perpendicular": perpendicular,
    }


def _zone_effect(zone_type: int, intensity: str) -> float:
    index = ZONE_INTENSITY_BUCKETS.index(intensity)
    if zone_type == 1:
        return (6.0, 10.0, 14.0)[index]
    if zone_type == 3:
        return (0.15, 0.3, 0.45)[index]
    return 0.0


def _fit_zone(
    center: np.ndarray, axes: np.ndarray, width: float, height: float
) -> tuple[np.ndarray, np.ndarray]:
    axes = np.maximum(np.asarray(axes, dtype=float), 1.5)
    axes = np.minimum(axes, np.array([width * 0.30, height * 0.35]))
    center = np.clip(
        center,
        np.array([-width / 2, -height / 2]) + axes + 1.0,
        np.array([width / 2, height / 2]) - axes - 1.0,
    )
    return center, axes


def _lava_clear_of_spawns(
    center: np.ndarray,
    axes: np.ndarray,
    scenario: dict[str, Any],
    perpendicular: np.ndarray,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(scenario["positions"], dtype=float)
    radii = np.asarray(scenario["body_radiuss"], dtype=float).reshape(-1)
    for shift in (0.0, 1.0, -1.0, 1.8, -1.8, 2.6, -2.6):
        candidate, fitted_axes = _fit_zone(
            center + perpendicular * shift * (axes[1] + 2.0), axes, width, height
        )
        normalized = ((positions - candidate) / (fitted_axes + radii[:, None])) ** 2
        if np.all(normalized.sum(axis=1) > 1.0):
            return candidate, fitted_axes
    raise ValueError("Could not place lava away from all spawn positions.")


def build_relational_zones(
    scenario: dict[str, Any],
    grid_info: dict[str, Any],
    archetype: str,
    intensity: str,
    relation: str,
) -> dict[str, Any]:
    """Generate zones from unit geometry rather than independent coordinates."""

    if archetype not in ZONE_ARCHETYPES:
        raise ValueError(f"Unknown zone archetype: {archetype!r}.")
    if intensity not in ZONE_INTENSITY_BUCKETS:
        raise ValueError(f"Unknown zone intensity: {intensity!r}.")
    if relation not in ZONE_RELATION_BUCKETS:
        raise ValueError(f"Unknown zone relation: {relation!r}.")
    if archetype == "void":
        return {"n_zone": 0, "zone_type": [], "position": [], "axes": [], "effect_value": []}

    geometry = _geometry(scenario)
    midpoint = geometry["midpoint"]
    direction = geometry["direction"]
    perpendicular = geometry["perpendicular"]
    ally_center = geometry["ally_center"]
    enemy_center = geometry["enemy_center"]
    width = float(grid_info["max_field_width"])
    height = float(grid_info["max_field_height"])
    scale = min(width / 121.0, height / 78.0)
    relation_factor = {"separated": 1.35, "tangent": 1.0, "overlap": 0.70}[relation]
    zones: list[tuple[int, np.ndarray, np.ndarray]] = []

    if archetype == "engagement_lava":
        zones = [(1, midpoint, np.array([7.0, 3.5]) * scale)]
    elif archetype == "flank_bush":
        flank = min(14.0 * relation_factor * scale, height * 0.28)
        zones = [
            (2, midpoint + perpendicular * flank, np.array([5.0, 7.0]) * scale),
            (2, midpoint - perpendicular * flank, np.array([5.0, 7.0]) * scale),
        ]
    elif archetype == "center_swamp":
        zones = [(3, midpoint, np.array([6.0, 18.0]) * scale)]
    elif archetype == "retreat_swamps":
        retreat = 7.0 * scale
        zones = [
            (3, ally_center - direction * retreat, np.array([6.0, 8.0]) * scale),
            (3, enemy_center + direction * retreat, np.array([6.0, 8.0]) * scale),
        ]
    elif archetype == "asymmetric_cover":
        zones = [
            (
                2,
                ally_center + perpendicular * 10.0 * scale + direction * 4.0 * scale,
                np.array([6.0, 8.0]) * scale,
            )
        ]
    elif archetype == "lava_corridor":
        gap = 10.0 * relation_factor * scale
        zones = [
            (1, midpoint - direction * gap, np.array([5.0, 3.0]) * scale),
            (1, midpoint, np.array([5.0, 3.0]) * scale),
            (1, midpoint + direction * gap, np.array([5.0, 3.0]) * scale),
        ]
    elif archetype == "channel_split":
        flank = 15.0 * relation_factor * scale
        zones = [
            (3, midpoint, np.array([7.0, 20.0]) * scale),
            (2, midpoint + perpendicular * flank, np.array([8.0, 4.0]) * scale),
            (2, midpoint - perpendicular * flank, np.array([8.0, 4.0]) * scale),
        ]
    else:
        # The relation bucket explicitly creates separated, tangent, or overlapping effects.
        pair_gap = 15.0 * relation_factor * scale
        zones = [
            (2, midpoint, np.array([8.0, 7.0]) * scale),
            (3, midpoint + direction * pair_gap, np.array([7.0, 6.0]) * scale),
        ]

    zone_types = []
    positions = []
    axes_values = []
    effects = []
    for zone_type, center, axes in zones:
        center, axes = _fit_zone(center, axes, width, height)
        if zone_type == 1:
            center, axes = _lava_clear_of_spawns(
                center, axes, scenario, perpendicular, width, height
            )
        zone_types.append([zone_type])
        positions.append(center.tolist())
        axes_values.append(axes.tolist())
        effects.append([_zone_effect(zone_type, intensity)])
    return {
        "n_zone": len(zones),
        "zone_type": zone_types,
        "position": positions,
        "axes": axes_values,
        "effect_value": effects,
    }


def generate_programmatic_task(
    rng: random.Random,
    *,
    composition: str,
    layout: str,
    distance: str,
    spread: str,
    zone: str,
    zone_intensity: str,
    zone_relation: str,
    map_scale: float,
    max_n_ally: int,
    max_n_enemy: int,
    match_mode: str = "unit_swap",
    price_rel_tol: float = 0.15,
    effective_rel_tol: float = 0.15,
    health_frac_buckets: Sequence[float] = (0.6, 0.7, 0.8, 0.9, 1.0),
    health_frac_min: float = 0.5,
    health_frac_max: float = 1.0,
) -> dict[str, Any]:
    """Generate one complete task covering 4.1 A/B/C/D."""

    ally, enemy, ally_fracs, enemy_fracs, match_info = sample_compositions(
        rng,
        composition,
        max_n_ally,
        max_n_enemy,
        match_mode=match_mode,
        price_rel_tol=price_rel_tol,
        effective_rel_tol=effective_rel_tol,
        health_frac_buckets=health_frac_buckets,
        health_frac_min=health_frac_min,
        health_frac_max=health_frac_max,
        layout=layout,
    )
    scenario, grid_info = build_scenario(
        rng,
        ally,
        enemy,
        layout,
        distance,
        spread,
        map_scale,
        ally_health_fracs=ally_fracs,
        enemy_health_fracs=enemy_fracs,
    )
    zone_scenario = build_relational_zones(scenario, grid_info, zone, zone_intensity, zone_relation)
    return {
        "grid_info": grid_info,
        "scenario": scenario,
        "zone_scenario": zone_scenario,
        "metadata": {
            "source": {"kind": "programmatic"},
            "composition_archetype": composition,
            "composition_match": match_info,
            "composition_features": composition_features(
                ally, enemy, ally_fracs, enemy_fracs
            ),
            "layout_archetype": layout,
            "distance_bucket": distance,
            "spread_bucket": spread,
            "map_bucket": {0.85: "small", 1.0: "medium", 1.15: "large"}.get(
                map_scale, f"scale_{map_scale:g}"
            ),
            "map_scale": map_scale,
            "zone_archetype": zone,
            "zone_intensity": zone_intensity,
            "zone_relation": zone_relation,
            "scenario_bucket": {
                "composition": composition,
                "layout": layout,
                "distance": distance,
                "spread": spread,
                "zone": zone,
                "zone_intensity": zone_intensity,
                "zone_relation": zone_relation,
                "match_mode": match_mode,
            },
        },
    }
