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


def sample_compositions(
    rng: random.Random,
    archetype: str,
    max_n_ally: int,
    max_n_enemy: int,
) -> tuple[list[int], list[int]]:
    """Sample two role-structured unit multisets for a composition archetype."""

    if archetype not in COMPOSITION_ARCHETYPES:
        raise ValueError(f"Unknown composition archetype: {archetype!r}.")

    if archetype == "frontline_backline":

        def make_team(capacity: int) -> list[int]:
            front = _bounded_count(rng, 2, 4, max(1, capacity - 1))
            back = _bounded_count(rng, 1, 3, max(1, capacity - front))
            return _cap_units(
                _sample_units(rng, FRONTLINE, front) + _sample_units(rng, BACKLINE, back),
                capacity,
            )

        ally, enemy = make_team(max_n_ally), make_team(max_n_enemy)
    elif archetype == "frontline_healer":

        def make_team(capacity: int) -> list[int]:
            healer_count = 1 if capacity < 5 else rng.randint(1, 2)
            combat_count = _bounded_count(rng, 2, 5, capacity - healer_count)
            combat_pool = FRONTLINE + (ASSASSIN,)
            return _cap_units(
                _sample_units(rng, combat_pool, combat_count)
                + _sample_units(rng, HEALERS, healer_count),
                capacity,
            )

        ally, enemy = make_team(max_n_ally), make_team(max_n_enemy)
    elif archetype == "assassin_fragile":
        ally_is_assassin = rng.random() < 0.5
        assassin_capacity, fragile_capacity = (
            (max_n_ally, max_n_enemy) if ally_is_assassin else (max_n_enemy, max_n_ally)
        )
        assassins = [ASSASSIN] * _bounded_count(rng, 2, 4, assassin_capacity)
        if len(assassins) < assassin_capacity and rng.random() < 0.6:
            assassins.append(rng.choice((FARMER, PALADIN)))
        fragile = _sample_units(rng, FRAGILE_BACKLINE, _bounded_count(rng, 2, 5, fragile_capacity))
        ally, enemy = (assassins, fragile) if ally_is_assassin else (fragile, assassins)
    elif archetype == "ranged_fort":

        def make_team(capacity: int) -> list[int]:
            n_total = _bounded_count(rng, 3, 7, capacity)
            n_front = 1 if n_total > 2 else 0
            return _sample_units(rng, FRONTLINE, n_front) + _sample_units(
                rng, RANGED, n_total - n_front
            )

        ally, enemy = make_team(max_n_ally), make_team(max_n_enemy)
    elif archetype == "elite_swarm":
        ally_is_elite = rng.random() < 0.5
        elite_capacity, swarm_capacity = (
            (max_n_ally, max_n_enemy) if ally_is_elite else (max_n_enemy, max_n_ally)
        )
        elite = _sample_units(rng, ELITE, _bounded_count(rng, 1, 3, elite_capacity))
        swarm = [FARMER] * _bounded_count(rng, 4, 9, swarm_capacity)
        ally, enemy = (elite, swarm) if ally_is_elite else (swarm, elite)
    else:
        ally_is_mobile = rng.random() < 0.5
        mobile_capacity, ranged_capacity = (
            (max_n_ally, max_n_enemy) if ally_is_mobile else (max_n_enemy, max_n_ally)
        )
        mobile_count = _bounded_count(rng, 3, 7, mobile_capacity)
        ranged_count = _bounded_count(rng, 3, 7, ranged_capacity)
        mobile = _sample_units(rng, MOBILE, mobile_count)
        if mobile:
            mobile[0] = ASSASSIN
        ranged = _sample_units(rng, RANGED, ranged_count)
        ally, enemy = (mobile, ranged) if ally_is_mobile else (ranged, mobile)

    ally = _cap_units(ally, max_n_ally)
    enemy = _cap_units(enemy, max_n_enemy)
    if not ally or not enemy:
        raise ValueError("Composition generation produced an empty team.")
    return ally, enemy


def _unit_specs() -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in get_all_unit_spec().items()}


def _unit_prices() -> np.ndarray:
    return _unit_specs()["prices"]


def composition_features(ally: Sequence[int], enemy: Sequence[int]) -> dict[str, Any]:
    """Return interpretable composition metadata."""

    prices = _unit_prices()

    def summarize(units: Sequence[int]) -> dict[str, Any]:
        counts = {name: 0 for name in ALL_UNIT_NAMES}
        for unit_id in units:
            counts[ALL_UNIT_NAMES[unit_id]] += 1
        return {
            "count": len(units),
            "unit_counts": {name: count for name, count in counts.items() if count},
            "total_price": float(prices[np.asarray(units)].sum()),
            "melee_ratio": float(
                sum(unit in FRONTLINE + (ASSASSIN,) for unit in units) / len(units)
            ),
            "ranged_ratio": float(sum(unit in RANGED for unit in units) / len(units)),
            "healer_ratio": float(sum(unit in HEALERS for unit in units) / len(units)),
        }

    return {"ally": summarize(ally), "enemy": summarize(enemy)}


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
        "healths": column(specs["healths"][unit_ids].astype(float)),
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
) -> dict[str, Any]:
    """Generate one complete task covering 4.1 A/B/C/D."""

    ally, enemy = sample_compositions(rng, composition, max_n_ally, max_n_enemy)
    scenario, grid_info = build_scenario(rng, ally, enemy, layout, distance, spread, map_scale)
    zone_scenario = build_relational_zones(scenario, grid_info, zone, zone_intensity, zone_relation)
    return {
        "grid_info": grid_info,
        "scenario": scenario,
        "zone_scenario": zone_scenario,
        "metadata": {
            "source": {"kind": "programmatic"},
            "composition_archetype": composition,
            "composition_features": composition_features(ally, enemy),
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
        },
    }
