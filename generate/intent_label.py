"""Intent labels for TABX explicit communication.

Packed ``msg_id`` is mode + global focus unit id. Annotate writes both
teacher (reference) and behavior action streams; MessageHead GT is teacher.
"""

from __future__ import annotations

import numpy as np

# Keep in sync with src.tabx.constants.UnitAction
UNIT_ACTION_ATTACK = 4
UNIT_ACTION_IDLE = 7

MODE_IDLE = 0
MODE_REPOSITION = 1
MODE_ATTACK = 2
N_INTENT_MODES = 3

MSG_PAD_ID = 0  # pack(idle, none); always masked by visible_ally_valid
ALLY_ID_PAD = -1
MOVE_NONE = 0  # not walking / not turning (idle, attack, dead, unknown)

DEFAULT_MAX_N_UNITS = 20
N_MOVE = 7  # 0=none, 1-6 = UP/DOWN/LEFT/RIGHT/TURN_R/TURN_L

# Env UnitAction 0-7 (+ airsoul dead=8) → move id.
# UP=0, DOWN=1, LEFT=2, RIGHT=3, ATTACK=4, TURN_RIGHT=5, TURN_LEFT=6, IDLE=7
_MOVE_FROM_ACTION = np.array([1, 2, 3, 4, 0, 5, 6, 0, 0], dtype=np.int32)


def msg_vocab_size(max_n_units: int = DEFAULT_MAX_N_UNITS) -> int:
    return int(N_INTENT_MODES) * (1 + int(max_n_units))


def pack_intent(
    mode: np.ndarray | int,
    focus: np.ndarray | int,
    *,
    max_n_units: int = DEFAULT_MAX_N_UNITS,
) -> np.ndarray | int:
    """Pack mode in {0,1,2} and focus in {-1} ∪ {0..max_n_units-1}."""

    max_n = int(max_n_units)
    span = 1 + max_n
    mode_a = np.asarray(mode, dtype=np.int32)
    focus_a = np.asarray(focus, dtype=np.int32)
    focus_a = np.clip(focus_a, -1, max_n - 1)
    packed = mode_a * span + (focus_a + 1)
    packed = np.clip(packed, 0, msg_vocab_size(max_n) - 1).astype(np.int32)
    if np.isscalar(mode) and np.isscalar(focus):
        return int(packed.reshape(()))
    return packed


def unpack_intent(
    msg_id: np.ndarray | int,
    *,
    max_n_units: int = DEFAULT_MAX_N_UNITS,
) -> tuple[np.ndarray, np.ndarray]:
    max_n = int(max_n_units)
    span = 1 + max_n
    mid = np.asarray(msg_id, dtype=np.int32)
    mode = mid // span
    focus = (mid % span) - 1
    return mode.astype(np.int32), focus.astype(np.int32)


def intent_from_reference(
    ref_action: np.ndarray,
    attack_target: np.ndarray,
    alive: np.ndarray | None = None,
    *,
    max_n_units: int = DEFAULT_MAX_N_UNITS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-step teacher intent.

    ``ref_action``: [...] int, env actions 0–7
    ``attack_target``: [...] int, global unit_keys index (broadcastable to ref)
    ``alive``: optional bool mask; dead steps keep a pad id (loss-masked later)

    Returns ``mode, focus, msg_id`` with the same leading shape as ``ref_action``.
    """

    ref = np.asarray(ref_action, dtype=np.int32)
    atk = np.asarray(attack_target, dtype=np.int32)
    if atk.shape != ref.shape:
        atk = np.broadcast_to(atk, ref.shape).copy()
    mode = np.full(ref.shape, MODE_REPOSITION, dtype=np.int32)
    mode[ref == UNIT_ACTION_IDLE] = MODE_IDLE
    mode[ref == UNIT_ACTION_ATTACK] = MODE_ATTACK
    focus = np.full(ref.shape, -1, dtype=np.int32)
    is_atk = mode == MODE_ATTACK
    focus[is_atk] = np.clip(atk[is_atk], 0, int(max_n_units) - 1)
    msg_id = pack_intent(mode, focus, max_n_units=max_n_units)
    if alive is not None:
        alive_b = np.asarray(alive) > 0.5
        msg_id = np.where(alive_b, msg_id, np.int32(MSG_PAD_ID))
        mode = np.where(alive_b, mode, np.int32(MODE_IDLE))
        focus = np.where(alive_b, focus, np.int32(-1))
    return mode, focus, np.asarray(msg_id, dtype=np.int32)


def move_from_action(action: np.ndarray | int) -> np.ndarray | int:
    """Map env action to ``{0=not-moving, 1-6}``.

    1-6 follow UnitAction UP/DOWN/LEFT/RIGHT/TURN_RIGHT/TURN_LEFT.
    IDLE, ATTACK, dead, and out-of-range ids map to 0.
    """

    scalar = np.isscalar(action)
    a = np.asarray(action, dtype=np.int32)
    table = _MOVE_FROM_ACTION
    clipped = np.clip(a, 0, int(table.shape[0]) - 1)
    out = table[clipped]
    if scalar:
        return int(np.asarray(out).reshape(()))
    return out.astype(np.int32)


def visible_ally_messages(
    *,
    receiver_index: int,
    ally_indices: np.ndarray,
    ally_unit_ids: np.ndarray,
    msg_all: np.ndarray,
    visible_matrix: np.ndarray,
    alive_all: np.ndarray,
    receiver_alive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-receiver teammate teacher messages in stable other-ally order.

    ``ally_indices``: agent_index of every *real* ally including receiver, sorted.
    ``msg_all`` / ``alive_all``: [T, n_schema_ally] (or [T, n_real] if already
    filtered to ``ally_indices`` columns — then pass matching unit ids).
    ``visible_matrix``: [T, n_units, n_units]
    ``ally_unit_ids``: [n_real] global unit id for each entry of ``ally_indices``.

    Returns ``msg [T, n_other], unit_id [T, n_other], valid [T, n_other]``.
    """

    recv = int(receiver_index)
    others = [int(a) for a in np.asarray(ally_indices).tolist() if int(a) != recv]
    t = int(msg_all.shape[0])
    n_other = len(others)
    out_msg = np.full((t, n_other), MSG_PAD_ID, dtype=np.int32)
    out_id = np.full((t, n_other), ALLY_ID_PAD, dtype=np.int32)
    out_valid = np.zeros((t, n_other), dtype=np.uint8)
    if n_other == 0:
        return out_msg, out_id, out_valid

    idx_map = {int(a): i for i, a in enumerate(np.asarray(ally_indices).tolist())}
    recv_col = idx_map[recv]
    recv_uid = int(ally_unit_ids[recv_col])
    vis = np.asarray(visible_matrix)
    alive_r = np.asarray(receiver_alive).reshape(t) > 0.5

    for slot, other in enumerate(others):
        col = idx_map[other]
        uid = int(ally_unit_ids[col])
        alive_j = np.asarray(alive_all[:, col]).reshape(t) > 0.5
        can_see = vis[:, recv_uid, uid] > 0.5
        valid = alive_r & alive_j & can_see
        out_valid[:, slot] = valid.astype(np.uint8)
        out_msg[:, slot] = np.where(valid, msg_all[:, col], np.int32(MSG_PAD_ID))
        out_id[:, slot] = np.where(valid, np.int32(uid), np.int32(ALLY_ID_PAD))
    return out_msg, out_id, out_valid
