"""Bitmask rules core shared by the compiled solver and the Python reference.

Every function here is ``numba``-compatible (``@njit``) and takes **plain ints and
numpy arrays**, never a :class:`domino.engine.GameState`, so one copy of the rules
drives both the fast (nopython) search and the slow reference search that verifies
it. The rules are *copied from* :mod:`domino.engine` — never re-invented — and
``tests/test_pimc_rules.py`` pins them against the engine position by position:

* a move is a ``(tile_id, end_index)`` placement, deduplicated exactly like
  ``GameState._placements_cached``: a placement that produces an end pair already
  seen collapses into the lower end index, and a double leaves its end unchanged;
* a seat with no playable tile must pass, and four consecutive passes block the
  deal (``pass_streak >= 4``);
* the deal value (team-0 point differential, matching ``env._deal_reward`` under
  ``reward_mode="differential"``): an emptied hand makes its seat's team bank the
  **other team's** remaining pips; a blocked deal banks the same pips for the
  fewest-pips seat's team; a blocked tie spanning both teams is void (0 points).

Free lead is encoded as ``left < 0`` (the engine's empty ``table``): the played
tile opens the line and exposes its own two values.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ..engine import GameState
from ..tiles import NUM_TILES, TILES, tile_pips

#: Sentinel returned by :func:`terminal_value` for a non-terminal position.
NOT_TERMINAL: int = -1000
#: Value of the blocked-tie (void) terminal: nobody scores, the deal is redealt.
VOID_VALUE: int = 0

# ---- nopython-visible constant tables --------------------------------------
#: Pip values of each tile id, split so the compiled code needs no tuple access.
TILE_A = np.array([a for a, _ in TILES], dtype=np.int8)
TILE_B = np.array([b for _, b in TILES], dtype=np.int8)
TILE_PIPS = np.array([tile_pips(t) for t in range(NUM_TILES)], dtype=np.int32)
#: Team of each seat (``engine.TEAM_OF_SEAT``): seats 0 and 2 are team 0.
TEAM = np.array([0, 1, 0, 1], dtype=np.int8)
#: The two seats of the *other* team than seat ``s``: ``(s + 1) % 4, (s + 3) % 4``.
OPP_A = np.array([1, 2, 3, 0], dtype=np.int8)
OPP_B = np.array([3, 0, 1, 2], dtype=np.int8)
#: ``VALUE_MASK[v]`` = bitmask of the tile ids carrying pip value ``v``.
VALUE_MASK = np.zeros(7, dtype=np.uint32)
for _t, (_a, _b) in enumerate(TILES):
    VALUE_MASK[_a] |= np.uint32(1) << np.uint32(_t)
    VALUE_MASK[_b] |= np.uint32(1) << np.uint32(_t)
del _t, _a, _b
#: All 28 tile bits.
ALL_MASK: np.uint32 = np.uint32((1 << NUM_TILES) - 1)


#: ``POP16[v]`` = number of set bits of ``v`` for a 16-bit ``v`` (64 KiB table).
_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int8)


@njit(cache=True, inline="always")
def popcount32(x: np.uint32) -> int:
    """Number of set bits of a 32-bit mask, as two 16-bit table lookups.

    The usual SWAR bit trick is *not* used here: under this numba build the final
    ``(x * 0x01010101) >> 24`` returned a byte-swapped value (verified by tracing the
    steps against plain Python), and a wrong popcount only degrades move ordering
    silently. The table costs 64 KiB and is exact.
    """
    return np.int64(_POP16[x & np.uint32(0xFFFF)]) + np.int64(_POP16[x >> np.uint32(16)])


@njit(cache=True, inline="always")
def playable_mask(hand: np.uint32, left: int, right: int) -> np.uint32:
    """Tiles of ``hand`` that may be played on the open ends ``(left, right)``."""
    if left < 0:                                     # free lead: every tile plays
        return hand
    return hand & (VALUE_MASK[left] | VALUE_MASK[right])


@njit(cache=True, inline="always")
def tile_placements(tid: int, left: int, right: int,
                    out_l: np.ndarray, out_r: np.ndarray, out_e: np.ndarray) -> int:
    """Fill the deduplicated placements of ``tid`` on ``(left, right)``.

    Returns the number of placements ``n``; for ``i < n`` the child ends are
    ``(out_l[i], out_r[i])`` and the exposed end index is ``out_e[i]``. Mirrors
    ``GameState._placements_cached`` exactly, including its dedup rule.
    """
    a = TILE_A[tid]
    b = TILE_B[tid]
    if left < 0:                                     # free lead: the tile is the line
        out_l[0] = a
        out_r[0] = b
        out_e[0] = 0
        return 1
    n = 0
    if a == b:                                       # a double leaves its end unchanged
        if left == a:
            out_l[0] = a
            out_r[0] = right
            out_e[0] = 0
            n = 1
        if right == a:
            if n == 0:
                out_l[0] = left
                out_r[0] = a
                out_e[0] = 1
                n = 1
            # else: same end pair -> collapsed into the lower end index
    else:
        new_left = -1
        if left == a:
            new_left = b
        elif left == b:
            new_left = a
        if new_left >= 0:
            out_l[0] = new_left
            out_r[0] = right
            out_e[0] = 0
            n = 1
        new_right = -1
        if right == a:
            new_right = b
        elif right == b:
            new_right = a
        if new_right >= 0:
            if n == 0:
                out_l[0] = left
                out_r[0] = new_right
                out_e[0] = 1
                n = 1
            else:
                # The engine dedups on the *sorted* end pair: keep the second
                # placement only when the pair really differs from the first one.
                first_low = out_l[0] if out_l[0] < out_r[0] else out_r[0]
                first_high = out_l[0] if out_l[0] > out_r[0] else out_r[0]
                second_low = left if left < new_right else new_right
                second_high = left if left > new_right else new_right
                if first_low != second_low or first_high != second_high:
                    out_l[1] = left
                    out_r[1] = new_right
                    out_e[1] = 1
                    n = 2
    return n


@njit(cache=True, inline="always")
def emptied_value(seat: int, pips: np.ndarray) -> int:
    """Team-0 differential when ``seat`` emptied its hand (``deal_outcome``)."""
    value = np.int64(pips[OPP_A[seat]] + pips[OPP_B[seat]])
    if TEAM[seat] == 0:
        return value
    return -value


@njit(cache=True, inline="always")
def blocked_value(pips: np.ndarray) -> int:
    """Team-0 differential of a blocked deal (fewest pips wins; cross-team tie voids)."""
    fewest = pips[0]
    for s in range(1, 4):
        if pips[s] < fewest:
            fewest = pips[s]
    winner = -1
    for s in range(4):
        if pips[s] == fewest:
            if winner < 0:
                winner = s
            elif TEAM[s] != TEAM[winner]:
                return VOID_VALUE                     # minimum spans both teams -> void
    value = np.int64(pips[OPP_A[winner]] + pips[OPP_B[winner]])
    if TEAM[winner] == 0:
        return value
    return -value


@njit(cache=True, inline="always")
def terminal_value(hands: np.ndarray, pips: np.ndarray, pass_streak: int) -> int:
    """Team-0 differential if the position is terminal, else :data:`NOT_TERMINAL`.

    The emptied branch is tested first, exactly like ``engine.deal_outcome``.
    """
    for s in range(4):
        if hands[s] == 0:
            return emptied_value(s, pips)
    if pass_streak >= 4:
        return blocked_value(pips)
    return NOT_TERMINAL


@njit(cache=True, inline="always")
def leaf_value(pips: np.ndarray, leaf_mode: int) -> int:
    """Default leaf estimate when the depth budget runs out (team-0 differential).

    ``leaf_mode == 0`` (*"blocked_now"*, the default) estimates the deal by asking
    who would win if it blocked right now — the fewest-pips team banks the other
    team's pips, ties void — which is the terminal rule with the pip counts read
    off the current hands. It is therefore in the reward's own units and range
    (``[-168, 168]``), not a rescaled score.

    ``leaf_mode == 1`` (*"pip_diff"*) is the plain pip-count difference
    ``pips[1] + pips[3] - pips[0] - pips[2]``: the same ordering, a different scale.
    """
    if leaf_mode == 1:
        return np.int64(pips[1] + pips[3] - pips[0] - pips[2])
    mine = pips[0] if pips[0] < pips[2] else pips[2]
    theirs = pips[1] if pips[1] < pips[3] else pips[3]
    if mine < theirs:
        return np.int64(pips[1] + pips[3])
    if mine > theirs:
        return -np.int64(pips[0] + pips[2])
    return np.int64(VOID_VALUE)


def compact(state: GameState) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Convert a :class:`GameState` into the compiled solver's arrays.

    Returns ``(hands, pips, left, right, pass_streak)`` with one ``uint32`` bitmask
    and one pip total per seat, and ``left < 0`` for a free lead, matching
    :func:`playable_mask`.
    """
    hands = np.zeros(4, dtype=np.uint32)
    pips = np.zeros(4, dtype=np.int32)
    for seat, hand in enumerate(state.hands):
        mask = np.uint32(0)
        total = 0
        for tid in hand:
            mask |= np.uint32(1) << np.uint32(int(tid))
            total += tile_pips(int(tid))
        hands[seat] = mask
        pips[seat] = total
    if state.table:
        left, right = int(state.ends[0]), int(state.ends[1])
    else:
        left, right = -1, -1
    return hands, pips, left, right, int(state.pass_streak)
