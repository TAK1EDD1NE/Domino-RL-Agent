"""Seat-relative 192-dim observation encoder (spec section 4).

Turns a :class:`domino.engine.GameState` into the fixed-length ``float32``
vector the policy network sees. The vector contains only public information
plus the acting seat's own hand, and it is *seat-relative*: the four seats are
ordered ``[self, left, partner, right]`` = ``(me+0, me+1, me+2, me+3) % 4``, so
a single network plays every seat from the same layout.

Layout (start:stop, ``me`` = acting seat)::

    0:28    my hand — 1 for each tile id I hold
    28:56   tiles already played this deal (1 = on the table)
    56:168  played_by[4][28] — which relative seat played which tile
    168:169 my remaining pips / TOTAL_PIPS
    169:170 unseen pips (TOTAL_PIPS - my pips - table pips) / TOTAL_PIPS
    170:171 current consecutive-pass streak / 4
    171:175 whether each relative seat passed on its most recent turn
    175:179 tile count in each relative seat's hand / 7
    179:183 one-hot: which relative seat led this deal
    183:184 1 if I am the deal leader
    184:191 per pip value v: 1 if any unseen tile still carries v
    191:192 1 if the board is empty (I am leading right now)

No gymnasium, torch or pygame is imported here: numpy and the engine only.
Encoding is deterministic and uses no global mutable state.
"""
from __future__ import annotations

import functools

import numpy as np

from .engine import GameState
from .tiles import ALL_TILE_IDS, NUM_TILES, TOTAL_PIPS, tile_values

# Public interface -----------------------------------------------------------
OBS_DIM: int = 192
SEATS: int = 4

# Slice constants (single source of truth for the layout).
HAND = slice(0, 28)
PLAYED = slice(28, 56)
PLAYED_BY = slice(56, 168)
MY_PIPS = slice(168, 169)
UNSEEN_PIPS = slice(169, 170)
PASS_STREAK = slice(170, 171)
LAST_PASS = slice(171, 175)
COUNTS = slice(175, 179)
DEAL_LEADER = slice(179, 183)
AM_LEADER = slice(183, 184)
LIVE_VALUE = slice(184, 191)
BOARD_EMPTY = slice(191, 192)


@functools.lru_cache(maxsize=1)
def _tile_value_matrix() -> np.ndarray:
    """Cached ``(NUM_TILES, 7)`` indicator of which pip values each tile carries.

    Built once from the tile table; never mutated, so it is safe to share.
    """
    matrix = np.zeros((NUM_TILES, 7), dtype=np.float32)
    for tid in ALL_TILE_IDS:
        a, b = tile_values(tid)
        matrix[tid, a] = 1.0
        matrix[tid, b] = 1.0
    return matrix


def encode(state: GameState, seat: int) -> np.ndarray:
    """Encode ``state`` from the point of view of ``seat`` as ``float32[192]``.

    ``seat`` is the acting seat (``me``); the four seats are read in the order
    ``[self, left, partner, right] = (seat + r) % 4``.
    """
    vector = np.zeros(OBS_DIM, dtype=np.float32)

    # 0:28 — my hand.
    for tid in state.hands[seat]:
        vector[HAND.start + tid] = 1.0

    # 28:56 — tiles on the table.
    for tid in state.table:
        vector[PLAYED.start + tid] = 1.0

    # 56:168 — played_by[4][28], one 28-block per relative seat.
    base = PLAYED_BY.start
    for r in range(SEATS):
        seat_r = (seat + r) % SEATS
        offset = base + r * NUM_TILES
        for tid in state.played_by[seat_r]:
            vector[offset + tid] = 1.0

    # 168:169 — my remaining pips.
    vector[MY_PIPS.start] = state.pips(seat) / TOTAL_PIPS

    # 169:170 — unseen pips (own hand and table are the only seen tiles).
    unseen = TOTAL_PIPS - state.pips(seat) - state.table_pips()
    vector[UNSEEN_PIPS.start] = unseen / TOTAL_PIPS

    # 170:171 — current consecutive-pass streak.
    vector[PASS_STREAK.start] = state.pass_streak / SEATS

    # 171:175 — who passed on its most recent turn.
    for r in range(SEATS):
        if state.last_pass[(seat + r) % SEATS]:
            vector[LAST_PASS.start + r] = 1.0

    # 175:179 — tile counts per relative seat.
    for r in range(SEATS):
        vector[COUNTS.start + r] = len(state.hands[(seat + r) % SEATS]) / 7

    # 179:183 — one-hot of the relative deal leader.
    vector[DEAL_LEADER.start + (state.leader - seat) % SEATS] = 1.0

    # 183:184 — am I the deal leader.
    if state.leader == seat:
        vector[AM_LEADER.start] = 1.0

    # 184:191 — per pip value, whether any unseen tile still carries it.
    unseen_tiles = state.unseen_tiles()
    if unseen_tiles:
        idx = np.fromiter(unseen_tiles, dtype=np.intp, count=len(unseen_tiles))
        live = _tile_value_matrix()[idx].any(axis=0)
        vector[LIVE_VALUE] = live.astype(np.float32)

    # 191:192 — is the board empty (I am leading right now).
    if not state.table:
        vector[BOARD_EMPTY.start] = 1.0

    return vector


def describe_observation() -> dict[str, tuple[int, int]]:
    """Return the observation layout as ``name -> (start, stop)``.

    The union of all ranges is asserted to be exactly ``0..OBS_DIM-1``.
    """
    layout: dict[str, tuple[int, int]] = {
        "hand": (HAND.start, HAND.stop),
        "played": (PLAYED.start, PLAYED.stop),
        "played_by": (PLAYED_BY.start, PLAYED_BY.stop),
        "my_pips": (MY_PIPS.start, MY_PIPS.stop),
        "unseen_pips": (UNSEEN_PIPS.start, UNSEEN_PIPS.stop),
        "pass_streak": (PASS_STREAK.start, PASS_STREAK.stop),
        "last_pass": (LAST_PASS.start, LAST_PASS.stop),
        "counts": (COUNTS.start, COUNTS.stop),
        "deal_leader": (DEAL_LEADER.start, DEAL_LEADER.stop),
        "am_leader": (AM_LEADER.start, AM_LEADER.stop),
        "live_value": (LIVE_VALUE.start, LIVE_VALUE.stop),
        "board_empty": (BOARD_EMPTY.start, BOARD_EMPTY.stop),
    }
    covered = [i for start, stop in layout.values() for i in range(start, stop)]
    if sorted(covered) != list(range(OBS_DIM)):
        raise AssertionError("observation layout does not cover 0..OBS_DIM-1 exactly once")
    return layout
