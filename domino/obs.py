"""Seat-relative 501-dim observation encoder (spec section 4 + v3 changes).

Turns a :class:`domino.engine.GameState` into the fixed-length ``float32`` vector
the network sees. The vector has a **public part** (public information plus the
acting seat's own hand — everything a policy may act on) and a **privileged part**
(the three opponents' true hands, consumed only by the centralised value network
and by the auxiliary belief head during training).

Layout (start:stop, ``me`` = acting seat):::

    0:28     my hand — 1 for each tile id I hold
    28:56    tiles already played this deal (1 = on the table)
    56:168   played_by[4][28] — which relative seat played which tile (no order)
    168:169  my remaining pips / TOTAL_PIPS
    169:170  unseen pips (TOTAL_PIPS - my pips - table pips) / TOTAL_PIPS
    170:171  current consecutive-pass streak / 4
    171:175  whether each relative seat passed on its most recent turn
    175:179  tile count in each relative seat's hand / 7
    179:183  one-hot: which relative seat led this deal
    183:184  1 if I am the deal leader
    184:191  per pip value v: 1 if any tile outside my hand and the table carries v
    191:192  1 if the board is empty (I am leading right now)
    192:213  void-lack matrix — 3 opponents x 7 values, 1 = known to lack that
             value (from forced passes; monotone, since there is no boneyard)
    213:219  match features — my score/100, opponent score/100, difference/100,
             deal_no/5 (capped), 1 if I lead the match, 1 if I trail
    219:423  ordered last-6 moves, 34 dims each: [mover one-hot 4][tile one-hot
             28][end one-hot 2], oldest first, zero-padded at the front
    423:507  privileged — the three opponents' true hands, 3 x 28 binary, in
             [left, partner, right] relative order

The four seats are always read in the order ``[self, left, partner, right]`` =
``(seat + r) % 4``, and the three opponents in ``[left, partner, right]`` order,
so one network plays every seat from the same layout.

Consumers slice: ``PUBLIC_SLICE`` is the actor's input, ``PRIVILEGED_SLICE`` the
critic's extra input. The policy therefore never sees hidden hands even though a
single observation array carries them.

No gymnasium, torch or pygame is imported here: numpy and the engine only.
Encoding is deterministic and uses no global mutable state.
"""
from __future__ import annotations

import functools

import numpy as np

from .engine import SEATS, TEAM_OF_SEAT, GameState
from .tiles import ALL_TILE_IDS, NUM_TILES, TOTAL_PIPS, tile_values

# ---- public interface ------------------------------------------------------
OBS_DIM_V1: int = 192                      # the original public block
VOID_DIM: int = 3 * 7                      # 3 opponents x 7 pip values
MATCH_DIM: int = 6
MOVES_N: int = 6                           # how many recent moves are encoded
MOVE_DIM: int = 4 + NUM_TILES + 2          # mover one-hot + tile one-hot + end
MOVES_DIM: int = MOVES_N * MOVE_DIM        # 204
PRIVILEGED_DIM: int = 3 * NUM_TILES        # 84
PUBLIC_DIM: int = OBS_DIM_V1 + VOID_DIM + MATCH_DIM + MOVES_DIM   # 423
OBS_DIM: int = PUBLIC_DIM + PRIVILEGED_DIM                        # 507
#: Alias kept for external units written against the v3 plan's earlier numbering.
OBS_DIM_V3: int = OBS_DIM
PUBLIC_SLICE: slice = slice(0, PUBLIC_DIM)
PRIVILEGED_SLICE: slice = slice(PUBLIC_DIM, OBS_DIM)

#: Relative offsets of the other seats, in the documented [left, partner, right] order.
OPPONENT_OFFSETS: tuple[int, int, int] = (1, 2, 3)
#: Match target used to normalise the score features (spec section 2).
SCORE_TARGET: float = 100.0
#: Deals per episode assumed when normalising the deal index (league_full preset).
DEALS_PER_EPISODE_REF: float = 5.0
#: One-hot width of the placement end (left / right).
END_DIM: int = 2

# ---- slice constants (single source of truth for the layout) ---------------
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
VOID_LACK = slice(OBS_DIM_V1, OBS_DIM_V1 + VOID_DIM)
MATCH = slice(VOID_LACK.stop, VOID_LACK.stop + MATCH_DIM)
MOVES = slice(MATCH.stop, MATCH.stop + MOVES_DIM)
PRIVILEGED = slice(MOVES.stop, MOVES.stop + PRIVILEGED_DIM)


def opponent_seats(seat: int) -> tuple[int, int, int]:
    """The other three seats in ``[left, partner, right]`` relative order."""
    return tuple((seat + offset) % SEATS for offset in OPPONENT_OFFSETS)  # type: ignore[return-value]


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


def _encode_public_v1(vector: np.ndarray, state: GameState, seat: int) -> None:
    """Fill the original 192 public features of ``vector`` in place."""
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

    # 184:191 — per pip value, whether any still-unseen tile carries it.
    # "Unseen" = not in my hand and not on the table (i.e. in another seat's
    # hand): with no boneyard this is the complement of hand | table, and is
    # deterministic public information (spec section 10: live[v] = count of
    # such tiles from 168, minus my hand, minus the table).
    seen = set(state.hands[seat]) | set(state.table)
    idx = np.fromiter((t for t in ALL_TILE_IDS if t not in seen),
                      dtype=np.intp)
    if idx.size:
        live = _tile_value_matrix()[idx].any(axis=0)
        vector[LIVE_VALUE] = live.astype(np.float32)

    # 191:192 — is the board empty (I am leading right now).
    if not state.table:
        vector[BOARD_EMPTY.start] = 1.0


def _encode_void_lacks(vector: np.ndarray, state: GameState, seat: int) -> None:
    """Fill the 3x7 void-lack matrix (values each opponent is known to lack).

    A forced pass is public proof that the seat held no tile carrying either open
    value; :meth:`GameState.pass_turn` accumulates those values per seat, and the
    lack is monotone because hands only shrink (no boneyard).
    """
    for r, other in enumerate(opponent_seats(seat)):
        base = VOID_LACK.start + r * 7
        for value in state.lacks[other]:
            vector[base + int(value)] = 1.0


def _encode_match(vector: np.ndarray, state: GameState, seat: int) -> None:
    """Fill the match-score block (team-relative, normalised)."""
    my_team = TEAM_OF_SEAT[seat]
    opponent_team = 1 - my_team
    mine = float(state.scores[my_team])
    theirs = float(state.scores[opponent_team])
    base = MATCH.start
    vector[base + 0] = mine / SCORE_TARGET
    vector[base + 1] = theirs / SCORE_TARGET
    vector[base + 2] = (mine - theirs) / SCORE_TARGET
    vector[base + 3] = min(float(state.deal_no), DEALS_PER_EPISODE_REF) / DEALS_PER_EPISODE_REF
    vector[base + 4] = 1.0 if mine > theirs else 0.0
    vector[base + 5] = 1.0 if mine < theirs else 0.0


def _encode_moves(vector: np.ndarray, state: GameState, seat: int) -> None:
    """Fill the ordered last-N move block (oldest first, zero-padded at the front)."""
    history = list(state.moves[-MOVES_N:])
    padded: list[tuple[int, int, int] | None] = [None] * (MOVES_N - len(history)) + history
    for i, move in enumerate(padded):
        if move is None:
            continue
        mover, tid, end = int(move[0]), int(move[1]), int(move[2])
        base = MOVES.start + i * MOVE_DIM
        vector[base + (mover - seat) % SEATS] = 1.0
        vector[base + 4 + tid] = 1.0
        vector[base + 4 + NUM_TILES + (1 if end else 0)] = 1.0


def _encode_privileged(
    vector: np.ndarray,
    state: GameState,
    seat: int,
    hands_override: list[list[int]] | None,
) -> None:
    """Fill the privileged block: the three opponents' true (or sampled) hands.

    ``hands_override`` supplies three hands in ``[left, partner, right]`` order —
    used by search to evaluate a determinisation instead of the true state.
    """
    if hands_override is None:
        hands = [state.hands[other] for other in opponent_seats(seat)]
    else:
        if len(hands_override) != 3:
            raise ValueError(f"hands_override must hold 3 hands, got {len(hands_override)}")
        hands = hands_override
    for r, hand in enumerate(hands):
        base = PRIVILEGED.start + r * NUM_TILES
        for tid in hand:
            vector[base + int(tid)] = 1.0


def encode(
    state: GameState,
    seat: int,
    hands_override: list[list[int]] | None = None,
) -> np.ndarray:
    """Encode ``state`` from the point of view of ``seat`` as ``float32[501]``.

    ``seat`` is the acting seat (``me``); the four seats are read in the order
    ``[self, left, partner, right] = (seat + r) % 4`` and the opponents in
    ``[left, partner, right]`` order. ``hands_override`` replaces the privileged
    block with sampled hands (search/determinisation only).
    """
    vector = np.zeros(OBS_DIM, dtype=np.float32)
    _encode_public_v1(vector, state, seat)
    _encode_void_lacks(vector, state, seat)
    _encode_match(vector, state, seat)
    _encode_moves(vector, state, seat)
    _encode_privileged(vector, state, seat, hands_override)
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
        "void_lack": (VOID_LACK.start, VOID_LACK.stop),
        "match": (MATCH.start, MATCH.stop),
        "moves": (MOVES.start, MOVES.stop),
        "privileged_hands": (PRIVILEGED.start, PRIVILEGED.stop),
    }
    covered = [i for start, stop in layout.values() for i in range(start, stop)]
    if sorted(covered) != list(range(OBS_DIM)):
        raise AssertionError("observation layout does not cover 0..OBS_DIM-1 exactly once")
    return layout
