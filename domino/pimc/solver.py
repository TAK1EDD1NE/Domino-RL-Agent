"""Phase-1 perfect-information solver: exact minimax over the *complete* deal.

The solver is given a fully known position (four hands, both open ends, the seat to
move, the pass streak) and returns

* the optimal **team point differential** of the position — the exact quantity the
  environment pays at the end of the deal (``env._deal_reward`` under
  ``reward_mode="differential"``), so no rescaling happens anywhere, and
* the value of **every legal move**, searched with a full window so each entry is a
  true value rather than an alpha-beta bound.

Two backends share one copy of the rules (:mod:`domino.pimc.rules`):

``Solver``
    The fast backend: a recursive alpha-beta search compiled with numba over
    28-bit hand bitmasks, with a size-capped always-replace transposition table.
    ``leaf_mode`` selects a compiled leaf estimate (the numba core cannot call back
    into Python). Used for the exact solve, the benchmarks and PIMC's inner search.
``reference_solve``
    The plain-minimax reference: no pruning, no table, and — unlike the compiled
    core — a **pluggable** ``leaf_eval(view)`` callable, which is the Phase-1
    contract for a learned or hand-written leaf estimate. Slow on purpose: it is
    the anchor the compiled search is verified against.

Conventions
-----------
* Values are always the **team-0 point differential** of the deal (positive = team 0
  banks points). ``SolveResult.value_to_move`` flips the sign for the team to move.
* ``max_depth`` counts **plies** (a play or a pass) from the root; ``None`` means
  exact (search the deal to its terminal) and is the default. A depth-limited
  search never reports ``exact=True``, and an exact search never evaluates a leaf
  (``leaf_evals == 0`` is asserted in the tests).
* A blocked deal whose fewest-pips minimum spans both teams is void: 0 points, and
  the environment redeals it, so the solver returns 0 for it.
* Determinism: the search reads no clock and no RNG. ``node_limit`` is a *node*
  budget (deterministic); a search that exceeds it is reported ``complete=False``
  and its values must be discarded — that is the hook Phase 3 uses for its
  per-move time budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numba import njit

from ..engine import GameState, TEAM_OF_SEAT
from ..masks import action_index
from ..tiles import NUM_TILES
from . import rules as R

#: Largest number of plies the scratch stack is dimensioned for (28 plays plus up
#: to three passes before each of them is 109; 128 leaves head room).
MAX_PLY: int = 128
#: Depth budget that means "search to the terminal" (never reached: a deal has at
#: most 109 plies).
EXACT_DEPTH: int = 1000
#: Search bound for alpha/beta windows.
INF: int = 1 << 30
#: Transposition-table probe length before an entry is replaced (always-replace).
PROBE_LIMIT: int = 8
#: Empty-slot sentinel of the table's meta word (meta values are < 1024).
EMPTY_META: np.uint32 = np.uint32(0xFFFFFFFF)
#: Value offset keeping the packed score non-negative (values live in [-168, 168]).
VALUE_OFFSET: int = 200

LEAF_MODES: dict[str, int] = {"blocked_now": 0, "pip_diff": 1}


# ---------------------------------------------------------------------------
# compiled core
# ---------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _mix(h0: np.uint32, h1: np.uint32, h2: np.uint32, h3: np.uint32,
         meta: np.uint32) -> np.int64:
    """Mix the five key words into a 63-bit hash (splitmix-style, wrapping).

    The result is signed on purpose: ``int()`` inside a numba kernel resolves to an
    *unsigned* cast in this toolchain, which silently typed the table slot as uint64
    (and unified the search's return type to float). Everything downstream of the
    hash therefore stays in ``np.int64``.
    """
    h = np.uint64(0x9E3779B97F4A7C15)
    h ^= (np.uint64(h0) + np.uint64(0x9E3779B9)) * np.uint64(0xBF58476D1CE4E5B9)
    h ^= (np.uint64(h1) + np.uint64(0x94D049BB)) * np.uint64(0x94D049BB133111EB)
    h ^= (np.uint64(h2) + np.uint64(0x85EBCA6B)) * np.uint64(0xC2B2AE3D27D4EB4F)
    h ^= (np.uint64(h3) + np.uint64(0x27D4EB2F)) * np.uint64(0x165667B19E3779F9)
    h ^= (np.uint64(meta) + np.uint64(0x9E3779B1)) * np.uint64(0x2545F4914F6CDD1D)
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return np.int64(h & np.uint64(0x7FFFFFFFFFFFFFFF))


@njit(cache=True, inline="always")
def _pack_meta(left: int, right: int, turn: int, pass_streak: int) -> np.uint32:
    """Pack ``(ends, turn, pass_streak)`` into the key's meta word.

    ``left < 0`` marks a free lead and encodes as 0, which no real end can produce
    (``left + 1`` is 1..7), so the free-lead position has its own key.
    """
    return np.uint32((left + 1) | ((right + 1) << 3) | (turn << 6) | (pass_streak << 8))


@njit(cache=True, inline="always")
def _tt_find(k0, k1, k2, k3, k4, h0: np.uint32, h1: np.uint32, h2: np.uint32,
             h3: np.uint32, meta: np.uint32, tmask: int, found: np.ndarray) -> int:
    """Return the slot for the key; set ``found[7]`` to 1 when the key is present.

    The hit flag travels through the caller's counter array instead of a tuple
    return: numba types the two-element tuple badly here (it unified the slot to a
    float index and refused to type the array lookup).
    """
    found[7] = 0
    start = _mix(h0, h1, h2, h3, meta) & tmask
    slot = start
    for i in range(PROBE_LIMIT):
        idx = (start + i) & tmask
        word = k4[idx]
        if word == EMPTY_META:
            return idx
        if (word == meta and k0[idx] == h0 and k1[idx] == h1
                and k2[idx] == h2 and k3[idx] == h3):
            found[7] = 1
            return idx
        slot = idx
    return slot


@njit(cache=True, inline="always")
def _tt_pack(value: int, flag: int, depth: int, tile: int, end: int, unbounded: int) -> np.int32:
    """Pack a table entry into 24 bits: value | flag | depth | best move | unbounded."""
    stored_depth = depth if depth < 63 else 63
    if tile < 0:
        tile = 31
    packed = (value + VALUE_OFFSET) & 0x1FF
    packed |= (flag & 3) << 9
    packed |= (stored_depth & 63) << 11
    packed |= (tile & 31) << 17
    packed |= (end & 1) << 22
    packed |= (unbounded & 1) << 23
    return np.int32(packed)


@njit(cache=False)   # caching a recursive numba kernel is not supported: with
#                      ``cache=True`` the compiled search crashed with an access
#                      violation on the first forced-pass position, while the same
#                      code ran correctly from a cold compile. Kept uncached on purpose.
def _search(hands: np.ndarray, ends: np.ndarray, turn: int, pass_streak: int,
            pips: np.ndarray, depth: int, alpha: int, beta: int, ply: int,
            sc_scores: np.ndarray, sc_moves: np.ndarray,
            sc_l: np.ndarray, sc_r: np.ndarray, sc_e: np.ndarray,
            k0: np.ndarray, k1: np.ndarray, k2: np.ndarray, k3: np.ndarray,
            k4: np.ndarray, tval: np.ndarray, tmask: int,
            counters: np.ndarray, leaf_mode: int, tt_enabled: bool,
            node_limit: int) -> int:
    """Team-0 differential of the position, searched to ``depth`` more plies.

    ``hands`` (uint32 masks), ``pips`` (per-seat pip totals) and ``ends`` are
    mutated in place and restored on the way out, so the recursion allocates
    nothing. ``counters`` = [nodes, tt_found_entries, tt_probes, leaf_evals, aborted,
    cutoffs, node_limit_hits, table-probe hit flag] — slot 7 doubles as
    :func:`_tt_find`'s output flag, which is why the table probe does not return a
    tuple (numba typed that tuple's slot as a float index).
    """
    counters[0] += 1
    if counters[4] == 1:
        return 0
    if counters[0] > node_limit:
        counters[4] = 1
        counters[6] += 1
        return 0
    terminal = R.terminal_value(hands, pips, pass_streak)
    if terminal != R.NOT_TERMINAL:
        return terminal
    if depth <= 0:
        counters[3] += 1
        return R.leaf_value(pips, leaf_mode)

    left = ends[0]
    right = ends[1]
    maximizing = R.TEAM[turn] == 0
    alpha0 = alpha
    beta0 = beta
    tt_tile = -1
    tt_end = 0

    if tt_enabled:
        counters[2] += 1
        meta = _pack_meta(left, right, turn, pass_streak)
        slot = _tt_find(k0, k1, k2, k3, k4, hands[0], hands[1], hands[2], hands[3],
                        meta, tmask, counters)
        if counters[7] == 1:
            counters[1] += 1
            packed = np.int64(tval[slot])
            entry_value = (packed & 0x1FF) - VALUE_OFFSET
            entry_flag = (packed >> 9) & 3
            entry_depth = (packed >> 11) & 63
            entry_unbounded = (packed >> 23) & 1
            tt_tile = (packed >> 17) & 31
            tt_end = (packed >> 22) & 1
            if entry_unbounded == 1 or entry_depth >= depth:
                if entry_flag == 0:
                    return entry_value
                if entry_flag == 1:                       # lower bound
                    if entry_value >= beta:
                        counters[5] += 1
                        return entry_value
                    if entry_value > alpha:
                        alpha = entry_value
                else:                                      # upper bound
                    if entry_value <= alpha:
                        counters[5] += 1
                        return entry_value
                    if entry_value < beta:
                        beta = entry_value
        else:
            tt_tile = -1

    # ---- move generation + ordering ----------------------------------------
    cand = R.playable_mask(hands[turn], left, right)
    if cand == 0:                                          # mandatory pass
        value = _search(hands, ends, (turn + 1) & 3, pass_streak + 1, pips, depth - 1,
                        alpha, beta, ply + 1, sc_scores, sc_moves, sc_l, sc_r, sc_e,
                        k0, k1, k2, k3, k4, tval, tmask, counters, leaf_mode,
                        tt_enabled, node_limit)
        return value

    n_moves = 0
    o1 = R.OPP_A[turn]
    o2 = R.OPP_B[turn]
    score_row = sc_scores[ply]
    move_row = sc_moves[ply]
    for tid in range(NUM_TILES):
        if (cand >> np.uint32(tid)) & np.uint32(1) == 0:
            continue
        count = R.tile_placements(tid, left, right, sc_l[ply], sc_r[ply], sc_e[ply])
        for i in range(count):
            child_l = sc_l[ply, i]
            child_r = sc_r[ply, i]
            after = R.VALUE_MASK[child_l] | R.VALUE_MASK[child_r]
            options = R.popcount32(hands[o1] & after) + R.popcount32(hands[o2] & after)
            move_row[n_moves, 0] = tid
            move_row[n_moves, 1] = sc_e[ply, i]
            move_row[n_moves, 2] = child_l
            move_row[n_moves, 3] = child_r
            score_row[n_moves] = np.float32(R.TILE_PIPS[tid] - 8 * options)
            n_moves += 1
    if tt_tile >= 0:                                       # table move first
        for i in range(n_moves):
            if move_row[i, 0] == tt_tile:
                score_row[i] = np.float32(1e9)
                break
    for i in range(1, n_moves):                            # insertion sort, best first
        key_score = score_row[i]
        k_tile = move_row[i, 0]
        k_end = move_row[i, 1]
        k_l = move_row[i, 2]
        k_r = move_row[i, 3]
        j = i - 1
        while j >= 0 and score_row[j] < key_score:
            score_row[j + 1] = score_row[j]
            move_row[j + 1, 0] = move_row[j, 0]
            move_row[j + 1, 1] = move_row[j, 1]
            move_row[j + 1, 2] = move_row[j, 2]
            move_row[j + 1, 3] = move_row[j, 3]
            j -= 1
        score_row[j + 1] = key_score
        move_row[j + 1, 0] = k_tile
        move_row[j + 1, 1] = k_end
        move_row[j + 1, 2] = k_l
        move_row[j + 1, 3] = k_r

    # ---- search the moves --------------------------------------------------
    best = -INF if maximizing else INF
    best_tile = -1
    best_end = 0
    for i in range(n_moves):
        tid = move_row[i, 0]
        end = move_row[i, 1]
        child_l = move_row[i, 2]
        child_r = move_row[i, 3]
        bit = np.uint32(1) << np.uint32(tid)
        hands[turn] ^= bit
        pips[turn] -= R.TILE_PIPS[tid]
        ends[0] = child_l
        ends[1] = child_r
        value = _search(hands, ends, (turn + 1) & 3, 0, pips, depth - 1, alpha, beta,
                        ply + 1, sc_scores, sc_moves, sc_l, sc_r, sc_e,
                        k0, k1, k2, k3, k4, tval, tmask, counters, leaf_mode,
                        tt_enabled, node_limit)
        hands[turn] ^= bit
        pips[turn] += R.TILE_PIPS[tid]
        ends[0] = left
        ends[1] = right
        if counters[4] == 1:
            return 0
        if maximizing:
            if value > best:
                best = value
                best_tile = tid
                best_end = end
                if best > alpha:
                    alpha = best
        else:
            if value < best:
                best = value
                best_tile = tid
                best_end = end
                if best < beta:
                    beta = best
        if alpha >= beta:
            counters[5] += 1
            break

    if tt_enabled and counters[4] == 0 and best_tile >= 0:
        if best <= alpha0:
            flag = 2                                       # upper bound
        elif best >= beta0:
            flag = 1                                       # lower bound
        else:
            flag = 0                                       # exact
        meta = _pack_meta(left, right, turn, pass_streak)
        slot = _tt_find(k0, k1, k2, k3, k4, hands[0], hands[1], hands[2], hands[3],
                        meta, tmask, counters)
        k0[slot] = hands[0]
        k1[slot] = hands[1]
        k2[slot] = hands[2]
        k3[slot] = hands[3]
        k4[slot] = meta
        tval[slot] = _tt_pack(best, flag, depth, best_tile, best_end,
                              1 if depth >= EXACT_DEPTH else 0)
    return best


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class SolveResult:
    """Outcome of one solve: the position's value and the value of its moves.

    ``value`` is the **team-0** point differential of the deal; ``value_to_move`` is
    the same number from the team-to-move's perspective (what that team can expect
    to bank). ``move_values`` holds one entry per legal move, keyed by
    ``(tile_id, end_index)`` exactly as ``GameState.legal_moves`` exposes them.
    """

    value: int
    value_to_move: int
    seat: int
    team_to_move: int
    move_values: dict[tuple[int, int], int] = field(default_factory=dict)
    best_move: Optional[tuple[int, int]] = None
    best_action: Optional[int] = None
    passed: bool = False
    depth: Optional[int] = None
    exact: bool = False
    complete: bool = True
    nodes: int = 0
    leaf_evals: int = 0
    tt_hits: int = 0
    tt_probes: int = 0
    cutoffs: int = 0
    elapsed_s: float = 0.0
    backend: str = "numba"
    leaf_mode: str = "blocked_now"

    def regret(self, move: tuple[int, int]) -> int:
        """How many points the move gives up against the best alternative (>= 0)."""
        return int(self.value_to_move_side(self.best_move) - self.value_to_move_side(move))

    def value_to_move_side(self, move: Optional[tuple[int, int]]) -> int:
        """Value of ``move`` from the moving team's perspective."""
        if move is None or move not in self.move_values:
            return 0
        raw = self.move_values[move]
        return raw if self.team_to_move == 0 else -raw


@dataclass(frozen=True)
class LeafView:
    """State handed to a pluggable ``leaf_eval`` callable (reference backend).

    Deliberately a frozen plain snapshot: masks, pip totals, the open ends and the
    seat to move — everything a leaf estimate may use, and nothing it must not
    (no table, no history, no RNG).
    """

    hands: tuple[int, int, int, int]
    pips: tuple[int, int, int, int]
    left: int
    right: int
    turn: int
    pass_streak: int

    @property
    def team_to_move(self) -> int:
        """Team of the seat to move."""
        return int(TEAM_OF_SEAT[self.turn])


def _root_expand(state: GameState) -> tuple[int, list[tuple[int, int]]]:
    """Return ``(seat, legal moves)`` for a position the solver may be called on."""
    if state.is_terminal():
        raise ValueError("solve() called on a terminal position")
    seat = int(state.turn)
    return seat, list(state.legal_moves(seat))


def _placements_py(tid: int, left: int, right: int) -> list[tuple[int, int, int]]:
    """``(child_left, child_right, end_index)`` placements of ``tid``, engine order.

    The compiled rule is used, so the Python-side root expansion and the search can
    never disagree about which placements exist or which end index they expose.
    """
    out_l = np.zeros(2, dtype=np.int32)
    out_r = np.zeros(2, dtype=np.int32)
    out_e = np.zeros(2, dtype=np.int32)
    count = int(R.tile_placements(int(tid), int(left), int(right), out_l, out_r, out_e))
    return [(int(out_l[i]), int(out_r[i]), int(out_e[i])) for i in range(count)]


def _finish(result: SolveResult, move_values: dict[tuple[int, int], int],
            passed: bool, team_to_move: int) -> SolveResult:
    """Select the best move (lowest action index on ties) and fill the result in.

    "Best" is the best **for the team to move**: at a node where team 0 is to move that
    is the largest team-0 differential, at a node where team 1 is to move it is the
    smallest one. Selecting on the raw value would report — and play — the *opponent's*
    preferred move whenever the seat to act is on team 1.
    """
    result.move_values = move_values
    result.passed = passed
    result.team_to_move = team_to_move
    if move_values:
        def own_value(move: tuple[int, int]) -> int:
            raw = int(move_values[move])
            return raw if team_to_move == 0 else -raw

        best_move = max(move_values, key=lambda m: (own_value(m), -action_index(*m)))
        result.best_move = best_move
        result.best_action = action_index(*best_move)
        raw = int(move_values[best_move])
    else:
        best_move = None
        raw = int(result.value)
    result.value = raw
    result.value_to_move = raw if team_to_move == 0 else -raw
    return result


def reference_solve(state: GameState, *, max_depth: Optional[int] = None,
                    leaf_eval: Optional[Callable[[LeafView], float]] = None,
                    node_limit: Optional[int] = None) -> SolveResult:
    """Plain minimax reference: no pruning, no table, pluggable ``leaf_eval``.

    This is the correctness anchor for :class:`Solver`: the same rules module, but
    an obviously-correct exhaustive recursion memoised on ``(state, depth)``. Slow
    by construction — it exists to be compared against, and to provide the
    pluggable-callable leaf evaluator the compiled backend cannot host.
    """
    seat, moves = _root_expand(state)
    hands, pips, left, right, pass_streak = R.compact(state)
    team_to_move = int(TEAM_OF_SEAT[seat])
    if max_depth is None:
        depth_budget = EXACT_DEPTH
    else:
        depth_budget = int(max_depth)
        if depth_budget < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")
        if depth_budget >= EXACT_DEPTH:
            raise ValueError(f"max_depth must be < {EXACT_DEPTH}, got {max_depth}")
    counters = {"nodes": 0, "leaf_evals": 0, "aborted": 0}
    limit = 100_000_000 if node_limit is None else int(node_limit)
    memo: dict[tuple, int] = {}
    pl_l = np.zeros(2, dtype=np.int32)
    pl_r = np.zeros(2, dtype=np.int32)
    pl_e = np.zeros(2, dtype=np.int32)
    #: One mutable box for the position the recursion walks over (arrays are mutated
    #: in place and restored, so a node allocates nothing).
    box = {"left": int(left), "right": int(right), "turn": int(seat), "pass": int(pass_streak)}

    def rec(depth: int) -> int:
        """Team-0 differential of the box position with ``depth`` plies left."""
        counters["nodes"] += 1
        if counters["nodes"] > limit:
            counters["aborted"] = 1
            return 0
        terminal = int(R.terminal_value(hands, pips, box["pass"]))
        if terminal != R.NOT_TERMINAL:
            return terminal
        if depth <= 0:
            counters["leaf_evals"] += 1
            if leaf_eval is None:
                return int(R.leaf_value(pips, LEAF_MODES["blocked_now"]))
            view = LeafView(tuple(int(h) for h in hands), tuple(int(p) for p in pips),
                            box["left"], box["right"], box["turn"], box["pass"])
            return int(round(float(leaf_eval(view))))
        key = (int(hands[0]), int(hands[1]), int(hands[2]), int(hands[3]),
               box["left"], box["right"], box["turn"], box["pass"], depth)
        cached = memo.get(key)
        if cached is not None:
            return cached
        turn = box["turn"]
        cand = int(R.playable_mask(hands[turn], box["left"], box["right"]))
        if cand == 0:                                       # mandatory pass
            box["turn"] = (turn + 1) & 3
            box["pass"] += 1
            value = rec(depth - 1)
            box["turn"] = turn
            box["pass"] -= 1
            memo[key] = value
            return value
        maximizing = int(R.TEAM[turn]) == 0
        best: Optional[int] = None
        for tid in range(NUM_TILES):
            if (cand >> tid) & 1 == 0:
                continue
            count = int(R.tile_placements(tid, box["left"], box["right"], pl_l, pl_r, pl_e))
            # Snapshot the placements *before* recursing: ``pl_*`` are shared scratch
            # arrays and the recursive call clobbers them, so reading ``pl_l[i]`` after
            # ``rec`` returned handed the search a stale (wrong) child position.
            placements = [(int(pl_l[i]), int(pl_r[i])) for i in range(count)]
            for child_left, child_right in placements:
                saved_left, saved_right = box["left"], box["right"]
                saved_pass = box["pass"]
                box["left"], box["right"] = child_left, child_right
                box["pass"] = 0                             # a play resets the streak
                hands[turn] ^= np.uint32(1) << np.uint32(tid)
                pips[turn] -= int(R.TILE_PIPS[tid])
                box["turn"] = (turn + 1) & 3
                value = rec(depth - 1)
                box["turn"] = turn
                hands[turn] ^= np.uint32(1) << np.uint32(tid)
                pips[turn] += int(R.TILE_PIPS[tid])
                box["left"], box["right"] = saved_left, saved_right
                box["pass"] = saved_pass
                if counters["aborted"] == 1:
                    return 0
                if best is None:
                    best = value
                elif maximizing:
                    best = value if value > best else best
                else:
                    best = value if value < best else best
        assert best is not None, "a hand with a playable tile always yields a move"
        memo[key] = best
        return best

    started = time.perf_counter()
    move_values: dict[tuple[int, int], int] = {}

    def make_result(value: int = 0) -> SolveResult:
        aborted = counters["aborted"] == 1
        return SolveResult(value=int(value), value_to_move=0, seat=seat,
                           team_to_move=team_to_move, nodes=counters["nodes"],
                           leaf_evals=counters["leaf_evals"], complete=not aborted,
                           depth=max_depth, exact=(max_depth is None and not aborted),
                           elapsed_s=time.perf_counter() - started, backend="python")

    if not moves:                                           # forced pass at the root
        box["turn"] = (seat + 1) & 3
        box["pass"] += 1
        value = rec(depth_budget - 1)
        box["turn"] = seat
        box["pass"] -= 1
        return _finish(make_result(value), {}, True, team_to_move)
    for tid, end in moves:
        placements = _placements_py(int(tid), box["left"], box["right"])
        chosen = next(((cl, cr) for cl, cr, e in placements if e == int(end)), None)
        if chosen is None:                                  # engine tolerates a wrong end
            chosen = (placements[0][0], placements[0][1])
        box["left"], box["right"] = chosen
        box["pass"] = 0                                     # a play resets the streak
        hands[seat] ^= np.uint32(1) << np.uint32(tid)
        pips[seat] -= int(R.TILE_PIPS[tid])
        box["turn"] = (seat + 1) & 3
        move_values[(int(tid), int(end))] = rec(depth_budget - 1)
        box["turn"] = seat
        hands[seat] ^= np.uint32(1) << np.uint32(tid)
        pips[seat] += int(R.TILE_PIPS[tid])
        box["left"], box["right"] = int(left), int(right)
        if counters["aborted"] == 1:
            break
    return _finish(make_result(), move_values, False, team_to_move)


# ---------------------------------------------------------------------------
# the fast backend
# ---------------------------------------------------------------------------
class Solver:
    """Alpha-beta solver over hand bitmasks with a size-capped transposition table.

    Parameters
    ----------
    max_depth:
        Ply budget from the root; ``None`` (default) solves the deal exactly.
    leaf_eval:
        ``None`` or a mode name (``"blocked_now"``, ``"pip_diff"``) for the compiled
        backend, or a callable ``leaf_eval(view) -> float`` to plug in a custom
        estimate. A callable forces the Python reference backend, because a numba
        nopython search cannot call back into Python — a limitation of the compiler,
        documented rather than hidden.
    tt_bits:
        Table size is ``2 ** tt_bits`` slots; each slot is 24 bytes, so the default
        20 is a hard 24 MiB cap, and the table never grows or spills.
    tt_enabled:
        Set ``False`` for the table-off control used by the tests.
    node_limit:
        Deterministic node budget per :meth:`solve` call (``None`` = unlimited).
        Exceeding it marks the result ``complete=False`` instead of returning a
        wrong "value".
    root_pruning:
        ``False`` (default) searches every root move with a full window, so every
        entry of ``move_values`` is exact. ``True`` shares one alpha-beta window
        across root moves: faster, but the values of non-best moves become bounds
        (only ``best_move`` is then guaranteed). The fast path for play.
    """

    def __init__(self, *, max_depth: Optional[int] = None, leaf_eval=None,
                 tt_bits: int = 20, tt_enabled: bool = True,
                 node_limit: Optional[int] = None, root_pruning: bool = False) -> None:
        if max_depth is not None:
            if not 0 <= int(max_depth) < EXACT_DEPTH:
                raise ValueError(f"max_depth must be in [0, {EXACT_DEPTH}), got {max_depth}")
        if not 12 <= int(tt_bits) <= 26:
            raise ValueError(f"tt_bits must be in [12, 26], got {tt_bits}")
        self.max_depth = None if max_depth is None else int(max_depth)
        self.tt_bits = int(tt_bits)
        self.tt_enabled = bool(tt_enabled)
        self.node_limit = None if node_limit is None else int(node_limit)
        self.root_pruning = bool(root_pruning)
        self.leaf_eval_callable: Optional[Callable[[LeafView], float]] = None
        self.leaf_mode = "blocked_now"
        if callable(leaf_eval):
            self.leaf_eval_callable = leaf_eval
            self.backend = "python"
        else:
            if leaf_eval is not None:
                if leaf_eval not in LEAF_MODES:
                    raise ValueError(
                        f"unknown leaf_eval {leaf_eval!r}; known modes: "
                        f"{sorted(LEAF_MODES)} or a callable"
                    )
                self.leaf_mode = str(leaf_eval)
            self.backend = "numba"
        self._tt: Optional[tuple[np.ndarray, ...]] = None
        self._tt_leaf_mode: Optional[str] = None
        self._scratch: Optional[tuple[np.ndarray, ...]] = None

    # ---- public API -----------------------------------------------------
    @property
    def tt_bytes(self) -> int:
        """Bytes currently held by the transposition table (0 before first use)."""
        if self._tt is None:
            return 0
        return sum(int(arr.nbytes) for arr in self._tt)

    def clear_tt(self) -> None:
        """Drop every table entry (O(size), no reallocation)."""
        if self._tt is not None:
            self._tt[4].fill(EMPTY_META)

    def solve(self, state: GameState) -> SolveResult:
        """Solve one position for the seat to move in ``state``."""
        seat, moves = _root_expand(state)
        if self.backend == "python":
            return reference_solve(state, max_depth=self.max_depth,
                                   leaf_eval=self.leaf_eval_callable,
                                   node_limit=self.node_limit)
        hands, pips, left, right, pass_streak = R.compact(state)
        return self._solve_core(hands, pips, int(left), int(right), int(seat),
                                int(pass_streak), moves)

    def solve_arrays(self, hands: np.ndarray, left: int, right: int, turn: int,
                     pass_streak: int, pips: np.ndarray,
                     moves: list[tuple[int, int]]) -> SolveResult:
        """Solve straight from compiled arrays, with no ``GameState`` round trip.

        The Phase-3 inner loop calls this once per (world, root move). Copies are
        taken, so the caller's world arrays are never mutated.
        """
        if self.backend == "python":
            raise ValueError("solve_arrays is only available on the compiled backend")
        if not moves:
            raise ValueError("solve_arrays needs the legal moves of the position")
        return self._solve_core(np.array(hands, dtype=np.uint32, copy=True),
                                np.array(pips, dtype=np.int32, copy=True),
                                int(left), int(right), int(turn), int(pass_streak),
                                list(moves))

    def _solve_core(self, hands: np.ndarray, pips: np.ndarray, left: int, right: int,
                    seat: int, pass_streak: int, moves: list[tuple[int, int]]) -> SolveResult:
        """Shared compiled-backend root: one full-window search per root move.

        With ``root_pruning=True`` the moves share a single alpha-beta window, which
        is faster but turns the non-best entries of ``move_values`` into bounds.
        """
        if self.leaf_mode != self._tt_leaf_mode:
            # Entries computed with another leaf estimate are not comparable.
            self.clear_tt()
            self._tt_leaf_mode = self.leaf_mode
        team_to_move = int(TEAM_OF_SEAT[seat])
        depth_budget = EXACT_DEPTH if self.max_depth is None else self.max_depth
        limit = INF * 64 if self.node_limit is None else self.node_limit
        counters = np.zeros(8, dtype=np.int64)
        tt = self._table()
        sc_scores, sc_moves, sc_l, sc_r, sc_e = self._scratch_arrays()
        ends = np.array([left, right], dtype=np.int32)
        leaf_mode = LEAF_MODES[self.leaf_mode]
        started = time.perf_counter()

        def run(turn: int, pass_value: int, alpha: int, beta: int) -> int:
            return int(_search(hands, ends, turn, pass_value, pips, depth_budget - 1,
                               alpha, beta, 0, sc_scores, sc_moves, sc_l, sc_r, sc_e,
                               tt[0], tt[1], tt[2], tt[3], tt[4], tt[5], tt[6],
                               counters, leaf_mode, self.tt_enabled, limit))

        move_values: dict[tuple[int, int], int] = {}
        if not moves:                                       # root forced pass
            value = run((seat + 1) & 3, pass_streak + 1, -INF, INF)
            result = self._result(seat, team_to_move, value, counters, started)
            result.complete = counters[4] == 0
            return _finish(result, {}, True, team_to_move)
        alpha, beta = -INF, INF
        aborted = False
        for tid, end in moves:
            placements = _placements_py(int(tid), left, right)
            chosen = next(((cl, cr) for cl, cr, e in placements if e == int(end)), None)
            if chosen is None:                              # engine tolerates a wrong end
                chosen = (placements[0][0], placements[0][1])
            bit = np.uint32(1) << np.uint32(int(tid))
            hands[seat] ^= bit
            pips[seat] -= int(R.TILE_PIPS[int(tid)])
            ends[0], ends[1] = chosen
            value = run((seat + 1) & 3, 0, alpha, beta)
            ends[0], ends[1] = left, right
            hands[seat] ^= bit
            pips[seat] += int(R.TILE_PIPS[int(tid)])
            move_values[(int(tid), int(end))] = value
            if counters[4] == 1:
                aborted = True
                break
            if self.root_pruning:
                # Narrow the bound the *moving team* cares about: a max node raises
                # alpha, a min node lowers beta. Raising alpha at a min node would prune
                # against the wrong bound and could pick the opponent's move.
                if team_to_move == 0:
                    if value > alpha:
                        alpha = value
                elif value < beta:
                    beta = value
        result = self._result(seat, team_to_move, 0, counters, started)
        result.complete = not aborted
        return _finish(result, move_values, False, team_to_move)

    # ---- internals ------------------------------------------------------
    def _table(self) -> tuple[np.ndarray, ...]:
        if self._tt is None:
            size = 1 << self.tt_bits
            self._tt = (
                np.zeros(size, dtype=np.uint32), np.zeros(size, dtype=np.uint32),
                np.zeros(size, dtype=np.uint32), np.zeros(size, dtype=np.uint32),
                np.full(size, EMPTY_META, dtype=np.uint32),
                np.zeros(size, dtype=np.int32), np.int64(size - 1),
            )
        return self._tt

    def _scratch_arrays(self) -> tuple[np.ndarray, ...]:
        if self._scratch is None:
            self._scratch = (
                np.zeros((MAX_PLY, 16), dtype=np.float32),
                np.zeros((MAX_PLY, 16, 4), dtype=np.int32),
                np.zeros((MAX_PLY, 4), dtype=np.int32),
                np.zeros((MAX_PLY, 4), dtype=np.int32),
                np.zeros((MAX_PLY, 4), dtype=np.int32),
            )
        return self._scratch

    def _result(self, seat: int, team_to_move: int, value: int, counters: np.ndarray,
                started: float) -> SolveResult:
        return SolveResult(
            value=int(value), value_to_move=0, seat=int(seat), team_to_move=team_to_move,
            nodes=int(counters[0]), leaf_evals=int(counters[3]), tt_hits=int(counters[1]),
            tt_probes=int(counters[2]), cutoffs=int(counters[5]),
            depth=self.max_depth, exact=(self.max_depth is None and counters[3] == 0),
            complete=(counters[4] == 0), elapsed_s=time.perf_counter() - started,
            backend="numba", leaf_mode=self.leaf_mode,
        )


def best_action(state: GameState, solver: Solver) -> int:
    """Best legal action index for the seat to move (oracle helper, exact by default)."""
    result = solver.solve(state)
    if result.best_action is None:
        raise ValueError("no legal move to choose from")
    return result.best_action
