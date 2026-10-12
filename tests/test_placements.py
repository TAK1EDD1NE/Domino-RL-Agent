"""Placed-move uniqueness (v3 item 7) and the placement cache (v3 item 8).

The proposal was to mask one of the two placements of a double because they are
duplicates. That is already the engine's contract: ``_tile_placements`` collapses
placements that produce the same position, so no duplicate action is ever exposed
and the policy spends no probability mass on a redundant move. These tests lock
that in as a property (it is a rule, not a coincidence) and cover the cache that
now backs it.
"""
from __future__ import annotations

import numpy as np

from domino.engine import GameState, legal_moves, new_deal
from domino.masks import ACTION_DIM, action_index, action_mask


def _positions_after(state: GameState, seat: int) -> dict[int, list[tuple[int, ...]]]:
    """Map each legal move of ``seat`` to its resulting open ends, per tile id."""
    outcome: dict[int, list[tuple[int, ...]]] = {}
    for tid, end in legal_moves(state, seat):
        clone = state.clone()
        clone.turn = seat
        clone.play(seat, tid, end)
        outcome.setdefault(tid, []).append(tuple(sorted(clone.ends)))
    return outcome


def test_no_two_legal_placements_of_a_tile_are_equivalent() -> None:
    """Over many deals: distinct legal moves always produce distinct positions.

    A double on a board whose two ends carry the same value has two geometrically
    equivalent placements; the engine collapses them, so the action space never
    contains a redundant move (v3 item 7) and the mask never exposes one.
    """
    checked = 0
    for seed in range(60):
        state = new_deal(deal_no=1, rng=np.random.default_rng(seed))
        while not state.is_terminal():
            seat = state.turn
            if state.must_pass(seat):
                state.pass_turn(seat)
                continue
            moves = legal_moves(state, seat)
            positions = _positions_after(state, seat)
            for tid, ends in positions.items():
                declared = sum(1 for t, _e in moves if t == tid)
                assert len(ends) == declared, (
                    f"seed {seed}: tile {tid} declares {declared} moves but reaches "
                    f"{len(ends)} distinct positions {ends}"
                )
                assert len(set(ends)) == len(ends), (
                    f"seed {seed}: tile {tid} exposes equivalent placements {ends}"
                )
            mask = action_mask(state, seat)
            assert int(mask.sum()) == len(moves)
            indices = [action_index(tid, end) for tid, end in moves]
            assert len(indices) == len(set(indices)) < ACTION_DIM
            checked += len(moves)
            tid, end = moves[0]
            state.play(seat, tid, end)
    assert checked > 100, f"only {checked} moves checked; the property test barely ran"


def test_placement_cache_is_used_and_returns_fresh_lists() -> None:
    """The (tile, ends) cache must serve repeats without handing out shared lists."""
    GameState._placements_cached.cache_clear()
    state = new_deal(deal_no=1, rng=np.random.default_rng(0))
    tid, _end = legal_moves(state, state.turn)[0]
    ends = (6, 4)

    first = GameState._placements_cached(tid, ends)
    second = GameState._placements_cached(tid, ends)
    assert first is second                              # served from the cache
    assert GameState._placements_cached.cache_info().hits >= 1

    state.ends = list(ends)
    list_a = state._tile_placements(tid)
    list_a[0][1].append(99)                             # mutate the returned list
    list_b = state._tile_placements(tid)
    assert 99 not in list_b[0][1], "the cache handed out its internal list"
    assert GameState._placements_cached(tid, ends) == first
