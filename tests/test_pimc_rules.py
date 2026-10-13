"""Pin the compiled rules core against ``domino.engine``, position by position.

The solver must never re-invent a rule: these tests compare the bitmask core with the
engine that the environment and the training path actually use, so a drift between
the two is a test failure rather than a silent value error in PIMC or the judge.
"""
from __future__ import annotations

import numpy as np
import pytest

from domino.engine import GameState, deal_outcome, new_deal
from domino.pimc import rules as R
from domino.tiles import NUM_TILES, SIX_SIX_ID, TILES, tile_pips

ALL_ENDS = [(left, right) for left in range(7) for right in range(7)]


def make_state(hands, ends=(), turn=0, pass_streak=0) -> GameState:
    """Build an engine state from explicit hands and open ends.

    A dummy first table tile is added whenever ``ends`` is given, because the engine
    (and ``rules.compact``) read "free lead" as an empty ``table``; the table's own
    tiles never affect the deal's value, only the open ends do.
    """
    flat = {tile for hand in hands for tile in hand}
    table = [t for t in range(NUM_TILES) if t not in flat][:1] if ends else []
    return GameState(
        hands=[sorted(int(t) for t in hand) for hand in hands],
        rng=np.random.default_rng(0),
        table=list(table),
        ends=[int(v) for v in ends],
        turn=int(turn),
        pass_streak=int(pass_streak),
    )


def pips_of(hands) -> list[int]:
    """Pip total per hand."""
    return [sum(tile_pips(t) for t in hand) for hand in hands]


def test_popcount_matches_python() -> None:
    """The SWAR popcount agrees with ``int.bit_count`` on every mask shape."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        value = int(rng.integers(0, 1 << 28))
        assert int(R.popcount32(np.uint32(value))) == bin(value).count("1")


@pytest.mark.parametrize("tid", range(NUM_TILES))
def test_tile_placements_match_engine(tid: int) -> None:
    """Every ``(tile, open ends)`` placement matches ``_placements_cached`` exactly."""
    out_l = np.zeros(2, dtype=np.int32)
    out_r = np.zeros(2, dtype=np.int32)
    out_e = np.zeros(2, dtype=np.int32)
    for left, right in ALL_ENDS:
        count = int(R.tile_placements(tid, left, right, out_l, out_r, out_e))
        mine = sorted((int(out_l[i]), int(out_r[i]), int(out_e[i])) for i in range(count))
        theirs = sorted((int(new_ends[0]), int(new_ends[1]), int(end))
                        for end, new_ends in GameState._placements_cached(tid, (left, right)))
        assert mine == theirs, f"tile {tid} on ends {(left, right)}"


@pytest.mark.parametrize("tid", range(NUM_TILES))
def test_free_lead_placement_matches_engine(tid: int) -> None:
    """On an empty table the tile opens the line and exposes both of its values."""
    out_l = np.zeros(2, dtype=np.int32)
    out_r = np.zeros(2, dtype=np.int32)
    out_e = np.zeros(2, dtype=np.int32)
    count = int(R.tile_placements(tid, -1, -1, out_l, out_r, out_e))
    assert count == 1 and int(out_e[0]) == 0
    assert (int(out_l[0]), int(out_r[0])) == TILES[tid]
    state = make_state([[tid], [0], [1], [2]])
    state.play(state.turn, tid)
    assert state.ends == list(TILES[tid])


def test_legal_moves_match_engine_on_real_deals() -> None:
    """``playable_mask`` + placements reproduce ``GameState.legal_moves`` move for move."""
    rng = np.random.default_rng(11)
    out_l = np.zeros(2, dtype=np.int32)
    out_r = np.zeros(2, dtype=np.int32)
    out_e = np.zeros(2, dtype=np.int32)
    checked = 0
    while checked < 400:
        state = new_deal(deal_no=1, rng=rng, lead_mode="six_six")
        while not state.is_terminal():
            for seat in range(4):
                simulated = set()
                hand = state.hands[seat]
                mask = np.uint32(0)
                for tile in hand:
                    mask |= np.uint32(1) << np.uint32(tile)
                if state.table:
                    ends = (int(state.ends[0]), int(state.ends[1]))
                else:
                    ends = (-1, -1)
                cand = int(R.playable_mask(mask, ends[0], ends[1]))
                if (not state.table and state.forced_leader_six_six
                        and seat == state.turn and SIX_SIX_ID in hand):
                    cand = 1 << SIX_SIX_ID
                for tile in range(NUM_TILES):
                    if (cand >> tile) & 1 == 0:
                        continue
                    count = int(R.tile_placements(tile, ends[0], ends[1], out_l, out_r, out_e))
                    for i in range(count):
                        simulated.add((tile, int(out_e[i])))
                assert simulated == set(state.legal_moves(seat)), (
                    f"seat {seat} on ends {ends} hand {sorted(hand)}"
                )
            checked += 1
            seat = state.turn
            moves = state.legal_moves(seat)
            if not moves:
                state.pass_turn(seat)
                continue
            tid, end = moves[int(rng.integers(0, len(moves)))]
            state.play(seat, tid, end)


def test_terminal_values_match_deal_outcome() -> None:
    """Emptied, blocked, blocked-tie and void shapes score exactly as the engine does."""
    cases = {
        "emptied_team0": ([[], [0, 1], [7], [13, 22]], (), 0),
        "emptied_team1": ([[27], [1], [7], []], (), 0),
        "blocked_unique_min": ([[7, 13], [8, 9], [14], [18]], (0, 0), 4),
        "blocked_other_team": ([[27], [7], [13], [18]], (0, 0), 4),
        "blocked_void_tie": ([[9], [13], [18], [20]], (0, 0), 4),
        "blocked_same_team_tie": ([[9], [18], [13], [20]], (0, 0), 4),
    }
    for name, (hands, ends, streak) in cases.items():
        state = make_state(hands, ends, turn=0, pass_streak=streak)
        assert state.is_terminal(), name
        outcome = deal_outcome(state)
        expected = 0
        if outcome.winner_team == 0:
            expected = outcome.points
        elif outcome.winner_team == 1:
            expected = -outcome.points
        compacted = R.compact(state)
        got = int(R.terminal_value(compacted[0], compacted[1], compacted[4]))
        assert got == expected, name
        assert got != R.NOT_TERMINAL
        assert [int(p) for p in compacted[1]] == pips_of(hands)


def test_terminal_value_is_sentinel_while_playable() -> None:
    """A playable position is not terminal, and the sentinel never collides with a value."""
    state = make_state([[27], [0], [7], [13]], (6, 6), turn=0)
    hands, pips, _left, _right, streak = R.compact(state)
    assert int(R.terminal_value(hands, pips, streak)) == R.NOT_TERMINAL


def test_leaf_modes_are_in_reward_units() -> None:
    """``blocked_now`` reads like the blocked terminal; ``pip_diff`` is the plain spread."""
    pips = np.array([6, 7, 5, 6], dtype=np.int32)          # team 0 lightest -> +13
    assert int(R.leaf_value(pips, 0)) == 13
    pips_heavy = np.array([9, 2, 4, 6], dtype=np.int32)    # team 1 lightest -> -13
    assert int(R.leaf_value(pips_heavy, 0)) == -13
    tied = np.array([4, 4, 8, 8], dtype=np.int32)          # cross-team tie -> void
    assert int(R.leaf_value(tied, 0)) == 0
    assert int(R.leaf_value(pips, 1)) == (7 + 6) - (6 + 5)
