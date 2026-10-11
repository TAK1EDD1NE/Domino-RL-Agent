"""Tests for the 56-dim action mask (spec section 5) and its index helpers."""
import numpy as np
import pytest

from domino.engine import GameState, new_deal
from domino.masks import (
    ACTION_DIM,
    action_index,
    action_mask,
    assert_consistent,
    decode_action,
    masked_actions,
)
from domino.tiles import NUM_TILES, SIX_SIX_ID, tile_id


def _synthetic_state(hand, ends, turn=0, table=None):
    """Build a minimal state with a given acting hand and open ends.

    ``table`` is set to a single arbitrary tile so the board counts as
    non-empty (a free lead ignores ``ends``).
    """
    return GameState(
        hands=[list(hand), [], [], []],
        rng=np.random.default_rng(0),
        table=list(table) if table is not None else [tile_id(0, 1)],
        ends=list(ends),
        turn=turn,
        leader=turn,
    )


def test_action_index_round_trip_and_bounds():
    seen = set()
    for tid in range(NUM_TILES):
        for end in (0, 1):
            a = action_index(tid, end)
            assert 0 <= a < ACTION_DIM == 56
            seen.add(a)
            assert decode_action(a) == (tid, end)
    assert seen == set(range(ACTION_DIM))


def test_decode_action_rejects_out_of_range():
    with pytest.raises(ValueError):
        decode_action(56)
    with pytest.raises(ValueError):
        decode_action(-1)


def test_action_index_matches_formula():
    for tid in range(NUM_TILES):
        assert action_index(tid, 0) == 2 * tid
        assert action_index(tid, 1) == 2 * tid + 1


def test_mask_shape_dtype_and_equivalence_on_a_real_deal():
    s = new_deal(deal_no=1, rng=np.random.default_rng(7))
    seat = s.turn
    mask = action_mask(s, seat)
    assert mask.shape == (ACTION_DIM,) == (56,)
    assert mask.dtype == np.bool_
    expected = {2 * t + e for (t, e) in s.legal_moves(seat)}
    assert set(int(i) for i in masked_actions(mask)) == expected
    # both directions: every marked index must decode to a legal move
    for a in masked_actions(mask):
        assert decode_action(int(a)) in s.legal_moves(seat)


def test_mask_nonempty_and_legal_across_a_full_random_deal():
    rng = np.random.default_rng(123)
    s = new_deal(deal_no=1, rng=rng)
    decisions = 0
    while not s.is_terminal():
        seat = s.turn
        if s.must_pass(seat):
            s.pass_turn(seat)
            continue
        decisions += 1
        mask = action_mask(s, seat)
        assert mask.sum() >= 1
        marked = set(int(i) for i in masked_actions(mask))
        legal = {2 * t + e for (t, e) in s.legal_moves(seat)}
        assert marked == legal
        action = int(rng.choice(sorted(marked)))
        tid, end = decode_action(action)
        s.play(seat, tid, end)
    assert decisions >= 1
    assert s.is_terminal()


def test_forced_opening_is_the_single_six_six_lead():
    s = new_deal(deal_no=1, rng=np.random.default_rng(5))
    assert s.forced_leader_six_six is True
    seat = s.turn
    mask = action_mask(s, seat)
    marked = masked_actions(mask)
    assert len(marked) == 1
    assert int(marked[0]) == action_index(SIX_SIX_ID, 0)


def test_dedupe_two_ends_with_a_matching_tile_yields_two_distinct_moves():
    t = tile_id(3, 5)
    s = _synthetic_state([t], ends=[3, 5], turn=0)
    mask = action_mask(s, 0)
    assert set(int(i) for i in masked_actions(mask)) == {action_index(t, 0), action_index(t, 1)}
    left = s.clone()
    left.play(0, t, 0)
    right = s.clone()
    right.play(0, t, 1)
    assert left.ends != right.ends


def test_dedupe_equal_ends_yield_a_single_move():
    t = tile_id(6, 3)
    s = _synthetic_state([t], ends=[6, 6], turn=0)
    mask = action_mask(s, 0)
    assert len(masked_actions(mask)) == 1


def test_assert_consistent_accepts_masked_and_rejects_masked_out():
    s = new_deal(deal_no=1, rng=np.random.default_rng(11))
    seat = s.turn
    legal_action = int(masked_actions(action_mask(s, seat))[0])
    assert_consistent(s, seat, legal_action)  # must not raise
    masked_out = next(
        a for a in range(ACTION_DIM)
        if a not in set(int(i) for i in masked_actions(action_mask(s, seat)))
    )
    with pytest.raises(ValueError):
        assert_consistent(s, seat, masked_out)
