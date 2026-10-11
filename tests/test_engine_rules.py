"""Rules-engine tests: state, dealing, legal moves, play, forced pass, block."""
import numpy as np
import pytest

from domino.engine import GameState, new_deal, legal_actions, legal_moves
from domino.tiles import ALL_TILE_IDS, SIX_SIX_ID, tile_id, tile_pips, TOTAL_PIPS


def fresh(seed=0, **kw):
    return new_deal(deal_no=1, rng=np.random.default_rng(seed), **kw)


def _vals(tid):
    from domino.tiles import tile_values
    return tile_values(tid)


def test_deal_gives_seven_tiles_each_and_loses_none():
    s = fresh()
    assert [len(h) for h in s.hands] == [7, 7, 7, 7]
    assert sorted(t for h in s.hands for t in h) == list(ALL_TILE_IDS)
    assert s.table == [] and s.ends == []


def test_deal_is_deterministic_for_a_seed():
    assert fresh(3).hands == fresh(3).hands
    assert fresh(3).hands != fresh(4).hands


def test_first_deal_leader_holds_six_six_and_must_play_it():
    s = fresh()
    assert s.leader == s.turn
    assert SIX_SIX_ID in s.hands[s.leader]
    assert legal_actions(s, s.leader) == [SIX_SIX_ID]


def test_lead_on_later_deal_is_free():
    s = new_deal(deal_no=2, rng=np.random.default_rng(0), lead_mode="winner", leader=2)
    assert s.turn == 2 and s.leader == 2
    assert sorted(legal_actions(s, 2)) == sorted(s.hands[2])


def test_play_requires_a_matching_end_and_is_mandatory():
    s = fresh()
    s.play(s.turn, SIX_SIX_ID)                      # forced opener
    nxt = s.turn
    assert s.ends == [6, 6]
    legal = legal_actions(s, nxt)
    assert legal == [t for t in s.hands[nxt] if 6 in _vals(t)]
    if legal:
        assert not s.must_pass(nxt)
        with pytest.raises(ValueError):
            s.play(nxt, next(t for t in s.hands[nxt] if t not in legal))
    else:
        assert s.must_pass(nxt)


def test_tile_matching_both_ends_gives_two_deduped_placements():
    s = new_deal(deal_no=2, rng=np.random.default_rng(0), lead_mode="winner", leader=0)
    dup = tile_id(3, 5)
    s.table = [dup]                                 # free lead of [3|5] exposes ends 3 and 5
    s.ends = [3, 5]
    s.hands[0] = [dup]
    s.turn = 0
    assert legal_moves(s, 0) == [(dup, 0), (dup, 1)]
    on_left = s.clone()
    on_left.play(0, dup, end=0)                     # attach to the 3-end -> [5, 5]
    assert on_left.ends == [5, 5]
    on_right = s.clone()
    on_right.play(0, dup, end=1)                    # attach to the 5-end -> [3, 3]
    assert on_right.ends == [3, 3]


def test_equal_ends_dedupe_to_one_placement():
    s = new_deal(deal_no=2, rng=np.random.default_rng(0), lead_mode="winner", leader=0)
    s.table = [tile_id(4, 4)]
    s.ends = [4, 4]
    t = tile_id(2, 4)
    s.hands[0] = [t]
    s.turn = 0
    assert legal_moves(s, 0) == [(t, 0)]


def test_pips_conservation_after_every_action():
    s = fresh(11)
    for _ in range(28):
        seat = s.turn
        la = legal_actions(s, seat)
        if la:
            s.play(seat, la[0])
        else:
            s.pass_turn(seat)
        hands = sum(tile_pips(t) for h in s.hands for t in h)
        assert hands + s.table_pips() == TOTAL_PIPS
        if s.blocks() or len(s.table) == 28:
            break


def test_block_after_exactly_four_consecutive_passes():
    s = new_deal(deal_no=1, rng=np.random.default_rng(7))
    # build a state nobody can extend: table ends (0, 0) while no hand holds a 0 twin
    s.table = [tile_id(0, 0)]
    s.ends = [0, 0]
    s.hands = [[t for t in h if 0 not in _vals(t)] for h in s.hands]
    s.turn = 0
    s.pass_streak = 0
    for i in range(3):
        seat = s.turn
        assert s.must_pass(seat)
        s.pass_turn(seat)
        assert not s.blocks()
    assert s.must_pass(s.turn)
    s.pass_turn(s.turn)
    assert s.blocks() and s.pass_streak == 4


def test_hand_emptying_is_detected():
    s = new_deal(deal_no=2, rng=np.random.default_rng(1), lead_mode="winner", leader=0)
    s.hands[0] = [tile_id(1, 2)]
    s.table = []
    s.ends = []
    s.turn = 0
    s.play(0, tile_id(1, 2))
    assert s.hand_size(0) == 0
