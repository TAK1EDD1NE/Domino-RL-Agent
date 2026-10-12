"""Tests for the 192-dim seat-relative observation encoder (spec section 4)."""
import numpy as np

from domino.engine import GameState, new_deal
from domino.obs import OBS_DIM, encode, describe_observation
from domino.tiles import ALL_TILE_IDS, tile_id, SIX_SIX_ID


def test_shape_dtype_and_range():
    s = new_deal(deal_no=1, rng=np.random.default_rng(0))
    v = encode(s, s.turn)
    assert v.shape == (OBS_DIM,) == (192,) and v.dtype == np.float32
    assert np.isfinite(v).all()


def test_hand_and_counts_are_seat_relative():
    s = new_deal(deal_no=1, rng=np.random.default_rng(1))
    for seat in range(4):
        v = encode(s, seat)
        hand = v[0:28]
        assert hand.sum() == 7
        assert hand[SIX_SIX_ID] == (1.0 if SIX_SIX_ID in s.hands[seat] else 0.0)
        assert list(v[175:179]) == [len(s.hands[(seat + k) % 4]) / 7 for k in range(4)]


def test_played_by_slice_tracks_the_table():
    s = new_deal(deal_no=1, rng=np.random.default_rng(2))
    leader = s.turn
    s.play(leader, SIX_SIX_ID)
    v = encode(s, leader)
    played_by = v[56:168].reshape(4, 28)
    assert played_by[0, SIX_SIX_ID] == 1.0        # "self" is always the acting seat
    v2 = encode(s, (leader + 2) % 4)
    played_by2 = v2[56:168].reshape(4, 28)
    assert played_by2[2, SIX_SIX_ID] == 1.0       # the leader is "partner" from that seat


def test_no_hidden_information_is_encodable():
    # The encoding must never depend on the *contents* of a seat the acting
    # seat cannot observe (only on its own hand, the table and public history).
    s = new_deal(deal_no=1, rng=np.random.default_rng(3))
    a = encode(s, 0).copy()

    # (1) Swap the contents of two hidden seats: public information (my hand,
    #     the table, the play history, per-seat counts) is unchanged, only the
    #     hidden tile assignment differs. The whole vector must be identical.
    s_swap = s.clone()
    s_swap.hands[1], s_swap.hands[2] = s_swap.hands[2], s_swap.hands[1]
    b = encode(s_swap, 0)
    assert np.array_equal(a[0:56], b[0:56])        # my hand + the table
    assert np.array_equal(a[56:168], b[56:168])    # public play history
    assert np.array_equal(a[169:192], b[169:192])  # pips, pass, counts, leader, liveness
    assert np.array_equal(a, b)

    # (2) Move one tile between two hidden seats: only the hand *sizes* (and
    #     thus the counts block) change; everything else is public and fixed.
    s_move = s.clone()
    s_move.hands[3].append(s_move.hands[1].pop())
    c = encode(s_move, 0)
    assert np.array_equal(a[0:56], c[0:56])
    assert np.array_equal(a[56:168], c[56:168])
    assert np.array_equal(a[169:175], c[169:175])  # unseen pips, pass streak/last pass
    assert np.array_equal(a[179:192], c[179:192])  # leader one-hots, liveness, board empty
    assert list(c[175:179]) == [len(s_move.hands[(0 + k) % 4]) / 7 for k in range(4)]
    assert not np.array_equal(a[175:179], c[175:179])  # the counts block did change


def test_live_value_all_seven_live_on_a_fresh_deal():
    # At deal start no tile is on the table and every pip value is carried by
    # at least one tile outside the acting seat's 7-tile hand, so all seven
    # liveness flags are set.
    s = new_deal(deal_no=1, rng=np.random.default_rng(4))
    v = encode(s, s.turn)
    assert list(v[184:191]) == [1.0] * 7


def test_live_value_zero_when_a_value_is_exhausted():
    # Value 0 is carried only by tile_id(0, b) == ids 0..6 (7 tiles). Put all
    # seven in the acting seat's hand: no tile outside hand|table carries 0,
    # so flag 0 is 0.0 while values 1..6 stay live: tile (v, v) has id
    # 7, 13, 18, 22, 25, 27 (all >= 7) and is therefore in another hand.
    zero_tiles = [tile_id(0, b) for b in range(7)]
    rest = [t for t in ALL_TILE_IDS if t not in zero_tiles]  # ids 7..27
    hands = [zero_tiles, rest[0:7], rest[7:14], rest[14:21]]
    s = GameState(hands=[sorted(h) for h in hands], rng=np.random.default_rng(0))
    # The engine's own unseen_tiles() is empty here (no boneyard), which is why
    # the old unconditional source produced an information-free all-zero slice.
    assert s.unseen_tiles() == set()
    v = encode(s, 0)
    assert list(v[184:191]) == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_describe_observation_covers_all_192_dims():
    d = describe_observation()
    covered = [i for sl in d.values() for i in range(sl[0], sl[1])]
    assert sorted(covered) == list(range(192))
