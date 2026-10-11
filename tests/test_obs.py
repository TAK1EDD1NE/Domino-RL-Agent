"""Tests for the 192-dim seat-relative observation encoder (spec section 4)."""
import numpy as np

from domino.engine import new_deal
from domino.obs import OBS_DIM, encode, describe_observation
from domino.tiles import tile_id, SIX_SIX_ID


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
    # the encoding must not depend on seats the acting seat cannot observe
    s = new_deal(deal_no=1, rng=np.random.default_rng(3))
    a = encode(s, 0).copy()
    s.hands[1] = [tile_id(0, 0)]
    b = encode(s, 0)
    assert np.array_equal(a, b) is False  # counts change (observable), but...
    # ...the content of seat 1's hand must never leak: only the size can differ
    s2 = new_deal(deal_no=1, rng=np.random.default_rng(3))
    s2.hands[1] = [tile_id(6, 6)]
    assert np.array_equal(encode(s2, 0)[0:56], encode(s, 0)[0:56])


def test_describe_observation_covers_all_192_dims():
    d = describe_observation()
    covered = [i for sl in d.values() for i in range(sl[0], sl[1])]
    assert sorted(covered) == list(range(192))
