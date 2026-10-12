"""Tests for the 192-dim seat-relative observation encoder (spec section 4)."""
import numpy as np

from domino.engine import GameState, new_deal
from domino.obs import (
    MATCH,
    MOVE_DIM,
    MOVES,
    MOVES_N,
    OBS_DIM,
    PRIVILEGED_DIM,
    PRIVILEGED_SLICE,
    PUBLIC_DIM,
    PUBLIC_SLICE,
    VOID_LACK,
    describe_observation,
    encode,
)
from domino.tiles import ALL_TILE_IDS, NUM_TILES, SIX_SIX_ID, tile_id, tile_values


def test_shape_dtype_and_range():
    s = new_deal(deal_no=1, rng=np.random.default_rng(0))
    v = encode(s, s.turn)
    assert v.shape == (OBS_DIM,) == (507,) and v.dtype == np.float32
    assert np.isfinite(v).all()
    # The vector is public part + privileged part (critic/belief-head input only).
    assert PUBLIC_DIM == 423 and PRIVILEGED_DIM == 84
    assert PUBLIC_SLICE == slice(0, 423) and PRIVILEGED_SLICE == slice(423, 507)


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


def test_no_hidden_information_in_the_public_slice():
    # v3 contract: the PUBLIC slice must never depend on the *contents* of a seat
    # the acting seat cannot observe (only on its own hand, the table and public
    # history). The privileged slice DOES carry the opponents' true hands by
    # design -- it exists for the centralised critic and the belief head, and the
    # actor's extractor never reads it (asserted in tests/test_policy.py).
    s = new_deal(deal_no=1, rng=np.random.default_rng(3))
    a = encode(s, 0).copy()

    # (1) Swap the contents of two hidden seats: public information (my hand,
    #     the table, the play history, per-seat counts) is unchanged, only the
    #     hidden tile assignment differs.
    s_swap = s.clone()
    s_swap.hands[1], s_swap.hands[2] = s_swap.hands[2], s_swap.hands[1]
    b = encode(s_swap, 0)
    assert np.array_equal(a[PUBLIC_SLICE], b[PUBLIC_SLICE])
    assert not np.array_equal(a[PRIVILEGED_SLICE], b[PRIVILEGED_SLICE])

    # (2) Move one tile between two hidden seats: only the hand *sizes* (and
    #     thus the counts block) change among the public features.
    s_move = s.clone()
    s_move.hands[3].append(s_move.hands[1].pop())
    c = encode(s_move, 0)
    assert np.array_equal(a[0:56], c[0:56])
    assert np.array_equal(a[56:168], c[56:168])
    assert np.array_equal(a[169:175], c[169:175])  # unseen pips, pass streak/last pass
    assert np.array_equal(a[179:192], c[179:192])  # leader one-hots, liveness, board empty
    assert list(c[175:179]) == [len(s_move.hands[(0 + k) % 4]) / 7 for k in range(4)]
    assert not np.array_equal(a[175:179], c[175:179])  # the counts block did change
    assert not np.array_equal(a[PRIVILEGED_SLICE], c[PRIVILEGED_SLICE])


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


def test_describe_observation_covers_every_dim():
    d = describe_observation()
    covered = [i for sl in d.values() for i in range(sl[0], sl[1])]
    assert sorted(covered) == list(range(OBS_DIM))
    assert set(d) >= {"void_lack", "match", "moves", "privileged_hands"}


def test_void_lack_matrix_tracks_forced_passes():
    """A pass proves the seat lacks both open-end values (v3 item 2).

    Synthetic, deterministic setup: ``[6|6]`` alone on the table (open ends 6, 6)
    and a passer whose hand holds no tile carrying a 6, so the pass is forced.
    """
    no_six = [t for t in ALL_TILE_IDS if 6 not in tile_values(t)]      # 21 tiles
    hands = [
        sorted(no_six[0:7]),      # seat 0 — the acting seat (holds no 6 either)
        sorted(no_six[7:14]),     # seat 1 — the forced passer
        sorted(no_six[14:21]),    # seat 2
        [SIX_SIX_ID] + [],        # seat 3 — holds [6|6], which the leader played
    ]
    hands[3] = sorted(set(ALL_TILE_IDS) - set(no_six) - {SIX_SIX_ID})  # the remaining sixes
    s = GameState(hands=hands, rng=np.random.default_rng(0))
    s.table = [SIX_SIX_ID]
    s.ends = [6, 6]
    s.turn = 1
    s.played_by[3] = [SIX_SIX_ID]
    s.forced_leader_six_six = False

    assert encode(s, 0)[VOID_LACK].sum() == 0.0
    assert s.must_pass(1), "seat 1 must be unable to play on ends [6, 6]"
    s.pass_turn(1)

    blocks = encode(s, 0)[VOID_LACK].reshape(3, 7)
    assert blocks[0][6] == 1.0                       # seat 1 is "left" of seat 0
    assert blocks[1:].sum() == 0.0
    assert s.lacks[1] == {6}
    # From seat 2, seat 1 sits at relative index 2 ([left, partner, right] = 3, 0, 1).
    assert encode(s, 2)[VOID_LACK].reshape(3, 7)[2][6] == 1.0

    # Monotone: a later pass on another value only adds lacks, never removes them.
    hand_values = {value for tile in hands[1] for value in tile_values(tile)}
    other = next(value for value in range(7) if value not in hand_values and value != 6)
    s.ends = [6, other]
    s.turn = 1
    s.pass_turn(1)
    assert s.lacks[1] == {6, other}
    row = encode(s, 0)[VOID_LACK].reshape(3, 7)[0]
    assert row[6] == 1.0 and row[other] == 1.0


def test_ordered_moves_block_keeps_sequence():
    """The move block encodes order, not just membership (v3 item 4)."""
    s = new_deal(deal_no=1, rng=np.random.default_rng(8))
    leader = s.turn
    first = s.legal_moves(leader)[0][0]
    s.play(leader, first, 0)
    v = encode(s, leader)
    pad = v[MOVES].reshape(MOVES_N, MOVE_DIM)
    assert pad[: MOVES_N - 1].sum() == 0.0          # zero-padded at the front
    last = pad[-1]
    assert last[(leader - leader) % 4] == 1.0       # mover one-hot, relative to me
    assert last[4 + first] == 1.0                   # tile one-hot
    assert last[4 + NUM_TILES:].sum() == 1.0        # end one-hot

    # A second move shifts the window: the older move is now in the previous slot.
    follower = s.turn
    second = s.legal_moves(follower)[0][0]
    s.play(follower, second, s.legal_moves(follower)[0][1])
    v2 = encode(s, leader)
    pad2 = v2[MOVES].reshape(MOVES_N, MOVE_DIM)
    assert pad2[-2][4 + first] == 1.0 and pad2[-1][4 + second] == 1.0


def test_match_features_carry_the_score():
    """Match-score features are team-relative and normalised (v3 item 6)."""
    s = new_deal(deal_no=3, rng=np.random.default_rng(9))
    s.scores = [40, 25]
    v = encode(s, 0)                     # seat 0 is team 0
    assert np.allclose(v[MATCH][:3], [0.40, 0.25, 0.15], atol=1e-6)
    assert v[MATCH][3] == 3 / 5
    assert v[MATCH][4] == 1.0 and v[MATCH][5] == 0.0
    w = encode(s, 1)                     # seat 1 is team 1: mirrored view
    assert np.allclose(w[MATCH][:3], [0.25, 0.40, -0.15], atol=1e-6)
    assert w[MATCH][4] == 0.0 and w[MATCH][5] == 1.0

