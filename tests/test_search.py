"""Tests for the PIMC root-selection policy (``domino.search``).

No model training and no long loops: the value function is an analytic stub, the
determiniser is exercised on hand-built states, and the whole module runs in a
few milliseconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from domino.engine import GameState
from domino.masks import action_index
from domino.obs import OBS_DIM, PRIVILEGED_SLICE
from domino.search import Determinizer, PIMCPolicyAgent, value_fn_from_model
from domino.tiles import NUM_TILES, tile_pips, tile_values


# ---- state construction ----------------------------------------------------
def _make_state(hands, table, played_by, ends, turn, leader, *,
                pass_streak=0, last_pass=None, lacks=None, moves=None):
    """Build a hand-crafted :class:`GameState`.

    ``lacks``/``moves`` default to the engine's empty values; pass them to model
    the public history a real deal would have accumulated.
    """
    return GameState(
        hands=[sorted(h) for h in hands],
        rng=np.random.default_rng(0),
        table=list(table),
        played_by=[list(p) for p in played_by],
        ends=list(ends),
        turn=turn,
        leader=leader,
        pass_streak=pass_streak,
        last_pass=list(last_pass if last_pass is not None else [False] * 4),
        lacks=[set(s) for s in (lacks if lacks is not None else [set(), set(), set(), set()])],
        moves=list(moves if moves is not None else []),
    )


# State A: seat 3 acts; ends (2,1) after seat2 played [0|2] on a [0|1] lead.
# seat1 was forced to pass (held no 0/1 tile) -> seat1 lacks {0,1}.  seat3 has
# six legal placements (tiles carrying 1 or 2).
STATE_A_HANDS = [
    [0, 3, 4, 5, 6, 7],            # seat0 (played the [0|1] lead, id 1)
    [13, 15, 18, 19, 20, 21, 23],  # seat1 (avoids 0/1 -> consistent with the lack)
    [9, 10, 24, 25, 26, 27],       # seat2 (played id 2 = [0|2])
    [8, 11, 12, 14, 16, 17, 22],   # seat3 = me
]


def _state_a() -> GameState:
    return _make_state(
        hands=STATE_A_HANDS,
        table=[1, 2],
        played_by=[[1], [], [2], []],
        ends=[2, 1],
        turn=3,
        leader=0,
        pass_streak=0,
        last_pass=[False, True, False, False],
        lacks=[set(), {0, 1}, set(), set()],
        moves=[(0, 1, 0), (2, 2, 0)],
    )


def _legal_actions(state: GameState, seat: int) -> set[int]:
    return {action_index(tid, end) for tid, end in state.legal_moves(seat)}


# ---- analytic value stubs --------------------------------------------------
def _pip_shed_value(observations) -> np.ndarray:
    """Value = minus my remaining pips (read straight off the public hand block).

    Deterministic and independent of the sampled hands, so PIMC must pick the
    move that sheds the heaviest legal tile.
    """
    obs = np.asarray(observations)
    out = np.empty(obs.shape[0], dtype=np.float64)
    for i in range(obs.shape[0]):
        out[i] = -sum(tile_pips(t) for t in range(NUM_TILES) if obs[i, t] > 0.5)
    return out


def _zero_value(observations) -> np.ndarray:
    return np.zeros(np.asarray(observations).shape[0], dtype=np.float64)


# ---- determiniser ----------------------------------------------------------
def test_determinizer_respects_counts_lacks_and_played_tiles():
    state = _state_a()
    seat = 3
    mine = set(state.hands[seat])
    table = set(state.table)
    unseen = set(range(NUM_TILES)) - mine - table
    det = Determinizer(seed=0)

    for _ in range(30):
        left, partner, right = det.sample(state, seat)
        # counts: seat0 played 1, seat1 none, seat2 played 1 (relative order
        # [left, partner, right] == seats (0, 1, 2) for acting seat 3).
        assert (len(left), len(partner), len(right)) == (6, 7, 6)
        # the three hands partition the unseen tiles exactly
        assert set(left) | set(partner) | set(right) == unseen
        assert len(set(left) | set(partner) | set(right)) == len(unseen)
        # no seat holds a tile it played (played tiles are on the table)
        for hand in (left, partner, right):
            assert mine.isdisjoint(hand)
            assert table.isdisjoint(hand)
        # the partner is known to lack 0 and 1 -> it may hold no such tile
        for tid in partner:
            a, b = tile_values(tid)
            assert 0 not in (a, b) and 1 not in (a, b)


def test_determinizer_is_seeded_reproducible():
    state = _state_a()
    first = Determinizer(seed=7)
    second = Determinizer(seed=7)
    assert first.sample(state, 3) == second.sample(state, 3)

    # a different seed yields a different draw (both over 30 tiles of slack)
    other = Determinizer(seed=8)
    assert other.sample(state, 3) != first.sample(state, 3)


def test_determinizer_draws_fresh_samples_across_calls():
    det = Determinizer(seed=0)
    state = _state_a()
    draws = {tuple(tuple(h) for h in det.sample(state, 3)) for _ in range(20)}
    assert len(draws) > 1  # the sampler is stochastic, not a fixed answer


def test_determinizer_raises_when_constraints_are_unsatisfiable():
    # seat2 acts; seats 0 (partner) and 1 (right) both lack {0,6} and each need 7
    # tiles, but only 8 unseen tiles avoid both values -> no valid assignment.
    state = _make_state(
        hands=[
            [0, 1, 2, 3, 4, 5, 11],       # seat0 (partner, count 7)
            [12, 15, 16, 17, 19, 20, 21],  # seat1 (right, count 7)
            [7, 8, 9, 10, 13, 14, 18],     # seat2 = me (count 7)
            [22, 23, 24, 25, 26, 27],      # seat3 (left, played the [0|6] lead)
        ],
        table=[6],
        played_by=[[], [], [], [6]],
        ends=[0, 6],
        turn=2,
        leader=3,
        pass_streak=2,
        last_pass=[True, True, False, False],
        lacks=[{0, 6}, {0, 6}, set(), set()],
        moves=[(3, 6, 0)],
    )
    with pytest.raises(ValueError):
        Determinizer(seed=0).sample(state, 2)


# ---- PIMC ------------------------------------------------------------------
def test_pimc_always_returns_a_legal_action():
    state = _state_a()
    legal = _legal_actions(state, 3)
    agent = PIMCPolicyAgent(_zero_value, n_determinizations=3, seed=0)
    for _ in range(5):
        assert agent.act(state, 3) in legal


def test_pimc_picks_the_move_the_stub_ranks_best():
    state = _state_a()
    agent = PIMCPolicyAgent(_pip_shed_value, n_determinizations=4, seed=0)
    # Heaviest legal tile is [2|6] (id 17, 8 pips); unique maximum, so the
    # expected action is action_index(17, 0) == 34.
    expected = action_index(17, 0)
    assert tile_pips(17) == 8
    legal = state.legal_moves(3)
    heaviest = max(legal, key=lambda move: tile_pips(move[0]))
    assert action_index(*heaviest) == expected
    assert agent.act(state, 3) == expected


def test_pimc_batches_value_fn_once_per_root_move():
    state = _state_a()
    calls: list[tuple[int, ...]] = []

    def counting_value(observations):
        array = np.asarray(observations)
        calls.append(array.shape)
        return np.zeros(array.shape[0], dtype=np.float64)

    n_determinizations = 4
    agent = PIMCPolicyAgent(counting_value, n_determinizations=n_determinizations, seed=0)
    agent.act(state, 3)

    n_moves = len(state.legal_moves(3))
    assert n_moves == 7
    assert len(calls) == n_moves                       # one batched call per move
    assert all(shape == (n_determinizations, OBS_DIM) for shape in calls)


def test_pimc_single_legal_move_returns_immediately_without_value_fn():
    # ends (0,1); seat3's only playable tile is [0|0] (id 0).
    state = _make_state(
        hands=[
            [2, 3, 4, 5, 6, 7],            # seat0 (played the [0|1] lead)
            [8, 9, 10, 11, 12, 19, 20],    # seat1
            [21, 23, 24, 25, 26, 27, 13],  # seat2
            [0, 14, 15, 16, 17, 18, 22],   # seat3 = me
        ],
        table=[1],
        played_by=[[1], [], [], []],
        ends=[0, 1],
        turn=3,
        leader=0,
        moves=[(0, 1, 0)],
    )
    assert state.legal_moves(3) == [(0, 0)]

    def exploding_value(observations):  # must never be called
        raise AssertionError("value_fn must not be called for a single legal move")

    agent = PIMCPolicyAgent(exploding_value, n_determinizations=8, seed=0)
    assert agent.act(state, 3) == action_index(0, 0)


def test_pimc_raises_on_a_forced_pass_position():
    # ends (0,1); seat3 holds no tile carrying 0 or 1 -> no legal move at all.
    state = _make_state(
        hands=[
            [2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 19, 20],
            [0, 21, 23, 24, 25, 26, 27],
            [13, 14, 15, 16, 17, 18, 22],  # seat3 = me
        ],
        table=[1],
        played_by=[[1], [], [], []],
        ends=[0, 1],
        turn=3,
        leader=0,
        moves=[(0, 1, 0)],
    )
    assert state.legal_moves(3) == []
    with pytest.raises(ValueError):
        PIMCPolicyAgent(_zero_value, seed=0).act(state, 3)


def test_pimc_leaf_observations_fill_the_privileged_block_with_the_sample():
    state = _state_a()
    observed: list[np.ndarray] = []

    def capture_value(observations):
        observed.append(np.asarray(observations).copy())
        return np.zeros(np.asarray(observations).shape[0], dtype=np.float64)

    # One determinisation -> the privileged block must equal that exact draw.
    # A fresh agent seeded 3 draws its first sample identically to a fresh
    # Determinizer(seed=3), so we can predict the exact hands it used.
    agent = PIMCPolicyAgent(capture_value, n_determinizations=1, seed=3)
    agent.act(state, 3)

    sample = Determinizer(seed=3).sample(state, 3)
    privileged = observed[0][0, PRIVILEGED_SLICE].reshape(3, NUM_TILES)
    for r, hand in enumerate(sample):          # [left, partner, right] order
        for tid in hand:
            assert privileged[r, tid] == 1.0
        assert privileged[r].sum() == len(hand)


# ---- model adapter ---------------------------------------------------------
def test_value_fn_from_model_wraps_predict_values():
    torch = pytest.importorskip("torch")

    class _Policy:
        def predict_values(self, tensor):
            return tensor.sum(dim=1)

    class _Model:
        policy = _Policy()

    value_fn = value_fn_from_model(_Model())
    obs = np.zeros((3, OBS_DIM), dtype=np.float32)
    obs[:, 0] = 1.0
    out = np.asarray(value_fn(obs))
    assert out.shape == (3,)
    assert np.allclose(out, 1.0)
