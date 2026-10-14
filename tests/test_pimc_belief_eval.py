"""Tests for the belief-head calibration: metrics, baseline, collector."""
from __future__ import annotations

import math

import numpy as np
import pytest

from domino.agents import RuleAgent
from domino.pimc.belief_eval import (
    SEAT_SLOT,
    baseline_marginals,
    binary_log_loss,
    brier_score,
    collect_belief_samples,
    expected_calibration_error,
    head_marginals,
    informative_mask,
    score,
)
from domino.pimc.sampler import PublicView, enumerate_worlds, sample_worlds
from domino.obs import OBS_DIM, OPPONENT_OFFSETS, PRIVILEGED_SLICE, encode
from domino.tiles import NUM_TILES


def make_state(hands: list[list[int]], turn: int = 0, table: tuple[int, int] | None = None):
    """A minimal opening state for the tests (mirrors the solver tests' helper)."""
    from domino.engine import GameState

    return GameState(hands=[list(hand) for hand in hands], rng=np.random.default_rng(0),
                     turn=turn, leader=turn, deal_no=1, forced_leader_six_six=False)


def late_position(plays_per_seat: int = 5, seed: int = 0, seat: int = 0):
    """A *valid* late-deal state: a real 28-tile deal advanced by ~``plays_per_seat`` each.

    Short hands cannot be faked in a fixture: the sampler's view assumes every tile outside
    my hand and the table belongs to an opponent, so a 2-tile-hand state leaves 20 tiles
    unaccounted for and the view's counts no longer match its own unseen pool. The only way
    to a small enumeration is a real deal played down, which is what this helper does (each
    seat plays its first legal move; forced passes are taken as they come).

    The seats do not all get the same number of plays - a seat that is forced to pass falls
    behind, and the deal can end early - so the stop condition is the *total* number of
    plays, not a per-seat count.
    """
    from domino.tiles import ALL_TILE_IDS, SIX_SIX_ID

    rng = np.random.default_rng(seed)
    tiles = list(ALL_TILE_IDS)
    rng.shuffle(tiles)
    hands = [sorted(tiles[index * 7:(index + 1) * 7]) for index in range(4)]
    leader = next(index for index, hand in enumerate(hands) if SIX_SIX_ID in hand)
    state = make_state(hands, turn=leader)
    state.forced_leader_six_six = True
    target = 4 * int(plays_per_seat)
    played = 0
    for _ in range(400):
        if played >= target or state.is_terminal():
            break
        turn = int(state.turn)
        moves = state.legal_moves(turn)
        if not moves:
            state.pass_turn(turn)
            continue
        tid, end = moves[0]
        state.play(turn, tid, end)
        played += 1
    assert not state.is_terminal(), "the deal ended before the fixture target"
    return state, seat


# ---- metrics ---------------------------------------------------------------
def test_log_loss_and_brier_have_known_values() -> None:
    """Spot values a reader can verify by hand."""
    assert binary_log_loss([0.5, 0.5], [1.0, 0.0]) == pytest.approx(math.log(2.0))
    assert binary_log_loss([0.9], [1.0]) == pytest.approx(-math.log(0.9))
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.25)
    assert binary_log_loss([0.0], [1.0]) < 30.0        # clipped, not inf


def test_ece_is_zero_when_calibrated_and_large_when_not() -> None:
    """A calibrated predictor has ECE ~ 0; an overconfident one does not."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    ece, table = expected_calibration_error(p, y, bins=10)
    assert ece < 0.05, ece
    assert sum(row[1] for row in table) == 4000
    biased, _ = expected_calibration_error(np.full(1000, 0.9), np.zeros(1000))
    assert biased == pytest.approx(0.9, abs=1e-6)


def test_score_reports_all_four_metrics() -> None:
    """The Score dataclass carries entries, loss, Brier, ECE and accuracy."""
    result = score(np.array([0.9, 0.1, 0.6]), np.array([1.0, 0.0, 1.0]))
    assert result.entries == 3
    assert result.accuracy == pytest.approx(1.0)
    assert 0.0 < result.log_loss < 1.0
    assert result.as_row("head")["predictor"] == "head"


# ---- the baseline ----------------------------------------------------------
def test_baseline_marginals_match_exact_enumeration() -> None:
    """On a small position the Monte-Carlo baseline must equal the exact marginals."""
    state, seat = late_position(plays_per_seat=5, seed=0)
    view = PublicView.from_state(state, seat)
    # A real position always satisfies the sampler's invariant, whatever the split is.
    assert sum(view.counts) == len(view.unseen) <= 8
    worlds = enumerate_worlds(view)
    exact = np.zeros((3, NUM_TILES))
    for world in worlds:
        hands = world.as_tuple()
        for offset, slot in SEAT_SLOT.items():
            for tile in hands[(view.seat + offset) % 4]:
                exact[slot, tile] += 1.0
    exact /= len(worlds)
    estimated, error = baseline_marginals(view, worlds=20_000, rng=np.random.default_rng(1))
    assert np.allclose(estimated, exact, atol=0.02), np.abs(estimated - exact).max()
    assert np.all(error <= 0.05)
    # Every unseen tile is held by exactly one of the three opponents, so the marginals
    # over those seats sum to the number of unseen tiles.
    assert estimated.sum() == pytest.approx(float(len(view.unseen)))


def test_informative_mask_excludes_my_hand_and_the_table() -> None:
    """Only tiles whose owner is genuinely unknown are scored: not mine, not played."""
    state, seat = late_position(plays_per_seat=5, seed=0)
    view = PublicView.from_state(state, seat)
    mask = informative_mask(view)
    assert mask.shape == (3, 28)
    assert not mask[:, list(view.my_hand)].any()           # my own hand
    assert not mask[:, list(view.table)].any()             # every tile already played
    assert mask[:, list(view.unseen)].all()                # the ones actually in doubt


def test_worlds_are_absolute_seat_ordered_in_both_methods() -> None:
    """``World`` hands are indexed by absolute seat - the baseline mapping depends on it."""
    rng = np.random.default_rng(3)
    state, seat = late_position(plays_per_seat=5, seed=0, seat=2)
    view = PublicView.from_state(state, seat)
    for method in ("rejection", "constrained"):
        for world in sample_worlds(view, 12, rng, method=method):
            hands = world.as_tuple()
            assert tuple(sorted(hands[view.seat])) == tuple(sorted(view.my_hand))
            others = sorted(tile for st in view.other_seats for tile in hands[st])
            assert others == sorted(view.unseen)
            assert tuple(len(hands[st]) for st in view.other_seats) == view.counts


def test_enumerate_worlds_rejects_an_unsatisfiable_view() -> None:
    """The guard that stops a mis-specified view from becoming a memory bomb."""
    state, seat = late_position(plays_per_seat=5, seed=0)
    wrong = PublicView.from_state(state, seat, tiles_per_seat=6)
    assert sum(wrong.counts) != len(wrong.unseen)       # the view contradicts its own pool
    with pytest.raises(ValueError, match="unsatisfiable"):
        enumerate_worlds(wrong)


# ---- the collector ---------------------------------------------------------
def test_collector_records_truthful_labels_and_views() -> None:
    """Labels are the real hands, the obs is the env's own encoding, plies increment."""
    agents = {seat: RuleAgent(seed=seat) for seat in range(4)}
    samples = collect_belief_samples(agents, deals=3, seed=0)
    assert samples, "no decision points collected"
    for sample in samples:
        assert sample.obs.shape == (OBS_DIM,)
        assert sample.obs.dtype == np.float32
        assert sample.view.seat == sample.seat
        assert sample.labels.shape == (3, 28)
        # 28 tiles = my hand + the table + the three opponents' hands.
        my_hand = float(sample.obs[0:28].sum())
        table = float(sample.obs[28:56].sum())
        assert sample.labels.sum() == pytest.approx(28.0 - my_hand - table)
    seats = {sample.seat for sample in samples}
    assert seats == {0, 2}
    assert {sample.phase for sample in samples} <= {"opening", "mid", "end"}


def test_collected_labels_agree_with_the_public_counts_in_relative_order() -> None:
    """The privileged block and the public counts describe the same hands, seat for seat.

    The counts must be read with the observation's ``[left, partner, right]`` offsets: the
    view's own ``counts`` tuple is in *absolute* seat order, which silently differs for the
    partner seat and looked exactly like a one-hand-size bug when compared directly.
    """
    agents = {seat: RuleAgent(seed=seat) for seat in range(4)}
    samples = collect_belief_samples(agents, deals=1, seed=0)
    assert samples
    for sample in samples:
        for slot, offset in enumerate(OPPONENT_OFFSETS):
            absolute = (sample.seat + offset) % 4
            held = sample.obs[PRIVILEGED_SLICE].reshape(3, NUM_TILES)[slot]
            assert held.sum() == sample.view.count_of(absolute)
            assert sample.view.counts_relative[slot] == sample.view.count_of(absolute)
            assert np.array_equal(held, sample.labels[slot])
        assert len(set(sample.view.unseen)) == sum(sample.view.counts)


def test_collected_labels_agree_with_the_absolute_counts() -> None:
    """The same check against the view's absolute-order counts, mapped explicitly."""
    agents = {seat: RuleAgent(seed=seat) for seat in range(4)}
    samples = collect_belief_samples(agents, deals=1, seed=0)
    for sample in samples:
        for slot, absolute in enumerate(sample.view.other_seats):
            held = sum(1 for tile in sample.view.unseen
                       if sample.obs[PRIVILEGED_SLICE].reshape(3, NUM_TILES)[
                           _slot_of(sample, absolute)][tile] > 0.5)
            assert held == sample.view.counts[slot]


def _slot_of(sample, absolute: int) -> int:
    """Label slot of an absolute seat, relative to the sample's acting seat."""
    return (absolute - sample.seat - 1) % 4


# ---- the head --------------------------------------------------------------
def test_head_marginals_shapes_and_range() -> None:
    """A stub policy with predict_beliefs returns (N, 3, 28) probabilities in [0, 1]."""
    import torch as th

    class Stub:
        def predict_beliefs(self, observations):           # noqa: D102 - test stub
            batch = th.as_tensor(np.asarray(observations), dtype=th.float32)
            return th.zeros((batch.shape[0], 84))

    out = head_marginals(Stub(), np.zeros((5, OBS_DIM), dtype=np.float32))
    assert out.shape == (5, 3, 28)
    assert np.allclose(out, 0.5)
