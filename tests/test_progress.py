"""Tests for the in-training progress gate (``domino/progress.py``).

Three things can silently go wrong with a callback that evaluates the live policy, and each has a
test here:

* it measures a *different protocol* than the project's own evaluation (pinned against
  :func:`domino.pimc.harness.play_duplicate_pair` on identical seeds);
* its verdict logic misfires (pure-function tests over every branch, including the frozen-deal
  signature that wasted a 10M-step run on this project);
* it disturbs training (a real short ``learn`` run through the real entry point, asserting the
  step budget is reached and that a broken run stops early instead of burning the budget).
"""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from domino.agents import RandomAgent, RuleAgent
from domino.config import TrainConfig
from domino.env import DominoEnv
from domino.pimc.harness import AgentSpec, play_duplicate_pair
from domino.policy import DominoMaskablePolicy, policy_kwargs_for
from domino.progress import (
    ANCHOR_EARLY,
    ANCHOR_RANDOM,
    ANCHOR_RULE,
    CSV_HEADER,
    PeriodicEvalCallback,
    VERDICT_BROKEN,
    VERDICT_KEEP,
    VERDICT_NO_GAIN,
    VERDICT_UNKNOWN,
    classify,
    play_duplicate_pairs,
    read_health,
    resolve_early_anchor,
)

N_ENVS = 2
N_STEPS = 8


def _vec_env(seed: int = 0) -> DummyVecEnv:
    """Two in-process workers against the rule bot, as the callback tests do."""
    factories = [partial(DominoEnv, opponent=RuleAgent(seed=rank), seed=seed + rank)
                 for rank in range(N_ENVS)]
    return DummyVecEnv(factories)


def _model(seed: int = 0) -> MaskablePPO:
    """A tiny real MaskablePPO on the project's custom policy."""
    model = MaskablePPO(
        DominoMaskablePolicy,
        _vec_env(seed),
        policy_kwargs=policy_kwargs_for(TrainConfig()),
        n_steps=N_STEPS,
        batch_size=N_STEPS,
        seed=seed,
        device="cpu",
    )
    return model


def _train(model: MaskablePPO, callbacks, steps: int = 1024) -> MaskablePPO:
    """Run a short real training pass with the given callbacks."""
    model.learn(total_timesteps=int(steps), callback=callbacks, progress_bar=False)
    return model


# --------------------------------------------------------------------------- #
# verdict logic (pure)
# --------------------------------------------------------------------------- #
def test_health_defaults_are_unknown_not_broken() -> None:
    """A logger that has just started must not read as a collapsed policy."""
    verdict, _reason = classify(entropy=None, approx_kl=None, clip_fraction=None,
                                policy_gradient_loss=None)
    assert verdict == VERDICT_UNKNOWN


def test_full_low_diversity_window_is_broken() -> None:
    """The frozen-deal signature outranks everything else."""
    verdict, reason = classify(
        entropy=0.4, approx_kl=1e-3, clip_fraction=0.1, policy_gradient_loss=-0.2,
        distinct_deals=1, window_full=True, rule_win_rate=0.5, rule_paired_low=-1.0,
    )
    assert verdict == VERDICT_BROKEN
    assert "distinct deal" in reason


def test_incomplete_diversity_window_is_not_broken() -> None:
    """A window that is not full yet says nothing about replaying deals."""
    verdict, _ = classify(
        entropy=0.4, approx_kl=1e-3, clip_fraction=0.1, policy_gradient_loss=-0.2,
        distinct_deals=1, window_full=False, rule_win_rate=0.5, rule_paired_low=1.0,
    )
    assert verdict == VERDICT_KEEP


def test_entropy_collapse_with_dead_gradient_is_broken() -> None:
    """Entropy ~0 with kl/clip/policy-loss ~0 is the no-gradient signature, not convergence."""
    verdict, reason = classify(
        entropy=0.0001, approx_kl=0.0, clip_fraction=0.0, policy_gradient_loss=-1e-7,
        distinct_deals=100, window_full=True, rule_win_rate=0.44, rule_paired_low=-3.0,
    )
    assert verdict == VERDICT_BROKEN
    assert "no gradient signal" in reason


def test_flat_anchor_with_zero_interval_is_no_gain() -> None:
    """No movement and a paired interval straddling zero is information, not a stop."""
    verdict, _reason = classify(
        entropy=0.3, approx_kl=1e-3, clip_fraction=0.05, policy_gradient_loss=-0.1,
        distinct_deals=100, window_full=True, rule_win_rate=0.51, rule_paired_low=-0.4,
        previous_rule_win_rate=0.51,
    )
    assert verdict == VERDICT_NO_GAIN


def test_rising_anchor_is_keep_even_with_zero_interval() -> None:
    """A rise in the win rate is progress even while the paired interval still includes zero."""
    verdict, _ = classify(
        entropy=0.3, approx_kl=1e-3, clip_fraction=0.05, policy_gradient_loss=-0.1,
        distinct_deals=100, window_full=True, rule_win_rate=0.55, rule_paired_low=-0.2,
        previous_rule_win_rate=0.50,
    )
    assert verdict == VERDICT_KEEP


def test_read_health_negates_the_logged_entropy_loss() -> None:
    """SB3 logs ``-mean(entropy)`` in nats; dividing by ent_coef is the documented trap."""
    logger = SimpleNamespace(name_to_value={
        "train/entropy_loss": -0.28, "train/approx_kl": 0.003, "train/clip_fraction": 0.02,
        "train/policy_gradient_loss": -0.15,
    })
    health = read_health(logger)
    assert health["entropy"] == pytest.approx(0.28)
    assert health["approx_kl"] == pytest.approx(0.003)
    assert health["policy_gradient_loss"] == pytest.approx(-0.15)


def test_read_health_returns_none_for_missing_keys() -> None:
    """Missing keys must not be read as 0.0 (which the collapse rule would misread)."""
    health = read_health(SimpleNamespace(name_to_value={}))
    assert all(value is None for value in health.values())


# --------------------------------------------------------------------------- #
# the duplicate-deal protocol
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_paired_loop_matches_the_harness_protocol(seed: int) -> None:
    """One pinned deal must give bit-identical results to ``harness.play_duplicate_pair``.

    The callback must not become a second, quieter protocol: if these two drift apart, every
    progress row means something different from the numbers the report quotes.
    """
    reference = play_duplicate_pair(AgentSpec("rule", "rule"), AgentSpec("random", "random"), seed)
    mine = play_duplicate_pairs(
        lambda seat_seed: RuleAgent(seed=seat_seed),
        lambda seat_seed: RandomAgent(seed=seat_seed),
        deals=1, seed=seed, bootstrap_draws=200,
    )
    expected_diff = (reference["differential_game1"] + reference["differential_game2"]) / 2.0
    assert mine["paired_differential"] == pytest.approx(expected_diff)
    assert mine["wins"] == int(reference["won_a_game1"]) + int(reference["won_a_game2"])
    assert mine["win_rate_dealt"] == pytest.approx(float(reference["won_a_game1"]))
    assert mine["win_rate_rotated"] == pytest.approx(float(reference["won_a_game2"]))
    assert mine["block_rate"] == pytest.approx(float(reference["blocked"]) / 2.0)


def test_paired_metrics_are_internally_consistent() -> None:
    """Games, win rate, Wilson band and the bootstrap ordering all have to line up."""
    metrics = play_duplicate_pairs(
        lambda seat_seed: RuleAgent(seed=seat_seed),
        lambda seat_seed: RandomAgent(seed=seat_seed),
        deals=20, seed=7, bootstrap_draws=500,
    )
    assert metrics["deals"] == 20
    assert metrics["games"] == 40
    assert metrics["wins"] <= metrics["games"]
    assert metrics["win_rate"] == pytest.approx(metrics["wins"] / metrics["games"])
    assert metrics["ci_low"] <= metrics["win_rate"] <= metrics["ci_high"]
    assert metrics["paired_low"] <= metrics["paired_differential"] <= metrics["paired_high"]
    assert 0.0 <= metrics["block_rate"] <= 1.0
    assert metrics["mean_pips_left"] >= 0.0


def test_paired_loop_rejects_a_zero_deal_count() -> None:
    """A silent zero would write a row with no games behind it."""
    with pytest.raises(ValueError):
        play_duplicate_pairs(lambda seat_seed: RuleAgent(seed=seat_seed),
                             lambda seat_seed: RandomAgent(seed=seat_seed), deals=0, seed=0)


# --------------------------------------------------------------------------- #
# the early anchor resolver
# --------------------------------------------------------------------------- #
def test_resolve_early_anchor_picks_the_run_start(tmp_path) -> None:
    """The earliest snapshot at or after the run's start step is the run's own starting policy."""
    for step in (0, 100_000, 500_000):
        (tmp_path / f"policy_{step}.zip").write_bytes(b"")
    (tmp_path / "not_a_snapshot.zip").write_bytes(b"")
    assert resolve_early_anchor(tmp_path, since_step=0).endswith("policy_0.zip")
    assert resolve_early_anchor(tmp_path, since_step=100_000).endswith("policy_100000.zip")
    assert resolve_early_anchor(tmp_path, since_step=600_000) is None


def test_resolve_early_anchor_without_a_directory(tmp_path) -> None:
    """A missing snapshot directory is a skipped leg, never a crash mid-training."""
    assert resolve_early_anchor(tmp_path / "nope", since_step=0) is None


# --------------------------------------------------------------------------- #
# the callback on a real (short) run
# --------------------------------------------------------------------------- #
def test_progress_callback_evaluates_the_live_policy(tmp_path) -> None:
    """A real run must produce one row per (evaluation, anchor) with sane numbers."""
    run_name = "progress_test"
    callback = PeriodicEvalCallback(
        run_name, results_dir=tmp_path, every=256, deals=4, anchors=(ANCHOR_RULE,),
        stop_on_broken=False,
    )
    model = _train(_model(), callback, steps=1024)

    assert model.num_timesteps == 1024          # the evaluation never touches the training clock
    assert callback.rows, "the progress gate produced no rows at all"
    frame = pd.read_csv(tmp_path / f"progress_{run_name}.csv")
    assert list(frame.columns) == list(CSV_HEADER)
    assert len(frame) == len(callback.rows)
    assert set(frame["anchor"]) == {ANCHOR_RULE}
    assert frame["step"].is_monotonic_increasing
    assert ((frame["win_rate"] >= 0.0) & (frame["win_rate"] <= 1.0)).all()
    assert (frame["games"] == 2 * callback.deals).all()
    assert frame["eval_seconds"].gt(0.0).all()
    assert callback.last_eval_seconds is not None
    # Every evaluation plays the same deals, so the win-rate column may repeat but must be
    # consistent with the games/wins it reports.
    assert np.allclose(frame["win_rate"], frame["wins"] / frame["games"])


def test_progress_callback_snapshots_its_own_starting_anchor(tmp_path) -> None:
    """The early anchor is this run's own starting weights, written next to the CSV.

    Not the league snapshot directory: that is pruned to the newest ``league_size`` files, which
    deletes a fresh run's ``policy_0.zip`` in the same call that writes it (measured -- the first
    version of this callback resolved to a previous run's 15.3M-step policy).
    """
    callback = PeriodicEvalCallback(
        "progress_anchor", results_dir=tmp_path, every=256, deals=2,
        anchors=(ANCHOR_RULE, ANCHOR_EARLY), stop_on_broken=False,
    )
    _train(_model(), callback, steps=512)

    assert callback.anchor_copy_path.exists()
    assert callback._resolved_anchor == str(callback.anchor_copy_path)
    frame = pd.read_csv(tmp_path / "progress_progress_anchor.csv")
    assert set(frame["anchor"]) == {ANCHOR_RULE, ANCHOR_EARLY}
    assert len(frame) == 2 * 2                  # two evaluations x two legs


def test_progress_callback_skips_the_anchor_leg_when_it_cannot_be_written(tmp_path) -> None:
    """An unwritable anchor directory skips that leg and says so; training must not die."""
    blocked = tmp_path / "a_file_not_a_dir"
    blocked.write_text("not a directory")
    callback = PeriodicEvalCallback(
        "progress_blocked", results_dir=tmp_path, every=256, deals=2,
        anchors=(ANCHOR_RULE, ANCHOR_EARLY), anchor_dir=blocked, stop_on_broken=False,
    )
    model = _train(_model(), callback, steps=512)

    assert model.num_timesteps == 512           # the run was not harmed by the skip
    frame = pd.read_csv(tmp_path / "progress_progress_blocked.csv")
    assert set(frame["anchor"]) == {ANCHOR_RULE}


def test_progress_callback_stops_a_run_with_the_frozen_deal_signature(tmp_path) -> None:
    """Two consecutive BROKEN readings halt training instead of burning the step budget."""
    stats = SimpleNamespace(diversity=SimpleNamespace(distinct=1, full=True))
    callback = PeriodicEvalCallback(
        "progress_broken", results_dir=tmp_path, every=128, deals=2,
        anchors=(ANCHOR_RULE,), stats=stats, stop_on_broken=True, broken_patience=2,
    )
    model = _train(_model(), callback, steps=4096)

    assert model.num_timesteps < 4096, "the run was not stopped despite a frozen-deal signature"
    frame = pd.read_csv(tmp_path / "progress_progress_broken.csv")
    assert set(frame["verdict"]) == {VERDICT_BROKEN}
    assert len(frame) == 2                      # exactly the readings the patience allows
