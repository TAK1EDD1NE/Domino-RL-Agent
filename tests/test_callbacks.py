"""Tests for the training callbacks (spec section 9, plan Tasks 10-11).

These exercise the assignment's convergence requirement end to end on a short
real ``MaskablePPO`` run:

* the CSV mirror has the exact header and >= 10 rows and its ``window_avg``
  column equals a pandas ``rolling(100, min_periods=1).mean()`` recomputation;
* the TensorBoard event file carries ``train/total_reward_per_episode`` and
  ``train/avg_episode_reward_window_100``;
* each per-episode total equals the sum of the rewards the env returned, checked
  independently against SB3's ``VecMonitor`` episode returns;
* snapshots appear, at most ``league_size`` survive, and the worker pool receives
  them through ``refresh_pool``.

The envs run in a ``DummyVecEnv`` (in-process) so the tests can inspect the
worker's ``OpponentPool`` directly; the ``SubprocVecEnv`` spawn path is covered
by the ``train.py`` smoke test.
"""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from domino.agents import OpponentPool
from domino.callbacks import (
    CSV_HEADER,
    EpisodeStatsCallback,
    SnapshotCallback,
    TAG_TOTAL_REWARD,
    TAG_WINDOW_AVG,
    enable_checkpoint_loading,
)
from domino.config import TrainConfig
from domino.env import DominoEnv
from domino.policy import DominoMaskablePolicy, policy_kwargs_for

# The league loads frozen snapshots during training; the torch/SB3 workaround
# must be active in this process too (see domino.callbacks.enable_checkpoint_loading).
enable_checkpoint_loading()

WORKER_WEIGHTS = {"snapshot": 0.6, "rule": 0.25, "random": 0.15}
RUN_NAME = "cb_test"
STEPS = 2048
N_ENVS = 2


def _make_env(seed: int) -> DominoEnv:
    """Module-level env factory for the in-process vector env used here."""
    pool = OpponentPool(weights=dict(WORKER_WEIGHTS), seed=seed)
    return DominoEnv(opponent=pool, seed=seed)


@pytest.fixture
def short_run(tmp_path):
    """Run a short MaskablePPO job with both callbacks into a temp directory."""
    runs_dir = tmp_path / "runs" / RUN_NAME
    results_dir = tmp_path / "results"
    snapshot_dir = tmp_path / "models" / "snapshots"

    raw_venv = DummyVecEnv([partial(_make_env, seed=i) for i in range(N_ENVS)])
    venv = VecMonitor(raw_venv)
    model = MaskablePPO(
        DominoMaskablePolicy,
        venv,
        policy_kwargs=policy_kwargs_for(TrainConfig()),
        tensorboard_log=str(runs_dir),
        n_steps=256,
        batch_size=256,
        n_epochs=2,
        seed=0,
        device="cpu",
        verbose=0,
    )

    stats_cb = EpisodeStatsCallback(run_name=RUN_NAME, results_dir=str(results_dir))
    snapshot_cb = SnapshotCallback(snapshot_every=512, snapshot_dir=str(snapshot_dir), league_size=2)
    # Seed the league exactly as train.py does (before learn).
    snapshot_cb.save(model, step=0, notify=True)

    model.learn(total_timesteps=STEPS, callback=CallbackList([stats_cb, snapshot_cb]))
    venv.close()

    return SimpleNamespace(
        model=model,
        raw_venv=raw_venv,
        stats=stats_cb,
        snapshot=snapshot_cb,
        csv_path=results_dir / f"training_{RUN_NAME}.csv",
        runs_dir=runs_dir,
        snapshot_dir=snapshot_dir,
    )


def _read_csv(path):
    return pd.read_csv(path)


# 1 ---------------------------------------------------------------------------
def test_csv_mirror_header_rows_and_window_average(short_run):
    df = _read_csv(short_run.csv_path)
    assert list(df.columns) == list(CSV_HEADER)
    assert len(df) >= 10, f"expected >= 10 episodes, got {len(df)}"
    assert df["episode"].tolist() == list(range(1, len(df) + 1))

    # window_avg must equal a pandas rolling(100, min_periods=1).mean() of the
    # total_reward column (the assignment's sliding-window average).
    expected = df["total_reward"].rolling(100, min_periods=1).mean()
    assert np.allclose(df["window_avg"].to_numpy(), expected.to_numpy(), atol=1e-9)

    # The in-memory mirror agrees with the file.
    assert len(short_run.stats.rows) == len(df)


# 2 ---------------------------------------------------------------------------
def test_tensorboard_event_file_has_convergence_scalars(short_run):
    events = list(short_run.runs_dir.rglob("events.out.tfevents.*"))
    assert events, "no TensorBoard event file was written"
    accumulator = EventAccumulator(str(events[0].parent))
    accumulator.Reload()
    scalars = set(accumulator.Tags()["scalars"])
    assert TAG_TOTAL_REWARD in scalars
    assert TAG_WINDOW_AVG in scalars
    assert "deals/block_rate" in scalars
    assert "deals/void_rate" in scalars
    assert "deals/points_per_deal" in scalars
    assert len(accumulator.Scalars(TAG_TOTAL_REWARD)) >= 10


# 3 ---------------------------------------------------------------------------
def test_per_episode_total_equals_sum_of_env_rewards(short_run):
    df = _read_csv(short_run.csv_path)
    csv_totals = df["total_reward"].to_numpy()

    # VecMonitor computes the episode return independently from the rewards it
    # sees; SB3 exposes them (undiscounted) in ep_info_buffer.
    reference = [float(info["r"]) for info in short_run.model.ep_info_buffer]
    assert len(reference) >= 10, f"ep_info_buffer too short: {len(reference)}"

    n = min(len(reference), len(csv_totals))
    assert np.allclose(csv_totals[-n:], np.asarray(reference[-n:]), atol=1e-6)


# 4 ---------------------------------------------------------------------------
def test_snapshots_written_pruned_and_delivered_to_pool(short_run):
    files = sorted(short_run.snapshot_dir.glob("policy_*.zip"))
    assert files, "no snapshot files were written"
    assert len(files) <= short_run.snapshot.league_size

    pool = short_run.raw_venv.envs[0]._opponent
    assert isinstance(pool, OpponentPool)
    assert pool._snapshot_paths, "the pool received no snapshot paths"
    assert sorted(pool._snapshot_paths) == sorted(short_run.snapshot.snapshots)
    # The pool can now sample a frozen snapshot opponent.
    assert "snapshot" in pool._available_weights()


# 5 ---------------------------------------------------------------------------
def test_callback_constructible_without_model_and_window_validation(tmp_path):
    cb = EpisodeStatsCallback(run_name="x", results_dir=str(tmp_path / "r"), window=5)
    assert cb.csv_path.name == "training_x.csv"
    assert cb.rows == []
    with pytest.raises(ValueError):
        EpisodeStatsCallback(run_name="x", results_dir=str(tmp_path), window=0)
    with pytest.raises(ValueError):
        SnapshotCallback(snapshot_every=0)
    with pytest.raises(ValueError):
        SnapshotCallback(snapshot_every=10, league_size=0)
