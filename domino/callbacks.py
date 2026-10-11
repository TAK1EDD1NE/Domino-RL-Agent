"""Training callbacks: undiscounted convergence logging and league snapshots.

The assignment (and spec section 9) requires the convergence metric to be the
**total, undiscounted reward per episode** logged to TensorBoard, together with
a sliding-window average. This module implements exactly that, plus the league
snapshotting that keeps the vectorised opponent pool fed with fresh frozen
checkpoints.

Three callbacks live here, all constructible with explicit paths/names and free
of global state so a test can point them at a temporary directory:

``EpisodeStatsCallback``
    Accumulates the reward the environment actually returns for every
    environment of a :class:`~stable_baselines3.common.vec_env.VecEnv` and, at
    every finished deal (episode), writes

    * the undiscounted episode total to ``train/total_reward_per_episode``,
    * the mean of the last ``window`` totals to
      ``train/avg_episode_reward_window_100``,
    * the running deal diagnostics ``deals/block_rate``, ``deals/void_rate`` and
      ``deals/points_per_deal``,

    and appends one row ``step,episode,total_reward,window_avg`` to
    ``results/training_<run_name>.csv``.

    The per-episode total is accumulated **directly** from ``self.locals[
    "rewards"]`` (the rewards the vector env just returned) so it is, by
    construction, the sum of the rewards the environment returned for that
    episode -- never a discounted return. SB3 reports undiscounted returns in
    ``ep_info_buffer`` as well; this callback reproduces that quantity exactly
    while additionally aligning each episode with its ``num_timesteps`` and with
    the deal diagnostics carried in the environment's ``info``.

``SnapshotCallback``
    Every ``snapshot_every`` steps saves ``policy_<num_timesteps>.zip`` under
    ``snapshot_dir``, keeps only the newest ``league_size`` snapshots, and then
    notifies every worker with
    ``vec_env.env_method("refresh_pool", paths)`` so each worker's
    :class:`~domino.agents.OpponentPool` sees the current frozen checkpoints.

``CheckpointCallback``
    Saves ``latest.zip`` plus ``ppo_<num_timesteps>_steps.zip`` every
    ``checkpoint_every`` steps, so a run can be resumed from the newest state.

Nothing here imports ``pygame``; the module is on the training path only.
"""
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

#: Width of the sliding window used for the average episode reward (assignment).
WINDOW_SIZE: int = 100
#: Exact header of the training CSV mirror (results/training_<run_name>.csv).
CSV_HEADER: tuple[str, ...] = ("step", "episode", "total_reward", "window_avg")

# TensorBoard scalar tags.
TAG_TOTAL_REWARD = "train/total_reward_per_episode"
TAG_WINDOW_AVG = "train/avg_episode_reward_window_100"
TAG_BLOCK_RATE = "deals/block_rate"
TAG_VOID_RATE = "deals/void_rate"
TAG_POINTS_PER_DEAL = "deals/points_per_deal"


def _tensorboard_writer(logger: Any) -> tuple[Any | None, bool]:
    """Return ``(writer, owned)`` for ``logger``'s TensorBoard output.

    Prefers the writer SB3 already created (so the convergence scalars land in
    the *same* event file as SB3's own ``rollout/*`` series); otherwise creates a
    fresh :class:`~torch.utils.tensorboard.SummaryWriter` in the run directory.
    ``owned`` is ``True`` when the caller must close the writer.
    """
    for output_format in getattr(logger, "output_formats", None) or []:
        writer = getattr(output_format, "writer", None)
        if writer is not None and hasattr(writer, "add_scalar"):
            return writer, False
    log_dir = getattr(logger, "dir", None)
    if log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(log_dir=str(log_dir)), True
        except Exception:  # pragma: no cover - tensorboard always installed here
            return None, False
    return None, False


class EpisodeStatsCallback(BaseCallback):
    """Log the undiscounted total reward per episode and a sliding-window mean.

    Parameters
    ----------
    run_name:
        Name used for the CSV mirror (``results/training_<run_name>.csv``).
    results_dir:
        Directory holding the CSV mirror (created if missing). Defaults to
        ``results``.
    window:
        Sliding-window width for the average (default :data:`WINDOW_SIZE`, 100).
    verbose:
        SB3 verbosity.
    """

    def __init__(
        self,
        run_name: str,
        results_dir: str | Path = "results",
        window: int = WINDOW_SIZE,
        append: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.run_name = str(run_name)
        self.results_dir = Path(results_dir)
        self.window = int(window)
        self.append = bool(append)

        self.csv_path: Path = self.results_dir / f"training_{self.run_name}.csv"
        #: In-memory mirror of the rows written to :attr:`csv_path`.
        self.rows: list[tuple[int, int, float, float]] = []

        self._file = None
        self._writer: Any | None = None
        self._owns_writer = False
        self._ep_reward: np.ndarray | None = None
        self._window: deque[float] = deque(maxlen=self.window)
        self._episode = 0
        self._deals = 0
        self._blocked = 0
        self._voids = 0
        self._points_sum = 0.0

    # ---- SB3 hooks ------------------------------------------------------
    def _init_callback(self) -> None:
        """Open the CSV mirror, reset the accumulators and resolve the writer."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.rows = []
        self._window = deque(maxlen=self.window)
        self._episode = 0
        self._deals = 0
        self._blocked = 0
        self._voids = 0
        self._points_sum = 0.0

        if self.append and self.csv_path.exists():
            self._load_existing()
            self._file = self.csv_path.open("a", newline="", encoding="utf-8")
        else:
            self._file = self.csv_path.open("w", newline="", encoding="utf-8")
            self._file.write(",".join(CSV_HEADER) + "\n")
        self._file.flush()

        self._ep_reward = np.zeros(int(self.training_env.num_envs), dtype=np.float64)
        self._writer, self._owns_writer = _tensorboard_writer(self.logger)

    def _load_existing(self) -> None:
        """Seed the episode counter and sliding window from an existing CSV.

        Used on ``--resume`` so the convergence series continues instead of being
        truncated; the deal diagnostics (block/void) restart because they are not
        recoverable from the CSV.
        """
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header
            for row in reader:
                if len(row) != len(CSV_HEADER):
                    continue
                try:
                    step, episode, total_reward, window_avg = (
                        int(row[0]),
                        int(row[1]),
                        float(row[2]),
                        float(row[3]),
                    )
                except ValueError:
                    continue
                self.rows.append((step, episode, total_reward, window_avg))
                self._window.append(total_reward)
                self._episode = episode
                self._deals += 1

    def _on_step(self) -> bool:
        """Accumulate the returned rewards and record every finished episode."""
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals["dones"]).reshape(-1).astype(bool)
        infos = self.locals["infos"]
        assert self._ep_reward is not None
        self._ep_reward += rewards
        for index in range(dones.shape[0]):
            info = infos[index]
            if bool(info.get("void_deal", False)):
                self._voids += 1
            if dones[index]:
                self._record_episode(float(self._ep_reward[index]), info)
                self._ep_reward[index] = 0.0
        return True

    def _on_training_end(self) -> None:
        """Flush and close the CSV mirror and the (possibly owned) writer."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        if self._writer is not None and self._owns_writer:
            self._writer.close()
        self._writer = None

    # ---- internals ------------------------------------------------------
    def _record_episode(self, total_reward: float, info: dict) -> None:
        """Record one finished episode (deal) across CSV, TB and the counters."""
        self._episode += 1
        self._deals += 1
        if bool(info.get("blocked", False)):
            self._blocked += 1
        points = info.get("points_learner")
        if points is not None:
            self._points_sum += float(points)

        self._window.append(total_reward)
        window_avg = float(np.mean(self._window))
        step = int(self.num_timesteps)
        self.rows.append((step, self._episode, total_reward, window_avg))

        if self._file is not None:
            # repr(float) round-trips exactly, so a pandas recomputation of the
            # rolling mean from this column matches window_avg to machine epsilon.
            self._file.write(f"{step},{self._episode},{total_reward!r},{window_avg!r}\n")
            self._file.flush()

        self._log_scalars(step, total_reward, window_avg)

    def _log_scalars(self, step: int, total_reward: float, window_avg: float) -> None:
        """Write the convergence scalars and the running deal diagnostics."""
        if self._writer is None:
            return
        self._writer.add_scalar(TAG_TOTAL_REWARD, total_reward, step)
        self._writer.add_scalar(TAG_WINDOW_AVG, window_avg, step)
        self._writer.add_scalar(TAG_BLOCK_RATE, self._blocked / self._deals, step)
        attempted = self._deals + self._voids
        self._writer.add_scalar(TAG_VOID_RATE, self._voids / attempted if attempted else 0.0, step)
        self._writer.add_scalar(TAG_POINTS_PER_DEAL, self._points_sum / self._deals, step)


class SnapshotCallback(BaseCallback):
    """Write league snapshots and keep every worker's opponent pool current.

    Parameters
    ----------
    snapshot_every:
        Save a snapshot every this many environment steps (>= 1).
    snapshot_dir:
        Directory for ``policy_<num_timesteps>.zip`` files (created if missing).
    league_size:
        Maximum number of snapshots kept; older ones are deleted.
    verbose:
        SB3 verbosity.
    """

    def __init__(
        self,
        snapshot_every: int,
        snapshot_dir: str | Path = "models/snapshots",
        league_size: int = 10,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if snapshot_every < 1:
            raise ValueError(f"snapshot_every must be >= 1, got {snapshot_every}")
        if league_size < 1:
            raise ValueError(f"league_size must be >= 1, got {league_size}")
        self.snapshot_every = int(snapshot_every)
        self.snapshot_dir = Path(snapshot_dir)
        self.league_size = int(league_size)
        #: Surviving snapshot paths after the most recent save (step order).
        self.snapshots: list[str] = []
        self._last_snapshot = 0

    def _init_callback(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_snapshot < self.snapshot_every:
            return True
        self._last_snapshot = int(self.num_timesteps)
        self.save(self.model, notify=True)
        return True

    def save(self, model: Any, step: int | None = None, notify: bool = False) -> list[str]:
        """Save a snapshot for ``model``, prune the league and optionally notify.

        Returns the sorted (step-order) list of surviving snapshot paths. Usable
        before ``learn`` (``self.model`` is not required) so the trainer can seed
        the league with an initial snapshot.
        """
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        step = int(model.num_timesteps if step is None else step)
        path = self.snapshot_dir / f"policy_{step}.zip"
        model.save(str(path))

        paths = self._prune()
        self.snapshots = paths
        if notify:
            env = self.training_env if getattr(self, "model", None) is not None else model.get_env()
            if env is not None:
                env.env_method("refresh_pool", paths)
        return paths

    def _prune(self) -> list[str]:
        """Delete all but the newest ``league_size`` snapshots; return survivors."""
        survivors = sorted(self.snapshot_dir.glob("policy_*.zip"), key=self._step_of)
        excess = len(survivors) - self.league_size
        for stale in survivors[: max(0, excess)]:
            stale.unlink(missing_ok=True)
        survivors = sorted(self.snapshot_dir.glob("policy_*.zip"), key=self._step_of)
        return [str(path) for path in survivors]

    @staticmethod
    def _step_of(path: Path) -> int:
        """Numeric step encoded in a ``policy_<step>.zip`` file name."""
        return int(path.stem.rsplit("_", 1)[-1])


class CheckpointCallback(BaseCallback):
    """Save ``latest.zip`` and ``ppo_<num_timesteps>_steps.zip`` periodically.

    Parameters
    ----------
    checkpoint_every:
        Save every this many environment steps (>= 1).
    checkpoint_dir:
        Directory for the checkpoint files (created if missing).
    verbose:
        SB3 verbosity.
    """

    def __init__(
        self,
        checkpoint_every: int,
        checkpoint_dir: str | Path = "models/checkpoints",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
        self.checkpoint_every = int(checkpoint_every)
        self.checkpoint_dir = Path(checkpoint_dir)
        self._last_checkpoint = 0

    def _init_callback(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_checkpoint < self.checkpoint_every:
            return True
        self._last_checkpoint = int(self.num_timesteps)
        self.save(self.model)
        return True

    def save(self, model: Any) -> None:
        """Write the step-numbered checkpoint plus the ``latest.zip`` alias."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        step = int(model.num_timesteps)
        model.save(str(self.checkpoint_dir / f"ppo_{step}_steps.zip"))
        model.save(str(self.checkpoint_dir / "latest.zip"))


__all__ = [
    "CSV_HEADER",
    "WINDOW_SIZE",
    "CheckpointCallback",
    "EpisodeStatsCallback",
    "SnapshotCallback",
]
