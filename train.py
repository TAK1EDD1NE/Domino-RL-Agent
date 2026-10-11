"""Train the domino 2v2 agent with MaskablePPO (spec section 9).

CLI entry point for the whole training pipeline. It loads a
:class:`~domino.config.TrainConfig` (YAML preset plus CLI overrides), builds a
vectorised environment, constructs ``MaskablePPO`` on the project's **custom**
network (:class:`~domino.policy.DominoMaskablePolicy` +
:func:`~domino.policy.policy_kwargs_for`), attaches the convergence/snapshot/
checkpoint callbacks and runs ``learn``. TensorBoard scalars, a CSV mirror of
the convergence series, checkpoints and league snapshots all land under the run
name; ``--resume`` continues from ``models/checkpoints/latest.zip``.

Vectorised league semantics (important, and reproduced in the project report)
----------------------------------------------------------------------------
The learner always occupies team A. Team B is drawn per deal from an
:class:`~domino.agents.OpponentPool` living **inside each worker**. A live
learner policy cannot be shared into a *spawned* worker (the process boundary
would copy the model and never see its updates), so with ``--n-envs > 1`` the
pool's self-play entry is the **newest frozen snapshot**: the worker pool's
default weights are ``{"snapshot": 0.6, "rule": 0.25, "random": 0.15}`` -- no
``"self"`` entry -- and :class:`~domino.callbacks.SnapshotCallback` keeps that
snapshot list current by calling ``vec_env.env_method("refresh_pool", paths)``
after every save. Self-play is therefore *delayed* self-play against a snapshot
that lags the learner by at most ``--snapshot-every`` steps.

With ``--n-envs 1`` the environment runs in-process (``DummyVecEnv``) and the
pool may instead hold a live ``"self"`` entry wired through
:meth:`~domino.agents.OpponentPool.set_learner`; that path uses the spec's full
league weights ``{"self": 0.5, "snapshot": 0.2, "rule": 0.2, "random": 0.1}``.
The frozen-snapshot path is the default for every multi-worker run because it is
the only spawn-safe option on Windows.

Determinism: ``--seed`` seeds numpy, torch, gymnasium and the environment (each
worker gets ``seed + rank``); the run logs its seed in ``config.json``.

Never imports pygame. ``SubprocVecEnv`` uses ``spawn`` on Windows, so the env
factory is a module-level function and the main body is guarded by
``if __name__ == "__main__":``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from domino.agents import OpponentPool
from domino.callbacks import (
    CheckpointCallback,
    EpisodeStatsCallback,
    SnapshotCallback,
    enable_checkpoint_loading,
)
from domino.config import TrainConfig, apply_overrides, load_config
from domino.env import DominoEnv
from domino.policy import DominoMaskablePolicy, param_count, policy_kwargs_for

# ---- project layout --------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = PROJECT_ROOT / "runs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
SNAPSHOT_DIR: Path = MODELS_DIR / "snapshots"
CHECKPOINT_DIR: Path = MODELS_DIR / "checkpoints"

# ---- opponent-league weight presets (spec section 8) -----------------------
#: Live self-play weights, used only for the in-process ``--n-envs 1`` path.
LIVE_LEAGUE_WEIGHTS: dict[str, float] = {"self": 0.5, "snapshot": 0.2, "rule": 0.2, "random": 0.1}
#: Spawn-safe frozen-snapshot league, the default for every multi-worker run.
FROZEN_LEAGUE_WEIGHTS: dict[str, float] = {"snapshot": 0.6, "rule": 0.25, "random": 0.15}

OPPONENT_CHOICES: tuple[str, ...] = ("league", "self", "rule", "random", "mixed")


# ---- module-level env factory (picklable for SubprocVecEnv spawn) ----------
def make_domino_env(
    cfg: TrainConfig,
    seed: int,
    weights: dict[str, float],
    mix_opponents: bool,
) -> DominoEnv:
    """Build one :class:`~domino.env.DominoEnv` with its own league pool.

    This is a **module-level** function so ``SubprocVecEnv`` can pickle it under
    the ``spawn`` start method (Windows). Each worker builds its own
    :class:`~domino.agents.OpponentPool` -- a live learner model is never shared
    across processes (see the module docstring on vectorised league semantics).
    """
    env_cfg = replace(cfg.env, seed=int(seed), opponent_weights=dict(weights))
    pool = OpponentPool(weights=dict(weights), seed=int(seed), mix_opponents=bool(mix_opponents))
    # Snapshot opponents are loaded lazily inside this worker, so the torch/SB3
    # checkpoint-loading workaround must be active in the worker process too.
    enable_checkpoint_loading()
    return DominoEnv(opponent=pool, env_cfg=env_cfg, seed=int(seed))


def opponent_weights(opponent: str, live_self: bool = False) -> dict[str, float]:
    """Map a ``--opponent`` value onto opponent-pool weights (ruling 2).

    ``league``/``mixed`` use the frozen-snapshot league (or the live-self league
    when ``live_self``); ``self`` is snapshot-only self-play, ``rule`` and
    ``random`` are single-source pools.
    """
    if opponent in ("league", "mixed"):
        return dict(LIVE_LEAGUE_WEIGHTS if live_self else FROZEN_LEAGUE_WEIGHTS)
    if opponent == "self":
        return {"snapshot": 1.0}
    if opponent == "rule":
        return {"rule": 1.0}
    if opponent == "random":
        return {"random": 1.0}
    raise ValueError(f"unknown opponent {opponent!r}; choose from {OPPONENT_CHOICES}")


def build_vec_env(cfg: TrainConfig, weights: dict[str, float], mix_opponents: bool) -> VecEnv:
    """Build the vectorised training environment.

    ``--n-envs 1`` uses an in-process :class:`DummyVecEnv` (so the league may
    hold a live learner); any other count uses :class:`SubprocVecEnv` with one
    module-level factory per worker.
    """
    if cfg.n_envs == 1:
        return DummyVecEnv([partial(make_domino_env, cfg, cfg.seed, weights, mix_opponents)])
    factories = [
        partial(make_domino_env, cfg, cfg.seed + rank, weights, mix_opponents)
        for rank in range(cfg.n_envs)
    ]
    return SubprocVecEnv(factories)


def effective_batch_size(cfg: TrainConfig) -> int:
    """Clamp ``batch_size`` to the rollout buffer size (``n_steps * n_envs``).

    SB3 only warns when ``batch_size`` exceeds the buffer (a truncated minibatch
    every epoch); clamping keeps the minibatches exact, which matters for the
    small smoke configs where ``n_envs`` is overridden on the CLI.
    """
    buffer_size = cfg.n_steps * cfg.n_envs
    return min(int(cfg.batch_size), buffer_size)


def build_model(cfg: TrainConfig, vec_env: VecEnv, runs_dir: Path) -> MaskablePPO:
    """Construct ``MaskablePPO`` on the project's custom policy and network."""
    return MaskablePPO(
        policy=DominoMaskablePolicy,
        env=vec_env,
        tensorboard_log=str(runs_dir),
        policy_kwargs=policy_kwargs_for(cfg),
        learning_rate=LinearSchedule(cfg.learning_rate, 0.0, 1.0),
        n_steps=int(cfg.n_steps),
        batch_size=effective_batch_size(cfg),
        n_epochs=int(cfg.n_epochs),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_range=float(cfg.clip_range),
        ent_coef=float(cfg.ent_coef),
        vf_coef=float(cfg.vf_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        target_kl=float(cfg.target_kl),
        seed=int(cfg.seed),
        device=cfg.device,
        verbose=1,
    )


def seed_everything(seed: int) -> None:
    """Seed numpy, torch, ``random`` and the CUDA generator for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def git_sha() -> str:
    """Return the current commit SHA, or ``"unknown"`` if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


# ---- CLI -------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the ``train.py`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train the domino 2v2 MaskablePPO agent.",
    )
    parser.add_argument("--config", default=None, help="YAML preset to load (optional).")
    parser.add_argument("--run-name", dest="run_name", default=None, help="Run name (runs/, results/).")
    parser.add_argument("--steps", type=int, default=None, help="Total environment steps to train.")
    parser.add_argument("--n-envs", dest="n_envs", type=int, default=None, help="Number of vector workers.")
    parser.add_argument("--seed", type=int, default=None, help="Global seed.")
    parser.add_argument(
        "--reward-mode",
        dest="reward_mode",
        choices=["differential", "banked"],
        default=None,
        help="Environment reward mode.",
    )
    parser.add_argument(
        "--opponent",
        choices=list(OPPONENT_CHOICES),
        default=None,
        help="Opponent league preset.",
    )
    parser.add_argument(
        "--snapshot-every", dest="snapshot_every", type=int, default=None, help="Snapshot interval (steps)."
    )
    parser.add_argument(
        "--ckpt-every", dest="ckpt_every", type=int, default=None, help="Checkpoint interval (steps)."
    )
    parser.add_argument("--device", default=None, help="Torch device ('auto', 'cpu', 'cuda').")
    parser.add_argument("--resume", action="store_true", help="Resume from models/checkpoints/latest.zip.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Build and print, do not train.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any TrainConfig field (repeatable), e.g. --set env.reward_mode=banked.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> TrainConfig:
    """Merge the YAML preset, the dedicated CLI flags and ``--set`` overrides."""
    cfg = load_config(args.config) if args.config else TrainConfig()
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.steps is not None:
        cfg.total_steps = int(args.steps)
    if args.n_envs is not None:
        cfg.n_envs = int(args.n_envs)
    if args.seed is not None:
        cfg.seed = int(args.seed)
        cfg.env.seed = int(args.seed)
    if args.reward_mode is not None:
        cfg.env.reward_mode = args.reward_mode
    if args.opponent is not None:
        cfg.env.opponent = args.opponent
    if args.snapshot_every is not None:
        cfg.snapshot_every = int(args.snapshot_every)
    if args.ckpt_every is not None:
        cfg.checkpoint_every = int(args.ckpt_every)
    if args.device is not None:
        cfg.device = args.device
    if args.overrides:
        apply_overrides(cfg, args.overrides)
    return cfg


def write_run_metadata(cfg: TrainConfig, runs_dir: Path) -> None:
    """Write ``config.json`` (resolved config + seed) and ``git_sha.txt``."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(cfg)
    payload["git_sha"] = git_sha()
    (runs_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (runs_dir / "git_sha.txt").write_text(payload["git_sha"] + "\n", encoding="utf-8")


# ---- main ------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Run the trainer end to end and return a process exit code."""
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)

    # torch 2.5 + SB3 checkpoint-loading workaround (needed for --resume here and
    # for snapshot opponents in the workers; see domino.callbacks).
    enable_checkpoint_loading()

    runs_dir = RUNS_DIR / cfg.run_name
    results_dir = RESULTS_DIR
    snapshot_dir = SNAPSHOT_DIR
    checkpoint_dir = CHECKPOINT_DIR
    write_run_metadata(cfg, runs_dir)

    seed_everything(cfg.seed)

    live_self = cfg.n_envs == 1
    weights = opponent_weights(cfg.env.opponent, live_self=live_self)
    mix_opponents = cfg.env.opponent == "mixed"
    cfg.env.opponent_weights = dict(weights)

    print(f"Run '{cfg.run_name}': {cfg.total_steps} steps, {cfg.n_envs} env(s), "
          f"opponent='{cfg.env.opponent}', reward_mode='{cfg.env.reward_mode}', seed={cfg.seed}")
    print(f"League weights: {weights}" + (" (live self, in-process)" if live_self else " (frozen snapshots)"))

    raw_env = build_vec_env(cfg, weights, mix_opponents)
    # VecMonitor gives SB3 its undiscounted episode returns (ep_info_buffer ->
    # rollout/ep_rew_mean); the custom EpisodeStatsCallback computes the same
    # quantity directly from the returned rewards.
    vec_env = VecMonitor(raw_env)
    try:
        model: MaskablePPO
        resume = bool(args.resume)
        latest = checkpoint_dir / "latest.zip"
        if resume and latest.exists():
            print(f"Resuming from {latest}")
            model = MaskablePPO.load(
                str(latest), env=vec_env, device=cfg.device, tensorboard_log=str(runs_dir)
            )
        else:
            if resume:
                print(f"--resume requested but {latest} does not exist; starting fresh")
            model = build_model(cfg, vec_env, runs_dir)

        # Wire the live learner into the in-process pool (n_envs == 1 only).
        if live_self:
            pool = getattr(raw_env.envs[0], "_opponent", None)
            if isinstance(pool, OpponentPool):
                pool.set_learner(model)

        n_params = param_count(model)
        print(f"Parameter count: {n_params}")
        print(f"Policy class: {type(model.policy).__name__} "
              f"(features_extractor={type(model.policy.features_extractor).__name__})")

        if args.dry_run:
            print("Resolved configuration:")
            print(json.dumps(dataclasses.asdict(cfg), indent=2))
            print("Dry run: configuration, environment and policy built; no training performed.")
            return 0

        stats_cb = EpisodeStatsCallback(
            run_name=cfg.run_name, results_dir=results_dir, append=resume
        )
        snapshot_cb = SnapshotCallback(
            snapshot_every=cfg.snapshot_every, snapshot_dir=snapshot_dir, league_size=cfg.env.league_size
        )
        checkpoint_cb = CheckpointCallback(
            checkpoint_every=cfg.checkpoint_every, checkpoint_dir=checkpoint_dir
        )

        # Seed the league with an initial snapshot *before* training so the pool
        # always has at least one frozen opponent to sample from.
        initial = snapshot_cb.save(model, step=int(model.num_timesteps), notify=True)
        print(f"Initial snapshot written; league has {len(initial)} snapshot(s)")

        model.learn(
            total_timesteps=int(cfg.total_steps),
            reset_num_timesteps=not resume,
            callback=CallbackList([stats_cb, snapshot_cb, checkpoint_cb]),
            tb_log_name=cfg.run_name,
        )
        model.save(str(checkpoint_dir / "latest.zip"))
        elapsed_fps = model.num_timesteps  # SB3 already printed the measured fps.
        print(f"Training finished: {elapsed_fps} total timesteps; latest checkpoint saved.")
    finally:
        vec_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
