"""Measure real training throughput, so the training budget is a measurement.

The project's first estimate assumed thousands of environment steps per second.
This tool replaces that guess with numbers: it measures (a) the raw engine's
deals/second with no learning machinery, (b) the end-to-end learner steps/second
with the real vectorised environment, PPO updates and the league active, and
(c) the same run with a cheaper opponent mix, so the league's cost is visible.

Usage
-----
    uv run python tools/bench_throughput.py --steps 30000 --envs 8
    uv run python tools/bench_throughput.py --variants rule,league --out runs/benchmark.json

The results land in ``runs/benchmark.json`` and are summarised in the design
spec's training-time section.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import platform
import statistics
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Steps per second table rows reported at the end.
EXTRAPOLATION_STEPS: tuple[int, ...] = (10_000_000, 30_000_000, 100_000_000, 300_000_000)


def _load_train_module():
    """Load the root-level ``train.py`` by path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("domino_train_script", REPO_ROOT / "train.py")
    assert spec and spec.loader, "cannot load train.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# (a) raw engine throughput
# --------------------------------------------------------------------------- #
def engine_throughput(deals: int, seed: int = 0) -> dict[str, float]:
    """Deals/second and moves/second playing directly on the engine (no gym, no PPO)."""
    from domino.agents import RandomAgent, RuleAgent
    from domino.engine import deal_outcome, new_deal, redeal_after_void
    from domino.masks import decode_action

    agents = {
        0: RuleAgent(seed=seed),
        1: RandomAgent(seed=seed + 1),
        2: RuleAgent(seed=seed + 2),
        3: RandomAgent(seed=seed + 3),
    }
    rng = np.random.default_rng(seed)
    moves = 0
    scored = 0
    started = time.perf_counter()
    while scored < deals:
        state = new_deal(deal_no=1, rng=rng, lead_mode="six_six")
        while True:
            while not state.is_terminal():
                seat = state.turn
                if state.must_pass(seat):
                    state.pass_turn(seat)
                    moves += 1
                    continue
                action = int(agents[seat].act(state, seat))
                moves += 1
                tid, end = decode_action(action)
                state.play(seat, tid, end)
            outcome = deal_outcome(state)
            if not outcome.void:
                break
            state = redeal_after_void(state)
        scored += 1
    wall = time.perf_counter() - started
    return {
        "deals": scored,
        "moves": moves,
        "seconds": wall,
        "deals_per_second": scored / wall,
        "moves_per_second": moves / wall,
        "moves_per_deal": moves / scored,
    }


# --------------------------------------------------------------------------- #
# (b) end-to-end learner throughput
# --------------------------------------------------------------------------- #
class _Deadline:
    """SB3 callback that stops training once a wall-clock budget is spent."""

    def __init__(self, max_seconds: float | None) -> None:
        self.max_seconds = max_seconds
        self.started = time.perf_counter()
        self.stopped_early = False

    def __call__(self) -> bool:
        if self.max_seconds is None:
            return True
        if time.perf_counter() - self.started >= self.max_seconds:
            self.stopped_early = True
            return False
        return True


def learner_throughput(
    *,
    envs: int,
    steps: int,
    opponent: str,
    seed: int = 0,
    max_seconds: float | None = 300.0,
    train_module=None,
) -> dict[str, object]:
    """Run a short real training job and report its measured steps/second."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    train = train_module or _load_train_module()
    cfg = train.TrainConfig(
        run_name=f"bench_{opponent}",
        total_steps=steps,
        n_envs=envs,
        seed=seed,
    )
    cfg.env.opponent = opponent
    train.seed_everything(seed)
    weights = train.opponent_weights(opponent, live_self=False)
    vec_env = train.build_vec_env(cfg, weights, mix_opponents=(opponent == "mixed"))
    model = train.build_model(cfg, vec_env, REPO_ROOT / "runs")

    deadline = _Deadline(max_seconds)

    class _Stop(BaseCallback):
        def _on_step(self) -> bool:  # noqa: D102 - SB3 hook
            return deadline()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    model.learn(total_timesteps=steps, callback=_Stop(), progress_bar=False)
    wall = time.perf_counter() - started
    steps_done = int(model.num_timesteps)
    vec_env.close()

    result: dict[str, object] = {
        "opponent": opponent,
        "weights": weights,
        "n_envs": envs,
        "requested_steps": steps,
        "steps": steps_done,
        "seconds": wall,
        "steps_per_second": steps_done / wall,
        "hit_deadline": deadline.stopped_early,
        "n_steps": cfg.n_steps,
        "batch_size": train.effective_batch_size(cfg),
        "param_count": None,
        "cuda_peak_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if torch.cuda.is_available() else None
        ),
    }
    try:
        from domino.policy import param_count

        result["param_count"] = param_count(model)
    except Exception:  # noqa: BLE001 - reported as unavailable
        pass
    return result


def summarise(engine: dict[str, float], learners: list[dict[str, object]]) -> str:
    """Render the measured numbers and the extrapolated training times."""
    lines = [
        "Measured throughput",
        "===================",
        "",
        f"raw engine (rule/random seats, no gym, no PPO): "
        f"{engine['deals_per_second']:.0f} deals/s, {engine['moves_per_second']:.0f} moves/s "
        f"({engine['moves_per_deal']:.1f} moves/deal)",
        "",
        f"{'opponent':<10}{'envs':>6}{'steps':>9}{'seconds':>10}{'steps/s':>10}",
        "-" * 45,
    ]
    for row in learners:
        lines.append(
            f"{str(row['opponent']):<10}{int(row['n_envs']):>6}{int(row['steps']):>9}"
            f"{float(row['seconds']):>10.1f}{float(row['steps_per_second']):>10.0f}"
        )
    rates = [float(row["steps_per_second"]) for row in learners]
    if rates:
        low, high = min(rates), max(rates)
        lines += [
            "",
            f"extrapolated wall clock at {low:.0f}-{high:.0f} steps/s:",
            f"{'steps':>12}{'low':>12}{'high':>12}",
            "-" * 36,
        ]
        for total in EXTRAPOLATION_STEPS:
            lines.append(
                f"{total:>12,}{total / high / 3600:>11.2f}h{total / low / 3600:>11.2f}h"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the measurements and write ``runs/benchmark.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", type=int, default=8, help="vectorised environments")
    parser.add_argument("--steps", type=int, default=30_000, help="learner steps per variant")
    parser.add_argument("--variants", default="rule,league",
                        help="comma-separated opponent mixes to measure")
    parser.add_argument("--engine-deals", type=int, default=2_000,
                        help="deals used for the raw engine measurement")
    parser.add_argument("--max-seconds", type=float, default=300.0,
                        help="wall-clock cap per learner variant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/benchmark.json")
    args = parser.parse_args(argv)

    train = _load_train_module()
    engine = engine_throughput(args.engine_deals, seed=args.seed)
    learners: list[dict[str, object]] = []
    for opponent in [name.strip() for name in args.variants.split(",") if name.strip()]:
        print(f"[bench] measuring opponent={opponent} ...", flush=True)
        learners.append(
            learner_throughput(
                envs=args.envs, steps=args.steps, opponent=opponent,
                seed=args.seed, max_seconds=args.max_seconds, train_module=train,
            )
        )

    payload = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "engine": engine,
        "learners": learners,
        "median_steps_per_second": statistics.median(
            [float(row["steps_per_second"]) for row in learners]
        ) if learners else None,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(summarise(engine, learners))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - measurement entry point
    raise SystemExit(main())
