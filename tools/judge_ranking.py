"""Phase-4 sanity: judge random / rule / network play and rank them by regret.

The brief's acceptance test for the judge is that it orders players the way their strength
does — random worst, the rule bot next, a trained checkpoint best. The judge is run on the
**same deals** for every player, against the same rule-bot opponents, and the report prints
the ordering it actually found (with the per-decision standard error) rather than assuming it.

Usage::

    uv run python tools/judge_ranking.py --deals 12 --worlds 12 --horizon 6
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import pathlib
import time

import numpy as np

from domino.agents import PolicyAgent, RandomAgent, RuleAgent
from domino.agents.pool import load_policy
from domino.pimc.judge import judge_deals


def build_player(kind: str, model_path: str | None, seed: int) -> dict[int, object]:
    """Seat -> agent for the judged team; opponents are always the rule bot."""
    opponents = {seat: RuleAgent(seed=seed + 10 + seat) for seat in (1, 3)}
    if kind == "random":
        team = {seat: RandomAgent(seed=seed + seat) for seat in (0, 2)}
    elif kind == "rule":
        team = {seat: RuleAgent(seed=seed + seat) for seat in (0, 2)}
    elif kind == "rl":
        if not model_path:
            raise SystemExit("the 'rl' player needs --model <checkpoint>")
        model = load_policy(model_path)
        team = {seat: PolicyAgent(model, deterministic=True, name="rl") for seat in (0, 2)}
    else:                                                    # pragma: no cover - guard
        raise SystemExit(f"unknown player kind {kind!r}")
    return {**team, **opponents}


def newest_checkpoint() -> str:
    """Highest-step ``ppo_<N>_steps.zip`` checkpoint (never the smoke-test ``latest.zip``)."""
    best: tuple[int, pathlib.Path] | None = None
    for path in pathlib.Path("models/checkpoints").glob("ppo_*_steps.zip"):
        try:
            steps = int(path.stem.split("_")[1])
        except (IndexError, ValueError):                     # pragma: no cover - odd name
            continue
        if steps >= 100_000 and (best is None or steps > best[0]):
            best = (steps, path)
    if best is None:
        raise SystemExit("no trained checkpoint found; pass --model explicitly")
    return str(best[1])


def main(argv: list[str] | None = None) -> int:
    """Judge the requested players and print the ranking."""
    parser = argparse.ArgumentParser(description="Rank players by judged regret.")
    parser.add_argument("--players", default="random,rule,rl",
                        help="comma-separated subset of random,rule,rl")
    parser.add_argument("--deals", type=int, default=12, help="deals judged per player")
    parser.add_argument("--worlds", type=int, default=12, help="sampled worlds per decision")
    parser.add_argument("--horizon", type=int, default=6, help="ply horizon of the judge")
    parser.add_argument("--seed", type=int, default=0, help="deal seed base (shared by players)")
    parser.add_argument("--model", default=None, help="checkpoint for the 'rl' player")
    parser.add_argument("--out", default=None, help="CSV path (default results/judge_ranking_<date>.csv)")
    parser.add_argument("--decisions-csv", default=None,
                        help="optional CSV path for the per-decision rows of every player")
    args = parser.parse_args(argv)

    players = [name.strip() for name in str(args.players).split(",") if name.strip()]
    model_path = args.model or (newest_checkpoint() if "rl" in players else None)
    if model_path:
        print(f"checkpoint: {model_path}")
    rows: list[dict] = []
    for kind in players:
        started = time.perf_counter()
        agents = build_player(kind, model_path, args.seed)
        report = judge_deals(agents, np.random.default_rng(args.seed + 5000), deals=args.deals,
                             worlds=args.worlds, horizon=args.horizon, label=kind,
                             eval_seats=(0, 2))
        print(report.summary())
        print(f"  ({time.perf_counter() - started:.0f}s)\n")
        rows.append({
            "player": kind,
            "deals": report.deals,
            "decisions": len(report.decisions),
            "graded": len(report.graded),
            "mean_regret": report.mean_regret(),
            "regret_se": report.regret_se(),
            "mean_regret_true": report.mean_regret_true(),
            "blunder_rate": report.blunder_rate(),
            "major_blunder_rate": report.major_blunder_rate(),
            "dominated": sum(1 for d in report.graded if d.dominance),
            "mean_luck": report.mean_luck(),
        })
        if args.decisions_csv:
            target = pathlib.Path(args.decisions_csv)
            if rows and target.exists() and len(rows) > 1:
                with target.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(
                        report.decisions[0].as_row().keys()))
                    writer.writerows(decision.as_row() for decision in report.decisions)
            else:
                report.write_csv(target)

    print(f"{'player':<8}{'decisions':>10}{'mean regret':>13}{'se':>7}"
          f"{'hindsight':>11}{'blunders':>10}{'dominated':>11}")
    for row in rows:
        print(f"{row['player']:<8}{row['graded']:>10}{row['mean_regret']:>13.2f}"
              f"{row['regret_se']:>7.2f}{row['mean_regret_true']:>11.2f}"
              f"{row['blunder_rate']:>10.1%}{row['dominated']:>11}")
    ordered = sorted(rows, key=lambda row: -row["mean_regret"])
    print("\nworst-to-best by mean regret:", " > ".join(row["player"] for row in ordered))
    expected = [name for name in ("random", "rule", "rl", "pimc") if name in players]
    found = [row["player"] for row in ordered if row["player"] in expected]
    print("matches the expected strength order (random worst, checkpoint best):",
          found == expected, f"(expected {expected})")
    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        "results") / f"judge_ranking_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
