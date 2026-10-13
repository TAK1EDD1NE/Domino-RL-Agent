"""Phase-5 CLI: duplicate-deal matchups for the search agents, plus a round-robin Elo table.

Every deal is played twice (normal seating, then the seating rotated by one seat, which
swaps the teams), the two games are pooled for the win rate with a Wilson interval, and the
per-deal margin is reported with a bootstrap interval over deals. ``--round-robin`` also fits
a Bradley-Terry model over all pairs and prints the Elo table.

Usage::

    uv run python tools/run_search_eval.py --matchups pimc_vs_rule,pimc_vs_random --deals 40
    uv run python tools/run_search_eval.py --round-robin --deals 30 --model <checkpoint>
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import pathlib
import time

from domino.pimc.harness import (
    AgentSpec,
    CSV_FIELDS,
    default_workers,
    elo_table,
    estimate_runtime,
    evaluate_matchup,
    append_row,
)

KNOWN_KINDS = ("random", "rule", "pimc", "rl")


def newest_checkpoint() -> str:
    """Highest-step ``ppo_<N>_steps.zip`` (never ``latest.zip``: the smoke test rewrites it)."""
    root = pathlib.Path("models/checkpoints")
    best: tuple[int, pathlib.Path] | None = None
    for path in root.glob("ppo_*_steps.zip"):
        try:
            steps = int(path.stem.split("_")[1])
        except (IndexError, ValueError):                     # pragma: no cover - odd name
            continue
        if steps < 100_000:                                  # skip smoke-test artefacts
            continue
        if best is None or steps > best[0]:
            best = (steps, path)
    if best is None:
        raise SystemExit("no trained checkpoint found; pass --model explicitly")
    return str(best[1])


def build_spec(name: str, args: argparse.Namespace) -> AgentSpec:
    """Map a matchup name fragment to an agent spec."""
    if name not in KNOWN_KINDS:
        raise SystemExit(f"unknown agent kind {name!r}; known: {', '.join(KNOWN_KINDS)}")
    if name == "pimc":
        return AgentSpec(name="pimc", kind="pimc", options={
            "worlds": args.pimc_worlds,
            "horizon": args.pimc_horizon,
            "time_budget": args.pimc_budget,
            "leaf_eval": args.pimc_leaf,
            "method": args.pimc_sampler,
            "opponent_model": args.pimc_opponent_model,
        })
    if name == "rl":
        model = args.model or newest_checkpoint()
        return AgentSpec(name="rl", kind="rl", options={"model_path": model})
    return AgentSpec(name=name, kind=name)


def format_table(rows: list[dict]) -> str:
    """Fixed-width summary of the matchup rows."""
    header = (f"{'matchup':<20}{'deals':>6}{'games':>6}{'W-L':>9}{'win_rate':>9}"
              f"{'wilson 95%':>17}{'paired diff':>12}{'boot 95%':>17}{'blk':>6}{'ms/act':>9}")
    lines = [header, "-" * len(header)]
    for row in rows:
        wins, losses = int(row["wins_a"]), int(row["wins_b"])
        wilson = f"[{row['wilson_low']:.3f}, {row['wilson_high']:.3f}]"
        boot = f"[{row['boot_low']:+.2f}, {row['boot_high']:+.2f}]"
        lines.append(
            f"{row['matchup']:<20}{int(row['deals']):>6}{int(row['games']):>6}"
            f"{f'{wins}-{losses}':>9}{row['win_rate_a']:>9.3f}{wilson:>17}"
            f"{row['paired_differential']:>12.2f}{boot:>17}"
            f"{row['block_rate']:>6.2f}{row['ms_per_action']:>9.1f}"
        )
    return "\n".join(lines)


def parse_matchups(text: str) -> list[tuple[str, str]]:
    """Parse ``a_vs_b,...`` into pairs of kinds."""
    pairs: list[tuple[str, str]] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "_vs_" not in chunk:
            raise SystemExit(f"matchup {chunk!r} must look like agent_vs_agent")
        left, right = chunk.split("_vs_")
        pairs.append((left, right))
    return pairs


def main(argv: list[str] | None = None) -> int:
    """Run the requested matchups (and optionally the round robin) with incremental CSV."""
    parser = argparse.ArgumentParser(description="Duplicate-deal evaluation for the search agents.")
    parser.add_argument("--matchups", default="pimc_vs_rule,pimc_vs_random,random_vs_rule",
                        help="comma-separated agent_vs_agent pairs")
    parser.add_argument("--deals", type=int, default=40, help="duplicate pairs per matchup")
    parser.add_argument("--seed", type=int, default=0, help="base deal seed")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"worker processes (default: cores - 2 = {default_workers()})")
    parser.add_argument("--model", default=None, help="checkpoint for the 'rl' agent kind")
    parser.add_argument("--round-robin", action="store_true",
                        help="play every pair and fit a Bradley-Terry/Elo table")
    parser.add_argument("--pimc-worlds", type=int, default=16, dest="pimc_worlds")
    parser.add_argument("--pimc-horizon", type=int, default=8, dest="pimc_horizon")
    parser.add_argument("--pimc-budget", type=float, default=0.25, dest="pimc_budget",
                        help="wall-clock seconds per PIMC decision (default 0.25 for sweeps)")
    parser.add_argument("--pimc-leaf", default="blocked_now", dest="pimc_leaf")
    parser.add_argument("--pimc-sampler", default="auto", dest="pimc_sampler")
    parser.add_argument("--pimc-opponent-model", default="min", dest="pimc_opponent_model")
    parser.add_argument("--out", default=None, help="CSV path (default results/search_eval_<date>.csv)")
    parser.add_argument("--out-elo", default=None, help="CSV path for the Elo table")
    args = parser.parse_args(argv)

    if args.round_robin:
        pairs = [(a, b) for index, a in enumerate(KNOWN_KINDS)
                 for b in KNOWN_KINDS[index + 1:]]
    else:
        pairs = parse_matchups(args.matchups)
    specs = {kind: build_spec(kind, args) for kind in KNOWN_KINDS
             if any(kind in pair for pair in pairs)}

    print(estimate_runtime(list(specs.values()), deals=args.deals, seed=args.seed,
                           workers=args.workers))
    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        "results") / f"search_eval_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    rows: list[dict] = []
    started = time.perf_counter()
    for left, right in pairs:
        spec_a, spec_b = specs[left], specs[right]
        mark = time.perf_counter()
        row = evaluate_matchup(spec_a, spec_b, deals=args.deals, seed=args.seed,
                               workers=args.workers)
        rows.append(row)
        append_row(out, row)
        print(f"[{row['matchup']}] {time.perf_counter() - mark:.1f}s "
              f"win {row['win_rate_a']:.3f} [{row['wilson_low']:.3f}, {row['wilson_high']:.3f}] "
              f"paired {row['paired_differential']:+.2f} "
              f"[{row['boot_low']:+.2f}, {row['boot_high']:+.2f}]")
    print()
    print(format_table(rows))
    if args.round_robin:
        elo = elo_table(rows)
        elo_path = pathlib.Path(args.out_elo) if args.out_elo else pathlib.Path(
            "results") / f"search_elo_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        elo_path.parent.mkdir(parents=True, exist_ok=True)
        with elo_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["agent", "elo"])
            writer.writeheader()
            for name, value in sorted(elo.items(), key=lambda item: -item[1]):
                writer.writerow({"agent": name, "elo": round(value, 1)})
        print("\nElo (Bradley-Terry, field mean 1500):")
        for name, value in sorted(elo.items(), key=lambda item: -item[1]):
            print(f"  {name:<10}{value:8.1f}")
        print(f"\nwrote {elo_path}")
    print(f"wrote {out} ({len(rows)} row(s) in {time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":                                   # spawn-safe entry point
    raise SystemExit(main())
