#!/usr/bin/env python
"""Aggregate evaluation CSVs across seeds: pooled Wilson CI + the between-seed spread.

A single 400-deal evaluation has a +/-5 pp interval and its point estimate swings by a few
points between seeds, so one seed is not a strength claim. This pools every row it is given
and also reports the per-seed rates and their standard deviation, which is the number that
tells you how much of a difference is real.

Usage:
  python aggregate_seeds.py 'results/eval_stage1_*.csv'
  python aggregate_seeds.py 'results/eval_*.csv' --matchup rl_vs_rule

Expected CSV schema (the one this project's test.py writes):
  matchup,deals,wins_a,win_rate_a,ci_low,ci_high,points_per_deal_a,points_per_deal_b,...
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
from pathlib import Path


FIELDS = ("matchup", "deals", "wins_a", "win_rate_a", "points_per_deal_a", "points_per_deal_b")


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load(pattern: str, matchup: str | None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(pattern)):
        for row in csv.DictReader(Path(path).open()):
            if matchup and row["matchup"] != matchup:
                continue
            record = {"file": Path(path).name}
            for field in FIELDS:
                record[field] = row[field]
            grouped.setdefault(row["matchup"], []).append(record)
    return grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", help="glob for the evaluation CSVs (quote it)")
    parser.add_argument("--matchup", default=None, help="restrict to one matchup name")
    args = parser.parse_args(argv)

    grouped = load(args.pattern, args.matchup)
    if not grouped:
        print(f"no rows matched {args.pattern}")
        return 1
    for matchup, rows in sorted(grouped.items()):
        rates = [float(r["win_rate_a"]) for r in rows]
        wins = sum(int(r["wins_a"]) for r in rows)
        deals = sum(int(r["deals"]) for r in rows)
        low, high = wilson(wins, deals)
        pts_a = statistics.fmean(float(r["points_per_deal_a"]) for r in rows)
        pts_b = statistics.fmean(float(r["points_per_deal_b"]) for r in rows)
        print(f"\n{matchup}: {len(rows)} evaluation(s), {deals} deals total")
        print("  per-run win rates: " + ", ".join(f"{r:.3f}" for r in rates))
        spread = statistics.stdev(rates) if len(rates) > 1 else float("nan")
        print(f"  mean {statistics.fmean(rates):.4f}   between-seed SD {spread:.4f}"
              f"   min {min(rates):.3f}   max {max(rates):.3f}")
        print(f"  pooled {wins}/{deals} = {wins / deals:.4f}   Wilson 95% [{low:.4f}, {high:.4f}]")
        verdict = ("above the baseline" if low > 0.5 else
                   "below the baseline" if high < 0.5 else "parity: CI straddles 0.5")
        print(f"  points/deal A {pts_a:.2f} vs B {pts_b:.2f} (differential {pts_a - pts_b:+.2f}) -- {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
