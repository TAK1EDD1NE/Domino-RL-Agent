"""Benchmark the Phase-2 sampler: acceptance rate by deal phase, throughput, bias.

Reports, per stage of a deal (tiles per hand), the measured rejection acceptance rate,
both dealers' throughput in worlds/second, and — where the valid-world support is small
enough to enumerate — the total-variation distance of each dealer from the exact
distribution. The rejection dealer is the unbiased control.

Usage::

    uv run python tools/bench_sampler.py --positions 8
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import time

import numpy as np

from domino.engine import new_deal
from domino.pimc.sampler import (
    ACCEPTANCE_THRESHOLD,
    PublicView,
    acceptance_rate,
    bias_report,
    enumerate_worlds,
    sample_worlds,
)

#: Cap on the enumerated support for the bias measurement (enumeration is factorial).
SUPPORT_CAP = 4000


def random_position(rng: np.random.Generator, max_tiles: int):
    state = new_deal(deal_no=1, rng=rng, lead_mode="six_six")
    while max(len(h) for h in state.hands) > max_tiles and not state.is_terminal():
        seat = state.turn
        moves = state.legal_moves(seat)
        if not moves:
            state.pass_turn(seat)
            continue
        tid, end = moves[int(rng.integers(0, len(moves)))]
        state.play(seat, tid, end)
    return state


def main(argv: list[str] | None = None) -> int:
    """Run the sampler benchmark and print (and optionally write) the table."""
    parser = argparse.ArgumentParser(description="Benchmark the Phase-2 belief sampler.")
    parser.add_argument("--sizes", default="7,5,3,2", help="tiles per hand to sample at")
    parser.add_argument("--positions", type=int, default=8, help="positions per size")
    parser.add_argument("--draws", type=int, default=400, help="worlds per timing run")
    parser.add_argument("--seed", type=int, default=20261013, help="position seed")
    parser.add_argument("--out", default=None, help="CSV path for the rows")
    args = parser.parse_args(argv)

    sizes = [int(chunk) for chunk in str(args.sizes).split(",") if chunk.strip()]
    rows: list[dict] = []
    header = (f"{'tiles':>5}{'accept':>9}{'rej_w/s':>10}{'con_w/s':>10}{'bias_pos':>10}"
              f"{'tv_rej':>8}{'tv_con':>8}{'use_rej':>8}")
    print(header)
    print("-" * len(header))
    for size in sizes:
        rng = np.random.default_rng(args.seed + size)
        rates, rej_rates, con_rates, tvs = [], [], [], []
        for _ in range(args.positions):
            state = random_position(rng, size)
            if state.is_terminal():
                continue
            view = PublicView.from_state(state, state.turn)
            rates.append(acceptance_rate(view, np.random.default_rng(0), trials=128))
            for method, bucket in (("rejection", rej_rates), ("constrained", con_rates)):
                started = time.perf_counter()
                sample_worlds(view, args.draws, rng, method=method)
                bucket.append(args.draws / max(time.perf_counter() - started, 1e-9))
            support = len(enumerate_worlds(view)) if len(view.unseen) <= 14 else None
            if support is not None and support and support <= SUPPORT_CAP:
                report = bias_report(view, np.random.default_rng(1), draws=1000)
                tvs.append((report["chi2_over_dof_rejection"],
                            report["chi2_over_dof_constrained"],
                            report["rejection_uniform_compatible"]))
        if not rates:
            continue
        row = {
            "size": size,
            "positions": len(rates),
            "accept_rate": float(np.mean(rates)),
            "rej_per_s": float(statistics.median(rej_rates)),
            "con_per_s": float(statistics.median(con_rates)),
            "chi2_rej": float(np.mean([t[0] for t in tvs])) if tvs else float("nan"),
            "chi2_con": float(np.mean([t[1] for t in tvs])) if tvs else float("nan"),
            "bias_positions": len(tvs),
            "rej_uniform_ok": int(all(t[2] for t in tvs)) if tvs else -1,
        }
        rows.append(row)
        use_rej = 1 if row["accept_rate"] >= ACCEPTANCE_THRESHOLD else 0
        print(f"{size:>5}{row['accept_rate']:>9.3f}{row['rej_per_s']:>10.0f}"
              f"{row['con_per_s']:>10.0f}{row['bias_positions']:>10}"
              f"{row['chi2_rej']:>8.2f}{row['chi2_con']:>8.2f}{use_rej:>8}")
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}")
    print("\n(accept = measured rejection acceptance rate; tv_* = total-variation distance "
          "from the exact distribution over enumerated worlds; use_rej = 1 means "
          "method='auto' picks the uniform rejection dealer at this stage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
