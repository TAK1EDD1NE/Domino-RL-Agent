"""Belief-head calibration: does the head beat the constraints, and are its numbers honest?

Plays deals, records every decision of the judged seats, and scores the belief head against
the sampler's exact uniform-over-feasible marginals on the *same* positions:

    uv run python tools/belief_calibration.py --model models/checkpoints/ppo_16002774_steps.zip
    uv run python tools/belief_calibration.py --driver rule --deals 60 --worlds 1024

Output: one CSV row per (predictor, subset) plus a printed table whose headline is the
**informative** subset - the tiles whose owner is actually in doubt. A head that does not
beat the baseline there has learned nothing beyond the constraints the sampler already knows.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import pathlib
import time
from typing import Any

import numpy as np

from domino.agents import PolicyAgent, RuleAgent
from domino.agents.pool import load_policy
from domino.pimc.belief_eval import (
    BeliefSample,
    Score,
    baseline_marginals,
    head_marginals,
    informative_mask,
    score,
)

SUBSETS = ("all", "informative", "left", "partner", "right", "opening", "mid", "end")


def build_agents(model: Any, *, driver: str, seed: int) -> dict[int, Any]:
    """Learner seats (0, 2) driven by ``driver``; the other seats by the rule bot."""
    agents: dict[int, Any] = {seat: RuleAgent(seed=seed + 10 + seat) for seat in (1, 3)}
    for seat in (0, 2):
        if driver == "rule":
            agents[seat] = RuleAgent(seed=seed + seat)
        else:
            agents[seat] = PolicyAgent(model, deterministic=False, name="rl")
    return agents


def _subset_mask(sample: BeliefSample, subset: str) -> np.ndarray:
    """The (3, 28) cell mask for one named subset."""
    mask = np.ones((3, 28), dtype=bool)
    if subset == "informative":
        mask &= informative_mask(sample.view)
    elif subset in ("left", "partner", "right"):
        slot = {"left": 0, "partner": 1, "right": 2}[subset]
        keep = np.zeros((3, 28), dtype=bool)
        keep[slot] = True
        mask &= keep
    elif subset in ("opening", "mid", "end"):
        if sample.phase != subset:
            mask[:] = False
    return mask


def main(argv: list[str] | None = None) -> int:
    """Run the calibration and write the CSV."""
    parser = argparse.ArgumentParser(description="Score the belief head against the sampler baseline.")
    parser.add_argument("--model", default="models/checkpoints/ppo_16002774_steps.zip")
    parser.add_argument("--deals", type=int, default=40, help="deals played to collect positions")
    parser.add_argument("--worlds", type=int, default=512,
                        help="uniform worlds per position for the baseline marginals")
    parser.add_argument("--driver", choices=("policy", "rule"), default="policy",
                        help="who drives the judged seats (positions come from this player)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="CSV path (default results/belief_calibration_<date>.csv)")
    args = parser.parse_args(argv)

    model = load_policy(args.model)
    rng = np.random.default_rng(args.seed)
    agents = build_agents(model, driver=args.driver, seed=args.seed)

    started = time.perf_counter()
    from domino.pimc.belief_eval import collect_belief_samples

    samples = collect_belief_samples(agents, deals=args.deals, seed=args.seed, rng=rng)
    if not samples:
        raise SystemExit("no samples collected; nothing to score")
    print(f"{len(samples)} decision(s) from {args.deals} deal(s) in {time.perf_counter() - started:.1f}s")

    obs = np.stack([sample.obs for sample in samples])                     # (N, OBS_DIM)
    head = head_marginals(model.policy, obs)                               # (N, 3, 28)

    mark = time.perf_counter()
    baseline = np.empty((len(samples), 3, 28), dtype=np.float64)
    baseline_error = np.empty((len(samples), 3, 28), dtype=np.float64)
    for index, sample in enumerate(samples):
        values, error = baseline_marginals(sample.view, worlds=args.worlds, rng=rng)
        baseline[index] = values
        baseline_error[index] = error
    print(f"baseline marginals: {time.perf_counter() - mark:.1f}s "
          f"({args.worlds} worlds/position), mean MC error {np.nanmean(baseline_error):.4f}")

    truth = np.stack([sample.labels for sample in samples])                # (N, 3, 28)
    rows: list[dict[str, Any]] = []
    for subset in SUBSETS:
        mask = np.stack([_subset_mask(sample, subset) for sample in samples])
        if not mask.any():
            continue
        head_score = score(head[mask], truth[mask])
        base_score = score(baseline[mask], truth[mask])
        for name, value in ((f"head[{args.driver}]", head_score), (f"baseline[uniform]", base_score)):
            row = value.as_row(name)
            row["subset"] = subset
            row["deals"] = int(args.deals)
            row["worlds"] = int(args.worlds)
            rows.append(row)

    header = (f"{'subset':<13}{'predictor':<20}{'cells':>8}{'log-loss':>10}{'brier':>9}"
              f"{'ece':>8}{'acc':>8}")
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['subset']:<13}{row['predictor']:<20}{row['entries']:>8}"
              f"{row['log_loss']:>10.4f}{row['brier']:>9.4f}{row['ece']:>8.4f}{row['accuracy']:>8.4f}")

    # Paired comparison on the informative subset: does the head win per decision?
    mask = np.stack([_subset_mask(sample, "informative") for sample in samples])
    head_cell = np.array([score(head[i][mask[i]], truth[i][mask[i]]).log_loss
                          for i in range(len(samples))])
    base_cell = np.array([score(baseline[i][mask[i]], truth[i][mask[i]]).log_loss
                          for i in range(len(samples))])
    delta = head_cell - base_cell
    wins = float(np.mean(delta < 0))
    se = float(np.std(delta, ddof=1) / np.sqrt(delta.size)) if delta.size > 1 else float("nan")
    print(f"\nper-decision (informative): head - baseline log-loss = {delta.mean():+.4f} "
          f"+/- {se:.4f} (SE), head better in {wins:.1%} of {delta.size} decisions")

    # Sharpness: a head that has learned "counts only" predicts nearly the same probability
    # for every tile of a seat. Compare each predictor's spread across the in-doubt tiles of
    # a decision, and how confident its single most likely tile is.
    def _spread(prediction: np.ndarray, cells: np.ndarray) -> tuple[float, float]:
        spreads, peaks = [], []
        for index in range(prediction.shape[0]):
            values = prediction[index][cells[index]]
            if values.size <= 1:
                continue
            spreads.append(float(np.std(values)))
            peaks.append(float(np.max(values)))
        return float(np.mean(spreads)), float(np.mean(peaks))

    head_spread, head_peak = _spread(head, mask)
    base_spread, base_peak = _spread(baseline, mask)
    print(f"sharpness (informative): head spread {head_spread:.4f} peak {head_peak:.3f} | "
          f"baseline spread {base_spread:.4f} peak {base_peak:.3f}")
    print("  (a head no sharper than the baseline is reproducing the constraints, not "
          "reading the history)")

    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        "results") / f"belief_calibration_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ("subset", "predictor", "entries", "log_loss", "brier", "ece", "accuracy",
              "deals", "worlds")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
