"""Elo ladder over a training run's checkpoints, scored with ``test.py`` itself.

A single checkpoint's win rate is noisier than the seed spread it is quoted
against, so this tool evaluates *every* checkpoint of a run against the **same**
fixed ladder of anchors and turns the resulting rows into one comparable number:

* ``random`` -- the uniform-random bot;
* ``rule``   -- the documented rule-based baseline;
* an explicitly named **early checkpoint** (``--anchor-early``), i.e. the frozen
  policy the run started from, which is the opponent that actually measures
  progress between late checkpoints.

All three anchors sit at a fixed Elo of :data:`ANCHOR_ELO` (1200), and each
checkpoint starts at the same 1200 and is updated once per row (K =
:data:`ELO_K`, 16) using the standard Elo expectation. A row's "score" is the
deal win rate ``win_rate_a`` the checkpoint achieved across ``--deals`` deals --
the natural generalisation of Elo to a partial score.

The evaluation is ``test.py``'s protocol, imported rather than re-implemented:
this module calls ``test.evaluate_matchup`` with ``test.build_parser`` arguments,
so a ladder row is byte-for-byte the schema ``test.py`` writes (plus the two
ladder columns). A bespoke evaluator around ``load_policy`` would silently
measure a different protocol.

Usage
-----
::

    uv run python tools/eval_ladder.py --run-name v3 --seeds 0,1,2
    uv run python tools/eval_ladder.py --checkpoints models/checkpoints/ppo_1000000_steps.zip --dry-run

Output goes to ``results/ladder_<run_name>.csv`` (override with ``--out``): one
row per ``(checkpoint, seed, matchup)`` with the ``test.py`` columns plus
``checkpoint`` and ``elo`` (the checkpoint's final rating, repeated on each of
its rows). A full run takes minutes per checkpoint; ``--dry-run`` prints the
planned evaluation list and runs no games.
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re
import sys
from typing import Any, Sequence

import numpy as np

# ``test.py`` is a mandatory root-level script, not a package module: force the
# repository root to the *front* of ``sys.path`` so ``import test`` resolves to
# the project's script and not the standard library's ``test`` package (which
# otherwise wins when this file is run as ``python tools/eval_ladder.py``), and
# so ``test.py``'s own ``domino.*`` imports resolve as they do from the repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from test import CSV_FIELDS, build_parser, evaluate_matchup, wilson_interval  # noqa: E402

#: Elo K-factor for every ladder row.
ELO_K: int = 16
#: Starting rating for a checkpoint (and the rating of every anchor).
ELO_BASE: float = 1200.0
#: Every ladder anchor is pinned here and never moves.
ANCHOR_ELO: float = 1200.0

#: Default early-anchor checkpoint (the explicitly named frozen baseline).
DEFAULT_ANCHOR_EARLY: str = "models/checkpoints/ppo_500000_steps.zip"
#: Default checkpoint glob, resolved relative to the working directory.
DEFAULT_CHECKPOINT_GLOB: str = "models/checkpoints/ppo_*_steps.zip"

#: Per-checkpoint legs: ``(matchup name, opponent label)``. The ``rl_vs_rl``
#: leg pairs the checkpoint (agent A) with ``--anchor-early`` (agent B), so every
#: row's ``win_rate_a`` is the checkpoint's *own* score.
LADDER_MATCHUPS: tuple[tuple[str, str], ...] = (
    ("rl_vs_rule", "rule"),
    ("rl_vs_random", "random"),
    ("rl_vs_rl", "anchor_early"),
)

#: The ``test.py`` CSV schema plus the two ladder columns.
LADDER_FIELDS: tuple[str, ...] = tuple(CSV_FIELDS) + ("checkpoint", "elo")

_CHECKPOINT_RE = re.compile(r"ppo_(\d+)_steps\.zip$")


# --------------------------------------------------------------------------- #
# Elo arithmetic (pure, so it is testable without any model)
# --------------------------------------------------------------------------- #
def expected_score(rating: float, opponent: float) -> float:
    """Standard Elo expectation of ``rating`` against ``opponent``."""
    return 1.0 / (1.0 + 10.0 ** ((float(opponent) - float(rating)) / 400.0))


def elo_update(rating: float, opponent: float, score: float, k: int = ELO_K) -> float:
    """One Elo update: ``rating + K * (score - expected_score)``.

    ``score`` is the achieved fraction in ``[0, 1]`` -- the deal win rate a row
    recorded, which extends Elo from win/loss to a partial score.
    """
    return float(rating) + float(k) * (float(score) - expected_score(rating, opponent))


def opponent_rating(row: dict[str, Any]) -> float:
    """Rating of the row's opponent; every ladder anchor is fixed at 1200."""
    value = row.get("opponent_elo")
    return ANCHOR_ELO if value is None else float(value)


def compute_elo(
    rows: Sequence[dict[str, Any]], *, initial: float = ELO_BASE, k: int = ELO_K
) -> dict[str, float]:
    """Final Elo per checkpoint after processing all of its rows in order.

    Checkpoints never play each other (the only cross-checkpoint opponent is the
    fixed anchor), so the per-checkpoint folds are independent and the result
    does not depend on the order of the rows *between* checkpoints.
    """
    ratings: dict[str, float] = {}
    for row in rows:
        ratings.setdefault(str(row["checkpoint"]), float(initial))
    for row in rows:
        checkpoint = str(row["checkpoint"])
        score = float(row["win_rate_a"])
        if math.isnan(score):  # a row with no scored deals carries no information
            continue
        ratings[checkpoint] = elo_update(ratings[checkpoint], opponent_rating(row), score, k)
    return ratings


def finalize_rows(
    rows: Sequence[dict[str, Any]], *, initial: float = ELO_BASE, k: int = ELO_K
) -> list[dict[str, Any]]:
    """Stamp each row with its checkpoint's final Elo (the same value per checkpoint)."""
    ratings = compute_elo(rows, initial=initial, k=k)
    out: list[dict[str, Any]] = []
    for row in rows:
        stamped = dict(row)
        stamped["elo"] = ratings[str(row["checkpoint"])]
        out.append(stamped)
    return out


# --------------------------------------------------------------------------- #
# Win-rate pooling for the summary table
# --------------------------------------------------------------------------- #
def pooled_win_rate(
    rows: Sequence[dict[str, Any]], matchup: str
) -> tuple[float, float, int, int]:
    """Pool one matchup across seeds: ``(rate, between-seed SD, wins, deals)``.

    The rate pools the deals (``sum(wins_a) / sum(deals)``); the SD is over the
    per-seed win rates, so a headline number never hides its seed spread.
    """
    subset = [row for row in rows if row["matchup"] == matchup]
    wins = sum(int(row["wins_a"]) for row in subset)
    deals = sum(int(row["deals"]) for row in subset)
    rate = wins / deals if deals else float("nan")
    per_seed = [float(row["win_rate_a"]) for row in subset]
    sd = float(np.std(per_seed, ddof=1)) if len(per_seed) >= 2 else 0.0
    return rate, sd, wins, deals


# --------------------------------------------------------------------------- #
# Checkpoint discovery and the evaluation plan
# --------------------------------------------------------------------------- #
def checkpoint_step(path: str | pathlib.Path) -> int:
    """Step number encoded in a ``ppo_<step>_steps.zip`` name (``-1`` if none)."""
    match = _CHECKPOINT_RE.search(pathlib.Path(str(path)).name)
    return int(match.group(1)) if match else -1


def resolve_checkpoints(
    explicit: Sequence[str] | None, checkpoint_glob: str = DEFAULT_CHECKPOINT_GLOB
) -> list[str]:
    """The checkpoints to evaluate: ``--checkpoints`` verbatim, else the glob.

    The globbed list is sorted by step so the ladder climbs in training order.
    """
    if explicit:
        return [str(path) for path in explicit]
    matches = sorted(pathlib.Path(".").glob(checkpoint_glob), key=checkpoint_step)
    return [str(path) for path in matches]


def parse_seeds(text: str | Sequence[int]) -> list[int]:
    """Parse ``"0,1,2"`` (or a sequence) into a de-duplicated int list."""
    if isinstance(text, (list, tuple)):
        seeds = [int(value) for value in text]
    else:
        seeds = [int(part) for part in str(text).split(",") if part.strip() != ""]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def build_plan(
    checkpoints: Sequence[str],
    seeds: Sequence[int],
    matchups: Sequence[tuple[str, str]] = LADDER_MATCHUPS,
) -> list[dict[str, Any]]:
    """The full ``(checkpoint, seed, matchup)`` evaluation list, in run order."""
    return [
        {"checkpoint": str(checkpoint), "seed": int(seed), "matchup": name, "opponent": label}
        for checkpoint in checkpoints
        for seed in seeds
        for name, label in matchups
    ]


# --------------------------------------------------------------------------- #
# Running the ladder
# --------------------------------------------------------------------------- #
def evaluate_entry(
    entry: dict[str, Any],
    *,
    parser: argparse.ArgumentParser,
    anchor_early: str,
    deals: int,
    matches: int,
    match_to: int,
) -> dict[str, Any]:
    """Evaluate one plan entry through ``test.evaluate_matchup`` and tag the row.

    The early-anchor leg is ``rl_vs_rl`` with the checkpoint as agent A and the
    anchor as agent B (``--model-b``), so ``win_rate_a`` is always the
    checkpoint's score; ``opponent_elo`` is carried in memory for the Elo pass
    and is not part of the written schema.
    """
    argv = [
        "--model", str(entry["checkpoint"]),
        "--matchups", str(entry["matchup"]),
        "--deals", str(int(deals)),
        "--matches", str(int(matches)),
        "--match-to", str(int(match_to)),
        "--seed", str(int(entry["seed"])),
        "--quiet",
    ]
    if entry["opponent"] == "anchor_early":
        argv += ["--model-b", str(anchor_early)]
    args = parser.parse_args(argv)
    row = evaluate_matchup(str(entry["matchup"]), args)
    row["checkpoint"] = str(entry["checkpoint"])
    row["opponent_elo"] = ANCHOR_ELO
    return row


def write_ladder_csv(rows: Sequence[dict[str, Any]], path: pathlib.Path) -> pathlib.Path:
    """Write the ladder rows with the fixed schema; returns the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LADDER_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in LADDER_FIELDS})
    return path


def _checkpoint_order(rows: Sequence[dict[str, Any]]) -> list[str]:
    """First-seen checkpoint order across the rows."""
    seen: list[str] = []
    for row in rows:
        checkpoint = str(row["checkpoint"])
        if checkpoint not in seen:
            seen.append(checkpoint)
    return seen


def format_ladder_table(rows: Sequence[dict[str, Any]]) -> str:
    """Render the per-checkpoint pooled win rates (vs rule / vs random) and Elo."""
    header = (
        f"{'checkpoint':<46}{'vs rule':>9}{'sd':>7}{'95% CI':>16}"
        f"{'vs rnd':>8}{'sd':>7}{'elo':>8}"
    )
    lines = [header, "-" * len(header)]
    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_checkpoint.setdefault(str(row["checkpoint"]), []).append(row)

    for checkpoint in _checkpoint_order(rows):
        group = by_checkpoint[checkpoint]
        rule_rate, rule_sd, rule_wins, rule_deals = pooled_win_rate(group, "rl_vs_rule")
        rnd_rate, rnd_sd, _, _ = pooled_win_rate(group, "rl_vs_random")
        ci_low, ci_high = wilson_interval(rule_wins, rule_deals)
        elo = float(group[0].get("elo", float("nan")))
        name = pathlib.Path(checkpoint).stem
        ci = "[" + format(ci_low, ".3f") + ", " + format(ci_high, ".3f") + "]"
        lines.append(
            f"{name:<46}{rule_rate:>9.3f}{rule_sd:>7.3f}{ci:>16}"
            f"{rnd_rate:>8.3f}{rnd_sd:>7.3f}{elo:>8.1f}"
        )
    return "\n".join(lines)


def build_ladder_parser() -> argparse.ArgumentParser:
    """CLI definition (kept separate so tests can construct args directly)."""
    parser = argparse.ArgumentParser(
        prog="eval_ladder",
        description="Elo ladder: every checkpoint of a run vs random, the rule bot and an early anchor.",
    )
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="explicit checkpoint paths (default: the --checkpoint-glob matches)")
    parser.add_argument("--checkpoint-glob", default=DEFAULT_CHECKPOINT_GLOB, dest="checkpoint_glob",
                        help=f"glob used when --checkpoints is omitted (default: {DEFAULT_CHECKPOINT_GLOB})")
    parser.add_argument("--anchor-early", default=DEFAULT_ANCHOR_EARLY, dest="anchor_early",
                        help=f"early checkpoint used as the frozen anchor (default: {DEFAULT_ANCHOR_EARLY})")
    parser.add_argument("--run-name", default="run", dest="run_name",
                        help="name in results/ladder_<run_name>.csv")
    parser.add_argument("--deals", type=int, default=400, help="scored deals per matchup")
    parser.add_argument("--matches", type=int, default=1, help="matches per matchup")
    parser.add_argument("--match-to", type=int, default=100, dest="match_to",
                        help="match target in points")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    parser.add_argument("--out", default=None, help="CSV path (default results/ladder_<run_name>.csv)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="print the planned evaluation list and run no games")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run (or dry-run) the ladder and write the CSV."""
    args = build_ladder_parser().parse_args(argv)
    seeds = parse_seeds(args.seeds)
    checkpoints = resolve_checkpoints(args.checkpoints, args.checkpoint_glob)
    if not checkpoints:
        raise SystemExit(
            f"no checkpoints found for {args.checkpoint_glob!r}; pass --checkpoints or --checkpoint-glob"
        )
    plan = build_plan(checkpoints, seeds)
    out = pathlib.Path(args.out) if args.out else pathlib.Path("results") / f"ladder_{args.run_name}.csv"

    if args.dry_run:
        print(
            f"Planned evaluation: {len(checkpoints)} checkpoint(s) x {len(seeds)} seed(s) "
            f"x {len(LADDER_MATCHUPS)} matchup(s) = {len(plan)} row(s)"
        )
        print(
            f"Anchors (fixed Elo {ANCHOR_ELO:g}): random, rule, early={args.anchor_early}"
        )
        print(f"CSV: {out}")
        for entry in plan:
            exists = "" if pathlib.Path(entry["checkpoint"]).exists() else "  [missing]"
            print(
                f"  {entry['checkpoint']} | seed {entry['seed']} | "
                f"{entry['matchup']} (opponent={entry['opponent']}){exists}"
            )
        return 0

    parser = build_parser()
    rows: list[dict[str, Any]] = []
    for entry in plan:
        rows.append(
            evaluate_entry(
                entry,
                parser=parser,
                anchor_early=args.anchor_early,
                deals=args.deals,
                matches=args.matches,
                match_to=args.match_to,
            )
        )
    rows = finalize_rows(rows)
    written = write_ladder_csv(rows, out)
    print(format_ladder_table(rows))
    print(f"\nwrote {written} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through tests
    raise SystemExit(main())
