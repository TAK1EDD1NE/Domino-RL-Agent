"""Reproducible coarse coordinate-ascent sweep over the rule bot's seven weights.

The rule bot (``domino/agents/rule_agent.py``) scores legal moves with a
seven-term evaluation function whose weights were originally chosen by hand.
This tool chooses them *by measurement*: it searches the weight space with
multiplicative coordinate ascent and ranks candidates by their **mean win rate
against a random opponent** across a fixed set of evaluation seeds, breaking
ties by mean points-per-deal.

Design
------
* Candidate evaluation plays ``Match(target=100)`` with the rule bot on team 0's
  seats ``{0, 2}`` and ``RandomAgent`` on seats ``{1, 3}``, exactly the harness
  the win-rate gate (``tests/test_agents.py::test_rule_bot_beats_random_over_300_deals``)
  uses, so the two numbers are comparable.
* Search: multiplicative coordinate ascent. For each of the seven weights in
  turn, try the factors ``{0.5, 1, 3}`` around its current value, keep the best,
  and repeat for ``--rounds`` rounds. A factor is only accepted if it strictly
  improves the ranking key (mean win rate, then points-per-deal), so the sweep
  never deviates gratuitously.
* Determinism: every run is fully reproducible from its seeds — the bot is
  deterministic (ties break to the lowest action index) and both the bot and the
  random opponent draw from explicit generators. There is no global mutable
  state.

Seed disjointness (important)
-----------------------------
The evaluation seed set is **disjoint** from the win-rate gate's test seeds
``(0, 1, 2, 3, 4, 5, 7, 42, 100, 123)``. The defaults are ``(1000, 1001, 1002,
1003, 1004)`` at 300 deals each. This guarantees the weights selected here are
not fitted to the seeds the gate scores on; the gate is therefore an honest
held-out check of the tuned weights.

Usage
-----
    uv run python tools/sweep_rule_weights.py
    uv run python tools/sweep_rule_weights.py --rounds 2 --deals 300 --workers 4
    uv run python tools/sweep_rule_weights.py --seeds 1000,1001 --deals 100

The full ranked candidate table is written to ``--out`` (default
``results/rule_weight_sweep.csv``). Dependencies: numpy and the package only —
no torch, no sb3, no pygame.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# Make `python tools/sweep_rule_weights.py` importable when the editable install
# has no path hook; `python -m tools.sweep_rule_weights` also works.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from domino.agents import RandomAgent, RuleAgent, RuleWeights  # noqa: E402
from domino.engine import GameState, Match  # noqa: E402
from domino.masks import decode_action  # noqa: E402

# The seven tunable fields, in sweep order.
WEIGHT_FIELDS: tuple[str, ...] = (
    "pips_shed",
    "kills_value",
    "opp_short_end",
    "keep_doubles",
    "protect_partner",
    "block_economics",
    "duplicate_lead",
)

# Evaluation seeds — deliberately disjoint from the gate's test seeds.
DEFAULT_SEEDS: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004)
DEFAULT_DEALS: int = 300
DEFAULT_ROUNDS: int = 2
DEFAULT_FACTORS: tuple[float, ...] = (0.5, 1.0, 3.0)
DEFAULT_OUT: str = "results/rule_weight_sweep.csv"
DEFAULT_MAX_SECONDS: float = 900.0
MATCH_TARGET: int = 100


@dataclass(frozen=True)
class EvalResult:
    """A candidate's measured performance over the evaluation seed set."""

    mean_win_rate: float
    sd_win_rate: float
    points_per_deal: float
    rates: tuple[float, ...]


@dataclass
class Candidate:
    """One evaluated weight vector, tagged with where in the search it arose."""

    label: str
    round: int
    field: str
    factor: float
    weights: RuleWeights
    result: EvalResult


def _match_policy(agent) -> Callable[[GameState, int], int]:
    """Adapt an ``Agent`` (56-dim action) to ``Match``'s tile-id policy."""
    def policy(state: GameState, seat: int) -> int:
        return decode_action(agent.act(state, seat))[0]
    return policy


def _match_stats(weights: RuleWeights, deals: int, seed: int) -> tuple[float, float]:
    """Play one seed's batch; return ``(win_rate, points_per_deal)`` for team 0.

    ``win_rate`` is the fraction of the ``deals`` deals won by team 0 (the bot's
    team); ``points_per_deal`` is the mean points team 0 banked per deal (the
    tiebreaker). A fresh ``Match`` is started whenever one finishes before the
    deal budget is met, mirroring the gate's harness.
    """
    rule = RuleAgent(weights=weights, seed=seed)
    rand = RandomAgent(seed=seed + 1)
    policies = {
        0: _match_policy(rule),
        2: _match_policy(rule),
        1: _match_policy(rand),
        3: _match_policy(rand),
    }
    wins = 0
    team0_points = 0
    played = 0
    match_seed = seed
    while played < deals:
        match = Match(target=MATCH_TARGET, seed=match_seed)
        match_seed += 1
        while not match.finished and played < deals:
            outcome = match.play_deal(policies)
            played += 1
            if outcome.winner_team == 0:
                wins += 1
                team0_points += outcome.points
    return wins / played, team0_points / played


def _seed_worker(job: tuple[RuleWeights, int, int]) -> tuple[float, float]:
    """Picklable top-level worker for the optional process pool."""
    weights, deals, seed = job
    return _match_stats(weights, deals, seed)


def evaluate(
    weights: RuleWeights,
    seeds: Sequence[int],
    deals: int,
    workers: int = 1,
) -> EvalResult:
    """Measure ``weights`` over ``seeds`` (``deals`` each); fully deterministic.

    Returns the mean win rate, its between-seed sample standard deviation
    (``ddof=1``), the mean points-per-deal and the per-seed win rates. With
    ``workers > 1`` the independent per-seed batches run in a process pool.
    """
    jobs = [(weights, deals, int(seed)) for seed in seeds]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_seed_worker, jobs))
    else:
        outcomes = [_seed_worker(job) for job in jobs]
    rates = np.array([rate for rate, _ in outcomes], dtype=float)
    ppds = np.array([ppd for _, ppd in outcomes], dtype=float)
    sd = float(rates.std(ddof=1)) if rates.size > 1 else 0.0
    return EvalResult(
        mean_win_rate=float(rates.mean()),
        sd_win_rate=sd,
        points_per_deal=float(ppds.mean()),
        rates=tuple(float(r) for r in rates),
    )


def _rank_key(result: EvalResult) -> tuple[float, float]:
    """Ranking key: mean win rate first, points-per-deal as the tiebreaker."""
    return result.mean_win_rate, result.points_per_deal


def _fmt(label: str, result: EvalResult, improved: bool | None = None) -> str:
    tag = "" if improved is None else ("  *best" if improved else "")
    return (
        f"{label:<24} mean={result.mean_win_rate:.4f} "
        f"sd={result.sd_win_rate:.4f} ppd={result.points_per_deal:.3f}{tag}"
    )


def coordinate_ascent(
    start: RuleWeights,
    seeds: Sequence[int],
    deals: int,
    rounds: int,
    factors: Sequence[float],
    workers: int = 1,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    log: Callable[[str], None] = print,
) -> tuple[RuleWeights, EvalResult, EvalResult, list[Candidate], bool]:
    """Run multiplicative coordinate ascent from ``start``.

    Returns ``(best_weights, best_result, baseline_result, rows, timed_out)``.
    On a wall-clock timeout the best-so-far is returned with ``timed_out=True``.
    """
    started = time.time()
    rows: list[Candidate] = []
    baseline = evaluate(start, seeds, deals, workers)
    rows.append(Candidate("baseline", 0, "(start)", 1.0, start, baseline))
    log(f"seeds={tuple(seeds)} deals={deals} rounds={rounds} factors={tuple(factors)}")
    log(_fmt("baseline", baseline))

    best = start
    best_result = baseline
    timed_out = False
    for rnd in range(1, rounds + 1):
        for field in WEIGHT_FIELDS:
            current_value = getattr(best, field)
            field_best = best
            field_best_result = best_result
            for factor in factors:
                if factor == 1.0:
                    continue  # the current value is already measured
                if max_seconds and (time.time() - started) > max_seconds:
                    timed_out = True
                    log(f"[max-seconds {max_seconds:g} reached — stopping]")
                    break
                candidate = replace(best, **{field: current_value * factor})
                result = evaluate(candidate, seeds, deals, workers)
                label = f"r{rnd}-{field}-x{factor:g}"
                improved = _rank_key(result) > _rank_key(field_best_result)
                rows.append(Candidate(label, rnd, field, factor, candidate, result))
                log(_fmt(label, result, improved))
                if improved:
                    field_best = candidate
                    field_best_result = result
            best = field_best
            best_result = field_best_result
            if timed_out:
                break
        if timed_out:
            break
    return best, best_result, baseline, rows, timed_out


def _paired_se_diff(best: EvalResult, baseline: EvalResult) -> float:
    """Standard error of the paired per-seed win-rate difference (best − baseline)."""
    diff = np.array(best.rates, dtype=float) - np.array(baseline.rates, dtype=float)
    if diff.size <= 1:
        return 0.0
    return float(diff.std(ddof=1) / np.sqrt(diff.size))


def write_csv(path: Path, rows: Sequence[Candidate]) -> Path:
    """Write the full ranked candidate table to ``path`` (parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda c: (-c.result.mean_win_rate, -c.result.points_per_deal))
    header = [
        "rank", "label", "round", "field", "factor",
        "mean_win_rate", "sd_win_rate", "points_per_deal",
        *WEIGHT_FIELDS,
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for rank, cand in enumerate(ranked, start=1):
            weights = asdict(cand.weights)
            writer.writerow([
                rank, cand.label, cand.round, cand.field, cand.factor,
                f"{cand.result.mean_win_rate:.6f}",
                f"{cand.result.sd_win_rate:.6f}",
                f"{cand.result.points_per_deal:.6f}",
                *[f"{weights[f]:g}" for f in WEIGHT_FIELDS],
            ])
    return path


def _parse_seeds(text: str) -> tuple[int, ...]:
    """Parse a comma/space separated seed list, e.g. ``1000,1001,1002``."""
    parts = text.replace(",", " ").split()
    if not parts:
        raise argparse.ArgumentTypeError("seed list is empty")
    return tuple(int(part) for part in parts)


def _weights_literal(weights: RuleWeights) -> str:
    inner = ", ".join(f"{field}={getattr(weights, field):g}" for field in WEIGHT_FIELDS)
    return f"RuleWeights({inner})"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: sweep the rule-bot weights and report the best."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS,
                        help=f"evaluation seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--deals", type=int, default=DEFAULT_DEALS,
                        help=f"deals per seed (default: {DEFAULT_DEALS})")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=f"coordinate-ascent rounds (default: {DEFAULT_ROUNDS})")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT),
                        help=f"ranked CSV output path (default: {DEFAULT_OUT})")
    parser.add_argument("--workers", type=int, default=1,
                        help="process-pool size for per-seed batches (default: 1, single-process)")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                        help=f"wall-clock safety cap (default: {DEFAULT_MAX_SECONDS:g})")
    args = parser.parse_args(argv)

    start = RuleWeights()
    print(f"start weights: {_weights_literal(start)}")
    started = time.time()
    best, best_result, baseline, rows, timed_out = coordinate_ascent(
        start,
        seeds=args.seeds,
        deals=args.deals,
        rounds=args.rounds,
        factors=DEFAULT_FACTORS,
        workers=args.workers,
        max_seconds=args.max_seconds,
    )
    out_path = write_csv(args.out, rows)

    improvement = best_result.mean_win_rate - baseline.mean_win_rate
    se_diff = _paired_se_diff(best_result, baseline)
    se_mean_best = best_result.sd_win_rate / np.sqrt(len(best_result.rates))
    real = improvement > 2.0 * se_diff

    print()
    print(f"baseline mean win rate : {baseline.mean_win_rate:.4f} "
          f"(sd {baseline.sd_win_rate:.4f}, ppd {baseline.points_per_deal:.3f})")
    print(f"best     mean win rate : {best_result.mean_win_rate:.4f} "
          f"(sd {best_result.sd_win_rate:.4f}, ppd {best_result.points_per_deal:.3f})")
    print(f"improvement            : {improvement:+.4f}")
    print(f"paired SE of difference: {se_diff:.4f}  (2*SE = {2.0 * se_diff:.4f})")
    print(f"SE of best mean        : {se_mean_best:.4f}  (2*SE = {2.0 * se_mean_best:.4f})")
    print(f"improvement > 2*paired-SE: {real}")
    print(f"best weights           : {_weights_literal(best)}")
    print(f"ranked table           : {out_path}")
    print(f"elapsed                : {time.time() - started:.1f}s"
          f"{'  [TIMED OUT — best-so-far]' if timed_out else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
