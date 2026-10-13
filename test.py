"""Evaluate the trained RL agent against the rule-based baseline and random play.

This is the assignment's required comparison script. It plays head-to-head series
between agents, computes several justified metrics, prints a table and writes one
CSV row per matchup that ``evaluate.ipynb`` turns into plots and tables.

Metrics and why each one is used
--------------------------------
``win_rate_a``
    Deals won by agent A's team divided by scored deals -- the cleanest strength
    measure, because a deal is the natural independent trial of this game.
``ci_low`` / ``ci_high``
    Wilson 95 % interval for ``win_rate_a``. Wilson (rather than the normal
    approximation) is used because the rate sits near 0.5 and the sample is
    modest, where the naive interval can exceed [0, 1].
``points_per_deal_a`` / ``points_per_deal_b``
    Mean points banked per deal, which captures the margin rather than only the
    win/loss outcome and is exactly the quantity the agent's reward optimises.
``match_win_rate_a``
    Matches won to ``--match-to`` points -- the human-meaningful score of the
    game as it is actually played.
``block_rate``
    Fraction of scored deals that ended blocked. Shows whether the agent learned
    the blocked endgame at all (the fewest-pips rule decides those deals).
``void_rate``
    Fraction of dealt rounds voided by the fewest-pips tie rule (no points,
    redeal). Reported so the reader can see how often the tie rule fires.
``mean_pips_left_a``
    Mean pips still in agent A's team's hands when a deal ends -- a direct
    efficiency measure of the blocked endgame, where the lowest-pip player wins.
``ms_per_action``
    Mean wall-clock milliseconds per decision, i.e. the inference cost that
    matters for the interactive mode.
``deals``
    Scored deals behind the row, so a headline rate is never read without its
    sample size.

Sample efficiency (how many deals an agent needs to pass the baseline) is *not*
computed here: the training log records per-episode reward, not periodic
head-to-head results, so it is presented in ``evaluate.ipynb`` as the convergence
curve (total reward per episode against steps) instead of a fabricated win-rate
series. Periodic in-training evaluation is listed as future work in the report.

Conventions
-----------
* Deal-phase deals are dealt fresh with deal-1 rules (the ``[6|6]`` holder leads
  and must play it), which is exactly the training episode's convention.
* A void deal scores nothing and is redealt; ``void_rate`` counts those rounds
  while ``deals`` counts only the scored ones.
* The match phase uses :class:`domino.engine.Match`, so the point cap and the
  "previous winner leads the next deal" rule apply.
* Agent A occupies team 0 (seats 0 and 2) unless ``--swap`` is passed, in which
  case it occupies team 1 -- this removes any seat/first-player bias from the
  comparison.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import pathlib
import time
from collections.abc import Callable

import numpy as np

from domino.agents import Agent, PolicyAgent, RandomAgent, RuleAgent
from domino.agents.pool import load_policy
from domino.engine import DealOutcome, GameState, Match, deal_outcome, new_deal, redeal_after_void
from domino.masks import decode_action

MATCHUPS: tuple[str, ...] = (
    "rl_vs_rule",
    "rl_vs_random",
    "rule_vs_random",
    "rl_vs_rl",
    "rule_vs_rule",
    "random_vs_random",
    "pimc_vs_rule",
    "pimc_vs_random",
    "pimc_vs_rl",
)

TEAM_A_SEATS: tuple[int, int] = (0, 2)
TEAM_B_SEATS: tuple[int, int] = (1, 3)

CSV_FIELDS: tuple[str, ...] = (
    "matchup",
    "agent_a",
    "agent_b",
    "deals",
    "wins_a",
    "wins_b",
    "win_rate_a",
    "ci_low",
    "ci_high",
    "points_per_deal_a",
    "points_per_deal_b",
    "match_win_rate_a",
    "matches_played",
    "block_rate",
    "void_rate",
    "mean_pips_left_a",
    "ms_per_action",
    "model_path",
    "seed",
)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _kinds(matchup: str) -> tuple[str, str]:
    """Split a matchup name into (agent A kind, agent B kind)."""
    if matchup not in MATCHUPS:
        raise ValueError(f"unknown matchup {matchup!r}; known: {', '.join(MATCHUPS)}")
    a, b = matchup.split("_vs_")
    return a, b


def _make_agent(kind: str, model: object | None, seed: int, search_k: int = 0,
                args: argparse.Namespace | None = None) -> Agent:
    """Build one agent of the requested kind."""
    if kind == "rl":
        if model is None:
            raise ValueError("matchup needs a trained model; pass --model")
        if search_k > 0:
            # v3 item 5: PIMC root selection with the trained value net as the leaf
            # evaluator. Deterministic and seeded, so an A/B against the plain policy
            # runs on identical deals.
            from domino.search import PIMCPolicyAgent, value_fn_from_model

            return PIMCPolicyAgent(
                value_fn_from_model(model), n_determinizations=int(search_k), seed=seed
            )
        return PolicyAgent(model, deterministic=True, name="rl")  # type: ignore[arg-type]
    if kind == "pimc":
        # Phase 3 of the search brief: sampled-world PIMC over the belief sampler, with the
        # exact/heuristic solver as the world evaluator.
        from domino.pimc import PIMCAgent

        worlds = int(getattr(args, "pimc_worlds", 32) or 32)
        horizon = int(getattr(args, "pimc_horizon", 10) or 0)
        budget = float(getattr(args, "pimc_budget", 1.0) or 1.0)
        leaf = str(getattr(args, "pimc_leaf", "blocked_now") or "blocked_now")
        method = str(getattr(args, "pimc_sampler", "auto") or "auto")
        opponent_model = str(getattr(args, "pimc_opponent_model", "min") or "min")
        opponent_policy = RuleAgent(seed=seed) if opponent_model == "policy" else None
        return PIMCAgent(worlds=worlds, horizon=horizon, time_budget=budget, leaf_eval=leaf,
                         method=method, seed=seed, opponent_model=opponent_model,
                         opponent_policy=opponent_policy)
    if kind == "rule":
        return RuleAgent(seed=seed)
    if kind == "random":
        return RandomAgent(seed=seed)
    raise ValueError(f"unknown agent kind {kind!r}")


def _build_agents(matchup: str, args: argparse.Namespace) -> tuple[dict[int, Agent], str, str]:
    """Return seat -> agent plus the two agent names for the CSV row."""
    a_kind, b_kind = _kinds(matchup)
    model_a = load_policy(args.model) if a_kind == "rl" else None
    model_b = load_policy(args.model_b or args.model) if b_kind == "rl" else None
    search_k = int(getattr(args, "search", 0) or 0)
    agents: dict[int, Agent] = {}
    a_seats, b_seats = (TEAM_A_SEATS, TEAM_B_SEATS) if not args.swap else (TEAM_B_SEATS, TEAM_A_SEATS)
    for seat in a_seats:
        agents[seat] = _make_agent(a_kind, model_a, seed=args.seed + seat, search_k=search_k,
                                   args=args)
    for seat in b_seats:
        agents[seat] = _make_agent(b_kind, model_b, seed=args.seed + 100 + seat, args=args)
    return agents, a_kind, b_kind


def play_deal_with_agents(
    agents: dict[int, Agent],
    rng: np.random.Generator,
    timings: list[float] | None = None,
) -> tuple[DealOutcome, list[DealOutcome], GameState]:
    """Play one fresh deal (deal-1 rules) to its scored end.

    Returns ``(outcome, void_outcomes, final_state)``; void deals are redealt
    inside the loop exactly as :meth:`domino.engine.Match.play_deal` does. Every
    rule decision is taken by the engine -- this loop only routes seats to
    agents.
    """
    voids: list[DealOutcome] = []
    state = new_deal(deal_no=1, rng=rng, lead_mode="six_six")
    while True:
        while not state.is_terminal():
            seat = state.turn
            if state.must_pass(seat):
                state.pass_turn(seat)
                continue
            agent = agents[seat]
            if timings is None:
                action = int(agent.act(state, seat))
            else:
                started = time.perf_counter()
                action = int(agent.act(state, seat))
                timings.append(time.perf_counter() - started)
            tid, end = decode_action(action)
            state.play(seat, tid, end)
        outcome = deal_outcome(state)
        if not outcome.void:
            return outcome, voids, state
        voids.append(outcome)
        state = redeal_after_void(state)


def _match_phase(agents: dict[int, Agent], args: argparse.Namespace) -> tuple[int, int]:
    """Play ``--matches`` matches to ``--match-to`` points; return (A wins, played)."""
    policy: dict[int, Callable[[GameState, int], int]] = {
        seat: agent.act for seat, agent in agents.items()
    }
    a_team = 1 if args.swap else 0
    wins = 0
    played = 0
    for index in range(args.matches):
        match = Match(target=args.match_to, seed=args.seed + 1000 + index)
        guard = 0
        while not match.finished and guard < 10_000:
            match.play_deal(policy, action_decoder=decode_action)
            guard += 1
        played += 1
        if match.winner_team == a_team:
            wins += 1
    return wins, played


def evaluate_matchup(matchup: str, args: argparse.Namespace) -> dict[str, object]:
    """Run the deal and match phases for one matchup and return its metrics row."""
    agents, a_kind, b_kind = _build_agents(matchup, args)
    rng = np.random.default_rng(args.seed)
    timings: list[float] = []

    wins_a = 0
    wins_b = 0
    points_a = 0
    points_b = 0
    blocked = 0
    voids = 0
    pips_left_a: list[int] = []

    for _ in range(args.deals):
        outcome, void_outcomes, state = play_deal_with_agents(agents, rng, timings)
        voids += len(void_outcomes)
        if outcome.winner_team == 0:
            wins_a += 1
            points_a += outcome.points
        else:
            wins_b += 1
            points_b += outcome.points
        if outcome.blocked:
            blocked += 1
        pips_left_a.append(sum(state.pips(seat) for seat in TEAM_A_SEATS))

    scored = wins_a + wins_b
    rounds = scored + voids
    match_wins, matches_played = _match_phase(agents, args)
    ci_low, ci_high = wilson_interval(wins_a, scored)

    return {
        "matchup": matchup,
        "agent_a": a_kind,
        "agent_b": b_kind,
        "deals": scored,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "win_rate_a": wins_a / scored if scored else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "points_per_deal_a": points_a / scored if scored else float("nan"),
        "points_per_deal_b": points_b / scored if scored else float("nan"),
        "match_win_rate_a": match_wins / matches_played if matches_played else float("nan"),
        "matches_played": matches_played,
        "block_rate": blocked / scored if scored else float("nan"),
        "void_rate": voids / rounds if rounds else float("nan"),
        "mean_pips_left_a": float(np.mean(pips_left_a)) if pips_left_a else float("nan"),
        "ms_per_action": 1000.0 * float(np.mean(timings)) if timings else float("nan"),
        "model_path": args.model,
        "seed": args.seed,
    }


def write_csv(rows: list[dict[str, object]], path: pathlib.Path) -> pathlib.Path:
    """Write the metric rows with the fixed schema; returns the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in CSV_FIELDS})
    return path


def format_table(rows: list[dict[str, object]]) -> str:
    """Render the metric rows as a plain fixed-width table."""
    header = (
        f"{'matchup':<16}{'deals':>6}{'W-L':>9}{'win_rate':>10}{'95% CI':>16}"
        f"{'pts/deal A':>12}{'pts/deal B':>12}{'match WR':>10}{'block':>8}{'void':>7}{'ms/act':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row['matchup']):<16}{int(row['deals']):>6}"
            f"{str(int(row['wins_a'])) + '-' + str(int(row['wins_b'])):>9}"
            f"{float(row['win_rate_a']):>10.3f}"
            f"{'[' + format(float(row['ci_low']), '.3f') + ', ' + format(float(row['ci_high']), '.3f') + ']':>16}"
            f"{float(row['points_per_deal_a']):>12.2f}{float(row['points_per_deal_b']):>12.2f}"
            f"{float(row['match_win_rate_a']):>10.3f}{float(row['block_rate']):>8.3f}"
            f"{float(row['void_rate']):>7.3f}{float(row['ms_per_action']):>9.2f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """CLI definition (kept separate so tests can construct args directly)."""
    parser = argparse.ArgumentParser(
        description="Evaluate the RL agent against the rule-based baseline and random play.",
    )
    parser.add_argument("--model", default="models/checkpoints/latest.zip",
                        help="checkpoint of agent A (and of both sides for rl_vs_rl)")
    parser.add_argument("--model-b", default=None,
                        help="optional second checkpoint (defaults to --model)")
    parser.add_argument("--matchups", default="all",
                        help=f"comma-separated subset of {','.join(MATCHUPS)} or 'all'")
    parser.add_argument("--deals", type=int, default=1000,
                        help="scored deals per matchup")
    parser.add_argument("--matches", type=int, default=5,
                        help="matches to play per matchup (to --match-to points)")
    parser.add_argument("--match-to", type=int, default=100, dest="match_to",
                        help="match target in points")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument("--swap", action="store_true",
                        help="put agent A on team 1 (seats 1 and 3) instead of team 0")
    parser.add_argument("--search", type=int, default=0,
                        help="PIMC determinizations for agent A's policy (0 = plain policy; "
                             "v3 item 5, needs a centralised value net)")
    parser.add_argument("--pimc-worlds", type=int, default=32, dest="pimc_worlds",
                        help="sampled worlds per decision for the 'pimc' agent kind")
    parser.add_argument("--pimc-horizon", type=int, default=10, dest="pimc_horizon",
                        help="ply horizon of the solver inside 'pimc' (exact once short)")
    parser.add_argument("--pimc-budget", type=float, default=1.0, dest="pimc_budget",
                        help="wall-clock seconds per 'pimc' decision")
    parser.add_argument("--pimc-leaf", default="blocked_now", dest="pimc_leaf",
                        help="leaf estimate of the inner search: blocked_now | pip_diff")
    parser.add_argument("--pimc-sampler", default="auto", dest="pimc_sampler",
                        help="belief sampler: auto | rejection | constrained")
    parser.add_argument("--pimc-opponent-model", default="min", dest="pimc_opponent_model",
                        help="opponents inside the tree: min (pessimistic) | policy (rule bot)")
    parser.add_argument("--out", default=None,
                        help="CSV path (default results/eval_<timestamp>.csv)")
    parser.add_argument("--quiet", action="store_true", help="suppress the printed table")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested matchups and write the CSV."""
    args = build_parser().parse_args(argv)
    names = list(MATCHUPS) if args.matchups == "all" else [
        name.strip() for name in args.matchups.split(",") if name.strip()
    ]
    for name in names:
        _kinds(name)  # fail fast on a typo before any game is played

    rows: list[dict[str, object]] = []
    for name in names:
        started = time.perf_counter()
        row = evaluate_matchup(name, args)
        rows.append(row)
        if not args.quiet:
            print(f"[{name}] {float(time.perf_counter() - started):.1f}s")

    if not args.quiet:
        print()
        print(format_table(rows))

    if args.out:
        out = pathlib.Path(args.out)
    else:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = pathlib.Path("results") / f"eval_{stamp}.csv"
    written = write_csv(rows, out)
    print(f"\nwrote {written} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
