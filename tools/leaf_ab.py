"""Leaf A/B: how much does each value leaf lose against the exact solver, in points?

Every candidate plays the *same* positions and the *same* sampled worlds (the sampler is
injected), and the arbiter is the exact solver: for each position it computes the true value of
every legal move, the best move's value, and the regret ``best - played`` of whatever each
candidate chose. Regret is in the environment's own reward units - the same units the judge
reports - so the table is directly comparable with earlier measurements.

    uv run python tools/leaf_ab.py --positions 30 --worlds 12 --model <checkpoint>

Candidates:
    exact        the compiled solver, exact when the deal is short (the current default)
    horizon      the compiled search to the ply horizon with the blocked-now heuristic
    critic       the network leaf, centralised critic on the sampled world (depth 1)
    public       the network leaf, public-information head (depth 1)
    random       a legal move at random (the noise floor)
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import pathlib
import time

import numpy as np

from domino.agents.pool import load_policy
from domino.engine import GameState, TEAM_OF_SEAT
from domino.masks import action_index
from domino.pimc.player import PIMCAgent
from domino.pimc.sampler import PublicView, World, sample_worlds
from domino.pimc.solver import Solver
from domino.tiles import ALL_TILE_IDS, SIX_SIX_ID


def fresh_state(seed: int) -> GameState:
    """A real opening deal."""
    rng = np.random.default_rng(seed)
    tiles = list(ALL_TILE_IDS)
    rng.shuffle(tiles)
    hands = [sorted(tiles[index * 7:(index + 1) * 7]) for index in range(4)]
    leader = next(index for index, hand in enumerate(hands) if SIX_SIX_ID in hand)
    return GameState(hands=[list(hand) for hand in hands], rng=rng, turn=leader, leader=leader,
                     deal_no=1, forced_leader_six_six=True)


def advance(state: GameState, rng: np.random.Generator, plies: int) -> GameState:
    """Play ``plies`` random legal plies so the positions are not all openings."""
    for _ in range(int(plies)):
        if state.is_terminal():
            break
        turn = int(state.turn)
        moves = state.legal_moves(turn)
        if not moves:
            state.pass_turn(turn)
            continue
        tid, end = moves[int(rng.integers(0, len(moves)))]
        state.play(turn, tid, end)
    return state


def own_value(raw: float, team: int) -> float:
    """Team-0 value -> the moving team's value."""
    return float(raw) if int(team) == 0 else -float(raw)


def main(argv: list[str] | None = None) -> int:
    """Run the A/B and print the regret table."""
    parser = argparse.ArgumentParser(description="Compare PIMC value leaves against the exact solver.")
    parser.add_argument("--model", default="models/checkpoints/ppo_16002774_steps.zip")
    parser.add_argument("--positions", type=int, default=30)
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=4, dest="max_plies",
                        help="positions are advanced by 0..max_plies random plies")
    parser.add_argument("--min-plies", type=int, default=0, dest="min_plies",
                        help="lower bound on the random plies (use ~8-12 for late positions)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=30.0, help="per-decision budget (loose)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    model = load_policy(args.model)
    solver = Solver(max_depth=None)                       # exact: the arbiter
    names = ["exact", "horizon", "critic", "public", "rule", "random"]
    regrets: dict[str, list[float]] = {name: [] for name in names}
    matches: dict[str, int] = {name: 0 for name in names}
    times: dict[str, list[float]] = {name: [] for name in names}
    rows: list[dict] = []

    for index in range(int(args.positions)):
        rng = np.random.default_rng(args.seed + 1000 * index)
        span = max(0, int(args.max_plies) - int(args.min_plies))
        plies = int(args.min_plies) + (int(rng.integers(0, span + 1)) if span else 0)
        state = advance(fresh_state(args.seed + index), rng, plies)
        if state.is_terminal():
            continue
        seat = int(state.turn)
        team = int(TEAM_OF_SEAT[seat])
        moves = [(int(tid), int(end)) for tid, end in state.legal_moves(seat)]
        if len(moves) < 2:
            continue
        mark = time.perf_counter()
        exact = solver.solve(state)
        times["exact"].append(time.perf_counter() - mark)
        if not exact.complete:
            continue
        per_move = {move: own_value(exact.move_values[move], team) for move in moves}
        best = max(moves, key=lambda move: per_move[move])
        best_value = per_move[best]
        worlds: list[World] = sample_worlds(PublicView.from_state(state, seat), int(args.worlds),
                                           rng, method="rejection")

        def injected(view: PublicView, count: int, generator) -> list[World]:
            """The same worlds for every candidate, so the comparison is paired."""
            return list(worlds[:count])

        plays: dict[str, tuple[int, int] | None] = {"exact": best}
        # The compiled horizon search is what the shipped PIMC does by default.
        horizon_agent = PIMCAgent(worlds=int(args.worlds), horizon=int(args.horizon),
                                  time_budget=args.budget, leaf_eval="blocked_now",
                                  exact_when_short=False, sampler=injected, seed=index)
        plays["horizon"] = _acted(horizon_agent, state, seat, times["horizon"])
        for mode in ("critic", "public"):
            agent = PIMCAgent(worlds=int(args.worlds), horizon=int(args.horizon),
                              time_budget=args.budget, leaf_eval=mode, model=model,
                              sampler=injected, seed=index)
            plays[mode] = _acted(agent, state, seat, times[mode])
        plays["random"] = moves[int(rng.integers(0, len(moves)))]
        # The hand-coded baseline, as the yardstick for "how far from perfect is anything".
        from domino.agents import RuleAgent

        rule = RuleAgent(seed=index)
        started = time.perf_counter()
        rule_action = int(rule.act(state, seat))
        times["rule"].append(time.perf_counter() - started)
        plays["rule"] = (rule_action // 2, rule_action % 2)

        row = {"position": index, "plies": plies, "seat": seat, "tiles": len(state.hands[seat]),
               "n_moves": len(moves), "best_value": best_value,
               "best": f"{best[0]}e{best[1]}"}
        for name in names:
            move = plays[name]
            regret = 0.0 if move is None else float(best_value - per_move[move])
            regrets[name].append(regret)
            matches[name] += int(move == best)
            row[f"{name}_regret"] = regret
            row[f"{name}_move"] = "" if move is None else f"{move[0]}e{move[1]}"
        rows.append(row)

    header = (f"{'candidate':<10}{'decisions':>10}{'mean regret':>13}{'se':>8}"
              f"{'best-move match':>17}{'ms/decision':>13}")
    print(header)
    print("-" * len(header))
    for name in names:
        values = np.asarray(regrets[name], dtype=np.float64)
        se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")
        share = matches[name] / max(1, values.size)
        spent = float(np.mean(times[name]) * 1000.0) if times[name] else float("nan")
        print(f"{name:<10}{values.size:>10}{values.mean():>13.3f}{se:>8.3f}"
              f"{share:>17.1%}{spent:>13.1f}")

    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        "results") / f"leaf_ab_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} position(s))")
    return 0


def _acted(agent: PIMCAgent, state: GameState, seat: int,
           sink: list[float]) -> tuple[int, int]:
    """Run one agent decision, timing it, and return the move it played."""
    started = time.perf_counter()
    action = agent.act(state, seat)
    sink.append(time.perf_counter() - started)
    return (action // 2, action % 2)


if __name__ == "__main__":
    raise SystemExit(main())
