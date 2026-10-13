"""Phase 5: the evaluation harness — duplicate deals, confidence intervals, Elo.

Duplicate-deal protocol
-----------------------
Tile luck is the dominant variance source in a single domino deal, so every deal is played
**twice**: once with the deal as dealt, and once with the whole seating rotated by one seat
(``new_hands[s + 1] = hands[s]``, leader rotated with it). Agent A keeps the *team-0* seats
in both games, so in the second game it plays exactly the tiles its opponents held in the
first — and no metric has to be "re-attributed" afterwards, which is the bug the project's
own ``test.py --swap`` rows still carry (there, team-0 columns stay team 0 while the labels
move). A void deal is redealt for the *pair*, so both orientations always see the same deal.

* ``games`` counts both orientations, so a deal contributes two independent game outcomes.
* ``win_rate_a`` is the share of games A's team won, with a Wilson interval.
* ``paired_differential`` is the mean over deals of ``(diff_game1 + diff_game2) / 2`` — the
  quantity that cancels the deal's tile luck — with a bootstrap interval over deals (the
  deals, not the games, are the independent unit).

Round robin and Elo
-------------------
Every pair of agents plays the same duplicate-deal scheme; the pairwise game counts feed a
Bradley-Terry fit (Hunter's MM iteration) and the Elo scale ``400 / ln 10 * ln(pi_i)``,
anchored on the field mean.

Parallelism, memory, resumability
---------------------------------
Deals are distributed over ``multiprocessing`` workers, capped at ``cpu_count() - 2`` so the
laptop stays usable, and every matchup row is appended to the CSV as soon as it is known, so
a crash loses nothing. Before a long run the caller gets a runtime estimate from a one-deal
pilot.
"""
from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
import pathlib
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from ..agents import Agent, PolicyAgent, RandomAgent, RuleAgent
from ..engine import SEATS, GameState, deal_outcome
from ..masks import decode_action
from ..tiles import ALL_TILE_IDS, SIX_SIX_ID

#: Agent kinds the harness can build inside a worker (a spec must stay picklable).
AGENT_KINDS: tuple[str, ...] = ("random", "rule", "pimc", "rl")

CSV_FIELDS: tuple[str, ...] = (
    "matchup", "agent_a", "agent_b", "deals", "games", "wins_a", "wins_b", "win_rate_a",
    "wilson_low", "wilson_high", "paired_differential", "boot_low", "boot_high",
    "points_per_game_a", "points_per_game_b", "paired_win_rate", "block_rate", "void_rate",
    "mean_pips_left_a", "ms_per_action", "workers", "seed",
)


@dataclass(frozen=True)
class AgentSpec:
    """A picklable recipe for building an agent inside a worker process."""

    name: str
    kind: str
    options: dict = field(default_factory=dict)

    def build(self, seed: int) -> Agent:
        """Instantiate the agent for one seat."""
        if self.kind == "random":
            return RandomAgent(seed=seed)
        if self.kind == "rule":
            return RuleAgent(seed=seed)
        if self.kind == "pimc":
            from .player import PIMCAgent

            return PIMCAgent(seed=seed, **self.options)
        if self.kind == "rl":
            from ..agents.pool import load_policy

            model_path = str(self.options.get("model_path", ""))
            if not model_path:
                raise ValueError("an 'rl' agent spec needs options['model_path']")
            model = load_policy(model_path)
            return PolicyAgent(model, deterministic=True, name=self.name)  # type: ignore[arg-type]
        raise ValueError(f"unknown agent kind {self.kind!r}")


# ---------------------------------------------------------------------------
# deal play
# ---------------------------------------------------------------------------
def deal_hands(rng: np.random.Generator) -> list[list[int]]:
    """Shuffle the 28 tiles into four hands of seven."""
    tiles = list(ALL_TILE_IDS)
    rng.shuffle(tiles)
    return [sorted(tiles[index * 7:(index + 1) * 7]) for index in range(SEATS)]


def rotate_hands(hands: Sequence[Sequence[int]]) -> list[list[int]]:
    """Rotate the seating by one: ``new_hands[s + 1] = hands[s]`` (teams swap)."""
    return [sorted(int(t) for t in hands[(seat - 1) % SEATS]) for seat in range(SEATS)]


def state_from_hands(hands: Sequence[Sequence[int]], rng: np.random.Generator) -> GameState:
    """Build the opening state of a deal (the ``[6|6]`` holder leads and must open)."""
    leader = next(seat for seat, hand in enumerate(hands) if SIX_SIX_ID in hand)
    return GameState(hands=[list(hand) for hand in hands], rng=rng, turn=leader, leader=leader,
                     deal_no=1, forced_leader_six_six=True)


def play_deal(agents: dict[int, Agent], hands: Sequence[Sequence[int]],
              rng: np.random.Generator,
              timings: Optional[list[float]] = None) -> tuple[object, GameState]:
    """Play one deal from explicit hands; returns ``(outcome, final_state)``.

    A void deal is *returned as such* rather than redealt here: the duplicate protocol must
    redeal **both** orientations of a deal, or the two games stop being mirror images (the
    first version of this module redealt inside ``play_deal`` and silently broke the pairing).
    """
    state = state_from_hands(hands, rng)
    while not state.is_terminal():
        seat = int(state.turn)
        moves = state.legal_moves(seat)
        if not moves:
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
    return deal_outcome(state), state


#: How many times a duplicate pair is redealt if the deal goes void.
MAX_PAIR_ATTEMPTS: int = 20


def play_duplicate_pair(spec_a: AgentSpec, spec_b: AgentSpec, seed: int,
                        timings: Optional[list[float]] = None) -> dict:
    """Play one deal twice (normal seating, then rotated) and report both games.

    The same deal is used for both orientations; if either orientation ends void the pair is
    dealt again, so the two games are always mirror images of one deal.
    """
    rng = np.random.default_rng(seed)
    voids = 0
    for _ in range(MAX_PAIR_ATTEMPTS):
        hands = deal_hands(rng)
        agents = {seat: (spec_a if seat in (0, 2) else spec_b).build(seed + seat)
                  for seat in range(SEATS)}
        outcome_a, final_a = play_deal(agents, hands, rng, timings)
        rotated = rotate_hands(hands)
        agents_rotated = {seat: (spec_a if seat in (0, 2) else spec_b).build(seed + 1000 + seat)
                          for seat in range(SEATS)}
        outcome_b, final_b = play_deal(agents_rotated, rotated, rng, timings)
        if not outcome_a.void and not outcome_b.void:
            break
        voids += int(bool(outcome_a.void)) + int(bool(outcome_b.void))
    else:                                                     # pragma: no cover - unlikely
        raise RuntimeError(f"duplicate pair for seed {seed} voided {voids} times in a row")

    def differential(outcome) -> int:
        if outcome.winner_team == 0:
            return int(outcome.points)
        if outcome.winner_team == 1:
            return -int(outcome.points)
        return 0

    return {
        "seed": int(seed),
        "differential_game1": differential(outcome_a),
        "differential_game2": differential(outcome_b),
        "won_a_game1": bool(outcome_a.winner_team == 0),
        "won_a_game2": bool(outcome_b.winner_team == 0),
        "points_game1": int(outcome_a.points),
        "points_game2": int(outcome_b.points),
        "blocked": bool(outcome_a.blocked) + bool(outcome_b.blocked),
        "pips_left_a_game1": int(sum(final_a.pips(seat) for seat in (0, 2))),
        "pips_left_a_game2": int(sum(final_b.pips(seat) for seat in (0, 2))),
        "voids": voids,
    }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values: Sequence[float], rng: np.random.Generator,
                      draws: int = 10_000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean, over the independent units given."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1:
        return (float(array[0]), float(array[0]))
    index = rng.integers(0, array.size, size=(draws, array.size))
    means = array[index].mean(axis=1)
    return (float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1 - alpha / 2.0)))


def bradley_terry(results: dict[tuple[str, str], tuple[int, int]],
                  *, iterations: int = 500, tolerance: float = 1e-10) -> dict[str, float]:
    """Bradley-Terry strengths from pairwise win counts (Hunter's MM iteration).

    ``results`` maps ``(name_i, name_j) -> (wins_i, wins_j)``; every pair should appear in
    both orders (or the counts are treated as symmetric information). Returns each name's
    ``pi`` (normalised to mean 1 over the connected field).
    """
    names: list[str] = []
    for left, right in results:
        for name in (left, right):
            if name not in names:
                names.append(name)
    if not names:
        return {}
    wins = {name: 0.0 for name in names}
    pair_counts: dict[tuple[str, str], int] = {}
    for (left, right), (win_left, win_right) in results.items():
        wins[left] += float(win_left)
        wins[right] += float(win_right)
        pair_counts[(left, right)] = pair_counts.get((left, right), 0) + int(win_left) + int(win_right)
    strength = {name: 1.0 for name in names}
    for _ in range(iterations):
        updated = {}
        for name in names:
            denominator = 0.0
            for (left, right), total in pair_counts.items():
                if name not in (left, right) or total == 0:
                    continue
                other = right if name == left else left
                denominator += total / (strength[name] + strength[other])
            updated[name] = (wins[name] / denominator) if denominator > 0 else strength[name]
        scale = float(np.mean(list(updated.values()))) or 1.0
        updated = {name: value / scale for name, value in updated.items()}
        delta = max(abs(updated[name] - strength[name]) for name in names)
        strength = updated
        if delta < tolerance:
            break
    return strength


def elo_from_strength(strength: dict[str, float], *, anchor: float = 1500.0) -> dict[str, float]:
    """Convert Bradley-Terry strengths to Elo points (field mean at ``anchor``)."""
    if not strength:
        return {}
    logs = {name: math.log(max(value, 1e-12)) for name, value in strength.items()}
    centre = float(np.mean(list(logs.values())))
    factor = 400.0 / math.log(10.0)
    return {name: anchor + factor * (value - centre) for name, value in logs.items()}


# ---------------------------------------------------------------------------
# matchups and round robin
# ---------------------------------------------------------------------------
def evaluate_matchup(spec_a: AgentSpec, spec_b: AgentSpec, *, deals: int, seed: int,
                     workers: int = 1) -> dict:
    """Play ``deals`` duplicate pairs between two agents and summarise them."""
    started = time.perf_counter()
    timings: Optional[list[float]] = [] if _times_agents(spec_a, spec_b) else None
    seeds = [int(seed) * 1_000_003 + index for index in range(int(deals))]
    if workers > 1:
        with mp.get_context("spawn").Pool(processes=int(workers)) as pool:
            pairs = pool.starmap(_worker_pairs, [(spec_a, spec_b, chunk) for chunk in
                                                 _chunks(seeds, workers)])
        rows = [row for chunk in pairs for row in chunk]
    else:
        rows = [play_duplicate_pair(spec_a, spec_b, value, timings) for value in seeds]
    return _summarise(spec_a, spec_b, rows, timings, seed=seed, workers=workers,
                      elapsed=time.perf_counter() - started)


def _times_agents(spec_a: AgentSpec, spec_b: AgentSpec) -> bool:
    """Only time the decisions of the search agents (the others are microseconds)."""
    return "pimc" in (spec_a.kind, spec_b.kind)


def _worker_pairs(spec_a: AgentSpec, spec_b: AgentSpec, seeds: list[int]) -> list[dict]:
    """Worker body: play a chunk of duplicate pairs (module level, so spawn can pickle it)."""
    return [play_duplicate_pair(spec_a, spec_b, value, None) for value in seeds]


def _chunks(values: list[int], count: int) -> list[list[int]]:
    """Split ``values`` into at most ``count`` contiguous chunks (one per worker)."""
    size = max(1, math.ceil(len(values) / max(1, int(count))))
    return [values[index:index + size] for index in range(0, len(values), size)]


def _summarise(spec_a: AgentSpec, spec_b: AgentSpec, rows: list[dict],
               timings: Optional[list[float]], *, seed: int, workers: int,
               elapsed: float) -> dict:
    """Aggregate the duplicate-pair rows of one matchup."""
    if not rows:
        raise ValueError("no deals were played")
    games = 2 * len(rows)
    wins_a = sum(1 for row in rows if row["won_a_game1"]) + \
        sum(1 for row in rows if row["won_a_game2"])
    wins_b = games - wins_a
    paired = [0.5 * (row["differential_game1"] + row["differential_game2"]) for row in rows]
    diffs = [row["differential_game1"] for row in rows] + \
        [row["differential_game2"] for row in rows]
    points_a = sum(row["points_game1"] if row["won_a_game1"] else 0 for row in rows) + \
        sum(row["points_game2"] if row["won_a_game2"] else 0 for row in rows)
    points_b = sum(row["points_game1"] if not row["won_a_game1"] else 0 for row in rows) + \
        sum(row["points_game2"] if not row["won_a_game2"] else 0 for row in rows)
    low, high = wilson_interval(wins_a, games)
    boot_low, boot_high = bootstrap_mean_ci(paired, np.random.default_rng(seed + 7))
    paired_wins = sum(1 for value in paired if value > 0)
    pips = ([row["pips_left_a_game1"] for row in rows] +
            [row["pips_left_a_game2"] for row in rows])
    return {
        "matchup": f"{spec_a.name}_vs_{spec_b.name}",
        "agent_a": spec_a.name,
        "agent_b": spec_b.name,
        "deals": len(rows),
        "games": games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "win_rate_a": wins_a / games,
        "wilson_low": low,
        "wilson_high": high,
        "paired_differential": float(np.mean(paired)),
        "boot_low": boot_low,
        "boot_high": boot_high,
        "points_per_game_a": points_a / games,
        "points_per_game_b": points_b / games,
        "paired_win_rate": paired_wins / len(rows),
        "block_rate": sum(row["blocked"] for row in rows) / games,
        "void_rate": sum(row["voids"] for row in rows) / games,
        "mean_pips_left_a": float(np.mean(pips)),
        "ms_per_action": 1000.0 * float(np.mean(timings)) if timings else float("nan"),
        "workers": int(workers),
        "seed": int(seed),
        "seconds": float(elapsed),
        "game_differentials": diffs,
    }


def _rows_of(matchup: dict) -> dict:
    """The CSV-safe view of a matchup row (drops the raw list)."""
    return {key: matchup[key] for key in CSV_FIELDS if key in matchup}


def append_row(path: str | pathlib.Path, row: dict) -> pathlib.Path:
    """Append one matchup row to the CSV, writing the header if the file is new."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    new = not target.exists()
    payload = _rows_of(row)
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        if new:
            writer.writeheader()
        writer.writerow({key: payload.get(key, "") for key in CSV_FIELDS})
    return target


def default_workers() -> int:
    """Cores to use: all but two, so the laptop stays usable, and never zero."""
    return max(1, (os.cpu_count() or 2) - 2)


def round_robin(specs: Sequence[AgentSpec], *, deals: int, seed: int,
                workers: Optional[int] = None,
                out_csv: Optional[str | pathlib.Path] = None) -> list[dict]:
    """Play every ordered pair once; each row is appended to ``out_csv`` immediately."""
    workers = default_workers() if workers is None else max(1, int(workers))
    rows: list[dict] = []
    for index, spec_a in enumerate(specs):
        for spec_b in specs[index + 1:]:
            if spec_a.name == spec_b.name:
                continue
            row = evaluate_matchup(spec_a, spec_b, deals=deals, seed=seed, workers=workers)
            rows.append(row)
            if out_csv is not None:
                append_row(out_csv, row)
    return rows


def elo_table(rows: Sequence[dict]) -> dict[str, float]:
    """Bradley-Terry strengths and Elo points from matchup rows."""
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    for row in rows:
        key = (row["agent_a"], row["agent_b"])
        wins = counts.get(key, (0, 0))
        counts[key] = (wins[0] + int(row["wins_a"]), wins[1] + int(row["wins_b"]))
    strength = bradley_terry(counts)
    return elo_from_strength(strength)


def estimate_runtime(specs: Sequence[AgentSpec], *, deals: int, seed: int,
                     workers: Optional[int] = None, pilot_deals: int = 1) -> str:
    """Time a one-deal pilot per matchup and extrapolate to the full run."""
    workers = default_workers() if workers is None else max(1, int(workers))
    pairs = [(a, b) for index, a in enumerate(specs) for b in specs[index + 1:]
             if a.name != b.name]
    started = time.perf_counter()
    pilot = 0.0
    for spec_a, spec_b in pairs[:2]:
        mark = time.perf_counter()
        play_duplicate_pair(spec_a, spec_b, seed)
        pilot += time.perf_counter() - mark
    pilot /= max(1, min(2, len(pairs)))
    total = pilot * deals * len(pairs) / workers
    return (f"estimate: {len(pairs)} matchup(s) x {deals} deal(s) at ~{pilot:.2f}s/deal "
            f"single-process, {workers} worker(s) -> ~{total / 60:.1f} min "
            f"(pilot took {time.perf_counter() - started:.2f}s)")
