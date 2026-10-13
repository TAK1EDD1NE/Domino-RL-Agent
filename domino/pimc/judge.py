"""Phase 4: the judge — grade every decision of a deal against sampled worlds.

For one decision the judge re-creates what the player *could* know: the public view at that
moment (its own hand, the table, the move history) and a set of worlds drawn from the belief
sampler. Every legal move is then priced in every one of those worlds by the Phase-1 solver,
using the **same worlds for all moves** so the comparison is paired, and the player's actual
move is graded against the best alternative:

``regret``
    ``best_value - played_value`` in the moving team's own perspective (never negative).
    This is the *agent's view* of the decision: how much it gave up given what it could see.
``regret_true`` — hindsight mode
    The same computation with the **true hands** (no sampling), i.e. what a player who saw
    everything would have lost by playing this move. Comparing the two separates *judgement*
    (regret under its own information) from *information* (how much the hidden hands cost).
``luck``
    Per deal: the actual point differential minus the sampled value of the position at the
    deal's first decision. That is the part of the result that came from the cards and the
    opponents rather than from the player's choices, so skill and luck are not conflated.
``se``
    Standard error of the regret across worlds, from the *paired* per-world differences
    (``best_w - played_w``), which is the correct pairing and much tighter than two
    independent means.
``dominance``
    True when the played move was worse than the best alternative in at least
    ``dominance_threshold`` of the sampled worlds — a move that is bad in almost every world
    the player thought possible is a blunder of judgement, not of the draw.

Aggregates: mean regret per decision, blunder rate (regret >= 10 points; "major" >= 20),
regret by deal phase (opening / mid / endgame) and by situation (lead / follow / forced
pass). The report writes a CSV and a short text summary.
"""
from __future__ import annotations

import csv
import pathlib
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from ..engine import DealOutcome, GameState, TEAM_OF_SEAT, deal_outcome, new_deal, redeal_after_void
from ..masks import action_index, decode_action
from .player import _materialise
from .sampler import PublicView, World, sample_worlds
from .solver import LEAF_MODES, Solver

#: Regret at or above which a decision is called a blunder (points).
BLUNDER_THRESHOLD: int = 10
#: Regret at or above which a decision is called a major blunder (points).
MAJOR_BLUNDER_THRESHOLD: int = 20
#: Fraction of sampled worlds in which the played move must be strictly worse to be dominated.
DOMINANCE_THRESHOLD: float = 0.9
#: Hand-tile totals bounding the deal phases (opening > MID, mid >= END).
PHASE_OPENING_MIN: int = 18
PHASE_MID_MIN: int = 9


@dataclass
class DecisionRecord:
    """One decision to be graded: the position, the seat and what it did."""

    state: GameState
    seat: int
    action: Optional[int]                 # None for a forced pass
    previous_event: str                   # "none" | "play" | "pass"
    ply: int


@dataclass
class DealTrace:
    """A played deal plus the decisions to judge."""

    decisions: list[DecisionRecord]
    outcome: DealOutcome
    final_state: GameState
    deal_seed: int = 0

    @property
    def differential(self) -> int:
        """Team-0 differential the deal actually paid (0 for a void deal)."""
        if self.outcome.winner_team == 0:
            return int(self.outcome.points)
        if self.outcome.winner_team == 1:
            return -int(self.outcome.points)
        return 0


@dataclass
class JudgedDecision:
    """The grading of one decision."""

    deal_index: int
    ply: int
    seat: int
    team: int
    phase: str
    situation: str
    played_move: Optional[tuple[int, int]]
    best_move: Optional[tuple[int, int]]
    n_moves: int
    worlds: int
    values: dict[tuple[int, int], float] = field(default_factory=dict)
    values_true: dict[tuple[int, int], float] = field(default_factory=dict)
    regret: float = 0.0
    regret_true: float = 0.0
    se: float = 0.0
    dominance: bool = False

    @property
    def blunder(self) -> bool:
        """True when the decision gave up at least :data:`BLUNDER_THRESHOLD` points."""
        return self.regret >= BLUNDER_THRESHOLD

    @property
    def major_blunder(self) -> bool:
        """True when the decision gave up at least :data:`MAJOR_BLUNDER_THRESHOLD` points."""
        return self.regret >= MAJOR_BLUNDER_THRESHOLD

    def as_row(self) -> dict:
        """Flat row for the CSV."""
        def name(move: Optional[tuple[int, int]]) -> str:
            return "" if move is None else f"t{move[0]}e{move[1]}"

        return {
            "deal": self.deal_index,
            "ply": self.ply,
            "seat": self.seat,
            "team": self.team,
            "phase": self.phase,
            "situation": self.situation,
            "n_moves": self.n_moves,
            "worlds": self.worlds,
            "played": name(self.played_move),
            "best": name(self.best_move),
            "value_played": round(
                self.values.get(self.played_move, float("nan")), 4) if self.played_move else "",
            "value_best": round(self.values.get(self.best_move, float("nan")), 4)
            if self.best_move else "",
            "regret": round(self.regret, 4),
            "regret_true": round(self.regret_true, 4),
            "se": round(self.se, 4),
            "dominance": int(self.dominance),
            "blunder": int(self.blunder),
            "major_blunder": int(self.major_blunder),
        }


@dataclass
class JudgeReport:
    """Every graded decision plus the aggregates the brief asks for."""

    decisions: list[JudgedDecision] = field(default_factory=list)
    deals: int = 0
    luck: list[int] = field(default_factory=list)
    label: str = "player"

    # ---- aggregates -----------------------------------------------------
    @property
    def graded(self) -> list[JudgedDecision]:
        """Decisions that involved a choice (forced passes excluded)."""
        return [d for d in self.decisions if d.situation != "forced_pass"]

    def mean_regret(self) -> float:
        """Mean regret per decision (points), forced passes excluded."""
        graded = self.graded
        return float(np.mean([d.regret for d in graded])) if graded else float("nan")

    def mean_regret_true(self) -> float:
        """Mean regret per decision with the true hands (hindsight), passes excluded."""
        graded = self.graded
        return float(np.mean([d.regret_true for d in graded])) if graded else float("nan")

    def blunder_rate(self) -> float:
        """Share of graded decisions that gave up >= BLUNDER_THRESHOLD points."""
        graded = self.graded
        return float(np.mean([d.blunder for d in graded])) if graded else float("nan")

    def major_blunder_rate(self) -> float:
        """Share of graded decisions that gave up >= MAJOR_BLUNDER_THRESHOLD points."""
        graded = self.graded
        return float(np.mean([d.major_blunder for d in graded])) if graded else float("nan")

    def regret_by(self, key: str) -> dict[str, float]:
        """Mean regret grouped by ``phase`` or ``situation``."""
        buckets: dict[str, list[float]] = defaultdict(list)
        for decision in self.graded:
            buckets[getattr(decision, key)].append(decision.regret)
        return {name: float(np.mean(values)) for name, values in sorted(buckets.items())}

    def regret_se(self) -> float:
        """Standard error of the mean regret across decisions."""
        graded = self.graded
        if len(graded) < 2:
            return float("nan")
        return float(np.std([d.regret for d in graded], ddof=1) / np.sqrt(len(graded)))

    def mean_luck(self) -> float:
        """Mean per-deal luck (actual differential minus the opening belief)."""
        return float(np.mean(self.luck)) if self.luck else float("nan")

    # ---- output ---------------------------------------------------------
    def summary(self) -> str:
        """Short text summary of the report."""
        graded = self.graded
        passes = len(self.decisions) - len(graded)
        lines = [
            f"judge[{self.label}]: {self.deals} deal(s), {len(self.decisions)} decision(s) "
            f"({len(graded)} graded, {passes} forced pass(es))",
            f"  mean regret {self.mean_regret():.2f} +/- {self.regret_se():.2f} points  "
            f"(hindsight {self.mean_regret_true():.2f})",
            f"  blunder rate {self.blunder_rate():.1%} (>= {BLUNDER_THRESHOLD} pts), "
            f"major {self.major_blunder_rate():.1%} (>= {MAJOR_BLUNDER_THRESHOLD} pts), "
            f"dominated {sum(1 for d in graded if d.dominance)}",
            f"  regret by phase {self.regret_by('phase')}",
            f"  regret by situation {self.regret_by('situation')}",
            f"  mean luck {self.mean_luck():.2f} points/deal",
        ]
        return "\n".join(lines)

    def write_csv(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write one row per graded decision."""
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = [decision.as_row() for decision in self.decisions]
        if not rows:
            target.write_text("", encoding="utf-8")
            return target
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return target


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------
def _ask(agent, state: GameState, seat: int) -> int:
    """Ask an agent for its action, accepting either an ``Agent`` or a bare callable."""
    act = getattr(agent, "act", None)
    if act is not None:
        return int(act(state, seat))
    return int(agent(state, seat))


def record_deal(agents: dict[int, Callable[[GameState, int], int]],
                rng: np.random.Generator, *, eval_seats: Sequence[int] = (0, 2),
                deal_no: int = 1, seed: int = 0) -> DealTrace:
    """Play one fresh deal (deal-1 rules) and capture the decisions of ``eval_seats``.

    Forced passes are recorded too (as decisions with no action), because the brief asks for
    regret broken down *by situation*, and "was stuck" is one of those situations. Void deals
    are redealt in place, exactly as the environment does, so the trace always ends in a
    scored deal.
    """
    state = new_deal(deal_no=deal_no, rng=rng, lead_mode="six_six")
    while True:
        decisions: list[DecisionRecord] = []
        ply = 0
        previous_event = "none"
        while not state.is_terminal():
            seat = int(state.turn)
            moves = [(int(t), int(e)) for t, e in state.legal_moves(seat)]
            if not moves:
                if seat in eval_seats:
                    decisions.append(DecisionRecord(state.clone(), seat, None, previous_event, ply))
                state.pass_turn(seat)
                previous_event = "pass"
                ply += 1
                continue
            action = int(_ask(agents[seat], state, seat))
            tid, end = decode_action(action)
            if seat in eval_seats:
                decisions.append(DecisionRecord(state.clone(), seat, action, previous_event, ply))
            state.play(seat, tid, end)
            previous_event = "play"
            ply += 1
        outcome = deal_outcome(state)
        if not outcome.void:
            return DealTrace(decisions=decisions, outcome=outcome, final_state=state.clone(),
                             deal_seed=int(seed))
        state = redeal_after_void(state)


def _phase_of(state: GameState) -> str:
    """Deal phase from the tiles still in hand."""
    total = sum(len(hand) for hand in state.hands)
    if total > PHASE_OPENING_MIN:
        return "opening"
    if total >= PHASE_MID_MIN:
        return "mid"
    return "endgame"


def _situation_of(record: DecisionRecord) -> str:
    """Situation of a decision: forced pass, lead (first play of a round) or follow."""
    if record.action is None:
        return "forced_pass"
    return "lead" if record.previous_event in ("none", "pass") else "follow"


def _outcome_value(state: GameState) -> int:
    """Team-0 differential of a terminal deal (0 for a void deal)."""
    outcome = deal_outcome(state)
    if outcome.winner_team == 0:
        return int(outcome.points)
    if outcome.winner_team == 1:
        return -int(outcome.points)
    return 0


def _child_value(solver: Solver, world_state: GameState, seat: int,
                 move: tuple[int, int]) -> int:
    """Value of the move in ``world_state`` (team-0 perspective)."""
    child = world_state.clone()
    child.play(int(seat), int(move[0]), int(move[1]))
    if child.is_terminal():
        return _outcome_value(child)
    result = solver.solve(child)
    return int(result.value) if result.complete else 0


def judge_game(trace: DealTrace, *, worlds: int = 32, horizon: int = 10,
               leaf_eval="blocked_now", rng: Optional[np.random.Generator] = None,
               hindsight: bool = True, tt_bits: int = 18, deal_index: int = 0,
               label: str = "player") -> JudgeReport:
    """Grade every recorded decision of one deal."""
    if rng is None:
        raise ValueError("judge_game needs an explicit np.random.Generator")
    if isinstance(leaf_eval, str) and leaf_eval not in LEAF_MODES:
        raise ValueError(f"unknown leaf_eval {leaf_eval!r}")
    solver = Solver(max_depth=None if horizon is None else int(horizon),
                    leaf_eval=leaf_eval, tt_bits=tt_bits)
    report = JudgeReport(deals=1, label=label)
    opening_belief: Optional[float] = None
    for record in trace.decisions:
        seat = int(record.seat)
        team = int(TEAM_OF_SEAT[seat])
        situation = _situation_of(record)
        moves = [(int(t), int(e)) for t, e in record.state.legal_moves(seat)]
        decision = JudgedDecision(
            deal_index=deal_index, ply=record.ply, seat=seat, team=team,
            phase=_phase_of(record.state), situation=situation,
            played_move=None if record.action is None else decode_action(record.action),
            best_move=None, n_moves=len(moves), worlds=0,
        )
        if not moves:                                        # forced pass
            pass_child = record.state.clone()
            pass_child.pass_turn(seat)
            value = _outcome_value(pass_child) if pass_child.is_terminal() else \
                float(solver.solve(pass_child).value)
            decision.values = {}
            decision.regret = 0.0
            decision.regret_true = 0.0
            decision.best_move = None
            report.decisions.append(decision)
            if opening_belief is None:
                opening_belief = float(value)
            continue
        view = PublicView.from_state(record.state, seat)
        world_list: list[World] = sample_worlds(view, int(worlds), rng, method="auto")
        per_world: dict[tuple[int, int], list[int]] = {move: [] for move in moves}
        true_values: dict[tuple[int, int], int] = {}
        if hindsight:
            for move in moves:
                true_values[move] = _child_value(solver, record.state, seat, move)
        for world in world_list:
            world_state = _materialise(record.state, world)
            for move in moves:
                per_world[move].append(_child_value(solver, world_state, seat, move))
        sign = 1 if team == 0 else -1
        own = {move: [sign * value for value in per_world[move]] for move in moves}
        means = {move: float(np.mean(values)) for move, values in own.items()}
        decision.values = means
        decision.worlds = len(world_list)
        decision.best_move = max(moves, key=lambda m: (means[m], -action_index(*m)))
        played = decision.played_move
        assert played is not None                            # a graded decision played a move
        decision.regret = float(means[decision.best_move] - means[played])
        paired_diffs = [max(own[m][i] for m in moves) - own[played][i]
                        for i in range(len(world_list))]
        decision.se = float(np.std(paired_diffs, ddof=1) / np.sqrt(len(paired_diffs))) \
            if len(paired_diffs) > 1 else float("nan")
        decision.dominance = bool(
            np.mean([1.0 if diff > 0 else 0.0 for diff in paired_diffs]) >= DOMINANCE_THRESHOLD)
        if hindsight:
            own_true = {move: sign * value for move, value in true_values.items()}
            best_true = max(own_true, key=lambda m: (own_true[m], -action_index(*m)))
            decision.values_true = {move: float(value) for move, value in own_true.items()}
            decision.regret_true = float(own_true[best_true] - own_true[played])
        if opening_belief is None:
            opening_belief = float(means[played])
        report.decisions.append(decision)
    report.luck.append(int(trace.differential - round(opening_belief or 0.0)))
    return report


def judge_deals(agents: dict[int, Callable[[GameState, int], int]],
                rng: np.random.Generator, *, deals: int = 10, worlds: int = 32,
                horizon: int = 10, leaf_eval="blocked_now", eval_seats: Sequence[int] = (0, 2),
                hindsight: bool = True, label: str = "player") -> JudgeReport:
    """Play ``deals`` fresh deals with ``agents`` and grade every decision of ``eval_seats``."""
    total = JudgeReport(label=label)
    for index in range(int(deals)):
        trace = record_deal(agents, rng, eval_seats=eval_seats, deal_no=index + 1)
        partial = judge_game(trace, worlds=worlds, horizon=horizon, leaf_eval=leaf_eval,
                             rng=rng, hindsight=hindsight, deal_index=index, label=label)
        total.decisions.extend(partial.decisions)
        total.luck.extend(partial.luck)
        total.deals += partial.deals
    return total
