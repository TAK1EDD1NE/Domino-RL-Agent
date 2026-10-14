"""Phase-3 PIMC player: sample worlds, solve each of them, average per move.

The agent plays the move with the best *average* value over worlds drawn from the belief
sampler (Phase 2). Every move is scored on the **same** worlds — a paired comparison, so the
tile luck a move benefits from is present in the alternatives too — and each world is played
perfect-information by the Phase-1 solver with a ply horizon.

Strategy fusion — read this before trusting the numbers
------------------------------------------------------
Fixed-world PIMC assumes each sampled world *is* the true one, which lets a move be scored as
if the agent will learn later which world it is in. It cannot: a line that wins in world A and
a different line that wins in world B can both look good, and averaging them can beat every
achievable line. That is **strategy fusion**, this module does not solve it (the brief says
not to), and Phase 5 measures the resulting strength rather than assuming it away.

Opponent model inside the tree
-----------------------------
``opponent_model="min"`` (default) has the opposing seats minimise the acting team's value:
pessimistic, and exactly what the compiled solver does. ``"policy"`` instead lets the
opposing seats follow a supplied agent (the rule bot, a network, anything implementing
``act(state, seat)``) while the acting team stays on the minimax path; it runs on the
engine-level Python search of :mod:`.policy_search` and is much slower.

Time budget
-----------
``time_budget`` is a hard per-move wall-clock budget. The agent prices one world first, then
reduces the number of worlds — and, if a single world does not fit, the ply horizon — to stay
inside it, logging every reduction in ``last_report.adjustments``. The solver itself never
reads a clock (its node counts stay deterministic); only the world count responds to time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..engine import GameState, TEAM_OF_SEAT
from ..masks import action_index
from ..tiles import tile_pips
from .leaf import NET_LEAF_MODES, NetLeaf
from .policy_search import PolicySearchResult, policy_solve
from .sampler import PublicView, World, sample_worlds
from .solver import LEAF_MODES, Solver

#: Default number of sampled worlds per decision.
DEFAULT_WORLDS: int = 32
#: Default ply horizon of the inner search (the deal is solved exactly once it is short).
DEFAULT_HORIZON: int = 10
#: Default wall-clock budget per move, seconds.
DEFAULT_BUDGET: float = 1.0


@dataclass
class MoveReport:
    """Diagnostics of one :meth:`PIMCAgent.act` call (also kept in ``last_report``)."""

    seat: int = 0
    depth: Optional[int] = None
    worlds_target: int = 0
    worlds_used: int = 0
    elapsed_s: float = 0.0
    values: dict[tuple[int, int], float] = field(default_factory=dict)
    best_move: Optional[tuple[int, int]] = None
    best_action: Optional[int] = None
    adjustments: list[str] = field(default_factory=list)
    opponent_model: str = "min"

    @property
    def per_move(self) -> dict[tuple[int, int], float]:
        """Mean value per move over the worlds actually used."""
        if self.worlds_used <= 0:
            return {}
        return {move: value / self.worlds_used for move, value in self.values.items()}


class PIMCAgent:
    """Perfect-Information-Monte-Carlo player for one seat.

    Implements the project's ``Agent`` protocol (``act(state, seat) -> action``), so it can
    be seated in ``DominoEnv``, ``play.py`` or ``test.py`` like any other agent.
    """

    name = "pimc"

    def __init__(self, *, worlds: int = DEFAULT_WORLDS, horizon: int = DEFAULT_HORIZON,
                 time_budget: float = DEFAULT_BUDGET, leaf_eval="blocked_now",
                 method: str = "auto", tt_bits: int = 20, node_limit: Optional[int] = None,
                 seed: Optional[int] = 0, opponent_model: str = "min",
                 opponent_policy: Optional[Callable] = None, exact_when_short: bool = True,
                 model: Optional[object] = None, root_all_moves: bool = False,
                 log: Optional[Callable[[str], None]] = None,
                 sampler: Optional[Callable[[PublicView, int, np.random.Generator],
                                            list[World]]] = None,
                 name: Optional[str] = None) -> None:
        if worlds < 1:
            raise ValueError(f"worlds must be >= 1, got {worlds}")
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        if time_budget <= 0:
            raise ValueError(f"time_budget must be positive, got {time_budget}")
        if opponent_model not in ("min", "policy"):
            raise ValueError(f"unknown opponent_model {opponent_model!r}")
        if opponent_model == "policy" and opponent_policy is None:
            raise ValueError("opponent_model='policy' needs an opponent_policy")
        if isinstance(leaf_eval, str) and leaf_eval not in (*LEAF_MODES, *NET_LEAF_MODES):
            raise ValueError(f"unknown leaf_eval {leaf_eval!r}")
        if leaf_eval in NET_LEAF_MODES and model is None:
            raise ValueError(
                f"leaf_eval={leaf_eval!r} needs a trained model (pass model=<MaskablePPO>); "
                "the compiled backend cannot call a network from inside its kernel"
            )
        self.worlds = int(worlds)
        self.horizon = int(horizon)
        self.time_budget = float(time_budget)
        self.leaf_eval = leaf_eval
        self.method = str(method)
        self.tt_bits = int(tt_bits)
        self.node_limit = None if node_limit is None else int(node_limit)
        self.seed = seed
        #: ``rng`` and ``_rng`` are the same object: the environment reseeds an owned
        #: opponent by resetting ``agent._rng`` to ``default_rng(agent.seed)``.
        self.rng = np.random.default_rng(seed)
        self._rng = self.rng
        self.opponent_model = str(opponent_model)
        self.opponent_policy = opponent_policy
        self.exact_when_short = bool(exact_when_short)
        #: One exact solve per world for the whole root instead of one per move. Only changes
        #: the *cost* (and the magnitudes of non-best entries, which become pruning bounds):
        #: the argmax over moves is identical. This is what makes an exact teacher affordable
        #: for offline target generation.
        self.root_all_moves = bool(root_all_moves)
        #: The trained network behind ``leaf_eval in NET_LEAF_MODES``.
        self.model = model
        self._net_leaf: Optional[NetLeaf] = None
        self._log = log
        self._sampler = sampler
        self.last_report = MoveReport()
        if name is not None:
            self.name = name
        self._solver: Optional[Solver] = None
        self._solver_depth: Optional[int] = None
        self._solver_leaf: Optional[str] = None

    # ---- public API -----------------------------------------------------
    def act(self, state: GameState, seat: int) -> int:
        """Return the best action index for ``seat`` (the seat to move)."""
        if int(seat) != int(state.turn):
            raise ValueError(
                f"PIMCAgent only acts for the seat to move ({state.turn}), got {seat}"
            )
        moves = [(int(tid), int(end)) for tid, end in state.legal_moves(int(seat))]
        if not moves:
            raise ValueError(f"PIMCAgent.act called for seat {seat} with no legal move")
        moves.sort(key=lambda move: action_index(*move))
        report = MoveReport(seat=int(seat), worlds_target=self.worlds,
                            opponent_model=self.opponent_model)
        started = time.perf_counter()
        depth = self._depth_for(state)
        view = PublicView.from_state(state, int(seat))
        if self.leaf_eval in NET_LEAF_MODES and depth is not None:
            self._score_net_leaf(state, moves, view, report, started)
        elif self.opponent_model == "policy":
            self._score_policy(state, moves, view, depth, report, started)
        else:
            self._score_minimax(state, moves, view, depth, report, started)
        report.elapsed_s = time.perf_counter() - started
        # Pick the best move **for the moving team**: at a team-1 seat the value to maximise
        # is the negated team-0 differential (the same trap as in Solver._finish).
        team = int(TEAM_OF_SEAT[int(seat)])

        def own_value(move: tuple[int, int]) -> float:
            raw = report.per_move[move]
            return raw if team == 0 else -raw

        best = max(moves, key=lambda m: (own_value(m), -action_index(*m)))
        report.best_move = best
        report.best_action = action_index(*best)
        self.last_report = report
        return report.best_action

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed the agent's generator (the environment does this between episodes)."""
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._rng = self.rng

    # ---- scoring paths --------------------------------------------------
    def _score_minimax(self, state: GameState, moves: list[tuple[int, int]],
                       view: PublicView, depth: Optional[int], report: MoveReport,
                       started: float) -> None:
        """Score every move on the same worlds with the compiled pessimistic search."""
        seat = int(state.turn)
        left = int(state.ends[0]) if state.table else -1
        right = int(state.ends[1]) if state.table else -1
        pass_streak = int(state.pass_streak)
        solver = self._solver_for(depth)
        used_depth = depth
        self._score_one_world(solver, view, moves, seat, left, right, pass_streak, report)
        completed = 1
        per_world = time.perf_counter() - started
        target = self.worlds
        if target > 1:
            affordable = int(max(1, self.time_budget / max(per_world, 1e-9)))
            if affordable < target:
                self._log_line(
                    f"reduced worlds {target} -> {affordable} to fit "
                    f"{self.time_budget:.3f}s (one world costs {per_world:.4f}s)", report)
                target = affordable
        if per_world > self.time_budget and depth is not None and depth > 0:
            new_depth = max(0, depth // 2)
            self._log_line(
                f"reduced horizon {depth} -> {new_depth} (one world costs {per_world:.4f}s "
                f"> budget {self.time_budget:.3f}s)", report)
            report.values.clear()
            solver = self._solver_for(new_depth)
            used_depth = new_depth
            self._score_one_world(solver, view, moves, seat, left, right, pass_streak, report)
            completed = 1
        while completed < target:
            if time.perf_counter() - started > self.time_budget:
                self._log_line(
                    f"stopped after {completed} world(s) of {target} "
                    f"({time.perf_counter() - started:.3f}s > {self.time_budget:.3f}s)",
                    report)
                break
            self._score_one_world(solver, view, moves, seat, left, right, pass_streak, report)
            completed += 1
        report.worlds_used = completed
        report.depth = used_depth

    def _score_one_world(self, solver: Solver, view: PublicView,
                         moves: list[tuple[int, int]], seat: int, left: int, right: int,
                         pass_streak: int, report: MoveReport) -> dict[tuple[int, int], int]:
        """Add one world's per-move values into ``report.values`` (returns the raw values)."""
        world = self._draw(view, 1)[0]
        masks = world.masks()
        pips = _pips_of_masks(masks)
        raw: dict[tuple[int, int], int] = {}
        if self.root_all_moves and solver.max_depth is None:
            # One exact solve for the whole root instead of one per move: with an exact
            # (depth-free) search the table is shared and the answer is the same ranking, at
            # roughly 1/len(moves) of the cost. Non-best entries in ``move_values`` can be
            # pruning bounds rather than exact magnitudes, which is all an argmax target needs.
            result = solver.solve_arrays(masks, left, right, seat, pass_streak, pips, moves)
            for move in moves:
                value = int(result.move_values.get(move, 0)) if result.complete else 0
                raw[move] = value
                report.values[move] = report.values.get(move, 0.0) + float(value)
            return raw
        for move in moves:
            result = solver.solve_arrays(masks, left, right, seat, pass_streak, pips, [move])
            value = int(result.value) if result.complete else 0
            raw[move] = value
            report.values[move] = report.values.get(move, 0.0) + float(value)
        return raw

    def _score_net_leaf(self, state: GameState, moves: list[tuple[int, int]],
                        view: PublicView, report: MoveReport, started: float) -> None:
        """Score every move with the **network leaf** on the same worlds (depth 1).

        Values are accumulated team-0 relative, like the compiled path, so ``act`` converts
        them with the same sign logic. The worlds are drawn in two batches: a small probe to
        measure the per-world cost, then as many as the remaining budget affords - the same
        adapt-then-log discipline the compiled path uses.
        """
        leaf = self._net_leaf_for()
        seat = int(state.turn)
        target = self.worlds
        probe = min(target, 4)
        used = 0
        per_world = 0.0
        while used < target:
            if used > 0:
                elapsed = time.perf_counter() - started
                if elapsed > self.time_budget:
                    self._log_line(
                        f"stopped after {used} world(s) of {target} "
                        f"({elapsed:.3f}s > {self.time_budget:.3f}s)", report)
                    break
                affordable = int(max(1, self.time_budget / max(per_world, 1e-9)))
                batch = min(target - used, max(1, affordable))
            else:
                batch = probe
            mark = time.perf_counter()
            worlds = self._draw(view, batch)
            values = leaf.value_of_moves(state, seat, moves, worlds)
            for move, value in values.items():
                report.values[move] = report.values.get(move, 0.0) + float(value)
            used += batch
            per_world = (time.perf_counter() - started) / used
        report.worlds_used = used
        report.depth = 1

    def _net_leaf_for(self) -> NetLeaf:
        """The lazily built network leaf for this agent's ``leaf_eval`` mode."""
        if self._net_leaf is None:
            self._net_leaf = NetLeaf(self.model, mode=str(self.leaf_eval))
        return self._net_leaf

    def _score_policy(self, state: GameState, moves: list[tuple[int, int]],
                      view: PublicView, depth: Optional[int], report: MoveReport,
                      started: float) -> None:
        """Score the moves with the opponents following ``opponent_policy``."""
        team = int(TEAM_OF_SEAT[int(state.turn)])
        leaf = self.leaf_eval if isinstance(self.leaf_eval, str) else "blocked_now"
        completed = 0
        target = self.worlds
        while completed < target:
            if completed and time.perf_counter() - started > self.time_budget:
                self._log_line(f"stopped after {completed} world(s) of {target}", report)
                break
            world = self._draw(view, 1)[0]
            materialised = _materialise(state, world)
            for move in moves:
                child = materialised.clone()
                child.play(int(state.turn), int(move[0]), int(move[1]))
                result: PolicySearchResult = policy_solve(
                    child, opponent_policy=self.opponent_policy, max_depth=depth,
                    leaf_eval=leaf, team=team)
                report.values[move] = report.values.get(move, 0.0) + float(result.value)
            completed += 1
        report.worlds_used = max(completed, 1)
        report.depth = depth

    # ---- helpers --------------------------------------------------------
    def _log_line(self, message: str, report: MoveReport) -> None:
        """Record an adjustment and forward it to the caller's logger."""
        report.adjustments.append(message)
        if self._log is not None:
            self._log(message)

    def _draw(self, view: PublicView, count: int) -> list[World]:
        """Draw ``count`` worlds through the injected sampler (or the module default)."""
        if self._sampler is not None:
            return list(self._sampler(view, count, self.rng))
        return sample_worlds(view, count, self.rng, method=self.method)

    def _depth_for(self, state: GameState) -> Optional[int]:
        """Ply horizon for this position; ``None`` when the whole deal fits the horizon."""
        remaining_plays = sum(len(hand) for hand in state.hands)
        if self.exact_when_short and remaining_plays <= self.horizon:
            return None
        return self.horizon

    def _solver_for(self, depth: Optional[int]) -> Solver:
        """Cache the solver per depth (its table is worth keeping between calls).

        A *net* leaf mode picks the compiled path for the short-deal case: when ``depth`` is
        ``None`` the decompiled solve runs to the terminal payoffs and never consults a leaf at
        all, so the mode only has to be one the kernel accepts.
        """
        leaf = self.leaf_eval if self.leaf_eval in LEAF_MODES else "blocked_now"
        if self._solver is None or self._solver_depth != depth or self._solver_leaf != leaf:
            self._solver = Solver(max_depth=depth, leaf_eval=leaf,
                                  tt_bits=self.tt_bits, node_limit=self.node_limit)
            self._solver_depth = depth
            self._solver_leaf = leaf
        return self._solver


def _pips_of_masks(masks: np.ndarray) -> np.ndarray:
    """Pip totals per seat from hand masks."""
    pips = np.zeros(4, dtype=np.int32)
    for seat in range(4):
        mask = int(masks[seat])
        total = 0
        for tile in range(28):
            if (mask >> tile) & 1:
                total += tile_pips(tile)
        pips[seat] = int(total)
    return pips


def materialise_world(state: GameState, world: World) -> GameState:
    """A clone of ``state`` holding the world's hands.

    Only the hidden half is replaced: the table, the open ends, the move history and the
    played counts stay as the acting seat really saw them (a world is a hypothesis about the
    *hidden* hands, never about the public record).
    """
    clone = state.clone()
    for seat in range(4):
        clone.hands[seat] = sorted(int(t) for t in world.hand(seat))
    return clone


#: Backwards-compatible private alias (the judge imports it under this name).
_materialise = materialise_world
