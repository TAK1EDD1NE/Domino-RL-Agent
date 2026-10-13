"""A perfect-information solver agent (oracle) for the domino engine.

:class:`SolverAgent` plays the move the exact solver prefers. It is **not** a legal
opponent for training or for a strength claim: it sees all four hands, which no real
player ever does. It exists for two jobs:

* verifying the solver against the environment (drive all four seats with it and the
  deal must pay exactly the value the solver reported), and
* providing the information-unlimited reference line for the Phase-4 judge and the
  Phase-5 harness ("how much did the visible-information agent give up against a
  player who knows everything?").

It implements the same ``Agent`` protocol as the other agents (``act(state, seat)``
returning a 56-dim action index), so it can be seated anywhere an agent can —
including ``DominoEnv(opponent=SolverAgent(...))``.
"""
from __future__ import annotations

from typing import Optional

from ..engine import GameState
from .solver import Solver


class SolverAgent:
    """Plays the exact solver's best move for the seat to act (omniscient oracle)."""

    name = "solver"

    def __init__(self, *, max_depth: Optional[int] = None, leaf_eval=None,
                 tt_bits: int = 20, node_limit: Optional[int] = None,
                 root_pruning: bool = True, seed: Optional[int] = None,
                 name: Optional[str] = None) -> None:
        """Build the oracle.

        Parameters mirror :class:`~domino.pimc.solver.Solver`. ``root_pruning``
        defaults to ``True`` here: play only needs the best move, so sharing one
        alpha-beta window across the root moves is strictly faster and still exact
        for that move. ``seed`` is accepted and unused (the solver is
        deterministic); it keeps the constructor signature uniform with the other
        agents.
        """
        self.solver = Solver(max_depth=max_depth, leaf_eval=leaf_eval, tt_bits=tt_bits,
                             node_limit=node_limit, root_pruning=root_pruning)
        self.seed = seed
        if name is not None:
            self.name = name
        self.last_result = None

    def act(self, state: GameState, seat: int) -> int:
        """Return the best legal action index for ``seat`` (the seat to move)."""
        if int(seat) != int(state.turn):
            raise ValueError(
                f"SolverAgent only acts for the seat to move ({state.turn}), got {seat}"
            )
        if not state.legal_moves(int(seat)):
            raise ValueError(
                f"SolverAgent.act called for seat {seat} with no legal move (forced pass)"
            )
        result = self.solver.solve(state)
        self.last_result = result
        if result.best_action is None:                      # pragma: no cover - defensive
            raise ValueError("solver returned no move for a position with legal moves")
        return int(result.best_action)
