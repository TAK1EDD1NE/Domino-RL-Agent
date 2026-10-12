"""Uniform random agent over the legal moves (spec section 6).

The cheapest league opponent and the sanity floor of the evaluation matrix: it
punishes degenerate policies that only exploit a specific line of play. All
randomness comes from an explicit :class:`numpy.random.Generator`, so a seeded
agent is fully deterministic. numpy and the engine only — no torch, no sb3,
no pygame.
"""
from __future__ import annotations

import numpy as np

from ..engine import GameState
from ..masks import action_index


class RandomAgent:
    """Plays a uniformly random legal placement."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, state: GameState, seat: int) -> int:
        """Return a uniformly drawn legal action index for ``seat``."""
        moves = state.legal_moves(seat)
        if not moves:
            raise ValueError(f"RandomAgent.act called for seat {seat} with no legal move (forced pass)")
        tid, end = moves[int(self._rng.integers(len(moves)))]
        return action_index(tid, end)
