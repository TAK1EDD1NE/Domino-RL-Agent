"""2v2 Algerian domino environment and deep-RL agents (RLOC 2026/2027 project).

This package is deliberately split into a pure-Python rules engine (``domino.tiles``,
later ``domino.engine``/``domino.env``) with no gymnasium/torch/pygame dependency at
the rules layer, and the learning code that builds on top of it.
"""
from __future__ import annotations

import gymnasium

from .engine import GameState, Match
from .env import DominoEnv

__version__ = "0.1.0"

gymnasium.register("Domino2v2-v0", entry_point="domino.env:DominoEnv")

__all__ = ["__version__", "DominoEnv", "GameState", "Match"]
