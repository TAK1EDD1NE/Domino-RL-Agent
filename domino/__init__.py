"""2v2 Algerian domino environment and deep-RL agents (RLOC 2026/2027 project).

This package is deliberately split into a pure-Python rules engine (``domino.tiles``,
later ``domino.engine``/``domino.env``) with no gymnasium/torch/pygame dependency at
the rules layer, and the learning code that builds on top of it.
"""
from __future__ import annotations

from .engine import GameState, Match

__version__ = "0.1.0"

__all__ = ["__version__", "GameState", "Match"]
