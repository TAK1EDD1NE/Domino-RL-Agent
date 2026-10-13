"""Search and analysis stack for 2v2 domino (phases 1-5 of the search brief).

This package is **additive**: nothing under ``domino/`` outside this directory was
changed to build it, so the training path, the environment and the existing
``domino/search.py`` PIMC hook (``test.py --search``) keep working untouched.

Modules
-------
``rules``
    Numba-compatible bitmask rules core, mirrored against ``domino.engine``.
``solver``
    Phase 1: exact/heuristic alpha-beta solver (compiled) plus the plain-minimax
    reference with a pluggable ``leaf_eval`` callable.
``agent``
    A perfect-information ``SolverAgent`` — an oracle used to verify the solver
    against the environment and as the information-unlimited reference line for the
    judge (Phase 4) and the evaluation harness (Phase 5).

Everything in this package is deterministic: randomness is only ever drawn from an
explicitly passed ``np.random.Generator`` (the solver itself draws none at all).
"""
from __future__ import annotations

from .agent import SolverAgent
from .solver import LEAF_MODES, LeafView, SolveResult, Solver, best_action, reference_solve

__all__ = [
    "LEAF_MODES",
    "LeafView",
    "SolveResult",
    "Solver",
    "SolverAgent",
    "best_action",
    "reference_solve",
]
