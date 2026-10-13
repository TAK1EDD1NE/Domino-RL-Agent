"""Phase-1 acceptance: the solver's value is exactly what the environment pays.

The environment is driven with the solver on **all four seats** — ``SolverAgent`` for the
opposing seats, the solver's own ``best_action`` for the learner's — so the deal is played
along the solver's optimal line and the terminal reward must equal the value the solver
reported at *every* decision point of that deal (an exact search has no horizon, so the
value of a position is the deal's outcome, not an estimate).

Deals containing a void redeal are counted and skipped for the value comparison: a void
deal pays nothing and the environment continues with a fresh deal, so the reward seen at
the end of the episode belongs to a later deal than the decisions being checked.

The deal count is read from ``PIMC_SOLVER_DEALS`` (default 30): the check is the expensive
one in this suite because it solves the opening position exactly, so the default is sized
from a measurement rather than a wish (see the Phase-1 report).
"""
from __future__ import annotations

import os

import numpy as np

from domino.env import DominoEnv
from domino.pimc.agent import SolverAgent
from domino.pimc.solver import Solver

#: Deals verified by the environment-facing check (cost grows with this, not linearly
#: only: the opening position is the most expensive one in every deal).
DEALS = int(os.environ.get("PIMC_SOLVER_DEALS", "30"))


def test_env_reward_equals_solver_value() -> None:
    """Reward at the end of the deal == the exact value at every decision point."""
    rng = np.random.default_rng(20261013)
    solver = Solver()
    oracle = SolverAgent(tt_bits=14)
    checked = 0
    skipped_void = 0
    decisions = 0
    for index in range(DEALS):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        env = DominoEnv(opponent=oracle, seed=seed)
        _observation, _info = env.reset(seed=seed)
        values: list[int] = []
        reward = 0.0
        terminated = False
        void_seen = False
        while not terminated:
            result = solver.solve(env.state)
            assert result.complete and result.exact
            assert result.leaf_evals == 0, "an exact solve must not bound with a leaf"
            assert result.seat == env.acting_seat
            values.append(result.value)
            _observation, reward, terminated, _truncated, info = env.step(int(result.best_action))
            if info.get("void_deal"):
                void_seen = True
        if void_seen:
            skipped_void += 1
            continue
        assert values, f"deal {index} produced no learner decision point"
        decisions += len(values)
        for value in values:
            assert value == int(reward), (
                f"deal {index} (seed {seed}): solver said {value}, env paid {int(reward)} "
                f"— decision-point values were {values}"
            )
        checked += 1
    assert checked + skipped_void == DEALS
    assert checked >= DEALS - 5, f"too many void deals ({skipped_void}) to trust this run"
    assert decisions > checked, "a deal should need more than one learner decision"


def test_env_pays_exactly_the_solver_value_on_a_seeded_deal() -> None:
    """The same check on one fixed seed, so a failure reproduces from the seed alone."""
    seed = 20261013
    solver = Solver()
    env = DominoEnv(opponent=SolverAgent(tt_bits=14), seed=seed)
    _observation, _info = env.reset(seed=seed)
    value = solver.solve(env.state).value
    terminated = False
    reward = 0.0
    while not terminated:
        result = solver.solve(env.state)
        _observation, reward, terminated, _truncated, info = env.step(int(result.best_action))
        if terminated and not info.get("void_deal"):
            assert int(reward) == value
