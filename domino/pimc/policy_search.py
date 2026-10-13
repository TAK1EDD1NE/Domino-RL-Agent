"""Engine-level search where the opposing seats follow a supplied policy.

The compiled solver assumes *every* seat minimises the acting team's value. That is the
pessimistic (and safest) opponent model, but not the only one worth measuring: against a
rule bot or a network, the true opponent plays a *policy*, and pricing a move against that
policy is a different question. This module answers it, at engine level:

* the acting team (the root seat and its partner) plays minimax over all legal moves;
* the opposing seats play exactly the move ``opponent_policy(state, seat)`` returns —
  which is the same ``Agent`` interface used everywhere else, so a ``RuleAgent``,
  ``PolicyAgent`` or any other agent can be plugged in unchanged;
* the deal's value comes from the engine's own rules (``deal_outcome``), and when the ply
  budget runs out the same leaf estimate as the compiled solver is used.

It is Python recursion over cloned ``GameState`` objects, so it is orders of magnitude
slower than the compiled path: it exists to *model a policy opponent*, and its cost is
reported rather than hidden (see the Phase-3 numbers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..engine import GameState, TEAM_OF_SEAT, deal_outcome
from ..masks import action_index, decode_action
from ..tiles import tile_pips
from . import rules as R
from .solver import LEAF_MODES, EXACT_DEPTH


@dataclass
class PolicySearchResult:
    """Value of a position when the opponents follow a policy (team-0 perspective)."""

    value: int
    move_values: dict[tuple[int, int], int] = field(default_factory=dict)
    best_move: Optional[tuple[int, int]] = None
    best_action: Optional[int] = None
    nodes: int = 0
    depth: Optional[int] = None
    opponent_team: int = -1


def _outcome_value(state: GameState) -> int:
    """Team-0 differential of a finished deal (0 for a void deal)."""
    outcome = deal_outcome(state)
    if outcome.winner_team == 0:
        return int(outcome.points)
    if outcome.winner_team == 1:
        return -int(outcome.points)
    return 0


def _leaf_of(state: GameState, leaf_mode: int) -> int:
    """The compiled leaf estimate, read off a ``GameState``."""
    pips = np.array([sum(tile_pips(t) for t in hand) for hand in state.hands], dtype=np.int32)
    return int(R.leaf_value(pips, leaf_mode))


def policy_solve(state: GameState, *, opponent_policy: Callable,
                 max_depth: Optional[int] = None, leaf_eval="blocked_now",
                 team: Optional[int] = None) -> PolicySearchResult:
    """Value of ``state`` for ``team`` (default: the seat to move's team).

    Parameters
    ----------
    opponent_policy:
        ``(state, seat) -> action_index`` (the project's ``Agent`` interface) or
        ``(state, seat, moves) -> action_index``. It is consulted only for seats on the
        opposing team; every other seat is searched exhaustively.
    max_depth:
        Ply budget; ``None`` searches to the terminal (exact for the modelled policy).
    leaf_eval:
        Compiled leaf mode name used when the budget runs out.
    team:
        Team whose differential is returned (0/1). Defaults to the moving seat's team.
    """
    if state.is_terminal():
        raise ValueError("policy_solve called on a terminal position")
    if isinstance(leaf_eval, str):
        if leaf_eval not in LEAF_MODES:
            raise ValueError(f"unknown leaf_eval {leaf_eval!r}")
        leaf_mode = LEAF_MODES[leaf_eval]
    else:                                                    # pragma: no cover - guard
        raise ValueError("policy_solve needs a compiled leaf mode name")
    root_seat = int(state.turn)
    root_team = int(TEAM_OF_SEAT[root_seat]) if team is None else int(team)
    opponent_team = 1 - root_team
    depth_budget = EXACT_DEPTH if max_depth is None else int(max_depth)
    counters = {"nodes": 0}

    def rec(node: GameState, depth: int) -> int:
        counters["nodes"] += 1
        if node.is_terminal():
            return _outcome_value(node)
        if depth <= 0:
            return _leaf_of(node, leaf_mode)
        seat = int(node.turn)
        moves = [(int(t), int(e)) for t, e in node.legal_moves(seat)]
        if not moves:
            child = node.clone()
            child.pass_turn(seat)
            return rec(child, depth - 1)
        if int(TEAM_OF_SEAT[seat]) == opponent_team:
            action = _policy_action(opponent_policy, node, seat, moves)
            tid, end = _decode_checked(action, moves)
            child = node.clone()
            child.play(seat, tid, end)
            return rec(child, depth - 1)
        maximizing = int(TEAM_OF_SEAT[seat]) == root_team
        best: Optional[int] = None
        for tid, end in moves:
            child = node.clone()
            child.play(seat, tid, end)
            value = rec(child, depth - 1)
            if best is None:
                best = value
            elif maximizing:
                best = max(best, value)
            else:
                best = min(best, value)
        return int(best)

    root_moves = [(int(t), int(e)) for t, e in state.legal_moves(root_seat)]
    result = PolicySearchResult(value=0, opponent_team=opponent_team, depth=max_depth)
    if not root_moves:
        child = state.clone()
        child.pass_turn(root_seat)
        result.value = rec(child, depth_budget - 1)
        result.nodes = counters["nodes"]
        return result
    for tid, end in root_moves:
        child = state.clone()
        child.play(root_seat, tid, end)
        result.move_values[(tid, end)] = rec(child, depth_budget - 1)
    result.nodes = counters["nodes"]
    # The root belongs to the acting team, so it picks its own best move.
    best_move = max(result.move_values,
                    key=lambda m: (result.move_values[m] if root_team == 0
                                   else -result.move_values[m], -action_index(*m)))
    result.best_move = best_move
    result.best_action = action_index(*best_move)
    result.value = int(result.move_values[best_move])
    return result


def _policy_action(policy: Callable, state: GameState, seat: int,
                   moves: list[tuple[int, int]]) -> int:
    """Ask the policy for an action, tolerating both supported signatures."""
    try:
        return int(policy(state, seat, moves))
    except TypeError:
        return int(policy(state, seat))


def _decode_checked(action: int, moves: list[tuple[int, int]]) -> tuple[int, int]:
    """Decode a policy action and refuse anything outside the legal moves."""
    tid, end = decode_action(int(action))
    if (tid, end) not in moves:
        if len([m for m in moves if m[0] == tid]) == 1:
            return next(m for m in moves if m[0] == tid)     # engine-tolerant end index
        raise ValueError(f"opponent policy returned illegal action {action}")
    return tid, end
