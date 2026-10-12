"""The single interface every playing agent implements.

A policy is anything that, given a :class:`~domino.engine.GameState` and the
seat about to act, returns an **action index in the 56-dimensional space**
(``2 * tile_id + end_index``, spec section 5). Agents are consulted only at real
decision points: the environment resolves forced passes itself, so an agent is
never asked for a move it cannot make.

Contract: :meth:`act` must always return a legal action (one whose index is set
in :func:`domino.masks.action_mask`). Implementations raise ``ValueError`` if
they are ever called with no legal move, which signalises an environment bug
rather than a legitimate pass.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..engine import GameState


@runtime_checkable
class Agent(Protocol):
    """Structural type for an acting policy."""

    name: str

    def act(self, state: GameState, seat: int) -> int:
        """Return a legal action index for ``seat`` in ``state``."""
        ...
