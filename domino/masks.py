"""Action masking over the ``Discrete(56)`` move space (spec section 5).

A move is a ``(tile_id, end_index)`` pair encoded as ``action = 2 * tile_id +
end_index`` with ``end_index`` ``0 = left`` and ``1 = right``. The mask is built
*from* :meth:`domino.engine.GameState.legal_moves`, never re-derived from tile
values, so a masked index is guaranteed to be a move the engine will accept —
masking and application can never disagree.

There is no pass action: whether the acting seat can play at all is decided by
the engine (a stuck seat is a forced pass and never a decision point).
"""
from __future__ import annotations

import numpy as np

from .engine import GameState
from .tiles import NUM_TILES

# Public interface -----------------------------------------------------------
ACTION_DIM: int = 2 * NUM_TILES  # Discrete(56): 2 * 28 tile/end combinations


def action_index(tid: int, end: int) -> int:
    """Return the action index for the move ``(tid, end)``: ``2 * tid + end``."""
    return 2 * tid + end


def decode_action(action: int) -> tuple[int, int]:
    """Return the ``(tile_id, end_index)`` for ``action``.

    Raises ``ValueError`` when ``action`` is not in ``0..ACTION_DIM-1``.
    """
    if not 0 <= action < ACTION_DIM:
        raise ValueError(f"action {action} is out of range 0..{ACTION_DIM - 1}")
    return divmod(action, 2)


def action_mask(state: GameState, seat: int) -> np.ndarray:
    """Return the ``np.bool_`` mask of length ``ACTION_DIM`` for ``seat``.

    ``True`` exactly at the indices of ``state.legal_moves(seat)``.
    """
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    for tid, end in state.legal_moves(seat):
        mask[action_index(tid, end)] = True
    return mask


def masked_actions(mask: np.ndarray) -> np.ndarray:
    """Return the indices where ``mask`` is ``True`` (an ``int`` array)."""
    return np.flatnonzero(np.asarray(mask))


def assert_consistent(state: GameState, seat: int, action: int) -> None:
    """Raise ``ValueError`` unless ``action`` is legal for ``seat``.

    Used by the environment in development mode as a belt-and-braces check
    that the mask and the applied action agree.
    """
    if not 0 <= action < ACTION_DIM:
        raise ValueError(f"action {action} is out of range 0..{ACTION_DIM - 1}")
    if action not in {action_index(t, e) for t, e in state.legal_moves(seat)}:
        tid, end = decode_action(action)
        raise ValueError(
            f"action {action} (tile {tid}, end {end}) is not legal for seat {seat}; "
            f"legal moves are {state.legal_moves(seat)}"
        )
