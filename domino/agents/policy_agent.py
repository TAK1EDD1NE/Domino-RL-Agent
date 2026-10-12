"""Wrapper turning an sb3 ``MaskablePPO`` model into an :class:`Agent`.

The same class serves the current learner (``name="self"``) and a frozen
snapshot (``name="snapshot"``): it encodes the state, builds the action mask and
lets the model choose. Importing this module pulls in no torch or sb3 — the
model is supplied by the caller and the heavy imports live inside :meth:`act`,
so the rule and random bots stay importable in a lightweight process.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..engine import GameState

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from sb3_contrib import MaskablePPO


class PolicyAgent:
    """An agent driven by a (maskable) policy model."""

    def __init__(self, model: "MaskablePPO", deterministic: bool = True, name: str = "self") -> None:
        self.model = model
        self.deterministic = deterministic
        self.name = name

    def act(self, state: GameState, seat: int) -> int:
        """Encode ``state`` for ``seat`` and return the model's (masked) action."""
        from ..masks import action_mask
        from ..obs import encode

        obs = encode(state, seat)
        mask = action_mask(state, seat)
        if not mask.any():
            raise ValueError(f"PolicyAgent.act called for seat {seat} with no legal move (forced pass)")
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return int(action)
