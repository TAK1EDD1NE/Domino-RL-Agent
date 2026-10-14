"""Network value leaves for PIMC: the brief's leaf option (b), plus the public-only twin.

Two evaluators, both batched in a single forward pass:

``critic``
    The centralised critic on a **fully specified** sampled world: the observation's
    privileged block is filled from the world's hands. This is the right function for a leaf -
    a determinised world *is* complete, and the critic was trained on exactly that input - and
    it estimates the return under the network's own imperfect-information play, so it does not
    import strategy fusion. The known risk is distributional: an implausible sampled world
    puts the critic off-distribution (the sampler's bias report is the instrument for that).

``public``
    The public-information head on the actor's features: no privileged input at all, so it
    cannot be fooled by an implausible world, but it is also the weaker function.

Both are evaluated at **depth 1**: for every legal root move the position *after* that move is
scored, averaged over the sampled worlds, and the caller ranks moves by that average. Deeper
net-driven trees need the batched-enumeration trick that the compiled solver cannot do; until
then, short deals are still solved exactly (stronger and cheaper than any learned leaf).

A child state that ends the deal is scored **exactly** - the deal outcome is not something a
network should get to approximate - so the leaf only ever substitutes the net where the value
is genuinely unknown.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ..engine import GameState, deal_outcome
from ..obs import OBS_DIM, OPPONENT_OFFSETS, encode
from .sampler import World

#: Leaf modes that need a trained model.
NET_LEAF_MODES: tuple[str, ...] = ("critic", "public")


def _world_hands_relative(world: World, seat: int) -> list[list[int]]:
    """The world's three opponent hands in the observation's ``[left, partner, right]`` order."""
    hands = world.as_tuple()
    return [list(hands[(seat + offset) % 4]) for offset in OPPONENT_OFFSETS]


class NetLeaf:
    """Batched network leaf for a root position: one value per legal move.

    Parameters
    ----------
    model:
        An sb3-contrib ``MaskablePPO`` (or any object with a ``policy`` exposing
        ``predict_values`` / ``predict_public_values``).
    mode:
        ``"critic"`` (privileged, on the sampled world) or ``"public"`` (actor features only).
    """

    def __init__(self, model: Any, *, mode: str = "critic") -> None:
        if mode not in NET_LEAF_MODES:
            raise ValueError(f"unknown net leaf mode {mode!r}; expected one of {NET_LEAF_MODES}")
        if not hasattr(model, "policy"):
            raise TypeError("the net leaf needs an sb3 model (an object with .policy)")
        self.model = model
        self.mode = mode
        #: Diagnostics: how many leaves were forwarded, and the wall-clock spent.
        self.leaves = 0
        self.elapsed = 0.0

    # ---- the leaf itself ------------------------------------------------
    def value_of_moves(self, state: GameState, seat: int,
                       moves: Sequence[tuple[int, int]], worlds: Sequence[World]) -> dict[tuple[int, int], float]:
        """Mean **team-0** value of each legal move over ``worlds``.

        The sign convention matches :class:`SolveResult` in ``domino.pimc.solver``: every
        number is the team-0 point differential, whatever seat is moving, and the caller
        converts to the moving team's perspective. A move whose child position is terminal is
        scored with the engine's own outcome for that world (exact), never with the network.
        """
        import time

        import torch as th

        if not worlds:
            raise ValueError("at least one world is required")
        seat = int(seat)
        started = time.perf_counter()
        observations: list[np.ndarray] = []
        exact: list[float | None] = []
        keys: list[tuple[int, int]] = []
        for world in worlds:
            materialised = _materialise(state, world)
            for tid, end in moves:
                child = materialised.clone()
                child.play(seat, int(tid), int(end))
                if child.is_terminal():
                    outcome = deal_outcome(child)
                    payout = 0.0 if outcome.void else float(outcome.points)
                    winner = getattr(outcome, "winner_team", None)
                    exact.append(0.0 if winner is None else (payout if winner == 0 else -payout))
                else:
                    exact.append(None)
                    observations.append(np.asarray(encode(
                        child, seat, hands_override=_world_hands_relative(world, seat)),
                        dtype=np.float32))
                keys.append((int(tid), int(end)))
        values = np.zeros(len(keys), dtype=np.float64)
        flat = np.zeros(0, dtype=np.float64)
        if observations:
            batch = th.as_tensor(np.stack(observations), dtype=th.float32).to(
                device=getattr(self.model.policy, "device", "cpu"))
            with th.no_grad():
                if self.mode == "critic":
                    predicted = self.model.policy.predict_values(batch)
                else:
                    predicted = self.model.policy.predict_public_values(batch)
            flat = predicted.detach().cpu().numpy().reshape(-1).astype(np.float64)
        # The exact entries are written back **outside** the `if observations` branch: a
        # position whose every legal move ends the deal (the endgame) forwards nothing at all,
        # and an earlier version left those values at zero - the highest-stakes case silently
        # scored as worthless.
        cursor = 0
        for index, value in enumerate(exact):
            if value is None:
                values[index] = flat[cursor]
                cursor += 1
            else:
                values[index] = value
        totals: dict[tuple[int, int], float] = {}
        for index, key in enumerate(keys):
            totals[key] = totals.get(key, 0.0) + float(values[index])
        self.leaves += len(observations)
        self.elapsed += time.perf_counter() - started
        return {key: value / float(len(worlds)) for key, value in totals.items()}


def _materialise(state: GameState, world: World) -> GameState:
    """Install a world's hands into a clone of ``state`` (the sampler's own helper)."""
    from .player import materialise_world

    return materialise_world(state, world)


__all__ = ["NET_LEAF_MODES", "NetLeaf"]
