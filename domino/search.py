"""Perfect-Information Monte Carlo (PIMC) root selection for inference.

At a decision point a single observation does **not** reveal the three
opponents' hands, so to evaluate a candidate root move this module samples
hidden hands that are *consistent with public information only*, makes the
position perfect-information for each sample, evaluates the resulting leaf with
a value function, and averages:

    for each legal root move (tile, end):
        for each determinisation:
            obs = encode(state_after_move, seat, hands_override=sampled_hands)
        value(move) = mean over determinisations of value_fn(obs)
    return the move with the highest mean value (ties: lowest action index)

Two building blocks:

* :class:`Determinizer` — draws the three opponents' hands from public
  information (counts, known lacks, tiles already played). It is the only source
  of randomness; seeding it makes the whole policy reproducible.
* :class:`PIMCPolicyAgent` — an :class:`~domino.agents.base.Agent` that ranks the
  legal root moves with a caller-supplied ``value_fn``.

Value-function dependency (important)
-------------------------------------
PIMC reasons from a *fictional* perfect-information state: only the root
player's own hand, the table and the play history are real, the opponents'
hands are guesses. A value function trained on the *public* observation cannot
tell a real leaf from a sampled one, so PIMC leaves are only meaningful once a
**centralised** value net exists — one that consumes the privileged slice
(``domino.obs.PRIVILEGED_SLICE``), i.e. is trained to value states knowing all
four hands. The constructor therefore accepts *any* callable
``(B, OBS_DIM) float32 -> (B,) float32`` and makes no assumption about it, so
tests can use an analytic stub. :func:`value_fn_from_model` adapts an
sb3-contrib ``MaskablePPO`` (``model.policy.predict_values``) once such a net is
trained.

This module imports numpy and the engine/obs only; torch is imported lazily
inside :func:`value_fn_from_model`.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .engine import GameState, legal_moves
from .masks import action_index
from .obs import (
    OBS_DIM,
    OBS_DIM_V3,
    PRIVILEGED_SLICE,
    PUBLIC_DIM,
    PUBLIC_SLICE,
    encode,
    opponent_seats,
)
from .tiles import ALL_TILE_IDS, NUM_TILES, tile_values

__all__ = [
    "Determinizer",
    "PIMCPolicyAgent",
    "value_fn_from_model",
    "OBS_DIM",
    "OBS_DIM_V3",
    "PUBLIC_DIM",
    "PUBLIC_SLICE",
    "PRIVILEGED_SLICE",
]

SEATS = 4
HAND_SIZE = 7          # tiles per seat at the start of a deal
NUM_OPPONENTS = 3


class Determinizer:
    """Samples the three opponents' hidden hands from public information only.

    The unseen tiles (the complement of the acting seat's own hand and the
    table) are partitioned over the other three seats subject to three public
    constraints:

    * **counts** — a seat's remaining tile count is ``HAND_SIZE -
      len(state.played_by[seat])`` (every seat starts with 7 and only ever
      plays tiles; there is no boneyard, so the counts are public even though
      the facing values are not);
    * **known lacks** — ``state.lacks[seat]`` holds the pip values a seat is
      proven to lack (it was forced to pass while those values were open); a
      tile carrying such a value may never be handed to that seat. The lack is
      monotone (hands only shrink), so it is a hard constraint;
    * **played tiles** — a seat can never hold a tile it already played; this is
      automatic because played tiles are on the table and therefore excluded
      from the unseen set.

    Hands are returned in the ``[left, partner, right]`` relative order used by
    the observation's privileged block (``domino.obs.opponent_seats``).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        max_restarts: int = 8,
        node_limit: int = 200_000,
    ) -> None:
        if max_restarts < 1:
            raise ValueError(f"max_restarts must be >= 1, got {max_restarts}")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.max_restarts = int(max_restarts)
        self.node_limit = int(node_limit)

    # ---- public interface ---------------------------------------------
    def sample(self, state: GameState, seat: int) -> tuple[list[int], list[int], list[int]]:
        """Draw one consistent ``(left, partner, right)`` hand assignment.

        Raises ``ValueError`` when the public constraints admit no assignment
        (an inconsistent state, or a lack that leaves too few legal tiles). The
        caller is expected to keep the last successful sample.
        """
        opponents = opponent_seats(seat)
        unseen = self._unseen_tiles(state, seat)
        counts = [HAND_SIZE - len(state.played_by[other]) for other in opponents]

        for other, count in zip(opponents, counts):
            if len(state.hands[other]) != count:
                raise ValueError(
                    f"inconsistent state: seat {other} holds {len(state.hands[other])} "
                    f"tiles but the public count is {count}"
                )
        if sum(counts) != len(unseen):
            raise ValueError(
                f"unsatisfiable determinisation: the {len(unseen)} unseen tiles do "
                f"not match the public counts {tuple(counts)}"
            )

        lacks = [frozenset(state.lacks[other]) for other in opponents]
        assignment = self._partition(unseen, counts, lacks)
        if assignment is None:
            raise ValueError(
                "unsatisfiable determinisation: no hidden-hand assignment respects "
                f"the counts {tuple(counts)} and the known lacks "
                f"{[sorted(lk) for lk in lacks]}"
            )
        return tuple(sorted(hand) for hand in assignment)  # type: ignore[return-value]

    # ---- internals -----------------------------------------------------
    @staticmethod
    def _unseen_tiles(state: GameState, seat: int) -> list[int]:
        """Public unseen tiles: the complement of my hand and the table."""
        mine = set(state.hands[seat])
        table = set(state.table)
        return [tid for tid in ALL_TILE_IDS if tid not in mine and tid not in table]

    def _partition(
        self,
        unseen: Sequence[int],
        counts: Sequence[int],
        lacks: Sequence[frozenset],
    ) -> Optional[list[list[int]]]:
        """Randomly partition ``unseen`` into three hands of the given sizes.

        Uses randomised backtracking (restarted a few times): reshuffling the
        tiles and the candidate seats on each restart both makes a fresh sample
        every call and lets the search find a valid assignment whenever one
        exists. Returns ``None`` when no assignment satisfies the lacks.
        """
        tiles = [int(t) for t in unseen]
        for _ in range(self.max_restarts):
            order = [int(t) for t in self.rng.permutation(tiles)]
            remaining = list(counts)
            assignment: list[list[int]] = [[], [], []]
            if self._backtrack(order, 0, remaining, assignment, lacks, [0]):
                return assignment
        return None

    def _backtrack(
        self,
        tiles: Sequence[int],
        index: int,
        remaining: list[int],
        assignment: list[list[int]],
        lacks: Sequence[frozenset],
        nodes: list[int],
    ) -> bool:
        nodes[0] += 1
        if nodes[0] > self.node_limit:
            return False
        if index == len(tiles):
            return True
        tid = tiles[index]
        a, b = tile_values(tid)
        candidates = [
            s
            for s in range(NUM_OPPONENTS)
            if remaining[s] > 0 and a not in lacks[s] and b not in lacks[s]
        ]
        if not candidates:
            return False
        for seat in (int(s) for s in self.rng.permutation(candidates)):
            assignment[seat].append(tid)
            remaining[seat] -= 1
            if self._backtrack(tiles, index + 1, remaining, assignment, lacks, nodes):
                return True
            assignment[seat].pop()
            remaining[seat] += 1
        return False


class PIMCPolicyAgent:
    """An :class:`Agent` that picks the legal move with the best PIMC leaf value.

    For every legal root move the agent, for each determinisation, encodes the
    resulting state from ``seat``'s point of view with the sampled hidden hands,
    batches those observations and calls ``value_fn`` **once per root move**,
    then averages. The highest mean value wins; ties are broken by the lowest
    action index. A position with a single legal move is returned immediately
    without touching the value function.
    """

    name = "pimc"

    def __init__(
        self,
        value_fn: Callable[[np.ndarray], np.ndarray],
        *,
        legal_moves_fn: Optional[Callable[[GameState, int], list[tuple[int, int]]]] = None,
        n_determinizations: int = 16,
        seed: Optional[int] = 0,
        name: str = "pimc",
    ) -> None:
        if n_determinizations < 1:
            raise ValueError(f"n_determinizations must be >= 1, got {n_determinizations}")
        self.value_fn = value_fn
        self.legal_moves_fn = legal_moves_fn if legal_moves_fn is not None else legal_moves
        self.n_determinizations = int(n_determinizations)
        self.seed = seed
        self.name = name
        self.determinizer = Determinizer(seed=seed)

    # ---- public interface ---------------------------------------------
    def act(self, state: GameState, seat: int) -> int:
        """Return the best legal action index for ``seat`` in ``state``."""
        moves = self.legal_moves_fn(state, seat)
        if not moves:
            raise ValueError(
                f"PIMCPolicyAgent.act called for seat {seat} with no legal move (forced pass)"
            )
        ordered = sorted(moves, key=lambda move: action_index(move[0], move[1]))
        if len(ordered) == 1:
            return action_index(ordered[0][0], ordered[0][1])

        samples = self._determinizations(state, seat)
        best_action: Optional[int] = None
        best_value = -np.inf
        for tid, end in ordered:
            after = state.clone()
            after.turn = seat
            after.play(seat, tid, end)
            observations = self._leaf_observations(after, seat, samples)
            values = np.asarray(self.value_fn(observations), dtype=np.float64).reshape(-1)
            mean_value = float(values.mean())
            # Strict ``>`` keeps the first (lowest action index) on ties.
            if mean_value > best_value:
                best_value = mean_value
                best_action = action_index(tid, end)
        assert best_action is not None  # ``ordered`` is non-empty
        return best_action

    # ---- internals -----------------------------------------------------
    def _determinizations(self, state: GameState, seat: int) -> list[tuple[list[int], list[int], list[int]]]:
        """Draw up to ``n_determinizations`` hands, reusing the last on failure."""
        samples: list[tuple[list[int], list[int], list[int]]] = []
        last: Optional[tuple[list[int], list[int], list[int]]] = None
        for _ in range(self.n_determinizations):
            try:
                last = self.determinizer.sample(state, seat)
            except ValueError:
                pass  # keep the last successful sample (per the PIMC contract)
            if last is not None:
                samples.append(last)
        if not samples:
            raise ValueError(
                f"PIMC could not draw any consistent hidden-hand sample for seat {seat}"
            )
        return samples

    @staticmethod
    def _leaf_observations(
        after: GameState,
        seat: int,
        samples: Sequence[tuple[list[int], list[int], list[int]]],
    ) -> np.ndarray:
        """Stack the ``(len(samples), OBS_DIM)`` float32 leaf observations.

        ``hands_override`` (in ``[left, partner, right]`` order) fills the
        privileged block with the sampled hands; every public feature is read
        off ``after`` and is identical across samples because the public counts
        match (determinisation never changes a hand *size*).
        """
        rows = np.empty((len(samples), OBS_DIM), dtype=np.float32)
        for row, hands in enumerate(samples):
            rows[row] = encode(after, seat, hands_override=[list(hand) for hand in hands])
        return rows


def value_fn_from_model(model) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap an sb3-contrib ``MaskablePPO`` as a PIMC value function.

    Uses ``model.policy.predict_values`` (the centralised critic) under
    ``torch.no_grad()``. torch is imported lazily so this module stays
    importable in torch-free processes; the model is supplied by the caller
    (``test.py`` / ``play.py`` build it from a checkpoint).
    """
    def value_fn(observations: np.ndarray) -> np.ndarray:
        import torch as th

        array = np.asarray(observations, dtype=np.float32)
        with th.no_grad():
            values = model.policy.predict_values(th.as_tensor(array))
        return values.detach().cpu().numpy().reshape(-1).astype(np.float32)

    return value_fn
