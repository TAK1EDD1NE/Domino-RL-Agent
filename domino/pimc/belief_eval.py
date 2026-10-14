"""Belief-head calibration against the sampler's exact uniform-over-feasible baseline.

The belief head predicts, from the public part of the observation, which of the three
other seats holds each of the 28 tiles (3 x 28 logits, ``[left, partner, right]`` order).
Before wiring that prediction into anything, this module answers two questions with
numbers instead of faith:

1. **Does it beat the constraints?** The baseline is the *exact* uniform distribution over
   worlds consistent with the public counts and the monotone void-lacks - i.e. exactly what
   :func:`domino.pimc.sampler.sample_worlds` draws under ``method="rejection"``. A head that
   does not beat it has learned nothing beyond the constraints it is handed.
2. **Are its probabilities honest?** Log-loss and Brier are proper scoring rules (they
   reward calibration *and* sharpness); ECE measures the calibration part alone.

Only the **informative** entries are scored by default: a tile in my own hand or already on
the table is deterministically absent from every opponent's hand, so including those 84-entry
cells would inflate accuracy and dilute the loss with free predictions. ``informative=True``
restricts the metrics to the unseen tiles (the ones whose owner is actually in doubt).

The baseline is Monte-Carlo over ``worlds`` rejection draws, so its own error is
``~sqrt(p(1-p)/worlds)`` per cell (about 0.02 at p = 0.25 with 512 worlds) - reported alongside,
because a head that ties the baseline within that noise has still learned nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from ..engine import GameState
from ..masks import decode_action
from ..obs import OBS_DIM, OPPONENT_OFFSETS, PRIVILEGED_SLICE, PUBLIC_SLICE, encode
from ..tiles import NUM_TILES
from .sampler import PublicView, sample_worlds

#: Relative seat index -> label slot in the privileged block ([left, partner, right]).
SEAT_SLOT: dict[int, int] = {offset: slot for slot, offset in enumerate(OPPONENT_OFFSETS)}


@dataclass
class BeliefSample:
    """One decision point where the belief head may be scored."""

    obs: np.ndarray            # (OBS_DIM,) float32, as the env emits it
    seat: int                  # absolute acting seat
    ply: int                   # decision index within the deal (0 = first decision)
    view: PublicView           # the public information at that moment
    labels: np.ndarray         # (3, 28) float32, the true opponents' hands

    @property
    def phase(self) -> str:
        """Opening / midgame / endgame bucket, matching the judge's convention."""
        tiles_left = int(self.obs[0:28].sum())
        if self.ply <= 1:
            return "opening"
        if tiles_left <= 2:
            return "end"
        return "mid"


def collect_belief_samples(agents: dict[int, Any], *, deals: int, seed: int,
                           learner_seats: Sequence[int] = (0, 2),
                           rng: Optional[np.random.Generator] = None) -> list[BeliefSample]:
    """Play deals and record every learner-seat decision as a :class:`BeliefSample`.

    The observation and the labels both come from :func:`domino.obs.encode`, so what the
    head is scored on is byte-identical to what it sees in training. The ``view`` is the
    sampler's own view object, so the baseline cannot drift from the sampler's semantics.
    """
    generator = np.random.default_rng(seed) if rng is None else rng
    samples: list[BeliefSample] = []
    for deal_index in range(int(deals)):
        state = _fresh_deal(agents, generator)
        while not state.is_terminal():
            seat = int(state.turn)
            moves = state.legal_moves(seat)
            if not moves:
                state.pass_turn(seat)
                continue
            if seat in learner_seats:
                obs = np.asarray(encode(state, seat), dtype=np.float32)
                labels = obs[PRIVILEGED_SLICE].reshape(3, NUM_TILES).astype(np.float32)
                samples.append(BeliefSample(
                    obs=obs, seat=seat, ply=sum(1 for s in samples if s.seat == seat),
                    view=PublicView.from_state(state, seat), labels=labels))
            action = int(agents[seat].act(state, seat))
            tid, end = decode_action(action)
            state.play(seat, tid, end)
        del deal_index                                            # noqa: F841 - clarity
    return samples


def _fresh_deal(agents: dict[int, Any], rng: np.random.Generator) -> GameState:
    """Deal 28 shuffled tiles into four hands, ``[6|6]`` holder leading."""
    from ..tiles import ALL_TILE_IDS, SIX_SIX_ID

    tiles = list(ALL_TILE_IDS)
    rng.shuffle(tiles)
    hands = [sorted(tiles[index * 7:(index + 1) * 7]) for index in range(4)]
    leader = next(seat for seat, hand in enumerate(hands) if SIX_SIX_ID in hand)
    return GameState(hands=[list(hand) for hand in hands], rng=rng, turn=leader,
                     leader=leader, deal_no=1, forced_leader_six_six=True)


# ---------------------------------------------------------------------------
# the two predictors
# ---------------------------------------------------------------------------
def baseline_marginals(view: PublicView, *, worlds: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Exact-uniform baseline marginals ``(3, 28)`` plus the standard error ``(3, 28)``.

    Counting how often each unseen tile lands in each opponent's hand over ``worlds``
    rejection draws estimates the marginal of the *uniform-over-feasible* distribution -
    the distribution a head must beat to have learned anything from the history.
    """
    counts = np.zeros((3, NUM_TILES), dtype=np.float64)
    drawn = sample_worlds(view, int(worlds), rng, method="rejection")
    for world in drawn:
        hands = world.as_tuple()
        for offset, slot in SEAT_SLOT.items():
            absolute = (view.seat + offset) % 4
            for tile in hands[absolute]:
                counts[slot, tile] += 1.0
    probabilities = counts / float(max(1, len(drawn)))
    error = np.sqrt(np.clip(probabilities * (1.0 - probabilities), 0.0, None) / float(max(1, len(drawn))))
    # A cell that can only ever be 0 or 1 gets no Monte-Carlo error, by construction.
    return probabilities, error


def head_marginals(policy: Any, observations: np.ndarray) -> np.ndarray:
    """The belief head's probabilities, ``(N, 3, 28)``, from the full observation batch."""
    import torch as th

    batch = np.asarray(observations, dtype=np.float32)
    with th.no_grad():
        logits = policy.predict_beliefs(batch)
    return th.sigmoid(logits).detach().cpu().numpy().reshape(-1, 3, NUM_TILES)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def informative_mask(view: PublicView) -> np.ndarray:
    """``(3, 28)`` boolean: tiles neither in my hand nor already on the table."""
    mask = np.zeros((3, NUM_TILES), dtype=bool)
    for tile in view.unseen:
        mask[:, int(tile)] = True
    return mask


def binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean binary cross-entropy, clipped to avoid ``log(0)``."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1 - 1e-9)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error of the probabilities."""
    return float(np.mean((np.asarray(probabilities, dtype=np.float64)
                          - np.asarray(labels, dtype=np.float64)) ** 2))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray,
                               *, bins: int = 10) -> tuple[float, list[tuple[float, int, float, float]]]:
    """Binned |confidence - accuracy|, weighted by bin population.

    Returns ``(ece, table)`` where each table row is ``(bin_upper_edge, count, mean_p, mean_y)``.
    """
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    total = p.size
    if total == 0:
        return float("nan"), []
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    table: list[tuple[float, int, float, float]] = []
    for index in range(int(bins)):
        low, high = edges[index], edges[index + 1]
        inside = (p >= low) & (p < high if index < bins - 1 else p <= high)
        count = int(np.count_nonzero(inside))
        if count == 0:
            table.append((float(high), 0, float("nan"), float("nan")))
            continue
        mean_p = float(p[inside].mean())
        mean_y = float(y[inside].mean())
        ece += (count / total) * abs(mean_p - mean_y)
        table.append((float(high), count, mean_p, mean_y))
    return float(ece), table


@dataclass
class Score:
    """The four headline numbers of one predictor on one subset of cells."""

    entries: int
    log_loss: float
    brier: float
    ece: float
    accuracy: float

    def as_row(self, name: str) -> dict[str, Any]:
        """Flat CSV row."""
        return {"predictor": name, "entries": self.entries, "log_loss": self.log_loss,
                "brier": self.brier, "ece": self.ece, "accuracy": self.accuracy}


def score(probabilities: np.ndarray, labels: np.ndarray) -> Score:
    """All four metrics over the entries given (already subset by a mask)."""
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return Score(0, float("nan"), float("nan"), float("nan"), float("nan"))
    ece, _ = expected_calibration_error(p, y)
    return Score(entries=int(p.size), log_loss=binary_log_loss(p, y), brier=brier_score(p, y),
                 ece=ece, accuracy=float(np.mean((p > 0.5) == (y > 0.5))))


__all__ = [
    "BeliefSample", "Score", "SEAT_SLOT", "baseline_marginals", "binary_log_loss",
    "brier_score", "collect_belief_samples", "expected_calibration_error",
    "head_marginals", "informative_mask", "score",
]
