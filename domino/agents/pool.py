"""Opponent league for training (spec section 8).

The learner always occupies team A's two seats; the opposing team is drawn per
deal from this pool. The pool exposes four sources behind the :class:`Agent`
interface:

* ``self``   — the current learner policy (mirror self-play, the main engine);
* ``snapshot`` — a frozen checkpoint, sampled uniformly from the known paths and
  loaded lazily through a cache holding the whole league window by default (see
  :attr:`OpponentPool.max_loaded`: a cache smaller than the window misses on most
  draws and reloads a checkpoint from disk per deal, which measured ~4x slower);
* ``rule``   — the documented rule-based baseline;
* ``random`` — the uniform-random agent (cheap diversity).

Weights are configurable: missing or zero keys are dropped and the rest are
renormalised, so ``{"self": 1.0}`` reproduces pure mirror self-play. A deal is a
coherent unit — both of team B's seats get the same opponent by default, and
``mix_opponents=True`` draws each seat independently for the ablation. All
randomness flows through an explicit :class:`numpy.random.Generator`, so a
seeded pool is deterministic. sb3 is imported only when a snapshot is actually
loaded, never at module import time.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np

from .base import Agent
from .policy_agent import PolicyAgent
from .random_agent import RandomAgent
from .rule_agent import RuleAgent

#: Default league weights (spec section 8): self-play dominates, snapshots and
#: the rule bot tie, random is the cheap diversity source.
DEFAULT_WEIGHTS: dict[str, float] = {"self": 0.5, "snapshot": 0.2, "rule": 0.2, "random": 0.1}
KNOWN_SOURCES: frozenset[str] = frozenset(DEFAULT_WEIGHTS)


def load_policy(path: str) -> Any:
    """Load a frozen snapshot checkpoint (sb3 imported lazily on first use)."""
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(path)


def _normalised_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Validate and renormalise user weights; drop missing/zero keys."""
    source = dict(DEFAULT_WEIGHTS) if weights is None else dict(weights)
    unknown = set(source) - KNOWN_SOURCES
    if unknown:
        raise ValueError(
            f"unknown opponent weight keys {sorted(unknown)}; allowed: {sorted(KNOWN_SOURCES)}"
        )
    for key, value in source.items():
        if value < 0:
            raise ValueError(f"opponent weight for {key!r} must be non-negative, got {value}")
    positive = {key: float(value) for key, value in source.items() if value > 0}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("opponent weights must not be all zero")
    return {key: value / total for key, value in positive.items()}


class OpponentPool:
    """Weighted league of opposing-team agents with a snapshot LRU cache."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        learner_model: Any = None,
        snapshot_paths: Sequence[str] | None = None,
        rule_agent: Agent | None = None,
        seed: int | None = None,
        max_loaded: int | None = None,
        mix_opponents: bool = False,
    ) -> None:
        if max_loaded is not None and max_loaded < 1:
            raise ValueError(f"max_loaded must be >= 1, got {max_loaded}")
        self._weights = _normalised_weights(weights)
        self._learner_model = learner_model
        self._rule_agent: Agent = rule_agent if rule_agent is not None else RuleAgent(seed=seed)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._max_loaded = max_loaded
        self.mix_opponents = mix_opponents
        self._snapshot_paths: list[str] = []
        self._snapshot_cache: "OrderedDict[str, Any]" = OrderedDict()
        if snapshot_paths:
            self.refresh_snapshots(snapshot_paths)

    # ---- sampling ------------------------------------------------------
    @property
    def max_loaded(self) -> int:
        """Effective snapshot cache size.

        ``None`` (the default) means "hold the whole league window": the pool
        samples uniformly over ``snapshot_paths``, so a cache *smaller* than that
        window misses on most draws and reloads a checkpoint from disk inside
        every worker -- measured at ~4x slower end-to-end training (400 vs 1500
        learner steps/s) once the window had grown to ten snapshots. Pass an int
        to bound the cache explicitly (the smaller-cache eviction behaviour is
        still exercised by the tests).
        """
        if self._max_loaded is not None:
            return int(self._max_loaded)
        return max(1, len(self._snapshot_paths))

    @property
    def weights(self) -> dict[str, float]:
        """The configured (renormalised, positive) source weights."""
        return dict(self._weights)

    def sample(self) -> Agent:
        """Draw a single opponent from the weighted league."""
        return self._draw(self._rng)

    def assign(self, deal_index: int) -> tuple[Agent, Agent]:
        """Return the two agents for the opposing team's seats, in seat order.

        Both seats share one opponent by default; with ``mix_opponents`` each
        seat is drawn independently. Given a seeded pool the assignment for a
        ``deal_index`` is reproducible.
        """
        rng = self._deal_rng(deal_index)
        first = self._draw(rng)
        second = self._draw(rng) if self.mix_opponents else first
        return first, second

    # ---- learner wiring ------------------------------------------------
    def set_learner(self, model: Any) -> None:
        """Attach the current learner policy used by the ``self`` source."""
        self._learner_model = model

    def learner_agent(self) -> PolicyAgent:
        """Wrap the current learner as an agent (the self-play entry)."""
        if self._learner_model is None:
            raise ValueError("no learner model is set; call set_learner(model) first")
        return PolicyAgent(self._learner_model, deterministic=True, name="self")

    # ---- snapshots -----------------------------------------------------
    def refresh_snapshots(self, paths: Sequence[str]) -> None:
        """Replace the snapshot list; load lazily, keep at most ``max_loaded``.

        Every path is checked up front so a bad path fails fast with the path in
        the message. Cached models for paths that remain are kept; models for
        paths that have been dropped are evicted.
        """
        new_paths = [str(path) for path in paths]
        for path in new_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"snapshot checkpoint not found: {path}")
        self._snapshot_paths = new_paths
        for path in list(self._snapshot_cache):
            if path not in new_paths:
                del self._snapshot_cache[path]

    # ---- internals -----------------------------------------------------
    def _deal_rng(self, deal_index: int) -> np.random.Generator:
        """Return the generator for one deal's draw.

        A seeded pool derives it from ``(seed, deal_index)`` so a deal's
        opponent is reproducible regardless of call order; an unseeded pool
        keeps a single persistent stream.
        """
        if self._seed is None:
            return self._rng
        return np.random.default_rng([int(self._seed), int(deal_index)])

    def _available_weights(self) -> dict[str, float]:
        """Weights restricted to sources that can actually produce an agent."""
        available: dict[str, float] = {}
        for key, weight in self._weights.items():
            if key == "self" and self._learner_model is None:
                continue
            if key == "snapshot" and not self._snapshot_paths:
                continue
            available[key] = weight
        return available

    def _draw(self, rng: np.random.Generator) -> Agent:
        available = self._available_weights()
        if not available:
            raise ValueError(
                "no opponent source is available: attach a learner, provide snapshots "
                "or assign a positive weight to 'rule'/'random'"
            )
        keys = list(available)
        probabilities = np.array([available[key] for key in keys], dtype=float)
        probabilities /= probabilities.sum()
        key = keys[int(rng.choice(len(keys), p=probabilities))]
        if key == "self":
            return self.learner_agent()
        if key == "rule":
            return self._rule_agent
        if key == "random":
            return RandomAgent(seed=int(rng.integers(0, 2**31 - 1)))
        return self._snapshot_agent(rng)

    def _snapshot_agent(self, rng: np.random.Generator) -> PolicyAgent:
        path = self._snapshot_paths[int(rng.integers(len(self._snapshot_paths)))]
        return PolicyAgent(self._get_snapshot(path), deterministic=True, name="snapshot")

    def _get_snapshot(self, path: str) -> Any:
        """Return the (lazily loaded) model for ``path``, evicting the LRU entry."""
        cache = self._snapshot_cache
        if path in cache:
            cache.move_to_end(path)
            return cache[path]
        model = load_policy(path)
        cache[path] = model
        cache.move_to_end(path)
        while len(cache) > self.max_loaded:
            cache.popitem(last=False)
        return model
