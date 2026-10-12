"""Tests for the opponent league pool (spec section 8).

Covers weighted sampling (honouring the default weights, dropping missing keys
and rejecting all-zero weights), the coherent per-deal team assignment (same
opponent type for both opposing seats unless per-seat mixing is enabled), the
snapshot LRU (lazy loading, at most ``max_loaded`` models in memory, a clear
``FileNotFoundError`` for a missing path) and seed determinism.
"""
from __future__ import annotations

import gc
import weakref

import pytest

import domino.agents.pool as pool_module
from domino.agents import OpponentPool, PolicyAgent


class _DummyModel:
    """Stand-in for a MaskablePPO: only the attributes the pool touches."""

    def __init__(self, path: str = "dummy") -> None:
        self.path = path


def _make_snapshot_files(tmp_path, n: int = 5) -> list[str]:
    paths = []
    for i in range(n):
        path = tmp_path / f"policy_{i}.zip"
        path.write_bytes(b"stub checkpoint")
        paths.append(str(path))
    return paths


def test_all_zero_weights_raise():
    with pytest.raises(ValueError):
        OpponentPool(weights={"self": 0.0, "snapshot": 0.0, "rule": 0.0, "random": 0.0})


def test_unknown_weight_key_raises():
    with pytest.raises(ValueError):
        OpponentPool(weights={"self": 0.5, "bogus": 0.5})


def test_weighted_sampling_honours_default_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_module, "load_policy", lambda path: _DummyModel(path))
    pool = OpponentPool(
        learner_model=_DummyModel("learner"),
        snapshot_paths=_make_snapshot_files(tmp_path),
        seed=12345,
    )
    draws = 10000
    counts = {"self": 0, "snapshot": 0, "rule": 0, "random": 0}
    for _ in range(draws):
        counts[pool.sample().name] += 1
    assert sum(counts.values()) == draws
    for name, weight in {"self": 0.5, "snapshot": 0.2, "rule": 0.2, "random": 0.1}.items():
        assert abs(counts[name] / draws - weight) <= 0.02, (name, counts[name] / draws)


def test_missing_weight_keys_are_dropped_and_renormalised():
    pool = OpponentPool(weights={"rule": 1.0, "random": 3.0}, seed=7)
    draws = 4000
    counts = {"rule": 0, "random": 0}
    for _ in range(draws):
        counts[pool.sample().name] += 1
    assert abs(counts["rule"] / draws - 0.25) <= 0.03
    assert abs(counts["random"] / draws - 0.75) <= 0.03


def test_assign_gives_same_type_by_default_and_can_mix(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_module, "load_policy", lambda path: _DummyModel(path))
    paths = _make_snapshot_files(tmp_path)
    default = OpponentPool(learner_model=_DummyModel("l"), snapshot_paths=paths, seed=7)
    for deal_index in range(50):
        first, second = default.assign(deal_index)
        assert first.name == second.name

    mixed = OpponentPool(
        learner_model=_DummyModel("l"), snapshot_paths=paths, seed=7, mix_opponents=True
    )
    pairs = {tuple(sorted((a.name, b.name))) for a, b in (mixed.assign(i) for i in range(500))}
    assert any(first != second for first, second in pairs), "mixed assignment never split the seats"


def test_assign_is_deterministic_under_a_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_module, "load_policy", lambda path: _DummyModel(path))
    paths = _make_snapshot_files(tmp_path)

    def run() -> list[tuple[str, str]]:
        pool = OpponentPool(learner_model=_DummyModel("l"), snapshot_paths=paths, seed=42)
        return [(a.name, b.name) for a, b in (pool.assign(i) for i in range(100))]

    assert run() == run()


def test_sampling_is_deterministic_under_a_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_module, "load_policy", lambda path: _DummyModel(path))
    paths = _make_snapshot_files(tmp_path)

    def run() -> list[str]:
        pool = OpponentPool(learner_model=_DummyModel("l"), snapshot_paths=paths, seed=99)
        return [pool.sample().name for _ in range(200)]

    assert run() == run()


def test_refresh_snapshots_missing_path_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_module, "load_policy", lambda path: _DummyModel(path))
    pool = OpponentPool(learner_model=_DummyModel("l"), seed=0)
    with pytest.raises(FileNotFoundError) as excinfo:
        pool.refresh_snapshots(["definitely-missing.zip"])
    assert "definitely-missing.zip" in str(excinfo.value)


def test_snapshot_lru_holds_at_most_max_loaded_models(tmp_path, monkeypatch):
    live = weakref.WeakSet()
    calls = {"n": 0}

    def loader(path: str) -> _DummyModel:
        calls["n"] += 1
        model = _DummyModel(path)
        live.add(model)
        return model

    monkeypatch.setattr(pool_module, "load_policy", loader)
    paths = _make_snapshot_files(tmp_path, n=5)
    pool = OpponentPool(weights={"snapshot": 1.0}, snapshot_paths=paths, seed=0, max_loaded=3)
    for _ in range(400):
        agent = pool.sample()
        assert agent.name == "snapshot"
        assert isinstance(agent.model, _DummyModel)
    gc.collect()
    assert calls["n"] > len(paths), "the loader was never asked to reload, so nothing was evicted"
    assert len(live) <= 3, f"pool held {len(live)} snapshot models, expected at most 3"


def test_set_learner_and_learner_agent_wrapping():
    pool = OpponentPool(seed=0)
    with pytest.raises(ValueError):
        pool.learner_agent()
    model = _DummyModel("learner")
    pool.set_learner(model)
    agent = pool.learner_agent()
    assert isinstance(agent, PolicyAgent)
    assert agent.name == "self"
    assert agent.model is model

    # With a learner attached, the "self" source becomes drawable.
    names = {pool.sample().name for _ in range(50)}
    assert names <= {"self", "rule", "random"}
