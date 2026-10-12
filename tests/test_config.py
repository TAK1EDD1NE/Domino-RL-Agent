"""Tests for typed configuration loading, saving and CLI overrides."""
from __future__ import annotations

from dataclasses import asdict

import pytest

from domino.config import (
    EnvConfig,
    EvalConfig,
    TrainConfig,
    apply_overrides,
    load_config,
    save_config,
)


def test_defaults_are_the_specified_values():
    env = EnvConfig()
    assert env.n_seats == 4
    assert env.learner_seats == (0, 2)
    assert env.reward_mode == "differential"
    train = TrainConfig()
    assert train.total_steps == 100_000_000
    assert train.env == env
    assert EvalConfig().match_to == 100


def test_yaml_round_trip(tmp_path):
    cfg = TrainConfig(
        run_name="unit",
        total_steps=1234,
        learning_rate=1e-3,
        env=EnvConfig(reward_mode="banked", seed=7, league_size=3),
    )
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    assert path.exists()
    loaded = load_config(path)
    assert loaded == cfg
    assert loaded.env.reward_mode == "banked"
    assert asdict(loaded) == asdict(cfg)


def test_load_config_nested_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "run_name: nested\n"
        "n_envs: 4\n"
        "env:\n"
        "  reward_mode: banked\n"
        "  seed: 42\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.run_name == "nested"
    assert cfg.n_envs == 4
    assert cfg.env.reward_mode == "banked"
    assert cfg.env.seed == 42


def test_apply_overrides_unknown_key_raises():
    cfg = TrainConfig()
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["does_not_exist=1"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["env.not_a_field=1"])


def test_apply_overrides_type_coercion():
    cfg = TrainConfig()
    apply_overrides(
        cfg,
        [
            "total_steps=1000",
            "learning_rate=0.0005",
            "gamma=0.9",
            "run_name=hello",
            "env.reward_mode=banked",
            "env.seed=99",
        ],
    )
    assert cfg.total_steps == 1000 and isinstance(cfg.total_steps, int)
    assert cfg.learning_rate == 0.0005 and isinstance(cfg.learning_rate, float)
    assert cfg.gamma == 0.9 and isinstance(cfg.gamma, float)
    assert cfg.run_name == "hello"
    assert cfg.env.reward_mode == "banked"
    assert cfg.env.seed == 99 and isinstance(cfg.env.seed, int)


def test_apply_overrides_rejects_bad_reward_mode():
    cfg = TrainConfig()
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["env.reward_mode=unknown"])


def test_load_config_rejects_bad_reward_mode(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("env:\n  reward_mode: bogus\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
