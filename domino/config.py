"""Typed configuration objects shared by train/test/play/observe.

Every entry point (training, evaluation, the Pygame UI) reads its parameters from
these dataclasses. ``load_config``/``save_config`` provide a YAML round-trip and
``apply_overrides`` implements the CLI ``--set key=value`` passthrough, coercing each
value to the type of the existing dataclass field and rejecting unknown keys.
"""
from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import yaml

REWARD_MODES: frozenset[str] = frozenset({"differential", "banked"})


@dataclass
class EnvConfig:
    n_seats: int = 4
    learner_seats: tuple[int, ...] = (0, 2)
    reward_mode: str = "differential"      # "differential" | "banked"
    opponent: str = "league"               # "league" | "self" | "rule" | "random" | "mixed"
    opponent_weights: dict[str, float] = field(
        default_factory=lambda: {"self": 0.5, "snapshot": 0.2, "rule": 0.2, "random": 0.1}
    )
    league_size: int = 10
    seed: int = 0
    #: Deals per training episode. 1 = the original one-deal episode contract used
    #: by the evaluation scripts and the UI; the training preset
    #: (configs/league_full.yaml) sets 5 so the match-score features carry signal.
    max_deals_per_episode: int = 1
    #: League hygiene (v3 item 9): draw snapshot opponents stochastically instead of
    #: argmax, so the opponent mix is a distribution rather than a fixed script.
    sample_snapshots: bool = True


@dataclass
class TrainConfig:
    run_name: str = "league_full"
    total_steps: int = 100_000_000
    n_envs: int = 8
    n_steps: int = 512
    batch_size: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    device: str = "auto"
    checkpoint_every: int = 500_000
    snapshot_every: int = 100_000
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    # ---- v3 architecture (docs/plans/2026-10-12-v3-architecture-plan.md) -------
    #: Auxiliary belief head (item 3): supervised ownership prediction of the hidden
    #: hands from the public trunk features.
    belief_enabled: bool = True
    belief_coef: float = 0.5          # weight applied to the belief BCE loss
    belief_lr: float = 3e-4           # its own optimiser's learning rate
    belief_batch: int = 256           # observations sampled from the rollout buffer
    #: Auxiliary updates every N rollout steps. Measured on this box: every=1 costs
    #: 3x throughput (376 vs 1164 steps/s with 6 workers), so the default trades
    #: shaping frequency for compute (an update every N*6 env steps).
    belief_every: int = 8
    #: Entropy floor (item 9): raise ent_coef back to the floor when the logged
    #: policy entropy drops under min_entropy nats, relax when it recovers.
    entropy_floor_enabled: bool = True
    entropy_floor: float = 0.05
    min_entropy: float = 0.1
    #: Public-information value head (search plan item (a)): a second value head on the
    #: actor's public features, trained on PPO's own returns by its own callback. It is the
    #: only play-time value the agent has; the centralised critic reads the true hands.
    public_value_enabled: bool = True
    #: MSE weight. Much smaller than belief_coef on purpose: this head's targets are point
    #: differences in the tens, so its loss is ~10^2 where the belief BCE is ~0.5 nats.
    public_value_coef: float = 0.1
    public_value_lr: float = 3e-4
    public_value_batch: int = 256
    #: Auxiliary updates every N rollout steps (same measured trade-off as belief_every:
    #: every=1 costs ~3x throughput at 6 workers).
    public_value_every: int = 8
    #: PIMC determinisations at inference (item 5); 0 disables search.
    search_k: int = 0
    # ---- in-training progress gate (domino/progress.py) -------------------------
    #: Evaluate the LIVE policy against fixed anchors (rule bot / random / this run's own
    #: earliest snapshot) while training, so strength is visible before the run ends.
    progress_eval_enabled: bool = True
    progress_eval_every: int = 250_000     # steps between evaluations (~5 min at 800 steps/s)
    progress_eval_deals: int = 200         # duplicate pairs per anchor (each pair = 2 games)
    #: Explicit checkpoint for the ``anchor`` leg. Empty = resolve this run's earliest league
    #: snapshot (the policy the run started from), which is the leg that measures progress.
    progress_anchor_path: str = ""
    #: Halt after ``progress_broken_patience`` consecutive BROKEN readings (frozen deals or an
    #: entropy collapse with no gradient signal). A stall only prints NO-GAIN and never stops.
    progress_stop_on_broken: bool = True
    progress_broken_patience: int = 2


@dataclass
class EvalConfig:
    deals_per_matchup: int = 1000
    match_to: int = 100
    seed: int = 0
    model_path: str = "models/checkpoints/latest.zip"
    out_dir: str = "results"


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _validate_env(env: EnvConfig) -> None:
    """Reject an unknown ``reward_mode`` (the only validated field for now)."""
    if env.reward_mode not in REWARD_MODES:
        raise ValueError(
            f"reward_mode must be one of {sorted(REWARD_MODES)}, got {env.reward_mode!r}"
        )


def _coerce_value(current: Any, value: Any) -> Any:
    """Coerce ``value`` to the type of ``current`` (bool before int, exact for the rest)."""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _BOOL_TRUE:
            return True
        if text in _BOOL_FALSE:
            return False
        raise ValueError(f"cannot parse boolean from {value!r}")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return str(value)
    if isinstance(current, tuple):
        return tuple(value)
    if isinstance(current, list):
        return list(value)
    if isinstance(current, dict):
        return dict(value)
    return value


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate ``cls`` from a mapping, coercing nested dataclasses and types."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"unknown configuration key(s) for {cls.__name__}: {sorted(unknown)}")
    base = cls()
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        if f.name == "env" and isinstance(raw, dict):
            kwargs[f.name] = _build_dataclass(EnvConfig, raw)
        else:
            kwargs[f.name] = _coerce_value(getattr(base, f.name), raw)
    return cls(**kwargs)


def load_config(path: str | pathlib.Path) -> TrainConfig:
    """Load a :class:`TrainConfig` from a YAML file (nested ``env:`` mapping supported)."""
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration file {path!s} must contain a mapping")
    cfg = _build_dataclass(TrainConfig, raw)
    _validate_env(cfg.env)
    return cfg


def save_config(cfg: TrainConfig, path: str | pathlib.Path) -> None:
    """Write ``cfg`` to ``path`` as YAML (round-trips through :func:`load_config`)."""
    path = pathlib.Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")


def apply_overrides(cfg: TrainConfig, pairs: Iterable[str]) -> TrainConfig:
    """Apply ``"key=value"`` overrides in place, coercing to the field's type.

    Dotted keys address nested configuration, e.g. ``env.reward_mode=banked``. Unknown
    field names raise :class:`ValueError`.
    """
    for pair in pairs:
        key, sep, raw = str(pair).partition("=")
        if not sep or not key:
            raise ValueError(f"override must be of the form key=value, got {pair!r}")
        _assign(cfg, key.split("."), raw)
    _validate_env(cfg.env)
    return cfg


def _assign(cfg: Any, parts: list[str], raw: str) -> None:
    obj = cfg
    for name in parts[:-1]:
        if not hasattr(obj, name):
            raise ValueError(f"unknown configuration section: {'.'.join(parts)}")
        obj = getattr(obj, name)
    field_name = parts[-1]
    if not hasattr(obj, field_name):
        raise ValueError(f"unknown configuration field: {'.'.join(parts)}")
    current = getattr(obj, field_name)
    setattr(obj, field_name, _coerce_value(current, raw))
