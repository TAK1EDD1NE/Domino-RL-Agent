"""Tests for :class:`domino.league.EntropyFloorCallback`.

The callback is exercised against a stub model carrying only ``ent_coef`` and a
logger-shaped ``name_to_value`` mapping, so the tests import no torch and train
nothing. They pin the two things that are easy to get wrong: the entropy is
``abs(entropy_loss)`` in nats (never divided by ``ent_coef``), and the switch
prints exactly one line per state transition.
"""
from __future__ import annotations

import pytest

from domino.league import EntropyFloorCallback, TAG_ENTROPY_LOSS


class _DummyLogger:
    """Stand-in for an SB3 ``Logger``: only ``name_to_value`` is read."""

    def __init__(self, values: dict[str, float] | None = None) -> None:
        self.name_to_value: dict[str, float] = dict(values or {})


class _DummyModel:
    """Minimal model: a settable ``ent_coef`` and a logger mapping."""

    def __init__(self, ent_coef: float | str = 0.01, entropy_loss: float | None = None) -> None:
        self.ent_coef = ent_coef
        values = {} if entropy_loss is None else {TAG_ENTROPY_LOSS: entropy_loss}
        self.logger = _DummyLogger(values)

    def log_entropy(self, entropy: float) -> None:
        """Record a mean entropy as SB3 would (the tag is its negation)."""
        self.logger.name_to_value[TAG_ENTROPY_LOSS] = -float(entropy)


def _callback(model: _DummyModel, **kwargs) -> EntropyFloorCallback:
    """Build the callback, attach the stub model and run SB3's init hook."""
    cb = EntropyFloorCallback(**kwargs)
    cb.model = model  # type: ignore[assignment]
    cb._init_callback()
    return cb


# --------------------------------------------------------------------------- #
# Reading the entropy
# --------------------------------------------------------------------------- #
def test_entropy_is_abs_of_entropy_loss_not_divided_by_ent_coef():
    """``train/entropy_loss`` = -mean(H); dividing by ent_coef would inflate 100x.

    With ``ent_coef=0.01`` a logged ``-0.05`` is an entropy of 0.05 nats and must
    engage the floor. A value divided by ``ent_coef`` would read as 5 nats and
    wrongly look healthy.
    """
    model = _DummyModel(ent_coef=0.01, entropy_loss=-0.05)
    cb = _callback(model)
    assert cb._on_step() is True
    assert cb.last_entropy == pytest.approx(0.05)
    assert model.ent_coef == pytest.approx(0.05)
    assert cb._engaged is True


def test_missing_entropy_key_is_a_noop_that_step():
    model = _DummyModel(ent_coef=0.01, entropy_loss=None)  # tag absent
    cb = _callback(model)
    assert cb._on_step() is True
    assert cb.last_entropy is None
    assert model.ent_coef == pytest.approx(0.01)
    assert cb._engaged is False


def test_empty_logger_is_a_noop_that_step():
    model = _DummyModel(ent_coef=0.01)
    model.logger.name_to_value = {}
    cb = _callback(model)
    cb._on_step()
    assert model.ent_coef == pytest.approx(0.01)


def test_healthy_entropy_does_not_engage_the_floor():
    model = _DummyModel(ent_coef=0.01, entropy_loss=-0.5)  # 0.5 nats
    cb = _callback(model)
    cb._on_step()
    assert cb.last_entropy == pytest.approx(0.5)
    assert cb._engaged is False
    assert model.ent_coef == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# Hysteresis
# --------------------------------------------------------------------------- #
def test_floor_engages_below_min_and_keeps_the_higher_coef():
    model = _DummyModel(ent_coef=0.2, entropy_loss=-0.05)  # already above floor
    cb = _callback(model, floor=0.05)
    cb._on_step()
    assert cb._engaged is True
    assert model.ent_coef == pytest.approx(0.2), "max(current, floor) must not lower it"


def test_recovery_relaxes_back_to_base_with_a_dead_band():
    model = _DummyModel(ent_coef=0.01, entropy_loss=-0.05)
    cb = _callback(model, min_entropy=0.1, floor=0.05, recover_entropy=0.3)
    cb._on_step()
    assert cb._engaged is True and model.ent_coef == pytest.approx(0.05)

    # Inside the dead band (0.1 <= H <= 0.3) nothing changes.
    model.log_entropy(0.2)
    cb._on_step()
    assert cb._engaged is True
    assert model.ent_coef == pytest.approx(0.05)

    # Above recover_entropy the coefficient falls back to the base.
    model.log_entropy(0.4)
    cb._on_step()
    assert cb._engaged is False
    assert model.ent_coef == pytest.approx(0.01)


def test_thresholds_are_strict():
    model = _DummyModel(ent_coef=0.01, entropy_loss=-0.1)  # exactly min_entropy
    cb = _callback(model)
    cb._on_step()
    assert cb._engaged is False, "entropy == min_entropy is not below it"

    model.log_entropy(0.05)
    cb._on_step()
    assert cb._engaged is True
    model.log_entropy(0.3)  # exactly recover_entropy
    cb._on_step()
    assert cb._engaged is True, "entropy == recover_entropy is not above it"


def test_reengagement_after_a_recovery():
    model = _DummyModel(ent_coef=0.01, entropy_loss=-0.02)
    cb = _callback(model)
    cb._on_step()
    model.log_entropy(0.9)
    cb._on_step()
    assert cb._engaged is False and model.ent_coef == pytest.approx(0.01)
    model.log_entropy(0.02)
    cb._on_step()
    assert cb._engaged is True and model.ent_coef == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# base_ent_coef resolution
# --------------------------------------------------------------------------- #
def test_base_ent_coef_is_latched_from_the_model_when_unset():
    model = _DummyModel(ent_coef=0.013, entropy_loss=-0.02)
    cb = _callback(model)  # base_ent_coef=None
    assert cb.base_ent_coef == pytest.approx(0.013)
    assert cb._base_resolved == pytest.approx(0.013)
    model.log_entropy(0.9)
    cb._on_step()
    assert model.ent_coef == pytest.approx(0.013)


def test_explicit_base_ent_coef_is_honoured():
    model = _DummyModel(ent_coef=0.01, entropy_loss=-0.02)
    cb = _callback(model, base_ent_coef=0.02)
    cb._on_step()
    assert model.ent_coef == pytest.approx(0.05)
    model.log_entropy(0.9)
    cb._on_step()
    assert model.ent_coef == pytest.approx(0.02)


def test_non_numeric_ent_coef_engages_but_relaxes_defensively():
    """SB3's ``"auto"`` has no numeric base: the floor still applies, relax is a no-op."""
    model = _DummyModel(ent_coef="auto", entropy_loss=-0.02)
    cb = _callback(model)
    assert cb._base_resolved is None
    cb._on_step()
    assert model.ent_coef == pytest.approx(0.05)
    model.log_entropy(0.9)
    cb._on_step()  # must not raise
    assert cb._engaged is False
    assert model.ent_coef == pytest.approx(0.05), "no base to restore to"


# --------------------------------------------------------------------------- #
# Logging discipline
# --------------------------------------------------------------------------- #
def test_one_line_per_transition_not_per_step(capsys):
    model = _DummyModel(ent_coef=0.01, entropy_loss=-0.02)
    cb = _callback(model)
    for _ in range(50):  # still inside the floor
        cb._on_step()
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "ent_coef 0.05" in lines[0]


def test_transition_lines_describe_both_edges(capsys):
    model = _DummyModel(ent_coef=0.01, entropy_loss=-0.02)
    cb = _callback(model)
    cb._on_step()
    model.log_entropy(0.9)
    cb._on_step()
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert all("[EntropyFloorCallback]" in line for line in lines)
    assert "entropy 0.0200" in lines[0] and "ent_coef 0.05" in lines[0]
    assert "entropy 0.9000" in lines[1] and "ent_coef 0.01" in lines[1]


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #
def test_constructor_rejects_a_missing_dead_band():
    with pytest.raises(ValueError, match="recover_entropy"):
        EntropyFloorCallback(min_entropy=0.3, recover_entropy=0.1)
    with pytest.raises(ValueError, match="recover_entropy"):
        EntropyFloorCallback(min_entropy=0.1, recover_entropy=0.1)


def test_constructor_rejects_negative_settings():
    with pytest.raises(ValueError):
        EntropyFloorCallback(min_entropy=-0.1)
    with pytest.raises(ValueError):
        EntropyFloorCallback(floor=-0.01)
    with pytest.raises(ValueError):
        EntropyFloorCallback(base_ent_coef=-1.0)
