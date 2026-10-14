"""Tests for the search-to-policy distillation path (item (e))."""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import Logger

from domino.agents import RandomAgent
from domino.config import TrainConfig
from domino.distill import (
    ACTION_DIM,
    Targets,
    distill,
    evaluate_agreement,
    expert_actions,
    expert_probabilities,
    load_targets,
    save_targets,
    split_by_deal,
)
from domino.env import DominoEnv
from domino.obs import OBS_DIM
from domino.policy import DominoMaskablePolicy, policy_kwargs_for


def _model() -> MaskablePPO:
    """A tiny real MaskablePPO around the custom policy."""
    env = DominoEnv(opponent=RandomAgent(seed=0), seed=0)
    model = MaskablePPO(
        DominoMaskablePolicy,
        env,
        policy_kwargs=policy_kwargs_for(TrainConfig()),
        n_steps=16,
        batch_size=16,
        seed=0,
        device="cpu",
    )
    model.set_logger(Logger(folder=None, output_formats=[]))
    return model


def synthetic(n_deals: int = 6, per_deal: int = 10, *, answer: int = 7,
              seed: int = 0) -> Targets:
    """A dataset whose answer is readable from the observation (dim ``answer + 28``).

    The expert value is highest for ``answer`` and the observation marks it, so a student that
    learns anything must reach a high agreement; a student that learns nothing cannot.
    """
    rng = np.random.default_rng(seed)
    rows = n_deals * per_deal
    obs = rng.random((rows, OBS_DIM)).astype(np.float32)
    obs[:, 28 + answer] = 2.0
    mask = np.zeros((rows, ACTION_DIM), dtype=bool)
    q = np.full((rows, ACTION_DIM), np.nan, dtype=np.float32)
    for row in range(rows):
        legal = rng.choice(ACTION_DIM, size=4, replace=False)
        legal[0] = answer
        mask[row, legal] = True
        q[row, legal] = rng.uniform(0.0, 1.0, size=legal.size)
        q[row, answer] = 5.0
    return Targets(obs=obs, mask=mask, q=q,
                   value=rng.normal(0.0, 5.0, size=rows).astype(np.float32),
                   seat=np.zeros(rows, dtype=np.int32),
                   ply=np.tile(np.arange(per_deal, dtype=np.int32), n_deals),
                   deal=np.repeat(np.arange(n_deals, dtype=np.int32), per_deal))


# ---- the dataset -----------------------------------------------------------
def test_targets_round_trip(tmp_path) -> None:
    """Writing and reading a dataset must preserve every field."""
    data = synthetic()
    path = tmp_path / "targets.npz"
    save_targets(str(path), obs=data.obs, mask=data.mask, q=data.q, value=data.value,
                 seat=data.seat, ply=data.ply, deal=data.deal)
    again = load_targets(str(path))
    assert len(again) == len(data) == 60
    assert np.array_equal(again.mask, data.mask)
    assert np.allclose(np.nan_to_num(again.q, nan=-1), np.nan_to_num(data.q, nan=-1))
    assert np.array_equal(again.deal, data.deal)
    with pytest.raises(ValueError):
        save_targets(str(tmp_path / "bad.npz"), obs=data.obs)


def test_split_by_deal_never_leaks_a_deal() -> None:
    """Rows of one deal share hidden information, so a deal must land on one side only."""
    data = synthetic(n_deals=8, per_deal=4)
    train, val = split_by_deal(data, val_fraction=0.25, rng=np.random.default_rng(3))
    assert train.deal.min() >= 0 and val.deal.min() >= 0
    assert not (set(train.deal.tolist()) & set(val.deal.tolist()))
    assert len(train) + len(val) == len(data)
    train2, val2 = split_by_deal(data, val_fraction=0.25, rng=np.random.default_rng(3))
    assert np.array_equal(train.deal, train2.deal) and np.array_equal(val.deal, val2.deal)
    with pytest.raises(ValueError):
        split_by_deal(synthetic(n_deals=1, per_deal=4), val_fraction=0.25)


def test_expert_actions_respect_the_mask_and_the_tie_break() -> None:
    """The expert is the masked argmax, lowest action index on a tie."""
    data = synthetic(n_deals=1, per_deal=3, answer=11)
    assert (expert_actions(data) == 11).all()
    illegal_best = synthetic(n_deals=1, per_deal=1)
    illegal_best.q[0, 40] = 99.0                      # best value, but not in the mask
    illegal_best.mask[0, 40] = False
    assert int(expert_actions(illegal_best)[0]) == 7
    empty = synthetic(n_deals=1, per_deal=1)
    empty.mask[0, :] = False
    with pytest.raises(ValueError):
        expert_actions(empty)


def test_expert_probabilities_are_a_distribution_on_the_legal_moves() -> None:
    """Softmax over q/τ: sums to one, zero outside the mask, sharper when colder."""
    data = synthetic()
    probabilities = expert_probabilities(data, temperature=1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert not probabilities[~data.mask].any()
    cold = expert_probabilities(data, temperature=0.1)
    assert cold.max(axis=1).mean() > probabilities.max(axis=1).mean()
    with pytest.raises(ValueError):
        expert_probabilities(data, temperature=0.0)


# ---- the loop --------------------------------------------------------------
def test_distillation_learns_the_expert(tmp_path) -> None:
    """End to end: a student must learn a readable expert, and never an illegal move."""
    data = synthetic(n_deals=8, per_deal=10, answer=9)
    train, val = split_by_deal(data, val_fraction=0.25, rng=np.random.default_rng(0))
    model = _model()
    before = evaluate_agreement(model, val)
    assert before["val_agreement"] < 0.6, before
    tracker = distill(model, train, val=val, epochs=6, lr=1e-3, batch_size=16,
                      soft_weight=0.5, value_weight=0.2, device="cpu")
    after = evaluate_agreement(model, val)
    assert after["val_agreement"] > 0.9, after
    assert after["val_ce"] < before["val_ce"]
    assert tracker.rows and "val_agreement" in tracker.rows[-1]
    assert tracker.rows[-1]["train_value_mse"] < 1e6
    # The masked distribution must never put mass on an illegal action.
    logits = model.policy.predict_action_logits(
        th.as_tensor(val.obs[:8], dtype=th.float32),
        th.as_tensor(val.mask[:8], dtype=th.bool))
    chosen = logits.argmax(dim=-1).numpy()
    assert val.mask[np.arange(8), chosen].all()


def test_distillation_without_the_value_term_leaves_the_head_alone() -> None:
    """``value_weight=0`` must not touch the public value head (that is PPO's or nothing)."""
    data = synthetic(n_deals=4, per_deal=6)
    train, _ = split_by_deal(data, val_fraction=0.5)
    model = _model()
    before = model.policy.public_value_head.weight.detach().clone()
    distill(model, train, epochs=1, lr=1e-2, batch_size=8, value_weight=0.0, device="cpu")
    after = model.policy.public_value_head.weight.detach().clone()
    assert th.allclose(before, after)


def test_distillation_value_term_moves_the_head() -> None:
    """With a value weight the head must actually receive gradient."""
    data = synthetic(n_deals=4, per_deal=6)
    train, _ = split_by_deal(data, val_fraction=0.5)
    model = _model()
    before = model.policy.public_value_head.weight.detach().clone()
    distill(model, train, epochs=2, lr=1e-2, batch_size=8, value_weight=1.0, device="cpu")
    after = model.policy.public_value_head.weight.detach().clone()
    assert not th.allclose(before, after)
