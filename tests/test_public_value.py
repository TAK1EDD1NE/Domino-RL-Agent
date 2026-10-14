"""Tests for the public-information value head and its auxiliary callback.

The head is the *play-time* value: same public features as the actor, same returns as the
centralised critic. These tests pin the three things that can silently go wrong: the GAE
targets, the "only the callback trains it" contract, and checkpoint round-tripping.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import Logger

from domino.agents import RandomAgent
from domino.config import TrainConfig
from domino.env import DominoEnv
from domino.obs import OBS_DIM
from domino.policy import DominoMaskablePolicy, policy_kwargs_for
from domino.public_value import TAG_PUBLIC_VALUE_LOSS, PublicValueHeadCallback, lambda_returns


def _model() -> MaskablePPO:
    """A tiny real MaskablePPO around the custom policy (as the belief tests build it)."""
    env = DominoEnv(opponent=RandomAgent(seed=0), seed=0)
    model = MaskablePPO(
        DominoMaskablePolicy,
        env,
        policy_kwargs=policy_kwargs_for(TrainConfig()),
        n_steps=8,
        batch_size=8,
        seed=0,
        device="cpu",
    )
    model.set_logger(Logger(folder=None, output_formats=[]))
    return model


def _fill_buffer(model: MaskablePPO, count: int = 8, *, reward: float = 1.0) -> None:
    """Fill the rollout buffer with plausible observations, rewards and values.

    SB3 2.x's buffer stores ``episode_starts`` (not the 1.x ``dones`` field that
    ``stable_baselines3``'s own docs still mention in places); the GAE target must be built
    from the field that exists.
    """
    buffer = model.rollout_buffer
    rng = np.random.default_rng(0)
    buffer.observations[:count] = rng.random((count, buffer.n_envs, OBS_DIM)).astype(np.float32)
    buffer.rewards[:count] = float(reward)
    buffer.values[:count] = 0.0
    buffer.episode_starts[:count] = 0.0
    buffer.pos = count


# ---- the targets -----------------------------------------------------------
def test_lambda_returns_reproduce_the_discounted_return() -> None:
    """With gamma=1 and zero values the target is the plain return-to-go."""
    rewards = np.array([[1.0], [1.0], [1.0]])
    starts = np.zeros((3, 1))
    values = np.zeros((3, 1))
    out = lambda_returns(rewards, starts, values, gamma=1.0, gae_lambda=1.0,
                         last_values=np.zeros(1))
    assert out.reshape(-1).tolist() == pytest.approx([3.0, 2.0, 1.0])
    discounted = lambda_returns(rewards, starts, values, gamma=0.5, gae_lambda=1.0,
                                last_values=np.zeros(1))
    assert discounted.reshape(-1).tolist() == pytest.approx([1.75, 1.5, 1.0])


def test_lambda_returns_respect_episode_boundaries() -> None:
    """A new episode at the next step cuts the bootstrap, as in SB3's own GAE."""
    rewards = np.array([[1.0], [5.0]])
    starts = np.array([[1.0], [1.0]])            # step 1 starts a new episode
    values = np.zeros((2, 1))
    out = lambda_returns(rewards, starts, values, gamma=1.0, gae_lambda=1.0,
                         last_values=np.zeros(1))
    assert out.reshape(-1).tolist() == pytest.approx([1.0, 5.0])


def test_lambda_returns_use_the_documented_bootstrap_by_default() -> None:
    """Without last_values the last stored value bootstraps the final step."""
    rewards = np.array([[0.0], [2.0]])
    starts = np.zeros((2, 1))
    values = np.array([[3.0], [1.0]])
    out = lambda_returns(rewards, starts, values, gamma=1.0, gae_lambda=1.0)
    # step 1: 2 + 1*1 (bootstrap = values[-1]) - 1 = 2 -> target 3
    # step 0: 0 + values[1] - 3 = -2, plus carry 2 -> 0 -> target 3
    assert out.reshape(-1).tolist() == pytest.approx([3.0, 3.0])


def test_lambda_returns_reject_mismatched_shapes() -> None:
    """A silent shape mismatch would train the head on shifted targets."""
    with pytest.raises(ValueError):
        lambda_returns(np.zeros((2, 1)), np.zeros((3, 1)), np.zeros((2, 1)),
                       gamma=0.99, gae_lambda=0.95)


# ---- the policy ------------------------------------------------------------
def test_head_exists_and_is_in_the_policy_optimizer() -> None:
    """The head must be part of the policy (saved with it) and of PPO's optimizer."""
    model = _model()
    head = model.policy.public_value_head
    assert isinstance(head, th.nn.Linear)
    assert head.out_features == 1
    in_optimizer = {id(p) for group in model.policy.optimizer.param_groups
                    for p in group["params"]}
    assert all(id(p) in in_optimizer for p in head.parameters())
    values = model.policy.predict_public_values(np.zeros((4, OBS_DIM), dtype=np.float32))
    assert tuple(values.shape) == (4, 1)
    assert th.isfinite(values).all()


def test_ppo_updates_do_not_touch_the_public_head() -> None:
    """PPO's loss has no path to this head: the callback is its only trainer.

    Run as real (tiny) training rather than a hand-built buffer: the claim is about PPO's
    gradient, and only a real update can settle it.
    """
    model = _model()
    head_before = model.policy.public_value_head.weight.detach().clone()
    trunk_before = [p.detach().clone() for p in model.policy.pi_features_extractor.parameters()]
    model.learn(total_timesteps=16, progress_bar=False)
    head_after = model.policy.public_value_head.weight.detach().clone()
    trunk_after = [p.detach() for p in model.policy.pi_features_extractor.parameters()]
    assert th.equal(head_before, head_after)
    # ... while the shared public trunk *is* trained by PPO, which is why the auxiliary
    # gradient (and its clipping) matters.
    assert any(not th.equal(before, after) for before, after in zip(trunk_before, trunk_after))


def test_older_checkpoints_still_load_with_the_new_head() -> None:
    """The regression that matters: a new head must not make old checkpoints unloadable.

    ``MaskablePPO.load`` uses ``exact_match=True``, so shipping the public value head broke
    every checkpoint written before it - including those test.py, play.py, the harness and the
    league load. ``load_policy`` now falls back to a ``strict=False`` copy and reports the gap.
    """
    import pathlib
    import warnings

    import torch as th

    from domino.agents.pool import load_policy

    path = pathlib.Path("models/checkpoints/ppo_16002774_steps.zip")
    if not path.exists():
        pytest.skip("the trained checkpoint is not present in this working tree")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = load_policy(str(path))
    assert model.policy.predict_public_values(np.zeros((2, OBS_DIM), np.float32)).shape == (2, 1)
    reported = " ".join(str(w.message) for w in caught)
    assert "public_value_head" in reported, "the missing head must be reported, not silently zeroed"
    # ... and the weights it *does* have must really be in the loaded model.
    import io
    import zipfile

    with zipfile.ZipFile(path) as archive:
        saved = th.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu",
                        weights_only=True)
    loaded = model.policy.action_net.weight.detach()
    assert th.allclose(loaded, saved["action_net.weight"].to(loaded.device), atol=1e-6)


def test_checkpoint_round_trip_keeps_the_head() -> None:
    """Save/load must carry the new head, or evaluation silently uses a random one."""
    model = _model()
    with th.no_grad():
        model.policy.public_value_head.weight.fill_(0.25)
    obs = np.zeros((2, OBS_DIM), dtype=np.float32)
    expected = model.policy.predict_public_values(obs).detach().numpy()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.zip"
        model.save(str(path))
        loaded = MaskablePPO.load(str(path), env=DominoEnv(opponent=RandomAgent(seed=0), seed=0),
                                  device="cpu")
    got = loaded.policy.predict_public_values(obs).detach().numpy()
    assert np.allclose(expected, got)
    assert th.equal(loaded.policy.public_value_head.weight.detach(),
                    model.policy.public_value_head.weight.detach())


# ---- the callback ----------------------------------------------------------
def test_callback_takes_a_step_and_learns_its_target() -> None:
    """Twenty updates on a fixed batch must reduce the loss it reports."""
    model = _model()
    callback = PublicValueHeadCallback(coef=1.0, lr=3e-3, batch_size=8, every=1)
    callback.init_callback(model)
    _fill_buffer(model, reward=1.0)
    losses = []
    for _ in range(20):
        assert callback._on_step() is True
        losses.append(callback.last_loss)
    assert callback.updates == 20
    assert losses[-1] < losses[0] * 0.5, losses
    assert np.isfinite(callback.last_corr) or np.isnan(callback.last_corr)
    assert callback.logger is not None
    assert model.logger.name_to_value[TAG_PUBLIC_VALUE_LOSS] == pytest.approx(losses[-1])


def test_callback_does_nothing_before_any_step_is_collected() -> None:
    """An empty buffer must not raise (the first callback call happens early)."""
    model = _model()
    callback = PublicValueHeadCallback(every=1)
    callback.init_callback(model)
    model.rollout_buffer.pos = 0
    assert callback._on_step() is True
    assert callback.updates == 0


def test_callback_honours_the_every_parameter() -> None:
    """``every=4`` means one update per four callback calls.

    SB3 advances ``n_calls`` in ``BaseCallback.__call__``, which is bypassed when a test
    calls ``_on_step`` directly, so the counter is advanced here as the training loop would.
    """
    model = _model()
    callback = PublicValueHeadCallback(every=4)
    callback.init_callback(model)
    _fill_buffer(model)
    for _ in range(8):
        callback.n_calls += 1
        callback._on_step()
    assert callback.updates == 2


def test_callback_rejects_bad_configuration() -> None:
    """Negative coefficients and zero batch sizes are configuration errors."""
    with pytest.raises(ValueError):
        PublicValueHeadCallback(coef=-1.0)
    with pytest.raises(ValueError):
        PublicValueHeadCallback(lr=0.0)
    with pytest.raises(ValueError):
        PublicValueHeadCallback(batch_size=0)
    with pytest.raises(ValueError):
        PublicValueHeadCallback(every=0)
