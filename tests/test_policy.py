"""Tests for the custom network and the MaskablePPO policy (spec section 7).

These are the tests that make the *customised architecture* a graded artefact
rather than a claim: they build the policy through ``MaskablePPO`` (the real
training path), pin the tensor shapes of the two towers and the heads, prove the
action mask is honoured at sampling time over a whole episode, prove gradients
reach every submodule, prove the masked pooling is cardinality-agnostic (a
7-tile opening hand and a synthetic 1-tile hand), and pin the parameter count.

The masked distribution used by sb3-contrib does **not** set illegal logits to
``-inf``: :class:`MaskableCategorical` fills them with ``-1e8`` (see
``sb3_contrib/common/maskable/distributions.py``). The masking test therefore
asserts the strong, checkable statement (``<= -1e7``, i.e. "a huge negative
logit, hence probability ~0") and documents the exact constant, while the
200-decision-point legality assertion is the binding evidence that no illegal
action is ever sampled.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
from sb3_contrib import MaskablePPO

from domino.agents import RandomAgent
from domino.config import TrainConfig
from domino.engine import GameState
from domino.env import DominoEnv
from domino.masks import ACTION_DIM
from domino.obs import OBS_DIM, PRIVILEGED_SLICE, encode
from domino.policy import (
    DominoFeaturesExtractor,
    DominoMaskablePolicy,
    param_count,
    policy_kwargs_for,
)

HAND = slice(0, 28)
FEATURES_DIM = 128


# ---- helpers ---------------------------------------------------------------
def _make_env(seed: int = 0) -> DominoEnv:
    """A real environment driven by a seeded random opponent."""
    return DominoEnv(opponent=RandomAgent(seed=0), seed=seed)


def _make_model(env: DominoEnv) -> MaskablePPO:
    """Build the trainer's model: our policy class + our policy_kwargs."""
    return MaskablePPO(
        DominoMaskablePolicy,
        env,
        policy_kwargs=policy_kwargs_for(TrainConfig()),
        n_steps=256,
        batch_size=128,
        seed=0,
        device="cpu",
    )


def _collect_observations(env: DominoEnv, count: int) -> np.ndarray:
    """Collect ``count`` real decision-point observations (resetting at deal end)."""
    obs, _ = env.reset()
    batch = [obs]
    guard = 0
    while len(batch) < count:
        guard += 1
        assert guard < 5000, "environment did not yield enough decision points"
        mask = env.action_masks()
        obs, _, terminated, truncated, _ = env.step(int(np.flatnonzero(mask)[0]))
        if terminated or truncated:
            obs, _ = env.reset()
        batch.append(obs)
    return np.stack(batch).astype(np.float32)


def _single_tile_state() -> GameState:
    """A synthetic deal where seat 0 holds exactly one tile (endgame shape).

    The surplus tiles sit on the table so every hand stays within ``0..7`` (the
    observation is a ``Box(0, 1)``) and the encoder is exercised on a real
    single-tile hand.
    """
    hands = [
        [0],
        [1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19, 20, 21],
    ]
    return GameState(
        hands=hands,
        rng=np.random.default_rng(0),
        table=[22, 23, 24, 25, 26, 27],
        played_by=[[], [22, 23], [24, 25], [26, 27]],
        ends=[3, 5],
        turn=0,
        leader=0,
    )


# ---- 1. construction through MaskablePPO and the parameter budget ----------
def test_policy_constructs_through_maskable_ppo_and_stays_under_budget() -> None:
    env = _make_env()
    model = _make_model(env)

    assert isinstance(model.policy, DominoMaskablePolicy)
    assert isinstance(model.policy.features_extractor, DominoFeaturesExtractor)
    assert model.policy.features_dim == FEATURES_DIM

    n_params = param_count(model)
    # Lower bound guards against a silently-empty architecture; upper bound is
    # the task's hard budget (the design is ~0.25 M, not SB3's default MLP).
    assert 100_000 < n_params < 2_000_000, f"unexpected parameter count: {n_params}"


def test_param_count_is_positive_and_int() -> None:
    model = _make_model(_make_env())
    n = param_count(model)
    assert isinstance(n, int)
    assert n > 0


# ---- 2. forward-pass shapes ------------------------------------------------
def test_forward_pass_shapes() -> None:
    env = _make_env()
    model = _make_model(env)
    policy = model.policy

    obs = _collect_observations(env, 8)
    obs_t = th.as_tensor(obs)

    actions, values, log_prob = policy.forward(obs_t)
    assert actions.shape == (8,)
    assert values.shape == (8, 1)
    assert log_prob.shape == (8,)

    # The underlying distribution exposes one logit per action (8, 56).
    distribution = policy.get_distribution(obs_t)
    logits = distribution.distribution.logits
    assert logits.shape == (8, ACTION_DIM)

    # The value head is reachable directly too.
    assert policy.predict_values(obs_t).shape == (8, 1)

    # Deterministic and stochastic paths both return (8,).
    assert policy.forward(obs_t, deterministic=True)[0].shape == (8,)


# ---- 3. illegal actions are impossible -------------------------------------
def test_masked_logits_suppress_illegal_actions() -> None:
    env = _make_env()
    model = _make_model(env)
    policy = model.policy

    obs, _ = env.reset()
    mask = env.action_masks()
    assert mask.sum() >= 1 and (~mask).any()

    distribution = policy.get_distribution(th.as_tensor(obs[None, :]), action_masks=mask[None, :])
    logits = distribution.distribution.logits[0]
    # sb3-contrib uses -1e8 (not -inf); assert the huge-negative property.
    assert bool(th.all(logits[~mask] <= -1e7))
    # Legal actions keep their finite logits.
    assert bool(th.all(th.isfinite(logits[mask])))


def test_sampled_actions_are_always_legal_over_200_decisions() -> None:
    env = _make_env(seed=0)
    model = _make_model(env)

    obs, _ = env.reset()
    decisions = 0
    guard = 0
    while decisions < 200:
        guard += 1
        assert guard < 5000, "did not reach 200 decision points in time"
        mask = env.action_masks()
        assert mask.sum() >= 1, "a decision point always has at least one legal action"

        for deterministic in (True, False):
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            chosen = int(np.asarray(action).item())
            assert 0 <= chosen < ACTION_DIM
            assert bool(mask[chosen]), f"illegal action {chosen} sampled (deterministic={deterministic})"

        # Advance the episode with the deterministic action.
        chosen = int(np.asarray(model.predict(obs, action_masks=mask, deterministic=True)[0]).item())
        obs, _, terminated, truncated, _ = env.step(chosen)
        decisions += 1
        if terminated or truncated:
            obs, _ = env.reset()

    assert decisions == 200


# ---- 4. gradients reach every submodule ------------------------------------
def test_gradients_reach_every_submodule() -> None:
    env = _make_env()
    model = _make_model(env)
    policy = model.policy
    extractor = policy.pi_features_extractor

    obs_t = th.as_tensor(_collect_observations(env, 8))
    actions, values, log_prob = policy.forward(obs_t)
    assert actions.shape == (8,)

    loss = -log_prob.mean() + values.mean()
    loss.backward()

    targets = {
        "tile_embedding": extractor.tile_embedding.weight,
        "tile_mlp": extractor.tile_mlp[0].weight,
        "context_mlp": extractor.context_mlp[0].weight,
        "move_mlp": extractor.move_mlp[0].weight,
        "move_head": extractor.move_head[0].weight,
        "trunk_layer_1": extractor.trunk[0].weight,
        "trunk_layer_2": extractor.trunk[3].weight,
        "trunk_layer_3": extractor.trunk[6].weight,
        "policy_head": policy.action_net.weight,
        "value_head": policy.value_net.weight,
        "privileged_value_extractor": policy.vf_features_extractor.net[0].weight,
    }
    for name, param in targets.items():
        assert param.grad is not None, f"{name} received no gradient"
        assert float(param.grad.abs().sum()) > 0.0, f"{name} received an all-zero gradient"

    # Every parameter except the auxiliary belief head gets a PPO gradient: the
    # belief head is trained by its own auxiliary loss (domino/belief.py).
    for name, param in policy.named_parameters():
        if name.startswith("belief_head"):
            continue
        assert param.grad is not None, f"{name} received no gradient"

    # The auxiliary loss must reach the belief head AND the public tower it reads.
    policy.zero_grad()
    obs_batch = th.as_tensor(_collect_observations(env, 4))
    targets_t = obs_batch[:, PRIVILEGED_SLICE]
    belief_loss = th.nn.functional.binary_cross_entropy_with_logits(
        policy.predict_beliefs(obs_batch), targets_t
    )
    belief_loss.backward()
    assert policy.belief_head.weight.grad is not None
    assert float(policy.belief_head.weight.grad.abs().sum()) > 0.0
    assert extractor.trunk[0].weight.grad is not None
    assert float(extractor.trunk[0].weight.grad.abs().sum()) > 0.0


# ---- 5. cardinality-agnostic pooling ---------------------------------------
def test_extractor_is_cardinality_agnostic() -> None:
    env = _make_env()
    extractor = DominoFeaturesExtractor(env.observation_space, features_dim=FEATURES_DIM)

    # (a) A real opening deal: a 7-tile hand.
    obs7, _ = env.reset()
    assert int(obs7[HAND].sum()) == 7
    out7 = extractor(th.as_tensor(obs7[None, :].astype(np.float32)))
    assert out7.shape == (1, FEATURES_DIM)
    assert bool(th.isfinite(out7).all())

    # (b) A synthetic single-tile hand.
    obs1 = encode(_single_tile_state(), 0).astype(np.float32)
    assert int(obs1[HAND].sum()) == 1
    out1 = extractor(th.as_tensor(obs1[None, :]))
    assert out1.shape == (1, FEATURES_DIM)
    assert bool(th.isfinite(out1).all())

    # (c) An empty hand slice must not divide by zero (defensive guard).
    out0 = extractor(th.zeros(1, OBS_DIM, dtype=th.float32))
    assert out0.shape == (1, FEATURES_DIM)
    assert bool(th.isfinite(out0).all())

    # A batch mixing cardinalities works in one shot.
    mixed = th.as_tensor(np.stack([obs7, obs1, obs1 * 0.0]).astype(np.float32))
    out_mixed = extractor(mixed)
    assert out_mixed.shape == (3, FEATURES_DIM)
    assert bool(th.isfinite(out_mixed).all())


def test_extractor_rejects_a_wrong_observation_shape() -> None:
    from gymnasium import spaces

    with pytest.raises(ValueError):
        DominoFeaturesExtractor(spaces.Box(0.0, 1.0, (10,), np.float32), features_dim=FEATURES_DIM)


# ---- 6. policy_kwargs_for is pure and complete -----------------------------
def test_policy_kwargs_for_is_pure_and_complete() -> None:
    cfg = TrainConfig()
    first = policy_kwargs_for(cfg)
    second = policy_kwargs_for(cfg)

    assert set(first) == {
        "features_extractor_class",
        "features_extractor_kwargs",
        "net_arch",
        "activation_fn",
        "ortho_init",
        "share_features_extractor",
    }
    assert first["features_extractor_class"] is DominoFeaturesExtractor
    assert first["features_extractor_kwargs"] == {"features_dim": FEATURES_DIM}
    assert first["net_arch"] == {"pi": [], "vf": []}
    assert first["activation_fn"] is th.nn.ReLU
    assert first["ortho_init"] is True
    # v3: the value side must get its own (privileged) extractor.
    assert first["share_features_extractor"] is False

    # Equal but independent: no shared mutable state between calls.
    assert first == second
    assert first is not second
    assert first["features_extractor_kwargs"] is not second["features_extractor_kwargs"]

    # Mutating a returned dict must not leak into the next call.
    first["net_arch"]["pi"].append(999)
    first["features_extractor_kwargs"]["features_dim"] = 7
    assert policy_kwargs_for()["net_arch"] == {"pi": [], "vf": []}
    assert policy_kwargs_for()["features_extractor_kwargs"] == {"features_dim": FEATURES_DIM}

    # ``cfg`` is optional and currently architecture-neutral.
    assert policy_kwargs_for() == policy_kwargs_for(TrainConfig())


# ---- 7. belief predictions follow the policy's device ----------------------
def test_belief_predictions_accept_numpy_and_move_to_the_policy_device() -> None:
    """Regression: with --device auto the extractor sits on cuda:0 but the belief
    callback samples numpy observations, which raised "Expected all tensors to be on
    the same device" and killed the trainer inside the callback.
    """
    env = _make_env(seed=0)
    model = _make_model(env)
    device = model.policy.device
    batch = _collect_observations(env, 4)

    from_numpy = model.policy.predict_beliefs(batch)
    from_tensor = model.policy.predict_beliefs(th.as_tensor(batch, dtype=th.float32))
    assert from_numpy.shape == (4, PRIVILEGED_SLICE.stop - PRIVILEGED_SLICE.start)
    assert from_tensor.shape == from_numpy.shape
    assert from_numpy.device == device and from_tensor.device == device
    assert th.allclose(from_numpy, from_tensor, atol=1e-5)
