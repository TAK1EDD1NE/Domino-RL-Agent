"""The auxiliary belief-head callback (v3 item 3).

Regression home for two bugs the full suite caught in the training path, both of
which the short `--device cpu` smoke runs missed:

1. the batch was built on the CPU while the extractor sat on ``cuda:0``
   (``--device auto``), raising a device mismatch inside the callback;
2. a CPU rollout buffer stores *numpy* observations, so a blanket ``.detach()``
   (added to fix 1) raised ``AttributeError`` on the numpy path.

The callback must therefore work from either storage, and place the batch on the
policy's device.
"""
from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import Logger

from domino.agents import RandomAgent
from domino.belief import TAG_BELIEF_ACCURACY, TAG_BELIEF_LOSS, BeliefHeadCallback
from domino.config import TrainConfig
from domino.env import DominoEnv
from domino.obs import OBS_DIM, PRIVILEGED_SLICE
from domino.policy import DominoMaskablePolicy, policy_kwargs_for


def _model() -> MaskablePPO:
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
    # `learn()` normally creates the logger; the callback records into it, so give the
    # model one here (no output formats: the values are read back from name_to_value).
    model.set_logger(Logger(folder=None, output_formats=[]))
    return model


def _fill_buffer(model: MaskablePPO, count: int = 4, *, as_tensor: bool = False) -> None:
    """Put ``count`` sweeps of plausible observations into the rollout buffer."""
    buffer = model.rollout_buffer
    rng = np.random.default_rng(0)
    data = rng.random((count, buffer.n_envs, OBS_DIM)).astype(np.float32)
    data[:, :, PRIVILEGED_SLICE] = (rng.random(
        (count, buffer.n_envs, PRIVILEGED_SLICE.stop - PRIVILEGED_SLICE.start)
    ) > 0.5).astype(np.float32)
    if as_tensor:
        buffer.observations = th.as_tensor(buffer.observations)
    buffer.observations[:count] = data if not as_tensor else th.as_tensor(data)
    buffer.pos = count


def _run_one_update(*, as_tensor: bool) -> BeliefHeadCallback:
    model = _model()
    callback = BeliefHeadCallback(coef=0.5, lr=1e-3, batch_size=4, every=1)
    callback.init_callback(model)
    _fill_buffer(model, as_tensor=as_tensor)
    assert callback._on_step() is True
    return callback


def test_belief_callback_updates_from_a_numpy_rollout_buffer() -> None:
    """A CPU rollout buffer hands out numpy arrays: no .detach(), no .to()."""
    callback = _run_one_update(as_tensor=False)
    assert callback.updates == 1
    assert callback.last_loss is not None and callback.last_loss > 0.0
    assert 0.0 <= callback.last_accuracy <= 1.0
    recorded = callback.logger.name_to_value
    assert TAG_BELIEF_LOSS in recorded and TAG_BELIEF_ACCURACY in recorded


def test_belief_callback_updates_from_a_tensor_rollout_buffer() -> None:
    """A cuda rollout buffer holds tensors, so the same code path must accept them."""
    callback = _run_one_update(as_tensor=True)
    assert callback.updates == 1
    assert callback.last_loss is not None


def test_belief_callback_respects_the_update_interval() -> None:
    """Cadence is keyed on SB3's ``n_calls``, which ``on_step()`` increments first."""
    model = _model()
    callback = BeliefHeadCallback(every=4)
    callback.init_callback(model)
    _fill_buffer(model)
    for _ in range(8):
        assert callback.on_step() is True
    assert callback.updates == 2          # at n_calls 4 and 8

    every_step = BeliefHeadCallback(every=1)
    every_step.init_callback(model)
    for _ in range(4):
        every_step.on_step()
    assert every_step.updates == 4
