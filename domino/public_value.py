"""Public-information value head: "how good is this position, given only what I see".

The v3 critic (:class:`domino.policy.DominoPrivilegedValueExtractor`) reads the privileged
slice - the three opponents' true hands - which is the right function for *training* but not
one the agent can ever compute at play time. That left the project with no calibrated answer
to the question a search leaf, a judge, or a self-evaluation needs. This module adds the other
head, exactly as the search plan asks: a second value head on the **actor's public features**,
trained on the **same returns** as the critic, with its own small loss weight.

Why a callback and not a subclass of ``MaskablePPO.train()``: the same reason
:mod:`domino.belief` gives - copying SB3's training loop is version-fragile. The callback owns
a small Adam optimiser over the public value head **and the public feature extractor**, so the
auxiliary gradient really does shape the shared trunk.

Targets
-------
The lambda-returns :func:`lambda_returns` computes mirror SB3's own GAE placeholder math
(``stable_baselines3.common.buffers.RolloutBuffer.compute_returns_and_advantage``) over the
buffer's stored rewards / episode starts / values. One documented approximation: SB3 bootstraps
the final step with the value of the *next* observation, which does not exist yet while the
callback runs mid-collection, so the bootstrap uses the value of the last stored step instead.
That changes one step's target out of ``n_steps`` per rollout.

Loss scale
----------
An episode's return is a point difference in the tens, so this head's MSE is ~10^2 while the
belief head's BCE is ~0.5 nats. The default ``coef`` is therefore much smaller than the belief
coefficient's, and the gradient norm is clipped per step so an early wild value head cannot
kick the shared trunk out from under PPO. (PPO's own value loss uses the same scale with
``vf_coef`` 0.5, but it is a *single* optimiser; here two optimisers share the trunk.)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from .obs import OBS_DIM

#: TensorBoard tag for the auxiliary value loss.
TAG_PUBLIC_VALUE_LOSS = "train/public_value_loss"
#: TensorBoard tag for the correlation between the public head and the true-seeing critic.
TAG_PUBLIC_VALUE_CORR = "train/public_value_critic_corr"
#: Gradient-norm clip for the auxiliary step (see "Loss scale" above).
GRAD_CLIP: float = 10.0


def lambda_returns(rewards: np.ndarray, episode_starts: np.ndarray, values: np.ndarray,
                   *, gamma: float, gae_lambda: float,
                   last_values: Optional[np.ndarray] = None) -> np.ndarray:
    """SB3's GAE target (``advantages + values``) over a rollout buffer slice.

    ``rewards`` / ``episode_starts`` / ``values`` are ``(T, N)`` for ``T`` steps and ``N``
    environments. ``last_values`` bootstraps the final step; without it the last stored
    value is used (documented approximation, see the module docstring).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    starts = np.asarray(episode_starts, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if rewards.shape != starts.shape or rewards.shape != values.shape:
        raise ValueError(f"shape mismatch: {rewards.shape}, {starts.shape}, {values.shape}")
    steps, n_envs = rewards.shape
    bootstrap = values[-1] if last_values is None else np.asarray(last_values, dtype=np.float64)
    advantages = np.zeros((steps, n_envs), dtype=np.float64)
    running = np.zeros(n_envs, dtype=np.float64)
    for step in range(steps - 1, -1, -1):
        if step == steps - 1:
            next_non_terminal = 1.0
            next_values = bootstrap
        else:
            next_non_terminal = 1.0 - starts[step + 1]
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        running = delta + gamma * gae_lambda * next_non_terminal * running
        advantages[step] = running
    return advantages + values


def _as_float_array(raw: Any) -> np.ndarray:
    """Buffer slice -> float64 numpy, whether the buffer stores numpy or cuda tensors."""
    if th.is_tensor(raw):
        return raw.detach().cpu().numpy().astype(np.float64)
    return np.asarray(raw, dtype=np.float64)


class PublicValueHeadCallback(BaseCallback):
    """Train the public value head (and the public trunk) on PPO's own returns.

    Parameters
    ----------
    coef:
        Multiplier applied to the MSE loss (the auxiliary task's reported weight).
    lr:
        Learning rate of the auxiliary optimiser.
    batch_size:
        Number of rollout transitions sampled per auxiliary update.
    every:
        Run the auxiliary update every ``every`` calls (SB3 calls ``_on_step`` once per
        environment step). The belief callback measured a 3x throughput cost at ``every=1``,
        so the default trades shaping frequency for compute.
    verbose:
        SB3 verbosity.
    """

    def __init__(self, coef: float = 0.1, lr: float = 3e-4, batch_size: int = 256,
                 every: int = 8, grad_clip: float = GRAD_CLIP, verbose: int = 0) -> None:
        super().__init__(verbose)
        if coef < 0:
            raise ValueError(f"coef must be non-negative, got {coef}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.coef = float(coef)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.every = int(every)
        self.grad_clip = float(grad_clip)
        self.optimizer: th.optim.Optimizer | None = None
        self._rng = np.random.default_rng(0)
        #: Diagnostics for the tests and the report.
        self.last_loss: float | None = None
        self.last_corr: float | None = None
        self.updates = 0

    # ---- SB3 hooks ------------------------------------------------------
    def _init_callback(self) -> None:
        """Create the auxiliary optimiser over the public value head and the public tower."""
        policy = self.model.policy
        parameters = list(policy.public_value_head.parameters())
        parameters += list(policy.pi_features_extractor.parameters())
        self.optimizer = th.optim.Adam(parameters, lr=self.lr)

    def _on_step(self) -> bool:
        """One auxiliary gradient step toward PPO's own returns."""
        if self.optimizer is None or self.n_calls % self.every != 0:
            return True
        buffer = self.model.rollout_buffer
        pos = int(getattr(buffer, "pos", 0))
        if pos < 1:
            return True
        gamma = float(getattr(buffer, "gamma", 0.99))
        gae_lambda = float(getattr(buffer, "gae_lambda", 0.95))
        starts = getattr(buffer, "episode_starts", None)
        if starts is None:                                  # pragma: no cover - old SB3
            starts = np.zeros((pos, int(getattr(buffer, "n_envs", 1))), dtype=np.float32)
        targets = lambda_returns(_as_float_array(buffer.rewards[:pos]),
                                 _as_float_array(starts[:pos]),
                                 _as_float_array(buffer.values[:pos]),
                                 gamma=gamma, gae_lambda=gae_lambda).reshape(-1)
        raw = buffer.observations[:pos]
        if not th.is_tensor(raw):
            raw = th.as_tensor(np.asarray(raw), dtype=th.float32)
        raw = raw.reshape(-1, OBS_DIM)
        values = _as_float_array(buffer.values[:pos]).reshape(-1)
        count = min(self.batch_size, int(raw.shape[0]))
        index = self._rng.integers(0, int(raw.shape[0]), size=count)     # numpy, shared
        device = th.device(getattr(self.model.policy, "device", "cpu"))
        observations = raw[th.as_tensor(index)].to(device=device, dtype=th.float32).detach()
        wanted = th.as_tensor(targets[index], dtype=th.float32, device=device).reshape(-1, 1)
        predicted = self.model.policy.predict_public_values(observations)
        loss = th.nn.functional.mse_loss(predicted, wanted) * self.coef
        self.optimizer.zero_grad()
        loss.backward()
        # Clip the *whole* auxiliary parameter set: the shared trunk is the part that can
        # hurt PPO, and it receives this gradient too.
        th.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.grad_clip)
        self.optimizer.step()
        with th.no_grad():
            critic = th.as_tensor(values[index], dtype=th.float32,
                                  device=device).reshape(-1, 1)
            corr = _pearson(predicted.detach().reshape(-1, 1), critic)
        self.last_loss = float(loss.item())
        self.last_corr = corr
        self.updates += 1
        self.logger.record(TAG_PUBLIC_VALUE_LOSS, self.last_loss)
        self.logger.record(TAG_PUBLIC_VALUE_CORR, corr)
        return True


def _pearson(left: th.Tensor, right: th.Tensor) -> float:
    """Pearson correlation of two 1-column batches (``nan`` when either is constant)."""
    a = left.detach().reshape(-1).float()
    b = right.detach().reshape(-1).float()
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(a.norm() * b.norm())
    if denominator <= 0.0:
        return float("nan")
    return float((a @ b) / denominator)


__all__ = ["GRAD_CLIP", "PublicValueHeadCallback", "TAG_PUBLIC_VALUE_CORR",
           "TAG_PUBLIC_VALUE_LOSS", "lambda_returns"]
