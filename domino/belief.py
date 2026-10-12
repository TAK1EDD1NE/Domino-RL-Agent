"""Auxiliary training losses (v3 item 3: the belief head).

The belief head predicts, from the **public** trunk features, which opponent holds
each unseen tile (3 x 28 = 84 logits, laid out exactly like the observation's
privileged slice). Training it jointly with PPO forces the actor's trunk to learn
hidden-information inference, which a centralised critic alone would remove the
pressure for.

Why a separate optimiser instead of subclassing ``MaskablePPO.train()``: copying
SB3's training loop into a subclass is fragile across versions (the loop changes
between SB3 releases), while SB3 exposes exactly what this needs - the policy, its
rollout buffer and its logger. The callback therefore owns a small Adam optimiser
over the belief head **and the public feature extractor**, so the auxiliary
gradient really does shape the shared trunk rather than only the head.

Ordering is safe with PPO's own optimiser: PPO zeroes its gradients at the start of
every minibatch backward, so the auxiliary gradient never leaks into the PPO step
(and the auxiliary step never uses PPO's gradient).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from .obs import OBS_DIM, PRIVILEGED_SLICE

#: TensorBoard tag for the auxiliary loss.
TAG_BELIEF_LOSS = "train/belief_loss"
#: TensorBoard tag for the belief accuracy over the sampled batch.
TAG_BELIEF_ACCURACY = "train/belief_accuracy"


class BeliefHeadCallback(BaseCallback):
    """Train the belief head (and the public trunk) on the hidden-hand labels.

    Parameters
    ----------
    coef:
        Multiplier applied to the BCE loss (reported weight of the auxiliary task).
    lr:
        Learning rate of the auxiliary optimiser.
    batch_size:
        Number of observations sampled from the rollout buffer each update.
    every:
        Run the auxiliary update every ``every`` calls (1 = every environment step,
        i.e. once per rollout since SB3 collects in ``_on_step``).
    verbose:
        SB3 verbosity.
    """

    def __init__(
        self,
        coef: float = 0.5,
        lr: float = 3e-4,
        batch_size: int = 256,
        every: int = 1,
        verbose: int = 0,
    ) -> None:
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
        self.optimizer: th.optim.Optimizer | None = None
        self._rng = np.random.default_rng(0)
        #: Diagnostics for the tests and the report.
        self.last_loss: float | None = None
        self.last_accuracy: float | None = None
        self.updates = 0

    # ---- SB3 hooks ------------------------------------------------------
    def _init_callback(self) -> None:
        """Create the auxiliary optimiser over the belief head and the public tower."""
        policy = self.model.policy
        parameters = list(policy.belief_head.parameters())
        parameters += list(policy.pi_features_extractor.parameters())
        self.optimizer = th.optim.Adam(parameters, lr=self.lr)

    def _on_step(self) -> bool:
        """Take one auxiliary gradient step on a sample of the rollout buffer."""
        if self.optimizer is None or self.n_calls % self.every != 0:
            return True
        buffer = self.model.rollout_buffer
        # Work on the buffer's raw arrays: ``buffer.observations`` is shaped
        # (buffer_size, n_envs, obs_dim) and ``pos`` counts the steps added so far,
        # so the auxiliary update runs during collection instead of only at the
        # rollout boundary. (``buffer.get`` is a minibatch *generator* here.)
        pos = int(getattr(buffer, "pos", 0))
        if pos < 1:
            return True
        raw = np.asarray(buffer.observations)[:pos].reshape(-1, OBS_DIM)
        count = min(self.batch_size, raw.shape[0])
        index = self._rng.integers(0, raw.shape[0], size=count)
        observations = th.as_tensor(raw[index], dtype=th.float32)
        targets = observations[:, PRIVILEGED_SLICE]
        logits = self.model.policy.predict_beliefs(observations)
        loss = th.nn.functional.binary_cross_entropy_with_logits(logits, targets) * self.coef
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with th.no_grad():
            predicted = (th.sigmoid(logits) > 0.5).float()
            accuracy = float((predicted == targets).float().mean())
        self.last_loss = float(loss.item())
        self.last_accuracy = accuracy
        self.updates += 1
        self.logger.record(TAG_BELIEF_LOSS, self.last_loss)
        self.logger.record(TAG_BELIEF_ACCURACY, accuracy)
        return True


__all__ = ["BeliefHeadCallback", "TAG_BELIEF_ACCURACY", "TAG_BELIEF_LOSS"]
