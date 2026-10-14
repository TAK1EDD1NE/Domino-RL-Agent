"""Search-to-policy distillation: turn PIMC's decisions into training targets.

The project's plateau is a teacher problem - every opponent in the curriculum is at or below
the rule bot - so the only component stronger than the bot (the search) is used as the teacher
here. ``tools/gen_pimc_targets.py`` writes one row per learner decision:

============  ==================================================================
``obs``       the 507-float observation exactly as the environment emits it
``mask``      the 56-dim legal-action mask
``q``         PIMC's value for every legal action, team-0 point differential, NaN elsewhere
``value``     the mean centralised-critic value over the sampled worlds (public-info target)
``seat/ply/deal``  bookkeeping; ``deal`` is the split unit
============  ==================================================================

Two losses share one optimiser:

* **policy** - masked cross-entropy against the expert's chosen action, optionally mixed with
  a KL against ``softmax(q / temperature)`` so the student also learns the expert's *ranking* of
  the near-equal moves rather than only its argmax;
* **value** - MSE of the public value head against the mean-critic target, which is the
  play-time value the search and the judge need.

Splitting is **by deal, never by row**: rows of one deal share the deal's hidden information, so
a random row split leaks the answer into the validation set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

#: Actions per decision (``2 * tile + end``).
ACTION_DIM: int = 56


@dataclass
class Targets:
    """One dataset of PIMC decisions."""

    obs: np.ndarray        # (N, 507) float32
    mask: np.ndarray       # (N, 56) bool
    q: np.ndarray          # (N, 56) float32, NaN where illegal
    value: np.ndarray      # (N,) float32
    seat: np.ndarray       # (N,) int32
    ply: np.ndarray        # (N,) int32
    deal: np.ndarray       # (N,) int32

    def __len__(self) -> int:
        """Number of decision rows."""
        return int(self.obs.shape[0])

    def subset(self, index: np.ndarray) -> "Targets":
        """A row-wise subset (used by the by-deal split)."""
        index = np.asarray(index, dtype=np.int64)
        return Targets(self.obs[index], self.mask[index], self.q[index], self.value[index],
                       self.seat[index], self.ply[index], self.deal[index])


def save_targets(path: str, **arrays: np.ndarray) -> str:
    """Write a target dataset as a compressed ``.npz``."""
    missing = {"obs", "mask", "q", "value", "seat", "ply", "deal"} - set(arrays)
    if missing:
        raise ValueError(f"missing arrays: {sorted(missing)}")
    np.savez_compressed(path, **arrays)
    return str(path)


def load_targets(path: str) -> Targets:
    """Read a target dataset written by :func:`save_targets`."""
    with np.load(path) as data:
        return Targets(
            obs=np.asarray(data["obs"], dtype=np.float32),
            mask=np.asarray(data["mask"], dtype=bool),
            q=np.asarray(data["q"], dtype=np.float32),
            value=np.asarray(data["value"], dtype=np.float32),
            seat=np.asarray(data["seat"], dtype=np.int32),
            ply=np.asarray(data["ply"], dtype=np.int32),
            deal=np.asarray(data["deal"], dtype=np.int32),
        )


def split_by_deal(targets: Targets, *, val_fraction: float = 0.25,
                  rng: Optional[np.random.Generator] = None) -> tuple[Targets, Targets]:
    """Split a dataset into (train, validation) **by deal**."""
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    generator = np.random.default_rng(0) if rng is None else rng
    deals = np.unique(targets.deal)
    if deals.size < 2:
        raise ValueError("at least two deals are needed to split by deal")
    shuffled = generator.permutation(deals)
    cut = int(max(1, min(deals.size - 1, round(deals.size * float(val_fraction)))))
    val_deals = set(int(deal) for deal in shuffled[:cut])
    val_index = np.array([index for index, deal in enumerate(targets.deal)
                          if int(deal) in val_deals], dtype=np.int64)
    train_index = np.array([index for index, deal in enumerate(targets.deal)
                            if int(deal) not in val_deals], dtype=np.int64)
    if train_index.size == 0 or val_index.size == 0:
        raise ValueError("the split left an empty side; use more deals")
    return targets.subset(train_index), targets.subset(val_index)


def expert_actions(targets: Targets) -> np.ndarray:
    """The expert's chosen action per row (argmax of ``q`` over the legal moves)."""
    q = np.nan_to_num(np.asarray(targets.q, dtype=np.float64), nan=-np.inf)
    q = np.where(np.asarray(targets.mask, dtype=bool), q, -np.inf)
    best = np.argmax(q, axis=1)
    if not np.isfinite(q[np.arange(q.shape[0]), best]).all():
        raise ValueError("a row has no legal move with a finite expert value")
    return best.astype(np.int64)


def expert_probabilities(targets: Targets, temperature: float = 1.0,
                         floor: float = 1e-6) -> np.ndarray:
    """Softmax over ``q / temperature`` among the legal moves, with a small uniform floor.

    The floor keeps a near-certain expert from producing infinite KL penalties and stops the
    student being pushed to absolute zero on moves the expert merely disliked.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    q = np.asarray(targets.q, dtype=np.float64) / float(temperature)
    q = np.where(np.asarray(targets.mask, dtype=bool), np.nan_to_num(q, nan=-1e9), -np.inf)
    shifted = q - np.max(q, axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights = np.where(np.asarray(targets.mask, dtype=bool), weights, 0.0)
    totals = weights.sum(axis=1, keepdims=True)
    probabilities = weights / np.where(totals > 0, totals, 1.0)
    if floor > 0:
        probabilities = (1.0 - floor) * probabilities + floor * np.asarray(
            targets.mask, dtype=np.float64)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


class DistillHistory:
    """Per-epoch numbers, for the report and for honest early stopping."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def add(self, **values: float) -> None:
        """Record one epoch's metrics."""
        self.rows.append({key: float(value) for key, value in values.items()})

    @property
    def best_agreement(self) -> float:
        """Best held-out top-1 agreement seen."""
        return max((row.get("val_agreement", 0.0) for row in self.rows), default=0.0)


def distill(model: Any, train: Targets, *, val: Optional[Targets] = None, epochs: int = 8,
            lr: float = 1e-4, batch_size: int = 128, temperature: float = 1.0,
            soft_weight: float = 0.5, value_weight: float = 0.2, weight_decay: float = 0.0,
            grad_clip: float = 5.0, device: Optional[str] = None,
            log: Optional[Callable[[str], None]] = None,
            rng: Optional[np.random.Generator] = None,
            history: Optional[DistillHistory] = None) -> DistillHistory:
    """Fine-tune ``model``'s policy on the PIMC targets; returns the per-epoch history.

    The gradient touches the public trunk, the action head and the public value head - never
    the privileged critic, which keeps its role as the training-time reference.
    """
    import torch as th

    generator = np.random.default_rng(0) if rng is None else rng
    policy = model.policy
    target_device = th.device(device or getattr(policy, "device", "cpu"))
    optimizer = th.optim.Adam(policy.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    expert = expert_actions(train)
    soft = expert_probabilities(train, temperature=temperature)
    val_expert = expert_actions(val) if val is not None else None
    val_soft = expert_probabilities(val, temperature=temperature) if val is not None else None
    tracker = history if history is not None else DistillHistory()
    rows = np.arange(len(train))

    def _batch_loss(subset: np.ndarray, targets: Targets, expert_row: np.ndarray,
                    soft_row: np.ndarray) -> tuple[Any, Any, Any]:
        obs = th.as_tensor(targets.obs[subset], dtype=th.float32, device=target_device)
        mask = th.as_tensor(targets.mask[subset], dtype=th.bool, device=target_device)
        logits = policy.predict_action_logits(obs, mask)
        wanted = th.as_tensor(expert_row[subset], dtype=th.long, device=target_device)
        loss_policy = th.nn.functional.cross_entropy(logits, wanted)
        loss_soft = th.zeros((), device=target_device)
        if soft_weight > 0:
            reference = th.as_tensor(soft_row[subset], dtype=th.float32, device=target_device)
            loss_soft = th.nn.functional.kl_div(
                th.nn.functional.log_softmax(logits, dim=-1), reference, reduction="batchmean")
        loss_value = th.zeros((), device=target_device)
        if value_weight > 0:
            values = th.as_tensor(targets.value[subset], dtype=th.float32,
                                  device=target_device).reshape(-1, 1)
            predicted = policy.predict_public_values(obs)
            loss_value = th.nn.functional.mse_loss(predicted, values)
        total = loss_policy + float(soft_weight) * loss_soft + float(value_weight) * loss_value
        return total, loss_policy, loss_value

    for epoch in range(int(epochs)):
        policy.train()
        order = generator.permutation(rows)
        totals = {"total": 0.0, "policy": 0.0, "value": 0.0}
        batches = 0
        for start in range(0, order.size, int(batch_size)):
            subset = order[start:start + int(batch_size)]
            if subset.size < 2:
                continue
            optimizer.zero_grad()
            loss, loss_policy, loss_value = _batch_loss(subset, train, expert, soft)
            loss.backward()
            th.nn.utils.clip_grad_norm_([p for p in policy.parameters()], float(grad_clip))
            optimizer.step()
            totals["total"] += float(loss.item())
            totals["policy"] += float(loss_policy.item())
            totals["value"] += float(loss_value.item())
            batches += 1
        row = {"epoch": epoch + 1,
               "train_loss": totals["total"] / max(1, batches),
               "train_ce": totals["policy"] / max(1, batches),
               "train_value_mse": totals["value"] / max(1, batches)}
        if val is not None and val_expert is not None and val_soft is not None:
            row.update(evaluate_agreement(model, val, val_expert, val_soft,
                                          batch_size=int(batch_size), device=str(target_device)))
        tracker.add(**row)
        if log is not None:
            compact = " ".join(f"{key}={value:.4f}" for key, value in row.items())
            log(f"epoch {row['epoch']}: {compact}")
    return tracker


def evaluate_agreement(model: Any, targets: Targets, expert: Optional[np.ndarray] = None,
                       soft: Optional[np.ndarray] = None, *, batch_size: int = 256,
                       device: str = "cpu") -> dict[str, float]:
    """Top-1 agreement, cross-entropy, KL and value MSE of the student on a dataset."""
    import torch as th

    policy = model.policy
    policy.eval()
    expert = expert_actions(targets) if expert is None else expert
    soft = expert_probabilities(targets) if soft is None else soft
    device_obj = th.device(device)
    agreed = 0
    ce_total = 0.0
    kl_total = 0.0
    value_total = 0.0
    count = 0
    with th.no_grad():
        for start in range(0, len(targets), int(batch_size)):
            stop = min(len(targets), start + int(batch_size))
            obs = th.as_tensor(targets.obs[start:stop], dtype=th.float32, device=device_obj)
            mask = th.as_tensor(targets.mask[start:stop], dtype=th.bool, device=device_obj)
            logits = policy.predict_action_logits(obs, mask)
            wanted = th.as_tensor(expert[start:stop], dtype=th.long, device=device_obj)
            agreement = (logits.argmax(dim=-1) == wanted)
            agreed += int(agreement.sum().item())
            ce_total += float(th.nn.functional.cross_entropy(logits, wanted).item()) * (stop - start)
            reference = th.as_tensor(soft[start:stop], dtype=th.float32, device=device_obj)
            kl_total += float(th.nn.functional.kl_div(
                th.nn.functional.log_softmax(logits, dim=-1), reference,
                reduction="batchmean").item()) * (stop - start)
            values = th.as_tensor(targets.value[start:stop], dtype=th.float32,
                                  device=device_obj).reshape(-1, 1)
            value_total += float(th.nn.functional.mse_loss(
                policy.predict_public_values(obs), values).item()) * (stop - start)
            count += stop - start
    total = max(1, count)
    return {"val_agreement": agreed / total, "val_ce": ce_total / total,
            "val_kl": kl_total / total, "val_value_mse": value_total / total}


__all__ = ["ACTION_DIM", "DistillHistory", "Targets", "distill", "evaluate_agreement",
           "expert_actions", "expert_probabilities", "load_targets", "save_targets",
           "split_by_deal"]
