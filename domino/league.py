"""League-training extras that would otherwise crowd ``domino.callbacks``.

One callback lives here today: :class:`EntropyFloorCallback`, the exploration
guard for a league curriculum. Keeping a floor under the policy entropy stops a
stage from collapsing to a near-deterministic script before it has faced enough
opponents to learn against them.

Why an entropy floor at all
---------------------------
PPO maximises entropy through ``ent_coef`` times the **negated** mean entropy,
which SB3 logs as ``train/entropy_loss`` (i.e. ``-mean(H)`` in nats, spec's
measured note). A run whose entropy drifts toward zero has a policy that barely
explores: the league stops producing the diverse opponents it exists to provide
and every reward window becomes memorisation of one line of play. Raising
``ent_coef`` while entropy is low -- and relaxing it once entropy recovers -- is
a cheap, loud, per-transition lever compared to restarting the stage.

Reading the entropy
-------------------
The value is taken from ``self.model.logger.name_to_value["train/entropy_loss"]``
(the mapping SB3 refreshes on every rollout). The mean entropy is ``abs(value)``
and must **not** be divided by ``ent_coef``: doing so inflates it by 100x for the
usual ``0.01`` and produces values above the ``ln(#actions)`` bound, which reads
as a healthy policy while the real entropy is pinned at ~0. When the key is
absent -- before the first rollout, or from a logger that does not carry the
scalar -- the callback does nothing that step. There is deliberately no other
source of truth to guess from.
"""
from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

#: SB3 scalar holding the negated mean policy entropy, in nats.
TAG_ENTROPY_LOSS = "train/entropy_loss"


class EntropyFloorCallback(BaseCallback):
    """Raise ``model.ent_coef`` when policy entropy falls, relax it on recovery.

    The callback is a hysteresis switch on one scalar:

    * while the mean entropy is below ``min_entropy`` it holds
      ``model.ent_coef`` at ``max(current, floor)`` (it never *lowers* an
      already-higher coefficient);
    * once entropy recovers above ``recover_entropy`` it puts ``ent_coef`` back
      to :attr:`base_ent_coef`.

    It prints one line per *transition* -- engaged, then relaxed -- never once
    per step, so the run log records the floor's activity without flooding.

    Parameters
    ----------
    min_entropy:
        Entropy (nats) below which the floor engages. Default ``0.1``.
    floor:
        ``ent_coef`` applied while the floor is engaged, as a lower bound
        (``max(current, floor)``). Default ``0.05`` -- 5x the usual ``0.01``.
    recover_entropy:
        Entropy (nats) above which the coefficient relaxes back toward
        :attr:`base_ent_coef`. Must exceed ``min_entropy`` so the two thresholds
        form a dead band and the callback does not chatter on noise. Default
        ``0.3``.
    base_ent_coef:
        Coefficient to restore on recovery. ``None`` (default) latches the
        model's own numeric ``ent_coef`` the first time the callback runs, so the
        relaxation target is whatever the run was configured with.
    verbose:
        SB3 verbosity (transitions are printed regardless of this value).

    Notes
    -----
    A non-numeric ``ent_coef`` (SB3's ``"auto"``) is handled defensively: the
    floor still applies, but on recovery the coefficient is left untouched
    because there is no base value to restore to.
    """

    def __init__(
        self,
        min_entropy: float = 0.1,
        floor: float = 0.05,
        recover_entropy: float = 0.3,
        base_ent_coef: float | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if min_entropy < 0:
            raise ValueError(f"min_entropy must be >= 0, got {min_entropy}")
        if floor < 0:
            raise ValueError(f"floor must be >= 0, got {floor}")
        if recover_entropy <= min_entropy:
            raise ValueError(
                f"recover_entropy ({recover_entropy}) must exceed min_entropy "
                f"({min_entropy}) for hysteresis"
            )
        if base_ent_coef is not None and base_ent_coef < 0:
            raise ValueError(f"base_ent_coef must be >= 0, got {base_ent_coef}")
        self.min_entropy = float(min_entropy)
        self.floor = float(floor)
        self.recover_entropy = float(recover_entropy)
        self.base_ent_coef = None if base_ent_coef is None else float(base_ent_coef)

        #: Last mean entropy observed (nats); ``None`` until a value is read.
        self.last_entropy: float | None = None
        #: Whether the floor is currently holding ``ent_coef`` up.
        self._engaged = False
        #: Resolved relaxation target (``base_ent_coef`` or the latched value).
        self._base_resolved: float | None = None

    # ---- SB3 hooks ------------------------------------------------------
    def _init_callback(self) -> None:
        """Reset the switch and resolve the relaxation target from the model."""
        self.last_entropy = None
        self._engaged = False
        if self.base_ent_coef is not None:
            self._base_resolved = self.base_ent_coef
            return
        latched = self._numeric_ent_coef()
        self._base_resolved = latched
        if latched is not None:
            # Expose the latched value so a report can say what it relaxes to.
            self.base_ent_coef = latched

    def _on_step(self) -> bool:
        """Apply the hysteresis to the freshly logged entropy, if it is there."""
        entropy = self._read_entropy()
        if entropy is None:
            return True
        self.last_entropy = entropy
        if entropy < self.min_entropy and not self._engaged:
            self._engage(entropy)
        elif entropy > self.recover_entropy and self._engaged:
            self._relax(entropy)
        return True

    # ---- internals ------------------------------------------------------
    def _read_entropy(self) -> float | None:
        """Mean policy entropy in nats from the logger, or ``None`` if absent.

        ``train/entropy_loss`` is ``-mean(H)``, so the entropy is its absolute
        value; ``ent_coef`` is deliberately not involved (see module docstring).
        """
        logger = getattr(self.model, "logger", None)
        values = getattr(logger, "name_to_value", None)
        if not values:
            return None
        raw = values.get(TAG_ENTROPY_LOSS)
        if raw is None:
            return None
        try:
            return abs(float(raw))
        except (TypeError, ValueError):
            return None

    def _numeric_ent_coef(self) -> float | None:
        """The model's current ``ent_coef`` as a float, or ``None`` if not numeric."""
        value = getattr(self.model, "ent_coef", None)
        if isinstance(value, bool):  # guard: bool is an int subclass
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _engage(self, entropy: float) -> None:
        """Hold ``ent_coef`` at or above :attr:`floor` and announce the change."""
        current = self._numeric_ent_coef()
        target = self.floor if current is None else max(current, self.floor)
        self.model.ent_coef = float(target)
        self._engaged = True
        was = "n/a" if current is None else format(float(current), "g")
        print(
            f"[EntropyFloorCallback] entropy {entropy:.4f} nats < min "
            f"{self.min_entropy:g} -> ent_coef {float(target):g} (was {was})"
        )

    def _relax(self, entropy: float) -> None:
        """Restore :attr:`base_ent_coef` and announce the change."""
        current = self._numeric_ent_coef()
        base = self._base_resolved
        if base is not None:
            self.model.ent_coef = float(base)
        self._engaged = False
        was = "n/a" if current is None else format(float(current), "g")
        now = "kept" if base is None else format(float(base), "g")
        print(
            f"[EntropyFloorCallback] entropy {entropy:.4f} nats > recover "
            f"{self.recover_entropy:g} -> ent_coef {now} (was {was})"
        )


__all__ = ["TAG_ENTROPY_LOSS", "EntropyFloorCallback"]
