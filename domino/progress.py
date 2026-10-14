"""In-training progress gate: measure the *live* policy's strength while the run is still going.

Why this module exists
----------------------
A reward window can only tell you that the optimiser is fitting *something*: it rises, flattens
and collapses identically well when the agent learns the task and when it memorises four frozen
deals. The project has already paid for that lesson once (a stage-3 window sat at a stable
+32.5 points/deal while the independent evaluation read 44.3%, i.e. random level). Waiting for a
ten-hour chain to finish before discovering that is the failure this callback removes.

Measured cost (this box, CPU, one process): playing a deal with the trained net in the loop costs
~23 ms, so a 200-deal evaluation against one opponent is ~2.3 s of play per 100 games
(``test.play_deal_with_agents``, 200 deals = 4.6 s). At the default cadence of 250k steps
(~5 min at ~800 steps/s) an evaluation every 250k steps costs a few percent of the wall clock.

What it measures
----------------
The **duplicate-deal protocol** of :mod:`domino.pimc.harness`, reused rather than re-implemented
(same ``deal_hands`` / ``state_from_hands`` / ``play_deal`` / ``rotate_hands`` / ``wilson_interval``
/ ``bootstrap_mean_ci``): every deal is played twice, once as dealt and once with the seating
rotated by one seat, the learner keeping the team-0 seats in both games. A void deal is redealt for
the *pair*, never for a single orientation. The headline number is the **paired differential** --
the mean over deals of ``(diff_game1 + diff_game2) / 2``, whose bootstrap interval is taken over
deals, the independent unit. That is the quantity in which the deal's tile luck cancels, and the
only one with the resolution to see a few points per deal of real progress.

Contrast, measured on the distilled checkpoint (2026-10-14): a plain 100-deal ``test.py`` row on
seed 0 gave 39.0% [30.0, 48.8] vs the rule bot while the duplicate protocol on 30 deals gave 65.0%
[52.4, 75.8]. One cheap win rate is a coin flip's worth of information; the paired differential
pooled over a fixed seed set -- the same deals in the same order on every evaluation, so
consecutive rows differ only by the policy -- is what a training curve needs.

Anchors
-------
``rule`` (the assignment's gate), ``random`` (a floor, and the calibration for the rule bot's own
57.5%) and ``anchor`` -- the run's own starting weights, snapshotted once into
``progress_anchor_<run_name>.zip`` on the first evaluation. The early anchor is the one that
measures *progress*: standing against the rule bot saturates, standing against your own starting
point does not. It is deliberately not read from the league snapshot directory, which
``SnapshotCallback`` prunes to the newest ``league_size`` files -- that deletes a fresh run's
``policy_0.zip`` in the same call that writes it, so "the earliest snapshot" resolves to a
*previous* run's near-final policy.

Verdicts
--------
``KEEP``          nothing wrong and the anchor is still moving.
``NO-GAIN``       the rule-anchor paired interval includes zero and the win rate did not rise.
                  Informational: this callback never stops a healthy run for stalling.
``BROKEN``        the frozen-deal signature (a full diversity window holding <= 2 distinct deals)
                  or an entropy collapse with no gradient signal left (entropy < 0.02 nats with
                  kl, clip fraction and policy loss all ~0). With ``stop_on_broken`` the run halts
                  after ``broken_patience`` consecutive readings instead of burning the rest of
                  the budget.
``UNKNOWN``       the metrics this verdict needs are not in the logger yet (first rollouts).

Interface notes
---------------
* The evaluation runs **in process** against the live policy. No second trainer, no temporary
  checkpoint, no shelling out: a competing process would starve the trainer for CPU and a temp
  checkpoint would need a path the tests own.
* It draws from its own generator, seeded by ``seed``, and never touches the training RNG, the
  rollout buffer or ``model.num_timesteps``. The policy is switched to ``eval()`` and back, and the
  whole evaluation runs under ``torch.no_grad()``.
* The CSV is ``results/progress_<run>.csv``: one row per (evaluation, anchor), appended, so a
  resumed run continues the curve instead of truncating it.
"""
from __future__ import annotations

import csv
import pathlib
import re
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .agents import PolicyAgent, RandomAgent, RuleAgent
from .callbacks import _tensorboard_writer
from .engine import SEATS
from .pimc.harness import (
    MAX_PAIR_ATTEMPTS,
    bootstrap_mean_ci,
    deal_hands,
    play_deal,
    rotate_hands,
    wilson_interval,
)

#: Anchor names this callback understands.
ANCHOR_RULE = "rule"
ANCHOR_RANDOM = "random"
ANCHOR_EARLY = "anchor"
DEFAULT_ANCHORS: tuple[str, ...] = (ANCHOR_RULE, ANCHOR_RANDOM, ANCHOR_EARLY)

#: Seats the learner always occupies (team A), in both orientations of a pair.
LEARNER_SEATS: tuple[int, int] = (0, 2)

CSV_HEADER: tuple[str, ...] = (
    "step", "anchor", "deals", "games", "wins", "win_rate", "ci_low", "ci_high",
    "win_rate_dealt", "win_rate_rotated", "points_per_game", "paired_differential",
    "paired_low", "paired_high", "block_rate", "void_rate", "mean_pips_left",
    "entropy", "approx_kl", "verdict", "eval_seconds",
)

#: TensorBoard scalar prefix.
TAG_PREFIX = "progress"

# ---- verdict vocabulary ----------------------------------------------------
VERDICT_KEEP = "KEEP"
VERDICT_NO_GAIN = "NO-GAIN"
VERDICT_BROKEN = "BROKEN"
VERDICT_UNKNOWN = "UNKNOWN"

# ---- health thresholds (the frozen-deal / collapsed-policy signature) ------
#: Same threshold :class:`domino.callbacks.EpisodeStatsCallback` warns on.
LOW_DIVERSITY_THRESHOLD: int = 2
#: Entropy below this (nats) counts as collapsed *only* together with a dead gradient.
MIN_ENTROPY_NATS: float = 0.02
KL_EPS: float = 1e-5
CLIP_EPS: float = 1e-6
PG_EPS: float = 1e-6
#: The rule bot's own measured ceiling against random (``rule_vs_random`` rows), quoted in the
#: log line when the agent sits inside this band and has stopped moving.
RULE_BOT_CEILING: float = 0.575
CEILING_BAND: tuple[float, float] = (0.54, 0.605)

_CHECKPOINT_RE = re.compile(r"policy_(\d+)\.zip$")


# --------------------------------------------------------------------------- #
# metrics read out of the trainer's own logger
# --------------------------------------------------------------------------- #
def read_health(logger: Any) -> dict[str, float | None]:
    """Entropy (nats), ``approx_kl``, clip fraction and policy loss from SB3's logger.

    SB3 records ``train/entropy_loss`` as ``-mean(policy entropy)`` in nats, so the entropy is the
    negation of that value -- dividing by ``ent_coef`` inflates it by 100x and is a documented trap
    in this project. A missing key yields ``None`` rather than 0.0, so "not logged yet" is never
    read as "collapsed".
    """
    values = getattr(logger, "name_to_value", {}) or {}

    def get(key: str) -> float | None:
        raw = values.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    entropy_loss = get("train/entropy_loss")
    return {
        "entropy": None if entropy_loss is None else -entropy_loss,
        "approx_kl": get("train/approx_kl"),
        "clip_fraction": get("train/clip_fraction"),
        "policy_gradient_loss": get("train/policy_gradient_loss"),
    }


def classify(
    *,
    entropy: float | None,
    approx_kl: float | None,
    clip_fraction: float | None,
    policy_gradient_loss: float | None,
    distinct_deals: int | None = None,
    window_full: bool = False,
    rule_win_rate: float | None = None,
    rule_paired_low: float | None = None,
    previous_rule_win_rate: float | None = None,
) -> tuple[str, str]:
    """Turn one evaluation's numbers into ``(verdict, reason)``.

    Pure function: no model, no logger, no filesystem, so every branch is unit-testable.
    """
    if (
        distinct_deals is not None
        and window_full
        and int(distinct_deals) <= LOW_DIVERSITY_THRESHOLD
    ):
        return (
            VERDICT_BROKEN,
            f"only {int(distinct_deals)} distinct deal(s) in a full diversity window: the "
            "environment is replaying deals and the reward window is memorisation, not skill",
        )

    collapsed_entropy = entropy is not None and float(entropy) < MIN_ENTROPY_NATS
    dead_kl = approx_kl is None or abs(float(approx_kl)) < KL_EPS
    dead_clip = clip_fraction is None or float(clip_fraction) < CLIP_EPS
    dead_pg = policy_gradient_loss is None or abs(float(policy_gradient_loss)) < PG_EPS
    if collapsed_entropy and dead_kl and dead_clip and dead_pg:
        return (
            VERDICT_BROKEN,
            f"entropy {float(entropy):.4f} nats with approx_kl/clip/policy-loss all ~0: no "
            "gradient signal left, so more steps cannot help",
        )

    if rule_win_rate is None or rule_paired_low is None:
        return VERDICT_UNKNOWN, "rule anchor not measured yet"

    improved = (
        previous_rule_win_rate is not None
        and float(rule_win_rate) > float(previous_rule_win_rate) + 1e-9
    )
    if float(rule_paired_low) <= 0.0 and not improved:
        return (
            VERDICT_NO_GAIN,
            "paired differential vs the rule bot includes zero and the win rate did not rise",
        )
    return VERDICT_KEEP, "anchors moving or the paired interval is clear of zero"


# --------------------------------------------------------------------------- #
# the duplicate-deal evaluation (harness protocol, live policy)
# --------------------------------------------------------------------------- #
def play_duplicate_pairs(
    learner_factory: Callable[[int], Any],
    opponent_factory: Callable[[int], Any],
    *,
    deals: int,
    seed: int,
    bootstrap_draws: int = 1000,
) -> dict[str, float | int]:
    """Play ``deals`` duplicate pairs and return the project's paired-deal metrics.

    This is :func:`domino.pimc.harness.play_duplicate_pair` generalised to take *factories* instead
    of picklable specs, so an in-process live policy can be seated. The protocol is otherwise
    identical -- same hands from the same generator, the same rotation, a pair redealt as a pair
    when it voids, the same differential sign convention -- and a test pins this function against
    the harness on identical seeds so the two cannot drift apart.

    ``hands`` are dealt once per attempt and *rotated* for the second game, so both games of a pair
    see the same tiles with the seats shifted; the learner keeps the team-0 seats in both, and a
    void deal is redealt for the pair.
    """
    if deals < 1:
        raise ValueError(f"deals must be >= 1, got {deals}")
    rng = np.random.default_rng(int(seed))
    wins = [0, 0]
    points = [0, 0]
    differentials: list[float] = []
    pips_left: list[int] = []
    blocked = 0
    voids = 0

    def differential(outcome: Any) -> int:
        if outcome.winner_team == 0:
            return int(outcome.points)
        if outcome.winner_team == 1:
            return -int(outcome.points)
        return 0

    for _ in range(int(deals)):
        for _attempt in range(MAX_PAIR_ATTEMPTS):
            hands = deal_hands(rng)
            agents = {
                seat: (learner_factory(int(seed) + seat)
                       if seat in LEARNER_SEATS else opponent_factory(int(seed) + seat))
                for seat in range(SEATS)
            }
            outcome_a, final_a = play_deal(agents, hands, rng)
            rotated = rotate_hands(hands)
            agents_rotated = {
                seat: (learner_factory(int(seed) + 1000 + seat)
                       if seat in LEARNER_SEATS else opponent_factory(int(seed) + 1000 + seat))
                for seat in range(SEATS)
            }
            outcome_b, final_b = play_deal(agents_rotated, rotated, rng)
            if not outcome_a.void and not outcome_b.void:
                voids += int(bool(outcome_a.void)) + int(bool(outcome_b.void))
                break
            voids += int(bool(outcome_a.void)) + int(bool(outcome_b.void))
        else:                                                  # pragma: no cover - unlikely
            raise RuntimeError(f"duplicate pair for seed {seed} voided {voids} times in a row")

        for game, (outcome, final) in enumerate(((outcome_a, final_a), (outcome_b, final_b))):
            won = int(outcome.winner_team == 0)
            wins[game] += won
            points[game] += int(outcome.points) if won else 0
            blocked += int(bool(outcome.blocked))
            pips_left.append(int(sum(final.pips(seat) for seat in LEARNER_SEATS)))
        differentials.append((differential(outcome_a) + differential(outcome_b)) / 2.0)

    games = 2 * int(deals)
    total_wins = wins[0] + wins[1]
    win_rate = total_wins / games
    low, high = wilson_interval(total_wins, games)
    mean_diff = float(np.mean(differentials)) if differentials else 0.0
    boot_low, boot_high = bootstrap_mean_ci(
        differentials, np.random.default_rng(int(seed) + 7), draws=int(bootstrap_draws)
    )
    return {
        "deals": int(deals),
        "games": games,
        "wins": total_wins,
        "win_rate": win_rate,
        "ci_low": low,
        "ci_high": high,
        "win_rate_dealt": wins[0] / int(deals),
        "win_rate_rotated": wins[1] / int(deals),
        "points_per_game": (points[0] + points[1]) / games,
        "paired_differential": mean_diff,
        "paired_low": boot_low,
        "paired_high": boot_high,
        "block_rate": blocked / games,
        "void_rate": voids / (int(deals) + voids) if (int(deals) + voids) else 0.0,
        "mean_pips_left": float(np.mean(pips_left)) if pips_left else 0.0,
    }


def resolve_early_anchor(snapshot_dir: str | pathlib.Path, *, since_step: int) -> str | None:
    """Path of the earliest league snapshot written at or after ``since_step`` (``None`` if none).

    A convenience for tooling that wants to pick a frozen opponent out of the league directory
    (``test.py --model-b``, a manual ladder). :class:`PeriodicEvalCallback` does **not** use it:
    ``SnapshotCallback._prune`` keeps only the newest ``league_size`` snapshots, so a fresh run's
    ``policy_0.zip`` is deleted in the same call that writes it and this resolver would return a
    previous run's checkpoint.
    """
    directory = pathlib.Path(snapshot_dir)
    if not directory.is_dir():
        return None
    candidates: list[tuple[int, pathlib.Path]] = []
    for path in directory.glob("policy_*.zip"):
        match = _CHECKPOINT_RE.search(path.name)
        if match is not None and int(match.group(1)) >= int(since_step):
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return str(min(candidates, key=lambda item: item[0])[1])


# --------------------------------------------------------------------------- #
# the callback
# --------------------------------------------------------------------------- #
class PeriodicEvalCallback(BaseCallback):
    """Evaluate the live policy against fixed anchors every ``every`` steps.

    Parameters
    ----------
    run_name:
        Run name; the CSV is ``results/progress_<run_name>.csv``.
    results_dir:
        Directory for the progress CSV.
    every:
        Evaluation cadence in environment steps. Also the cadence of the health check that can
        halt a broken run, so a collapse is caught within ``every`` steps (~5 min at 250k).
    deals:
        Duplicate pairs per anchor (each pair is two games).
    anchors:
        Subset of :data:`DEFAULT_ANCHORS` to measure.
    anchor_path:
        Explicit checkpoint for the ``anchor`` leg. ``None`` (the default) snapshots the run's
        **own starting weights** into ``progress_anchor_<run_name>.zip`` next to the CSV and uses
        that -- the league's snapshot directory cannot be used for this, because
        :meth:`domino.callbacks.SnapshotCallback.save` prunes it to the newest ``league_size``
        files and the fresh run's ``policy_0.zip`` is the first to go.
    anchor_dir:
        Where the run's own starting snapshot is written (default: ``results_dir``).
    stats:
        The run's :class:`~domino.callbacks.EpisodeStatsCallback`, read for the frozen-deal
        guard's diversity window. Without it the diversity half of the verdict is skipped and the
        log line says so.
    stop_on_broken:
        Halt training after ``broken_patience`` consecutive ``BROKEN`` readings.
    broken_patience:
        Consecutive broken readings required before halting.
    seed:
        Seed of the evaluation's deal stream. Fixed on purpose: every evaluation plays the same
        deals in the same order, so consecutive rows differ only by the policy.
    bootstrap_draws:
        Bootstrap draws for the paired interval.
    """

    def __init__(
        self,
        run_name: str,
        *,
        results_dir: str | pathlib.Path = "results",
        every: int = 250_000,
        deals: int = 200,
        anchors: Sequence[str] = DEFAULT_ANCHORS,
        anchor_path: str | None = None,
        anchor_dir: str | pathlib.Path | None = None,
        stats: Any | None = None,
        stop_on_broken: bool = True,
        broken_patience: int = 2,
        seed: int = 12345,
        bootstrap_draws: int = 1000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if int(every) < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        if int(deals) < 1:
            raise ValueError(f"deals must be >= 1, got {deals}")
        if int(broken_patience) < 1:
            raise ValueError(f"broken_patience must be >= 1, got {broken_patience}")
        unknown = set(anchors) - set(DEFAULT_ANCHORS)
        if unknown:
            raise ValueError(f"unknown anchor(s) {sorted(unknown)}; allowed: {list(DEFAULT_ANCHORS)}")
        self.run_name = str(run_name)
        self.results_dir = pathlib.Path(results_dir)
        self.every = int(every)
        self.deals = int(deals)
        self.anchors = tuple(anchors)
        self.anchor_path = None if anchor_path is None else str(anchor_path)
        self.anchor_dir = pathlib.Path(anchor_dir) if anchor_dir is not None else self.results_dir
        self.stats = stats
        self.stop_on_broken = bool(stop_on_broken)
        self.broken_patience = int(broken_patience)
        self.seed = int(seed)
        self.bootstrap_draws = int(bootstrap_draws)

        self.csv_path: pathlib.Path = self.results_dir / f"progress_{self.run_name}.csv"
        #: Where the run's own starting policy is snapshotted for the ``anchor`` leg.
        self.anchor_copy_path: pathlib.Path = (
            self.anchor_dir / f"progress_anchor_{self.run_name}.zip"
        )
        #: Rows written so far, in order (each evaluation writes one row per anchor).
        self.rows: list[dict[str, Any]] = []
        #: Seconds spent in the last evaluation (the overhead account).
        self.last_eval_seconds: float | None = None

        self._file: Any | None = None
        self._writer: Any | None = None
        self._owns_writer = False
        self._next_eval: int | None = None
        self._anchor_model: Any | None = None
        self._resolved_anchor: str | None = None
        self._broken_streak = 0
        #: Set when a broken run must end; read by :meth:`_on_step` (SB3 2.7.1 honours no stop flag).
        self._stop_requested = False
        self._previous: dict[str, dict[str, Any]] = {}

    # ---- SB3 hooks ------------------------------------------------------
    def _init_callback(self) -> None:
        """Resolve the anchors, open the CSV mirror and schedule the first evaluation."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        start = int(self.num_timesteps)
        if ANCHOR_EARLY in self.anchors and self._anchor_model is None:
            self._resolve_early_anchor()

        if self.csv_path.exists():
            self._file = self.csv_path.open("a", newline="", encoding="utf-8")
        else:
            self._file = self.csv_path.open("w", newline="", encoding="utf-8")
            self._file.write(",".join(CSV_HEADER) + "\n")
        self._file.flush()

        self._writer, self._owns_writer = _tensorboard_writer(self.logger)
        self._next_eval = ((start // self.every) + 1) * self.every
        games = 2 * self.deals
        legs = [name for name in self.anchors
                if name != ANCHOR_EARLY or self._anchor_model is not None]
        print(
            f"[progress] every {self.every:,} steps: {len(legs)} anchor(s) "
            f"({', '.join(legs)}) x {self.deals} duplicate pairs ({games} games), fixed deal seed "
            f"{self.seed}; early anchor = {self.anchor_path or self._resolved_anchor or 'none'}; "
            f"first evaluation at step {self._next_eval:,}"
        )

    def _resolve_early_anchor(self) -> None:
        """Load the run's own starting policy, snapshotting it once into ``anchor_copy_path``.

        The league snapshot directory cannot serve this purpose: ``SnapshotCallback.save`` prunes
        to the newest ``league_size`` files in the same call that writes the initial snapshot, so
        a fresh run's ``policy_0.zip`` is deleted immediately and the "earliest snapshot" would be
        some *previous* run's near-final policy. Writing our own copy also keeps the anchor fixed
        across a ``--resume`` (a moved anchor would break the curve's comparability).

        A failure here is a skipped leg with a printed reason, never a dead ten-hour run.
        """
        path = self.anchor_path
        if path is None:
            path = str(self.anchor_copy_path)
            resuming = self.csv_path.exists() and self.anchor_copy_path.exists()
            if not resuming:
                try:
                    self.anchor_copy_path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(path)
                except Exception as error:                      # noqa: BLE001 - never fatal
                    print(
                        f"[progress] could not snapshot the starting policy to {path} "
                        f"({type(error).__name__}: {error}); the early-anchor leg is skipped "
                        "(pass progress_anchor_path to use an existing checkpoint instead)"
                    )
                    return
            self._resolved_anchor = path
        try:
            from .agents.pool import load_policy

            self._anchor_model = load_policy(path)
        except Exception as error:                              # noqa: BLE001 - never fatal
            print(
                f"[progress] could not load the early anchor {path} "
                f"({type(error).__name__}: {error}); that leg is skipped"
            )
            self._anchor_model = None

    def _on_step(self) -> bool:
        """Run an evaluation when the cadence is reached; never touch the training clock.

        Returns ``False`` once a broken run has been halted: SB3 2.7.1 has no ``stop_training``
        flag (nothing in the package reads one), the callback's own return value is the documented
        way to end a run.
        """
        if self._next_eval is not None and int(self.num_timesteps) >= int(self._next_eval):
            self._run_evaluation()
            self._next_eval = int(self._next_eval) + self.every
        if self._stop_requested:
            print(
                f"[progress] halting at step {int(self.num_timesteps):,} through the callback "
                "contract (returning False from on_step); the trainer saves its checkpoint next"
            )
            return False
        return True

    def _on_training_end(self) -> None:
        """Flush and close the CSV mirror and the (possibly owned) writer."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        if self._writer is not None and self._owns_writer:
            self._writer.close()
        self._writer = None

    # ---- evaluation -----------------------------------------------------
    def _learner_factory(self) -> Callable[[int], Any]:
        """Build agents that drive the *live* model (the point of this callback)."""
        model = self.model
        return lambda seed: PolicyAgent(model, deterministic=True, name="rl")

    def _anchor_factory(self, anchor: str) -> Callable[[int], Any]:
        """Build agents for one anchor name."""
        if anchor == ANCHOR_RULE:
            return lambda seed: RuleAgent(seed=seed)
        if anchor == ANCHOR_RANDOM:
            return lambda seed: RandomAgent(seed=seed)
        model = self._anchor_model
        if model is None:                                      # pragma: no cover - guarded at init
            raise RuntimeError("the early-anchor leg was requested without a resolved checkpoint")
        return lambda seed: PolicyAgent(model, deterministic=True, name=ANCHOR_EARLY)

    def _run_evaluation(self) -> None:
        """Evaluate every anchor in process and write rows, scalars and one verdict line."""
        import torch as th

        policy = self.model.policy
        was_training = policy.training
        policy.eval()
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        try:
            with th.no_grad():
                for anchor in self.anchors:
                    if anchor == ANCHOR_EARLY and self._anchor_model is None:
                        continue
                    metrics = play_duplicate_pairs(
                        self._learner_factory(), self._anchor_factory(anchor),
                        deals=self.deals, seed=self.seed,
                        bootstrap_draws=self.bootstrap_draws,
                    )
                    rows.append({"step": int(self.num_timesteps), "anchor": anchor, **metrics})
        finally:
            if was_training:
                policy.train()
        self.last_eval_seconds = time.perf_counter() - started

        health = read_health(self.logger)
        distinct, window_full = self._diversity()
        verdict, reason = self._verdict(rows, health, distinct, window_full)
        for row in rows:
            row.update({
                "entropy": health["entropy"],
                "approx_kl": health["approx_kl"],
                "verdict": verdict,
                "eval_seconds": self.last_eval_seconds,
            })
            self._write_row(row)
        self._log_scalars(rows)
        print(self._log_line(rows, health, verdict, reason, distinct, window_full))
        self._handle_verdict(verdict, reason)

    def _diversity(self) -> tuple[int | None, bool]:
        """``(distinct deals, window full)`` from the stats callback, or ``(None, False)``."""
        diversity = getattr(self.stats, "diversity", None)
        if diversity is None:
            return None, False
        return int(diversity.distinct), bool(diversity.full)

    def _verdict(self, rows: Iterable[dict[str, Any]],
                 health: dict[str, float | None],
                 distinct: int | None,
                 window_full: bool) -> tuple[str, str]:
        """The verdict for this evaluation, using the previous row of the rule anchor."""
        by_anchor = {str(row["anchor"]): row for row in rows}
        rule = by_anchor.get(ANCHOR_RULE)
        previous = self._previous.get(ANCHOR_RULE)
        verdict, reason = classify(
            entropy=health["entropy"],
            approx_kl=health["approx_kl"],
            clip_fraction=health["clip_fraction"],
            policy_gradient_loss=health["policy_gradient_loss"],
            distinct_deals=distinct,
            window_full=window_full,
            rule_win_rate=None if rule is None else float(rule["win_rate"]),
            rule_paired_low=None if rule is None else float(rule["paired_low"]),
            previous_rule_win_rate=None if previous is None else float(previous["win_rate"]),
        )
        for name, row in by_anchor.items():
            self._previous[name] = row
        return verdict, reason

    def _handle_verdict(self, verdict: str, reason: str) -> None:
        """Count broken readings and halt the run once the patience is exhausted."""
        if verdict != VERDICT_BROKEN:
            self._broken_streak = 0
            return
        self._broken_streak += 1
        remaining = self.broken_patience - self._broken_streak
        if remaining > 0:
            print(f"[progress] BROKEN reading {self._broken_streak}/{self.broken_patience}: {reason}")
            return
        if not self.stop_on_broken:
            print(
                f"[progress] BROKEN ({self._broken_streak} consecutive readings) but "
                "stop_on_broken is off: continuing. " + reason
            )
            return
        print(
            f"[progress] STOPPING TRAINING after {self._broken_streak} consecutive BROKEN "
            f"readings: {reason}. The last checkpoint is on disk; fix the environment before "
            "resuming, because more steps cannot recover this."
        )
        self._stop_requested = True

    # ---- output ---------------------------------------------------------
    def _extras(self, row: dict[str, Any], verdict: str,
                distinct: int | None, window_full: bool) -> str:
        """Advisory suffix for the log line (ceiling note, missing diversity counter)."""
        notes: list[str] = []
        previous = self._previous.get(str(row["anchor"]))
        if (
            str(row["anchor"]) == ANCHOR_RULE
            and verdict == VERDICT_NO_GAIN
            and CEILING_BAND[0] <= float(row["win_rate"]) <= CEILING_BAND[1]
            and previous is not None
        ):
            notes.append(
                f"at the rule bot's own ceiling ({RULE_BOT_CEILING:.1%} vs random): more steps are "
                "unlikely to pay -- distil from the search, or strengthen the opponent mix"
            )
        if distinct is None:
            notes.append("no diversity counter wired: frozen-deal check skipped")
        return "" if not notes else " | " + "; ".join(notes)

    def _log_line(self, rows: Iterable[dict[str, Any]], health: dict[str, float | None],
                  verdict: str, reason: str, distinct: int | None, window_full: bool) -> str:
        """One human-readable line per evaluation, the thing you read at 2 a.m."""
        parts: list[str] = []
        for row in rows:
            parts.append(
                f"{row['anchor']} {float(row['win_rate']):.1%} "
                f"[{float(row['ci_low']):.1%},{float(row['ci_high']):.1%}] "
                f"paired {float(row['paired_differential']):+.1f} "
                f"[{float(row['paired_low']):+.1f},{float(row['paired_high']):+.1f}] "
                f"dealt {float(row['win_rate_dealt']):.1%}/rot {float(row['win_rate_rotated']):.1%}"
            )
        entropy = health["entropy"]
        kl = health["approx_kl"]
        health_text = (
            f"ent {entropy:.3f} kl {kl:.1e}" if entropy is not None and kl is not None
            else "ent n/a kl n/a"
        )
        step = int(self.num_timesteps)
        suffix = self._extras(next(iter(rows), {"anchor": ""}), verdict, distinct, window_full)
        return (
            f"[progress @{step / 1e6:.2f}M] " + " | ".join(parts) + f" | {health_text} | "
            f"{verdict} ({reason})"
            f"{suffix} | {self.last_eval_seconds:.1f}s"
        )

    def _write_row(self, row: dict[str, Any]) -> None:
        """Append one row to the CSV mirror."""
        self.rows.append(dict(row))
        if self._file is None:
            return
        self._file.write(",".join(_format(row[key]) for key in CSV_HEADER) + "\n")
        self._file.flush()

    def _log_scalars(self, rows: Iterable[dict[str, Any]]) -> None:
        """Write the per-anchor scalars to TensorBoard."""
        if self._writer is None:
            return
        step = int(self.num_timesteps)
        for row in rows:
            anchor = str(row["anchor"])
            self._writer.add_scalar(f"{TAG_PREFIX}/{anchor}/win_rate", float(row["win_rate"]), step)
            self._writer.add_scalar(
                f"{TAG_PREFIX}/{anchor}/paired_differential",
                float(row["paired_differential"]), step,
            )
            self._writer.add_scalar(
                f"{TAG_PREFIX}/{anchor}/paired_low", float(row["paired_low"]), step,
            )
            self._writer.add_scalar(
                f"{TAG_PREFIX}/{anchor}/paired_high", float(row["paired_high"]), step,
            )
            self._writer.add_scalar(
                f"{TAG_PREFIX}/{anchor}/points_per_game", float(row["points_per_game"]), step,
            )
            self._writer.add_scalar(
                f"{TAG_PREFIX}/{anchor}/mean_pips_left", float(row["mean_pips_left"]), step,
            )


def _format(value: Any) -> str:
    """CSV field formatting: floats via ``repr`` (exact round-trip), ints verbatim."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


__all__ = [
    "ANCHOR_EARLY",
    "ANCHOR_RANDOM",
    "ANCHOR_RULE",
    "CSV_HEADER",
    "DEFAULT_ANCHORS",
    "PeriodicEvalCallback",
    "VERDICT_BROKEN",
    "VERDICT_KEEP",
    "VERDICT_NO_GAIN",
    "VERDICT_UNKNOWN",
    "classify",
    "play_duplicate_pairs",
    "read_health",
    "resolve_early_anchor",
]
