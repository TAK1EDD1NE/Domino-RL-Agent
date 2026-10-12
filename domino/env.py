"""Gymnasium environment exposing the learner team's decision points (spec section 6).

``DominoEnv`` wraps the pure-Python engine as the **learner-team MDP**: the
learner controls ``env_cfg.learner_seats`` (default ``(0, 2)``) while the opposing
seats are played internally by the league opponent. :meth:`DominoEnv.step` applies
the learner's move and then advances the game -- every opponent turn is played by
the assigned policy and every forced pass is consumed silently -- so the
environment only ever returns at a real decision point (a learner seat with at
least one legal placement) or at the end of a scored deal. A turn with no legal
move is never a decision point.

The observation and the action mask are seat-relative to whichever learner seat is
to act: :func:`domino.obs.encode` and :func:`domino.masks.action_mask` are called
with the acting seat. The action space is ``Discrete(56)``
(``2 * tile_id + end_index``); the observation is a 192-dim ``float32`` vector in
``[0, 1]``.

Reward is the deal-level banked points (raw, undiscounted, no shaping): under
``reward_mode="differential"`` (default) the learner's team gets ``+points`` when
it wins the deal and ``-points`` when it loses, while ``reward_mode="banked"``
gives ``+points`` on a win and ``0`` otherwise. Every scored deal emits its reward
on the transition that ends it -- including the intermediate deals of a
``max_deals_per_episode > 1`` episode -- and every other transition is exactly
``0.0``. A void deal scores nothing, is redealt and does not end the episode
(mirroring :class:`domino.engine.Match`), and the lead rules are always taken from
the engine (deal 1 and post-void redeals force the ``[6|6]`` lead, later deals are
led by the previous deal's winner with a free lead). The deal shuffle and the
per-deal opponent assignment flow through the environment's own
:class:`numpy.random.Generator`, and the environment owns its opponent: a bare
:class:`~domino.agents.Agent` is deep-copied (the caller's object is never touched)
and that copy is restored to its construction seed on ``reset`` -- so a seeded
environment (with seeded opponents) is fully reproducible.

Only ``gymnasium``, ``numpy`` and the engine/observation/mask/agent layers are
imported here -- never torch, sb3 or pygame -- so the module is safe to import in
a training worker (it is built from a picklable factory inside ``SubprocVecEnv``)
and in a headless process. The environment deliberately provides no ``render``
method: the Pygame UI drives its own renderer.
"""
from __future__ import annotations

import copy
import zlib
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from .agents import Agent, OpponentPool
from .config import REWARD_MODES, EnvConfig
from .engine import (
    DealOutcome,
    GameState,
    deal_outcome,
    new_deal,
    next_leader,
    redeal_after_void,
)
from .masks import ACTION_DIM, action_mask, decode_action
from .obs import OBS_DIM, encode


def _opening_key(state: GameState) -> int:
    """Stable fingerprint of a deal's opening position (hands, turn, leader).

    Reported per step as ``info["deal_key"]`` so a trainer can count how many
    distinct deals it is actually playing: an environment that re-dealt the same
    shuffle every episode used to look like a converged run (flat reward window,
    entropy collapsed to zero) while the policy was only memorising a handful of
    positions. ``crc32`` of a canonical string keeps the id small, cheap and
    identical across processes and machines.
    """
    hands = "|".join(",".join(str(tile) for tile in sorted(hand)) for hand in state.hands)
    payload = f"{hands}|table={','.join(str(tile) for tile in state.table)}"
    payload += f"|turn={state.turn}|leader={state.leader}|six_six={int(state.forced_leader_six_six)}"
    return int(zlib.crc32(payload.encode("ascii")))


class DominoEnv(gym.Env):
    """The learner-team 2v2 domino MDP (spec sections 3, 5, 6)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: Agent | OpponentPool | None = None,
        env_cfg: EnvConfig | None = None,
        seed: int | None = None,
        strict: bool = True,
    ) -> None:
        """Build the environment.

        Parameters
        ----------
        opponent:
            The opposing team's source. A bare :class:`~domino.agents.Agent`
            plays both opposing seats; an :class:`~domino.agents.OpponentPool`
            is sampled per deal. ``None`` builds the default league pool from
            ``env_cfg``. A bare agent is deep-copied so the environment owns the
            object it reseeds on ``reset`` (the caller's agent is never mutated).
        env_cfg:
            Typed configuration (reward mode, learner seats, deal cap). Defaults
            to :class:`domino.config.EnvConfig`.
        seed:
            Initial seed for the environment's own generator. ``reset(seed=...)``
            overrides it.
        strict:
            With ``True`` (default) an action outside the current mask raises
            :class:`ValueError`. With ``False`` it is coerced to the lowest-index
            legal action, which lets ``gymnasium.utils.env_checker.check_env``
            (which samples random actions) validate the API.
        """
        super().__init__()
        self.env_cfg = env_cfg if env_cfg is not None else EnvConfig()
        if self.env_cfg.reward_mode not in REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {sorted(REWARD_MODES)}, "
                f"got {self.env_cfg.reward_mode!r}"
            )
        if self.env_cfg.max_deals_per_episode < 1:
            raise ValueError(
                f"max_deals_per_episode must be >= 1, got {self.env_cfg.max_deals_per_episode}"
            )
        self.strict = bool(strict)
        self._reward_mode = self.env_cfg.reward_mode
        self._max_deals = int(self.env_cfg.max_deals_per_episode)
        self._learner_seats: tuple[int, ...] = tuple(self.env_cfg.learner_seats)
        self._opponent_seats: tuple[int, ...] = tuple(
            seat for seat in range(self.env_cfg.n_seats) if seat not in self._learner_seats
        )

        self._seed = None if seed is None else int(seed)
        # All episode randomness flows through gymnasium's own generator: the
        # constructor seed seeds the first episode and ``reset(seed=...)``
        # re-seeds it, so ``self._seed`` is bookkeeping only. Deriving the deal
        # shuffle from a sticky seed instead made every episode replay one deal.
        self.np_random, self._np_random_seed = seeding.np_random(self._seed)
        self._rng = np.random.default_rng(int(self.np_random.integers(0, 2**31 - 1)))
        self._opponent: Agent | OpponentPool = self._adopt_opponent(
            opponent if opponent is not None else self._default_opponent()
        )

        self.observation_space = spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(ACTION_DIM)

        # Lifetime state: deal_index and match_scores advance monotonically
        # across the deals of an episode; reset() starts a fresh episode (and a
        # fresh match) so a seeded reset is fully reproducible, as the Gymnasium
        # environment checker requires.
        self._deal_index = 0
        self._scores: list[int] = [0, 0]

        # Per-episode state.
        self._state: GameState | None = None
        self._opponent_agents: tuple[Agent, Agent] | None = None
        self._pending_leader: int | None = None
        self._deals_scored = 0
        self._scored_this_step: DealOutcome | None = None
        self._void_this_step = False
        self._done = True
        # The last learner seat to act: pinned as acting_seat on the terminal
        # transition, where there is no decision point.
        self._last_learner_seat: int | None = None
        # Opening-deal fingerprint of the deal in play, and of the deal whose
        # outcome is reported on this transition (info["deal_key"]).
        self._deal_key = 0
        self._scored_key: int | None = None

    # ---- construction helpers ------------------------------------------
    def _default_opponent(self) -> OpponentPool:
        """Build the default league pool from the environment configuration."""
        return OpponentPool(weights=self.env_cfg.opponent_weights, seed=self.env_cfg.seed)

    def _adopt_opponent(self, opponent: Agent | OpponentPool) -> Agent | OpponentPool:
        """Take ownership of the opponent without mutating the caller's object.

        ``reset`` reseeds a bare stochastic opponent, so the environment must own
        the object it reseeds: a bare :class:`~domino.agents.Agent` is deep-copied
        (the caller's agent is never rewound, nor shared between environments).
        An :class:`~domino.agents.OpponentPool` is kept as-is -- its per-deal draw
        already flows through the environment's own generator.
        """
        if isinstance(opponent, OpponentPool):
            return opponent
        return copy.deepcopy(opponent)

    # ---- gymnasium API --------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode and return the first learner decision point.

        Draws this episode's deal seed from gymnasium's generator and routes the
        per-deal opponent assignment through the resulting generator, so
        ``reset(seed=...)`` reproduces an episode exactly while an unseeded
        reset -- what a vector-env worker does after every episode -- deals a
        fresh shuffle. Deals the first deal and advances through opponent turns
        and forced passes to the first learner decision.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
        # ``self.np_random`` advances on every reset and is rebuilt from ``seed``
        # when one is given: one fresh deal seed per episode, reproducibly.
        self._rng = np.random.default_rng(int(self.np_random.integers(0, 2**31 - 1)))

        self._done = False
        self._pending_leader = None
        self._deals_scored = 0
        self._scored_this_step = None
        self._void_this_step = False
        self._deal_index = 0
        self._scores = [0, 0]
        self._last_learner_seat = None
        self._scored_key = None

        self._opponent_agents = self._sample_opponent_team()
        self._state = self._deal_new()
        self._reseed_opponent()
        # Advance to the first learner decision. A deal cannot end before the
        # leader plays (a free lead always has a legal move), so the loop is a
        # defensive guard ensuring reset always returns a decision point.
        while not self._advance_to_decision():
            self._pending_leader = None
            self._opponent_agents = self._sample_opponent_team()
            self._state = self._deal_new()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply the learner's move and advance to the next decision point.

        Returns ``(observation, reward, terminated, truncated, info)``. The reward
        is non-zero only when a scored deal ends on this transition; ``truncated``
        is always ``False`` (episodes end only at a deal end).
        """
        if self._done:
            raise RuntimeError("step() called after the episode ended; call reset() first")
        action = int(action)
        mask = self.action_masks()
        if not 0 <= action < ACTION_DIM or not bool(mask[action]):
            if self.strict:
                legal = np.flatnonzero(mask).tolist()
                raise ValueError(
                    f"action {action} is not legal at this decision point; "
                    f"legal actions are {legal}"
                )
            action = int(np.flatnonzero(mask)[0])
        seat = self._state.turn
        tile_id, end = decode_action(action)
        self._state.play(seat, tile_id, end)
        reward, terminated = self._advance()
        return self._observation(), float(reward), bool(terminated), False, self._info()

    def action_masks(self) -> np.ndarray:
        """Return the ``np.bool_`` action mask for the acting learner seat."""
        return action_mask(self._state, self._state.turn)

    @property
    def state(self) -> GameState:
        """The current deal's engine state."""
        return self._state

    @property
    def acting_seat(self) -> int:
        """The learner seat to act.

        At a decision point this is the seat about to act (always a learner
        seat). On the terminal transition there is no decision point, so it is
        pinned to the last learner seat that acted -- the property always names
        a learner seat.
        """
        return self._acting_seat()

    def set_opponent(self, opponent: Agent | OpponentPool) -> None:
        """Swap the opposing team's source mid-life (takes effect next deal).

        The environment takes ownership of a bare opponent by copying it, so the
        caller's object is never mutated.
        """
        self._opponent = self._adopt_opponent(opponent)

    def refresh_pool(self, paths: Sequence[str]) -> None:
        """Notify the league pool of the current snapshot paths (spec section 8).

        The training :class:`~domino.callbacks.SnapshotCallback` calls this on
        every worker through ``vec_env.env_method("refresh_pool", paths)`` after
        it has written a new frozen checkpoint. When the current opponent is an
        :class:`~domino.agents.OpponentPool` the call is delegated to
        :meth:`OpponentPool.refresh_snapshots`; when it is a bare
        :class:`~domino.agents.Agent` (a single fixed opponent) this is a no-op,
        so the method is safe to call unconditionally on every worker.
        """
        opponent = self._opponent
        if isinstance(opponent, OpponentPool):
            opponent.refresh_snapshots(paths)

    # ---- turn loop ------------------------------------------------------
    def _acting_seat(self) -> int:
        """The seat whose perspective the observation uses.

        The current turn at a decision point, or -- once the episode has ended,
        where there is no decision point -- the last learner seat that acted.
        """
        if self._done:
            return self._last_learner_seat
        return self._state.turn

    def _advance(self) -> tuple[float, bool]:
        """Resolve the game to the next learner decision or a scored deal end.

        Returns ``(reward, terminated)``. Void deals are redealt in place (no
        reward, no episode end). Every scored deal emits its reward on the
        transition that ends it: the final deal of the episode ends it, while an
        intermediate deal (``max_deals_per_episode > 1``) deals the next deal,
        advances to its first decision point and still returns the reward.
        """
        self._scored_this_step = None
        self._scored_key = None
        self._void_this_step = False
        while True:
            if self._advance_to_decision():
                return 0.0, False
            outcome = deal_outcome(self._state)
            if outcome.void:
                self._void_this_step = True
                self._pending_leader = None
                self._opponent_agents = self._sample_opponent_team()
                self._state = redeal_after_void(self._state)
                self._deal_key = _opening_key(self._state)
                self._deal_index += 1
                continue
            reward = self._deal_reward(outcome)
            self._scores[outcome.winner_team] += outcome.points
            self._pending_leader = next_leader(self._state, outcome)
            self._scored_this_step = outcome
            self._scored_key = self._deal_key
            self._deals_scored += 1
            if self._deals_scored >= self._max_deals:
                self._done = True
                return reward, True
            # Non-final deal end: emit this deal's reward on this transition and
            # return at the next deal's first decision point -- do not fall
            # through to the top-of-loop 0.0, which would discard the reward.
            self._opponent_agents = self._sample_opponent_team()
            self._state = self._deal_new()
            while not self._advance_to_decision():
                self._pending_leader = None
                self._opponent_agents = self._sample_opponent_team()
                self._state = self._deal_new()
            return reward, False

    def _advance_to_decision(self) -> bool:
        """Play out opponent turns and forced passes until a learner decision.

        Returns ``True`` when a learner seat with a legal move is to act, and
        ``False`` when the deal has ended (emptied or blocked).
        """
        while not self._state.is_terminal():
            seat = self._state.turn
            if not self._state.legal_actions(seat):
                self._state.pass_turn(seat)          # forced pass: never a decision
                continue
            if seat in self._learner_seats:
                self._last_learner_seat = seat
                return True
            action = int(self._agent_for_seat(seat).act(self._state, seat))
            tile_id, end = decode_action(action)
            self._state.play(seat, tile_id, end)
        return False

    # ---- deal bookkeeping ----------------------------------------------
    def _deal_new(self) -> GameState:
        """Deal a fresh deal, delegating the lead rule to the engine."""
        self._deal_index += 1
        if self._pending_leader is None:
            state = new_deal(
                deal_no=self._deal_index,
                rng=self._rng,
                lead_mode="six_six",
                scores=list(self._scores),
            )
        else:
            state = new_deal(
                deal_no=self._deal_index,
                rng=self._rng,
                lead_mode="winner",
                leader=self._pending_leader,
                scores=list(self._scores),
            )
        self._deal_key = _opening_key(state)
        return state

    def _deal_reward(self, outcome: DealOutcome) -> float:
        """Apply the configured reward mode to a scored deal."""
        points_learner = outcome.points if outcome.winner_team == 0 else 0
        points_opponent = outcome.points if outcome.winner_team == 1 else 0
        if self._reward_mode == "banked":
            return float(points_learner)
        return float(points_learner - points_opponent)

    # ---- opponents ------------------------------------------------------
    def _sample_opponent_team(self) -> tuple[Agent, Agent]:
        """Draw the opposing team's two agents for the coming deal.

        A bare :class:`~domino.agents.Agent` plays both opposing seats. An
        :class:`~domino.agents.OpponentPool` is drawn through its documented
        :meth:`~domino.agents.OpponentPool.assign` API, but the per-deal seed is
        taken from the environment's own generator (never the pool's persistent
        stream), so ``reset(seed=...)`` reproduces the assignment exactly.
        """
        opponent = self._opponent
        if isinstance(opponent, OpponentPool):
            opponent._seed = int(self._rng.integers(0, 2**31 - 1))
            return opponent.assign(self._deal_index)
        return opponent, opponent

    def _reseed_opponent(self) -> None:
        """Restore the env-owned bare opponent to its construction seed.

        The environment always owns its opponent (a bare ``Agent`` is
        deep-copied in ``__init__``/``set_opponent``), so this never mutates a
        caller-supplied object. The pool path needs no reset here: its per-deal
        assignment is already drawn from the environment's generator.
        """
        opponent = self._opponent
        if isinstance(opponent, OpponentPool):
            return
        if isinstance(getattr(opponent, "_rng", None), np.random.Generator):
            opponent._rng = np.random.default_rng(getattr(opponent, "seed", None))

    def _agent_for_seat(self, seat: int) -> Agent:
        """Return the assigned opponent agent for an opposing seat."""
        return self._opponent_agents[self._opponent_seats.index(seat)]

    # ---- observation and info ------------------------------------------
    def _observation(self) -> np.ndarray:
        """Seat-relative 192-dim encoding for the seat to act.

        On the terminal transition the seat is the last learner seat that acted
        (there is no decision point), so the encoding stays learner-relative.
        """
        return encode(self._state, self._acting_seat())

    def _info(self) -> dict:
        """Build the ``info`` dict for the current position.

        ``acting_seat``, ``deal_index`` and ``deal_key`` are always present.
        ``deal_key`` is the opening-position fingerprint of the deal in play, or
        -- on a transition that reports a scored deal -- of the deal that just
        ended, so a trainer can count how many distinct deals it has seen. The
        deal-level keys are populated when a scored deal ended on this transition
        (the terminal step, or an intermediate deal end of a multi-deal episode)
        and ``None`` otherwise; a void redeal is reported through ``void_deal``
        (``True``) and ``deal_outcome="void_tie"`` on the otherwise non-terminal
        step it happens.
        """
        outcome = self._scored_this_step
        if outcome is None:
            points_learner: int | None = None
            points_opponent: int | None = None
            deal_outcome: str | None = "void_tie" if self._void_this_step else None
        else:
            points_learner = outcome.points if outcome.winner_team == 0 else 0
            points_opponent = outcome.points if outcome.winner_team == 1 else 0
            deal_outcome = outcome.reason
        return {
            "acting_seat": int(self._acting_seat()),
            "deal_index": int(self._deal_index),
            "deal_key": int(self._deal_key if outcome is None else self._scored_key),
            "deal_outcome": deal_outcome,
            "void_deal": bool(self._void_this_step) if outcome is None else bool(outcome.void),
            "blocked": None if outcome is None else bool(outcome.blocked),
            "points_learner": points_learner,
            "points_opponent": points_opponent,
            "differential": None if outcome is None else points_learner - points_opponent,
            "match_scores": list(self._scores),
            "learner_won": None if outcome is None else bool(outcome.winner_team == 0),
        }
