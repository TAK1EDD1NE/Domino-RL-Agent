"""Tests for the learner-team Gymnasium environment (spec sections 3, 5, 6).

Covers the environment API (spaces, ``reset``/``step`` contracts), the masked
``Discrete(56)`` action space, the deal-terminal differential/banked reward
accounting, the forced-pass rule (a stuck seat is never a decision point), the
bounded episode length, void-deal redeals, seed determinism, the league wiring,
``gymnasium.utils.env_checker`` and ``SubprocVecEnv`` spawn-safety. The tests
assert on the public surface plus ``info``; a couple inspect the private
per-deal opponent assignment, which is the only way to observe league wiring
without a full model.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from domino.agents import OpponentPool, RandomAgent
from domino.config import EnvConfig
from domino.env import DominoEnv
from domino.masks import action_index
from domino.obs import OBS_DIM, encode

LEARNER_SEATS = (0, 2)


class _RecordingAgent:
    """Records its calls and plays the lowest-index legal move."""

    name = "recording"

    def __init__(self) -> None:
        self.calls = 0

    def act(self, state, seat: int) -> int:
        self.calls += 1
        tid, end = state.legal_moves(seat)[0]
        return action_index(tid, end)


def _step_lowest(env: DominoEnv):
    """Step with the lowest legal action; returns the full step tuple."""
    mask = env.action_masks()
    return env.step(int(np.flatnonzero(mask)[0]))


def _run_deals(env: DominoEnv, deals: int, guard: int = 5000) -> None:
    """Play until ``deals`` deals have ended, resetting between episodes."""
    ended = 0
    steps = 0
    while ended < deals:
        _, _, term, trunc, info = _step_lowest(env)
        steps += 1
        assert steps < guard, "env did not finish the requested deals in time"
        if info["points_learner"] is not None:      # a scored deal ended
            ended += 1
        if term or trunc:
            env.reset()


# 1 ---------------------------------------------------------------------------
def test_spaces_and_reset():
    env = DominoEnv(opponent=RandomAgent(seed=0), seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.observation_space.dtype == np.float32
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
    assert env.action_space.n == 56
    assert env.acting_seat in LEARNER_SEATS
    assert info["acting_seat"] == env.acting_seat
    assert info["deal_index"] == 1
    assert isinstance(info["match_scores"], list) and len(info["match_scores"]) == 2


# 2 ---------------------------------------------------------------------------
def test_mask_matches_legal_moves_at_every_decision_point():
    env = DominoEnv(opponent=RandomAgent(seed=1), seed=1)
    env.reset(seed=1)
    steps = 0
    for _ in range(200):
        mask = env.action_masks()
        assert mask.shape == (56,) and mask.dtype == np.bool_
        assert mask.sum() >= 1
        seat = env.acting_seat
        expected = sorted(2 * t + e for t, e in env.state.legal_moves(seat))
        assert sorted(np.flatnonzero(mask).tolist()) == expected
        _, _, term, trunc, _ = env.step(int(np.flatnonzero(mask)[0]))
        steps += 1
        if term or trunc:
            env.reset()
    assert steps == 200


# 3 ---------------------------------------------------------------------------
def test_illegal_action_raises_strict_and_is_coerced_lenient():
    strict = DominoEnv(opponent=RandomAgent(seed=2), seed=2, strict=True)
    strict.reset(seed=2)
    mask = strict.action_masks()
    illegal = int(np.flatnonzero(~mask)[0])
    with pytest.raises(ValueError):
        strict.step(illegal)
    with pytest.raises(ValueError):
        strict.step(999)                       # out of range
    # A rejected action must not have advanced the game.
    assert strict.action_masks().tolist() == mask.tolist()
    assert strict.acting_seat == strict.state.turn

    # strict=False coerces to the lowest legal index: an illegal action and the
    # lowest legal one drive identical trajectories.
    a = DominoEnv(opponent=RandomAgent(seed=2), seed=2, strict=False)
    b = DominoEnv(opponent=RandomAgent(seed=2), seed=2, strict=False)
    a.reset(seed=2)
    b.reset(seed=2)
    ma = a.action_masks()
    low = int(np.flatnonzero(ma)[0])
    bad = int(np.flatnonzero(~ma)[0])
    obs_a, r_a, term_a, trunc_a, _ = a.step(low)
    obs_b, r_b, term_b, trunc_b, _ = b.step(bad)
    assert np.array_equal(obs_a, obs_b)
    assert (r_a, term_a, trunc_a) == (r_b, term_b, trunc_b)


# 4 ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["differential", "banked"])
def test_terminal_reward_matches_house_scoring(mode):
    env = DominoEnv(opponent=RandomAgent(seed=3), env_cfg=EnvConfig(reward_mode=mode), seed=3)
    env.reset(seed=3)
    total = 0.0
    for _ in range(300):
        _, r, term, trunc, info = _step_lowest(env)
        if not term:
            assert r == 0.0                 # only the terminal step carries reward
        total += r
        if term:
            break
    assert term and not trunc
    assert info["deal_outcome"] in ("emptied", "blocked")
    assert info["void_deal"] is False
    pl, po = info["points_learner"], info["points_opponent"]
    assert info["differential"] == pl - po
    assert info["learner_won"] == (pl > po)
    if mode == "differential":
        assert total == pytest.approx(pl - po)
    else:
        assert total == pytest.approx(pl)


# 5 ---------------------------------------------------------------------------
def test_forced_pass_is_never_a_decision_point_and_decisions_bounded():
    env = DominoEnv(opponent=RandomAgent(seed=4), seed=4)
    env.reset(seed=4)
    decisions = 0
    while True:
        seat = env.acting_seat
        assert env.state.legal_actions(seat), "a forced pass must not be a decision point"
        mask = env.action_masks()
        assert mask.sum() >= 1
        decisions += 1
        _, _, term, trunc, _ = env.step(int(np.flatnonzero(mask)[0]))
        if term or trunc:
            break
    assert 1 <= decisions <= 14


# 6 ---------------------------------------------------------------------------
def test_episode_length_is_bounded_and_only_terminated():
    for seed in range(8):
        env = DominoEnv(opponent=RandomAgent(seed=seed), seed=seed)
        env.reset(seed=seed)
        steps = 0
        while True:
            _, _, term, trunc, _ = _step_lowest(env)
            assert trunc is False
            steps += 1
            assert 1 <= steps <= 14
            if term:
                break
        assert term is True


# 7 ---------------------------------------------------------------------------
def test_multi_deal_episode_and_invalid_max_deals():
    cfg = EnvConfig(max_deals_per_episode=3)
    env = DominoEnv(opponent=RandomAgent(seed=5), env_cfg=cfg, seed=5)
    env.reset(seed=5)
    deal_ends = 0
    indices = []
    reward_sum = 0.0
    differential_sum = 0.0
    deal_end_rewards: list[float] = []
    while True:
        _, r, term, trunc, info = _step_lowest(env)
        reward_sum += r
        indices.append(info["deal_index"])
        if info["points_learner"] is not None:      # a scored deal ended
            assert info["deal_outcome"] in ("emptied", "blocked")
            deal_ends += 1
            deal_end_rewards.append(r)
            differential_sum += info["differential"]
        if term or trunc:
            break
    assert deal_ends == 3                       # three scored deals in one episode
    assert indices == sorted(indices)
    assert indices[-1] > indices[0]             # deal_index advances
    # Every deal-end transition carries its own reward -- including the
    # intermediate deals -- and the episode's reward equals the summed
    # per-deal differentials reported in info.
    assert len(deal_end_rewards) == 3
    assert all(r != 0.0 for r in deal_end_rewards)
    assert reward_sum == pytest.approx(differential_sum)
    assert reward_sum == pytest.approx(sum(deal_end_rewards))

    with pytest.raises(ValueError):
        DominoEnv(env_cfg=EnvConfig(max_deals_per_episode=0))


# 8 ---------------------------------------------------------------------------
def test_determinism_same_seed_and_opponents():
    def run(seed: int) -> list[tuple]:
        env = DominoEnv(opponent=RandomAgent(seed=seed), seed=seed)
        traces: list[tuple] = []
        for episode in range(20):
            env.reset(seed=seed + episode)
            while True:
                _, _, term, _, info = _step_lowest(env)
                if term:
                    break
            traces.append(
                (
                    info["deal_outcome"],
                    info["points_learner"],
                    info["points_opponent"],
                    info["differential"],
                    tuple(info["match_scores"]),
                )
            )
        return traces

    assert run(123) == run(123)


# 8b -------------------------------------------------------------------------
def test_unseeded_reset_deals_a_fresh_shuffle():
    """Regression: an unseeded ``reset`` must deal a new shuffle.

    ``reset`` used to rebuild the deal generator from the sticky ``self._seed``,
    so every episode after the first replayed one identical deal. A vector-env
    worker auto-resets with ``seed=None`` (SB3's subproc worker), so a whole run
    trained on a handful of frozen deals: the policy memorised them (entropy ->
    0, policy/value losses -> 0, a flat reward window) and then scored at
    random level on the fresh deals ``test.py`` deals.
    """
    def deal_key(env: DominoEnv) -> tuple:
        return tuple(tuple(sorted(hand)) for hand in env._state.hands)

    env = DominoEnv(opponent=RandomAgent(seed=0), seed=0)
    keys = set()
    for _ in range(25):
        env.reset()
        keys.add(deal_key(env))
    assert len(keys) >= 20, f"unseeded resets replayed deals: {len(keys)} distinct in 25"

    # reset(seed=...) keeps its contract: same seed reproduces, other seeds differ.
    env.reset(seed=0)
    first = deal_key(env)
    env.reset(seed=1)
    other = deal_key(env)
    env.reset(seed=0)
    assert deal_key(env) == first
    assert other != first

    # The constructor seed still seeds the first episode reproducibly.
    a = DominoEnv(opponent=RandomAgent(seed=0), seed=7)
    b = DominoEnv(opponent=RandomAgent(seed=0), seed=7)
    a.reset()
    b.reset()
    assert deal_key(a) == deal_key(b)


# 8c -------------------------------------------------------------------------
def test_info_deal_key_identifies_the_opening_position():
    """The per-deal fingerprint the frozen-deal guard is built on.

    A run whose env re-dealt the same shuffle every episode trains on a handful
    of positions and looks converged (flat window, entropy -> 0). The guard needs
    a stable id for the deal a step belongs to, not for the position in it.
    """
    env = DominoEnv(opponent=RandomAgent(seed=0), seed=0)
    keys = set()
    for _ in range(12):
        _, info = env.reset()
        key = info["deal_key"]
        assert isinstance(key, int)
        while True:                          # constant for every step of the deal
            _, _, term, trunc, step_info = _step_lowest(env)
            assert step_info["deal_key"] == key
            if term or trunc:
                break
        keys.add(key)
    assert len(keys) == 12, "a fresh shuffle must give a new deal key"

    env.reset(seed=5)
    seeded = env._info()["deal_key"]
    env.reset(seed=6)
    assert env._info()["deal_key"] != seeded
    env.reset(seed=5)
    assert env._info()["deal_key"] == seeded


# 9 ---------------------------------------------------------------------------
def test_league_wiring_rule_random_and_set_opponent():
    # rule source: the pool hands back the (recording) rule agent for both seats
    recorder = _RecordingAgent()
    rule_pool = OpponentPool(weights={"rule": 1.0}, rule_agent=recorder, seed=0)
    env = DominoEnv(opponent=rule_pool, seed=0)
    env.reset(seed=0)
    assert env._opponent_agents == (recorder, recorder)
    _run_deals(env, deals=5)
    assert recorder.calls > 0

    # random source: both opposing seats are RandomAgents
    random_pool = OpponentPool(weights={"random": 1.0}, seed=0)
    env2 = DominoEnv(opponent=random_pool, seed=0)
    env2.reset(seed=0)
    assert [agent.name for agent in env2._opponent_agents] == ["random", "random"]

    # set_opponent swaps the opponent mid-life without breaking the loop. The
    # env owns a copy, so the caller's recorder is never used or mutated.
    env3 = DominoEnv(opponent=RandomAgent(seed=0), seed=0)
    env3.reset(seed=0)
    _step_lowest(env3)
    recorder2 = _RecordingAgent()
    env3.set_opponent(recorder2)
    assert env3._opponent is not recorder2     # the env owns a copy
    _run_deals(env3, deals=5)
    assert env3._opponent.calls > 0
    assert recorder2.calls == 0


# 9b --------------------------------------------------------------------------
def test_env_owns_its_opponent_and_leaves_the_caller_untouched():
    agent = RandomAgent(seed=0)
    caller_rng = agent._rng
    env = DominoEnv(opponent=agent, seed=0)
    assert env._opponent is not agent               # the env owns a copy
    env.reset(seed=0)
    env.reset(seed=0)                               # double reset (check_env style)
    assert agent._rng is caller_rng                 # caller's generator never rewound
    assert env._opponent._rng is not caller_rng     # the copy owns its own stream

    # Two envs given the same caller object never share an opponent.
    other = DominoEnv(opponent=agent, seed=0)
    assert other._opponent is not env._opponent
    assert other._opponent is not agent

    # The owned copy is what plays, and it stays deterministic across same-seed envs.
    def trace() -> tuple:
        e = DominoEnv(opponent=RandomAgent(seed=0), seed=0)
        e.reset(seed=0)
        while True:
            _, _, term, _, info = _step_lowest(e)
            if term:
                return (info["deal_outcome"], info["points_learner"], info["points_opponent"])

    assert trace() == trace()


# 9c --------------------------------------------------------------------------
def test_terminal_step_acting_seat_is_the_last_learner_seat():
    env = DominoEnv(opponent=RandomAgent(seed=6), seed=6)
    env.reset(seed=6)
    last_seat = None
    while True:
        obs, _, term, _, info = _step_lowest(env)
        assert info["acting_seat"] in LEARNER_SEATS      # a learner seat at every step
        if term:
            # No decision point at termination: pinned to the last learner seat.
            assert info["acting_seat"] == last_seat
            break
        last_seat = info["acting_seat"]
    assert env.acting_seat in LEARNER_SEATS
    assert info["acting_seat"] == env.acting_seat
    # The returned terminal observation is seat-relative to that pinned seat.
    assert np.array_equal(obs, encode(env.state, env.acting_seat))


# 10 --------------------------------------------------------------------------
def test_gymnasium_env_checker_passes():
    env = DominoEnv(opponent=RandomAgent(seed=0), strict=False)
    check_env(env, skip_render_check=True)


# 11 --------------------------------------------------------------------------
def _subproc_factory() -> DominoEnv:
    """Module-level factory so ``SubprocVecEnv`` (spawn on Windows) can pickle it."""
    return DominoEnv(opponent=RandomAgent(seed=7), seed=7, strict=False)


def test_subproc_vec_env_spawn_safe():
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = SubprocVecEnv([_subproc_factory, _subproc_factory])
    try:
        obs = venv.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        for _ in range(200):
            actions = np.array([venv.action_space.sample() for _ in range(venv.num_envs)])
            obs, _, _, _ = venv.step(actions)
        assert obs.shape == (venv.num_envs, OBS_DIM)
        assert np.isfinite(obs).all()
    finally:
        venv.close()


# 12 --------------------------------------------------------------------------
def test_env_module_imports_no_torch_or_pygame():
    code = (
        "import sys; import domino.env; "
        "banned = {'torch', 'sb3_contrib', 'stable_baselines3', 'pygame'}; "
        "leaked = banned & set(sys.modules); "
        "assert not leaked, leaked"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# 13 --------------------------------------------------------------------------
def test_refresh_pool_delegates_for_pool_and_is_noop_for_bare_agent(tmp_path):
    # A league pool: refresh_pool forwards the paths to its snapshot list.
    fake = tmp_path / "policy_1.zip"
    fake.write_bytes(b"placeholder")
    pool = OpponentPool(weights={"snapshot": 0.5, "rule": 0.5}, seed=0)
    env = DominoEnv(opponent=pool, seed=0)
    assert "snapshot" not in env._opponent._available_weights()   # no snapshots yet
    env.refresh_pool([str(fake)])
    assert env._opponent._snapshot_paths == [str(fake)]
    assert "snapshot" in env._opponent._available_weights()       # now drawable

    # An empty refresh clears the list (still a valid delegation).
    env.refresh_pool([])
    assert env._opponent._snapshot_paths == []

    # A bare agent: refresh_pool must be a harmless no-op.
    env2 = DominoEnv(opponent=RandomAgent(seed=0), seed=0)
    env2.refresh_pool([str(fake)])              # must not raise
    assert not isinstance(env2._opponent, OpponentPool)
