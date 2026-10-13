"""Phase-3 tests: paired worlds, budget control, exactness when short, opponent models."""
from __future__ import annotations

import time

import numpy as np
import pytest

from domino.agents import RuleAgent
from domino.engine import GameState, new_deal
from domino.masks import action_index
from domino.pimc import PIMCAgent, PublicView, SolverAgent, World
from domino.pimc.policy_search import policy_solve
from domino.tiles import NUM_TILES


def make_state(hands, ends=(), turn=0, pass_streak=0) -> GameState:
    """Engine state from explicit hands and open ends (local copy: pytest 9 does not put
    the tests directory on ``sys.path``, so test modules cannot import each other)."""
    flat = {tile for hand in hands for tile in hand}
    table = [t for t in range(NUM_TILES) if t not in flat][:1] if ends else []
    return GameState(
        hands=[sorted(int(t) for t in hand) for hand in hands],
        rng=np.random.default_rng(0),
        table=list(table),
        ends=[int(v) for v in ends],
        turn=int(turn),
        pass_streak=int(pass_streak),
    )


def random_position(rng: np.random.Generator, max_tiles: int) -> GameState:
    """Deal a real deal and play random legal moves until hands are <= ``max_tiles``."""
    state = new_deal(deal_no=1, rng=rng, lead_mode="six_six")
    guard = 0
    while max(len(h) for h in state.hands) > max_tiles and not state.is_terminal():
        guard += 1
        if guard > 500:                                     # pragma: no cover - defensive
            break
        seat = state.turn
        moves = state.legal_moves(seat)
        if not moves:
            state.pass_turn(seat)
            continue
        tid, end = moves[int(rng.integers(0, len(moves)))]
        state.play(seat, tid, end)
    return state


def playable_position(rng: np.random.Generator, max_tiles: int, tries: int = 25) -> GameState:
    """A non-terminal position whose seat to move has at least one legal move."""
    for _ in range(tries):
        state = random_position(rng, max_tiles)
        if not state.is_terminal() and state.legal_moves(state.turn):
            return state
    raise AssertionError("no actionable position")


def true_world_sampler(counter: list | None = None):
    """A sampler hook that returns the *real* hands (perfect information, one world)."""

    def sampler(view: PublicView, count: int, rng: np.random.Generator) -> list[World]:
        state_ref = sampler.state                             # type: ignore[attr-defined]
        hands = tuple(tuple(sorted(int(t) for t in state_ref.hands[s])) for s in range(4))
        if counter is not None:
            counter.append(count)
        return [World(hands) for _ in range(count)]

    return sampler


def test_acts_legally_on_real_positions() -> None:
    """Every decision is a legal action, on every stage of a deal."""
    rng = np.random.default_rng(41)
    agent = PIMCAgent(worlds=4, horizon=6, time_budget=30.0, seed=3)
    for size in (5, 3, 2):
        state = playable_position(rng, size)
        action = agent.act(state, state.turn)
        legal = {action_index(tid, end) for tid, end in state.legal_moves(state.turn)}
        assert action in legal
        assert agent.last_report.worlds_used >= 1
        assert set(agent.last_report.per_move) == set(state.legal_moves(state.turn))


def test_perfect_information_single_world_matches_the_oracle() -> None:
    """With the true hands and one world, PIMC must play the exact solver's move."""
    rng = np.random.default_rng(42)
    oracle = SolverAgent()
    for _ in range(6):
        state = playable_position(rng, 2)                    # <= 10 plies: the agent is exact
        sampler = true_world_sampler()
        sampler.state = state                                # type: ignore[attr-defined]
        agent = PIMCAgent(worlds=1, horizon=10, time_budget=30.0, seed=1, sampler=sampler)
        action = agent.act(state, state.turn)
        assert agent.last_report.depth is None, "a short deal must be solved exactly"
        assert action == oracle.act(state, state.turn)


def test_worlds_are_paired_across_moves() -> None:
    """One draw per world, and every move is scored on those same worlds."""
    rng = np.random.default_rng(43)
    state = playable_position(rng, 4)
    calls: list[int] = []
    sampler = true_world_sampler(calls)
    sampler.state = state                                    # type: ignore[attr-defined]
    agent = PIMCAgent(worlds=5, horizon=4, time_budget=30.0, seed=2, sampler=sampler)
    agent.act(state, state.turn)
    assert calls == [1] * 5, f"expected 5 single-world draws, got {calls}"
    assert agent.last_report.worlds_used == 5
    assert len(agent.last_report.per_move) == len(state.legal_moves(state.turn))


def test_budget_reduces_worlds_and_logs_it() -> None:
    """A tiny budget still returns an answer, quickly, and says what it gave up."""
    rng = np.random.default_rng(44)
    state = playable_position(rng, 6)
    seen: list[str] = []
    agent = PIMCAgent(worlds=64, horizon=10, time_budget=0.02, seed=4, log=seen.append)
    started = time.perf_counter()
    action = agent.act(state, state.turn)
    elapsed = time.perf_counter() - started
    assert action in {action_index(tid, end) for tid, end in state.legal_moves(state.turn)}
    assert elapsed < 3.0, f"a 20 ms budget took {elapsed:.2f}s"
    assert agent.last_report.worlds_used >= 1
    assert agent.last_report.adjustments, "a reduction must be reported"
    assert seen == agent.last_report.adjustments


def test_sampler_is_drawn_from_the_injected_generator() -> None:
    """Two seeded agents act identically; reset() replays the same sequence."""
    rng = np.random.default_rng(45)
    state = playable_position(rng, 4)
    first = PIMCAgent(worlds=6, horizon=4, time_budget=30.0, seed=11)
    second = PIMCAgent(worlds=6, horizon=4, time_budget=30.0, seed=11)
    a = first.act(state, state.turn)
    b = second.act(state, state.turn)
    assert a == b
    first.reset()
    assert first.act(state, state.turn) == a


def test_horizon_is_used_when_the_deal_is_long() -> None:
    """A long deal gets a horizon; a short one is exact."""
    rng = np.random.default_rng(46)
    long_state = playable_position(rng, 7)
    short_state = playable_position(rng, 1)
    long_agent = PIMCAgent(worlds=2, horizon=6, time_budget=30.0, seed=5)
    long_agent.act(long_state, long_state.turn)
    assert long_agent.last_report.depth == 6
    short_agent = PIMCAgent(worlds=2, horizon=6, time_budget=30.0, seed=5)
    short_agent.act(short_state, short_state.turn)
    assert short_agent.last_report.depth is None


def test_opponent_policy_mode_plays_legally() -> None:
    """The policy opponent model returns legal actions and reports its own depth."""
    rng = np.random.default_rng(47)
    state = playable_position(rng, 3)
    agent = PIMCAgent(worlds=2, horizon=2, time_budget=30.0, seed=6,
                      opponent_model="policy", opponent_policy=RuleAgent(seed=0))
    action = agent.act(state, state.turn)
    assert action in {action_index(tid, end) for tid, end in state.legal_moves(state.turn)}
    assert agent.last_report.worlds_used >= 1
    assert agent.last_report.opponent_model == "policy"


def test_opponent_model_requires_a_policy() -> None:
    """A missing policy is refused instead of silently searching pessimistically."""
    with pytest.raises(ValueError):
        PIMCAgent(opponent_model="policy")
    with pytest.raises(ValueError):
        PIMCAgent(opponent_model="bogus")
    with pytest.raises(ValueError):
        PIMCAgent(worlds=0)


def test_policy_solve_reads_the_opponent_policy() -> None:
    """``policy_solve`` values a hand-checked position, per opponent policy.

    Position: ends (6,1); seat 0 (team 0) holds [2|6] and [1|6]; seat 1 (opponent) holds
    [2|2] and [2|3]; seat 2 (partner) holds [0|3]; seat 3 (opponent) holds [4|4].

    * (12, 0) -> ends (1,1): nobody can follow, four passes block the deal with pips
      [8, 9, 3, 8] -> seat 2 (team 0) is lightest and banks seats 1+3 = 9 + 8 = +17.
    * (12, 1) -> ends (6,6): the opponents pass out, seat 0 plays [2|6] and empties for
      pips(1) + pips(3) = 9 + 8 = +17 (a tie with (12, 0), broken by the lower action).
    * (17, 0) -> ends (2,1) and then the *opponent's policy* decides: playing [2|2] leaves
      seat 1 holding [2|3] and team 0 still banks 5 + 8 = +13, while playing [2|3] lets the
      partner's [0|3] empty the deal first for 4 + 8 = +12.
    """
    hands = [[17, 12], [13, 14], [3], [22]]
    state = make_state(hands, (6, 1), turn=0)

    def first_policy(node, seat):
        return action_index(*sorted(node.legal_moves(seat))[0])

    def last_policy(node, seat):
        return action_index(*sorted(node.legal_moves(seat))[-1])

    first = policy_solve(state, opponent_policy=first_policy)
    last = policy_solve(state, opponent_policy=last_policy)
    assert first.move_values[(12, 0)] == 17
    assert first.move_values[(12, 1)] == 17
    assert last.move_values[(12, 0)] == 17
    assert last.move_values[(12, 1)] == 17
    assert first.move_values[(17, 0)] == 13, first.move_values
    assert last.move_values[(17, 0)] == 12, last.move_values
    assert first.best_move == (12, 0) and last.best_move == (12, 0)
    assert first.value == 17 and last.value == 17
    assert first.nodes > 0 and last.nodes > 0


def test_policy_solve_refuses_an_illegal_policy_action() -> None:
    """A policy that returns a move nobody can make is reported, not played."""
    hands = [[17, 12], [13, 14], [3], [22]]
    state = make_state(hands, (6, 1), turn=0)
    with pytest.raises(ValueError):
        policy_solve(state, opponent_policy=lambda node, seat: 55)   # tile 27, end 1
