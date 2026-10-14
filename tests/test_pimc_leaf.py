"""Tests for the network value leaves and their wiring into ``PIMCAgent``.

The leaf is the piece that lets a *trained* value function stand in for the compiled
solver when the deal is still long. What can go wrong is quiet: the privileged block not
carrying the sampled world, a sign flip on a team-1 seat, or a terminal child being
approximated by a network instead of paid out exactly. Each of those has a test here.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th

from domino.engine import GameState, deal_outcome
from domino.obs import OBS_DIM, PRIVILEGED_SLICE
from domino.pimc.leaf import NetLeaf
from domino.pimc.player import PIMCAgent
from domino.pimc.sampler import PublicView, World, sample_worlds
from domino.tiles import ALL_TILE_IDS, SIX_SIX_ID


# ---- stubs -----------------------------------------------------------------
class StubPolicy:
    """A ``policy`` with the two value entry points and a call counter."""

    device = "cpu"

    def __init__(self, value: float = 1.0, fn=None) -> None:
        self.value = float(value)
        self.fn = fn
        self.calls = 0

    def _out(self, observations) -> th.Tensor:
        self.calls += 1
        batch = np.asarray(observations.detach().cpu().numpy(), dtype=np.float32)
        if self.fn is None:
            return th.full((batch.shape[0], 1), self.value)
        return th.as_tensor(np.asarray(self.fn(batch), dtype=np.float32)).reshape(-1, 1)

    def predict_values(self, observations) -> th.Tensor:              # noqa: D102
        return self._out(observations)

    def predict_public_values(self, observations) -> th.Tensor:       # noqa: D102
        return self._out(observations)


class StubModel:
    """Minimal sb3-shaped model: just a ``policy``."""

    def __init__(self, **kwargs) -> None:
        self.policy = StubPolicy(**kwargs)


# ---- fixtures --------------------------------------------------------------
def fresh_state(seed: int = 0) -> GameState:
    """A real opening deal: four 7-tile hands, the ``[6|6]`` holder to move."""
    rng = np.random.default_rng(seed)
    tiles = list(ALL_TILE_IDS)
    rng.shuffle(tiles)
    hands = [sorted(tiles[index * 7:(index + 1) * 7]) for index in range(4)]
    leader = next(index for index, hand in enumerate(hands) if SIX_SIX_ID in hand)
    return GameState(hands=[list(hand) for hand in hands], rng=rng, turn=leader, leader=leader,
                     deal_no=1, forced_leader_six_six=True)


def deal_down(stop_remaining: int = 8, seed: int = 0) -> GameState:
    """A real deal advanced until at most ``stop_remaining`` tiles are left in hand.

    A hand-crafted short-hand state cannot be used with ``PIMCAgent``: the sampler's view
    assumes every tile outside the acting seat's hand and the table belongs to an opponent,
    so only a real deal played down gives a consistent position. The loop stops *before* the
    play that would end the deal, so the fixture is always a live position - and with
    ``stop_remaining=4`` that leaves the acting seat holding a tile whose play ends the deal,
    which is exactly what the terminal-leaf test needs.
    """
    state = fresh_state(seed)
    for _ in range(400):
        if state.is_terminal() or sum(len(hand) for hand in state.hands) <= int(stop_remaining):
            break
        turn = int(state.turn)
        moves = state.legal_moves(turn)
        if not moves:
            state.pass_turn(turn)
            continue
        tid, end = moves[0]
        child = state.clone()
        child.play(turn, tid, end)
        if child.is_terminal():
            break
        state.play(turn, tid, end)
    assert not state.is_terminal(), "the fixture must stop before the deal ends"
    return state


def worlds_for(state: GameState, seat: int, count: int = 2, seed: int = 1) -> list[World]:
    """Consistent worlds for a live position."""
    view = PublicView.from_state(state, seat)
    return sample_worlds(view, count, np.random.default_rng(seed), method="rejection")


# ---- the leaf --------------------------------------------------------------
def test_constant_leaf_averages_over_worlds() -> None:
    """With a constant evaluator every move scores that constant, per world."""
    state = fresh_state(0)
    seat = int(state.turn)
    moves = [(int(tid), int(end)) for tid, end in state.legal_moves(seat)]
    leaf = NetLeaf(StubModel(value=2.5), mode="critic")
    values = leaf.value_of_moves(state, seat, moves, worlds_for(state, seat, 3))
    assert set(values) == set(moves)
    assert all(value == pytest.approx(2.5) for value in values.values())
    assert leaf.leaves == 3 * len(moves)


def craft_world(state: GameState, seat: int, holder_of_tile0: int) -> World:
    """A world of the right *sizes* with tile 0 placed in a chosen opponent's hand.

    The opponents' hands are built from the unseen pool only (not from their true hands, which
    would leak the real owner of tile 0 and silently defeat the point of the test).
    """
    view = PublicView.from_state(state, seat)
    pool = [int(t) for t in view.unseen if int(t) != 0]
    opponents = [index for index in range(4) if index != int(seat)]
    sizes = {index: 7 for index in opponents}
    sizes[int(holder_of_tile0)] -= 1                     # room for tile 0
    hands: list[list[int]] = [list(state.hands[index]) for index in range(4)]
    cursor = 0
    for index in opponents:
        hands[index] = sorted(pool[cursor:cursor + sizes[index]])
        cursor += sizes[index]
    hands[int(holder_of_tile0)] = sorted(hands[int(holder_of_tile0)] + [0])
    assert cursor == len(pool), (cursor, len(pool))
    return World(tuple(tuple(hand) for hand in hands))


def test_leaf_reads_the_sampled_world_not_the_true_hands() -> None:
    """The privileged block must be filled from the world: that is what the critic reads."""
    state = fresh_state(0)
    seat = int(state.turn)
    moves = [(int(tid), int(end)) for tid, end in state.legal_moves(seat)]
    left = (seat + 1) % 4
    partner = (seat + 2) % 4
    # Tile 0 with the left opponent in one world and with the partner in the other: the stub
    # answers "is tile 0 in the left opponent's slot?", so the mean must be exactly 0.5.
    worlds = [craft_world(state, seat, left), craft_world(state, seat, partner)]
    stub = StubModel(fn=lambda batch: batch[:, PRIVILEGED_SLICE][:, 0].reshape(-1, 1))
    values = NetLeaf(stub, mode="critic").value_of_moves(state, seat, moves, worlds)
    assert all(value == pytest.approx(0.5) for value in values.values()), values
    # Flip both worlds so the left opponent holds tile 0 in both: the mean must be 1.
    again = [craft_world(state, seat, left), craft_world(state, seat, left)]
    values = NetLeaf(stub, mode="critic").value_of_moves(state, seat, moves, again)
    assert all(value == pytest.approx(1.0) for value in values.values()), values


def test_terminal_children_are_paid_exactly_not_estimated() -> None:
    """A move that ends the deal must use the engine's outcome, never the network.

    The payout is the *world's* outcome, not the true state's: inside PIMC a determinised
    world is treated as the truth, so the emptying move pays whatever the sampled hands make
    it worth, averaged over the worlds. Comparing against the true state's outcome would be
    the wrong expectation (and a cross-team tie inside a sampled world pays 0).
    """
    from domino.pimc.player import materialise_world

    state = deal_down(stop_remaining=4, seed=0)          # one tile per seat, deal alive
    seat = int(state.turn)
    moves = [(int(tid), int(end)) for tid, end in state.legal_moves(seat)]
    assert moves, "the fixture's acting seat must have a legal move"
    worlds = worlds_for(state, seat, 2)
    stub = StubModel(value=999.0)
    leaf = NetLeaf(stub, mode="critic")
    values = leaf.value_of_moves(state, seat, moves, worlds)

    def expected_of(tid: int, end: int) -> float:
        total = 0.0
        for world in worlds:
            child = materialise_world(state, world)
            child.play(seat, int(tid), int(end))
            if not child.is_terminal():
                return 999.0
            outcome = deal_outcome(child)
            payout = 0.0 if outcome.void else float(outcome.points)
            winner = getattr(outcome, "winner_team", None)
            total += 0.0 if winner is None else (payout if winner == 0 else -payout)
        return total / float(len(worlds))

    termini = 0
    for tid, end in moves:
        child = state.clone()
        child.play(seat, tid, end)
        if child.is_terminal():
            termini += 1
            assert values[(tid, end)] == pytest.approx(expected_of(tid, end))
        else:
            assert values[(tid, end)] == pytest.approx(999.0)
    assert leaf.leaves == (len(moves) - termini) * len(worlds), "terminal leaves not forwarded"
    assert termini > 0, "the fixture must contain an emptying move"


def test_public_mode_uses_the_public_head() -> None:
    """``mode="public"`` must call the public value head, not the critic."""
    seen: list[str] = []

    class Recorder(StubPolicy):
        def predict_values(self, observations):                      # noqa: D102
            seen.append("critic")
            return super().predict_values(observations)

        def predict_public_values(self, observations):                # noqa: D102
            seen.append("public")
            return super().predict_public_values(observations)

    model = StubModel()
    model.policy = Recorder(value=1.0)
    state = fresh_state(0)
    seat = int(state.turn)
    moves = [(int(tid), int(end)) for tid, end in state.legal_moves(seat)]
    NetLeaf(model, mode="public").value_of_moves(state, seat, moves, worlds_for(state, seat, 1))
    assert seen and set(seen) == {"public"}


def test_leaf_rejects_bad_configuration() -> None:
    """Unknown modes and non-models are configuration errors, not silent defaults."""
    with pytest.raises(ValueError):
        NetLeaf(StubModel(), mode="rollout")
    with pytest.raises(TypeError):
        NetLeaf(object(), mode="critic")


# ---- the agent -------------------------------------------------------------
def test_agent_uses_the_net_leaf_only_when_the_deal_is_long() -> None:
    """Short deals stay exact (the compiled solver), long ones go to the network leaf."""
    model = StubModel(value=0.25)
    long_state = fresh_state(0)
    agent = PIMCAgent(worlds=2, horizon=8, time_budget=5.0, leaf_eval="critic", model=model,
                      seed=0)
    action = agent.act(long_state, int(long_state.turn))
    assert agent.last_report.depth == 1, "a long deal must take the net-leaf path"
    assert model.policy.calls >= 1
    assert 0 <= action < 56

    calls_after = model.policy.calls
    short_state = deal_down(stop_remaining=8, seed=0)
    assert sum(len(hand) for hand in short_state.hands) <= 8
    agent.act(short_state, int(short_state.turn))
    assert agent.last_report.depth is None, "a short deal must be solved exactly"
    assert model.policy.calls == calls_after, "the network must not be consulted when exact"


def test_agent_requires_a_model_for_net_leaf_modes() -> None:
    """Without a model the compiled kernel cannot call a network: refuse, do not guess."""
    with pytest.raises(ValueError, match="needs a trained model"):
        PIMCAgent(leaf_eval="critic")
    with pytest.raises(ValueError, match="needs a trained model"):
        PIMCAgent(leaf_eval="public")


def test_agent_picks_the_move_the_leaf_prefers() -> None:
    """The chosen action must be the argmax of the leaf's per-move values (sign included)."""
    state = fresh_state(0)
    seat = int(state.turn)
    moves = [(int(tid), int(end)) for tid, end in state.legal_moves(seat)]
    # A stub that prefers a specific move: score = the tile id, so the highest tile wins.
    def by_tile(batch: np.ndarray) -> np.ndarray:
        return np.ones((batch.shape[0], 1), dtype=np.float32)

    fixed_worlds = worlds_for(state, seat, 2, seed=5)
    agent = PIMCAgent(worlds=2, horizon=8, time_budget=5.0, leaf_eval="critic",
                      model=StubModel(fn=by_tile), seed=0,
                      sampler=lambda view, count, rng: list(fixed_worlds)[:count])
    action = agent.act(state, seat)
    from domino.masks import action_index

    assert agent.last_report.depth == 1
    chosen = agent.last_report.best_move
    assert action == action_index(*chosen)
    # Every move scored identically here, so the lowest action index wins the tie.
    assert action == min(action_index(*move) for move in moves)


def test_root_all_moves_matches_the_per_move_exact_path() -> None:
    """One exact solve for the whole root must choose exactly what one-solve-per-move does.

    The offline teacher relies on this: solving the root once per world is ~1/len(moves) of the
    cost, and it is only legitimate if the argmax is unchanged.
    """
    state = None
    for seed in range(8):
        candidate = deal_down(stop_remaining=6, seed=seed)
        if len(candidate.legal_moves(int(candidate.turn))) >= 2:
            state = candidate
            break
    assert state is not None, "no fixture seed left the acting seat with a real choice"
    seat = int(state.turn)
    worlds = worlds_for(state, seat, 3, seed=11)
    common = dict(worlds=3, horizon=28, time_budget=60.0, leaf_eval="blocked_now", seed=0,
                  sampler=lambda view, count, rng: list(worlds)[:count])
    per_move = PIMCAgent(**common)
    whole_root = PIMCAgent(**common, root_all_moves=True)
    assert per_move.act(state, seat) == whole_root.act(state, seat)
    assert per_move.last_report.best_move == whole_root.last_report.best_move
    assert per_move.last_report.depth is None and whole_root.last_report.depth is None
