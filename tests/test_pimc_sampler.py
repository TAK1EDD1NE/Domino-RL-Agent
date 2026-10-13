"""Phase-2 sampler tests: validity, exact uniformity, history-derived lacks, throughput.

The uniformity and "true deal" checks compare against the *enumerated* set of valid
worlds, which is the strongest available form of the brief's "statistically
indistinguishable from samples" requirement: the sampler's empirical distribution is
compared with the exact one, not with a second sampler.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from domino.engine import new_deal
from domino.pimc.sampler import (
    ACCEPTANCE_THRESHOLD,
    PublicView,
    acceptance_rate,
    bias_report,
    enumerate_worlds,
    sample_worlds,
)
from domino.tiles import NUM_TILES, tile_pips


def random_position(rng: np.random.Generator, max_tiles: int):
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


def playable_position(rng: np.random.Generator, max_tiles: int, tries: int = 25):
    """A non-terminal position of at most ``max_tiles`` tiles per hand."""
    for _ in range(tries):
        state = random_position(rng, max_tiles)
        if not state.is_terminal():
            return state
    raise AssertionError("no non-terminal position")


def assert_world_valid(world, view: PublicView) -> None:
    """Counts, lacks, disjointness and completeness of one sampled world."""
    for seat in range(4):
        hand = world.hand(seat)
        assert len(hand) == len(set(hand)), f"duplicate tile in seat {seat}"
    assert set(world.hand(view.seat)) == set(view.my_hand)
    hidden = [world.hand(seat) for seat in view.other_seats]
    for seat, hand in zip(view.other_seats, hidden):
        assert len(hand) == view.tiles_per_seat - view.played_by[seat]
        for tile in hand:
            a, b = _values(tile)
            assert a not in view.lacks[seat] and b not in view.lacks[seat]
    assert set(view.table).isdisjoint(t for hand in hidden for t in hand)
    assert set(view.my_hand).isdisjoint(t for hand in hidden for t in hand)
    pool = sorted(t for hand in hidden for t in hand)
    assert pool == sorted(view.unseen)


def _values(tile: int) -> tuple[int, int]:
    """Pip values of a tile id."""
    from domino.tiles import tile_values

    return tile_values(int(tile))


def test_public_view_reads_the_public_information() -> None:
    """Counts, unseen pool and lacks come out of the live state unchanged."""
    rng = np.random.default_rng(1)
    state = playable_position(rng, 4)
    seat = state.turn
    view = PublicView.from_state(state, seat)
    assert view.my_hand == tuple(sorted(state.hands[seat]))
    assert view.table == tuple(state.table)
    assert view.ends == (state.ends[0], state.ends[1])
    assert view.counts == tuple(len(state.hands[s]) for s in view.other_seats)
    assert view.lacks == tuple(frozenset(state.lacks[s]) for s in range(4))
    assert len(view.unseen) == sum(view.counts)
    assert set(view.unseen).isdisjoint(view.my_hand) and set(view.unseen).isdisjoint(view.table)


def test_sampled_worlds_satisfy_every_constraint() -> None:
    """Every sampled world of both dealers obeys counts, lacks and disjointness."""
    rng = np.random.default_rng(2)
    positions = [playable_position(rng, size) for size in (6, 4, 3, 2)]
    for state in positions:
        seat = state.turn
        view = PublicView.from_state(state, seat)
        for method in ("rejection", "constrained", "auto"):
            for world in sample_worlds(view, 25, rng, method=method):
                assert_world_valid(world, view)


def test_the_true_deal_is_one_of_the_valid_worlds() -> None:
    """The real hidden hands satisfy the public constraints (the sampler must contain them)."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        state = playable_position(rng, 4)
        seat = state.turn
        view = PublicView.from_state(state, seat)
        true_hands = tuple(tuple(sorted(int(t) for t in state.hands[s])) for s in range(4))
        assert set(view.unseen) == {t for s in view.other_seats for t in true_hands[s]}
        for s in view.other_seats:
            assert len(true_hands[s]) == view.tiles_per_seat - view.played_by[s]
            assert not (set(view.lacks[s]) & {v for t in true_hands[s] for v in _values(t)})


def test_rejection_sampler_is_uniform_over_valid_worlds() -> None:
    """Empirical distribution of the rejection dealer == exact over the enumerated support."""
    rng = np.random.default_rng(4)
    state = playable_position(rng, 3)
    view = PublicView.from_state(state, state.turn)
    report = bias_report(view, np.random.default_rng(5), draws=3000)
    assert report["worlds"] > 20, "the test position must have a non-trivial support"
    noise = report["expected_tv_noise"]
    assert report["tv_rejection"] <= 4.0 * noise + 0.05, report
    assert report["rejection_uniform_compatible"], report
    assert report["chi2_over_dof_rejection"] < 2.0, report
    assert report["chi2_over_dof_constrained"] >= 0.0


def test_true_hand_pip_distribution_matches_the_samples() -> None:
    """The exact pip distribution of a seat's hand matches the sampler's empirical one."""
    rng = np.random.default_rng(6)
    state = playable_position(rng, 3)
    seat = state.turn
    view = PublicView.from_state(state, seat)
    exact_worlds = enumerate_worlds(view)
    target = view.other_seats[0]
    exact: dict[int, float] = {}
    for world in exact_worlds:
        total = sum(tile_pips(t) for t in world.hand(target))
        exact[total] = exact.get(total, 0.0) + 1.0 / len(exact_worlds)
    draws = 4000
    sampled = sample_worlds(view, draws, np.random.default_rng(7), method="rejection")
    empirical: dict[int, float] = {}
    for world in sampled:
        total = sum(tile_pips(t) for t in world.hand(target))
        empirical[total] = empirical.get(total, 0.0) + 1.0 / draws
    for total, probability in exact.items():
        assert abs(empirical.get(total, 0.0) - probability) < 0.06, (
            f"pip total {total}: exact {probability:.3f} sampled "
            f"{empirical.get(total, 0.0):.3f}"
        )


def test_from_events_rebuilds_table_ends_and_lacks() -> None:
    """A play/pass log reproduces the live view (passes included)."""
    rng = np.random.default_rng(8)
    state = new_deal(deal_no=1, rng=rng, lead_mode="six_six")
    events: list[tuple] = []
    for _ in range(40):
        if state.is_terminal() or max(len(h) for h in state.hands) <= 3:
            break
        seat = state.turn
        moves = state.legal_moves(seat)
        if not moves:
            state.pass_turn(seat)
            events.append(("pass", seat))
            continue
        tid, end = moves[int(rng.integers(0, len(moves)))]
        state.play(seat, tid, end)
        events.append(("play", seat, tid, end))
    seat = state.turn
    live = PublicView.from_state(state, seat)
    rebuilt = PublicView.from_events(seat=seat, my_hand=live.my_hand, events=events)
    assert rebuilt.table == live.table
    assert rebuilt.ends == live.ends
    assert rebuilt.played_by == live.played_by
    assert rebuilt.lacks == live.lacks


def test_tight_positions_switch_the_dealer() -> None:
    """A tight view has a low acceptance rate and ``auto`` still returns valid worlds."""
    rng = np.random.default_rng(9)
    view = None
    for _ in range(60):
        state = playable_position(rng, 2)
        candidate = PublicView.from_state(state, state.turn)
        if acceptance_rate(candidate, np.random.default_rng(10), trials=64) < ACCEPTANCE_THRESHOLD:
            view = candidate
            break
    assert view is not None, "no tight position found in 60 deals"
    for world in sample_worlds(view, 30, rng, method="auto"):
        assert_world_valid(world, view)


def test_tile_weights_bias_the_dealer_and_are_recorded() -> None:
    """Weighted sampling concentrates on the preferred tile and reports its weight."""
    rng = np.random.default_rng(11)
    state = playable_position(rng, 3)
    view = PublicView.from_state(state, state.turn)
    target_seat = view.other_seats[1]
    target_tile = max(view.unseen)

    def weights(seat: int, tile: int) -> float:
        return 25.0 if (seat == target_seat and tile == target_tile) else 1.0

    worlds = sample_worlds(view, 60, rng, method="constrained", tile_weights=weights)
    hits = sum(1 for w in worlds if target_tile in w.hand(target_seat))
    baseline = view.counts[list(view.other_seats).index(target_seat)] / len(view.unseen)
    assert hits / len(worlds) > 2.0 * baseline
    assert any(w.weight > 1.0 for w in worlds)


def test_rejection_refuses_weights_and_bad_input() -> None:
    """Weights must not silently break the uniform path, and bad inputs are refused."""
    rng = np.random.default_rng(12)
    state = playable_position(rng, 3)
    view = PublicView.from_state(state, state.turn)
    with pytest.raises(ValueError):
        sample_worlds(view, 5, rng, method="rejection", tile_weights=np.ones(NUM_TILES))
    with pytest.raises(ValueError):
        sample_worlds(view, 0, rng)
    with pytest.raises(ValueError):
        sample_worlds(view, 5, rng, method="bogus")
    with pytest.raises(TypeError):
        sample_worlds(view, 5, 12345)                        # not a Generator


def test_unsatisfiable_view_is_reported() -> None:
    """A view whose counts contradict the unseen pool raises instead of guessing."""
    rng = np.random.default_rng(13)
    state = playable_position(rng, 3)
    view = PublicView.from_state(state, state.turn)
    other = view.other_seats[0]                              # a seat that must be sampled
    played = list(view.played_by)
    played[other] += 1                                       # counts no longer add up
    broken = PublicView(
        seat=view.seat, my_hand=view.my_hand, table=view.table, ends=view.ends,
        turn=view.turn, pass_streak=view.pass_streak,
        played_by=tuple(played),                             # type: ignore[arg-type]
        lacks=view.lacks,
    )
    with pytest.raises(ValueError):
        sample_worlds(broken, 1, rng)


def test_sampler_determinism_contract() -> None:
    """Seeded generators reproduce the worlds; unseeded ones differ."""
    state = playable_position(np.random.default_rng(14), 3)
    view = PublicView.from_state(state, state.turn)
    first = sample_worlds(view, 5, np.random.default_rng(99))
    second = sample_worlds(view, 5, np.random.default_rng(99))
    assert [w.as_tuple() for w in first] == [w.as_tuple() for w in second]
    left = sample_worlds(view, 5, np.random.default_rng())
    right = sample_worlds(view, 5, np.random.default_rng())
    assert [w.as_tuple() for w in left] != [w.as_tuple() for w in right]


def test_sampler_throughput_floor() -> None:
    """A cheap floor on worlds/second (the real numbers live in tools/bench_sampler.py)."""
    rng = np.random.default_rng(15)
    state = playable_position(rng, 5)
    view = PublicView.from_state(state, state.turn)
    for method, floor in (("rejection", 200.0), ("constrained", 200.0)):
        started = time.perf_counter()
        worlds = sample_worlds(view, 400, rng, method=method)
        elapsed = time.perf_counter() - started
        rate = len(worlds) / elapsed
        assert rate > floor, f"{method}: {rate:.0f} worlds/s"
