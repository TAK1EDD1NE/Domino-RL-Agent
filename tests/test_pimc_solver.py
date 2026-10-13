"""Phase-1 solver tests: hand-checked endgames, table/symmetry controls, reference parity.

Every number in the hand-built cases is derived on paper from the engine's own rules
and stated in the fixture; the fixtures that would need a full search to justify are
instead asserted structurally and against :func:`reference_solve`, which is an
independent exhaustive implementation of the same rules.
"""
from __future__ import annotations

import numpy as np
import pytest

from domino.engine import GameState, new_deal
from domino.masks import action_index
from domino.pimc.solver import EXACT_DEPTH, LeafView, Solver, reference_solve
from domino.tiles import NUM_TILES, tile_pips

# Hand-built positions and their values (team-0 point differential), derived from the
# engine's rules. ``moves`` is the *complete* expectation for the position: values of
# every legal (tile, end) placement. ``passed`` marks a position where the acting seat
# is stuck (mandatory play leaves it no choice).
HAND_CASES = {
    # seat 0 leads its only tile, empties, and its team banks seats 1+3 pips (1 + 12).
    "emptied_on_the_lead": dict(hands=[[27], [0, 1], [7], [13, 22]], ends=(), turn=0,
                                value=13, moves={(27, 0): 13}, passed=False),
    # ends (0,0) and nobody holds a 0: all four pass -> blocked; pip totals
    # [6, 7, 5, 6] -> seat 2 (team 0) is lightest -> seats 1+3 bank 7 + 6.
    "blocked_all_pass_team0_wins": dict(hands=[[7, 13], [8, 9], [14], [18]], ends=(0, 0),
                                        turn=0, value=13, moves={}, passed=True),
    # pip totals [12, 2, 4, 6] -> seat 1 (team 1) lightest -> it banks 12 + 4.
    "blocked_all_pass_team1_wins": dict(hands=[[27], [7], [13], [18]], ends=(0, 0),
                                        turn=0, value=-16, moves={}, passed=True),
    # pip totals [4, 4, 6, 8] -> minimum shared across both teams -> void, 0 points.
    "blocked_cross_team_tie_is_void": dict(hands=[[9], [13], [18], [20]], ends=(0, 0),
                                           turn=0, value=0, moves={}, passed=True),
    # pip totals [4, 6, 4, 8] -> the minimum is inside team 0 -> team 0 still banks
    # seats 1+3: 6 + 8. A same-team tie is *not* void.
    "blocked_same_team_tie_scores": dict(hands=[[9], [18], [13], [20]], ends=(0, 0),
                                         turn=0, value=14, moves={}, passed=True),
    # ends (1,1); seat 0 holds [0|1] and [1|1] (both playable).
    #   play [1|1] -> ends stay (1,1) -> seats 1,2,3 pass -> seat 0 plays [0|1] and
    #     empties: team 0 banks pips(1)+pips(3) = 0 + 3 = +3.
    #   play [0|1] -> ends (0,1) -> seat 1 plays its [0|0] and empties: team 1 banks
    #     pips(0)+pips(2) = 2 + 2 = -4.
    # The same shape with team 1 to move: the value must flip with the seating, and the
    # chosen move must be the *moving* team's best (not the opponent's).
    "emptied_on_the_lead_team1": dict(hands=[[0, 1], [27], [7], [13, 22]], ends=(), turn=1,
                                      value=-3, moves={(27, 0): -3}, passed=False),
    "two_ply_choice": dict(hands=[[1, 7], [0], [2], [3]], ends=(1, 1), turn=0,
                           value=3, moves={(1, 0): -4, (7, 0): 3}, passed=False),
}


def make_state(hands, ends=(), turn=0, pass_streak=0, table_extra=()) -> GameState:
    """Engine state from explicit hands and open ends (dummy table tile when needed).

    The table's own tiles never affect the deal's value in these positions, so a
    single filler tile stands in for the line; only ``ends`` and the hands matter.
    """
    flat = {tile for hand in hands for tile in hand} | set(table_extra)
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
    while max(len(hand) for hand in state.hands) > max_tiles and not state.is_terminal():
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
    """A non-terminal position of at most ``max_tiles`` tiles per hand.

    A random prefix can block the deal, and a finished deal is not solvable, so the
    generator retries instead of handing a terminal state to the solver.
    """
    for _ in range(tries):
        state = random_position(rng, max_tiles)
        if not state.is_terminal():
            return state
    raise AssertionError(f"could not build a non-terminal position at {max_tiles} tiles")


@pytest.mark.parametrize("name", sorted(HAND_CASES))
def test_hand_built_endgames(name: str) -> None:
    """Exact values and per-move values on positions whose arithmetic is checkable."""
    case = HAND_CASES[name]
    state = make_state(case["hands"], case["ends"], turn=case["turn"])
    result = Solver().solve(state)
    assert result.complete and result.exact
    assert result.value == case["value"], f"{name}: got {result.value} of {result.move_values}"
    assert result.move_values == case["moves"]
    assert result.passed == case["passed"]
    expected_to_move = case["value"] if result.team_to_move == 0 else -case["value"]
    assert result.value_to_move == expected_to_move
    if case["moves"]:
        best = max(case["moves"], key=lambda m: (case["moves"][m], -action_index(*m)))
        assert result.best_move == best
        assert result.best_action == action_index(*best)


def test_exact_search_never_evaluates_a_leaf() -> None:
    """``max_depth=None`` must reach the terminals: a leaf call would silently bound."""
    rng = np.random.default_rng(3)
    for _ in range(12):
        state = playable_position(rng, 3)
        result = Solver().solve(state)
        assert result.exact and result.complete
        assert result.leaf_evals == 0


def test_move_values_match_engine_legal_moves() -> None:
    """The solver exposes exactly the engine's legal moves, and its value is the best."""
    rng = np.random.default_rng(5)
    checked_stuck = 0
    for _ in range(12):
        state = playable_position(rng, 4)
        result = Solver().solve(state)
        legal = set(state.legal_moves(state.turn))
        assert set(result.move_values) == legal
        if not legal:                                       # a stuck seat has no choice
            checked_stuck += 1
            assert result.passed and result.best_move is None
            continue
        assert result.value == result.move_values[result.best_move]
        assert result.best_action in set(action_index(tid, end) for tid, end in legal)
        if result.team_to_move == 0:
            assert result.value == max(result.move_values.values())
        else:
            assert result.value == min(result.move_values.values())
    assert checked_stuck < 12, "no position exercised the move-value path"


def test_transposition_table_does_not_change_answers() -> None:
    """Table on and table off agree on the value and on the value of every move."""
    rng = np.random.default_rng(7)
    with_tt = Solver()
    without = Solver(tt_enabled=False)
    for _ in range(10):
        state = playable_position(rng, 3)
        a = with_tt.solve(state)
        b = without.solve(state)
        assert a.value == b.value
        assert a.move_values == b.move_values
        assert a.best_move == b.best_move


def test_repeated_solves_are_identical() -> None:
    """A warm table must not change a later answer (determinism, no state leaks)."""
    rng = np.random.default_rng(9)
    state = playable_position(rng, 3)
    solver = Solver()
    first = solver.solve(state)
    second = solver.solve(state)
    fresh = Solver().solve(state)
    assert first.value == second.value == fresh.value
    assert first.move_values == second.move_values == fresh.move_values


def test_team_swap_negates_the_value() -> None:
    """Rotating every seat by one swaps the teams: every value flips sign.

    The rotation must move the *hands and the turn the same way*: seat ``s + 1`` takes
    over what seat ``s`` held and the turn advances with it. (Shifting the hands one way
    and the turn the other leaves a position that is *not* an image of the original —
    an earlier version of this test did exactly that and "found" a symmetry break.)
    """
    rng = np.random.default_rng(13)
    for _ in range(10):
        state = playable_position(rng, 3)
        if not state.table:                                 # keep the lead rules out of it
            continue
        swapped = make_state([state.hands[(seat - 1) % 4] for seat in range(4)],
                             tuple(state.ends), turn=(state.turn + 1) % 4,
                             pass_streak=state.pass_streak)
        original = Solver().solve(state)
        rotated = Solver().solve(swapped)
        assert rotated.value == -original.value
        assert rotated.move_values == {m: -v for m, v in original.move_values.items()}
        assert rotated.best_move == original.best_move


def test_end_swap_preserves_play() -> None:
    """Which of the two open ends is called left is arbitrary: values must not move."""
    rng = np.random.default_rng(17)
    for _ in range(10):
        state = playable_position(rng, 3)
        if not state.table:
            continue
        mirrored = make_state(state.hands, (int(state.ends[1]), int(state.ends[0])),
                              turn=state.turn, pass_streak=state.pass_streak)
        original = Solver().solve(state)
        flipped = Solver().solve(mirrored)
        assert flipped.value == original.value
        by_tile_original: dict[int, list[int]] = {}
        for (tid, _end), value in original.move_values.items():
            by_tile_original.setdefault(tid, []).append(value)
        by_tile_flipped: dict[int, list[int]] = {}
        for (tid, _end), value in flipped.move_values.items():
            by_tile_flipped.setdefault(tid, []).append(value)
        assert {t: sorted(v) for t, v in by_tile_flipped.items()} == \
               {t: sorted(v) for t, v in by_tile_original.items()}


def test_depth_zero_evaluates_the_children_with_the_leaf() -> None:
    """``max_depth=0`` means one-ply lookahead scored by the leaf estimate."""
    case = HAND_CASES["two_ply_choice"]
    state = make_state(case["hands"], case["ends"], turn=case["turn"])
    result = Solver(max_depth=0).solve(state)
    assert not result.exact and result.complete
    assert result.leaf_evals > 0
    # Leaf values of the two children, in the leaf's own units (see rules.leaf_value):
    # after [0|1] the hands are [7],[0],[2],[3] -> team 0 heavier -> -(2 + 2) = -4;
    # after [1|1] the hands are [1],[0],[2],[3] -> team 0 heavier -> -(1 + 2) = -3.
    assert result.move_values == {(1, 0): -4, (7, 0): -3}
    assert result.value == -3


def test_depth_limit_matches_reference_depth_limit() -> None:
    """The same ply budget gives the same answers on both backends."""
    rng = np.random.default_rng(19)
    for _ in range(6):
        state = playable_position(rng, 3)
        for depth in (1, 2, 4):
            fast = Solver(max_depth=depth).solve(state)
            reference = reference_solve(state, max_depth=depth)
            assert fast.value == reference.value, f"depth {depth}"
            assert fast.move_values == reference.move_values, f"depth {depth}"


def test_reference_parity_on_real_deals() -> None:
    """Exhaustive reference and compiled alpha-beta agree exactly, move by move."""
    rng = np.random.default_rng(23)
    checked = 0
    for _ in range(60):
        state = playable_position(rng, 3)
        fast = Solver().solve(state)
        reference = reference_solve(state)
        assert fast.value == reference.value
        assert fast.move_values == reference.move_values
        assert fast.best_move == reference.best_move
        checked += 1
    assert checked >= 20


def test_node_limit_marks_the_result_incomplete() -> None:
    """A budget stop is reported, never returned as if it were a value."""
    rng = np.random.default_rng(29)
    state = random_position(rng, 3)
    result = Solver(node_limit=1).solve(state)
    assert not result.complete
    assert result.nodes >= 1


def test_callable_leaf_eval_runs_on_the_python_backend() -> None:
    """A pluggable callable is honoured (reference backend) and receives a LeafView."""
    case = HAND_CASES["two_ply_choice"]
    state = make_state(case["hands"], case["ends"], turn=case["turn"])
    case_pips = [sum(tile_pips(t) for t in hand) for hand in case["hands"]]
    seen: list[LeafView] = []

    def leaf(view: LeafView) -> float:
        seen.append(view)
        return -1.0

    solver = Solver(max_depth=0, leaf_eval=leaf)
    result = solver.solve(state)
    assert solver.backend == "python"
    assert result.backend == "python"
    assert len(seen) == 2                                   # one leaf per legal move
    assert all(isinstance(view, LeafView) for view in seen)
    assert set(result.move_values.values()) == {-1}
    assert sum(seen[0].pips) < sum(case_pips)  # a tile was played before the leaf
    for view in seen:
        assert view.turn in (1, 3)                          # the opponent is to move
        assert view.team_to_move == 1


def test_leaf_eval_argument_is_validated() -> None:
    """An unknown mode name is rejected instead of silently falling back."""
    with pytest.raises(ValueError):
        Solver(leaf_eval="nope")


def test_hand_masks_are_canonical() -> None:
    """Compaction is a bijection between a hand and its bitmask (no silent tile loss)."""
    from domino.pimc.rules import compact
    from domino.tiles import tile_pips

    rng = np.random.default_rng(31)
    for _ in range(20):
        state = playable_position(rng, 4)
        hands, pips, _left, _right, _streak = compact(state)
        for seat in range(4):
            mask = int(hands[seat])
            tiles = [t for t in range(NUM_TILES) if (mask >> t) & 1]
            assert tiles == sorted(state.hands[seat])
            assert int(pips[seat]) == sum(tile_pips(t) for t in tiles)
        assert int(pips.sum()) == sum(sum(tile_pips(t) for t in hand)
                                      for hand in state.hands)


def test_generator_determinism_contract() -> None:
    """Seeded generators reproduce a position; unseeded ones differ (the project's bug)."""
    first = random_position(np.random.default_rng(4242), 3)
    second = random_position(np.random.default_rng(4242), 3)
    assert [list(h) for h in first.hands] == [list(h) for h in second.hands]
    assert first.ends == second.ends and first.table == second.table
    left = random_position(np.random.default_rng(), 5)
    right = random_position(np.random.default_rng(), 5)
    assert [list(h) for h in left.hands] != [list(h) for h in right.hands]


def test_max_depth_bounds_are_validated() -> None:
    """The depth sentinel is not a legal budget, and silly table sizes are refused."""
    with pytest.raises(ValueError):
        Solver(max_depth=EXACT_DEPTH)
    with pytest.raises(ValueError):
        Solver(max_depth=-1)
    with pytest.raises(ValueError):
        Solver(tt_bits=4)


def test_solve_refuses_a_terminal_position() -> None:
    """Solving a finished deal is a caller bug, not a 0."""
    state = make_state([[], [0, 1], [7], [13, 22]], ())
    with pytest.raises(ValueError):
        Solver().solve(state)
