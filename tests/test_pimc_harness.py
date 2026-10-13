"""Phase-5 harness tests: duplicate-deal protocol, CI math, Elo fit, incremental CSV."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from domino.pimc.harness import (
    AgentSpec,
    append_row,
    bootstrap_mean_ci,
    bradley_terry,
    default_workers,
    elo_from_strength,
    estimate_runtime,
    evaluate_matchup,
    play_duplicate_pair,
    rotate_hands,
    round_robin,
    wilson_interval,
)


def test_rotation_swaps_the_teams_and_is_a_fourth_root_of_identity() -> None:
    """Rotating the seating swaps the teams; four rotations return the deal."""
    from domino.pimc.harness import deal_hands
    from domino.tiles import SIX_SIX_ID

    hands = deal_hands(np.random.default_rng(0))
    rotated = rotate_hands(hands)
    for seat in range(4):
        assert sorted(rotated[(seat + 1) % 4]) == sorted(hands[seat])
    twice = rotate_hands(rotated)
    for seat in range(4):
        assert sorted(twice[seat]) == sorted(hands[(seat - 2) % 4])
    four = rotate_hands(rotate_hands(twice))
    assert [sorted(hand) for hand in four] == [sorted(hand) for hand in hands]
    holder = next(seat for seat, hand in enumerate(hands) if SIX_SIX_ID in hand)
    holder_rotated = next(seat for seat, hand in enumerate(rotated) if SIX_SIX_ID in hand)
    assert holder_rotated == (holder + 1) % 4


def test_duplicate_pair_is_antisymmetric_for_identical_agents() -> None:
    """With the same agent on both sides the two orientations are exact mirror images."""
    spec = AgentSpec(name="rule", kind="rule")
    for seed in range(5):
        row = play_duplicate_pair(spec, spec, seed)
        assert row["differential_game2"] == -row["differential_game1"]


def test_identity_matchup_is_even() -> None:
    """Rule vs rule over 40 deals: win rate ~50 % and the paired margin ~0."""
    spec = AgentSpec(name="rule", kind="rule")
    row = evaluate_matchup(spec, spec, deals=40, seed=1, workers=1)
    assert row["deals"] == 40 and row["games"] == 80
    assert row["wilson_low"] <= 0.5 <= row["wilson_high"]
    assert abs(row["paired_differential"]) < 4.0
    assert abs(row["win_rate_a"] - 0.5) < 0.15


def test_random_loses_to_the_rule_bot() -> None:
    """The baseline the brief asks for: the rule bot beats random, pooled over seeds.

    Deliberately *pooled* and deliberately checking direction rather than an interval: this
    project has measured a between-seed SD of ~0.02-0.04 on a 400-deal evaluation, so a
    single-seed interval would be claiming more than the sample supports.
    """
    wins = games = 0
    margins = []
    for seed in range(3):
        row = evaluate_matchup(AgentSpec(name="rule", kind="rule"),
                               AgentSpec(name="random", kind="random"),
                               deals=40, seed=seed, workers=1)
        assert row["wilson_low"] <= row["win_rate_a"] <= row["wilson_high"]   # plumbing
        wins += row["wins_a"]
        games += row["games"]
        margins.append(row["paired_differential"])
    pooled = wins / games
    assert pooled > 0.5, f"pooled rule-vs-random win rate {pooled:.3f}"
    assert sum(margins) / len(margins) > 0


def test_statistics_match_their_definitions() -> None:
    """Wilson and bootstrap intervals behave on inputs with known answers."""
    low, high = wilson_interval(500, 1000)
    assert 0.46 < low < 0.47 and 0.53 < high < 0.54
    assert low <= 0.5 <= high
    assert wilson_interval(10, 10)[0] > 0.7
    assert all(math.isnan(value) for value in wilson_interval(0, 0))
    rng = np.random.default_rng(3)
    constant = bootstrap_mean_ci([5.0, 5.0, 5.0], rng)
    assert constant == (5.0, 5.0)
    spread = bootstrap_mean_ci([-10.0, 10.0] * 50, rng, draws=2000)
    assert spread[0] < 0 < spread[1]
    assert bootstrap_mean_ci([], rng)[0] != bootstrap_mean_ci([], rng)[0]  # nan


def test_bradley_terry_recovers_a_known_order() -> None:
    """A clearly ordered set of pairwise results must come back in that order."""
    results = {
        ("a", "b"): (90, 10),
        ("b", "a"): (10, 90),
        ("b", "c"): (80, 20),
        ("c", "b"): (20, 80),
        ("a", "c"): (95, 5),
        ("c", "a"): (5, 95),
    }
    strength = bradley_terry(results)
    assert strength["a"] > strength["b"] > strength["c"]
    elo = elo_from_strength(strength)
    assert elo["a"] > elo["b"] > elo["c"]
    assert abs(np.mean(list(elo.values())) - 1500.0) < 1e-6
    assert elo["a"] - elo["c"] > 200


def test_round_robin_appends_rows_incrementally(tmp_path) -> None:
    """Every matchup row lands in the CSV as soon as it is known."""
    path = tmp_path / "rr.csv"
    specs = [AgentSpec("rule", "rule"), AgentSpec("random", "random")]
    rows = round_robin(specs, deals=6, seed=4, workers=1, out_csv=path)
    assert len(rows) == 1
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    header = lines[0].split(",")
    assert "paired_differential" in header and "wilson_low" in header
    # A second run appends instead of replacing (a crash loses nothing).
    round_robin(specs, deals=6, seed=5, workers=1, out_csv=path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_workers_are_capped_below_the_core_count() -> None:
    """The default worker count leaves two cores for the laptop."""
    assert default_workers() <= max(1, (os.cpu_count() or 2) - 2)
    assert default_workers() >= 1


def test_runtime_estimate_is_printed() -> None:
    """A long run is announced before it starts."""
    text = estimate_runtime([AgentSpec("rule", "rule"), AgentSpec("random", "random")],
                            deals=10, seed=6, workers=2)
    assert "estimate" in text and "matchup" in text and "min" in text


def test_append_row_tolerates_missing_optional_columns(tmp_path) -> None:
    """Rows missing an optional column still write a well-formed CSV."""
    path = tmp_path / "row.csv"
    append_row(path, {"matchup": "x_vs_y", "agent_a": "x", "agent_b": "y", "deals": 1,
                      "games": 2, "wins_a": 1, "wins_b": 1})
    line = path.read_text(encoding="utf-8").strip().splitlines()[1]
    assert line.startswith("x_vs_y,x,y,1,2,1,1")
    assert len(line.split(",")) == len(path.read_text(encoding="utf-8")
                                        .strip().splitlines()[0].split(","))


def test_invalid_spec_is_refused() -> None:
    """An unknown kind or a missing model path is an error, not a silent default."""
    with pytest.raises(ValueError):
        AgentSpec("z", "bogus").build(0)
    with pytest.raises(ValueError):
        AgentSpec("rl", "rl", {}).build(0)
