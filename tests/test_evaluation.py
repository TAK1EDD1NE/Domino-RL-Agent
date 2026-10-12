"""Tests for the evaluation script ``test.py`` and its engine hooks.

``test.py`` is a mandatory root-level deliverable and cannot live in ``tests/``,
so it is loaded here by path and also exercised end-to-end as a subprocess.
"""
from __future__ import annotations

import csv
import importlib.util
import math
import pathlib
import subprocess
import sys

import pytest

from domino.agents import RandomAgent
from domino.engine import Match
from domino.masks import action_index, decode_action
from domino.tiles import tile_id

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "test.py"


def _load_script():
    """Import the root-level ``test.py`` by path (it is not a package module)."""
    spec = importlib.util.spec_from_file_location("domino_eval_script", SCRIPT)
    assert spec and spec.loader, "cannot load test.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


# --------------------------------------------------------------------------- #
# Wilson interval
# --------------------------------------------------------------------------- #
def test_wilson_interval_brackets_the_rate_and_stays_in_range(script):
    low, high = script.wilson_interval(57, 100)
    assert 0.0 <= low < 0.57 < high <= 1.0
    assert low < high
    # A 50 % sample must bracket 0.5 symmetrically.
    low, high = script.wilson_interval(50, 100)
    assert low < 0.5 < high
    assert math.isclose(0.5 - low, high - 0.5, rel_tol=1e-9)


def test_wilson_interval_handles_zero_and_perfect_samples(script):
    low, high = script.wilson_interval(0, 10)
    assert low == 0.0 and 0.0 < high < 0.35
    low, high = script.wilson_interval(10, 10)
    assert high == 1.0 and 0.65 < low < 1.0
    low, high = script.wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


# --------------------------------------------------------------------------- #
# Engine hook: Match.play_deal with the 56-dim (tile, end) action space
# --------------------------------------------------------------------------- #
def test_match_play_deal_accepts_56_space_actions(script):
    match = Match(target=100, seed=3)
    policy = {seat: RandomAgent(seed=100 + seat).act for seat in range(4)}
    outcome = match.play_deal(policy, action_decoder=decode_action)
    assert outcome.points > 0
    assert outcome.winner_team in (0, 1)
    assert match.deal_no == 2
    assert match.scores[outcome.winner_team] == outcome.points


def test_match_play_deal_decoder_rejects_an_unplayable_tile(script):
    match = Match(target=100, seed=3)
    # The forced [6|6] opener cannot be satisfied by [0|0].
    policy = {seat: (lambda state, seat: action_index(tile_id(0, 0), 0)) for seat in range(4)}
    with pytest.raises(ValueError, match="illegal action"):
        match.play_deal(policy, action_decoder=decode_action)


def test_match_play_deal_without_decoder_still_takes_tile_ids(script):
    """The pre-existing tile-id contract must keep working (no silent behaviour change)."""
    match = Match(target=100, seed=5)
    policy = {seat: (lambda state, seat: state.legal_actions(seat)[0]) for seat in range(4)}
    outcome = match.play_deal(policy)
    assert outcome.points > 0
    assert outcome.winner_team in (0, 1)
    assert match.deals == [outcome]


# --------------------------------------------------------------------------- #
# Deal driver
# --------------------------------------------------------------------------- #
def test_play_deal_with_agents_is_deterministic_and_scores(script):
    def run():
        agents = {
            seat: RandomAgent(seed=7 + seat) for seat in range(4)
        }
        rng = __import__("numpy").random.default_rng(11)
        results = []
        for _ in range(20):
            outcome, voids, state = script.play_deal_with_agents(agents, rng)
            results.append((outcome.winner_team, outcome.points, outcome.reason, len(voids)))
        return results

    first, second = run(), run()
    assert first == second
    assert all(points > 0 for _, points, reason, _ in first if reason != "void_tie")


def test_evaluate_matchup_metrics_are_self_consistent(script):
    args = script.build_parser().parse_args(
        ["--matchups", "rule_vs_random", "--deals", "20", "--matches", "1", "--seed", "2",
         "--model", "models/checkpoints/latest.zip"]
    )
    row = script.evaluate_matchup("rule_vs_random", args)

    assert row["deals"] == row["wins_a"] + row["wins_b"]
    assert row["deals"] == 20
    assert 0.0 <= row["win_rate_a"] <= 1.0
    assert row["ci_low"] <= row["win_rate_a"] <= row["ci_high"]
    assert 0.0 <= row["block_rate"] <= 1.0
    assert 0.0 <= row["void_rate"] <= 1.0
    assert row["mean_pips_left_a"] >= 0.0
    assert row["ms_per_action"] > 0.0
    assert row["points_per_deal_a"] >= 0.0 and row["points_per_deal_b"] >= 0.0
    assert row["matches_played"] == 1
    assert row["match_win_rate_a"] in (0.0, 1.0)


# --------------------------------------------------------------------------- #
# End-to-end script
# --------------------------------------------------------------------------- #
def test_test_py_writes_the_documented_schema(tmp_path):
    out = tmp_path / "eval.csv"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--matchups", "rule_vs_random", "--deals", "40",
         "--matches", "1", "--seed", "4", "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.exists()

    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert list(row.keys()) == list(_load_script().CSV_FIELDS)
    assert row["matchup"] == "rule_vs_random"
    assert int(row["deals"]) == int(row["wins_a"]) + int(row["wins_b"]) == 40
    assert 0.0 <= float(row["win_rate_a"]) <= 1.0
    assert float(row["ci_low"]) <= float(row["win_rate_a"]) <= float(row["ci_high"])


def test_test_py_rejects_an_unknown_matchup(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--matchups", "rl_vs_typo", "--deals", "5",
         "--matches", "0", "--out", str(tmp_path / "x.csv")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode != 0
    assert "unknown matchup" in (proc.stdout + proc.stderr)
