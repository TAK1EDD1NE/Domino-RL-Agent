"""Tests for ``tools/eval_ladder.py``: the Elo arithmetic and the row schema.

No model is loaded and no game is played: the ladder's pure helpers and the CSV
writer are exercised on synthetic ``test.py``-shaped rows, and ``--dry-run`` is
proven to avoid the evaluation path entirely (``build_parser`` is booby-trapped).
"""
from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

import pytest

import tools.eval_ladder as ladder
from tools.eval_ladder import (
    ANCHOR_ELO,
    CSV_FIELDS,
    ELO_BASE,
    LADDER_FIELDS,
    LADDER_MATCHUPS,
    build_plan,
    checkpoint_step,
    compute_elo,
    elo_update,
    expected_score,
    finalize_rows,
    format_ladder_table,
    main,
    parse_seeds,
    pooled_win_rate,
    resolve_checkpoints,
    write_ladder_csv,
)


def _synthetic_row(
    checkpoint: str, matchup: str, win_rate: float, seed: int = 0, deals: int = 100
) -> dict[str, object]:
    """One ``test.py``-shaped row with every documented column present."""
    wins_a = int(round(win_rate * deals))
    return {
        "matchup": matchup,
        "agent_a": "rl",
        "agent_b": "rule" if matchup != "rl_vs_random" else "random",
        "deals": deals,
        "wins_a": wins_a,
        "wins_b": deals - wins_a,
        "win_rate_a": win_rate,
        "ci_low": 0.0,
        "ci_high": 1.0,
        "points_per_deal_a": 1.0,
        "points_per_deal_b": 1.0,
        "match_win_rate_a": win_rate,
        "matches_played": 1,
        "block_rate": 0.1,
        "void_rate": 0.01,
        "mean_pips_left_a": 5.0,
        "ms_per_action": 0.5,
        "model_path": checkpoint,
        "seed": seed,
        "checkpoint": checkpoint,
    }


# --------------------------------------------------------------------------- #
# Row schema
# --------------------------------------------------------------------------- #
def test_ladder_fields_is_the_test_schema_plus_checkpoint_and_elo():
    assert LADDER_FIELDS == tuple(CSV_FIELDS) + ("checkpoint", "elo")
    assert "checkpoint" in LADDER_FIELDS and "elo" in LADDER_FIELDS
    assert LADDER_FIELDS[:-2] == tuple(CSV_FIELDS), "test.py column order is preserved"


def test_write_ladder_csv_headers_and_round_trips(tmp_path):
    rows = finalize_rows([
        _synthetic_row("models/a.zip", "rl_vs_rule", 0.6),
        _synthetic_row("models/a.zip", "rl_vs_random", 0.9, seed=1),
    ])
    out = write_ladder_csv(rows, tmp_path / "ladder_x.csv")
    with out.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)
    assert list(reader.fieldnames or []) == list(LADDER_FIELDS)
    assert len(written) == 2
    assert written[0]["checkpoint"] == "models/a.zip"
    assert written[0]["matchup"] == "rl_vs_rule"
    assert float(written[0]["elo"]) == float(rows[0]["elo"])
    assert set(written[0]) == set(LADDER_FIELDS)


# --------------------------------------------------------------------------- #
# Elo arithmetic
# --------------------------------------------------------------------------- #
def test_expected_score_is_symmetric_and_monotonic():
    assert expected_score(1200, 1200) == pytest.approx(0.5)
    assert expected_score(1400, 1200) == pytest.approx(1.0 / (1.0 + 10.0 ** -0.5))
    assert expected_score(1000, 1200) == pytest.approx(1.0 / (1.0 + 10.0 ** 0.5))
    # A 200-point edge means the underdog's expectation is the favourite's complement.
    assert expected_score(1000, 1200) == pytest.approx(1.0 - expected_score(1200, 1000))
    assert expected_score(1300, 1200) > expected_score(1250, 1200) > 0.5


def test_elo_update_equal_ratings_gives_k_over_two():
    assert elo_update(1200, 1200, 1.0, k=16) == pytest.approx(1208.0)
    assert elo_update(1200, 1200, 0.0, k=16) == pytest.approx(1192.0)
    assert elo_update(1200, 1200, 0.5, k=16) == pytest.approx(1200.0)


def test_elo_update_uses_the_partial_score_linearly():
    # A 0.75 score is halfway between a win (+8) and a draw (0) at equal ratings.
    assert elo_update(1200, 1200, 0.75, k=16) == pytest.approx(1204.0)
    assert elo_update(1200, 1200, 0.25, k=16) == pytest.approx(1196.0)


def test_compute_elo_folds_each_checkpoints_rows_in_order():
    rows = [_synthetic_row("models/a.zip", "rl_vs_rule", 1.0, seed=s) for s in range(3)]
    ratings = compute_elo(rows)
    expected = ELO_BASE
    for _ in range(3):
        expected = elo_update(expected, ANCHOR_ELO, 1.0, k=16)
    assert ratings == {"models/a.zip": pytest.approx(expected)}
    assert ratings["models/a.zip"] > ELO_BASE, "three wins must raise the rating"


def test_compute_elo_checkpoints_are_independent():
    strong = [_synthetic_row("strong.zip", "rl_vs_rule", 1.0, seed=s) for s in range(3)]
    weak = [_synthetic_row("weak.zip", "rl_vs_rule", 0.0, seed=s) for s in range(3)]
    interleaved = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]
    ratings = compute_elo(interleaved)
    assert ratings["strong.zip"] > ELO_BASE > ratings["weak.zip"]
    assert ratings["strong.zip"] == pytest.approx(compute_elo(strong)["strong.zip"])
    assert ratings["weak.zip"] == pytest.approx(compute_elo(weak)["weak.zip"])


def test_compute_elo_skips_rows_without_a_score():
    rows = [
        _synthetic_row("models/a.zip", "rl_vs_rule", 1.0),
        {**_synthetic_row("models/a.zip", "rl_vs_random", 0.0), "win_rate_a": float("nan")},
    ]
    ratings = compute_elo(rows)
    assert ratings["models/a.zip"] == pytest.approx(elo_update(ELO_BASE, ANCHOR_ELO, 1.0))


def test_all_anchor_opponents_are_pinned_at_1200():
    # The rule, random and early-anchor rows all move the rating by the same K step.
    rule = compute_elo([_synthetic_row("a.zip", "rl_vs_rule", 1.0)])["a.zip"]
    random = compute_elo([_synthetic_row("a.zip", "rl_vs_random", 1.0)])["a.zip"]
    anchor = compute_elo([_synthetic_row("a.zip", "rl_vs_rl", 1.0)])["a.zip"]
    assert rule == pytest.approx(random) == pytest.approx(anchor) == pytest.approx(1208.0)


def test_finalize_rows_stamps_the_same_final_elo_on_every_row():
    rows = [
        _synthetic_row("models/a.zip", "rl_vs_rule", 1.0, seed=0),
        _synthetic_row("models/a.zip", "rl_vs_random", 1.0, seed=1),
        _synthetic_row("models/b.zip", "rl_vs_rule", 0.0, seed=0),
    ]
    stamped = finalize_rows(rows)
    assert {r["checkpoint"] for r in stamped} == {"models/a.zip", "models/b.zip"}
    a_elos = {r["elo"] for r in stamped if r["checkpoint"] == "models/a.zip"}
    assert len(a_elos) == 1, "all rows of a checkpoint carry its final Elo"
    assert next(iter(a_elos)) > ELO_BASE
    assert all(r["elo"] < ELO_BASE for r in stamped if r["checkpoint"] == "models/b.zip")
    # The original rows are not mutated.
    assert "elo" not in rows[0]


def test_finalize_rows_can_override_the_initial_rating():
    rows = [_synthetic_row("a.zip", "rl_vs_rule", 0.5)]
    stamped = finalize_rows(rows, initial=1000.0)
    # A draw against a 1200 anchor from 1000 raises the rating.
    assert stamped[0]["elo"] == pytest.approx(elo_update(1000.0, ANCHOR_ELO, 0.5))


# --------------------------------------------------------------------------- #
# Pooling for the summary table
# --------------------------------------------------------------------------- #
def test_pooled_win_rate_pools_deals_and_reports_the_seed_sd():
    rows = [
        _synthetic_row("a.zip", "rl_vs_rule", rate, seed=s)
        for s, rate in enumerate((0.4, 0.5, 0.6))
    ]
    rate, sd, wins, deals = pooled_win_rate(rows, "rl_vs_rule")
    assert rate == pytest.approx(0.5)
    assert wins == 150 and deals == 300
    assert sd == pytest.approx(0.1)  # sample SD of (0.4, 0.5, 0.6)


def test_pooled_win_rate_single_seed_has_zero_sd():
    rows = [_synthetic_row("a.zip", "rl_vs_random", 0.7)]
    rate, sd, wins, deals = pooled_win_rate(rows, "rl_vs_random")
    assert rate == pytest.approx(0.7)
    assert sd == 0.0
    assert wins == 70 and deals == 100


def test_pooled_win_rate_missing_matchup_is_nan():
    rows = [_synthetic_row("a.zip", "rl_vs_rule", 0.7)]
    rate, sd, wins, deals = pooled_win_rate(rows, "rl_vs_random")
    assert rate != rate  # NaN
    assert wins == 0 and deals == 0


# --------------------------------------------------------------------------- #
# Discovery and the plan
# --------------------------------------------------------------------------- #
def test_parse_seeds_handles_text_and_sequences():
    assert parse_seeds("0,1,2") == [0, 1, 2]
    assert parse_seeds(" 0 , 1 , ") == [0, 1]
    assert parse_seeds([3, 4]) == [3, 4]
    with pytest.raises(ValueError):
        parse_seeds("")


def test_checkpoint_step_reads_the_step_or_returns_minus_one():
    assert checkpoint_step("models/checkpoints/ppo_1000000_steps.zip") == 1000000
    assert checkpoint_step("models/checkpoints/latest.zip") == -1


def test_resolve_checkpoints_globs_and_sorts_by_step(tmp_path, monkeypatch):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    for step in (2000000, 500000, 1000000):
        (ckpt / f"ppo_{step}_steps.zip").write_bytes(b"stub")
    monkeypatch.chdir(tmp_path)
    found = resolve_checkpoints(None, "checkpoints/ppo_*_steps.zip")
    assert [checkpoint_step(path) for path in found] == [500000, 1000000, 2000000]


def test_resolve_checkpoints_passes_explicit_paths_through():
    assert resolve_checkpoints(["a.zip", "b.zip"], "ignored/*") == ["a.zip", "b.zip"]


def test_build_plan_is_every_checkpoint_seed_matchup_combination():
    plan = build_plan(["a.zip", "b.zip"], [0, 1])
    assert len(plan) == 2 * 2 * len(LADDER_MATCHUPS)
    assert plan[0] == {"checkpoint": "a.zip", "seed": 0, "matchup": "rl_vs_rule", "opponent": "rule"}
    assert [entry["matchup"] for entry in plan[:3]] == [m[0] for m in LADDER_MATCHUPS]
    # The early-anchor leg is the only one that names a second model.
    assert {e["opponent"] for e in plan} == {"rule", "random", "anchor_early"}


# --------------------------------------------------------------------------- #
# Table and dry run
# --------------------------------------------------------------------------- #
def test_format_ladder_table_shows_both_pooled_rates_sd_and_elo():
    rows = finalize_rows(
        [_synthetic_row("models/checkpoints/ppo_1000000_steps.zip", m, 0.6, seed=s)
         for m in ("rl_vs_rule", "rl_vs_random", "rl_vs_rl") for s in (0, 1)]
    )
    table = format_ladder_table(rows)
    assert "ppo_1000000_steps" in table
    assert "vs rule" in table and "vs rnd" in table and "elo" in table
    # Two rows per matchup -> a non-zero between-seed SD is shown.
    line = [ln for ln in table.splitlines() if "ppo_1000000_steps" in ln][0]
    assert "0.600" in line


def test_dry_run_prints_the_plan_and_never_evaluates(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def _boom(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("--dry-run must not touch the evaluation path")

    monkeypatch.setattr(ladder, "build_parser", _boom)
    rc = main(["--dry-run", "--run-name", "v3", "--seeds", "0,1", "--checkpoints", "a.zip", "b.zip"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 checkpoint(s) x 2 seed(s) x 3 matchup(s) = 12 row(s)" in out
    assert "ladder_v3.csv" in out
    assert "a.zip | seed 0 | rl_vs_rule" in out


def test_dry_run_reports_missing_checkpoints_but_still_plans(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rc = main(["--dry-run", "--checkpoints", "definitely-missing.zip"])
    assert rc == 0
    assert "[missing]" in capsys.readouterr().out


def test_main_errors_when_no_checkpoint_matches(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main(["--dry-run", "--checkpoint-glob", "empty/ppo_*_steps.zip"])


# --------------------------------------------------------------------------- #
# ``test.py`` import hygiene (stdlib ``test`` package shadowing)
# --------------------------------------------------------------------------- #
def test_the_project_test_script_is_imported_not_the_stdlib_package():
    """``import test`` must resolve to the root ``test.py``, never ``Lib/test``."""
    import test as test_module

    assert pathlib.Path(test_module.__file__).resolve() == ladder.REPO_ROOT / "test.py"
    assert str(ladder.REPO_ROOT) in sys.path


def test_script_mode_dry_run_imports_the_project_test_script():
    """Running the file directly (``python tools/eval_ladder.py``) must not import
    the standard library's ``test`` package, which otherwise shadows the root
    ``test.py`` and makes the ladder ``ImportError`` before it does anything."""
    proc = subprocess.run(
        [sys.executable, str(ladder.REPO_ROOT / "tools" / "eval_ladder.py"),
         "--dry-run", "--checkpoints", "a.zip"],
        cwd=ladder.REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Planned evaluation: 1 checkpoint(s) x 3 seed(s) x 3 matchup(s) = 9 row(s)" in proc.stdout
