"""Phase-4 judge tests: regret bookkeeping, hindsight exactness, CSV, ranking of play."""
from __future__ import annotations

import numpy as np
import pytest

from domino.agents import RandomAgent, RuleAgent
from domino.pimc import SolverAgent
from domino.pimc.judge import (
    BLUNDER_THRESHOLD,
    judge_deals,
    judge_game,
    record_deal,
)


def agents_for(kind: str, seed: int) -> dict[int, object]:
    """Seat -> agent for the kinds used in these tests."""
    if kind == "random":
        return {seat: RandomAgent(seed=seed + seat) for seat in range(4)}
    if kind == "rule":
        return {seat: RuleAgent(seed=seed + seat) for seat in range(4)}
    if kind == "oracle":
        return {seat: SolverAgent(tt_bits=14) for seat in range(4)}
    raise ValueError(kind)


def test_record_deal_captures_decisions_and_isolates_them() -> None:
    """A trace holds one record per decision of the evaluated seats, on frozen clones."""
    rng = np.random.default_rng(1)
    trace = record_deal(agents_for("rule", 0), rng, eval_seats=(0, 2), deal_no=1)
    assert trace.decisions, "the rule bot must make decisions"
    assert all(record.seat in (0, 2) for record in trace.decisions)
    assert trace.outcome.void is False
    plies = [record.ply for record in trace.decisions]
    assert plies == sorted(plies) and plies[0] < plies[-1]
    # A recorded state is a snapshot: mutating the *final* deal must not touch it.
    first = trace.decisions[0]
    hands_before = [list(hand) for hand in first.state.hands]
    trace.final_state.hands[0] = [1, 2, 3]
    assert [list(hand) for hand in first.state.hands] == hands_before
    assert trace.final_state.is_terminal()


def test_judge_grades_a_decision_with_paired_worlds() -> None:
    """Every graded decision carries values per move, a regret, a SE and a flag."""
    rng = np.random.default_rng(2)
    trace = record_deal(agents_for("rule", 1), rng, eval_seats=(0, 2), deal_no=1)
    report = judge_game(trace, worlds=6, horizon=4, rng=rng, hindsight=True)
    graded = report.graded
    assert graded, "the rule bot makes graded decisions"
    for decision in graded:
        assert decision.worlds == 6
        assert decision.played_move in decision.values
        assert decision.best_move in decision.values
        assert decision.regret >= -1e-9
        assert decision.regret_true >= -1e-9
        assert decision.values[decision.best_move] >= decision.values[decision.played_move] - 1e-9
        assert np.isnan(decision.se) or decision.se >= 0.0
        assert decision.situation in ("lead", "follow")
        assert decision.phase in ("opening", "mid", "endgame")
    passes = [d for d in report.decisions if d.situation == "forced_pass"]
    for decision in passes:
        assert decision.regret == 0.0 and decision.played_move is None


def test_oracle_has_no_hindsight_regret() -> None:
    """Judged with the true hands, the exact solver's moves cannot be improved on."""
    rng = np.random.default_rng(3)
    trace = record_deal(agents_for("oracle", 2), rng, eval_seats=(0, 2), deal_no=1)
    report = judge_game(trace, worlds=3, horizon=None, rng=rng, hindsight=True)
    graded = report.graded
    assert graded
    for decision in graded:
        assert decision.regret_true == 0.0, (
            f"ply {decision.ply} seat {decision.seat}: hindsight regret "
            f"{decision.regret_true} (values {decision.values_true})"
        )


def test_random_play_is_seen_as_worse_than_the_rule_bot() -> None:
    """The judge must rank random play above the rule bot by regret."""
    rng = np.random.default_rng(4)
    deals, worlds, horizon = 3, 8, 5
    random_report = judge_deals(
        agents_for("random", 10), np.random.default_rng(11), deals=deals, worlds=worlds,
        horizon=horizon, label="random")
    rule_report = judge_deals(
        agents_for("rule", 10), np.random.default_rng(11), deals=deals, worlds=worlds,
        horizon=horizon, label="rule")
    assert random_report.graded and rule_report.graded
    assert random_report.mean_regret() > rule_report.mean_regret(), (
        f"random {random_report.mean_regret():.2f} vs rule {rule_report.mean_regret():.2f} "
        "— the judge failed its own sanity check"
    )
    assert random_report.blunder_rate() >= rule_report.blunder_rate() - 1e-9


def test_report_writes_a_csv_and_a_summary(tmp_path) -> None:
    """The CSV has one row per decision with the documented columns."""
    rng = np.random.default_rng(5)
    trace = record_deal(agents_for("rule", 3), rng, eval_seats=(0, 2), deal_no=1)
    report = judge_game(trace, worlds=4, horizon=4, rng=rng, hindsight=True)
    path = report.write_csv(tmp_path / "judge.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(report.decisions) + 1
    header = lines[0].split(",")
    for column in ("regret", "regret_true", "se", "dominance", "phase", "situation",
                   "played", "best", "worlds"):
        assert column in header
    text = report.summary()
    assert "mean regret" in text and "blunder rate" in text
    assert f">= {BLUNDER_THRESHOLD} pts" in text
    assert len(report.luck) == 1


def test_forced_pass_is_recorded_and_excluded_from_the_mean() -> None:
    """A stuck seat is recorded as a forced pass whose regret is not averaged in."""
    rng = np.random.default_rng(6)
    trace = None
    for seed in range(12):
        candidate = record_deal(agents_for("rule", 20 + seed), rng, eval_seats=(0, 2),
                                deal_no=1)
        if any(d.situation == "forced_pass" for d in
               judge_game(candidate, worlds=2, horizon=2, rng=rng, hindsight=False).decisions):
            trace = candidate
            break
    assert trace is not None, "no deal with a forced pass in 12 tries"
    report = judge_game(trace, worlds=2, horizon=2, rng=rng, hindsight=False)
    forced = [d for d in report.decisions if d.situation == "forced_pass"]
    assert forced
    assert all(d.regret == 0.0 for d in forced)
    graded = report.graded
    if graded:
        assert abs(report.mean_regret() - float(np.mean([d.regret for d in graded]))) < 1e-9


def test_judge_needs_an_explicit_generator() -> None:
    """No global randomness: a missing generator is an error, not a default."""
    rng = np.random.default_rng(7)
    trace = record_deal(agents_for("rule", 4), rng, eval_seats=(0,), deal_no=1)
    with pytest.raises(ValueError):
        judge_game(trace, worlds=2, horizon=2, rng=None)
    with pytest.raises(ValueError):
        judge_game(trace, worlds=2, horizon=2, rng=rng, leaf_eval="bogus")
