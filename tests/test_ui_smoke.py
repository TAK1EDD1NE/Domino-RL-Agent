"""Headless smoke tests for the Pygame modes.

The video driver is forced to SDL's ``dummy`` backend before pygame is imported,
so these tests run on a machine with no display (and in CI).
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from domino.agents import RandomAgent, RuleAgent  # noqa: E402
from domino.engine import new_deal  # noqa: E402
from domino.tiles import SIX_SIX_ID, tile_id  # noqa: E402
from domino.ui.observe_loop import ObserveConfig, action_probabilities, run_observe  # noqa: E402
from domino.ui.play_loop import (  # noqa: E402
    PlayConfig,
    _build_agents,
    agent_label,
    parse_seat_agent,
    run_play,
    seat_labels,
)
from domino.ui.render import Renderer, describe_hand, draw_tile, format_action  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mid_deal_state(seed: int = 0):
    state = new_deal(deal_no=1, rng=np.random.default_rng(seed), lead_mode="six_six")
    state.play(state.turn, SIX_SIX_ID)
    return state


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
def test_renderer_draws_a_mid_deal_state_and_maps_clicks():
    state = _mid_deal_state()
    renderer = Renderer(size=(1024, 640))
    try:
        renderer.draw(
            state,
            acting_seat=state.turn,
            human_seat=1,
            legal=state.legal_actions(1),
            banner="smoke",
            last_actions=["seat 0 (rule): [6|6]"],
        )
        assert len(renderer.hand_rects) == 7
        tile, rect = next(iter(renderer.hand_rects.items()))
        assert renderer.tile_at(rect.center) == tile
        assert renderer.tile_at((0, 0)) is None
        renderer.tick(fps=1)
    finally:
        renderer.close()


def test_doubles_are_drawn_crosswise_and_others_long():
    import pygame

    pygame.init()
    surface = pygame.Surface((400, 400))
    double = draw_tile(surface, tile_id(3, 3), (10, 10), crosswise=True)
    assert double.height > double.width
    normal = draw_tile(surface, tile_id(2, 5), (10, 200), crosswise=False)
    assert normal.width > normal.height


def test_action_text_helpers():
    assert format_action(tile_id(6, 6)) == "[6|6] (12 pips)"
    assert format_action(tile_id(3, 5), 1) == "[3|5] (8 pips) on end 1"
    assert describe_hand([]) == "(empty)"
    assert describe_hand([tile_id(6, 6)]) == "[6|6] (12 pips)"


def test_ambiguous_tile_draws_the_end_boxes_it_makes_clickable():
    """A tile fitting both ends must show the two targets the click tests."""
    from domino.ui.render import ACTING, end_marker_rects

    state = _mid_deal_state()
    renderer = Renderer(size=(1280, 720))
    try:
        renderer.draw(state, acting_seat=1, human_seat=1, legal=[], selected=tile_id(6, 3),
                      ambiguous_ends=[0, 1], banner="fits both ends")
        left, right = end_marker_rects(renderer.size)
        # the boxes are actually painted (highlighted border) where they are hit
        assert renderer.screen.get_at((left.x + 1, left.centery))[:3] == ACTING
        assert renderer.screen.get_at((right.x + 1, right.centery))[:3] == ACTING
        assert renderer.end_at(left.center, state, state.ends) == 0
        assert renderer.end_at(right.center, state, state.ends) == 1
        # nowhere else, and no end choice at all on a single-ended table
        assert renderer.end_at((600, 400), state, state.ends) is None
        assert renderer.end_at(left.center, state, [6]) is None
    finally:
        renderer.close()


# --------------------------------------------------------------------------- #
# Loops (headless)
# --------------------------------------------------------------------------- #
def test_observe_loop_runs_headless():
    cfg = ObserveConfig(agent="rule", deals=1, overlay=False, fps=30)
    assert run_observe(cfg, max_frames=40) == 0


def test_play_loop_runs_headless_in_autoplay():
    cfg = PlayConfig(partner="rule", opponents="random", autoplay=True, fps=30)
    assert run_play(cfg, max_frames=80) == 0


# --------------------------------------------------------------------------- #
# Per-seat agents (three different agents at the three AI seats)
# --------------------------------------------------------------------------- #
def test_parse_seat_agent_reads_kind_and_checkpoint():
    assert parse_seat_agent("rule") == ("rule", None)
    assert parse_seat_agent("RL") == ("rl", None)
    assert parse_seat_agent("rl:models/checkpoints/ppo_4001502_steps.zip") == (
        "rl", "models/checkpoints/ppo_4001502_steps.zip",
    )
    # a Windows path keeps all of its characters: only the first colon splits
    assert parse_seat_agent(r"rl:C:\Users\pc\models\latest.zip") == (
        "rl", r"C:\Users\pc\models\latest.zip",
    )
    with pytest.raises(ValueError, match="unknown agent kind"):
        parse_seat_agent("mcts")


def test_agent_label_names_the_training_stage():
    assert agent_label("rl", "models/checkpoints/ppo_4001502_steps.zip") == "rl-4.0M"
    assert agent_label("rl", "models/checkpoints/ppo_15002766_steps.zip") == "rl-15.0M"
    assert agent_label("rl", "models/checkpoints/latest.zip") == "rl-latest"
    assert agent_label("rl", None) == "rl"
    assert agent_label("rule") == "rule"
    assert agent_label("random") == "random"


def test_seat_agents_can_differ_and_defaults_are_untouched():
    """Each AI seat gets its own kind; unspecified seats keep partner/opponents."""
    cfg = PlayConfig(human_seat=1, partner="rule", opponents="random",
                     seat_agents={0: "rule", 2: "random"})
    agents = _build_agents(cfg)
    assert sorted(agents) == [0, 2, 3]
    assert isinstance(agents[3], RuleAgent)          # partner default
    assert agents[3].name == "rule"
    # the two opponents are distinguishable now: seat 0 was overridden, seat 2
    # kept the configured default, and neither shares an object with the other
    assert agents[0] is not agents[2]
    assert {agents[0].name, agents[2].name} == {"rule", "random"}
    labels = seat_labels(agents, cfg.human_seat)
    assert labels[1] == "you" and labels[3] == "rule"


def test_seat_agents_reject_the_human_seat_and_bad_specs():
    with pytest.raises(ValueError, match="human seat"):
        _build_agents(PlayConfig(human_seat=1, partner="rule", opponents="rule",
                                 seat_agents={1: "rule"}))
    with pytest.raises(ValueError, match="unknown agent kind"):
        _build_agents(PlayConfig(human_seat=1, partner="rule", opponents="rule",
                                 seat_agents={0: "nope"}))


def test_seat_agents_missing_checkpoint_fails_before_the_window_opens():
    with pytest.raises(FileNotFoundError, match="checkpoint for seat 0 not found"):
        _build_agents(PlayConfig(human_seat=1, partner="rule", opponents="rule",
                                 seat_agents={0: "rl:models/checkpoints/nope.zip"},
                                 model="models/checkpoints/nope.zip"))


def test_play_cli_parses_repeated_agent_flags(monkeypatch):
    """`--agent` reaches PlayConfig as {seat: spec}; the window is not opened."""
    import play

    seen: dict[str, object] = {}

    def fake_run_play(cfg):
        seen["cfg"] = cfg
        return 0

    monkeypatch.setattr(play, "run_play", fake_run_play)
    code = play.main(["--human-seat", "1", "--agent", "0=rule", "--agent",
                      "2=rl:models/checkpoints/ppo_4001502_steps.zip"])
    assert code == 0
    assert seen["cfg"].seat_agents == {0: "rule", 2: "rl:models/checkpoints/ppo_4001502_steps.zip"}
    with pytest.raises(SystemExit, match="your own seat"):
        play.main(["--human-seat", "1", "--agent", "1=rule"])
    with pytest.raises(SystemExit, match="SEAT=KIND"):
        play.main(["--human-seat", "1", "--agent", "rule"])


def test_action_probabilities_is_none_without_a_policy():
    state = _mid_deal_state()
    assert action_probabilities(RuleAgent(seed=0), state, 0) is None
    assert action_probabilities(RandomAgent(seed=0), state, 0) is None


# --------------------------------------------------------------------------- #
# CLIs
# --------------------------------------------------------------------------- #
def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["SDL_VIDEODRIVER"] = "dummy"
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, env=env,
    )


def test_play_and_observe_help_exit_zero():
    for script in ("play.py", "observe.py"):
        proc = _run(script, "--help")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "usage" in proc.stdout.lower()


# --------------------------------------------------------------------------- #
# The training path must stay pygame-free
# --------------------------------------------------------------------------- #
def test_training_path_never_imports_pygame():
    training_files = [
        "train.py",
        "test.py",
        "domino/engine.py",
        "domino/obs.py",
        "domino/masks.py",
        "domino/env.py",
        "domino/policy.py",
        "domino/callbacks.py",
        "domino/config.py",
    ]
    for relative in training_files:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        imports = [
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith(("import pygame", "from pygame"))
        ]
        assert not imports, f"{relative} must not import pygame (found {imports})"
