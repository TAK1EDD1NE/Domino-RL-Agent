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

from domino.agents import RandomAgent, RuleAgent  # noqa: E402
from domino.engine import new_deal  # noqa: E402
from domino.tiles import SIX_SIX_ID, tile_id  # noqa: E402
from domino.ui.observe_loop import ObserveConfig, action_probabilities, run_observe  # noqa: E402
from domino.ui.play_loop import PlayConfig, run_play  # noqa: E402
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


# --------------------------------------------------------------------------- #
# Loops (headless)
# --------------------------------------------------------------------------- #
def test_observe_loop_runs_headless():
    cfg = ObserveConfig(agent="rule", deals=1, overlay=False, fps=30)
    assert run_observe(cfg, max_frames=40) == 0


def test_play_loop_runs_headless_in_autoplay():
    cfg = PlayConfig(partner="rule", opponents="random", autoplay=True, fps=30)
    assert run_play(cfg, max_frames=80) == 0


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
