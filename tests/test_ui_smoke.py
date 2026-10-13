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
import pygame  # noqa: E402
import pytest  # noqa: E402

from domino.agents import RandomAgent, RuleAgent  # noqa: E402
from domino.engine import new_deal  # noqa: E402
from domino.tiles import SIX_SIX_ID, tile_id, tile_values  # noqa: E402
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


def test_the_panel_shows_how_many_tiles_each_seat_has_left():
    """Only the count per seat -- never which tiles a seat is holding."""
    from domino.ui.render import seat_line

    state = _mid_deal_state()          # whoever led has already shed one tile
    for seat in range(4):
        line = seat_line(state, seat, label="rule")
        assert f"{state.hand_size(seat)} tiles left" in line
        assert "[" not in line, "the panel must not name a seat's tiles"
    assert seat_line(state, 0, label="rl-4.0M", acting=True) == "-> seat 0  6 tiles left  rl-4.0M"
    assert seat_line(state, 1, human=True) == "   seat 1  7 tiles left  you"  # aligns with '->'


def test_ambiguous_tile_draws_the_end_frames_it_makes_clickable():
    """A tile fitting both ends must show the two targets the click tests."""
    from domino.ui.render import ACTING

    state = _mid_deal_state()          # [6|6] on the table: ends [6, 6]
    renderer = Renderer(size=(1280, 720))
    try:
        renderer.draw(state, acting_seat=1, human_seat=1, legal=[], selected=tile_id(6, 3),
                      ambiguous_ends=[0, 1], banner="fits both ends")
        assert set(renderer.end_rects) == {0, 1}
        left_rect, right_rect = renderer.end_rects[0], renderer.end_rects[1]
        # the frames are painted where they are hit, and they sit on the line
        assert renderer.screen.get_at((left_rect.x + 1, left_rect.centery))[:3] == ACTING
        assert renderer.screen.get_at((right_rect.right - 2, right_rect.centery))[:3] == ACTING
        assert renderer.end_at(left_rect.center, state, state.ends) == 0
        assert renderer.end_at(right_rect.center, state, state.ends) == 1
        # nowhere else, and no end choice at all on a single-ended table
        assert renderer.end_at((640, 560), state, state.ends) is None
        assert renderer.end_at(left_rect.center, state, [6]) is None
    finally:
        renderer.close()


def test_board_draws_the_line_on_both_sides_of_the_opener():
    """The reported bug, end-to-end on the pixels: nothing piles onto one side."""
    from domino.ui import render as render_module

    state = _scripted_state(seed=0)
    assert len(state.table) >= 4, "the scripted deal should put a few tiles down"
    renderer = Renderer(size=(1280, 720))
    try:
        renderer.draw(state, acting_seat=1, human_seat=1, legal=[], last_actions=[])
        painted = [
            x for x in range(0, 1280 - 330) for y in range(190, 520, 7)
            if renderer.screen.get_at((x, y))[:3] == render_module.TILE
        ]
        assert painted, "the board must have drawn tiles"
        opener = _opener_rect(state, renderer)
        assert min(painted) < opener.left, "no tile drawn left of the opener"
        assert max(painted) > opener.right, "no tile drawn right of the opener"
    finally:
        renderer.close()


def test_the_renderer_draws_a_reversed_tile_reversed():
    """The line's orientation must reach the screen, not just the layout."""
    from domino.ui.board import layout_line
    from domino.ui.render import table_geometry

    state = _scripted_state(seed=0)
    x, y, width, _height = table_geometry((1280, 720))["line"]
    layout = layout_line(state, (1280, 720), top=y, left=x, right=x + width)
    # skip the two extreme tiles: the open-end frames are painted over them
    reversed_tiles = [
        placed for index, placed in enumerate(layout.tiles)
        if placed.flip and 0 < index < len(layout.tiles) - 1
    ]
    assert reversed_tiles, "the scripted deal should need at least one tile turned round"
    renderer = Renderer(size=(1280, 720))
    try:
        renderer.draw(state, acting_seat=1, human_seat=1, legal=[], last_actions=[])
        for placed in reversed_tiles[:2]:
            drawn = renderer.screen.subsurface(placed.rect).copy()
            straight = _render_tile_offline(placed.tile, flip=False)
            turned = _render_tile_offline(placed.tile, flip=True)
            assert _pixels(drawn) == _pixels(turned), (
                f"tile {tile_values(placed.tile)} must be drawn turned round"
            )
            assert _pixels(drawn) != _pixels(straight)
    finally:
        renderer.close()


def _render_tile_offline(tile: int, *, flip: bool):
    """The same tile drawn alone on the board background, for pixel comparison."""
    import pygame

    from domino.ui import render as render_module
    from domino.ui.board import tile_height, tile_width

    surface = pygame.Surface((tile_width(tile), tile_height(tile)))
    surface.fill(render_module.BG)
    draw_tile(surface, tile, (0, 0), crosswise=tile_values(tile)[0] == tile_values(tile)[1],
              flipped=flip)
    return surface


def _pixels(surface) -> list[tuple[int, int, int]]:
    return [
        surface.get_at((x, y))[:3]
        for x in range(surface.get_width())
        for y in range(surface.get_height())
    ]


def _opener_rect(state, renderer):
    """The opener's rect in the layout the renderer actually draws."""
    from domino.ui.board import line_tiles, layout_line
    from domino.ui.render import table_geometry

    x, y, width, _height = table_geometry(renderer.size)["line"]
    return next(
        p.rect for p in layout_line(state, renderer.size, top=y, left=x, right=x + width).tiles
        if p.tile == line_tiles(state)[0][line_tiles(state)[1]][0]
    )


def _scripted_state(seed: int = 0):
    """A deal with a handful of tiles played on both ends."""
    state = new_deal(deal_no=1, rng=np.random.default_rng(seed), lead_mode="six_six")
    state.play(state.turn, SIX_SIX_ID)
    for wanted in (0, 1, 1, 0):
        seat = state.turn
        options = state.legal_moves(seat)
        if not options:
            state.pass_turn(seat)
            continue
        tile, end = next(((t, e) for t, e in options if e == wanted), options[0])
        state.play(seat, tile, end)
    return state


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


# --------------------------------------------------------------------------- #
# The table view: your hand face up, every other seat face down
# --------------------------------------------------------------------------- #
def _region_colours(renderer, rect) -> set[tuple[int, int, int]]:
    return {
        renderer.screen.get_at((x, y))[:3]
        for x in range(rect.x + 2, rect.right - 2, 3)
        for y in range(rect.y + 2, rect.bottom - 2, 3)
    }


def _table_draw(state, renderer, **kwargs) -> None:
    renderer.draw(state, acting_seat=1, human_seat=1, legal=state.legal_actions(1),
                  seat_labels={0: "rl-4.0M", 2: "rule", 3: "rl-latest"}, **kwargs)


def test_the_table_regions_fit_the_window_without_overlapping():
    from domino.ui.render import table_geometry

    for size in ((1280, 720), (1024, 640)):
        regions = table_geometry(size)
        width, height = size
        for name, (x, y, w, h) in regions.items():
            assert x >= 0 and y >= 0 and x + w <= width and y + h <= height, name
        bar, partner = regions["bar"], regions["partner"]
        line, hand = regions["line"], regions["hand"]
        left, right = regions["left"], regions["right"]
        assert partner[1] >= bar[1] + bar[3], "the partner sits under the status bar"
        assert line[1] >= partner[1] + partner[3], "the line band starts below the partner"
        assert hand[1] >= line[1] + line[3], "your hand is at the bottom"
        assert left[0] + left[2] <= line[0], "the left stack is outside the line band"
        assert right[0] >= line[0] + line[2], "the right stack is outside the line band"
        assert line[2] > width // 2, "the line band must be worth drawing in"


def test_the_table_shows_your_hand_face_up_and_the_others_face_down():
    from domino.ui.render import BACK, TILE, TILE_DIM, seat_sides

    state = _scripted_state(seed=0)
    renderer = Renderer(size=(1280, 720))
    try:
        _table_draw(state, renderer)
        sides = seat_sides(1)
        assert sides["partner"] == 3, "your partner sits opposite you"
        assert {sides["left"], sides["right"]} == {0, 2}, "the opponents sit left and right"

        assert len(renderer.hand_rects) == state.hand_size(1)
        playable = set(state.legal_actions(1))
        for tile, rect in renderer.hand_rects.items():
            colours = _region_colours(renderer, rect)
            assert BACK not in colours, "your tiles are face up"
            assert colours & {TILE, TILE_DIM}, "a face-up tile body must be drawn"
            if tile in playable:
                assert TILE in colours, "a playable tile is drawn bright"
            else:
                assert TILE_DIM in colours, "an unplayable tile is drawn dimmed"

        for side in ("partner", "left", "right"):
            seat = sides[side]
            rects = renderer.seat_rects[seat]
            assert len(rects) == state.hand_size(seat), "one face-down tile per tile held"
            for rect in rects:
                colours = _region_colours(renderer, rect)
                assert BACK in colours and TILE not in colours, "their tiles are face down"
        assert 1 not in renderer.seat_rects, "your seat is the hand, not a stack of backs"
    finally:
        renderer.close()


def test_the_face_down_count_follows_the_engine_hand_sizes():
    from domino.engine import GameState
    from domino.ui.render import seat_sides

    hands = [[tile_id(0, 4)], [tile_id(0, 1)], [tile_id(4, 1)], [tile_id(1, 1)]]
    state = GameState(hands=hands, rng=np.random.default_rng(0), turn=0, leader=0)
    state.play(0, tile_id(0, 4))              # seat 0 goes out: no tiles to show
    renderer = Renderer(size=(1280, 720))
    try:
        _table_draw(state, renderer)
        sides = seat_sides(1)
        assert len(renderer.seat_rects[sides["partner"]]) == 1        # seat 3 holds one
        assert renderer.seat_rects[0] == [], "a seat that went out shows no tiles"
        assert len(renderer.hand_rects) == 1                          # your own single tile
    finally:
        renderer.close()


def test_the_summary_is_drawn_over_the_table_and_does_not_crash():
    from domino.ui.render import ACTING, PANEL

    state = _mid_deal_state()
    renderer = Renderer(size=(1280, 720))
    try:
        _table_draw(state, renderer)
        renderer.draw_summary(["your team takes 12 points", "match    12 : 4"],
                              title="hand 3 — your team takes 12 points",
                              hint="press SPACE (or click) to deal the next hand")
        box = pygame.Rect(0, 0, int(1280 * 0.68), int(720 * 0.54))
        box.center = (640, 360)
        assert renderer.screen.get_at(box.center)[:3] in (PANEL, ACTING)
        assert renderer.screen.get_at((6, 6))[:3] != (24, 84, 52), "the table is dimmed behind it"
    finally:
        renderer.close()
