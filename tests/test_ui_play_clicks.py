"""The human's click flow in the interactive mode.

Regression test for the reported defect: clicking a tile that fits both open ends
selected it, but the second click (on the end) was swallowed -- the click handler
required the click to land on a *hand* tile before it ever asked which end was
hit, so the end choice was unreachable and the tile could never be played.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402

from domino.engine import GameState  # noqa: E402
from domino.tiles import tile_id  # noqa: E402
from domino.ui.play_loop import PlayConfig, _human_click, run_play  # noqa: E402
from domino.ui.render import Renderer  # noqa: E402

#: The ambiguous tile both tests play, and the two ends it could attach to.
BRIDGE = tile_id(4, 1)


def _both_ends_state() -> GameState:
    """A two-tile line with open ends 1 and 4; seat 2 holds ``[4|1]`` to play."""
    hands = [[tile_id(0, 4)], [tile_id(0, 1)], [BRIDGE], []]
    state = GameState(hands=hands, rng=np.random.default_rng(0), turn=0, leader=0)
    state.play(0, tile_id(0, 4))               # opener: ends [0, 4]
    state.play(1, tile_id(0, 1), 0)            # 0 attaches to the left end -> [1, 4]
    assert state.ends == [1, 4] and state.turn == 2
    return state


def _draw(renderer: Renderer, state: GameState, *, pending: int | None = None) -> None:
    placements = [end for (tile, end) in state.legal_moves(2) if tile == pending] if pending else []
    renderer.draw(
        state,
        acting_seat=2,
        human_seat=2,
        legal=state.legal_actions(2),
        selected=pending,
        ambiguous_ends=placements,
        last_actions=[],
    )


def _pick_tile(renderer: Renderer, state: GameState) -> int | None:
    """Click ``[4|1]`` in the hand; returns the resulting pending tile."""
    cfg = PlayConfig(human_seat=2, partner="rule", opponents="rule")
    _draw(renderer, state)
    pending, banner = _human_click(state, cfg, renderer, renderer.hand_rects[BRIDGE].center, None)
    assert "both ends" in banner, banner
    return pending


def test_a_tile_fitting_both_ends_waits_for_the_end_click():
    state = _both_ends_state()
    renderer = Renderer(size=(1280, 720))
    try:
        assert _pick_tile(renderer, state) == BRIDGE
        assert BRIDGE not in state.table, "selecting a tile must not play it"
        assert BRIDGE in state.hands[2]
    finally:
        renderer.close()


def test_clicking_the_framed_left_end_plays_on_the_left():
    state = _both_ends_state()
    cfg = PlayConfig(human_seat=2, partner="rule", opponents="rule")
    renderer = Renderer(size=(1280, 720))
    try:
        pending = _pick_tile(renderer, state)
        _draw(renderer, state, pending=pending)
        assert set(renderer.end_rects) == {0, 1}
        pending, banner = _human_click(state, cfg, renderer, renderer.end_rects[0].center, pending)
        assert pending is None and banner == "played"
        assert state.table[-1] == BRIDGE and state.table_ends[-1] == 0
        assert state.ends == [4, 4], "attaching the 1 to the left end exposes the 4"
    finally:
        renderer.close()


def test_clicking_the_framed_right_end_plays_on_the_right():
    state = _both_ends_state()
    cfg = PlayConfig(human_seat=2, partner="rule", opponents="rule")
    renderer = Renderer(size=(1280, 720))
    try:
        pending = _pick_tile(renderer, state)
        _draw(renderer, state, pending=pending)
        pending, banner = _human_click(state, cfg, renderer, renderer.end_rects[1].center, pending)
        assert pending is None and banner == "played"
        assert state.table[-1] == BRIDGE and state.table_ends[-1] == 1
        assert state.ends == [1, 1], "attaching the 4 to the right end exposes the 1"
    finally:
        renderer.close()


def test_the_two_ends_really_lead_to_different_positions():
    """"Where do I put it" must be a real decision, not a cosmetic one."""
    results = []
    for end in (0, 1):
        state = _both_ends_state()
        cfg = PlayConfig(human_seat=2, partner="rule", opponents="rule")
        renderer = Renderer(size=(1280, 720))
        try:
            pending = _pick_tile(renderer, state)
            _draw(renderer, state, pending=pending)
            _, banner = _human_click(state, cfg, renderer, renderer.end_rects[end].center, pending)
            assert banner == "played"
            results.append(list(state.ends))
        finally:
            renderer.close()
    assert results == [[4, 4], [1, 1]]


def test_clicking_the_table_keeps_the_selection_and_the_tile():
    state = _both_ends_state()
    cfg = PlayConfig(human_seat=2, partner="rule", opponents="rule")
    renderer = Renderer(size=(1280, 720))
    try:
        pending = _pick_tile(renderer, state)
        _draw(renderer, state, pending=pending)
        # the bottom-right corner of the table: no hand tile, no end frame
        pending, banner = _human_click(state, cfg, renderer, (1250, 690), pending)
        assert pending == BRIDGE and banner == ""
        assert BRIDGE in state.hands[2] and BRIDGE not in state.table
    finally:
        renderer.close()


def test_clicking_the_selected_tile_again_clears_the_selection():
    state = _both_ends_state()
    cfg = PlayConfig(human_seat=2, partner="rule", opponents="rule")
    renderer = Renderer(size=(1280, 720))
    try:
        pending = _pick_tile(renderer, state)
        _draw(renderer, state, pending=pending)
        pending, banner = _human_click(
            state, cfg, renderer, renderer.hand_rects[BRIDGE].center, pending
        )
        assert pending is None and banner == "selection cleared"
        assert BRIDGE in state.hands[2]
    finally:
        renderer.close()


def test_an_end_frame_click_without_a_pending_tile_plays_nothing():
    state = _both_ends_state()
    cfg = PlayConfig(human_seat=2, partner="rule", opponents="rule")
    renderer = Renderer(size=(1280, 720))
    try:
        _draw(renderer, state)
        pending, banner = _human_click(state, cfg, renderer, renderer.end_rects[0].center, None)
        assert pending is None and banner == ""
        assert state.table == [tile_id(0, 4), tile_id(0, 1)]
        assert BRIDGE in state.hands[2]
    finally:
        renderer.close()


def test_a_one_ended_tile_is_played_by_a_single_click():
    """The common case must not need two clicks."""
    hands = [[tile_id(0, 4)], [tile_id(0, 1)], [tile_id(4, 5)], []]
    state = GameState(hands=hands, rng=np.random.default_rng(0), turn=0, leader=0)
    state.play(0, tile_id(0, 4))
    state.play(1, tile_id(0, 1), 0)              # ends [1, 4]
    cfg = PlayConfig(human_seat=2, partner="rule", opponents="rule")
    renderer = Renderer(size=(1280, 720))
    try:
        _draw(renderer, state)
        pending, banner = _human_click(
            state, cfg, renderer, renderer.hand_rects[tile_id(4, 5)].center, None
        )
        assert pending is None and banner == "played"
        assert state.table[-1] == tile_id(4, 5)
        assert state.ends == [1, 5]
    finally:
        renderer.close()


# --------------------------------------------------------------------------- #
# Pacing: the AI waits between its moves
# --------------------------------------------------------------------------- #
def test_the_ai_waits_between_its_moves():
    """With a long delay only the first move happens, then the AI holds."""
    cfg = PlayConfig(partner="rule", opponents="rule", autoplay=True, fps=0, ai_delay=30.0)
    run_play(cfg, max_frames=60)
    assert len(cfg.last_actions) <= 1, cfg.last_actions


def test_with_no_delay_the_ai_plays_every_frame():
    """ai_delay=0 keeps the old behaviour: one move per frame, for scripts."""
    cfg = PlayConfig(partner="rule", opponents="rule", autoplay=True, fps=0, ai_delay=0.0)
    run_play(cfg, max_frames=60)
    assert len(cfg.last_actions) > 5, cfg.last_actions


def test_the_delay_is_configurable_from_the_cli():
    import play

    seen = {}
    original = play.run_play

    def fake(cfg):
        seen["cfg"] = cfg
        return 0

    play.run_play = fake
    try:
        play.main(["--ai-delay", "1.25"])
    finally:
        play.run_play = original
    assert seen["cfg"].ai_delay == 1.25


# --------------------------------------------------------------------------- #
# A stuck human seat must not freeze the deal
# --------------------------------------------------------------------------- #
def test_your_seat_passes_itself_when_it_is_stuck():
    """Reported: "the game stops when I can't play, it should skip me".

    Seed 31 makes seat 1 (the human's seat) unplayable on its first turn, so the
    loop has to take the forced pass itself -- waiting for a click that can never
    come left the deal frozen forever.
    """
    cfg = PlayConfig(human_seat=1, partner="rule", opponents="rule", seed=31,
                     fps=0, ai_delay=0.0)
    run_play(cfg, max_frames=400)
    actions = cfg.last_actions
    forced = [index for index, line in enumerate(actions) if "you: pass" in line]
    assert forced, f"your seat never took its forced pass: {actions}"
    # the deal carried on: the AI kept moving after the pass instead of stalling
    assert len(actions) > forced[0] + 2, actions


def test_your_seat_does_not_pass_when_it_has_a_legal_tile():
    """A soft check that the auto-pass cannot fire for a playable seat."""
    cfg = PlayConfig(human_seat=1, partner="rule", opponents="rule", seed=0,
                     fps=0, ai_delay=0.0)
    run_play(cfg, max_frames=200)
    assert cfg.last_actions, "the loop should have played something"
    assert not [line for line in cfg.last_actions if "you: pass" in line], cfg.last_actions


def test_a_stuck_seat_in_autoplay_is_passed_as_before():
    """With autoplay the human seat is an agent, so the AI path takes the pass."""
    cfg = PlayConfig(human_seat=1, partner="rule", opponents="rule", seed=31,
                     autoplay=True, fps=0, ai_delay=0.0)
    run_play(cfg, max_frames=400)
    assert not [line for line in cfg.last_actions if "you: pass" in line]
    assert any("pass (stuck)" in line for line in cfg.last_actions), cfg.last_actions


# --------------------------------------------------------------------------- #
# The end-of-hand report
# --------------------------------------------------------------------------- #
def _emptied_state():
    """Seat 2 (the human's seat) goes out on its only tile."""
    hands = [[tile_id(0, 4), tile_id(5, 5)], [tile_id(0, 1), tile_id(6, 6)],
             [tile_id(4, 1)], [tile_id(2, 2)]]
    state = GameState(hands=hands, rng=np.random.default_rng(0), turn=0, leader=0)
    state.play(0, tile_id(0, 4))              # ends [0, 4]
    state.play(1, tile_id(0, 1), 0)           # ends [1, 4]
    state.play(2, tile_id(4, 1), 1)           # seat 2 is out: ends [1, 1]
    assert state.is_terminal() and state.hands[2] == []
    return state


def test_the_report_says_who_took_the_points_and_why():
    from domino.engine import deal_outcome
    from domino.ui.play_loop import deal_summary

    state = _emptied_state()
    outcome = deal_outcome(state)
    scores = [0, 0]
    scores[outcome.winner_team] += outcome.points
    title, lines, match_over = deal_summary(
        state, outcome, human_seat=2, labels={0: "rl-4.0M", 1: "b", 2: "you", 3: "d"},
        scores=scores, match_to=100,
    )
    assert outcome.points == 16, "the two opponents' pips"
    assert title == "hand 1 — your team takes 16 points"
    assert any("went out first" in line for line in lines)
    assert any(line.startswith("your team left") for line in lines)
    assert any(line.startswith("the AI team left") for line in lines)
    assert any("match    your team 16 : 0 the AI team" in line for line in lines)
    assert not match_over and not any("MATCH OVER" in line for line in lines)


def test_the_report_names_a_blocked_hand_and_the_team_that_lost():
    from domino.engine import DealOutcome
    from domino.ui.play_loop import deal_summary

    state = _emptied_state()
    blocked = DealOutcome(winner_team=0, points=9, blocked=True, void=False,
                          reason="blocked", winning_seat=2)
    title, lines, match_over = deal_summary(
        state, blocked, human_seat=1, labels={0: "a", 1: "you", 2: "c", 3: "d"},
        scores=[9, 0], match_to=100,
    )
    assert title == "hand 1 — the AI team takes 9 points"
    assert any("blocked" in line and "fewest pips" in line for line in lines)
    assert not match_over


def test_the_report_announces_the_end_of_the_match():
    from domino.engine import DealOutcome
    from domino.ui.play_loop import deal_summary

    state = _emptied_state()
    outcome = DealOutcome(winner_team=0, points=10, blocked=False, void=False,
                          reason="emptied", winning_seat=2)
    title, lines, match_over = deal_summary(
        state, outcome, human_seat=2, labels={0: "a", 1: "b", 2: "you", 3: "d"},
        scores=[100, 84], match_to=100,
    )
    assert match_over
    assert any("MATCH OVER" in line and "your team wins 100 : 84" in line for line in lines)


def test_the_report_explains_a_void_hand():
    from domino.engine import DealOutcome
    from domino.ui.play_loop import deal_summary

    state = _emptied_state()
    void = DealOutcome(None, 0, True, True, "void_tie", None)
    title, lines, match_over = deal_summary(
        state, void, human_seat=1, labels={0: "a", 1: "you", 2: "c", 3: "d"},
        scores=[20, 20], match_to=100,
    )
    assert title == "hand 1 — void, no points"
    assert any("different teams" in line for line in lines)
    assert not match_over


def test_a_finished_hand_is_reported_and_the_next_one_waits_for_you():
    """Reported: "at the end of the hand it shows me what happened and who won"."""
    def play(frames: int):
        cfg = PlayConfig(human_seat=1, partner="rule", opponents="rule", autoplay=True,
                         fps=0, ai_delay=0.0, seed=0)
        run_play(cfg, max_frames=frames)
        return cfg

    short = play(1500)
    assert short.last_summary, "the finished hand must be reported"
    assert "hand 1" in short.last_summary[0]
    assert any("takes" in line for line in short.last_summary)
    played = len(short.last_actions)
    assert played > 5, short.last_actions

    # three times the frames and not one extra move: nothing is dealt behind it
    longer = play(4500)
    assert len(longer.last_actions) == played, "the next hand must wait for the player"
    assert longer.last_summary == short.last_summary


# --------------------------------------------------------------------------- #
# The replay: every position of the hand is kept and can be walked through
# --------------------------------------------------------------------------- #
def test_the_whole_hand_is_recorded_so_it_can_be_replayed():
    """One position per move, plus the deal: the replay has something to show."""
    cfg = PlayConfig(human_seat=1, partner="rule", opponents="rule", autoplay=True,
                     fps=0, ai_delay=0.0, seed=0)
    run_play(cfg, max_frames=1500)
    assert cfg.last_summary, "the hand should have finished"
    assert cfg.history_len == len(cfg.last_actions) + 1, (
        f"{cfg.history_len} positions kept for {len(cfg.last_actions)} moves"
    )


def test_replay_step_stays_inside_the_hand():
    from domino.ui.play_loop import replay_step

    assert replay_step(0, 1, 10) == 1
    assert replay_step(9, 1, 10) == 9, "the last position is the end"
    assert replay_step(0, -1, 10) == 0, "the first position is the deal"
    assert replay_step(4, -2, 10) == 2
    assert replay_step(0, 5, 1) == 0, "a one-position hand cannot move"


def test_pressing_r_starts_the_replay_at_the_deal():
    """R (or the report's button) walks the hand from its first position."""
    import pygame

    pygame.init()
    pygame.display.set_mode((1, 1))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    cfg = PlayConfig(human_seat=1, partner="rule", opponents="rule", autoplay=True,
                     fps=0, ai_delay=5.0, seed=0)
    run_play(cfg, max_frames=3)
    assert cfg.replay_index == 0, "the replay starts at the dealt position"
    # the same frame's AI move is recorded too, so the index must be inside the hand
    assert 0 <= cfg.replay_index < cfg.history_len
