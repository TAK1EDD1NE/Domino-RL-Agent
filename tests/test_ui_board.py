"""Tests for the physical line layout (both sides, matching numbers, wrapping).

Regression tests for the reported bug: the board used to be drawn as the deal's
*play order* on a single side, with every tile's halves in their canonical order,
so all tiles piled up to the right of the opener and a tile was frequently drawn
with two different numbers touching.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402

from domino.engine import new_deal  # noqa: E402
from domino.tiles import SIX_SIX_ID, tile_values  # noqa: E402
from domino.ui.board import (  # noqa: E402
    PANEL_W,
    line_tiles,
    line_values,
    layout_line,
    tile_width,
)


def _scripted_line(seed: int = 0, moves: int = 10, sides: tuple[int, ...] = (0, 1, 0, 1, 1, 0, 1, 0, 1, 0)):
    """Play a deal, choosing the requested end whenever the rules allow it."""
    state = new_deal(deal_no=1, rng=np.random.default_rng(seed), lead_mode="six_six")
    state.play(state.turn, SIX_SIX_ID)
    used: list[int] = []
    for wanted in sides:
        if len(state.table) >= moves:
            break
        seat = state.turn
        options = state.legal_moves(seat)
        if not options:
            state.pass_turn(seat)
            continue
        tile, side = next(((t, s) for t, s in options if s == wanted), options[0])
        state.play(seat, tile, side)
        used.append(side)
    return state, used


# --------------------------------------------------------------------------- #
# Both ends of the line are used
# --------------------------------------------------------------------------- #
def test_the_line_grows_on_both_sides_of_the_opener():
    state, used = _scripted_line(seed=0, moves=8)
    assert set(used) == {0, 1}, "the script should exercise both ends of the line"
    ordered, anchor = line_tiles(state)
    assert ordered[anchor][0] == SIX_SIX_ID
    layout = layout_line(state, (1280, 720))
    opener = next(p for p in layout.tiles if p.tile == SIX_SIX_ID)
    left = [p for p in layout.tiles if p.rect.right <= opener.rect.left]
    right = [p for p in layout.tiles if p.rect.left >= opener.rect.right]
    assert left, "tiles played on the left end must be drawn left of the opener"
    assert right, "tiles played on the right end must be drawn right of the opener"
    assert len(left) + len(right) + 1 == len(layout.tiles)


def test_the_opener_stays_where_it_is_drawn_as_the_line_grows():
    """The reference tile must not drift: the third tile never moves the opener."""
    positions = []
    for moves in (2, 5, 9):
        state, _ = _scripted_line(seed=3, moves=moves)
        layout = layout_line(state, (1280, 720))
        opener = next(p for p in layout.tiles if p.tile == SIX_SIX_ID)
        positions.append(opener.rect.topleft)
    assert len(set(positions)) == 1


# --------------------------------------------------------------------------- #
# Matching numbers touch (the reported bug)
# --------------------------------------------------------------------------- #
def test_equal_numbers_touch_along_the_drawn_line():
    state, _ = _scripted_line(seed=2, moves=9)
    layout = layout_line(state, (1280, 720))
    values = line_values(layout)
    assert len(values) == 2 * len(layout.tiles)
    checked = 0
    for index in range(len(layout.tiles) - 1):
        if layout.tiles[index].row != layout.tiles[index + 1].row:
            continue                      # a row break is a visual discontinuity
        assert values[2 * index + 1] == values[2 * index + 2], (
            f"tiles {index} and {index + 1} touch with "
            f"{values[2 * index + 1]}/{values[2 * index + 2]}"
        )
        checked += 1
    assert checked >= len(layout.tiles) - 2


def test_both_halves_of_every_tile_are_the_tile_it_represents():
    state, _ = _scripted_line(seed=5, moves=7)
    layout = layout_line(state, (1280, 720))
    values = line_values(layout)
    for index, placed in enumerate(layout.tiles):
        a, b = tile_values(placed.tile)
        assert values[2 * index] in (a, b)
        assert {values[2 * index], values[2 * index + 1]} == {a, b}


def test_the_open_ends_drawn_are_the_ends_the_engine_reports():
    """Whatever value is framed must be the value the rules expose there."""
    state, _ = _scripted_line(seed=7, moves=10)
    ordered, anchor = line_tiles(state)
    left_end = ordered[0]
    right_end = ordered[-1]
    # the framed halves are the first tile's left half and the last tile's right
    a, b = tile_values(left_end[0])
    left_value = b if left_end[1] else a
    x, y = tile_values(right_end[0])
    right_value = x if right_end[1] else y
    assert [left_value, right_value] == list(state.ends)


# --------------------------------------------------------------------------- #
# Wrapping
# --------------------------------------------------------------------------- #
def _long_line(tiles_target: int = 20):
    """A hand-built line of ``tiles_target`` tiles, alternating both ends.

    Built directly rather than dealt, because a real deal often blocks well
    before 20 tiles and the wrap behaviour is what is under test here. Tiles are
    fed to whichever seat is on turn, and every move is validated by the engine.
    """
    from domino.engine import GameState
    from domino.tiles import ALL_TILE_IDS

    used = {SIX_SIX_ID}
    hands: list[list[int]] = [[SIX_SIX_ID], [], [], []]
    state = GameState(hands=hands, rng=np.random.default_rng(0), turn=0, leader=0)
    state.play(0, SIX_SIX_ID)
    while len(state.table) < tiles_target:
        seat = state.turn
        end = len(state.table) % 2                  # alternate left / right
        target = state.ends[end]
        candidates = [
            tid for tid in ALL_TILE_IDS
            if tid not in used and target in tile_values(tid)
        ]
        if not candidates:
            break

        def room(tid: int) -> int:
            """How many unused tiles could follow this one (one-step lookahead)."""
            a, b = tile_values(tid)
            exposed = b if a == target else a
            return sum(
                1 for t in ALL_TILE_IDS
                if t not in used and t != tid and exposed in tile_values(t)
            )

        pick = max(candidates, key=room)
        used.add(pick)
        hands[seat].append(pick)
        state.play(seat, pick, end)                 # falls back to the only legal end
    return state


def test_a_long_line_wraps_into_rows_without_covering_the_panel():
    state = _long_line(20)
    assert len(state.table) >= 18, "the synthetic line should be long"
    size = (1280, 720)
    layout = layout_line(state, size)
    board_right = size[0] - PANEL_W
    assert layout.rows >= 2, "18+ tiles cannot fit in one row of this board"
    for placed in layout.tiles:
        assert placed.rect.left >= 0
        assert placed.rect.right <= board_right, "no tile may run under the info panel"
    tops = {placed.row: placed.rect.top for placed in layout.tiles}
    assert sorted(tops) == list(range(layout.rows)), "rows are 0..n-1"
    assert [tops[row] for row in sorted(tops)] == sorted(tops.values()), "rows stack downwards"


def test_matching_numbers_still_touch_inside_each_wrapped_row():
    state = _long_line(20)
    layout = layout_line(state, (1280, 720))
    values = line_values(layout)
    for index in range(len(layout.tiles) - 1):
        if layout.tiles[index].row != layout.tiles[index + 1].row:
            continue
        assert values[2 * index + 1] == values[2 * index + 2]


def test_every_tile_keeps_its_drawn_size():
    state, _ = _scripted_line(seed=13, moves=12)
    layout = layout_line(state, (1280, 720))
    for placed in layout.tiles:
        a, b = tile_values(placed.tile)
        if a == b:
            assert placed.rect.width == 42 and placed.rect.height == 84
        else:
            assert placed.rect.width == tile_width(placed.tile) == 84
            assert placed.rect.height == 42


def test_an_empty_table_has_no_layout():
    from domino.engine import GameState

    state = GameState(hands=[[SIX_SIX_ID]], rng=np.random.default_rng(0))
    layout = layout_line(state, (1280, 720))
    assert layout.tiles == () and layout.ends == {} and layout.rows == 0
