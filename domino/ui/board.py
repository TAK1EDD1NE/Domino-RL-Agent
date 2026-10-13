"""Physical layout of the domino line: sides, orientation, wrapping and ends.

The engine keeps a deal's tiles in *play order* plus the values of the two open
ends, which is not enough to draw a table: a tile played on the left end and a
tile played on the right end both land in ``state.table`` in the order they were
played, and a tile's two halves must be oriented so its matching number touches
the number it was played against. This module rebuilds the physical line from
``(state.table, state.table_ends)`` -- screen-left to screen-right, every tile
oriented, wrapped into rows -- and reports where the two open ends ended up, so
the UI can draw them and the player can click them.

The invariant that makes a drawn line read like a real table is asserted in the
tests: for every adjacent pair of drawn tiles, the **right half of the first
equals the left half of the second**, i.e. equal numbers always touch.
"""
from __future__ import annotations

from dataclasses import dataclass

import pygame

from ..engine import GameState
from ..tiles import tile_values

#: Tile geometry (pixels).
TILE_LONG: int = 84
TILE_SHORT: int = 42
TILE_GAP: int = 6
#: A row is as tall as a crosswise double, so the tall tiles of a row fit it.
ROW_H: int = TILE_LONG
ROW_PITCH: int = ROW_H + 16
#: Where the first row of the line starts, and the margin the rows keep from the
#: board's left/right edges.
BOARD_TOP: int = 196
BOARD_MARGIN: int = 40
#: Width of the right-hand info panel; the board stops before it.
PANEL_W: int = 330


def tile_width(tile: int) -> int:
    """Drawn width of a tile: a double is crosswise, hence half as wide."""
    a, b = tile_values(tile)
    return TILE_SHORT if a == b else TILE_LONG


def tile_height(tile: int) -> int:
    """Drawn height of a tile: a double is crosswise, hence twice as tall."""
    a, b = tile_values(tile)
    return TILE_LONG if a == b else TILE_SHORT


@dataclass(frozen=True)
class PlacedTile:
    """One tile of the line: which tile, how it is oriented, where it sits."""

    tile: int
    flip: bool
    row: int
    rect: pygame.Rect


@dataclass(frozen=True)
class LineLayout:
    """A laid-out line: the tiles left-to-right plus the two clickable ends."""

    tiles: tuple[PlacedTile, ...]
    ends: dict[int, pygame.Rect]

    @property
    def rows(self) -> int:
        """Number of rows the line occupies."""
        return 0 if not self.tiles else max(p.row for p in self.tiles) + 1

    def row_values(self, ends: list[int] | None = None) -> list[list[int]]:  # pragma: no cover
        """Debug helper: the pip values shown on each row, left to right."""
        rows: list[list[int]] = []
        for index in range(self.rows):
            values: list[int] = []
            for placed in self.tiles:
                if placed.row != index:
                    continue
                a, b = tile_values(placed.tile)
                values.extend((b, a) if placed.flip else (a, b))
            rows.append(values)
        return rows


def line_tiles(state: GameState) -> tuple[list[tuple[int, bool]], int]:
    """Rebuild the physical line as ``([(tile, flip)], opener_index)``.

    The returned list is in screen order (left to right): the tiles attached to
    the left end first, newest first, then the opener, then the right end's tiles
    in play order. ``flip`` says the tile must be drawn with its halves swapped so
    that its matching half touches the line.
    """
    table = list(state.table)
    if not table:
        return [], 0
    attached = list(state.table_ends)
    if len(attached) != len(table):          # a state built by hand in a test
        attached = [0] * len(table)

    first = table[0]
    a, b = tile_values(first)
    current = [a, b]                          # the opener defines both ends
    entries: list[tuple[int, bool, int]] = [(first, False, -1)]
    for tile, end in zip(table[1:], attached[1:]):
        end = 1 if int(end) == 1 else 0
        target = current[end]
        x, y = tile_values(tile)
        if x == y:                            # a double: both halves match, either way up
            flip, exposed = False, x
        elif target == x:                     # x touches the line, y stays open
            flip, exposed = (end == 0), y
        else:                                 # y touches the line, x stays open
            flip, exposed = (end == 1), x
        current[end] = exposed
        entries.append((tile, flip, end))

    left = [(t, f) for t, f, e in entries if e == 0]     # oldest first
    right = [(t, f) for t, f, e in entries if e == 1]
    anchor = len(left)                                   # index of the opener
    return list(reversed(left)) + [(first, False)] + right, anchor


def layout_line(
    state: GameState,
    size: tuple[int, int],
    *,
    top: int = BOARD_TOP,
    margin: int = BOARD_MARGIN,
    gap: int = TILE_GAP,
) -> LineLayout:
    """Lay the line out on screen, growing away from the opener on both sides.

    The opener stays where it was first drawn (the line does not jump around as
    tiles are added) and each side extends outwards from it; a side that reaches
    the board edge overflows onto the next row, laid out in reading order. Tiles
    that a side pushes past its edge keep their relative order, so the only place
    the "matching numbers touch" invariant breaks is the row break itself -- the
    same break a real table has when a line is restarted on a new row.
    """
    tiles, anchor = line_tiles(state)
    if not tiles:
        return LineLayout((), {})

    board_w = size[0] - PANEL_W
    avail = board_w - 2 * margin
    widths = [tile_width(t) for t, _ in tiles]
    heights = [tile_height(t) for t, _ in tiles]
    total = len(tiles)
    placed: list[PlacedTile | None] = [None] * total

    def put(index: int, x: int, row: int) -> None:
        y = top + row * ROW_PITCH + (ROW_H - heights[index]) // 2
        placed[index] = PlacedTile(
            tiles[index][0], tiles[index][1], row,
            pygame.Rect(x, y, widths[index], heights[index]),
        )

    # Row 0: the opener is centred, so the line has room to grow both ways.
    anchor_x = margin + max(0, (avail - widths[anchor]) // 2)
    put(anchor, anchor_x, 0)

    cursor = anchor_x + widths[anchor] + gap
    fitted_right = 0
    for index in range(anchor + 1, total):
        if cursor + widths[index] > margin + avail:
            break
        put(index, cursor, 0)
        cursor += widths[index] + gap
        fitted_right += 1

    cursor = anchor_x - gap
    fitted_left = 0
    for index in range(anchor - 1, -1, -1):
        x = cursor - widths[index]
        if x < margin:
            break
        put(index, x, 0)
        cursor = x - gap
        fitted_left += 1

    # Overflow, one row per group, in reading order (left to right).
    overflow_right = list(range(anchor + 1 + fitted_right, total))
    overflow_left = list(range(0, anchor - fitted_left))
    row = 1
    for group in (overflow_right, overflow_left):
        if not group:
            continue
        cursor = margin
        for index in group:
            if cursor + widths[index] > margin + avail:
                row += 1
                cursor = margin
            put(index, cursor, row)
            cursor += widths[index] + gap
        row += 1

    laid = [p for p in placed if p is not None]
    first, last = laid[0], laid[-1]
    first_rect, last_rect = first.rect, last.rect
    ends = {
        0: pygame.Rect(first_rect.x - 10, first_rect.y - 8,
                       first_rect.width // 2 + 10, first_rect.height + 16),
        1: pygame.Rect(last_rect.right - last_rect.width // 2, last_rect.y - 8,
                       last_rect.width // 2 + 10, last_rect.height + 16),
    }
    return LineLayout(tuple(laid), ends)


def line_values(layout: LineLayout) -> list[int]:
    """The pip values drawn along the line, left to right (one per tile half).

    Used by the tests to check the "equal numbers touch" invariant: consecutive
    entries belong to the same tile when their index is even.
    """
    values: list[int] = []
    for placed in layout.tiles:
        a, b = tile_values(placed.tile)
        values.extend((b, a) if placed.flip else (a, b))
    return values
