"""Pygame rendering for the 2v2 domino table (interactive mode and replays).

The window *is* the table: you sit at the bottom with your tiles face up, your
partner sits opposite and the two opponents sit left and right. The other seats
show one tile per tile they hold -- face down by default (a real opponent's hand
is hidden information), or face up when the caller asks for the open-hands view
the owner uses to watch a game. The line is laid out by :mod:`domino.ui.board`,
which rebuilds the physical line from the engine's ``(table, table_ends)`` so the
numbers that touch each other always match, and frames the two open ends where
they really are.

A finished hand is reported by :meth:`Renderer.draw_summary` -- with clickable
buttons -- rather than silently replaced by the next one, and any position that
has already been played can be drawn again, which is what the replay mode walks
through.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pygame

from ..engine import GameState, TEAM_OF_SEAT
from ..tiles import tile_pips, tile_values
from .board import TILE_LONG, TILE_SHORT, LineLayout, layout_line

PIP_RADIUS: int = 4
HAND_GAP: int = 8

#: Colours.
BG = (24, 84, 52)
PANEL = (18, 30, 24)
TILE = (245, 243, 232)
#: A face-up tile that cannot be played right now.
TILE_DIM = (170, 168, 160)
TILE_EDGE = (120, 118, 110)
PIP = (32, 34, 38)
TEAM_A = (86, 156, 240)
TEAM_B = (232, 108, 108)
ACTING = (255, 214, 96)
LEGAL = (110, 220, 140)
TEXT = (238, 238, 238)
MUTED = (168, 178, 172)
BUTTON = (44, 74, 60)
BUTTON_EDGE = (120, 200, 150)
#: A face-down tile (the other seats' hands, hidden).
BACK = (26, 44, 88)
BACK_EDGE = (12, 22, 44)
BACK_MARK = (78, 110, 176)
#: The dimming layer under the end-of-hand report.
SCRIM = (6, 14, 10)

#: Face-down tile size and the pitch of a stack of them.
BACK_LONG: int = 46
BACK_SHORT: int = 22
BACK_PITCH: int = 31

#: Scale of the tiles drawn in the other seats' regions (the line and your own
#: hand are drawn full size).
SEAT_TILE_SCALE: float = 0.62

#: Normalised pip positions per pip count (x, y in 0..1 inside a half).
_PIP_LAYOUT: dict[int, tuple[tuple[float, float], ...]] = {
    0: (),
    1: ((0.5, 0.5),),
    2: ((0.28, 0.28), (0.72, 0.72)),
    3: ((0.28, 0.28), (0.5, 0.5), (0.72, 0.72)),
    4: ((0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)),
    5: ((0.28, 0.28), (0.72, 0.28), (0.5, 0.5), (0.28, 0.72), (0.72, 0.72)),
    6: ((0.28, 0.22), (0.72, 0.22), (0.28, 0.5), (0.72, 0.5), (0.28, 0.78), (0.72, 0.78)),
}

#: Cache of pre-scaled tile surfaces, keyed by ``(tile, scale)``.
_TILE_SURFACES: dict[tuple[int, float], pygame.Surface] = {}


def _draw_pips(surface: pygame.Surface, rect: pygame.Rect, count: int) -> None:
    """Draw ``count`` pips inside ``rect`` using the standard domino patterns."""
    for fx, fy in _PIP_LAYOUT[count]:
        pygame.draw.circle(
            surface,
            PIP,
            (rect.x + int(fx * rect.width), rect.y + int(fy * rect.height)),
            PIP_RADIUS,
        )


def draw_tile(
    surface: pygame.Surface,
    tile: int,
    top_left: tuple[int, int],
    *,
    crosswise: bool = False,
    frame: tuple[int, int, int] | None = None,
    highlight: bool = False,
    dim: bool = False,
    flipped: bool = False,
) -> pygame.Rect:
    """Draw one domino tile and return its rect.

    A normal tile is drawn long-and-horizontal with its two values left/right; a
    double is drawn crosswise (long-and-vertical), which is how it is laid on a
    real table. ``flipped`` swaps the two halves, which is how the line orients a
    tile so that its matching number touches the tile it was played against.
    """
    a, b = tile_values(tile)
    if flipped:
        a, b = b, a
    width, height = (TILE_SHORT, TILE_LONG) if crosswise else (TILE_LONG, TILE_SHORT)
    rect = pygame.Rect(top_left[0], top_left[1], width, height)
    body = TILE if not dim else TILE_DIM
    pygame.draw.rect(surface, body, rect, border_radius=6)
    pygame.draw.rect(surface, frame or TILE_EDGE, rect, width=3, border_radius=6)
    if highlight:
        pygame.draw.rect(surface, LEGAL, rect.inflate(6, 6), width=3, border_radius=8)

    half_long = height // 2 if crosswise else width // 2
    if crosswise:
        first = pygame.Rect(rect.x, rect.y, width, half_long)
        second = pygame.Rect(rect.x, rect.y + half_long, width, half_long)
        pygame.draw.line(surface, TILE_EDGE, (rect.x, rect.y + half_long),
                         (rect.right, rect.y + half_long), 2)
    else:
        first = pygame.Rect(rect.x, rect.y, half_long, height)
        second = pygame.Rect(rect.x + half_long, rect.y, half_long, height)
        pygame.draw.line(surface, TILE_EDGE, (rect.x + half_long, rect.y),
                         (rect.x + half_long, rect.bottom), 2)
    _draw_pips(surface, first, a)
    _draw_pips(surface, second, b)
    return rect


def draw_back(surface: pygame.Surface, rect: pygame.Rect) -> None:
    """Draw one face-down tile (a 'reversed' tile) filling ``rect``."""
    pygame.draw.rect(surface, BACK, rect, border_radius=5)
    pygame.draw.rect(surface, BACK_EDGE, rect, width=2, border_radius=5)
    inner = rect.inflate(-8, -8)
    if inner.width > 3 and inner.height > 3:
        pygame.draw.rect(surface, BACK_MARK, inner, width=2, border_radius=3)


def tile_surface(tile: int, scale: float = SEAT_TILE_SCALE) -> pygame.Surface:
    """A tile drawn on its own surface, scaled, for the seats around the table.

    Cached per ``(tile, scale)``: the same handful of surfaces is asked for on
    every frame, and re-rendering 28 of them per frame is pure waste.
    """
    key = (int(tile), round(float(scale), 3))
    cached = _TILE_SURFACES.get(key)
    if cached is not None:
        return cached
    a, b = tile_values(tile)
    crosswise = a == b
    width = TILE_SHORT if crosswise else TILE_LONG
    height = TILE_LONG if crosswise else TILE_SHORT
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    draw_tile(surface, tile, (0, 0), crosswise=crosswise)
    if scale != 1.0:
        surface = pygame.transform.smoothscale(
            surface, (max(1, int(width * scale)), max(1, int(height * scale)))
        )
    _TILE_SURFACES[key] = surface
    return surface


def tile_back_size() -> tuple[int, int]:
    """Width and height of a face-down tile."""
    return BACK_LONG, BACK_SHORT


def seat_line(state: GameState, seat: int, *, label: str = "", acting: bool = False,
              human: bool = False) -> str:
    """Caption for one seat: how many tiles it holds and who plays it.

    Only the *number* of remaining tiles is shown in the caption; the tiles
    themselves are drawn either face down (hidden) or face up (open-hands view).
    """
    marker = "->" if acting else "  "
    who = "you" if human else label
    return f"{marker} seat {seat}  {state.hand_size(seat)} tiles left  {who}"


def format_action(tile: int, end: int | None = None) -> str:
    """Human-readable action text used in the info panel."""
    a, b = tile_values(tile)
    suffix = "" if end is None else f" on end {end}"
    return f"[{a}|{b}] ({tile_pips(tile)} pips){suffix}"


def describe_hand(hand: Iterable[int]) -> str:
    """Compact description of a hand, used by the CLI banners."""
    return " ".join(format_action(tile) for tile in sorted(hand, reverse=True)) or "(empty)"


def table_geometry(size: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """Regions of the table for a window of ``size``, each ``(x, y, w, h)``.

    Scales with the window, so the same layout holds at the 1024x640 the tests
    use and at the 1280x720 the game opens with: a status strip on top, the
    partner opposite, the two opponents left and right (wide enough for a
    revealed tile plus its caption), the line in the middle and your hand across
    the bottom.
    """
    width, height = size
    bar_h = max(56, int(height * 0.096))
    strip_h = max(78, int(height * 0.11))          # fits a revealed crosswise double
    side_w = max(76, int(width * 0.059))
    hand_h = TILE_LONG + 26
    line_top = bar_h + strip_h + 8
    line_bottom = height - hand_h - 12
    return {
        "bar": (0, 0, width, bar_h),
        "partner": (0, bar_h, width, strip_h),
        "left": (10, line_top, side_w, line_bottom - line_top),
        "right": (width - 10 - side_w, line_top, side_w, line_bottom - line_top),
        "line": (10 + side_w + 26, line_top,
                 width - 2 * (10 + side_w + 26), line_bottom - line_top),
        "hand": (0, height - hand_h, width, hand_h),
    }


def seat_sides(human_seat: int | None) -> dict[str, int]:
    """Which seat sits on which side of the table (the partner sits opposite)."""
    bottom = 0 if human_seat is None else human_seat
    return {
        "bottom": bottom,
        "partner": (bottom + 2) % 4,
        "left": (bottom + 1) % 4,
        "right": (bottom + 3) % 4,
    }


class Renderer:
    """Draws the table and collects input events for the pygame modes."""

    def __init__(self, size: tuple[int, int] = (1280, 720), *, title: str = "Domino 2v2") -> None:
        pygame.init()
        self.size = size
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.small = pygame.font.Font(None, 17)
        self.big = pygame.font.Font(None, 26)
        self.hand_rects: dict[int, pygame.Rect] = {}
        #: Click regions of the two open ends, refreshed by every board draw.
        self.end_rects: dict[int, pygame.Rect] = {}
        #: This seat's tile rects, refreshed by every table draw.
        self.seat_rects: dict[int, list[pygame.Rect]] = {}
        #: Click regions of the report's buttons, refreshed by ``draw_summary``.
        self.summary_buttons: dict[str, pygame.Rect] = {}

    # -- input ------------------------------------------------------------
    def poll(self) -> list[pygame.event.Event]:
        """Return the events seen since the last poll."""
        return list(pygame.event.get())

    def tick(self, fps: int = 30) -> None:
        """Cap the frame rate and present the drawn frame."""
        self.clock.tick(fps)
        pygame.display.flip()

    def close(self) -> None:
        """Shut the display down (safe to call more than once)."""
        pygame.display.quit()

    # -- drawing ----------------------------------------------------------
    def draw(
        self,
        state: GameState,
        *,
        acting_seat: int | None = None,
        human_seat: int | None = None,
        legal: Sequence[int] | None = None,
        overlay: Sequence[float] | None = None,
        banner: str = "",
        last_actions: Sequence[str] = (),
        selected: int | None = None,
        ambiguous_ends: Sequence[int] | None = None,
        seat_labels: Mapping[int, str] | None = None,
        reveal_hands: bool = False,
    ) -> None:
        """Render one frame of ``state`` as a table.

        ``human_seat`` puts your hand face up at the bottom; ``reveal_hands``
        shows what the other three seats are holding instead of face-down tiles.
        ``legal`` marks playable tiles, ``overlay`` draws per-action probability
        bars (replay mode), ``banner`` is the status message and ``seat_labels``
        names the agent in each seat.
        """
        regions = table_geometry(self.size)
        self.screen.fill(BG)
        self._draw_top_bar(state, regions, acting_seat, banner, seat_labels, human_seat)
        self._draw_seats(state, regions, acting_seat, human_seat, seat_labels,
                         reveal=reveal_hands)
        self._draw_board(state, regions, ambiguous_ends or ())
        if human_seat is not None:
            self._draw_hand(state, human_seat, legal or [], selected, ambiguous_ends or [])
        self._draw_footer(regions, last_actions, overlay)

    def draw_summary(
        self,
        lines: Sequence[str],
        *,
        title: str = "",
        hint: str = "",
        buttons: Sequence[tuple[str, str]] = (),
    ) -> dict[str, pygame.Rect]:
        """Draw the end-of-hand report over the finished table.

        ``buttons`` are ``(key, label)`` pairs drawn along the bottom of the
        panel; the returned ``{key: rect}`` map is also kept in
        :attr:`summary_buttons` so a caller can hit-test clicks. The caller decides
        when to move on, so a hand is never replaced before it has been read.
        """
        width, height = self.size
        box = pygame.Rect(0, 0, int(width * 0.68), int(height * 0.56))
        box.center = (width // 2, height // 2)
        scrim = pygame.Surface(self.size, pygame.SRCALPHA)
        scrim.fill((*SCRIM, 205))
        self.screen.blit(scrim, (0, 0))
        pygame.draw.rect(self.screen, PANEL, box, border_radius=10)
        pygame.draw.rect(self.screen, ACTING, box, width=3, border_radius=10)

        y = box.y + 20
        if title:
            self.screen.blit(self.big.render(title, True, ACTING), (box.x + 26, y))
            y += 36
        for line in lines:
            if line:
                self.screen.blit(self.font.render(line, True, TEXT), (box.x + 26, y))
            y += 25

        self.summary_buttons = {}
        if buttons:
            button_y = box.bottom - 62
            x = box.x + 26
            for key, label in buttons:
                text = self.font.render(label, True, TEXT)
                rect = pygame.Rect(x, button_y, text.get_width() + 28, 36)
                pygame.draw.rect(self.screen, BUTTON, rect, border_radius=8)
                pygame.draw.rect(self.screen, BUTTON_EDGE, rect, width=2, border_radius=8)
                self.screen.blit(text, (rect.x + 14, rect.y + 9))
                self.summary_buttons[key] = rect
                x = rect.right + 16
        if hint:
            self.screen.blit(self.small.render(hint, True, LEGAL),
                             (box.x + 26, box.bottom - 24))
        return dict(self.summary_buttons)

    # -- the table's parts ------------------------------------------------
    def _draw_top_bar(
        self,
        state: GameState,
        regions: Mapping[str, tuple[int, int, int, int]],
        acting_seat: int | None,
        banner: str,
        seat_labels: Mapping[int, str] | None,
        human_seat: int | None,
    ) -> None:
        """The status strip: the match state on one line, the message on the next."""
        x, y, width, height = regions["bar"]
        pygame.draw.rect(self.screen, PANEL, pygame.Rect(x, y, width, height))
        your_team = TEAM_OF_SEAT[1 if human_seat is None else human_seat]
        ends = state.ends if state.ends else "-"
        if acting_seat is None:
            turn = "-"
        elif acting_seat == human_seat:
            turn = "you"
        else:
            turn = f"seat {acting_seat} ({(seat_labels or {}).get(acting_seat, 'AI')})"
        status = (
            f"deal {state.deal_no}    match  your team {state.scores[your_team]}"
            f" : {state.scores[1 - your_team]} the AI team    open ends {ends}"
            f"    pass streak {state.pass_streak}/4    to play: {turn}"
        )
        self.screen.blit(self.small.render(status, True, TEXT), (x + 16, y + 10))
        if banner:
            self.screen.blit(self.big.render(banner, True, ACTING), (x + 16, y + 32))

    def _draw_seats(
        self,
        state: GameState,
        regions: Mapping[str, tuple[int, int, int, int]],
        acting_seat: int | None,
        human_seat: int | None,
        seat_labels: Mapping[int, str] | None,
        *,
        reveal: bool = False,
    ) -> None:
        """The seats that are not yours, around the table.

        One tile per tile held, face down by default or face up when ``reveal``:
        the partner in a row opposite you, the two opponents in columns left and
        right, each with its (rotated) caption along the table edge.
        """
        self.seat_rects = {}
        sides = seat_sides(human_seat)
        for side in ("partner", "left", "right"):
            seat = sides[side]
            if seat == human_seat:
                continue
            caption = seat_line(
                state, seat, label=(seat_labels or {}).get(seat, ""), acting=seat == acting_seat
            )
            colour = ACTING if seat == acting_seat else MUTED
            held = sorted(state.hands[seat], reverse=True)
            if side == "partner":
                x, y, width, _ = regions["partner"]
                text = caption + ("   (your partner)" if human_seat is not None else "")
                surface = self.small.render(text, True, colour)
                self.screen.blit(surface, (x + (width - surface.get_width()) // 2, y + 2))
                self.seat_rects[seat] = self._draw_seat_row(held, width // 2, y + 26, reveal)
                continue
            x, y, width, height = regions[side]
            stack_x = x if side == "left" else x + width - self._seat_tile_width(reveal)
            self.seat_rects[seat] = self._draw_seat_column(
                held, stack_x, y + 4, height - 8, reveal
            )
            rotated = pygame.transform.rotate(
                self.small.render(caption, True, colour), 90 if side == "left" else -90
            )
            label_x = (x + self._seat_tile_width(reveal) + 8 if side == "left"
                       else x + width - self._seat_tile_width(reveal)
                       - rotated.get_width() - 8)
            self.screen.blit(rotated, (label_x, y + (height - rotated.get_height()) // 2))

    @staticmethod
    def _seat_tile_width(reveal: bool) -> int:
        """Width taken up by one seat tile (a revealed tile is wider than a back)."""
        if not reveal:
            return BACK_SHORT
        return int(TILE_LONG * SEAT_TILE_SCALE)

    def _draw_seat_row(
        self, held: Sequence[int], centre_x: int, y: int, reveal: bool
    ) -> list[pygame.Rect]:
        """One seat's tiles in a horizontal row (the partner, opposite you)."""
        surfaces = [tile_surface(tile) for tile in held] if reveal else []
        widths = [s.get_width() for s in surfaces] if reveal else [BACK_LONG] * len(held)
        total = sum(width + 6 for width in widths) - 6 if widths else 0
        x = centre_x - total // 2
        rects: list[pygame.Rect] = []
        for index, tile in enumerate(held):
            if reveal:
                surface = surfaces[index]
                rect = pygame.Rect(x, y, surface.get_width(), surface.get_height())
                self.screen.blit(surface, rect.topleft)
            else:
                rect = pygame.Rect(x, y, BACK_LONG, BACK_SHORT)
                draw_back(self.screen, rect)
            rects.append(rect)
            x += widths[index] + 6
        return rects

    def _draw_seat_column(
        self, held: Sequence[int], x: int, y: int, height: int, reveal: bool
    ) -> list[pygame.Rect]:
        """One seat's tiles in a vertical column (the opponents, left and right)."""
        surfaces = [tile_surface(tile) for tile in held] if reveal else []
        heights = [s.get_height() for s in surfaces] if reveal else [BACK_LONG] * len(held)
        total = sum(one + 6 for one in heights) - 6 if heights else 0
        top = y + max(0, (height - total) // 2)
        rects: list[pygame.Rect] = []
        for index, tile in enumerate(held):
            if reveal:
                surface = surfaces[index]
                rect = pygame.Rect(x, top, surface.get_width(), surface.get_height())
                self.screen.blit(surface, rect.topleft)
            else:
                rect = pygame.Rect(x, top, BACK_SHORT, BACK_LONG)
                draw_back(self.screen, rect)
            rects.append(rect)
            top += heights[index] + 6
        return rects

    def _draw_open_ends(
        self, state: GameState, layout: LineLayout, ambiguous_ends: Sequence[int]
    ) -> None:
        """Frame the two ends of the line and label them with their value.

        A selected hand tile that fits both ends makes the choice explicit: the
        two frames light up in the 'acting' colour, and each frame covers exactly
        the region :meth:`end_at` maps back to that end.
        """
        for end, rect in sorted(layout.ends.items()):
            active = end in ambiguous_ends
            value = state.ends[end] if end < len(state.ends) else None
            colour = ACTING if active else LEGAL
            pygame.draw.rect(self.screen, colour, rect, width=3, border_radius=6)
            label = f"end {end} = {value}" if value is not None else f"end {end}"
            text = self.small.render(f"{label} - click here" if active else label, True, colour)
            self.screen.blit(text, (rect.x, min(rect.bottom + 2, self.size[1] - TILE_LONG - 62)))

    def _draw_board(
        self,
        state: GameState,
        regions: Mapping[str, tuple[int, int, int, int]],
        ambiguous_ends: Sequence[int] = (),
    ) -> None:
        """Draw the played line inside the middle of the table."""
        self.end_rects = {}
        x, y, width, height = regions["line"]
        if not state.table:
            text = self.big.render("table empty — the leader opens", True, MUTED)
            self.screen.blit(text, (x + (width - text.get_width()) // 2, y + height // 2))
            return
        layout = layout_line(state, self.size, top=y, left=x, right=x + width)
        for placed in layout.tiles:
            a, b = tile_values(placed.tile)
            draw_tile(
                self.screen, placed.tile, placed.rect.topleft,
                crosswise=a == b, flipped=placed.flip,
            )
        self.end_rects = dict(layout.ends)
        self._draw_open_ends(state, layout, ambiguous_ends)

    def _draw_hand(
        self,
        state: GameState,
        human_seat: int,
        legal: Sequence[int],
        selected: int | None,
        ambiguous_ends: Sequence[int],
    ) -> None:
        """Your own tiles, face up along the bottom of the table."""
        self.hand_rects = {}
        legal_set = set(legal)
        tiles = sorted(state.hands[human_seat], reverse=True)
        steps = [(TILE_SHORT if tile_values(t)[0] == tile_values(t)[1] else TILE_LONG) + HAND_GAP
                 for t in tiles]
        total = sum(steps) - HAND_GAP if steps else 0
        x = (self.size[0] - total) // 2
        y = self.size[1] - TILE_LONG - 4

        caption = seat_line(state, human_seat, human=True, acting=True)
        caption += f"   (pips {state.pips(human_seat)})   —   click a highlighted tile"
        self.screen.blit(self.small.render(caption, True, ACTING), (x, y - 22))
        if ambiguous_ends:
            hint = self.small.render(
                "that tile fits both ends — click the framed end to attach it to", True, ACTING
            )
            self.screen.blit(hint, (x, y - 42))

        for step, tile in zip(steps, tiles):
            a, b = tile_values(tile)
            rect = draw_tile(
                self.screen, tile, (x, y),
                crosswise=a == b,
                frame=TEAM_B if TEAM_OF_SEAT[human_seat] else TEAM_A,
                highlight=tile in legal_set,
                dim=tile not in legal_set,
            )
            if tile == selected:
                pygame.draw.rect(self.screen, ACTING, rect.inflate(10, 10), width=3,
                                 border_radius=10)
            self.hand_rects[tile] = rect
            x += step

    def _draw_footer(
        self,
        regions: Mapping[str, tuple[int, int, int, int]],
        last_actions: Sequence[str],
        overlay: Sequence[float] | None,
    ) -> None:
        """Recent moves in the bottom-left corner, and the replay overlay."""
        x, y, width, height = regions["hand"]
        log_y = y + 8
        for text in list(last_actions)[-3:]:
            self.screen.blit(self.small.render(self._short_action(text), True, MUTED), (14, log_y))
            log_y += 19
        if overlay is not None:
            self._draw_overlay(overlay, pygame.Rect(x + 14, y + 4, width // 2, height - 8))

    @staticmethod
    def _short_action(text: str) -> str:
        """Compact form of an action line, for the corner log."""
        return text.replace(" pips)", "p").replace(" on end ", " e").replace(" (", " ")[:44]

    def _draw_overlay(self, overlay: Sequence[float], area: pygame.Rect) -> None:
        """Draw probability bars for the actions with non-trivial probability."""
        entries = [(index, float(prob)) for index, prob in enumerate(overlay) if prob > 0.005]
        entries.sort(key=lambda item: item[1], reverse=True)
        y = area.y
        self.screen.blit(self.small.render("policy (tile/end)", True, TEXT), (area.x, y))
        y += 20
        for index, prob in entries[:8]:
            tid, end = index // 2, index % 2
            label = f"[{tile_values(tid)[0]}|{tile_values(tid)[1]}] e{end} {prob:5.1%}"
            self.screen.blit(self.small.render(label, True, MUTED), (area.x, y))
            bar = pygame.Rect(area.x + 176, y + 3, int(110 * min(1.0, prob * 2)), 10)
            pygame.draw.rect(self.screen, LEGAL, bar)
            y += 19

    # -- geometry ---------------------------------------------------------
    def tile_at(self, pos: tuple[int, int]) -> int | None:
        """Return the hand tile under ``pos`` (or ``None``)."""
        for tile, rect in self.hand_rects.items():
            if rect.collidepoint(pos):
                return tile
        return None

    def end_at(self, pos: tuple[int, int], state: GameState, ends: Sequence[int]) -> int | None:
        """Map a click on one of the two framed ends to that end's index.

        The frames are the ones :meth:`_draw_board` drew (the outer half of the
        tile at each end of the line), so the target the player sees is the target
        the click tests. Returns ``None`` when there is no end to choose, or the
        click missed both."""
        if len(ends) < 2:
            return None
        for end, rect in sorted(self.end_rects.items()):
            if rect.collidepoint(pos):
                return end
        return None

    def button_at(self, pos: tuple[int, int]) -> str | None:
        """Return the key of the report button under ``pos`` (or ``None``)."""
        for key, rect in self.summary_buttons.items():
            if rect.collidepoint(pos):
                return key
        return None
