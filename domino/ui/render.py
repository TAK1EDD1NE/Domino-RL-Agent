"""Pygame rendering for the 2v2 domino table (interactive mode and replays).

The renderer is the only place pygame is used; the training path never imports
this package (there is a test asserting that). The line is laid out by
:mod:`domino.ui.board`, which rebuilds the physical line from the engine's
``(table, table_ends)``: tiles extend away from the opener on both sides, each
one oriented so its matching number touches the line, and the two open ends are
drawn where they really are (with their value) so a click picks the end it looks
like. Doubles are drawn crosswise, exactly as they sit on a physical table.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import os

import pygame

from ..engine import GameState, TEAM_OF_SEAT
from ..tiles import tile_pips, tile_values
from .board import (
    PANEL_W,
    TILE_LONG,
    TILE_SHORT,
    LineLayout,
    layout_line,
)

PIP_RADIUS: int = 4
HAND_GAP: int = 8

#: Colours.
BG = (24, 84, 52)
PANEL = (18, 30, 24)
TILE = (245, 243, 232)
TILE_EDGE = (120, 118, 110)
PIP = (32, 34, 38)
TEAM_A = (86, 156, 240)
TEAM_B = (232, 108, 108)
ACTING = (255, 214, 96)
LEGAL = (110, 220, 140)
TEXT = (238, 238, 238)
MUTED = (168, 178, 172)

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
    body = TILE if not dim else (170, 168, 160)
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


class Renderer:
    """Draws a game state and collects input events for the pygame modes."""

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
    ) -> None:
        """Render one frame of ``state``.

        ``human_seat`` enables the clickable hand row; ``legal`` marks playable
        tiles; ``overlay`` draws per-action probability bars (replay mode);
        ``banner`` is the status line; ``seat_labels`` names the agent sitting in
        each seat (interactive mode -- so you can see who you are facing).
        """
        self.screen.fill(BG)
        self._draw_board(state, acting_seat, ambiguous_ends or ())
        self._draw_panel(state, acting_seat, banner, last_actions, overlay, human_seat, seat_labels)
        if human_seat is not None:
            self._draw_hand(state, human_seat, legal or [], selected, ambiguous_ends or [])

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
            text = self.small.render(
                f"{label} - click here" if active else label, True, colour
            )
            y = min(rect.bottom + 2, self.size[1] - TILE_LONG - 60)
            self.screen.blit(text, (rect.x, y))

    def _draw_board(
        self,
        state: GameState,
        acting_seat: int | None,
        ambiguous_ends: Sequence[int] = (),
    ) -> None:
        self.end_rects = {}
        if not state.table:
            text = self.big.render("table empty — the leader opens", True, MUTED)
            self.screen.blit(text, (40, 180))
            return
        layout = layout_line(state, self.size)
        for placed in layout.tiles:
            a, b = tile_values(placed.tile)
            draw_tile(
                self.screen, placed.tile, placed.rect.topleft,
                crosswise=a == b, flipped=placed.flip,
            )
        self.end_rects = dict(layout.ends)
        self._draw_open_ends(state, layout, ambiguous_ends)
        if acting_seat is not None:
            bottom = max(placed.rect.bottom for placed in layout.tiles)
            label = self.small.render(
                f"seat {acting_seat} (team {TEAM_OF_SEAT[acting_seat]}) to play",
                True, ACTING,
            )
            self.screen.blit(label, (40, min(bottom + 26, self.size[1] - TILE_LONG - 50)))

    def _draw_panel(
        self,
        state: GameState,
        acting_seat: int | None,
        banner: str,
        last_actions: Sequence[str],
        overlay: Sequence[float] | None,
        human_seat: int | None = None,
        seat_labels: Mapping[int, str] | None = None,
    ) -> None:
        panel = pygame.Rect(self.size[0] - PANEL_W, 0, PANEL_W, self.size[1])
        pygame.draw.rect(self.screen, PANEL, panel)
        lines: list[tuple[str, tuple[int, int, int]]] = [
            (f"deal {state.deal_no}", TEXT),
            (f"match  A {state.scores[0]} : {state.scores[1]} B", TEXT),
            (f"open ends  {state.ends if state.ends else '-'}", TEXT),
            (f"pass streak  {state.pass_streak}/4", TEXT),
            ("", TEXT),
        ]
        for seat in range(4):
            lines.append(
                (
                    seat_line(
                        state, seat,
                        label=(seat_labels or {}).get(seat, ""),
                        acting=seat == acting_seat,
                        human=seat == human_seat,
                    ),
                    ACTING if seat == acting_seat else MUTED,
                )
            )
        if human_seat is not None:
            lines.append((f"  your pips  {state.pips(human_seat)}", MUTED))
        lines.append(("", TEXT))
        for text in last_actions[-6:]:
            lines.append((text, MUTED))
        if banner:
            lines.append(("", TEXT))
            lines.append((banner, LEGAL))
        y = 16
        for text, colour in lines:
            if text:
                self.screen.blit(self.small.render(text, True, colour), (panel.x + 14, y))
            y += 20
        if overlay is not None:
            self._draw_overlay(overlay, panel)

    def _draw_overlay(self, overlay: Sequence[float], panel: pygame.Rect) -> None:
        """Draw probability bars for the actions with non-trivial probability."""
        entries = [(index, float(prob)) for index, prob in enumerate(overlay) if prob > 0.005]
        entries.sort(key=lambda item: item[1], reverse=True)
        y = panel.y + 470
        self.screen.blit(self.small.render("policy (tile/end)", True, TEXT), (panel.x + 14, y))
        y += 22
        for index, prob in entries[:8]:
            tid, end = index // 2, index % 2
            label = f"[{tile_values(tid)[0]}|{tile_values(tid)[1]}] e{end} {prob:5.1%}"
            self.screen.blit(self.small.render(label, True, MUTED), (panel.x + 14, y))
            bar = pygame.Rect(panel.x + 190, y + 3, int(110 * min(1.0, prob * 2)), 10)
            pygame.draw.rect(self.screen, LEGAL, bar)
            y += 20

    def _draw_hand(
        self,
        state: GameState,
        human_seat: int,
        legal: Sequence[int],
        selected: int | None,
        ambiguous_ends: Sequence[int],
    ) -> None:
        self.hand_rects = {}
        legal_set = set(legal)
        x, y = 40, self.size[1] - TILE_LONG - 30
        title = self.small.render(
            f"your hand (seat {human_seat}, team {TEAM_OF_SEAT[human_seat]}) — click a highlighted tile",
            True, TEXT,
        )
        self.screen.blit(title, (40, y - 26))
        for tile in sorted(state.hands[human_seat], reverse=True):
            a, b = tile_values(tile)
            crosswise = a == b
            rect = draw_tile(
                self.screen, tile, (x, y),
                crosswise=crosswise,
                frame=TEAM_A if TEAM_OF_SEAT[human_seat] == 0 else TEAM_B,
                highlight=tile in legal_set,
                dim=tile not in legal_set,
            )
            if tile == selected:
                pygame.draw.rect(self.screen, ACTING, rect.inflate(10, 10), width=3, border_radius=10)
            self.hand_rects[tile] = rect
            x += rect.width + HAND_GAP
        if ambiguous_ends:
            hint = self.small.render(
                "tile matches both ends — click the end to attach it to", True, ACTING
            )
            self.screen.blit(hint, (40, y - 48))

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


def seat_line(state: GameState, seat: int, *, label: str = "", acting: bool = False,
              human: bool = False) -> str:
    """Panel line for one seat: how many tiles it holds and who plays it.

    Only the *number* of remaining tiles is shown -- every player can see the
    backs on a real table -- never which tiles they are.
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
