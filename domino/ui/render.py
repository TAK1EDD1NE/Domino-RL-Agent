"""Pygame rendering for the 2v2 domino table (interactive mode and replays).

The renderer is the only place pygame is used; the training path never imports
this package (there is a test asserting that). Doubles are drawn crosswise,
exactly as they sit on a physical table, and the two open ends are shown
explicitly in the info panel.

One honest simplification: the engine stores the tiles of a deal in *play order*
and keeps the two open end values, but it does not store which end each tile was
attached to. The board is therefore drawn as the chronological line rather than
reconstructing the physical left/right layout; the authoritative open ends are
always displayed, so the game state shown is exact even though the tile order on
screen is not the physical order.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import os

import pygame

from ..engine import GameState, PARTNER_OF_SEAT, TEAM_OF_SEAT
from ..tiles import tile_pips, tile_values

#: Tile geometry (pixels).
TILE_LONG: int = 84
TILE_SHORT: int = 42
PIP_RADIUS: int = 4
HAND_GAP: int = 8

#: Size of the two "attach to this end" boxes drawn while a tile selected in the
#: hand fits both open ends (the player clicks one of them to choose the end).
END_MARKER_W: int = 176
END_MARKER_H: int = 56
END_MARKER_Y: int = 124
#: Width of the right-hand info panel; the board stops there.
PANEL_W: int = 330

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
) -> pygame.Rect:
    """Draw one domino tile and return its rect.

    A normal tile is drawn long-and-horizontal with its two values left/right; a
    double is drawn crosswise (long-and-vertical), which is how it is laid on a
    real table.
    """
    a, b = tile_values(tile)
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
        self._draw_board(state, acting_seat)
        self._draw_panel(state, acting_seat, banner, last_actions, overlay, human_seat, seat_labels)
        if ambiguous_ends:
            self._draw_end_markers(state, ambiguous_ends)
        if human_seat is not None:
            self._draw_hand(state, human_seat, legal or [], selected, ambiguous_ends or [])

    def _draw_end_markers(self, state: GameState, ambiguous_ends: Sequence[int]) -> None:
        """Draw the two clickable end boxes for an ambiguous tile.

        The selected tile fits both open ends, so the player picks which value
        stays open by clicking one of these boxes; each shows the value that end
        currently exposes.
        """
        left, right = end_marker_rects(self.size)
        for rect, end in ((left, 0), (right, 1)):
            active = end in ambiguous_ends
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
            pygame.draw.rect(
                self.screen, ACTING if active else TILE_EDGE, rect, width=3 if active else 2,
                border_radius=8,
            )
            value = state.ends[end] if end < len(state.ends) else "?"
            colour = ACTING if active else MUTED
            self.screen.blit(
                self.small.render(f"end {end} = {value}", True, colour), (rect.x + 12, rect.y + 8)
            )
            self.screen.blit(
                self.small.render("click to attach here" if active else "not available", True,
                                  TEXT if active else MUTED),
                (rect.x + 12, rect.y + 30),
            )

    def _draw_board(self, state: GameState, acting_seat: int | None) -> None:
        if not state.table:
            text = self.big.render("table empty — the leader opens", True, MUTED)
            self.screen.blit(text, (40, 180))
            return
        x, y = 40, 200
        for tile in state.table:
            a, b = tile_values(tile)
            crosswise = a == b
            rect = draw_tile(self.screen, tile, (x, y), crosswise=crosswise)
            x += rect.width + 4
            if x > self.size[0] - PANEL_W - 20:  # wrap before running under the panel
                x, y = 40, y + TILE_LONG + 12
        if acting_seat is not None:
            label = self.small.render(
                f"seat {acting_seat} (team {TEAM_OF_SEAT[acting_seat]}) to play",
                True, ACTING,
            )
            self.screen.blit(label, (40, y + TILE_LONG + 20))

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
            marker = "->" if seat == acting_seat else "  "
            partner = PARTNER_OF_SEAT[seat]
            who = "you" if seat == human_seat else (seat_labels or {}).get(seat, "")
            lines.append(
                (
                    f"{marker} seat {seat}  {state.hand_size(seat)}t {state.pips(seat)}p"
                    f"  {who} (p{partner})",
                    ACTING if seat == acting_seat else MUTED,
                )
            )
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
        """Map a click on one of the two end boxes to that end's index.

        The boxes are the ones :func:`end_marker_rects` draws, so the target the
        player sees is the target the click tests.
        """
        if len(ends) < 2:
            return None
        left, right = end_marker_rects(self.size)
        if left.collidepoint(pos):
            return 0
        if right.collidepoint(pos):
            return 1
        return None


def end_marker_rects(size: tuple[int, int]) -> tuple[pygame.Rect, pygame.Rect]:
    """The two clickable 'attach to this end' boxes (end 0 left, end 1 right).

    They sit above the board rows so they never cover the tiles, and they are the
    hit boxes :meth:`Renderer.end_at` tests, so what is drawn is exactly what is
    clickable.
    """
    left = pygame.Rect(20, END_MARKER_Y, END_MARKER_W, END_MARKER_H)
    right = pygame.Rect(
        size[0] - PANEL_W - 20 - END_MARKER_W, END_MARKER_Y, END_MARKER_W, END_MARKER_H
    )
    return left, right


def format_action(tile: int, end: int | None = None) -> str:
    """Human-readable action text used in the info panel."""
    a, b = tile_values(tile)
    suffix = "" if end is None else f" on end {end}"
    return f"[{a}|{b}] ({tile_pips(tile)} pips){suffix}"


def describe_hand(hand: Iterable[int]) -> str:
    """Compact description of a hand, used by the CLI banners."""
    return " ".join(format_action(tile) for tile in sorted(hand, reverse=True)) or "(empty)"
