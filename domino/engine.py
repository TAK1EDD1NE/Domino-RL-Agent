"""Pure-Python rules engine for Algerian 2v2 domino.

No gymnasium, no torch, no pygame: the engine is a deterministic state machine
so it can be unit-tested exhaustively and reused by the environment, the
evaluation scripts and the UI.

Rules implemented (see docs/specs/2026-10-11-domino-2v2-rl-design.md section 2):
28 tiles, 7 per seat, no boneyard, mandatory play, pass only when stuck, a deal
is blocked after four consecutive passes, doubles are ordinary line moves that
leave their end value unchanged. A move is a ``(tile, end)`` pair because a tile
matching both open ends can be oriented two ways and the two placements are
legally distinct unless the ends carry the same value.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .tiles import (
    ALL_TILE_IDS,
    NUM_TILES,
    SIX_SIX_ID,
    TOTAL_PIPS,
    tile_pips,
    tile_values,
)

SEATS = 4
TEAM_OF_SEAT = (0, 1, 0, 1)            # seats 0, 2 -> team 0 ; seats 1, 3 -> team 1
PARTNER_OF_SEAT = (2, 3, 0, 1)
OPPONENTS_OF_SEAT = ((1, 3), (0, 2), (1, 3), (0, 2))


@dataclass
class GameState:
    """A single deal in progress.

    Attributes
    ----------
    hands:
        Four sorted tile-id lists (seat -> tiles still held).
    rng:
        The explicit generator used for any reshuffle; never global state.
    table:
        Tiles in play order (the line).
    played_by:
        Per seat, the tile ids that seat played, in play order.
    ends:
        The open pip values of the line: index 0 = left, index 1 = right.
    turn:
        Seat to act next.
    leader:
        Seat that led (plays first) the current deal.
    pass_streak:
        Number of consecutive passes immediately before the next turn.
    last_pass:
        Whether each seat passed on its most recent turn.
    deal_no:
        1-based deal index within the match.
    scores:
        Running match score, per team (length 2).
    forced_leader_six_six:
        True while a deal still requires its leader to open with ``[6|6]``.
    """

    hands: list[list[int]]
    rng: np.random.Generator
    table: list[int] = field(default_factory=list)
    played_by: list[list[int]] = field(default_factory=lambda: [[], [], [], []])
    ends: list[int] = field(default_factory=list)
    turn: int = 0
    leader: int = 0
    pass_streak: int = 0
    last_pass: list[bool] = field(default_factory=lambda: [False] * SEATS)
    deal_no: int = 1
    scores: list[int] = field(default_factory=lambda: [0, 0])
    forced_leader_six_six: bool = False

    # ---- construction -------------------------------------------------
    def clone(self) -> "GameState":
        """Return an independent deep copy (RNG state included)."""
        return GameState(
            hands=[list(h) for h in self.hands],
            rng=copy.deepcopy(self.rng),
            table=list(self.table),
            played_by=[list(p) for p in self.played_by],
            ends=list(self.ends),
            turn=self.turn,
            leader=self.leader,
            pass_streak=self.pass_streak,
            last_pass=list(self.last_pass),
            deal_no=self.deal_no,
            scores=list(self.scores),
            forced_leader_six_six=self.forced_leader_six_six,
        )

    # ---- queries ------------------------------------------------------
    @staticmethod
    def team_of(seat: int) -> int:
        return TEAM_OF_SEAT[seat]

    def hand_size(self, seat: int) -> int:
        return len(self.hands[seat])

    def pips(self, seat: int) -> int:
        return sum(tile_pips(t) for t in self.hands[seat])

    def table_pips(self) -> int:
        return sum(tile_pips(t) for t in self.table)

    def unseen_tiles(self) -> set[int]:
        played = set(self.table) | {t for h in self.hands for t in h}
        return set(ALL_TILE_IDS) - played

    def legal_moves(self, seat: int) -> list[tuple[int, int]]:
        return legal_moves(self, seat)

    def legal_actions(self, seat: int) -> list[int]:
        return legal_actions(self, seat)

    def must_pass(self, seat: int) -> bool:
        return not legal_actions(self, seat)

    def blocks(self) -> bool:
        return self.pass_streak >= SEATS

    def is_terminal(self) -> bool:
        return self.blocks() or any(len(h) == 0 for h in self.hands) or len(self.table) == NUM_TILES

    # ---- placement helper ---------------------------------------------
    def _tile_placements(self, tid: int) -> list[tuple[int, list[int]]]:
        """Return deduplicated ``(end_index, new_ends)`` placements for ``tid``.

        The board must be non-empty. Placements that produce the same position
        (attaching to two ends that carry the same value) collapse to the lower
        end index, so no duplicate move is ever exposed.
        """
        a, b = tile_values(tid)
        candidates: list[tuple[int, list[int]]] = []
        for e in (0, 1):
            target = self.ends[e]
            if target != a and target != b:
                continue
            if a == b:
                new_value = a
            else:
                new_value = b if target == a else a
            new_ends = list(self.ends)
            new_ends[e] = new_value
            candidates.append((e, new_ends))
        placements: list[tuple[int, list[int]]] = []
        seen: set[tuple[int, ...]] = set()
        for e, new_ends in candidates:
            key = tuple(sorted(new_ends))
            if key in seen:
                continue
            seen.add(key)
            placements.append((e, new_ends))
        return placements

    # ---- transitions --------------------------------------------------
    def play(self, seat: int, tid: int, end: int = 0) -> None:
        """Play ``tid`` from ``seat``'s hand onto the line.

        ``end`` is the target end index (0 = left, 1 = right). It is honoured
        when the tile can attach to either open end; a tile that matches only
        one end ignores it. Playing a double on an end leaves that end's value
        unchanged.
        """
        if seat != self.turn:
            raise ValueError(f"seat {seat} cannot play on seat {self.turn}'s turn")
        if tid not in self.hands[seat]:
            raise ValueError(f"tile {tid} is not in seat {seat}'s hand")
        if tid not in self.legal_actions(seat):
            raise ValueError(f"tile {tid} is not playable on ends {self.ends}")
        if not self.table:                       # free lead: expose the tile's two values
            a, b = tile_values(tid)
            new_ends = [a, b]                     # a double leads to two equal ends, e.g. [6, 6]
        else:
            placements = self._tile_placements(tid)
            chosen = next((p for p in placements if p[0] == end), placements[0])
            new_ends = chosen[1]
        self.hands[seat].remove(tid)
        self.table.append(tid)
        self.played_by[seat].append(tid)
        self.ends = new_ends
        self.pass_streak = 0
        self.last_pass[seat] = False
        if len(self.table) == 1:
            self.forced_leader_six_six = False
        self.turn = (seat + 1) % SEATS

    def pass_turn(self, seat: int) -> None:
        """Record a forced pass; illegal when a legal tile exists in hand."""
        if seat != self.turn:
            raise ValueError("out of turn pass")
        if not self.must_pass(seat):
            raise ValueError("mandatory play violated: passing with a legal tile in hand")
        self.pass_streak += 1
        self.last_pass[seat] = True
        self.turn = (seat + 1) % SEATS


def legal_moves(state: GameState, seat: int) -> list[tuple[int, int]]:
    """Return the deduplicated ``(tile_id, end_index)`` placements for ``seat``."""
    hand = state.hands[seat]
    if not hand:
        return []
    if not state.table:                          # free lead
        if state.forced_leader_six_six and seat == state.turn and SIX_SIX_ID in hand:
            return [(SIX_SIX_ID, 0)]
        return [(t, 0) for t in sorted(hand)]
    moves: list[tuple[int, int]] = []
    for t in sorted(hand):
        for e, _ in state._tile_placements(t):
            moves.append((t, e))
    return moves


def legal_actions(state: GameState, seat: int) -> list[int]:
    """Tile-level convenience view: sorted unique playable tile ids."""
    return sorted({t for t, _ in legal_moves(state, seat)})


def new_deal(*, deal_no: int, rng: np.random.Generator, lead_mode: str = "six_six",
             leader: int | None = None, scores: list[int] | None = None) -> GameState:
    """Deal 28 tiles into four hands of seven and build the initial state.

    ``lead_mode="six_six"`` makes the ``[6|6]`` holder lead and obliges that
    forced opening; ``lead_mode="winner"`` starts with the explicit ``leader``
    seat and a free lead.
    """
    if lead_mode not in ("six_six", "winner"):
        raise ValueError(f"unknown lead_mode {lead_mode!r}")
    tiles = list(ALL_TILE_IDS)
    rng.shuffle(tiles)
    hands = [sorted(tiles[i * 7:(i + 1) * 7]) for i in range(SEATS)]
    if lead_mode == "six_six":
        leader_seat = next(i for i, h in enumerate(hands) if SIX_SIX_ID in h)
    else:
        if leader is None:
            raise ValueError("lead_mode='winner' requires an explicit leader seat")
        leader_seat = leader
    return GameState(
        hands=hands,
        rng=rng,
        turn=leader_seat,
        leader=leader_seat,
        deal_no=deal_no,
        scores=list(scores or [0, 0]),
        forced_leader_six_six=(lead_mode == "six_six"),
    )
