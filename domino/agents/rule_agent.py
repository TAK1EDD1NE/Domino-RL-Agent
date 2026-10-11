"""Documented rule-based baseline agent (spec section 10).

A transparent, hand-tuned evaluation function over the legal placements, in the
spirit of classic domino play. Every component is a named weight in
:class:`RuleWeights` so the report can show the exact decision-making strategy.

The bot maintains the same public-information opponent model the RL agent gets:
``live[v]`` is the number of tiles *not in its own hand and not on the track*
carrying pip value ``v`` (each pip value is carried by exactly 7 tiles, so
``live[v] = 7 - (my v-tiles) - (v-tiles already played)``). A move is scored by
the spec's term table; toward a block (``sum(live) <= 12`` or ``pass_streak >=
2``) the "kills a value" and "block economics" terms are doubled. Ties are
broken by the lowest action index, so the agent is fully deterministic; numpy
and the engine only — no torch, no sb3, no pygame.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..engine import OPPONENTS_OF_SEAT, PARTNER_OF_SEAT, GameState
from ..masks import action_index
from ..tiles import ALL_TILE_IDS, DOUBLE_IDS, NUM_TILES, TOTAL_PIPS, tile_pips, tile_values

# Block-awareness thresholds (spec section 10).
BLOCK_LIVE_THRESHOLD = 12      # sum(live) <= 12 -> the deal is likely to block
BLOCK_PASS_STREAK = 2          # pass_streak >= 2 -> the deal is likely to block
SHORT_SUPPLY = 2               # live[v] <= 2 -> the other seats are thin on v


@dataclass
class RuleWeights:
    """Named weights of the spec section 10 scoring table (defaults as specified)."""

    pips_shed: float = 1.0          # shed heavy tiles: they are what you lose on a block
    kills_value: float = 3.0        # play so an open end value is dead table-wide
    opp_short_end: float = 2.0      # leave an end value the opponents are short of
    keep_doubles: float = 0.5       # do not spend a double unless it pays off
    protect_partner: float = 1.5    # do not open a value the partner is short of
    block_economics: float = 2.0    # become the lowest-pip player before a block
    duplicate_lead: float = 1.5     # lead a value you still hold a duplicate of


def _value_counts(tile_ids) -> list[int]:
    """Count how many of ``tile_ids`` carry each pip value (a double counts once)."""
    counts = [0] * 7
    for tid in tile_ids:
        a, b = tile_values(tid)
        counts[a] += 1
        if b != a:
            counts[b] += 1
    return counts


def _live_counts(state: GameState, seat: int) -> list[int]:
    """Return the ``live[v]`` opponent model of ``seat`` (spec section 10).

    ``live[v]`` counts tiles outside ``seat``'s hand and the table, i.e. the
    tiles held by the other three seats — public information, never a hand.
    """
    seen = set(state.hands[seat]) | set(state.table)
    live = [0] * 7
    for tid in ALL_TILE_IDS:
        if tid in seen:
            continue
        a, b = tile_values(tid)
        live[a] += 1
        if b != a:
            live[b] += 1
    return live


class RuleAgent:
    """Hand-tuned, fully deterministic evaluation-function baseline."""

    name = "rule"

    def __init__(self, weights: RuleWeights | None = None, seed: int | None = None) -> None:
        self.weights = weights if weights is not None else RuleWeights()
        # Decisions are deterministic (ties by lowest action index); the seed is
        # stored for API symmetry with the other agents and future tie-break use.
        self.seed = seed

    # ---- public interface ---------------------------------------------
    def act(self, state: GameState, seat: int) -> int:
        """Return the best-scoring legal action index for ``seat`` (ties: lowest index)."""
        if not state.legal_moves(seat):
            raise ValueError(f"RuleAgent.act called for seat {seat} with no legal move (forced pass)")
        return self.explain(state, seat)[0][0]

    def explain(self, state: GameState, seat: int) -> list[tuple[int, float]]:
        """Return ``(action_index, score)`` for every legal action, best first."""
        entries = [
            (action_index(tid, end), self.score_move(state, seat, tid, end))
            for tid, end in state.legal_moves(seat)
        ]
        entries.sort(key=lambda item: (-item[1], item[0]))
        return entries

    def score_move(self, state: GameState, seat: int, tid: int, end: int) -> float:
        """Score playing ``tid`` on end ``end``; finite for every legal placement."""
        weights = self.weights

        # Block-awareness is a property of the current position, not of the
        # candidate move (playing resets the pass streak), so read it off `state`.
        live_now = _live_counts(state, seat)
        block_close = (
            sum(live_now) <= BLOCK_LIVE_THRESHOLD
            or state.pass_streak >= BLOCK_PASS_STREAK
        )
        kills_weight = weights.kills_value * (2.0 if block_close else 1.0)

        # Evaluate the hypothetical position after the placement.
        after = state.clone()
        after.turn = seat                       # allow scoring from any seat
        after.play(seat, tid, end)
        live = _live_counts(after, seat)
        end_values = set(after.ends)

        score = weights.pips_shed * tile_pips(tid)

        # Kills a value: an open end whose value no live tile can match.
        score += kills_weight * sum(1 for value in after.ends if live[value] == 0)

        # Opponents short of an open end / partner short of an open end.
        partner = PARTNER_OF_SEAT[seat]
        opponents = OPPONENTS_OF_SEAT[seat]
        shown_partner = _value_counts(after.played_by[partner])
        shown_opponents = [_value_counts(after.played_by[o]) for o in opponents]
        opp_short_ends = 0
        partner_short_ends = 0
        for value in end_values:
            if live[value] <= SHORT_SUPPLY and all(shown[value] == 0 for shown in shown_opponents):
                opp_short_ends += 1
            if live[value] <= SHORT_SUPPLY and shown_partner[value] == 0:
                partner_short_ends += 1
        score += weights.opp_short_end * opp_short_ends
        score -= weights.protect_partner * partner_short_ends

        # Keeps my doubles in hand.
        if tid not in DOUBLE_IDS:
            score += weights.keep_doubles

        # Block economics: be lighter than the lightest estimated opponent.
        score += weights.block_economics * (2.0 if block_close else 1.0) * self._block_term(
            state, after, seat, block_close
        )

        # Leading: prefer a value you still hold a duplicate of (end-switcher).
        if not state.table:
            played_values = set(tile_values(tid))
            if any(played_values & set(tile_values(other)) for other in after.hands[seat]):
                score += weights.duplicate_lead

        return float(score)

    # ---- internals -----------------------------------------------------
    def _block_term(self, state: GameState, after: GameState, seat: int, block_close: bool) -> float:
        """Raw, unweighted "reduce my expected loss on a block" term in ``[-1, 1]``.

        The term estimates the acting seat's pip position against the lightest
        opponent it can infer from public information, and is a property of the
        *resulting* position. Losing a blocked deal is only a real risk once a
        block is a prospect, so the term contributes only when the caller flags
        ``block_close`` (``sum(live) <= 12`` or ``pass_streak >= 2``, spec
        section 10); the caller then doubles the weight.
        """
        if not block_close:
            return 0.0
        return self._block_advantage(after, seat)

    @staticmethod
    def _block_advantage(after: GameState, seat: int) -> float:
        """Signed, bounded ``[-1, 1]`` measure of being the lightest-pip player.

        The pips in the three other hands are public (``TOTAL_PIPS`` minus own
        hand minus table), so an opponent's pip total is estimated as the mean
        pips-per-tile in those hands times their hand size; the lightest of the
        two opponents is the block threat. ``+1`` means far lighter than that
        threat, ``-1`` far heavier, ``0`` even.
        """
        my_pips = after.pips(seat)
        others_tiles = NUM_TILES - len(after.table) - len(after.hands[seat])
        unseen_pips = TOTAL_PIPS - my_pips - after.table_pips()
        if others_tiles <= 0 or unseen_pips <= 0:
            estimate_lightest = 0.0
        else:
            per_tile = unseen_pips / others_tiles
            estimate_lightest = min(
                per_tile * len(after.hands[opponent]) for opponent in OPPONENTS_OF_SEAT[seat]
            )
        denominator = my_pips + estimate_lightest
        if denominator <= 0:
            return 0.0
        return 1.0 - 2.0 * (my_pips / denominator)
