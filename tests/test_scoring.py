"""House-rule scoring, deal outcomes, leaders, void deals and the match to 100."""
import numpy as np
import pytest

from domino.engine import GameState, Match, deal_outcome, new_deal, next_leader, redeal_after_void
from domino.tiles import tile_id


def mk(hands, table=(), ends=(), turn=0, leader=0, deal_no=2, scores=(0, 0), pass_streak=0):
    return GameState(hands=[sorted(h) for h in hands], rng=np.random.default_rng(0),
                     table=list(table), ends=list(ends), turn=turn, leader=leader,
                     deal_no=deal_no, scores=list(scores), pass_streak=pass_streak)


def test_emptied_hand_banks_both_opponents():
    # seat 0 (team 0) has finished; opponents are seats 1 and 3
    s = mk(hands=[[], [tile_id(6, 6)], [], [tile_id(4, 4)]],
           table=[tile_id(1, 1)], ends=[1])
    out = deal_outcome(s)
    assert (out.reason, out.winner_team, out.points, out.winning_seat) == ("emptied", 0, 12 + 8, 0)


def test_block_unique_minimum_wins_opposing_pair_only():
    s = mk(hands=[[tile_id(0, 0)], [tile_id(6, 6)], [tile_id(0, 1)], [tile_id(6, 5)]],
           table=[tile_id(2, 2)], ends=[2], pass_streak=4)
    out = deal_outcome(s)
    # seat 0 has 0 pips (unique minimum, team 0) -> team 0 banks pips of seats 1 and 3 only
    assert (out.reason, out.winner_team, out.points, out.winning_seat) == ("blocked", 0, 12 + 11, 0)


def test_block_tie_inside_one_team_still_scores():
    # seats 0 and 2 both hold 4 pips and are team-mates -> team 0 banks pips of seats 1 and 3
    s = mk(hands=[[tile_id(2, 2)], [tile_id(6, 6)], [tile_id(1, 3)], [tile_id(6, 5)]],
           table=[tile_id(1, 1)], ends=[1], pass_streak=4)
    out = deal_outcome(s)
    assert (out.reason, out.winner_team, out.points) == ("blocked", 0, 12 + 11)


def test_block_tie_across_teams_is_void():
    # seats 0 and 1 both hold 4 pips and are on different teams -> void deal
    s = mk(hands=[[tile_id(2, 2)], [tile_id(1, 3)], [tile_id(4, 4)], [tile_id(6, 6)]],
           table=[tile_id(0, 0)], ends=[0], pass_streak=4)
    out = deal_outcome(s)
    assert (out.reason, out.winner_team, out.points, out.void) == ("void_tie", None, 0, True)


def test_void_deal_redeals_with_six_six_leader_and_keeps_scores():
    s = mk(hands=[[tile_id(2, 2)], [tile_id(1, 3)], [tile_id(4, 4)], [tile_id(6, 6)]],
           table=[tile_id(0, 0)], ends=[0], pass_streak=4, scores=(37, 44), deal_no=3)
    out = deal_outcome(s)
    assert next_leader(s, out) is None
    nxt = redeal_after_void(s)
    assert nxt.scores == [37, 44] and nxt.deal_no == 4
    assert nxt.leader == nxt.turn and tile_id(6, 6) in nxt.hands[nxt.leader]


def test_next_leader_is_the_winner():
    s = mk(hands=[[], [tile_id(6, 6)], [tile_id(1, 1)], [tile_id(2, 2)]], table=[tile_id(0, 0)], ends=[0])
    out = deal_outcome(s)
    assert next_leader(s, out) == 0


def test_match_reaches_100_and_stops():
    m = Match(target=100, seed=5)
    m.reset()
    while not m.finished:
        m.play_deal({seat: (lambda st, s=seat: st.legal_actions(s)[0]) for seat in range(4)})
    assert m.finished and max(m.scores) >= 100
    assert m.winner_team in (0, 1)
    assert m.scores[0] + m.scores[1] == sum(o.points for o in m.deals if not o.void)
    for o in m.deals:
        if o.void:
            assert o.points == 0
    assert all(o.points > 0 for o in m.deals if not o.void)
