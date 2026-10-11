"""Interactive pygame mode: a human plays one seat against AI seats.

The human's team-mate is by default the trained network, and the two opposing
seats are configurable (trained network, the rule-based baseline, or random
play). The engine owns every rule; this module only routes seats to agents, maps
clicks to ``(tile, end)`` moves and draws the table.

Clicking a tile that can be attached to either open end selects it and asks for a
second click on the end to attach it to -- that choice is a real move in this
game (it decides which value stays open), so it is never made for the player.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import pygame

from ..agents import Agent, PolicyAgent, RandomAgent, RuleAgent
from ..agents.pool import load_policy
from ..engine import (
    GameState,
    TEAM_OF_SEAT,
    deal_outcome,
    new_deal,
    next_leader,
    redeal_after_void,
)
from ..masks import decode_action
from .render import Renderer, format_action


@dataclass
class PlayConfig:
    """Everything the interactive mode needs to start."""

    human_seat: int = 1
    partner: str = "rl"
    opponents: str = "rl"
    model: str = "models/checkpoints/latest.zip"
    seed: int = 0
    match_to: int = 100
    fps: int = 30
    autoplay: bool = False
    quiet: bool = False
    last_actions: list[str] = field(default_factory=list)


def _make_agent(kind: str, model: object | None, seed: int) -> Agent:
    if kind == "rl":
        if model is None:
            raise ValueError("an 'rl' seat needs a trained model; pass --model")
        return PolicyAgent(model, deterministic=True, name="rl")
    if kind == "rule":
        return RuleAgent(seed=seed)
    if kind == "random":
        return RandomAgent(seed=seed)
    raise ValueError(f"unknown agent kind {kind!r}")


def _build_agents(cfg: PlayConfig) -> dict[int, Agent]:
    """Seats other than the human's: the partner and the two opponents.

    With ``autoplay`` the human's seat also gets an agent, so the demo can run
    itself (used by the live demo and by the headless smoke test).
    """
    partner_seat = (cfg.human_seat + 2) % 4
    opponents = [seat for seat in range(4) if seat not in (cfg.human_seat, partner_seat)]
    model = None
    if cfg.partner == "rl" or cfg.opponents == "rl":
        model = load_policy(cfg.model)
    agents = {partner_seat: _make_agent(cfg.partner, model, cfg.seed + 10)}
    for offset, seat in enumerate(opponents):
        agents[seat] = _make_agent(cfg.opponents, model, cfg.seed + 20 + offset)
    if cfg.autoplay:
        agents[cfg.human_seat] = _make_agent(cfg.opponents, model, cfg.seed + 30)
    return agents


def _placements(state: GameState, seat: int, tile: int) -> list[int]:
    """End indices on which ``tile`` may be played by ``seat``."""
    return [end for (candidate, end) in state.legal_moves(seat) if candidate == tile]


def run_play(cfg: PlayConfig | None = None, *, max_frames: int | None = None) -> int:
    """Run the interactive mode until the window is closed.

    ``max_frames`` is a test hook: when set, the loop returns after that many
    frames (used by the headless smoke test with ``SDL_VIDEODRIVER=dummy``).
    """
    cfg = cfg or PlayConfig()
    agents = _build_agents(cfg)
    renderer = Renderer(title=f"Domino 2v2 — you are seat {cfg.human_seat}")
    rng = np.random.default_rng(cfg.seed)
    scores = [0, 0]
    state = new_deal(deal_no=1, rng=rng, lead_mode="six_six", scores=scores)
    banner = "your turn" if state.turn == cfg.human_seat else "AI thinking"
    pending_tile: int | None = None
    frames = 0
    running = True

    while running:
        legal = state.legal_actions(cfg.human_seat) if state.turn == cfg.human_seat else []
        ambiguous = [end for end in _placements(state, cfg.human_seat, pending_tile or -1)]
        renderer.draw(
            state,
            acting_seat=state.turn,
            human_seat=cfg.human_seat,
            legal=legal,
            banner=banner,
            last_actions=cfg.last_actions,
            selected=pending_tile,
            ambiguous_ends=ambiguous if pending_tile is not None else [],
        )

        for event in renderer.poll():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                cfg.autoplay = not cfg.autoplay
                banner = f"autoplay {'on' if cfg.autoplay else 'off'}"
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                scores = [0, 0]
                state = new_deal(deal_no=1, rng=rng, lead_mode="six_six", scores=scores)
                pending_tile = None
                banner = "new deal"
            elif event.type == pygame.MOUSEBUTTONDOWN and state.turn == cfg.human_seat:
                tile = renderer.tile_at(event.pos)
                if tile is None:
                    continue
                placements = _placements(state, cfg.human_seat, tile)
                if not placements:
                    banner = f"{format_action(tile)} cannot be played on either end"
                    pending_tile = None
                elif len(placements) == 1:
                    state.play(cfg.human_seat, tile, placements[0])
                    cfg.last_actions.append(f"you: {format_action(tile, placements[0])}")
                    pending_tile = None
                    banner = "played"
                elif pending_tile == tile:
                    end = renderer.end_at(event.pos, state, state.ends)
                    if end in placements:
                        state.play(cfg.human_seat, tile, end)
                        cfg.last_actions.append(f"you: {format_action(tile, end)}")
                        pending_tile = None
                        banner = "played"
                else:
                    pending_tile = tile
                    banner = f"{format_action(tile)} fits both ends — click the end to use"

        if running and (state.turn != cfg.human_seat or cfg.autoplay):
            seat = state.turn
            if state.must_pass(seat):
                state.pass_turn(seat)
                cfg.last_actions.append(f"seat {seat}: pass (stuck)")
            else:
                agent = agents[seat]
                action = int(agent.act(state, seat))
                tile, end = decode_action(action)
                state.play(seat, tile, end)
                cfg.last_actions.append(f"seat {seat} ({agent.name}): {format_action(tile, end)}")

        if state.is_terminal():
            outcome = deal_outcome(state)
            if outcome.void:
                banner = "void deal (fewest-pips tie across teams) — redealt"
                state = redeal_after_void(state)
            else:
                scores[outcome.winner_team] += outcome.points
                winner = "your team" if outcome.winner_team == TEAM_OF_SEAT[cfg.human_seat] else "the AI team"
                banner = f"{winner} takes {outcome.points} points ({outcome.reason})"
                if max(scores) >= cfg.match_to:
                    banner = f"match over — {'your team' if scores.index(max(scores)) == TEAM_OF_SEAT[cfg.human_seat] else 'the AI team'} wins {scores[0]}-{scores[1]} (press n for a new match)"
                    scores = [0, 0]
                leader = next_leader(state, outcome)
                state = new_deal(
                    deal_no=state.deal_no + 1,
                    rng=rng,
                    lead_mode="six_six" if leader is None else "winner",
                    leader=leader,
                    scores=scores,
                )
            pending_tile = None

        renderer.tick(cfg.fps)
        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False
        if max_frames is None:
            time.sleep(0)

    renderer.close()
    return 0
