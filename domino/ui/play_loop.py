"""Interactive pygame mode: a human plays one seat against AI seats.

The human's team-mate and the two opposing seats are configurable (trained
network, the rule-based baseline, or random play) and can each be given a
*different* agent through ``PlayConfig.seat_agents``, so the three AI seats can
hold, say, two checkpoints and the rule bot. The engine owns every rule; this
module only routes seats to agents, maps clicks to ``(tile, end)`` moves and
draws the table.

Clicking a tile that can be attached to either open end selects it and asks for a
second click on the end to attach it to -- that choice is a real move in this
game (it decides which value stays open), so it is never made for the player.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import time
from typing import Any

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


#: Agent kinds a seat can be assigned.
AGENT_KINDS: tuple[str, ...] = ("rl", "rule", "random")


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
    #: Per-seat overrides, ``{seat: "KIND[:checkpoint]"}``, so the three AI seats
    #: can be three *different* agents (e.g. two checkpoints and the rule bot).
    seat_agents: dict[int, str] = field(default_factory=dict)
    last_actions: list[str] = field(default_factory=list)


def parse_seat_agent(spec: str) -> tuple[str, str | None]:
    """Split a ``KIND[:CHECKPOINT]`` seat spec into its kind and optional path.

    Only the *first* colon separates the two parts, so a Windows checkpoint path
    (``rl:C:/models/x.zip``) survives untouched. A checkpoint is only meaningful
    for ``rl``; ``rule``/``random`` ignore one if it is given.
    """
    kind, _, path = str(spec).partition(":")
    kind = kind.strip().lower()
    if kind not in AGENT_KINDS:
        raise ValueError(
            f"unknown agent kind {kind!r} in seat spec {spec!r}; expected one of {AGENT_KINDS}"
        )
    return kind, (path.strip() or None)


def agent_label(kind: str, path: str | None = None) -> str:
    """Short HUD name for an agent, derived from its checkpoint name.

    ``.../ppo_15002766_steps.zip`` reads as ``rl-15.0M`` so the panel tells you
    which training stage you are facing, and ``latest.zip`` as ``rl-latest``.
    """
    if kind != "rl":
        return kind
    if not path:
        return "rl"
    match = re.search(r"ppo_(\d+)_steps", os.path.basename(path))
    if match:
        return f"rl-{int(match.group(1)) / 1e6:.1f}M"
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"rl-{stem[:12]}"


def _make_agent(kind: str, model: Any, seed: int, name: str | None = None) -> Agent:
    if kind == "rl":
        if model is None:
            raise ValueError("an 'rl' seat needs a trained model; pass --model")
        return PolicyAgent(model, deterministic=True, name=name or "rl")
    if kind == "rule":
        return RuleAgent(seed=seed)
    if kind == "random":
        return RandomAgent(seed=seed)
    raise ValueError(f"unknown agent kind {kind!r}")


def _seat_specs(cfg: PlayConfig) -> dict[int, tuple[str, str | None]]:
    """Resolve every AI seat to a ``(kind, checkpoint)`` pair.

    The defaults are ``cfg.partner`` for the teammate and ``cfg.opponents`` for
    both opponents (with ``autoplay`` also filling the human's seat);
    ``cfg.seat_agents`` overrides individual seats.
    """
    partner_seat = (cfg.human_seat + 2) % 4
    opponents = [seat for seat in range(4) if seat not in (cfg.human_seat, partner_seat)]
    specs: dict[int, tuple[str, str | None]] = {partner_seat: (cfg.partner, cfg.model)}
    for seat in opponents:
        specs[seat] = (cfg.opponents, cfg.model if cfg.opponents == "rl" else None)
    if cfg.autoplay:
        specs[cfg.human_seat] = (cfg.opponents, cfg.model if cfg.opponents == "rl" else None)
    for seat, spec in cfg.seat_agents.items():
        seat = int(seat)
        if seat not in specs:
            raise ValueError(
                f"seat {seat} cannot be assigned an agent: it is the human seat "
                f"({cfg.human_seat}); only seats {sorted(specs)} are played by the AI"
            )
        specs[seat] = parse_seat_agent(spec)
    return specs


def _build_agents(cfg: PlayConfig) -> dict[int, Agent]:
    """Seats other than the human's: the partner and the two opponents.

    Every ``rl`` seat loads its own checkpoint (each path loaded once and shared
    between seats that name it). With ``autoplay`` the human's seat also gets an
    agent, so the demo can run itself.
    """
    partner_seat = (cfg.human_seat + 2) % 4
    opponents = [seat for seat in range(4) if seat not in (cfg.human_seat, partner_seat)]
    models: dict[str, Any] = {}
    agents: dict[int, Agent] = {}
    for seat, (kind, path) in sorted(_seat_specs(cfg).items()):
        if seat == partner_seat:
            seed = cfg.seed + 10
        elif seat == cfg.human_seat:
            seed = cfg.seed + 30
        else:
            seed = cfg.seed + 20 + opponents.index(seat)
        model = None
        if kind == "rl":
            path = path or cfg.model
            if path not in models:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"checkpoint for seat {seat} not found: {path}")
                models[path] = load_policy(path)
            model = models[path]
        agents[seat] = _make_agent(kind, model, seed, agent_label(kind, path))
    return agents


def seat_labels(agents: dict[int, Agent], human_seat: int | None = None) -> dict[int, str]:
    """HUD labels per seat: each agent's name, and 'you' for the human's seat.

    Pass ``human_seat=None`` in autoplay mode, where the human's seat is played
    by an agent too.
    """
    labels = {seat: agent.name for seat, agent in agents.items()}
    if human_seat is not None:
        labels[human_seat] = "you"
    return labels


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
    labels = seat_labels(agents, None if cfg.autoplay else cfg.human_seat)
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
            seat_labels=labels,
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
