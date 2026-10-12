"""Replay mode: watch one deal played by the trained agent (or the rule bot).

Step with space, toggle continuous playback with p, change speed with +/- and
start a fresh deal with n. When a trained network is in play, an optional overlay
shows its policy over the legal moves and the value estimate for the acting seat;
the overlay degrades to nothing if the installed sb3-contrib does not expose the
masked distribution (that is a display nicety, never a correctness path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import pygame

from ..agents import Agent, PolicyAgent, RandomAgent, RuleAgent
from ..agents.pool import load_policy
from ..engine import GameState, deal_outcome, new_deal, redeal_after_void
from ..masks import action_mask, decode_action
from ..obs import encode
from .render import Renderer, format_action


@dataclass
class ObserveConfig:
    """Everything the replay mode needs to start."""

    agent: str = "rl"
    model: str = "models/checkpoints/latest.zip"
    seed: int = 0
    fps: int = 4
    overlay: bool = True
    deals: int = 1
    quiet: bool = False
    last_actions: list[str] = field(default_factory=list)


def _make_agent(kind: str, model: object | None, seed: int) -> Agent:
    if kind == "rl":
        if model is None:
            raise ValueError("agent 'rl' needs a trained model; pass --model")
        return PolicyAgent(model, deterministic=True, name="rl")
    if kind == "rule":
        return RuleAgent(seed=seed)
    if kind == "random":
        return RandomAgent(seed=seed)
    raise ValueError(f"unknown agent kind {kind!r}")


def action_probabilities(agent: Agent, state: GameState, seat: int) -> list[float] | None:
    """Masked action probabilities of ``agent`` for ``seat``, or ``None``.

    Best-effort: returns ``None`` for non-policy agents and for any sb3-contrib
    version whose masked distribution is not reachable this way.
    """
    model = getattr(agent, "model", None)
    if model is None:
        return None
    try:
        import torch

        obs = torch.as_tensor(encode(state, seat), dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(action_mask(state, seat)).unsqueeze(0)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs)
            categorical = distribution.distribution
            categorical.apply_masking(mask)
            probs = categorical.probs.detach().cpu().numpy().reshape(-1)
        return [float(value) for value in probs]
    except Exception:  # noqa: BLE001 - display-only path, never fatal
        return None


def value_estimate(agent: Agent, state: GameState, seat: int) -> float | None:
    """Value-head estimate for ``seat`` (``None`` when unavailable)."""
    model = getattr(agent, "model", None)
    if model is None:
        return None
    try:
        import torch

        obs = torch.as_tensor(encode(state, seat), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, value, _ = model.policy.evaluate_actions(obs, torch.zeros(1, dtype=torch.long))
        return float(value.item())
    except Exception:  # noqa: BLE001 - display-only path, never fatal
        return None


def run_observe(cfg: ObserveConfig | None = None, *, max_frames: int | None = None) -> int:
    """Run the replay mode until the window is closed.

    ``max_frames`` is a test hook: when set, the loop returns after that many
    frames (used by the headless smoke test with ``SDL_VIDEODRIVER=dummy``).
    """
    cfg = cfg or ObserveConfig()
    model = load_policy(cfg.model) if cfg.agent == "rl" else None
    agents = {seat: _make_agent(cfg.agent, model, cfg.seed + seat) for seat in range(4)}
    renderer = Renderer(title=f"Domino 2v2 replay — {cfg.agent} agents")
    rng = np.random.default_rng(cfg.seed)
    state = new_deal(deal_no=1, rng=rng, lead_mode="six_six")
    paused = False
    frames = 0
    deals_done = 0
    running = True
    banner = "space: step   p: play/pause   +/-: speed   n: new deal   q: quit"

    while running:
        seat = state.turn
        overlay = None
        if cfg.overlay and not state.is_terminal():
            overlay = action_probabilities(agents[seat], state, seat)
        renderer.draw(
            state,
            acting_seat=seat,
            overlay=overlay,
            banner=banner,
            last_actions=cfg.last_actions,
        )

        for event in renderer.poll():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = not paused
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = True
                _step(state, agents, cfg)
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                cfg.fps = min(60, cfg.fps + 2)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_MINUS:
                cfg.fps = max(1, cfg.fps - 2)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                state = new_deal(deal_no=1, rng=rng, lead_mode="six_six")
                cfg.last_actions.clear()

        if running and not paused and not state.is_terminal():
            _step(state, agents, cfg)

        if state.is_terminal():
            outcome = deal_outcome(state)
            if outcome.void:
                banner = "void deal — redealt"
                state = redeal_after_void(state)
            else:
                deals_done += 1
                value = value_estimate(agents[0], state, 0)
                extra = "" if value is None else f"   V(seat 0) = {value:+.2f}"
                banner = (
                    f"deal {deals_done}: team {outcome.winner_team} +{outcome.points}"
                    f" ({outcome.reason}){extra}   n: new deal"
                )
                if deals_done >= cfg.deals:
                    paused = True

        renderer.tick(cfg.fps)
        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False
        if max_frames is None:
            time.sleep(0)

    renderer.close()
    return 0


def _step(state: GameState, agents: dict[int, Agent], cfg: ObserveConfig) -> None:
    """Advance the deal by one seat's decision (or a forced pass)."""
    seat = state.turn
    if state.must_pass(seat):
        state.pass_turn(seat)
        cfg.last_actions.append(f"seat {seat}: pass (stuck)")
        return
    action = int(agents[seat].act(state, seat))
    tile, end = decode_action(action)
    state.play(seat, tile, end)
    cfg.last_actions.append(f"seat {seat} ({agents[seat].name}): {format_action(tile, end)}")
