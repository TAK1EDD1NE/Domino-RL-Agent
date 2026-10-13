"""play.py — play a 2v2 domino match against the trained agent (interactive mode).

You take one seat; your team-mate is by default the trained network and the two
opposing seats are configurable. Clicking a tile that fits both open ends asks
for a second click on the end to attach it to, because that choice decides which
value stays open.

Each of the three AI seats can be given its own agent with a repeatable
``--agent SEAT=KIND[:CHECKPOINT]``, so you can face, for example, the 15M network
on one side and the rule bot on the other while the 4M network partners you.
``--ai-delay`` slows the AI down so you can follow the deal. The window is a
table: your tiles are face up at the bottom, the other three seats sit around it
and show what they hold (``--hide-hands`` keeps them face down), and a finished
hand is reported -- who took the points, why, what each side was left holding --
with buttons to replay that hand move by move or deal the next one.

Controls: click a tile (then the framed end if it fits both), H shows/hides the
other hands, R replays the hand (LEFT/RIGHT step, P plays it, R or ESC leaves),
SPACE continues after a report, A autoplay, N new match, Q quits.

Examples
--------
    uv run python play.py --partner rl --opponents rl
    uv run python play.py --partner rule --opponents random --human-seat 3
    uv run python play.py --ai-delay 1.5 --agent 2=rule
    uv run python play.py --agent 3=rl:models/checkpoints/ppo_4001502_steps.zip \\
        --agent 0=rl:models/checkpoints/ppo_15002766_steps.zip --agent 2=rule
"""
from __future__ import annotations

import argparse

from domino.ui.play_loop import PlayConfig, parse_seat_agent, run_play


def _seat_agents(items: list[str]) -> dict[int, str]:
    """Turn ``["0=rule", "2=rl:path.zip"]`` into ``{0: "rule", 2: "rl:path.zip"}``."""
    parsed: dict[int, str] = {}
    for item in items:
        seat_text, sep, spec = item.partition("=")
        if not sep or not spec.strip():
            raise SystemExit(f"--agent expects SEAT=KIND[:CHECKPOINT], got {item!r}")
        try:
            seat = int(seat_text)
        except ValueError:
            raise SystemExit(f"--agent seat must be an integer 0-3, got {seat_text!r}")
        if not 0 <= seat <= 3:
            raise SystemExit(f"--agent seat must be between 0 and 3, got {seat}")
        try:
            parse_seat_agent(spec)
        except ValueError as exc:  # fail before the window opens
            raise SystemExit(str(exc))
        parsed[seat] = spec.strip()
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """CLI definition for the interactive mode."""
    parser = argparse.ArgumentParser(
        description="Play 2v2 domino against the trained agent (Pygame window).",
    )
    parser.add_argument("--human-seat", type=int, default=1, dest="human_seat",
                        help="seat you play (0-3); seats 0 and 2 are one team")
    parser.add_argument("--partner", choices=("rl", "rule", "random"), default="rl",
                        help="your team-mate")
    parser.add_argument("--opponents", choices=("rl", "rule", "random"), default="rl",
                        help="the two opposing seats")
    parser.add_argument("--agent", action="append", default=[], dest="seat_agents",
                        metavar="SEAT=KIND[:CHECKPOINT]",
                        help="give one AI seat its own agent, e.g. --agent 0=rule or "
                             "--agent 2=rl:models/checkpoints/ppo_4001502_steps.zip "
                             "(repeatable; overrides --partner/--opponents for that seat)")
    parser.add_argument("--model", default="models/checkpoints/latest.zip",
                        help="checkpoint used by every 'rl' seat without its own path")
    parser.add_argument("--seed", type=int, default=0, help="deal seed")
    parser.add_argument("--match-to", type=int, default=100, dest="match_to",
                        help="match target in points")
    parser.add_argument("--fps", type=int, default=30, help="frame rate")
    parser.add_argument("--ai-delay", type=float, default=1.2, dest="ai_delay",
                        help="seconds the AI waits between its moves (0 = as fast as possible)")
    parser.add_argument("--hide-hands", action="store_false", dest="reveal_hands",
                        help="keep the other seats face down instead of showing their tiles "
                             "(they are shown by default; H toggles while playing)")
    parser.add_argument("--autoplay", action="store_true",
                        help="let the AI play your seat too (demo mode)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the CLI and by the tests."""
    args = build_parser().parse_args(argv)
    if not 0 <= args.human_seat <= 3:
        raise SystemExit("--human-seat must be between 0 and 3")
    seat_agents = _seat_agents(args.seat_agents)
    if args.human_seat in seat_agents:
        raise SystemExit(
            f"--agent cannot assign seat {args.human_seat}: that is your own seat"
        )
    cfg = PlayConfig(
        human_seat=args.human_seat,
        partner=args.partner,
        opponents=args.opponents,
        model=args.model,
        seed=args.seed,
        match_to=args.match_to,
        fps=args.fps,
        ai_delay=args.ai_delay,
        autoplay=args.autoplay,
        seat_agents=seat_agents,
        reveal_hands=args.reveal_hands,
    )
    return run_play(cfg)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
