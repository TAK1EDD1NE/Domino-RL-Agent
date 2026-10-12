"""play.py — play a 2v2 domino match against the trained agent (interactive mode).

You take one seat; your team-mate is by default the trained network and the two
opposing seats are configurable. Clicking a tile that fits both open ends asks
for a second click on the end to attach it to, because that choice decides which
value stays open.

Examples
--------
    uv run python play.py --partner rl --opponents rl
    uv run python play.py --partner rule --opponents random --human-seat 3
"""
from __future__ import annotations

import argparse

from domino.ui.play_loop import PlayConfig, run_play


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
    parser.add_argument("--model", default="models/checkpoints/latest.zip",
                        help="checkpoint used by every 'rl' seat")
    parser.add_argument("--seed", type=int, default=0, help="deal seed")
    parser.add_argument("--match-to", type=int, default=100, dest="match_to",
                        help="match target in points")
    parser.add_argument("--fps", type=int, default=30, help="frame rate")
    parser.add_argument("--autoplay", action="store_true",
                        help="let the AI play your seat too (demo mode)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the CLI and by the tests."""
    args = build_parser().parse_args(argv)
    if not 0 <= args.human_seat <= 3:
        raise SystemExit("--human-seat must be between 0 and 3")
    cfg = PlayConfig(
        human_seat=args.human_seat,
        partner=args.partner,
        opponents=args.opponents,
        model=args.model,
        seed=args.seed,
        match_to=args.match_to,
        fps=args.fps,
        autoplay=args.autoplay,
    )
    return run_play(cfg)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
