"""observe.py — watch a deal played by the trained agent (or the rule bot).

Replays one deal step by step in a Pygame window, optionally overlaying the
acting seat's masked action probabilities and its value estimate so the agent's
decisions can be explained during a live demo.

Examples
--------
    uv run python observe.py --agent rl --model models/checkpoints/latest.zip
    uv run python observe.py --agent rule --deals 3 --no-overlay
"""
from __future__ import annotations

import argparse

from domino.ui.observe_loop import ObserveConfig, run_observe


def build_parser() -> argparse.ArgumentParser:
    """CLI definition for the replay mode."""
    parser = argparse.ArgumentParser(
        description="Replay a deal played by the trained agent or the rule-based bot.",
    )
    parser.add_argument("--agent", choices=("rl", "rule", "random"), default="rl",
                        help="which agent plays all four seats")
    parser.add_argument("--model", default="models/checkpoints/latest.zip",
                        help="checkpoint used when --agent rl")
    parser.add_argument("--seed", type=int, default=0, help="deal seed")
    parser.add_argument("--fps", type=int, default=4,
                        help="replay speed when not stepping manually")
    parser.add_argument("--deals", type=int, default=1, help="deals to replay before pausing")
    parser.add_argument("--no-overlay", action="store_true", dest="no_overlay",
                        help="hide the policy/value overlay")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the CLI and by the tests."""
    args = build_parser().parse_args(argv)
    cfg = ObserveConfig(
        agent=args.agent,
        model=args.model,
        seed=args.seed,
        fps=args.fps,
        overlay=not args.no_overlay,
        deals=args.deals,
    )
    return run_observe(cfg)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
