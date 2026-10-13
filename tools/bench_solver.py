"""Benchmark the Phase-1 solver: cost of an exact solve at 7, 5 and 3 tiles per hand.

Positions are *real* deals: a fresh deal is dealt and then played with random legal
moves until every hand is down to at most ``size`` tiles, so the position is one the
game can actually reach (same pass/lack rules, same ends). The seating seat is the
one to move, exactly as at a real decision point.

Reports per size: positions solved, how many were solved exactly (a position that
hits ``--node-limit`` is reported, never silently truncated), nodes, nodes/second,
wall time per position, and the transposition-table hit rate.

Usage::

    uv run python tools/bench_solver.py --sizes 7,5,3 --positions 20
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import time
import tracemalloc

import numpy as np

from domino.engine import GameState, new_deal
from domino.pimc.solver import Solver


def random_position(rng: np.random.Generator, max_tiles: int) -> GameState:
    """Deal a real deal and play random legal moves until hands are <= ``max_tiles``."""
    state = new_deal(deal_no=1, rng=rng, lead_mode="six_six")
    guard = 0
    while max(len(hand) for hand in state.hands) > max_tiles and not state.is_terminal():
        guard += 1
        if guard > 500:                                     # pragma: no cover - defensive
            break
        seat = state.turn
        moves = state.legal_moves(seat)
        if not moves:
            state.pass_turn(seat)
            continue
        tid, end = moves[int(rng.integers(0, len(moves)))]
        state.play(seat, tid, end)
    return state


def bench_size(size: int, positions: int, seed: int, args: argparse.Namespace) -> dict:
    """Solve ``positions`` positions of the given size and return the aggregate row."""
    rng = np.random.default_rng(seed + size)
    solver = Solver(max_depth=None if not args.depth else args.depth,
                    tt_bits=args.tt_bits, node_limit=args.node_limit,
                    root_pruning=args.root_pruning)
    # Warm up the compiled kernel on a throwaway position: the first call of a fresh
    # process pays the numba compile (~10 s), which would otherwise be billed to the
    # first solve and make the reported nodes/s meaningless.
    warmup = GameState(
        hands=[[27, 11], [0, 13], [1, 7], [2, 18]],
        rng=np.random.default_rng(0), table=[4], ends=[1, 5], turn=0,
    )
    started = time.perf_counter()
    solver.solve(warmup)
    warmup_s = time.perf_counter() - started
    rows = []
    for index in range(positions):
        state = random_position(rng, size)
        if state.is_terminal():                             # pragma: no cover - guard
            continue
        result = solver.solve(state)
        rows.append({
            "size": size,
            "index": index,
            "deal_no": state.deal_no,
            "seat": int(state.turn),
            "value": int(result.value),
            "nodes": int(result.nodes),
            "seconds": float(result.elapsed_s),
            "exact": bool(result.exact),
            "complete": bool(result.complete),
            "tt_hits": int(result.tt_hits),
            "tt_probes": int(result.tt_probes),
            "leaf_evals": int(result.leaf_evals),
            "cutoffs": int(result.cutoffs),
            "moves": len(result.move_values),
            "tt_mb": solver.tt_bytes / 1e6,
        })
        if args.verbose:
            print(f"  size {size} #{index}: value={result.value} nodes={result.nodes} "
                  f"{result.elapsed_s * 1e3:.1f} ms complete={result.complete}")
    print(f"{'':>5}(warm-up solve on a 2-tile position: {warmup_s:.2f}s — this is the "
          f"first-call cost, excluded from the table)")
    return {"size": size, "rows": rows}


def summarize(batch: dict) -> str:
    """One fixed-width summary line for a batch of solves."""
    rows = batch["rows"]
    size = batch["size"]
    if not rows:
        return f"{size:>5}{'(no positions)':>40}"
    nodes = [r["nodes"] for r in rows]
    seconds = [r["seconds"] for r in rows]
    total_s = sum(seconds)
    total_nodes = sum(nodes)
    exact = sum(1 for r in rows if r["exact"])
    complete = sum(1 for r in rows if r["complete"])
    rate = total_nodes / total_s if total_s > 0 else float("inf")
    probes = sum(r["tt_probes"] for r in rows)
    hits = sum(r["tt_hits"] for r in rows)
    hit_rate = hits / probes if probes else 0.0
    return (
        f"{size:>5}{len(rows):>8}{complete:>9}{exact:>7}"
        f"{statistics.median(nodes):>14.0f}{max(nodes):>12}"
        f"{statistics.median(seconds) * 1e3:>12.1f}{total_s:>10.2f}"
        f"{rate:>12.0f}{hit_rate:>8.2f}"
    )


def run_sizes(sizes: list[int], args: argparse.Namespace, writer) -> list[dict]:
    """Benchmark every requested size, writing rows as they are produced."""
    all_rows: list[dict] = []
    for size in sizes:
        started = time.perf_counter()
        batch = bench_size(size, args.positions, args.seed, args)
        print(summarize(batch))
        if writer is not None:
            for row in batch["rows"]:
                writer.writerow(row)
        all_rows.extend(batch["rows"])
        print(f"{'':>5}({time.perf_counter() - started:.1f}s wall for {len(batch['rows'])} "
              f"position(s) of size {size})")
    return all_rows


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print (and optionally write) the table."""
    parser = argparse.ArgumentParser(description="Benchmark the exact domino solver.")
    parser.add_argument("--sizes", default="7,5,3", help="tiles per hand to solve at")
    parser.add_argument("--positions", type=int, default=20, help="positions per size")
    parser.add_argument("--seed", type=int, default=20261013, help="position seed")
    parser.add_argument("--tt-bits", type=int, default=20, help="transposition table size")
    parser.add_argument("--node-limit", type=int, default=None,
                        help="per-solve node budget (default: none)")
    parser.add_argument("--depth", type=int, default=None,
                        help="ply horizon instead of an exact solve")
    parser.add_argument("--root-pruning", action="store_true",
                        help="share one alpha-beta window across root moves")
    parser.add_argument("--out", default=None, help="CSV path for the per-position rows")
    parser.add_argument("--verbose", action="store_true", help="one line per position")
    args = parser.parse_args(argv)

    sizes = [int(chunk) for chunk in str(args.sizes).split(",") if chunk.strip()]
    header = (f"{'tiles':>5}{'solved':>8}{'complete':>9}{'exact':>7}{'nodes_med':>14}"
              f"{'nodes_max':>12}{'ms_med':>12}{'s_total':>10}{'nodes/s':>12}{'tt_hit':>8}")
    print(header)
    print("-" * len(header))
    tracemalloc.start()
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "size", "index", "deal_no", "seat", "value", "nodes", "seconds", "exact",
                "complete", "tt_hits", "tt_probes", "leaf_evals", "cutoffs", "moves", "tt_mb"])
            writer.writeheader()
            rows = run_sizes(sizes, args, writer)
    else:
        rows = run_sizes(sizes, args, None)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tt_mb = rows[-1]["tt_mb"] if rows else 0.0
    print(f"\npeak traced memory {peak / 1e6:.1f} MB (tt = {tt_mb:.1f} MB) "
          f"over {len(rows)} solve(s)")
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
