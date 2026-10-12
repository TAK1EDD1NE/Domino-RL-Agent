"""Tile identity for the double-six domino set (28 unique tiles).

Tile ids are the single source of truth shared by the engine, the observation encoder,
the action space and the renderer:

    id(a, b) = a * 7 + b - a * (a + 1) // 2      for a <= b, 0 <= a, b <= 6
"""
from __future__ import annotations

NUM_TILES = 28
TOTAL_PIPS = 168


def tile_id(a: int, b: int) -> int:
    if not (0 <= a <= 6 and 0 <= b <= 6):
        raise ValueError(f"pip values must be in 0..6, got ({a}, {b})")
    if a > b:
        a, b = b, a
    return a * 7 + b - a * (a + 1) // 2


TILES: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in range(7) for b in range(a, 7)
)
ALL_TILE_IDS: tuple[int, ...] = tuple(range(NUM_TILES))
assert len(TILES) == NUM_TILES, "tile table must contain 28 tiles"
assert all(tile_id(a, b) == i for i, (a, b) in enumerate(TILES)), "tile table must be id-ordered"

DOUBLE_IDS: tuple[int, ...] = tuple(tile_id(v, v) for v in range(7))
SIX_SIX_ID: int = tile_id(6, 6)


def tile_values(tid: int) -> tuple[int, int]:
    if not 0 <= tid < NUM_TILES:
        raise ValueError(f"tile id out of range: {tid}")
    return TILES[tid]


def tile_pips(tid: int) -> int:
    a, b = tile_values(tid)
    return a + b


def tile_name(tid: int) -> str:
    a, b = tile_values(tid)
    return f"[{a}|{b}]"


def tile_from_name(name: str) -> int:
    try:
        a, b = name.strip().strip("[]").split("|")
        return tile_id(int(a), int(b))
    except Exception as exc:  # noqa: BLE001 - normalised into ValueError
        raise ValueError(f"cannot parse tile name {name!r}") from exc
