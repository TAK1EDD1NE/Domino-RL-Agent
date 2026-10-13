"""Phase 2: the belief sampler — hidden hands consistent with everything a seat saw.

The acting seat knows its own hand, the tiles on the table and the whole move history.
Everything else is inference. A *world* is one complete deal (all four hands) that agrees
with all of it; ``sample_worlds`` draws worlds from the public constraints:

* **counts** — every seat starts with seven tiles and only ever plays them, so a seat's
  remaining hand size is ``7 - played_by[seat]``, which is public even though the faces
  are not;
* **unseen tiles** — the complement of the acting seat's hand and the table (there is no
  boneyard, so the unseen set is exactly the tiles the other three seats hold);
* **void constraints from the history** — a forced pass at a turn proves the passing seat
  held no tile matching either open end *at that turn*. ``GameState.pass_turn`` accumulates
  those values in ``state.lacks[seat]`` as the deal is played, i.e. from the whole history
  rather than from the current board, and the lack is monotone because hands only shrink.
  :meth:`PublicView.from_events` rebuilds the same set from an explicit
  play/pass log for callers that replay a history from outside the engine.

Two dealers ship, and which one runs is never hidden:

``rejection``
    Shuffle the unseen tiles and split them by the public counts; reject and reshuffle
    until the lacks hold. **Exactly uniform** over the valid worlds (measured in
    ``tests/test_pimc_sampler.py`` against the enumerated distribution), but its
    acceptance rate collapses when the endgame makes the lacks tight.
``constrained``
    Deal the most constrained tile first — the tiles are ordered by how many seats may
    still hold them — and backtrack when a seat's remaining capacity cannot be filled.
    Much faster on tight positions, **not uniform**, and its deviation from uniform is
    measured (``bias_report``) rather than assumed away.
``auto``
    Measures the rejection acceptance rate on the position and picks: rejection while the
    rate is at least ``acceptance_threshold``, constrained below it.

Randomness always comes from an explicitly passed ``np.random.Generator``; the module
never touches global state, so a seeded run is reproducible and an unseeded one differs.

Plumbing for a learned belief head: ``tile_weights`` (per-tile, or ``(seat, tile) -> w``)
weights the seat choice in the constrained dealer and is recorded on each
:class:`World` as ``weight`` (the product of its tile weights). Weights *bias* the dealer
by design — they are the hook for a later belief-head-driven PIMC, which must then correct
for them with importance weights; the uniform path stays the default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np

from ..engine import GameState, TEAM_OF_SEAT
from ..tiles import ALL_TILE_IDS, NUM_TILES, tile_values

#: Tiles per seat at the start of a deal.
HAND_SIZE: int = 7
#: Acceptance rate below which ``method="auto"`` switches to the constrained dealer.
ACCEPTANCE_THRESHOLD: float = 0.25
#: Backtracking budget of the constrained dealer before it gives up on a world.
CONSTRAINED_NODE_LIMIT: int = 200_000

TileWeights = Union[None, np.ndarray, Sequence[float], Callable[[int, int], float]]


@dataclass(frozen=True)
class PublicView:
    """Everything one seat can see, in the form the sampler needs.

    ``played_by`` counts how many tiles each seat has played (public), and
    ``lacks[s]`` holds the pip values seat ``s`` is proven to be void in. ``ends`` is
    ``None`` for a free lead (empty board).
    """

    seat: int
    my_hand: tuple[int, ...]
    table: tuple[int, ...]
    ends: Optional[tuple[int, int]]
    turn: int
    pass_streak: int
    played_by: tuple[int, int, int, int]
    lacks: tuple[frozenset[int], ...]
    last_events: tuple[tuple[str, int, Optional[int], Optional[int]], ...] = ()
    tiles_per_seat: int = HAND_SIZE

    # ---- constructors --------------------------------------------------
    @classmethod
    def from_state(cls, state: GameState, seat: int,
                   tiles_per_seat: int = HAND_SIZE) -> "PublicView":
        """Read the view off a live engine state (the usual entry point)."""
        events = tuple(("play", int(s), int(t), int(e)) for s, t, e in state.moves[-6:])
        return cls(
            seat=int(seat),
            my_hand=tuple(sorted(int(t) for t in state.hands[int(seat)])),
            table=tuple(int(t) for t in state.table),
            ends=(int(state.ends[0]), int(state.ends[1])) if state.table else None,
            turn=int(state.turn),
            pass_streak=int(state.pass_streak),
            played_by=tuple(len(state.played_by[s]) for s in range(4)),   # type: ignore[arg-type]
            lacks=tuple(frozenset(int(v) for v in state.lacks[s]) for s in range(4)),  # type: ignore[arg-type]
            last_events=events,
            tiles_per_seat=int(tiles_per_seat),
        )

    @classmethod
    def from_events(cls, *, seat: int, my_hand: Sequence[int], events: Sequence[tuple],
                    tiles_per_seat: int = HAND_SIZE) -> "PublicView":
        """Rebuild the view from an explicit log of plays **and** passes.

        ``events`` is an ordered sequence of either ``("play", seat, tile, end)`` with
        ``end`` ``0``/``1``/``None`` or ``("pass", seat)``. The table and the open ends
        are reconstructed from the plays, and the lacks from the passes — the same
        accumulation the engine does live, for callers that replay a history from a log
        (the engine's own ``moves`` list records plays only, so passes must come from the
        log or from ``state.lacks``).
        """
        table: list[int] = []
        ends: list[int] = []
        played = [0, 0, 0, 0]
        lacks: list[set[int]] = [set(), set(), set(), set()]
        for event in events:
            kind = str(event[0])
            if kind == "pass":
                owner = int(event[1])
                if not ends:
                    continue
                lacks[owner].update(int(v) for v in ends)
                continue
            if kind != "play":
                raise ValueError(f"unknown event {event!r}")
            owner, tile = int(event[1]), int(event[2])
            end = event[3] if len(event) > 3 else None
            a, b = tile_values(int(tile))
            if not table:
                new_ends = [a, b]
            else:
                candidates: list[tuple[int, list[int]]] = []
                for index in (0, 1):
                    target = ends[index]
                    if target != a and target != b:
                        continue
                    value = a if a == b else (b if target == a else a)
                    updated = list(ends)
                    updated[index] = value
                    candidates.append((index, updated))
                if not candidates:                          # pragma: no cover - bad log
                    raise ValueError(f"tile {tile} does not fit the open ends {tuple(ends)}")
                chosen = next((c for c in candidates if c[0] == end), candidates[0])
                new_ends = chosen[1]
            table.append(int(tile))
            ends = list(new_ends)
            played[owner] += 1
        return cls(
            seat=int(seat),
            my_hand=tuple(sorted(int(t) for t in my_hand)),
            table=tuple(table),
            ends=(ends[0], ends[1]) if table else None,
            turn=int(seat),
            pass_streak=0,
            played_by=tuple(played),                       # type: ignore[arg-type]
            lacks=tuple(frozenset(values) for values in lacks),  # type: ignore[arg-type]
            last_events=tuple(tuple(e) for e in events),   # type: ignore[arg-type]
            tiles_per_seat=int(tiles_per_seat),
        )

    # ---- derived quantities --------------------------------------------
    @property
    def other_seats(self) -> tuple[int, int, int]:
        """The three seats whose hands must be sampled, in seat order."""
        return tuple(s for s in range(4) if s != self.seat)  # type: ignore[return-value]

    @property
    def counts(self) -> tuple[int, int, int]:
        """Remaining hand size of each other seat (public)."""
        return tuple(max(0, self.tiles_per_seat - self.played_by[s]) for s in self.other_seats)  # type: ignore[return-value]

    @property
    def unseen(self) -> tuple[int, ...]:
        """Tiles outside the acting seat's hand and the table (the hidden pool)."""
        seen = set(self.my_hand) | set(self.table)
        return tuple(t for t in ALL_TILE_IDS if t not in seen)


@dataclass(frozen=True)
class World:
    """A complete deal: one hand per absolute seat, own hand included."""

    hands: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    weight: float = 1.0

    def hand(self, seat: int) -> tuple[int, ...]:
        """The tiles held by ``seat``."""
        return self.hands[int(seat)]

    def masks(self) -> np.ndarray:
        """Four ``uint32`` bitmasks, one per seat (the solver's input form)."""
        masks = np.zeros(4, dtype=np.uint32)
        for seat, hand in enumerate(self.hands):
            mask = np.uint32(0)
            for tile in hand:
                mask |= np.uint32(1) << np.uint32(int(tile))
            masks[seat] = mask
        return masks

    def as_tuple(self) -> tuple[tuple[int, ...], ...]:
        """Plain tuple form, for equality checks in tests."""
        return self.hands


def _tile_values_set(tile: int) -> frozenset[int]:
    """The pip values of a tile, as a set."""
    a, b = tile_values(int(tile))
    return frozenset((a, b))


def _allowed_for(seat: int, lacks: frozenset[int], tile: int) -> bool:
    """True when ``seat`` may hold ``tile`` given its void constraints."""
    return not (_tile_values_set(tile) & set(lacks))


def _split(pool: Sequence[int], counts: Sequence[int]) -> list[list[int]]:
    """Split ``pool`` (already in the desired order) into hands of the given sizes."""
    hands: list[list[int]] = []
    cursor = 0
    for count in counts:
        hands.append([int(t) for t in pool[cursor:cursor + count]])
        cursor += count
    return hands


def _weight_of(tile_weights: TileWeights, seat: int, tile: int) -> float:
    """Weight of putting ``tile`` in ``seat``'s hand (1.0 when unweighted)."""
    if tile_weights is None:
        return 1.0
    if callable(tile_weights):
        return float(tile_weights(seat, tile))
    return float(tile_weights[tile])


def _rejection_world(view: PublicView, rng: np.random.Generator,
                     attempts: int = 200_000) -> World:
    """Exactly uniform rejection sampler (shuffle-and-split, reject on a lack)."""
    seats = view.other_seats
    counts = view.counts
    pool = list(view.unseen)
    for _ in range(attempts):
        order = rng.permutation(len(pool))
        shuffled = [pool[int(i)] for i in order]
        hands = _split(shuffled, counts)
        if all(all(_allowed_for(seat, view.lacks[seat], tile) for tile in hand)
               for seat, hand in zip(seats, hands)):
            complete = [list(view.my_hand)] * 4
            for seat, hand in zip(seats, hands):
                complete[seat] = sorted(hand)
            complete[view.seat] = sorted(view.my_hand)
            return World(tuple(tuple(hand) for hand in complete))
    raise ValueError(
        "rejection sampling found no consistent world in "
        f"{attempts} attempts for counts {tuple(counts)} and lacks "
        f"{[sorted(view.lacks[s]) for s in seats]}"
    )


def _constrained_world(view: PublicView, rng: np.random.Generator,
                       tile_weights: TileWeights = None) -> World:
    """Most-constrained-tile-first dealer with randomized backtracking.

    Tiles are placed in order of how many seats may still hold them (ties broken by a
    random draw), each into a seat drawn with probability proportional to its weight among
    the seats that are allowed and still have capacity. Not uniform over the valid worlds
    — ``bias_report`` measures the deviation.
    """
    seats = view.other_seats
    remaining = list(view.counts)
    unseen = list(view.unseen)
    allowed_counts = []
    for tile in unseen:
        allowed_counts.append(sum(1 for index, seat in enumerate(seats)
                                  if _allowed_for(seat, view.lacks[seat], tile)))
    jitter = rng.random(len(unseen))
    order = sorted(range(len(unseen)),
                   key=lambda i: (allowed_counts[i], jitter[i]))
    hands: list[list[int]] = [[], [], []]
    nodes = [0]

    def place(position: int, weights: list[float]) -> Optional[list[float]]:
        nodes[0] += 1
        if nodes[0] > CONSTRAINED_NODE_LIMIT:
            raise ValueError("constrained dealer exhausted its backtracking budget")
        if position == len(order):
            return weights
        tile = unseen[order[position]]
        candidates = [index for index in range(len(seats))
                      if remaining[index] > 0
                      and _allowed_for(seats[index], view.lacks[seats[index]], tile)]
        if not candidates:
            return None
        probs = np.array([max(_weight_of(tile_weights, seats[i], tile), 0.0)
                          for i in candidates], dtype=np.float64)
        if probs.sum() <= 0:
            probs = np.ones(len(candidates), dtype=np.float64)
        probs = probs / probs.sum()
        draw = rng.choice(len(candidates), p=probs)
        # Try the drawn seat first, then the rest (backtracking must not change the
        # *distribution* of the first choice, only the feasibility of the world).
        ordered = [int(candidates[draw])] + [int(c) for i, c in enumerate(candidates)
                                             if i != int(draw)]
        for index in ordered:
            seat = seats[index]
            hands[index].append(tile)
            remaining[index] -= 1
            result = place(position + 1,
                           weights + [max(_weight_of(tile_weights, seat, tile), 0.0)])
            if result is not None:
                return result
            hands[index].pop()
            remaining[index] += 1
        return None

    weights = place(0, [])
    if weights is None:                                      # pragma: no cover - fallback
        return _rejection_world(view, rng)
    complete = [list(view.my_hand) for _ in range(4)]
    for seat, hand in zip(seats, hands):
        complete[seat] = sorted(hand)
    product = 1.0
    for value in weights:
        product *= value
    return World(tuple(tuple(hand) for hand in complete), weight=product)


def acceptance_rate(view: PublicView, rng: np.random.Generator,
                    trials: int = 64) -> float:
    """Fraction of shuffle-and-split draws that satisfy the lacks (measured, cheap)."""
    seats = view.other_seats
    counts = view.counts
    pool = list(view.unseen)
    if not pool:
        return 1.0
    accepted = 0
    for _ in range(int(trials)):
        order = rng.permutation(len(pool))
        shuffled = [pool[int(i)] for i in order]
        hands = _split(shuffled, counts)
        if all(all(_allowed_for(seat, view.lacks[seat], tile) for tile in hand)
               for seat, hand in zip(seats, hands)):
            accepted += 1
    return accepted / float(trials)


def sample_worlds(view: PublicView, n: int, rng: np.random.Generator, *,
                  method: str = "auto", tile_weights: TileWeights = None,
                  acceptance_threshold: float = ACCEPTANCE_THRESHOLD) -> list[World]:
    """Draw ``n`` consistent worlds for ``view`` using ``rng``.

    Parameters
    ----------
    method:
        ``"rejection"`` (exactly uniform), ``"constrained"`` (fast, measurably biased)
        or ``"auto"`` (measure the acceptance rate on this position and choose).
    tile_weights:
        Optional per-tile weights (array-like indexed by tile id) or ``(seat, tile)``
        callable. Only the constrained dealer consumes them, and they bias it by design.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if method not in ("auto", "rejection", "constrained"):
        raise ValueError(f"unknown method {method!r}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator (no global randomness)")
    counts = view.counts
    if sum(counts) != len(view.unseen):
        raise ValueError(
            f"unsatisfiable view: {len(view.unseen)} unseen tile(s) do not match the "
            f"public counts {counts}"
        )
    chosen = method
    if chosen == "auto":
        rate = acceptance_rate(view, rng)
        chosen = "rejection" if rate >= acceptance_threshold else "constrained"
    if chosen == "rejection" and tile_weights is not None:
        raise ValueError(
            "tile_weights bias the dealer: use method='constrained' (the rejection path is "
            "reserved for exactly uniform sampling)"
        )
    if chosen == "rejection":
        return [_rejection_world(view, rng) for _ in range(int(n))]
    return [_constrained_world(view, rng, tile_weights) for _ in range(int(n))]


def enumerate_worlds(view: PublicView) -> list[World]:
    """Every valid world, each once (the exact support; small positions only)."""
    from itertools import combinations

    seats = view.other_seats
    counts = view.counts
    pool = list(view.unseen)
    worlds: list[World] = []
    remaining = list(pool)

    def rec(index: int, rest: list[int], hands: list[list[int]]) -> None:
        if index == len(seats):
            complete = [list(view.my_hand) for _ in range(4)]
            for seat, hand in zip(seats, hands):
                complete[seat] = sorted(hand)
            worlds.append(World(tuple(tuple(hand) for hand in complete)))
            return
        seat = seats[index]
        for combo in combinations(rest, counts[index]):
            if any(not _allowed_for(seat, view.lacks[seat], tile) for tile in combo):
                continue
            left = [t for t in rest if t not in set(combo)]
            hands.append(list(combo))
            rec(index + 1, left, hands)
            hands.pop()

    rec(0, remaining, [])
    return worlds


def chi_square_uniformity(observed: np.ndarray, draws: int) -> dict:
    """Goodness of fit of ``observed`` counts against uniform-over-support.

    Returns the statistic, the degrees of freedom and ``chi2 / dof``: an unbiased dealer
    leaves ``chi2 / dof`` near 1 (and ``uniform_compatible`` true), while a biased one
    inflates it. ``dof == 0`` means the support has a single world (nothing to test).
    """
    counts = np.asarray(observed, dtype=np.float64)
    support = int(counts.size)
    total = float(counts.sum())
    if support <= 1 or total <= 0:
        return {"support": support, "chi2": 0.0, "dof": 0, "chi2_over_dof": 0.0,
                "uniform_compatible": True}
    expected = total / support
    statistic = float(((counts - expected) ** 2 / expected).sum())
    dof = support - 1
    ratio = statistic / dof
    # A crude but honest calibration: with dof >= 20 the statistic of an unbiased dealer
    # sits near dof (sd ~ sqrt(2/dof)), so a ratio above 2 is a real deviation.
    compatible = ratio < 2.0 if dof >= 20 else ratio < 4.0
    return {"support": support, "chi2": statistic, "dof": dof, "chi2_over_dof": ratio,
            "uniform_compatible": bool(compatible)}


def bias_report(view: PublicView, rng: np.random.Generator, draws: int = 4000,
                *, method: str = "auto", min_draws_per_world: float = 6.0,
                **kwargs) -> dict:
    """Measure a dealer against the exact distribution (total variation + chi-square).

    Enumerates the valid worlds (they must be few enough to enumerate), draws worlds by
    rejection and by the constrained dealer on the *same* generator stream, and reports
    each empirical distribution's total-variation distance from uniform and the
    chi-square goodness of fit. The rejection dealer is the unbiased control: if its own
    ``chi2 / dof`` is far above 1 the sample is too small to say anything about the other
    dealer either, which is why the draws are raised to ``min_draws_per_world`` per world
    of support.
    """
    exact = enumerate_worlds(view)
    if not exact:
        raise ValueError("no valid world to enumerate")
    index = {world.as_tuple(): i for i, world in enumerate(exact)}
    uniform = 1.0 / len(exact)
    draws = int(max(draws, min_draws_per_world * len(exact)))

    def measure(worlds: Iterable[World]) -> dict:
        counts = np.zeros(len(exact), dtype=np.float64)
        total = 0
        unknown = 0
        for world in worlds:
            key = world.as_tuple()
            if key in index:
                counts[index[key]] += 1.0
                total += 1
            else:                                            # pragma: no cover - defensive
                unknown += 1
        if total == 0:
            return {"tv": float("nan"), "unknown": unknown, **chi_square_uniformity(counts, 0)}
        empirical = counts / total
        tv = float(0.5 * np.abs(empirical - uniform).sum())
        stats = chi_square_uniformity(counts, total)
        return {"tv": tv, "unknown": unknown, **stats}

    rejection = measure(sample_worlds(view, draws, rng, method="rejection", **kwargs))
    constrained = measure(sample_worlds(view, draws, rng, method="constrained", **kwargs))
    return {
        "worlds": len(exact),
        "draws": draws,
        "tv_rejection": rejection["tv"],
        "tv_constrained": constrained["tv"],
        "chi2_over_dof_rejection": rejection["chi2_over_dof"],
        "chi2_over_dof_constrained": constrained["chi2_over_dof"],
        "dof": rejection["dof"],
        "rejection_uniform_compatible": rejection["uniform_compatible"],
        "constrained_uniform_compatible": constrained["uniform_compatible"],
        "expected_tv_noise": float(np.sqrt(len(exact) / (2.0 * np.pi * draws))),
        "method": method,
    }


def world_to_masks(world: World) -> np.ndarray:
    """Four ``uint32`` hand masks for ``world`` (solver `solve_arrays` input)."""
    return world.masks()
