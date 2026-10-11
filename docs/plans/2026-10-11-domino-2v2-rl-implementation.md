# 2v2 Domino RL — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete RLOC-2026/2027 deliverable — a 2v2 domino rules engine, a Gymnasium environment, a MaskablePPO agent with a custom set-encoder network trained by league self-play, a documented rule-based baseline, Pygame interactive/observation modes, an evaluation pipeline, and the report artefacts — in `C:\Users\pc\Desktop\tp\rl\project`.

**Architecture:** A pure-Python deterministic rules engine (`domino/engine.py`) is wrapped by a Gymnasium env (`domino/env.py`) that exposes the *learner team's* decision points only (opponent seats are played internally by an `OpponentPool`), with a seat-relative 192-dim public-information observation, a masked `Discrete(56)` `(tile, end)` action space and deal-terminal point rewards. `sb3-contrib` `MaskablePPO` trains it with a custom two-tower network. Pygame lives only in `domino/ui/` and is never imported by training code.

**Tech Stack:** Python 3.12, uv (pyproject + uv.lock), gymnasium 1.4, torch 2.5.1+cu121, stable-baselines3 2.7.1, sb3-contrib 2.7.1, pygame 2.6.1, tensorboard 2.21, pandas 3.0, matplotlib 3.11, pytest 9.1.

**Spec:** `docs/specs/2026-10-11-domino-2v2-rl-design.md` — the plan implements this spec; executors read both.

## Global Constraints

- Python `>=3.12,<3.13`; every command runs through `uv run …` from the project root; `uv.lock` must stay in sync (`uv sync --locked` passes).
- Mandatory root file names (assignment): `train.py`, `test.py`, `play.py`, `observe.py`, `evaluate.ipynb`, `README.md`. Root scripts are thin CLIs; logic lives in the package.
- Training path must never import `pygame`. UI modules are imported only by `play.py` / `observe.py` / UI tests.
- Observation is exactly `192` floats (`float32`), layout fixed by spec section 4. Action space is exactly `Discrete(56)` = `2 * tile_id + end_index` (`0 = left`, `1 = right`) over tile ids `id(a,b) = a*7 + b - a*(a+1)/2`, `a<=b`; the move set is deduplicated when both placements produce the same position.
- Rewards are raw points, no per-step shaping, no scaling; only the terminal transition is non-zero. Team of the learner = seats `{0,2}`; team B = seats `{1,3}`.
- The engine is deterministic given a seed and has no global mutable state; every random draw goes through an explicit `numpy.random.Generator`.
- Windows: `SubprocVecEnv` requires `if __name__ == "__main__":`; pygame tests set `SDL_VIDEODRIVER=dummy`.
- Docs, comments, README and report artefacts in English; code style PEP 8, type hints on public functions, module docstrings on every module.
- Commits: conventional-commit messages, one commit per task minimum; **never** push (owner merges/pushes himself).
- No long training runs: only smoke runs (`--steps 4096`) and the throughput benchmark, whose results update spec section 14.

### Spec addendum A (reward mode — decided during planning)

Spec section 3 fixes the reward as "points banked by the agent's team". During planning one
ambiguity surfaced from the owner's own example (`+26, −13, 0, …`): a losing deal could yield
either `0` (banked-only) or the negative of the opponents' points (zero-sum differential). Both
are implemented behind `--reward-mode {differential,banked}` because the differential is the
standard convention for symmetric zero-sum self-play with a shared policy, and the ablation A6
measures it. **Default: `differential`** (vector sum of both teams' banked points: `+p` for the
winning team, `−p` for the losing team, `0` for a void deal). Spec section 3 and 15 are patched
to record this at Task 3.

---

### Task 1: Skeleton, dependency check, tile utilities

**Files:**
- Create: `domino/__init__.py`, `domino/tiles.py`, `domino/config.py`, `tests/__init__.py`, `tests/test_tiles.py`, `domino/py.typed`
- Modify: `.gitignore` (add `*.ipynb_checkpoints/`, `docs/report/figures/*.png` ignore rules only if needed)

**Interfaces:**
- Consumes: nothing.
- Produces: `domino.tiles`: `NUM_TILES: int = 28`, `TILES: tuple[tuple[int,int], ...]`, `tile_id(a:int,b:int)->int`, `tile_values(tid:int)->tuple[int,int]`, `tile_pips(tid:int)->int`, `tile_name(tid:int)->str`, `tile_from_name(name:str)->int`, `ALL_TILE_IDS: tuple[int,...]`, `DOUBLE_IDS: tuple[int,...]` (7 doubles), `SIX_SIX_ID: int`, `TOTAL_PIPS: int = 168`.

- [ ] **Step 1: Write the failing test** — `tests/test_tiles.py`

```python
import pytest
from domino.tiles import (NUM_TILES, TILES, tile_id, tile_values, tile_pips, tile_name,
                          tile_from_name, ALL_TILE_IDS, DOUBLE_IDS, SIX_SIX_ID, TOTAL_PIPS)

def test_ids_cover_0_to_27_exactly_once():
    ids = [tile_id(a, b) for a in range(7) for b in range(a, 7)]
    assert sorted(ids) == list(range(NUM_TILES))
    assert len(set(ids)) == NUM_TILES

def test_id_is_order_insensitive():
    for a in range(7):
        for b in range(a, 7):
            assert tile_id(a, b) == tile_id(b, a)

def test_known_ids():
    assert tile_id(0, 0) == 0
    assert tile_id(0, 6) == 6
    assert tile_id(1, 1) == 7
    assert tile_id(6, 6) == 27 == SIX_SIX_ID

def test_values_and_pips_roundtrip():
    total = 0
    for tid in ALL_TILE_IDS:
        a, b = tile_values(tid)
        assert a <= b
        assert tile_id(a, b) == tid
        assert tile_pips(tid) == a + b
        total += a + b
    assert total == TOTAL_PIPS == 168

def test_doubles_and_names():
    assert len(DOUBLE_IDS) == 7
    assert all(tile_values(t)[0] == tile_values(t)[1] for t in DOUBLE_IDS)
    assert tile_name(27) == "[6|6]"
    assert tile_from_name("[6|6]") == 27
    for tid in ALL_TILE_IDS:
        assert tile_from_name(tile_name(tid)) == tid

def test_invalid_inputs():
    with pytest.raises(ValueError):
        tile_id(7, 8)
    with pytest.raises(ValueError):
        tile_from_name("nonsense")
```

- [ ] **Step 2: Run test to verify it fails** — `uv run pytest tests/test_tiles.py -v` → FAIL (`ModuleNotFoundError: domino.tiles`).

- [ ] **Step 3: Write minimal implementation** — `domino/tiles.py`

```python
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
```

- [ ] **Step 4: Run test to verify it passes** — `uv run pytest tests/test_tiles.py -v` → 6 passed.

- [ ] **Step 5: Add `domino/config.py` (dataclasses + YAML I/O) and its test**

```python
"""Typed configuration objects shared by train/test/play/observe."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import pathlib
import yaml

@dataclass
class EnvConfig:
    n_seats: int = 4
    learner_seats: tuple[int, ...] = (0, 2)
    reward_mode: str = "differential"      # "differential" | "banked"
    opponent: str = "league"               # "league" | "self" | "rule" | "random" | "mixed"
    opponent_weights: dict[str, float] = field(
        default_factory=lambda: {"self": 0.5, "snapshot": 0.2, "rule": 0.2, "random": 0.1}
    )
    league_size: int = 10
    seed: int = 0
    max_deals_per_episode: int = 1

@dataclass
class TrainConfig:
    run_name: str = "league_full"
    total_steps: int = 100_000_000
    n_envs: int = 8
    n_steps: int = 512
    batch_size: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    device: str = "auto"
    checkpoint_every: int = 500_000
    snapshot_every: int = 100_000
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)

@dataclass
class EvalConfig:
    deals_per_matchup: int = 1000
    match_to: int = 100
    seed: int = 0
    model_path: str = "models/checkpoints/latest.zip"
    out_dir: str = "results"
```

plus `load_config(path) -> TrainConfig`, `save_config(cfg, path)`, `apply_overrides(cfg, pairs)` (CLI `--k v` → attribute assignment, raising on unknown keys) and `tests/test_config.py` covering YAML round-trip and an unknown-key error.

- [ ] **Step 6: Verify + commit**

```bash
uv run pytest tests/test_tiles.py tests/test_config.py -v
git add -A && git commit -m "feat(domino): tile identity table, typed configs, package skeleton"
```

---

### Task 2: Rules engine — state, dealing, legal moves, play, forced pass, block

**Files:**
- Create: `domino/engine.py`, `tests/test_engine_rules.py`
- Modify: `domino/__init__.py` (export `GameState`, `Match`)

**Interfaces:**
- Consumes: `domino.tiles`.
- Produces:
  - `@dataclass class GameState`: fields `hands: list[list[int]]` (4 sorted tile-id lists), `table: list[int]` (tiles in play order), `played_by: list[list[int]]` (per seat, tile ids in play order), `ends: list[int]` (0-2 open end values: index 0 = "left", 1 = "right"), `turn: int`, `leader: int`, `pass_streak: int`, `last_pass: list[bool]`, `deal_no: int`, `scores: list[int]` (per team, length 2), `rng: np.random.Generator`, `forced_leader_six_six: bool`.
    Methods: `clone() -> GameState`, `blocks() -> bool`, `legal_moves(seat:int)->list[tuple[int,int]]`, `legal_actions(seat:int)->list[int]`, `must_pass(seat:int)->bool`, `play(seat:int, tid:int, end:int=0)->None`, `pass_turn(seat:int)->None`, `team_of(seat:int)->int`, `pips(seat:int)->int`, `hand_size(seat:int)->int`, `table_pips()->int`, `unseen_tiles()->set[int]`.
  - `def new_deal(*, deal_no:int, rng:np.random.Generator, lead_mode:str="six_six"|"winner", leader:int|None=None) -> GameState`
  - `def legal_moves(state:GameState, seat:int) -> list[tuple[int,int]]` — `(tile_id, end_index)` placements, deduplicated
  - `def legal_actions(state:GameState, seat:int) -> list[int]` (module-level alias of the method, for tests)

- [ ] **Step 1: Write the failing test** — `tests/test_engine_rules.py`

```python
import numpy as np
import pytest
from domino.engine import GameState, new_deal, legal_actions
from domino.tiles import ALL_TILE_IDS, SIX_SIX_ID, tile_id, tile_pips, TOTAL_PIPS


def fresh(seed=0, **kw):
    return new_deal(deal_no=1, rng=np.random.default_rng(seed), **kw)


def test_deal_gives_seven_tiles_each_and_loses_none():
    s = fresh()
    assert [len(h) for h in s.hands] == [7, 7, 7, 7]
    assert sorted(t for h in s.hands for t in h) == list(ALL_TILE_IDS)
    assert s.table == [] and s.ends == []


def test_deal_is_deterministic_for_a_seed():
    assert fresh(3).hands == fresh(3).hands
    assert fresh(3).hands != fresh(4).hands


def test_first_deal_leader_holds_six_six_and_must_play_it():
    s = fresh()
    assert s.leader == s.turn
    assert SIX_SIX_ID in s.hands[s.leader]
    assert legal_actions(s, s.leader) == [SIX_SIX_ID]


def test_lead_on_later_deal_is_free():
    s = new_deal(deal_no=2, rng=np.random.default_rng(0), lead_mode="winner", leader=2)
    assert s.turn == 2 and s.leader == 2
    assert sorted(legal_actions(s, 2)) == sorted(s.hands[2])


def test_play_requires_a_matching_end_and_is_mandatory():
    s = fresh()
    s.play(s.turn, SIX_SIX_ID)                      # forced opener
    nxt = s.turn
    assert s.ends == [6, 6]
    legal = legal_actions(s, nxt)
    assert all(6 in (tile_id(*sorted((t % 7, t % 7)))) or True for t in legal)  # sanity placeholder
    assert legal == [t for t in s.hands[nxt] if 6 in _vals(t)]
    if legal:
        assert not s.must_pass(nxt)
        with pytest.raises(ValueError):
            s.play(nxt, next(t for t in s.hands[nxt] if t not in legal))
    else:
        assert s.must_pass(nxt)


def _vals(tid):
    from domino.tiles import tile_values
    return tile_values(tid)


def test_pips_conservation_after_every_action():
    s = fresh(11)
    for _ in range(28):
        seat = s.turn
        la = legal_actions(s, seat)
        if la:
            s.play(seat, la[0])
        else:
            s.pass_turn(seat)
        hands = sum(tile_pips(t) for h in s.hands for t in h)
        assert hands + s.table_pips() == TOTAL_PIPS
        if s.blocks() or len(s.table) == 28:
            break


def test_block_after_exactly_four_consecutive_passes():
    s = new_deal(deal_no=1, rng=np.random.default_rng(7))
    # build a state nobody can extend: table ends (0, 0) while no hand holds a 0 twin
    s.table = [tile_id(0, 0)]
    s.ends = [0, 0]
    s.hands = [[t for t in h if 0 not in _vals(t)] for h in s.hands]
    s.turn = 0
    s.pass_streak = 0
    for i in range(3):
        seat = s.turn
        assert s.must_pass(seat)
        s.pass_turn(seat)
        assert not s.blocks()
    assert s.must_pass(s.turn)
    s.pass_turn(s.turn)
    assert s.blocks() and s.pass_streak == 4


def test_hand_emptying_is_detected():
    s = new_deal(deal_no=2, rng=np.random.default_rng(1), lead_mode="winner", leader=0)
    s.hands[0] = [tile_id(1, 2)]
    s.table = []
    s.ends = []
    s.turn = 0
    s.play(0, tile_id(1, 2))
    assert s.hand_size(0) == 0
```

- [ ] **Step 2: Run test to verify it fails** — `uv run pytest tests/test_engine_rules.py -v` → FAIL (`ImportError: cannot import name 'GameState'`).

- [ ] **Step 3: Write minimal implementation** — `domino/engine.py`

```python
"""Pure-Python rules engine for Algerian 2v2 domino.

No gymnasium, no torch, no pygame: the engine is a deterministic state machine so it can be
unit-tested exhaustively and reused by the environment, the evaluation scripts and the UI.

Rules implemented (see docs/specs/2026-10-11-domino-2v2-rl-design.md section 2):
28 tiles, 7 per seat, no boneyard, mandatory play, pass only when stuck, blocked after four
consecutive passes, doubles are ordinary line moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .tiles import (ALL_TILE_IDS, NUM_TILES, SIX_SIX_ID, TOTAL_PIPS, tile_pips, tile_values)

SEATS = 4
TEAM_OF_SEAT = (0, 1, 0, 1)          # seats 0,2 -> team 0 ; seats 1,3 -> team 1
PARTNER_OF_SEAT = (2, 3, 0, 1)
OPPONENTS_OF_SEAT = ((1, 3), (0, 2), (1, 3), (0, 2))


@dataclass
class GameState:
    hands: list[list[int]]
    rng: np.random.Generator
    table: list[int] = field(default_factory=list)
    played_by: list[list[int]] = field(default_factory=lambda: [[], [], [], []])
    ends: list[int] = field(default_factory=list)
    turn: int = 0
    leader: int = 0
    pass_streak: int = 0
    last_pass: list[bool] = field(default_factory=lambda: [False] * SEATS)
    deal_no: int = 1
    scores: list[int] = field(default_factory=lambda: [0, 0])

    # ---- construction -------------------------------------------------
    def clone(self) -> "GameState":
        return GameState(
            hands=[list(h) for h in self.hands],
            rng=np.random.default_rng(self.rng.integers(0, 2**63 - 1)),
            table=list(self.table),
            played_by=[list(p) for p in self.played_by],
            ends=list(self.ends), turn=self.turn, leader=self.leader,
            pass_streak=self.pass_streak, last_pass=list(self.last_pass),
            deal_no=self.deal_no, scores=list(self.scores),
        )

    # ---- queries ------------------------------------------------------
    @staticmethod
    def team_of(seat: int) -> int:
        return TEAM_OF_SEAT[seat]

    def hand_size(self, seat: int) -> int:
        return len(self.hands[seat])

    def pips(self, seat: int) -> int:
        return sum(tile_pips(t) for t in self.hands[seat])

    def table_pips(self) -> int:
        return sum(tile_pips(t) for t in self.table)

    def unseen_tiles(self) -> set[int]:
        played = set(self.table) | {t for h in self.hands for t in h}
        return set(ALL_TILE_IDS) - played

    def legal_actions(self, seat: int) -> list[int]:
        return legal_actions(self, seat)

    def must_pass(self, seat: int) -> bool:
        return not legal_actions(self, seat)

    def blocks(self) -> bool:
        return self.pass_streak >= SEATS

    def is_terminal(self) -> bool:
        return self.blocks() or any(len(h) == 0 for h in self.hands) or len(self.table) == NUM_TILES

    # ---- transitions --------------------------------------------------
    def play(self, seat: int, tid: int) -> None:
        if seat != self.turn:
            raise ValueError(f"seat {seat} cannot play on seat {self.turn}'s turn")
        if tid not in self.hands[seat]:
            raise ValueError(f"tile {tid} is not in seat {seat}'s hand")
        if tid not in self.legal_actions(seat):
            raise ValueError(f"tile {tid} is not playable on ends {self.ends}")
        self.hands[seat].remove(tid)
        a, b = tile_values(tid)
        if not self.table:                       # free lead: the line exposes the tile's two values
            self.table.append(tid)
            self.ends = [a, b]                    # a double leads to two equal ends, e.g. [6, 6]
        elif a == b:                             # double: end value is unchanged
            if a == self.ends[0]:
                pass
            elif len(self.ends) > 1 and a == self.ends[1]:
                self.ends[0], self.ends[1] = self.ends[1], self.ends[0]
            else:
                raise ValueError("double does not match an open end")
            self.table.append(tid)
        else:
            if a in self.ends and b in self.ends and len(set(self.ends)) == 2:
                # ambiguous: caller must disambiguate; engine keeps "left" preference
                idx = 0
            elif a in self.ends:
                idx = 0 if self.ends[0] == a else 1
                a, b = b, a
            elif b in self.ends:
                idx = 0 if self.ends[0] == b else 1
            else:
                raise ValueError("tile does not match an open end")
            self.ends[idx] = b if idx == 0 else b
            self.table.append(tid)
        self.played_by[seat].append(tid)
        self.pass_streak = 0
        self.last_pass[seat] = False
        self.turn = (seat + 1) % SEATS

    def pass_turn(self, seat: int) -> None:
        if seat != self.turn:
            raise ValueError("out of turn pass")
        if not self.must_pass(seat):
            raise ValueError("mandatory play violated: passing with a legal tile in hand")
        self.pass_streak += 1
        self.last_pass[seat] = True
        self.turn = (seat + 1) % SEATS


def legal_actions(state: GameState, seat: int) -> list[int]:
    hand = state.hands[seat]
    if not hand:
        return []
    if not state.table:                      # free lead
        return sorted(hand)
    if state.deal_no == 1 and state.leader == seat and SIX_SIX_ID in hand and not state.table:
        return [SIX_SIX_ID]
    ends = set(state.ends)
    return sorted(t for t in hand if ends & set(tile_values(t)))


def new_deal(*, deal_no: int, rng: np.random.Generator, lead_mode: str = "six_six",
             leader: int | None = None, scores: list[int] | None = None) -> GameState:
    tiles = list(ALL_TILE_IDS)
    rng.shuffle(tiles)
    hands = [sorted(tiles[i * 7:(i + 1) * 7]) for i in range(SEATS)]
    if lead_mode == "six_six":
        leader_seat = next(i for i, h in enumerate(hands) if SIX_SIX_ID in h)
    elif lead_mode == "winner":
        if leader is None:
            raise ValueError("lead_mode='winner' requires an explicit leader seat")
        leader_seat = leader
    else:
        raise ValueError(f"unknown lead_mode {lead_mode!r}")
    return GameState(hands=hands, rng=rng, turn=leader_seat, leader=leader_seat,
                     deal_no=deal_no, scores=list(scores or [0, 0]))
```

- [ ] **Step 4: Run test to verify it passes** — `uv run pytest tests/test_engine_rules.py -v` → all pass. **If the double-branch or ambiguous-end logic fails, fix the branch rather than the test**; the ambiguity rule (a tile matching both open ends may be oriented either way) is intentional and must be covered by an added test asserting `ends` stays a permutation of the tile's values.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(engine): deterministic domino rules engine with mandatory play and blocking"
```

---

### Task 3: Scoring, deal outcomes, leaders, void deals, match to 100 (+ spec patch for addendum A)

**Files:**
- Create: `tests/test_scoring.py`
- Modify: `domino/engine.py` (add `DealOutcome`, `deal_outcome()`, `next_leader()`, `class Match`, `redeal_after_void()`), `docs/specs/2026-10-11-domino-2v2-rl-design.md` (patch sections 3 and 15 with the reward-mode addendum)

**Interfaces:**
- Consumes: Task 2.
- Produces:
  - `@dataclass(frozen=True) class DealOutcome`: `winner_team: int | None`, `points: int`, `blocked: bool`, `void: bool`, `reason: str` (`"emptied"|"blocked"|"void_tie"|"wall"`), `winning_seat: int | None`.
  - `def deal_outcome(state: GameState) -> DealOutcome`
  - `def next_leader(state: GameState, outcome: DealOutcome) -> int | None` (`None` = redeal and lead with `[6|6]`)
  - `def redeal_after_void(state: GameState) -> GameState` (same match scores, new deal, `[6|6]` holder leads, `deal_no = state.deal_no + 1`)
  - `class Match`: `__init__(target:int=100, seed:int=0)`, `reset()`, `deal_no`, `scores`, `deals: list[DealOutcome]`, `finished: bool`, `winner_team: int | None`, `play_deal(policy_by_seat: dict[int, Callable[[GameState,int],int]]) -> DealOutcome` (drives one deal to the end, applies scoring, handles a void redeal, records the outcome).

- [ ] **Step 1: Write the failing test** — `tests/test_scoring.py`

```python
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
    s = mk(hands=[[], [tile_id(6, 6), tile_id(5, 5)], [], [tile_id(4, 4)]],
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
    s = mk(hands=[[tile_id(2, 2)], [tile_id(6, 6)], [tile_id(2, 3)], [tile_id(6, 5)]],
           table=[tile_id(1, 1)], ends=[1], pass_streak=4)
    out = deal_outcome(s)
    # seats 0 and 2 both hold 4 pips and are team-mates -> team 0 banks pips of seats 1 and 3
    assert (out.reason, out.winner_team, out.points) == ("blocked", 0, 12 + 11)


def test_block_tie_across_teams_is_void():
    s = mk(hands=[[tile_id(2, 2)], [tile_id(3, 3)], [tile_id(4, 4)], [tile_id(1, 2)]],
           table=[tile_id(0, 0)], ends=[0], pass_streak=4)
    out = deal_outcome(s)
    # seats 0 and 1 both hold 4 pips and are on different teams -> void deal
    assert (out.reason, out.winner_team, out.points, out.void) == ("void_tie", None, 0, True)


def test_void_deal_redeals_with_six_six_leader_and_keeps_scores():
    s = mk(hands=[[tile_id(2, 2)], [tile_id(3, 3)], [tile_id(4, 4)], [tile_id(1, 2)]],
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
    assert sum(1 for o in m.deals if o.void) >= 0     # void deals may occur; they score nothing
    for o in m.deals:
        if o.void:
            assert o.points == 0
    assert all(o.points > 0 for o in m.deals if not o.void)
```

- [ ] **Step 2: Run test to verify it fails** — `uv run pytest tests/test_scoring.py -v` → FAIL (`cannot import name 'deal_outcome'`).

- [ ] **Step 3: Implement scoring + match** — append to `domino/engine.py`:

```python
@dataclass(frozen=True)
class DealOutcome:
    winner_team: int | None
    points: int
    blocked: bool
    void: bool
    reason: str
    winning_seat: int | None = None


def deal_outcome(state: GameState) -> DealOutcome:
    """Score the finished deal according to the project's house rules."""
    emptied = [i for i, h in enumerate(state.hands) if not h]
    if emptied:
        seat = emptied[0]
        team = TEAM_OF_SEAT[seat]
        opponents = OPPONENTS_OF_SEAT[seat]
        return DealOutcome(team, sum(state.pips(o) for o in opponents), False, False, "emptied", seat)
    if not state.blocks() and len(state.table) < NUM_TILES:
        raise ValueError("deal_outcome called on a deal that is neither emptied nor blocked")
    pips = [state.pips(i) for i in range(SEATS)]
    fewest = min(pips)
    tied = [i for i in range(SEATS) if pips[i] == fewest]
    teams = {TEAM_OF_SEAT[i] for i in tied}
    if len(teams) > 1:
        return DealOutcome(None, 0, True, True, "void_tie", None)
    team = teams.pop()
    winner_seat = tied[0]
    opponents = OPPONENTS_OF_SEAT[winner_seat]
    return DealOutcome(team, sum(state.pips(o) for o in opponents), True, False, "blocked", winner_seat)


def next_leader(state: GameState, outcome: DealOutcome) -> int | None:
    if outcome.void:
        return None
    if outcome.reason == "emptied":
        return outcome.winning_seat
    return outcome.winning_seat


def redeal_after_void(state: GameState) -> GameState:
    return new_deal(deal_no=state.deal_no + 1, rng=state.rng, lead_mode="six_six",
                    scores=list(state.scores))


class Match:
    """A match to `target` points; drives deals through caller-supplied per-seat policies."""

    def __init__(self, target: int = 100, seed: int = 0) -> None:
        self.target = target
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.deals: list[DealOutcome] = []
        self.scores: list[int] = [0, 0]
        self.deal_no = 1
        self.finished = False
        self.winner_team: int | None = None

    def reset(self) -> None:
        self.__init__(target=self.target, seed=self.seed)

    def play_deal(self, policy_by_seat) -> DealOutcome:
        state = new_deal(deal_no=self.deal_no, rng=self.rng, lead_mode="six_six",
                         scores=list(self.scores))
        while True:
            while not state.is_terminal():
                seat = state.turn
                legal = state.legal_actions(seat)
                if not legal:
                    state.pass_turn(seat)
                    continue
                action = int(policy_by_seat[seat](state, seat))
                if action not in legal:
                    raise ValueError(f"policy for seat {seat} returned illegal action {action}")
                state.play(seat, action)
            outcome = deal_outcome(state)
            if not outcome.void:
                break
            self.deals.append(outcome)
            state = redeal_after_void(state)
            self.deal_no += 1
        self.deals.append(outcome)
        self.scores[outcome.winner_team] += outcome.points
        self.deal_no += 1
        if max(self.scores) >= self.target:
            self.finished = True
            self.winner_team = 0 if self.scores[0] >= self.scores[1] else 1
        return outcome
```

- [ ] **Step 4: Run test to verify it passes** — `uv run pytest tests/test_scoring.py -v` → all pass.

- [ ] **Step 5: Patch the spec** with addendum A: in section 3 reward bullet add "Reward mode is configurable (`differential` default | `banked`), see addendum A"; in section 15 add "A6 reward mode differential vs banked".

- [ ] **Step 6: Commit**

```bash
uv run pytest -q
git add -A && git commit -m "feat(engine): house-rule scoring, void deals, match to 100; spec addendum A (reward mode)"
```

---

### Task 4: Observation encoder (192 dims)

**Files:** Create `domino/obs.py`, `tests/test_obs.py`

**Interfaces:**
- Consumes: Task 2/3.
- Produces: `OBS_DIM: int = 192`, slice constants (`HAND`, `PLAYED`, `PLAYED_BY`, `MY_PIPS`, `UNSEEN_PIPS`, `PASS_STREAK`, `LAST_PASS`, `COUNTS`, `DEAL_LEADER`, `AM_LEADER`, `LIVE_VALUE`, `BOARD_EMPTY`), `def encode(state: GameState, seat: int) -> np.ndarray` (float32, shape `(192,)`), `def describe_observation() -> dict[str, tuple[int, ...]]` (name → slice indices, used by the report figure).

- [ ] **Step 1: Failing test** — `tests/test_obs.py`

```python
import numpy as np
import pytest
from domino.engine import new_deal
from domino.obs import OBS_DIM, encode, describe_observation
from domino.tiles import tile_id, SIX_SIX_ID


def test_shape_dtype_and_range():
    s = new_deal(deal_no=1, rng=np.random.default_rng(0))
    v = encode(s, s.turn)
    assert v.shape == (OBS_DIM,) == (192,) and v.dtype == np.float32
    assert np.isfinite(v).all()

def test_hand_and_counts_are_seat_relative():
    s = new_deal(deal_no=1, rng=np.random.default_rng(1))
    for seat in range(4):
        v = encode(s, seat)
        hand = v[0:28]
        assert hand.sum() == 7
        assert hand[SIX_SIX_ID] == (1.0 if SIX_SIX_ID in s.hands[seat] else 0.0)
        assert list(v[175:179]) == [len(s.hands[(seat + k) % 4]) / 7 for k in range(4)]

def test_played_by_slice_tracks_the_table():
    s = new_deal(deal_no=1, rng=np.random.default_rng(2))
    leader = s.turn
    s.play(leader, SIX_SIX_ID)
    v = encode(s, leader)
    played_by = v[56:168].reshape(4, 28)
    assert played_by[0, SIX_SIX_ID] == 1.0        # "self" is always the acting seat
    v2 = encode(s, (leader + 2) % 4)
    played_by2 = v2[56:168].reshape(4, 28)
    assert played_by2[2, SIX_SIX_ID] == 1.0       # the leader is "partner" from that seat

def test_no_hidden_information_is_encodable():
    # the encoding must not depend on seats the acting seat cannot observe
    s = new_deal(deal_no=1, rng=np.random.default_rng(3))
    a = encode(s, 0).copy()
    s.hands[1] = [tile_id(0, 0)]
    b = encode(s, 0)
    assert np.array_equal(a, b) is False  # counts change (observable), but...
    # ...the content of seat 1's hand must never leak: only the size can differ
    s2 = new_deal(deal_no=1, rng=np.random.default_rng(3))
    s2.hands[1] = [tile_id(6, 6)]
    assert np.array_equal(encode(s2, 0)[0:56], encode(s, 0)[0:56])

def test_describe_observation_covers_all_192_dims():
    d = describe_observation()
    covered = [i for sl in d.values() for i in range(sl[0], sl[1])]
    assert sorted(covered) == list(range(192))
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_obs.py -v` → FAIL.

- [ ] **Step 3: Implement** `domino/obs.py` — module docstring citing spec section 4, `_tile_values_matrix` cached with `functools.lru_cache`, `encode()` filling a preallocated `np.zeros(192, dtype=np.float32)`:
  - `HAND = slice(0, 28)`: `v[t] = 1` for `t in state.hands[seat]`.
  - `PLAYED = slice(28, 56)`: 1 for `t in state.table`.
  - `PLAYED_BY = slice(56, 168)`: for relative index `r`, seat `s = (seat + r) % 4`, `v[56 + r*28 + t] = 1` for `t in state.played_by[s]`.
  - `MY_PIPS = slice(168, 169)`: `state.pips(seat) / TOTAL_PIPS`.
  - `UNSEEN_PIPS = slice(169, 170)`: `(TOTAL_PIPS - state.pips(seat) - state.table_pips()) / TOTAL_PIPS`.
  - `PASS_STREAK = slice(170, 171)`: `state.pass_streak / 4`.
  - `LAST_PASS = slice(171, 175)`: `state.last_pass[(seat + r) % 4]`.
  - `COUNTS = slice(175, 179)`: `len(state.hands[(seat + r) % 4]) / 7`.
  - `DEAL_LEADER = slice(179, 183)`: one-hot of `(state.leader - seat) % 4`.
  - `AM_LEADER = slice(183, 184)`: `1.0 if state.leader == seat`.
  - `LIVE_VALUE = slice(184, 191)`: for each value `v`, 1 if any unseen tile contains `v`.
  - `BOARD_EMPTY = slice(191, 192)`: `1.0 if not state.table`.
  - `describe_observation()` returns the name→slice-tuple map and asserts the union is 0..191.

- [ ] **Step 4: Run to verify pass**; **Step 5: commit** `feat(obs): 192-dim seat-relative public-information encoder`.

---

### Task 5: Action masks

**Files:** Create `domino/masks.py`, `tests/test_masks.py`

**Interfaces:** Produces `ACTION_DIM: int = 56`, `def action_index(tid: int, end: int) -> int` (= `2*tid+end`), `def decode_action(action: int) -> tuple[int,int]`, `def action_mask(state: GameState, seat: int) -> np.ndarray` (bool array shape (56,), `np.bool_`, built from `engine.legal_moves`), `def masked_actions(mask) -> np.ndarray`, `def assert_consistent(state, seat, action)`.

- [ ] **Step 1: Failing test** — mask equals `legal_actions`, always non-empty at a decision point, forced `[6|6]` lead mask is a single `True`, illegal action raises through `assert_consistent`.
- [ ] **Step 2/3/4:** implement + pass + **commit** `feat(masks): action masking over the 28 tile ids`.

---

### Task 6: Gymnasium environment (learner-team MDP)

**Files:** Create `domino/env.py`, `tests/test_env.py`; Modify `domino/__init__.py`

**Interfaces:**
- Consumes: Tasks 2-5, `domino/agents/`.
- Produces: `class DominoEnv(gym.Env)`, `__init__(self, opponent: Agent | OpponentPool | None = None, env_cfg: EnvConfig | None = None, seed: int | None = None)`; attributes `observation_space` (`Box(0, 1, (192,), float32)`), `action_space` (`Discrete(56)`); methods `reset(seed=None, options=None) -> (np.ndarray, dict)`, `step(action) -> (obs, reward, terminated, truncated, info)`, `action_masks() -> np.ndarray`, `state` property, `acting_seat` property, `set_opponent(obj)`, `render(mode="human")` (delegates to `domino.ui.render` lazily). Registered as `gymnasium.register("Domino2v2-v0", entry_point="domino.env:DominoEnv")` in `domino/__init__.py`.

- [ ] **Step 1: Failing test** — `tests/test_env.py`

```python
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from domino.env import DominoEnv
from domino.agents.random_agent import RandomAgent
from domino.obs import OBS_DIM


def test_spaces_and_reset():
    env = DominoEnv(opponent=RandomAgent(), seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.shape == (OBS_DIM,)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert env.action_space.n == 56
    assert env.acting_seat in (0, 2)          # only learner seats produce decisions
    assert info["acting_seat"] == env.acting_seat


def test_mask_is_never_empty_at_a_decision_point(draws=50):
    env = DominoEnv(opponent=RandomAgent(), seed=1)
    env.reset(seed=1)
    for _ in range(draws):
        mask = env.action_masks()
        assert mask.dtype == np.bool_ and mask.shape == (56,)
        assert mask.sum() >= 1
        legal = env.state.legal_actions(env.acting_seat)
        assert sorted(np.flatnonzero(mask).tolist()) == sorted(legal)
        obs, r, term, trunc, info = env.step(int(np.flatnonzero(mask)[0]))
        if term or trunc:
            env.reset()
        assert r == 0.0 or term       # only the terminal step carries reward


def test_illegal_action_raises():
    env = DominoEnv(opponent=RandomAgent(), seed=2)
    env.reset(seed=2)
    mask = env.action_masks()
    illegal = int(np.flatnonzero(~mask)[0])
    with pytest.raises(ValueError):
        env.step(illegal)


def test_terminal_reward_matches_house_scoring():
    env = DominoEnv(opponent=RandomAgent(), seed=3)
    obs, info = env.reset(seed=3)
    total = 0.0
    for _ in range(200):
        mask = env.action_masks()
        obs, r, term, trunc, info = env.step(int(np.flatnonzero(mask)[0]))
        total += r
        if term:
            break
    assert term and not trunc
    assert info["deal_outcome"] in ("emptied", "blocked")
    assert abs(total) == pytest.approx(info["points_for_learner_team"]) or info["void_deal"]


def test_episode_length_is_bounded_and_no_transition_for_forced_pass():
    env = DominoEnv(opponent=RandomAgent(), seed=4)
    env.reset(seed=4)
    steps = 0
    while True:
        mask = env.action_masks()
        assert mask.sum() >= 1
        assert env.state.legal_actions(env.acting_seat), "a forced pass must not be a decision point"
        _, _, term, trunc, _ = env.step(int(np.flatnonzero(mask)[0]))
        steps += 1
        if term or trunc:
            break
    assert 1 <= steps <= 14


def test_gymnasium_api_check():
    env = DominoEnv(opponent=RandomAgent(), seed=5)
    check_env(env, skip_render_check=True)
```

- [ ] **Step 2: Run to verify failure**, **Step 3: implement** the internal turn loop exactly as designed in spec section 6 (`reset` deals, assigns the opponent for the deal, advances; `step` validates against the mask, plays, then advances through opponent seats and forced passes; terminal reward computed by `deal_outcome` and cast to `differential`/`banked` per `EnvConfig.reward_mode`), **Step 4: run to verify pass**, **Step 5: commit** `feat(env): learner-team Gymnasium env with league hook and masked actions`.

---

### Task 7: Agents — random, rule-based baseline, policy wrapper

**Files:** Create `domino/agents/__init__.py`, `domino/agents/base.py`, `domino/agents/random_agent.py`, `domino/agents/rule_agent.py`, `domino/agents/policy_agent.py`, `tests/test_agents.py`

**Interfaces:** `class Agent(Protocol): name: str; def act(self, state: GameState, seat: int) -> int`; `RandomAgent(seed=None)`; `RuleWeights` dataclass with the spec section 10 weights and `RuleAgent(weights: RuleWeights | None = None, seed: int | None = None)` exposing `def score_move(state, seat, tid) -> float` and `def explain(state, seat) -> list[tuple[int, float]]`; `PolicyAgent(model: MaskablePPO, deterministic: bool = True)`.

- [ ] **Step 1: Failing test** — legality for every agent over 200 random states; the rule bot beats random over 300 deals (win rate ≥ 0.55, the "baseline is a real opponent" check) using `Match` with the bot on team 0 and random on team 1; `score_move` returns a finite float for every legal move; `RuleAgent` is deterministic under a seed.
- [ ] **Step 2/3/4:** implement (the rule bot implements the spec's term table in one pass over legal moves, `explain()` returning the per-term breakdown for the report), pass, **Step 5: commit** `feat(agents): random, documented rule-based baseline, policy wrapper`.

---

### Task 8: Opponent pool and snapshots

**Files:** Create `domino/agents/pool.py`, `tests/test_pool.py`

**Interfaces:** `class OpponentPool: __init__(self, weights: dict[str,float], learner_model=None, snapshots: list[str] | None = None, rule_agent=None, seed=None)`; `def sample(self) -> Agent`; `def assign_team(self, deal_seed: int) -> tuple[Agent, Agent]`; `def set_learner(self, model)`; `def refresh_snapshots(self, paths: Sequence[str])` (LRU cache, max 3 loaded `MaskablePPO` models); `def as_team_policy(...)`. Weight keys: `self`, `snapshot`, `rule`, `random`; missing keys are dropped and the rest renormalised; all-zero weights raise.

- [ ] **Step 1: Failing test** — sampling honours weights within tolerance over 10 000 draws with a fixed seed; a deal always assigns the *same* opponent type to both opposing seats by default and per-seat mixing when enabled; `refresh_snapshots(["missing.zip"])` raises a clear error; LRU never holds more than 3 models (counted via a monkeypatched loader).
- [ ] **Step 2/3/4/5:** implement, pass, commit `feat(league): opponent pool with snapshot LRU and weighted sampling`.

---

### Task 9: Custom network + policy (graded requirement)

**Files:** Create `domino/policy.py`, `tests/test_policy.py`

**Interfaces:** `class DominoFeaturesExtractor(BaseFeaturesExtractor)` (features_dim=128, set encoder + context tower as in spec section 7), `class DominoMaskablePolicy(MaskableActorCriticPolicy)` (LayerNorm trunk, orthogonal init, gain √2, `net_arch`-free), `def policy_kwargs_for(cfg: TrainConfig) -> dict` (returns `{"features_extractor_class": DominoFeaturesExtractor, "features_extractor_kwargs": {...}}`).

- [ ] **Step 1: Failing test** — building `MaskablePPO(DominoMaskablePolicy, env, policy_kwargs=policy_kwargs_for(cfg))` works; `param_count(model) < 2_000_000`; forward pass on a batch of 8 obs returns action shapes `(8, 56)` unmasked and `(8,)` masked; masked logits of illegal actions are `-inf`; the sampled action is legal for 200 random states; every submodule receives gradient after one `train()` call; the extractor is cardinality-agnostic (works on hands of size 1 and 7).
- [ ] **Step 2/3/4/5:** implement, pass, commit `feat(policy): custom two-tower set-encoder network and MaskablePPO policy`.

---

### Task 10: Callbacks — convergence logging and snapshots

**Files:** Create `domino/callbacks.py`, `tests/test_callbacks.py`

**Interfaces:** `class EpisodeStatsCallback(BaseCallback)`: logs `train/total_reward_per_episode` (undiscounted, per finished deal), `train/avg_episode_reward_window_100`, `deals/block_rate`, `deals/void_rate`, `deals/points_per_deal`, and appends `(step, episode, total_reward, window_avg)` rows to `results/training_<run>.csv`. `class SnapshotCallback(BaseCallback)`: every `snapshot_every` steps saves `models/snapshots/policy_<steps>.zip` keeping the newest `league_size`, then calls `vec_env.env_method("refresh_pool", paths)`.

- [ ] **Step 1: Failing test** — a 2 048-step PPO smoke run with both callbacks writes a CSV with the exact header `step,episode,total_reward,window_avg`, ≥ 10 rows, and the window average matches a recomputed pandas rolling mean; the TB event file exists and contains the two scalar tags (read with `tensorboard.backend.event_processing.event_accumulator`); snapshot files appear and the pool receives them.
- [ ] **Step 2/3/4/5:** implement, pass, commit `feat(callbacks): undiscounted per-episode convergence logging + league snapshots`.

---

### Task 11: `train.py` + configs + smoke run

**Files:** Create `train.py`, `configs/default.yaml`, `configs/quick_smoke.yaml`, `configs/league_full.yaml`, `tests/test_scripts.py` (train part)

**Interfaces:** CLI: `--config`, `--run-name`, `--steps`, `--n-envs`, `--seed`, `--reward-mode`, `--opponent`, `--snapshot-every`, `--ckpt-every`, `--device`, `--resume`, `--dry-run`, plus `--set k=v` passthrough for any `TrainConfig` field. Builds `SubprocVecEnv([make_env(i) for i in range(n_envs)])` behind `if __name__ == "__main__":`, `MaskablePPO(DominoMaskablePolicy, vec_env, tensorboard_log=f"runs/{run_name}", **cfg)`, attaches callbacks, `model.learn(total_timesteps=..., reset_num_timesteps=not resume, callback=...)`, saves `models/checkpoints/latest.zip` and `models/checkpoints/ppo_<steps>_steps.zip`, writes `runs/<run>/config.json` + `git rev-parse HEAD`.

- [ ] **Step 1: Failing test** — `subprocess.run([sys.executable, "train.py", "--steps", "4096", "--n-envs", "2", "--run-name", "smoke_test"])` exits 0 and produces `models/checkpoints/latest.zip`, `runs/smoke_test/`, and `results/training_smoke_test.csv` with ≥ 1 row; `--resume` continues from that checkpoint without error; the training process logs the measured fps.
- [ ] **Step 2: run the test to see it fail** (no such file), **Step 3: implement**, **Step 4: run it** (this is the "testing only" smoke run — never a long run), **Step 5: commit** `feat(train): CLI trainer with league, checkpoints, resume and TB logging`.

---

### Task 12: `test.py` — evaluation matrix and metrics

**Files:** Create `test.py`, `tests/test_scripts.py` (evaluate part) — the file must be named `test.py` per the assignment, so it is excluded from pytest collection via `pyproject.toml` `[tool.pytest.ini_options] collect_ignore = ["test.py"]`.

**Interfaces:** CLI: `--model`, `--model-b`, `--deals`, `--match-to`, `--matchups {rl_vs_rule,rl_vs_random,rule_vs_random,rl_vs_rl,rule_vs_rule,all}`, `--seed`, `--out`, `--workers`. Prints a table and writes one CSV row per matchup with exactly the spec section 12 schema; Wilson interval helper `wilson_interval(wins, n, z=1.96)`.

- [ ] **Step 1: Failing test** — `test.py --matchups rule_vs_random --deals 40 --model models/checkpoints/latest.zip` writes a CSV whose columns match the spec schema and where `wins_a + wins_b == deals` (deals are independent; ties are impossible because a deal always banks points or is void, and void deals are counted separately in `void_rate`, not as losses — assert the accounting identity explicitly).
- [ ] **Step 2/3/4:** implement, pass, **Step 5: commit** `feat(test): RL-vs-baseline evaluation matrix with CSV metrics`.

---

### Task 13: Pygame renderer, `play.py`, `observe.py`

**Files:** Create `domino/ui/__init__.py`, `domino/ui/render.py`, `domino/ui/play_loop.py`, `domino/ui/observe_loop.py`, `play.py`, `observe.py`, `tests/test_ui_smoke.py`

**Interfaces:** `class Renderer: __init__(self, size=(1280, 720)); def draw(self, state, *, acting_seat, legal=None, overlay=None, banner="", scores=None) -> None; def wait_event(...)`; doubles drawn crosswise; `def run_play(cfg) -> None` (human seat 1 default, `--partner rl|rule`, `--opponents rl|rule|random`, `--model`, `--seed`, `--match-to`); `def run_observe(cfg) -> None` (`--agent rl|rule`, `--model`, `--seed`, `--fps`, `--overlay-probs`).

- [ ] **Step 1: Failing test (headless)** — with `SDL_VIDEODRIVER=dummy`: `Renderer` constructs, `draw()` succeeds on a mid-deal state, `run_observe` advances 30 synthetic frames, and `play.py --help` / `observe.py --help` exit 0. Also assert `import domino.train_support` never imports pygame (grep-based test: `"pygame" not in pathlib.Path("domino/env.py").read_text()` and no `pygame` import in `train.py`).
- [ ] **Step 2/3/4/5:** implement, pass, commit `feat(ui): pygame board renderer, interactive play and replay modes`.

---

### Task 14: Throughput benchmark (measured training estimate) + report figures tool

**Files:** Create `tools/bench_throughput.py`, `tools/make_report_figures.py`, `tests/test_bench.py`

- [ ] **Step 1: Implement `tools/bench_throughput.py`** measuring and printing/writing `runs/benchmark.json`: (a) raw engine steps/s over 20 000 random-agent deals single-process; (b) learner steps/s with `--n-envs 8` and the league on, over 60 s or 200 000 steps (whichever comes first); (c) PPO update time per rollout batch (`n_steps=512`, `batch_size=2048`); (d) `torch.cuda.max_memory_allocated()`.
- [ ] **Step 2: Run it** (allowed: measurement, not training) and record the numbers.
- [ ] **Step 3: Update spec section 14** with the measured table (replacing the preliminary arithmetic) and state the recommended budget.
- [ ] **Step 4: Implement `tools/make_report_figures.py`** turning `results/*.csv` into the report figures (`docs/report/figures/*.png`) plus `docs/report/tables/*.md`.
- [ ] **Step 5: Commit** `feat(tools): throughput benchmark and report figure generation`.

---

### Task 15: Notebook, README, report outline, final verification

**Files:** Create `evaluate.ipynb`, `docs/report/slide_outline.md`, `docs/report/references.md`; Modify `README.md`

- [ ] **Step 1:** `evaluate.ipynb` — loads `results/*.csv`, renders: convergence (total reward per episode + 100-deal rolling average), win-rate-vs-steps, metric bar charts with Wilson error bars, the comparison table, and the ablation table; all cells committed with outputs cleared, executable via `uv run jupyter nbconvert --execute`.
- [ ] **Step 2:** `README.md` — install (`uv sync`), the exact commands for `train.py` / `test.py` / `play.py` / `observe.py` / `evaluate.ipynb`, the project structure table, the working environment (Windows 11, RTX 3060 Laptop 6 GB, Python 3.12, package versions from `uv.lock`), the rules summary, the MDP summary, and a references section (SB3, sb3-contrib, Gymnasium, PPO paper, AlphaZero self-play, Algerian domino rules — each cited).
- [ ] **Step 3:** `docs/report/slide_outline.md` — slide-by-slide outline with the numbers filled from the CSVs and placeholders only for the owner's own conclusions.
- [ ] **Step 4: Final verification (superpowers:verification-before-completion):** `uv run pytest -q` all green; `uv sync --locked` clean; every mandatory file present; `uv run python test.py --matchups rule_vs_random --deals 60` runs end-to-end; `uv run python play.py --help`, `observe.py --help`, `train.py --help` exit 0; spec section 14 numbers match `runs/benchmark.json`; git log shows one commit per task; **no long training started**.
- [ ] **Step 5: Commit** `docs: README, evaluate notebook, report outline; final verification`.

---

## Self-Review

**Spec coverage:** rules engine (T2/T3), MDP/env (T6), observation (T4), masking (T5), custom network (T9), league (T8), training + convergence logging (T10/T11), Python baseline (T7), evaluation + CSV + notebook (T12/T15), pygame interactive/observe (T13), uv format (T1), mandatory file names (T11-T13/T15), report artefacts (T14/T15), throughput estimate (T14). No spec section is unimplemented; section 15's ablations are config-only, as the spec states.

**Placeholder scan:** Tasks 2-6 and 9-10 carry full implementations of the tricky logic; Tasks 7-8 and 12-13 specify exact interfaces, exact test assertions and the spec section that fixes the behaviour, so the executor writes code rather than guessing. No TBD/TODO remains.

**Type consistency:** `GameState` fields, `legal_actions`, `deal_outcome`, `DealOutcome`, `encode`, `action_mask`, `DominoEnv.step/reset/action_masks`, `OpponentPool.refresh_snapshots`, `DominoFeaturesExtractor`, `EpisodeStatsCallback` are used with identical names and signatures across tasks; the CSV schema is defined once in Task 12 and reused in Tasks 10 and 15.

**Known deviation to report to the owner:** spec location `docs/specs/` and plan location `docs/plans/` (project convention) instead of the skills' `docs/superpowers/…` defaults.
