# Domino-RL — a 2v2 domino environment and a self-play deep-RL agent

Project for **Reinforcement Learning and Optimal Control 2026/2027** (3CS AI & DS,
N. Farhi & M. Bouali). A 2v2 Algerian domino game is implemented as a Gymnasium
environment; a MaskablePPO agent with a **custom network architecture** is trained
by league self-play against frozen snapshots of itself, a documented rule-based
bot and random play; the trained agent is then compared with the rule-based
baseline on several justified metrics.

Everything is reproducible with `uv sync`, and the same rules engine drives the
headless training path, the evaluation scripts and the Pygame interfaces.

---

## 1. Quick start

```bash
uv sync                                  # creates .venv and installs the pinned stack
uv run pytest -q                         # the full test suite (see section 6)

uv run python train.py --dry-run         # print the resolved config + parameter count
uv run python train.py --steps 4096 --n-envs 2 --run-name smoke   # a short smoke run
uv run python test.py --matchups all --deals 300 --out results/eval.csv
uv run python play.py --partner rl --opponents rl        # play a match yourself (Pygame)
uv run python observe.py --agent rl --deals 2            # watch the agent, with its policy overlay
uv run jupyter lab evaluate.ipynb                        # plots and tables from the CSVs
```

## 2. The game (rules implemented by `domino/engine.py`)

* 28 tiles (`[0|0]`…`[6|6]`), 7 dealt to each of four seats, **no boneyard**.
* Teams are seats `{0, 2}` vs `{1, 3}`; partners sit opposite each other.
* A player **must** play if a tile matches either open end; a pass is only legal
  when genuinely stuck. A deal **blocks** after four consecutive passes.
* The line exposes exactly **two ends**; a move is a **(tile, end)** pair. A double
  played on an end leaves that end's value unchanged (it is drawn crosswise, as on
  a real table). When a tile fits both ends the two placements are different moves
  — they decide which value stays open — and the action space keeps them distinct.
* Deal 1 is opened by the holder of `[6|6]`, who must play it; every later deal is
  opened with a free lead by the previous deal's winner.
* **Scoring.** Hand emptied → that team banks the two opponents' remaining pips.
  Deal blocked → the *individual* with the fewest pips wins it for their team,
  banking the opposing pair's pips; if the fewest-pip players span both teams the
  deal is **void** (no points, redeal, `[6|6]` holder opens). Matches run to 100.

## 3. The RL formulation

* **Agent** — the learner's team; one policy plays both of its seats from a
  seat-relative view.
* **Episode** — one scored deal (~10-14 decisions for the team).
* **Observation** — 192 float32 features (`domino/obs.py`): own hand, tiles played,
  who played what, per-seat tile counts, open ends, pass history, per-value
  liveness. Public information only — no opponent or partner hand contents.
* **Action** — `Discrete(56)`: `2 * tile_id + end_index`. Illegal moves are masked
  by `MaskablePPO`; a forced pass is not a decision point at all.
* **Reward** — the deal's points, undiscounted, on the terminal transition only:
  `+points` for the winning team and `-points` for the losing team
  (`--reward-mode differential`, the default), or `0` for the loser
  (`--reward-mode banked`).
* **Network** — a custom two-tower set encoder: a per-tile embedding with masked
  mean/max pooling over the hand, a context tower over the remaining 164 features,
  a LayerNorm trunk, and policy/value heads (≈252 k parameters). This replaces
  SB3's default `[64, 64]` MLP, as the assignment requires.
* **Training** — `MaskablePPO` (sb3-contrib) on 8 vectorised environments. Because
  a live policy cannot be shared into spawned workers, the league's self-play
  opponent is the **newest frozen snapshot** (`{"snapshot": 0.6, "rule": 0.25,
  "random": 0.15}`), refreshed every `--snapshot-every` steps; a live `self`
  opponent is available in single-environment mode (`--n-envs 1`).

## 4. Files (the assignment's mandatory names are the root scripts)

| path | what it does |
|---|---|
| `train.py` | trains the agent (CLI, config-driven, checkpoints, resume, TensorBoard) |
| `test.py` | evaluates RL vs rule vs random: win rate + Wilson CI, points/deal, match series, block/void rates, pips left, latency → `results/eval_*.csv` |
| `play.py` | interactive Pygame mode: you play one seat, the AI fills the others |
| `observe.py` | Pygame replay of a deal with the agent's policy/value overlay |
| `evaluate.ipynb` | plots and tables built from the CSVs |
| `domino/engine.py` | the rules engine (deterministic, dependency-free) |
| `domino/obs.py`, `domino/masks.py` | observation encoder and action masking |
| `domino/env.py` | the Gymnasium environment (learner-team decision points) |
| `domino/agents/` | `RandomAgent`, `RuleAgent` (documented heuristic), `PolicyAgent`, `OpponentPool` |
| `domino/policy.py` | the custom network and the MaskablePPO policy class |
| `domino/callbacks.py` | convergence logging (TensorBoard + CSV) and league snapshots |
| `domino/ui/` | the Pygame renderer and the two UI loops |
| `configs/` | `default.yaml`, `quick_smoke.yaml`, `league_full.yaml` |
| `tools/` | `sweep_rule_weights.py`, `bench_throughput.py`, `make_report_figures.py` |
| `docs/specs/`, `docs/plans/` | the design specification and the implementation plan |
| `docs/report/` | generated figures, metric tables and the slide outline |
| `tests/` | the pytest suite |

Artefacts: TensorBoard logs in `runs/<run-name>/`, checkpoints in
`models/checkpoints/` (`latest.zip`, `ppo_<steps>_steps.zip`), league snapshots in
`models/snapshots/`, CSV results in `results/`, figures/tables in `docs/report/`.

## 5. Rule-based baseline

`domino/agents/rule_agent.py` scores every legal placement with seven named terms
(shed pips, kill a value, prefer ends the opponents are short of, keep doubles,
protect the partner, block economics, duplicate leads) and doubles the block terms
when a block looks close. The weights were chosen by a **documented sweep**
(`tools/sweep_rule_weights.py`, seeds disjoint from the test gate's), which raised
its mean win rate against random play from 0.5527 to 0.5853 on held-out seeds.

## 6. Tests

`uv run pytest -q` covers: the tile table; the rules engine (legality, mandatory
play, blocking, every scoring path, void redeals, match to 100); the observation
layout slice by slice; action masking; the agents and the league (weighted
sampling, snapshot LRU); the environment (mask legality, reward accounting under
both modes, forced passes, determinism, `check_env`, spawn-safety, no
torch/pygame in the training path); the network (shapes, masked legality over 200
decision points, gradient flow, cardinality-agnostic pooling); the callbacks
(CSV/TensorBoard convergence series, snapshot delivery); the CLI scripts (a
4096-step subprocess smoke run with resume fidelity); the evaluation script; and
the Pygame modes (headless, SDL dummy driver).

## 7. Working environment

Windows 11, Python 3.12, 12 logical CPUs, NVIDIA RTX 3060 Laptop (6 GB, CUDA 12.1).
Pinned by `uv.lock`: torch 2.5.1+cu121, stable-baselines3 2.7.1, sb3-contrib 2.7.1,
gymnasium 1.2.3, pygame 2.6.1, tensorboard 2.21, pandas 3.0, matplotlib 3.11.

## 8. Honest limitations

* Rewards are terminal-only, so credit assignment across a deal is sparse; the
  league mitigates it, and reward shaping is an ablation that was not run.
* The learned policy optimises points per deal, not match-level play near the
  100-point cap (the match score is deliberately not an observation feature).
* The environment is Python-bound: measured throughput is a few hundred learner
  steps/s, so the reported training budget is hours, not minutes (see
  `runs/benchmark.json` and the design spec's training-time section).
* A tile that fits both ends is a genuine choice; the engine keeps both placements
  distinct, but a rule-bot strength measurement that goes through tile-only
  adapters cannot exercise the bot's end choice.
* Hidden information is handled by giving the network the full public history, not
  by an explicit belief model.

## 9. References

* Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — the PPO objective.
* Raffin et al., *Stable-Baselines3* (JMLR 2021) and *sb3-contrib* — the
  implementations used, with our own policy and feature extractor.
* Huang et al., *A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms* (2022) — why masked logits are set to a large negative constant.
* Silver et al., *Mastering the game of Go without human knowledge* (2017) — the
  self-play/opponent-pool idea the league follows.
* Gymnasium documentation — the environment API contract.
* Algerian/Maghrebi 2v2 domino rules as specified by the project owner (recorded in
  `docs/specs/2026-10-11-domino-2v2-rl-design.md`, section 2).
