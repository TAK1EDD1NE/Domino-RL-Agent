# 2v2 Domino — Reinforcement Learning Design Specification

Project: RLOC 2026/2027 — "Development of a Game-Based Environment for Reinforcement Learning"
Course: Reinforcement Learning and Optimal Control, 3CS AI & DS (N. Farhi & M. Bouali)
Submission deadline: 31/12/2026
Status: approved design, implementation not yet started.
Date: 2026-10-11

---

## 1. Purpose and scope

Build a complete, reproducible RL system around a 2v2 domino game:

1. a headless, deterministic rules engine and a Gymnasium environment,
2. a deep RL agent (MaskablePPO) whose **policy network is a custom architecture**,
3. a documented **rule-based baseline agent**,
4. a **Pygame interactive mode** (human plays) and a **Pygame observation mode** (watch the agent),
5. an evaluation pipeline comparing RL versus the baseline, with CSV outputs, plots and a notebook,
6. documentation: README, this spec, and report-ready figures/tables plus a slide outline.

Out of scope: any change to the assignment's mandatory file names; training more than one
algorithm (DQN/CFR) — they are listed in section 15 as possible future ablations only.
The user builds the slide deck themselves; we deliver figures, tables, CSVs and an outline.

---

## 2. Game rules (source of truth for the engine)

Algerian/Maghrebi 2v2 domino ("domino", not the draw variant).

* Tiles: the standard double-six set, 28 unique tiles `(a,b)` with `0 <= a <= b <= 6`.
* Players: 4, in turn order 0 → 1 → 2 → 3 → 0. Teams are seats `{0, 2}` (team A) and
  `{1, 3}` (team B). Partners sit opposite each other.
* Deal: all 28 tiles are dealt, 7 per player. **No boneyard, no drawing.**
* Play: players place tiles one at a time, extending a single line. Each played tile must
  match the pip value of one of the two open ends of the line. The first tile of a deal is
  placed as a free "lead".
* Doubles: mechanically identical to any other tile — playing `[3|3]` on a 3-end leaves the
  3-end open. There is no obligation to "close" a double and no extra end is created.
  Rendering only differs (doubles are drawn crosswise, section 11).
* The line exposes exactly **two ends**. A tile is placed on one of them, and the ends are the
  outer pip values of the line: after a free lead of `[a|b]` the ends are `a` and `b` (both `a`
  when the lead is a double).
* A move is therefore a pair **(tile, end)**. When a tile matches both ends the two placements are
  legally distinct and can lead to different positions (ends `[3,5]` with the tile `[3|5]` in hand:
  attaching to the 3-end leaves `[5,5]`, attaching to the 5-end leaves `[3,3]`). The engine
  enumerates both; when the two placements are equivalent (the ends carry the same value) only one
  is kept, so no duplicate move is ever exposed.
* Passing: a player **must** play a legal tile if they hold one. A pass is legal only when
  the player holds no tile matching either open end.
* Block: the deal is blocked when the last **four consecutive turns were all passes**
  (equivalently: no player can move).
* Round lead:
  * deal 1 of a match: the holder of `[6|6]` leads and must play `[6|6]`;
  * every later deal: the winner of the previous deal leads, and may lead with any tile.
* Scoring — hand emptied: if a player plays their last tile, that player's team banks the
  sum of the pips remaining in the two opponents' hands. No bonus of any kind.
* Scoring — blocked deal: the **individual player with the fewest pips in hand** wins the
  deal for their team, and their team banks the sum of the pips remaining in the **two
  opposing** hands.
  * If the players holding the minimum pip count all belong to the same team, that team banks
    as above.
  * If the minimum is shared by players from **both** teams, the deal is **void**: nobody
    scores, tiles are reshuffled and a new deal starts, led by the `[6|6]` holder playing `[6|6]`.
* Match: deals repeat until a team reaches **100 points**. The match score is carried between
  deals.

All rule decisions above were confirmed interactively with the project owner (see section 17).

---

## 3. RL problem formulation (MDP)

The assignment requires an explicit MDP formulation; this is the one implemented and
reported.

* **Agent**: the learner's *team* (both seats of team A). One parameterised policy plays
  every seat it controls, in a seat-relative representation.
* **Episode**: one deal. Episodes always terminate (a player empties their hand, or the deal
  blocks). There is no time-limit truncation.
* **Horizon**: at most 28 tile plays plus passes; the learner's team makes at most ~14
  decisions per deal.
* **State (observation)**: a 192-dimensional float32 vector containing only public
  information plus the acting seat's own hand (section 4). It is *seat-relative*: the same
  vector layout describes the situation from the point of view of whichever controlled seat
  is about to act, so one network can play both seats.
* **Action**: an integer in `Discrete(56)` encoding the move `(tile, end)`: `action = 2 tile_id + end`
  with `end ∈ {0 = left, 1 = right}` (section 5).
* **Action mask**: a 56-dimensional boolean vector; `False` for any tile that is not in the
  acting seat's hand or is not playable on either open end (and, when leading, any tile not
  in hand). Whether the acting seat can play at all is decided by the engine: a turn with no
  legal tile is a **forced pass and is not a decision point** for the agent (there is no
  choice to learn), so it never appears as a transition.
* **Reward**: points banked by the agent's team at the end of the deal
  (`+` sum of the opponents' remaining pips, `0` on a void deal, `0` for every non-terminal
  transition). Rewards are **raw points, undiscounted, no shaping** — the assignment's
  convergence metric (total reward per episode) is then directly "points per deal".
  Reward mode is configurable (`differential` default | `banked`); the environment applies it,
  the engine always reports the points banked by the winning team (see addendum A).
* **Discount**: `gamma = 0.99` by default (exposed as a CLI flag; 1.0 is defensible for a
  finite-horizon episodic task and is discussed in the report). SB3's advantage
  normalisation handles the reward magnitude (returns are order 10-60 points), which is why
  no reward scaling is applied.
* **Termination**: deal end. The terminal transition carries the deal reward.
* **Partial observability**: the agent never sees opponents' or partner's tiles. Beliefs are
  left to the network; the observation supplies the full public history (who played which
  tile, pass history, per-seat tile counts, per-pip-value liveness) from which the network can
  infer what remains. This is stated explicitly as a POMDP approximated by a reactive policy
  on a rich public-history state (no recurrent policy in v1).

---

## 4. Observation vector (192 dims, float32, seat-relative)

Let `me` be the controlled seat about to act; the four seats are ordered relative to `me` as
`[self, left, partner, right]` where `left = me+1`, `partner = me+2`, `right = me+3` (mod 4).
`left` and `right` are always opponents, `partner` is always the team-mate.

| slice     | dims | content |
|-----------|------|---------|
| 0:28      | 28   | my hand — 1 for each tile id I hold |
| 28:56     | 28   | tiles already played this deal (1 = on the table) |
| 56:168    | 112  | `played_by[4][28]` — which relative seat played which tile (public information) |
| 168:169   | 1    | my remaining pips / 168 |
| 169:170   | 1    | unseen pips (168 − my pips − pips on the table) / 168 |
| 170:171   | 1    | current consecutive-pass streak / 4 |
| 171:175   | 4    | whether each relative seat passed on its most recent turn |
| 175:179   | 4    | tile count in each relative seat's hand / 7 |
| 179:183   | 4    | one-hot: which relative seat led this deal |
| 183:184   | 1    | 1 if I am the deal leader (I lead the deal) |
| 184:191   | 7    | per pip value `v`: 1 if any unseen tile with value `v` is still live |
| 191:192   | 1    | 1 if the board is empty (I am leading right now) |

Deliberately excluded, with reasons:

* **Match score / deal index** — episodes are single deals, so a running match score would be
  a constant/never-trained feature that only appears at evaluation time (distribution shift).
  Consequence, documented as a limitation: the learned policy maximises expected points per
  deal, which is not identical to optimal match play near the 100-point cap.
* **Partner's pip count / any opponent hand content** — not observable. Only public
  information is provided; observability is not faked.
* **Action mask as an input feature** — it is passed separately through MaskablePPO's
  `action_masks()` API and consumed by the policy's masked distribution, so duplicating it in
  the observation would be redundant. The custom extractor still uses the hand slice.

Tile id convention: `id(a,b) = a*7 + b - a*(a+1)/2` for `a <= b`, giving ids 0..27
(`[0|0] = 0`, `[0|6] = 6`, `[1|1] = 7`, …, `[6|6] = 27`). One shared constant table is
imported everywhere; the mapping is a unit test.

---

## 5. Action space and masking

* `gym.spaces.Discrete(56)`; `action = 2 * tile_id + end_index`, where `end_index` is
  `0 = left end`, `1 = right end`. A lead (empty board) uses `end_index = 0` only.
* The engine exposes `legal_moves(state, seat) -> list[tuple[int, int]]` — `(tile_id, end_index)`
  pairs for every placement that is legal, deduplicated when the two placements would produce the
  same position — plus the tile-level convenience view `legal_actions(state, seat) -> list[int]`
  used by the rule bot, the Pygame UI and the diagnostics. Tiles that match no open end are never
  listed; the forced opening lead of deal 1 (and of a redeal after a void deal) yields exactly the
  one move `(id(6,6), 0)`.
* `action_masks()` returns a `np.bool_` array of length 56 built from `legal_moves`, which is
  exactly the interface `MaskablePPO` expects. Illegal logits are set to `-inf` by SB3.
* Pass actions are **not** in the action space (see section 3).
* Invariant enforced by a test: `mask.sum() >= 1` at every decision point, and the sampled
  action is always in `legal_moves`.

---

## 6. Environment architecture

Two layers, so the rules are testable without RL and the RL wrapper stays thin:

```
domino/engine.py     pure-Python rules engine (no gym, no torch, no numpy requirement)
domino/obs.py        state -> 192-dim observation encoder (numpy)
domino/masks.py      state -> action mask
domino/env.py        gymnasium.Env: turn loop, league opponents, reward, seat-relative view
domino/agents/       acting policies behind one interface: act(state, seat) -> action
domino/ui/           pygame rendering + interactive loop (never imported by training code)
```

`domino/engine.py` holds `GameState` (hands, table ends, table tiles in play order,
`played_by`, pass streak, deal winner, running match score, RNG seed) and pure functions
`deal()`, `legal_actions()`, `apply_action()`, `deal_result()`, `next_leader()`,
`scores()`. It is deterministic given a seed and free of global state; every mutation returns
a new state (or an explicit in-place `apply_action` with documented semantics) so it can be
unit-tested exhaustively.

`domino/env.py` implements the **learner-team MDP**:

* `reset()` deals a new deal, assigns the opponent team for this deal through the
  `OpponentPool`, sets the leader, and advances to the first decision point of the learner's
  team (opponent turns and forced passes are resolved internally).
* `step(action)` verifies the action is legal (assert in dev, mask in production), applies it,
  then advances the game; every opponent turn is played immediately by the assigned opponent
  policy; consecutive forced passes are consumed silently; when the deal ends the terminal
  reward is returned; otherwise the next learner decision point is returned.
* `action_masks()` for the acting learner seat.
* `info` carries per-deal diagnostics used by logging and evaluation: `deal_winner`,
  `deal_points`, `blocked`, `void_deal`, `deal_steps`, `deals_played`, `match_scores`.
* Options: `{"max_deals_per_episode": 1, "opponent": "league"|"self"|"rule"|"random"|"mixed",
  "seed": int}`.

`domino/agents/` — one small interface `act(state, seat) -> int` with implementations:

| agent  | implementation |
|--------|----------------|
| `RandomAgent` | uniform over legal actions |
| `RuleAgent`  | documented heuristic, section 10 |
| `PolicyAgent`| wraps an sb3 `MaskablePPO` model (current learner or a frozen snapshot) |
| `OpponentPool` | samples the opposing team's agent per deal from configurable weights |

---

## 7. Custom neural network architecture (graded requirement)

SB3 is used, but the network is ours — a two-tower **set encoder** that exploits the structure
of the game (a hand is a *set* of tiles, plus a fixed-size context):

```
hand  : 28-dim binary  -> per-tile embedding table E[28, 32]
                       -> shared tile MLP   (32 -> 64, ReLU)
                       -> masked mean-pool and masked max-pool  -> 128
context: remaining 164 context features
                       -> MLP (164 -> 256, LayerNorm, ReLU)     -> 256
trunk  : concat(128 + 256 = 384)
                       -> Linear 384->256, LayerNorm, ReLU
                       -> Linear 256->256, LayerNorm, ReLU
                       -> Linear 256->128, LayerNorm, ReLU      -> 128 features
heads  : SB3 policy head Linear(128->56)  (masked at sampling time)
         SB3 value  head Linear(128->1)
```

Implementation: a `TorchFeaturesExtractor` subclass registered as
`features_extractor_class` plus a custom `MaskableActorCriticPolicy` subclass that builds the
trunk with LayerNorm and orthogonal initialisation (gain √2) instead of SB3's default
`[64, 64]` MLP with Tanh. Parameter count is roughly 0.3 M — small on purpose, so the GPU is
never the bottleneck and training stays in the minutes-to-hours envelope.

Justification for the report: the hand slice is a set with variable cardinality (7 → 1 tiles),
which a fixed-size MLP over a 28-bit mask handles poorly as the endgame arrives; a shared
per-tile embedding with masked pooling is cardinality-agnostic, and it lets the same weights
be reused for the "which tiles are still live" structure. The context tower keeps board and
history features out of the embedding path so they are not diluted by pooling.

---

## 8. Opponent league (training)

The learner always occupies team A's two seats. Team B is drawn per deal from an
`OpponentPool` with configurable weights (default):

| opponent | default weight | notes |
|----------|----------------|-------|
| `self` (current learner policy) | 0.50 | mirror self-play, the main engine of improvement |
| frozen snapshots | 0.20 | last K = 10 checkpoints, sampled uniformly |
| rule-based bot | 0.20 | guarantees the trained agent has faced the evaluation opponent |
| random agent | 0.10 | cheap diversity, punishes degenerate policies |

* Snapshots are written every `--snapshot-every` PPO updates (default: every 100 000 env
  steps) to `models/snapshots/policy_<steps>.zip`; the pool keeps the newest 10.
* Workers are notified with `vec_env.env_method("refresh_pool")` on an SB3 callback; each
  worker loads snapshots lazily and caches at most 3 models (they are tiny).
* A deal is a coherent unit: both of team B's seats use the same opponent type by default
  (`--mix-opponents` allows per-seat mixing for an ablation).
* `--opponent self` reproduces pure mirror self-play (section 15 ablation A1).

Known cost: opponent inference runs on CPU inside the vector workers and lowers throughput by
roughly 10-25 % versus mirror-only self-play; measured in the throughput benchmark.

---

## 9. Training pipeline

* Algorithm: `MaskablePPO` from `sb3-contrib` (chosen over DQN: native action masking for our
  variable legal-move sets, on-policy stability, no replay-buffer plumbing for masks, and it
  is the framework explicitly blessed by the assignment).
* Vectorisation: `SubprocVecEnv` with `--n-envs 8` (12 logical cores; leaves headroom for the
  trainer process and GPU feed). `spawn` start method with a `__main__` guard (Windows).
* Defaults: `n_steps=512`, `batch_size=2048`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`,
  `clip_range=0.2`, `ent_coef=0.01`, `lr=3e-4` (linear schedule), `vf_coef=0.5`,
  `max_grad_norm=0.5`, `target_kl=0.02`. All overridable by CLI/config file.
* Checkpointing: `models/checkpoints/ppo_<steps>_steps.zip` every `--ckpt-every` steps
  (default 500 000) plus `latest.zip`; `--resume` restores the observation config, RNG state
  and step counter.
* Seeding: `--seed` controls numpy, torch, gym and the engine RNG; every run logs its seed.
* Logging (assignment requirement 4): TensorBoard at `runs/<run_name>/` containing
  * `train/total_reward_per_episode` — undiscounted total reward of every finished deal,
  * `train/avg_episode_reward_window_100` — sliding-window average (the assignment's
    "average episode reward for a given sliding window"),
  * SB3's own `rollout/ep_rew_mean`, `train/entropy_loss`, `train/policy_gradient_loss`,
    `train/value_loss`, `train/explained_variance`, `time/fps`,
  * custom deal diagnostics: `deals/block_rate`, `deals/void_rate`, `deals/points_per_deal`,
    `deals/win_rate_vs_<opponent>` (rolling).
  Implemented with a `StatsCallback` that reads `info`/`ep_info_buffer` (SB3 reports
  undiscounted episode returns, so this is exact rather than discounted).
* Plus a CSV mirror of the convergence series at `results/training_<run>.csv` (step,
  episode, total reward, window average) so `evaluate.ipynb` does not depend on TB.

---

## 10. Rule-based baseline agent (documented strategy)

A transparent, hand-tuned evaluation function over legal moves, in the spirit of classic
domino play. Every component is a named, documented weight in a dataclass so the report can
show the exact decision-making strategy.

Opponent model maintained by the bot (same public information the RL agent receives):

* `live[v]` — count of unseen tiles carrying pip value `v` (from `168`, minus its hand, minus
  tiles on the table);
* `opp_short[v]` / `partner_short[v]` — same quantity per relative seat, using tiles each seat
  has already shown (a seat that has not played value `v` while many `v` tiles are gone is
  likely short);
* own hand composition, table ends, pass streak, and the block situation.

Scoring of each legal move (higher is better), with default weights:

| term | weight | intent |
|------|--------|--------|
| pips shed (value of the tile) | +0.5 | get heavy tiles out of your hand, they are what you lose on a block |
| kills a value (after the move, that end's value is dead table-wide) | +9.0 | strangle the table, force passes |
| end value the opponents are short of | +6.0 | make them draw from a shrinking set |
| keeps my doubles in hand | +0.5 | doubles are the most flexible tiles, useful as end-switchers later |
| protects the partner (does not open a value the partner is short of) | +1.5 | team play: do not hand the partner an impossible end |
| reduces my expected loss on a future block (`my_pips` vs table minimum) | +2.0 | if a block looks close, become the lowest-pip player |
| leading a value I hold a duplicate of | +1.5 | keeps both ends available |
| forced opening `[6|6]` | n/a | the only legal move in deal 1's lead |

These weights were chosen by the documented sweep `tools/sweep_rule_weights.py` (coordinate
ascent against a random opponent on evaluation seeds disjoint from the win-rate gate's), not
by hand; the previous hand-chosen values were `+1.0 / +3.0 / +2.0 / +0.5 / +1.5 / +2.0 / +1.5`.

Block-awareness: if `sum(live) <= 12` or the pass streak is ≥ 2, the "reduce my expected loss"
and "kills a value" terms are doubled, because the deal is likely to end blocked and the
fewest-pips player's team takes the pot.

Leading: among hand tiles, prefer a duplicate-valued tile with the highest pip count unless the
block terms dominate.

The bot is described in the report with this exact table, plus worked examples from
`test.py` logs. It is deliberately strong (the comparison is only meaningful if the baseline
is a real opponent) and it is also one of the league opponents during training.

---

## 11. Pygame interfaces

Two modes, both headless-safe (`SDL_VIDEODRIVER=dummy` for smoke tests and for any CI-like
run); training never imports pygame.

`play.py` — human plays seat 1 (`--human-seat` configurable).

* `--partner rl|rule` (default `rl`), `--opponents rl|rule|random` (default `rl`),
  `--model models/checkpoints/latest.zip`, `--seed`, `--match-to 100`.
* Board rendered as a single line; doubles drawn crosswise (perpendicular) exactly as on a
  physical table, with a small gap marker so the line stays readable.
* Legal tiles highlighted, illegal ones dimmed; clicking a tile with a single legal placement plays it
  immediately, while clicking an ambiguous one (matching both ends) highlights both target ends and the
  second click chooses the end; a "pass" indicator is shown automatically when the human is stuck
  (mandatory-play rule).
* Info panel: hands' tile counts, table ends, pass streak, remaining points, deal and match
  scores, and a text line reporting the last action of each AI seat.
* Buttons/keys: new deal, restart match, toggle AI autoplay for the human seat (useful for a
  live demo), quit.

`observe.py` — replays one deal played by the trained agent (or `--agent rule`) with no human
input: step-by-step (`space`) and continuous playback (`p`, `+`/`-` for speed) plus an overlay
showing the acting seat's policy over legal moves (probability bars per legal tile) and its
value estimate. This doubles as the "explain the agent" artefact for the defence.

Tile rendering: a rounded rectangle divided by the middle bar, pips drawn as circles, doubles
crosswise, team colour on the frame (team A / team B), the acting seat outlined.

---

## 12. Evaluation protocol and metrics

`test.py` runs head-to-head series with seed control. Default matrix (`--deals 1000` per
matchup, both seat assignments balanced):

| matchup | purpose |
|---------|---------|
| RL(team A) vs Rule(team B) | the headline comparison required by the assignment |
| RL vs Random | sanity floor |
| Rule vs Random | establishes that the baseline is genuinely strong |
| RL vs Rule (RL as team B) | removes any first-player / seat bias |
| Rule vs Rule | variance reference for significance |
| RL(early ckpt) vs RL(final) | learning progress, optional |

Metrics (each justified in the report):

1. **Win rate per deal** (and Wilson 95 % confidence interval) — the cleanest strength measure;
   deals are the natural independent trial.
2. **Points per deal (mean)** — captures margin, not only win/lose; directly the reward the
   agent optimises.
3. **Match win rate to 100 points** — the human-meaningful score, at a smaller sample size.
4. **Block rate and void-deal rate** — shows whether the agent learned the block endgame at
   all, and whether the tie rule is exercised.
5. **Mean pips left in hand at deal end** — measures efficiency in the blocked endgame
   specifically.
6. **Decision latency (ms/action)** — inference cost, relevant for the interactive mode.
7. **Sample efficiency**: deals needed to reach 50 % win rate vs the rule bot (from the
   training CSV) — the classic RL comparison axis.

Outputs: one row per (matchup, seed-batch) in `results/eval_<timestamp>.csv` with a fixed
schema (matchup, agent_a, agent_b, deals, wins_a, wins_b, win_rate_a, ci_low, ci_high,
points_per_deal_a, points_per_deal_b, block_rate, void_rate, match_win_rate_a, ms_per_action,
model_path, seed). `evaluate.ipynb` reads the CSVs and produces: convergence curve (total
reward per episode with the 100-deal window average), win-rate-vs-training-steps curve, metric
bar charts with error bars, the metric table, and the ablation table.

---

## 13. Tests and verification (TDD)

Written before the implementation they cover. `pytest`, run with `uv run pytest`.

Engine (`tests/test_engine.py`):
1. tile id mapping covers 0..27 exactly once;
2. a deal gives 7 tiles to each of 4 seats and loses none;
3. pip conservation invariant (hands + table = 168) after every action;
4. `legal_actions` correctness: matching either end, free lead, forced `[6|6]` lead;
5. mandatory-play: passing is impossible when a legal tile exists;
6. block detection after exactly 4 consecutive passes (and not before);
7. scoring, emptied hand: opponents' pips banked, own team only;
8. scoring, block, unique minimum: fewest-pips player's team banks both opponents' hands;
9. scoring, block, tie inside one team: same as 8 (no void);
10. scoring, block, tie across teams: void deal, no points, redeal led by `[6|6]` holder;
11. lead of deal 2+ goes to the previous deal's winner; match ends at 100;
12. determinism: same seed ⇒ identical deal and identical action sequence under a random agent.

Environment (`tests/test_env.py`): obs shape/dtype/range; mask has at least one legal action at
every decision point; `step` asserts on masked-out actions; reward is exactly the banked points
and zero on every non-terminal step; a forced pass never yields a transition; deal length is
bounded; `gymnasium.utils.env_checker` and SB3's `check_env` pass; league sampling is
deterministic under a seed; snapshot loading/reloading works; `SubprocVecEnv` with 2 workers
produces identical episodes to a single env under the same seed.

Network (`tests/test_policy.py`): the custom extractor + policy instantiate through
`MaskablePPO`; forward pass shapes; masked logits are `-inf` for illegal actions and the sampled
action is legal; gradients reach every submodule.

Scripts (`tests/test_scripts.py`): `train.py --steps 4096` smoke run completes and writes a
checkpoint + TB event file; `test.py --deals 20` writes a valid CSV row; `play.py` and
`observe.py` construct their pygame surface under `SDL_VIDEODRIVER=dummy` and simulate N
frames with synthetic events.

Definition of done for the build: all tests green, one short training run demonstrating the
pipeline, and a measured throughput benchmark (section 14) — with no long training run
started, since the training budget is the owner's decision.

---

## 14. Training-time estimate

Method: measure, do not guess. `tools/bench_throughput.py` measures (a) raw engine
steps/second single-process, (b) end-to-end learner steps/second with 8 `SubprocVecEnv`
workers and the league active, (c) PPO update time per rollout batch, (d) GPU utilisation.
Results go to `runs/benchmark.json` and the table below is replaced by the measured numbers
before any budget decision is taken.

Preliminary arithmetic (to be replaced):

* The policy has ~0.3 M parameters and the observation is 192-dim, so gradient work is trivial
  for an RTX 3060 Laptop (6 GB) — memory is not a constraint and the GPU is not the bottleneck.
* The bottleneck is the Python env: the engine is pure Python with precomputed legal moves,
  so expect roughly 3 000-15 000 learner steps/s per worker, i.e. **~25 000-120 000 steps/s
  aggregate with 8 workers** (minus 10-25 % for league opponent inference).
* One deal is ~20-30 raw moves, so ~10-14 learner decisions; 1 M learner steps ≈ 80 000-100 000
  deals.

| steps (learner) | aggregate 25 k steps/s | aggregate 60 k steps/s | aggregate 120 k steps/s |
|-----------------|------------------------|------------------------|-------------------------|
| 10 M            | ~6.7 min               | ~2.8 min               | ~1.4 min                |
| 30 M            | ~20 min                | ~8.3 min               | ~4.2 min                |
| 100 M           | ~67 min                | ~28 min                | ~14 min                 |
| 300 M           | ~3.3 h                 | ~83 min                | ~42 min                 |

Convergence expectation: with masking, a 0.3 M-parameter net, terminal deal-level rewards and a
league of opponents, a first competent policy (beats random decisively, competitive with the
strong rule bot) should appear in the 10-50 M step range; beating the rule bot reliably is
plausible in the 50-300 M range. Budget recommendation: plan a 100-300 M step run (~15 min to
3.3 h at the measured rate, single overnight run at worst) instead of an open-ended one, and
keep checkpoints every 500 k steps so any stopping point is usable.

---

## 15. Ablations and future work (report material, off by default)

* A1 self-play only vs league (`--opponent self` vs `--opponent league`).
* A2 reward shaping off (default) vs a small pip-penalty variant.
* A3 architecture: custom set encoder vs a default SB3 MLP of similar size.
* A4 `gamma` 0.99 vs 1.0.
* A5 opponent mixing within a deal.
* A6 reward mode: `differential` (default) vs `banked`.
* A7 rule-bot weight sweep: hand-chosen defaults vs swept weights (tool: tools/sweep_rule_weights.py).
Not to be run unless the training budget allows; the harness supports them from day one
because they are config values, not code branches.

---

## 16. Risks and limitations (for the report's honest discussion)

1. **Sparse rewards**: only the terminal transition carries signal, so credit assignment over
   ~10-14 decisions is hard; the league mitigates it, and ablation A2 quantifies it.
2. **Deal-level optimality ≠ match-level optimality** (match score deliberately excluded from
   the observation).
3. **Non-stationary self-play**: the league exists because opponents change as the learner
   does; snapshot staleness is a knob, not a solved problem.
4. **Hidden information is not explicitly modelled**: no belief-state or recurrent layer in v1;
   the network sees public history only.
5. **Rule bot is heuristic, not optimal** — the comparison bounds strength above by the bot's
   own ability.
6. **Windows specifics**: `SubprocVecEnv` uses spawn (slower startup, needs the `__main__`
   guard), pygame needs the dummy video driver for headless smoke tests.
7. **Disk**: ~2.5-3 GB of wheels in `.venv`; the machine had 25 GB free at design time —
   checkpoints and TB logs are kept small (models are ~1 MB each).

---

## 17. Decision log (interactive answers that fix the design)

| # | decision |
|---|----------|
| Q1 | Algerian 2v2 ruleset, with the owner's blocked-deal modification |
| Q2 | Deep RL self-play, one policy over all 4 seats |
| Q3 | sb3-contrib + Gymnasium |
| Q4 | All four deliverable groups: training core, evaluation suite, playable demo, report docs |
| Q5 | Training-time estimate to be delivered, budget decided afterwards |
| Q6 | Match to 100 points; deal 1 led by `[6|6]` holder; later deals led by the previous winner |
| Q7 | Block tie inside a team: that team banks the opposing two hands; tie across teams: void deal, redeal, `[6|6]` holder leads |
| Q8 | Emptied hand: team banks both opponents' pips, no bonus |
| Q9 | Rich observation + full public history |
| Q10 | Round-level team reward, raw points, no shaping |
| Q11 | Doubles behave as normal line moves (same end value); crosswise rendering only |
| Q12 | Round 1 lead is the forced `[6|6]`; later leads are free |
| Q13 | Mandatory play; block after 4 consecutive passes |
| Q14 | Graded coursework: RLOC 2026/2027, deadline 31/12/2026, mandatory file names, uv format, custom network required |
| Q15 | Playable demo against the AI (Pygame, per the assignment's interactive mode) |
| Q16 | Custom architecture: set encoder + context tower + trunk |
| Q17 | Episode = one deal |
| Q18 | League self-play as the default training setup |
| Q19 | MaskablePPO |
| Q20 | Pygame: human plays seat 1, partner is the trained net, opponents behind flags |
| Q21 | Strong, documented rule-based heuristic |
| Q22 | Design approved as written: spec → implementation plan → build |
| Q23 | Create the uv environment now; local git repo with commits, never a push without asking |
| Q24 | Report artefacts: figures, tables, CSV results and a slide-by-slide outline; the owner builds the deck |

---

## 18. Assignment traceability

| assignment requirement | where satisfied |
|------------------------|-----------------|
| build the game with a suitable framework (Pygame or similar) | sections 6, 11 (`pygame` for UI; pure-Python engine for logic) |
| interactive mode with a graphical interface | section 11, `play.py` |
| headless mode for training/testing | sections 3, 6, 9 (engine/env import no pygame) |
| train an RL agent with a DRL algorithm of choice | section 9 (`MaskablePPO`) |
| convergence via total reward per episode (undiscounted), TensorBoard, sliding-window average | section 9 (custom callback, exact undiscounted returns) |
| design and implement a rule-based agent | section 10 |
| compare RL vs rule-based with several justified metrics, CSV output, plots/tables in a notebook | sections 12, 13, `test.py`, `evaluate.ipynb` |
| report: game description, MDP, algorithm/architecture/hyperparameters, baseline strategy, code description, results, references | sections 2, 3, 7, 9, 10, 12 + slide outline artefact |
| uv project format reproducible with `uv sync` | section 19, `pyproject.toml` + `uv.lock` |
| `train.py` / `test.py` / `play.py` / `observe.py` / `evaluate.ipynb` / `README.md` | section 19 |
| customized neural network architecture, not a predefined one | section 7 |
| cite resources that inspired the design | README references section + report references |

## 19. Repository layout (final)

```
project/
├── pyproject.toml, uv.lock, README.md
├── train.py            # trains the RL agent (CLI, config-driven)
├── test.py             # evaluates RL vs rule vs random, writes results/*.csv
├── play.py             # Pygame interactive mode (human + AI seats)
├── observe.py          # Pygame replay of one deal by the trained/rule agent
├── evaluate.ipynb      # plots and tables from results/*.csv
├── domino/             # package: engine, env, obs, masks, agents/, ui/, config.py
├── configs/            # yaml presets (default, quick_smoke, league_full)
├── tools/              # bench_throughput.py, make_report_figures.py
├── tests/              # pytest suite
├── docs/specs/         # this specification + report slide outline
├── runs/               # tensorboard logs (git-ignored)
├── models/             # checkpoints + snapshots (git-ignored)
└── results/            # eval CSVs, training CSVs, figures (CSVs/figures kept, big files ignored)
```

---

*Spec self-review: no placeholders, no TBDs; every rule, tensor slice and metric above is
fully specified; the only quantity intentionally deferred is the measured throughput number in
section 14, which has a stated measurement method and will regenerate the table.*
