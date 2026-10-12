# v3 architecture plan — 9 changes in one pass (2026-10-12)

Owner's decision: implement all nine proposed changes together, debug afterwards. This file fixes
the interfaces so the units don't fight each other, and records which judgement calls were made
where the proposal left a choice.

## Observation layout v2 (the one interface everything touches)

The observation grows from 192 to a single flat vector that carries a **public part** (what the
policy may act on) and a **privileged part** (training-time critic only). Consumers slice it:

```
0:192     public v1 (unchanged: hand, table, played_by, pips, passes, counts, leader, liveness)
192:213   void-lack matrix — 3 relative opponents x 7 pip values, 1 = "known to lack v"   [item 2]
213:219   match features — my score/100, opponent score/100, difference/100, deal_no/5,
          leads-the-match flag, trails-the-match flag                                      [item 6]
219:423   ordered last-6 moves, flattened: per move (seat one-hot 4, tile one-hot 28,
          end 2) -> 34 x 6 = 204                                                          [item 4]
423:507   privileged: the three opponents' true hands, 3 x 28 binary                       [item 1]
PUBLIC_DIM = 423, OBS_DIM = 507
```

(The first draft of this plan said 501 with slightly different offsets; the implemented layout
above is the authoritative one — 219:423 for moves and 423:507 for the privileged block.)

- `PUBLIC_SLICE = slice(0, 417)` — the actor path consumes exactly this. The spec's rule
  "policy sees public information plus its own hand" stays true and is asserted by a test.
- `PRIVILEGED_SLICE = slice(417, 501)` — the value network consumes the whole vector
  (`share_features_extractor=False`, a second extractor with its own slice), so the critic is
  centralised while the actor stays decentralised (MAPPO pattern).
- The belief head is supervised from `obs[417:501]`, so no extra `info` plumbing for labels.
- Item 7 (double redundancy) is already implemented: `GameState._tile_placements` collapses
  placements that produce the same position, and the mask is built from `legal_moves`. A test
  locks that in; no code change beyond the regression test.
- Item 3 (belief head) predicts those same 3x28 owner labels from the **public** trunk features.

## Net (DominoFeaturesExtractor v3)

- hand tower: unchanged (tile embedding 28x32 -> shared 32->64 MLP -> masked mean+max pool -> 128)
- context tower: input 187 (164 public context + 21 void + 2 match) -> Linear 187->256, LayerNorm, ReLU
- move tower [item 4]: per move concat(emb(tile) 32, seat one-hot 4, end flag 2) = 38 -> shared
  38->64 ReLU MLP, 6 moves flattened = 384 -> Linear 384->128, LayerNorm, ReLU
- trunk: concat(128 + 256 + 128 = 512) -> 256 LN ReLU -> 256 LN ReLU -> 128 LN ReLU
- heads: policy Linear(128->56); value head is fed by the **privileged extractor** (501 -> 256
  LN ReLU -> 256 LN ReLU -> 128) then Linear(128->1)
- belief head [item 3]: Linear(128 -> 84) with sigmoid, trained by BCE against obs[417:501]

## Training loop

- **Aux belief loss** (item 3) is applied by a callback (`BeliefHeadCallback`) with its own Adam
  optimiser and its own lr (`belief_lr`, default 1e-3): each rollout it samples the rollout
  buffer, runs `policy.extract_features` -> belief head -> BCE vs the privileged labels, and
  steps. Chosen over subclassing `MaskablePPO.train()` because copying SB3's train loop is
  version-fragile; both optimisers update the *shared* public trunk, which is what item 3 asks
  for. Weight: `belief_coef` (default 0.5 on the loss reported to TensorBoard; the callback's
  step size is `belief_lr`).
- **Match episodes** (item 6): `max_deals_per_episode` becomes a real training setting (default
  5, `configs/league_full.yaml`), so the score features carry information. The env already emits
  per-deal rewards on the transition that ends each deal and ends the episode after the last.
  Convergence is still total undiscounted reward per episode (now a 5-deal episode); the eval
  protocol (`test.py`) keeps its one-deal phase and its match phase unchanged.
- **League hygiene** (item 9): snapshot/policy opponents act *stochastically*
  (`PolicyAgent(deterministic=False)` for the pool, argmax kept for evaluation); an entropy floor
  callback raises `ent_coef` back to the floor when the logged entropy drops under
  `min_entropy` nats (default 0.1) and relaxes it again when entropy recovers; `tools/eval_ladder.py`
  evaluates checkpoints against a fixed ladder (random / rule / a named early checkpoint) with N
  seeds and writes `results/ladder_<run>.csv` plus Elo numbers.
- **Throughput** (item 8): `n_envs` 4 -> 6 (RAM-bounded, heavy imports are already lazy), and an
  LRU cache on `_tile_placements` keyed by (tile, ends). A vectorised/JAX engine rewrite is
  explicitly *out of scope* and recorded as future work in the report: it means reimplementing the
  graded rules as array maths for a 2-4x end-to-end gain over the 650-950 steps/s we measure.

## Inference

- **PIMC** (item 5): `domino/search.py` — for each root legal move: sample K hidden-hand
  determinizations consistent with the public information (hand counts, void-lack matrix, tiles
  already played), evaluate the resulting state with the *privileged* value net, average, pick the
  best move. No deep rollouts: K x #legal-moves value calls, ~50-120 ms/move at K=16, which is the
  "thinking" mode, not the 2.5 ms default. Exposed as `--search K` on `test.py`/`play.py` so the
  A/B ("policy only" vs "policy + PIMC") runs on identical deals.

## Verification plan

1. Full `pytest -q` green after each unit (obs layout, policy shapes, belief loss, ladder, PIMC).
2. Pre-flight gate: `preflight.py --factory domino.env:DominoEnv` must PASS (episode diversity).
3. Smoke training run (40k steps, 4-6 workers) -> `deals/distinct_deals_window` = window, no
   WARNING, entropy above the floor, parameter count reported.
4. 5-seed pooled eval (`aggregate_seeds.py`) of the new checkpoint vs the v2 52.4% vs-rule baseline.
5. Overnight chain via `tools/run_curriculum.sh` with a fresh `v3` run name (never `--resume`
   across an observation change).
