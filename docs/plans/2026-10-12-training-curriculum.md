# Teaching the agent to win ~80% — staged curriculum

Target metric: **win rate vs the tuned RuleAgent in the standard eval gate** (`test.py`
`rl_vs_rule`, ≥5 seeds, reported with Wilson CI — never single-seed numbers; the
true between-seed SD measured earlier was 0.037).

## Why staged, not one long run

A single 10M league run trains against 60% favorite snapshots from step zero —
the opponent mix is too hard too early, so entropy keeps the policy a slow
rule-follower instead of a rule-breaker. The 3-stage plan changes only the
opponent mix (one `--set env.opponent=...` per stage); **no PPO hyperparameter
changes between stages**, so the training curves are comparable across stages.

## Stages

| Stage | Steps | Opponent mix | Purpose | Command |
|---|---|---|--|---|
| 1 | 1M | `random` (pure) | dense, unambiguous reward — seed the value function | `train.py --steps 1000000 --n-envs 4 --device cpu --run-name stage1_random --set env.opponent=random --resume` |
| 2 | 4M | `rule` (pure) | the real skill gate: which end, pip-counting, leading decisions | same with `--set env.opponent=rule --run-name stage2_rule --resume` |
| 3 | 5-20M | `league` (frozen snapshots 60% / rule 25% / random 15%) | beat *your own past selves* once rule-bot is beaten, no overfit to one style | same with `--set env.opponent=league --run-name stage3_league --resume` |

## Hyperparameter-effect protocol (for the report's hyperparam chapter)

Baseline = the league_full preset above. Each variant below runs 500k steps with
`run_name=hp_<name>`, then is scored by `test.py --matchups rl_vs_rule` at that
checkpoint. Results land in `results/eval_hp_sweep.csv` — one row per
(variant, seed). Variants chosen for their expected leverage on a
masks-fixed small-net PPO:

| variant | what it changes | what it tests |
|---|---|---|
| `hp_nsteps2048` | `n_steps 1024→2048` | larger collect amortizes the CPU-bound PPO update |
| `hp_nsteps256` | `n_steps 512→256` | the symmetric smaller-batch probe |
| `hp_ent0.02` | `ent_coef 0.01→0.02` | does higher entropy slow the plateau? |
| `hp_lr1e-4` | `learning_rate 3e-4→1e-4` | stability vs speed |

## Data artifacts produced (graph-ready CSVs)

* `results/training_stage1_random.csv` / `stage2_rule` / `stage3_league` —
  per-episode `step,episode,total_reward,window_avg` (the learning curve).
* `results/eval_hp_sweep.csv` — hyperparam variant × seed → win rate/CI/points/deal
  (the hyperparam-effect chart).
* `results/eval_<run>.csv` — standard RL-vs-baseline comparison rows per stage,
  straight into `evaluate.ipynb`'s existing plots.

## v2 relaunch (2026-10-12): the first attempt was invalid

The first pass of this curriculum (10.2M learner steps: 1M random, 5M rule, 4.2M of the
planned 10M league) is **not a result**. `DominoEnv.reset()` rebuilt its deal generator
from a sticky `self._seed`, and SB3 seeds a vector env exactly once — every later
auto-reset calls `env.reset()` with no seed (`common/vec_env/subproc_vec_env.py:42`) — so
each of the 4 workers re-dealt a single frozen deal for the whole run. The policy
memorised those 4 deals (stage CSVs contain exactly 4 distinct reward values), entropy
collapsed to 0.01 nats with losses at ~1e-7, the reward window sat flat at +32.5/deal,
and fresh-deal evaluation stayed at 42-45% vs the rule bot (overlapping Wilson CIs) and
51-52% vs random. Evidence and the invalidated artifacts live in
`results/invalid_frozen_deal_2026-10-12/` (README there).

Fix + guard before relaunching:

1. `reset()` draws each episode's deal seed from gymnasium's `np_random` (advances every
   reset, rebuilt from `seed` when given) → `reset(seed=X)` reproduces, unseeded resets
   re-deal. Regression test: `tests/test_env.py::test_unseeded_reset_deals_a_fresh_shuffle`
   (1 distinct deal in 25 unseeded resets on the old code, >= 20 on the fixed one).
2. `deals/distinct_deals_window` (TensorBoard) + a run-log warning from
   `EpisodeStatsCallback` when a full window holds <= 2 distinct deals, so a replayed
   deal can never again look like convergence.
3. Stage 1 starts **fresh** (`no --resume`); every old checkpoint is a memorised lookup
   table. Stages 2 and 3 chain with `--resume` as before.

Launcher: `bash tools/run_curriculum.sh [log]` — runs the three stages, one `test.py`
evaluation after each stage (`results/eval_stage<N>_<opponent>_v2.csv`), appending
`STAGE_<name>_EXIT=<code>` / `EVAL_<name>_EXIT=<code>` markers to the log
(default `runs/logs/curriculum_v2.log`). Judge each stage by its eval row, never by the
reward window; both stages and evaluations are light enough to run in one chain on this
box (4 workers, ~520-980 fps measured).

