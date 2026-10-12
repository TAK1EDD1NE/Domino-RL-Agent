Invalidated artifacts — frozen-deal environment bug (2026-10-12)
================================================================

Everything in this directory comes from the first training attempt
(stage1_random, stage2_rule, stage3_league, 10.2M learner steps total) and is
kept only as evidence for the report's "what went wrong and how it was found"
section. None of it should be used as a result.

Why it is invalid
-----------------
`DominoEnv.reset()` rebuilt its deal generator from a sticky `self._seed` on
every reset. SB3 seeds a vector env exactly once; every auto-reset after an
episode calls `env.reset()` with no seed (stable_baselines3 2.7.1,
`common/vec_env/subproc_vec_env.py:42`). Each of the 4 workers therefore re-dealt
the same shuffle for the whole run: 4 frozen deals in total. The policy memorised
them, so the reward window looked like convergence while the agent scored at
random level on the fresh deals `test.py` deals.

Evidence
--------
* 25 consecutive unseeded resets produced 1 distinct deal (regression test
  `tests/test_env.py::test_unseeded_reset_deals_a_fresh_shuffle`, which fails on
  the old code and passes on the fixed one).
* Replaying the frozen deal with the final policy gave exactly +39 pts/deal
  (sd 0.0) in 300/300 deals vs the rule bot and -45 in 300/300 vs random.
* The per-stage training CSVs contain only 4 distinct reward values each: the
  4 workers' memorised deals (stage 3: 39 = 53%, 18 = 26%, 34 = 20%).
* Entropy collapsed (62.8 -> 0.055 nats in stage 1; 0.0125 nats at the end of
  stage 3) with approx_kl = 0, clip_fraction = 0 and losses ~1e-7: no gradient
  left to take.
* `test.py` on the checkpoints (400 fresh deals): 45.3% (1M), 42.0% (6.0M),
  44.3% (10.2M) vs the rule bot, 51-52% vs random -- the three Wilson intervals
  overlap, so 10.2M steps bought nothing.

Fix
---
`domino/env.py` now draws each episode's deal seed from gymnasium's `np_random`
(advances on every reset, rebuilt from `seed` when one is passed), so
`reset(seed=X)` stays reproducible and an unseeded reset re-deals. Guarded by
`deals/distinct_deals_window` in TensorBoard plus its callback warning, so a
replayed deal can never again masquerade as convergence.

Contents
--------
checkpoints/  the 25 stage checkpoints (latest.zip included)
snapshots/    the 10 league snapshots
*.csv         the stage reward mirrors, the 1M/interim/stage-2-final evaluations
*.png         the learning curves of the invalidated run
