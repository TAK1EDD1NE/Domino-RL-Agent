# Slide outline — RLOC 2026/2027 (Domino 2v2 RL agent)

A slide-by-slide skeleton for the PDF deck the assignment asks for, with the measured numbers
already filled in. The **wording, the emphasis and the final conclusions are the author's** —
this is a scaffold, not a finished deck. Anything marked `[FILL]` needs a number that only a
real training run can produce.

The assignment's required report content maps onto slides 1-5, 9-13 (description of the game and
the learning problem, the algorithm, the architecture and hyper-parameters, the code, the
results, the references).

---

## 1. Title

* Domino-RL — a 2v2 domino agent trained by self-play.
* Names, 3CS AI & DS, Reinforcement Learning and Optimal Control 2026/2027, N. Farhi & M. Bouali.
* One-line claim: *"a rules engine, a Gymnasium environment, a masked PPO agent with a custom
  set-encoder network, and a documented rule-based baseline, compared on six metrics."*

## 2. The game and why it is a hard RL problem

* 28 tiles, 7 per seat, teams `{0,2}` vs `{1,3}`, no boneyard, mandatory play, pass only when
  stuck, block after four passes.
* Two open ends → a move is a **(tile, end)** pair; a double leaves its end unchanged.
* Scoring: hand emptied banks the opponents' pips; a blocked deal is won by the fewest-pips
  *individual* (banking the opposing pair's pips); a tie across teams voids the deal and forces
  a redeal. Matches to 100.
* Why it is interesting: **imperfect information** (you never see three hands), **cooperation**
  (the partner is not controllable), **sparse terminal reward** (points arrive once per deal,
  after 10-14 decisions) and **variable action sets** (0-14 legal placements).

## 3. Problem formulation (the MDP)

* Table: agent / episode / observation 192-dim / action `Discrete(56)` / reward / horizon.
* Observation: own hand, public play history, per-seat counts, open ends, pass streak, per-value
  liveness — **public information only** (a test asserts no hidden hand leaks into the encoding).
* Reward: deal points, undiscounted, on the terminal transition; differential (zero-sum) by
  default, `banked` as an ablation.
* Masking: illegal placements are masked by `MaskablePPO`; a forced pass is not a decision.

## 4. Algorithm and architecture

* `MaskablePPO` (sb3-contrib) with `gamma=1.0`-style undiscounted setting, `n_steps=512`,
  `batch_size` = effective rollout size, `ent_coef` and `clip_range` from `configs/default.yaml`.
* **Custom network** (the assignment forbids a stock MLP): per-tile embedding with masked
  mean/max pooling over the hand → context tower over the remaining features → LayerNorm trunk →
  policy/value heads. **251 641 parameters**, one policy shared by both seats of the team.
* Diagram slot: two-tower set encoder.

## 5. Self-play league

* Because a live policy cannot be shared into spawned workers, the opponent pool is
  `{snapshot 0.6, rule 0.25, random 0.15}`, refreshed every `--snapshot-every` steps; a live
  `self` opponent is available in single-env mode.
* Snapshot store is an LRU of `max_loaded` checkpoints, sampled by weight.
* Why: a stationary opponent lets the policy overfit; the league keeps it moving.

## 6. Rule-based baseline

* Seven named scoring terms (pips shed, kill a value, opponent-short ends, keep doubles, protect
  partner, block economics, duplicate leads) + doubling of the block terms near a block.
* Weights chosen by a coordinate-ascent sweep on **seeds disjoint from the test gate**:
  mean win rate vs random 0.5527 → **0.5853** on held-out seeds.
* This matters because the baseline is the bar the RL agent has to clear.

## 7. Training setup and cost

* Hardware: Windows 11, 12 logical CPUs, RTX 3060 Laptop 6 GB; torch 2.5.1+cu121.
* Measured (not estimated) throughput: raw engine ~5 000 moves/s; **end-to-end learner
  ~800 steps/s with 8 envs** (league costs nothing measurable over the rule-only mix).
* Wall clock: 10 M steps ≈ **3.5 h**, 30 M ≈ **10.5 h**, 100 M ≈ **35 h**.
* Budget used: `[FILL: the run you actually launch]`, checkpoints every 500 k steps.

## 8. Convergence

* Figure: total undiscounted reward per episode + 100-episode sliding average.
* What to look for: the window average rising and then flattening, the per-episode spread
  staying wide (deal luck is large: 0-100+ points swing on the last tile).
* `[FILL: the plateau value and the steps at which it appeared]`

## 9. Evaluation results

* Table from `test.py`: win rate + Wilson 95 % CI, points/deal, match win rate, blocked/void
  rates, mean pips left, ms/action, for RL vs rule, RL vs random, rule vs random, rule vs rule,
  random vs random (the last two are sanity baselines: ~0.5).
* Figures: win rates with error bars; margin per deal; blocked/void rates.
* `[FILL: the trained agent's row]`

## 10. RL vs the rule-based baseline

* Did the agent beat 0.5 against the rule bot, and by how much against the Wilson interval?
* Where does it lose? (blocks, the endgame, near the 100-point cap.)
* `[FILL: your reading of the numbers]`

## 11. Code and interfaces

* Mandatory files: `train.py`, `test.py`, `play.py`, `observe.py`, `evaluate.ipynb`, `README.md`.
* `uv sync` reproducibility, `uv.lock` pins, 120-test pytest suite.
* Screenshots: interactive mode (you vs the agent) and observation mode (policy/value overlay).

## 12. Limitations and future work

* Terminal-only reward → sparse credit assignment; reward shaping untested.
* Match score not observed → no match-level tactics near 100.
* Python-bound environment → ~800 steps/s; levers listed in the spec.
* No explicit belief model over hidden hands.
* Future: periodic in-training evaluation (true sample-efficiency curve), PFSP opponent
  selection, an end-only action head for the both-ends case, GPU-vectorised environments.

## 13. References

* PPO (Schulman et al. 2017); Stable-Baselines3 (Raffin et al. 2021) + sb3-contrib;
  invalid-action masking (Huang et al. 2022); self-play (Silver et al. 2017);
  Gymnasium API; Algerian 2v2 domino rules as specified by the project owner.

---

## Numbers to fill in after the real run

| placeholder | where it comes from |
|---|---|
| steps actually trained | `train.py` console / TensorBoard `time/total_timesteps` |
| plateau of the 100-episode average | `results/training_<run>.csv` |
| win rate vs rule bot + CI | `results/eval_<run>.csv` |
| match win rate | same file, `match_win_rate_a` |
| blocked/void rates | same file |
| screenshots | `play.py` and `observe.py` windows |
