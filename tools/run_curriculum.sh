#!/usr/bin/env bash
# Staged training curriculum for the 2v2 domino agent (memos: docs/plans/2026-10-12-training-curriculum.md).
#
# Three stages chained through models/checkpoints/latest.zip, changing ONLY the
# opponent per stage (identical PPO hyperparameters, so the stage curves are
# comparable), with an evaluation of the checkpoint against the rule bot and
# random play after every stage.
#
# Stage 1 starts FRESH (no --resume): after the frozen-deal env bug the old
# checkpoints are memorised lookup tables and must not be resumed into.
#
# Each stage writes STAGE_<name>_EXIT=<code> into the log; each evaluation writes
# EVAL_<name>_EXIT=<code>. A log that ends without the marker means the stage was
# killed rather than finished.
#
# Usage:  bash tools/run_curriculum.sh [log-path]
set -u

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
LOG="${1:-runs/logs/curriculum_v2.log}"
mkdir -p "$(dirname "$LOG")"

echo "=== curriculum start $(date -Is) ===" >> "$LOG"

run_stage () {           # name steps opponent fresh|resume
    local name="$1" steps="$2" opponent="$3" mode="$4"
    local -a cmd=(uv run python -u train.py --config configs/league_full.yaml
                  --steps "$steps" --n-envs 4 --device cpu --run-name "$name"
                  --set "env.opponent=$opponent")
    if [ "$mode" = "resume" ]; then
        cmd+=(--resume)
    fi
    echo "--- stage $name: $steps steps vs opponent=$opponent ($mode) $(date -Is)" >> "$LOG"
    "${cmd[@]}" >> "$LOG" 2>&1
    echo "STAGE_${name}_EXIT=$?" >> "$LOG"

    uv run python -u test.py --model models/checkpoints/latest.zip \
        --matchups rl_vs_rule,rl_vs_random --deals 400 --matches 1 --quiet \
        --out "results/eval_${name}.csv" >> "$LOG" 2>&1
    echo "EVAL_${name}_EXIT=$?" >> "$LOG"
}

run_stage stage1_random_v2 1000000  random fresh
run_stage stage2_rule_v2   5000000  rule   resume
run_stage stage3_league_v2 10000000 league resume

echo "=== curriculum done $(date -Is) ===" >> "$LOG"
