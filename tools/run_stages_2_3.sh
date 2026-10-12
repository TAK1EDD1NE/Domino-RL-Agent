#!/usr/bin/env bash
# Stages 2 and 3 of the v3 curriculum, to be run after stage 1 (and separately from it).
#
# Stage 2: 5,000,000 steps against the rule bot, resuming from the stage-1 checkpoint.
# Stage 3: 10,000,000 steps in the league, resuming from the stage-2 checkpoint.
# After each stage the resulting checkpoint is evaluated with test.py -- 5 seeds x 400 deals
# against the rule bot and random play, pooled with a Wilson CI by tools/aggregate_seeds.py.
# Both stages of the v3 architecture (centralised critic, void-lack/match/move features,
# belief head, stochastic league, entropy floor) are already active in the checkpoint.
#
# Prerequisite: models/checkpoints/latest.zip from stage 1 (tools/run_curriculum.sh wrote it).
# Usage:        bash tools/run_stages_2_3.sh [log-path]
set -u

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
LOG="${1:-runs/logs/curriculum_stages23.log}"
mkdir -p "$(dirname "$LOG")"

LATEST="models/checkpoints/latest.zip"
STAGE1_LOG="${STAGE1_LOG:-runs/logs/curriculum_v3b.log}"
STAGE1_RUN="${STAGE1_RUN:-stage1_random_v3}"
STAGE1_STEPS="${STAGE1_STEPS:-1000000}"
STAGE1_CSV="results/training_${STAGE1_RUN}.csv"

# Prerequisite check: the *finished* stage 1, not merely a checkpoint. train.py writes
# models/checkpoints/latest.zip every --ckpt-every steps, so that file exists in the middle of
# stage 1 too -- resuming from it silently trains stage 2 on a half-finished stage-1 policy.
# Completion is proven by data (the stage-1 CSV reaching its step budget) or by the launcher's
# exit marker; the checkpoint file alone proves nothing.
last_step=0
if [ -f "$STAGE1_CSV" ]; then
    last_step=$(tail -1 "$STAGE1_CSV" | cut -d, -f1 | tr -dc '0-9')
fi
if ! grep -q "STAGE_${STAGE1_RUN}_EXIT=0" "$STAGE1_LOG" 2>/dev/null \
   && [ "${last_step:-0}" -lt $((STAGE1_STEPS - 10000)) ]; then
    echo "ERROR: stage 1 does not look finished:" >&2
    echo "       no 'STAGE_${STAGE1_RUN}_EXIT=0' in $STAGE1_LOG and $STAGE1_CSV" >&2
    echo "       only reached step ${last_step:-0} of ${STAGE1_STEPS}." >&2
    echo "       (latest.zip is written every --ckpt-every steps, so it does not mean stage 1 is done.)" >&2
    exit 1
fi
echo "stage 1 verified complete (step ${last_step:-0} of ${STAGE1_STEPS})" >&2
if [ ! -f "$LATEST" ]; then
    echo "ERROR: $LATEST not found -- stage 1 must leave a final checkpoint behind." >&2
    exit 1
fi

echo "=== stages 2-3 start $(date -Is) ===" >> "$LOG"
echo "--- stage 1 verified complete in $STAGE1_LOG; resuming from $LATEST ($(date -Is))" >> "$LOG"

run_stage () {           # name steps opponent
    local name="$1" steps="$2" opponent="$3"
    echo "--- stage $name: $steps steps vs opponent=$opponent (resume) $(date -Is)" >> "$LOG"
    uv run python -u train.py --config configs/league_full.yaml \
        --steps "$steps" --n-envs 6 --device cpu --run-name "$name" --resume \
        --set "env.opponent=$opponent" >> "$LOG" 2>&1
    echo "STAGE_${name}_EXIT=$?" >> "$LOG"

    for s in 0 1 2 3 4; do
        uv run python -u test.py --model "$LATEST" \
            --matchups rl_vs_rule,rl_vs_random --deals 400 --matches 1 --quiet \
            --seed "$s" --out "results/eval_${name}_seed${s}.csv" >> "$LOG" 2>&1
        echo "EVAL_${name}_seed${s}_EXIT=$?" >> "$LOG"
    done
    uv run python -u tools/aggregate_seeds.py "results/eval_${name}_seed*.csv" >> "$LOG" 2>&1
    echo "EVAL_${name}_EXIT=$?" >> "$LOG"
}

run_stage stage2_rule_v3   5000000  rule
run_stage stage3_league_v3 10000000 league

echo "=== stages 2-3 done $(date -Is) ===" >> "$LOG"
