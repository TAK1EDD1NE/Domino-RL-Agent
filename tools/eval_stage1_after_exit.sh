#!/usr/bin/env bash
# Wait for the running stage-1 trainer to exit, then evaluate its final checkpoint.
#
# The launcher's evaluation step normally does this, but if the chain is stopped (or the
# launcher dies) the stage-1 numbers still have to be produced. This runs the same
# evaluation the launcher runs: 5 seeds x 400 deals against the rule bot and random play,
# pooled into a Wilson CI by tools/aggregate_seeds.py.
#
# Usage:  bash tools/eval_stage1_after_exit.sh [run-name] [log-path]
set -u

# Resolve uv explicitly (see tools/run_curriculum.sh): a shell without the Hermes bin dir on
# PATH fails on a bare "uv: command not found".
for _dir in "$HOME/AppData/Local/hermes/bin" "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -d "$_dir" ] && PATH="$_dir:$PATH"
done
export PATH
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found on PATH; refusing to start." >&2; exit 127; }

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
RUN_NAME="${1:-stage1_random_v3}"
LOG="${2:-runs/logs/curriculum_v3b.log}"
EVAL_LOG="runs/logs/${RUN_NAME}_eval.log"
LATEST="models/checkpoints/latest.zip"

echo "=== waiting for the ${RUN_NAME} trainer to exit $(date -Is) ===" >> "$EVAL_LOG"
# Match on a pattern that cannot match this command's own text: the run name with its last
# character in a character class ("stage1_random_v[3]"), plus the requirement that the process
# is a train.py process (the script's own bash/uv processes mention the run name as well).
LAST_CHAR="${RUN_NAME: -1}"
SELF_SAFE="${RUN_NAME%?}[${LAST_CHAR}]"
while powershell -NoProfile -Command "(Get-CimInstance Win32_Process |
        Where-Object { \$_.CommandLine -match 'trai[n]\.py' -and \$_.CommandLine -match '${SELF_SAFE}' } |
        Measure-Object).Count" | tr -d '\r' | grep -qv '^0$'; do
    sleep 10
done
echo "=== trainer gone $(date -Is); evaluating ${LATEST} ===" >> "$EVAL_LOG"

for s in 0 1 2 3 4; do
    uv run python -u test.py --model "$LATEST" \
        --matchups rl_vs_rule,rl_vs_random --deals 400 --matches 1 --quiet \
        --seed "$s" --out "results/eval_${RUN_NAME}_seed${s}.csv" >> "$EVAL_LOG" 2>&1
    echo "EVAL_${RUN_NAME}_seed${s}_EXIT=$?" >> "$EVAL_LOG"
done

uv run python -u tools/aggregate_seeds.py "results/eval_${RUN_NAME}_seed*.csv" >> "$EVAL_LOG" 2>&1
AGG=$?
echo "EVAL_${RUN_NAME}_EXIT=$AGG" >> "$EVAL_LOG"
# Mirror the marker into the chain log so anything watching that file (the launcher, or
# tools/stop_after_stage1.sh) sees the same fact. The real exit code is preserved.
echo "EVAL_${RUN_NAME}_EXIT=$AGG (written by tools/eval_stage1_after_exit.sh $(date -Is))" >> "$LOG"
echo "=== evaluation done $(date -Is) (aggregate exit $AGG) ===" >> "$EVAL_LOG"
