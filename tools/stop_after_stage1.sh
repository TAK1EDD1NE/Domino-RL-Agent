#!/usr/bin/env bash
# Stop the curriculum chain once stage 1 (and its evaluation) is done.
#
# The chain launched by tools/run_curriculum.sh runs stages 1-3 back to back. When only
# stage 1 is wanted tonight, this watchdog waits for the stage-1 evaluation marker in the
# log and then kills the launcher and any stage-2 trainer that has already started (stage 2
# resumes from models/checkpoints/latest.zip, so killing it early costs nothing).
#
# Usage:  bash tools/stop_after_stage1.sh [log-path]
set -u

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
LOG="${1:-runs/logs/curriculum_v3b.log}"
STATUS="${LOG%.log}.watchdog.log"
# Anchored, and every status line goes to a *separate* file: a watchdog that echoes the marker
# text it waits for into the file it greps matches its own message and fires immediately (this
# bug killed the launcher 0 s after arming the first version of this script).
TARGET='^EVAL_stage1_random_v3_EXIT='

echo "watchdog armed $(date -Is): waiting for the stage-1 evaluation exit marker in ${LOG}" >> "$STATUS"
while ! grep -qE "$TARGET" "$LOG" 2>/dev/null; do
    sleep 5
done
echo "watchdog fired $(date -Is): stage 1 + evaluation finished; stopping the chain" >> "$STATUS"

# Kill the launcher tree and any stage-2/3 trainer that has already started. The patterns use
# a character class ('run_curriculu[m]') so this command's own command line cannot match
# them, and the PowerShell process excludes itself.
powershell -NoProfile -Command "\$self = \$PID;
    Get-CimInstance Win32_Process |
    Where-Object { \$_.ProcessId -ne \$self -and
        (\$_.CommandLine -match 'run_curriculu[m]' -or
         \$_.CommandLine -match 'stage2_rul[e]_v3' -or
         \$_.CommandLine -match 'stage3_leagu[e]_v3') } |
    ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" >/dev/null 2>&1
sleep 3
REMAINING=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process |
    Where-Object { \$_.CommandLine -match 'train\.py' } | Measure-Object).Count")
echo "watchdog done $(date -Is): remaining train.py processes = ${REMAINING:-unknown} (0 expected)" >> "$STATUS"
