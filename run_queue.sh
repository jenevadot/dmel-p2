#!/bin/bash
# Sequential experiment queue. Kept simple and restartable: run() skips any
# experiment whose metrics_dev.json already exists, so re-running this script
# resumes rather than redoing work.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=experiments/queue.log
mkdir -p experiments
echo "=== queue started $(date) ===" | tee -a "$LOG"
for exp in "$@"; do
  echo "" | tee -a "$LOG"
  echo "--- $exp  $(date +%H:%M:%S) ---" | tee -a "$LOG"
  $PY run_ablation.py --only "$exp" 2>&1 | tail -22 | tee -a "$LOG"
done
echo "" | tee -a "$LOG"
echo "=== queue finished $(date) ===" | tee -a "$LOG"
