#!/bin/bash
# FINAL MODEL — Machine A (MPS). 6 of the 9 ensemble members.
# Machine B trains the other 3 (run_queue_final_machineB.sh).
#
#   member                    est. time here
#   final_lstm_huber05_s42     ~3.3 h
#   final_resid_huber05_s42    ~6.4 h
#   final_mae_s42              ~6.4 h
#   final_lstm_huber05_s123    ~3.3 h
#   final_resid_huber05_s123   ~6.4 h
#   final_lstm_huber05_s177    ~3.3 h   (last: the 6-member set is done first)
#
# Restartable: a member with metrics_dev.json is skipped.
# Launch:  nohup caffeinate -i ./run_queue_final_machineA.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=experiments/queue_final_machineA.log

QUEUE=(
  "final_lstm_huber05 42"
  "final_resid_huber05 42"
  "final_mae 42"
  "final_lstm_huber05 123"
  "final_resid_huber05 123"
  "final_lstm_huber05 177"
)

echo "" | tee -a "$LOG"
echo "=== final machine-A started $(date) ===" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do
  read -r exp seed <<< "$item"
  echo "" | tee -a "$LOG"
  echo "--- ${exp}_s${seed}  $(date +%Y-%m-%d\ %H:%M:%S) ---" | tee -a "$LOG"
  $PY run_ablation.py --only "$exp" --seeds "$seed" --always_suffix 2>&1 \
    | tail -25 | tee -a "$LOG"
done
echo "=== final machine-A finished $(date) ===" | tee -a "$LOG"
