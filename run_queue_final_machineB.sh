#!/bin/bash
# FINAL MODEL — Machine B (CUDA). 3 of the 9 ensemble members.
# Start AFTER exp_012__mae_s177 (iteration 2) has finished.
#
#   final_mae_s123             ~7.7 h
#   final_resid_huber05_s177   ~7.7 h
#   final_mae_s177             ~7.7 h
#
# Then sync each experiments/final_*/ folder back to machine A INCLUDING
# best_model.pt (gitignored; copy it deliberately), where the ensemble is
# scored:
#   python src/ensemble_eval.py --exps final_* --out experiments/final_ens9_dev.json
#
# Launch:  nohup ./run_queue_final_machineB.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=experiments/queue_final_machineB.log

if [ ! -f experiments/exp_012__mae_s177/metrics_dev.json ]; then
  echo "=== waiting for exp_012__mae_s177 ($(date)) ===" | tee -a "$LOG"
  while [ ! -f experiments/exp_012__mae_s177/metrics_dev.json ]; do
    sleep 120
  done
fi

QUEUE=(
  "final_mae 123"
  "final_resid_huber05 177"
  "final_mae 177"
)

echo "" | tee -a "$LOG"
echo "=== final machine-B started $(date) ===" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do
  read -r exp seed <<< "$item"
  echo "" | tee -a "$LOG"
  echo "--- ${exp}_s${seed}  $(date +%Y-%m-%d\ %H:%M:%S) ---" | tee -a "$LOG"
  $PY run_ablation.py --only "$exp" --seeds "$seed" --always_suffix 2>&1 \
    | tail -25 | tee -a "$LOG"
done
echo "=== final machine-B finished $(date) ===" | tee -a "$LOG"
