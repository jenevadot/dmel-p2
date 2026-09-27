#!/bin/bash
# Iteration 2 — Machine A (M4 Pro, MPS). See NEXT_ITERATION.md §4.
#
# Sequential and restartable: experiment.run() skips any experiment whose
# metrics_dev.json exists. Never co-schedule a second training on this GPU —
# earlier MPS timings were inflated exactly that way.
#
#   0. re-score the s42 reference runs with the extended evaluation
#      (persistence per bucket, dev_clean, skill by lead, high-flow bias) so
#      the new runs are compared like for like. ~2 min each, no training.
#   1. exp_050__resid_huber05       vs exp_010 — residual output (R1)
#   2. exp_051__resid_mse           vs exp_001 s42 — isolates residual vs loss
#   3. exp_052__rimf_append_huber05 vs exp_010 — the one CEEMDAN test (R6)
#   4. exp_053__lstm_huber05        optional LSTM baseline
#
# Launch:  nohup caffeinate -i ./run_queue_iter2_machineA.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=experiments/queue_iter2_machineA.log
mkdir -p experiments/rescore

echo "" | tee -a "$LOG"
echo "=== iter2 machine-A started $(date) ===" | tee -a "$LOG"

for ref in exp_001__baseline exp_009__huber exp_010__huber_d05 exp_012__mae; do
  out=experiments/rescore/${ref}__dev.json
  if [ ! -f "$out" ]; then
    echo "--- rescore $ref  $(date +%H:%M:%S) ---" | tee -a "$LOG"
    $PY src/ensemble_eval.py --exps "$ref" --out "$out" 2>&1 \
      | tr '\r' '\n' | grep -E "PRIMARY|subset|beta" | tee -a "$LOG"
  fi
done

QUEUE=(
  exp_050__resid_huber05
  exp_051__resid_mse
  exp_052__rimf_append_huber05
  exp_053__lstm_huber05
)
for exp in "${QUEUE[@]}"; do
  echo "" | tee -a "$LOG"
  echo "--- $exp  $(date +%Y-%m-%d\ %H:%M:%S) ---" | tee -a "$LOG"
  $PY run_ablation.py --only "$exp" 2>&1 | tail -25 | tee -a "$LOG"
  $PY iter2_report.py 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== iter2 machine-A finished $(date) ===" | tee -a "$LOG"
