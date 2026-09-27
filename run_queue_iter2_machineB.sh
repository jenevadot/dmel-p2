#!/bin/bash
# Iteration 2 — Machine B (CUDA). See NEXT_ITERATION.md §4.
#
# Seed replication of the robust-loss claim. These pair with the EXISTING
# exp_001__baseline_s123 / _s177 runs on this machine: same san_val
# partition (split seed = seed), same training basins, same device.
#
# PREREQUISITE: pull the iteration-2 code first (EMA logging, extended
# evaluation). The training path of exp_010/exp_012 is unchanged — new
# config keys default to the old behaviour — so pairs stay matched.
#
#   0. exp_034__no_distil: score the existing checkpoint (eval only)
#   0b. re-score the s123/s177 baselines with the extended evaluation
#   1-2. exp_010__huber_d05  seeds 123, 177
#   3-4. exp_012__mae        seeds 123, 177
#   5-6. exp_050__resid_huber05 seeds 123, 177 — ONLY if the A1 gate passed.
#        Copy experiments/exp_050__resid_huber05/{summary,metrics_dev}.json
#        and experiments/rescore/exp_010__huber_d05__dev.json from machine A
#        into this checkout; iter2_report.py then decides. Or force with
#        RUN_R1_SEEDS=1.
#
# Launch:  nohup ./run_queue_iter2_machineB.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=experiments/queue_iter2_machineB.log
mkdir -p experiments/rescore

echo "" | tee -a "$LOG"
echo "=== iter2 machine-B started $(date) ===" | tee -a "$LOG"

if [ -f experiments/exp_034__no_distil/best_model.pt ] && \
   [ ! -f experiments/exp_034__no_distil/metrics_dev.json ]; then
  echo "--- exp_034 eval-only  $(date +%H:%M:%S) ---" | tee -a "$LOG"
  $PY src/ensemble_eval.py --exps exp_034__no_distil \
      --override use_distil=false --write_into 2>&1 \
    | tr '\r' '\n' | grep -E "PRIMARY|h1_12|h37|subset" | tee -a "$LOG"
fi

for ref in exp_001__baseline_s123 exp_001__baseline_s177; do
  out=experiments/rescore/${ref}__dev.json
  if [ -f experiments/$ref/best_model.pt ] && [ ! -f "$out" ]; then
    echo "--- rescore $ref ---" | tee -a "$LOG"
    $PY src/ensemble_eval.py --exps "$ref" --out "$out" 2>&1 \
      | tr '\r' '\n' | grep -E "PRIMARY|subset" | tee -a "$LOG"
  fi
done

for exp in exp_010__huber_d05 exp_012__mae; do
  echo "" | tee -a "$LOG"
  echo "--- $exp seeds 123 177  $(date +%Y-%m-%d\ %H:%M:%S) ---" | tee -a "$LOG"
  $PY run_ablation.py --only "$exp" --seeds 123 177 2>&1 | tail -30 \
    | tee -a "$LOG"
done

if [ "${RUN_R1_SEEDS:-0}" = "1" ] || $PY iter2_report.py --gate_r1; then
  echo "--- R1 gate passed: exp_050 seeds 123 177 ---" | tee -a "$LOG"
  $PY run_ablation.py --only exp_050__resid_huber05 --seeds 123 177 2>&1 \
    | tail -30 | tee -a "$LOG"
else
  echo "--- R1 gate not passed (or exp_050 not synced) — skipping B5/B6 ---" \
    | tee -a "$LOG"
fi

$PY iter2_report.py 2>&1 | tee -a "$LOG"
echo "=== iter2 machine-B finished $(date) ===" | tee -a "$LOG"
