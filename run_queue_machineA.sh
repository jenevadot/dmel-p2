#!/bin/bash
# Machine A queue — runs AFTER exp_038__no_clip finishes.
#
# Sequential and restartable: experiment.run() skips any experiment whose
# metrics_dev.json already exists, so re-running this script resumes rather
# than redoing work. Kill it and restart freely.
#
# WAITS for the in-flight exp_038__no_clip before starting, so two trainings
# never share the GPU. exp_038 is at epoch 33/40 with best sanval 0.5736 @ ep15
# and 18 epochs of decline since — it is already a decided negative result
# (clip=1.0 baseline is 0.6186), but it is left to finish so the record is
# complete rather than truncated.
#
# ORDER AND RATIONALE
# ───────────────────
# Machine B holds the seed-123/177 exp_001__baseline arm (T1), so this queue
# deliberately avoids exp_001 and avoids the 5 CEEMDAN experiments
# (exp_020-024, priority "ceemdan") which Machine B cannot run and which this
# machine already has the cache for — those go in a LATER queue, after
# exp_020__dmel, because T1's second arm depends on exp_020 finishing first.
#
#   1. exp_002__global_norm      critical — norm_strategy is the single most
#                                load-bearing preprocessing choice; a wrong
#                                answer here invalidates everything downstream.
#   2. exp_003__discharge_only   critical — do the 11 meteo channels earn their
#   3. exp_004__precip_discharge critical   keep? Cheapest large-effect answer.
#   4. exp_009__huber            high — loss shape. NSE is peak-dominated, so
#   5. exp_010__huber_d05        high   these test peak handling directly.
#   6. exp_012__mae              high
#   7. exp_011__nse_loss         high — train the metric being scored.
#   8. exp_005__aux_task         high — y_aux as auxiliary TARGET (legitimate;
#                                the head is discarded at inference).
#   9. exp_008__seq168           high — 168h vs 336h history.
#  10. exp_015__dropout02        high — regularisation. Promoted in priority by
#                                exp_038's overfit signature (train_loss 0.295
#                                while sanval decays), which suggests capacity
#                                control matters more than first assumed.
#
# DELIBERATELY EXCLUDED
# ─────────────────────
#   exp_013__lr_5e4 / exp_014__lr_5e5 — BLOCKED. clip_grad=1.0 fires on 100%
#     of steps (measured grad_norm 2.1-2.8), so effective LR is 0.35-0.47x
#     nominal and a nominal 10x LR difference is compressed to far less. These
#     two would measure clipping, not learning rate. Run them only after the
#     clip_grad fix (PROMPT_IMPLEMENT.md Task 1) lands.
#   exp_006__basin_emb / exp_007__aux_basin_emb — BLOCKED on the selection_split
#     fix (Task 3). san_val has 0 basin overlap with train, so their basin
#     embeddings are evaluated as randomly initialised noise and both are
#     guaranteed to look worse than they are.
#   exp_030-040 (medium) — architecture sweep, better suited to Machine B's
#     CUDA + VRAM. Not blocked, just lower value per GPU-hour here.
#
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=experiments/queue.log
mkdir -p experiments

# ── Wait for the in-flight run to finish ──────────────────────────────
# Poll for exp_038's metrics_dev.json, which experiment.run() writes only
# after training completes and both splits are evaluated.
if [ ! -f experiments/exp_038__no_clip/metrics_dev.json ]; then
  echo "=== waiting for exp_038__no_clip to finish ($(date)) ===" | tee -a "$LOG"
  while [ ! -f experiments/exp_038__no_clip/metrics_dev.json ]; do
    sleep 120
  done
  echo "=== exp_038__no_clip finished ($(date)) ===" | tee -a "$LOG"
fi

QUEUE=(
  exp_002__global_norm
  exp_003__discharge_only
  exp_004__precip_discharge
  exp_009__huber
  exp_010__huber_d05
  exp_012__mae
  exp_011__nse_loss
  exp_005__aux_task
  exp_008__seq168
  exp_015__dropout02
)

echo "" | tee -a "$LOG"
echo "=== machine-A queue started $(date) — ${#QUEUE[@]} experiments ===" \
  | tee -a "$LOG"

for exp in "${QUEUE[@]}"; do
  echo "" | tee -a "$LOG"
  echo "--- $exp  $(date +%Y-%m-%d\ %H:%M:%S) ---" | tee -a "$LOG"
  $PY run_ablation.py --only "$exp" 2>&1 | tail -25 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== machine-A queue finished $(date) ===" | tee -a "$LOG"
