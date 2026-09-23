#!/bin/bash
# Machine B queue (Ryzen 9 + NVIDIA, CUDA). Run AFTER the seed-123/177
# exp_001__baseline arm finishes.
#
# Sequential and restartable: experiment.run() skips any experiment whose
# metrics_dev.json already exists, so re-running resumes rather than redoing.
#
# No overlap with run_queue_machineA.sh, and nothing here needs the CEEMDAN
# cache (exp_020-024 are Machine A only — it holds data/rimf_*.h5).
#
# SETUP — do this once before the first run
# ─────────────────────────────────────────
#   num_workers: TRAIN_CFG defaults to 0 because Machine A's sandbox blocks
#   torch_shm_manager. That is a Machine-A workaround and costs real
#   throughput here: HDF5 reads happen serially in the training process while
#   the GPU idles. On this box set it to 8-12. Do NOT commit the change —
#   export it per-machine:
#
#       export DMEL_NUM_WORKERS=10        # if the config reads an env var
#       # otherwise pass --num_workers 10 to run_experiment.py
#
#   caffeinate is macOS-only. It is deliberately absent below.
#
# ORDER AND RATIONALE
# ───────────────────
#   1-2. exp_013/014 (LR) — FIRST. Both clipping endpoints are now measured
#        (clip=1.0 -> 0.6419, clip=0 -> 0.6296), so LR results are
#        interpretable against them. Highest-priority unblocked items.
#   3-4. exp_003/004 (channel ablation) — these CRASHED on Machine A with
#        "weight of size [256,12,3], expected input[64,1,338]": c_in stayed 12
#        while the dataset subset X to 1-2 channels. Fixed by deriving c_in
#        from input_channels in build_model. Cheap and high-information: do
#        the 11 meteo channels earn their keep at all?
#   5-7. exp_031/032/036 — d_model 128/512 and batch 128. The most
#        CUDA-favoured work available: more VRAM and real tensor cores.
#        exp_032__d_model_512 in particular is painful on MPS.
#   8-10. exp_030/033/034 — ProbSparse attention, 3 encoder layers, no
#        distillation. Architecture sweep, medium priority.
#
# EXCLUDED
# ────────
#   exp_006/007 (basin embeddings) — BLOCKED on the selection_split fix.
#     san_val has 0 basin overlap with train, so their embeddings are scored
#     as randomly initialised noise and both are guaranteed to look worse
#     than they are.
#   exp_020-024 — need the CEEMDAN cache, Machine A only.
#   Everything in run_queue_machineA.sh.
#
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=experiments/queue_machineB.log
mkdir -p experiments

QUEUE=(
  exp_013__lr_5e4
  exp_014__lr_5e5
  exp_003__discharge_only
  exp_004__precip_discharge
  exp_031__d_model_128
  exp_032__d_model_512
  exp_036__batch_128
  exp_030__probsparse
  exp_033__enc_layers_3
  exp_034__no_distil
)

echo "" | tee -a "$LOG"
echo "=== machine-B queue started $(date) — ${#QUEUE[@]} experiments ===" \
  | tee -a "$LOG"

for exp in "${QUEUE[@]}"; do
  echo "" | tee -a "$LOG"
  echo "--- $exp  $(date +%Y-%m-%d\ %H:%M:%S) ---" | tee -a "$LOG"
  $PY run_ablation.py --only "$exp" 2>&1 | tail -25 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== machine-B queue finished $(date) ===" | tee -a "$LOG"
