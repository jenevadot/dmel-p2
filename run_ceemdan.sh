#!/bin/bash
# ── CEEMDAN / MRS precompute ────────────────────────────────────────────
# Run this in YOUR terminal, with the training queue stopped, so all cores
# are available:
#
#     ./run_ceemdan.sh            # 12 procs (default)
#     ./run_ceemdan.sh 8          # fewer, if the machine is under pressure
#
# Resumable: each block sets its `done` flag LAST and flushes, so a crash or
# ctrl-c loses at most one block. Re-running picks up where it stopped. Safe
# to stop and restart at any time.
#
# READ THIS BEFORE RE-LAUNCHING (2026-09-21)
# ──────────────────────────────────────────
# The first attempt died at 256/272,142 windows (0.094%) with BrokenPipeError
# in the worker pool and 6 leaked semaphores. Two causes, both now fixed in
# src/decompose.py:
#
#   1. --block was 256. At a measured 1.74 s/window that is ~7.4 min of work
#      inside a worker before it returns anything, so nothing was ever
#      checkpointed before the pool collapsed. Default is now 32 (~56 s).
#   2. A single failed block propagated out of the pool and discarded every
#      other worker's completed-but-unflushed work. Failures are now caught
#      per block, logged, and left for the next run to retry.
#
# It was also launched while training held the machine. Do not do that: the
# box was already ~5.9 GB into swap. Stop the queue first.
#
# Timing measured on this machine (M4, 14 cores):
#     1 proc,  idle              :  1.74 s/window
#    12 procs, machine idle      : ~5.0 win/s  -> ~15-17 h for train.h5
# Training co-scheduled roughly triples that and risks the same crash.
set -eu
cd "$(dirname "$0")"
PY=.venv/bin/python
NPROC="${1:-12}"

echo "=== CEEMDAN precompute: $NPROC processes ==="
echo "train.h5: 272,142 windows   test.h5: 27,983 windows"
echo

# Refuse to start if the training queue is alive — this is the failure mode
# that killed the first run, so make it impossible rather than documented.
if pgrep -f run_queue.sh > /dev/null 2>&1; then
  echo "ERROR: run_queue.sh is running. CEEMDAN needs the machine to itself."
  echo "       Stop it first:  pkill -f run_queue.sh"
  exit 1
fi

caffeinate -dimsu $PY src/decompose.py --split train --n-proc "$NPROC" \
    --block 32 --verify 2>&1 | tee -a experiments/ceemdan_train.log

caffeinate -dimsu $PY src/decompose.py --split test  --n-proc "$NPROC" \
    --block 32 --verify 2>&1 | tee -a experiments/ceemdan_test.log

echo
echo "=== done. Now run the DMEL ablations: ==="
echo "  ./run_queue.sh exp_020__dmel exp_021__dmel_mlp_combiner \\"
echo "                 exp_023__dmel_se_threshold exp_024__dmel_univariate"
