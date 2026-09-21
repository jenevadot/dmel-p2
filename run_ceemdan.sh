#!/bin/bash
# ── CEEMDAN / MRS precompute ────────────────────────────────────────────
# Run this in YOUR terminal, with the training queue stopped, so all 14
# cores are available:
#
#     ./run_ceemdan.sh
#
# Resumable: each 256-row block sets its `done` flag LAST and flushes, so a
# crash or ctrl-c loses at most 256 windows. Re-running picks up where it
# stopped. Safe to stop and restart at any time.
#
# Timing measured on this machine:
#     4 procs, training running :  1.5 win/s  -> ~56 h
#    12 procs, machine idle     : ~5.0 win/s  -> ~17 h
set -eu
cd "$(dirname "$0")"
PY=.venv/bin/python
NPROC="${1:-12}"

echo "=== CEEMDAN precompute: $NPROC processes ==="
echo "train.h5: 272,142 windows   test.h5: 27,983 windows"
echo "Stop the training queue first:  pkill -f run_queue.sh"
echo

caffeinate -dimsu $PY src/decompose.py --split train --n-proc "$NPROC" \
    --block 256 --verify 2>&1 | tee -a experiments/ceemdan_train.log

caffeinate -dimsu $PY src/decompose.py --split test  --n-proc "$NPROC" \
    --block 256 --verify 2>&1 | tee -a experiments/ceemdan_test.log

echo
echo "=== done. Now run the DMEL ablations: ==="
echo "  ./run_queue.sh exp_020__dmel exp_021__dmel_mlp_combiner \\"
echo "                 exp_023__dmel_se_threshold exp_024__dmel_univariate"
