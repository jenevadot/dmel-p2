# TASKS

Open work items with enough context to act on without re-deriving anything.
Closed items move to the bottom with their outcome.

---

## T1 — Multi-seed `exp_020__dmel` vs `exp_001__baseline` (the thesis comparison)

**Status:** BLOCKED on `exp_020__dmel` completing (queued on Machine A, ETA below)
**Priority:** highest — this is the comparison the paper's central claim rests on
**Cost:** ~19 h per arm at 3 seeds on MPS; ~6 h/arm if run on Machine B (CUDA)

### Why this one and not all 31

Wang et al. average every reported number over **ten independent runs** and back
the DMEL-vs-baseline claim with a Wilcoxon signed-rank test at α=0.05 paired
across those runs (`dmel.txt:517`). Every run in this repo is **seed 42**, so we
currently cannot make any statement about seed variance.

Multi-seeding all 31 experiments is ~2,000 h and not worth it. Multi-seeding the
ONE comparison that carries the contribution is ~38 h and is.

### Preconditions

1. `exp_020__dmel` finished (`experiments/exp_020__dmel/metrics_dev.json` exists)
2. `exp_020` median NSE **beats** `exp_001` (0.6419). If it does not, this task
   changes shape — a negative result needs a different write-up, not more seeds.

### Steps

```bash
# 1. Both arms, three seeds each. Skips whatever already exists.
./run_queue.sh exp_001__baseline exp_020__dmel     # seed 42 already done
python run_ablation.py --only exp_001__baseline --seeds 123 777
python run_ablation.py --only exp_020__dmel    --seeds 123 777

# 2. Paired test. wilcoxon_nse() already exists and pairs BY BASIN.
python -c "
import sys; sys.path.insert(0,'src')
from evaluate import EvalResult, wilcoxon_nse
a=EvalResult.load('experiments/exp_020__dmel/metrics_dev.json')
b=EvalResult.load('experiments/exp_001__baseline/metrics_dev.json')
print(wilcoxon_nse(a,b))"
```

### Note on what the test actually claims

`wilcoxon_nse` (`src/evaluate.py:371`) pairs across **basins** (n≈508), not
across runs. That is a different — arguably stronger — claim than the paper's:
"DMEL is better across catchments" vs "DMEL is better across restarts". Report
both axes and say which is which. Do not describe the basin-paired p-value as
evidence of seed robustness.

### Done when

- 3 seeds per arm, all with `metrics_dev.json`
- Basin-paired Wilcoxon p-value recorded
- Across-seed median/IQR recorded per arm (the paper's box-plot equivalent)
- Per-horizon table for both arms — `h37_48` is where the claim lives

---

## T2 — Fix the `--limit` truncated-cache trap

**Status:** open
**Priority:** medium — cost us a full day once, currently dormant
**Cost:** ~15 min

`src/decompose.py:699` sizes the h5 datasets from `N` **only on creation**:

```python
N = min(N_total, limit) if limit else N_total
```

A `--limit 256` validation run therefore creates a 256-row cache. On a later
full run, `N` becomes 272,142 but `out["done"][:N]` on a 256-length dataset
silently returns 256 elements, all True — so `todo` is empty and it reports
`0 / 272,142 rows remaining` and exits. The built-in `--verify` then PASSES,
because the 200 rows it samples are real.

Two changes:
1. `--limit` writes to a distinct path (suffix the cache key) so it can never
   poison the canonical cache.
2. On open, raise if `len(out["done"]) != N_total` instead of trusting the
   `done` fraction.

Until then the rule is: **verify the SHAPE, not the done fraction.**

```bash
.venv/bin/python -c "
import h5py; f=h5py.File('data/rimf_train_110b9c26af82.h5','r')
d=f['done'][:]; print(f['rimf'].shape, int(d.sum()), '/', len(d))"
# want (272142, 3, 336) 272142 / 272142
```

---

## T3 — Reconcile `patience_requested` vs `patience_used`

**Status:** open
**Priority:** low
**Cost:** ~10 min to decide, 0 to implement if the override is intended

`exp_001__baseline/summary.json` records `patience_requested: 8` but
`patience_used: 20`. The log explains it — "cosine-family schedule needs to
reach its lr floor" — so it is deliberate, but the config says one thing and the
run does another. Either change the default to 20 or document the override at
the definition site so it is not read as a bug later.

---

## Closed

**Ridge baseline (B4)** — added `d02eaf8`. dev median NSE 0.4169. Fills the gap
between free arithmetic and a ~2.8M-param Informer; `informer − ridge` isolates
nonlinearity+attention. ARIMA rejected with reasoning recorded in
`src/baselines.py`.

**CEEMDAN crash** — fixed `f618808`, cache rebuilt and verified 2026-09-22.
`(272142, 3, 336)`, 272,142/272,142, 0 failed blocks, superposition max abs err
7.65e-03 mm/h on 400 independent windows.
