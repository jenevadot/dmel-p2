# HANDOFF — DMEL Streamflow Forecasting: Review Findings & Action Items

> **Purpose**: This document hands off three completed technical assessments of the
> `paper2/` DMEL implementation. It contains measured evidence, conclusions, and a
> prioritised action list. Nothing in these assessments has been implemented yet.
>
> **Audience**: an engineering agent picking up this codebase.
>
> **Status**: implementation is code-complete but **has never been run**. Two hard
> blockers prevent training from starting.

---

## 0. Project context

**Task**: multi-step streamflow forecasting — predict 48 hours of specific discharge
(mm/h) from 336 hours of history, across 508 anonymous river basins.

**Method being replicated**: DMEL (Wang et al., *Applied Soft Computing* 192, 2026) —
CEEMDAN signal decomposition → route high-frequency IMFs to an Informer and
low-frequency IMFs to an LSTM → sum predictions.

### Data

| File | Samples | Contents |
|---|---|---|
| `data/train.h5` | 272,142 | `X` (336×12), `y` (48,), `y_aux` (48×11), `split`, `basin_id` |
| `data/test.h5` | 27,983 | `X`, `basin_id` only — **no targets** |

- `split=0` → 254,000 samples (exactly 500/basin)
- `split=1` → 18,142 samples (11–40/basin) — the competition's own validation
- Channel 11 of `X` = `specific_discharge` = the target variable
- `y_aux` = 48h of **future** meteorological forcing, available at train/dev, **absent at test**

### Split discipline in use

| Split | Source | Size | Role |
|---|---|---|---|
| `train` | `split=0`, 88% | 223,520 | gradient updates |
| `san_val` | `split=0`, 12% | 30,480 | early stopping only |
| `dev` | `split=1` | 18,142 | ablation score — untouched during training |
| `test` | `test.h5` | 27,983 | final submission, unscoreable locally |

### Codebase layout

```
paper2/
├── src/
│   ├── config.py          # hyperparameters, seeds, device (CUDA>MPS>CPU)
│   ├── dataset.py         # HDF5 lazy load, BasinNormalizer, make_loader
│   ├── train.py           # optimizer groups, schedulers, losses, epoch loop
│   ├── evaluate.py        # NSE/KGE/RMSE/MAE, EvalResult, wilcoxon_nse
│   ├── early_stopping.py  # EarlyStopping, CheckpointManager
│   ├── experiment.py      # single-experiment orchestrator
│   └── models/
│       ├── informer.py    # ProbSparse Informer   <-- CONTAINS A BLOCKER BUG
│       ├── lstm_branch.py
│       └── dmel.py        # assembler + build_model() factory
├── run_experiment.py      # CLI, one experiment, ~30 flags
├── run_ablation.py        # CLI, 32-experiment grid, priority filtering
├── compare_results.py     # CLI, ranked scoreboard
└── experiments/<name>/    # per-run artefacts
```

Recently adopted (from a sibling project review): weight-decay parameter groups,
gradient-norm logging, cosine-aware patience relaxation, independent DataLoader RNG,
per-run normaliser persistence, and a `summary.json` run audit trail.

---

## 1. ASSESSMENT ONE — four proposed optimisations

A reviewer proposed four changes. Verdict: **one critical, one wrong-reasoning-right-
conclusion, one premature, one unnecessary.** Investigating them surfaced a blocker
that outranks all four.

### 1.0 🔴 BLOCKER FOUND (not in the proposal) — ProbSparse indexing bug

`src/models/informer.py`, method `_prob_qk`, ~line 145:

```python
K_samp = K[:, :, idx.view(B, H, -1)].view(B, H, T_q, sample_k, d)
```

This is **mixed basic + advanced indexing**. PyTorch retains the leading sliced dims
*and* inserts the advanced-index shape, yielding `(B,H, B,H,N, d)` instead of
`(B,H,N,d)`. Verified with numpy (identical semantics):

```
K shape            : (2,3,10,4)
idx.view(B,H,-1)   : (2,3,50)
K[:, :, idx]       : (2,3, 2,3,50, 4)   <-- wrong
elements produced  : 7,200
elements needed    : 1,200
```

Scaling to production shapes (`H=8, T=336, sample_k=25, d=32`):

| Batch | Allocation | Result |
|---|---|---|
| 4 (self-test) | 1.0 GB | passes |
| 16 | 16.4 GB | heavy |
| 32 | 65.6 GB | OOM |
| **64 (our default)** | **262.5 GB** | **OOM** |

**Why the self-test passed**: it used `B=4` and asserted only the *final output shape*.
`.view()` forcibly reshapes the wrong tensor back, so shape assertions pass while the
values are garbage. The test validated shape, not correctness.

Line 143 immediately above is also dead code — `K.unsqueeze(-2).expand(...)` is
overwritten on the next line without being used.

**Training cannot start until this is fixed.**

### 1.1 ✅ "CEEMDAN must be offline" — correct, understated, currently moot

Measured cost at our scale:

```
per 336h window : 100 trials × 9 IMFs × ~7 sifts = 6,300 sifting passes
dataset         : × 272,142 windows = 1.71 BILLION sifting passes
PyEMD benchmark : ~2–8 s per 336-pt window, single core
                → ~378 core-hours for ONE pass
                → ~38 h on 10 cores
```

Online (in-DataLoader) CEEMDAN is not "slow" — it is impossible. The reviewer is right.

**Two things the proposal missed:**

**(a) It is moot right now.** Every experiment in the grid except
`exp_020__shared_informer` runs `use_ceemdan=False`. Ablation `A-OUR-1` (*is CEEMDAN
needed at all?*) is already ranked **critical**. If the baseline matches DMEL without
decomposition, this work never needs doing.

**(b) A conceptual problem outranks the compute problem.** DMEL decomposes **one
continuous multi-year series**. We have **272k independent 336h windows**. CEEMDAN's
IMFs are defined relative to the full record's extrema; decomposing a 336h slice
produces IMFs that are not comparable across windows — IMF₄ in window A ≠ IMF₄ in
window B. The entire "route IMF *k* to model *M*" premise may not survive this.

**Action**: run `A-OUR-1` first. Only if decomposition demonstrably helps, resolve the
windowing semantics, *then* build an offline precompute → `data/train_ceemdan.h5`.

### 1.2 ⚠️ "Enforce chronological splitting" — right instinct, wrong prescription

**Finding A — the competition's own `split=1` is randomly interleaved, not chronological:**

```
basin   0 split codes: [0 0 0 1 0 1 0 0 0 0 ... 1 0 0 1 1 0 1 0]
basin   7 split codes: [1 0 0 0 0 0 0 0 0 0 ...]
val positions basin 0: [3, 5, 22, 25, 26, 28, 61, 65, 100, 120, ...]
```

Forcing `san_val` to be chronological would make it a **different distribution** from
the `dev` set we are scored on. That is a mismatch, not a fix.

**Finding B — windows overlap massively. Exhaustively verified** (6-gram index, every
sample pair within a basin):

| Basin | Samples | Overlapping pairs | Cross train/val |
|---|---|---|---|
| 0 | 540 | 2,919 | 311 |
| 7 | 535 | 3,816 | 159 |
| 200 | 528 | 7,773 | 375 |
| **Total (3 basins)** | | **14,508** | **845** |

Severity, basin 200 (61,398 overlapping pairs):

```
p1  : shift=  4h → 332h shared (98.8% of window)
p5  : shift= 17h → 319h shared (94.9%)
p25 : shift= 85h → 251h shared (74.7%)
p50 : shift=168h → 168h shared (50.0%)
p95 : shift=314h →  22h shared ( 6.5%)

6,090 pairs share >90% of their window
30,508 pairs share >50% of their window
```

**Conclusion**: leakage is real and severe — but it is **baked into the competition's
own split**, not introduced by our carve. 845 cross-split overlapping pairs exist
before we touch anything.

**Action**: do **not** impose chronological splitting (breaks distribution-match with
`dev`). Instead add an **overlap-aware `san_val` carve** — exclude from `train` any
sample whose 336h window overlaps a `san_val` window — so our *internal selection
signal* is clean. Document that `dev` itself remains optimistic.

### 1.3 🟡 Mixed precision (AMP) — valid on CUDA, risky on MPS, premature

- **CUDA**: genuine ~2× throughput on Ampere+. Worth adopting eventually.
- **MPS**: `bf16` autocast has had correctness bugs through recent PyTorch. Not a drop-in.
- **Interaction hazard**: AMP + `clip_grad_norm_` requires `scaler.unscale_()` first,
  or clipping operates on scaled gradients and the `mean_grad_norm` logging just added
  becomes meaningless.

We have **zero timing measurements**. Model is only 2.4M params. Unknown whether we are
compute-bound or data-bound.

**Action**: defer until the baseline reports seconds/epoch. Then adopt CUDA-only,
guarded, with correct unscale-before-clip ordering.

### 1.4 ⚪ Gradient accumulation — solving a problem we do not have

Memory budget: model 10 MB, AdamW states 20 MB, SDPA attention O(L) not O(L²), 8 GB
available. Nothing indicates batch=64 will not fit.

Adds a silent-breakage risk: the LR scheduler must step per **optimizer** step, not per
micro-batch, or the cosine schedule runs N× too fast.

**Action**: skip. Revisit only if a specific config actually OOMs.

### 1.5 ✅ Replace ProbSparse with SDPA — strongest proposal, reasoning partly wrong

The stated reason (*"ProbSparse needs custom CUDA kernels that won't run on MPS"*) is
**false for our code** — it is written in pure PyTorch ops. But the conclusion is right
for better reasons.

**At our sequence length the sparsity collapses:**

```
L = 336, factor = 5
u = 5 × int(ln 337) = 5 × 5 = 25 active queries

  25 of 336 queries ( 7.4%) receive real attention
 311 of 336 queries (92.6%) receive V.mean() — a CONSTANT vector
```

**The memory argument inverts:**

| | Elements | Memory |
|---|---|---|
| Full attention matrix | 57.8 M | 220 MB |
| **ProbSparse sampling intermediate** | **137.6 M** | **525 MB** |
| SDPA / FlashAttention | never materialised | O(L) |

**ProbSparse's intermediate is 2.4× larger than the full matrix it replaces.**
ProbSparse pays off at L≈5000+, not L=336.

**Action**: adopt SDPA. Since the current implementation is broken anyway, this is the
*cheaper* fix. Make attention a swappable backend (`--attention sdpa|probsparse`) so
ablation `I7` remains measurable rather than assumed.

---

## 2. ASSESSMENT TWO — loss function for hydrology extremes

**Proposal**: *"MSE is heavily skewed by flood peaks. While NSE is correctly tracked as
a validation metric, you should explicitly plan to test an NSE-Loss or Huber Loss as
the training objective (T3). Aligning the training loss with the physical/hydrological
evaluation metric often yields the best empirical results."*

Verdict: **one assertion correct, one already satisfied, one a non-sequitur** — and
verifying it exposed a second bug.

### 2.1 ✅ "MSE is peak-skewed" — CORRECT

Measured on 1.92M z-scored training targets:

```
z-scored target distribution:
  p50   :  -0.20 σ
  p90   :   0.64 σ
  p99   :   3.89 σ
  p99.9 :  10.60 σ
  p100  :  82.18 σ
```

| Slice | Share of total **squared** error | Share of total **absolute** error |
|---|---|---|
| top 0.1% | **30.4%** | 3.4% |
| top 1% | **62.1%** | 14.1% |
| top 5% | **83.5%** | 32.6% |

1% of samples drive 62% of the gradient. Premise is sound.

### 2.2 ❌ "NSE-Loss fixes it" — NON-SEQUITUR

```
NSE = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²
           └───┬───┘   └───┬───┘
       same squared term  constant w.r.t. ŷ
```

The numerator is **identical to MSE**. The denominator has zero gradient.
**NSE-loss is exactly as peak-skewed as MSE.** It cannot be the remedy for peak skew.

### 2.3 🔴 BUG — our `NSELoss` is MSE in disguise

`src/train.py`, class `NSELoss`:

```python
ss_res = ((y_true - y_pred) ** 2).sum()
ss_tot = ((y_true - y_true.mean()) ** 2).sum() + 1e-8
return ss_res / ss_tot
```

`ss_tot` depends only on `y_true` → it is a **constant** w.r.t. `y_pred`. Verified
numerically:

```
d(NSELoss)/dŷ = 2(ŷ−y) / ss_tot
d(MSE)/dŷ     = 2(ŷ−y) / n

gradient ratio g_nse/g_mse: min=1.007567  max=1.007567
constant across ALL elements: True
```

**The loss is MSE rescaled by a random per-batch factor** — effective-learning-rate
jitter, not a distinct objective. Also, it aggregates over the whole batch while the
evaluation metric computes NSE **per basin** then takes the **median across basins**.
`exp_012__nse_loss` as written would measure noise.

### 2.4 ⚠️ "Align train loss with eval metric" — ALREADY DONE, in the normaliser

```
Training MSE on z-scored y :  (1/N) Σ (y−ŷ)² / σ_b²
Kratzert (2019) NSE* loss  :        Σ (y−ŷ)² / (σ_b + ε)²
```

**Identical when ε = 0.** Per-basin z-scoring **is** the basin-variance-weighted NSE
loss. The requested alignment already exists — it lives in `BasinNormalizer`, not in
`build_loss()`.

**The real gap is ε = 0.** Per-basin σ measured across 508 basins:

```
p0  : 0.001064      p50 : 0.100715
p1  : 0.002755      p75 : 0.154494
p5  : 0.014495      p95 : 0.308516
p25 : 0.059619      p100: 0.554344

ratio max/min = 521×
14 basins (2.8%) have σ < 0.01  →  essentially flat lines
```

Dividing a flat basin's error by σ² amplifies **pure noise** to the same gradient
magnitude as a genuinely dynamic basin. That is exactly what Kratzert's ε damps.

### 2.5 ✅ "Test Huber" — CORRECT, and the only real lever

Huber is the only proposed option that changes the **error exponent** (quadratic →
linear past δ), which is the only mechanism that actually reduces peak dominance.

Default sanity-check:

```
median |z| = 0.311 σ        p99 |z| = 3.89 σ        max |z| = 82.2 σ
93.1% of z-scored targets have |z| < 1.0
```

`HuberLoss(delta=1.0)` is quadratic for 93% of data, linear only for extremes —
reasonable, but **δ must match the error scale**, not be left at the PyTorch default.

### 2.6 Summary table

| Assertion | Verdict |
|---|---|
| MSE is peak-skewed | ✅ Correct — 1% of samples → 62% of gradient |
| NSE-loss fixes that | ❌ Non-sequitur — identical squared numerator |
| Align train loss with eval metric | ⚠️ Already satisfied via per-basin z-score |
| Test Huber | ✅ Correct — the only genuine peak-skew lever |

---

## 3. ASSESSMENT THREE — baseline model and targets

### 3.1 Measured baselines on `dev` (508 basins, 18,142 samples)

| ID | Baseline | median NSE | mean NSE | >0.5 | >0.7 |
|---|---|---|---|---|---|
| **B0** | Predict basin mean | **0.0000** | 0.0000 | 0.0% | 0.0% |
| **B1** | **Persistence** (last value, flat 48h) | **0.5251** | 0.2292 | 52.6% | **35.8%** |
| B2 | Climatology (train hourly profile) | −0.0221 | −14.3134 | 0.0% | 0.0% |
| B3 | Persistence + 24h decay to mean | 0.4460 | −4.7089 | 43.9% | 10.0% |

**B1 (persistence) is the bar — not B0.** A model scoring median NSE 0.52 has learned
nothing beyond *"tomorrow looks like today."*

B2's catastrophic mean (−14.31) is diagnostic: a fixed hourly profile is *actively
harmful*, confirming discharge is driven by **event timing**, not diurnal cycle.

### 3.2 The difficulty is entirely in the long leads

```
persistence NSE by forecast lead time
  h+1    0.9985  #######################################
  h+6    0.9574  ######################################
  h+12   0.8408  #################################
  h+24   0.5810  #######################
  h+36   0.3453  #############
  h+48   0.2228  ########
```

Discharge is so autocorrelated that persistence is **near-perfect at h+1** and collapses
to 0.22 by h+48.

**This reframes the evaluation.** A single aggregate median-NSE is dominated by easy
early hours. The model's entire value lies in **h+24 → h+48**, where persistence fails —
and precisely where `y_aux` (future precipitation) should pay off, since only incoming
rainfall can predict a rise the history window cannot see.

**Recommendation**: report NSE by lead-time bucket (h+1–12, h+13–24, h+25–36, h+37–48),
not only the aggregate. Otherwise a model that merely learns persistence will look strong.

### 3.3 Targets

**Aggregate:**

| Tier | median NSE | Meaning |
|---|---|---|
| Floor | 0.53 | Ties persistence — **no value added** |
| Minimum viable | 0.65 | Beats persistence meaningfully |
| Good | 0.75 | Competitive |
| Strong | 0.85 | DMEL-paper territory |

**Per-horizon — where it actually matters:**

| Lead | Persistence | Target |
|---|---|---|
| h+1–12 | 0.84–1.00 | ≥ 0.95 (should be easy) |
| h+13–24 | 0.58 | **≥ 0.75** |
| h+25–36 | 0.35 | **≥ 0.65** |
| h+37–48 | 0.22 | **≥ 0.55** ← the real test |

### 3.4 The baseline model

`exp_001__baseline`, already defined in `run_ablation.py`:

```
single Informer · all 12 channels · per-basin z-score
AdamW · warmup-cosine · MSE · batch 64
no CEEMDAN · no y_aux · no basin_emb
```

Deliberately excludes `y_aux`, `basin_emb`, and CEEMDAN so each can be measured as a
clean delta against it.

---

## 4. CONSOLIDATED ACTION ITEMS

### 🔴 P0 — Blockers. Training cannot start.

**A1. Fix the attention implementation.**
- File: `src/models/informer.py`, `ProbSparseSelfAttention._prob_qk`
- Current code allocates 262 GB at batch=64 due to mixed basic/advanced indexing
- **Recommended**: replace with `torch.nn.functional.scaled_dot_product_attention`
  as the default backend, since at L=336 ProbSparse is both broken *and* slower
  (525 MB intermediate vs 220 MB full matrix; only 25/336 queries active)
- Keep ProbSparse behind `--attention probsparse` (correctly implemented) so `I7`
  stays a measurable ablation
- Remove the dead `K.unsqueeze(-2).expand(...)` line
- **Add a self-test at batch=64**, not batch=4 — the current test only validates shape

**A2. Fix or remove `NSELoss`.**
- File: `src/train.py`, class `NSELoss`
- Currently mathematically identical to MSE up to a random per-batch scalar
- **Recommended**: reimplement as per-basin with ε damping:
  ```
  loss = Σ (y_n − ŷ_n)² · σ_b² / (σ_b + ε)²     with ε ≈ 0.1
  ```
  (requires passing `basin_id` and the normaliser's `σ_b` into the criterion)
- If not reimplemented, **delete it** and remove `exp_012__nse_loss` — shipping it
  as-is produces a meaningless ablation

### 🟠 P1 — Correctness, before the baseline run

**A3. Overlap-aware `san_val` carve.**
- File: `src/dataset.py`, `build_splits()`
- Exclude from `train` any sample whose 336h window overlaps a `san_val` window
- Measured: 14,508 overlapping pairs across just 3 basins; 6,090 pairs share >90%
- Do **not** switch to chronological splitting — the competition's `split=1` is
  randomly interleaved, so that would break distribution-match with `dev`
- Document in `summary.json` that `dev` itself retains inherent overlap

**A4. Per-horizon NSE reporting.**
- File: `src/evaluate.py`
- Add NSE by lead-time bucket: h+1–12, h+13–24, h+25–36, h+37–48
- Without this, aggregate NSE is dominated by trivially-easy early hours and a
  persistence-equivalent model will appear strong

**A5. Record persistence baseline as a first-class artefact.**
- Write `experiments/baseline_persistence/metrics_dev.json` using the same
  `EvalResult` schema so `compare_results.py` ranks every model against B1=0.5251
  automatically rather than against B0=0

### 🟡 P2 — First runs

**A6. Run `exp_001__baseline`, single seed (42).**
- Capture: seconds/epoch, `mean_grad_norm`, `sanval_to_dev_gap`, best epoch
- Everything in P3 depends on these measurements

**A7. Run `A-OUR-1` (CEEMDAN necessity check).**
- Already `exp_004__no_decomp_check` in the grid
- Gates ~378 core-hours of precompute work

**A8. Add `--huber_delta` and sweep {0.5, 1.0, 2.0}.**
- The only proposed change that genuinely addresses peak skew
- Do not ship the PyTorch default unexamined: 93.1% of z-scored targets have |z|<1.0

### ⚪ P3 — Conditional / deferred

| Item | Condition |
|---|---|
| **A9.** AMP (bf16 autocast) | Only after A6 shows we are compute-bound. CUDA-only. Must `scaler.unscale_()` before `clip_grad_norm_` or `mean_grad_norm` logging breaks |
| **A10.** CEEMDAN offline precompute → `data/train_ceemdan.h5` | Only if A7 shows decomposition helps. **First** resolve whether per-window IMFs are comparable across windows |
| **A11.** ε-damping ablation, ε ∈ {0, 0.1, 0.5} | After A2. Tests whether the 14 low-σ basins are degrading training |
| **A12.** SDPA vs ProbSparse ablation | After A1, once both backends are correct |
| **A13.** Gradient accumulation | **Skip.** No evidence of VRAM pressure. Revisit only on an actual OOM |

---

## 5. THINGS TO EXPLICITLY NOT DO

| Do not | Reason |
|---|---|
| Impose chronological train/val splitting | Competition's `split=1` is randomly interleaved; would break distribution-match with `dev` |
| Add gradient accumulation | Memory budget shows no pressure; risks silent scheduler-stepping bug |
| Run CEEMDAN inside the DataLoader | 1.71 billion sifting passes ≈ 378 core-hours per epoch |
| Trust `exp_012__nse_loss` as currently written | It is MSE with random per-batch rescaling |
| Judge models against B0 (NSE=0) | The real bar is B1 persistence at **0.5251** |
| Report only aggregate median NSE | Dominated by h+1–12 where persistence already scores 0.84–1.00 |
| Trust the `informer.py` self-test | Runs at batch=4 and asserts shape only; masks the 262 GB bug |

---

## 6. REPRODUCING THE EVIDENCE

All findings are reproducible with read-only scripts against `data/train.h5`.

```bash
# Environment (not yet created — torch is NOT installed)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # M4
uv pip install -r requirements.txt

# Grid inspection works without torch (lazy import)
python run_ablation.py --dry_run
```

Key verifications performed:

| Finding | Method |
|---|---|
| ProbSparse 262 GB bug | numpy mixed-indexing shape reproduction (same semantics as torch) |
| Window overlap | 6-gram index of discharge channel, exhaustive pairwise within basin |
| `split=1` interleaved | direct inspection of `split` codes ordered by row index per basin |
| NSELoss ≡ MSE | analytic gradient ratio, confirmed constant across all elements |
| Peak skew shares | squared vs absolute error contribution by percentile on z-scored targets |
| Baselines B0–B3 | per-basin NSE on `dev`, four closed-form predictors |
| Persistence by lead | per-basin NSE at h ∈ {1,6,12,24,36,48} |

---

## 7. ONE-PARAGRAPH SUMMARY

The implementation is code-complete but unrun, and contains **two bugs that must be
fixed first**: a ProbSparse attention indexing error that would allocate 262 GB at the
default batch size, and an `NSELoss` that is mathematically identical to MSE. Of four
proposed optimisations, replacing ProbSparse with SDPA is correct and now doubly
justified; offline CEEMDAN precompute is correct but gated behind an ablation that may
render it unnecessary; chronological splitting is the wrong fix for a real leakage
problem (windows overlap heavily, but the competition's own split is randomly
interleaved, so an overlap-aware carve is the right response); AMP is worth doing after
timing data exists; and gradient accumulation is unnecessary. On the loss question, MSE
*is* peak-skewed as claimed, but NSE-loss shares the identical squared numerator and
cannot fix it, while per-basin z-scoring already makes MSE equivalent to Kratzert's NSE*
loss at ε=0 — so the requested train/eval alignment is already present and the real gap
is ε-damping for the 14 near-flat basins. Huber is the only proposed change that genuinely
reduces peak dominance, and its δ must be tuned rather than left at default. Finally, the
baseline to beat is **persistence at median NSE 0.5251**, not the zero-skill mean
predictor, and because persistence scores 0.9985 at h+1 but only 0.2228 at h+48, all
evaluation must be reported **per lead-time bucket** or a persistence-equivalent model
will look deceptively strong.
