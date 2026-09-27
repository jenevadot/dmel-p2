# Conclusions & Next Experiments
> Auto-generated from technical review · 25 Sep 2026

---

## Conclusions

### What Works

1. **Loss function is the dominant hyperparameter.**
   MAE > Huber-δ0.5 > Huber-δ1.0 > MSE > NSE loss for this distribution.
   The heavy-tailed, multi-basin runoff data responds strongly to robust loss
   functions. MAE reaches dev median NSE **0.6830** (+0.043 above the 3-seed
   baseline mean of 0.6399), which is the only single-change result that
   clears the 2σ significance bar. This is a publishable finding on its own.

2. **336h context window is necessary.**
   Halving to 168h drops dev NSE by −0.072 (significant) and collapses
   h37-48 from 0.468 → 0.312. Long antecedent soil-moisture and precipitation
   accumulation is genuinely captured in the second week of history.

3. **Per-basin z-score normalisation is non-negotiable.**
   Global z-score causes early stopping at epoch 6 and a +0.093
   sanval-to-dev gap. Discharge spans orders of magnitude across 508 basins;
   a shared scale destroys basin-specific dynamics.

4. **Model size is not a constraint.**
   d_model=128 (719 K params) matches d_model=256 (2.85 M) within noise.
   d_model=512 (11.3 M) actually *underfits* on unseen basins (+0.072 gap).
   The smallest tested architecture is sufficient — the problem is data-limited,
   not capacity-limited.

5. **Auxiliary task regularises without hurting.**
   aux_task=True gives +0.020 on dev, tightens the generalisation gap from
   +0.023 → +0.011, and trains stably for all 40 epochs. The cleanest
   single-change improvement after MAE.

6. **Gradient clipping (clip_grad=1.0) is necessary.**
   Removing it hurts both dev (−0.012) and sanval (−0.045) and inflates the
   gap to +0.056. The mean grad norm of ~2.2–2.9 confirms the clipper is
   active and beneficial across all MSE/MAE runs.

---

### What Doesn't Work (Yet)

1. **DMEL — the paper's core contribution — is currently failing.**

   | Experiment | dev NSE | sanval | gap | h37-48 |
   |---|---|---|---|---|
   | exp_020__dmel (fixed_split) | 0.3971 | 0.5873 | −0.190 | 0.113 |
   | exp_023__dmel_se_threshold | 0.5039 | 0.5821 | −0.078 | 0.291 |
   | exp_001__baseline (MSE) | 0.6419 | 0.6186 | +0.023 | 0.468 |

   The −0.243 gap below baseline mean and the massive sanval-to-dev collapse
   is a hard blocker for any submission. DMEL *trains* on sanval (0.587) but
   **fails to generalise** to the 61 held-out dev basins. The RIMF
   decomposition structure is basin-specific; high/low-frequency branch
   routing that works for training basins does not transfer to unseen ones.

   **Root cause hypotheses (ranked):**
   - **H1 (most likely):** RIMF cache quality on dev basins. Verify
     superposition error per-basin and check whether dev basins have
     pathological CEEMDAN outputs (flat signals, extreme outliers).
   - **H2:** K=3 fixed branches (2 HF → Informer, 1 LF → LSTM) misalign
     for diverse basin types (snowmelt vs rain-dominated). A basin-adaptive
     routing or larger K may be needed.
   - **H3:** The 15.3 M parameter dual-branch model is overparameterised
     for ~223 K training windows. Model too large relative to data.

2. **NSE loss (train the metric you score) is a negative result.**
   dev NSE drops to 0.6075 (−0.032 below baseline mean, significant).
   Per-basin variance normalisation in the NSE loss creates gradient variance
   that hurts unseen basins despite looking decent on sanval (0.654).

---

### The Risk Picture

The honest current state: **a strong single-branch Informer with per-basin
normalisation and MAE loss reaches 0.683 dev NSE**, beating persistence
by +0.158 and ridge by +0.266. DMEL as implemented is worse than the
baseline at every horizon.

Two paths forward:

| Path | Requirement | Risk |
|---|---|---|
| **A — Fix DMEL** | Diagnose root cause, fix branch routing or cache quality, re-run multi-seed comparison | High effort, necessary for the original paper claim |
| **B — Reframe contribution** | "Per-basin Informer + MAE is a strong simple baseline for multi-basin runoff" | Lower effort, different but publishable paper |

Path A is worth pursuing — the SE-threshold variant (0.504) is already
meaningfully better than fixed_split (0.397), suggesting the decomposition
*can* be made to work with the right routing logic.

---

## Recommended Next Experiments

### 🔴 Priority 1 — High Value, Fast to Run

**EXP-040 · `mae + aux_task` combination**
```python
overrides = {"loss": "mae", "aux_task": True, "aux_loss_weight": 0.3}
```
Combine the two cleanest positive results. If effects are additive, expect
~0.690+. Each individually gives +0.043 and +0.020 over baseline; even
partial additivity would set a new high-water mark. ~500 min on MPS.

---

**EXP-041 · `mae` extended to 60 epochs**
```python
overrides = {"loss": "mae", "epochs": 60}
```
MAE best epoch was **32 of 40** with sanval still at 0.6386 and grad_norm
still falling (2.93 → 1.21 over the run). The cosine schedule had not
fully converged. Budget 20 more epochs. ~1150 min total (incremental ~380 min).

---

**EXP-034 re-run · `no_distil`** *(crashed — missing summary.json)*
```python
overrides = {"use_distil": False}  # identical config
```
History.csv has 38 epochs, best sanval 0.625 at ep28 — the run completed
training but crashed during post-training evaluation. Re-execute to recover
the dev/sanval metrics. ~710 min but already paid for.

---

### 🟠 Priority 2 — DMEL Diagnosis & Recovery

**DMEL cache audit** *(do this before any new DMEL run)*
```bash
python -c "
import h5py, numpy as np
f = h5py.File('data/rimf_train_*.h5', 'r')
rimf = f['rimf']
done = f['done'][:]
print('Shape:', rimf.shape)       # want (272142, 3, 336)
print('Done:', done.sum(), '/', len(done))
# Superposition error: reconstruct and compare to original
"
# Separately: check superposition error distribution specifically on the
# 61 dev basins vs the 447 train basins to see if dev basins are OOD.
```

---

**EXP-042 · `dmel + mae` loss**
```python
overrides = {"use_ceemdan": True, "loss": "mae"}
```
If the DMEL collapse is partly driven by MSE amplifying RIMF reconstruction
noise (outlier errors squared), switching to MAE should dampen it.
Diagnostic: if dev NSE improves substantially → H1/noise sensitivity.
If not → H2/H3 (structural branch mismatch or overparameterisation).

---

**EXP-043 · `dmel + d_model=128`** *(reduce overparameterisation)*
```python
overrides = {"use_ceemdan": True, "d_model": 128, "d_ff": 512}
```
15.3 M params for ~223 K training windows is an unfavourable ratio.
d_model=128 baseline matched d_model=256 on 2.85 M params; a smaller
DMEL model (≈3.8 M) may generalise better to dev basins. Faster to train
(~900 min vs 1537 min).

---

### 🟡 Priority 3 — Statistical Confirmation

**EXP-010 second seed · `huber_d05 + seed=123`**
```python
overrides = {"loss": "huber", "huber_delta": 0.5, "seed": 123}
```
Best generalisation gap in the board (−0.003), converged late at ep37,
gradient never clipped (gnorm=0.28). Needs a second seed to confirm.
Also consider extending to 60 epochs.

---

**EXP-005 second seed · `aux_task + seed=123`**
```python
overrides = {"aux_task": True, "aux_loss_weight": 0.3, "seed": 123}
```
Cleanest improvement after MAE with tightest gap (+0.011). Confirming
with a second seed validates it for the paper's ablation table.

---

**EXP-044 · `huber_d02`** *(explore the δ spectrum)*
```python
overrides = {"loss": "huber", "huber_delta": 0.2}
```
At δ=0.5 the gradient clip never fires (gnorm=0.28 ≪ 1.0). At δ=0.2
Huber becomes near-MAE for most errors while retaining quadratic
behaviour very close to zero. The optimal δ may sit between 0.2 and 0.5.

---

### 🟢 Priority 4 — Safe to Deprioritise

| Experiment | Reason |
|---|---|
| d_model variants (128 / 512) | Confirmed null — size doesn't matter in this range |
| LR variants (5e-4, 5e-5) | 1e-4 confirmed optimal; both deviations lose |
| ProbSparse attention | Neutral result, computationally heavier at seq_len=336 |
| batch_size=128 | Borderline negative (−0.028), stick with 64 |
| enc_layers=3 | Neutral, marginally better gap but no dev gain |
| dropout=0.2 | Suspicious large gap (+0.082); don't trust without 2nd seed |
| Channel ablations (exp_003, exp_004) | Large gaps suggest dev/sanval distribution artefact |

---

## Quick Reference — Key Numbers

| Metric | Value | Experiment |
|---|---|---|
| Best dev median NSE | **0.6830** | exp_012 (MAE) |
| Best generalisation gap | **−0.003** | exp_010 (Huber δ=0.5) |
| Baseline mean ± std (3 seeds) | 0.6399 ± 0.0146 | exp_001 s42/123/177 |
| Significance threshold (2σ) | ±0.029 | — |
| Persistence ceiling (h1-12) | 0.938 | baseline_persistence |
| Best h37-48 | **0.500** | exp_012 (MAE) |
| DMEL h37-48 | 0.113 | exp_020 |
| DMEL vs baseline delta | **−0.243** | exp_020 vs baseline mean |
