# Session Notes — DMEL vs MoE Comparison & CEEMDAN Sifting Deep Dive

> **Papers covered**
> - `dmel.txt` → Wang et al., *DMEL: A novel dual-modal ensemble learning architecture for multi-step runoff prediction*, Applied Soft Computing 192 (2026) 114792
> - `Informer-LTSM.pdf` → Rong et al., *Mixture of experts leveraging Informer and LSTM variants for enhanced daily streamflow forecasting*, Journal of Hydrology 653 (2025) 132737

---

## Part 1 — DMEL vs MoE: Implementation Differences

Both papers build an ensemble of **Informer + LSTM** for hydrological forecasting, but the philosophy, architecture, and preprocessing are fundamentally different.

### 1.1 Quick Reference Card

| Dimension | **DMEL** | **MoE-Informer-LSTM** |
|---|---|---|
| Journal / Year | Applied Soft Computing 2026 | Journal of Hydrology 2025 |
| Study area | Shuangpai & Fenghuang, China | Quinebaug River Basin, USA |
| Data period | 2002–2005 (~4 years) | 1950–1970 (~20 years) |
| Input variables | Univariate (runoff only) | Multivariate (streamflow + precip + evap + Tmax/Tmin) |
| Train / val / test split | 7 : 1.5 : 1.5 | 80 : 20 (train : val) |

---

### 1.2 Core Ensemble Paradigm

| | DMEL | MoE |
|---|---|---|
| **Strategy** | **Frequency-domain split** — decompose the signal, assign frequency bands to models | **Sample-wise routing** — all models predict every point; a router picks the best |
| **Key idea** | Each model is permanently specialized to a frequency band | Each model is a candidate expert; a meta-classifier picks the winner per sample |
| **Analogy** | Signal equalizer — different frequency bands go to different channels | Election — multiple candidates compete and the winner handles each case |

---

### 1.3 Preprocessing Pipeline

**DMEL** introduces the full **Modal Recognition Strategy (MRS)**:

```
Raw runoff q(t)
    │
    ▼
CEEMDAN (noise_std=0.2, N=100 trials)
    → IMF₁, IMF₂, ..., IMF₉  +  residual
    │
    ▼
Sample Entropy (SE) per IMFᵢ   [m=2, r=0.2·std]
    │
    ▼
Group IMFs with similar SE → Reconstructed IMFs (RIMFs)
    RIMFⱼ = Σ_{i∈Gⱼ} IMFᵢ
    │
    ▼
Split by SE relative to SE(original sequence):
    SE ≥ SE(q) → HIGH-frequency  → Informer
    SE ≈ 0–0.1 → LOW-frequency   → LSTM
```

Dimensionality reduction achieved:
- Shuangpai: 9 IMFs → 6 RIMFs (−33.3%)
- Fenghuang:  9 IMFs → 4 RIMFs (−55.6%)
- Runtime saving: 28.7–44.2% vs using all CEEMDAN components directly

**MoE** has **no signal decomposition**:

```
Raw multivariate features
    │
    ▼
Min-max normalization only
    │
    ▼
Fed directly to all 4 expert models
```

---

### 1.4 Model Components

| | DMEL | MoE |
|---|---|---|
| **Models** | 2: Informer + LSTM | 4 experts: Informer, LSTM, GRU, LSTM-S2S-Attention |
| **Router** | None — deterministic frequency split | 3 routers tested: Random Forest, LSTM, Transformer |
| **Informer role** | Handles **all high-frequency** RIMFs | One candidate expert among four |
| **LSTM role** | Handles **all low-frequency** RIMFs | One candidate expert among four |
| **Extra LSTM variants** | Standard LSTM only | GRU + LSTM-Seq2Seq with Attention |

---

### 1.5 Output Combination Method

**DMEL** — additive superposition (physics-motivated: IMFs sum back to original signal):

```
ŷ(t) = ŷ_Informer(RIMF_High(t))  +  ŷ_LSTM(RIMF_Low(t))
```

**MoE** — winner-takes-all routing:

```
Router classifies each sample → outputs expert label (0-3 for 4-class, 0-1 for 2-class)
    → Only that one expert's prediction is the final output
```

---

### 1.6 Training Procedure

**DMEL** — single-phase parallel training:
1. Apply MRS → decompose and split into RIMFs
2. Train Informer on each high-frequency RIMF
3. Train LSTM on each low-frequency RIMF
4. At inference: sum all outputs

**MoE** — three-phase sequential training:
1. Train all 4 expert models independently on the full dataset
2. For each training sample, compare residuals of all experts → assign ground-truth label = best-performing expert index
3. Train a router classifier (RF, LSTM, or Transformer) on those labels
4. At inference: router classifies the sample → route to winning expert

---

### 1.7 Prediction Horizons

| | DMEL | MoE |
|---|---|---|
| Lead times tested | 1, 3, 5, 7 days | 1, 3, 5, 7, 8 days |
| Key claim | Extends effective horizon 3 → 7 days without accuracy loss | Improves NSE by ~3% over best single model at lead ≥ 5 days |
| Anti-accumulation mechanism | Informer's **generative decoder** outputs all h steps in one forward pass | No unified mechanism; each expert uses its own decoder |

---

### 1.8 Evaluation Metrics

| Metric | DMEL | MoE | Notes |
|---|---|---|---|
| RMSE | ✅ | ✅ | Absolute error, peak-sensitive |
| NSE | ✅ | ✅ | Standard hydrologic skill score |
| KGE | ✅ | ❌ | Decomposes into correlation + variability + bias |
| R (Pearson) | ❌ | ✅ | Correlation |
| MAE | ❌ | ✅ | Robust to outliers |
| FHV | ❌ | ✅ | Bias for top 2% flows — flood peak performance |
| FLV | ❌ | ❌→✅ | Bias for bottom 30% flows — drought/low-flow performance |

MoE's FHV/FLV metrics provide explicit evaluation under extreme-flow conditions that DMEL does not address.

---

### 1.9 Core Innovations Summary

| | DMEL | MoE |
|---|---|---|
| **Novel contribution** | **MRS** — entropy-guided signal regrouping before model assignment | **MoE router** — a learned classifier routing each sample to the best expert |
| **Knowledge source** | Signal frequency characteristics (physics-informed preprocessing) | Learned per-sample residual performance patterns (data-driven routing) |
| **Computational cost** | Lower — MRS reduces components; fixed 2-model pipeline | Higher — 4 experts + router, all trained separately + grid search |
| **Interpretability** | High — rule: high-freq → Informer, low-freq → LSTM | Lower — router decisions depend on learned classification boundaries |

---

### 1.10 Bottom-Line Difference

> **DMEL** is a **signal decomposition-driven ensemble**: it solves the problem *before* modeling — by restructuring the signal into frequency bands and mapping each band to the architecturally best-suited model. The models are passive receivers of pre-separated signals.

> **MoE** is a **learned routing ensemble**: it solves the problem *during/after* modeling — by letting all experts compete on every sample and training a meta-classifier to pick the winner. The router is the active intelligence; the models are interchangeable experts.

---

---

## Part 2 — CEEMDAN Preprocessing: Deep Dive

### 2.1 The Algorithm Family Tree

```
EMD (Huang 1998)
 │  Problem: mode mixing (intermittent signals pollute frequency bands)
 │
 ▼
EEMD (Wu & Huang 2009)
 │  Fix: add white noise, average N trials → separates scales
 │  Problem: averaged IMFs don't sum back to original (residual noise)
 │
 ▼
CEEMDAN (Torres 2011)
   Fix: adaptive noise matched to each residual level → exact reconstruction
   Used in DMEL with: noise_std = 0.2,  N = 100 trials
```

---

### 2.2 Generation 1 — EMD Core Concepts

**The Intrinsic Mode Function (IMF)** is valid when it satisfies both:

1. `|#extrema − #zero-crossings| ≤ 1`  over the whole signal
2. Mean of upper and lower envelopes = 0 at every point

IMFs represent **AM-FM oscillations** — their amplitude and instantaneous frequency can drift over time. This is exactly what a river discharge (or any geophysical signal) does. Fourier decomposition cannot capture this because its basis functions have fixed frequency forever.

```
IMF₁  /\/\/\/\/\/\/\/\/\/\/\   fastest — storm responses, noise
IMF₂   /\  /\  /\  /\  /\     weekly flood pulses
IMF₃     /‾\    /‾\    /‾\    monthly cycles
IMF₄       /‾‾‾\    /‾‾‾\     seasonal swing
  ...
IMFₙ           ___/‾‾‾\___    annual or multi-year variation
residual  ─────────────────    long-term trend (monotonic)
```

**Reconstruction** — perfect, zero information lost:

```
         k
x(t) =  Σ  IMFᵢ(t)  +  rₙ(t)
        i=1
```

This is Equation (1) from the DMEL paper:
```
       k
q(n) = Σ IMFᵢ(t) + rₙ(t)
       i=1
```

---

### 2.3 Generation 2 — EEMD Problem

EEMD averages N noisy trials to reduce mode mixing:
```
For i = 1 to N:
    xᵢ(t) = x(t) + εᵢwᵢ(t)    ← different white noise each time
    EMD(xᵢ) → {IMFᵢ₁, IMFᵢ₂, ...}

Final IMFⱼ = (1/N) Σᵢ IMFᵢⱼ
```

**Problem:** `Σ averaged_IMFⱼ ≠ x(t)` — noise doesn't perfectly cancel, leaving a reconstruction residual.

---

### 2.4 Generation 3 — CEEMDAN Full Mathematics

**Notation:**
- `Eⱼ(·)` = operator that applies EMD and returns the **j-th mode**
- `wⁱ(t)` = i-th white noise realization, `wⁱ(t) ~ N(0,1)`
- `ε₀` = noise standard deviation (set to 0.2 in DMEL)
- `N` = number of ensemble trials (set to 100 in DMEL)

---

**Stage 1 — Extract IMF₁:**

Add noise to the original signal; average the first EMD mode across all trials:

```
         1   N
IMF₁ =  ─── Σ  E₁( x(t) + ε₀·wⁱ(t) )
          N  i=1
```

Stage 1 residual:
```
r₁(t) = x(t) − IMF₁(t)
```

---

**Stage k (k ≥ 2) — Extract IMFₖ with adaptive noise:**

The critical innovation: add noise **proportional to the current residual's frequency content**, not to the original signal:

```
           1   N
IMFₖ  =   ─── Σ  E₁( rₖ₋₁(t) + εₖ₋₁ · Eₖ(wⁱ(t)) )
            N  i=1
```

where `Eₖ(wⁱ(t))` is the k-th EMD mode of the white noise `wⁱ(t)`, then scaled by `εₖ₋₁`.

This means the injected noise is **matched to the frequency scale of the current residual** — as lower-frequency residuals are processed, finer-scale noise is injected that fits those longer oscillations.

Stage k residual:
```
rₖ(t) = rₖ₋₁(t) − IMFₖ(t)
```

---

**Stopping condition:**

Stop when `rₙ(t)` has fewer than 3 extrema (cannot be further decomposed by sifting).

---

**Exact reconstruction guarantee:**

```
         k
x(t) =  Σ  IMFᵢ(t)  +  rₙ(t)      ✓  (exact, no residual noise)
        i=1
```

This is the key advantage over EEMD.

---

### 2.5 CEEMDAN Parameters in DMEL

| Parameter | Value | Role | Sensitivity (Sobol) |
|---|---|---|---|
| Noise std (ε₀) | 0.2 | Controls noise injection amplitude | **Low** at both stations — robust |
| Ensemble trials (N) | 100 | Number of noisy realizations to average | — |
| SE embedding dim (m) | 2 | Pattern length for entropy calculation | **Dominant** at Shuangpai |
| SE tolerance (r) | 0.2·std | Matching threshold for entropy | **Dominant** at Fenghuang |

The Sobol sensitivity analysis confirms that `ε₀ = 0.2` is a safe fixed choice — the algorithm is robust to small changes in noise amplitude.

---

### 2.6 CEEMDAN Output in DMEL — Actual Numbers

After applying CEEMDAN to Shuangpai Station (SE of original sequence = 0.1769):

| Component | SE Value | Classification | Assigned To |
|---|---|---|---|
| IMF₁ | 0.1885 | SE > 0.1769 → High-freq | Informer (RIMF₁) |
| IMF₂ | 0.2226 | SE > 0.1769 → High-freq | Informer (RIMF₂) |
| IMF₃ | 0.1911 | SE > 0.1769 → High-freq | Informer (RIMF₃) |
| IMF₄ | 0.2593 | SE > 0.1769 → High-freq | Informer (RIMF₄) |
| IMF₅ | 0.2167 | SE > 0.1769 → High-freq | Informer (RIMF₅) |
| IMF₆ | 0.0509 | SE ≈ 0 → Low-freq | ⬐ grouped together |
| IMF₇ | 0.0305 | SE ≈ 0 → Low-freq | ⬐ → RIMF₆ (SE=0.099) |
| IMF₈ | 0.0162 | SE ≈ 0 → Low-freq | ⬐ |
| IMF₉ | 0.0014 | SE ≈ 0 → Low-freq | ⬐ → LSTM |

**Note the discontinuous jump** between IMF₅ (SE=0.2167) and IMF₆ (SE=0.0509) — this is the natural separation boundary that MRS exploits.

---

---

## Part 3 — The Sifting Process: Visual Walkthrough

### 3.1 What Sifting Is

> Sifting is the iterative **local mean subtraction** that extracts one IMF from a signal. It is the inner loop of EMD and CEEMDAN.

**One sifting iteration (sub-steps A → D):**

```
Input: hₖ₋₁(t)
  │
  ├─ A: Find all local maxima ▲ and local minima ▼
  │
  ├─ B: Fit cubic spline through all ▲  →  upper envelope  e_up(t)
  │     Fit cubic spline through all ▼  →  lower envelope  e_lo(t)
  │
  ├─ C: Compute local mean:
  │         e_up(t) + e_lo(t)
  │     m(t) = ─────────────────
  │                   2
  │
  └─ D: Subtract mean:
            hₖ(t) = hₖ₋₁(t) − m(t)
```

Then **check IMF conditions**. If both pass → `hₖ(t)` is the IMF. If not → use `hₖ(t)` as the new input and repeat.

---

### 3.2 Why Cubic Splines for Envelopes?

Cubic splines are used because they:
- Pass **exactly** through every knot (peak or valley) — the envelope touches every extremum
- Have **continuous 1st and 2nd derivatives** — no kinks in the skin
- Produce the smoothest possible curve through the knot set (minimum curvature energy)

This ensures:
```
e_up(t) ≥ x(t)  for all t    (upper skin never dips below the signal)
e_lo(t) ≤ x(t)  for all t    (lower skin never rises above the signal)
```

---

### 3.3 Why Subtract the Mean?

The local mean `m(t)` represents the **DC offset / lopsidedness** of the signal at each instant.

- If the signal oscillates symmetrically around zero → `m(t) ≈ 0` → already IMF-like
- If `m(t)` is large → the signal is asymmetric (rides on a trend or low-frequency component)

Subtracting `m(t)` **centers** the oscillations. After enough sifting iterations the mean converges toward zero and the result satisfies both IMF conditions.

---

### 3.4 IMF Condition Check (the stopping criterion per sift)

```
Condition 1:  |count(extrema) − count(zero-crossings)| ≤ 1
Condition 2:  mean(|m(t)|) / max(|hₖ(t)|)  <  5%  (relative mean deviation)
```

Condition 1 ensures the signal oscillates without "riding waves" (no extra wiggles that don't cross zero).

Condition 2 ensures the signal is symmetric enough around zero (the residual mean is negligible).

---

### 3.5 Complete Sifting Flowchart

```
                     ┌──────────────────────┐
                     │    Input signal       │
                     │    h₀(t) = x(t)       │
                     └──────────┬───────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  A: Find all local MAXIMA ▲   │
                │     Find all local MINIMA ▼   │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  B: Spline through ▲ → e_up   │
                │     Spline through ▼ → e_lo   │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  C: m(t) = (e_up + e_lo) / 2  │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  D: hₖ(t) = hₖ₋₁(t) − m(t)  │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  Check IMF conditions:        │
        ┌──NO───┤  C1: |extrema−zeros| ≤ 1 ?   ├──YES──┐
        │       │  C2: rel. mean dev < 5% ?     │       │
        │       └───────────────────────────────┘       │
        │                                               ▼
        │                          ┌───────────────────────────────┐
        ▼                          │  IMFₙ(t) = hₖ(t)  ✓ FOUND   │
 ┌────────────────┐                │  rₙ(t) = prev_input − IMFₙ   │
 │  hₖ(t) → new  │                └───────────────────────────────┘
 │  input, sift ↺ │
 └────────────────┘
```

---

### 3.6 The Onion Analogy (Full Decomposition Loop)

```
x(t)   ──sift──►  IMF₁  +  r₁(t)    [fastest oscillation peeled off]
r₁(t)  ──sift──►  IMF₂  +  r₂(t)    [next layer, slower]
r₂(t)  ──sift──►  IMF₃  +  r₃(t)    [next layer, slower still]
r₃(t)  ──sift──►  IMF₄  +  r₄(t)    ...
  ⋮                   ⋮
rₙ₋₁(t)           residual rₙ(t)     [monotone long-term trend = the core]
```

Each layer is a physically meaningful oscillatory MODE. The reconstruction is exact:

```
x(t) = IMF₁ + IMF₂ + IMF₃ + … + IMFₙ + rₙ(t)
```

---

### 3.7 Observed Sifting Convergence (from Python simulation)

On the synthetic runoff-like signal (N=80 samples):

| Iteration | Rel. deviation | |extrema−zeros| | Status |
|---|---|---|---|
| 1 | 27.27% | 9 | ✗ sift again |
| 2 | 11.27% | 1 | C1✓ C2✗ |
| 3 | 8.50% | 1 | C1✓ C2✗ |
| 4 | 5.79% | 1 | C1✓ C2✗ |
| 5 | 14.89% | 5 | ✗ sift again |
| 6 | 5.85% | 1 | C1✓ C2✗ |
| **7** | **2.35%** | **1** | **✓ IMF FOUND** |

IMF₁ extracted: 21 zero-crossings (fastest).
Residual r₁: 5 zero-crossings (slower remaining content).
Reconstruction error: `4.44e-16` ≈ machine zero ✓

---

### 3.8 Sparkline Summary of Full Decomposition (from simulation)

```
Component           Zero-cross   Character
──────────────────  ──────────   ──────────────────────────
x(t) [original]           5     messy mix of all frequencies
IMF₁                     21     ← high freq  (noise/storms)
IMF₂                      9     ← high freq  (storm cycles)
IMF₃                      3     ← mid  freq  (seasonal swing)
Residual                  1     ← trend      (long-term drift)
```

Reconstruction verified: `max|x(t) − Σ IMFᵢ(t) − r(t)| ≈ 0` ✓

---

### 3.9 How CEEMDAN Improves on Plain EMD Sifting

In CEEMDAN, the same sifting process is applied **N=100 times** per stage, each time on a slightly noisy version of the residual:

```
For each stage k:
    For i = 1 to 100:
        noisy_input = rₖ₋₁(t) + εₖ₋₁ · Eₖ(wⁱ(t))  ← adaptive noise
        sift noisy_input → get its first mode
    IMFₖ = average of all 100 first modes
```

**Why this prevents mode mixing:**

Intermittent features (e.g., a flood spike) are brief. In plain EMD, the sifting may "accidentally" fold their frequency content into the wrong scale because there's no reference for what counts as "same scale". The added noise acts as a **uniform frequency reference grid** — it forces scale separation by making sure that oscillations at different time scales consistently appear in different IMFs across all 100 trials. After averaging, the noise cancels but the scale-separation structure remains.

---

## Part 4 — Python Simulation Code

The sifting visualizer script (pure standard library, no numpy/scipy) is saved at:

```
/tmp/sifting_viz.py
```

**What it produces:**
- Step 0: Raw signal `x(t)` — a synthetic runoff-like mix (slow cosine + faster sine + burst + trend)
- Step A: Extrema detection (▲ peaks, ▼ valleys) on a 72×18 ASCII canvas
- Step B: Upper `▀▀▀` and lower `▄▄▄` cubic spline envelopes plotted together
- Step C: Mean envelope `━` overlaid on the signal — showing the lopsidedness before sifting
- Step D: First subtracted result `h₁(t)` — IMF condition check showing both conditions FAIL
- Iteration 2: Second sift on `h₁(t)` with new envelopes and condition check
- Convergence table: 7 iterations with progress bars until both conditions pass
- IMF₁ result: extracted first mode + residual, with reconstruction error check
- IMF₂ extraction: running sifting on `r₁(t)` to peel the next layer
- Final sparkline summary: all IMFs + residual with zero-crossing frequency staircase

**To re-run:**
```bash
python3 /tmp/sifting_viz.py
```

---

## Part 5 — Key Equations Reference

### CEEMDAN reconstruction (DMEL Eq. 1)
```
       k
q(n) = Σ IMFᵢ(t) + rₙ(t)
       i=1
```

### Sample Entropy (DMEL Eq. 2)
```
SE(m, r, L) = −ln( A(m+1) / B(m) )

  m = embedding dimension = 2
  r = similarity tolerance = 0.2 × std(IMFᵢ)
  L = time series length
  A(m+1) = number of matching vector pairs of length m+1
  B(m)   = number of matching vector pairs of length m
```

Higher SE → more complex, more irregular → high-frequency character.
Lower SE → more predictable, smoother → low-frequency character.

### RIMF grouping (DMEL Eq. 12)
```
RIMFⱼ = Σ_{i∈Gⱼ} IMFᵢ
```

### Final prediction (DMEL Eq. 14)
```
ŷ(t) = ŷ_Informer(RIMF_High(t))  +  ŷ_LSTM(RIMF_Low(t))
```

### CEEMDAN Stage k
```
           1   N
IMFₖ  =   ─── Σ  E₁( rₖ₋₁(t) + εₖ₋₁ · Eₖ(wⁱ(t)) )
            N  i=1

rₖ(t) = rₖ₋₁(t) − IMFₖ(t)
```

### Sifting local mean
```
m(t) = ( e_up(t) + e_lo(t) ) / 2

hₖ(t) = hₖ₋₁(t) − m(t)
```

---

## Part 6 — Connections to Existing Notes

| Topic | See also |
|---|---|
| DMEL architecture details, ablations, metrics | `DMEL_study_notes.md` |
| Full EMD/EEMD/CEEMDAN/VMD/STL comparison | `decomposition_methods_explained.md` |
| Informer model internals (ProbSparse, distilling, generative decoder) | `Informer_internals_and_reading_list.md` |
| EMD/CEEMDAN Python simulation with plots | `runoff_emd_deep_dive.py` + `runoff_emd_output.txt` |
| Dyadic filter bank (Wavelet vs EMD comparison) | `dyadic_filter_bank_demo.py` + `dyadic_filter_bank_output.txt` |

---

*Session date: 2025 — Pi coding agent session.*
