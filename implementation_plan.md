# DMEL Implementation Plan — Decision Registry & Data Setup

> **Task**: Reproduce and improve DMEL for the competition dataset.
> **Data**: 508 basins · 336-h history · 48-h forecast · 12 channels · 272,142 train+val samples.

---

## 1. What the Paper Does — Architecture Recap

```
q(t) ──CEEMDAN──► IMF₁…IMFₙ
                     │
               SampleEntropy
                     │
              group similar SE → RIMFⱼ
                     │
          ┌──────────┴──────────┐
       SE ≥ SE(q)           SE ≈ 0-0.1
       HIGH-FREQ              LOW-FREQ
          │                      │
       Informer               LSTM
    (ProbSparse attn)      (gating memory)
          │                      │
          └──────── + ───────────┘
                    │
                 ŷ(t)   (sum = reconstruction identity)
```

Key numbers from the paper (single-basin, daily runoff, 7-day horizon):
- Input window: 7 days, Output: 1/3/5/7 steps
- CEEMDAN: noise_std=0.2, N=100 trials → 9 IMFs
- SE params: m=2, r=0.2·std per IMF
- Informer: hidden=256, heads=8, enc_layers=2, dec_layers=1, ff_dim=1024, GELU
- LSTM: hidden=256, 2 layers
- Optimizer: Adam, lr=1e-4, epochs=100, batch=32, loss=MSE

---

## 2. Our Dataset vs Paper Dataset

| Property | Paper (DMEL) | Our Dataset |
|---|---|---|
| Granularity | Daily (24h) | Hourly (1h) |
| History window | 7 days | 336 h = 14 days |
| Forecast horizon | 1–7 days | 48 h = 2 days |
| Input type | Univariate (runoff only) | Multivariate (12 channels) |
| Basins | 2 stations | 508 basins |
| Total train samples | ~1,050 (per station) | 254,000 |
| Target variable | Daily runoff m³/s | Specific discharge mm/h |
| Normalization | Not applied (our data) | Not applied → **must normalize** |
| y_aux (future meteo) | Not available | Available at train/val, NOT at test |

Critical difference: our data is **multivariate, hourly, 508-basin** — DMEL was designed for
univariate daily single-station. Every design decision below accounts for this.

---

## 3. Hyperparameter & Architectural Decisions to Evaluate

### 3.1 CEEMDAN / MRS Decisions

| ID | Decision | Paper Value | Candidates to Test | Priority |
|---|---|---|---|---|
| C1 | CEEMDAN noise_std (ε₀) | 0.2 | {0.1, 0.2, 0.3} | Low (Sobol: insensitive) |
| C2 | CEEMDAN ensemble trials N | 100 | {50, 100, 200} | Low |
| C3 | SE embedding dim m | 2 | {1, 2, 3} | **High** (Sobol: dominant at Shuangpai) |
| C4 | SE tolerance r | 0.2·std | {0.1, 0.2, 0.3}·std | **High** (Sobol: dominant at Fenghuang) |
| C5 | Number of IMFs k | auto (9) | {6, 9, 12} | Medium |
| C6 | Decompose channel | discharge only (ch 11) | discharge only vs all 12 channels | **High** (novel to our data) |
| C7 | SE grouping criterion | similar SE → RIMF | {SE bucketing, k-means on SE, fixed-k split} | Medium |
| C8 | HF/LF boundary | SE ≥ SE(original) | {SE > original, top-K IMFs are HF, energy ratio} | Medium |
| C9 | Decompose on full series vs per-sample | full series (paper) | full series vs per-window (leakage concern) | **Critical** |

> ⚠️ **C9 is the most important architectural risk**: The paper decomposes the full time
> series before the train/val split, which causes data leakage through the spline envelopes.
> For a competition, we should evaluate: (a) decompose only on train split, apply same IMF
> structure to val/test; or (b) skip CEEMDAN entirely and let the model learn decomposition.

### 3.2 Informer (High-Frequency Branch) Decisions

| ID | Decision | Paper Value | Candidates to Test | Priority |
|---|---|---|---|---|
| I1 | Hidden dimension d_model | 256 | {128, 256, 512} | Medium |
| I2 | Number of attention heads | 8 | {4, 8, 16} | Medium |
| I3 | Encoder layers | 2 | {2, 3, 4} | Medium |
| I4 | Decoder layers | 1 | {1, 2} | Low |
| I5 | Feed-forward dimension | 1024 | {512, 1024, 2048} | Low |
| I6 | Activation function | GELU | {GELU, ReLU} | Low |
| I7 | ProbSparse factor (c) | default | {3, 5, 10} | Medium |
| I8 | Input length | 7 (days) → 168h | {168, 336} (7 or 14 days) | **High** |
| I9 | Generative decoder vs autoregressive | generative (one-shot) | generative vs AR | **High** (affects 48-step quality) |
| I10 | Distillation layers on/off | on | on vs off | Medium |

### 3.3 LSTM (Low-Frequency Branch) Decisions

| ID | Decision | Paper Value | Candidates to Test | Priority |
|---|---|---|---|---|
| L1 | Hidden size | 256 | {128, 256, 512} | Medium |
| L2 | Number of layers | 2 | {1, 2, 3} | Medium |
| L3 | Dropout | not specified | {0, 0.1, 0.2, 0.3} | **High** |
| L4 | Bidirectional | no | uni vs bi | Medium |
| L5 | Replace LSTM with GRU | LSTM | LSTM vs GRU | Medium |
| L6 | Add attention over LSTM | no | w/ and w/o attention | Medium |

### 3.4 Ensemble / Combination Decisions

| ID | Decision | Paper Value | Candidates to Test | Priority |
|---|---|---|---|---|
| E1 | Combination method | sum (additive) | {sum, learned linear, MLP combiner} | **High** |
| E2 | Loss per branch vs joint | per branch MSE | per-branch vs joint | Medium |
| E3 | Number of HF models | one Informer per RIMF_High | shared Informer vs one per RIMF | **High** (compute) |
| E4 | Number of LF models | one LSTM per RIMF_Low | shared LSTM vs one per RIMF | **High** (compute) |
| E5 | Use y_aux (future meteo) in HF branch | not in paper | feed y_aux to Informer decoder | **High** (novel to our data) |

> ⚠️ **E5 is the biggest opportunity**: `y_aux` contains 48h of future meteorological
> forcing (precip, temp, radiation, etc.) available at train/val time. This is exactly
> what the Informer generative decoder is designed to consume. The paper doesn't have this.

### 3.5 Training / Optimization Decisions

| ID | Decision | Paper Value | Candidates to Test | Priority |
|---|---|---|---|---|
| T1 | Learning rate | 1e-4 | {1e-3, 5e-4, 1e-4, 5e-5} | **High** |
| T2 | LR scheduler | not specified | {none, cosine, reduce-on-plateau} | **High** |
| T3 | Loss function | MSE | {MSE, MAE, Huber, NSE-loss, log-cosh} | **High** |
| T4 | Batch size | 32 | {32, 64, 128, 256} | Medium |
| T5 | Epochs | 100 | {50, 100, 200} + early stopping | Medium |
| T6 | Gradient clipping | not specified | {none, 0.5, 1.0} | Medium |
| T7 | Weight decay | not specified | {0, 1e-5, 1e-4} | Low |
| T8 | Normalization strategy | none in paper | {z-score global, z-score per basin, min-max, robust} | **Critical** |
| T9 | Normalization scope | — | normalize X all channels vs normalize only discharge for CEEMDAN | **High** |
| T10 | Basin embedding | none | {none, learned basin embedding added to input} | **High** (508 basins!) |

> ⚠️ **T8 is critical**: channels span completely different scales (pressure ~94k, humidity
> ~0.007). Without normalization the model cannot learn. Per-basin z-score is standard
> in hydrology (LSTM-for-Hydrology / CAMELS benchmark).

> ⚠️ **T10 is a major opportunity**: with 508 basins, a learned basin ID embedding
> (analogous to entity embeddings) allows the model to adapt its behavior per watershed
> without separate model copies.

### 3.6 Data / Input Decisions

| ID | Decision | Paper Value | Candidates to Test | Priority |
|---|---|---|---|---|
| D1 | Input channels to model | discharge only | discharge only vs all 12 channels | **Critical** |
| D2 | Include y_aux as decoder input | no y_aux in paper | yes (future meteo available) vs no | **High** |
| D3 | Time encoding | positional encoding | {sinusoidal, learnable, hour-of-day + day-of-year} | Medium |
| D4 | Channel normalization | — | per-basin running stats vs global stats | **High** |

---

## 4. Ablations from the Paper (Must Replicate)

The paper runs two explicit ablation studies:

### Ablation A — MRS vs. CEEMDAN-only (Table 8, Table 9)
Tests whether the entropy-grouping step adds value beyond raw CEEMDAN decomposition.

```
CEEMDAN-Informer   : 9 IMFs, one Informer per IMF, sum predictions
MRS-Informer       : 6 RIMFs, one Informer per RIMF, sum predictions
DMEL               : 6 RIMFs split HF→Informer + LF→LSTM, sum predictions
```

Key result: MRS-Informer reduces runtime 28.7–44.2%, improves 7-step NSE by 9.7%.

### Ablation B — Model Swap (Table 10)
Tests whether the Informer→HF / LSTM→LF assignment is optimal.

```
DMEL                  : HF → Informer,  LF → LSTM    ← claimed optimal
MRS-LSTM-Informer     : HF → LSTM,      LF → Informer ← swapped (baseline)
```

Key result: swap causes 7-step KGE to drop from 0.8759 → 0.6711 (−23.4%).

### Ablation C — Sensitivity Analysis (Appendix A)
Sobol variance decomposition over {noise_std, embed_dim, tolerance}:
- noise_std: low sensitivity at both stations → fix at 0.2
- embed_dim: dominant at Shuangpai
- tolerance: dominant at Fenghuang

### Ablation D — Robustness / Stress Tests (Appendix B)
- Inject 5% and 10% Gaussian noise into input → MRS adaptively reclassifies IMFs
- Introduce 5% and 10% missing values (linear interpolation) → marginal degradation

---

## 5. Our Additional Ablations to Run

Beyond what the paper tests, we should add:

| Ablation | What it tests |
|---|---|
| A-OUR-1 | DMEL vs no decomposition (raw input to Informer) — is CEEMDAN actually needed? |
| A-OUR-2 | With vs without y_aux as future forcing in Informer decoder |
| A-OUR-3 | With vs without basin embedding (shared model vs basin-conditioned) |
| A-OUR-4 | Per-basin normalization vs global normalization |
| A-OUR-5 | Decompose discharge channel only vs decompose all 12 channels |
| A-OUR-6 | Sum combiner vs learned combiner |

---

## 6. Data Split Strategy

### Existing splits in train.h5
```
split=0  →  254,000 samples  (train)     → 500 samples × 508 basins
split=1  →   18,142 samples  (validation) → 11–40 samples × 508 basins
test.h5  →   27,983 samples  (held-out)   → no targets
```

### Our strategy: carve a dev set out of split=0
We keep `split=1` as our internal **dev set** (matches the competition eval distribution).
We further split `split=0` into **train** and **sanity-val** using a reproducible seed:

```
split=0 (254,000) ──► train   : ~88% per basin = ~440 samples/basin
                  └──► san_val : ~12% per basin = ~60 samples/basin  (for quick loss checks)

split=1 (18,142)  ──► dev (never touches training)

test.h5 (27,983)  ──► test (submit predictions, no targets)
```

Stratify by basin so every basin is represented in both train and san_val.

---

## 7. Optimizer: Adam vs AdamW

### What the paper uses
The paper uses plain **Adam** (lr=1e-4, no weight decay specified).

### Why AdamW is strictly better

Adam with L2 regularisation has a well-known bug: the weight decay term is
entangled with the adaptive learning rate, so heavier parameters get
disproportionately less regularisation than lighter ones. **AdamW** (Loshchilov
& Hutter, 2019) fixes this by decoupling weight decay from the gradient update:

```
Adam update:
    θ ← θ − lr · m̂/(√v̂ + ε)   − lr · λ · θ    ← weight decay INSIDE adaptive step
                                 ↑ scaled by 1/√v̂  (wrong: heavy params get less decay)

AdamW update:
    θ ← θ − lr · m̂/(√v̂ + ε)   − lr · λ · θ    ← weight decay OUTSIDE adaptive step
                                 ↑ constant factor  (correct: all params equally decayed)
```

In practice AdamW:
- Generalises better on held-out data
- Is the default optimizer for every modern Transformer (BERT, GPT, PatchTST)
- Requires no extra compute — same cost as Adam
- Adds one tunable hyperparameter: `weight_decay λ` (typical range 1e-4 to 1e-2)

**Decision**: Use **AdamW** as baseline. Compare against Adam only as an ablation.

### Weight decay candidates
```
λ = 0       → equivalent to plain Adam (no regularisation)
λ = 1e-5    → very light
λ = 1e-4    → standard default (PyTorch AdamW default)  ← our starting point
λ = 1e-3    → moderate (common for Transformer fine-tuning)
λ = 1e-2    → strong (good if model overfits badly)
```

| ID  | Decision         | Paper value | Our baseline | Candidates         | Priority |
|-----|------------------|-------------|--------------|--------------------|---------|
| T1a | Optimizer        | Adam        | **AdamW**    | Adam vs AdamW      | High    |
| T7  | Weight decay (λ) | not set     | **1e-4**     | {1e-5, 1e-4, 1e-3} | High    |

---

## 8. Learning Rate Scheduling: Cosine and Variants

### Why scheduling matters
A fixed learning rate is a blunt instrument:
- Too high early → unstable, oscillates around minima
- Too high late  → never converges to a sharp minimum
- Too low early  → wastes epochs on slow initial progress

A scheduler changes the learning rate automatically during training.

### The cosine schedule (our default)

CosineLR anneals the learning rate following a half-cosine curve from
`lr_max` down to `lr_min` over `T_max` steps:

```
                lr_max
  lr(t) = lr_min + ────── · (1 + cos(π · t / T_max))
                    2

  t     = current step (or epoch)
  T_max = total steps (or epochs)
```

Visualized:
```
  lr
  │
max│\                          warmup optional
   │ \                         ↓
   │  \                 ╭─────╮ \          ← cosine with warm restarts
   │   \               ╱       \  \          (CosineAnnealingWarmRestarts)
   │    ────────────╯           ╲──╲
min│                                 ───
   └───────────────────────────────────── epoch
      0    T/4   T/2   3T/4    T
```

### Variants we will test

**1. CosineAnnealingLR** (simple, no restarts)
```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
# lr decays from lr_max → eta_min over all epochs in one arc
```
- Simplest option, no extra hyperparameters
- Risk: lr is near zero in last 30% of training → very slow updates

**2. CosineAnnealingWarmRestarts / SGDR** (Loshchilov & Hutter 2017)
```python
scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)
# T_0=20:   first cycle = 20 epochs
# T_mult=2: each restart doubles the cycle length (20, 40, 80...)
```
- Restarts allow the model to escape shallow local minima
- Useful for our multi-component DMEL (Informer + LSTM trained in parallel)

**3. LinearWarmup + CosineDecay** (standard Transformer recipe)
```python
def lr_lambda(step):
    if step < N_warmup:
        return step / N_warmup              # linear ramp-up
    progress = (step - N_warmup) / (T_max - N_warmup)
    return 0.5 * (1 + cos(pi * progress))  # cosine decay
scheduler = LambdaLR(optimizer, lr_lambda)
```
- Warmup prevents large early gradients from corrupting attention heads
- Standard recipe for BERT, GPT, PatchTST
- Recommended when using batch=64+

**4. ReduceLROnPlateau** (reactive, data-driven)
```python
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)
# Halves lr whenever val_loss doesn't improve for 5 epochs
```
- No T_max to configure, reacts to actual loss dynamics
- Can be combined with early stopping naturally

### Recommended progression for our experiments
```
Phase 1 — Exploration (single seed):    LinearWarmup + CosineDecay
Phase 2 — Ablations  (3-seed runs):     CosineAnnealingWarmRestarts
Phase 3 — Final eval (5-seed runs):     best from Phase 1/2
```

| ID  | Decision         | Paper value | Our baseline            | Candidates                              | Priority |
|-----|------------------|-------------|-------------------------|-----------------------------------------|---------|
| T2  | LR scheduler     | none stated | **LinearWarmup+Cosine** | {none, CosineAnnealingLR, SGDR, OnPlateau} | High |
| T2a | Warmup steps     | —           | 5% of total steps       | {0%, 3%, 5%, 10%}                       | Medium  |
| T2b | Min lr (eta_min) | —           | 1e-6                    | {1e-6, 1e-7}                            | Low     |
| T2c | Restart T_0      | —           | 20 epochs               | {10, 20, 30}                            | Medium  |

---

## 9. Batch Size — Dependence on Input Size and MPS Performance

### Data memory per batch

```
  Tensor   Shape           Memory formula              B=32     B=64     B=128
  ───────  ──────────────  ──────────────────────────  ──────   ──────   ──────
  X        (B, 336, 12)    B×336×12×4 bytes             0.49MB   0.98MB   1.97MB
  y        (B, 48)         B×48×4 bytes                 6.0KB   12.0KB   24.0KB
  y_aux    (B, 48, 11)     B×48×11×4 bytes             64.0KB  129.0KB  258.0KB
  ───────  ──────────────  ──────────────────────────  ──────   ──────   ──────
  Total data per batch                                  0.56MB   1.12MB   2.25MB
```

Data tensors alone are small. The real memory cost is **model activations**.

### Why attention is the real memory bottleneck

A standard Transformer attention matrix is `(B, heads, seq_len, seq_len)`:
```
  B=64, heads=8, seq=336 (full attention):  64×8×336×336×4 bytes = 220 MB
  B=64, heads=8, seq=336 (ProbSparse):      64×8×336×log2(336)×4 ≈  37 MB
```

ProbSparse is not just a performance trick — it is what makes the Informer
feasible at seq_len=336 without running out of VRAM.

### Is batch size dependent on input size? Yes, two ways

**Way 1 — Memory:** Longer sequences → larger activations → fewer samples fit
per batch. Our seq_len=336 is 48× longer than the paper's 7 days. This is why
the paper uses batch=32 comfortably but we need to check at batch=64.

**Way 2 — Gradient quality:**
```
  Small batch → noisy gradient → acts as regularisation, better generalisation
  Large batch → smooth gradient → faster convergence but sharper minima risk
  Rule of thumb: scale lr ∝ √batch_size when changing batch size
```

### Why batch=64 is the MPS sweet spot on Apple M4

MPS (Metal Performance Shaders) is Apple's GPU compute framework for M-series
chips. Its hardware SIMD lanes are 32 elements wide internally. Matrix
multiplications are most efficient when the batch dimension is a multiple of 64:

```
  batch=32  → fills 1 SIMD group     → ~75-85% hardware utilisation
  batch=64  → fills 2 SIMD groups    → ~90-95% hardware utilisation  ← sweet spot
  batch=128 → bandwidth limited      → ~80-85% (memory wall)
  batch=256 → OOM risk with d_model=256 and seq=336
```

Batch=64 gives higher **samples/second** throughput on M4 vs batch=32, even
though both fit in unified memory. This was confirmed in a previous
implementation run on this machine.

### On the Linux/NVIDIA machine (8 GB VRAM, Ryzen 9)

CUDA is less sensitive to specific batch size alignment than MPS. Optimal
batch is determined by VRAM budget:
```
  Available VRAM: 8 GB
  Model params (Informer d=256): ~2.4M params ≈ 10 MB
  AdamW optimizer states:        ~20 MB (2 moment vectors)
  Remaining for activations:     ~7.97 GB
  Safe batch with ProbSparse:    128–256
```

For cross-machine reproducibility, canonical value is **batch=64**.
Scale up to 128 on the Linux machine only if training is slow.

| ID  | Decision      | Paper  | M4 Mac | Linux/NVIDIA | Candidates  | Priority |
|-----|---------------|--------|--------|--------------|-------------|----------|
| T4  | Batch size    | 32     | **64** | **64–128**   | {32,64,128} | Medium   |
| T4a | LR adjustment | —      | 1e-4   | scale with B | √B scaling  | Medium   |

---

## 10. Sobol Sensitivity Analysis

### What it is

Sobol answers: *Given output Y that depends on parameters (p1, p2, p3), how much
of the variance in Y is caused by each parameter?*

This is **global** sensitivity analysis — it samples the full hyperparameter
space rather than checking local gradients around one point.

### The core idea: variance decomposition

Any output Y = f(p1, p2, p3) can be decomposed exactly as:
```
Var(Y) = V₁ + V₂ + V₃ + V₁₂ + V₁₃ + V₂₃ + V₁₂₃

  V₁   = variance caused by p1 alone            (first-order effect)
  V₁₂  = extra variance from p1×p2 interaction  (second-order effect)
  ...etc
```

Sobol indices normalise by total variance:
```
  S1ᵢ = Vᵢ / Var(Y)      ← first-order index  (main effect of pᵢ alone)
  STᵢ = (Vᵢ + all interactions involving i) / Var(Y)   ← total-effect index
```

| Index value  | Meaning                               | Action              |
|--------------|---------------------------------------|---------------------|
| S1 ≈ 0       | Parameter has no effect on output     | Fix at any value    |
| S1 ≈ 0.1–0.3 | Secondary influence                   | Light tuning        |
| S1 ≈ 0.5–1.0 | Dominant driver of output             | Must tune carefully |
| ST >> S1     | Strong interactions with other params | Tune jointly        |

### Concrete example: DMEL Appendix A

The paper ran Sobol on three MRS parameters, using 32 Saltelli-sampled
combinations:

```
Parameters:  noise_std ∈ [0.1, 0.5]
             embed_dim ∈ [2, 5]       (m in sample entropy)
             tolerance ∈ [0.1, 0.3]   (r in sample entropy)

Output:      average sample entropy of IMF components after decomposition

Results:
  Station       S1(noise_std)  S1(embed_dim)  S1(tolerance)  dominant
  ─────────────────────────────────────────────────────────────────────
  Shuangpai     ≈ 0.03 (3%)    ≈ 0.85 (85%)   ≈ 0.12 (12%)  embed_dim
  Fenghuang     ≈ 0.04 (4%)    ≈ 0.35 (35%)   ≈ 0.61 (61%)  tolerance

Conclusion:
  noise_std → insensitive at both stations → fix at 0.2, no tuning needed
  embed_dim → dominant at Shuangpai        → must be tuned
  tolerance → dominant at Fenghuang        → must be tuned
  sensitivity is DATA-DRIVEN, not universal
```

### Why Saltelli sampling with only 32 points

Naive Monte Carlo Sobol requires N×(2k+2) model evaluations (k=parameters).
Saltelli's quasi-random scheme achieves accurate variance estimates with
only 32 structured samples by ensuring uniform coverage of the parameter
space — no random clumping.

### Implications for our implementation

1. **Do not tune noise_std** — insensitive at both stations, fix at 0.2
2. **Do tune embed_dim and tolerance** — they are dataset-dependent; our
   hourly 508-basin data may respond differently than the paper's stations
3. After implementing MRS, run a mini Sobol (32 combinations) on a small
   subset of our basins to identify which parameters matter most

---

## 11. Updated Training Config Summary

All decisions from sections 7–10 reflected in `src/config.py`:

```python
TRAIN_CFG = dict(
    seed              = 42,
    optimizer         = 'adamw',         # T1a: AdamW, not Adam
    weight_decay      = 1e-4,            # T7
    lr                = 1e-4,            # T1
    scheduler         = 'warmup_cosine', # T2: LinearWarmup + CosineDecay
    warmup_frac       = 0.05,            # T2a: 5% of total steps as warmup
    eta_min           = 1e-6,            # T2b
    batch_size        = 64,              # T4: MPS sweet spot, CUDA-safe
    epochs            = 100,             # T5
    clip_grad         = 1.0,             # T6
    loss              = 'mse',           # T3
    norm_strategy     = 'per_basin_zscore',  # T8
    early_stop_patience = 10,
    num_workers       = 4,
    pin_memory        = True,            # auto-disabled on MPS in make_loader()
)
```

### Seed strategy per experiment phase

| Phase            | Seeds   | Purpose                                  | Comparison valid? |
|------------------|---------|------------------------------------------|------------------|
| Exploration      | 1 (=42) | Fast iteration, rough architecture check | Directional only |
| Ablations        | 3       | Confirm differences are real vs noise    | Moderate         |
| Final evaluation | 5–10    | Report mean ± std, run Wilcoxon test     | Statistically valid |

A single seed is not statistically meaningful for claiming one model is better
than another. Use 1 seed during development, 5–10 for any result worth reporting.

---

## 12. Early Stopping

Early stopping halts training when the validation metric stops improving,
preventing overfitting even as training loss continues to decrease.

### Configuration

```
Monitored metric : median NSE on san_val   (mode = max, higher is better)
Patience         : 10 epochs
Min delta        : 1e-4  (improvement must exceed 0.0001 to reset counter)
Restore weights  : YES — checkpoint at best epoch is reloaded before evaluation
```

### What patience=10 means in practice

```
Steps per epoch  =  223,520 samples ÷ batch=64  =  3,492 steps
Patience window  =  10 epochs  =  34,920 gradient steps
```

Patience=10 (not 5) is deliberate: hydrological signals have strong seasonal
structure. The model may plateau temporarily during a training phase that
under-samples flood events. Patience=5 would terminate too early. Patience=10
also gives the cosine LR schedule time to anneal into a local minimum.

### Why NSE — not MSE — triggers early stopping

Training MSE is computed on **z-score normalised** values. It has no physical
meaning across basins: MSE=0.1 on a flashy basin and MSE=0.1 on a flat
baseflow basin are incomparable. NSE on `san_val` is computed on denormalised
discharge, per basin, then aggregated — it reflects actual hydrological skill
and is the correct quantity to guard against overfitting.

### Logic per epoch

```
IF  median_NSE(san_val) > best_NSE + 1e-4:
    best_NSE         ← current NSE
    best_epoch       ← current epoch
    patience_counter ← 0
    SAVE  experiments/exp_XXX/best_model.pt
ELSE:
    patience_counter += 1
    IF patience_counter >= 10:
        LOAD best_model.pt
        STOP training

Final evaluation always uses best_model.pt — never the last-epoch weights.
```

---

## 13. Evaluation Metrics

Four metrics are computed at every evaluation point. Each captures a
different failure mode.

### Primary: NSE (Nash-Sutcliffe Efficiency)

```
          Σ (y − ŷ)²
NSE = 1 − ─────────────      range: (−∞, 1]     perfect = 1
          Σ (y − ȳ)²
```

The denominator is the variance of observed discharge around its own mean.
NSE measures how much better the model is than predicting the basin mean.

| NSE value | Interpretation |
|-----------|----------------|
| < 0       | Worse than predicting the mean — model is useless |
| 0         | Same skill as predicting the mean — baseline |
| 0.5       | Acceptable for operational hydrology |
| 0.65      | Good |
| 0.75      | Very good |
| 0.85      | Excellent |
| 0.9+      | Outstanding (DMEL paper achieves this for 1-step daily) |

**Why NSE is primary**: scale-invariant (NSE=0.8 means the same for a wet
basin and a dry basin), has a clear interpretable baseline, standard in
hydrology since 1970, and directly comparable across all 508 basins.

**Known weakness**: dominated by flood peaks. A single discharge value 100×
the mean contributes 10,000× more to the sum than a dry-day sample. NSE
can be high even with poor low-flow performance. Acceptable for our
flood forecasting task.

### Secondary: KGE (Kling-Gupta Efficiency)

```
KGE = 1 − √( (r−1)² + (α−1)² + (β−1)² )

  r = Pearson correlation    — timing accuracy
  α = σ_pred / σ_obs         — variability ratio (does model reproduce spread?)
  β = μ_pred / μ_obs         — bias ratio (does mean level match?)
```

KGE decomposes error into three independent components. When two ablations
have similar NSE, KGE diagnoses **why** one is better: is it timing (r),
variability under/over-prediction (α), or systematic bias (β)?

### Supplementary: RMSE and MAE

- **RMSE** (units: mm/h) — absolute error, peak-sensitive. Not comparable
  across basins (scale-dependent). Use for detecting catastrophic failures.
- **MAE** (units: mm/h) — robust to outliers, good for checking systematic
  bias on normal days. Same scale-dependence caveat as RMSE.

```
  Metric   Units   Scale-invariant   Rank ablations?   Role
  ──────   ──────  ───────────────   ───────────────   ──────────────────
  NSE      none    YES               YES (primary)     Overall skill
  KGE      none    YES               YES (secondary)   Diagnose failure
  RMSE     mm/h    NO                NO                Absolute error
  MAE      mm/h    NO                NO                Bias on normal days
```

### Aggregation across 508 basins

All metrics are computed **per basin first**, then aggregated. NSE has
extreme outliers (a few basins with NSE < −1 due to low variance).

```
PRIMARY:    Median NSE   ← robust to outliers, reflects typical basin
SECONDARY:  Mean NSE     ← shows outlier sensitivity (mean << median = problem)
DIAGNOSTIC: % basins with NSE > 0.5   (acceptable skill threshold)
            % basins with NSE > 0.7   (good skill threshold)
```

### Baselines on dev split (empirical)

```
Predict basin mean for all 48h  → NSE = 0.000  (by definition)
Persistence (last known value)  → NSE median ≈ 0.53
Good model target               → NSE median > 0.70
Excellent model target          → NSE median > 0.85
```

---

## 14. Ablation Comparison Protocol

### Step 1 — Single-seed exploration (seed=42)

For every ablation variant, train once and record:
- Median NSE on dev, mean NSE on dev, % basins > 0.5, % basins > 0.7
- Best epoch (when did early stopping fire?)
- Training time (minutes)

**Threshold**: an ablation is *interesting* if it improves median NSE by
**> +0.01** over the baseline. Below that, the difference may be seed noise.

### Step 2 — Filter and sort

Rank all variants by median NSE descending. Keep top-5 for multi-seed.

### Step 3 — Multi-seed confirmation (seeds = 42, 123, 777)

For each top-5 candidate, run 3 seeds. Report mean ± std of median NSE.

```
Decision rule:
  A beats B  if  mean_NSE(A) > mean_NSE(B)  AND  intervals don't overlap

  A: NSE = 0.821 ± 0.008 )  intervals overlap → not conclusive at 3 seeds
  B: NSE = 0.810 ± 0.012 )  → escalate to 5 seeds

  A: NSE = 0.821 ± 0.004 )  clear gap, no overlap → A wins
  B: NSE = 0.793 ± 0.006 )
```

### Step 4 — Statistical significance (5+ seeds, Wilcoxon test)

For final paper-quality comparisons:
- Run 5 seeds per model
- Apply Wilcoxon signed-rank test (paired, non-parametric)
- H0: same median NSE per basin across models
- Significance level α = 0.05

Wilcoxon is used over t-test because NSE is not normally distributed across
basins, and we have paired samples (same 508 basins in both models).

### Step 5 — Compound ablation testing

Once individual winners are identified, test combinations:
```
Winner(T8) + Winner(D1) + Winner(T1a) → compound model
```
Verify the compound is better than its parts — sometimes two improvements
cancel (e.g. basin embedding may be redundant once per-basin norm is applied).

### Full decision flow per experiment

```
TRAIN (seed=42, max 100 epochs)
  ↓ every epoch
  evaluate san_val → median NSE → early stopping check
  ↓ (stopped or completed)
  load best_model.pt
  ↓
  evaluate dev split (split=1)
    → NSE per basin → median, mean, %>0.5, %>0.7
    → KGE, RMSE, MAE per basin (median aggregation)
    → save experiments/exp_XXX/metrics.json
  ↓
  compare to other ablations
    single-seed: direction check (threshold +0.01)
    3-seed: confidence interval check
    5-seed: Wilcoxon test (p < 0.05)
```

---

## 15. Implementation Files Plan

```
paper2/
├── data/                  ← train.h5, test.h5, metadata.json
├── src/
│   ├── config.py          ← hyperparameters, seeds, device detection
│   ├── dataset.py         ← HDF5 loading, BasinNormalizer, make_loader
│   ├── train.py           ← optimizer groups, schedulers, losses, epoch loop
│   ├── evaluate.py        ← NSE/KGE/RMSE/MAE, EvalResult, wilcoxon_nse
│   ├── early_stopping.py  ← EarlyStopping, CheckpointManager
│   ├── experiment.py      ← single-experiment orchestrator
│   └── models/            ← informer.py, lstm_branch.py, dmel.py
├── run_experiment.py      ← CLI, one experiment
├── run_ablation.py        ← CLI, ablation grid
├── compare_results.py     ← CLI, scoreboard
└── experiments/<name>/    ← per-run artefacts
```

---

## 16. Training Rigour Patterns (adopted from paper1)

Seven patterns were ported from the paper1 (HPSS/AnuraSet) implementation
after a side-by-side review of both training paths. Each closes a specific
hole where a decision was being made silently or unobservably.

### 16.1 Weight-decay parameter groups

**Was:** `weight_decay` applied uniformly to every parameter via
`model.parameters()`.

**Now:** `build_optimizer()` splits parameters into three groups.

```
decay      : multi-dimensional weight tensors (Conv1d, Linear, LSTM weight_*,
             Embedding)
no_decay   : LayerNorm gamma/beta · BatchNorm gamma/beta · all biases ·
             any 1-D combiner weight
combiner   : combiner params only, at lr x combiner_lr_mult (opt-in)
```

**Why it matters here specifically:** the Informer has `LayerNorm` in every
encoder and decoder block, plus `BatchNorm1d` inside every `DistilLayer`.
Decaying a normalisation layer's `gamma` toward zero directly fights the
layer's ability to rescale features — which is its entire function. Under
the old code every one of those parameters was being decayed.

`nn.LSTM` needs no special-casing: it exposes `weight_ih_l*`/`weight_hh_l*`
as 2-D (decayed) and `bias_ih_l*`/`bias_hh_l*` as 1-D (not decayed), so the
`ndim <= 1` rule handles it.

The optimiser logs the split at startup:
```
[optim] adamw  lr=0.0001
[optim] weight_decay=0.0001 on 47 tensors (2,381,824 params)
[optim] weight_decay=0.0 on 34 tensors (9,472 params)  <- LayerNorm/BatchNorm/bias
```

### 16.2 Gradient-norm logging

**Was:** `clip_grad_norm_` called, return value discarded. No visibility.

**Now:** the mean pre-clip gradient L2 norm is measured every epoch,
written to `history.csv`, and aggregated into `summary.json` as
`mean_grad_norm`.

- **Clipping ON**  → `clip_grad_norm_` already returns the pre-clip norm,
  so the measurement is free.
- **Clipping OFF** → the norm is computed on every 50th step
  (`GRAD_NORM_SAMPLE_EVERY`), costing <1% and still giving a usable
  per-epoch mean.

**Why:** `grad_norm` is the only signal that says whether `clip_grad` is
doing anything.

| Observation | Meaning | Action |
|---|---|---|
| `mean_grad_norm << clip_grad` | Clipper never fires | Set `clip_grad=0`, save the overhead |
| `mean_grad_norm → clip_grad` | Clipper fires every step | It is silently rescaling every update and changing the effective lr |
| `mean_grad_norm` climbing | Instability developing | Investigate before it diverges |

Turning clipping off must not make gradient behaviour unobservable — that
is why the sampling branch exists.

### 16.3 Patience relaxed for cosine schedules

**Was:** `early_stop_patience=10` applied regardless of schedule.

**Now:** `adjust_patience_for_scheduler()` raises patience to
`epochs // 2` whenever a batch-stepped cosine-family schedule is active.

**The conflict:** a cosine schedule anneals lr to its floor at exactly
`epochs`. The low-lr tail is where the model settles into a minimum. A
stopper with patience=10 fires during the mid-cosine plateau — discarding
precisely the phase the schedule was configured to reach. The two
components want opposite things:

```
scheduler: "let me run all `epochs` so lr reaches the floor"
stopper  : "stop as soon as the metric stalls"
```

Resolution: require `patience >= epochs // 2` for cosine / warmup_cosine /
sgdr. `plateau` and `none` are untouched — plateau reacts to the metric
itself so the two agree, and a flat lr has no anneal to protect.

The stopper is **not** removed; it remains a runaway guard. Both values are
recorded:
```json
{"patience_requested": 10, "patience_used": 50}
```

### 16.4 Independent RNG stream for the DataLoader

**Was:** shuffle order drawn from the global torch RNG.

**Now:** a dedicated `torch.Generator` seeded from the experiment seed is
passed to `make_loader()`, plus `worker_init_fn=seed_worker` re-seeds numpy
and stdlib `random` inside each worker.

**Why:** with a shared global RNG, batch order depends on how many draws
model init and any stochastic layer have already consumed. Changing an
unrelated hyperparameter that alters the draw count — adding a layer,
moving dropout — would silently reshuffle the data, making two runs
incomparable for a reason that has nothing to do with the thing being
ablated.

`seed_worker` is required because macOS **spawns** workers rather than
forking, so each worker would otherwise start with a fresh numpy RNG seeded
from OS entropy.

### 16.5 Measured justifications in config

Every non-obvious constant in `TRAIN_CFG` now carries the reasoning that
chose it, including the Adam-vs-AdamW update equations, why cosine steps
per batch rather than per epoch, and what to check before trusting
`clip_grad`. A reader should not have to reverse-engineer intent from a
bare number.

### 16.6 Normalizer persisted per run

**Was:** `BasinNormalizer` held in memory only; lost when the process exited.

**Now:** saved to `experiments/<name>/norm.npz` immediately after fitting.

**Why:** inference must apply the *identical* transform. Recomputing stats
on test data would be test-time leakage; recomputing from train would risk
silent drift if the split seed or `SAN_VAL_FRACTION` ever changes. The
statistics used at training time are now a durable artefact of the run.

### 16.7 `summary.json` — run audit trail

**Was:** no single file recorded what the run actually did.

**Now:** `summary.json` per experiment, with the split-discipline audit
trail as first-class fields:

```json
{
  "selection_split"   : "san_val",
  "report_split"      : "dev",
  "dev_median_nse"    : 0.8401,
  "sanval_median_nse" : 0.8455,
  "sanval_to_dev_gap" : -0.0054,
  "mean_grad_norm"    : 0.132,
  "clip_grad"         : 1.0,
  "patience_requested": 10,
  "patience_used"     : 50,
  "early_stopped"     : false,
  "best_epoch"        : 38,
  "epochs_run"        : 51
}
```

Three fields carry most of the diagnostic weight:

- **`selection_split` / `report_split`** must always differ. If they were
  ever the same, the reported metric would be optimistically biased by
  construction. Recording them makes that property auditable rather than
  assumed.
- **`sanval_to_dev_gap`** is the selection-overfitting detector. Strongly
  negative means the checkpoint was chosen on san_val noise and does not
  transfer.
- **`mean_grad_norm`** vs `clip_grad` settles whether clipping is active
  (see 16.2).

Surface across all runs with:
```bash
python compare_results.py --gap
```

### 16.8 Side effect: single-pass `quick_nse`

While wiring 16.2, `quick_nse()` was found to iterate the san_val loader
**twice** — once for `ss_res`, again for `ss_tot` after the per-basin mean
was known. It is now single-pass using the algebraic identity:

```
ss_tot = Σ(obs − mean)²  =  Σobs² − (Σobs)² / n
```

Verified exact to 0.00e+00 against the two-pass version. Since `quick_nse`
runs on ~30k samples after every epoch, this halves early-stopping
overhead for the entire ablation grid.

### 16.9 New ablation entries

| Experiment | Tests |
|---|---|
| `exp_011b__linear_combiner_lr100` | Is a learned combiner unhelpful, or merely under-trained? A handful of scalars competing against 2.4M parameters at one shared lr may never leave their initialisation |
| `exp_029b__no_clip` | Disable clipping. Read `mean_grad_norm` from the baseline's `summary.json` first — if it is far below 1.0 the clipper was never firing |

### 16.10 Not adopted, and why

| paper1 pattern | Reason |
|---|---|
| Per-class threshold tuning | Classification-only; DMEL is regression, there is no decision threshold |
| SpecAugment / MixUp | Domain-specific audio augmentation. Time-series augmentation for streamflow is a separate research question, not a port |
| `ZERO_POSITIVE` / `ULTRA_RARE` class exclusion | Already covered — `aggregate()` drops non-finite NSE basins, which is the regression analogue |
