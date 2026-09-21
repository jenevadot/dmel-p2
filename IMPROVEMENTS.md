# IMPROVEMENTS — implementation trail

> A running record of what was found, what was done about it, and the measured
> evidence for each decision. Written so the reasoning survives, including the
> attempts that failed.
>
> Format per item: **Started with** → **Found this problem** → **Moved to this**
> → **Evidence**.

---

## Tier 0 — Environment

### 0.1 Nothing had ever been run

**Started with** a code-complete repo: `src/` fully written, a 32-entry ablation
grid, a README describing five experiment phases.

**Found this problem**: `experiments/` was empty and `torch` was not installed
anywhere on the machine. Every number in the plan document was a projection, not
a measurement. This explains the bug density in Tier 1 — none of the code had
ever executed.

`pip` was additionally blocked by a security-sandbox proxy:

```
ProxyError: Tunnel connection failed: 403 Forbidden
🚫 Domain blocked by Apple Claude Code security sandbox
```

**Moved to** a `uv` virtual environment. The block was an allowlist policy, not a
broken network, so it was resolved by policy rather than worked around.

**Evidence**
```
.venv/bin/python -V   ->  Python 3.11.1
torch                 ->  2.14.0
torch.backends.mps    ->  True
numpy 2.4.6 · scipy 1.17.1 · h5py 3.16.0 · pandas · tqdm · matplotlib
```

A consequence worth recording: `PyEMD` / `EMD-signal` could not be installed. This
turned out not to matter — see §4.1, where CEEMDAN is vendored from a verified
reference already present in the repo.

---

## Tier 1 — The split (largest single correction)

This section is long because the first two answers were wrong, and the wrongness
is the instructive part.

### 1.1 Attempt one: aligned 24h chunks — a FALSE NEGATIVE

**Started with** `dataset.build_splits()`, which carves `san_val` out of `split=0`
by shuffling each basin's rows and taking 12%.

**Found this problem**: the windows are pre-cut 336h slices in shuffled row order
with no timestamps. If they overlap in time, a random carve puts near-duplicates
on both sides of the selection boundary.

I tested for overlap using aligned 24-hour chunks at stride 24, and built
`src/splits.py` around overlap-connected components. It reported:

```
random split leak rate : 68.98%
group  split leak rate :  0.00%
```

That looked like a clean fix. **It was not.** Aligned chunks only detect a pair of
windows when their time shift happens to be a multiple of 24 hours. Real shifts
are arbitrary — 1h, 4h, 17h.

**Evidence of the false negative** — re-testing the *same* "clean" split with
6-grams at stride 1:

```
basin   0: 55/60 san_val windows STILL overlap train
basin   7: 60/60
basin 200: 60/60
```

A split reported as 0.00% contaminated was in fact ~98% contaminated. The
detector, not the split, was the problem.

> Credit: `HANDOFF_REVIEW.md` independently flagged window overlap using a 6-gram
> index, which is what prompted the re-test.

### 1.2 Attempt two: connected components at stride 1 — correct but unusable

**Moved to** stride-1 fingerprinting, which finds overlap at any alignment.

**Found this problem**: leakage went to zero, but `san_val` ballooned to **50.5%**
against a 12% target — it would have thrown away 38% of the training data.

Diagnosis: at stride 1 the windows chain (w1 overlaps w2 overlaps w3...), so
transitively **each basin collapses to a single component**.

```
basin  0:  3 components, largest 495 of 500
basin  3:  1 component,  largest 500
basin  7:  1 component,  largest 500
basin  9:  1 component,  largest 500
basin 11:  1 component,  largest 500
mean components/basin: 3.4
```

There is no such thing as a *small* leak-free within-basin holdout in this
dataset. You take all 500 windows of a basin, or none.

### 1.3 Attempt three: held-out basins — correct

**Moved to** holding out whole **basins**. Different basins are different rivers
and share no temporal content, so this is leak-free by construction.

**Found one more problem**: the first cross-basin measurement reported 18.50%
leakage. Inspecting the actual matches showed every one was a **constant run**:

```
distinct matching gram values: 36 — all of the form
   [0.       0.       0.       0.       0.       0.      ]
   [0.00074679] * 6
   [0.00020367] * 6
```

Dry-spell plateaus coincide across unrelated basins. Filtering to grams with
>1 distinct value dropped it to 0.42%; the survivors were recession plateaus with
exactly 2 near-identical values. Final rule: a gram must have **≥3 distinct
values**, and a window needs **≥3 matching grams** (`MIN_MATCH_RUN`) before it
counts. Genuine overlap produces 18–331 matching grams; coincidence produces 1–2.

**Evidence** (`python src/splits.py`)
```
random within-basin : 97.00% of val windows leak
basin holdout       :  0.00% of val windows leak
train = 223,500 · san_val = 30,500 (12.0%)
```

### 1.4 `dev` is contaminated too — and we are NOT fixing it

**Found this problem**: the same strict test shows **78.8% of dev windows overlap
a training window**. The competition's `split=1` is randomly interleaved, not a
temporal tail — verified by inspecting split codes by row position:

```
basin   0: val positions span p0=1%  p50=42% p100=95%
basin   7:                    p0=0%  p50=44% p100=99%
basin 200:                    p0=9%  p50=69% p100=98%
```

**Moved to** measuring and documenting it rather than fixing it. `dev` is the
distribution we are scored on; forcing a chronological split would make our
selection signal a *different* distribution, which is a mismatch, not a fix.
`measure_dev_contamination()` records the number so the reported dev NSE is known
to be optimistic instead of silently trusted.

### 1.5 The tradeoff, stated plainly

`san_val` now measures **generalisation to unseen basins**. `dev` measures
**unseen time in known basins**. These are different questions. That mismatch is
the price of an uncontaminated selection signal, and it is recorded in the split
stats as `selection_measures` so it travels with the number.

---

## Tier 2 — CEEMDAN + MRS (`src/decompose.py`)

### 2.1 Vendored rather than written from scratch

**Started with** the intention to write CEEMDAN fresh, and a worry that it might
be computationally infeasible.

**Found this**: the repo already contains a CEEMDAN verified against Torres,
Colominas, Schlotthauer & Flandrin (ICASSP 2011) Eqs. 1–5 at
`runoff_emd_deep_dive.py:86-145` — including the subtle stage-≥2 rule that
perturbs with the *noise's own k-th EMD mode* rescaled to hold SNR constant,
rather than with fresh white noise. That rule is the single easiest thing to get
wrong, and it was already right.

**Moved to** porting it to numpy + `scipy.interpolate.CubicSpline`. This also
sidestepped the blocked `PyEMD` install entirely.

**Evidence** — completeness (Eq. 5) holds to machine precision:
```
EMD     : 3 IMFs, max|sum+res-x| = 1.78e-15
CEEMDAN : 6 IMFs, max|sum+res-x| = 1.78e-15
real window: n_imf=7, identity max err = 7.45e-09
```

### 2.2 Cost: my first estimate was wrong by 4×

**Started with** an estimate of ~2.8 h wall for the full precompute.

**Found this problem**: that came from a toy EMD that bailed after 1 IMF. With a
correct implementation:

```
n_trials= 20:  173 ms/window  ->  1.20 h wall @12 proc
n_trials= 50:  368 ms/window  ->  2.56 h
n_trials=100:  679 ms/window  ->  4.71 h    <- paper's setting
breakdown @100: ceemdan 594 ms (87%) | sampen 3.8 ms x ~8 calls
```

**Moved to** keeping the paper's `n_trials=100` and accepting 4.7 h. Reducing
trials was tested and rejected — it perturbs individual RIMF values materially:

```
n_trials=50 vs 100: mean rel.diff 0.005, MAX 0.255
n_trials=25 vs 100: mean rel.diff 0.008, MAX 0.427
```

(HF/LF *assignment* was stable at 40/40 windows, so the cheaper settings remain
viable as an ablation — but the default stays faithful to the paper.)

One optimisation makes the 4.7 h number hold: the noise modes `E_j(w_i)` depend
only on `(seed, n_trials, T)` and **not on the signal**, so the bank is built once
per worker and reused. Without it the per-window cost roughly doubles.

### 2.3 K=3, not the paper's 9 — forced by the data

**Started with** the paper's 9 IMFs → 6 RIMFs structure.

**Found this problem**: the paper decomposes ~1400-point *daily* series. We have
336-point *hourly* windows. Measured over 600 random windows:

```
0 IMFs:  1.0%      3 IMFs: 21.0%      6 IMFs:  4.5%
1 IMF :  9.8%      4 IMFs: 35.0%
2 IMFs:  3.7%      5 IMFs: 25.0%
```

`n_imf` varies in [0,6] and is never 9. Zero-padding a variable count into fixed
branches would make branch *i* mean a different frequency band per sample,
destroying the per-branch specialisation that justifies one-model-per-RIMF.

**Moved to** grouping into exactly **K=3** RIMFs (`n_high=2, n_low=1`) — a mild
specialisation of the paper's own operator, which already maps 9→6 (Shuangpai)
and 9→4 (Fenghuang).

```
K=2: 99.0% of windows have >= K components
K=3: 89.2%   <- chosen
K=4: 85.5%
```

Grouping is **contiguous** (exact O(n·K) DP over the SE vector), not k-means:
EMD orders components by decreasing frequency and SE follows, so k-means could
place IMF1 with IMF7 and destroy the band structure. The DP is also deterministic.

The 10.8% shortfall at K=3 is handled explicitly with degenerate codes, never
silently. The invariant `sum(RIMFs) == q` is asserted in **every** path including
all degenerate branches — if it breaks, the additive ensemble is no longer
reconstructing the target and every downstream number is meaningless.

### 2.4 Cache design

Disk was a non-issue once measured — K=3 float16 is ~605 MiB against 21 GB free.
Decomposition runs on the **raw** series so the cache stays independent of the
split seed; normalisation is applied at read time. `hf_rule`/`n_high`/`n_low` are
deliberately excluded from the cache key so the HF/LF-rule ablation re-derives
from cached SE values instead of triggering a 4.7 h rebuild.

---

## Corrections to my own earlier review

Recorded because the review is referenced elsewhere and two of its claims were
wrong.

| Claim I made | Status | Correction |
|---|---|---|
| "dev is essentially clean (0–11% overlap)" | **Wrong** | 78.8% of dev windows overlap train. My aligned-24h detector produced a false negative. |
| "group-holdout within basin is viable, 35–94 components/basin" | **Wrong** | That component count was an artefact of coarse chunking. At stride 1 it is ~3.4/basin and each basin is effectively one component. Basin-level holdout is the only workable answer. |
| "CEEMDAN may take weeks — likely prohibitive" | **Wrong** | 4.7 h wall at the paper's settings. My benchmark had used an EMD that bailed after one IMF. |
| "CEEMDAN ≈ 2.8 h" | **Understated** | 4.7 h at `n_trials=100`. |
| ProbSparse loses 93% of positions | **Confirmed** | 25 of 336 queries active; 311 inactive positions receive one byte-identical vector. |
| `NSELoss` denominator is batch-wide | **Confirmed and sharpened** | It is constant w.r.t. predictions, so the gradient is exactly MSE's up to a per-batch scalar. Verified: ratio min=max=1.007567. |

---

## Findings adopted from `HANDOFF_REVIEW.md`

Independently verified before adoption.

| Finding | Verification |
|---|---|
| ProbSparse mixed basic+advanced indexing allocates 262 GB at batch=64 | Reproduced in numpy: `K[:,:,idx.view(B,H,-1)]` yields `(B,H,B,H,N,d)`, a 6× blowup at toy scale → **281.9 GB** at production shapes. My review caught the *information* loss but missed the *allocation* bug. |
| `NSELoss` gradient ≡ MSE up to a scalar | Gradient ratio constant at 1.007567 across all elements. |
| `split=1` is randomly interleaved, not chronological | Val row positions span p0=0% → p100=99% per basin. |
| Persistence (median NSE 0.5251) is the real bar, not the mean predictor | Adopted as the baseline to beat. |
| Report NSE per lead-time bucket | Persistence scores 0.9985 at h+1 but 0.2228 at h+48; an aggregate median hides this. |

---

## Tier 3 — The code fixes

### 3.1 The 262 GB allocation bug

**Started with** my own review finding that ProbSparse loses information: at
L=336, only 25 of 336 queries are active and the other 311 receive one
byte-identical constant vector.

**Found this problem** (credit: `HANDOFF_REVIEW.md`) — there was a second, worse
bug on the same line. `K[:, :, idx.view(B, H, -1)]` is **mixed basic + advanced
indexing**: PyTorch keeps the leading sliced dims *and* inserts the
advanced-index shape. Verified in torch at B=8:

```
OLD (mixed indexing) shape: (8, 8, 8, 8, 8400, 32)  elems=1,101,004,800
NEW (gather)         shape: (8, 8, 336, 25, 32)     elems=   17,203,200
needed                                                        17,203,200
blowup factor at B=8: 64x   ->  281.9 GB at batch=64
```

The self-test passed because it ran at **batch=4** (1.1 GB) and asserted only
the output *shape* — and `.view()` reshapes the wrong tensor back, so shape
checks succeed on garbage.

**Moved to** three changes:
- `attention: "full"` is now the default, via
  `F.scaled_dot_product_attention` (fused kernels, matrix never materialised).
  At L=336 the ProbSparse sampling intermediate (525 MB) is **2.4× larger**
  than the full matrix it replaces (220 MB); it pays off around L≥5000.
- ProbSparse repaired (`torch.gather`, dead `expand` removed) and kept as
  ablation I7, so the comparison is between two *working* backends.
- **The self-test now runs at batch=64** and asserts finiteness. That is the
  regression test.

**Evidence**
```
attention=full  batch=64 -> (64, 48) finite ✓
attention=prob  batch=64 -> (64, 48) finite ✓
54/54 parameter tensors received gradient ✓
```

### 3.2 `NSELoss` was MSE in disguise

**Found this problem** (credit: `HANDOFF_REVIEW.md`): `ss_tot` depends only on
`y_true`, so it is a **constant** w.r.t. predictions. Verified:

```
gradient ratio g_nse/g_mse: min=1.007567 max=1.007567 constant=True
```

It was MSE rescaled by a random per-batch scalar — effective-learning-rate
jitter, not a distinct objective. `exp_012__nse_loss` would have measured noise.

A second realisation matters more: **plain MSE on per-basin z-scored targets
already IS Kratzert's NSE\* loss at ε=0**, because dividing by σ_b during
normalisation is exactly the basin-variance weighting. The train/eval alignment
the ablation was reaching for already existed — in `BasinNormalizer`, not in
`build_loss`.

**Moved to** the real remaining gap, the **ε**. Per-basin σ spans a 521× range
and 14 basins (2.8%) have σ < 0.01 — essentially flat lines whose pure noise
gets amplified to the same gradient magnitude as a genuinely dynamic basin.
`NSELoss` now implements `loss = mean[(y-ŷ)² · σ_b²/(σ_b+ε)²]` with ε=0.1, and
`exp_040__nse_eps0` (ε=0) is the control that recovers plain MSE.

### 3.3 Three forward paths that disagreed

**Found this problem**: `evaluate.py` called `model(x, y_aux=y_aux)` — a kwarg
`DMEL.forward` does not accept (instant `TypeError`), never passed `basin_ids`
(so `use_basin_emb` crashed), and never built `x_dec` at all. So evaluation
crashed on two flags and, once naively fixed, would still have fed the model
**different inputs than training did**.

**Moved to** a single `train.forward_batch(model, batch, cfg, device)` used by
`train_one_epoch`, `quick_nse`, and `evaluate_model`. Also removed the bare
`except Exception` in `run_ablation.py` that printed only `str(exc)` — the
mechanism by which this class of bug could burn a full training run and leave
one unhelpful line in a scoreboard.

### 3.4 `y_aux` → auxiliary target

`metadata.json` says `"y_aux_role": "future_supervision_only_not_inference_inputs"`
and `y_aux` is absent from `test.h5`. Feeding it to the decoder would inflate
dev NSE and collapse at submission, and `sanval_to_dev_gap` could not detect it
because *both* splits have `y_aux`.

**Moved to** an optional `aux_head` predicting future forcing from the encoder
representation, with `loss = mse(q̂,q) + 0.3·mse(â,a)`. The head is discarded at
inference. Multi-task regularisation, not conditioning. The dataset only loads
`y_aux` when `want_aux=True`, which also removed a per-sample 11-channel
normalisation loop from every default run.

### 3.5 Data pipeline: 4.1 GB and 153×

Two measured wins:

| | Before | After |
|---|---|---|
| Normalizer fit peak RSS | **4.1 GB** materialised then discarded | **627 MB** (streaming Σx, Σx², n) |
| `transform` per sample | 0.612 ms (loop over all 508 basins) | 0.004 ms (`transform_one`) — **153×** |

At 223k samples/epoch the old transform burned ~137 s/epoch of pure overhead,
~3.8 h per 100-epoch run. The streaming fit reuses the `var = E[x²] − E[x]²`
identity already present in `quick_nse`.

Verified bit-identical to the old batched path before replacing it.

### 3.6 Four ablations that could not measure anything

- `global_zscore` / `none` were **accepted by the CLI but never implemented** —
  `build_splits` only matched `"per_basin_zscore"` and left `normalizer=None`
  otherwise, feeding the model raw data (pressure ~94,000 beside humidity
  ~0.007). Now implemented; unknown strategies raise.
- **Channel subsetting did not exist**, so `exp_003__all_channels` was
  byte-identical to the baseline. Now `input_channels=[...]`.
- **`seq_len` did not truncate** — the dataset returned all 336 steps and the
  conv/attention stack is length-agnostic, so `exp_009__seq168` compared 336
  against 336.
- The **combiner was unreachable** (`dmel.py` returned before building it when
  `use_ceemdan=False`), making three combiner ablations no-ops. It now raises
  if you request a non-`sum` combiner with a single branch, instead of silently
  ignoring it.

**Moved to** a rebuilt grid plus `validate_grid()`, which fails if any entry's
overrides are unknown keys or all equal their defaults. That is the regression
test for this whole bug class:

```
grid validation: 26 experiments, all differ from baseline ✓
```

### 3.7 Per-horizon metrics and the real baseline

**Found this problem**: the pooled median NSE is dominated by the trivially easy
early hours. Persistence alone scores 0.9382 at h+1–12 and 0.2495 at h+37–48, so
a model that learned nothing but persistence still posts a respectable
aggregate.

**Moved to** reporting NSE per lead-time bucket everywhere, and writing all four
baselines as first-class `EvalResult` artefacts so the scoreboard ranks against
**persistence, not zero**.

Also fixed `pct_nse_05`/`pct_nse_07`, which divided by the *valid* basin count —
a run producing NaN on 200 basins reported percentages over the surviving 308,
i.e. looked better for having failed.

**Evidence** — baselines reproduce the handoff's independent numbers exactly:

```
persistence  median NSE +0.5251   h1_12 +0.9382  h37_48 +0.2495
decay                   +0.4889         +0.9136         +0.1940
climatology             +0.1546         +0.2678         +0.1290
mean                    -0.0071         -0.0097         -0.0101
```

`persistence = 0.5251` matching to four decimals is strong cross-validation of
the whole metric path.

---

## Tier 3b — Problems the first smoke run exposed

Running end-to-end is what surfaced these; none were visible from reading code.

### 3b.1 Sandbox blocks DataLoader workers

`num_workers=4` dies at the first collate:
```
RuntimeError: torch_shm_manager ... Operation not permitted
```
Set `num_workers=0`. Side benefit: removes the 16 concurrent HDF5 handles that
4 loaders × 4 workers would have opened.

### 3b.2 `json.dump` failed AFTER a completed run

`TypeError: Object of type int64 is not JSON serializable` — `basin_id` came
from a numpy array, and `asdict` preserved `np.int64`. It surfaced only after
training *and* both evaluations had finished: the most expensive possible place
to find a type error. Fixed at the boundary (`int(basin_id)`) plus a
`_json_safe` fallback encoder on every writer.

### 3b.3 The basin holdout broke the normalizer — a flaw in my own design

**This is the most important finding of the smoke run.** First real run reported:

```
san_val med NSE : -53.1510
dev     med NSE :   0.5766
```

Diagnosis: with a **basin** holdout, the 61 san_val basins are unseen by
definition — so a train-only normalizer fit left every one of them at the
unfitted default `mean=0, std=1`:

```
UNFITTED held-out basins: 61/61
p0 = -35,623   p50 = -53.15   p100 = -4.15
basins with NSE > 0: 0/61
```

Their raw inputs (pressure ~94,000) went straight into a model expecting
z-scores. Early stopping was selecting checkpoints on pure noise.

**Moved to** fitting the normalizer over **all** split=0 basins (train +
san_val). This leaks no targets: the normalizer holds per-basin *input* scaling,
and `test.h5` forces the identical operation anyway — we must produce statistics
for its basins without ever seeing their targets. What would be leakage, and is
not done: fitting on dev/test windows, or using selection-split targets to pick
a checkpoint.

**Evidence**: `UNFITTED held-out basins now: 0/61`.

### 3b.4 Epoch budget: 100 → 12 → 40 (the 12 was wrong)

Measured **~11 min/epoch** (223,500 windows, batch 64, `num_workers=0`). The
paper's 100 epochs is ~18 h *per experiment*, ~400 h for a 31-experiment grid —
a budget that does not exist here.

**First answer: 12 epochs.** Justified on the grounds that the smoke run beat
persistence at every lead-time bucket after one epoch, so "the ranking signal
arrives early."

**That was wrong, and the first real run showed it:**

```
  ep   train_loss   sanval_NSE   delta      lr
   8     0.4317       0.5598              1.70e-05
   9     0.4192       0.5799    +0.0201   8.33e-06
  10     0.4114       0.5941    +0.0142   2.87e-06   <- best epoch is the LAST
```

Three diagnostics, all pointing the same way:

1. **Val NSE still rising** at the final epoch (+0.0142).
2. **Train loss still falling** monotonically — so this is not overfitting, it
   is truncation.
3. **The lr had been annealed to 2.87e-06**, ~35× below its start. Cosine
   anneals to its floor at exactly `epochs`, so telling the scheduler the run
   was 12 epochs long *starved* the model rather than converging it. The budget
   manufactured the appearance of convergence.

The reasoning behind the original choice was also weaker than I presented it:
"ablation rankings stabilise early" was an assumption, not a measurement. A
config that converges *slowly* looks worse at 12 epochs purely because it was
cut off sooner — so a short grid can invert the very ranking it exists to
establish.

**Moved to 40 epochs on the ~8 highest-value experiments** (~7.3 h each, ~59 h
total) instead of 12 epochs across all 31. Fewer converged experiments beat more
starved ones. `patience` raised to 8 (auto-relaxed to 20 by the cosine rule).

The 12-epoch run is kept as `experiments/_ep12__exp_001__baseline` so the
budget comparison itself is reproducible.

### 3b.5 Gradient clipping is active, not a no-op

Smoke run reported `mean_grad_norm = 2.4474` against `clip_grad = 1.0`. The
clipper fires on **every step**, so the effective learning rate is lower than
configured. This is exactly what §16.2's logging was built to reveal.
`exp_038__no_clip` is the controlled test; `results.py --gap` flags the
condition automatically.

---

## Tier 4 — CEEMDAN at scale: two more of my own errors

### 4.1 "The sandbox blocks multiprocessing" — WRONG

**Started with** a background CEEMDAN run at 6 procs alongside training. After
9 minutes it had printed no progress and `pgrep` showed only the parent, no
workers. I concluded the sandbox blocked `multiprocessing` the same way it had
blocked the DataLoader's shared-memory manager, and told the user serial
execution (~57 h) was the only option.

**That was wrong, and I tested it properly instead of assuming:**

| Test | Result |
|---|---|
| Plain `spawn` Pool, `map(f,[1,2,3])` | works, 0.1 s |
| Pool + `make_noise_bank` initializer | works, 1.7 s/worker |
| Pool + HDF5 read inside worker | works, 0.0 s/block |
| Real `build_cache`, 4 procs, 256 rows | **works, 2.8 min** |

Two real causes, neither of them the sandbox:

1. **CPU contention.** 6 CEEMDAN workers plus a training run on 14 cores. Each
   worker also pays a 1.7 s noise-bank init.
2. **No output until a block completes.** With `--block 256` at ~0.7 s/window,
   the first progress line is ~3 minutes away per worker. It looked hung
   because it printed nothing, not because it was hung.

**Moved to** measuring the actual rates and handing the job over:

```
 4 procs, training running :  1.5 win/s  -> ~56 h
12 procs, machine idle     : ~5.0 win/s  -> ~17 h
```

The user runs `./run_ceemdan.sh` on an idle machine. The cache is resumable —
each block sets its `done` flag last and flushes — so stopping and restarting
is safe.

Threads were also tested and rejected: 0.7–0.8× *slower* than serial, because
the sifting loop is Python-level and holds the GIL.

### 4.2 The IMF-count measurement used the wrong algorithm

**Started with** a K=3 decision justified by "CEEMDAN yields 1–6 IMFs, 9.8% of
windows yield only 1".

**Found this problem**: that measurement ran **plain EMD**, not CEEMDAN. The
ensemble noise creates extrema, so more modes are extractable. Real CEEMDAN at
`n_trials=100`, measured on 256 windows:

| IMFs | plain EMD (wrong) | CEEMDAN (actual) |
|---|---|---|
| 0 (constant) | 1.0% | 2.0% |
| 1–5 | 69.5% | **0%** |
| 6 | 4.5% | 7.0% |
| 7 | — | 39.8% |
| 8 | — | **45.7%** |
| 9 | — | 5.5% |

CEEMDAN yields **6–9 IMFs, mode 8** — much closer to the paper's 9 than I had
claimed, and the "variable IMF count is the central design problem" framing was
overstated.

**K=3 survives**, but for a cleaner reason than the one I gave: every
non-constant window has ≥6 components, so K=3 is *always* satisfiable rather
than "89.2% satisfiable with a 10.8% degenerate remainder". `DEG_FEW_COMPONENTS`
is now effectively dead code kept as a guard. Documentation in `config.py` and
`decompose.py` corrected in place.

### 4.3 Cache verified correct

```
verify: 256 rows, max rel recon err = 5.56e-03   (float16 storage limit)
degenerate codes: {normal: 251, constant: 5}
```

The Eq. 14 identity `sum(RIMFs) == q` survives the float16 round-trip at the
expected precision.

---

## Corrections to my own work — running tally

Kept because the review and plan are referenced elsewhere.

| Claim | Status | Correction |
|---|---|---|
| "dev is essentially clean (0–11% overlap)" | **Wrong** | 78.8% of dev windows overlap train. Aligned-24h chunking gave a false negative. |
| "group-holdout within basin viable, 35–94 components/basin" | **Wrong** | Artefact of coarse chunking. At stride 1 it is ~3.4/basin and each basin is one component; basin-level holdout is the only workable answer. |
| "CEEMDAN may take weeks" | **Wrong** | ~17 h at 12 procs. |
| "CEEMDAN ≈ 2.8 h at 12 procs" | **Wrong** | ~17 h. The 2.8 h figure came from a broken EMD that bailed after one IMF. |
| "The sandbox blocks multiprocessing" | **Wrong** | It does not. CPU contention plus block-buffered output. |
| "CEEMDAN yields 1–6 IMFs" | **Wrong** | 6–9, mode 8. That measurement used plain EMD. |
| "Train-only normalizer fit is correct" | **Wrong** | With a basin holdout it leaves all 61 san_val basins unfitted (median NSE −53). Must fit on all split=0 basins. |
| ProbSparse loses 93% of positions | Confirmed | 25/336 queries active. |
| `NSELoss` denominator is batch-wide | Confirmed & sharpened | Gradient is exactly MSE's × a per-batch scalar (ratio 1.007567, constant). |

---

## Status

| Tier | Item | State |
|---|---|---|
| 0 | venv + torch 2.14.0 + MPS | ✅ |
| 1 | Leak-free split (`src/splits.py`) | ✅ 97% → 0% |
| 1 | dev contamination measured (78.8%, unfixable) | ✅ |
| 2 | CEEMDAN + SE + MRS (`src/decompose.py`) | ✅ written, self-tested, cache verified |
| 3 | Attention 262 GB bug + batch=64 regression test | ✅ |
| 3 | `NSELoss` → basin-averaged with ε | ✅ |
| 3 | `forward_batch` unification | ✅ |
| 3 | `y_aux` → auxiliary target | ✅ |
| 3 | Streaming fit (4.1 GB → 627 MB), `transform_one` (153×) | ✅ |
| 3 | Grid rebuilt + `validate_grid()` regression test | ✅ |
| 3 | Per-horizon metrics + 4 baselines | ✅ |
| 3b | Normalizer fitted on all basins | ✅ |
| 4 | `results.py` inspection shell | ✅ |
| 5 | Informer grid running (8 experiments, caffeinate) | 🔄 in progress |
| 5 | CEEMDAN precompute (~17 h, 12 procs) | ⏳ **user runs `./run_ceemdan.sh`** |
| 5 | DMEL ablations (need the cache first) | ⏳ blocked on above |


