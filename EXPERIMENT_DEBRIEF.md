# Experiment Debrief: DMEL replication on the 508-basin hourly dataset

Date: 2026-09-26
Sources used: only `experiments/*` (summary.json, history.csv, metrics_dev/sanval/test.json, log.txt,
queue logs), `src/*`, the run scripts, and read-only analyses on the saved checkpoints, caches, and h5
files. No existing review document was used (HANDOFF_REVIEW.md, IMPROVEMENTS.md, README.md, TASKS.md,
WORKFLOW.md, and the other notes). All numbers below were recomputed from raw artefacts. Section 10 lists
the method behind each ad-hoc analysis.

Terminology
- dev: the competition validation split (`split=1`). It has 18,142 windows across all 508 basins, about
  38 windows per basin. It is the headline metric.
- san_val: 61 held-out basins with their `split=0` windows (30,500 windows). It is the early-stopping
  signal.
- NSE: the per-basin Nash-Sutcliffe efficiency, aggregated as the median across basins unless stated
  otherwise.
- sigma: the standard deviation across the three baseline seeds.
- Buckets: forecast lead-time buckets h1-12, h13-24, h25-36, h37-48.

---

## 0. Decision summary

1. Most measured differences are noise. The three baseline seeds span dev median NSE from 0.621 to 0.657
   (sigma = 0.018). For the h37-48 bucket they span 0.395 to 0.468 (sigma = 0.041). With one treatment
   run against three baseline seeds, an effect needs to be at least about 0.04 before it can be trusted,
   and about 0.09 under a strict t-test. Across 19 configs on the same partition, the dev ranking and the
   san_val ranking are uncorrelated (Spearman rho = 0.00). No tuning experiment clearly beats the
   baseline. Several experiments are clearly worse.
2. CEEMDAN/DMEL is not validated. Both DMEL runs are worse than the plain Informer on san_val (0.587 and
   0.582 vs 0.619 on the same basins) and collapse on dev (0.397 and 0.504, which is below persistence at
   0.525). The checkpoints show why: the two high-frequency Informer branches output near-constants
   (their output varies by only 0.01-0.07 z across windows), the LSTM does all the work, and the sum
   over-predicts by about +0.08 to +0.14 z. That bias is what destroys dev NSE. Each run costs 3.5-4x
   the baseline. Do not run more DMEL variants as they are built now (Section 5.1).
3. The best signal so far is the robust losses. MAE (dev 0.683, +2.4 sigma), Huber delta=0.5 (0.666,
   +1.4 sigma), and Huber delta=1 (0.658, +1.0 sigma) improve in the same direction, and more strongly
   as the loss gets more robust. All four buckets are positive for MAE and Huber 0.5. On the same
   san_val basins, Huber 0.5 has the best smoothed curve of any run (+0.068 over the baseline). This
   needs 2 more seeds before you adopt it.
4. The largest structural weakness is short lead times and steady flow. All four checkpoints re-run
   here (MSE, Huber 1, Huber 0.5, MAE) are worse than persistence at h+1 to h+3: at h+1 the MSE model's
   error is 3.8x persistence's, and its RMSE is 0.30 z vs 0.155 z. On the 87% of windows with steady
   flow, model MSE is 2.2x persistence's. In 25-44% of basins, the model is worse than persistence
   overall. The model earns its skill on recessions (skill +0.79) and at long lead times (+0.33 at
   h24-h48). It cannot anticipate rises: it captures a median of only 4-11% of the observed rise. An
   output that predicts the change from the last observed value (a "residual to persistence" output)
   targets exactly this gap, and it is the best single architectural experiment to run next.
5. Ensembling existing checkpoints costs no training and helps. Averaging the 4 local checkpoints (MSE,
   Huber 1, Huber 0.5, MAE) gives dev median NSE 0.708, vs 0.642 for the baseline and 0.683 for the best
   single model. Every bucket improves (0.949 / 0.802 / 0.649 / 0.526). This beats every
   hyperparameter change you have tested.
6. More epochs is not the binding constraint. MSE runs peak between epochs 14 and 31 and then plateau or
   decline. Only Huber 0.5 and MAE are still rising at epoch 37-40, and only slowly: about +0.01 per 10
   epochs, which is below seed noise. Spend compute on seeds before spending it on epochs.
7. Protocol problems limit what the current runs can tell you:
   - Changing the seed also changes the san_val partition. The seed-123 and seed-177 partitions share
     only 9 and 7 of their 61 basins with seed 42.
   - The per-basin Wilcoxon test reports p = 1e-8 between two seeds of the same configuration, so it is
     not a valid significance test here.
   - The persistence reference in the logs is wrong.
   - 61 test basins were never seen in training by any model.
   - ProbSparse evaluation is non-deterministic.
   - `exp_034` is incomplete.

---

## 1. What was actually run (from the code)

| Item | Implementation |
|---|---|
| Data | 336 h input window, 48 h target, 12 channels (ch 11 = discharge), 508 basins |
| Train | 223,500 windows from 447 basins (`split=0`, san_val basins removed) |
| san_val (selection) | 30,500 windows from 61 held-out basins. Used for early stopping only. |
| dev (headline) | 18,142 windows (`split=1`), all 508 basins, about 38 windows per basin (range 11-40) |
| test | 27,983 windows, targets in CSV. Scored once, for `exp_001__baseline` only. |
| Normalization | Per-basin, per-channel z-score, fitted on all `split=0` windows of all 508 basins |
| Baseline model | Informer: d_model 256, 8 heads, 2 encoder layers with distillation (336 to 168), 1 decoder layer, full attention, 2.85M params |
| Decoder input | Last 24 h of the encoder input followed by 48 zeros. No future forcings, no leakage. |
| Optimization | AdamW lr 1e-4, weight decay 1e-4 on matrices only, 5% warmup then cosine decay over 40 epochs, gradient clipping at 1.0, batch 64 |
| Selection | Maximum san_val median NSE (patience relaxed to 20), then the best checkpoint is restored |
| DMEL | CEEMDAN per 336 h window (100 trials, eps 0.2) and MRS grouping into K=3 RIMFs. The 2 highest-SE RIMFs go to 2 Informers and the lowest to an LSTM. In `replace` mode each branch gets the 11 meteo channels plus its RIMF in place of discharge. Branch outputs are summed and trained end-to-end on the total target y. |

Cost per run: about 6.4 h for the baseline, on both MPS (M4, about 11 min/epoch) and CUDA (about
9.7 min/epoch). DMEL took about 25 h per run as measured: 3.5x the baseline per epoch, while sharing
the GPU with another run. The MPS wall-clock times for `exp_002`, `exp_009`,
`exp_010`, `exp_011`, and `exp_012` are inflated, because they ran at the same time as `exp_020` and
`exp_023` on the same GPU (confirmed from the timestamps in `queue.log`).

---

## 2. The noise floor

### 2.1 Baseline seeds

| seed | device | san_val partition | best epoch | san_val best | dev median | h1-12 | h13-24 | h25-36 | h37-48 |
|---|---|---|---|---|---|---|---|---|---|
| 42 | MPS | P42 | 14 | 0.619 | 0.642 | 0.927 | 0.757 | 0.576 | 0.468 |
| 123 | CUDA | P123 (9 of 61 basins shared with P42) | 31 | 0.638 | 0.621 | 0.920 | 0.743 | 0.549 | 0.395 |
| 177 | CUDA | P177 (7 of 61 basins shared with P42) | 29 | 0.660 | 0.657 | 0.935 | 0.774 | 0.592 | 0.399 |
| **mean / sigma** | | | | | **0.640 / 0.018** | 0.927 / 0.008 | 0.758 / 0.016 | 0.572 / 0.021 | **0.421 / 0.041** |

- The seed is passed to `build_splits(seed=seed)` (`src/experiment.py:131-141`). That drives
  `build_group_split`'s `default_rng(seed)` basin permutation (`src/splits.py:367-379`). So the three
  seeds mix initialization, data order, the training-basin set, the san_val set, and the device. This is
  a fair estimate of total variance, but it is not a pure seed estimate.
- The noise is largest at h37-48, the bucket where the model adds the most over persistence.
- The shape of the training curve depends on the partition. On P42, san_val peaks at epoch 14 and then
  falls to about 0.55 while training loss falls 47%. On P123 and P177, the same configuration keeps
  improving until epochs 29-31 and ends on a plateau (last-5 average 0.628 and 0.651). Whether a run
  looks like it overfits depends on which 61 basins are held out.

### 2.2 Why the per-basin paired tests are misleading

Here is the per-basin comparison between two seeds of the same config, restricted to the basins that
both runs trained on:

| pair | median per-basin delta NSE | fraction of basins better | Wilcoxon p |
|---|---|---|---|
| s123 vs s42 | -0.021 | 0.38 | **1.3e-8** |
| s177 vs s42 | +0.001 | 0.51 | 0.97 |
| s177 vs s123 | +0.018 | 0.60 | **8.8e-6** |

The per-basin Wilcoxon (`evaluate.wilcoxon_nse`) treats basins as independent replicates and ignores
training randomness. It will call two seeds of the same model "significantly different". Only
seed-level replication gives valid inference.

### 2.3 Selection noise and the winner's curse

- San_val NSE jumps from one epoch to the next with a standard deviation of 0.015-0.037 in the second
  half of training. These are real changes in the model at high learning rate, not measurement noise.
- The selected "best epoch" is often a single spike. For `exp_001` s42, the best is 0.619 while the
  3-epoch smoothed maximum is 0.600. For `exp_005`, the best is 0.649 at epoch 24, but its neighbours
  are 0.590 and 0.575. `exp_036` and `exp_009` behave the same way.
- Across the 19 configs that share partition P42 (excluding DMEL), the Spearman correlation between dev
  median and san_val is -0.01 for the best value and 0.00 for the smoothed value. The two validation
  signals give unrelated rankings.

### 2.4 Detectable effect size

| Design | Minimum dev median delta you can trust (95%) |
|---|---|
| 1 treatment run vs 3 baseline seeds, sigma treated as known | about 0.041 |
| same design, t-test with 2 degrees of freedom (sigma estimated from 3 runs) | about 0.089 |
| 3 vs 3 seeds, t-test with 4 degrees of freedom | about 0.041 |
| 5 vs 5 seeds | about 0.026 |

None of the tuning experiments clears 0.089. Only MAE (+0.043) passes the lenient 0.041 threshold.

---

## 3. Scoreboard

The dev z-score is the difference from the 3-seed baseline mean divided by the seed sigma (per bucket
for the bucket columns). The last san_val column (sm3 delta) compares each run's 3-epoch smoothed maximum
with the same quantity for baseline s42, which uses the same partition P42. The s123 and s177 rows use
other partitions.

| Experiment | dev median (z) | h1-12 | h13-24 | h25-36 | h37-48 | san_val best / sm3 | sm3 delta vs s42 | Verdict |
|---|---|---|---|---|---|---|---|---|
| persistence | 0.525 | 0.938 | 0.686 | 0.416 | 0.249 | — | — | reference |
| ridge (lambda tuned on dev) | 0.417 | 0.666 | 0.547 | 0.313 | 0.233 | — | — | weak reference (Section 6) |
| exp_001 baseline s42 | 0.642 (+0.1) | 0.927 | 0.757 | 0.576 | 0.468 | 0.619 / 0.600 | 0 | reference |
| exp_001 s123 | 0.621 (-1.1) | 0.920 | 0.743 | 0.549 | 0.395 | 0.638 / 0.630 | (P123) | seed |
| exp_001 s177 | 0.657 (+0.9) | 0.935 | 0.774 | 0.592 | 0.399 | 0.660 / 0.654 | (P177) | seed |
| **exp_012 MAE** | **0.683 (+2.4)** | 0.946 (+2.3) | 0.786 (+1.8) | 0.625 (+2.5) | 0.500 (+2.0) | 0.639 / 0.634 | +0.034 | **PROMISING: replicate** |
| **exp_010 Huber delta=0.5** | **0.666 (+1.4)** | 0.945 (+2.2) | 0.779 (+1.3) | 0.596 (+1.1) | 0.474 (+1.3) | **0.669 / 0.668** | **+0.068** | **PROMISING: replicate** |
| exp_009 Huber delta=1 | 0.658 (+1.0) | 0.938 (+1.3) | 0.756 (-0.1) | 0.602 (+1.4) | 0.477 (+1.4) | 0.633 / 0.603 | +0.003 | supports the robust-loss trend |
| exp_005 aux task | 0.660 (+1.1) | 0.935 (+1.0) | 0.784 (+1.6) | 0.606 (+1.6) | 0.466 (+1.1) | 0.649 (spike) / 0.607 | +0.007 | MIXED, +15% time per epoch |
| exp_004 precip+discharge | 0.672 (+1.8) | 0.948 (+2.7) | 0.794 (+2.3) | 0.610 (+1.8) | 0.477 (+1.4) | 0.500 / 0.498 | **-0.102** | MIXED: dev up, unseen basins down |
| exp_003 discharge only | 0.656 (+0.9) | 0.949 (+2.8) | 0.744 (-0.9) | 0.578 (+0.3) | 0.450 (+0.7) | 0.514 / 0.509 | **-0.091** | MIXED, same pattern |
| exp_015 dropout 0.2 | 0.661 (+1.2) | 0.906 (-2.7) | 0.750 (-0.5) | 0.581 (+0.4) | 0.475 (+1.3) | 0.579 / 0.565 | -0.035 | MIXED: drop |
| exp_030 ProbSparse | 0.644 (+0.2) | 0.914 (-1.7) | 0.729 (-1.8) | 0.548 (-1.1) | 0.446 (+0.6) | 0.620 / 0.606 | +0.006 | NEUTRAL, non-deterministic eval |
| exp_031 d_model 128 | 0.640 (0.0) | 0.911 (-2.0) | 0.744 (-0.9) | 0.570 (-0.1) | 0.414 (-0.2) | 0.625 / 0.601 | +0.001 | NEUTRAL: 4x fewer params, 34% faster |
| exp_033 3 encoder layers | 0.637 (-0.1) | 0.926 (-0.2) | 0.719 (-2.5) | 0.534 (-1.8) | 0.412 (-0.2) | 0.640 / 0.628 | +0.028 | NEUTRAL: drop |
| exp_032 d_model 512 | 0.635 (-0.2) | 0.924 | 0.735 | 0.537 | 0.413 | 0.564 / 0.556 | -0.043 | INVALID: optimization stalled |
| exp_038 no gradient clip | 0.630 (-0.6) | **0.883 (-5.6)** | 0.700 (-3.7) | 0.554 (-0.8) | 0.467 (+1.1) | 0.574 / 0.557 | -0.043 | NEGATIVE: keep clipping |
| exp_002 global z-score | 0.619 (-1.2) | **0.875 (-6.7)** | 0.685 (-4.7) | 0.530 (-2.0) | 0.425 (+0.1) | 0.527 / 0.513 | -0.087 | NEGATIVE: keep per-basin z-score |
| exp_013 lr 5e-4 | 0.615 (-1.4) | 0.907 (-2.5) | 0.710 (-3.1) | 0.500 (-3.4) | 0.374 (-1.1) | 0.566 / 0.562 | -0.038 | NEGATIVE: optimization stalled |
| exp_014 lr 5e-5 | 0.615 (-1.4) | 0.917 (-1.3) | 0.747 (-0.7) | 0.540 (-1.5) | 0.385 (-0.9) | 0.634 / 0.630 | +0.031 | NEUTRAL/NEGATIVE: keep 1e-4 |
| exp_036 batch 128 (lr x sqrt 2) | 0.612 (-1.6) | 0.910 (-2.2) | 0.716 (-2.7) | 0.518 (-2.5) | 0.402 (-0.4) | 0.636 (spike) / 0.595 | -0.005 | NEGATIVE/NEUTRAL: no faster per epoch |
| exp_011 NSE loss (eps 0.1) | 0.608 (-1.8) | 0.914 (-1.7) | 0.740 (-1.2) | 0.548 (-1.1) | 0.346 (-1.8) | 0.654 / 0.630 | +0.030 | MIXED/NEGATIVE: drop |
| exp_008 seq_len 168 | **0.568 (-4.0)** | 0.903 (-3.1) | 0.677 (-5.1) | 0.456 (-5.4) | 0.312 (-2.7) | 0.621 / 0.611 | +0.011 | NEGATIVE on dev: keep 336 |
| exp_023 DMEL, SE threshold | **0.504 (-7.6)** | 0.904 | 0.639 | 0.387 | 0.291 | 0.582 / 0.555 | -0.045 | NEGATIVE |
| exp_020 DMEL | **0.397 (-13.6)** | 0.864 | 0.513 | 0.227 | 0.113 | 0.587 / 0.564 | -0.036 | NEGATIVE, below persistence |
| exp_034 no distillation | — | — | — | — | — | 0.625 @ epoch 28 | — | INCOMPLETE: 38 epochs, no eval |

Test (the only readout, `exp_001` s42): median NSE **0.632**, buckets 0.919 / 0.743 / 0.588 / 0.462.
For that model, dev is 0.642 and san_val is 0.619, so dev is a close but slightly optimistic proxy for
test.

---

## 4. Buckets: where the model earns skill and where it loses it

### 4.1 Lead-time buckets vs persistence (dev, median of per-basin NSE)

| | h1-12 | h13-24 | h25-36 | h37-48 |
|---|---|---|---|---|
| persistence | 0.938 | 0.686 | 0.416 | 0.249 |
| baseline, 3-seed mean | 0.927 (**-0.011**) | 0.758 (+0.072) | 0.572 (+0.156) | 0.421 (+0.172) |
| best single run (MAE) | 0.946 | 0.786 | 0.625 | 0.500 |
| 4-checkpoint ensemble | 0.949 | 0.802 | 0.649 | 0.526 |

### 4.2 Skill by lead hour

Pooled MSE skill on dev in z-space, defined as 1 - MSE_model / MSE_persistence:

| | h1 | h2 | h3 | h4 | h6 | h12 | h24 | h36 | h48 |
|---|---|---|---|---|---|---|---|---|---|
| MSE baseline | **-2.81** | -0.51 | -0.11 | +0.07 | +0.21 | +0.12 | +0.28 | +0.33 | +0.33 |
| Huber 0.5 | -2.08 | -0.34 | -0.05 | +0.12 | +0.25 | +0.12 | +0.23 | +0.29 | +0.32 |
| MAE | -4.14 | -1.03 | -0.36 | -0.13 | +0.08 | +0.06 | +0.24 | +0.30 | +0.32 |

At h+1 the model's RMSE is 0.30 z against 0.155 z for persistence. Its mean offset from the last
observed value is about 0, so this is variance, not bias. The model does not reproduce "the current
level". The code shows why this is hard for it:
- Future decoder positions are filled with zeros (`src/models/informer.py:499-504`).
- The output head has no path from the last observed value (`informer.py:521`).
- The token embedding and distillation convolutions use circular padding
  (`informer.py:65-66`, `informer.py:264-265`), so the embedding of the most recent hour mixes in the
  value from 336 h earlier.

A fixed per-horizon blend with persistence does not help. When fitted on san_val and applied to dev it
changes the median by -0.003 to +0.004, and the fitted weight is about 1.0 from h6 onward. The deficit
varies by sample and by basin, so it needs a per-sample fix: predict the change from `x[:, -1, 11]`.

### 4.3 Flow-regime buckets (dev, pooled skill vs persistence, mean bias in z)

| Regime (share of windows) | MSE baseline | Huber 0.5 | MAE |
|---|---|---|---|
| Steady, abs(mean(y) - last) < 0.25 z (87.4%) | **-1.19**, bias +0.02 | -1.32 | -0.85 |
| Falling / recession (6.4%) | **+0.79**, bias +0.25 | +0.80 | +0.79 |
| Rising (6.1%) | -0.05, **bias -0.93** | -0.11 | -0.12 |
| Low-flow tercile (within basin) | **-3.12** | -1.79 | **-0.75** |
| High-flow tercile | **+0.27**, bias -0.10 | +0.24, bias -0.13 | +0.23, **bias -0.15** |

- Rises are limited by missing information. The median model captures only 4-11% of the observed rise
  (MSE 0.11, Huber 1: 0.09, Huber 0.5: 0.075, MAE 0.04). Without future forcings (`y_aux` is not
  available at test time), a 48 h model can only react to rain that has already fallen. Do not expect
  architecture changes to fix this bucket.
- Robust losses trade high-flow accuracy for low-flow accuracy. MAE improves low-flow skill from -3.1 to
  -0.75 but under-predicts high flows more (KGE beta median 0.906, versus 1.003 for MSE s42). Huber 0.5
  sits between the two. This is the mechanism behind their dev gains, because dev has only about
  38 windows per basin and often samples quiet periods.

### 4.4 Basin buckets

- The model is worse than persistence (per-basin NSE on dev) in 32% of basins for the baseline s42,
  40% for s123, 25% for precip+discharge, and 59% for DMEL. The median per-basin skill vs persistence is
  only +0.07 to +0.19.
- The catastrophic basins are flashy, intermittent basins whose dev sample happens to fall in a quiet
  period. Basin 377 (train-period mean 0.003, std 0.114) scores NSE -11,155, and persistence also scores
  -19 there. Basins 404, 165, 188, and 17 behave similarly. The mean NSE (-9 to -526) is meaningless.
  Use median, the share of basins above 0.5, or mean(max(NSE, -1)).

---

## 5. Findings by experiment axis

### 5.1 CEEMDAN / DMEL, the core claim: not validated

What the numbers show:
- On P42, san_val is 0.587 (`exp_020`) and 0.582 (`exp_023`), against 0.619 for the baseline. That is
  below all three baseline seeds (0.619-0.660). On the same basins, DMEL is better in only 26% of them.
- Dev is 0.397 and 0.504, below persistence (0.525). The h25-36 and h37-48 buckets reach only 0.23 and
  0.11.
- Dev KGE beta (bias ratio) median is **1.29** and 1.17, against 1.00 for the baseline. San_val beta is
  1.15 and 1.05. The model over-predicts in both splits, and the effect is amplified on dev's small,
  quiet per-basin samples.

What the checkpoints show (512 random dev windows, `exp_020` / `exp_023`):

| Branch | Mean output (z) | Variation across windows (std, z) | Correlation with y |
|---|---|---|---|
| HF Informer #1 | +1.14 / +0.98 | **0.012 / 0.014** (effectively constant) | 0.20 / 0.17 |
| HF Informer #2 | +0.97 / +0.92 | **0.070 / 0.065** | 0.17 / 0.13 |
| LSTM (LF) | -1.99 / -1.84 | 0.76 / 0.68 | 0.63 / 0.60 |
| Sum vs y | **bias +0.14 / +0.08** | | 0.63 / 0.60 |

- The ensemble has degenerated into "LSTM plus constants". An R^2 using the LSTM alone with the right
  constant (0.374) is as good as or better than the full sum (0.349).
- The sum combiner cannot identify the branch offsets: the +2.1 and -2.0 constants must cancel, and at
  the selected checkpoint they do not. The resulting bias persists even on training-period windows
  (+0.07 to +0.09 z on 25 known basins). DMEL also fits the training period better than the baseline
  (MSE_z 0.301 vs 0.372) but generalizes worse to dev windows (0.547 vs 0.433).

Implementation causes, verified in code and data:
1. **Endpoint pinning.** `_anchor` forces both spline envelopes through the series endpoints
   (`src/decompose.py:137-151`). As a result every IMF is exactly 0 at t=0 and t=335. In all 600 sampled
   windows, the maximum absolute value at the endpoints was 0. The high-frequency branches therefore see
   exactly 0 at the forecast origin, which is the most informative hour, and the whole current level
   goes to the LF residual.
2. **Inconsistent routing.** With `fixed_split`, the torch path orders HF branches by SE, not by
   frequency (`decompose.py:884-887`). SE ordering is non-monotone in 22-24% of windows, so the two HF
   branches swap bands in those windows. In 4.1-4.7% of windows the level-carrying RIMF_3, with the mean
   subtracted, is routed to an Informer while the LSTM receives a zero-mean HF band
   (`src/dataset.py:366-372`). The `se_threshold` path returns index-sorted groups
   (`decompose.py:501-515`), which is more consistent, and it scored better (dev 0.504 vs 0.397).
   That is still only one seed.
3. **Dead parameters.** `build_model` defaults to `n_high=5` (`src/models/dmel.py:323`), but
   `assemble_branch_inputs` defaults to 2 (`decompose.py:876`). Three Informers are built and never
   used. Only 6.76M of the reported 15.3M parameters (44%) are in use.
4. **No per-component supervision.** Each branch is trained only through the sum on the total y. This
   leak-free setup is structurally different from forecasting each IMF separately. According to the
   notes in `config.py`, the paper decomposes whole daily series. Decomposing each 336 h window on its
   own, as done here, is the only leak-free option. A lookahead-free decomposition cannot add
   information beyond the input window, only inductive bias. Low expected gain is the prior.
5. **Smaller issues.** RIMFs are cached as float16 (`decompose.py:705`). The median relative
   reconstruction error is 3e-4, and the worst sampled window reached 6%.

Decision:
- Stop running DMEL variants as they are built now (`exp_021`, `exp_022`, `exp_024`). Each costs about
  25 h, and the architecture has already collapsed.
- The cheapest fair test of whether CEEMDAN carries usable information is one baseline-cost run: the
  plain Informer with the 3 cached RIMFs added as extra input channels (c_in = 15, raw discharge kept).
  If that run does not beat baseline plus noise, close the CEEMDAN line and write it up as a documented
  negative result with the mechanisms above.
- Only if you want to push DMEL itself, fix all of these first:
  - endpoint handling (mirror or extrapolated extrema);
  - frequency-order routing;
  - a single shared output bias or zero-mean branch outputs;
  - the `n_high` mismatch;
  - optionally, per-branch auxiliary targets from a leak-free 384 h decomposition, used for training
    only (this needs a new cache, about 17 h).

### 5.2 Loss function: the strongest signal

- The effect grows with robustness: MSE 0.640 (seed mean), Huber 1 at 0.658, Huber 0.5 at 0.666, and
  MAE at 0.683.
- The paired analysis explains how. On dev windows whose targets fall in quiet or recession periods,
  robust losses are far better: MAE scores 0.802 on them vs 0.505 for MSE. On other windows they are
  moderately better (0.714 vs 0.680). The gain comes from low- and mid-flow accuracy (Section 4.3).
- Huber 0.5 is the most stable training run. Its san_val curve is smooth and rises until epoch 37, and
  training loss barely changes after the best epoch. It gives the best smoothed san_val on P42.
- Confound: the Huber runs are rarely clipped (mean pre-clip gradient norm 0.28-0.40, below 1.0),
  whereas MSE and MAE are always clipped (2.3-2.9).
- The NSE loss (eps 0.1) down-weights low-variance basins, which are exactly where the model already
  loses to persistence. It is negative on dev in every bucket.

### 5.3 Inputs (channel ablation): a strong dev/san_val conflict

- Precip+discharge and discharge-only models are better on dev. They have the best h1-12 scores of all
  runs (0.948-0.949), the fewest catastrophic basins (19 vs 39 with NSE < 0), and the highest clipped
  mean (0.581).
- They are clearly worse on unseen basins: san_val drops 0.09-0.10, 80-84% of san_val basins get worse,
  and they lose at h25-48 (0.37-0.41 vs 0.48-0.49).
- Reading: the 10 extra forcings help the model transfer to new basins and help at long horizons, but
  in known basins they add noise at short lead times.
- Do not remove the channels. Test the residual-to-persistence output instead: it should give the
  short-lead benefit of these runs while keeping the forcings.

### 5.4 Normalization

Global z-score is clearly worse: -6.7 sigma at h1-12 and -0.087 on san_val. Per-basin z-score is
confirmed.

### 5.5 Optimization

- **Clipping.** With AdamW, global-norm clipping does not cut the effective learning rate: Adam is
  approximately invariant to a rescaled gradient, so clipping acts as a per-step normalization. It is
  still helpful. Turning it off costs -5.6 sigma at h1-12 on dev and drops san_val h1-12 from 0.864 to
  0.762. Keep 1.0.
- **Learning rate.**
  - At 5e-4, training stalls: loss stays between 0.48 and 0.53 from about epoch 5 (the baseline reaches
    0.21), and the gradient norm drops to 0.87.
  - At 5e-5 the run fits less (final training loss 0.263 vs 0.21) and is not better on dev.
  - Keep 1e-4. The LR runs are interpretable as they stand; clipping does not make them meaningless.
- **Batch size.** Batch 128 with sqrt-scaled LR is slightly negative on dev and no faster per epoch
  (572 vs 579 s), because data loading with `num_workers=0` is the bottleneck. Drop it.

### 5.6 Architecture and capacity

- d_model 128 matches d_model 256 (dev 0.640 on both), with 0.72M vs 2.85M params and 34% less time.
  Capacity is not the bottleneck.
- d_model 512 is not a valid capacity test. Training loss stalls at 0.457 (the baseline reaches 0.21).
  The larger post-LN model needs a lower LR or longer warmup.
- 3 encoder layers and ProbSparse attention are neutral at L=336.
- ProbSparse evaluation is also random at inference, because it samples keys with `torch.randint` in
  eval mode (`informer.py:215`). The restored checkpoint re-scored 0.6169 against a logged 0.6204.
- Use d_model 128 as the screening model; it lets you run more seeds per GPU-hour.

### 5.7 Window length

`seq_len` 168 is -4 sigma on dev (-5 sigma at h13-36) and neutral on san_val. It is about 40% faster
per epoch (382 vs 653 s), but it is not worth the accuracy. Keep 336.

### 5.8 Auxiliary target (`y_aux` future meteo, used as a training target only, which is legitimate)

- Dev is +1.1 sigma, with all buckets between +1.0 and +1.6 sigma.
- The san_val "best" of 0.649 is a single-epoch spike; the smoothed maximum is 0.607, level with the
  baseline.
- It costs about 15% more time per epoch (750 vs 653 s). Treat it as optional, to combine with the
  winner later.

### 5.9 Epochs and convergence

| Pattern | Runs | Would more epochs help? |
|---|---|---|
| Peak mid-schedule, then decline while training loss keeps falling 35-48% | s42, 002, 005, 009, 015, 030, 036, 038, 020, 023 (all on P42) | No. They need regularization or weight averaging, not epochs. |
| Late peak, then plateau (best within the last 25% of the schedule, flat training loss) | s123, s177, 014, 011, 008 | Marginal |
| Still rising at epochs 37-40, no overfitting | **010 Huber 0.5** (+0.012 per 10 epochs), **012 MAE** (+0.010 per 10 epochs) | Possibly, a small gain below single-seed noise |
| Optimization stalled (training loss plateau 0.46-0.53) | 013 (lr 5e-4), 032 (d_model 512) | No. Fix LR and warmup. |
| Information-limited (high training loss because inputs are missing) | 003, 004 | No |

Only one extension is justified: 60 epochs for the robust-loss winner, and only after seeds confirm
that the effect is real. Extending cosine decay from 40 to 60 epochs changes the whole schedule, not
just the tail, so treat it as a new configuration.

---

## 6. Evaluation-protocol audit (code-level issues that affect conclusions)

| # | Issue | Evidence | Impact | Fix |
|---|---|---|---|---|
| P1 | Changing the seed changes the san_val partition | `experiment.py:134` passes the seed to `build_splits`. P42, P123, and P177 share only 9/7/5 basins. | The 3-seed arm mixes partition variance with seed variance, and paired seed comparisons are confounded. | Add a separate `split_seed`, or match seeds across arms (Section 8) |
| P2 | The logged persistence reference is point-horizon, not bucket-level | `evaluate.py:245` hardcodes 0.84 / 0.58 / 0.35 / 0.22. The bucket values on dev are 0.938 / 0.686 / 0.416 / 0.249. | The logs report h1-12 "beats persistence" when it does not. DMEL's logged "+0.024" is really -0.074. | Compute persistence per bucket from the same windows |
| P3 | The per-basin Wilcoxon is used as a significance test | seed vs seed gives p = 1.3e-8 | False positives | Test at seed level; report per-basin statistics as descriptive only |
| P4 | Mean NSE is dominated by quiet dev samples of flashy basins | basin 377: -11,155 | Mean, std, and to a lesser degree the share above 0.5 are unstable | Report mean(max(NSE, -1)) or an NSE with an epsilon term, and keep the median |
| P5 | Dev overlaps the training record | 55.6% of dev windows share inputs with `split=0`; 24.7% have their 48 h target inside the training record. On 8 sampled basins, about 41% of test windows overlap train.h5. | Dev is somewhat optimistic and imbalanced across flow regimes. There is **no sign** that the selected checkpoints exploit the overlap: on target-seen windows, MSE skill vs persistence is +0.005, against +0.214 on clean windows. | Also report dev-clean (non-overlapping windows) |
| P6 | 61 test basins were never in training | Training uses 447 basins; test covers all 508 | About 12% of test basins are scored zero-shot (dev: known 0.646 vs unseen 0.622) | Retrain the final model on all 508 basins with a fixed schedule |
| P7 | ProbSparse evaluation is stochastic | `informer.py:215`; re-score 0.6169 vs logged 0.6204 | The best-epoch choice was made on a random draw | Fix the sampling seed in eval, or drop ProbSparse |
| P8 | The ridge baseline scores below persistence (0.417 vs 0.525) | lambda = 3e5 on 4,033 standardized features shrinks the last-value coefficient; lambda was tuned on dev | "Informer beats the linear model by 0.22" overstates the value of nonlinearity | Fit ridge on y - q_last (a residual ridge) |
| P9 | Circular padding in the input convolutions | `informer.py:65-66, 264-265` | The most recent hour's embedding sees x[0], from 14 days earlier. This may contribute to the h1 deficit (unproven). | Use replicate or causal padding |
| P10 | DMEL uses 3 of 5 Informers and reports 15.3M params | Section 5.1 | Cost is misreported, and changing `n_high` would silently misbehave | Share one default |
| P11 | `exp_034__no_distil` is incomplete | Only `history.csv` (38 epochs) and `best_metrics.json` were synced | No verdict possible | Finish the run and evaluate it on machine B |
| P12 | Selection uses a spiky maximum | Section 2.3 | Winner's curse; the dev score comes from a noisy checkpoint | Use EMA or SWA weights, or select on a 3-epoch smoothed curve |

---

## 7. Zero-training-cost result: ensembling

Predictions come from the saved checkpoints on the full dev set. As a sanity check, the single-model
medians recomputed exactly: 0.6419, 0.6581, 0.6657, and 0.6830.

| Ensemble (simple average of z predictions) | dev median | share of basins > 0.5 | h1-12 | h13-24 | h25-36 | h37-48 |
|---|---|---|---|---|---|---|
| MSE alone (s42) | 0.642 | 64.0% | 0.927 | 0.757 | 0.576 | 0.468 |
| MAE alone | 0.683 | 65.9% | 0.946 | 0.786 | 0.625 | 0.500 |
| mean(Huber 0.5, MAE) | 0.702 | 66.7% | 0.951 | 0.792 | 0.626 | 0.504 |
| **mean(MSE, Huber 1, Huber 0.5, MAE)** | **0.708** | **67.9%** | 0.949 | 0.802 | 0.649 | **0.526** |

No weights were fitted and no dev-based choices were made, so this is a fair dev estimate. The gain
concentrates in the long buckets, which have the highest seed variance, exactly where averaging should
help. Plan the final submission as an ensemble. Keep test untouched until the final readout.

---

## 8. Recommended plan (ranked by expected value per GPU-hour)

### Tier 0: no training (do first, less than a day of work)

1. Extend `evaluate.py`:
   - persist per-basin x per-bucket NSE;
   - compute bucket-level persistence on the same windows (fixes P2);
   - add dev-clean and dev-overlap masks;
   - add rising/steady/falling and low/mid/high flow buckets;
   - add skill vs persistence per lead hour;
   - add mean(max(NSE, -1)).

   Then re-score every local checkpoint. The one-off analysis in this document can become the standard
   report.
2. Fix P7 (seed the ProbSparse eval), P10 (`n_high` default), and P3 (label the Wilcoxon output as
   descriptive only).
3. Finish `exp_034` on machine B. Sync the machine B checkpoints (`best_model.pt`) so their runs can
   join the re-scoring and the ensemble.

### Tier 1: the next training runs (about 6.4 h each; no co-scheduling on MPS)

| ID | Change | Why | Seeds |
|---|---|---|---|
| **R1** | Residual output: `pred = decoder_out + x_enc[:, -1, 11]`, with Huber 0.5 | Targets the largest measured deficit: h1-3, steady windows (87% of windows), and the 25-44% of basins below persistence. The reduced-input runs already show the h1-12 headroom (0.949). | 42, then 123 and 177 if positive |
| R2 | Residual output with MSE | Isolates the residual effect from the loss effect | 42 |
| **R3** | Huber 0.5 with seeds 123 and 177, using the current coupled split seed, on machine B (CUDA, the same device as the existing s123/s177) | Each gives a clean pair with the existing baseline s123 and s177 (same partition, same training basins, same device), so you get 3 matched pairs without re-running the baseline | 123, 177 |
| R4 | MAE with seeds 123 and 177, also on machine B | Same design; decides between MAE and Huber 0.5 (the MAE option has a peak-bias risk) | 123, 177 |
| R5 | EMA of weights (for example decay 0.999 per step; average BatchNorm buffers too, or recompute them) and select on the EMA model | Cuts checkpoint noise (P12) and is usually a free gain. Can be added to R1 or R3. | combine |

Decision rule: adopt a change if the mean of its 3 seed-matched dev differences is at least 0.02, all
3 are positive, and the h25-48 buckets do not get worse.

### Tier 2: the CEEMDAN answer (one run)

- R6: the Informer with RIMF_1..3 as extra channels (c_in = 15), using the existing cache and the same
  loss as the Tier 1 winner.
- If the result is at most baseline plus noise, close the DMEL/CEEMDAN line and report it as a negative
  result. The evidence is the branch collapse, endpoint pinning, and cost from Section 5.1.

### Tier 3: finalization

- R7: pick the winning configuration and use a fixed epoch budget (the median best epoch across its
  seeds, or the full cosine schedule with EMA).
- Train K = 3 to 5 seeds on **all 508 basins** (a no-holdout mode), average them, and score test once.
- Alternative: 3 disjoint 61-basin holdouts (rotating san_val). This gives an out-of-basin score on 183
  basins, a variance estimate, and ensemble members from the same compute.

### Optional, after Tier 1

- A plain LSTM on the raw 12 channels as a cheap, paper-relevant baseline. DMEL effectively became an
  LSTM, so it is worth knowing where an LSTM stands on its own.
- 60 epochs for the confirmed winner.
- The aux task on top of the winner.
- d_model 512 with LR 5e-5 and 10% warmup (low priority, since capacity is not the bottleneck).

### Stop spending compute on

- DMEL as currently implemented (`exp_021`, `exp_022`, `exp_024`)
- global z-score, seq_len 168, disabling clipping, lr 5e-4, batch 128, dropout 0.2, the NSE loss,
  ProbSparse attention, 3 encoder layers
- removing input channels (fold that idea into R1 instead)

Budget: R1 to R5 are 8 runs, about 52 GPU-hours, or about 3 days on two machines. R6 is 1 run. R7 is
3 to 5 runs.

---

## 9. What can be claimed today (for the report or paper)

- A leak-free Informer beats persistence by +0.12 dev median NSE (3-seed mean 0.640 ± 0.018 vs 0.525).
  On test (a single readout) it scores 0.632. The skill comes entirely from lead times of 4 h and more,
  and mainly from recessions. It is worse than persistence at h1-3.
- The CEEMDAN-based DMEL, implemented with leak-free per-window decomposition, does **not** improve on
  the single Informer (san_val 0.587 vs 0.619, dev 0.397 vs 0.640). The mechanism has been diagnosed:
  the high-frequency branches collapse, and the sum carries a +0.08 to +0.14 z bias.
- Per-basin z-score normalization, gradient clipping, the 336 h context, and lr 1e-4 are all supported
  by clearly negative ablations.
- Robust losses (Huber 0.5 and MAE) are promising but not yet significant with the current number of
  seeds.

---

## 10. Methods of the ad-hoc analyses (read-only on repository artefacts; this file is the only repository change)

1. **Metrics tables.** Parsed every `summary.json`, `metrics_{dev,sanval}.json` (per-basin lists),
   `history.csv`, and `best_metrics.json`. The known/unseen dev split uses each run's own san_val basin
   list from `metrics_sanval.json`.
2. **Noise.** Sample sigma across the 3 baseline seeds, per metric. Bootstrap over basins (4,000
   resamples) for differences of medians. Paired Wilcoxon on basins known to both runs.
3. **Curves.** 3-epoch moving-average maximum, mean of the last 5 epochs, slope over the last 10
   epochs, and the standard deviation of epoch-to-epoch differences in the second half of training.
4. **Overlap.** Informative 6-grams (at least 3 distinct values) of discharge at stride 1, with at least
   3 matching grams counting as an overlap, using the same criterion as `src/splits.py`:
   - dev vs `split=0` inputs, all 508 basins;
   - "target-seen": dev targets vs `split=0` inputs and targets;
   - test inputs vs train.h5 inputs, 8 random basins (415 windows).
5. **Checkpoint inference** on MPS for `exp_001`, `exp_009`, `exp_010`, and `exp_012`: all 18,142 dev
   windows and 8,000 san_val windows. Scratch outputs went to `/tmp/dmel_review/`. DMEL branch
   decomposition was run on CPU: 512 random dev windows for
   `exp_020` and `exp_023`, plus 915 dev and 915 training-period windows from 25 known basins for
   `exp_020` and the baseline.
6. **RIMF cache checks** on `data/rimf_train_110b9c26af82.h5`: completeness (all 272,142 rows done),
   the reconstruction identity on 600 windows, endpoint values of the HF RIMFs, SE monotonicity
   (all rows), and the routing statistics for LF != RIMF_3.
