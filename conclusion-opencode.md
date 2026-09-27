# Conclusions and Follow-up Items

Source: an independent review of `experiments/` and `src/`, with read-only re-runs of the saved
checkpoints. No other review document was used. The full evidence, tables, and methods are in
`EXPERIMENT_DEBRIEF.md`.

Terms used below:
- dev: the competition validation split (`split=1`), all 508 basins, about 38 windows per basin.
- san_val: the 61 basins held out for early stopping.
- NSE: per-basin Nash-Sutcliffe efficiency, reported as the median across basins.
- sigma: the standard deviation across the three baseline seeds.
- Buckets: forecast lead-time buckets h1-12, h13-24, h25-36, h37-48.
- z: normalized units (per-basin z-score).

---

## 1. Conclusions

### 1.1 Most measured differences are noise

- The three baseline seeds give a dev median NSE of 0.621 / 0.642 / 0.657 (sigma = 0.018).
- In the h37-48 bucket the seeds range from 0.395 to 0.468 (sigma = 0.041).
- With one run per experiment against three baseline seeds, a difference must be at least about 0.04
  before you can trust it. Under a strict t-test (2 degrees of freedom) it must be at least about 0.09.
- Across 19 configurations on the same basin split, the dev ranking and the san_val ranking are
  uncorrelated (Spearman rho = 0.00).
- No tuning experiment clearly beats the baseline. Several are clearly worse.

### 1.2 CEEMDAN/DMEL, the core claim of the paper, is not validated

Results:
- san_val: 0.587 and 0.582 against 0.619 for the baseline on the same basins. This is below all three
  baseline seeds.
- dev: 0.397 and 0.504, which is below persistence (0.525).
- Each run costs 3.5-4x the baseline's compute.

What the checkpoints show:
- The two high-frequency Informer branches output near-constants. Their outputs vary by only
  0.01-0.07 z across windows.
- The LSTM does all the useful work. The summed output over-predicts by +0.08 to +0.14 z. That bias
  is what destroys the dev NSE.

Implementation causes found in the code:
- The high-frequency components are exactly 0 at the last input hour. The spline edge anchoring in
  `decompose.py:137-151` forces this.
- Routing by sample entropy (SE) swaps the two high-frequency bands between branches in 22-24% of
  windows.
- 3 of the 5 Informers that are built are never used. Only 44% of the reported parameters are active.
- The branches are trained only through their sum, so the individual branch offsets are not
  identifiable.

### 1.3 Robust losses are the best signal so far, but still unconfirmed

- MAE scores 0.683 on dev (+2.4 sigma). Huber delta=0.5 scores 0.666 (+1.4 sigma). Huber delta=1
  scores 0.658 (+1.0 sigma).
- The gain grows steadily as the loss becomes more robust.
- Huber 0.5 also has the best smoothed san_val curve on the same basins: +0.068 over the baseline.
- The gain comes from low- and mid-flow accuracy.
- MAE under-predicts high flows. Its KGE beta (bias ratio) median is 0.906, against 1.003 for MSE.

### 1.4 The largest structural weakness is short lead times and steady flow

Where the model loses to persistence:
- It is worse than persistence at h+1 to h+3. At h+1 its MSE is 3.8x persistence's.
- It is worse on steady-flow windows, which are 87% of all windows. There its MSE is 2.2x
  persistence's.
- It is worse than persistence in 25-44% of basins overall.

Where the model earns its skill:
- On recessions (skill +0.79 against persistence).
- At long lead times (+0.33 at h24-h48).

What the model cannot do:
- It captures only 4-11% of observed rises. This is an information limit: the model has no future
  rainfall input.

### 1.5 Ensembling is the largest free gain

Averaging the 4 existing checkpoints (MSE, Huber 1, Huber 0.5, MAE) gives:
- dev 0.708, against 0.642 for the baseline and 0.683 for the best single run.
- An improvement in every bucket (0.949 / 0.802 / 0.649 / 0.526).

### 1.6 More epochs is not the binding constraint

- MSE runs peak between epochs 14 and 31, then plateau or decline.
- Only Huber 0.5 and MAE are still rising at epochs 37-40, and only by about +0.01 per 10 epochs.
  That is below seed noise.
- Spend compute on seeds before spending it on epochs.

### 1.7 Protocol issues limit what the current runs can prove

- **Seed and basin split are coupled.** Changing the seed also changes the san_val basin set.
  The seed-123 and seed-177 splits share only 9 and 7 basins with seed 42.
- **Invalid significance test.** A per-basin Wilcoxon test gives p = 1e-8 between two seeds of the
  same configuration, so it cannot be used to judge configurations.
- **Wrong persistence reference in the logs.** `evaluate.py:245` hardcodes values that are not
  bucket-level persistence.
- **Unseen test basins.** 61 of the 508 test basins were never used in training by any model.
- **Non-deterministic evaluation.** ProbSparse evaluation is random.
- **Incomplete run.** `exp_034` has training history but no evaluation.
- **Weak ridge baseline.** The ridge baseline scores below persistence, so "the Informer beats the
  linear model" overstates the value of nonlinearity.

### 1.8 What can be claimed today

- A leak-free Informer beats persistence by +0.12 dev median NSE (3-seed mean 0.640 ± 0.018 vs 0.525).
  Its single test readout is 0.632.
- All of its skill comes from lead times of 4 h or more, and mostly from recessions. It is worse than
  persistence at h1-3.
- DMEL with leak-free per-window CEEMDAN does not improve on the single Informer (san_val 0.587 vs
  0.619). The mechanism is diagnosed: the branches collapse and the sum is biased.
- These choices are supported by clearly negative ablations:
  - per-basin z-score normalization;
  - gradient clipping at 1.0;
  - the 336 h input context;
  - learning rate 1e-4.
- Robust losses (Huber 0.5, MAE) are promising but not yet statistically significant.

---

## 2. Follow-up items (ranked by expected value per GPU-hour)

### Tier 0: no training (do first, less than a day)

- [ ] **Extend `evaluate.py`**, then re-score every local checkpoint:
  - persist per-basin, per-bucket NSE;
  - compute bucket-level persistence from the same windows (fixes the log reference);
  - report dev-clean and dev-overlap windows separately;
  - add rising / steady / falling and low / mid / high flow buckets;
  - add skill against persistence per lead hour;
  - add mean(max(NSE, -1)) in place of the unusable mean NSE.
- [ ] **Code fixes:**
  - seed the ProbSparse evaluation;
  - use a single `n_high` default in `dmel.py:323` and `decompose.py:876`;
  - label the per-basin Wilcoxon output as descriptive only.
- [ ] **Machine B housekeeping:**
  - finish and evaluate `exp_034__no_distil`;
  - sync the machine B `best_model.pt` files so those runs can be re-scored and ensembled.

### Tier 1: next training runs (about 6.4 h each; do not co-schedule on MPS)

| ID | Change | Why | Seeds |
|---|---|---|---|
| **R1** | Residual output: `pred = decoder_out + x_enc[:, -1, 11]`, with Huber 0.5 | Targets the largest measured deficit: h1-3, steady windows (87%), and the 25-44% of basins below persistence | 42, then 123 and 177 if positive |
| R2 | Residual output with MSE | Separates the residual effect from the loss effect | 42 |
| **R3** | Huber 0.5, seeds 123 and 177, on machine B (CUDA) | Pairs with the existing baseline s123/s177 runs: same basin split, same training basins, same device. Gives 3 matched pairs without re-running the baseline. | 123, 177 |
| R4 | MAE, seeds 123 and 177, on machine B | Same design. Decides between MAE and Huber 0.5, given MAE's peak-bias risk. | 123, 177 |
| R5 | EMA of weights (for example decay 0.999 per step; also average the BatchNorm buffers) and select checkpoints on the EMA model | Reduces checkpoint noise and the winner's-curse effect of picking the best epoch. Can be added to R1/R3. | combined |

**Decision rule:** adopt a change only if all three of these hold:
- the mean of its 3 seed-matched dev differences is at least 0.02;
- all 3 differences are positive;
- the h25-48 buckets do not get worse.

### Tier 2: settle the CEEMDAN question (one run)

- [ ] **R6: CEEMDAN components as extra input channels.**
  - Setup: the plain Informer with RIMF_1..3 added as extra input channels (c_in = 15). Keep the raw
    discharge channel and reuse the existing cache.
  - Loss: the same as the Tier 1 winner.
- [ ] **If R6 is at most baseline plus noise:** close the DMEL/CEEMDAN line. Report it as a negative
  result backed by the diagnosed mechanisms (branch collapse, zero components at the last hour,
  3.5-4x cost).
- [ ] **Only if you still want to push DMEL itself**, fix these first:
  - edge handling in the decomposition (mirror or extrapolated extrema);
  - routing by frequency order instead of SE order;
  - a single shared output bias;
  - the `n_high` mismatch;
  - optionally, per-branch auxiliary targets from a leak-free 384 h decomposition, used for training
    only. This needs a new cache (about 17 h).

### Tier 3: finalization

- [ ] **R7: final ensemble.**
  - Pick the winning configuration with a fixed epoch budget.
  - Train 3-5 seeds on all 508 basins, with no holdout.
  - Average the seeds.
  - Score test once, at the very end.
- [ ] **Alternative:** use 3 disjoint 61-basin holdouts. This gives an out-of-basin score on 183
  basins, a variance estimate, and ensemble members from the same compute.

### Optional, after Tier 1

- [ ] A plain LSTM on the raw 12 channels. It is a cheap baseline and relevant to the paper, since
  DMEL effectively collapsed into an LSTM.
- [ ] 60 epochs for the confirmed winner. Treat it as a new schedule, not an extension of the old one.
- [ ] The auxiliary task on top of the winner (+15% time per epoch).
- [ ] d_model 512 with lr 5e-5 and 10% warmup. The current d_model 512 run was an optimization
  failure, not a capacity result.

### Stop spending compute on

- DMEL as currently implemented (`exp_021`, `exp_022`, `exp_024`).
- global z-score normalization.
- seq_len 168.
- disabling gradient clipping.
- lr 5e-4.
- batch size 128.
- dropout 0.2.
- the NSE loss.
- ProbSparse attention.
- 3 encoder layers.
- removing input channels. Fold that idea into R1 instead.

### Budget

- R1-R5: 8 runs, about 52 GPU-hours, about 3 days on two machines.
- R6: 1 run.
- R7: 3-5 runs.
