# Next Iteration: Last Experiment Round Before the Final Model

Date: 2026-09-26
Inputs: `conclusion-pi.md`, `conclusion-opencode.md`, `conclusion-claude.md`, `EXPERIMENT_DEBRIEF.md`,
`metric.md`, the code in `src/`, and new checks on `data/` (Section 6).

Budget: **one** more iteration of experiments. After it, the config is frozen and the final model is
trained.

Terms:
- dev: `split=1`, all 508 basins, about 38 windows per basin.
- san_val: the 61 held-out basins (their `split=0` windows), used for early stopping.
- σ: the standard deviation across the 3 baseline seeds (0.018 dev, 0.041 at h37-48).

---

## 1. What we already know (agreed by all reviews; do not re-test)

| Fact | Evidence |
|---|---|
| Robust loss is the only real gain | MAE 0.683 (+2.4σ), Huber 0.5 0.666, Huber 1 0.658, MSE 0.640 (3-seed mean). One seed each. |
| DMEL as built fails | dev 0.397 / 0.504, below persistence (0.525). san_val 0.587 / 0.582 vs 0.619 for the baseline. |
| Keep per-basin z-score, clipping at 1.0, seq 336, lr 1e-4, batch 64 | Clearly negative ablations |
| Keep all 12 input channels | Reduced-channel runs gain on dev but lose 0.09-0.10 on san_val |
| Capacity is not the bottleneck | d_model 128 = d_model 256 |
| Ensembling is the largest free gain | Mean of 4 existing checkpoints = **0.708** dev |
| All skill is at h ≥ 4 | The model is worse than persistence at h1-3 and on steady windows (87% of windows) |

## 2. Confirmed bugs (fix before the next runs)

| ID | Bug | Location | Affects |
|---|---|---|---|
| B1 | HF components are exactly 0 at t=0 and t=335 (the spline envelopes are pinned to the endpoints) | `src/decompose.py:137-151` | DMEL / any RIMF use |
| B2 | `n_high` default 5 vs 2; `CEEMDAN_CFG` never reaches `build_model` | `src/models/dmel.py:323`, `src/decompose.py:876`, `src/experiment.py:128` | DMEL param count (15.3M reported, 6.76M real) |
| B3 | HF branches are ordered by SE, not frequency: the two bands swap in 23% of windows | `src/decompose.py:884-887` | DMEL |
| B4 | Persistence reference in the logs is hardcoded and wrong | `src/evaluate.py:245` | Logged "beats persistence" claims |
| B5 | Circular padding: the last hour's embedding mixes in the value from t=0 | `src/models/informer.py:65-66, 264-265` | Probably the h1 deficit |
| B6 | No path from the last observed discharge to the output | `src/models/informer.py:499-521` | h1-3 and steady-flow deficit |
| B7 | ProbSparse eval samples randomly | `src/models/informer.py:215` | Only ProbSparse (dropped) |
| B8 | The seed also changes the san_val basin set | `src/experiment.py:134` → `splits.build_group_split` | Seed comparisons |
| B9 | exp_034 was never evaluated | `experiments/exp_034__no_distil/` | Housekeeping |

---

## 3. Step 0: code changes (no GPU, about 1 day)

Do these in order. C1-C3 block the first machine-A run.

| # | Change | Where | Done when |
|---|---|---|---|
| **C1** | `residual_output: bool` flag. `pred = dec_out + x_enc[:, -1, TARGET_CHANNEL]`. Valid because `x[..., 11]` and `y` use the same per-basin stats (`dataset.py:342`, `:384`). | `informer.py:521`, `dmel.build_model` | A model with zeroed weights predicts persistence exactly |
| **C2** | `pad_mode: "circular" \| "replicate"` flag for TokenEmbedding and the distil conv | `informer.py:65`, `:264` | Default unchanged ("circular") so old checkpoints still load |
| **C3** | EMA shadow weights (decay 0.999 per step). Evaluate san_val on **both** the raw and EMA weights every epoch; save `best_model.pt` and `best_model_ema.pt`; log both in `history.csv`. Selection and comparisons use raw weights, so results stay comparable with old runs. | `train.py` | `history.csv` has a `sanval_nse_ema` column |
| **C4** | `rimf_input_mode="append"`: plain Informer, c_in=15 (12 raw channels + 3 RIMFs, normalized as in `dataset.py:368-372`) | `dataset.py`, `build_model`, `forward_batch` | Shape test passes with `use_ceemdan=False`, `rimf_append=True` |
| **C5** | Evaluation report: bucket persistence from the same windows (fixes B4), dev-clean vs dev-overlap, mean(max(NSE,−1)), a peak-flow metric (high-flow-tercile KGE β and FHV), per-basin × per-bucket NSE saved to JSON | `evaluate.py` | Re-scoring exp_001 s42 reproduces 0.6419 |
| **C6** | `ensemble_eval.py`: load N checkpoints, average z-predictions, score with C5 | new file | Mean of {001, 009, 010, 012} reproduces 0.708 |
| **C7** | `train_all_basins: bool` (no san_val; fixed epochs; save last and EMA) and `split_seed` (defaults to `seed`, so behaviour is unchanged) | `experiment.py`, `dataset.build_splits` | Final-model mode runs 1 epoch on a subset |
| C8 | Fix B2: `n_high` default 2, merge `CEEMDAN_CFG` into `cfg`, assert `len(hf_branches) == len(x_high)` | `dmel.py:323`, `experiment.py:128` | `build_model({"use_ceemdan": True})` reports 6,764,338 params |
| C9 | Label `wilcoxon_nse` output as descriptive only | `evaluate.py:371` | — |

New registry entries in `run_ablation.py`:

```python
dict(name="exp_050__resid_huber05", priority="high",
     description="residual-to-persistence output + replicate pad, Huber 0.5",
     overrides={"loss": "huber", "huber_delta": 0.5,
                "residual_output": True, "pad_mode": "replicate"}),
dict(name="exp_051__resid_mse", priority="high",
     description="residual-to-persistence output + replicate pad, MSE",
     overrides={"loss": "mse",
                "residual_output": True, "pad_mode": "replicate"}),
dict(name="exp_052__rimf_append_huber05", priority="medium",
     description="Informer + 3 RIMFs as extra input channels (c_in=15)",
     overrides={"loss": "huber", "huber_delta": 0.5, "rimf_append": True}),
dict(name="exp_053__lstm_huber05", priority="low",
     description="plain 2-layer LSTM on the 12 raw channels",
     overrides={"loss": "huber", "huber_delta": 0.5, "base_model": "lstm"}),
```

---

## Status (2026-09-26, evening)

- Step 0 is implemented: C1-C9, plus B1 (opt-in `edge_mode="mirror"`) and B3 (opt-in
  `hf_route="frequency"`). The legacy defaults keep old runs and the existing RIMF cache reproducible.
- Verified: re-scoring exp_001/009/010/012 reproduces 0.6419 / 0.6581 / 0.6657 / 0.6830, and the
  4-model ensemble reproduces 0.7076.
- Dev overlap flags are built for all 18,142 windows (`data/dev_overlap_flags.npz`): 55.6% input
  overlap, 24.5% target overlap, 43.4% clean. Clean windows score *higher* (exp_010: 0.710 clean vs
  0.506 overlap), so there is no sign of memorization.
- Machine A queue `run_queue_iter2_machineA.sh` started at 20:54 with exp_050.
- Machine B: run `run_queue_iter2_machineB.sh` after pulling this code.
- Progress and verdicts at any time: `python iter2_report.py`.
- The registry numbers are exp_050-053 because exp_040 was already taken (`exp_040__nse_eps0`).

## 4. Step 1: the runs (priority order)

One run takes about 6.4 h. **Never run two jobs on the same MPS GPU at once.**

### Machine B (CUDA): seed replication of the loss claim

These pair with the existing baseline s123/s177, which were trained on the same device with the same
partitions. That gives 3 matched pairs per loss without re-running the baseline.

| Order | Run | Command | Question |
|---|---|---|---|
| B0 | exp_034 eval only | score the existing `best_model.pt` on dev | Housekeeping (minutes) |
| **B1** | Huber 0.5, s123 | `run_ablation.py --only exp_010__huber_d05 --seeds 123 177` | Is the robust-loss gain real? |
| **B2** | Huber 0.5, s177 | (same command) | |
| **B3** | MAE, s123 | `run_ablation.py --only exp_012__mae --seeds 123 177` | MAE or Huber 0.5? |
| **B4** | MAE, s177 | (same command) | |
| B5 | exp_050, s123 | only if A1 passes its gate (below) | Replicates the residual output |
| B6 | exp_050, s177 | only if A1 passes | |

### Machine A (MPS): architecture and the CEEMDAN answer

| Order | Run | Compare against | Question |
|---|---|---|---|
| **A1** | exp_050 resid + Huber 0.5, s42 | exp_010 (Huber 0.5, s42) | Does the residual output fix h1-3 and steady windows? |
| **A2** | exp_051 resid + MSE, s42 | exp_001 s42 | Is the gain from the residual output or from the loss? |
| **A3** | exp_052 RIMF-append + Huber 0.5, s42 | exp_010 | Does CEEMDAN carry any usable information? |
| A4 (optional) | exp_053 plain LSTM | exp_010 | Paper baseline (DMEL effectively became an LSTM) |

Total: 9-10 runs, about 2.5-3 days of wall-clock on two machines.

### Decision rules (fixed now, before any results)

| Decision | Rule |
|---|---|
| **Adopt robust loss** | Mean of the 3 seed-matched dev deltas vs baseline ≥ 0.02, all 3 positive, and h25-48 not worse |
| **MAE vs Huber 0.5** | Pick MAE only if its 3-seed dev mean beats Huber 0.5 by ≥ 0.01 **and** its high-flow KGE β ≥ 0.9. Otherwise pick Huber 0.5 (better san_val gap and peaks). |
| **Gate for running B5/B6** | A1 beats exp_010 on dev by ≥ 0.01 **and** skill vs persistence at h1-3 improves |
| **Adopt residual output** | Same rule as for the loss, on the 3 seeds (A1, B5, B6) |
| **Adopt EMA** | EMA san_val sm3 ≥ raw san_val sm3 in ≥ 4 of the runs above (free, since it's logged in every run) |
| **CEEMDAN** | If A3 beats exp_010 by less than 0.02 on dev **and** san_val, close the CEEMDAN line and write it up as a diagnosed negative result (Section 6.2) |

---

## 5. Step 2: the final model (after this iteration)

1. **Config:** the winning loss, plus the residual output if adopted. d_model 256, 40-epoch
   warmup-cosine schedule, clip 1.0, lr 1e-4, batch 64.
2. **Training:** `train_all_basins=True`, all 508 basins, no holdout, fixed 40 epochs, keep the EMA
   weights (or the last epoch if EMA was not adopted). 5 seeds: 42, 123, 177, 777, 2024.
3. **Ensemble:** average the 5 seeds with `ensemble_eval.py`. If both losses passed the adoption rule,
   use 3 Huber 0.5 + 2 MAE members: that pair averaged to 0.702 vs 0.683 for MAE alone.
4. **Test:** score `test.h5` **once**. Report median NSE, per-bucket NSE with correct persistence,
   the share of basins with NSE > 0.5, mean(max(NSE,−1)), high-flow KGE β, and dev-clean.
5. **Before step 4:** confirm the competition's official metric. `metadata.json` doesn't define one;
   we have been assuming median per-basin NSE.

## 6. Verified findings behind this plan

### 6.1 Leakage between san_val, dev, and train

Measured on 15 random san_val basins (seed-42 partition, 541 dev windows), with the same 6-gram
criterion as `src/splits.py`:

| Pair | Overlap | Is it leakage? |
|---|---|---|
| train ↔ san_val | 0% (basin-disjoint) | No. The split is correct. |
| **san_val ↔ dev** (the 61 held-out basins) | **51.4%** of dev inputs share hours with a san_val window; **24.0%** of dev *targets* fall inside san_val data | **Selection leakage only.** The model never trains on these hours, but early stopping looked at them when picking the best epoch. It's weak (one epoch choice per run), but "dev on unseen basins" is not fully independent of selection. |
| **train ↔ dev** (the 447 training basins) | 55.6% inputs / 24.7% targets (debrief); 78.8% by the looser `splits.py` count | Built into the benchmark. `test.h5` has the same property (~41% overlap on 8 basins), so dev is a *faithful* proxy for test, not an unfair one. No sign that the model exploits it (debrief P5). |
| Target → model input | None | The decoder future is zeros, `y_aux` is only a training target, and the RIMF cache is decomposed from `X` only (336 h, never `y`) |
| Normalizer | Fit on `split=0` of all 508 basins | Fine. It uses no dev or test windows, and test basins need these stats anyway. |

Consequence: san_val measures *unseen basins*, while dev and test measure *unseen time in known basins*.
That's why their rankings don't correlate (ρ = 0.00). The final model trains on all 508 basins and is
scored on known basins, so **dev is the right proxy for the final decision**. san_val stays useful as
the early-stopping signal and as a check against overfitting.

### 6.2 Why CEEMDAN/DMEL fails (first principles plus measurements)

Measured on 1,934 non-degenerate cached windows (`data/rimf_train_110b9c26af82.h5`):

| Check | Result |
|---|---|
| Variance share of each RIMF (median) | RIMF1 **0.2%**, RIMF2 **0.6%**, RIMF3 (LF) **97.9%** |
| HF value at the forecast hour (t=335) | **Exactly 0** in every window (max 0.0 / 5e-4, i.e. float16 rounding) |
| RIMF3 at t=335 | Equals q(335) (the whole current level goes to the LSTM) |
| The two HF branches swap bands | 23.5% of windows |
| LF branch gets a non-slowest band | 2.2% of windows |
| Reconstruction sum(RIMF) = q | Holds (float16 error ~1e-3 relative) |

The argument:
1. **No new information.** Each RIMF is a deterministic function of the 336 h discharge that the
   baseline already sees. A leak-free, per-window decomposition can only add inductive bias, never
   information. The paper decomposes the *whole* series before splitting, so its IMF values at time t
   depend on later hours, including the forecast targets. That is a known source of leakage in
   EMD-based forecasting papers, and it would explain why its gains don't replicate here.
2. **The Transformer branches get almost nothing.** At hourly scale, 98% of discharge variance is the
   slow component. The two Informers, the paper's key branches, forecast a signal holding <1% of the
   variance and exactly 0 at the forecast hour. The loss is on the total y, so they learn constants
   (measured output std 0.01-0.07 z). The LSTM does all the work.
3. **The sum can't identify the branch offsets.** The +1.0 / +1.0 / −2.0 z constants have to cancel,
   and they don't at the selected checkpoint. That leaves a +0.08-0.14 z bias, which is what destroys
   dev NSE on the quiet 38-window samples.
4. **More parameters on the same data** (3 branches that each re-learn the meteo→discharge map) mean
   worse generalization: DMEL fits training windows better (MSE_z 0.301 vs 0.372) and dev windows worse
   (0.547 vs 0.433).

exp_052 (RIMF-append) is the cheapest fair test of point 1. The prior is that it's null.

### 6.3 Is the evaluation correct for this dataset?

| Item | Verdict |
|---|---|
| NSE per basin, in mm/h after inverting the same z-score, pooled over windows × 48 h | Correct |
| Median across basins as the headline | Correct. The mean is destroyed by flashy basins with quiet dev samples (basin 377: −11,155). |
| Train and eval share one input builder (`train.forward_batch`) | Correct; no train/eval drift |
| dev uses `rimf_train_*.h5`, test uses `rimf_test_*.h5`, both indexed by row | Correct |
| Persistence reference in the logs | **Wrong** (B4). The true bucket values are 0.938 / 0.686 / 0.416 / 0.249. |
| Per-basin Wilcoxon | **Invalid as a significance test** (p = 1e-8 between seeds of one config). Use seed-level comparisons only. |
| About 38 dev windows per basin | Per-basin NSE is noisy; report the share of basins with NSE > 0.5 and mean(max(NSE,−1)) too |
| Competition metric | **Not defined in `metadata.json`.** Confirm before the final readout. |

## 7. Stop spending compute on

- DMEL as built, and every variant of it (exp_021, 022, 024; dmel+mae; dmel+d128; the shared-informer
  rescue)
- global z-score, seq 168, no-clip, lr 5e-4 / 5e-5, batch 128, dropout 0.2, the NSE loss, ProbSparse,
  3 encoder layers, d_model 128 / 512
- channel removal
- basin embeddings (they can't work on held-out basins)
- epoch extensions (+0.01 per 10 epochs is below seed noise; EMA replaces this)
- Huber δ = 0.2
- MAE + aux (the aux gain comes from a single-epoch spike)
