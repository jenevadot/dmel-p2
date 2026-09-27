# DMEL Streamflow Forecasting — Conclusions & Next Actions

**Date:** 2026-09-26
**Scope:** ~23 experiments, 208.7 GPU/MPS hours, 508 basins, 336h history → 48h forecast.
**Constraints as run:** 1 seed per experiment (3 on baseline), 40 epochs.

---

## 1. Conclusions

### C1 — The paper's core contribution (CEEMDAN/DMEL) fails on this dataset

`exp_020__dmel` scores **0.3971** dev median NSE — *below* `baseline_persistence`
(0.5251). At the longest horizon it is 0.113 vs persistence 0.249.
`exp_023__dmel_se_threshold` is only marginally better at 0.5039, still below
persistence. Together these two runs consumed **24% of all compute spent**
(49.6 h of 208.7 h) at 4× the cost of a baseline run.

This is a real result, not an artifact. Three checks rule out the easy excuses:

- **Decomposition is correct** — CEEMDAN reconstruction verified at max relative
  error **5.56e-03**.
- **No label leakage** — the RIMF cache is `(N, 3, 336)`, i.e. 336 not 384; the
  decomposition worker reads only `f["X"]` (`src/decompose.py:638`), never `y`
  or `y_aux`. Decomposition over the target window would have produced 384.
- **The failure mode is overfitting, not under-training** — train loss falls
  0.705 → 0.234 while san_val NSE plateaus at ~0.55 from epoch 6 and then
  oscillates for 30+ epochs.

Root cause is architectural. In the default `rimf_input_mode="replace"`, each of
the 3 branches receives all 12 channels with only channel 11 swapped for its
RIMF — so every branch re-learns the same meteo→discharge mapping, on ~6.76M
parameters, with independent (unshared) weights and no branch-specific
regularization.

### C2 — Convergence: more epochs help the good runs, and will not rescue DMEL

This answers the 40-epoch question, but the answer is split by family:

- **Baseline family — headroom is real.** san_val NSE climbs steadily
  0.54 → 0.65 and is *still rising* at epoch 39. Best epochs land late
  (MAE 32/40, Huber δ=0.5 37/40).
- **DMEL — no headroom.** Plateaued by epoch 6, then noise. Additional epochs
  are wasted compute.

Do not extend the epoch budget uniformly. Extend it only for the robust-loss
family.

### C3 — Robust losses are the only reproducible gain, with a confirmed mechanism

MAE / Huber(0.5) / Huber(1.0) take **3 of the top 4 slots**. Three independent
runs moving the same direction is substantially stronger evidence than any
single top rank.

| loss | Δ devNSE vs baseline | Δ h37-48 |
|---|---|---|
| MAE | +0.043 | **+0.080** |
| Huber δ=0.5 | +0.026 | +0.053 |
| Huber δ=1.0 | +0.018 | +0.056 |
| NSE loss | −0.032 | −0.075 |

The mechanism was verified directly on `data/train.h5` (1.92M target hours):
**skew 11.2, kurtosis 386 (normal = 3), max/median = 942×. The top 1% of hours
own 60.6% of the total squared magnitude; the top 0.1% own 28.3%.**

Under MSE the gradient is effectively a flood-peak regressor. MAE/Huber
rebalance it toward the median basin — which is precisely what
median-across-basin NSE rewards. The gain is causally explained, not a
coincidence of ranking, and it concentrates at the horizons that matter.

**Caveat to state openly in the paper:** this is partly metric-shopping. Median
basin NSE rewards de-emphasizing peaks, but flood peaks are the operationally
important events. MAE is the correct default *for this metric*; a peak-flow
metric must be reported alongside it so the tradeoff is visible.

### C4 — Short horizons are saturated; all model value is at h25-48

No model beats persistence at h1-12 (persistence 0.938, best model 0.946).
Persistence collapses to 0.249 at h37-48 where the best model reaches 0.500.
Aggregate NSE is therefore ~60% a restatement of persistence and is misleading
as a headline number.

### C5 — A config bug makes every reported DMEL parameter count wrong

`build_model(cfg)` (`src/experiment.py:157`) receives TRAIN_CFG+overrides, which
never contains `n_high` — because `n_high=2` lives in `CEEMDAN_CFG`
(`src/config.py:237`), and that dict is merged into a *separate* `ceemdan_cfg`
used only by `build_splits` (`src/experiment.py:128-130`). So
`src/models/dmel.py:323` falls back to `n_high=5` while
`src/decompose.py:876` uses 2.

Confirmed empirically:

```
build_model({use_ceemdan: True})         = 15,308,341 params, hf_branches = 5
build_model({..., n_high: 2, n_low: 1})  =  6,764,338 params, hf_branches = 2
DEAD (never fed, never gradient)         =  8,544,003 = 56% of the checkpoint
```

15,308,341 matches the `n_params` reported in `exp_020` and `exp_023` exactly.

**Scoping note:** `forward` loops over `x_high` (length 2), so the 3 extra
branches are **inert** — no input, no gradient. This bug therefore does **not**
explain DMEL's poor score (C1 does). What it corrupts is every reported DMEL
parameter count, any params-vs-performance or efficiency claim, and 56% of
checkpoint size.

### C6 — `dev` is contaminated, and it inverts the channel-ablation conclusion

`dev` is **all 508 basins** — the 447 trained basins ∪ the 61 san_val basins —
and **78.8% of dev windows overlap a training window** (`src/splits.py:60`).
Only `san_val` (61 unseen basins) is a clean held-out split. The comment at
`src/experiment.py:301` calling dev "never touched during training" is
misleading.

Usually this is mild optimism (baseline: dev 0.6419, test 0.6316, san_val
0.6186 — test sits between, so dev runs ~+0.01 optimistic). **But it is severe
for the channel ablations**, which carry the two largest gaps in the entire set:

| exp | dev rank | devNSE | san_val | gap |
|---|---|---|---|---|
| exp_004__precip_discharge | #2 | 0.6720 | 0.5002 | **+0.172** |
| exp_003__discharge_only | #7 | 0.6555 | 0.5141 | **+0.141** |
| exp_001__baseline | — | 0.6419 | 0.6186 | +0.023 |

On the clean split these are the two *worst* non-DMEL runs.
**Do not conclude the 11 meteo channels are unnecessary — the clean split says
the opposite.**

### C7 — Seed noise sets the significance floor

The only 3-seed cell (baseline) gives **σ = 0.018 devNSE, σ = 0.041 on h37-48**.
Any devNSE difference below ~0.036 (2σ) is indistinguishable at n=1. MAE's
+0.043 is **+2.4σ** — suggestive, not yet significant.

### C8 — Reporting artifacts to clean up

- `mean_nse` values of −24.8 / −81 / −128 are a pure artifact: 7 of 508 basins
  (flat baseflow, tiny `ss_tot` that still clears `_EPS=1e-10`) contribute
  −25.36 of the −24.84 mean. Median is the correct headline.
- `exp_034__no_distil` is **incomplete but recoverable** — it trained 38 epochs
  and peaked at san_val **0.6250** (epoch 28); it only missed the dev-eval step.
- `test.h5` has been scored **exactly once** (`exp_001__baseline`), so it
  remains essentially clean for one final comparison.

---

## 2. Next actions

### Tier 0 — do before spending any more compute (hours, no GPU)

- [ ] **Fix the `n_high` config path.** Merge `CEEMDAN_CFG` into the cfg reaching
      `build_model` (`src/experiment.py:157`); align the default at
      `src/models/dmel.py:323` from 5 → 2 to match `src/decompose.py:876` and
      `src/config.py:237`.
- [ ] **Add a loud failure.** Assert `len(hf_branches) == len(x_high)` at
      `src/models/dmel.py:270` so a branch-count mismatch raises instead of
      silently allocating dead weights.
- [ ] **Switch the headline split to san_val**, dev secondary with the 78.8%
      overlap stated. Wire the already-implemented but never-called
      `measure_dev_contamination()` (`src/splits.py:574-622`) into
      `summary.json`, and correct the false "never touched" comment
      (`src/experiment.py:301`).
- [ ] **Drop `mean_nse`** from reporting, or tighten the variance guard at
      `src/evaluate.py:148-152`.
- [ ] **Recover `exp_034__no_distil` for free** — run only the dev-eval step
      against the existing epoch-28 checkpoint. No retraining.

### Tier 1 — highest-value compute (~26 GPU-h)

- [ ] **Seed-replicate MAE and Huber(0.5): 2 extra seeds each** (4 runs ×
      ~750 min). This is the single best use of the next compute block — it
      converts the paper's main positive claim from n=1 at +2.4σ into a 3-seed
      result with error bars, and it is the only candidate with a confirmed
      mechanism (C3).
- [ ] **Prefer Huber(0.5) as the production default if the two tie.** It has the
      best generalization signature in the set (san_val 0.669, gap −0.003);
      MAE's gap is +0.044.

### Tier 2 — run if compute allows

- [ ] **Extend the winning loss to 80–100 epochs, 1 seed (~25 GPU-h).**
      Justified specifically by C2; keep early stopping on san_val. Confirm
      san_val is still rising past epoch 40 before committing to a full
      100-epoch final run.
- [ ] **One DMEL rescue attempt, then stop (~26 GPU-h).** Single run with
      `shared_informer=True` (already implemented at
      `src/models/dmel.py:201-218`, never used by any experiment) +
      `rimf_input_mode="univariate"` + dropout 0.2. This attacks the C1
      diagnosis directly: weight sharing and univariate branches cut capacity
      and stop branches re-learning one shared mapping.
      **If it still lands below persistence, report DMEL as a negative result**
      — legitimate and publishable on 508 basins with a verified-correct,
      leakage-free implementation.

### Do not spend more compute on

- **Architecture sweep** — `d_model` 128/512, `enc_layers`, ProbSparse, batch
  size, LR, `seq168`, `no_clip`. Every variant lands within ~2σ of baseline or
  below; `d_model=512` costs 2.3× for −0.004. The architecture is not the
  bottleneck, the loss function is. `exp_013__lr_5e4` and `exp_008__seq168` are
  clearly worse — settled, no replication needed.
- **`exp_003` / `exp_004` channel ablation as-is** — contaminated (C6). If the
  "do the meteo channels earn their keep" question matters, re-run scored on
  san_val.
- **`exp_006` / `exp_007` basin embeddings** — structurally invalid under
  held-out-basin selection: san_val has 0 train-basin overlap, so embeddings for
  unseen basins are scored as random initialization. Needs a protocol change,
  not a rerun.

### Reporting changes for the paper

- [ ] Headline = **san_val** median NSE; dev secondary, with the 78.8% window
      overlap stated explicitly.
- [ ] Lead with **per-horizon** results; note that persistence ties at h1-12
      and that all model value is at h25-48 (C4).
- [ ] Report the DMEL parameter count as **6.76M**, not 15.3M (C5).
- [ ] Add a **peak-flow metric** beside median NSE so the MAE tradeoff is
      explicit (C3).
- [ ] Touch `test.h5` **once**, at the end, for the final chosen model only.

---

## 3. Verification

- **Config fix:** `build_model({"use_ceemdan": True})` reports **6,764,338**
  params and `len(hf_branches) == 2`; assert it equals the explicit
  `n_high=2, n_low=1` build.
- **Contamination reporting:** a fresh `summary.json` carries the contamination
  figure and labels san_val as the headline split.
- **exp_034 recovery:** `metrics_dev.json` + `summary.json` appear, with
  san_val NSE ≈ 0.6250 matching `best_metrics.json` epoch 28.
- **Seed replication:** report mean ± sd over 3 seeds for MAE and Huber(0.5).
  The claim holds only if the gain exceeds ~0.036 devNSE (2σ) **and** reproduces
  on san_val, not just dev.
- **Extended epochs:** confirm san_val NSE is still rising past epoch 40 before
  committing to the 100-epoch run.
