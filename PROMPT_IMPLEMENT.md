# TASK: Implement 4 validated improvements to the DMEL streamflow forecasting pipeline

You are working in `/Users/jent/UTEC/ciclo5/deepLearning/paper2`.

This is a 508-basin hourly streamflow forecasting project (336h history → 48h
forecast) replicating the DMEL architecture (Informer + LSTM dual-branch with
CEEMDAN decomposition). The codebase is ~6,100 lines and mature: split leakage
has been solved, per-horizon metrics exist, config carries measured
justifications.

A detailed review has already been completed. **Nine improvement proposals were
evaluated; five were rejected or deferred with reasons.** Your job is to
implement only the four that survived, in the stated order.

**Do not implement anything in the "DO NOT IMPLEMENT" section at the bottom.**
Each has a specific reason for rejection, and at least one would silently break
the competition submission.

---

## CONTEXT YOU NEED

### Data facts (measured, do not re-derive)

```
train.h5 : 272,142 windows. keys = X, y, y_aux, split, basin_id
           X shape (336, 12), channel 11 = specific_discharge = target
           y shape (48,)
test.h5  : 27,983 windows. keys = X, basin_id ONLY  ← no y, no y_aux

split=0 → 254,000 windows (train pool)
split=1 →  18,142 windows (dev — the reported/scored split)

Split partition (cached, basin-disjoint, leak-free):
  train   : 223,500 windows across 447 basins
  san_val :  30,500 windows across  61 basins
  overlap : 0 basins          ← THIS MATTERS, see Task 3
```

### Metric facts

- Primary metric: **median NSE across basins**, computed on **raw** (denormalised) discharge.
- NSE is effectively a **peak-flow metric**: top 1% of timesteps hold 22–85% of the NSE denominator variance depending on basin.
- Baselines on dev: persistence **0.5251**, decay 0.4889, climatology 0.1546, mean −0.0071.
- Targets are per-basin z-scored. On z-scored targets: median |z| = 0.311, p99 = 3.89, **max = 82.2**.

### Current training state

A 40-epoch `exp_001__baseline` run is in progress. Measured behaviour:

```
ep  train_loss  sanval_nse   grad_norm
 0     0.62785    0.48855      2.822
 1     0.53566    0.52022      2.571
 2     0.52182    0.52705      2.437
 3     0.50918    0.52491      2.264
 4     0.49973    0.54146      2.201
 5     0.49028    0.55337      2.110
```

Model: 2,848,001 params. ~11 min/epoch, 189 ms/step, batch 64, MPS.

---

## TASK 1 — Fix gradient clipping (do this FIRST, it distorts everything else)

### The problem

`clip_grad = 1.0` in `src/config.py`, but measured `grad_norm` never drops
below 2.0. **The clipper fires on 100% of steps in every run.**

```
epoch 0: grad_norm 2.822 → update rescaled by 0.354x
epoch 5: grad_norm 2.110 → update rescaled by 0.474x
```

Consequences:
1. **Effective LR is ~0.35–0.47x the configured value** and drifts upward as
   gradients shrink. The nominal `1e-4` is really running near `4e-5`.
2. **The cosine schedule is corrupted.** What the optimiser sees is
   `cosine(step) x clip_factor(step)` — two entangled schedules.
3. **The LR ablations (`exp_013__lr_5e4`, `exp_014__lr_5e5`) cannot produce
   valid conclusions.** At higher LR, gradients grow, clipping bites harder,
   and the effective LR difference is compressed far below the nominal 10x.
   Those experiments would measure clipping behaviour, not learning rate.

Gradients are **stable and monotonically decreasing** (2.82 → 2.11). There is
no explosion to protect against.

### What to do

Change the default to `clip_grad = 5.0` in `TRAIN_CFG` — comfortably above the
observed max (2.82) so the clipper becomes a genuine runaway guard rather than
a per-step rescaler.

**This changes the effective learning rate, so it must be validated as an
ablation, not applied silently.** Add two grid entries:

```python
dict(
    name        = "exp_016__clip5",
    priority    = "critical",
    description = "[T6] clip_grad=5.0 — clipper becomes a guard, not a "
                  "per-step rescaler. Measured grad_norm 2.1-2.8 means "
                  "clip=1.0 fires every step at 0.35-0.47x.",
    overrides   = {"clip_grad": 5.0},
),
dict(
    name        = "exp_017__noclip",
    priority    = "critical",
    description = "[T6] clip_grad=0 — no clipping at all. grad_norm is "
                  "still logged (sampled every 50th step) so instability "
                  "stays visible.",
    overrides   = {"clip_grad": 0.0},
),
```

Update the `clip_grad` comment in `config.py` to record the measurement and
why 5.0 was chosen. Keep the existing `exp_001__baseline` (clip=1.0) as the
comparison point — **do not delete or overwrite it.**

### Acceptance

- `clip_grad` default is 5.0 with a comment citing measured grad_norm 2.1–2.8
- Two new critical-priority grid entries exist
- `python run_ablation.py --dry_run` lists them
- `mean_grad_norm` still appears in `summary.json`

---

## TASK 2 — Add asymmetric MSE loss

### Why

MSE is symmetric, but the costs are not. Under-predicting a flood peak is
operationally worse than over-predicting it, and because NSE is dominated by
peak errors, systematic peak under-prediction directly depresses the score.

This is the **only** loss-shape change that survived review. (A differentiable
KGE loss was rejected: KGE needs per-basin correlation `r`, variability `α`,
and bias `β`, but a batch of 64 drawn from 508 basins gives most basins 0–1
samples, so `r` is undefined without basin-grouped batch sampling — a data
pipeline restructure, not a loss change.)

### What to do

Add to `src/train.py`, registered in `build_loss`:

```python
class AsymmetricMSELoss(nn.Module):
    """
    MSE with a heavier penalty on UNDER-prediction.

    loss = mean( w * (y - yhat)^2 ),  w = under_weight if yhat < y else 1.0

    Why asymmetric here
    ───────────────────
    NSE is computed on raw discharge and its denominator is raw variance, so
    it is dominated by the largest flows (top 1% of timesteps hold 22-85% of
    that variance depending on basin). A model that systematically under-shoots
    peaks loses NSE fast. Plain MSE has no reason to prefer over- to
    under-prediction; this does.

    under_weight = 1.0 recovers plain MSE exactly, so the ablation
    {1.0, 2.0, 3.0} is a clean test of whether the asymmetry helps.
    """
    def __init__(self, under_weight: float = 2.0):
        super().__init__()
        self.under_weight = under_weight

    def forward(self, y_pred, y_true):
        err = y_true - y_pred
        w = torch.where(err > 0,                       # under-prediction
                        torch.full_like(err, self.under_weight),
                        torch.ones_like(err))
        return (w * err.pow(2)).mean()
```

Wire `"asym_mse"` into `build_loss`, add `asym_under_weight = 2.0` to
`TRAIN_CFG`, add `--asym_under_weight` to `run_experiment.py`.

Add grid entries:

```python
dict(
    name        = "exp_018__asym_mse_w2",
    priority    = "high",
    description = "[T3] Asymmetric MSE, under-prediction weighted 2x. NSE "
                  "is peak-dominated, so systematic peak under-shoot is "
                  "costly; plain MSE is indifferent to error sign.",
    overrides   = {"loss": "asym_mse", "asym_under_weight": 2.0},
),
dict(
    name        = "exp_019__asym_mse_w3",
    priority    = "high",
    description = "[T3] Asymmetric MSE, under-prediction weighted 3x.",
    overrides   = {"loss": "asym_mse", "asym_under_weight": 3.0},
),
```

### Acceptance

- `AsymmetricMSELoss` exists with the docstring rationale
- `under_weight=1.0` reproduces `nn.MSELoss()` to within 1e-6 — **write a test asserting this**
- Both grid entries appear in `--dry_run`
- Loss takes only `(pred, true)`, so `needs_basin_ids()` returns False and no `forward_batch` change is required

---

## TASK 3 — Fix the evaluation protocol for basin-conditioned models

### The problem

`src/experiment.py` already evaluates **both** splits and writes both to
`summary.json`. The bug is narrower than it looks: **the selection and ranking
logic is hardcoded to san_val, and san_val cannot evaluate basin embeddings.**

```
train   : 447 basins
san_val :  61 basins
overlap :   0 basins        ← san_val basin embeddings NEVER receive gradient
```

Any basin-conditioned model is selected and ranked using **randomly
initialised embedding rows** for every san_val basin.

But at submission time:

```
train.h5 basins: 508    test.h5 basins: 508    test basins not in train: 0
```

**All 508 test basins have trained embeddings.** So san_val systematically
under-rates basin-conditioned models relative to their true submission
performance. `exp_006__basin_emb` and `exp_007__aux_basin_emb` are currently
guaranteed to look worse than they are.

### What to do

Make the selection metric **explicit and per-experiment**, rather than an
invisible global assumption.

1. Add to `TRAIN_CFG`:
   ```python
   # Which split drives early stopping and checkpoint selection.
   #   "san_val" — held-out BASINS. Leak-free. Correct default.
   #   "dev"     — held-out TIME in known basins. Required for
   #               basin-conditioned models, whose san_val embeddings
   #               are never trained (0 basin overlap). Using dev here
   #               means dev is no longer an unbiased report for that
   #               experiment — summary.json must record this.
   selection_split = "san_val",
   ```

2. In `experiment.py`, pick the early-stopping loader from this setting instead
   of hardcoding san_val. Write the **actual** value into
   `summary.json["selection_split"]` (it currently writes the literal string
   `"san_val"` unconditionally — that becomes wrong the moment this is
   configurable).

3. When `selection_split == "dev"`, add to `summary.json`:
   ```python
   "report_is_biased": True,
   "bias_reason": "selection and reporting both used dev"
   ```

4. Change the two basin-embedding experiments to `selection_split: "dev"`, and
   extend their descriptions to state why.

5. In `compare_results.py`, flag any experiment whose `summary.json` has
   `report_is_biased: True` with a marker in the scoreboard, so a biased number
   is never silently compared against unbiased ones.

### Acceptance

- `selection_split` is configurable and defaults to `"san_val"`
- `summary.json` records the real value, not a constant
- `exp_006` / `exp_007` use `dev` and are flagged biased
- `compare_results.py` visibly marks biased rows
- Non-basin-embedding experiments are **unchanged** — verify `exp_001` still reports `selection_split: "san_val"`

---

## TASK 4 — Profile the training step (measurement only, no optimisation)

### Why

189 ms/step for a 2.85M-param model at batch 64 is slow. Before any
performance work, establish *where* the time goes. A proposal to add AMP was
**deferred** because the memory premise does not hold:

```
model params  10.9 MB  |  AdamW states 21.7 MB  |  grads 10.9 MB
data/batch     0.98 MB |  total ~43 MB on a 24 GB machine
attention = "full" → SDPA → LxL never materialised → peak O(L), not O(L^2)
```

Memory is not binding. AMP accelerates compute; if the bottleneck is I/O it
changes nothing. The prime suspect is in the config already:

```python
num_workers = 0   # sandbox blocks torch_shm_manager
```

meaning HDF5 reads happen serially in the training process while the GPU idles.

### What to do

Write `src/profile_step.py` that runs ~100 steps and reports the split between:

- **data loading** (time blocked waiting on the loader)
- **forward**
- **backward**
- **optimizer step**

Use `time.perf_counter()` with `torch.mps.synchronize()` / `torch.cuda.synchronize()`
before each boundary. Report mean ms and percentage per phase.

Then test one hypothesis: **is the loader the bottleneck?** Time an epoch of
pure iteration with no model at all:

```python
for batch in train_loader:
    pass
```

If that alone approaches 11 minutes, the pipeline is data-bound and no compute
optimisation will help.

### Acceptance

- `src/profile_step.py` runs standalone and prints a per-phase breakdown
- It reports the loader-only epoch time
- It prints an explicit verdict line: `BOTTLENECK: data | compute | balanced`
- **Do not implement AMP, gradient accumulation, or loader changes in this
  task.** Measure only. The fix follows from the measurement.

---

## GENERAL REQUIREMENTS

- **Do not delete or overwrite `experiments/exp_001__baseline/`** — it is the comparison baseline and may still be running. Check before touching `experiments/`.
- Follow existing code style: measured justifications in comments, docstrings explaining *why* not just *what*.
- Every new config value needs a comment saying what it does and why the default was chosen.
- After each task, verify: `python run_ablation.py --dry_run` succeeds and lists the expected experiments.
- Run existing self-tests where they exist (`python src/early_stopping.py`, `python src/evaluate.py`).
- Torch lives in `.venv`. Activate with `source .venv/bin/activate`.

---

## DO NOT IMPLEMENT — rejected with reasons

### ❌ Feeding `y_aux` to the LSTM branch (or any branch) as an INPUT

**This would break the submission.**

```
metadata.json: y_aux_role = "future_supervision_only_not_inference_inputs"
test.h5 keys : ['X', 'basin_id']        ← y_aux DOES NOT EXIST
```

The current code rejects this deliberately (`src/train.py`, `forward_batch`):

```python
# NOTE: x_dec is deliberately NOT built from y_aux. y_aux does not exist
# at test time, so using it as a decoder input is target leakage.
```

A model trained this way posts a strong dev NSE and collapses at submission
with **no local signal warning you**. The legitimate use — `y_aux` as an
auxiliary *target* with the head discarded at inference — is already
implemented as `aux_task`. Leave it alone.

### ❌ Log / Box-Cox transform of the target

Distributionally it works (skew 9.9 → 0.7, max |z| 82 → 16). But **NSE is
computed on raw discharge and is peak-dominated** (top 1% of timesteps hold
22–85% of denominator variance). Log-transform deliberately down-weights
exactly the samples NSE rewards. You would stabilise the loss curve while
optimising a different objective than the one being scored.

Also: **3.26% of targets are exactly zero**, making `ε` a load-bearing
hyperparameter.

If revisited later, `sqrt` is the defensible middle ground (skew 10.7 → 3.35,
preserves peak ordering) — and it should be run as an ablation with the
expectation of *lower* NSE, not adopted as a default.

### ❌ FiLM / cross-attention basin embedding

The technique is sound, but it **cannot be evaluated correctly until Task 3
lands**. FiLM conditions more of the network on the basin vector — on san_val,
where those vectors are random noise, that makes the handicap *worse*, not
better. Revisit only after Task 3, and only if plain concat embedding shows
promise on the corrected protocol.

### ❌ AMP / mixed precision

Deferred pending Task 4. The stated premise (attention-matrix memory pressure)
is false: total footprint ~43 MB on 24 GB, and SDPA never materialises the
L×L matrix. Also, AMP requires `scaler.unscale_()` before
`clip_grad_norm_` — adding it before Task 1 would compound the clipping bug.

### ❌ Gated / attention-based combiner

Best architectural idea in the set and hydrologically well-motivated (HF
matters during storms, LF during recession). But it is blocked on two unproven
prerequisites: CEEMDAN mode has never run end-to-end (`exp_020` pending), and
a prior review found the combiner ablations were **silent no-ops** because
`build_combiner` was unreachable. Upgrading a combiner that may not execute is
premature. Revisit after `exp_020` demonstrates decomposition helps.

### ❌ SWA (Stochastic Weight Averaging)

Two problems. "Average the top 3–5 checkpoints" is checkpoint ensembling, not
SWA — real SWA averages along a low-LR trajectory. And `DistilLayer` uses
`nn.BatchNorm1d` (`src/models/informer.py:266`), so SWA requires an
`update_bn()` pass over the training set or the averaged weights carry wrong
running statistics. Modest gain, non-trivial cost, low priority.

### ❌ Offline CEEMDAN precompute / hybrid decomposition

**Already implemented.** `src/decompose.py` (1,010 lines) with
`multiprocessing.Pool`, cache at `data/rimf_train_110b9c26af82.h5`, driver
`run_ceemdan.sh`. Config already decomposes only channel 11 and uses
`rimf_input_mode="replace"` to keep the 11 meteo channels raw — exactly the
recommended hybrid.

---

## ORDER AND RATIONALE

1. **Task 1 (clipping)** — first, because it distorts every measurement below it.
2. **Task 2 (asymmetric MSE)** — cheap, isolated, addresses a real peak-bias problem.
3. **Task 3 (evaluation protocol)** — unblocks basin-conditioned experiments and prevents a predictably wrong conclusion.
4. **Task 4 (profiling)** — measurement only; decides whether any performance work is worth doing.

Tasks 1–3 change results. Task 4 changes nothing and only produces information.
