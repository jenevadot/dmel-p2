# DMEL — Streamflow Forecasting

Reimplementation and ablation study of the **Dual-Modal Ensemble Learning** (DMEL) architecture for multi-step streamflow forecasting, applied to a 508-basin hourly discharge dataset.

> Paper: Wang et al., *DMEL: A novel dual-modal ensemble learning architecture for multi-step runoff prediction*, Applied Soft Computing 192 (2026)

---

## Project Structure

```
paper2/
├── data/
│   ├── train.h5          ← 272,142 samples (split=0 train + split=1 dev)
│   ├── test.h5           ← 27,983 samples (no targets — final submission)
│   └── metadata.json     ← channel names, shapes, split codes
│
├── src/
│   ├── config.py         ← all hyperparameters, seeds, device detection
│   ├── dataset.py        ← HDF5 loading, per-basin normalizer, DataLoaders
│   ├── train.py          ← optimizer, scheduler, loss, training loop
│   ├── evaluate.py       ← NSE/KGE/RMSE/MAE per basin, EvalResult, Wilcoxon
│   ├── early_stopping.py ← EarlyStopping callback, CheckpointManager
│   ├── experiment.py     ← single-experiment orchestrator
│   └── models/
│       ├── informer.py   ← ProbSparse Informer (high-frequency branch)
│       ├── lstm_branch.py← LSTM (low-frequency branch)
│       └── dmel.py       ← DMEL assembler + build_model() factory
│
├── run_experiment.py     ← CLI: run one experiment with flags
├── run_ablation.py       ← CLI: run the full ablation grid
├── compare_results.py    ← CLI: print scoreboard from all experiments
├── requirements.txt
└── experiments/          ← auto-created; one folder per run
    └── exp_001__baseline/
        ├── config.json
        ├── best_model.pt
        ├── metrics_dev.json
        ├── metrics_sanval.json
        ├── history.csv
        └── log.txt
```

---

## Quick-start

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

### 2. Create virtual environment and install dependencies

```bash
cd /path/to/paper2

# Create venv with Python 3.11 (stable for torch + h5py)
uv venv --python 3.11 .venv

# Activate
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install PyTorch — choose ONE of the following:

# Apple M4 (MPS)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Linux NVIDIA GPU (CUDA 12.1)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only fallback
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
uv pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('MPS :', torch.backends.mps.is_available())
print('CUDA:', torch.cuda.is_available())
"
```

### 4. Verify data and infrastructure

```bash
# Test config + device detection
python src/config.py

# Test data pipeline (splits, normalizer, loaders)
python src/dataset.py

# Test evaluation metrics
python src/evaluate.py

# Test early stopping
python src/early_stopping.py

# Test model components
python src/models/informer.py
python src/models/lstm_branch.py
python src/models/dmel.py
```

---

## Running Experiments

### Run one experiment (CLI flags)

```bash
# Baseline — paper defaults, single Informer, per-basin z-score
python run_experiment.py --name exp_001__baseline

# Key architectural flags
python run_experiment.py --name exp_006__y_aux     --use_y_aux
python run_experiment.py --name exp_007__basin_emb --use_basin_emb
python run_experiment.py --name exp_012__nse_loss  --loss nse
python run_experiment.py --name exp_009__seq168    --seq_len 168

# Combine flags
python run_experiment.py \
  --name exp_008__y_aux_basin_emb \
  --use_y_aux --use_basin_emb --loss nse

# Different seed (for multi-seed validation)
python run_experiment.py --name exp_001__baseline_s123 --seed 123
python run_experiment.py --name exp_001__baseline_s777 --seed 777

# Force re-run an existing experiment
python run_experiment.py --name exp_001__baseline --force
```

### All available flags

| Flag | Default | Plan ID | What it tests |
|------|---------|---------|---------------|
| `--norm_strategy` | `per_basin_zscore` | T8 | `global_zscore` \| `none` |
| `--use_y_aux` | off | E5/D2 | Feed future meteo to decoder |
| `--use_basin_emb` | off | T10 | Learned basin identity embedding |
| `--ensemble_method` | `sum` | E1 | `linear` \| `mlp` combiner |
| `--combiner_lr_mult` | `1.0` | E1b | lr multiplier for combiner params only (own group, no decay) |
| `--use_ceemdan` | off | — | Enable CEEMDAN decomposition |
| `--shared_informer` | off | E3 | One Informer across all HF branches |
| `--seq_len` | `336` | I8 | `168` = 7 days instead of 14 |
| `--loss` | `mse` | T3 | `mae` \| `huber` \| `nse` |
| `--optimizer` | `adamw` | T1a | `adam` |
| `--lr` | `1e-4` | T1 | `5e-4`, `5e-5` |
| `--scheduler` | `warmup_cosine` | T2 | `cosine` \| `sgdr` \| `plateau` \| `none` |
| `--dropout` | `0.1` | I6/L3 | `0.2`, `0.3` |
| `--lstm_bidir` | off | L4 | Bidirectional LSTM |
| `--d_model` | `256` | I1 | `128`, `512` |
| `--enc_layers` | `2` | I3 | `3`, `4` |
| `--batch_size` | `64` | T4 | `32`, `128` |
| `--epochs` | `100` | T5 | Any integer |
| `--seed` | `42` | — | Reproducibility |
| `--force` | off | — | Re-run even if complete |

---

## Running the Ablation Grid

```bash
# Dry run — print grid without executing anything
python run_ablation.py --dry_run

# Run only CRITICAL experiments first (fastest path to baseline)
python run_ablation.py --priority critical

# Run CRITICAL + HIGH (main ablation study)
python run_ablation.py --priority critical high

# Run everything
python run_ablation.py

# Run one specific experiment by name
python run_ablation.py --only exp_006__y_aux

# Run with multiple seeds (3-seed confirmation phase)
python run_ablation.py --priority critical high --seeds 42 123 777

# Force re-run all completed experiments
python run_ablation.py --force
```

---

## Comparing Results

```bash
# Full scoreboard (ranked by median NSE on dev split)
python compare_results.py

# Show KGE alongside NSE
python compare_results.py --kge

# Show generalisation gap (dev - san_val) and mean gradient norm
python compare_results.py --gap

# Wilcoxon statistical test vs baseline
python compare_results.py --wilcoxon --baseline exp_001__baseline

# Filter to a subset
python compare_results.py --filter exp_00
```

Example output:
```
  Rank  Experiment                                   med NSE  mean NSE   >0.7%   ep
  ────  ──────────────────────────────────────────  ────────  ─────────  ──────  ───
     1  exp_008__y_aux_basin_emb                     0.8821    0.8103   74.2%    38
     2  exp_006__y_aux                               0.8650    0.7941   71.5%    42
     3  exp_007__basin_emb                           0.8612    0.7888   70.1%    45
     4  exp_001__baseline                            0.8401    0.7712   66.3%    51  ← baseline
  ...
    29  exp_002__global_norm                         0.6102    0.5014   38.1%    30
```

---

## Experiment Output Files

Every experiment writes to `experiments/<name>/`:

| File | Contents |
|------|----------|
| `summary.json` | **Run audit trail** — selection/report split, generalisation gap, gradient health, resolved config |
| `config.json` | Exact hyperparameters used (fully reproducible) |
| `best_model.pt` | Model weights at best san\_val NSE epoch |
| `best_metrics.json` | Epoch and NSE that triggered checkpoint |
| `norm.npz` | Per-basin normalisation stats (train-split only) — required for inference |
| `metrics_dev.json` | Full EvalResult on dev split — the ablation score |
| `metrics_sanval.json` | Full EvalResult on san\_val at best epoch |
| `history.csv` | `epoch, train_loss, sanval_nse, lr, grad_norm, elapsed_s` per epoch |
| `log.txt` | Full training stdout |

### `summary.json` — the fields that matter

```json
{
  "selection_split"   : "san_val",   // what early stopping read
  "report_split"      : "dev",       // where the headline came from
  "dev_median_nse"    : 0.8401,      // the reportable number
  "sanval_median_nse" : 0.8455,      // what selection saw
  "sanval_to_dev_gap" : -0.0054,     // transfer check
  "mean_grad_norm"    : 0.132,       // vs clip_grad=1.0 -> clipper never fires
  "patience_requested": 10,
  "patience_used"     : 50,          // auto-relaxed for cosine
  "early_stopped"     : false
}
```

**`selection_split` vs `report_split`** must always differ. If they were ever
the same, the reported metric would be optimistically biased by construction.

**`sanval_to_dev_gap`** is the selection-overfitting detector. Strongly negative
means the checkpoint was chosen on san\_val noise and does not transfer.

**`mean_grad_norm`** vs `clip_grad` tells you whether clipping is active:
far below → the clipper is a no-op you are paying for; close to it → the clipper
fires every step and is silently changing the effective learning rate.

View these across all runs with:
```bash
python compare_results.py --gap
```

---

## Data Splits

| Split | Source | Samples | Role |
|-------|--------|---------|------|
| `train` | `train.h5` split=0, 88% | 223,520 | Gradient updates |
| `san_val` | `train.h5` split=0, 12% | 30,480 | Early stopping signal |
| `dev` | `train.h5` split=1 | 18,142 | Ablation evaluation (untouched during training) |
| `test` | `test.h5` | 27,983 | Final submission (no targets) |

Split is stratified by basin with `seed=42`. Per-basin z-score normalizer is fitted on `train` only.

---

## Evaluation Metrics

| Metric | Scale-invariant | Used for ranking |
|--------|----------------|-----------------|
| **median NSE** | ✅ | **Primary — ablation ranking** |
| mean NSE | ✅ | Outlier sensitivity check |
| % NSE > 0.7 | ✅ | How many basins reach "good" skill |
| median KGE | ✅ | Secondary diagnostic (timing / variability / bias) |
| median RMSE | ❌ | Supplementary |

Early stopping monitors **median NSE on san\_val**, patience=10 epochs.

---

## Recommended Workflow

```
Phase 1 — Foundation (1 seed = 42)
  python run_ablation.py --priority critical
  python compare_results.py
  → identify which critical choices matter most

Phase 2 — Architecture (1 seed = 42)
  python run_ablation.py --priority high
  python compare_results.py --kge
  → identify top-5 configurations

Phase 3 — Confirmation (3 seeds = 42, 123, 777)
  python run_ablation.py --only <top_exp> --seeds 42 123 777
  python compare_results.py --wilcoxon
  → confirm improvements are not seed-luck (p < 0.05)

Phase 4 — Final (5 seeds, best config)
  python run_ablation.py --only <best_exp> --seeds 42 123 456 777 999
  → report mean ± std

Phase 5 — Submission
  python predict_test.py --exp <best_exp>   # (to be implemented)
```

---

## Inspecting results

`results.py` is the single entry point for looking at what ran.

```bash
./results.py                    # scoreboard, ranked by dev median NSE
./results.py --horizon          # per-lead-time NSE (h+1-12 ... h+37-48)
./results.py --gap              # split-discipline + gradient-health audit
./results.py --list             # inventory: done / running / stale / corrupt
./results.py --show exp_001__baseline   # everything about one run
./results.py --watch            # live refresh every 30s
./results.py --wilcoxon         # significance vs the persistence baseline
./results.py --csv out.csv      # export
```

**Every model is ranked against persistence (median NSE 0.5251), not zero.**
Measured baselines on dev:

| baseline | median NSE | h+1-12 | h+37-48 |
|---|---|---|---|
| persistence (last value held flat) | **0.5251** | 0.9382 | 0.2495 |
| persistence decaying to basin mean | 0.4889 | 0.9136 | 0.1940 |
| climatology (hourly profile) | 0.1546 | 0.2678 | 0.1290 |
| basin mean | -0.0071 | -0.0097 | -0.0101 |

Discharge is so autocorrelated that persistence is near-perfect at h+1 and
collapses by h+48. A single aggregate median NSE is therefore dominated by the
easy early hours — always read `--horizon` before concluding a model is good.

Regenerate the baselines with `python src/baselines.py`.

---

## Running experiments

```bash
python run_ablation.py --validate            # check the grid for no-op entries
python run_ablation.py --dry_run             # list without running
python run_ablation.py --only exp_001__baseline
./run_queue.sh exp_001__baseline exp_005__aux_task   # sequential, restartable
```

`--validate` is a regression test, not a convenience: ten entries in the
original grid measured nothing (byte-identical to the baseline, or setting
flags no code path read). The grid now refuses to run if any entry's overrides
are unknown or all equal their defaults.

Measured cost: **~11 min/epoch** (223,500 windows, batch 64, `num_workers=0`
because the sandbox blocks torch's shared-memory manager). `epochs` defaults to
12, not the paper's 100, which would be ~18 h per experiment.

### CEEMDAN / MRS (the paper's DMEL branch)

The dual-branch experiments need a precomputed RIMF cache:

```bash
pkill -f run_queue.sh          # free the cores first
./run_ceemdan.sh 12            # ~17 h at 12 procs on an idle machine
./run_queue.sh exp_020__dmel exp_021__dmel_mlp_combiner \
               exp_023__dmel_se_threshold exp_024__dmel_univariate
```

The cache is resumable — each 256-row block sets its `done` flag last and
flushes — so stopping and restarting is safe.

---

## Known caveats

**dev is contaminated and it cannot be fixed.** 78.8% of dev windows overlap a
training window (measured, 6-gram stride-1). That leak lives in the
competition's own `split=1`, which is randomly interleaved rather than a
temporal tail. Absolute dev NSE is therefore optimistic; it remains valid for
*ranking*, since every model pays the same penalty.

**san_val measures something different from dev.** It is a held-out *basin*
split (generalisation to unseen basins); dev is unseen time in known basins.
A within-basin split was tried first and leaked 97% of its windows. See
`src/splits.py` for the derivation and the two failed attempts.

Full record of every fix and its measured evidence: **`IMPROVEMENTS.md`**.

---

## Hardware Notes

| Machine | Device | Recommended batch | Notes |
|---------|--------|-------------------|-------|
| Apple M4 | MPS | 64 | `pin_memory` auto-disabled |
| Linux + NVIDIA 8GB VRAM | CUDA | 64–128 | Scale lr ∝ √(B/64) if B > 64 |
| CPU fallback | CPU | 32 | Use for debugging only |

Device is auto-detected: CUDA → MPS → CPU.
