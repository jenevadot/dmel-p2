"""
config.py
─────────
Central configuration: seeds, device detection, paths, and the
hyperparameter registry that all other modules import.

Supports:
  - Apple M4 (MPS backend)
  - NVIDIA GPU  (CUDA backend)
  - CPU fallback
"""

import os
import random
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 0. Project paths
# ─────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent   # paper2/
DATA_DIR  = ROOT / "data"
SRC_DIR   = ROOT / "src"
EXP_DIR   = ROOT / "experiments"
EXP_DIR.mkdir(exist_ok=True)

TRAIN_H5  = DATA_DIR / "train.h5"
TEST_H5   = DATA_DIR / "test.h5"
META_JSON = DATA_DIR / "metadata.json"

# Ground-truth targets for test.h5, shipped separately from the inputs.
#
# test.h5 holds only X and basin_id; this CSV supplies the 48h targets that
# would otherwise be missing. Layout: Id,q_01..q_48 with Id == the zero-based
# row index in test.h5 (metadata.json: id_rule), so the join is positional.
#
# Verified raw mm/h, NOT z-scored (metadata.json: normalization_applied=false):
# CSV median 0.0245 vs train.h5 y median 0.0202 — same scale. Continuity check
# against the history channel gives corr 0.996 between X[:,-1,11] and q_01,
# median |gap| 1.05e-4 mm/h, which is what confirms the row alignment.
#
# TEST IS A FINAL READ-OUT, NOT A SELECTION SPLIT. Nothing in training or
# checkpointing may read this file; 508 basins with no held-out remainder
# means tuning against it destroys its only value.
TEST_TARGETS_CSV = DATA_DIR / "test_targets.csv"

# ─────────────────────────────────────────────────────────────
# 1. Reproducibility seeds
#    Set these ONCE here; call seed_everything() at start of
#    every script / notebook.
# ─────────────────────────────────────────────────────────────

GLOBAL_SEED = 42

def seed_everything(seed: int = GLOBAL_SEED) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU + GPU/MPS).
    Call this at the very top of every training script.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic ops on CUDA (slight speed penalty)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't expose a manual_seed directly;
            # torch.manual_seed() covers it via the global generator.
            pass
    except ImportError:
        pass  # torch not available yet — that's fine at import time

    print(f"[config] Seeds set to {seed}")


# ─────────────────────────────────────────────────────────────
# 2. Device detection
#    Priority: CUDA > MPS > CPU
# ─────────────────────────────────────────────────────────────

def get_device(verbose: bool = True):
    """
    Return the best available torch.device.

    Hierarchy
    ---------
    1. CUDA   (Linux + NVIDIA RTX, 8 GB VRAM)
    2. MPS    (macOS Apple Silicon M4)
    3. CPU    (fallback)

    Usage
    -----
        device = get_device()
        model  = MyModel().to(device)
        x      = x.to(device)
    """
    import torch

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if verbose:
            print(f"[config] Device: CUDA  →  {name}  ({mem:.1f} GB)")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        if verbose:
            print("[config] Device: MPS   →  Apple Silicon (M-series)")

    else:
        dev = torch.device("cpu")
        if verbose:
            print("[config] Device: CPU   →  no GPU found")

    return dev


# ─────────────────────────────────────────────────────────────
# 3. Data constants  (from metadata.json)
# ─────────────────────────────────────────────────────────────

HISTORY_HOURS  = 336          # input sequence length (14 days × 24 h)
FORECAST_HOURS = 48           # output sequence length (2 days × 24 h)
N_CHANNELS     = 12           # total input channels
TARGET_CHANNEL = 11           # specific_discharge index in X
N_AUX_CHANNELS = 11           # y_aux channels (future meteo, training only)
N_BASINS       = 508          # number of anonymous basins

CHANNEL_NAMES = [
    "convective_fraction",   # 0
    "longwave_radiation",    # 1
    "potential_energy",      # 2
    "potential_evaporation", # 3
    "pressure",              # 4
    "shortwave_radiation",   # 5
    "specific_humidity",     # 6
    "temperature",           # 7
    "total_precipitation",   # 8
    "wind_u",                # 9
    "wind_v",                # 10
    "specific_discharge",    # 11  ← target
]

# split codes stored in train.h5["split"]
SPLIT_TRAIN = 0
SPLIT_VAL   = 1

# ─────────────────────────────────────────────────────────────
# 4. Split fractions
#    split=0 (254,000) is further divided into train / san_val
#    split=1 (18,142)  is kept as dev (never seen during training)
# ─────────────────────────────────────────────────────────────

# Fraction of split=0 samples PER BASIN held out for sanity validation
# 500 samples/basin × 0.12 ≈ 60 samples → enough for a quick loss curve check
SAN_VAL_FRACTION = 0.12

# ─────────────────────────────────────────────────────────────
# 5. Hyperparameter registry
#    One dict per component — imported by train.py and model files.
#    Change values here; no need to hunt through model code.
# ─────────────────────────────────────────────────────────────

# ── CEEMDAN / MRS ──────────────────────────────────────────
#
# K (n_rimf) is the load-bearing decision here, and it is NOT the paper's.
#
# The paper decomposes ~1400-point DAILY series and gets 9 IMFs. We have
# 336-point HOURLY windows.
#
# CORRECTION (important): an earlier measurement here reported 1-6 IMFs with
# 9.8% of windows yielding a single IMF, and the K decision was justified on
# that. Those numbers came from PLAIN EMD (no ensemble noise). Real CEEMDAN at
# n_trials=100 is very different, because the added noise creates extrema and
# therefore more extractable modes. Measured on 256 real windows:
#
#       0 IMFs:  2.0%  (constant windows)
#       6 IMFs:  7.0%      8 IMFs: 45.7%
#       7 IMFs: 39.8%      9 IMFs:  5.5%
#
# So CEEMDAN yields 6-9 IMFs, mode 8 — much closer to the paper's 9 than the
# plain-EMD figure suggested. Every non-constant window has >= 6 components.
#
# K=3 (n_high=2, n_low=1) is still the choice, now for a cleaner reason: it is
# satisfiable in EVERY non-constant window, and grouping to a constant K is a
# mild specialisation of the paper's own operator, which already maps
# 9->6 (Shuangpai) and 9->4 (Fenghuang). A fixed branch list cannot take a
# variable component count: zero-padding would make branch i mean a different
# frequency band per sample, destroying the per-branch specialisation that
# justifies one-model-per-RIMF at all.
#
# Constant windows (~2%) are handled explicitly via DEG_CONSTANT, not
# silently. K stays configurable so C5/C7 remain real ablations.
CEEMDAN_CFG = dict(
    noise_std      = 0.2,    # C1: ε₀ — paper value; Sobol shows low sensitivity
    n_trials       = 100,    # C2: N ensemble trials
    max_imf        = 9,      # C5: upper bound on number of IMFs
    se_embed_dim   = 2,      # C3: m — embedding dimension for sample entropy
    se_tolerance   = 0.2,    # C4: r factor (multiplied by std of each IMF)
    decompose_ch   = TARGET_CHANNEL,  # C6: which channel to decompose (discharge)

    # ── added for the MRS implementation ─────────────────────────────
    n_rimf         = 3,      # K: RIMFs after grouping (see note above)
    n_sift         = 8,      # sifting iterations per IMF. Fixed count, not
                             # Huang's SD threshold: bounded and fully
                             # deterministic, so a multi-hour precompute is
                             # reproducible. Matches runoff_emd_deep_dive.py.
    noise_seed     = 1234,   # seeds the CEEMDAN noise bank. The bank depends
                             # only on (seed, n_trials, T) — not on the signal
                             # — so it is built ONCE per worker and reused.
                             # Without that reuse the precompute doubles.
    hf_rule        = "fixed_split",  # "fixed_split" | "se_threshold"
                             # fixed_split: top n_high by SE -> Informer.
                             #   Gives a CONSTANT branch count, which the
                             #   fixed-branch model requires.
                             # se_threshold: paper rule (SE >= SE(q)), then
                             #   reconciled to n_high/n_low. Kept as an
                             #   ablation; re-derived from cached SE, so it
                             #   costs no recomputation.
    rimf_input_mode = "replace",  # "replace" | "univariate"
                             # replace: keep channels 0-10, swap ch 11 for
                             #   RIMF_j. The paper is univariate, but our
                             #   target is driven by 11 meteo forcings that
                             #   every branch needs; a single-channel RIMF
                             #   would discard precipitation and guarantee the
                             #   branch underperforms the baseline.
                             # univariate: (B,T,1) per branch, paper-faithful.
    n_high         = 2,      # RIMFs routed to the Informer (high frequency)
    n_low          = 1,      # RIMFs routed to the LSTM      (low frequency)
    cache_dir      = DATA_DIR,
)

# ── Informer (High-Frequency branch) ─────────────────────
INFORMER_CFG = dict(
    d_model        = 256,    # I1: hidden dimension
    n_heads        = 8,      # I2: attention heads
    enc_layers     = 2,      # I3: encoder stack depth
    dec_layers     = 1,      # I4: decoder stack depth
    d_ff           = 1024,   # I5: feed-forward dimension
    activation     = "gelu", # I6: activation function
    prob_factor    = 5,      # I7: ProbSparse c factor
    # Attention backend. DEFAULT IS "full", not the paper's ProbSparse.
    #
    # Two measured reasons:
    #  1. At L=336, factor=5 gives u = 5*int(ln 337) = 25 active queries.
    #     25/336 (7.4%) of positions get real attention; the other 311 receive
    #     ONE byte-identical constant vector (verified: "distinct vectors among
    #     the 311 inactive positions: 1").
    #  2. The memory argument inverts at this length: full attention matrix is
    #     57.8M elements (220 MB), the ProbSparse sampling intermediate is
    #     137.6M (525 MB) — 2.4x LARGER. ProbSparse pays off at L>=5000.
    #
    # "full" uses F.scaled_dot_product_attention (fused/flash kernels, matrix
    # never materialised). "prob" is kept, now correctly implemented, so
    # ablation I7 compares two working backends.
    attention      = "full", # "full" | "prob"
    dropout        = 0.1,    # regularisation
    seq_len        = HISTORY_HOURS,    # I8: encoder input length
    label_len      = FORECAST_HOURS // 2,  # decoder start token length
    pred_len       = FORECAST_HOURS,   # output length
    use_distil     = True,   # I10: distillation layers
)

# ── LSTM (Low-Frequency branch) ───────────────────────────
LSTM_CFG = dict(
    hidden_size    = 256,    # L1
    num_layers     = 2,      # L2
    dropout        = 0.1,    # L3
    bidirectional  = False,  # L4
    pred_len       = FORECAST_HOURS,
)

# ── Ensemble combiner ────────────────────────────────────
ENSEMBLE_CFG = dict(
    method         = "sum",  # E1: "sum" | "linear" | "mlp"
)

# ── Training ─────────────────────────────────────────────
TRAIN_CFG = dict(
    seed              = GLOBAL_SEED,

    # Optimizer (T1a / T7)
    # AdamW decouples weight decay from the adaptive lr step, fixing a known
    # Adam bug. Standard for all modern Transformers (BERT, GPT, PatchTST).
    #
    # Adam:  theta -= lr * mhat/(sqrt(vhat)+eps) - lr*lambda*theta
    #                                               ^ scaled by 1/sqrt(vhat)
    #        => effective decay is INVERSELY proportional to gradient
    #           magnitude, so heavily-updated params get least regularised.
    # AdamW: decay applied outside the adaptive step, uniform for all params.
    optimizer         = "adamw",     # "adam" | "adamw"  — AdamW is our default
    lr                = 1e-4,        # T1: initial learning rate
    weight_decay      = 1e-4,        # T7: AdamW decoupled weight decay

    # Weight decay is applied ONLY to multi-dimensional weight tensors.
    # build_optimizer() splits parameters into decay / no-decay groups:
    #   no-decay: LayerNorm gamma/beta, BatchNorm gamma/beta, all biases,
    #             and any 1-D combiner weight.
    # Rationale: decaying LayerNorm gamma toward zero directly fights the
    # layer's ability to rescale features — which is its entire function.
    # The Informer has LayerNorm in every encoder/decoder block, so this is
    # not a marginal detail here.

    # Optional lr multiplier for the ensemble combiner parameters only.
    # A 1-D combiner weight (LinearCombiner.w) is a handful of scalars
    # competing against ~2.4M other parameters at one shared lr, so it may
    # not travel far enough from its initialisation during training — which
    # would make ablation E1 ("does a learned combiner beat a fixed sum?")
    # measure optimisation difficulty rather than combiner value.
    # 1.0 = identity. Raise to 10-100 to test whether the combiner is
    # under-trained rather than unhelpful.
    combiner_lr_mult  = 1.0,

    # LR Scheduler (T2 / T2a / T2b)
    # LinearWarmup + CosineDecay: standard Transformer recipe.
    # Warmup prevents large early gradients from corrupting attention heads
    # before LayerNorm statistics have stabilised.
    # After warmup, cosine annealing smoothly decays to eta_min.
    #
    # Cosine steps PER BATCH, not per epoch. Per-epoch stepping would give
    # only `epochs` discrete lr values (a staircase); per-batch gives
    # epochs * steps_per_epoch and a genuinely smooth decay. This matters
    # most in the final phase, where per-epoch stepping would hold a
    # still-too-large lr for an entire 3,492-step epoch.
    scheduler         = "warmup_cosine",  # "none"|"warmup_cosine"|"cosine"|"sgdr"|"plateau"
    warmup_frac       = 0.05,        # T2a: fraction of total steps used for linear warmup
    eta_min           = 1e-6,        # T2b: minimum lr at end of cosine decay
    # For SGDR (CosineAnnealingWarmRestarts) only:
    sgdr_T0           = 20,          # T2c: first cycle length in epochs
    sgdr_T_mult       = 2,           # each restart doubles the cycle length

    # Batch / epochs (T4 / T5)
    # batch=64 is the MPS sweet spot on Apple M4 (fills 2 SIMD groups = ~90-95% utilisation)
    # vs batch=32 which leaves half the SIMD capacity idle.
    # On CUDA (Linux 8GB) batch=64 is also safe; scale to 128 if training is slow.
    batch_size        = 64,          # T4: canonical cross-machine value

    # EPOCH BUDGET — measured, and revised once after looking at a real run.
    #
    # ~11 min/epoch on this M4 (223,500 train windows, batch 64,
    # num_workers=0 because the sandbox blocks torch's shm manager).
    #
    # First attempt was 12 epochs, chosen to fit a 31-experiment grid. A real
    # run showed that was too short:
    #
    #    ep   train_loss   sanval_NSE   delta
    #     8     0.4317       0.5598
    #     9     0.4192       0.5799     +0.0201
    #    10     0.4114       0.5941     +0.0142   <- best epoch is the LAST
    #
    # Train loss still falling AND val NSE still rising = not overfitting, just
    # truncated. Worse, cosine anneals lr to its floor at exactly `epochs`, so
    # by epoch 10 lr was 2.9e-06 (~35x below start): the model was STARVED of
    # learning rate, not converged. The budget manufactured the appearance of
    # convergence.
    #
    # 40 epochs (~7.3 h/experiment) on the ~8 highest-value experiments instead
    # of 12 epochs across all 31. Rationale: a truncated run penalises configs
    # that converge slowly, so cheap-but-short comparisons can invert the
    # ranking they are supposed to establish. Fewer, converged experiments beat
    # more, starved ones.
    epochs            = 40,          # T5

    # Regularisation / stability
    loss              = "mse",       # T3: "mse"|"mae"|"huber"|"nse"

    # Huber delta must match the ERROR scale rather than sit at the torch
    # default. Measured on z-scored targets: median |z| = 0.311, p99 = 3.89,
    # max = 82.2, and 93.1% of targets have |z| < 1.0. So delta=1.0 is
    # quadratic for ~93% of the data and linear only for the extremes.
    huber_delta       = 1.0,

    # Epsilon for the basin-averaged NSE loss (Kratzert 2019).
    # Per-basin sigma spans a 521x range and 14 basins (2.8%) have
    # sigma < 0.01 — essentially flat lines. At eps=0 (which is what plain MSE
    # on z-scored targets already is) their pure noise is amplified to the same
    # gradient magnitude as a genuinely dynamic basin. eps damps that.
    nse_eps           = 0.1,

    # AUXILIARY TASK — the only legitimate use of y_aux.
    # metadata.json: "y_aux_role": "future_supervision_only_not_inference_inputs"
    # and y_aux is absent from test.h5. Feeding it to the decoder would inflate
    # dev NSE and collapse at submission. As an auxiliary TARGET it instead
    # regularises the encoder; the head is discarded at inference.
    aux_task          = False,
    aux_loss_weight   = 0.3,

    # CEEMDAN / MRS dual-branch mode. Requires a prebuilt RIMF cache:
    #   python src/decompose.py --split train
    use_ceemdan       = False,

    # Restrict the input channels (D1). None = all 12.
    input_channels    = None,

    # Gradient clipping max norm. 0 disables.
    # The mean PRE-clip gradient norm is logged every epoch (sampled every
    # 50th step when clipping is off) and recorded in summary.json as
    # mean_grad_norm. Read it before trusting this value:
    #   mean_grad_norm << clip_grad  -> clipper never fires, paying for a no-op
    #   mean_grad_norm -> clip_grad  -> clipper fires constantly and silently
    #                                   rescales every update, changing the
    #                                   effective lr with no log line saying so
    clip_grad         = 1.0,         # T6

    # Early stopping (see implementation_plan.md §12)
    # Monitors median NSE on san_val (mode=max). Restores best checkpoint.
    # patience=10 chosen to survive seasonal plateaus in hydrological signals.
    #
    # NOTE: automatically relaxed to epochs//2 when a cosine-family schedule
    # is active (see train.adjust_patience_for_scheduler). Cosine anneals to
    # its floor at exactly `epochs`; stopping mid-anneal discards the low-lr
    # phase where convergence happens. summary.json records both
    # patience_requested and patience_used.
    early_stop_patience   = 8,
    early_stop_min_delta  = 1e-4,   # must improve by > 0.0001 to reset counter
    early_stop_metric     = "median_nse",  # what to monitor (not training loss)

    # Data pipeline
    norm_strategy     = "per_basin_zscore",  # T8: per-basin z-score is critical
    use_basin_emb     = False,       # T10: learned basin embedding (508 basins)
    basin_emb_dim     = 16,          # embedding dimension if use_basin_emb=True
    # DataLoader workers.
    # 0, not 4: torch's shared-memory manager (torch_shm_manager) is blocked
    # in this sandbox — any num_workers>0 dies with
    #   "torch_shm_manager: Operation not permitted"
    # at the first collate. Single-process loading also removes the 16
    # concurrent HDF5 handles that 4 loaders x 4 workers would have opened.
    num_workers       = 0,
    pin_memory        = True,        # faster CPU→CUDA transfer; auto-disabled on MPS
)


# ─────────────────────────────────────────────────────────────
# 6. Quick self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seed_everything()
    dev = get_device()
    print()
    print("Paths:")
    print(f"  ROOT      = {ROOT}")
    print(f"  TRAIN_H5  = {TRAIN_H5}  exists={TRAIN_H5.exists()}")
    print(f"  TEST_H5   = {TEST_H5}   exists={TEST_H5.exists()}")
    print()
    print("Data constants:")
    print(f"  History  : {HISTORY_HOURS} h  ({HISTORY_HOURS//24} days)")
    print(f"  Forecast : {FORECAST_HOURS} h  ({FORECAST_HOURS//24} days)")
    print(f"  Channels : {N_CHANNELS}  (target = ch {TARGET_CHANNEL} = specific_discharge)")
    print(f"  Basins   : {N_BASINS}")
    print()
    print("CEEMDAN cfg :", CEEMDAN_CFG)
    print("Informer cfg:", INFORMER_CFG)
    print("LSTM cfg    :", LSTM_CFG)
    print("Train cfg   :", TRAIN_CFG)
