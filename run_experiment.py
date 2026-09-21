#!/usr/bin/env python3
"""
run_experiment.py
─────────────────
CLI entry point for running a single experiment.

Usage examples
──────────────
  # Baseline (paper defaults, no CEEMDAN)
  python run_experiment.py --name exp_001__baseline

  # Override loss function
  python run_experiment.py --name exp_011__nse_loss --loss nse

  # All 12 channels, AdamW, warmup cosine
  python run_experiment.py --name exp_003__all_channels

  # With future meteo in decoder
  python run_experiment.py --name exp_007__y_aux --use_y_aux

  # With basin embedding
  python run_experiment.py --name exp_010__basin_emb --use_basin_emb

  # Change LR and dropout
  python run_experiment.py --name exp_013__lr_5e4 --lr 5e-4

  # Force re-run even if results exist
  python run_experiment.py --name exp_001__baseline --force

  # Different seed
  python run_experiment.py --name exp_001__baseline_s123 --seed 123

Each flag maps to exactly one key in the config dict.
The full list of flags is the complete experiment decision space.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from experiment import run


def parse_args():
    p = argparse.ArgumentParser(
        description="Run one DMEL experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Identity ──────────────────────────────────────────────────────
    p.add_argument("--name",   required=True,
                   help="Experiment name → experiments/<name>/ folder")
    p.add_argument("--force",  action="store_true",
                   help="Re-run even if metrics_dev.json already exists")
    p.add_argument("--seed",   type=int, default=42)

    # ── Data ──────────────────────────────────────────────────────────
    p.add_argument("--norm_strategy", default="per_basin_zscore",
                   choices=["per_basin_zscore", "global_zscore", "none"],
                   help="[T8] Normalisation strategy")

    # ── Model ─────────────────────────────────────────────────────────
    p.add_argument("--use_ceemdan",     action="store_true",
                   help="[A-OUR-1] Use CEEMDAN decomposition (requires decompose.py)")
    p.add_argument("--use_basin_emb",   action="store_true",
                   help="[T10] Append learned basin embedding to input")
    p.add_argument("--basin_emb_dim",   type=int,   default=16)
    p.add_argument("--use_y_aux",       action="store_true",
                   help="[E5/D2] Feed future meteo (y_aux) to Informer decoder")
    p.add_argument("--ensemble_method", default="sum",
                   choices=["sum", "linear", "mlp"],
                   help="[E1] How to combine Informer + LSTM outputs")
    p.add_argument("--shared_informer", action="store_true",
                   help="[E3] Share one Informer across all HF branches")
    p.add_argument("--shared_lstm",     action="store_true",
                   help="[E4] Share one LSTM across all LF branches")

    # ── Informer architecture ─────────────────────────────────────────
    p.add_argument("--seq_len",    type=int,   default=336,
                   help="[I8] Encoder input length in hours (168 or 336)")
    p.add_argument("--d_model",    type=int,   default=256)
    p.add_argument("--n_heads",    type=int,   default=8)
    p.add_argument("--enc_layers", type=int,   default=2)
    p.add_argument("--dec_layers", type=int,   default=1)
    p.add_argument("--d_ff",       type=int,   default=1024)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--use_distil", action="store_true", default=True,
                   help="[I10] Use distillation layers in encoder")
    p.add_argument("--no_distil",  action="store_false", dest="use_distil")

    # ── LSTM architecture ─────────────────────────────────────────────
    p.add_argument("--lstm_hidden",  type=int,   default=256, help="[L1]")
    p.add_argument("--lstm_layers",  type=int,   default=2,   help="[L2]")
    p.add_argument("--lstm_dropout", type=float, default=0.1, help="[L3]")
    p.add_argument("--lstm_bidir",   action="store_true",     help="[L4]")

    # ── Training ──────────────────────────────────────────────────────
    p.add_argument("--optimizer",  default="adamw",
                   choices=["adamw", "adam"],          help="[T1a]")
    p.add_argument("--lr",         type=float, default=1e-4, help="[T1]")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="[T7]")
    p.add_argument("--combiner_lr_mult", type=float, default=1.0,
                   help="lr multiplier for combiner params only (own group, no decay)")
    p.add_argument("--scheduler",  default="warmup_cosine",
                   choices=["warmup_cosine", "cosine", "sgdr",
                             "plateau", "none"],        help="[T2]")
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--loss",        default="mse",
                   choices=["mse", "mae", "huber", "nse"], help="[T3]")
    p.add_argument("--batch_size",  type=int,   default=64,  help="[T4]")
    p.add_argument("--epochs",      type=int,   default=100, help="[T5]")
    p.add_argument("--clip_grad",   type=float, default=1.0, help="[T6]")
    p.add_argument("--early_stop_patience",  type=int,   default=10)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    return p.parse_args()


def main():
    args = parse_args()

    # Build overrides dict from all non-default flags
    overrides = {
        "seed"                  : args.seed,
        "norm_strategy"         : args.norm_strategy,
        "use_ceemdan"           : args.use_ceemdan,
        "use_basin_emb"         : args.use_basin_emb,
        "basin_emb_dim"         : args.basin_emb_dim,
        "use_y_aux"             : args.use_y_aux,
        "ensemble_method"       : args.ensemble_method,
        "shared_informer"       : args.shared_informer,
        "shared_lstm"           : args.shared_lstm,
        "seq_len"               : args.seq_len,
        "d_model"               : args.d_model,
        "n_heads"               : args.n_heads,
        "enc_layers"            : args.enc_layers,
        "dec_layers"            : args.dec_layers,
        "d_ff"                  : args.d_ff,
        "dropout"               : args.dropout,
        "use_distil"            : args.use_distil,
        "lstm_hidden"           : args.lstm_hidden,
        "lstm_layers"           : args.lstm_layers,
        "lstm_dropout"          : args.lstm_dropout,
        "lstm_bidir"            : args.lstm_bidir,
        "optimizer"             : args.optimizer,
        "lr"                    : args.lr,
        "weight_decay"          : args.weight_decay,
        "combiner_lr_mult"      : args.combiner_lr_mult,
        "scheduler"             : args.scheduler,
        "warmup_frac"           : args.warmup_frac,
        "loss"                  : args.loss,
        "batch_size"            : args.batch_size,
        "epochs"                : args.epochs,
        "clip_grad"             : args.clip_grad,
        "early_stop_patience"   : args.early_stop_patience,
        "early_stop_min_delta"  : args.early_stop_min_delta,
    }

    run(name=args.name, overrides=overrides,
        force=args.force, verbose=True)


if __name__ == "__main__":
    main()
