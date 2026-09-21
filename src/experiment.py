"""
src/experiment.py
─────────────────
Orchestrates a single experiment end to end:
  1. Merge base config + overrides
  2. Seed everything
  3. Build data splits, normalizer, loaders
  4. Build model, optimizer, scheduler, criterion
  5. Training loop with early stopping on san_val NSE
  6. Final evaluation on dev split
  7. Save all artefacts to experiments/<name>/

Called by run_experiment.py (CLI) and run_ablation.py (grid).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

# Make src/ importable regardless of working directory
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import torch

from config       import (TRAIN_CFG, INFORMER_CFG, LSTM_CFG, ENSEMBLE_CFG,
                           EXP_DIR, GLOBAL_SEED, HISTORY_HOURS,
                           seed_everything, get_device)
from dataset      import build_splits, make_loader
from models.dmel  import build_model
from train        import (build_optimizer, build_scheduler,
                          build_loss, train_one_epoch, quick_nse,
                          scheduler_granularity, adjust_patience_for_scheduler)
from early_stopping import EarlyStopping
from evaluate       import evaluate_model, EvalResult


# ─────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────

def _merge_cfg(overrides: dict) -> dict:
    """Return a flat config dict: TRAIN_CFG defaults + overrides."""
    cfg = deepcopy(TRAIN_CFG)
    cfg.update(deepcopy(overrides))
    return cfg


def _save_config(cfg: dict, path: Path) -> None:
    # Convert non-serialisable values to strings
    safe = {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                else str(v))
            for k, v in cfg.items()}
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def run(
    name      : str,
    overrides : dict  = None,
    force     : bool  = False,
    verbose   : bool  = True,
) -> EvalResult:
    """
    Run one complete experiment.

    Parameters
    ----------
    name      : experiment name, becomes the folder name under experiments/
    overrides : dict of hyperparameter overrides on top of TRAIN_CFG
    force     : if True, re-run even if metrics_dev.json already exists
    verbose   : print progress

    Returns
    -------
    EvalResult on the dev split (split=1).
    """
    overrides  = overrides or {}
    cfg        = _merge_cfg(overrides)
    exp_dir    = EXP_DIR / name
    dev_metric = exp_dir / "metrics_dev.json"

    # ── Skip if already done ──────────────────────────────────────────
    if dev_metric.exists() and not force:
        if verbose:
            print(f"  [skip] {name} — already complete. "
                  "Use force=True to rerun.")
        return EvalResult.load(str(dev_metric))

    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Redirect stdout to log file + console ─────────────────────────
    log_path = exp_dir / "log.txt"
    logger   = _Tee(str(log_path)) if verbose else open(os.devnull, "w")

    with logger:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {name}")
        print(f"  Overrides : {overrides}")
        print(f"{'='*60}")

        # 1. Reproducibility
        seed = cfg.get("seed", GLOBAL_SEED)
        seed_everything(seed)
        device = get_device(verbose=verbose)

        # Dedicated generator for DataLoader shuffle. Keeps batch order
        # reproducible AND independent of how many draws model init or any
        # stochastic layer takes from the global torch RNG — so changing an
        # unrelated hyperparameter cannot silently reshuffle the data.
        loader_gen = torch.Generator()
        loader_gen.manual_seed(seed)

        # 2. Data
        print("\n[data] Building splits and normalizer...")
        from config import CEEMDAN_CFG
        ceemdan_cfg = {**CEEMDAN_CFG, **{k: v for k, v in cfg.items()
                                         if k in CEEMDAN_CFG}}
        ds_train, ds_sanval, ds_dev, ds_test, normalizer, split_stats = \
            build_splits(
                norm_strategy  = cfg.get("norm_strategy", "per_basin_zscore"),
                seed           = seed,
                seq_len        = cfg.get("seq_len", HISTORY_HOURS),
                input_channels = cfg.get("input_channels"),
                use_ceemdan    = cfg.get("use_ceemdan", False),
                want_aux       = cfg.get("aux_task", False),
                ceemdan_cfg    = ceemdan_cfg,
                verbose        = verbose,
            )
        train_loader  = make_loader(ds_train,  cfg["batch_size"], shuffle=True,
                                     generator=loader_gen)
        sanval_loader = make_loader(ds_sanval, cfg["batch_size"], shuffle=False)
        dev_loader    = make_loader(ds_dev,    cfg["batch_size"], shuffle=False)

        # Persist the TRAIN-split normalisation beside the checkpoint.
        # Inference (predict_test.py) must apply the identical transform;
        # recomputing stats there would be test-time leakage, and
        # recomputing them from train would risk silent drift if the split
        # seed or fraction ever changes.
        if normalizer is not None:
            normalizer.save(str(exp_dir / "norm"))

        # 3. Model
        print("\n[model] Building DMEL...")
        model = build_model(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        # 4. Optimiser / scheduler / criterion
        optimizer = build_optimizer(model, cfg, log=print)
        n_epochs  = cfg.get("epochs", 100)
        total_steps = len(train_loader) * n_epochs
        scheduler = build_scheduler(optimizer, cfg, total_steps)
        # NSE loss needs per-basin target std; see train.NSELoss for why the
        # previous batch-wide version was just MSE in disguise.
        basin_std = None
        if cfg.get("loss") == "nse":
            basin_std = torch.tensor(normalizer.target_std(), device=device)
        criterion = build_loss(cfg, basin_std=basin_std)
        sched_gran = scheduler_granularity(cfg)
        print(f"  [sched] {cfg.get('scheduler','warmup_cosine')} "
              f"(steps per {sched_gran}), total_steps={total_steps:,}")

        # 5. Early stopping
        # Patience is relaxed automatically when a cosine-family schedule
        # is active — stopping mid-anneal would discard the low-lr phase
        # where convergence actually happens.
        base_patience = cfg.get("early_stop_patience", 10)
        patience = adjust_patience_for_scheduler(
            base_patience, n_epochs, cfg, log=print,
        )
        stopper = EarlyStopping(
            patience  = patience,
            min_delta = cfg.get("early_stop_min_delta", 1e-4),
            save_dir  = str(exp_dir),
            verbose   = verbose,
        )

        # 6. History CSV
        hist_path = exp_dir / "history.csv"
        hist_f    = open(hist_path, "w", newline="")
        hist_csv  = csv.writer(hist_f)
        hist_csv.writerow(["epoch", "train_loss", "sanval_nse",
                            "lr", "grad_norm", "elapsed_s"])

        # 7. Training loop
        print(f"\n[train] Starting — max {n_epochs} epochs, "
              f"patience={patience}")
        history = []
        for epoch in range(n_epochs):
            ep_t0 = time.time()

            tr = train_one_epoch(
                model, train_loader, optimizer, criterion,
                scheduler, device, cfg, epoch,
            )
            avg_loss  = tr["train_loss"]
            grad_norm = tr["grad_norm"]

            nse_val = quick_nse(model, sanval_loader, normalizer, device, cfg)

            # ReduceLROnPlateau needs the metric, not a bare step
            if isinstance(scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(nse_val)

            cur_lr  = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - ep_t0

            gn_str = f" | gnorm {grad_norm:.3f}" if grad_norm is not None else ""
            print(f"  epoch {epoch:3d} | loss {avg_loss:.5f} "
                  f"| sanval_nse {nse_val:.5f} "
                  f"| lr {cur_lr:.2e}{gn_str} | {elapsed:.1f}s")

            hist_csv.writerow([epoch, avg_loss, nse_val, cur_lr,
                                "" if grad_norm is None else f"{grad_norm:.6f}",
                                f"{elapsed:.1f}"])
            hist_f.flush()
            history.append({
                "epoch": epoch, "train_loss": avg_loss,
                "sanval_nse": nse_val, "lr": cur_lr,
                "grad_norm": grad_norm,
            })

            stopper.step(nse_val, model, optimizer, scheduler, epoch,
                         extra={"train_loss": avg_loss,
                                "grad_norm": grad_norm})

            if stopper.should_stop:
                break

        hist_f.close()

        # 8. Restore best weights
        print(f"\n[train] Restoring best weights (epoch {stopper.best_epoch})")
        stopper.restore_best(model, optimizer, scheduler, device)

        # 9. Final evaluation on dev split
        print("\n[eval] Evaluating on dev split (split=1)...")
        eval_dev = evaluate_model(
            model, dev_loader, normalizer, device,
            split_name = "dev",
            best_epoch = stopper.best_epoch,
            cfg        = cfg,
            verbose    = verbose,
        )
        eval_dev.save(str(dev_metric))

        # 10. Evaluation on san_val at best epoch
        print("\n[eval] Evaluating on san_val...")
        eval_sanval = evaluate_model(
            model, sanval_loader, normalizer, device,
            split_name = "san_val",
            best_epoch = stopper.best_epoch,
            cfg        = cfg,
            verbose    = False,
        )
        eval_sanval.save(str(exp_dir / "metrics_sanval.json"))

        # 11. Save config
        _save_config(cfg, exp_dir / "config.json")

        # 12. Save run summary
        # This is the self-describing record of the run: what was selected
        # on, what was reported, and the gap between them. A reader must
        # never have to infer which split drove which decision.
        total_time = time.time() - t0
        mean_gnorm = [h["grad_norm"] for h in history
                      if h["grad_norm"] is not None]
        summary = {
            # identity
            "name"            : name,
            "seed"            : seed,
            "device"          : str(device),
            "overrides"       : {k: (v if isinstance(
                                      v, (int, float, str, bool, type(None)))
                                     else str(v))
                                 for k, v in overrides.items()},

            # SPLIT DISCIPLINE — the audit trail
            # selection_split : the split early stopping and checkpointing read
            # report_split    : the split the headline number comes from
            # These must differ. If they were ever the same, the reported
            # metric would be optimistically biased by construction.
            "selection_split" : "san_val",
            "report_split"    : "dev",

            # headline (dev — never touched during training)
            "dev_median_nse"  : eval_dev.median_nse,
            "dev_mean_nse"    : eval_dev.mean_nse,
            "dev_median_kge"  : eval_dev.median_kge,
            "dev_pct_nse_05"  : eval_dev.pct_nse_05,
            "dev_pct_nse_07"  : eval_dev.pct_nse_07,

            # selection split metrics — kept so tuning can be compared
            # WITHOUT reading dev. Prefer these for any iteration decision.
            "sanval_median_nse": eval_sanval.median_nse,
            "sanval_mean_nse"  : eval_sanval.mean_nse,

            # generalisation gap: dev minus san_val.
            # Large negative gap = the model was selected on san_val noise
            # and does not transfer. This is the number that reveals
            # selection overfitting.
            "sanval_to_dev_gap": eval_dev.median_nse - eval_sanval.median_nse,

            # training trajectory
            "best_epoch"      : stopper.best_epoch,
            "best_sanval_nse" : stopper.best_nse,
            "epochs_run"      : len(history),
            "epochs_max"      : n_epochs,
            "early_stopped"   : stopper.should_stop,
            "final_lr"        : history[-1]["lr"] if history else None,
            "final_train_loss": history[-1]["train_loss"] if history else None,

            # gradient health. If mean_grad_norm approaches clip_grad the
            # clipper is rescaling every step and silently changing the
            # effective lr; if it is far below, clipping is a no-op.
            "mean_grad_norm"  : (sum(mean_gnorm) / len(mean_gnorm)
                                 if mean_gnorm else None),
            "clip_grad"       : cfg.get("clip_grad"),

            # resolved training config (post-adjustment)
            "optimizer"       : cfg.get("optimizer"),
            "lr"              : cfg.get("lr"),
            "weight_decay"    : cfg.get("weight_decay"),
            "scheduler"       : cfg.get("scheduler"),
            "scheduler_steps_per": sched_gran,
            "loss"            : cfg.get("loss"),
            "batch_size"      : cfg.get("batch_size"),
            "patience_requested": base_patience,
            "patience_used"   : patience,
            "norm_strategy"   : cfg.get("norm_strategy"),
            "aux_task"        : cfg.get("aux_task"),
            "aux_loss_weight" : cfg.get("aux_loss_weight"),
            "attention"       : cfg.get("attention"),

            # Split discipline, measured rather than asserted.
            "split_method"    : split_stats.get("method"),
            "selection_measures": split_stats.get("selection_measures"),
            "n_train_basins"  : split_stats.get("n_train_basins"),
            "n_val_basins"    : split_stats.get("n_val_basins"),

            # Per-lead-time NSE. The aggregate is dominated by the easy early
            # hours (persistence alone scores 0.94 at h+1..12), so a single
            # median hides whether the model beats persistence where it counts.
            "dev_horizon_nse" : eval_dev.horizon_nse,
            "use_basin_emb"   : cfg.get("use_basin_emb"),
            "use_ceemdan"     : cfg.get("use_ceemdan"),
            "ensemble_method" : cfg.get("ensemble_method"),

            # cost
            "n_params"        : n_params,
            "total_minutes"   : round(total_time / 60, 2),
        }
        with open(exp_dir / "summary.json", "w") as f:
            from evaluate import _json_safe
            json.dump(summary, f, indent=2, default=_json_safe)

        print(f"\n{'='*60}")
        print(f"  DONE  {name}")
        print(f"  selection split : san_val  (early stopping)")
        print(f"  report split    : dev      (headline, untouched)")
        print(f"  san_val med NSE : {eval_sanval.median_nse:.4f}")
        print(f"  dev     med NSE : {eval_dev.median_nse:.4f}")
        print(f"  gap (dev-sanval): {summary['sanval_to_dev_gap']:+.4f}")
        print(f"  best epoch      : {stopper.best_epoch} "
              f"of {len(history)} run")
        if summary["mean_grad_norm"] is not None:
            print(f"  mean grad norm  : {summary['mean_grad_norm']:.4f} "
                  f"(clip={cfg.get('clip_grad')})")
        print(f"  total time      : {total_time/60:.1f} min")
        print(f"{'='*60}\n")

    return eval_dev


# ─────────────────────────────────────────────────────────────────────
# Tee: write stdout to file AND console simultaneously
# ─────────────────────────────────────────────────────────────────────

class _Tee:
    """Context manager: duplicates stdout to a log file."""

    def __init__(self, path: str):
        self.path     = path
        self._file    = None
        self._stdout  = sys.stdout

    def __enter__(self):
        self._file  = open(self.path, "w", buffering=1)
        sys.stdout  = self
        return self

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __exit__(self, *args):
        sys.stdout = self._stdout
        self._file.close()
