"""
evaluate_test.py
────────────────
Final read-out of a trained checkpoint on test.h5, scored against the
ground-truth targets in data/test_targets.csv.

Why this is a separate script and not a flag on experiment.py
─────────────────────────────────────────────────────────────
Test must stay unseen. It is 508 basins with no held-out remainder behind
them, so the moment a test number influences a checkpoint, a hyperparameter,
or an architecture choice, it stops estimating generalisation and becomes the
same optimistic bias that the san_val/dev split discipline exists to prevent.

Keeping it out of the training entry point is a structural guarantee rather
than a promise: experiment.py cannot read test targets because it has no code
path that does. This script runs AFTER a run is finished, on a checkpoint that
is already frozen.

What it reports
───────────────
  * test median NSE          ← the headline
  * the san_val / dev / test triple, side by side
  * per-horizon NSE decay (h1-12 ... h37-48)
  * the generalisation gap, and which direction it points

Usage
-----
    python src/evaluate_test.py --exp exp_001__baseline
    python src/evaluate_test.py --exp exp_001__baseline --save
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    TEST_H5, TEST_TARGETS_CSV, TRAIN_H5, EXP_DIR,
    FORECAST_HOURS, TARGET_CHANNEL, TRAIN_CFG, GLOBAL_SEED,
    seed_everything, get_device,
)


# ─────────────────────────────────────────────────────────────────────
# 1.  Target loading + alignment checks
# ─────────────────────────────────────────────────────────────────────

def load_test_targets(csv_path: Path = TEST_TARGETS_CSV,
                      test_h5: Path = TEST_H5,
                      verbose: bool = True) -> np.ndarray:
    """
    Load data/test_targets.csv into an (n_test, 48) raw-mm/h array, ordered by
    test.h5 row index.

    The CSV is a positional join (metadata.json: id_rule =
    zero_based_row_index_within_file), so every check here is about proving
    that assumption instead of trusting it. A silently misaligned target file
    produces a plausible-looking but meaningless NSE, which is the single
    worst failure mode available in this script.
    """
    import h5py
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(
            f"test targets not found: {csv_path}\n"
            f"This file ships with the dataset alongside test.h5.")

    df = pd.read_csv(csv_path)

    with h5py.File(test_h5, "r") as f:
        n_test = len(f["X"])
        last_hist = f["X"][:, -1, TARGET_CHANNEL].astype(np.float64)

    # ── shape / id contract ───────────────────────────────────────────
    if "Id" not in df.columns:
        raise ValueError(f"expected an 'Id' column, got {list(df.columns)[:5]}")

    qcols = [c for c in df.columns if c != "Id"]
    if len(qcols) != FORECAST_HOURS:
        raise ValueError(
            f"expected {FORECAST_HOURS} target columns, got {len(qcols)}")

    if len(df) != n_test:
        raise ValueError(
            f"row count mismatch: csv has {len(df):,}, "
            f"test.h5 has {n_test:,}")

    ids = df["Id"].to_numpy()
    if not np.array_equal(ids, np.arange(n_test)):
        # Recoverable — sort by Id — but loud, because it means the file is
        # not in the order the id_rule implies.
        print("[warn] Id column is not 0..N-1 in order; sorting by Id")
        df = df.sort_values("Id").reset_index(drop=True)
        if not np.array_equal(df["Id"].to_numpy(), np.arange(n_test)):
            raise ValueError("Id column is not a permutation of 0..N-1")

    Y = df[qcols].to_numpy(np.float32)            # (n_test, 48) raw mm/h

    # ── sanity: raw scale, not z-scored ───────────────────────────────
    # z-scored targets would sit near mean 0 with negatives present. Raw
    # specific discharge is non-negative and small (median ~0.02 mm/h).
    n_neg = int((Y < 0).sum())
    if n_neg:
        raise ValueError(
            f"{n_neg:,} negative target values — discharge cannot be "
            f"negative, so this file is not raw mm/h as assumed.")
    if not np.isfinite(Y).all():
        raise ValueError(f"{int((~np.isfinite(Y)).sum()):,} non-finite targets")

    # ── sanity: physical continuity proves the row alignment ──────────
    # The last observed history hour and the first forecast hour are adjacent
    # hours of one discharge series, so they must nearly coincide. Under a
    # shuffled join this correlation collapses toward 0.
    corr = float(np.corrcoef(last_hist, Y[:, 0].astype(np.float64))[0, 1])
    if corr < 0.90:
        raise ValueError(
            f"continuity check FAILED: corr(X[:,-1,{TARGET_CHANNEL}], q_01) "
            f"= {corr:.4f}, expected > 0.90. The targets do not line up with "
            f"the test.h5 rows — scoring against them would be meaningless.")

    if verbose:
        print(f"[targets] {csv_path.name}: {Y.shape[0]:,} x {Y.shape[1]}  "
              f"raw mm/h")
        print(f"[targets] median {np.median(Y):.5f}  mean {Y.mean():.5f}  "
              f"max {Y.max():.3f}  zeros {(Y == 0).mean() * 100:.2f}%")
        print(f"[targets] continuity corr(last_hist, q_01) = {corr:.4f}  ✓")

    return Y


# ─────────────────────────────────────────────────────────────────────
# 2.  Checkpoint loading
# ─────────────────────────────────────────────────────────────────────

def load_checkpoint(exp_dir: Path, device, verbose: bool = True):
    """
    Rebuild the model exactly as the run configured it and load best_model.pt.

    The config is read from the run's own config.json rather than from the
    live TRAIN_CFG, so editing the defaults later cannot retroactively change
    how an old checkpoint is reconstructed.
    """
    import torch
    from models.dmel import build_model

    ckpt_path = exp_dir / "best_model.pt"
    cfg_path  = exp_dir / "config.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")

    cfg = dict(TRAIN_CFG)
    if cfg_path.exists():
        with open(cfg_path) as f:
            saved = json.load(f)
        cfg.update(saved)
    else:
        print(f"[warn] {cfg_path} missing — falling back to current TRAIN_CFG. "
              f"If the run used non-default architecture flags this will "
              f"build the wrong model and fail to load the state dict.")

    model = build_model(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        ep = ckpt.get("epoch", "?")
        print(f"[model] {exp_dir.name}: {n_params:,} params, "
              f"checkpoint epoch {ep}")

    return model, cfg


# ─────────────────────────────────────────────────────────────────────
# 3.  Main
# ─────────────────────────────────────────────────────────────────────

def evaluate_on_test(exp_name: str,
                     save: bool = False,
                     batch_size: Optional[int] = None,
                     verbose: bool = True) -> dict:
    import torch
    from dataset  import RunoffDataset, make_loader, BasinNormalizer
    from evaluate import evaluate_model

    seed_everything(GLOBAL_SEED)
    device = get_device(verbose=False)

    exp_dir = EXP_DIR / exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"no such experiment: {exp_dir}")

    print("=" * 68)
    print(f"  FINAL TEST EVALUATION — {exp_name}")
    print("=" * 68)

    # ── 1. targets (loaded and validated BEFORE any prediction) ───────
    Y = load_test_targets(verbose=verbose)

    # ── 2. normalizer — the run's own, not a refit ────────────────────
    # Refitting here would change the input scaling the model was trained
    # under and quietly degrade it.
    norm_path = exp_dir / "norm.npz"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"no normalizer at {norm_path} — cannot reproduce the run's "
            f"input scaling, and refitting would not match training.")
    normalizer = BasinNormalizer().load(str(norm_path))
    print(f"[norm] loaded {norm_path.name}")

    # ── 3. model ──────────────────────────────────────────────────────
    model, cfg = load_checkpoint(exp_dir, device, verbose=verbose)

    # ── 4. test loader, now WITH targets ──────────────────────────────
    import h5py
    with h5py.File(TEST_H5, "r") as f:
        n_test = len(f["X"])

    rimf_cache = None
    if cfg.get("use_ceemdan"):
        from decompose import RIMFCache, cache_path
        p = cache_path(str(TEST_H5), cfg)
        if not Path(p).exists():
            raise FileNotFoundError(
                f"run used CEEMDAN but the test RIMF cache is missing: {p}")
        rimf_cache = RIMFCache(str(p))

    ds_test = RunoffDataset(
        str(TEST_H5),
        np.arange(n_test, dtype=np.int64),
        normalizer,
        has_targets    = False,       # test.h5 genuinely has no f["y"]
        external_y     = Y,           # ... the labels come from the CSV
        seq_len        = cfg.get("seq_len", 336),
        input_channels = cfg.get("input_channels"),
        rimf_cache     = rimf_cache,
        want_aux       = False,       # aux head is discarded at inference
    )
    loader = make_loader(
        ds_test,
        batch_size  = batch_size or cfg.get("batch_size", 64),
        shuffle     = False,
        num_workers = cfg.get("num_workers", 0),
    )

    # ── 5. score ──────────────────────────────────────────────────────
    print(f"\n[eval] scoring {n_test:,} test windows...")
    result = evaluate_model(
        model, loader, normalizer, device,
        split_name = "test",
        best_epoch = -1,
        cfg        = cfg,
        verbose    = verbose,
    )

    # ── 6. the comparison that matters ────────────────────────────────
    summary_path = exp_dir / "summary.json"
    prior = {}
    if summary_path.exists():
        with open(summary_path) as f:
            prior = json.load(f)

    test_nse   = float(np.median(result.nse_array()))
    sanval_nse = prior.get("sanval_median_nse")
    dev_nse    = prior.get("dev_median_nse")

    print("\n" + "=" * 68)
    print("  SPLIT COMPARISON  (median NSE, raw mm/h)")
    print("=" * 68)
    rows = [
        ("san_val", sanval_nse, "held-out BASINS — drove selection"),
        ("dev",     dev_nse,    "held-out TIME, known basins"),
        ("test",    test_nse,   "UNSEEN until now — final read-out"),
    ]
    for label, val, note in rows:
        shown = f"{val:.4f}" if isinstance(val, (int, float)) else "   n/a"
        print(f"  {label:<9} {shown}   {note}")

    if isinstance(sanval_nse, (int, float)):
        gap = test_nse - sanval_nse
        print(f"\n  test - san_val : {gap:+.4f}")
        if gap > 0.02:
            print("  → test scores HIGHER. Expected here: test basins all have "
                  "trained\n    embeddings/statistics, san_val basins are "
                  "genuinely unseen, so san_val\n    is the conservative bound.")
        elif gap < -0.02:
            print("  → test scores LOWER than the split used for selection. "
                  "Worth a look:\n    possible overfit to the selection split, "
                  "or a harder test period.")
        else:
            print("  → the two agree closely; selection generalised cleanly.")

    out = {
        "exp"                : exp_name,
        "test_median_nse"    : test_nse,
        "test_mean_nse"      : float(np.mean(result.nse_array())),
        "test_median_kge"    : float(np.median(result.kge_array())),
        "test_n_basins"      : int(result.n_basins_total),
        "test_horizon_nse"   : result.horizon_nse,
        "sanval_median_nse"  : sanval_nse,
        "dev_median_nse"     : dev_nse,
        "test_minus_sanval"  : (test_nse - sanval_nse
                                if isinstance(sanval_nse, (int, float))
                                else None),
        "targets_file"       : TEST_TARGETS_CSV.name,
    }

    if save:
        result.save(str(exp_dir / "metrics_test.json"))
        with open(exp_dir / "test_report.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[save] {exp_dir / 'metrics_test.json'}")
        print(f"[save] {exp_dir / 'test_report.json'}")
        # Deliberately NOT written into summary.json: that file is the record
        # of the run's own selection discipline, and folding a test number in
        # would invite it being compared against dev/san_val as a peer.

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score a finished checkpoint on test.h5 vs test_targets.csv")
    ap.add_argument("--exp", required=True,
                    help="experiment dir name under experiments/")
    ap.add_argument("--save", action="store_true",
                    help="write metrics_test.json + test_report.json")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    evaluate_on_test(args.exp, save=args.save,
                     batch_size=args.batch_size, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
