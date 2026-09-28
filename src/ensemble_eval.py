"""
src/ensemble_eval.py
────────────────────
Score one checkpoint, or the average of several, on dev / san_val / test with
the full evaluation report (evaluate.score_predictions).

Averaging happens in per-basin z-space, where every member was trained, and
uses the normalizer of the first member. The normalizer is fitted on the
split=0 rows of all 508 basins regardless of seed or holdout, so members must
agree on it — this is checked, not assumed.

Usage
─────
  # the free ensemble of the four existing loss runs
  python src/ensemble_eval.py --exps exp_001__baseline exp_009__huber \\
      exp_010__huber_d05 exp_012__mae --out experiments/ens_4loss_dev.json

  # one run, EMA weights
  python src/ensemble_eval.py --exps exp_050__resid_huber05 --ema

  # re-score a run in place and write its missing summary (exp_034)
  python src/ensemble_eval.py --exps exp_034__no_distil \\
      --override use_distil=false --write_into

  # FINAL test read-out — refuses to run without --confirm_test
  python src/ensemble_eval.py --exps final_s42 final_s123 ... \\
      --split test --confirm_test --out experiments/final_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (EXP_DIR, GLOBAL_SEED, TEST_H5, TRAIN_H5, SPLIT_VAL,
                    seed_everything, get_device)


def _parse_override(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        low = v.lower()
        if low in ("true", "false"):
            out[k] = low == "true"
        elif low in ("none", "null"):
            out[k] = None
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _dataset(split, cfg, normalizer, split_seed):
    """The rows a single run would have scored, built the same way."""
    import h5py
    from dataset import RunoffDataset

    rimf = None
    if cfg.get("use_ceemdan") or cfg.get("rimf_append"):
        from config import CEEMDAN_CFG
        from decompose import RIMFCache, cache_path
        # Older config.json files omit the CEEMDAN keys the cache key needs.
        ccfg = {**CEEMDAN_CFG, **cfg}
        rimf = RIMFCache(str(cache_path(
            str(TEST_H5 if split == "test" else TRAIN_H5), ccfg)))

    kw = dict(seq_len=cfg.get("seq_len", 336),
              input_channels=cfg.get("input_channels"),
              rimf_cache=rimf, want_aux=False)

    if split == "test":
        from evaluate_test import load_test_targets
        Y = load_test_targets(verbose=False)
        with h5py.File(TEST_H5, "r") as f:
            n = len(f["X"])
        return RunoffDataset(str(TEST_H5), np.arange(n, dtype=np.int64),
                             normalizer, has_targets=False, external_y=Y, **kw)

    if split == "dev":
        with h5py.File(TRAIN_H5, "r") as f:
            idx = np.where(f["split"][:] == SPLIT_VAL)[0].astype(np.int64)
    else:  # san_val
        from splits import build_group_split
        _, idx, _ = build_group_split(seed=split_seed, verbose=False)
    return RunoffDataset(str(TRAIN_H5), idx, normalizer, has_targets=True,
                         **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--split", default="dev",
                    choices=["dev", "san_val", "test"])
    ap.add_argument("--ema", action="store_true",
                    help="use best_model_ema.pt instead of best_model.pt")
    ap.add_argument("--override", nargs="*", default=[],
                    help="key=value config patches applied to every member")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    ap.add_argument("--write_into", action="store_true",
                    help="single member: write metrics_<split>.json and, if "
                         "missing, summary.json into its experiment dir")
    ap.add_argument("--confirm_test", action="store_true")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        print("Refusing to score test.h5 without --confirm_test. Test is a "
              "single final read-out; see config.TEST_TARGETS_CSV.")
        return 1

    import torch
    from dataset import BasinNormalizer, make_loader
    from evaluate import predict, score_predictions, dev_subsets
    from evaluate_test import load_checkpoint

    seed_everything(GLOBAL_SEED)
    device = get_device(verbose=False)
    overrides = _parse_override(args.override)
    ckpt_name = "best_model_ema.pt" if args.ema else "best_model.pt"

    normalizer, preds, ref = None, [], None
    for name in args.exps:
        exp_dir = EXP_DIR / name
        norm = BasinNormalizer().load(str(exp_dir / "norm.npz"))
        if normalizer is None:
            normalizer = norm
        elif not (np.allclose(norm.mean_, normalizer.mean_)
                  and np.allclose(norm.std_, normalizer.std_)):
            raise ValueError(f"{name}: normalizer differs from "
                             f"{args.exps[0]} — members are not comparable")

        model, cfg = load_checkpoint(exp_dir, device, ckpt_name=ckpt_name,
                                     overrides=overrides)
        split_seed = cfg.get("split_seed")
        split_seed = cfg.get("seed", GLOBAL_SEED) if split_seed is None \
            else split_seed
        ds = _dataset(args.split, cfg, norm, split_seed)
        loader = make_loader(ds, args.batch_size, shuffle=False)
        P = predict(model, loader, device, cfg, verbose=True, label=name)

        if ref is None:
            ref = P
        elif not np.array_equal(P["sample_id"], ref["sample_id"]):
            raise ValueError(f"{name}: scored different windows than "
                             f"{args.exps[0]} (san_val partitions differ?)")
        preds.append(P["pred"])
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    P = dict(ref)
    P["pred"] = np.mean(preds, axis=0).astype(np.float32)
    label = (args.exps[0] if len(args.exps) == 1
             else f"ensemble[{len(args.exps)}]")
    subsets = dev_subsets(P["sample_id"]) if args.split == "dev" else {}
    print(f"\n=== {label} on {args.split} ({ckpt_name}) ===")
    result = score_predictions(P, normalizer, split_name=args.split,
                               subsets=subsets, verbose=True)

    if args.out:
        result.save(args.out)
        with open(Path(args.out).with_suffix(".members.json"), "w") as f:
            json.dump({"members": args.exps, "checkpoint": ckpt_name,
                       "split": args.split, "overrides": overrides}, f,
                      indent=2)

    if args.write_into:
        if len(args.exps) != 1:
            raise ValueError("--write_into needs exactly one member")
        exp_dir = EXP_DIR / args.exps[0]
        suffix = "_ema" if args.ema else ""
        result.save(str(exp_dir / f"metrics_{args.split}{suffix}.json"))
        summ = exp_dir / "summary.json"
        if not summ.exists() and args.split == "dev" and not args.ema:
            hist = exp_dir / "best_metrics.json"
            best = json.load(open(hist)) if hist.exists() else {}
            json.dump({"name": args.exps[0], "overrides": overrides,
                       "dev_median_nse": result.median_nse,
                       "dev_horizon_nse": result.horizon_nse,
                       "dev_mean_nse_clip": result.mean_nse_clip,
                       "dev_subset_median_nse": result.subset_median_nse,
                       "best_sanval_nse": best.get("best_nse",
                                                   best.get("nse")),
                       "best_epoch": best.get("best_epoch",
                                              best.get("epoch")),
                       "note": "re-scored post hoc by ensemble_eval.py"},
                      open(summ, "w"), indent=2)
            print(f"[save] {summ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
