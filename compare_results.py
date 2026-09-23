#!/usr/bin/env python3
"""
compare_results.py
──────────────────
Reads all experiments/*/metrics_dev.json and prints a ranked scoreboard.

Usage
─────
  # Full scoreboard
  python compare_results.py

  # Filter by prefix
  python compare_results.py --filter exp_0

  # Show KGE components alongside NSE
  python compare_results.py --kge

  # Show selection->report generalisation gap and gradient health
  python compare_results.py --gap

  # Run Wilcoxon test: compare each experiment against the baseline
  python compare_results.py --wilcoxon --baseline exp_001__baseline
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evaluate import EvalResult, wilcoxon_nse
from config import EXP_DIR


def load_all(filt: str = "") -> dict:
    results = {}
    for p in sorted(EXP_DIR.glob("*/metrics_dev.json")):
        name = p.parent.name
        if filt and filt not in name:
            continue
        try:
            results[name] = EvalResult.load(str(p))
        except Exception as e:
            print(f"  [warn] Could not load {p}: {e}")
    return results


def load_summaries(names) -> dict:
    """Load summary.json per experiment (absent on runs predating it)."""
    out = {}
    for name in names:
        p = EXP_DIR / name / "summary.json"
        if p.exists():
            try:
                out[name] = json.loads(p.read_text())
            except Exception:
                pass
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter",   default="",
                        help="Only show experiments whose name contains this string")
    parser.add_argument("--kge",      action="store_true",
                        help="Show KGE alongside NSE")
    parser.add_argument("--gap",      action="store_true",
                        help="Show san_val->dev gap and mean gradient norm")
    parser.add_argument("--wilcoxon", action="store_true",
                        help="Run Wilcoxon test vs baseline")
    parser.add_argument("--horizon", action="store_true",
                        help="Show median NSE per lead-time bucket "
                             "(h1_12 / h13_24 / h25_36 / h37_48)")
    parser.add_argument("--rank_by", default="median_nse",
                        choices=["median_nse", "h37_48"],
                        help="Ranking metric. h37_48 is the 37-48h lead-time "
                             "bucket — the largest margin over persistence "
                             "(+0.242 vs +0.079 at h1_12) and where method "
                             "differences actually show up, since h1_12 is "
                             "saturated by persistence.")
    parser.add_argument("--baseline", default="exp_001__baseline",
                        help="Baseline experiment name for Wilcoxon")
    args = parser.parse_args()

    results = load_all(args.filter)
    if not results:
        print(f"  No results found in {EXP_DIR}")
        sys.exit(0)

    summaries = load_summaries(results.keys())

    # Sort by the chosen ranking metric.
    #
    # median_nse is the headline, but it is dominated by the short lead times
    # that persistence already solves (h1_12 sits at ~0.92 for every model).
    # Ranking by h37_48 instead surfaces the experiments that actually improve
    # the forecast where the contribution lives.
    def _h37(res) -> float:
        # Missing bucket sorts last rather than crashing: older runs predate
        # horizon_nse, and a scoreboard must still print.
        return (res.horizon_nse or {}).get("h37_48", float("-inf"))

    if args.rank_by == "h37_48":
        ranked = sorted(results.items(), key=lambda x: _h37(x[1]),
                        reverse=True)
    else:
        ranked = sorted(results.items(),
                        key=lambda x: x[1].median_nse, reverse=True)

    baseline_result = results.get(args.baseline)

    # ── Header ────────────────────────────────────────────────────────
    kge_col = "  med KGE" if args.kge else ""
    gap_col = f"  {'gap':>7}  {'gnorm':>5}" if args.gap else ""
    w_col   = "  p-val  sig" if args.wilcoxon else ""
    hz_col  = (f"  {'h1_12':>7}  {'h13_24':>7}  {'h25_36':>7}  {'h37_48':>7}"
               if args.horizon else "")
    print(f"\n  {'Rank':>4}  {'Experiment':<42}  "
          f"{'med NSE':>8}  {'mean NSE':>9}  {'>0.7%':>6}"
          f"  {'ep':>3}{kge_col}{hz_col}{gap_col}{w_col}")
    print("  " + "-" * (75 + (10 if args.kge else 0)
                           + (36 if args.horizon else 0)
                           + (16 if args.gap else 0)
                           + (12 if args.wilcoxon else 0)))

    for rank, (name, res) in enumerate(ranked, 1):
        kge_str = f"  {res.median_kge:>8.4f}" if args.kge else ""

        hz_str = ""
        if args.horizon:
            hz = res.horizon_nse or {}
            for b in ("h1_12", "h13_24", "h25_36", "h37_48"):
                v = hz.get(b)
                hz_str += f"  {v:>7.4f}" if v is not None else f"  {'-':>7}"

        gap_str = ""
        if args.gap:
            s = summaries.get(name, {})
            g = s.get("sanval_to_dev_gap")
            n = s.get("mean_grad_norm")
            gap_str  = f"  {g:>+7.4f}" if g is not None else f"  {'-':>7}"
            gap_str += f"  {n:>5.3f}"  if n is not None else f"  {'-':>5}"

        w_str = ""
        if args.wilcoxon and baseline_result and name != args.baseline:
            w = wilcoxon_nse(res, baseline_result)
            sig = "YES" if w.get("significant") else " no"
            pv  = w.get("p_value", 1.0)
            w_str = f"  {pv:>6.4f}  {sig}"

        marker = "  <- baseline" if name == args.baseline else ""
        print(
            f"  {rank:>4}  {name:<42}  "
            f"{res.median_nse:>8.4f}  {res.mean_nse:>9.4f}  "
            f"{res.pct_nse_07:>5.1f}%  "
            f"{res.best_epoch:>3}"
            f"{kge_str}{hz_str}{gap_str}{w_str}{marker}"
        )

    print(f"\n  {len(results)} experiments loaded from {EXP_DIR}")

    if baseline_result:
        best_name, best_res = ranked[0]
        delta = best_res.median_nse - baseline_result.median_nse
        print(f"  Best vs baseline: {delta:+.4f} NSE  ({best_name})")

        if args.horizon:
            b37 = _h37(baseline_result)
            d37 = _h37(best_res) - b37
            if b37 != float("-inf"):
                print(f"  Best vs baseline @ h37_48: {d37:+.4f} NSE")

    if args.horizon:
        print("\n  Lead-time buckets = median NSE over that slice of the 48h")
        print("          forecast, pooled per basin then medianed across")
        print("          basins. Each bucket has its OWN variance denominator,")
        print("          so the four do NOT average to med NSE.")
        print("  h37_48  the bucket that matters. Margin over persistence is")
        print("          +0.242 there vs +0.079 at h1_12, where persistence")
        print("          alone already scores ~0.84 and every model saturates")
        print("          near 0.92. Rank with --rank_by h37_48.")
        print("  CAUTION single seed (42) per experiment: a gap under ~0.01 is")
        print("          not distinguishable from seed noise. Shortlist here,")
        print("          then multi-seed the finalists before claiming an effect.")

    if args.gap:
        print("\n  gap   = dev median NSE - san_val median NSE")
        print("          strongly negative => the model was selected on")
        print("          san_val noise and does not transfer to dev")
        print("  gnorm = mean pre-clip gradient L2 norm over training")
        print("          compare to clip_grad: if close, the clipper is")
        print("          rescaling every step and changing the effective lr")
    print()


if __name__ == "__main__":
    main()
