#!/usr/bin/env python3
"""
iter2_report.py
───────────────
Apply the iteration-2 decision rules (NEXT_ITERATION.md §4) to whatever
results exist so far. Safe to run at any time; missing runs show as "—".

Every comparison is SEED-MATCHED: seed s of a treatment against seed s of its
reference, which on the coupled split seed means the same san_val partition,
the same training basins, and (for s123/s177) the same machine.

  python iter2_report.py             # print the tables and verdicts
  python iter2_report.py --gate_r1   # exit 0 iff the R1 gate passes
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent / "experiments"
SEEDS = (42, 123, 177)

ARMS = {
    "mse":   "exp_001__baseline",
    "huber": "exp_010__huber_d05",
    "mae":   "exp_012__mae",
    "resid": "exp_050__resid_huber05",
}


def run_name(base: str, seed: int) -> str:
    return base if seed == 42 else f"{base}_s{seed}"


def metrics(name: str):
    """Extended dev metrics: the run's own file if it has the new fields,
    else a rescore, else the old file (median + buckets only)."""
    own = EXP / name / "metrics_dev.json"
    re_ = EXP / "rescore" / f"{name}__dev.json"
    for p in (own, re_):
        if p.exists():
            d = json.load(open(p))
            if d.get("skill_by_lead") or p == re_:
                return d
    return json.load(open(own)) if own.exists() else None


def summary(name: str):
    p = EXP / name / "summary.json"
    return json.load(open(p)) if p.exists() else None


def h25_48(d) -> float:
    h = d.get("horizon_nse", {})
    return float(np.mean([h["h25_36"], h["h37_48"]]))


def skill_h1_3(d):
    s = d.get("skill_by_lead") or []
    return float(np.mean(s[:3])) if len(s) >= 3 else None


def paired(treat: str, ref: str):
    """[(seed, d_dev, d_h25_48)] for every seed where both runs exist."""
    out = []
    for s in SEEDS:
        a, b = metrics(run_name(treat, s)), metrics(run_name(ref, s))
        if a and b:
            out.append((s, a["median_nse"] - b["median_nse"],
                        h25_48(a) - h25_48(b)))
    return out


def adopt(pairs) -> str:
    if len(pairs) < 3:
        return f"PENDING ({len(pairs)}/3 seed pairs)"
    d = np.array([p[1] for p in pairs])
    h = np.array([p[2] for p in pairs])
    ok = d.mean() >= 0.02 and (d > 0).all() and h.mean() >= 0
    return (f"{'ADOPT' if ok else 'REJECT'}  mean Δdev {d.mean():+.4f}, "
            f"all>0 {bool((d > 0).all())}, mean Δh25-48 {h.mean():+.4f}")


def fmt_pairs(pairs) -> str:
    return "  ".join(f"s{s}: {d:+.4f} (h25-48 {h:+.4f})"
                     for s, d, h in pairs) or "—"


def gate_r1():
    """A1 beats exp_010 by >= 0.01 on dev AND improves h1-3 skill."""
    a, b = metrics(ARMS["resid"]), metrics(ARMS["huber"])
    if not (a and b):
        return None, "exp_050 or exp_010 (s42) results missing"
    d = a["median_nse"] - b["median_nse"]
    sa, sb = skill_h1_3(a), skill_h1_3(b)
    if sa is None or sb is None:
        return None, "skill_by_lead missing (re-score exp_010 first)"
    ok = d >= 0.01 and sa > sb
    return ok, (f"Δdev {d:+.4f} (need ≥ +0.01), h1-3 skill "
                f"{sb:+.3f} → {sa:+.3f}")


def ema_votes():
    """Runs whose EMA 3-epoch-smoothed max san_val >= the raw one."""
    votes, total = 0, 0
    for p in sorted(EXP.glob("*/history.csv")):
        rows = list(csv.DictReader(open(p)))
        if not rows or not rows[0].get("sanval_nse_ema"):
            continue
        raw = np.array([float(r["sanval_nse"]) for r in rows])
        ema = np.array([float(r["sanval_nse_ema"]) for r in rows])
        if len(raw) < 3 or np.isnan(raw).all():
            continue
        sm = lambda v: np.convolve(v, np.ones(3) / 3, "valid").max()
        total += 1
        votes += int(sm(ema) >= sm(raw))
    return votes, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_r1", action="store_true")
    args = ap.parse_args()

    if args.gate_r1:
        ok, why = gate_r1()
        print(f"R1 gate: {ok}  {why}")
        return 0 if ok else 1

    print("\n" + "=" * 72)
    print("  ITERATION 2 — decision rules (NEXT_ITERATION.md §4)")
    print("=" * 72)

    print("\n  dev median NSE by arm and seed")
    print(f"  {'arm':<8}" + "".join(f"{'s'+str(s):>10}" for s in SEEDS)
          + f"{'mean':>10}")
    for arm, base in ARMS.items():
        vals = [metrics(run_name(base, s)) for s in SEEDS]
        nums = [v["median_nse"] for v in vals if v]
        cells = "".join(f"{v['median_nse']:>10.4f}" if v else f"{'—':>10}"
                        for v in vals)
        print(f"  {arm:<8}{cells}"
              f"{(np.mean(nums) if nums else float('nan')):>10.4f}")

    for label, treat, ref in (("Huber 0.5 vs MSE", "huber", "mse"),
                              ("MAE vs MSE", "mae", "mse"),
                              ("Residual vs Huber 0.5", "resid", "huber")):
        pr = paired(ARMS[treat], ARMS[ref])
        print(f"\n  {label}\n    {fmt_pairs(pr)}\n    → {adopt(pr)}")

    mae = [metrics(run_name(ARMS["mae"], s)) for s in SEEDS]
    hub = [metrics(run_name(ARMS["huber"], s)) for s in SEEDS]
    if all(mae) and all(hub):
        dm = np.mean([m["median_nse"] for m in mae]) - \
             np.mean([h["median_nse"] for h in hub])
        betas = [m.get("median_highflow_beta") for m in mae]
        beta = (float(np.nanmedian([b for b in betas if b is not None]))
                if any(b is not None for b in betas) else float("nan"))
        pick = "MAE" if dm >= 0.01 and beta >= 0.9 else "Huber 0.5"
        print(f"\n  MAE vs Huber 0.5: Δmean {dm:+.4f} (need ≥ +0.01), "
              f"MAE high-flow β {beta:.3f} (need ≥ 0.9) → {pick}")
    else:
        print("\n  MAE vs Huber 0.5: PENDING")

    ok, why = gate_r1()
    print(f"\n  R1 gate (run exp_050 on s123/s177?): "
          f"{'PASS' if ok else ('FAIL' if ok is False else 'PENDING')}  {why}")

    a, b = metrics("exp_052__rimf_append_huber05"), metrics(ARMS["huber"])
    sa, sb = summary("exp_052__rimf_append_huber05"), summary(ARMS["huber"])
    if a and b and sa and sb:
        dd = a["median_nse"] - b["median_nse"]
        ds = sa["sanval_median_nse"] - sb["sanval_median_nse"]
        close = dd < 0.02 and ds < 0.02
        print(f"\n  CEEMDAN (exp_052 vs exp_010): Δdev {dd:+.4f}, "
              f"Δsan_val {ds:+.4f} → "
              f"{'CLOSE the line (negative result)' if close else 'KEEP — signal'}")
    else:
        print("\n  CEEMDAN (exp_052 vs exp_010): PENDING")

    v, t = ema_votes()
    print(f"\n  EMA: smoothed EMA san_val ≥ raw in {v}/{t} runs "
          f"(adopt at ≥ 4) → {'ADOPT' if v >= 4 else 'PENDING/REJECT'}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
