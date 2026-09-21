#!/usr/bin/env python3
"""
results.py
──────────
Inspect experiment results. One command, several views.

    ./results.py                 # scoreboard, ranked by dev median NSE
    ./results.py --horizon       # add per-lead-time NSE columns
    ./results.py --gap           # split-discipline + gradient health audit
    ./results.py --show NAME     # everything about one experiment
    ./results.py --list          # what has run / is running / failed
    ./results.py --watch         # live-refresh the scoreboard
    ./results.py --wilcoxon      # significance vs the persistence baseline
    ./results.py --csv out.csv   # export the table

Every model is ranked against the PERSISTENCE baseline (median NSE 0.5251),
not against zero. A model at 0.52 has learned nothing beyond "the next two
days look like right now"; judged against 0.0 that would look like success.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
sys.path.insert(0, str(ROOT / "src"))

# Persistence, measured on dev by src/baselines.py. THE bar to beat.
PERSISTENCE_NSE = 0.5251
HORIZON_KEYS = ("h1_12", "h13_24", "h25_36", "h37_48")
PERSISTENCE_HORIZON = {"h1_12": 0.9382, "h13_24": 0.6865,
                       "h25_36": 0.4162, "h37_48": 0.2495}

C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m",
    "cyan": "\033[36m", "grey": "\033[90m",
}


def _c(s, colour: str, on: bool = True) -> str:
    return f"{C[colour]}{s}{C['reset']}" if on else str(s)


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────

def load_runs() -> List[dict]:
    """Collect every experiment folder, whether finished or not."""
    runs = []
    if not EXP.exists():
        return runs

    for d in sorted(EXP.iterdir()):
        if not d.is_dir():
            continue
        rec: Dict = {"name": d.name, "dir": d, "state": "unknown"}

        summ = d / "summary.json"
        devm = d / "metrics_dev.json"
        hist = d / "history.csv"

        if summ.exists():
            try:
                rec.update(json.load(open(summ)))
                rec["state"] = "done"
            except json.JSONDecodeError:
                rec["state"] = "corrupt"
        elif devm.exists():
            try:
                m = json.load(open(devm))
                rec["dev_median_nse"] = m.get("median_nse")
                rec["dev_mean_nse"] = m.get("mean_nse")
                rec["dev_median_kge"] = m.get("median_kge")
                rec["dev_pct_nse_05"] = m.get("pct_nse_05")
                rec["dev_pct_nse_07"] = m.get("pct_nse_07")
                rec["horizon_nse"] = m.get("horizon_nse", {})
                rec["state"] = "done"
            except json.JSONDecodeError:
                rec["state"] = "corrupt"
        elif hist.exists():
            rec["state"] = "running"
            try:
                rows = list(csv.DictReader(open(hist)))
                if rows:
                    rec["epochs_run"] = len(rows)
                    rec["last_sanval_nse"] = float(rows[-1]["sanval_nse"])
                    mtime = hist.stat().st_mtime
                    rec["stale_min"] = (time.time() - mtime) / 60
            except Exception:
                pass
        else:
            rec["state"] = "empty"

        if "horizon_nse" not in rec:
            hn = rec.get("dev_horizon_nse")
            rec["horizon_nse"] = hn if isinstance(hn, dict) else {}

        runs.append(rec)
    return runs


# ─────────────────────────────────────────────────────────────────────
# Views
# ─────────────────────────────────────────────────────────────────────

def view_scoreboard(runs, horizon=False, colour=True, baseline_nse=None):
    done = [r for r in runs if r["state"] == "done"
            and r.get("dev_median_nse") is not None]
    done.sort(key=lambda r: -r["dev_median_nse"])

    bar = baseline_nse if baseline_nse is not None else PERSISTENCE_NSE

    if horizon:
        hdr = (f"  {'#':>3}  {'experiment':<34} {'medNSE':>8} {'vs pers':>8}  "
               + "".join(f"{k:>9}" for k in HORIZON_KEYS))
    else:
        hdr = (f"  {'#':>3}  {'experiment':<34} {'medNSE':>8} {'vs pers':>8} "
               f"{'meanNSE':>8} {'KGE':>7} {'>0.7%':>7} {'ep':>4} {'min':>6}")

    print()
    print(_c("  DEV SCOREBOARD", "bold", colour)
          + _c(f"   (persistence baseline = {bar:.4f})", "grey", colour))
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for i, r in enumerate(done, 1):
        nse = r["dev_median_nse"]
        delta = nse - bar
        is_base = r.get("is_baseline", False)

        name = r["name"][:34].ljust(34)
        if is_base:
            name = _c(name, "grey", colour)

        dcol = "green" if delta > 0.01 else ("red" if delta < -0.01 else "yellow")
        # Pad BEFORE colouring: ANSI escapes count toward f-string width and
        # would otherwise shift every column to the right of this one.
        dstr = _c(f"{delta:+.4f}".rjust(8), dcol, colour)

        if horizon:
            cells = ""
            for k in HORIZON_KEYS:
                v = (r.get("horizon_nse") or {}).get(k)
                cells += f"{v:>9.4f}" if isinstance(v, (int, float)) else f"{'—':>9}"
            print(f"  {i:>3}  {name} {nse:>8.4f} {dstr}  {cells}")
        else:
            mean = r.get("dev_mean_nse")
            kge = r.get("dev_median_kge")
            p07 = r.get("dev_pct_nse_07")
            ep = r.get("best_epoch")
            mn = r.get("total_minutes")
            print(f"  {i:>3}  {name} {nse:>8.4f} {dstr} "
                  f"{_f(mean, 8, 4)} {_f(kge, 7, 3)} {_f(p07, 6, 1)}% "
                  f"{_i(ep, 4)} {_f(mn, 6, 1)}")

    if not done:
        print(_c("  no completed experiments yet", "grey", colour))
    print()


def _f(v, w, p):
    return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'—':>{w}}"


def _i(v, w):
    return f"{v:>{w}}" if isinstance(v, int) else f"{'—':>{w}}"


def view_gap(runs, colour=True):
    """Split-discipline and gradient-health audit."""
    done = [r for r in runs if r["state"] == "done"
            and not r.get("is_baseline")]
    done.sort(key=lambda r: -(r.get("dev_median_nse") or -9e9))

    print()
    print(_c("  SPLIT DISCIPLINE & TRAINING HEALTH", "bold", colour))
    print(f"  {'experiment':<30} {'sel':>7} {'dev':>8} {'gap':>8} "
          f"{'gnorm':>7} {'clip':>5} {'ep':>4} {'stop':>5}")
    print("  " + "─" * 84)
    for r in done:
        sel = r.get("sanval_median_nse")
        dev = r.get("dev_median_nse")
        gap = r.get("sanval_to_dev_gap")
        gn = r.get("mean_grad_norm")
        cl = r.get("clip_grad")

        warn = ""
        if isinstance(gn, (int, float)) and isinstance(cl, (int, float)) and cl > 0:
            if gn > cl:
                warn = _c(" clipper fires every step", "yellow", colour)
            elif gn < 0.1 * cl:
                warn = _c(" clipping is a no-op", "grey", colour)

        print(f"  {r['name'][:30]:<30} {_f(sel,7,3)} {_f(dev,8,4)} "
              f"{_f(gap,8,4)} {_f(gn,7,3)} {_f(cl,5,1)} "
              f"{_i(r.get('best_epoch'),4)} "
              f"{str(r.get('early_stopped','—')):>5}{warn}")

    print()
    print(_c("  Reading this table", "bold", colour))
    print("    sel/dev  : selection split vs report split. MUST differ.")
    print("    gap      : dev - san_val. Large negative = the checkpoint was")
    print("               chosen on selection-split noise and did not transfer.")
    print("    gnorm    : mean pre-clip gradient norm. Compare to clip.")
    print("               > clip  -> every update is being rescaled, so the")
    print("                          effective lr is lower than configured.")
    print("               << clip -> clipping never fires; it is pure overhead.")
    print()
    print(_c("  Caveat that applies to every dev number above:", "yellow", colour))
    print("    78.8% of dev windows overlap a training window (measured, 6-gram")
    print("    stride-1). That leak is in the competition's own split=1 and is")
    print("    not fixable. dev NSE is therefore optimistic in absolute terms;")
    print("    it remains valid for RANKING, since every model pays it equally.")
    print()


def view_list(runs, colour=True):
    order = {"running": 0, "done": 1, "empty": 2, "corrupt": 3, "unknown": 4}
    runs = sorted(runs, key=lambda r: (order.get(r["state"], 9), r["name"]))

    print()
    print(_c("  EXPERIMENT INVENTORY", "bold", colour))
    print(f"  {'state':<9} {'experiment':<36} {'detail'}")
    print("  " + "─" * 78)
    for r in runs:
        st = r["state"]
        col = {"done": "green", "running": "cyan", "corrupt": "red",
               "empty": "grey"}.get(st, "grey")
        if st == "done":
            nse = r.get("dev_median_nse")
            detail = (f"dev medNSE {nse:.4f}" if isinstance(nse, float)
                      else "no metrics")
            if isinstance(nse, float):
                detail += f"   ({nse - PERSISTENCE_NSE:+.4f} vs persistence)"
        elif st == "running":
            ep = r.get("epochs_run", 0)
            sv = r.get("last_sanval_nse")
            stale = r.get("stale_min", 0)
            detail = f"epoch {ep}"
            if isinstance(sv, float):
                detail += f", san_val NSE {sv:.4f}"
            if stale > 30:
                detail += _c(f"  STALE {stale:.0f} min", "red", colour)
        elif st == "empty":
            detail = "folder exists, nothing written"
        else:
            detail = st
        print(f"  {_c(f'{st:<9}', col, colour)} {r['name'][:36]:<36} {detail}")

    n = {k: sum(1 for r in runs if r["state"] == k)
         for k in ("done", "running", "empty", "corrupt")}
    print()
    print(f"  {n['done']} done · {n['running']} running · "
          f"{n['empty']} empty · {n['corrupt']} corrupt")
    print()


def view_show(runs, name, colour=True):
    match = [r for r in runs if r["name"] == name]
    if not match:
        match = [r for r in runs if name in r["name"]]
    if not match:
        print(f"  no experiment matching {name!r}")
        return
    r = match[0]

    print()
    print(_c(f"  {r['name']}", "bold", colour) + f"   [{r['state']}]")
    print("  " + "─" * 66)

    groups = [
        ("headline", ["dev_median_nse", "dev_mean_nse", "dev_median_kge",
                      "dev_pct_nse_05", "dev_pct_nse_07"]),
        ("split discipline", ["selection_split", "report_split",
                             "split_method", "selection_measures",
                             "n_train_basins", "n_val_basins",
                             "sanval_median_nse", "sanval_to_dev_gap"]),
        ("training", ["best_epoch", "epochs_run", "epochs_max",
                      "early_stopped", "mean_grad_norm", "clip_grad",
                      "final_lr", "final_train_loss", "total_minutes"]),
        ("config", ["optimizer", "lr", "weight_decay", "scheduler", "loss",
                    "huber_delta", "nse_eps", "batch_size", "norm_strategy",
                    "attention", "aux_task", "aux_loss_weight",
                    "use_basin_emb", "use_ceemdan", "ensemble_method",
                    "n_params", "patience_requested", "patience_used"]),
    ]
    for title, keys in groups:
        present = [(k, r[k]) for k in keys if k in r and r[k] is not None]
        if not present:
            continue
        print(f"\n  {_c(title, 'cyan', colour)}")
        for k, v in present:
            if isinstance(v, float):
                v = f"{v:.6g}"
            print(f"    {k:<22} {v}")

    hn = r.get("horizon_nse") or {}
    if hn:
        print(f"\n  {_c('NSE by lead time', 'cyan', colour)}")
        print(f"    {'bucket':<10} {'model':>9} {'persist':>9} {'delta':>9}")
        for k in HORIZON_KEYS:
            if k in hn:
                p = PERSISTENCE_HORIZON[k]
                d = hn[k] - p
                col = "green" if d > 0 else "red"
                print(f"    {k:<10} {hn[k]:>9.4f} {p:>9.4f} "
                      f"{_c(f'{d:+9.4f}', col, colour)}")

    ov = r.get("overrides")
    if ov:
        print(f"\n  {_c('overrides', 'cyan', colour)}")
        for k, v in (ov.items() if isinstance(ov, dict) else []):
            print(f"    {k:<22} {v}")

    h = r["dir"] / "history.csv"
    if h.exists():
        rows = list(csv.DictReader(open(h)))
        if rows:
            print(f"\n  {_c('training curve (san_val NSE)', 'cyan', colour)}")
            vals = [float(x["sanval_nse"]) for x in rows]
            lo, hi = min(vals), max(vals)
            spark = "".join(
                " ▁▂▃▄▅▆▇█"[min(8, int((v - lo) / (hi - lo + 1e-9) * 8))]
                for v in vals)
            print(f"    {spark}")
            print(f"    epoch 0 → {len(vals)-1}   range [{lo:.4f}, {hi:.4f}]")
    print()


def view_wilcoxon(runs, baseline="baseline_persistence", colour=True):
    from evaluate import EvalResult, wilcoxon_nse

    base_dir = EXP / baseline / "metrics_dev.json"
    if not base_dir.exists():
        print(f"  baseline {baseline!r} not found. Run: "
              f"python src/baselines.py")
        return
    base = EvalResult.load(str(base_dir))

    done = [r for r in runs if r["state"] == "done"
            and not r.get("is_baseline")
            and (r["dir"] / "metrics_dev.json").exists()]
    done.sort(key=lambda r: -(r.get("dev_median_nse") or -9e9))

    print()
    print(_c(f"  WILCOXON SIGNED-RANK vs {baseline}", "bold", colour))
    print(f"  paired on per-basin NSE · alpha = 0.05")
    print(f"  {'experiment':<34} {'delta':>9} {'p':>10} {'significant':>12}")
    print("  " + "─" * 70)
    for r in done:
        res = EvalResult.load(str(r["dir"] / "metrics_dev.json"))
        w = wilcoxon_nse(res, base)
        if "error" in w:
            continue
        sig = w["significant"]
        print(f"  {r['name'][:34]:<34} {w['delta_median']:>+9.4f} "
              f"{w['p_value']:>10.2e} "
              f"{_c('YES' if sig else 'no', 'green' if sig else 'grey', colour):>12}")
    print()


def export_csv(runs, path):
    done = [r for r in runs if r["state"] == "done"]
    cols = ["name", "dev_median_nse", "dev_mean_nse", "dev_median_kge",
            "dev_pct_nse_05", "dev_pct_nse_07", "sanval_median_nse",
            "sanval_to_dev_gap", "best_epoch", "epochs_run",
            "mean_grad_norm", "clip_grad", "loss", "lr", "optimizer",
            "scheduler", "batch_size", "attention", "aux_task",
            "use_basin_emb", "use_ceemdan", "n_params", "total_minutes"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + list(HORIZON_KEYS))
        for r in done:
            hn = r.get("horizon_nse") or {}
            w.writerow([r.get(c, "") for c in cols]
                       + [hn.get(k, "") for k in HORIZON_KEYS])
    print(f"  wrote {len(done)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Inspect DMEL experiment results.")
    ap.add_argument("--horizon", action="store_true",
                    help="show per-lead-time NSE instead of summary columns")
    ap.add_argument("--gap", action="store_true",
                    help="split-discipline and gradient-health audit")
    ap.add_argument("--list", action="store_true",
                    help="inventory: done / running / failed")
    ap.add_argument("--show", metavar="NAME",
                    help="full detail for one experiment")
    ap.add_argument("--wilcoxon", action="store_true",
                    help="significance test vs the persistence baseline")
    ap.add_argument("--baseline", default="baseline_persistence")
    ap.add_argument("--watch", action="store_true",
                    help="refresh every 30s")
    ap.add_argument("--csv", metavar="PATH", help="export table to CSV")
    ap.add_argument("--no-colour", action="store_true")
    a = ap.parse_args()

    colour = not a.no_colour and sys.stdout.isatty()

    def render():
        runs = load_runs()
        if a.csv:
            export_csv(runs, a.csv)
            return
        if a.show:
            view_show(runs, a.show, colour)
            return
        if a.list:
            view_list(runs, colour)
            return
        if a.gap:
            view_gap(runs, colour)
            return
        if a.wilcoxon:
            view_wilcoxon(runs, a.baseline, colour)
            return
        view_scoreboard(runs, horizon=a.horizon, colour=colour)

    if a.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")
                print(_c(f"  watching · {time.strftime('%H:%M:%S')}"
                         f" · ctrl-c to stop", "grey", colour))
                render()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n  stopped.")
    else:
        render()


if __name__ == "__main__":
    main()
