#!/usr/bin/env python3
"""
make_report.py
──────────────
Figures + tables for the report, from the saved result files only (no model
is run here). Re-run any time; it uses whatever results exist.

Inputs
  experiments/test/<run>.json, <run>__dev.json   (run_test_readout.py)
  experiments/rescore/*.json, experiments/*/metrics_dev.json, history.csv

Outputs
  reports/figures/fig*.png|pdf     static, light surface, for the paper
  reports/tables/*.csv             the table twin of every figure
  reports/RESULTS.md               the tables as markdown

Chart conventions: validated categorical palette (colour-blind-safe order),
one y-axis per panel, 2 px lines, >= 8 px markers with a surface ring,
hairline solid grid, persistence always drawn as the grey reference, and
direct labels on the few marks that matter; every value is also in the CSV.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
TEST = EXP / "test"
OUT = ROOT / "reports"
FIG, TAB = OUT / "figures", OUT / "tables"

# ── palette (reference instance, validated light mode) ────────────────
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
REF = MUTED            # persistence reference: grey, never a series colour
MARKERS = ["o", "s", "D", "^", "v"]

BUCKETS = ["h1_12", "h13_24", "h25_36", "h37_48"]
BUCKET_LABELS = ["1–12 h", "13–24 h", "25–36 h", "37–48 h"]

LABELS = {
    "exp_001__baseline": "Informer, MSE (baseline)",
    "exp_009__huber": "Huber δ=1",
    "exp_010__huber_d05": "Huber δ=0.5",
    "exp_012__mae": "MAE",
    "exp_050__resid_huber05": "Residual output + Huber 0.5",
    "exp_051__resid_mse": "Residual output + MSE",
    "exp_052__rimf_append_huber05": "CEEMDAN RIMFs as inputs",
    "exp_053__lstm_huber05": "LSTM + Huber 0.5",
    "exp_020__dmel": "DMEL (fixed split)",
    "exp_023__dmel_se_threshold": "DMEL (SE threshold)",
    "exp_002__global_norm": "Global z-score",
    "exp_005__aux_task": "Auxiliary forcing task",
    "exp_008__seq168": "History 168 h",
    "exp_011__nse_loss": "NSE loss",
    "exp_015__dropout02": "Dropout 0.2",
    "exp_038__no_clip": "No gradient clipping",
    "exp_034__no_distil": "No distillation",
    "final_ens9": "FINAL: 9-model ensemble",
    "final_lstm_huber05_ens3": "Final LSTM ×3",
    "final_resid_huber05_ens3": "Final residual Informer ×3",
    "final_mae_ens3": "Final MAE Informer ×3",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def label(run: str) -> str:
    for sfx in ("_s123", "_s177"):
        if run.endswith(sfx):
            return f"{LABELS.get(run[:-len(sfx)], run)} ({sfx[1:]})"
    return LABELS.get(run, run)


def load(path: Path):
    return json.load(open(path)) if path.exists() else None


def pairs():
    """{run: (dev, test)} for every run with both read-outs."""
    out = {}
    for t in sorted(TEST.glob("*.json")):
        if t.name.endswith(("__dev.json", ".members.json")):
            continue
        run = t.stem
        d = load(TEST / f"{run}__dev.json")
        if d:
            out[run] = (d, load(t))
    return out


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote reports/figures/{name}.png|pdf")


def write_csv(name, header, rows):
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────
# Fig 1 — dev vs test, every scored model (dumbbell)
# ─────────────────────────────────────────────────────────────────────

def fig_scoreboard(P):
    runs = [r for r in P if not r.endswith(("_s123", "_s177"))]
    runs.sort(key=lambda r: P[r][1]["median_nse"])
    dev = [P[r][0]["median_nse"] for r in runs]
    tst = [P[r][1]["median_nse"] for r in runs]
    pers_d = P[runs[0]][0]["median_nse_persistence"]
    pers_t = P[runs[0]][1]["median_nse_persistence"]

    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(runs) + 1.4))
    y = np.arange(len(runs))
    for yi, a, b in zip(y, dev, tst):
        ax.plot([a, b], [yi, yi], color=AXIS, lw=1.5, zorder=1)
    ax.scatter(dev, y, s=64, color=S1, edgecolor=SURFACE, lw=2, zorder=3,
               label="dev (split=1)", marker="o")
    ax.scatter(tst, y, s=64, color=S2, edgecolor=SURFACE, lw=2, zorder=3,
               label="test", marker="s")
    ax.axvline(pers_t, color=REF, lw=1.2, zorder=0)
    ax.text(pers_t + 0.003, -0.9, f"persistence (test) {pers_t:.3f}",
            color=INK2, fontsize=8, va="center")
    ax.set_ylim(-1.4, len(runs) - 0.5)
    ax.set_yticks(y, [label(r) for r in runs])
    ax.tick_params(axis="y", colors=INK2, length=0)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("FINAL"):
            t.set_fontweight("bold")
            t.set_color(INK)
    for yi, r, a, b in zip(y, runs, dev, tst):
        if r in ("final_ens9", "exp_001__baseline"):
            ax.text(max(a, b) + 0.006, yi, f"test {b:.3f}", va="center",
                    fontsize=8, color=INK)
    ax.set_xlabel("median per-basin NSE (higher is better)")
    ax.set_title("Every scored model: dev vs test")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig1_dev_vs_test")

    write_csv("fig1_dev_vs_test",
              ["run", "label", "dev_median_nse", "test_median_nse",
               "test_minus_dev", "dev_persistence", "test_persistence"],
              [[r, label(r), f"{a:.4f}", f"{b:.4f}", f"{b - a:+.4f}",
                f"{pers_d:.4f}", f"{pers_t:.4f}"]
               for r, a, b in zip(runs, dev, tst)])


# ─────────────────────────────────────────────────────────────────────
# Fig 2 — NSE by lead-time bucket, dev | test small multiples
# ─────────────────────────────────────────────────────────────────────

KEY = [("exp_001__baseline", S1), ("exp_012__mae", S2),
       ("exp_050__resid_huber05", S3), ("exp_053__lstm_huber05", S4),
       ("final_ens9", S5)]


def fig_horizon(P):
    key = [(r, c) for r, c in KEY if r in P]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    rows = []
    x = np.arange(len(BUCKETS))
    for ax, split, idx in zip(axes, ("dev", "test"), (0, 1)):
        ref = P[key[0][0]][idx]["horizon_nse_persistence"]
        ax.plot(x, [ref[b] for b in BUCKETS], color=REF, lw=2,
                marker="o", ms=5, label="persistence")
        for i, (r, c) in enumerate(key):
            h = P[r][idx]["horizon_nse"]
            ax.plot(x, [h[b] for b in BUCKETS], color=c,
                    marker=MARKERS[i], ms=8, mec=SURFACE, mew=2,
                    label=label(r))
            rows.append([split, r, label(r)] + [f"{h[b]:.4f}" for b in BUCKETS])
        rows.append([split, "persistence", "persistence"]
                    + [f"{ref[b]:.4f}" for b in BUCKETS])
        ax.set_xticks(x, BUCKET_LABELS)
        ax.set_title(split)
        ax.set_xlabel("forecast lead time")
    axes[0].set_ylabel("median per-basin NSE")
    handles = [Line2D([], [], color=REF, lw=2, marker="o", ms=5,
                      label="persistence")]
    handles += [Line2D([], [], color=c, lw=2, marker=MARKERS[i], ms=7,
                       label=label(r)) for i, (r, c) in enumerate(key)]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Skill decays with lead time; the models' value is at 13–48 h",
                 x=0.06, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_nse_by_lead_bucket")
    write_csv("fig2_nse_by_lead_bucket", ["split", "run", "label"] + BUCKETS,
              rows)


# ─────────────────────────────────────────────────────────────────────
# Fig 3 — skill vs persistence per lead hour (test)
# ─────────────────────────────────────────────────────────────────────

def fig_skill(P):
    key = [(r, c) for r, c in KEY if r in P]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    lead = np.arange(1, 49)
    lo = -3.0
    rows = []
    for i, (r, c) in enumerate(key):
        s = np.array(P[r][1]["skill_by_lead"])
        ax.plot(lead, np.maximum(s, lo), color=c,
                label=f"{label(r)}   (h+1: {s[0]:+.1f})")
        ax.plot(lead[-1], s[-1], marker=MARKERS[i], ms=8, color=c,
                mec=SURFACE, mew=2)
        rows.append([r, label(r)] + [f"{v:.4f}" for v in s])
    ax.axhline(0, color=REF, lw=1.2)
    ax.text(48.5, 0.02, "= persistence", color=INK2, fontsize=8, va="bottom",
            ha="right")
    ax.set_ylim(lo, 0.6)
    ax.set_xlim(0.5, 48.5)
    ax.set_xlabel("lead hour")
    ax.set_ylabel("skill vs persistence\n1 − MSE / MSE(persistence)")
    ax.set_title("Test: skill against persistence by lead hour")
    ax.text(12, lo + 0.08, "axis clipped at −3; exact h+1 values in legend",
            fontsize=7.5, color=INK2, va="bottom")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig3_skill_by_lead_hour")
    write_csv("fig3_skill_by_lead_hour",
              ["run", "label"] + [f"h{i}" for i in lead], rows)


# ─────────────────────────────────────────────────────────────────────
# Fig 4 — seed robustness of the loss choice (dev; test where synced)
# ─────────────────────────────────────────────────────────────────────

ARMS = [("MSE", "exp_001__baseline", S1), ("Huber δ=0.5", "exp_010__huber_d05", S3),
        ("MAE", "exp_012__mae", S2)]


def _dev_of(run):
    for p in (EXP / "rescore" / f"{run}__dev.json", TEST / f"{run}__dev.json",
              EXP / run / "metrics_dev.json"):
        d = load(p)
        if d:
            return d["median_nse"]
    return None


def fig_seeds(P):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rows = []
    for i, (name, base, c) in enumerate(ARMS):
        for j, (split, off, mk) in enumerate((("dev", -0.12, "o"),
                                             ("test", 0.12, "s"))):
            vals = []
            for s in (42, 123, 177):
                run = base if s == 42 else f"{base}_s{s}"
                v = (_dev_of(run) if split == "dev"
                     else (P[run][1]["median_nse"] if run in P else None))
                if v is not None:
                    vals.append(v)
                    rows.append([name, split, s, f"{v:.4f}"])
            if not vals:
                continue
            xs = np.full(len(vals), i + off)
            ax.scatter(xs, vals, s=64, color=c, marker=mk, edgecolor=SURFACE,
                       lw=2, zorder=3)
            m = float(np.mean(vals))
            if len(vals) > 1:
                ax.plot([i + off - 0.08, i + off + 0.08], [m, m], color=INK,
                        lw=2, zorder=2)
                ax.text(i + off + 0.1, m, f"mean {m:.3f} (n={len(vals)})",
                        fontsize=8, va="center", color=INK2)
            else:
                ax.text(i + off + 0.1, m, f"{m:.3f} (s42)", fontsize=8,
                        va="center", color=INK2)
    ax.set_xticks(range(len(ARMS)), [a[0] for a in ARMS])
    ax.scatter([], [], marker="o", color=MUTED, label="dev, per seed")
    ax.scatter([], [], marker="s", color=MUTED, label="test, per seed")
    ax.plot([], [], color=INK, lw=2, label="mean over seeds")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-0.4, len(ARMS) - 0.1)
    ax.set_ylabel("median per-basin NSE")
    ax.set_title("Loss choice across seeds 42 / 123 / 177")
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_loss_seeds")
    write_csv("fig4_loss_seeds", ["loss", "split", "seed", "median_nse"], rows)


# ─────────────────────────────────────────────────────────────────────
# Fig 5 — training curves (san_val, seed 42)
# ─────────────────────────────────────────────────────────────────────

def fig_curves():
    runs = [("exp_001__baseline", S1), ("exp_012__mae", S2),
            ("exp_050__resid_huber05", S3), ("exp_053__lstm_huber05", S4),
            ("exp_020__dmel", S5)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    rows = []
    for i, (r, c) in enumerate(runs):
        p = EXP / r / "history.csv"
        if not p.exists():
            continue
        h = list(csv.DictReader(open(p)))
        ep = [int(x["epoch"]) + 1 for x in h]
        sv = [float(x["sanval_nse"]) for x in h]
        ax.plot(ep, sv, color=c, label=label(r))
        b = int(np.argmax(sv))
        ax.plot(ep[b], sv[b], marker=MARKERS[i], ms=8, color=c, mec=SURFACE,
                mew=2)
        rows += [[r, e, f"{v:.4f}"] for e, v in zip(ep, sv)]
    ax.set_xlabel("epoch")
    ax.set_ylabel("san_val median NSE (61 unseen basins)")
    ax.set_title("Training curves on held-out basins (marker = selected epoch)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig5_training_curves")
    write_csv("fig5_training_curves", ["run", "epoch", "sanval_median_nse"],
              rows)


# ─────────────────────────────────────────────────────────────────────
# RESULTS.md
# ─────────────────────────────────────────────────────────────────────

def results_md(P):
    runs = sorted(P, key=lambda r: -P[r][1]["median_nse"])
    lines = ["# Results — dev and test read-outs", "",
             "Generated by `make_report.py` from `experiments/test/`. Median "
             "per-basin NSE; persistence is scored on the same windows. "
             "The final model (9-member equal-weight ensemble) was fixed "
             "before any test number existed; test was never used to choose.",
             "",
             "| model | dev | test | test − dev | test h1-12 | test h37-48 "
             "| test % basins NSE>0.5 | test high-flow β | test mean(max(NSE,−1)) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        d, t = P[r]
        h = t["horizon_nse"]
        lines.append(
            f"| {label(r)} | {d['median_nse']:.4f} | **{t['median_nse']:.4f}** "
            f"| {t['median_nse'] - d['median_nse']:+.4f} | {h['h1_12']:.3f} "
            f"| {h['h37_48']:.3f} | {t['pct_nse_05']:.1f} "
            f"| {t['median_highflow_beta']:.3f} | {t['mean_nse_clip']:.3f} |")
    any_run = next(iter(P.values()))
    lines += ["", f"Persistence: dev {any_run[0]['median_nse_persistence']:.4f}, "
              f"test {any_run[1]['median_nse_persistence']:.4f}.", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print("  wrote reports/RESULTS.md")


def main():
    style()
    P = pairs()
    if not P:
        print("no read-outs yet — run run_test_readout.py first")
        return 1
    print(f"{len(P)} models with dev + test read-outs")
    fig_scoreboard(P)
    fig_horizon(P)
    fig_skill(P)
    fig_seeds(P)
    fig_curves()
    results_md(P)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
