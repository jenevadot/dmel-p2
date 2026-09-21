"""
src/baselines.py
────────────────
Closed-form baselines, computed on `dev` and written as first-class
`EvalResult` artefacts so `compare_results.py` ranks every trained model
against them automatically.

Why this module exists
──────────────────────
The plan document quoted "persistence ~ NSE 0.53" as an empirical fact, but
nothing computed it, and the scoreboard implicitly compared models against
NSE=0 (predict-the-mean). That is the wrong bar by a wide margin.

    B0  predict basin mean            NSE = 0.0000  by definition
    B1  persistence (last value)      NSE ~ 0.5251  <- THE BAR
    B2  climatology (hourly profile)  NSE ~ -0.0221
    B3  persistence decaying to mean  NSE ~ 0.4460

A model scoring median NSE 0.52 has learned nothing beyond "the next two days
look like right now". Judging it against 0.0 would call that a triumph.

The lead-time story matters more than the aggregate
───────────────────────────────────────────────────
Discharge is so autocorrelated that persistence is near-perfect at short lead
and collapses at long lead:

    h+1   0.9985        h+24  0.5810
    h+6   0.9574        h+36  0.3453
    h+12  0.8408        h+48  0.2228

So an aggregate median NSE is dominated by the trivially easy early hours. The
model's entire value lives in h+24..h+48. That is why `evaluate.py` reports
per-horizon buckets and why these baselines do too.

B2's catastrophic mean NSE is diagnostic rather than a bug: a fixed
hour-of-day profile is actively harmful, which confirms discharge is driven by
event timing (rainfall) and not by a diurnal cycle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EXP_DIR, FORECAST_HOURS, TARGET_CHANNEL
from evaluate import (HORIZON_BUCKETS, BasinMetrics, EvalResult, _EPS,
                      _json_safe, aggregate, compute_metrics)


def _predict(kind: str, x_raw: np.ndarray, basin_mean: float) -> np.ndarray:
    """
    Produce a (48,) forecast from one RAW (un-normalised) history window.

    x_raw : (T, C) raw window; discharge is column TARGET_CHANNEL.
    """
    q = x_raw[:, TARGET_CHANNEL]

    if kind == "mean":
        return np.full(FORECAST_HOURS, basin_mean, dtype=np.float64)

    if kind == "persistence":
        return np.full(FORECAST_HOURS, q[-1], dtype=np.float64)

    if kind == "climatology":
        # Mean of each hour-of-day position over the history window.
        prof = q[-24 * 7:] if len(q) >= 24 * 7 else q
        hourly = prof.reshape(-1, 24).mean(axis=0) if len(prof) % 24 == 0 \
            else np.full(24, prof.mean())
        return np.tile(hourly, int(np.ceil(FORECAST_HOURS / 24))
                       )[:FORECAST_HOURS]

    if kind == "decay":
        # Persistence relaxing exponentially toward the basin mean over 24h.
        last = q[-1]
        t = np.arange(FORECAST_HOURS)
        w = np.exp(-t / 24.0)
        return last * w + basin_mean * (1 - w)

    raise ValueError(f"Unknown baseline: {kind!r}")


def run_baseline(kind: str, loader, verbose: bool = True) -> EvalResult:
    """
    Evaluate one baseline over a loader that yields RAW (un-normalised) data.

    The loader MUST be built with `normalizer` whose strategy is "none", or the
    predictions and targets will be on different scales. `compute_all` handles
    this.
    """
    obs_store: Dict[int, List[np.ndarray]] = {}
    pred_store: Dict[int, List[np.ndarray]] = {}
    sums: Dict[int, List[float]] = {}

    # Pass 1: per-basin mean of the observed history (needed by mean/decay).
    for batch in loader:
        x = batch["x"].numpy()
        bids = batch["basin_id"].numpy()
        for i, b in enumerate(bids):
            s = sums.setdefault(int(b), [0.0, 0.0])
            s[0] += float(x[i, :, TARGET_CHANNEL].sum())
            s[1] += x.shape[1]
    basin_mean = {b: (v[0] / v[1] if v[1] else 0.0) for b, v in sums.items()}

    # Pass 2: predict.
    for batch in loader:
        x = batch["x"].numpy()
        y = batch["y"].numpy()
        bids = batch["basin_id"].numpy()
        for i, b in enumerate(bids):
            b = int(b)
            p = _predict(kind, x[i], basin_mean[b])
            obs_store.setdefault(b, []).append(y[i].astype(np.float64))
            pred_store.setdefault(b, []).append(p)

    metrics: List[BasinMetrics] = []
    horizon: Dict[str, List[float]] = {k: [] for k in HORIZON_BUCKETS}

    for b in sorted(obs_store):
        o2 = np.stack(obs_store[b])
        p2 = np.stack(pred_store[b])
        metrics.append(compute_metrics(o2.ravel(), p2.ravel(), basin_id=b))
        for label, (lo, hi) in HORIZON_BUCKETS.items():
            o = o2[:, lo:hi].ravel()
            p = p2[:, lo:hi].ravel()
            sst = ((o - o.mean()) ** 2).sum()
            if sst > _EPS:
                horizon[label].append(float(1.0 - ((o - p) ** 2).sum() / sst))

    res = aggregate(metrics, split_name=f"dev::{kind}")
    res.horizon_nse = {k: float(np.median(v)) for k, v in horizon.items() if v}
    res.n_basins_total = len(metrics)

    if verbose:
        print(f"\n  baseline={kind}")
        print(f"    median NSE {res.median_nse:+.4f}   mean {res.mean_nse:+.4f}"
              f"   >0.5 {res.pct_nse_05:.1f}%   >0.7 {res.pct_nse_07:.1f}%")
        for k, v in res.horizon_nse.items():
            print(f"      {k:<8} {v:+.4f}")
    return res


BASELINES = ("mean", "persistence", "climatology", "decay")


def compute_all(save: bool = True, verbose: bool = True) -> Dict[str, EvalResult]:
    """Run every baseline on dev and persist each as an experiment folder."""
    from dataset import build_splits, make_loader

    # strategy="none" -> the loader yields RAW values, which is what the
    # closed-form predictors need. Metrics are therefore already in mm/h.
    _, _, ds_dev, _, _, _ = build_splits(norm_strategy="none",
                                         verbose=verbose)
    loader = make_loader(ds_dev, batch_size=256, shuffle=False,
                         num_workers=0)

    out: Dict[str, EvalResult] = {}
    for kind in BASELINES:
        res = run_baseline(kind, loader, verbose=verbose)
        out[kind] = res
        if save:
            d = EXP_DIR / f"baseline_{kind}"
            d.mkdir(parents=True, exist_ok=True)
            res.save(str(d / "metrics_dev.json"))
            import json
            with open(d / "summary.json", "w") as f:
                json.dump({
                    "name": f"baseline_{kind}",
                    "is_baseline": True,
                    "selection_split": "none",
                    "report_split": "dev",
                    "dev_median_nse": res.median_nse,
                    "dev_mean_nse": res.mean_nse,
                    "dev_median_kge": res.median_kge,
                    "dev_pct_nse_05": res.pct_nse_05,
                    "dev_pct_nse_07": res.pct_nse_07,
                    "horizon_nse": res.horizon_nse,
                    "n_params": 0,
                    "total_minutes": 0.0,
                }, f, indent=2, default=_json_safe)

    if verbose:
        print("\n  ┌─────────────────────────────────────────────────┐")
        print("  │  THE BAR TO BEAT                                │")
        print("  ├─────────────────────────────────────────────────┤")
        for k, r in sorted(out.items(), key=lambda kv: -kv[1].median_nse):
            print(f"  │  {k:<14} median NSE {r.median_nse:+.4f}          │")
        print("  └─────────────────────────────────────────────────┘")
    return out


if __name__ == "__main__":
    compute_all(save=True, verbose=True)
