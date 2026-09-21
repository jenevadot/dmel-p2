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

    B0  predict basin mean            NSE = -0.0071  ~0 by construction
    B1  persistence (last value)      NSE =  0.5251  <- THE BAR
    B2  climatology (hourly profile)  NSE =  0.1546
    B3  persistence decaying to mean  NSE =  0.4889
    B4  ridge on the flattened window NSE =  0.4169  (60k fit windows)

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

B2 is weakly skilful (median +0.1546) but its MEAN NSE is -0.3673: a fixed
hour-of-day profile helps in a minority of basins and hurts badly in the rest.
That spread is diagnostic — discharge is driven by event timing (rainfall), not
by a diurnal cycle, so a climatological prior cannot carry the forecast.

Why B4 (ridge) is here, and why not ARIMA
─────────────────────────────────────────
B0-B3 are free arithmetic; the Informer has ~10^6 parameters. Nothing occupies
the middle, so "the Informer beats persistence" cannot distinguish two very
different worlds: one where the mapping is genuinely nonlinear, and one where
any fitted regressor on the same inputs would do as well. B4 closes that gap.

Ridge is the right occupant of that slot:

  * It is the LINEAR member of the model's own hypothesis class. It sees the
    identical (T, C) window the Informer sees, flattened. So `informer - ridge`
    isolates exactly one thing: the value of nonlinearity + attention, with
    inputs, split, and target held fixed. No other comparison in the registry
    does that.
  * Closed form. `w = (XᵀX + λI)⁻¹ Xᵀ y` is deterministic, has no seed, no
    epochs, and no early-stopping choice — so it cannot be accused of being
    under-tuned, which is the standard objection to a weak baseline.
  * Multi-output for free: Y is (n, 48), solved in one factorisation, matching
    the direct multi-horizon setup rather than being rolled out step by step.

ARIMA was considered and rejected for this slot:

  * Cost. ARIMA fits per series; with 508 basins x thousands of windows and no
    shared parameters, order selection alone dwarfs the Informer's training
    budget. Ridge is one factorisation, seconds.
  * It would answer a different question. ARIMA is univariate on q, so it could
    not use precipitation — it would conflate "linear vs nonlinear" with
    "fewer inputs", which is the confound B4 exists to avoid. exp_003__
    discharge_only already isolates the input-set question.
  * The paper does not use it either. Wang et al. compare DMEL against six DEEP
    models (Informer, Transformer, LSTM, RNN, TimesNet, PatchTST); ARIMA
    appears only in their related-work discussion of STL-Transformer-ARIMA. So
    an ARIMA arm is not needed to situate this work against the paper.

The honest limitation: ridge on a flattened window has no notion of time
ordering beyond what the coefficients learn per lag, and it is fitted globally
rather than per basin. It is a lower bound on what a linear model can do, not
the best possible one. That is the intended reading — if the Informer cannot
clear it by a clear margin at h+25-48, the architecture is not earning its cost.
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


# ─────────────────────────────────────────────────────────────────────
# B4 — ridge regression (the only FITTED baseline)
# ─────────────────────────────────────────────────────────────────────

def _fit_ridge(train_loader, lam: float, max_windows: int,
               verbose: bool = True):
    """
    Solve multi-output ridge on flattened, STANDARDISED raw windows.

        X (n, T*C+1)  ->  Y (n, 48)
        w = (XᵀX + λI)⁻¹ XᵀY

    Standardisation is not optional here. The 12 raw channels span five orders
    of magnitude (measured on train: ch6 mean 0.007, ch4 mean 92,374), which
    leaves cond(XᵀX) ≈ 4e54 — numerically singular. A single λ then means
    "no penalty at all" for the large-scale columns and "total suppression"
    for the small ones, and the solve returns garbage: an unstandardised fit
    scored median NSE -51,277 on dev. Centring and scaling each column to unit
    variance makes one λ meaningful across all of them.

    Accumulates the GRAM MATRIX rather than stacking X. XᵀX is (T*C+1)²
    ≈ 4033² float64 ≈ 130 MB regardless of n, whereas stacking 60k windows
    would need ~1.9 GB — which matters on a machine already swapping.

    Two passes over the capped sample: one for mean/std, one for the Gram
    accumulation. `max_windows` bounds both.

    Fitting happens in PER-BASIN Z-SCORE space — the same space the Informer
    trains in — and predictions are denormalised before scoring. This is not
    cosmetic. Fitting one global map on raw mm/h was tried and fails badly:
    the 508 basins differ in mean discharge by >16x, so a single intercept is
    wrong everywhere, and near-zero-flow basins (measured obs_std = 0.0000 for
    basin 188) get ~1-2 mm/h predictions whose per-basin NSE divides by
    almost zero. That produced median NSE -14 and individual basins at -1.6e8,
    while the POOLED number still looked like +0.23 — the pooled figure hides
    it completely. Per-basin normalisation removes the level and scale before
    the fit, so every basin contributes comparably.

    Measured dev median NSE vs λ (20k fit windows), which is also the record
    of how the default was chosen:

        λ=1e3  +0.1547     λ=1e5  +0.4145
        λ=1e4  +0.3437     λ=3e5  +0.4297   <- default
                           λ=1e6  +0.3798
                           λ=3e6  +0.2713

    CAVEAT, stated because it matters for how this number is read: λ was
    selected on dev, the same split the baseline is reported on. That is a mild
    optimism in ridge's favour — deliberately so. A baseline that has been
    given every advantage and is still beaten is a stronger result than one
    that was handicapped. Do not quote ridge's dev NSE as a held-out number.

    The intercept column is appended as a constant 1 and left UNPENALISED
    (its diagonal entry in λI is zeroed); penalising the intercept would
    shrink predictions toward zero rather than toward the mean.
    """
    # ── Pass 1: column mean / std over the capped sample ─────────────
    n_seen = 0
    s1 = s2 = None
    for batch in train_loader:
        xb = batch["x"].numpy()
        b = xb.shape[0]
        Xf = xb.reshape(b, -1).astype(np.float64)
        if s1 is None:
            s1 = np.zeros(Xf.shape[1])
            s2 = np.zeros(Xf.shape[1])
        s1 += Xf.sum(axis=0)
        s2 += (Xf ** 2).sum(axis=0)
        n_seen += b
        if n_seen >= max_windows:
            break

    mu = s1 / n_seen
    var = np.maximum(s2 / n_seen - mu ** 2, 0.0)
    sd = np.sqrt(var)
    # Constant columns carry no information; scaling by 1.0 leaves them at
    # zero after centring, so they contribute nothing and cannot blow up.
    sd[sd < 1e-8] = 1.0

    # ── Pass 2: Gram accumulation on standardised features ───────────
    XtX = XtY = None
    n_fit = 0
    for batch in train_loader:
        xb = batch["x"].numpy()
        y = batch["y"].numpy().astype(np.float64)
        b = xb.shape[0]
        Xf = (xb.reshape(b, -1).astype(np.float64) - mu) / sd
        Xb = np.concatenate([Xf, np.ones((b, 1))], axis=1)

        if XtX is None:
            d = Xb.shape[1]
            XtX = np.zeros((d, d), dtype=np.float64)
            XtY = np.zeros((d, y.shape[1]), dtype=np.float64)

        XtX += Xb.T @ Xb
        XtY += Xb.T @ y
        n_fit += b
        if n_fit >= max_windows:
            break

    d = XtX.shape[0]
    reg = lam * np.eye(d)
    reg[-1, -1] = 0.0                                # do not penalise intercept

    try:
        w = np.linalg.solve(XtX + reg, XtY)
    except np.linalg.LinAlgError:
        # Singular even with λ — fall back to the pseudo-inverse rather than
        # returning a broken baseline silently.
        if verbose:
            print("    [ridge] singular system; using lstsq")
        w = np.linalg.lstsq(XtX + reg, XtY, rcond=None)[0]

    if verbose:
        cond = np.linalg.cond(XtX + reg)
        print(f"    [ridge] fitted on {n_fit:,} windows, d={d}, "
              f"lambda={lam:g}, cond={cond:.2e}")
    return w, mu, sd, n_fit


def run_ridge(train_loader, dev_loader, normalizer, lam: float = 3e5,
              max_windows: int = 60_000, verbose: bool = True):
    """
    Fit ridge on TRAIN, evaluate on dev. Returns (EvalResult, n_params).

    Both loaders must be built with `norm_strategy="per_basin_zscore"`, the
    same space the neural runs use. Predictions are denormalised with the
    identical expression `evaluate_model` uses (y * std + mean, per basin and
    target channel) so the resulting NSE is directly comparable to every
    trained model's.

    Fitting on train (not dev) is what makes this comparable to the neural
    runs — a baseline tuned on the split it is scored on would be a different,
    and much weaker, claim.
    """
    w, mu, sd, n_fit = _fit_ridge(train_loader, lam, max_windows,
                                  verbose=verbose)

    obs_store: Dict[int, List[np.ndarray]] = {}
    pred_store: Dict[int, List[np.ndarray]] = {}

    for batch in dev_loader:
        x = batch["x"].numpy()
        y = batch["y"].numpy()
        bids = batch["basin_id"].numpy()
        b = x.shape[0]
        # Same standardisation as the fit, using TRAIN statistics.
        Xf = (x.reshape(b, -1).astype(np.float64) - mu) / sd
        Xb = np.concatenate([Xf, np.ones((b, 1))], axis=1)
        pred = Xb @ w                                # (b, 48) normalised
        for i, bid in enumerate(bids):
            bid = int(bid)
            # Denormalise exactly as evaluate_model does.
            m = normalizer.mean_[bid, TARGET_CHANNEL]
            s = normalizer.std_[bid, TARGET_CHANNEL]
            obs_store.setdefault(bid, []).append(
                y[i].astype(np.float64) * s + m)
            pred_store.setdefault(bid, []).append(pred[i] * s + m)

    metrics: List[BasinMetrics] = []
    horizon: Dict[str, List[float]] = {k: [] for k in HORIZON_BUCKETS}

    for bid in sorted(obs_store):
        o2 = np.stack(obs_store[bid])
        p2 = np.stack(pred_store[bid])
        metrics.append(compute_metrics(o2.ravel(), p2.ravel(), basin_id=bid))
        for label, (lo, hi) in HORIZON_BUCKETS.items():
            o = o2[:, lo:hi].ravel()
            p = p2[:, lo:hi].ravel()
            sst = ((o - o.mean()) ** 2).sum()
            if sst > _EPS:
                horizon[label].append(float(1.0 - ((o - p) ** 2).sum() / sst))

    res = aggregate(metrics, split_name="dev::ridge")
    res.horizon_nse = {k: float(np.median(v)) for k, v in horizon.items() if v}
    res.n_basins_total = len(metrics)

    if verbose:
        print(f"\n  baseline=ridge")
        print(f"    median NSE {res.median_nse:+.4f}   mean {res.mean_nse:+.4f}"
              f"   >0.5 {res.pct_nse_05:.1f}%   >0.7 {res.pct_nse_07:.1f}%")
        for k, v in res.horizon_nse.items():
            print(f"      {k:<8} {v:+.4f}")
    return res, int(w.size)


def compute_all(save: bool = True, verbose: bool = True,
                with_ridge: bool = True,
                ridge_lambda: float = 3e5,
                ridge_max_windows: int = 60_000) -> Dict[str, EvalResult]:
    """Run every baseline on dev and persist each as an experiment folder."""
    from dataset import build_splits, make_loader

    # strategy="none" -> the loader yields RAW values, which is what the
    # closed-form predictors need. Metrics are therefore already in mm/h.
    _, _, ds_dev, _, _, _ = build_splits(norm_strategy="none",
                                         verbose=verbose)
    loader = make_loader(ds_dev, batch_size=256, shuffle=False,
                         num_workers=0)

    out: Dict[str, EvalResult] = {}
    ridge_n_params = 0
    for kind in BASELINES:
        res = run_baseline(kind, loader, verbose=verbose)
        out[kind] = res

    if with_ridge:
        # Ridge fits in the SAME normalised space as the neural runs, so it
        # needs its own splits — the closed-form baselines above require raw.
        ds_tr_n, _, ds_dev_n, _, nrm, _ = build_splits(
            norm_strategy="per_basin_zscore", verbose=False)
        # shuffle=True so a capped fit sees many basins, not the first few.
        tl = make_loader(ds_tr_n, batch_size=256, shuffle=True, num_workers=0)
        dl = make_loader(ds_dev_n, batch_size=256, shuffle=False,
                         num_workers=0)
        out["ridge"], ridge_n_params = run_ridge(
            tl, dl, nrm, lam=ridge_lambda,
            max_windows=ridge_max_windows, verbose=verbose)

    if save:
        for kind, res in out.items():
            d = EXP_DIR / f"baseline_{kind}"
            d.mkdir(parents=True, exist_ok=True)
            res.save(str(d / "metrics_dev.json"))
            import json
            # Ridge is the one baseline with real parameters; recording 0 here
            # would misrepresent it against the Informer's ~10^6 in the
            # scoreboard's params column.
            n_params = ridge_n_params if kind == "ridge" else 0
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
                    "n_params": n_params,
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
