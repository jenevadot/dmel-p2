"""
evaluate.py
───────────
Per-basin evaluation: NSE, KGE, RMSE, MAE.
All metrics are computed on DENORMALISED discharge (mm/h).

Design
------
  evaluate_model()  — main entry point: runs model over a DataLoader,
                      denormalises predictions, computes all metrics per basin,
                      returns an EvalResult with aggregated stats.

  compute_metrics() — pure numpy function: given arrays of observed and
                      predicted values (already denormalised), returns a
                      MetricsDict for one basin.

  aggregate()       — given a list of per-basin MetricsDicts, returns the
                      full summary used for ablation comparison:
                        median NSE  ← primary ranking metric
                        mean   NSE  ← outlier-sensitivity check
                        % NSE > 0.5 ← acceptable-skill threshold
                        % NSE > 0.7 ← good-skill threshold
                        median KGE  ← secondary diagnostic

Usage
-----
    from evaluate import evaluate_model
    result = evaluate_model(model, dev_loader, normalizer, device)
    print(result.summary())            # human-readable table
    result.save("experiments/exp_001/metrics_dev.json")
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── make sure sibling modules are importable when run directly ────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import N_BASINS, TARGET_CHANNEL, FORECAST_HOURS


# ─────────────────────────────────────────────────────────────────────
# 1.  Per-basin metric container
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BasinMetrics:
    """
    Metrics for a single basin, computed over all its samples in a split.

    All values are on DENORMALISED discharge (mm/h).
    """
    basin_id : int
    n_samples: int          # number of samples evaluated

    # ── Primary ──────────────────────────────────────────────────────
    nse      : float        # Nash-Sutcliffe Efficiency   ∈ (−∞, 1]
    kge      : float        # Kling-Gupta Efficiency      ∈ (−∞, 1]

    # ── KGE components (diagnostic) ──────────────────────────────────
    kge_r    : float        # Pearson r  (timing accuracy)
    kge_alpha: float        # σ_pred / σ_obs  (variability)
    kge_beta : float        # μ_pred / μ_obs  (bias)

    # ── Supplementary ─────────────────────────────────────────────────
    rmse     : float        # Root Mean Squared Error  (mm/h)
    mae      : float        # Mean Absolute Error      (mm/h)

    def is_valid(self) -> bool:
        return math.isfinite(self.nse) and math.isfinite(self.kge)


# ─────────────────────────────────────────────────────────────────────
# 2.  Core metric computation (pure numpy, no torch)
# ─────────────────────────────────────────────────────────────────────

def _json_safe(o):
    """
    Fallback encoder for numpy scalar / array types.

    Metrics are computed in numpy, so np.float32 / np.int64 leak into the
    saved records. Without this, `json.dump` fails only at the very END of a
    run — after training and dev evaluation have both completed — which is the
    most expensive possible place to discover a type error. (It did exactly
    that once; see IMPROVEMENTS.md.)
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(
        f"Object of type {type(o).__name__} is not JSON serializable")


_EPS = 1e-10   # guard against division by zero

# Lead-time buckets for per-horizon reporting.
#
# WHY THIS EXISTS: a single aggregate NSE is dominated by the trivially easy
# early hours and hides where the model actually earns its keep. Measured
# persistence (last observed value held flat) by lead time:
#
#     h+1   0.9985        h+24  0.5810
#     h+6   0.9574        h+36  0.3453
#     h+12  0.8408        h+48  0.2228
#
# Discharge is so autocorrelated that "tomorrow looks like today" is nearly
# perfect at h+1 and collapses by h+48. A model that has learned nothing but
# persistence will still post a respectable aggregate median NSE. Reporting by
# bucket is what exposes that.
HORIZON_BUCKETS = {
    "h1_12": (0, 12),
    "h13_24": (12, 24),
    "h25_36": (24, 36),
    "h37_48": (36, 48),
}

def compute_metrics(
    obs   : np.ndarray,   # (N,)  observed discharge, denormalised, mm/h
    pred  : np.ndarray,   # (N,)  predicted discharge, denormalised, mm/h
    basin_id: int = -1,
) -> BasinMetrics:
    """
    Compute NSE, KGE (+ components), RMSE, MAE for one basin.

    Parameters
    ----------
    obs, pred : 1-D arrays of any length (all samples × all timesteps
                for that basin, flattened).
    """
    obs  = obs.astype(np.float64)
    pred = pred.astype(np.float64)
    n    = len(obs)

    # ── NSE ──────────────────────────────────────────────────────────
    obs_mean = obs.mean()
    ss_res   = ((obs - pred) ** 2).sum()
    ss_tot   = ((obs - obs_mean) ** 2).sum()
    nse      = float(1.0 - ss_res / ss_tot) if ss_tot > _EPS else 0.0

    # ── KGE components ───────────────────────────────────────────────
    obs_std  = obs.std()
    pred_std = pred.std()
    pred_mean = pred.mean()

    # Pearson r
    if obs_std > _EPS and pred_std > _EPS:
        r = float(np.corrcoef(obs, pred)[0, 1])
    else:
        r = 1.0   # both constant → perfect correlation by convention

    # Variability ratio α
    alpha = float(pred_std / obs_std) if obs_std > _EPS else 1.0

    # Bias ratio β
    beta  = float(pred_mean / obs_mean) if abs(obs_mean) > _EPS else 1.0

    kge = float(1.0 - math.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

    # ── RMSE / MAE ───────────────────────────────────────────────────
    rmse = float(math.sqrt(((obs - pred) ** 2).mean()))
    mae  = float(np.abs(obs - pred).mean())

    return BasinMetrics(
        # int() is load-bearing: `basin_id` arrives from a numpy array, so it
        # is an np.int64. dataclasses.asdict preserves that type, and
        # json.dump then raises "Object of type int64 is not JSON
        # serializable" — after the full training run and dev evaluation have
        # already completed. Coerce at the boundary where it enters the record.
        basin_id  = int(basin_id),
        n_samples = int(n),
        nse       = nse,
        kge       = kge,
        kge_r     = r,
        kge_alpha = alpha,
        kge_beta  = beta,
        rmse      = rmse,
        mae       = mae,
    )


# ─────────────────────────────────────────────────────────────────────
# 3.  Aggregation across basins
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """
    Full evaluation result over all basins in one split.

    The 'summary' fields are what we use to rank ablations:
      median_nse   ← PRIMARY   ranking metric
      mean_nse     ← secondary (outlier sensitivity)
      pct_nse_05   ← % basins with NSE > 0.5  (acceptable skill)
      pct_nse_07   ← % basins with NSE > 0.7  (good skill)
      median_kge   ← KGE diagnostic

    The full per-basin list is kept for Wilcoxon tests.
    """
    # ── aggregate summary (used for ranking) ─────────────────────────
    median_nse  : float = 0.0
    mean_nse    : float = 0.0
    std_nse     : float = 0.0
    pct_nse_05  : float = 0.0    # % basins NSE > 0.5
    pct_nse_07  : float = 0.0    # % basins NSE > 0.7
    median_kge  : float = 0.0
    mean_kge    : float = 0.0
    median_rmse : float = 0.0    # supplementary
    median_mae  : float = 0.0    # supplementary
    n_basins    : int   = 0
    n_samples   : int   = 0

    # Median NSE per lead-time bucket. See HORIZON_BUCKETS for why the
    # aggregate alone is misleading.
    horizon_nse : Dict[str, float] = field(default_factory=dict)

    # Total basins INCLUDING those dropped as invalid. pct_nse_* divide by
    # this, not by the survivor count — see `aggregate`.
    n_basins_total : int = 0

    # ── per-basin detail (for Wilcoxon / violin plots) ────────────────
    per_basin   : List[BasinMetrics] = field(default_factory=list)

    # ── metadata ─────────────────────────────────────────────────────
    split_name  : str = ""
    best_epoch  : int = -1

    def horizon_summary(self) -> str:
        """Per-lead-time table, with the persistence baseline for reference."""
        if not self.horizon_nse:
            return ""
        ref = {"h1_12": 0.84, "h13_24": 0.58, "h25_36": 0.35, "h37_48": 0.22}
        lines = ["", "  NSE by lead time (median across basins)",
                 "  ─────────────────────────────────────────────────",
                 f"  {'bucket':<10} {'NSE':>8}  {'persistence':>12}  {'delta':>8}"]
        for k in HORIZON_BUCKETS:
            if k in self.horizon_nse:
                v = self.horizon_nse[k]
                r = ref[k]
                lines.append(f"  {k:<10} {v:>8.4f}  {r:>12.2f}  {v - r:>+8.4f}")
        return "\n".join(lines)

    def nse_array(self) -> np.ndarray:
        """Return NSE values as a numpy array (one value per basin)."""
        return np.array([b.nse for b in self.per_basin])

    def kge_array(self) -> np.ndarray:
        return np.array([b.kge for b in self.per_basin])

    def summary(self, width: int = 62) -> str:
        """Human-readable summary table."""
        bar = '─' * width
        lines = [
            f"  ┌{bar}┐",
            f"  │  Evaluation — {self.split_name:<{width-17}}│",
            f"  ├{bar}┤",
            f"  │  {'Metric':<28} {'Value':>10}  {'Notes':<18} │",
            f"  ├{bar}┤",
            f"  │  {'median NSE  [PRIMARY]':<28} {self.median_nse:>10.4f}  {'higher = better':<18} │",
            f"  │  {'mean NSE':<28} {self.mean_nse:>10.4f}  {'outlier check':<18} │",
            f"  │  {'std NSE':<28} {self.std_nse:>10.4f}  {'across basins':<18} │",
            f"  │  {'% NSE > 0.50 (acceptable)':<28} {self.pct_nse_05:>9.1f}%  {'':<18} │",
            f"  │  {'% NSE > 0.70 (good)':<28} {self.pct_nse_07:>9.1f}%  {'':<18} │",
            f"  ├{bar}┤",
            f"  │  {'median KGE  [secondary]':<28} {self.median_kge:>10.4f}  {'timing+var+bias':<18} │",
            f"  ├{bar}┤",
            f"  │  {'median RMSE (mm/h)':<28} {self.median_rmse:>10.6f}  {'supplementary':<18} │",
            f"  │  {'median MAE  (mm/h)':<28} {self.median_mae:>10.6f}  {'supplementary':<18} │",
            f"  ├{bar}┤",
            f"  │  {'basins evaluated':<28} {self.n_basins:>10}  {'':<18} │",
            f"  │  {'total samples':<28} {self.n_samples:>10,}  {'':<18} │",
            f"  └{bar}┘",
        ]
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save to JSON — per-basin list included for Wilcoxon tests later."""
        d = {
            "median_nse" : self.median_nse,
            "mean_nse"   : self.mean_nse,
            "std_nse"    : self.std_nse,
            "pct_nse_05" : self.pct_nse_05,
            "pct_nse_07" : self.pct_nse_07,
            "median_kge" : self.median_kge,
            "mean_kge"   : self.mean_kge,
            "median_rmse": self.median_rmse,
            "median_mae" : self.median_mae,
            "n_basins"   : self.n_basins,
            "n_basins_total": self.n_basins_total,
            "n_samples"  : self.n_samples,
            "horizon_nse": self.horizon_nse,
            "split_name" : self.split_name,
            "best_epoch" : self.best_epoch,
            "per_basin"  : [asdict(b) for b in self.per_basin],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=_json_safe)
        print(f"[evaluate] Saved metrics → {path}")

    @classmethod
    def load(cls, path: str) -> "EvalResult":
        with open(path) as f:
            d = json.load(f)
        per_basin = [BasinMetrics(**b) for b in d.pop("per_basin")]
        result = cls(**{k: v for k, v in d.items()})
        result.per_basin = per_basin
        return result


def aggregate(basin_metrics: List[BasinMetrics],
              split_name: str = "",
              best_epoch: int = -1) -> EvalResult:
    """
    Aggregate a list of per-basin BasinMetrics into one EvalResult.
    Invalid basins (non-finite NSE) are excluded from aggregation
    but are still stored in per_basin for transparency.
    """
    valid = [b for b in basin_metrics if b.is_valid()]
    n_total = len(basin_metrics)
    if not valid:
        return EvalResult(split_name=split_name, best_epoch=best_epoch,
                          per_basin=basin_metrics, n_basins_total=n_total)

    nse_arr  = np.array([b.nse  for b in valid])
    kge_arr  = np.array([b.kge  for b in valid])
    rmse_arr = np.array([b.rmse for b in valid])
    mae_arr  = np.array([b.mae  for b in valid])

    # pct_* divide by ALL basins, not just the valid ones. Dividing by the
    # survivors would let a run that produced NaN on 200 basins report its
    # percentages over the remaining 308 and look better for having failed.
    denom = max(n_total, 1)

    return EvalResult(
        median_nse  = float(np.median(nse_arr)),
        mean_nse    = float(np.mean(nse_arr)),
        std_nse     = float(np.std(nse_arr)),
        pct_nse_05  = float((nse_arr > 0.5).sum() / denom * 100),
        pct_nse_07  = float((nse_arr > 0.7).sum() / denom * 100),
        median_kge  = float(np.median(kge_arr)),
        mean_kge    = float(np.mean(kge_arr)),
        median_rmse = float(np.median(rmse_arr)),
        median_mae  = float(np.median(mae_arr)),
        n_basins    = len(valid),
        n_basins_total = n_total,
        n_samples   = sum(b.n_samples for b in valid),
        per_basin   = basin_metrics,
        split_name  = split_name,
        best_epoch  = best_epoch,
    )


# ─────────────────────────────────────────────────────────────────────
# 4.  Wilcoxon signed-rank test helper
# ─────────────────────────────────────────────────────────────────────

def wilcoxon_nse(result_a: EvalResult,
                 result_b: EvalResult,
                 alpha: float = 0.05) -> Dict:
    """
    Non-parametric Wilcoxon signed-rank test comparing two models
    on the same set of basins.

    Tests H1: model_a has higher per-basin NSE than model_b.
    Uses scipy.stats.wilcoxon if available, otherwise reports
    a simple sign-test approximation.

    Returns
    -------
    dict with keys: statistic, p_value, significant, n_pairs,
                    delta_median (a.median_nse - b.median_nse)
    """
    a_nse = result_a.nse_array()
    b_nse = result_b.nse_array()

    # Match basins that appear in both results
    a_ids = {b.basin_id: b.nse for b in result_a.per_basin if b.is_valid()}
    b_ids = {b.basin_id: b.nse for b in result_b.per_basin if b.is_valid()}
    common = sorted(set(a_ids) & set(b_ids))

    if not common:
        return {"error": "no common basins", "significant": False}

    a_paired = np.array([a_ids[i] for i in common])
    b_paired = np.array([b_ids[i] for i in common])
    diffs    = a_paired - b_paired

    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
        stat, pval = scipy_wilcoxon(diffs, alternative="greater")
    except ImportError:
        # Fallback: sign test (less powerful but always available)
        n_pos = (diffs > 0).sum()
        n_neg = (diffs < 0).sum()
        n     = n_pos + n_neg
        # Two-tailed binomial approximation
        import math as _math
        if n == 0:
            pval = 1.0
        else:
            pval  = 2 * min(n_pos, n_neg) / n   # crude approximation
        stat  = float(n_pos)

    return {
        "statistic"    : float(stat),
        "p_value"      : float(pval),
        "significant"  : bool(pval < alpha),
        "alpha"        : alpha,
        "n_pairs"      : len(common),
        "delta_median" : float(np.median(a_paired) - np.median(b_paired)),
        "a_median_nse" : float(np.median(a_paired)),
        "b_median_nse" : float(np.median(b_paired)),
    }


# ─────────────────────────────────────────────────────────────────────
# 5.  Model evaluation loop (torch-dependent)
# ─────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    loader,
    normalizer,
    device,
    split_name  : str = "dev",
    best_epoch  : int = -1,
    cfg         : Optional[dict] = None,
    verbose     : bool = True,
) -> EvalResult:
    """
    Run model over all batches, denormalise, compute per-basin metrics.

    Inputs are built by `train.forward_batch` — the SAME helper training uses.
    Previously this function called `model(x, y_aux=y_aux)`, a kwarg
    DMEL.forward does not accept (instant TypeError), never passed basin_ids
    (so use_basin_emb crashed), and never built a decoder input at all. Sharing
    one builder is what prevents train/eval input drift.
    """
    import torch

    from train import forward_batch

    cfg = cfg or {}
    model.eval()

    obs_store: Dict[int, List[np.ndarray]] = defaultdict(list)
    pred_store: Dict[int, List[np.ndarray]] = defaultdict(list)

    n_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            y_true = batch["y"].to(device)
            basin_ids = batch["basin_id"].cpu().numpy()

            y_pred, _ = forward_batch(model, batch, cfg, device)

            y_pred_np = y_pred.cpu().numpy().astype(np.float32)
            y_true_np = y_true.cpu().numpy().astype(np.float32)

            for i, bid in enumerate(basin_ids):
                m = normalizer.mean_[bid, TARGET_CHANNEL]
                s = normalizer.std_[bid, TARGET_CHANNEL]
                obs_store[bid].append(y_true_np[i] * s + m)
                pred_store[bid].append(y_pred_np[i] * s + m)

            if verbose and (batch_idx % max(1, n_batches // 20) == 0):
                pct = (batch_idx + 1) / n_batches * 100
                bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
                print(f"\r  [evaluate] {split_name} [{bar}] {pct:5.1f}%", end="")

    if verbose:
        print()

    # ── per-basin metrics, pooled and per-horizon ─────────────────────
    basin_metrics: List[BasinMetrics] = []
    horizon_nse: Dict[str, List[float]] = defaultdict(list)

    for bid in sorted(obs_store.keys()):
        obs_2d = np.stack(obs_store[bid])      # (n_samples, 48)
        pred_2d = np.stack(pred_store[bid])

        bm = compute_metrics(obs_2d.ravel(), pred_2d.ravel(), basin_id=bid)
        basin_metrics.append(bm)

        for label, (lo, hi) in HORIZON_BUCKETS.items():
            o = obs_2d[:, lo:hi].ravel()
            p = pred_2d[:, lo:hi].ravel()
            ss_tot = ((o - o.mean()) ** 2).sum()
            if ss_tot > _EPS:
                horizon_nse[label].append(
                    float(1.0 - ((o - p) ** 2).sum() / ss_tot))

    result = aggregate(basin_metrics, split_name=split_name,
                       best_epoch=best_epoch)
    result.horizon_nse = {
        k: float(np.median(v)) for k, v in horizon_nse.items() if v
    }
    result.n_basins_total = len(basin_metrics)

    if verbose:
        print(result.summary())
        print(result.horizon_summary())

    return result


# ─────────────────────────────────────────────────────────────────────
# 6.  Self-test (numpy only, no torch)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("\n── compute_metrics: perfect prediction ──────────────────────")
    obs  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = obs.copy()
    m    = compute_metrics(obs, pred, basin_id=0)
    assert abs(m.nse  - 1.0) < 1e-9, f"NSE should be 1.0, got {m.nse}"
    assert abs(m.kge  - 1.0) < 1e-9, f"KGE should be 1.0, got {m.kge}"
    assert abs(m.rmse - 0.0) < 1e-9
    assert abs(m.mae  - 0.0) < 1e-9
    print(f"  NSE={m.nse:.4f}  KGE={m.kge:.4f}  RMSE={m.rmse:.6f}  ✓")

    print("\n── compute_metrics: mean prediction (NSE = 0) ───────────────")
    obs   = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred  = np.full_like(obs, obs.mean())
    m     = compute_metrics(obs, pred, basin_id=1)
    assert abs(m.nse - 0.0) < 1e-9, f"NSE should be 0.0, got {m.nse}"
    print(f"  NSE={m.nse:.4f}  KGE={m.kge:.4f}  RMSE={m.rmse:.6f}  ✓")

    print("\n── compute_metrics: systematic over-prediction ──────────────")
    obs  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = obs * 2.0     # 2× the true values
    m    = compute_metrics(obs, pred, basin_id=2)
    # NSE < 0 (predicting 2x is worse than predicting the mean)
    # KGE beta = 2.0, so KGE < 0
    print(f"  NSE={m.nse:.4f}  KGE={m.kge:.4f}  beta={m.kge_beta:.4f}")
    assert m.kge_beta > 1.8, "beta should be ~2"
    print(f"  ✓  over-prediction detected via KGE beta={m.kge_beta:.4f}")

    print("\n── aggregate: 5 fake basins ──────────────────────────────────")
    rng    = np.random.default_rng(42)
    basins = []
    for bid in range(5):
        n   = 200
        obs  = rng.exponential(0.1, n)
        noise = rng.normal(0, 0.02, n)
        pred  = obs + noise
        basins.append(compute_metrics(obs, pred, basin_id=bid))

    result = aggregate(basins, split_name="test_split")
    print(result.summary())
    assert 0 < result.median_nse <= 1.0
    assert result.n_basins == 5
    print("  aggregate ✓")

    print("\n── save / load round-trip ────────────────────────────────────")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "metrics.json")
        result.save(path)
        loaded = EvalResult.load(path)
        assert abs(loaded.median_nse - result.median_nse) < 1e-9
        assert len(loaded.per_basin) == len(result.per_basin)
    print("  save/load ✓")

    print("\n── wilcoxon_nse: model_a clearly better than model_b ─────────")
    rng2   = np.random.default_rng(123)
    basins_a, basins_b = [], []
    for bid in range(20):
        obs   = rng2.exponential(0.1, 500)
        pred_a = obs + rng2.normal(0, 0.01, 500)    # good
        pred_b = obs + rng2.normal(0, 0.05, 500)    # worse
        basins_a.append(compute_metrics(obs, pred_a, basin_id=bid))
        basins_b.append(compute_metrics(obs, pred_b, basin_id=bid))
    res_a = aggregate(basins_a, split_name="model_a")
    res_b = aggregate(basins_b, split_name="model_b")
    w = wilcoxon_nse(res_a, res_b)
    print(f"  model_a median NSE = {res_a.median_nse:.4f}")
    print(f"  model_b median NSE = {res_b.median_nse:.4f}")
    print(f"  Δmedian = {w['delta_median']:+.4f}  p={w['p_value']:.4f}  "
          f"significant={w['significant']}")
    assert w["delta_median"] > 0, "model_a should be better"
    print("  wilcoxon ✓")

    print("\n✓ evaluate.py self-test passed\n")
