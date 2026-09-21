"""
src/decompose.py
────────────────
CEEMDAN + Sample Entropy + the paper's Modal Recognition Strategy (MRS).

This is the paper's actual contribution (Wang et al. 2026, §2.2.1-2.2.2), which
was absent from the implementation: the code reproduced the Informer baseline
(the paper's M2), not DMEL (M1).

Pipeline
────────
    q(t)  ──CEEMDAN──►  IMF_1 … IMF_n , residual
                              │
                        SampleEntropy per component
                              │
                  contiguous DP grouping into exactly K
                              │
                         RIMF_1 … RIMF_K          (Eq. 12)
                              │
                   split by SE into HF / LF        (Step 2)
                              │
                   Informer(HF)  +  LSTM(LF)       (Eq. 14)

Provenance
──────────
The CEEMDAN core is a numpy port of `runoff_emd_deep_dive.py:86-145`, which is
already verified in this repo against Torres, Colominas, Schlotthauer &
Flandrin (ICASSP 2011), Eqs. 1-5 — including the subtle stage->=2 rule that
perturbs with the NOISE'S OWN k-th EMD mode rescaled to hold SNR constant, not
with fresh white noise. Writing that from scratch is the main correctness risk
in the whole module, so it is ported rather than reinvented.

Design decisions forced by OUR data (all measured, see IMPROVEMENTS.md)
──────────────────────────────────────────────────────────────────────
The paper decomposes ~1400-point DAILY univariate series and obtains 9 IMFs.
We have 336-point HOURLY windows.

CORRECTION worth recording: an earlier note here claimed 1-6 IMFs with 9.8% of
windows yielding a single IMF. Those figures came from PLAIN EMD. Real CEEMDAN
at n_trials=100 behaves differently — the added noise creates extrema, so more
modes are extractable. Measured on 256 real windows:

      0 IMFs:  2.0%  (constant)     8 IMFs: 45.7%
      6 IMFs:  7.0%                 9 IMFs:  5.5%
      7 IMFs: 39.8%

So CEEMDAN yields 6-9 IMFs, mode 8 — close to the paper's 9. Consequences:

1.  K is FIXED at 3 (n_high=2, n_low=1), not equal to n_imf.
    Zero-padding a variable count into fixed branches would make branch i mean
    a different frequency band depending on the sample, destroying the
    per-branch specialisation that justifies one-model-per-RIMF in the first
    place. Grouping to a constant K is a mild specialisation of the paper's
    own operator, which already maps 9->6 (Shuangpai) and 9->4 (Fenghuang).
    Every non-constant window has >= 6 components, so K=3 is always
    satisfiable; constant windows are handled explicitly below.

2.  Grouping must be CONTIGUOUS in EMD order.
    EMD emits components in decreasing frequency and SE follows that order
    (paper Fig. 9). k-means on SE could place IMF1 with IMF7, which would
    destroy the band structure. Exact O(n*K) dynamic programming over
    contiguous segments is used instead — and it is deterministic, unlike
    k-means with random init.

3.  Degenerate windows are first-class, not edge cases.
    Constant windows, monotone windows and 1-IMF windows are ~11% of the data.
    Each has an explicit code and an explicit handling rule.

The invariant
─────────────
    sum(RIMF_1..K) == q      exactly, in every path, including every
                             degenerate branch.

This is the paper's Eq. 14 superposition identity. If it ever breaks, the
additive ensemble is no longer reconstructing the target and every downstream
number is meaningless. It is asserted, not assumed.

Leakage
───────
`decompose_window` is a pure function of ONE 336-point window. The classic EMD
leakage mechanism — cubic-spline envelopes fitted across a whole series,
carrying future information backwards over the train/test boundary — cannot
occur here because the window is self-contained and the rows carry no shared
time axis to leak along. Registry item C9 is resolved by construction.

Honest caveat, stated because a reviewer will look for it: RIMF_j at time t
depends on the entire window including t+1..336. That is ACAUSAL WITHIN THE
INPUT, which is not label leakage — the 336h window is entirely input, and the
targets (the following 48h) live in `y` and are never touched here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CEEMDAN_CFG, DATA_DIR, TARGET_CHANNEL, TEST_H5, TRAIN_H5

# Bump whenever the sifting / grouping MATH changes. Part of the cache key, so
# a code change invalidates a stale cache even when no hyperparameter moved.
ALGO_VERSION = 1

# SampleEntropy returns -ln(A/B); A or B == 0 gives +inf, which would poison
# the DP grouping. Cap instead, and record that it happened.
SE_MAX = 3.0

# Below this std a window is treated as constant (no oscillation to extract).
CONST_EPS = 1e-8

# Degenerate codes, stored per row in the cache.
DEG_NORMAL = 0
DEG_CONSTANT = 1        # std(q) < CONST_EPS
DEG_NO_IMF = 2          # monotone: CEEMDAN extracted nothing
DEG_FEW_COMPONENTS = 3  # n_imf + 1 < K
DEG_SE_CAPPED = 5       # at least one SE hit SE_MAX


# ═════════════════════════════════════════════════════════════════════
# 1. EMD primitives
# ═════════════════════════════════════════════════════════════════════

def _extrema(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interior local maxima / minima indices."""
    return argrelextrema(y, np.greater)[0], argrelextrema(y, np.less)[0]


def _anchor(idx: np.ndarray, y: np.ndarray, n: int
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add the two endpoints as spline anchors.

    Without endpoint anchoring a cubic spline extrapolates wildly past the
    outermost extremum, injecting a large spurious swing into the first and
    last few samples of every IMF. Clamping to the series endpoints is the
    standard cheap boundary treatment and is what the verified reference in
    `runoff_emd_deep_dive.py` uses.
    """
    xs = np.concatenate(([0], idx, [n - 1]))
    # A repeated endpoint index would make CubicSpline raise on non-increasing x
    xs = np.unique(xs)
    return xs, y[xs]


def _sift(x: np.ndarray, n_sift: int = 8) -> np.ndarray:
    """
    Extract ONE IMF — the E_1(.) operator of the CEEMDAN equations.

    Fixed iteration count rather than Huang's SD threshold: bounded, fully
    deterministic, and already validated in this repo's deep-dive. A variable
    stopping rule would make the 2.8h precompute non-reproducible.
    """
    n = len(x)
    grid = np.arange(n)
    h = x.astype(np.float64, copy=True)
    for _ in range(n_sift):
        mx, mn = _extrema(h)
        if len(mx) < 1 or len(mn) < 1:
            break
        xu, yu = _anchor(mx, h, n)
        xl, yl = _anchor(mn, h, n)
        up = CubicSpline(xu, yu)(grid)
        lo = CubicSpline(xl, yl)(grid)
        h = h - 0.5 * (up + lo)
    return h


def emd(x: np.ndarray, max_imf: int = 9, n_sift: int = 8
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plain EMD. Returns (imfs (n, T), residual (T,)).

    Completeness `imfs.sum(0) + residual == x` holds by construction: every
    mode is subtracted from the running residual.
    """
    x = np.asarray(x, dtype=np.float64)
    res = x.copy()
    imfs = []
    for _ in range(max_imf):
        h = _sift(res, n_sift)
        imfs.append(h)
        res = res - h
        mx, mn = _extrema(res)
        if len(mx) + len(mn) < 3:
            break
    arr = (np.array(imfs, dtype=np.float64) if imfs
           else np.zeros((0, len(x)), dtype=np.float64))
    return arr, res


# ═════════════════════════════════════════════════════════════════════
# 2. CEEMDAN
# ═════════════════════════════════════════════════════════════════════

def make_noise_bank(n_trials: int, length: int, max_imf: int,
                    seed: int) -> list:
    """
    Pre-compute the EMD modes E_j(w_i) of the noise realisations.

    THE key optimisation. The noise modes depend only on (seed, n_trials,
    length) and NOT on the signal, so this bank is built once per worker
    process and reused for every window it handles. Without it, each window
    pays n_trials extra EMDs and the full precompute roughly doubles from
    ~2.8h to ~6h.

    Fixing the bank across all windows also makes the entire cache
    bit-reproducible from one integer seed.
    """
    rng = np.random.default_rng(seed)
    bank = []
    for _ in range(n_trials):
        w = rng.standard_normal(length)
        modes, _ = emd(w, max_imf=max_imf + 2)
        bank.append(modes)
    return bank


def ceemdan(
    x: np.ndarray,
    n_trials: int = 100,
    noise_std: float = 0.2,
    max_imf: int = 9,
    n_sift: int = 8,
    noise_bank: Optional[list] = None,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete EEMD with Adaptive Noise (Torres et al. 2011).

        step 1   IMF~_1 = mean_i E_1( x + eps_0 * w_i )      <- RAW white noise
        step 2   r_1    = x - IMF~_1                                   (Eq. 1)
        step 3   IMF~_k = mean_i E_1( r_{k-1} + eps_k * E_k(w_i) )  <- NOISE
                                                                      MODES
        step 4   r_k    = r_{k-1} - IMF~_k                             (Eq. 2)

    until the residue has fewer than 3 extrema, giving the EXACT
    reconstruction x = sum(IMF~_k) + r  (Eq. 5) — the "complete" in CEEMDAN.

    eps_k rescales by std(r_{k-1})/std(E_k(w_i)) so the SNR is held constant
    at every stage, which is what distinguishes CEEMDAN from plain EEMD and
    what removes EEMD's residual-noise problem.

    Ported from the verified reference at runoff_emd_deep_dive.py:86-145.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    sd = x.std()

    if sd < CONST_EPS:
        # Constant input: nothing to decompose. Identity still holds.
        return np.zeros((0, n), dtype=np.float64), x.copy()

    if noise_bank is None:
        noise_bank = make_noise_bank(n_trials, n, max_imf, seed)

    # Stage-1 uses raw noise; regenerate it deterministically from the bank's
    # seed so the function is self-contained when a bank is passed in.
    rng = np.random.default_rng(seed)
    raw_noise = rng.standard_normal((len(noise_bank), n))

    imfs = []
    res = x.copy()

    for i in range(max_imf):
        sd_r = res.std() or sd
        acc = np.zeros(n, dtype=np.float64)
        used = 0

        for k, modes in enumerate(noise_bank):
            if i == 0:
                pert = res + noise_std * sd * raw_noise[k]
            else:
                if i - 1 >= len(modes):
                    continue  # this realisation ran out of modes
                nm = modes[i - 1]
                nm_sd = nm.std()
                if nm_sd < CONST_EPS:
                    continue
                pert = res + (noise_std * sd_r / nm_sd) * nm
            acc += _sift(pert, n_sift)
            used += 1

        if used == 0:
            break

        imf = acc / used
        imfs.append(imf)
        res = res - imf

        mx, mn = _extrema(res)
        if len(mx) + len(mn) < 3:
            break

    arr = (np.array(imfs, dtype=np.float64) if imfs
           else np.zeros((0, n), dtype=np.float64))
    return arr, res


# ═════════════════════════════════════════════════════════════════════
# 3. Sample entropy
# ═════════════════════════════════════════════════════════════════════

def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    SampEn(m, r, N) = -ln(A / B),  r = r_factor * std(x)      (paper Eq. 2)

    A = # of (m+1)-length vector pairs within Chebyshev distance r
    B = # of  m   -length vector pairs within Chebyshev distance r

    Vectorised over the 336x336 distance matrix; measured at ~5 ms/series.

    Returns SE_MAX when A or B is zero (-ln(0) = inf would poison the DP
    grouping) and 0.0 for a constant series.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    sd = x.std()
    if sd < CONST_EPS or N <= m + 1:
        return 0.0
    r = r_factor * sd

    def _count(mm: int) -> int:
        # (N-mm+1, mm) sliding embedding
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)
        # Chebyshev distance between all pairs, done in blocks to bound memory
        total = 0
        M = len(emb)
        block = 512
        for s in range(0, M, block):
            d = np.abs(emb[s:s + block, None, :] - emb[None, :, :]).max(axis=2)
            total += int((d <= r).sum())
        return total - M  # remove self-matches

    B = _count(m)
    A = _count(m + 1)
    if A <= 0 or B <= 0:
        return SE_MAX
    return float(min(-np.log(A / B), SE_MAX))


# ═════════════════════════════════════════════════════════════════════
# 4. MRS — grouping and HF/LF split
# ═════════════════════════════════════════════════════════════════════

def _dp_segment(values: np.ndarray, k: int) -> list:
    """
    Partition `values` into exactly k CONTIGUOUS segments minimising the
    within-segment sum of squared deviations. Exact O(n^2 k) DP; n <= 10 here
    so cost is negligible.

    Contiguity is the point: it is what keeps a group a frequency BAND.
    """
    n = len(values)
    if k >= n:
        return [[i] for i in range(n)]
    if k <= 1:
        return [list(range(n))]

    # cost[i][j] = SSE of values[i..j]
    cost = np.full((n, n), np.inf)
    for i in range(n):
        s = 0.0
        s2 = 0.0
        for j in range(i, n):
            s += values[j]
            s2 += values[j] ** 2
            cnt = j - i + 1
            cost[i, j] = s2 - (s * s) / cnt

    dp = np.full((k + 1, n + 1), np.inf)
    back = np.zeros((k + 1, n + 1), dtype=int)
    dp[0, 0] = 0.0
    for kk in range(1, k + 1):
        for j in range(1, n + 1):
            for i in range(kk - 1, j):
                v = dp[kk - 1, i] + cost[i, j - 1]
                if v < dp[kk, j]:
                    dp[kk, j] = v
                    back[kk, j] = i

    # backtrack
    segs = []
    j = n
    for kk in range(k, 0, -1):
        i = back[kk, j]
        segs.append(list(range(i, j)))
        j = i
    return segs[::-1]


def mrs_group(
    imfs: np.ndarray,
    residual: np.ndarray,
    original: np.ndarray,
    k_rimf: int = 3,
    m: int = 2,
    r_factor: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Modal Recognition Strategy: IMFs -> exactly `k_rimf` RIMFs.  (Eq. 12-13)

    Steps
    -----
    1. Fold the residual in as the slowest component. It has near-zero SE by
       construction and therefore always lands in the low-frequency group,
       which is the behaviour the paper describes.
    2. SampEn of every component.
    3. Contiguous DP segmentation of the SE vector into exactly k groups.
    4. RIMF_j = sum of its members  (Eq. 12).
    5. SE of each RIMF recomputed on the SUMMED signal — the paper reports the
       SE of the reconstructed component (0.099, 0.1074), not the mean of its
       parts.

    Guarantees  sum(RIMFs) == original.
    """
    T = len(original)
    comps = (np.vstack([imfs, residual[None, :]]) if len(imfs)
             else residual[None, :].copy())

    n_comp = len(comps)
    deg = DEG_NORMAL
    se_capped = False

    se_comp = np.array([sample_entropy(c, m, r_factor) for c in comps])
    if np.any(se_comp >= SE_MAX):
        se_capped = True

    if n_comp < k_rimf:
        # Fewer components than groups. Pad the FAST end with zeros: the
        # window genuinely has no high-frequency content, and zeros land in
        # the HF branches where that is the semantically correct statement.
        deg = DEG_FEW_COMPONENTS
        pad = np.zeros((k_rimf - n_comp, T), dtype=np.float64)
        rimfs = np.vstack([pad, comps])
        groups = [[] for _ in range(k_rimf - n_comp)] + \
                 [[i] for i in range(n_comp)]
    else:
        # DP on SE; comps are already ordered fast -> slow by EMD.
        segs = _dp_segment(se_comp, k_rimf)
        rimfs = np.array([comps[s].sum(axis=0) for s in segs])
        groups = segs

    se_rimf = np.array([sample_entropy(r, m, r_factor) for r in rimfs])

    # THE invariant (Eq. 14 superposition).
    recon = rimfs.sum(axis=0)
    max_err = float(np.abs(recon - original).max())
    assert max_err < 1e-6 * max(1.0, float(np.abs(original).max())), (
        f"MRS broke the reconstruction identity: max|sum(RIMF)-q| = {max_err}")

    if se_capped and deg == DEG_NORMAL:
        deg = DEG_SE_CAPPED

    meta = {
        "n_comp": int(n_comp),
        "groups": groups,
        "se_comp": se_comp,
        "degenerate": int(deg),
        "recon_err": max_err,
    }
    return rimfs.astype(np.float32), se_rimf.astype(np.float32), meta


def split_high_low(
    se_rimf: np.ndarray,
    se_original: float,
    n_high: int,
    n_low: int,
    rule: str = "fixed_split",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign RIMFs to the Informer (HF) or the LSTM (LF) branch.  (Step 2)

    rule="fixed_split"   (default)
        Top `n_high` RIMFs by SE -> HF, rest -> LF. Produces a CONSTANT branch
        count, which the fixed-branch model requires.

    rule="se_threshold"  (paper-faithful, ablation)
        HF iff SE(RIMF_j) >= SE(q), per the paper. This yields a VARIABLE HF
        count, which would reintroduce the variable-branch problem, so the
        boundary is then forced back to n_high/n_low. The disagreement rate
        between the two rules is itself reportable, and the per-sample
        adaptivity is exactly the mechanism behind the paper's Appendix-B
        noise-robustness claim.

    Both rules read only CACHED SE values, so switching between them costs no
    recomputation.
    """
    k = len(se_rimf)
    assert n_high + n_low == k, f"n_high+n_low ({n_high}+{n_low}) != K ({k})"

    order = np.argsort(-se_rimf)  # descending SE == fastest first

    if rule == "se_threshold":
        hf_mask = se_rimf >= se_original
        if hf_mask.sum() != n_high:
            # Reconcile to the fixed branch structure.
            hf = np.sort(order[:n_high])
            lf = np.sort(order[n_high:])
            return hf, lf
        return np.where(hf_mask)[0], np.where(~hf_mask)[0]

    if rule != "fixed_split":
        raise ValueError(f"Unknown hf_rule: {rule!r}")

    return np.sort(order[:n_high]), np.sort(order[n_high:])


# ═════════════════════════════════════════════════════════════════════
# 5. Single-window driver
# ═════════════════════════════════════════════════════════════════════

def decompose_window(
    q: np.ndarray,
    cfg: dict,
    noise_bank: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    """
    Full CEEMDAN + MRS for one raw discharge window.

    Returns (rimfs (K,T) f32, se_rimf (K,) f32, se_original, n_imf, degenerate)

    Operates on the RAW series, never the normalised one, so the cache stays
    independent of the train/san_val split seed — otherwise changing
    SAN_VAL_FRACTION would invalidate a 2.8-hour precompute.

    The constant check uses the RAW std deliberately: BasinNormalizer maps a
    zero-variance basin to std->1.0 via its eps guard (dataset.py), which would
    hide exactly this case.
    """
    q = np.asarray(q, dtype=np.float64)
    T = len(q)
    K = cfg["n_rimf"]
    m = cfg["se_embed_dim"]
    r_fac = cfg["se_tolerance"]

    if q.std() < CONST_EPS:
        # Constant window: all content is "trend". Identity holds trivially.
        rimfs = np.zeros((K, T), dtype=np.float32)
        rimfs[-1] = q.astype(np.float32)
        return rimfs, np.zeros(K, dtype=np.float32), 0.0, 0, DEG_CONSTANT

    imfs, residual = ceemdan(
        q,
        n_trials=cfg["n_trials"],
        noise_std=cfg["noise_std"],
        max_imf=cfg["max_imf"],
        n_sift=cfg["n_sift"],
        noise_bank=noise_bank,
        seed=cfg["noise_seed"],
    )
    n_imf = len(imfs)

    rimfs, se_rimf, meta = mrs_group(imfs, residual, q, K, m, r_fac)
    se_original = sample_entropy(q, m, r_fac)

    deg = meta["degenerate"]
    if n_imf == 0 and deg == DEG_NORMAL:
        deg = DEG_NO_IMF

    return rimfs, se_rimf, float(se_original), n_imf, deg


# ═════════════════════════════════════════════════════════════════════
# 6. Cache
# ═════════════════════════════════════════════════════════════════════

KEY_FIELDS = ("noise_std", "n_trials", "max_imf", "se_embed_dim",
              "se_tolerance", "n_rimf", "n_sift", "decompose_ch", "noise_seed")


def cache_key(cfg: dict) -> str:
    """
    12-hex config hash.

    `hf_rule`, `n_high` and `n_low` are deliberately EXCLUDED: the HF/LF split
    is re-derived at load time from the cached SE values, so those ablations
    reuse one cache instead of triggering a 2.8-hour rebuild each.
    """
    payload = {k: cfg[k] for k in KEY_FIELDS}
    payload["algo_version"] = ALGO_VERSION
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def cache_path(h5_path: str, cfg: dict,
               cache_dir: Optional[Path] = None) -> Path:
    stem = Path(h5_path).stem
    d = Path(cache_dir or cfg.get("cache_dir", DATA_DIR))
    return d / f"rimf_{stem}_{cache_key(cfg)}.h5"


# Worker-local state. Module level because macOS SPAWNS processes: closures
# are not picklable and each worker re-imports this module fresh.
_W: dict = {}


def _worker_init(h5_path: str, cfg: dict) -> None:
    # Stop numpy from spawning N threads inside each of 12 processes.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _W["path"] = h5_path
    _W["cfg"] = cfg
    _W["file"] = None
    _W["pid"] = None
    _W["bank"] = make_noise_bank(
        cfg["n_trials"], cfg["window_len"], cfg["max_imf"], cfg["noise_seed"])


def _worker_file():
    """Worker-safe lazy handle — mirrors RunoffDataset._open (dataset.py)."""
    import h5py as _h5
    if _W["file"] is None or _W["pid"] != os.getpid():
        if _W["file"] is not None:
            _W["file"].close()
        _W["file"] = _h5.File(_W["path"], "r")
        _W["pid"] = os.getpid()
    return _W["file"]


def _worker(rows: np.ndarray):
    """Decompose a block of rows. Returns arrays for the parent to write."""
    cfg = _W["cfg"]
    f = _worker_file()
    ch = cfg["decompose_ch"]
    K = cfg["n_rimf"]

    rows = np.sort(np.asarray(rows))
    block = f["X"][rows][:, :, ch].astype(np.float64)

    n = len(rows)
    rimf = np.zeros((n, K, block.shape[1]), dtype=np.float32)
    se_r = np.zeros((n, K), dtype=np.float32)
    se_o = np.zeros(n, dtype=np.float32)
    nimf = np.zeros(n, dtype=np.uint8)
    degc = np.zeros(n, dtype=np.uint8)

    for i, q in enumerate(block):
        r, s, so, ni, dg = decompose_window(q, cfg, noise_bank=_W["bank"])
        rimf[i], se_r[i], se_o[i], nimf[i], degc[i] = r, s, so, ni, dg

    return rows, rimf, se_r, se_o, nimf, degc


def build_cache(
    h5_path: str,
    cfg: dict,
    out_path: Optional[Path] = None,
    n_proc: int = 12,
    block: int = 256,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Path:
    """
    Precompute RIMFs for every row of `h5_path`. Resumable.

    Workers compute, the PARENT writes: h5py has no concurrent-writer safety
    without MPI. Each block sets its `done` flag LAST, so a crash mid-block
    loses at most `block` windows.
    """
    import multiprocessing as mp

    import h5py

    out_path = Path(out_path or cache_path(h5_path, cfg))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        N_total, T, _ = f["X"].shape
    N = min(N_total, limit) if limit else N_total
    K = cfg["n_rimf"]
    cfg = {**cfg, "window_len": T}

    with h5py.File(out_path, "a") as out:
        if "rimf" not in out:
            out.create_dataset("rimf", shape=(N, K, T), dtype=np.float16,
                               chunks=(64, K, T))
            out.create_dataset("se_rimf", shape=(N, K), dtype=np.float32)
            out.create_dataset("se_original", shape=(N,), dtype=np.float32)
            out.create_dataset("n_imf", shape=(N,), dtype=np.uint8)
            out.create_dataset("degenerate", shape=(N,), dtype=np.uint8)
            out.create_dataset("done", shape=(N,), dtype=bool)
            out.attrs["config_json"] = json.dumps(
                {k: cfg[k] for k in KEY_FIELDS}, sort_keys=True)
            out.attrs["algo_version"] = ALGO_VERSION
            out.attrs["source"] = str(h5_path)

        todo = np.where(~out["done"][:N])[0]
        if verbose:
            print(f"[decompose] {out_path.name}: {len(todo):,} / {N:,} rows "
                  f"remaining")
        if len(todo) == 0:
            return out_path

        blocks = [todo[i:i + block] for i in range(0, len(todo), block)]
        t0 = time.time()
        done_rows = 0

        ctx = mp.get_context("spawn")
        with ctx.Pool(n_proc, initializer=_worker_init,
                      initargs=(h5_path, cfg)) as pool:
            for rows, rimf, se_r, se_o, nimf, degc in pool.imap_unordered(
                    _worker, blocks, chunksize=1):
                out["rimf"][rows] = rimf.astype(np.float16)
                out["se_rimf"][rows] = se_r
                out["se_original"][rows] = se_o
                out["n_imf"][rows] = nimf
                out["degenerate"][rows] = degc
                out["done"][rows] = True      # LAST — crash-safe ordering
                out.flush()

                done_rows += len(rows)
                if verbose:
                    el = time.time() - t0
                    rate = done_rows / el
                    eta = (len(todo) - done_rows) / rate / 60 if rate else 0
                    print(f"\r[decompose] {done_rows:,}/{len(todo):,}  "
                          f"{rate:.1f} win/s  ETA {eta:.0f} min", end="")
        if verbose:
            print(f"\n[decompose] done in {(time.time()-t0)/60:.1f} min")

    return out_path


class RIMFCache:
    """Worker-safe reader. Mirrors RunoffDataset's per-process handle logic."""

    def __init__(self, path: str):
        self.path = str(path)
        self._file = None
        self._pid = None
        with self._h5py().File(self.path, "r") as f:
            self.n = len(f["rimf"])
            self.k = f["rimf"].shape[1]

    @staticmethod
    def _h5py():
        import h5py
        return h5py

    def _open(self):
        if self._file is None or self._pid != os.getpid():
            if self._file is not None:
                self._file.close()
            self._file = self._h5py().File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def get(self, row: int):
        f = self._open()
        return (f["rimf"][row].astype(np.float32),
                f["se_rimf"][row].astype(np.float32),
                float(f["se_original"][row]))

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_file"] = None
        s["_pid"] = None
        return s

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

    def verify(self, h5_path: str, cfg: dict, n_samples: int = 200,
               verbose: bool = True) -> dict:
        """
        Re-read random rows and confirm sum(RIMFs) == raw discharge.

        This is the acceptance test for a multi-hour precompute: it checks the
        Eq. 14 identity actually survived the float16 round-trip.
        """
        h5py = self._h5py()
        rng = np.random.default_rng(0)
        f = self._open()
        done = np.where(f["done"][:])[0]
        pick = rng.choice(done, min(n_samples, len(done)), replace=False)
        pick = np.sort(pick)

        with h5py.File(h5_path, "r") as src:
            q = src["X"][pick][:, :, cfg["decompose_ch"]].astype(np.float64)

        r = f["rimf"][pick].astype(np.float64)
        err = np.abs(r.sum(axis=1) - q).max(axis=1)
        scale = np.maximum(np.abs(q).max(axis=1), 1e-6)
        rel = err / scale

        out = {"n_checked": len(pick), "max_abs_err": float(err.max()),
               "max_rel_err": float(rel.max()),
               "degenerate_counts": np.bincount(
                   f["degenerate"][pick], minlength=6).tolist()}
        if verbose:
            print(f"[decompose] verify: {len(pick)} rows, "
                  f"max rel recon err = {rel.max():.2e}")
        return out


def assemble_branch_inputs(x, rimf, se_rimf, se_orig, cfg):
    """
    Turn cached RIMFs into the `x_high` / `x_low` lists DMEL.forward expects.

    x        : (B, T, C) normalised full input
    rimf     : (B, K, T) normalised RIMFs, sum == x[:, :, TARGET]
    se_rimf  : (B, K)
    se_orig  : (B,)

    Returns (x_high, x_low) — lists of (B, T, C) tensors.

    Channel-replacement, not univariate
    ───────────────────────────────────
    Each branch receives the full 12-channel input with ONLY the discharge
    channel swapped for its RIMF. The paper is univariate, but our target is
    driven by 11 meteorological forcings; handing the HF Informer a
    single-channel RIMF would discard precipitation entirely and guarantee it
    underperforms the existing baseline. This also means zero changes to
    Informer / LSTMBranch / DMEL.

    `rimf_input_mode="univariate"` restores the paper-faithful (B,T,1) form as
    an ablation.
    """
    import torch

    mode = cfg.get("rimf_input_mode", "replace")
    rule = cfg.get("hf_rule", "fixed_split")
    n_high = cfg.get("n_high", 2)
    n_low = cfg.get("n_low", 1)
    ch = cfg.get("decompose_ch", TARGET_CHANNEL)

    B, K, T = rimf.shape

    # HF/LF assignment is derived from CACHED SE values, so switching
    # `hf_rule` costs no recomputation.
    if rule == "fixed_split":
        order = torch.argsort(se_rimf, dim=1, descending=True)
        hf_idx = order[:, :n_high]
        lf_idx = order[:, n_high:n_high + n_low]
    else:
        se_np = se_rimf.detach().cpu().numpy()
        so_np = se_orig.detach().cpu().numpy()
        hf_l, lf_l = [], []
        for i in range(B):
            h, l = split_high_low(se_np[i], float(so_np[i]),
                                  n_high, n_low, rule)
            hf_l.append(h)
            lf_l.append(l)
        hf_idx = torch.as_tensor(np.array(hf_l), device=rimf.device)
        lf_idx = torch.as_tensor(np.array(lf_l), device=rimf.device)

    def build(idx_col):
        comp = torch.gather(
            rimf, 1, idx_col.view(B, 1, 1).expand(B, 1, T)).squeeze(1)
        if mode == "univariate":
            return comp.unsqueeze(-1)
        out = x.clone()
        out[:, :, ch] = comp
        return out

    x_high = [build(hf_idx[:, j]) for j in range(hf_idx.shape[1])]
    x_low = [build(lf_idx[:, j]) for j in range(lf_idx.shape[1])]
    return x_high, x_low


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Build the RIMF cache.")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--n-proc", type=int, default=12)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None,
                    help="only decompose the first N rows (smoke test)")
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--n-rimf", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    cfg = dict(CEEMDAN_CFG)
    if a.n_trials:
        cfg["n_trials"] = a.n_trials
    if a.n_rimf:
        cfg["n_rimf"] = a.n_rimf

    src = str(TRAIN_H5 if a.split == "train" else TEST_H5)
    out = build_cache(src, cfg, n_proc=a.n_proc, block=a.block,
                      limit=a.limit, verbose=True)
    if a.verify:
        RIMFCache(str(out)).verify(src, cfg, n_samples=200)


# ═════════════════════════════════════════════════════════════════════
# 7. Self-test
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        _cli()
        _sys.exit(0)

    rng = np.random.default_rng(42)
    T = 336
    t = np.arange(T)

    print("\n── EMD completeness on a synthetic 3-tone signal ────────────")
    sig = (np.sin(2 * np.pi * t / 12) + 0.5 * np.sin(2 * np.pi * t / 60)
           + 0.02 * t)
    imfs, res = emd(sig)
    err = np.abs(imfs.sum(axis=0) + res - sig).max()
    assert err < 1e-9, f"EMD not complete: {err}"
    print(f"  {len(imfs)} IMFs, max|sum+res-x| = {err:.2e}  ✓")

    print("\n── CEEMDAN completeness (Torres Eq. 5) ──────────────────────")
    bank = make_noise_bank(8, T, 6, seed=1234)
    imfs, res = ceemdan(sig, n_trials=8, max_imf=6, noise_bank=bank)
    err = np.abs(imfs.sum(axis=0) + res - sig).max()
    assert err < 1e-9, f"CEEMDAN not complete: {err}"
    print(f"  {len(imfs)} IMFs, max|sum+res-x| = {err:.2e}  ✓")

    print("\n── SampleEntropy sanity ─────────────────────────────────────")
    se_const = sample_entropy(np.ones(T))
    se_sine = sample_entropy(np.sin(2 * np.pi * t / 24))
    se_noise = sample_entropy(rng.standard_normal(T))
    print(f"  constant={se_const:.4f}  sine={se_sine:.4f}  "
          f"noise={se_noise:.4f}")
    assert se_const == 0.0, "constant series must have SE 0"
    assert se_noise > se_sine, "white noise must be more complex than a sine"
    print("  ordering constant < sine < noise  ✓")

    print("\n── MRS grouping: identity + contiguity ──────────────────────")
    rimfs, se_r, meta = mrs_group(imfs, res, sig, k_rimf=3)
    assert rimfs.shape == (3, T)
    err = np.abs(rimfs.sum(axis=0) - sig).max()
    assert err < 1e-5, f"identity broken: {err}"
    for g in meta["groups"]:
        if g:
            assert g == list(range(g[0], g[-1] + 1)), f"non-contiguous {g}"
    print(f"  K=3, groups={meta['groups']}, max|sum-q|={err:.2e}  ✓")
    print(f"  SE per RIMF: {np.round(se_r, 4)}")

    print("\n── Degenerate: constant window ──────────────────────────────")
    cfg = {**CEEMDAN_CFG, "n_rimf": 3, "n_sift": 8, "noise_seed": 1234,
           "n_trials": 8}
    const = np.full(T, 0.37)
    r, s, so, ni, dg = decompose_window(const, cfg)
    assert dg == DEG_CONSTANT and ni == 0
    assert np.abs(r.sum(axis=0) - const).max() < 1e-6
    print(f"  code={dg} n_imf={ni}  identity holds  ✓")

    print("\n── Degenerate: monotone window ──────────────────────────────")
    mono = np.linspace(0.0, 1.0, T)
    r, s, so, ni, dg = decompose_window(mono, cfg)
    assert np.abs(r.sum(axis=0) - mono).max() < 1e-5
    print(f"  code={dg} n_imf={ni}  identity holds  ✓")

    print("\n── Degenerate: few components (K > n_comp) ──────────────────")
    r2, s2, meta2 = mrs_group(np.zeros((0, T)), mono, mono, k_rimf=3)
    assert meta2["degenerate"] == DEG_FEW_COMPONENTS
    assert np.abs(r2.sum(axis=0) - mono).max() < 1e-5
    assert np.allclose(r2[0], 0) and np.allclose(r2[1], 0)
    print(f"  zeros padded at the FAST end, trend in RIMF_K  ✓")

    print("\n── split_high_low: both rules ───────────────────────────────")
    se = np.array([0.9, 0.4, 0.02], dtype=np.float32)
    hf, lf = split_high_low(se, 0.5, 2, 1, "fixed_split")
    assert list(hf) == [0, 1] and list(lf) == [2]
    print(f"  fixed_split -> HF={list(hf)} LF={list(lf)}  ✓")
    hf2, lf2 = split_high_low(se, 0.5, 2, 1, "se_threshold")
    assert len(hf2) == 2 and len(lf2) == 1, "must reconcile to fixed counts"
    print(f"  se_threshold -> HF={list(hf2)} LF={list(lf2)} (reconciled)  ✓")

    print("\n── cache_key stability ──────────────────────────────────────")
    k1 = cache_key(cfg)
    k2 = cache_key({**cfg, "hf_rule": "se_threshold", "n_high": 1})
    k3 = cache_key({**cfg, "n_trials": 50})
    assert k1 == k2, "hf_rule must NOT change the key (re-derived at load)"
    assert k1 != k3, "n_trials MUST change the key"
    print(f"  key={k1}  invariant to hf_rule, sensitive to n_trials  ✓")

    print("\n── REAL DATA: decompose one window ──────────────────────────")
    if Path(TRAIN_H5).exists():
        import h5py
        with h5py.File(TRAIN_H5, "r") as f:
            q = f["X"][0][:, TARGET_CHANNEL].astype(np.float64)
        cfg_real = {**CEEMDAN_CFG, "n_rimf": 3, "n_sift": 8,
                    "noise_seed": 1234, "n_trials": 20}
        t0 = time.time()
        r, s, so, ni, dg = decompose_window(q, cfg_real)
        el = time.time() - t0
        err = np.abs(r.sum(axis=0) - q).max()
        print(f"  n_imf={ni} deg={dg} SE(q)={so:.4f} SE(RIMF)={np.round(s,4)}")
        print(f"  identity max err = {err:.2e}   ({el*1000:.0f} ms at "
              f"n_trials=20)")
        assert err < 1e-5
        print("  real-window decomposition  ✓")
    else:
        print("  [skip] train.h5 not found")

    print("\n✓ decompose.py self-test passed\n")
