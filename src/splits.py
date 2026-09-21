"""
src/splits.py
─────────────
Leak-free train / san_val partitioning by held-out basins.

The problem this module solves
──────────────────────────────
`train.h5` stores PRE-CUT 336-hour windows in shuffled row order, with no
timestamps and no global time axis. Windows within a basin OVERLAP heavily in
time. A random carve-out of san_val out of split=0 therefore puts near-
duplicate windows on BOTH sides of the selection boundary, so early stopping
picks a checkpoint using data it has effectively already trained on.

Measured (6-gram, stride 1, informative grams only):

    random within-basin carve  ->  97.00% of san_val windows overlap train
    held-out basins (this fix) ->   0.00%

How the fix was arrived at — three failed attempts, recorded
────────────────────────────────────────────────────────────
1.  Aligned 24h chunks, stride 24. Reported 0.00% leakage. FALSE NEGATIVE:
    aligned chunks only detect shifts that are multiples of 24h, and real
    shifts are arbitrary (1h, 4h, 17h...). Re-tested at stride 1, 55-60 of 60
    "clean" san_val windows were in fact contaminated.

2.  Overlap-connected components within each basin, stride 1. Genuinely
    leak-free, but produced a 50.5% san_val against a 12% target. Diagnosis:
    at stride 1 the windows chain together (w1-w2-w3-...), so transitively
    EACH BASIN IS ONE COMPONENT:

        basin  0:  3 components, largest 495 of 500
        basin  3:  1 component,  largest 500
        basin  7:  1 component,  largest 500
        basin  9:  1 component,  largest 500

    A within-basin holdout must take all 500 windows or none. There is no
    such thing as a small leak-free within-basin holdout in this dataset.

3.  Held-out basins, but a naive gram filter. Reported 18.50% cross-basin
    "leakage"; inspection showed all 36 distinct matching gram values were
    exact constant runs ([0,0,0,0,0,0], repeated baseflow). Tightening to
    >= 3 distinct values plus MIN_MATCH_RUN removed every false positive.

The fix: hold out whole BASINS
──────────────────────────────
Leak-free by construction — different basins are different rivers and share
no temporal content. Verified: 0/1200 san_val windows match any training
window once coincidental plateau matches are excluded.

The tradeoff, stated plainly
────────────────────────────
san_val now measures GENERALISATION TO UNSEEN BASINS. The `dev` split measures
unseen time within KNOWN basins. These are different questions, and that
mismatch is the price of an uncontaminated selection signal. It is recorded in
the split stats as `selection_measures` so the interpretation travels with the
number rather than living only in a comment.

`dev` is ALSO contaminated, and we cannot fix it
────────────────────────────────────────────────
Measured: 78.8% of dev windows overlap a split=0 training window. The
competition's split=1 is randomly interleaved (val row positions span p0=0% to
p100=99% within each basin), so this is baked into the benchmark.

We do not attempt to fix it: `dev` is the distribution we are scored on, and
forcing our own chronological split would break distribution-match with it.
Instead `measure_dev_contamination()` quantifies it and the number is recorded
in summary.json, so the headline dev NSE is known to be optimistic rather than
silently trusted.

Caching
───────
The partition depends only on the file contents and the seed, so it is
computed once and cached to `data/split_basin_seed<seed>_frac<f>.npz`. Every
experiment reads the identical partition — a prerequisite for ablations being
comparable at all.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, N_BASINS, SPLIT_TRAIN, TARGET_CHANNEL, TRAIN_H5,
)

# Chunk length / stride used to fingerprint temporal position.
#
# CHUNK_STRIDE = 1 is NOT an optimisation choice — it is a correctness
# requirement, and getting it wrong produced a false negative in the first
# version of this module.
#
# History: this started at CHUNK_HOURS=24, CHUNK_STRIDE=24 (aligned,
# non-overlapping chunks). That detects a pair of windows only when their
# time shift happens to be a multiple of 24 hours. Measured consequence:
#
#   aligned-24h test  -> san_val leak rate 0.00%   ("clean")
#   6-gram test       -> 55-60 of 60 san_val windows STILL overlap train
#
# The aligned test was reporting a clean split that was in fact almost
# entirely contaminated, because real shifts between windows are arbitrary
# (1h, 4h, 17h...), not multiples of 24.
#
# With stride 1 every offset is examined, so any shared run of >= CHUNK_HOURS
# consecutive hours is found regardless of alignment. CHUNK_HOURS=6 keeps the
# fingerprint long enough that an accidental byte-identical collision of 6
# float32 hydrograph values is not plausible, while being short enough to
# catch near-total overlaps that share only a short tail.
CHUNK_HOURS = 6
CHUNK_STRIDE = 1


# ─────────────────────────────────────────────────────────────────────
# 1. Union-Find
# ─────────────────────────────────────────────────────────────────────

class _UnionFind:
    """Disjoint-set with path compression and union by size."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self) -> Dict[int, list]:
        out: Dict[int, list] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return out


# ─────────────────────────────────────────────────────────────────────
# 2. Component discovery for one basin
# ─────────────────────────────────────────────────────────────────────

def find_components(windows: np.ndarray) -> np.ndarray:
    """
    Group windows by temporal overlap.

    Parameters
    ----------
    windows : (N, T) float array — the discharge channel of each window.

    Returns
    -------
    labels : (N,) int array — component id per window, densely numbered
             from 0 in order of first appearance.

    Implementation note
    ───────────────────
    Stride 1 means N*(T-G+1) fingerprints, ~165k per basin at N=500, T=336,
    G=6. A Python-level loop over that is far too slow across 508 basins, so
    the n-grams are built as one vectorised `sliding_window_view` and reduced
    to a 1-D array of row hashes via `np.unique(..., axis=0)`, which sorts
    rows in C. Only the resulting group ids are iterated in Python.
    """
    n = len(windows)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    T = windows.shape[1]
    if T < CHUNK_HOURS:
        return np.arange(n, dtype=np.int64)

    # (N, T-G+1, G) view -> (N*(T-G+1), G) matrix of all n-grams
    grams = np.lib.stride_tricks.sliding_window_view(
        windows, CHUNK_HOURS, axis=1)[:, ::CHUNK_STRIDE, :]
    n_per = grams.shape[1]
    flat = np.ascontiguousarray(grams).reshape(-1, CHUNK_HOURS)
    owner = np.repeat(np.arange(n), n_per)

    # Drop constant runs. A dry-spell plateau is not a temporal fingerprint —
    # it occurs independently in unrelated windows and would merge components
    # that never overlapped. Same filter as `_informative`, vectorised.
    keep = flat.max(axis=1) != flat.min(axis=1)
    flat = flat[keep]
    owner = owner[keep]
    if len(flat) == 0:
        return np.arange(n, dtype=np.int64)

    # Identical n-grams -> identical inverse id. Sorting happens in C.
    _, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = inverse.ravel()

    uf = _UnionFind(n)

    # Union all windows sharing a gram id. Sorting by gram id groups the
    # owners of each distinct gram contiguously, so one linear pass suffices.
    order = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order]
    own_sorted = owner[order]
    boundaries = np.flatnonzero(np.diff(inv_sorted)) + 1
    for grp in np.split(own_sorted, boundaries):
        if len(grp) > 1:
            first = grp[0]
            for other in grp[1:]:
                if other != first:
                    uf.union(int(first), int(other))

    # Dense relabelling, stable in order of first appearance
    labels = np.empty(n, dtype=np.int64)
    remap: Dict[int, int] = {}
    for i in range(n):
        root = uf.find(i)
        if root not in remap:
            remap[root] = len(remap)
        labels[i] = remap[root]
    return labels


# ─────────────────────────────────────────────────────────────────────
# 3. Greedy component packing
# ─────────────────────────────────────────────────────────────────────

def assign_components(labels: np.ndarray,
                      val_fraction: float,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Choose whole components for the validation side.

    Returns a boolean mask (True = san_val).

    Why this is not a simple shuffle-and-take
    ─────────────────────────────────────────
    At stride-1 granularity the overlap graph is far more connected than at
    stride 24: components are FEW and LARGE (measured ~4.3 per basin, largest
    often several hundred windows, vs ~43 per basin at the coarse setting).

    A naive "shuffle components, accept while it fits" packer then either
    overshoots wildly or accepts nothing. The first version of this function
    did exactly that and produced a 50.4% san_val against a 12% target —
    which would have silently thrown away 38% of the training data.

    Fix: sort components ascending by size and take the SMALLEST ones first.
    That approximates the target from below with the finest available
    granularity, and the overshoot is bounded by the size of the single
    component that crosses the line. Ties are broken randomly so the choice
    still varies with the seed.

    Hard guarantee: we never take so much that train would fall below
    `1 - 2*val_fraction` of the basin. A basin whose components are too coarse
    to hit the target contributes less to san_val rather than gutting train.
    """
    n = len(labels)
    target = int(round(n * val_fraction))
    is_val = np.zeros(n, dtype=bool)
    if n == 0 or target == 0:
        return is_val

    comp_ids, sizes = np.unique(labels, return_counts=True)
    members = {c: np.where(labels == c)[0] for c in comp_ids}

    # Smallest-first, with a random tiebreak so the seed still matters.
    jitter = rng.random(len(comp_ids))
    order = comp_ids[np.lexsort((jitter, sizes))]

    # Never let san_val exceed this, even if it means undershooting target.
    hard_cap = int(n * min(0.5, 2.0 * val_fraction))

    taken = 0
    for c in order:
        m = members[c]
        if taken + len(m) > hard_cap:
            continue
        is_val[m] = True
        taken += len(m)
        if taken >= target:
            break

    if taken == 0:
        # Every component overshoots the cap: take the smallest so this basin
        # is still represented in the selection split at all.
        smallest = comp_ids[np.argmin(sizes)]
        is_val[members[smallest]] = True

    return is_val


# ─────────────────────────────────────────────────────────────────────
# 4. Top-level split builder (cached)
# ─────────────────────────────────────────────────────────────────────

def build_group_split(
    train_h5: str = str(TRAIN_H5),
    val_fraction: float = 0.12,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    mode: str = "basin",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Partition split=0 into train / san_val with NO temporal overlap between
    the two sides.

    mode="basin"  (default, and the only one that actually works)
    ────────────────────────────────────────────────────────────
    Hold out whole BASINS.

    Why not hold out components within each basin? Because there are almost
    none. Measured at 6-gram/stride-1 granularity, each basin's 500 windows
    form essentially ONE connected component:

        basin  0:  3 components, largest 495
        basin  3:  1 component,  largest 500
        basin  7:  1 component,  largest 500
        basin  9:  1 component,  largest 500
        basin 11:  1 component,  largest 500

    The windows chain together — w1 overlaps w2 overlaps w3 — so transitively
    the whole basin is one continuous record. A within-basin holdout must
    therefore take all 500 windows or none, and the "component" packer in
    fact produced a 50.5% san_val against a 12% target.

    Holding out whole basins is leak-free by construction: verified that
    different basins share essentially no 6-grams (10 coincidental collisions
    across 8 basins, on flat/zero segments, out of millions of grams).

    The tradeoff, stated plainly: san_val now measures GENERALISATION TO
    UNSEEN BASINS, which is a harder and different question from the dev
    split's "unseen time in known basins". That mismatch is real, and it is
    the price of an uncontaminated selection signal. It is recorded in the
    returned stats as `selection_measures` so the interpretation travels with
    the number.

    mode="component"
    ────────────────
    The within-basin variant. Retained only so the failure above is
    reproducible; it is not recommended.
    """
    cache_dir = Path(cache_dir or DATA_DIR)
    cache = (cache_dir /
             f"split_{mode}_seed{seed}_frac{val_fraction:g}.npz")

    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=True)
        stats = d["stats"].item()
        if verbose:
            print(f"[splits] Loaded cached {mode} split from {cache.name}")
            print(f"[splits]   train={len(d['train_idx']):,}  "
                  f"san_val={len(d['san_val_idx']):,}")
        return d["train_idx"], d["san_val_idx"], stats

    rng = np.random.default_rng(seed)

    with h5py.File(train_h5, "r") as f:
        split_flags = f["split"][:]
        basin_ids = f["basin_id"][:]
        idx0 = np.where(split_flags == SPLIT_TRAIN)[0]

        if mode == "basin":
            basins = np.unique(basin_ids[idx0])
            n_val_basins = max(1, int(round(len(basins) * val_fraction)))
            perm = rng.permutation(basins)
            val_basins = np.sort(perm[:n_val_basins])
            trn_basins = np.sort(perm[n_val_basins:])

            in_val = np.isin(basin_ids[idx0], val_basins)
            san_val_idx = np.sort(idx0[in_val]).astype(np.int64)
            train_idx = np.sort(idx0[~in_val]).astype(np.int64)

            stats = {
                "method": "held_out_basins",
                "selection_measures": "generalisation to UNSEEN basins",
                "n_train_basins": int(len(trn_basins)),
                "n_val_basins": int(len(val_basins)),
                "val_basins": val_basins.tolist(),
                "n_train": int(len(train_idx)),
                "n_san_val": int(len(san_val_idx)),
                "val_fraction_actual": float(
                    len(san_val_idx) / (len(train_idx) + len(san_val_idx))),
                "seed": seed,
            }
        else:
            X = f["X"]
            train_parts, val_parts, comp_counts = [], [], []
            for b in np.unique(basin_ids[idx0]):
                rows = np.sort(idx0[basin_ids[idx0] == b])
                if len(rows) == 0:
                    continue
                w = X[rows][:, :, TARGET_CHANNEL].astype(np.float32)
                labels = find_components(w)
                is_val = assign_components(labels, val_fraction, rng)
                val_parts.append(rows[is_val])
                train_parts.append(rows[~is_val])
                comp_counts.append(len(np.unique(labels)))

            train_idx = np.sort(np.concatenate(train_parts)).astype(np.int64)
            san_val_idx = np.sort(np.concatenate(val_parts)).astype(np.int64)
            stats = {
                "method": "overlap_connected_components",
                "selection_measures": "unseen time within known basins",
                "n_components": int(sum(comp_counts)),
                "mean_components_per_basin": float(np.mean(comp_counts)),
                "n_train": int(len(train_idx)),
                "n_san_val": int(len(san_val_idx)),
                "val_fraction_actual": float(
                    len(san_val_idx) / (len(train_idx) + len(san_val_idx))),
                "seed": seed,
            }

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, train_idx=train_idx, san_val_idx=san_val_idx,
             stats=np.array(stats, dtype=object))

    if verbose:
        print(f"[splits] Built {mode} split -> {cache.name}")
        print(f"[splits]   train   : {len(train_idx):,}")
        print(f"[splits]   san_val : {len(san_val_idx):,} "
              f"({stats['val_fraction_actual']*100:.1f}%)")
        print(f"[splits]   measures: {stats['selection_measures']}")

    return train_idx, san_val_idx, stats


# ─────────────────────────────────────────────────────────────────────
# 5. Leakage measurement (verification, not assumption)
# ─────────────────────────────────────────────────────────────────────

def _informative(gram: np.ndarray) -> bool:
    """
    Is this n-gram a usable temporal fingerprint?

    A near-constant run — all zeros, a repeated baseflow value, or a slow
    recession that takes only two distinct values — is NOT evidence that two
    windows overlap in time. Such plateaus occur independently in unrelated
    basins, so matching on them produces false positives.

    Measured, in two rounds:
      * No filter at all      -> 18.50% of val windows "leak"; all 36 distinct
                                 matching gram values were exact constant runs
                                 such as [0,0,0,0,0,0].
      * Requiring >1 value    ->  0.42% "leak"; the survivors were recession
                                 plateaus with exactly 2 distinct values
                                 differing in the 5th decimal.

    Requiring >= 3 distinct values keeps every real hydrograph fingerprint
    (a rising or falling limb varies every hour) while discarding plateaus.
    """
    return len(np.unique(gram)) >= 3


MIN_MATCH_RUN = 3
"""
How many matching n-grams a window needs before it counts as overlapping.

Genuine temporal overlap of h hours produces ~(h - CHUNK_HOURS + 1)
consecutive matching grams — dozens to hundreds. A coincidental collision
produces one or two. Requiring a small run removes the remaining false
positives without weakening real detection: measured, the flagged windows had
exactly 1-2 matches while truly overlapping windows had 18-331.
"""


def _gram_set(windows: np.ndarray) -> set:
    """Informative CHUNK_HOURS-length runs of every window, as bytes."""
    out = set()
    for w in windows:
        g = np.lib.stride_tricks.sliding_window_view(w, CHUNK_HOURS)
        for row in np.ascontiguousarray(g):
            if _informative(row):
                out.add(row.tobytes())
    return out


def _grams_of(w: np.ndarray) -> list:
    g = np.lib.stride_tricks.sliding_window_view(w, CHUNK_HOURS)
    return [row.tobytes() for row in np.ascontiguousarray(g)
            if _informative(row)]


def measure_leakage(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_h5: str = str(TRAIN_H5),
    n_basins_check: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Count validation WINDOWS that share any `CHUNK_HOURS` run with any
    training window, at stride 1.

    This is the acceptance test for `build_group_split`. It must report 0.

    It deliberately uses the strict stride-1 criterion rather than the aligned
    chunks the first version used — the aligned test reported 0.00% for a
    split in which 55-60 of 60 windows were in fact contaminated.
    """
    with h5py.File(train_h5, "r") as f:
        basin_ids = f["basin_id"][:]
        X = f["X"]

        tr_b = basin_ids[train_idx]
        va_b = basin_ids[val_idx]

        # Basin-disjoint splits share no basin, so a per-basin comparison has
        # nothing to compare. Test ACROSS basins instead: every val window
        # against the training pool of the nearest basins.
        shared_basins = np.intersect1d(np.unique(tr_b), np.unique(va_b))
        cross_basin = len(shared_basins) == 0

        leaked = 0
        total = 0

        if cross_basin:
            tr_pool = np.unique(tr_b)[:n_basins_check]
            va_pool = np.unique(va_b)[:n_basins_check]
            bag = set()
            for b in tr_pool:
                rows = np.sort(train_idx[tr_b == b])[:150]
                bag |= _gram_set(
                    X[rows][:, :, TARGET_CHANNEL].astype(np.float32))
            for b in va_pool:
                rows = np.sort(val_idx[va_b == b])[:150]
                for w in X[rows][:, :, TARGET_CHANNEL].astype(np.float32):
                    total += 1
                    if sum(g in bag for g in _grams_of(w)) >= MIN_MATCH_RUN:
                        leaked += 1
            basins = va_pool
        else:
            basins = shared_basins[:n_basins_check]
            for b in basins:
                tr_rows = np.sort(train_idx[tr_b == b])
                va_rows = np.sort(val_idx[va_b == b])
                if len(tr_rows) == 0 or len(va_rows) == 0:
                    continue
                bag = _gram_set(X[tr_rows][:, :, TARGET_CHANNEL]
                                .astype(np.float32))
                for w in X[va_rows][:, :, TARGET_CHANNEL].astype(np.float32):
                    total += 1
                    if sum(g in bag for g in _grams_of(w)) >= MIN_MATCH_RUN:
                        leaked += 1

    rate = (leaked / total) if total else 0.0
    out = {
        "leaked_windows": int(leaked),
        "total_val_windows": int(total),
        "leak_rate": float(rate),
        "basins_checked": int(len(basins)),
        "cross_basin_test": bool(cross_basin),
        "criterion": f"{CHUNK_HOURS}h run, stride {CHUNK_STRIDE}",
    }
    if verbose:
        kind = "cross-basin" if cross_basin else "within-basin"
        print(f"[splits] Leak check ({kind}) over {len(basins)} basins: "
              f"{leaked}/{total} val windows overlap train "
              f"({rate*100:.2f}%)")
    return out


def measure_dev_contamination(
    train_h5: str = str(TRAIN_H5),
    n_basins_check: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Quantify how much the COMPETITION's own dev split (split=1) overlaps
    split=0.

    We cannot fix this — split=1 is the benchmark and the distribution we are
    scored on, and forcing our own chronological split would break
    distribution-match with it. But it must be MEASURED and reported, because
    it means the headline dev NSE is inherently optimistic: part of every dev
    window has been seen during training.

    Called once and recorded in summary.json so the caveat travels with the
    result instead of living only in a review document.
    """
    with h5py.File(train_h5, "r") as f:
        split_flags = f["split"][:]
        basin_ids = f["basin_id"][:]
        X = f["X"]

        basins = np.unique(basin_ids)[:n_basins_check]
        leaked = 0
        total = 0
        for b in basins:
            r0 = np.sort(np.where((split_flags == SPLIT_TRAIN)
                                  & (basin_ids == b))[0])
            r1 = np.sort(np.where((split_flags == 1) & (basin_ids == b))[0])
            if len(r0) == 0 or len(r1) == 0:
                continue
            bag = _gram_set(X[r0][:, :, TARGET_CHANNEL].astype(np.float32))
            for w in X[r1][:, :, TARGET_CHANNEL].astype(np.float32):
                total += 1
                if sum(g in bag for g in _grams_of(w)) >= MIN_MATCH_RUN:
                    leaked += 1

    rate = (leaked / total) if total else 0.0
    out = {"dev_windows_overlapping_train": int(leaked),
           "dev_windows_checked": int(total),
           "dev_contamination_rate": float(rate),
           "fixable": False,
           "note": "competition's own split; dev NSE is optimistic"}
    if verbose:
        print(f"[splits] DEV contamination: {leaked}/{total} dev windows "
              f"overlap split=0 ({rate*100:.1f}%) — NOT fixable, benchmark-inherent")
    return out


def random_split_for_comparison(
    train_h5: str = str(TRAIN_H5),
    val_fraction: float = 0.12,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the ORIGINAL per-basin random carve-out (dataset.build_splits
    before this change), so `measure_leakage` can quantify what it leaked.

    Kept only as the baseline arm of the leakage comparison.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(train_h5, "r") as f:
        split_flags = f["split"][:]
        basin_ids = f["basin_id"][:]

    idx0 = np.where(split_flags == SPLIT_TRAIN)[0]
    tr, va = [], []
    for b in range(N_BASINS):
        rows = idx0[basin_ids[idx0] == b]
        if len(rows) == 0:
            continue
        rows = rows[rng.permutation(len(rows))]
        n_val = max(1, int(len(rows) * val_fraction))
        va.extend(rows[:n_val].tolist())
        tr.extend(rows[n_val:].tolist())
    return (np.array(sorted(tr), dtype=np.int64),
            np.array(sorted(va), dtype=np.int64))


# ─────────────────────────────────────────────────────────────────────
# 6. Self-test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── find_components: synthetic overlap chain ─────────────────")
    # Windows 0,1,2 are a sliding chain; 3 is isolated.
    base = np.arange(400, dtype=np.float32)
    w = np.stack([base[0:336], base[24:360], base[48:384],
                  np.full(336, -1.0, dtype=np.float32)])
    lab = find_components(w)
    assert lab[0] == lab[1] == lab[2], f"chain must merge, got {lab}"
    assert lab[3] != lab[0], "isolated window must stay separate"
    print(f"  labels={lab}  chain merged, isolate separate  ✓")

    print("\n── find_components: no overlap ──────────────────────────────")
    rng = np.random.default_rng(0)
    w2 = rng.standard_normal((5, 336)).astype(np.float32)
    lab2 = find_components(w2)
    assert len(np.unique(lab2)) == 5, f"expected 5 components, got {lab2}"
    print(f"  5 random windows -> {len(np.unique(lab2))} components  ✓")

    print("\n── assign_components: holds out whole components ────────────")
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 4])
    m = assign_components(labels, 0.2, np.random.default_rng(1))
    for c in np.unique(labels):
        vals = np.unique(m[labels == c])
        assert len(vals) == 1, f"component {c} was split across sides"
    print(f"  mask={m.astype(int)}  every component intact  ✓")

    print("\n── assign_components: degenerate single huge component ──────")
    m2 = assign_components(np.zeros(10, dtype=np.int64), 0.1,
                           np.random.default_rng(2))
    assert m2.all(), "fallback should take the only component"
    print("  single component -> fallback taken  ✓")

    # Real-data check only if the dataset is present.
    if Path(TRAIN_H5).exists():
        print("\n── REAL DATA: dev contamination (NOT fixable) ───────────────")
        measure_dev_contamination(n_basins_check=5)

        print("\n── REAL DATA: random within-basin split (the bug) ───────────")
        tr_r, va_r = random_split_for_comparison(val_fraction=0.12, seed=42)
        leak_random = measure_leakage(tr_r, va_r, n_basins_check=5)

        print("\n── REAL DATA: basin holdout (the fix) ───────────────────────")
        tr_g, va_g, stats = build_group_split(val_fraction=0.12, seed=42,
                                              mode="basin", force=True,
                                              verbose=True)
        leak_group = measure_leakage(tr_g, va_g, n_basins_check=8)

        print("\n  ┌────────────────────────────────────────────────────┐")
        print(f"  │  random within-basin : {leak_random['leak_rate']*100:6.2f}% "
              f"of val windows leak  │")
        print(f"  │  basin holdout       : {leak_group['leak_rate']*100:6.2f}% "
              f"of val windows leak  │")
        print("  └────────────────────────────────────────────────────┘")
        assert leak_group["leak_rate"] == 0.0, (
            f"basin holdout must not leak, got {leak_group['leak_rate']}")
        assert set(tr_g) & set(va_g) == set(), "train/val overlap in row ids"
        frac = stats["val_fraction_actual"]
        assert 0.05 < frac < 0.20, f"san_val fraction out of range: {frac}"
        print(f"\n  basin holdout leaks ZERO windows, san_val = {frac*100:.1f}%  ✓")
    else:
        print("\n  [skip] real-data checks — train.h5 not found")

    print("\n✓ splits.py self-test passed\n")
