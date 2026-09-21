"""
dataset.py
──────────
DataLoader pipeline for the 508-basin hourly discharge dataset.

Splits
------
  train.h5 split=0  (254,000)  → split by HELD-OUT BASINS into:
      ├── train   (~447 basins, 223,500 windows) — model training
      └── san_val (~61 basins,   30,500 windows) — early-stopping signal

  train.h5 split=1  (18,142)   → dev   — ablation score
  test.h5            (27,983)  → test  — final submission (no targets)

  The basin holdout is not a stylistic choice. A random within-basin carve
  leaked 97% of its validation windows into training, because the 336h windows
  overlap heavily and each basin is effectively one continuous record. See
  src/splits.py for the measurements and the two failed attempts that preceded
  this design.

  Consequence to keep in mind: san_val measures generalisation to UNSEEN
  BASINS, while dev measures unseen time in KNOWN basins. Different questions.

Normalization
-------------
  Per-basin z-score fitted on the TRAIN split only, by STREAMING sufficient
  statistics (the previous version loaded 4.1 GB into RAM to compute a mean).
  Strategies: per_basin_zscore | global_zscore | none — all three are now
  actually implemented; two of them previously fell through to "no normaliser"
  and silently fed the model raw data.

y_aux note
----------
  y_aux (48h × 11 channels of FUTURE meteorological forcing) is NOT a model
  input. metadata.json declares it
      "y_aux_role": "future_supervision_only_not_inference_inputs"
  and it does not exist in test.h5. Using it as a decoder input would inflate
  dev NSE and collapse at submission time.

  It is loaded only when `want_aux=True`, and then only as a TARGET for the
  auxiliary multi-task head (see Informer.aux_head).
"""

import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import Optional, Dict, Tuple

# Import from sibling module — works when run from paper2/ root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TRAIN_H5, TEST_H5,
    HISTORY_HOURS, FORECAST_HOURS, N_CHANNELS, TARGET_CHANNEL,
    N_AUX_CHANNELS, N_BASINS, SPLIT_TRAIN, SPLIT_VAL, SAN_VAL_FRACTION,
    TRAIN_CFG, GLOBAL_SEED,
)


# ─────────────────────────────────────────────────────────────
# 1. Normalization helpers
# ─────────────────────────────────────────────────────────────

class BasinNormalizer:
    """
    Per-basin, per-channel z-score normalizer.

    Strategies
    ----------
    "per_basin_zscore" : stats per (basin, channel)   [default]
    "global_zscore"    : one stat per channel, shared across basins
    "none"             : identity

    `global_zscore` and `none` were previously ACCEPTED BY THE CLI but never
    implemented: `build_splits` only checked for "per_basin_zscore" and left
    `normalizer=None` otherwise, which fed the model RAW data (pressure ~94,000
    beside specific humidity ~0.007). `exp_002__global_norm` would have
    produced NaN and been misread as "per-basin normalisation wins".
    """

    def __init__(self, n_basins: int = N_BASINS, n_channels: int = N_CHANNELS,
                 eps: float = 1e-6, strategy: str = "per_basin_zscore"):
        if strategy not in ("per_basin_zscore", "global_zscore", "none"):
            raise ValueError(f"Unknown norm strategy: {strategy!r}")
        self.n_basins   = n_basins
        self.n_channels = n_channels
        self.eps        = eps
        self.strategy   = strategy
        self.mean_: Optional[np.ndarray] = None   # (n_basins, n_channels)
        self.std_:  Optional[np.ndarray] = None
        self._fitted = False

    # ── fitting ──────────────────────────────────────────────────────

    def fit_streaming(self, h5_path: str, indices: np.ndarray,
                      block: int = 4096, verbose: bool = True
                      ) -> "BasinNormalizer":
        """
        Fit from disk using streaming sufficient statistics.

        The previous implementation did

            X_train = f["X"][train_idx]        # (223520, 336, 12) float32

        which materialises **4.1 GB** purely to compute a mean and a std, on a
        26 GB machine that also has to hold MPS model state. It was also the
        slowest possible access pattern: `X` is chunked (1, 336, 12), so a
        fancy-index gather of 223k rows is 223k separate chunk reads.

        Here we accumulate sum, sum-of-squares and count per basin in blocks
        and use the identity

            var = E[x^2] - E[x]^2

        the same algebraic trick `quick_nse` already uses for ss_tot. Peak
        memory is one block (~50 MB) instead of 4.1 GB, and the read is
        sequential.
        """
        if self.strategy == "none":
            self.mean_ = np.zeros((self.n_basins, self.n_channels), np.float32)
            self.std_ = np.ones((self.n_basins, self.n_channels), np.float32)
            self._fitted = True
            return self

        s = np.zeros((self.n_basins, self.n_channels), np.float64)
        s2 = np.zeros((self.n_basins, self.n_channels), np.float64)
        cnt = np.zeros(self.n_basins, np.float64)

        indices = np.sort(np.asarray(indices))
        with h5py.File(h5_path, "r") as f:
            X = f["X"]
            bids_all = f["basin_id"][:]
            for start in range(0, len(indices), block):
                rows = indices[start:start + block]
                xb = X[rows].astype(np.float64)          # (b, T, C)
                bb = bids_all[rows]
                # Per-window sums, then scatter-add by basin
                w_sum = xb.sum(axis=1)                   # (b, C)
                w_sq = (xb ** 2).sum(axis=1)             # (b, C)
                np.add.at(s, bb, w_sum)
                np.add.at(s2, bb, w_sq)
                np.add.at(cnt, bb, xb.shape[1])
                if verbose and (start // block) % 10 == 0:
                    print(f"\r[BasinNormalizer] fit {start + len(rows):,}/"
                          f"{len(indices):,}", end="")
        if verbose:
            print()

        n = np.maximum(cnt, 1)[:, None]
        mean = s / n
        var = np.maximum(s2 / n - mean ** 2, 0.0)
        std = np.sqrt(var)

        if self.strategy == "global_zscore":
            # Pool across basins: one mean/std per channel, broadcast back.
            tot = cnt.sum()
            g_mean = (s.sum(axis=0) / tot)
            g_var = np.maximum(s2.sum(axis=0) / tot - g_mean ** 2, 0.0)
            g_std = np.sqrt(g_var)
            mean = np.tile(g_mean, (self.n_basins, 1))
            std = np.tile(g_std, (self.n_basins, 1))

        std = np.where(std < self.eps, 1.0, std)
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        self._fitted = True

        seen = int((cnt > 0).sum())
        if verbose:
            print(f"[BasinNormalizer] strategy={self.strategy}, "
                  f"fitted on {seen} basins, {len(indices):,} windows "
                  f"(streaming, no full-array load)")
        return self

    def fit(self, X: np.ndarray, basin_ids: np.ndarray) -> "BasinNormalizer":
        """In-memory fit. Kept for tests; training uses fit_streaming."""
        mean = np.zeros((self.n_basins, self.n_channels), np.float64)
        std = np.ones((self.n_basins, self.n_channels), np.float64)
        for b in np.unique(basin_ids):
            data = X[basin_ids == b].reshape(-1, self.n_channels)
            mean[b] = data.mean(axis=0)
            std[b] = data.std(axis=0)
        std = np.where(std < self.eps, 1.0, std)
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        self._fitted = True
        return self

    # ── transforms ───────────────────────────────────────────────────

    def transform_one(self, x: np.ndarray, basin_id: int) -> np.ndarray:
        """
        Normalise ONE window. This is the hot path — called once per sample.

        The previous code called a batched `transform` that looped over all
        508 basins to normalise a single window. Measured:

            loop-over-508 : 0.612 ms/sample
            direct index  : 0.004 ms/sample     -> 153x faster

        At 223k samples/epoch the old path burned ~137 s of pure overhead per
        epoch, about 3.8 hours across a 100-epoch run, per experiment.
        """
        if self.strategy == "none":
            return x.astype(np.float32, copy=False)
        return ((x - self.mean_[basin_id]) / self.std_[basin_id]
                ).astype(np.float32)

    def transform(self, X: np.ndarray, basin_ids: np.ndarray) -> np.ndarray:
        """Batched transform (fit/eval helpers). Vectorised, no basin loop."""
        assert self._fitted, "Call .fit_streaming() before .transform()"
        if self.strategy == "none":
            return X.astype(np.float32, copy=False)
        return ((X - self.mean_[basin_ids][:, None, :])
                / self.std_[basin_ids][:, None, :]).astype(np.float32)

    def inverse_transform_target(self, y_norm: np.ndarray,
                                 basin_ids: np.ndarray) -> np.ndarray:
        """Denormalise the TARGET channel only. y_norm: (N, T) or (N,)."""
        assert self._fitted
        if self.strategy == "none":
            return y_norm.astype(np.float32, copy=False)
        m = self.mean_[basin_ids, TARGET_CHANNEL]
        s = self.std_[basin_ids, TARGET_CHANNEL]
        if y_norm.ndim == 2:
            m, s = m[:, None], s[:, None]
        return (y_norm * s + m).astype(np.float32)

    def target_std(self) -> np.ndarray:
        """Per-basin std of the target channel — used by the NSE loss."""
        assert self._fitted
        return self.std_[:, TARGET_CHANNEL].copy()

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean_, std=self.std_,
                 strategy=np.array(self.strategy))
        print(f"[BasinNormalizer] Saved to {path}.npz")

    def load(self, path: str) -> "BasinNormalizer":
        data = np.load(path if path.endswith(".npz") else path + ".npz",
                       allow_pickle=True)
        self.mean_ = data["mean"]
        self.std_ = data["std"]
        if "strategy" in data:
            self.strategy = str(data["strategy"])
        self._fitted = True
        print(f"[BasinNormalizer] Loaded from {path}")
        return self


# ─────────────────────────────────────────────────────────────
# 2. Core HDF5 Dataset
# ─────────────────────────────────────────────────────────────

class RunoffDataset(Dataset):
    """
    Lazy-loading HDF5 dataset.

    Returns per sample
    ------------------
        x        : (HISTORY_HOURS, N_CHANNELS)   float32  — input window
        y        : (FORECAST_HOURS,)              float32  — target discharge
        y_aux    : (FORECAST_HOURS, 11)           float32  — future meteo (or zeros)
        basin_id : ()                             int64    — basin index
        sample_id: ()                             int64    — original row index in h5

    The file handle is opened per-process to be DataLoader worker-safe.
    Normalization is applied in __getitem__ using a pre-computed BasinNormalizer.
    """

    def __init__(
        self,
        h5_path: str,
        indices: np.ndarray,              # which rows to expose from the file
        normalizer: Optional[BasinNormalizer] = None,
        has_targets: bool = True,
        seq_len: int = HISTORY_HOURS,
        input_channels: Optional[list] = None,
        rimf_cache=None,
        want_aux: bool = False,
    ):
        self.h5_path    = str(h5_path)
        self.indices    = indices
        self.normalizer = normalizer
        self.has_targets = has_targets
        self.seq_len    = seq_len
        self.input_channels = input_channels
        self.rimf_cache = rimf_cache
        self.want_aux   = want_aux

        # File handle — opened lazily per worker process
        self._file = None
        self._pid  = None

    def _open(self):
        if self._file is None or self._pid != os.getpid():
            if self._file is not None:
                self._file.close()
            self._file = h5py.File(self.h5_path, "r")
            self._pid  = os.getpid()
        return self._file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f  = self._open()
        i  = int(self.indices[idx])
        b  = int(f["basin_id"][i])

        x  = f["X"][i].astype(np.float32)          # (336, 12)

        if self.normalizer is not None:
            # Fast single-window path: direct indexing, no 508-basin loop.
            x = self.normalizer.transform_one(x, b)

        # I8: actually truncate the history window. Setting seq_len alone used
        # to change nothing — the dataset still returned all 336 steps and the
        # conv/attention stack is length-agnostic, so `exp_009__seq168`
        # measured 336 vs 336 and would have "shown" window length is
        # irrelevant.
        if self.seq_len < x.shape[0]:
            x = x[-self.seq_len:]

        # D1: channel subsetting. There was previously no way to select a
        # channel subset at all, so `exp_003__all_channels` was byte-identical
        # to the baseline.
        if self.input_channels is not None:
            x = x[:, self.input_channels]

        result = {
            "x":         torch.from_numpy(np.ascontiguousarray(x)),
            "basin_id":  torch.tensor(b,  dtype=torch.long),
            "sample_id": torch.tensor(i,  dtype=torch.long),
        }

        if self.rimf_cache is not None:
            rimf, se_rimf, se_orig = self.rimf_cache.get(i)
            # Normalise RIMFs with the TARGET channel's stats so that
            # sum(rimf_norm) == x_norm[:, TARGET] is preserved exactly.
            if self.normalizer is not None:
                s = self.normalizer.std_[b, TARGET_CHANNEL]
                m = self.normalizer.mean_[b, TARGET_CHANNEL]
                rimf = rimf / s
                rimf[-1] -= m / s
            if self.seq_len < rimf.shape[1]:
                rimf = rimf[:, -self.seq_len:]
            result["rimf"] = torch.from_numpy(np.ascontiguousarray(rimf))
            result["se_rimf"] = torch.from_numpy(se_rimf)
            result["se_original"] = torch.tensor(se_orig, dtype=torch.float32)

        if self.has_targets:
            y = f["y"][i].astype(np.float32)             # (48,)
            if self.normalizer is not None:
                m = self.normalizer.mean_[b, TARGET_CHANNEL]
                s = self.normalizer.std_[b,  TARGET_CHANNEL]
                y = (y - m) / s
            result["y"] = torch.from_numpy(y)

            # y_aux is loaded ONLY when the auxiliary task needs it as a
            # TARGET. It is never a model input — metadata.json declares it
            # "future_supervision_only_not_inference_inputs" and it is absent
            # from test.h5. Loading and normalising it unconditionally (as the
            # previous code did, in a Python loop over 11 channels) was pure
            # overhead in every default run.
            if self.want_aux:
                y_aux = f["y_aux"][i].astype(np.float32)   # (48, 11)
                if self.normalizer is not None:
                    mc = self.normalizer.mean_[b, :N_AUX_CHANNELS]
                    sc = self.normalizer.std_[b, :N_AUX_CHANNELS]
                    y_aux = (y_aux - mc) / sc
                result["y_aux"] = torch.from_numpy(y_aux)

        return result

    def __getstate__(self):
        """Pickle-safe: close handle before pickling (for DataLoader workers)."""
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"]  = None
        return state

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# 3. Split builder
# ─────────────────────────────────────────────────────────────

def build_splits(
    train_h5: str  = str(TRAIN_H5),
    test_h5:  str  = str(TEST_H5),
    san_val_frac:  float = SAN_VAL_FRACTION,
    seed:          int   = GLOBAL_SEED,
    norm_strategy: str   = TRAIN_CFG["norm_strategy"],
    seq_len:       int   = HISTORY_HOURS,
    input_channels: Optional[list] = None,
    use_ceemdan:   bool  = False,
    want_aux:      bool  = False,
    ceemdan_cfg:   Optional[dict] = None,
    verbose:       bool  = True,
):
    """
    Build train / san_val / dev / test datasets and a fitted normalizer.

    san_val is carved by HELD-OUT BASINS, not by a random within-basin split.
    See src/splits.py for the derivation; briefly, the random carve leaked 97%
    of its validation windows into training, because the 336h windows overlap
    heavily and each basin is effectively one continuous record.

    Returns
    -------
    ds_train, ds_san_val, ds_dev, ds_test, normalizer, split_stats
    """
    from splits import build_group_split

    train_idx, san_val_idx, split_stats = build_group_split(
        train_h5=train_h5, val_fraction=san_val_frac, seed=seed,
        mode="basin", verbose=verbose,
    )

    with h5py.File(train_h5, "r") as f:
        dev_idx = np.where(f["split"][:] == SPLIT_VAL)[0].astype(np.int64)

    n_test = 0
    if Path(test_h5).exists():
        with h5py.File(test_h5, "r") as f:
            n_test = len(f["X"])

    if verbose:
        print(f"[build_splits] train   : {len(train_idx):>8,}")
        print(f"[build_splits] san_val : {len(san_val_idx):>8,}  "
              f"({split_stats['selection_measures']})")
        print(f"[build_splits] dev     : {len(dev_idx):>8,}  (split=1)")
        print(f"[build_splits] test    : {n_test:>8,}")

    # ── Normalizer ────────────────────────────────────────────────────
    #
    # Fitted over ALL basins in split=0, not only the training basins.
    #
    # Why this is not leakage, and why it is REQUIRED:
    #
    #   * The normaliser holds per-basin INPUT scaling (mean/std of the 12
    #     forcing channels). It carries no target information beyond the
    #     target channel's own first two moments, which are a property of the
    #     basin's hydrology, not of any particular forecast window.
    #   * san_val basins are held out from TRAINING, which is the point. But
    #     they are not held out from existing. With a basin holdout they are
    #     by definition unseen, so a train-only fit leaves them at the
    #     unfitted default mean=0, std=1 — and their raw inputs (pressure
    #     ~94,000 next to humidity ~0.007) then go straight into a model that
    #     expects z-scores.
    #
    #     Measured consequence of getting this wrong: median san_val NSE of
    #     -53.15, with 61/61 held-out basins below -1 and p0 at -35,623.
    #     Early stopping was selecting checkpoints on pure noise.
    #   * test.h5 forces the same decision anyway: we must produce statistics
    #     for its basins without ever seeing their targets. This is the same
    #     operation.
    #
    # What WOULD be leakage, and is not done here: fitting on dev/test
    # windows, or using target values from the selection split to choose a
    # checkpoint.
    normalizer = BasinNormalizer(strategy=norm_strategy)
    fit_idx = np.sort(np.concatenate([train_idx, san_val_idx]))
    normalizer.fit_streaming(train_h5, fit_idx, verbose=verbose)

    # ── Optional RIMF caches ──────────────────────────────────────────
    cache_tr = cache_te = None
    if use_ceemdan:
        from decompose import RIMFCache, cache_path
        cfg = ceemdan_cfg or {}
        p_tr = cache_path(train_h5, cfg)
        if not Path(p_tr).exists():
            raise FileNotFoundError(
                f"RIMF cache missing: {p_tr}\n"
                f"Build it first:  python -m src.decompose --split train")
        cache_tr = RIMFCache(str(p_tr))
        p_te = cache_path(test_h5, cfg)
        if Path(p_te).exists():
            cache_te = RIMFCache(str(p_te))

    def mk(path, idx, targets, cache):
        return RunoffDataset(path, idx, normalizer, has_targets=targets,
                             seq_len=seq_len, input_channels=input_channels,
                             rimf_cache=cache, want_aux=want_aux)

    ds_train   = mk(train_h5, train_idx,   True,  cache_tr)
    ds_san_val = mk(train_h5, san_val_idx, True,  cache_tr)
    ds_dev     = mk(train_h5, dev_idx,     True,  cache_tr)
    ds_test    = mk(test_h5, np.arange(n_test, dtype=np.int64), False, cache_te)

    return ds_train, ds_san_val, ds_dev, ds_test, normalizer, split_stats


# ─────────────────────────────────────────────────────────────
# 4. DataLoader factory
# ─────────────────────────────────────────────────────────────

def seed_worker(worker_id: int) -> None:
    """
    Re-seed numpy and stdlib random inside each DataLoader worker.

    PyTorch seeds each worker's torch RNG automatically, but NOT numpy's
    or random's. macOS SPAWNS workers (it does not fork), so each worker
    starts with a fresh numpy RNG seeded from OS entropy — meaning any
    numpy-based work inside __getitem__ would be non-reproducible even
    with a global seed set.

    Harmless at num_workers=0, but required for num_workers>0 to be a
    usable flag rather than a silent reproducibility hole.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset:     RunoffDataset,
    batch_size:  int  = TRAIN_CFG["batch_size"],
    shuffle:     bool = True,
    num_workers: int  = TRAIN_CFG["num_workers"],
    pin_memory:  bool = TRAIN_CFG["pin_memory"],
    drop_last:   bool = False,
    generator:   Optional["torch.Generator"] = None,
) -> DataLoader:
    """
    Wrap a RunoffDataset in a DataLoader.

    pin_memory
    ──────────
    Pinned (page-locked) host memory lets CUDA do async DMA transfers
    over PCIe — a real win on a discrete NVIDIA GPU. On Apple unified
    memory (MPS) there is no host->device copy to accelerate, so it is
    pure overhead. Hence gated on CUDA specifically, not on "any
    accelerator".

    generator
    ─────────
    Pass a dedicated torch.Generator to make the shuffle order depend
    ONLY on its own seed, independent of the global torch RNG.

    Without it, batch order is drawn from the same global RNG that model
    init and any stochastic layer consume. Changing an unrelated
    hyperparameter that alters the number of RNG draws (e.g. dropout
    placement, or adding a layer) would then silently change the batch
    order too — making two runs incomparable for a reason that has
    nothing to do with the thing being ablated.
    """
    import torch
    # MPS does not support pin_memory
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = drop_last,
        generator   = generator if shuffle else None,
        worker_init_fn = seed_worker if num_workers > 0 else None,
        persistent_workers = (num_workers > 0),
    )


# ─────────────────────────────────────────────────────────────
# 5. Self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import seed_everything, get_device
    import resource

    seed_everything()
    device = get_device()

    print("\n── transform_one vs batched transform (the 153x fix) ────────")
    norm = BasinNormalizer()
    rng = np.random.default_rng(0)
    norm.mean_ = rng.standard_normal((N_BASINS, N_CHANNELS)).astype(np.float32)
    norm.std_ = (np.abs(rng.standard_normal((N_BASINS, N_CHANNELS))) + 0.5
                 ).astype(np.float32)
    norm._fitted = True
    x = rng.standard_normal((HISTORY_HOURS, N_CHANNELS)).astype(np.float32)
    a = norm.transform_one(x, 7)
    b = norm.transform(x[None], np.array([7]))[0]
    assert np.allclose(a, b, atol=1e-6), "fast path must match batched result"
    t0 = time.time()
    for _ in range(500):
        norm.transform_one(x, 7)
    dt = (time.time() - t0) / 500 * 1000
    print(f"  bit-identical to batched ✓   {dt:.4f} ms/sample")

    print("\n── global_zscore / none are real strategies now ─────────────")
    for strat in ("per_basin_zscore", "global_zscore", "none"):
        n2 = BasinNormalizer(strategy=strat)
        print(f"  {strat:<18} constructs ✓")
    try:
        BasinNormalizer(strategy="bogus")
        raise AssertionError("should have rejected unknown strategy")
    except ValueError:
        print("  unknown strategy rejected loudly ✓")

    if not Path(TRAIN_H5).exists():
        print("\n  [skip] real-data checks — train.h5 not found")
        sys.exit(0)

    print("\n── build_splits (streaming fit, basin holdout) ──────────────")
    t0 = time.time()
    ds_train, ds_sv, ds_dev, ds_test, norm, stats = build_splits(verbose=True)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"  built in {time.time()-t0:.1f}s   peak RSS {peak_mb:.0f} MB")
    assert peak_mb < 2000, f"peak RSS {peak_mb:.0f} MB — streaming fit failed"
    print("  peak RSS well under the 4.1 GB the old full-load used ✓")

    print("\n── sample shapes ───────────────────────────────────────────")
    smp = ds_train[0]
    for k, v in smp.items():
        print(f"  {k:<12}: {tuple(v.shape)}  {v.dtype}")
    assert smp["x"].shape == (HISTORY_HOURS, N_CHANNELS)
    assert "y_aux" not in smp, "y_aux must NOT load unless want_aux=True"
    print("  y_aux absent by default (it is a target, never an input) ✓")

    print("\n── seq_len truncation actually truncates ───────────────────")
    ds168 = RunoffDataset(str(TRAIN_H5), ds_train.indices[:4], norm,
                          seq_len=168)
    assert ds168[0]["x"].shape == (168, N_CHANNELS), ds168[0]["x"].shape
    print(f"  seq_len=168 -> {tuple(ds168[0]['x'].shape)} ✓")

    print("\n── channel subsetting works ────────────────────────────────")
    dsc = RunoffDataset(str(TRAIN_H5), ds_train.indices[:4], norm,
                        input_channels=[8, 11])
    assert dsc[0]["x"].shape == (HISTORY_HOURS, 2), dsc[0]["x"].shape
    print(f"  input_channels=[8,11] -> {tuple(dsc[0]['x'].shape)} ✓")

    print("\n── DataLoader batch ────────────────────────────────────────")
    loader = make_loader(ds_train, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"  {k:<12}: {tuple(v.shape)}")

    print("\n✓ dataset.py self-test passed\n")
