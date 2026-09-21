"""
src/train.py
────────────
Training loop components:
  build_optimizer()  — AdamW/Adam with decay/no-decay param groups
  build_scheduler()  — warmup_cosine | cosine | sgdr | plateau | none
  build_loss()       — MSE | MAE | Huber | NSE-loss
  train_one_epoch()  — one full pass; returns dict incl. grad_norm
  quick_nse()        — fast per-basin NSE on san_val for early stopping
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRAIN_CFG, TARGET_CHANNEL, FORECAST_HOURS


# ─────────────────────────────────────────────────────────────────────
# 1. Optimizer — with decay / no-decay parameter groups
# ─────────────────────────────────────────────────────────────────────

def build_optimizer(
    model : nn.Module,
    cfg   : dict,
    log   = print,
) -> torch.optim.Optimizer:
    """
    AdamW (default) or Adam, with weight decay applied ONLY to
    multi-dimensional weight tensors.

    Why parameter groups matter
    ───────────────────────────
    Weight decay must NOT touch 1-D parameters:

      * LayerNorm gamma/beta — the Informer has LayerNorm in every encoder
        and decoder block. Decaying gamma toward zero directly fights the
        layer's ability to rescale features, which is its entire purpose.
      * BatchNorm gamma/beta — same argument; DistilLayer uses BatchNorm1d.
      * All biases           — decaying a bias shifts the function's offset
        toward zero for no regularisation benefit.
      * Combiner weight      — LinearCombiner.w is the DMEL analogue of a
        fusion weight. Decay would bias it toward softmax-uniform, i.e.
        toward the fixed-sum baseline, confounding the very quantity
        ablation E1 exists to measure.

    LSTM note: nn.LSTM exposes weight_ih_l*/weight_hh_l* (2-D, decayed)
    and bias_ih_l*/bias_hh_l* (1-D, not decayed), so the ndim split works
    without special-casing.

    Optional `combiner_lr_mult` puts the combiner weight in its own group
    at lr x mult. A single scalar (or n_branches-length vector) competing
    against millions of other parameters at one shared lr often cannot
    travel far enough during training — the same problem paper1 measured
    for its fusion alpha. Default 1.0 = identity, so the effect is
    measured rather than assumed.
    """
    opt_name  = cfg.get("optimizer", "adamw").lower()
    lr        = cfg.get("lr", 1e-4)
    wd        = cfg.get("weight_decay", 1e-4)
    comb_mult = cfg.get("combiner_lr_mult", 1.0)

    decay, no_decay, combiner = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_combiner = (".combiner." in name or name.startswith("combiner."))

        if is_combiner and comb_mult != 1.0:
            # Own group so the combiner can take larger steps without
            # touching the rest of the network. Never decayed.
            combiner.append(param)
        elif param.ndim <= 1 or name.endswith(".bias"):
            # LayerNorm/BatchNorm gamma,beta + all biases + 1-D combiner
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay,    "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if combiner:
        groups.append({"params": combiner, "weight_decay": 0.0,
                       "lr": lr * comb_mult})

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=lr)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name!r}")

    n_decay    = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    log(f"  [optim] {opt_name}  lr={lr:g}")
    log(f"  [optim] weight_decay={wd:g} on {len(decay)} tensors "
        f"({n_decay:,} params)")
    log(f"  [optim] weight_decay=0.0 on {len(no_decay)} tensors "
        f"({n_no_decay:,} params)  <- LayerNorm/BatchNorm/bias")
    if combiner:
        log(f"  [optim] combiner in own group at lr x{comb_mult:g} "
            f"({len(combiner)} tensors, no decay)")

    return optimizer


# ─────────────────────────────────────────────────────────────────────
# 2. LR Scheduler
# ─────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer  : torch.optim.Optimizer,
    cfg        : dict,
    total_steps: int,
):
    """
    Builds the LR scheduler.

    Supported modes (cfg["scheduler"]):
      "warmup_cosine" — linear warmup for warmup_frac of steps, then cosine
      "cosine"        — CosineAnnealingLR (no warmup)
      "sgdr"          — CosineAnnealingWarmRestarts
      "plateau"       — ReduceLROnPlateau
      "none"          — constant LR (returns None)
    """
    mode = cfg.get("scheduler", "warmup_cosine").lower()

    if mode == "none":
        return None

    elif mode == "warmup_cosine":
        warmup_steps = max(1, int(total_steps * cfg.get("warmup_frac", 0.05)))
        eta_min      = cfg.get("eta_min", 1e-6)
        base_lr      = cfg.get("lr", 1e-4)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / warmup_steps          # linear ramp
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return eta_min / base_lr + (1 - eta_min / base_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif mode == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = cfg.get("epochs", 100),
            eta_min = cfg.get("eta_min", 1e-6),
        )

    elif mode == "sgdr":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0     = cfg.get("sgdr_T0", 20),
            T_mult  = cfg.get("sgdr_T_mult", 2),
            eta_min = cfg.get("eta_min", 1e-6),
        )

    elif mode == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",   # we track NSE (higher = better)
            factor   = 0.5,
            patience = 5,
            min_lr   = cfg.get("eta_min", 1e-6),
        )

    else:
        raise ValueError(f"Unknown scheduler: {mode!r}")


def scheduler_granularity(cfg: dict) -> str:
    """
    Return "batch", "epoch", or "none" — how often the scheduler steps.

    Used by experiment.py to decide whether early-stopping patience needs
    relaxing (see adjust_patience_for_scheduler).
    """
    mode = cfg.get("scheduler", "warmup_cosine").lower()
    if mode == "none":
        return "none"
    if mode == "plateau":
        return "epoch"
    return "batch"      # warmup_cosine | cosine | sgdr


def adjust_patience_for_scheduler(
    patience : int,
    epochs   : int,
    cfg      : dict,
    log      = print,
) -> int:
    """
    Relax early-stopping patience when a cosine-family schedule is active.

    The problem
    ───────────
    A cosine schedule anneals lr to its floor at exactly `epochs`. The
    low-lr tail is where the model actually settles into a minimum. An
    early stopper with short patience will fire during the mid-cosine
    plateau — before the anneal has delivered any of its benefit — and
    discard precisely the phase the schedule was configured to reach.

    The schedule and the stopper want opposite things:
      * scheduler: "let me run the full `epochs` so lr reaches the floor"
      * stopper  : "stop as soon as the metric stalls"

    Resolution
    ──────────
    Require patience >= epochs // 2 whenever the schedule is batch-stepped
    (cosine / warmup_cosine / sgdr), so a transient dip cannot abort the
    anneal. The stopper is NOT removed — it stays as a runaway guard for
    genuine divergence.

    Plateau and "none" schedules are left alone: plateau reacts to the
    metric itself (so the stopper and scheduler agree), and a flat lr has
    no anneal to protect.
    """
    gran = scheduler_granularity(cfg)
    if gran != "batch":
        return patience

    floor = epochs // 2
    if patience < floor:
        log(f"  [patience] {patience} -> {floor} "
            f"(cosine-family schedule needs to reach its lr floor)")
        return floor
    return patience


# ─────────────────────────────────────────────────────────────────────
# 3. Loss functions
# ─────────────────────────────────────────────────────────────────────

class NSELoss(nn.Module):
    """
    Basin-averaged NSE loss (Kratzert et al. 2019).

    WHY THE PREVIOUS VERSION WAS BROKEN
    ───────────────────────────────────
    It computed

        ss_res = ((y_true - y_pred)**2).sum()
        ss_tot = ((y_true - y_true.mean())**2).sum() + 1e-8
        return ss_res / ss_tot

    `ss_tot` depends only on `y_true`, so it is a CONSTANT with respect to the
    predictions. The gradient is therefore exactly MSE's gradient times a
    per-batch scalar. Verified numerically:

        d(NSELoss)/dy_hat  =  2(y_hat - y) / ss_tot
        d(MSE)/dy_hat      =  2(y_hat - y) / n
        ratio: min = max = 1.007567   (constant across every element)

    So it was MSE with a random per-batch rescaling — effective-learning-rate
    jitter, not a distinct objective. `exp_012__nse_loss` would have measured
    noise. (Credit: HANDOFF_REVIEW.md.)

    WHY PLAIN MSE IS ALREADY BASIN-WEIGHTED HERE
    ────────────────────────────────────────────
    Targets are per-basin z-scored, so MSE on normalised values is

        (1/N) * sum (y - y_hat)^2 / sigma_b^2

    which is Kratzert's NSE* loss at epsilon = 0. The alignment the ablation
    was reaching for already exists — it lives in BasinNormalizer, not here.

    WHAT THIS CLASS ACTUALLY ADDS: the epsilon
    ──────────────────────────────────────────
    Per-basin sigma spans a 521x range across the 508 basins, and 14 basins
    (2.8%) have sigma < 0.01 — essentially flat lines. At epsilon = 0, dividing
    a flat basin's error by sigma^2 amplifies PURE NOISE to the same gradient
    magnitude as a genuinely dynamic basin. Kratzert's epsilon damps exactly
    that.

    Given z-scored inputs, re-weighting by sigma_b^2 / (sigma_b + eps)^2
    recovers the damped form:

        loss = mean_n  (y_n - y_hat_n)^2 * sigma_b^2 / (sigma_b + eps)^2

    eps = 0 reduces to plain MSE on z-scored targets, so the ablation
    eps in {0, 0.1, 0.5} is a clean test of whether the near-flat basins are
    degrading training.
    """

    def __init__(self, basin_std: torch.Tensor, eps: float = 0.1):
        super().__init__()
        # (n_basins,) raw per-basin target std, from BasinNormalizer.
        self.register_buffer("basin_std", basin_std)
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                basin_ids: torch.Tensor) -> torch.Tensor:
        sd = self.basin_std[basin_ids].clamp_min(1e-8)      # (B,)
        w = (sd / (sd + self.eps)) ** 2                     # (B,)
        se = (y_true - y_pred) ** 2                         # (B, T)
        return (se.mean(dim=1) * w).mean()


def build_loss(cfg: dict, basin_std: Optional[torch.Tensor] = None
               ) -> nn.Module:
    """
    Build the training criterion.

    Losses taking only (pred, true) are plain nn modules. `nse` additionally
    needs per-basin std, so callers must pass `basin_std`; `forward_batch`
    handles the extra argument.
    """
    loss_name = cfg.get("loss", "mse").lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        # delta must match the ERROR scale, not be left at the torch default.
        # Measured on z-scored targets: median |z| = 0.311, p99 = 3.89,
        # max = 82.2, and 93.1% of targets have |z| < 1.0. So delta=1.0 is
        # quadratic for ~93% of data and linear only for the extremes —
        # a defensible default, but it is now tunable rather than implicit.
        return nn.HuberLoss(delta=cfg.get("huber_delta", 1.0))
    if loss_name == "nse":
        if basin_std is None:
            raise ValueError(
                "loss='nse' requires basin_std (per-basin target std). "
                "Pass it from the fitted BasinNormalizer.")
        return NSELoss(basin_std, eps=cfg.get("nse_eps", 0.1))
    raise ValueError(f"Unknown loss: {loss_name!r}")


def needs_basin_ids(criterion: nn.Module) -> bool:
    """True when the criterion's forward takes basin_ids."""
    return isinstance(criterion, NSELoss)


# ─────────────────────────────────────────────────────────────────────
# 3b. Single source of truth for building model inputs
# ─────────────────────────────────────────────────────────────────────

def forward_batch(model, batch, cfg, device, normalizer=None):
    """
    Build model inputs and run the forward pass. ONE implementation, used by
    train_one_epoch, quick_nse and evaluate_model.

    Why this exists
    ───────────────
    The three call sites each built their own inputs, and they disagreed:

      * train_one_epoch  built x_dec from y_aux and passed basin_ids
      * quick_nse        built x_dec from y_aux and passed basin_ids
      * evaluate_model   called model(x, y_aux=y_aux) — a kwarg DMEL.forward
                         does not accept (instant TypeError), never passed
                         basin_ids (so use_basin_emb crashed), and never built
                         x_dec at all

    So evaluation would have crashed on two flags and, once fixed naively,
    would still have fed the model DIFFERENT inputs than training did. A single
    helper makes that class of drift impossible.

    Returns
    -------
    (y_pred, aux_pred) — aux_pred is None unless the aux task is enabled.
    """
    x = batch["x"].to(device, non_blocking=True)
    basin_ids = batch["basin_id"].to(device, non_blocking=True)

    kwargs = {}
    if cfg.get("use_basin_emb", False):
        kwargs["basin_ids"] = basin_ids

    if cfg.get("use_ceemdan", False):
        from decompose import assemble_branch_inputs
        x_high, x_low = assemble_branch_inputs(
            x,
            batch["rimf"].to(device, non_blocking=True),
            batch["se_rimf"].to(device, non_blocking=True),
            batch["se_original"].to(device, non_blocking=True),
            cfg,
        )
        kwargs["x_high"] = x_high
        kwargs["x_low"] = x_low

    # NOTE: x_dec is deliberately NOT built from y_aux. See Informer.forward
    # and the aux-task rationale — y_aux does not exist at test time, so using
    # it as a decoder input is target leakage.
    if cfg.get("aux_task", False):
        return model(x, return_aux=True, **kwargs)
    return model(x, **kwargs), None


def compute_loss(criterion, y_pred, y_true, basin_ids,
                 aux_pred=None, batch=None, cfg=None, device=None):
    """Main loss plus the optional auxiliary-forcing term."""
    if needs_basin_ids(criterion):
        loss = criterion(y_pred, y_true, basin_ids)
    else:
        loss = criterion(y_pred, y_true)

    if aux_pred is not None and batch is not None and "y_aux" in batch:
        w = cfg.get("aux_loss_weight", 0.3)
        y_aux = batch["y_aux"].to(device, non_blocking=True)
        loss = loss + w * nn.functional.mse_loss(aux_pred, y_aux)
    return loss


# ─────────────────────────────────────────────────────────────────────
# 4. One training epoch
# ─────────────────────────────────────────────────────────────────────

# Sample the gradient norm every Nth step when clipping is DISABLED.
# Measuring on every step would cost a full extra reduction over all
# parameters; every 50th is <1% overhead and plenty for a per-epoch mean.
GRAD_NORM_SAMPLE_EVERY = 50


def train_one_epoch(
    model      : nn.Module,
    loader     : DataLoader,
    optimizer  : torch.optim.Optimizer,
    criterion  : nn.Module,
    scheduler,                           # LambdaLR / None / ReduceLROnPlateau
    device     : torch.device,
    cfg        : dict,
    epoch      : int = 0,
) -> dict:
    """
    One full pass over the training DataLoader.

    Returns
    -------
    dict with keys:
      train_loss : float        — mean loss over all batches
      grad_norm  : float | None — mean PRE-clip gradient L2 norm
      n_batches  : int

    Why grad_norm is always measured
    ───────────────────────────────
    The gradient norm is the ONLY signal that tells you whether
    clip_grad is doing anything at all:

      * If mean grad_norm << clip_grad, the clipper never fires and you
        are paying for a no-op on every step.
      * If mean grad_norm climbs toward clip_grad, the clipper is firing
        constantly and silently rescaling every update — which changes
        the effective learning rate without any log line saying so.

    When clipping is ON we get the norm for free (clip_grad_norm_ returns
    it). When clipping is OFF we compute it on a sparse sample of steps,
    so turning clipping off never makes gradient behaviour unobservable.
    """
    model.train()
    total_loss    = 0.0
    n_batches     = len(loader)
    clip_grad     = cfg.get("clip_grad", 1.0)

    grad_sum, n_grad = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]",
                leave=False, ncols=80)

    for step, batch in enumerate(pbar):
        y_true    = batch["y"].to(device)            # (B, 48)
        basin_ids = batch["basin_id"].to(device)     # (B,)

        # set_to_none=True frees the grad buffers instead of zero-filling:
        # slightly faster and makes "no gradient" distinguishable from
        # "zero gradient" in the sampling branch below.
        optimizer.zero_grad(set_to_none=True)

        # Single shared input builder — see forward_batch().
        y_pred, aux_pred = forward_batch(model, batch, cfg, device)
        loss = compute_loss(criterion, y_pred, y_true, basin_ids,
                            aux_pred, batch, cfg, device)
        loss.backward()

        # Gradient norm / clipping.
        # Clip BEFORE optimizer.step(), AFTER backward() — the only valid
        # position. clip_grad_norm_ returns the PRE-clip total norm, so
        # when clipping is on the measurement is free.
        if clip_grad > 0:
            total_norm = nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad,
            )
            grad_sum += float(total_norm)
            n_grad   += 1
        elif step % GRAD_NORM_SAMPLE_EVERY == 0:
            with torch.no_grad():
                sq = sum((p.grad.detach() ** 2).sum()
                         for p in model.parameters()
                         if p.grad is not None)
                grad_sum += float(sq.sqrt())
            n_grad += 1

        optimizer.step()

        # Step per-batch schedulers AFTER optimizer.step(), per PyTorch's
        # documented order. Stepping before would apply the next step's lr
        # to this update and emit a warning.
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "grad_norm" : (grad_sum / n_grad) if n_grad else None,
        "n_batches" : n_batches,
    }


# ─────────────────────────────────────────────────────────────────────
# 5. Quick NSE eval on san_val (for early stopping)
# ─────────────────────────────────────────────────────────────────────

def quick_nse(
    model      : nn.Module,
    loader     : DataLoader,
    normalizer,
    device     : torch.device,
    cfg        : dict,
) -> float:
    """
    Fast per-basin NSE on san_val. Returns MEDIAN NSE across basins.
    Called by the early stopping callback after every epoch.

    Single pass over the loader. ss_tot is accumulated algebraically:

        ss_tot = sum((obs - mean)^2)
               = sum(obs^2) - n * mean^2
               = sum_sq - (sum^2 / n)

    which is exact and avoids the second pass an explicit mean would need.
    At ~30k san_val samples evaluated every epoch, that halves the
    early-stopping overhead.
    """
    from collections import defaultdict
    import numpy as np

    model.eval()

    # stats[basin_id] = [ss_res, sum_obs, sum_obs_sq, n]
    stats: Dict = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    with torch.no_grad():
        for batch in loader:
            y_true_t  = batch["y"].to(device)
            basin_ids = batch["basin_id"].cpu().numpy()

            # Same input builder as training — see forward_batch().
            y_pred_t, _ = forward_batch(model, batch, cfg, device)

            y_pred_np = y_pred_t.cpu().numpy().astype(np.float64)
            y_true_np = y_true_t.cpu().numpy().astype(np.float64)

            for i, bid in enumerate(basin_ids):
                m = normalizer.mean_[bid, TARGET_CHANNEL]
                s = normalizer.std_[ bid, TARGET_CHANNEL]
                obs  = y_true_np[i] * s + m
                pred = y_pred_np[i] * s + m

                st = stats[bid]
                st[0] += float(((obs - pred) ** 2).sum())   # ss_res
                st[1] += float(obs.sum())                    # sum_obs
                st[2] += float((obs ** 2).sum())             # sum_obs_sq
                st[3] += len(obs)                            # n

    nse_vals = []
    for ss_res, sum_obs, sum_sq, n in stats.values():
        if n < 2:
            continue
        ss_tot = sum_sq - (sum_obs ** 2) / n
        if ss_tot > 1e-10:
            nse_vals.append(1.0 - ss_res / ss_tot)

    return float(np.median(nse_vals)) if nse_vals else 0.0
