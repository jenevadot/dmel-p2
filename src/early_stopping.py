"""
early_stopping.py
─────────────────
EarlyStopping callback: monitors median NSE on san_val, saves the best
checkpoint, and signals the training loop to stop when patience runs out.

Design
------
  EarlyStopping   — stateful callback, one instance lives across all epochs.
                    Call .step(nse, model, optimizer, epoch) after each
                    san_val evaluation. Check .should_stop to halt training.

  CheckpointManager — lightweight helper that saves / loads best_model.pt
                      and a companion best_metrics.json alongside it.

Usage in training loop
----------------------
    stopper = EarlyStopping(
        patience  = TRAIN_CFG["early_stop_patience"],
        min_delta = TRAIN_CFG["early_stop_min_delta"],
        save_dir  = "experiments/exp_001",
    )

    for epoch in range(max_epochs):
        train_one_epoch(model, train_loader, ...)
        nse = quick_eval_nse(model, san_val_loader, normalizer, device)

        stopper.step(nse, model, optimizer, scheduler, epoch)

        if stopper.should_stop:
            print(f"Early stop at epoch {epoch}, best was epoch {stopper.best_epoch}")
            break

    # After loop: restore best weights automatically
    stopper.restore_best(model, optimizer, scheduler)
    # Now run final evaluation on dev split
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── make sure sibling modules are importable when run directly ────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRAIN_CFG


# ─────────────────────────────────────────────────────────────────────
# 1.  Checkpoint manager
# ─────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Saves and loads model + optimizer + scheduler state to disk.

    Files written to save_dir:
      best_model.pt       — full torch checkpoint (state_dicts)
      best_metrics.json   — the NSE and epoch that triggered the save
    """

    def __init__(self, save_dir: str):
        self.save_dir   = Path(save_dir)
        self.model_path = self.save_dir / "best_model.pt"
        self.meta_path  = self.save_dir / "best_metrics.json"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch: int,
             nse: float, extra: Optional[dict] = None) -> None:
        """
        Save model + optimizer + scheduler state dicts.
        Overwrites any previous checkpoint in this directory.
        """
        import torch

        ckpt = {
            "epoch"          : epoch,
            "nse"            : nse,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        if scheduler is not None:
            ckpt["scheduler_state"] = scheduler.state_dict()

        torch.save(ckpt, self.model_path)

        meta = {"epoch": epoch, "nse": nse}
        if extra:
            meta.update(extra)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [checkpoint] Saved  epoch={epoch:3d}  NSE={nse:.6f}"
              f"  →  {self.model_path}")

    def load(self, model, optimizer=None,
             scheduler=None, device=None) -> dict:
        """
        Restore weights (and optionally optimizer/scheduler) from checkpoint.
        Returns the checkpoint dict so the caller can read 'epoch' and 'nse'.
        """
        import torch

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {self.model_path}"
            )

        map_loc = device if device is not None else "cpu"
        ckpt    = torch.load(self.model_path, map_location=map_loc,
                             weights_only=False)

        model.load_state_dict(ckpt["model_state"])

        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        print(f"  [checkpoint] Loaded epoch={ckpt['epoch']:3d}"
              f"  NSE={ckpt['nse']:.6f}  ←  {self.model_path}")
        return ckpt

    def exists(self) -> bool:
        return self.model_path.exists()


# ─────────────────────────────────────────────────────────────────────
# 2.  EarlyStopping callback
# ─────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitors median NSE on san_val after every epoch.

    Stops training and restores the best checkpoint when the metric
    has not improved by at least `min_delta` for `patience` epochs.

    Parameters
    ----------
    patience   : int   — epochs without improvement before stopping (default 10)
    min_delta  : float — minimum NSE improvement to count as progress (default 1e-4)
    save_dir   : str   — directory for best_model.pt and logs
    verbose    : bool  — print per-epoch status line

    State (read after training)
    ---------------------------
    .should_stop  : bool  — True when patience is exhausted
    .best_nse     : float — best median NSE seen so far
    .best_epoch   : int   — epoch at which best_nse was achieved
    .counter      : int   — how many epochs since last improvement

    Log format
    ----------
    Every epoch prints one line:
      [ES] epoch=  5  NSE=0.7832  best=0.7832  counter=0  ✓ saved
      [ES] epoch=  6  NSE=0.7801  best=0.7832  counter=1
      ...
      [ES] epoch= 16  NSE=0.7701  best=0.7832  counter=10  ✗ STOP
    """

    def __init__(
        self,
        patience   : int   = TRAIN_CFG["early_stop_patience"],
        min_delta  : float = TRAIN_CFG["early_stop_min_delta"],
        save_dir   : str   = "experiments/default",
        verbose    : bool  = True,
    ):
        self.patience    = patience
        self.min_delta   = min_delta
        self.verbose     = verbose
        self.ckpt        = CheckpointManager(save_dir)

        # mutable state
        self.best_nse    : float = -float("inf")
        self.best_epoch  : int   = -1
        self.counter     : int   = 0
        self.should_stop : bool  = False
        self._history    : list  = []   # (epoch, nse) for plotting

    # ── main entry ────────────────────────────────────────────────────

    def step(
        self,
        nse       : float,
        model,
        optimizer,
        scheduler,
        epoch     : int,
        extra     : Optional[dict] = None,
    ) -> bool:
        """
        Call once per epoch with the current san_val median NSE.

        Returns True if this epoch set a new best (checkpoint saved).
        Sets self.should_stop = True when patience is exhausted.
        """
        self._history.append((epoch, nse))
        improved = nse > self.best_nse + self.min_delta

        if improved:
            self.best_nse   = nse
            self.best_epoch = epoch
            self.counter    = 0
            self.ckpt.save(model, optimizer, scheduler, epoch, nse,
                           extra=extra)
            tag = "✓ saved"
        else:
            self.counter += 1
            tag = f"  (patience {self.counter}/{self.patience})"

        if self.verbose:
            print(f"  [ES] epoch={epoch:3d}  NSE={nse:.6f}"
                  f"  best={self.best_nse:.6f}"
                  f"  counter={self.counter:2d}  {tag}")

        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"\n  [ES] ✗ STOP — no improvement for {self.patience}"
                      f" epochs.  Best epoch = {self.best_epoch}"
                      f"  NSE = {self.best_nse:.6f}\n")

        return improved

    # ── restore ───────────────────────────────────────────────────────

    def restore_best(self, model, optimizer=None,
                     scheduler=None, device=None) -> dict:
        """
        Restore weights from the best checkpoint.
        Call this after the training loop exits (normal or early-stopped).
        """
        if not self.ckpt.exists():
            print("  [ES] WARNING: no checkpoint found — using current weights")
            return {}
        return self.ckpt.load(model, optimizer, scheduler, device)

    # ── diagnostics ───────────────────────────────────────────────────

    def summary(self) -> str:
        if not self._history:
            return "  [EarlyStopping] No epochs recorded yet."
        epochs = [e for e, _ in self._history]
        nses   = [n for _, n in self._history]
        width  = 50
        mx, mn = max(nses), min(nses)
        lines  = [
            f"\n  [EarlyStopping] NSE history ({len(epochs)} epochs)",
            f"  Best: epoch={self.best_epoch}  NSE={self.best_nse:.6f}",
            f"  Stopped: {self.should_stop}   Counter: {self.counter}/{self.patience}",
            "  " + "─" * width,
        ]
        # ascii sparkline
        spark = ""
        for n in nses:
            frac = (n - mn) / (mx - mn + 1e-9)
            idx  = min(7, int(frac * 8))
            spark += " ▁▂▃▄▅▆▇█"[idx]
        lines.append(f"  NSE: {spark[:width]}")
        lines.append(f"       {'↑ epoch 0':<{width//2}}epoch {epochs[-1]} ↑")
        return "\n".join(lines)

    def history_csv(self) -> str:
        """Return epoch,nse CSV string for logging."""
        rows = ["epoch,nse"] + [f"{e},{n:.8f}" for e, n in self._history]
        return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────
# 3.  Self-test (no torch required for logic)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import types

    # ── mock torch so we can test without it installed ────────────────
    torch_mock = types.ModuleType("torch")

    class _FakeModel:
        _state = {"w": 1.0}
        def state_dict(self):    return dict(self._state)
        def load_state_dict(self, d): self._state.update(d)

    class _FakeOptim:
        def state_dict(self):    return {}
        def load_state_dict(self, d): pass

    # Patch torch.save / torch.load
    def _fake_save(obj, path, **kw):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def _fake_load(path, map_location=None, weights_only=False):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    torch_mock.save = _fake_save
    torch_mock.load = _fake_load
    import sys
    sys.modules["torch"] = torch_mock

    # ── simulate 20 epochs ────────────────────────────────────────────
    print("\n── EarlyStopping: simulate 20 epochs with plateau ───────────")

    nse_curve = (
        # ramp up then plateau then slightly worsen
        [0.50, 0.60, 0.68, 0.74, 0.79, 0.83, 0.86, 0.87, 0.875, 0.878,
         0.877, 0.876, 0.872, 0.869, 0.865, 0.860, 0.855, 0.850, 0.845, 0.840]
    )

    model  = _FakeModel()
    optim  = _FakeOptim()

    with tempfile.TemporaryDirectory() as td:
        stopper = EarlyStopping(patience=5, min_delta=1e-4,
                                save_dir=td, verbose=True)

        for epoch, nse in enumerate(nse_curve):
            stopper.step(nse, model, optim, scheduler=None, epoch=epoch)
            if stopper.should_stop:
                break

        print(stopper.summary())
        assert stopper.should_stop, "Should have stopped"
        assert stopper.best_epoch == 9, f"Best should be epoch 9, got {stopper.best_epoch}"
        assert abs(stopper.best_nse - 0.878) < 1e-6

        # Restore
        stopper.restore_best(model, optim)
        print(f"\n  Restored model state: {model._state}")

    print("\n── EarlyStopping: monotone improvement (no early stop) ──────")
    with tempfile.TemporaryDirectory() as td:
        stopper2 = EarlyStopping(patience=3, min_delta=1e-4,
                                 save_dir=td, verbose=False)
        for epoch in range(10):
            stopper2.step(0.5 + epoch * 0.01, model, optim, None, epoch)
        assert not stopper2.should_stop, "Should NOT have stopped"
        assert stopper2.best_epoch == 9
        print(f"  Never stopped.  best_epoch={stopper2.best_epoch}"
              f"  best_nse={stopper2.best_nse:.4f}  ✓")

    print("\n✓ early_stopping.py self-test passed\n")
