#!/usr/bin/env python3
"""
run_test_readout.py
───────────────────
The single test.h5 read-out, for the ablation checkpoints and the final model.

The rule this script enforces (NEXT_ITERATION.md §5): test is scored only
AFTER the configuration is frozen, and nothing scored here feeds back into a
choice. The final model is FIXED before any test number exists:

    FINAL = equal-weight mean of all 9 members
            final_{lstm_huber05, resid_huber05, mae}_s{42, 123, 177}

The ensemble is scored only when all 9 members are present. There is no
"best subset" option, deliberately.

Every result lands in experiments/test/<name>.json, and a dev twin (same
scorer, same windows) in experiments/test/<name>__dev.json, so make_report.py
compares dev and test like for like.

  python run_test_readout.py --ablation     # every local ablation checkpoint
  python run_test_readout.py --final        # the 9-member final ensemble
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
OUT = EXP / "test"
PY = sys.executable

# Ablation read-out: one checkpoint per question the report answers. Only
# runs whose best_model.pt is on this machine are scored; the rest are
# skipped with a note (machine-B seeds need their checkpoints synced).
ABLATION = [
    "exp_001__baseline", "exp_001__baseline_s123", "exp_001__baseline_s177",
    "exp_009__huber",
    "exp_010__huber_d05", "exp_010__huber_d05_s123", "exp_010__huber_d05_s177",
    "exp_012__mae", "exp_012__mae_s123", "exp_012__mae_s177",
    "exp_050__resid_huber05", "exp_051__resid_mse",
    "exp_052__rimf_append_huber05", "exp_053__lstm_huber05",
    "exp_020__dmel", "exp_023__dmel_se_threshold",
    "exp_002__global_norm", "exp_005__aux_task", "exp_008__seq168",
    "exp_011__nse_loss", "exp_015__dropout02", "exp_038__no_clip",
    "exp_034__no_distil",
]

FINAL = [f"final_{fam}_s{s}"
         for fam in ("lstm_huber05", "resid_huber05", "mae")
         for s in (42, 123, 177)]


def score(members, name, split):
    out = OUT / (f"{name}.json" if split == "test" else f"{name}__dev.json")
    if out.exists():
        print(f"  [skip] {out.name} exists")
        return True
    cmd = [PY, str(ROOT / "src" / "ensemble_eval.py"), "--exps", *members,
           "--split", split, "--out", str(out)]
    if split == "test":
        cmd.append("--confirm_test")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [FAIL] {name} on {split}:\n{r.stdout[-800:]}{r.stderr[-800:]}")
        return False
    line = next((l for l in r.stdout.splitlines() if "PRIMARY" in l), "")
    print(f"  {name:<34} {split:<5} {line.split()[4] if line else '?'}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.ablation:
        print("\n== ablation checkpoints ==")
        for n in ABLATION:
            if not (EXP / n / "best_model.pt").exists():
                print(f"  [missing] {n}: no local best_model.pt")
                continue
            score([n], n, "dev")
            score([n], n, "test")

    if args.final:
        missing = [m for m in FINAL
                   if not (EXP / m / "best_model.pt").exists()]
        if missing:
            print(f"\nFinal ensemble NOT scored — {len(missing)}/9 members "
                  f"missing: {missing}")
            return 1
        print("\n== FINAL: 9-member ensemble ==")
        score(FINAL, "final_ens9", "dev")
        score(FINAL, "final_ens9", "test")
        for fam in ("lstm_huber05", "resid_huber05", "mae"):
            mem = [m for m in FINAL if f"_{fam}_" in m]
            score(mem, f"final_{fam}_ens3", "dev")
            score(mem, f"final_{fam}_ens3", "test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
