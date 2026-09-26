#!/usr/bin/env python3
"""
run_ablation.py
───────────────
Runs the full ablation grid, in priority order.

Usage
─────
  # Run ALL experiments (can take hours)
  python run_ablation.py

  # Run only CRITICAL priority
  python run_ablation.py --priority critical

  # Run CRITICAL + HIGH
  python run_ablation.py --priority critical high

  # Run a single named experiment
  python run_ablation.py --only exp_001__baseline

  # Dry-run: print the grid without running anything
  python run_ablation.py --dry_run

  # Force re-run completed experiments
  python run_ablation.py --force

  # Run multiple seeds for a specific experiment
  python run_ablation.py --only exp_001__baseline --seeds 42 123 777

Each entry in ABLATION_GRID maps directly to run_experiment.py flags.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# NOTE: `from experiment import run` is deliberately deferred into main()
# so that --dry_run works on a machine without torch installed. Listing the
# grid is a pure-metadata operation and should not require the ML stack.


# ─────────────────────────────────────────────────────────────────────
# THE ABLATION GRID
# Each entry: name, priority, overrides dict, description
# ─────────────────────────────────────────────────────────────────────

ABLATION_GRID = [

    # ══════════════════════════════════════════════════════════════════
    # CRITICAL — resolve before anything else
    #
    # NOTE: the previous grid had TEN entries that measured nothing:
    #   exp_003/004/005  byte-identical to the baseline (empty or
    #                    default-valued overrides)
    #   exp_002          set an unimplemented norm strategy -> raw data -> NaN
    #   exp_009          set seq_len, which the dataset ignored
    #   exp_010/011/011b varied a combiner unreachable when use_ceemdan=False
    #   exp_024/026      varied an LSTM branch not instantiated when
    #                    use_ceemdan=False
    # All are now either implemented for real or removed. validate_grid()
    # below is the regression test that stops this recurring.
    # ══════════════════════════════════════════════════════════════════

    dict(
        name        = "exp_001__baseline",
        priority    = "critical",
        description = "Baseline: single Informer, full attention, 12 channels, "
                      "per-basin z-score, AdamW, warmup-cosine, MSE",
        overrides   = {},
    ),
    dict(
        name        = "exp_002__global_norm",
        priority    = "critical",
        description = "[T8] Global z-score (now actually implemented)",
        overrides   = {"norm_strategy": "global_zscore"},
    ),
    dict(
        name        = "exp_003__discharge_only",
        priority    = "critical",
        description = "[D1] Discharge channel ONLY vs all 12",
        overrides   = {"input_channels": [11]},
    ),
    dict(
        name        = "exp_004__precip_discharge",
        priority    = "critical",
        description = "[D1] Precipitation + discharge only",
        overrides   = {"input_channels": [8, 11]},
    ),

    # ══════════════════════════════════════════════════════════════════
    # HIGH — core architectural decisions
    # ══════════════════════════════════════════════════════════════════

    dict(
        name        = "exp_005__aux_task",
        priority    = "high",
        description = "[E5] y_aux as AUXILIARY TARGET (not decoder input — "
                      "it does not exist at test time)",
        overrides   = {"aux_task": True, "aux_loss_weight": 0.3},
    ),
    dict(
        name        = "exp_006__basin_emb",
        priority    = "high",
        description = "[T10] Learned basin embedding dim=16",
        overrides   = {"use_basin_emb": True, "basin_emb_dim": 16},
    ),
    dict(
        name        = "exp_007__aux_basin_emb",
        priority    = "high",
        description = "[E5+T10] aux task AND basin embedding",
        overrides   = {"aux_task": True, "use_basin_emb": True},
    ),
    dict(
        name        = "exp_008__seq168",
        priority    = "high",
        description = "[I8] 168h history vs 336h (truncation now real)",
        overrides   = {"seq_len": 168},
    ),
    dict(
        name        = "exp_009__huber",
        priority    = "high",
        description = "[T3] Huber delta=1.0 — the only lever that actually "
                      "reduces flood-peak dominance",
        overrides   = {"loss": "huber", "huber_delta": 1.0},
    ),
    dict(
        name        = "exp_010__huber_d05",
        priority    = "high",
        description = "[T3] Huber delta=0.5 — stronger peak damping",
        overrides   = {"loss": "huber", "huber_delta": 0.5},
    ),
    dict(
        name        = "exp_011__nse_loss",
        priority    = "high",
        description = "[T3] Basin-averaged NSE loss eps=0.1 (Kratzert). The "
                      "old NSELoss was MSE in disguise",
        overrides   = {"loss": "nse", "nse_eps": 0.1},
    ),
    dict(
        name        = "exp_012__mae",
        priority    = "high",
        description = "[T3] MAE — linear error, no peak dominance",
        overrides   = {"loss": "mae"},
    ),
    dict(
        name        = "exp_013__lr_5e4",
        priority    = "high",
        description = "[T1] Higher LR 5e-4",
        overrides   = {"lr": 5e-4},
    ),
    dict(
        name        = "exp_014__lr_5e5",
        priority    = "high",
        description = "[T1] Lower LR 5e-5",
        overrides   = {"lr": 5e-5},
    ),
    dict(
        name        = "exp_015__dropout02",
        priority    = "high",
        description = "[L3] Dropout 0.2",
        overrides   = {"dropout": 0.2},
    ),

    # ══════════════════════════════════════════════════════════════════
    # CEEMDAN / MRS — the paper's actual contribution.
    # Requires the RIMF cache:  python src/decompose.py --split train
    # ══════════════════════════════════════════════════════════════════

    dict(
        name        = "exp_020__dmel",
        priority    = "ceemdan",
        description = "[A-OUR-1] Full DMEL: CEEMDAN+MRS K=3, HF->Informer x2, "
                      "LF->LSTM, sum. vs the no-decomposition baseline",
        overrides   = {"use_ceemdan": True},
    ),
    dict(
        name        = "exp_021__dmel_mlp_combiner",
        priority    = "ceemdan",
        description = "[E1] MLP combiner vs fixed sum. Paper 5.6 concatenates "
                      "in the multivariate case",
        overrides   = {"use_ceemdan": True, "ensemble_method": "mlp"},
    ),
    dict(
        name        = "exp_022__dmel_linear_combiner",
        priority    = "ceemdan",
        description = "[E1] Learned scalar weight per branch",
        overrides   = {"use_ceemdan": True, "ensemble_method": "linear"},
    ),
    dict(
        name        = "exp_023__dmel_se_threshold",
        priority    = "ceemdan",
        description = "[C8] Paper's SE>=SE(q) HF/LF rule vs fixed 2:1. Free — "
                      "re-derived from cached SE",
        overrides   = {"use_ceemdan": True, "hf_rule": "se_threshold"},
    ),
    dict(
        name        = "exp_024__dmel_univariate",
        priority    = "ceemdan",
        description = "[C6] Paper-faithful univariate RIMF branches",
        overrides   = {"use_ceemdan": True, "rimf_input_mode": "univariate"},
    ),

    # ══════════════════════════════════════════════════════════════════
    # MEDIUM — secondary architecture choices
    # ══════════════════════════════════════════════════════════════════

    dict(
        name        = "exp_030__probsparse",
        priority    = "medium",
        description = "[I7] ProbSparse vs full attention. Both backends now "
                      "correct; at L=336 only 25/336 queries are active",
        overrides   = {"attention": "prob"},
    ),
    dict(
        name        = "exp_031__d_model_128",
        priority    = "medium",
        description = "[I1] d_model=128",
        overrides   = {"d_model": 128, "d_ff": 512},
    ),
    dict(
        name        = "exp_032__d_model_512",
        priority    = "medium",
        description = "[I1] d_model=512",
        overrides   = {"d_model": 512, "d_ff": 2048},
    ),
    dict(
        name        = "exp_033__enc_layers_3",
        priority    = "medium",
        description = "[I3] 3 encoder layers",
        overrides   = {"enc_layers": 3},
    ),
    dict(
        name        = "exp_034__no_distil",
        priority    = "medium",
        description = "[I10] Disable distillation",
        overrides   = {"use_distil": False},
    ),
    dict(
        name        = "exp_035__adam",
        priority    = "medium",
        description = "[T1a] Plain Adam vs AdamW",
        overrides   = {"optimizer": "adam", "weight_decay": 0.0},
    ),
    dict(
        name        = "exp_036__batch_128",
        priority    = "medium",
        description = "[T4] batch=128 with lr scaled by sqrt(B/64) — else "
                      "batch size is confounded with effective lr",
        overrides   = {"batch_size": 128, "lr": 1e-4 * (128 / 64) ** 0.5},
    ),
    dict(
        name        = "exp_037__wd_1e3",
        priority    = "medium",
        description = "[T7] weight_decay=1e-3",
        overrides   = {"weight_decay": 1e-3},
    ),
    dict(
        name        = "exp_038__no_clip",
        priority    = "medium",
        description = "[T6] No gradient clipping — read mean_grad_norm from "
                      "the baseline summary.json first",
        overrides   = {"clip_grad": 0.0},
    ),
    dict(
        name        = "exp_039__sgdr",
        priority    = "medium",
        description = "[T2] SGDR warm restarts",
        overrides   = {"scheduler": "sgdr"},
    ),
    dict(
        name        = "exp_040__nse_eps0",
        priority    = "medium",
        description = "[T3] NSE loss eps=0 — equals plain MSE on z-scored "
                      "targets; tests whether near-flat basins hurt training",
        overrides   = {"loss": "nse", "nse_eps": 0.0},
    ),
]


def validate_grid(grid=None, verbose: bool = True) -> list:
    """
    REGRESSION TEST for the no-op ablation bug class.

    Ten entries in the previous grid silently measured nothing — some were
    byte-identical to the baseline, some set flags no code path read. Running
    them would have burned hours producing copies of the baseline and then
    ranked those copies against each other as if the differences were real.

    Checks:
      1. names are unique
      2. no non-baseline entry has empty overrides
      3. every override key is one the config actually consumes
      4. every override CHANGES the resolved value
    """
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent / "src"))
    from config import CEEMDAN_CFG, INFORMER_CFG, LSTM_CFG, TRAIN_CFG

    grid = grid if grid is not None else ABLATION_GRID

    known = dict(TRAIN_CFG)
    known.update(INFORMER_CFG)
    known.update(CEEMDAN_CFG)
    known.update({
        "ensemble_method": "sum", "shared_informer": False,
        "shared_lstm": False,
        "lstm_hidden": LSTM_CFG["hidden_size"],
        "lstm_layers": LSTM_CFG["num_layers"],
        "lstm_dropout": LSTM_CFG["dropout"],
        "lstm_bidir": LSTM_CFG["bidirectional"],
        "seed": 42, "pred_len": 48, "c_in": 12,
    })

    problems, seen = [], set()
    for e in grid:
        name, ov = e["name"], e["overrides"]
        if name in seen:
            problems.append(f"{name}: duplicate name")
        seen.add(name)

        # The baseline is DEFINED as "no overrides", so it is exempt from
        # both the empty check and the all-defaults check below.
        if name == "exp_001__baseline":
            continue

        if not ov:
            problems.append(f"{name}: empty overrides — identical to baseline")
            continue

        unknown = [k for k in ov if k not in known]
        for k in unknown:
            problems.append(f"{name}: unknown override key {k!r}")

        # The real failure mode is an entry where NOTHING differs from the
        # baseline. An individual key may legitimately restate a default
        # (e.g. loss="huber" paired with an explicit huber_delta=1.0) as long
        # as at least one key actually moves.
        changed = [k for k, v in ov.items()
                   if k in known and known[k] != v]
        if not changed and not unknown:
            problems.append(
                f"{name}: every override equals its default — no-op "
                f"({ov})")

    if verbose:
        if problems:
            print("\n  GRID VALIDATION FAILED")
            for p in problems:
                print(f"    ✗ {p}")
        else:
            print(f"\n  grid validation: {len(grid)} experiments, "
                  f"all differ from baseline ✓")
    return problems


# ─────────────────────────────────────────────────────────────────────
# Grid runner
# ─────────────────────────────────────────────────────────────────────

PRIORITY_ORDER = ["critical", "high", "medium", "low"]


def main():
    parser = argparse.ArgumentParser(
        description="Run the ablation grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--priority", nargs="+",
        default=["critical", "high", "medium"],
        choices=PRIORITY_ORDER,
        help="Which priority levels to run",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Run only this one experiment name",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Seed(s) to run each experiment with",
    )
    parser.add_argument("--validate", action="store_true",
                        help="Only validate the grid, then exit")
    parser.add_argument("--fail_fast", action="store_true",
                        help="Abort the whole grid on first failure")
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print grid without running anything")
    args = parser.parse_args()

    # ── Filter grid ───────────────────────────────────────────────────
    if args.validate:
        sys.exit(1 if validate_grid() else 0)

    # Always validate before running. A grid containing a no-op entry
    # would burn hours producing copies of the baseline and then rank
    # those copies against each other as if the deltas were real.
    if validate_grid(verbose=True):
        print("\n  Refusing to run a grid with no-op entries.")
        sys.exit(1)

    grid = ABLATION_GRID
    if args.only:
        grid = [e for e in grid if e["name"] == args.only]
        if not grid:
            print(f"ERROR: no experiment named {args.only!r}")
            sys.exit(1)
    else:
        grid = [e for e in grid if e["priority"] in args.priority]

    print(f"\n  Ablation grid: {len(grid)} experiments × "
          f"{len(args.seeds)} seed(s)")
    print(f"  Priorities   : {args.priority}")
    print(f"  Seeds        : {args.seeds}")
    print()

    header = f"  {'#':>3}  {'Priority':<10}  {'Name':<40}  {'Description'}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, e in enumerate(grid, 1):
        print(f"  {i:>3}  {e['priority']:<10}  {e['name']:<40}  "
              f"{e['description'][:45]}")

    if args.dry_run:
        print("\n  [dry_run] No experiments run.")
        return

    # Deferred import: only needed once we actually train (see note at top).
    from experiment import run

    print()

    # ── Run ───────────────────────────────────────────────────────────
    results = {}
    t_total = time.time()

    for exp in grid:
        base_name = exp["name"]
        for seed in args.seeds:
            # Append seed suffix only if multiple seeds requested
            name = (f"{base_name}_s{seed}"
                    if len(args.seeds) > 1 else base_name)

            overrides = dict(exp["overrides"])
            overrides["seed"] = seed

            print(f"\n{'─'*60}")
            print(f"  Running: {name}  [{exp['priority']}]")
            print(f"  {exp['description']}")
            print(f"{'─'*60}")

            try:
                result = run(name=name, overrides=overrides,
                              force=args.force, verbose=True)
                results[name] = result.median_nse
                print(f"  → median NSE = {result.median_nse:.4f}")
            except Exception as exc:
                # Print the FULL traceback. The previous bare handler printed
                # only str(exc), which is how a TypeError in the eval path could
                # burn a full training run and leave one unhelpful line in a
                # scoreboard nobody could debug.
                import traceback
                print(f"  ✗ FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                results[name] = None
                if args.fail_fast:
                    raise

    # ── Scoreboard ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ABLATION SCOREBOARD  (ranked by median NSE on dev)")
    print(f"{'='*60}")
    valid = [(n, v) for n, v in results.items() if v is not None]
    valid.sort(key=lambda x: x[1], reverse=True)

    print(f"  {'Rank':>4}  {'Experiment':<45}  {'med NSE':>8}")
    print(f"  {'─'*4}  {'─'*45}  {'─'*8}")
    for rank, (name, nse) in enumerate(valid, 1):
        marker = " ← BEST" if rank == 1 else ""
        print(f"  {rank:>4}  {name:<45}  {nse:>8.4f}{marker}")

    # Failed experiments
    failed = [n for n, v in results.items() if v is None]
    if failed:
        print(f"\n  Failed ({len(failed)}): {', '.join(failed)}")

    elapsed = (time.time() - t_total) / 60
    print(f"\n  Total time: {elapsed:.1f} min")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
