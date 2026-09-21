# WORKFLOW — running experiments across two machines

Two machines, one repo, no duplicated work.

| | Machine A | Machine B |
|---|---|---|
| Hardware | Apple M4, 14 cores, 26 GB unified | Ryzen 9 + NVIDIA, 8 GB VRAM |
| Device | `mps` | `cuda` |
| Best at | CEEMDAN precompute (CPU, 12 procs) | Training (CUDA is ~3-5× MPS here) |
| Measured | ~11 min/epoch, ~5 win/s CEEMDAN | expect ~3-4 min/epoch |

---

## Answering the sharing question directly

**Yes — commit results, pull them on the other machine. That is sufficient.**
It works because of what an experiment actually produces:

| Artefact | Size | In git? | Why |
|---|---|---|---|
| `metrics_dev.json` | ~200 KB | **yes** | the result; per-basin NSE needed for Wilcoxon |
| `metrics_sanval.json` | ~200 KB | **yes** | selection-split metrics |
| `summary.json` | ~2 KB | **yes** | the audit trail |
| `history.csv` | ~1 KB | **yes** | loss/NSE curve per epoch |
| `config.json` | ~1 KB | **yes** | exact reproducibility |
| `best_model.pt` | **36 MB** | no | only needed to *predict*, not to compare |
| `log.txt` | ~50 KB | no | `history.csv` has the numbers |
| `norm.npz` | 50 KB | no | regenerated from split + seed |
| `data/*.h5` | **5.1 GB** | no | both machines fetch independently |
| `data/rimf_*.h5` | ~600 MB | no | rebuilt by `./run_ceemdan.sh` |

A full result set is **~1.3 MB**, so `git pull` is instant and
`./results.py` merges both machines' runs automatically — it just reads every
folder under `experiments/`.

**When you DO need the weights** (final test-set submission only), copy that one
file directly rather than committing it:

```bash
scp machineB:~/paper2/experiments/exp_009__huber/best_model.pt \
    experiments/exp_009__huber/
```

### Why not commit checkpoints

31 experiments × 36 MB = 1.1 GB of binaries that git cannot delta-compress, in a
repo whose text content is ~1 MB. If you later need every checkpoint, use
`git lfs track "*.pt"` — but you almost certainly won't: comparison needs
metrics, and re-running a config from `config.json` is deterministic.

---

## One-time setup on Machine B (Linux + NVIDIA)

```bash
# 1. Repo
git clone git@github.com:jenevadot/dmel-p2.git paper2 && cd paper2

# 2. Data — NOT in git. Copy from machine A or re-download.
#    5.1 GB total; scp over LAN is usually fastest.
mkdir -p data
scp machineA:~/paper2/data/train.h5 data/
scp machineA:~/paper2/data/test.h5  data/
scp machineA:~/paper2/data/metadata.json data/

# 3. Environment. Same uv flow, CUDA wheel instead of CPU.
#    Install torch from the CUDA index FIRST so the lockfile's CPU/MPS
#    torch==2.14.0 does not get pulled in, then pin everything else.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --index-url \
    https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python matplotlib \
    -r <(grep -v '^torch==' requirements-lock-machineA.txt)

# 3b. Confirm the majors match Machine A — differing numpy/pandas majors
#     make results incomparable.
.venv/bin/python -c "import numpy,pandas,scipy,h5py; \
print('numpy',numpy.__version__,'pandas',pandas.__version__, \
'scipy',scipy.__version__,'h5py',h5py.__version__)"
# Machine A: numpy 2.4.6  pandas 3.0.6  scipy 1.17.1  h5py 3.16.0

# 4. Verify — must print cuda True
.venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"

# 5. Self-tests. All six must pass before training anything.
for m in config evaluate early_stopping models/informer \
         models/lstm_branch models/dmel; do
  .venv/bin/python src/$m.py > /dev/null && echo "$m PASS" || echo "$m FAIL"
done

# 6. Regenerate the split (deterministic — must match machine A)
.venv/bin/python src/splits.py

# 7. Baselines (fast, and they are the bar every model is judged against)
.venv/bin/python src/baselines.py
```

### Two things to check on Machine B

**`num_workers`.** Machine A is forced to 0 because its sandbox blocks torch's
shared-memory manager. Linux has no such restriction, so raise it:

```python
# src/config.py
num_workers = 4       # Linux only; keep 0 on machine A
pin_memory  = True    # real win on discrete VRAM over PCIe
```

**Batch size.** 8 GB VRAM fits batch 64 comfortably. If you go to 128, scale the
lr — `exp_036__batch_128` already does `lr * sqrt(128/64)`. Changing batch size
without scaling lr confounds the two.

---

## Dividing the work

The split follows the hardware, not preference.

> **Status 2026-09-21: CEEMDAN on Machine A failed and is 0.094% done.**
> The 12-process run was launched while training held the machine. Workers died
> with `BrokenPipeError` on `send_bytes` plus 6 leaked semaphores — the pool
> could not return results under memory pressure (26 GB unified, ~5.9 GB already
> swapped). `ceemdan_train.log` opens with `272,142 / 272,142 rows remaining`
> and never recovers.
>
> Resume worked as designed: **256 of 272,142 windows** are complete and flagged
> `done`. Nothing is corrupt — but ~17 h of work remains, it is not 17 h spent.
>
> Do not co-schedule CEEMDAN with training again. Either stop the queue first,
> or run it on Machine B (see revised division below).

**Machine A (M4) — training, which it is already mid-queue on:**
```bash
./run_queue.sh exp_005__aux_task exp_006__basin_emb exp_009__huber
```

**Machine B (Ryzen 9) — CEEMDAN, which is a pure CPU job it wins at:**
```bash
./run_ceemdan.sh 12            # ~17 h, resumable, needs the box to itself
```
Then `scp` the ~600 MB `data/rimf_*.h5` back to A rather than paying the 17 h
twice. If you must run it on A, stop the queue first and start at `8` procs —
12 is what broke it.

Claim experiments explicitly so neither machine repeats work. `run()` already
skips any experiment whose `metrics_dev.json` exists, so a `git pull` before
launching is itself the lock:

```bash
git pull                        # see what the other machine finished
./results.py --list             # done / running / stale
./run_queue.sh <only unclaimed names>
```

### Suggested division

| Machine | Experiments | Why |
|---|---|---|
| A (M4) | `exp_001__baseline` (running, ep 9/40), then `exp_005__aux_task`, `exp_006__basin_emb`, `exp_009__huber` | already mid-queue; do not restart what is in flight |
| B (CUDA) | `exp_011__nse_loss`, `exp_003__discharge_only`, `exp_002__global_norm`, `exp_013__lr_5e4` | fastest at training once its env is built |
| B (CPU) | `./run_ceemdan.sh` first — it blocks every DMEL ablation | Ryzen 9 is faster, and A cannot run it alongside training |

Once the cache exists on B, `scp` `data/rimf_*.h5` to A (~600 MB) so the DMEL
ablations (`exp_020__dmel` …) can run on either machine.

---

## The share loop

```bash
# on whichever machine just finished something
git add experiments/
git commit -m "results: exp_009__huber (dev medNSE 0.6xxx, 40 ep, cuda)"
git push

# on the other machine
git pull
./results.py                 # scoreboard now spans BOTH machines
./results.py --horizon       # per-lead-time, which is what actually matters
./results.py --gap           # split discipline + gradient health
```

Put the headline number in the commit message. It makes `git log --oneline` a
usable experiment history on its own.

### Keep results comparable across machines

Two runs are only comparable if these match:

- `seed` (default 42)
- `data/split_basin_seed42_frac0.12.npz` — deterministic from the seed, and it
  **is** committed, so both machines load the identical partition
- `epochs` — a run cut short is not comparable to a converged one (this bit us
  once; see `IMPROVEMENTS.md` §3b.4)
- normalizer strategy

`summary.json` records `device`, so a CUDA/MPS difference is always visible.
Expect small numeric differences between backends from non-deterministic
reductions — that is why `epochs` and `seed` matching matters more than the
device being identical.

---

## Reading the results

```bash
./results.py --horizon
```

Always read the per-horizon view before concluding anything. Persistence scores
**0.9382** at h+1-12 and **0.2495** at h+37-48, so an aggregate median NSE is
dominated by the easy early hours — a model that merely learned persistence will
still look respectable. The value is in h+25-48.

```bash
./results.py --gap
```

Watch two fields:
- `gap` = dev − san_val. Strongly negative means the checkpoint was chosen on
  selection-split noise and did not transfer.
- `gnorm` vs `clip`. The baseline measured `mean_grad_norm ≈ 2.1-2.8` against
  `clip_grad = 1.0`, so the clipper fires on *every* step and the effective lr
  is below what is configured. `exp_038__no_clip` is the controlled test.

---

## Caveats that travel with every number

**dev is contaminated and cannot be fixed.** 78.8% of dev windows overlap a
training window (measured, 6-gram stride-1). The leak is in the competition's
own `split=1`. Absolute dev NSE is optimistic; it stays valid for *ranking*
because every model pays it equally.

**san_val answers a different question than dev.** It is a held-out *basin*
split (unseen basins); dev is unseen time in known basins. A within-basin split
was tried first and leaked 97% of its windows.

Full record of every fix, measurement, and correction: **`IMPROVEMENTS.md`**.
