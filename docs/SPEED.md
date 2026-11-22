# Training speed (MFU) report, 2026-11-22

## Problem
The H100 test reached about **5% MFU**, under our 10% kill rule, which means a setup bug.

## Causes found in `scripts/train.py`, and fixes
| # | Cause | Fix | Expected effect |
|---|---|---|---|
| 1 | `liger-kernel` was never installed on the GPU (`run_all.sh` skipped it), so the loss fell back to **256 tiny 256-token chunks per microbatch, each with checkpoint recompute** | `run_all.sh` installs `liger-kernel` (this is the frozen "fused cross-entropy"); the fallback now uses 8,192-token chunks and prints a loud WARNING | biggest win, likely 2× or more |
| 2 | 3+ GPU↔CPU syncs per microbatch (`.item()`, `isfinite`) stopped the GPU from queueing work | target counts come from CPU tensors; loss is accumulated on the GPU; **one sync per step** | 10–30% |
| 3 | Data was read from disk inside the timed step | background thread prefetches the next 2 steps into pinned memory; copies are `non_blocking` | small–medium; now logged as `data_wait_frac` |
| 4 | Validation used 256-token chunks with a sync per chunk | 8,192-token chunks, one sync per batch | faster evals |
| 5 | MFU formula ignored attention and assumed an H100 | counts attention (+46% at 2k context) and looks up the card's own peak (`--peak-tflops` to override) | honest number; the old "5%" ≈ 7% real |
| 6 | Compile mode not tunable | `--compile-mode` (`default` / `max-autotune-no-cudagraphs` / `max-autotune`) | 0–15% |

Nothing in the training recipe changed: same batch, LR, data order, loss maths and checkpoints. A change in microbatch size only changes how the 524,288-token batch is split.

## What to aim for
30–40% MFU with the corrected formula. This model is thin (640 hidden), so above about 45% would need custom kernels or FP8, which isn't worth it.

## 15-minute GPU check (do this before the full run)
```
bash scripts/run_all.sh ...   # or just: pip install liger-kernel
python scripts/speed_sweep.py --storage <persistent volume>
```
- This runs 3 configs × 20 steps of the real 181M model (microbatch 65k/131k, two compile modes), prints tok/s, MFU and data-wait, writes `reports/speed-sweep.json`, then **pauses the machine**.
- `run_chain.py` automatically uses the fastest config for the speed, LR and pretraining stages.
- Read it like this: data-wait above 5% means data loading is the bottleneck. A failed 131k config means out of memory, so keep 65k.
- Go / no-go for 10B tokens on ₹3,600 with H200 Spot (₹217/h): you need about **≥ 180k tok/s** (about 15 h).

## Tests
`pytest -q`: 115 passed. New tests cover the MFU formula and prefetch order/error handling. The size gate (180,925,056) passes. 1 failure is local only: `lm_eval` isn't installed on the Mac.
Not verified: real GPU speed. That is the point of the sweep above.
