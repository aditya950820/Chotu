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

## Review, 2026-11-22 (pranay): can we reach 30–35% MFU?

**Short answer: 30% on an H100 SXM is realistic once the fixes above are live; 35% is the top of the realistic range, not a safe plan; above ~40% is unlikely for a model this thin.** Only the sweep can say for sure. Nothing here has run on a GPU yet.

### How the estimate is built
- Work per token (the formula `train.py` uses): 1.59 GFLOP at 2k context (6N = 1.09 G, attention 0.50 G), 3.10 GFLOP at 8k.
- About two thirds of the time goes into matrix multiplies. With 640 hidden they are small: the q/k/v projections (640→640/320) are limited by memory speed, not compute, so they run well below peak. MLP (640→1,792) and the LM head are better.
- About a third of the counted work is attention. The SDPA flash kernel on H100 is the older FlashAttention-2 design and reaches a fraction of peak at head_dim 64. cuDNN attention can be faster on H100, so the sweep now tries it.
- The rest is memory-bound small steps per layer (4 RMSNorms including QK-norm, RoPE, SiLU, adds). Under autocast the residual stream stays fp32, which doubles that traffic. `torch.compile` fuses most of it, but a thin model spends a larger share of time here than a wide one.
- Honest caveat on the number itself: the attention term counts the full square, not the causal half, the same convention as nanoGPT/PaLM. Under a weights-only (6N) count, every MFU figure is about 0.68× lower.

| Card | Expected MFU | tok/s at 2k | 10B tokens (2k speed) |
|---|---|---|---|
| H100 SXM (989 TFLOPS) | 25–35% | 155k–218k | 13–18 h |
| H200 (same compute, faster memory) | a bit above H100, because the memory-bound parts speed up | ~170k–240k | 12–16 h |
| A100-80GB (312 TFLOPS) | 40–50% (easier to keep busy) | 78k–98k | 28–35 h |

### Things the plan does not yet account for
1. **The 8k stage is half the speed in tok/s.** Attention is two thirds of the work at 8k. Same MFU, about 51% of the 2k tok/s. Over the final 8% of tokens this adds about 8% to the main run. It is not a bug when tok/s halves after `SWITCH`.
2. **Chosen card: H200 Spot (pranay, 2026-11-22; pending Adi in `DECISIONS.md`).** Its ₹3,600 go/no-go of ≥180k tok/s is too low. It counts only the main 10B. Add the 8k slowdown, the LR test (2 × 0.5B tokens) and the sweep, and 16.6 h (₹3,600 / ₹217) needs about **200k tok/s at 2k (≈32% MFU) before evals, chat tuning and Spot restarts**. Plan on ~210k, or a bigger budget.
3. **Logged tok/s leaves out eval and checkpoint time.** Each checkpoint is ~2.2 GB (weights + AdamW). On a network volume that can be 5–20 s every 100 steps (~4.5 min at 2k speed), so 2–7% of wall time. New `wall_tokens_per_second` and `CKPT seconds=` lines show the real cost. On Spot, keep `--ckpt-every 100`; on an on-demand card, 250 is a reasonable trade.
4. **The A100 fallback would have hit the speed kill rule.** The fixed 74k tok/s floor is an H100 number; an A100 at a healthy 40% does ~78k, and 35% would have been killed at step 50. The floor now scales with the card's peak (H100 SXM still 74k, A100 23k). See the proposal in `DECISIONS.md`.

### Added in this review (speed only, recipe unchanged)
- `--attention {auto,flash,cudnn,efficient}`: picks the SDPA kernel for pretraining. Anything but `auto` turns the slow math kernel off, so a silent fallback turns into a loud error. At startup `fused_kernels_ok=` reports which fast kernels accept our real shapes (causal, 10 query / 5 KV heads, head_dim 64). If it says `math only`, stop: that alone would explain a very low MFU.
- The sweep tries auto/flash/cuDNN at a 65k microbatch, flash at 131k, and max-autotune. It saves each config's log and reports why a config failed (OOM or unsupported kernel), plus peak GPU memory. Takes about 20 minutes.
- `liger-kernel==0.8.5` is pinned. That version handles fp32 weights with bf16 activations under autocast, as checked in its source. `train.py` now refuses to train the full model without it unless `--allow-unfused-ce` is passed, so the main run can't silently fall back to the slow path again.

### What would move MFU further (not done, bigger changes)
- Fusing q/k/v into one matrix multiply (and gate/up into one) would make the small, memory-bound matmuls bigger. That needs a custom attention module instead of stock HF Qwen3. It's a readability cost; consider it only if the sweep lands under 25%.
- Keeping the residual stream in bf16 (bf16 weights with fp32 master copies) cuts memory traffic but changes numerics. That would need a decision entry.
- FP8 and custom kernels are not worth it at this size.
