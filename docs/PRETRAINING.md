# Pretraining (chain step 1)

One readable loop: `scripts/train.py --stage pretrain`. Chat tuning, evaluations,
export and the chain driver are later steps. The 8k tail is still a proposal
pending partner agreement; `--long-frac 0` disables it.

```sh
source ~/chotu-venv/bin/activate
python scripts/train.py --stage pretrain --model chotu --storage /persistent/chotu \
  --name pretrain --lr 0.003 --kill-rules
```

The LR above is an example, not a decision: **run the LR test first**. For each
candidate use a separate name and `--warmup-steps 100 --decay-frac 0.15
--total-tokens 5e8`. Default batch/micro sizes are 524,288 / 65,536 tokens,
warmup 700 updates, final linear decay 20%, long tail 8%. Tiny CPU runs use
fp32, no compile or fused kernels; set `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`
on small hosts. CUDA uses bf16, SDPA, fused AdamW, compile (unless
`--no-compile`) and Liger fused linear CE if installed. The fallback chunks
and recomputes vocabulary projections to limit memory. Install `liger-kernel`
on the GPU host if desired; it is not required for CPU rehearsal.

## Data, budget and reproducibility

`chotu/data.py` memmaps `<storage>/shards/{train,val}_NNN.bin`. Seed shuffles
whole train shards once; windows then move sequentially, never sampling with
replacement. Targets are shifted by one token. Adjacent windows can share
one context token, but no target position is trained twice. Validation uses
the first fixed 8,192 target tokens (whole 2k windows) in sorted val shards,
at both stages. `--val-tokens` controls this budget. BPB sums CE in nats and
divides by ln(2) and actual UTF-8 bytes from decoding each target row, including
EOT text; no average bytes/token approximation. Incomplete final windows are
excluded. This is a fixed-window metric, not a claim of full-document BPB.

The token budget defaults to all available training targets; `--total-tokens`
is an upper bound. Whole optimizer batches only. The short stage consumes its
prefix; only the unread tail supplies long windows. Scan EOTs in bounded-memory
chunks, report complete-document counts/token sums ≥2,048 / ≥8,192, and prefer
starts of documents ≥4,096 when one is within 4,096 tokens of the next available
document start. Every long window begins at a document start; it may continue
into following documents. Skip gaps without overlapping targets. Boundary scraps and batch
rounding mean some tokens are unused: `TAIL` explicitly reports planned vs
actual steps, unused tokens and preferred starts. WSD uses actual steps;
switch position stays fixed at the planned short-prefix boundary. The realized
long fraction can therefore be slightly lower than requested. Fail if no full
long batch fits rather than silently omit the long stage.

At the switch rebuild Qwen3's rotary embedding from `long_context_config()`,
with theta 500,000; optimizer weights/state are retained. Short windows use
`PRETRAINING_WINDOW` (2,048), not the config's declared 8k capacity.

AdamW betas (0.9, 0.95), weight decay 0.1 on matrices except tied embeddings;
no decay on vectors/norms or embeddings. Clip gradient norm at 1.0. The
schedule warms linearly to peak LR, holds, then decays linearly to zero.
Use a warmup shorter than the run for meaningful tiny rehearsals.

## Checkpoints, telemetry and stopping

All training output lives under storage, default `<storage>/runs/<name>`;
`--out` must be inside storage. Atomic checkpoint renames every `--ckpt-every`
updates and at completion; keep the last two. Checkpoints contain model,
optimizer, step, config, short/long stage, CPU/CUDA/Python/NumPy RNG, rule
history and best validation BPB. The latest checkpoint resumes automatically;
command/seed and shard manifests must match. CPU deterministic resume is tested
bit-for-bit; CUDA bitwise reproducibility is not claimed. A `done.json` marker
makes completed reruns no-ops. Metrics beyond the last durable checkpoint are
removed on resume, then recomputed. SIGKILL/power failure loses only work since
that checkpoint; SIGTERM/SIGINT finish the current update, checkpoint and exit.

`metrics.jsonl` and stdout contain step, loss, LR, gradient norm, tokens/s,
sequence length, theta, validation and best BPB. MFU is the approximate
`6 * parameters * tokens/s / 989e12`, even on CPU (not CPU utilization);
attention overhead is omitted. W&B is optional via `.env` `WANDB_API_KEY`,
`WANDB_PROJECT`, `WANDB_ENTITY`; telemetry failures do not block training.

Kill rules are opt-in (`--kill-rules`), including for the real model. Loss
below 6 must have occurred by update 300. Throughput uses the mean of updates
20–50, past initial compilation. Spike threshold is 1.5× the preceding-200
minimum; if it does not recover within 200 updates, alert **and stop** rather
than continue spending. Rules/history resume with the checkpoint. Failed
updates are not checkpointed; rerunning a killed run requires investigating
the cause, not blindly looping. Nonfinite loss/gradients always stop, even
with rules disabled.

Exit codes:
- 0: complete / already complete.
- 1: ordinary crash, invalid run configuration, or nonfinite values with rules off.
- 2: CLI argument error.
- 20: nonfinite loss/gradient (rules enabled).
- 21: loss never below 6 by update 300.
- 22: throughput below `--min-tok-s` (default 74,000).
- 23: unrecovered loss spike (also an alert).
- 24: graceful signal stop with checkpoint.
- 25: Jarvis pause failed; overrides the training code, which is printed. **Pause manually.**

CUDA jobs always invoke `scripts/stop_machine.py` in `finally`, including after
failure. Configure `.env` `JL_API_KEY`, `JL_MACHINE_ID`, install `jarvislabs`, and
verify `stop_machine.py --dry-run` **before renting time**. CPU runs never pause
a machine. SIGKILL cannot run cleanup: the later chain driver must own an
independent shutdown guard. The future orchestrator must explicitly transfer
shutdown ownership before composing GPU stages; do not use this standalone
entry point unchanged as a mid-chain subprocess.

## Checks

`python -m pytest -q`: synthetic shards test deterministic/no-repeat batches,
long-tail starts, WSD, exact UTF-8 BPB, stop rules, and exact CPU model/loss
resume before and after the 8k switch. No synthetic data is presented as a
model-quality result. Real FineWeb-Edu CPU smoke results go in `PROGRESS.md`.

The private prepared-shard smoke uses only train_013 and the validation shards:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/train.py --stage pretrain \
  --model tiny --storage /home/ubuntu/chotu-smoke --name private-resume-60 \
  --total-tokens 491520 --lr .003 --batch-tokens 8192 --micro-tokens 8192 \
  --long-frac .15 --warmup-steps 5 --decay-frac .2 --ckpt-every 5 \
  --eval-every 20 --val-tokens 2048 --device cpu
```

For crash rehearsal, SIGKILL the training PID after update 27 and rerun that
same command; it resumes update 25's checkpoint. The completed run is a no-op
on rerun: choose a fresh name to repeat the experiment. On 2026-11-19 it finished
58 updates (doc-start packing scraps), switched after update 51, and lowered
loss from 10.84 to 7.40. It is a plumbing check, not a useful trained checkpoint.
