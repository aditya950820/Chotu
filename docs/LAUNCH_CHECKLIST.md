# Launch checklist: H200, one command

Goal: the GPU is never idle waiting on something that could have been done earlier or in parallel.

## Before renting (free)
- [x] Chat data prepared on the Mac: `~/chotu-chat/` (done 2026-11-22, 1.2 GB).
- [ ] Mac `.env` has `HF_TOKEN` (write access to your own repos) **and `JL_API_KEY`**. W&B is optional; we don't use it (2026-11-24).
- [ ] Jarvis dashboard: no old instance billing.
- [ ] Both agree: keep or skip the LR test (the chain runs it unless we change the frozen decision).

## Rent
Jarvis **H200**, Spot, **PyTorch template (newest)**, persistent storage of at least 100 GB. Copy the SSH command and the machine ID.

## Launch (on the Mac)
```
bash scripts/launch.sh <user@host> <ssh-port> <jarvis-machine-id>
```
It copies the code and `.env` (secrets never printed), starts `scripts/gpu_up.sh` in tmux, and copies the chat data in parallel.

What `gpu_up.sh` does on the H200:
| Time | GPU | In parallel |
|---|---|---|
| 0–3 min | checks (Python ≥ 3.11, CUDA, ≥ 100 GiB free, Jarvis key) | packages, first shards, chat copy |
| ~3–15 min | **speed sweep** on the real model (65k → 131k → 262k microbatch, then autotuned compile) | rest of shards, llama.cpp prebuild |
| ~15 min | **go/no-go**: ≥ 180k tok/s (`MIN_TOK_S`), otherwise **pause and stop** | |
| then | full chain: preflight → … → pretrain → chat → evals → export → upload → **pause** | CPU rehearsal of the later stages on spare cores (early warning) |

## Watch / recover
- Progress, Mac closed or not: open the Jarvis web terminal (or ssh from any device) and run `cd /home/Chotu && python scripts/status.py`. It shows stages done, step/total, loss, tok/s, MFU, validation BPB, time and ₹ left, and GPU use. Use `watch -n 60 python scripts/status.py` to refresh.
- Full log: `ssh -p <port> <user@host> -t tmux attach -t chotu`; logs are in `<storage>/setup-logs/` and `<storage>/logs/`.
- No W&B means no phone alerts: on NaN, a kill rule or a failure the chain stops and **pauses the machine**, so a paused instance in the Jarvis dashboard is the alert.
- Early warning: if `setup-logs/rehearsal.log` fails, fix the code **before pretraining ends**. Later stages read the current files.
- Spot interruption: resume the instance, then rerun the same `launch.sh` command. Done work is skipped, and training resumes from its last checkpoint (≤ 100 steps ≈ 4–5 min lost).
- If code changes after the first full-chain launch, the chain refuses to resume until `<storage>/chain-config.json` is deleted. Delete it only if the change doesn't affect training maths.
