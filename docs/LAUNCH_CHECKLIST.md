# Launch checklist: H200, one command

Goal: the GPU is never idle waiting on something that could have been done earlier or in parallel.

## Before renting (free)
- [x] Chat data prepared on the Mac: `~/chotu-chat/` (done 2026-11-22, 1.2 GB).
- [ ] Mac `.env` has `HF_TOKEN` (write access to your own repos) **and `JL_API_KEY`**. W&B is optional; we don't use it (2026-11-24).
- [ ] Jarvis dashboard: no old instance billing.
- [x] LR test: **kept** (frozen decision unchanged, 2026-11-24).
- [ ] Jarvis balance: full chain with LR test ≈ ₹3,750 GPU + ≈ ₹35 storage. **Top up to at least ₹4,200** so the run can't stop on a zero balance.

## Rent
Jarvis **H200**, Spot, **PyTorch template (newest)**, **150 GB storage** (the chain needs ≈ 87 GB in total: shards 20, 10 checkpoints 22, chat 6, exports 4, eval data 4, llama.cpp 4, caches 9, +20% margin; 100 GB is the minimum). Wait until it is fully up, then send the SSH command and the machine ID.

## Launch (on the Mac)
```
bash scripts/launch.sh <user@host> <ssh-port> <jarvis-machine-id>
```
It copies the code and `.env` (secrets never printed), starts `scripts/gpu_up.sh` in tmux, copies the chat data in parallel, and prints the **dashboard link**.

What `gpu_up.sh` does on the H200:
| Time | GPU | In parallel |
|---|---|---|
| second 1 | **live dashboard** on port 6006 (records GPU/CPU/disk every 30 s) | |
| 0–3 min | checks (Python ≥ 3.11, CUDA, disk, Jarvis key, private HF backup repo), restore from backup if this is a new instance | packages, first shards, chat copy, backup loop |
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

## Dashboard and backups (work with the Mac off)
- **Dashboard:** Jarvis dashboard → your instance → **API** button → copy the URL, add `/<key>/` (launch.sh prints it; the key is in `<storage>/dashboard/key`). Refreshes every 30 s: stage timeline, step/total, time and ₹ left, billed ₹ so far, loss, validation BPB, tok/s, MFU, LR, gradient norm, LR-trial comparison, chat loss, GPU busy/memory, disk, speed-sweep table, alerts, backup status, rehearsal status, log tails. Without the key the port answers 404.
- **Everything is logged** on the volume: `gpu_up.log`, `setup-logs/`, `logs/<stage>.log`, `logs/chain.jsonl` (every event), `logs/system.jsonl` (GPU/CPU/RAM/disk every 30 s), `runs/*/metrics.jsonl` (every step), `reports/`, `stages/`.
- **Everything is backed up** to the private HF repo `<you>/chotu-run-backup`: logs/reports/metrics every 10 min, newest checkpoint of the running stage every 60 min, a full copy before every pause. The repo must be private, otherwise setup stops.
- **If Spot deletes the instance:** create a new H200 and run the same `launch.sh`. It restores the backup, re-downloads the shards, and training resumes from the last backed-up checkpoint (at most about 1 h lost).
