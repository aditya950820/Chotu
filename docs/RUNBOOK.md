# One-command chain on Jarvis

**Do the cheap-GPU rehearsal first.** Both partners agreed the 8k/tool recipe
on 2026-11-19 (Aditya confirmed); read the data samples before the paid full
run. H100 SXM 80 GB is ₹330/hour, plus storage charges even while paused.
The dry-run budget is an estimate, not a spending limit.

## Fresh instance

Choose a Python 3.13 template with working **CUDA torch** and a persistent
volume with **at least 100 GiB free**. Confirm the mount survives pause/resume;
set the actual mount below. Do not use an ephemeral container disk.

```bash
git clone https://github.com/aditya950820/Chotu.git && cd Chotu
# During review, before merge:
git checkout devin/1795103279-e2e-training-chain
git config user.name "YOUR OWN NAME"     # check_setup verifies human identity
git config user.email "YOUR OWN EMAIL"
cp .env.example .env && nano .env
# Fill HF_TOKEN (read dataset + write private model repo), WANDB_API_KEY,
# JL_API_KEY, JL_MACHINE_ID; optionally WANDB_PROJECT / WANDB_ENTITY.
export CHOTU_STORAGE=/path/to/persistent/chotu
bash scripts/run_all.sh --dry-run       # plan only: no install/network/pause
```

`run_all.sh` installs requirements + Jarvis SDK using the active Python,
removing the CPU torch pin and constraining the **already installed torch
version** so the CUDA build is kept. It then transfers shutdown ownership to
the Python chain. Installation failure attempts a pause too; if installing
the pause dependencies itself fails, **pause manually immediately**.

## Rehearse, then launch

On an L4/A100 first; resume and inspect its reports before using an H100:

```bash
tmux new -s chotu
bash scripts/run_all.sh --rehearsal
# Rehearsal uses $CHOTU_STORAGE/rehearsal and OWNER/chotu-v1-rehearsal.
# Pass --hf-repo OWNER/NAME to choose a private repo; rehearsal adds -rehearsal.
```

On the H100, after the gates/recipe are accepted:

```bash
tmux new -s chotu
bash scripts/run_all.sh
```

Detach with Ctrl-B then D; `tmux attach -t chotu` to watch. Alternatively:
`nohup bash scripts/run_all.sh > "$CHOTU_STORAGE/launcher.log" 2>&1 &`.
Watch `tail -f "$CHOTU_STORAGE/logs/chain.jsonl"`, `status.json`, and the current
stage log under `logs/`. Rehearsal paths have the extra `/rehearsal` component.
W&B is optional telemetry for the chain; real-GPU preflight still verifies
the configured W&B key via `check_setup.py`. Rehearsals use a separate
`WANDB_PROJECT-rehearsal` project so child run IDs cannot mix with production.

## Stages, results and resume

Order: **preflight → shards → chat_data → speed → lr_test → pretrain →
base_evals → chat_tune → final_evals → export_gguf → upload → final_report**.
Preflight checks keys, CUDA/GPU memory, computed disk requirement, the size
test and Jarvis dry-run. Disk budget includes 20.21 GB shards, two optimizer
checkpoints for each of five runs, HF exports, atomic-write scratch, chat/eval
data, build/cache and 20% margin. Shards are revision-pinned, resumable via HF,
and checked against remote sizes/LFS hashes plus `.done` token/document counts.

Speed uses 50 real-model updates and the 20–50 throughput window. LR trials
use 1.5e-3/3e-3, 0.5B tokens each, 100-update warmup and 15% decay; lower final
BPB wins (ties pick 1.5e-3). Main pretraining enables kill rules. Eval/tool
gates are reported, not fatal. Upload refuses public repos and requires a
write-capable token; it is **not** tested for real by the CPU smoke. The final
report is republished after the artifact upload so its stage verdicts are
included remotely too. See `EVALUATION_EXPORT.md` for scoring/export details.

After a crash/pause, resume the instance, activate the same environment,
and rerun **the exact same command**. Stage markers in `stages/*.done.json`
skip completed stages; training resumes its latest durable checkpoint. Do
not delete checkpoints or edit the code/settings mid-run: the chain records
a code/config signature and refuses mismatches. Changing keys is allowed.
Stage durations report the successful invocation, not cumulative billed time
across interrupted attempts. Do not start a second chain while one is running.
`--from-stage NAME` / `--only-stage NAME` are expert manual controls, not a
way to generate missing prerequisite artifacts. They also skip preflight if
you request a later stage, so verify the machine/resources yourself first.

Results: `reports/final_report.md`, LM/tool reports, `runs/pretrain/final/`,
`runs/chat/best/`, `export/chotu-q8_0.gguf`, `export/Modelfile`. All live on the
volume; HF exports are relative symlinks, so keep `hf/` or copy dereferencing.
`logs/stop.log` records the final pause request. Upload is private by default;
`--skip-upload` explicitly disables **all** repo creation/upload calls.

## Stops, alerts and local CPU check

- `0`: completed/selected stages already done. `1`: ordinary error; inspect
  the stage log. Training child exit 1 (or SIGKILL) retries twice by default,
  using checkpoints; `--retries N` changes the extra-attempt limit.
- `20–23`: nonfinite, loss gate, speed gate or unrecovered spike: **no retry**,
  alert and pause. Diagnose before rerunning. Non-training failures never retry.
- `24`: child checkpointed a graceful stop; `130/143`: chain Ctrl-C/SIGTERM.
  The chain terminates its child's process group and pauses in `finally`.
- `25`: **PAUSE FAILED. Pause manually now**, then inspect `logs/stop.log`.
  Checkpoints/volume persist; storage billing continues after a successful pause.
- SIGKILL/power loss cannot run Python/shell cleanup. Use provider monitoring;
  no local program can guarantee pause if its entire VM/process is killed.

The real CPU smoke uses the same stages, tiny model and 8k switch, but skips
live GPU/W&B/Jarvis preflight and disables quality/speed kill thresholds:

```bash
python scripts/run_chain.py --storage ~/chotu-chain-cpu --rehearsal \
  --device cpu --no-stop --skip-upload --shards-subset
```

Defaults: 524,288 pretrain tokens, 2,000 chat rows/source, four speed updates,
two 65,536-token LR trials, eight chat updates, two eval examples/category.
`--rehearsal-tokens 2097152` increases the rehearsal. `--shards-subset` downloads
only train_013 plus all validation shards. **Never use `--no-stop` on a rented
GPU without another shutdown owner.** Fresh GPU/SDK/Ollama/full-size execution
still needs its real rehearsal; CPU success is only a plumbing check.
