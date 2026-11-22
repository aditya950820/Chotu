# Launch checklist (A100 test → H200 run)

Goal: no paid minute spent waiting on something we could have done for free.

## Before renting anything (free)
- [ ] Chat data prepared on the Mac: `python scripts/prepare_chat.py --storage ~/chotu-chat`
      (creates `~/chotu-chat/chat/` + `reports/chat_report.md`; copy only those, not `chat-cache/`).
- [ ] Jarvis dashboard: no old instance billing; check whether a paused instance can resume on another GPU type.
- [ ] Keys ready for the machine's `.env`: `HF_TOKEN` (read dataset + write private model), `WANDB_API_KEY`,
      `WANDB_ENTITY`, `WANDB_PROJECT`, `JL_API_KEY`, `JL_MACHINE_ID` (of the new machine).
- [ ] Decide with partner: keep or skip the LR test (budget: skipping saves ~₹330 on H200 Spot).

## A100-40GB test (~1–1.5 h, ~₹100–130)
Template: Jarvis PyTorch, newest version, Python ≥ 3.11, persistent volume ≥ 100 GB.
1. `python --version; python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
2. Clone, own git name/email, `.env`, `export CHOTU_STORAGE=<volume>/chotu`.
3. In parallel: `run_all.sh --only-stage preflight --no-stop`, then shards download; `scp` chat data from the Mac.
4. `bash scripts/run_all.sh --rehearsal --skip-upload --no-stop`: whole chain on the tiny model.
5. `python scripts/speed_sweep.py --storage $CHOTU_STORAGE --no-stop`: real 181M speed / MFU / data wait.
6. Kill-and-resume test, then `python scripts/stop_machine.py` and confirm in the dashboard it really paused.

**Gotcha:** the real (non-rehearsal) chain writes `chain-config.json` with hashes of the code. If we fix code after
the A100 test, delete that file before the H200 launch. It's safe only because data stages don't depend on the code.

## H200 Spot run
1. Resume the same volume if Jarvis allows it; otherwise repeat A100 steps 1–3.
2. Speed check (sweep, ~10 min). Go only if ≥ ~180k tok/s.
3. `tmux new -s chotu; bash scripts/run_all.sh` (pauses itself at the end or on failure).
4. Watch steps 50 and 300 against the kill rules; after a Spot interruption, resume and rerun the same command.
