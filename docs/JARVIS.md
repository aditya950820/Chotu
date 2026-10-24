# Jarvis Labs (proposed GPU provider)

Prices checked 2026-10-24 ([pricing](https://jarvislabs.ai/in/gpu-cloud-pricing)), per GPU per hour:
H100 SXM 80 GB ₹330, A100 80 GB ₹141, A100 40 GB ₹84, L4 ₹41. Storage ₹0.013/GB/hour (about ₹940/month for 100 GB), billed **even while paused**.

## Rules
- **Pause, don't leave running.** A paused instance keeps files and packages and bills storage only. An idle H100 costs about ₹7,900 a day.
- **Use the cheapest GPU that does the job.** Data prep is CPU work, and the rehearsals need a small GPU: use an L4 or A100 40 GB for Sessions 2–4, and the H100 only for Session 5.
- **Every GPU job ends with `python scripts/stop_machine.py`**, which pauses the instance through the [Jarvis SDK](https://docs.jarvislabs.ai/sdk). Test it with `--dry-run` first, then for real in the Session 4 dry run.
- Record every running or paused instance under "Paid resources running" in `docs/PROGRESS.md`.

## First-time setup on the instance (JupyterLab terminal)

```bash
git clone https://github.com/aditya950820/Chotu.git && cd Chotu
git config user.name "<your GitHub username>" && git config user.email "<your email>"
cp .env.example .env && nano .env      # HF_TOKEN, WANDB_API_KEY, JL_API_KEY, JL_MACHINE_ID
grep -vE '^torch==' requirements.txt > /tmp/req.txt && pip install -q -r /tmp/req.txt jarvislabs   # keep the template's CUDA torch
python scripts/check_setup.py          # expect 4x [OK]
python scripts/stop_machine.py --dry-run
nproc && df -h ~ && nvidia-smi --query-gpu=name,memory.total --format=csv
```

## Session 2 on Jarvis (instead of the Mac)
Data made here is already where training happens, so there's no 20 GB upload later.

```bash
mkdir -p ~/chotu-data
python scripts/prepare_data.py --storage ~/chotu-data --max-docs 2000 --max-files 2   # 1-min test
nohup python scripts/prepare_data.py --storage ~/chotu-data > ~/chotu-data/run.log 2>&1 &
tail -f ~/chotu-data/run.log            # Ctrl-C stops watching, not the run
```

To verify before relying on it:
- Which folder survives a pause (check that `~/chotu-data` is still there after pause and resume).
- Whether resuming lets you switch GPU type (L4 → H100). If not, put data on a Jarvis filesystem that attaches to any instance.
