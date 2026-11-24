"""One-screen progress check that runs on the GPU machine, so it works with the Mac closed
(Jarvis web terminal, or ssh from a phone). Reads only files the chain already writes.

    python scripts/status.py                 # storage from CHOTU_STORAGE in .env
    watch -n 60 python scripts/status.py     # refresh every minute
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from dotenv import load_dotenv

STAGES = ("preflight", "shards", "chat_data", "speed", "lr_test", "pretrain",
          "base_evals", "chat_tune", "final_evals", "export_gguf", "upload", "final_report")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hours(seconds):
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage", default=os.getenv("CHOTU_STORAGE"))
    ap.add_argument("--rupees-per-hour", type=float, default=217)
    a = ap.parse_args()
    if not a.storage:
        raise SystemExit("pass --storage or set CHOTU_STORAGE in .env")
    s = Path(a.storage)
    done = [x for x in STAGES if (s / "stages" / f"{x}.done.json").exists()]
    print(f"Stages done {len(done)}/{len(STAGES)}: " + " ".join(f"[{'x' if x in done else ' '}]{x}" for x in STAGES))

    sweep = s / "reports/speed-sweep.json"
    if sweep.exists() and (best := json.loads(sweep.read_text()).get("best")):
        print(f"Speed sweep: {best['tokens_per_second']:,.0f} tok/s, MFU {best['mfu']:.1%}, micro {best['micro_tokens']}")

    logs = sorted((s / "runs").glob("*/metrics.jsonl"), key=lambda p: p.stat().st_mtime) if (s / "runs").exists() else []
    logs = [p for p in logs if not p.parent.name.startswith("sweep-")]
    if not logs:
        print("No training run yet.")
    else:
        log = logs[-1]
        run, data = log.parent, rows(log)
        plan = json.loads((run / "plan.json").read_text()) if (run / "plan.json").exists() else {}
        last, recent = data[-1], data[-50:]
        age = time.time() - log.stat().st_mtime
        total = plan.get("steps")
        print(f"\nRun '{run.name}': step {last['step']:,}" + (f" / {total:,} ({last['step'] / total:.1%})" if total else "")
              + ("   DONE" if (run / "done.json").exists() else f"   last update {hours(age)} ago"))
        speed = statistics.median(r["tokens_per_second"] for r in recent)
        print(f"  loss {last['loss']:.3f} (first {data[0]['loss']:.2f})   grad norm {last['grad_norm']:.2f}   lr {last['lr']:.2e}")
        print(f"  speed {speed:,.0f} tok/s   MFU {statistics.median(r['mfu'] for r in recent):.1%}   "
              f"data wait {statistics.median(r.get('data_wait_frac', 0) for r in recent):.1%}   context {last['seq_len']}")
        evals = [r for r in data if r.get("val_bpb") is not None or r.get("val_loss") is not None]
        if evals:
            e = evals[-1]
            key = "val_bpb" if e.get("val_bpb") is not None else "val_loss"
            print(f"  {key} {e[key]:.4f} at step {e['step']:,}   best {e.get('best_' + key)}")
        if total and not (run / "done.json").exists():
            left = (total - last["step"]) * plan.get("batch_tokens", 524288) / speed
            print(f"  time left for this run ≈ {hours(left)}  (≈ ₹{left / 3600 * a.rupees_per_hour:,.0f} at ₹{a.rupees_per_hour:.0f}/h)")
        if age > 600 and not (run / "done.json").exists():
            print("  WARNING: no new step for over 10 minutes; check the tmux session / logs")

    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.split(",")
        print(f"\nGPU busy {gpu[0].strip()}%, memory {int(gpu[1]) / 1024:.0f}/{int(gpu[2]) / 1024:.0f} GiB")
    except Exception:
        pass
    rehearsal = s / "setup-logs/rehearsal.log"
    if rehearsal.exists():
        state = "finished" if (s / "rehearsal/stages/final_report.done.json").exists() else "running or failed"
        print(f"CPU rehearsal: {state} (tail {rehearsal})")


if __name__ == "__main__":
    main()
