"""Live Session 2 dashboard on W&B, so both of us can watch the data run from any browser or phone.

Run it next to prepare_data.py (e.g. in a second tmux window). It only reads files, never changes them:

    python scripts/watch_data.py --storage ~/chotu-data --log ~/data.log

Every minute it logs: files done (of 14), tokens written, speed, hours left, CPU load, free disk,
plus the newest log lines. It stops by itself once the final report appears.
Needs WANDB_API_KEY (and WANDB_ENTITY / WANDB_PROJECT) in .env.
"""
import argparse
import glob
import html
import os
import shutil
import socket
import time

TARGET_TOKENS = 10.0e9


def snapshot(storage):
    """What's on disk right now: finished files, tokens written (incl. the file in progress)."""
    shard_dir = os.path.join(storage, "shards")
    done = glob.glob(os.path.join(shard_dir, "*.done"))  # count only: a marker may be mid-write
    bins = glob.glob(os.path.join(shard_dir, "*.bin")) + glob.glob(os.path.join(shard_dir, "*.bin.tmp"))
    tokens = sum(os.path.getsize(b) for b in bins) // 2  # uint16 = 2 bytes per token
    finished = os.path.exists(os.path.join(storage, "reports", "data_report.md"))
    return {"files_done": len(done), "tokens": tokens, "report_written": finished}


def rate_and_eta(history, tokens):
    """Speed over the last ~15 minutes and hours left to ~10B tokens."""
    window = [h for h in history if h[0] >= history[-1][0] - 900]
    dt = window[-1][0] - window[0][0]
    rate = (window[-1][1] - window[0][1]) / dt if dt > 0 else 0.0
    eta_h = (TARGET_TOKENS - tokens) / rate / 3600 if rate > 0 else None
    return rate, eta_h


def tail(path, n=5):
    if not path or not os.path.exists(path):
        return "(no log yet)"
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", default=os.path.expanduser("~/chotu-data"))
    ap.add_argument("--log", default=os.path.expanduser("~/data.log"))
    ap.add_argument("--every", type=int, default=60, help="seconds between updates")
    a = ap.parse_args()

    from dotenv import load_dotenv
    import wandb

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    run = wandb.init(project=os.getenv("WANDB_PROJECT", "chotu"), entity=os.getenv("WANDB_ENTITY") or None,
                     name=f"session2-data-{socket.gethostname()}", job_type="data",
                     config={"storage": a.storage, "cpus": os.cpu_count(), "target_tokens": TARGET_TOKENS})
    print(f"Live dashboard: {run.url}")

    history, t0 = [], time.time()
    while True:
        s = snapshot(a.storage)
        now = time.time()
        history.append((now, s["tokens"]))
        rate, eta_h = rate_and_eta(history, s["tokens"])
        wandb.log({
            "files_done": s["files_done"],
            "tokens_B": s["tokens"] / 1e9,
            "percent_done": 100 * min(1.0, s["tokens"] / TARGET_TOKENS),
            "speed_M_tok_per_s": rate / 1e6,
            "hours_left": eta_h if eta_h is not None else float("nan"),
            "hours_running": (now - t0) / 3600,
            "cpu_load_1min": os.getloadavg()[0],
            "disk_free_GB": shutil.disk_usage(a.storage if os.path.isdir(a.storage) else os.path.dirname(a.storage)).free / 1e9,
            "last_log_lines": wandb.Html(f"<pre>{html.escape(tail(a.log))}</pre>"),
        })
        if s["report_written"]:  # written after the last file, so wait for it rather than counting files
            with open(os.path.join(a.storage, "reports", "data_report.md")) as f:
                wandb.log({"data_report": wandb.Html(f"<pre>{html.escape(f.read())}</pre>")})
            run.summary["finished"] = True
            break
        time.sleep(a.every)
    run.finish()
    print("All files done; dashboard closed.")


if __name__ == "__main__":
    main()
