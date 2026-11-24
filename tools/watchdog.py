"""Watchdog (runs on the Mac): if the H200 gets paused by a Spot interruption, resume it and carry on.

    caffeinate -i python tools/watchdog.py          # keep the Mac awake while it watches
    touch ~/.chotu-watchdog-hold                       # pause the watchdog (e.g. before pausing by hand)

Every --every seconds it asks Jarvis for the instance status. When the instance is Paused it reads
what the chain saved to the private HF backup just before stopping, and decides:

  resume     no deliberate stop was recorded (Spot interruption), or the chain stopped because it got
             a stop signal (SIGTERM/SIGINT); training restarts from its last checkpoint
  never      NaN / kill rule / failed stage / NO-GO / setup failure: resuming would only repeat the
             failure and spend money, so it notifies you and waits for a human
  finished   the whole chain completed: the watchdog exits

Resume = Jarvis resume on the same volume (Spot H200, 150 GB) + the usual scripts/launch.sh, which
skips finished work. At most --max-resumes per 6 hours, so a broken setup can never loop.
Nothing secret is printed; keys come from the Mac .env.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
HOLD = Path.home() / ".chotu-watchdog-hold"
LOG = Path.home() / "chotu-watchdog.log"
STOP_STATES = {"ALERT", "final_backup", "pausing", "finished", "pause_skipped"}


def log(message, notify=False):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as file:
        file.write(line + "\n")
    if notify:
        safe = message.replace('"', "'")[:200]
        subprocess.run(["osascript", "-e", f'display notification "{safe}" with title "Chotu watchdog" sound name "Glass"'],
                       capture_output=True)


def decide(events, setup_exit, finished_marker):
    """events: chain.jsonl rows (oldest first); setup_exit: gpu_up's exit record or None.
    Returns (action, reason) with action in {"resume", "never", "finished"}."""
    if finished_marker:
        return "finished", "final_report stage is done"
    last_chain = events[-1] if events else None
    # Whichever wrote last (gpu_up's setup phase or the chain) explains the pause.
    if setup_exit and (not last_chain or setup_exit["time"] > last_chain["time"]):
        if setup_exit.get("code") == 0:
            return "resume", "setup exited cleanly before the chain took over"
        return "never", f"setup stopped on purpose (exit {setup_exit.get('code')}): {setup_exit.get('reason', '')}"
    if not last_chain or last_chain["state"] not in STOP_STATES:
        return "resume", "no deliberate stop recorded: looks like a Spot interruption"
    alerts = [e for e in events if e["state"] == "ALERT"]
    reason = alerts[-1].get("message", "") if alerts else "chain stopped without an alert"
    if reason.startswith("signal"):
        return "resume", f"chain was stopped by a signal ({reason})"
    return "never", f"chain stopped on purpose: {reason}"


class Watchdog:
    def __init__(self, a):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=True)
        from jarvislabs import Client
        from huggingface_hub import HfApi
        self.a, self.client, self.hf = a, Client(api_key=os.environ["JL_API_KEY"]), HfApi()
        self.repo = os.getenv("CHOTU_BACKUP_REPO") or f"{self.hf.whoami()['name']}/chotu-run-backup"
        self.resumes = []

    def instance(self):
        rows = [i if isinstance(i, dict) else i.__dict__ for i in self.client.instances.list()]
        rows = [r for r in rows if r.get("gpu_type") == self.a.gpu] or rows
        if len(rows) != 1:
            raise RuntimeError(f"expected one {self.a.gpu} instance, found {len(rows)}")
        return rows[0]

    def backup_state(self):
        from huggingface_hub import hf_hub_download
        files = set(self.hf.list_repo_files(self.repo))
        with tempfile.TemporaryDirectory() as tmp:
            def read(name):
                if f"storage/{name}" not in files:
                    return None
                return Path(hf_hub_download(self.repo, f"storage/{name}", local_dir=tmp, force_download=True)).read_text()
            text = read("logs/chain.jsonl") or ""
            events = [json.loads(line) for line in text.splitlines() if line.strip()]
            setup = read("dashboard/setup-exit.json")
            return events, json.loads(setup) if setup else None, "storage/stages/final_report.done.json" in files

    def resume(self, machine_id):
        try:
            self.client.instances.resume(int(machine_id), gpu_type=self.a.gpu, is_spot=not self.a.on_demand,
                                         storage=self.a.storage_gb)
        except Exception as error:  # Jarvis gives the resumed machine a new id, so its lookup can fail
            log(f"resume call returned {type(error).__name__}; checking instance list")
        deadline = time.time() + 900
        while time.time() < deadline:
            inst = self.instance()
            if str(inst.get("status")).lower() == "running" and inst.get("public_ip"):
                return inst
            time.sleep(15)
        raise RuntimeError("instance did not come back within 15 minutes (no Spot capacity?)")

    def relaunch(self, inst):
        host = f"root@{inst['public_ip']}"
        for _ in range(40):  # wait for SSH
            if subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                               "-o", "StrictHostKeyChecking=accept-new", host, "true"], capture_output=True).returncode == 0:
                break
            time.sleep(10)
        result = subprocess.run(["bash", str(ROOT / "scripts/launch.sh"), host, "22", str(inst["machine_id"])],
                                cwd=ROOT, capture_output=True, text=True)
        log("launch.sh: " + " | ".join(line for line in result.stdout.splitlines() if line.startswith("==")))
        if result.returncode:
            raise RuntimeError(f"launch.sh failed: {result.stderr[-300:]}")

    def run(self):
        log(f"watching ({self.a.gpu}, every {self.a.every}s, backup repo {self.repo}); hold file: {HOLD}")
        last_status, waiting_human = None, None
        while True:
            try:
                if HOLD.exists():
                    if last_status != "hold":
                        log("hold file present: not acting until it is removed")
                    last_status = "hold"
                    time.sleep(self.a.every)
                    continue
                inst = self.instance()
                status = str(inst.get("status"))
                if status != last_status:
                    log(f"instance {inst.get('machine_id')} is {status}")
                    last_status = status
                if status.lower() == "running":
                    waiting_human = None
                elif status.lower() == "paused" and waiting_human != inst.get("machine_id"):
                    time.sleep(20)  # let the chain's final backup land first
                    action, reason = decide(*self.backup_state())
                    if action == "finished":
                        log(f"run finished ({reason}). Watchdog exiting.", notify=True)
                        return
                    if action == "never":
                        log(f"NOT resuming: {reason}. Needs a human.", notify=True)
                        waiting_human = inst.get("machine_id")
                    else:
                        cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
                        self.resumes = [t for t in self.resumes if t > cutoff]
                        if len(self.resumes) >= self.a.max_resumes:
                            log(f"{len(self.resumes)} resumes in 6 h: stopping auto-resume. Needs a human.", notify=True)
                            waiting_human = inst.get("machine_id")
                        else:
                            log(f"resuming: {reason}", notify=True)
                            self.resumes.append(datetime.now(timezone.utc))
                            new = self.resume(inst["machine_id"])
                            log(f"resumed as {new['machine_id']} at {new['public_ip']}; relaunching")
                            self.relaunch(new)
                            log("relaunched: training continues from its last checkpoint", notify=True)
            except Exception as error:
                log(f"watchdog error (will retry): {type(error).__name__}: {error}")
                time.sleep(max(self.a.every, 60))
            time.sleep(self.a.every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=20, help="seconds between Jarvis status checks")
    ap.add_argument("--gpu", default="H200")
    ap.add_argument("--storage-gb", type=int, default=150)
    ap.add_argument("--max-resumes", type=int, default=4, help="per 6 hours")
    ap.add_argument("--on-demand", action="store_true", help="resume as on-demand instead of Spot (costs more)")
    Watchdog(ap.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
