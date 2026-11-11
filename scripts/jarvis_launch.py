"""Rent a Jarvis Labs GPU, run a twin-day job on it with no SSH, and make sure it stops.

    python scripts/jarvis_launch.py --only "speed main5x"            # create the machine and start the job
    python scripts/jarvis_launch.py --only "speed main5x" --dry-run  # show the plan, rent nothing

How it works:
1. A startup script is stored in your Jarvis account. It clones this repo, writes .env on the machine
   (keys from your local .env), builds the Python env (scripts/setup_gpu.sh), finds its own machine id,
   then runs `ONLY=... bash scripts/twin_day.sh`, which pauses the machine at the end, even after a failure.
2. A new GPU machine is created with that script.
3. This script then watches the `live-runs` branch on GitHub (the machine pushes there every 30 s) and is the
   safety net: it pauses the machine if nothing arrives within --boot-minutes, if the job ends but the machine
   is still running, or at --max-minutes no matter what. The stored startup script (it holds keys) is removed.

Needs in .env: JL_API_KEY, HF_TOKEN, GH_TOKEN (Contents read/write on this repo), GPU_PRICE.
"""
import argparse, json, os, pathlib, subprocess, sys, time

from dotenv import dotenv_values
from jarvislabs import Client

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "aditya950820/Chotu"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--only", default="speed main5x", help="twin_day.sh steps to run")
ap.add_argument("--branch", default="claude/chotu-project-setup-t9dncn", help="branch the machine checks out")
ap.add_argument("--gpu", default="H100")
ap.add_argument("--template", default="vm", help="'vm' = bare VM (H100s are offered as VMs)")
ap.add_argument("--storage", type=int, default=100, help="disk in GB")
ap.add_argument("--name", default="chotu-5x")
ap.add_argument("--boot-minutes", type=float, default=25, help="pause if the job has not reported by then")
ap.add_argument("--beat-minutes", type=float, default=8, help="pause if the machine has sent no boot heartbeat by then")
ap.add_argument("--max-minutes", type=float, default=75, help="pause no matter what after this long")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

env = dotenv_values(ROOT / ".env")
missing = [k for k in ("JL_API_KEY", "HF_TOKEN", "GH_TOKEN") if not env.get(k)]
if missing:
    sys.exit(f"missing in .env: {', '.join(missing)}")
client = Client(api_key=env["JL_API_KEY"])


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------- 1. the startup script (runs as the machine boots) ----------
BOOT = f"""#!/bin/bash
# Chotu: startup script written by scripts/jarvis_launch.py. Holds keys: removed once the machine has booted.
export HOME=${{HOME:-/root}}
cd "$HOME"
exec >> "$HOME/chotu_boot.log" 2>&1   # no `set -x`: it would print the keys into the log
GH="https://aditya950820:{env['GH_TOKEN']}@github.com/{REPO}.git"
beat() {{  # sign of life: step name + log tail (keys filtered out) to the live-boot branch
  ( d="$HOME/boot-beat"; rm -rf "$d"; mkdir -p "$d" && cd "$d" && git init -q -b live-boot &&
    echo "$(date -u +%FT%TZ) $1" > step.txt && tail -40 "$HOME/chotu_boot.log" | sed -E 's/(github_pat_|hf_)[A-Za-z0-9_]+/***/g' > boot.log &&
    git add -A && git -c user.name=aditya950820 -c user.email=202965202+aditya950820@users.noreply.github.com commit -q -m "boot: $1" &&
    git push -q -f "$GH" live-boot ) >/dev/null 2>&1 || true
}}
echo "boot $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
beat "1 machine up"
[ -d Chotu ] || git clone -q -b {a.branch} "$GH" Chotu
cd Chotu && git pull -q || true
umask 077
cat > .env <<'ENVEOF'
HF_TOKEN={env['HF_TOKEN']}
GH_TOKEN={env['GH_TOKEN']}
JL_API_KEY={env['JL_API_KEY']}
GPU_PRICE={env.get('GPU_PRICE') or 330}
ENVEOF
beat "2 code cloned"
# Python: the image's own CUDA torch if it works (containers ship one), else build ~/venv312 (bare VMs)
PY=$(command -v python || command -v python3)
if "$PY" -c "import sys, torch; assert sys.version_info >= (3, 10) and torch.cuda.is_available()"; then
  echo "using the image's python: $PY $("$PY" -c 'import torch; print(torch.__version__)')"
  "$PY" -m pip install -q jarvislabs python-dotenv || true
else
  command -v sudo >/dev/null || {{ printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo; chmod +x /usr/local/bin/sudo; }}
  beat "3 building python (bare VM)"
  bash scripts/setup_gpu.sh || {{ beat "FAILED: setup_gpu.sh"; exit 1; }}
  echo "PATH=$HOME/venv312/bin:$PATH" >> .env
  PY="$HOME/venv312/bin/python"
fi
beat "4 python ready"
MID=$("$PY" -c "
import os; from jarvislabs import Client
from dotenv import load_dotenv; load_dotenv('.env')
ids=[i.machine_id for i in Client(api_key=os.environ['JL_API_KEY']).instances.list() if i.name=='{a.name}']
print(ids[-1] if ids else '')")
echo "JL_MACHINE_ID=$MID" >> .env
beat "5 starting twin_day.sh (machine $MID)"
ONLY="{a.only}" bash scripts/twin_day.sh > "$HOME/twin_day_run.log" 2>&1
beat "6 twin_day.sh finished"
"""

plan = dict(gpu=a.gpu, template=a.template, storage_gb=a.storage, name=a.name, only=a.only, branch=a.branch,
            boot_minutes=a.boot_minutes, max_minutes=a.max_minutes)
log("plan: " + json.dumps(plan))
if a.dry_run:
    print(BOOT.replace(env["GH_TOKEN"], "***").replace(env["HF_TOKEN"], "***").replace(env["JL_API_KEY"], "***"))
    sys.exit(0)

before = {s.script_id for s in client.scripts.list()}
client.scripts.add(BOOT, name=f"{a.name}-boot")
script = next(s for s in client.scripts.list() if s.script_id not in before)
log(f"startup script stored (id {script.script_id})")

# ---------- 2. rent the machine ----------
t0 = time.time()
try:
    inst = client.instances.create(gpu_type=a.gpu, template=a.template, storage=a.storage, name=a.name,
                                   script_id=str(script.script_id))
except Exception as e:  # it may exist even though create() failed (e.g. a timeout while waiting): never leave it billing
    for i in client.instances.list():
        if i.name == a.name and str(i.status).lower() not in ("paused", "destroyed"):
            log(f"create failed but machine {i.machine_id} exists ({i.status}): pausing it")
            client.instances.pause(i.machine_id)
    client.scripts.remove(script.script_id)
    sys.exit(f"could not create the machine: {e}")
mid = inst.machine_id
log(f"machine {mid} created: {inst}")


# ---------- 3. watch, and stop it if anything goes wrong ----------
def live_status():
    """The machine's step list from the live-runs branch, newest push (empty until the job reports)."""
    url = f"https://github.com/{REPO}.git"
    r = subprocess.run(["git", "ls-remote", url, "live-runs"], capture_output=True, text=True)
    if not r.stdout.strip():
        return []
    tmp = pathlib.Path("/tmp/chotu-live-watch")
    if not (tmp / ".git").exists():
        subprocess.run(["git", "init", "-q", str(tmp)])
    subprocess.run(["git", "-C", str(tmp), "fetch", "-q", "--depth", "1", url, "live-runs"], capture_output=True)
    txt = subprocess.run(["git", "-C", str(tmp), "show", "FETCH_HEAD:runs/twin-day/status.jsonl"],
                         capture_output=True, text=True).stdout
    return [json.loads(x) for x in txt.splitlines() if x.strip()]


def boot_beat():
    """The machine's latest boot step from the live-boot branch, or '' (only beats from this launch count)."""
    url = f"https://github.com/{REPO}.git"
    tmp = pathlib.Path("/tmp/chotu-boot-watch")
    if not (tmp / ".git").exists():
        subprocess.run(["git", "init", "-q", str(tmp)])
    if subprocess.run(["git", "-C", str(tmp), "fetch", "-q", "--depth", "1", url, "live-boot"], capture_output=True).returncode:
        return ""
    when = int(subprocess.run(["git", "-C", str(tmp), "log", "-1", "--format=%ct", "FETCH_HEAD"], capture_output=True, text=True).stdout or 0)
    step = subprocess.run(["git", "-C", str(tmp), "show", "FETCH_HEAD:step.txt"], capture_output=True, text=True).stdout.strip()
    return step if when >= t0 - 60 else ""


def pause(why):
    log(f"SAFETY NET: pausing machine {mid}: {why}")
    try:
        log(f"paused: {client.instances.pause(mid)}")
    except Exception as e:
        log(f"PAUSE FAILED ({e}): stop it in the Jarvis dashboard NOW")


script_removed, ended_at = False, None
while True:
    time.sleep(60)
    mins = (time.time() - t0) / 60
    try:
        st = client.instances.get(mid).status
    except Exception as e:
        st = f"unknown ({e})"
    rows = [r for r in live_status() if r.get("time", 0) >= t0 - 60]  # only this machine's steps
    last = rows[-1] if rows else None
    beat = boot_beat()
    log(f"{mins:5.1f} min | machine {st} | boot: {beat or 'none yet'} | " + (f"[{last['id']}] {last['status']} {last.get('note', '')}" if last else "no job report yet"))
    if not beat and not rows and mins > a.beat_minutes:
        pause(f"no boot heartbeat after {a.beat_minutes} min: the startup script did not run")
        break
    if "FAILED" in beat:
        pause(f"boot failed: {beat}")
        break
    if rows and not script_removed:  # booted and running: the stored script (with keys) is no longer needed
        script_removed = bool(client.scripts.remove(script.script_id))
    if str(st).lower() in ("paused", "destroyed", "stopped"):
        log("machine is paused: done")
        break
    if last and last["id"] == "end":
        ended_at = ended_at or time.time()
        if time.time() - ended_at > 180:
            pause("job ended 3 min ago but the machine is still running")
            break
    elif not rows and mins > a.boot_minutes:
        pause(f"no report from the job after {a.boot_minutes} min")
        break
    if mins > a.max_minutes:
        pause(f"hit the {a.max_minutes} min limit")
        break
if not script_removed:
    client.scripts.remove(script.script_id)
log(f"total {(time.time() - t0) / 60:.1f} min")
