"""Run on any new machine: python scripts/check_setup.py
Checks keys, git identity and the model size without printing secrets."""
import os
import subprocess
import sys

import requests
from dotenv import load_dotenv

load_dotenv()
ok = True


def report(name, good, detail=""):
    global ok
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] {name} {detail}")


hf = os.getenv("HF_TOKEN", "")
try:
    r = requests.get("https://huggingface.co/api/whoami-v2",
                     headers={"Authorization": f"Bearer {hf}"}, timeout=20).json()
    report("Hugging Face token", "name" in r, f"user={r.get('name')}")
except Exception as e:
    report("Hugging Face token", False, str(e))

wb = os.getenv("WANDB_API_KEY", "")
try:
    r = requests.post("https://api.wandb.ai/graphql", auth=("api", wb), timeout=20,
                      json={"query": "{ viewer { username entity } }"}).json()
    v = (r.get("data") or {}).get("viewer") or {}
    report("W&B key", bool(v), f"user={v.get('username')} entity={v.get('entity')}")
except Exception as e:
    report("W&B key", False, str(e))

name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).stdout.strip()
report("git identity is a person, not Claude", bool(name) and "claude" not in name.lower(), f"user.name={name!r}")

rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True)
report("model size tests", rc.returncode == 0, rc.stdout.strip().splitlines()[-1] if rc.stdout else "")

sys.exit(0 if ok else 1)
