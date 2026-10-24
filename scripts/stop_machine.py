"""Pause this Jarvis Labs instance so GPU billing stops (storage is kept and still billed).

Every GPU job ends by calling this, even after a failure:
    python scripts/stop_machine.py            # pause now
    python scripts/stop_machine.py --dry-run  # check the key and machine id without pausing

Needs in .env: JL_API_KEY (jarvislabs.ai/settings/api-keys) and JL_MACHINE_ID (from the
dashboard or `python scripts/stop_machine.py --list`).
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--list", action="store_true")
a = ap.parse_args()

try:
    from jarvislabs import Client
except ImportError:
    sys.exit("pip install jarvislabs")

client = Client(api_key=os.getenv("JL_API_KEY"))
if a.list:
    for inst in client.instances.list():
        print(inst)
    sys.exit(0)
mid = os.getenv("JL_MACHINE_ID")
if not mid:
    sys.exit("Set JL_MACHINE_ID in .env (run with --list to find it)")
inst = client.instances.get(mid)
print("this machine:", inst)
if a.dry_run:
    print("dry run: key and machine id work; not pausing")
else:
    print("pausing:", client.instances.pause(mid))
