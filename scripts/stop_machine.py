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
def field(inst, name):
    return inst.get(name) if isinstance(inst, dict) else getattr(inst, name, None)


def this_machine():
    """Jarvis gives an instance a NEW id on every resume, so a saved id goes stale. Use it if it is
    still valid, otherwise the account's only running instance (or the one with this machine's IP)."""
    mid = os.getenv("JL_MACHINE_ID")
    if mid:
        try:
            return client.instances.get(int(mid))
        except Exception:
            print(f"saved JL_MACHINE_ID {mid} is stale (Jarvis changes it on resume); looking it up")
    running = [i for i in client.instances.list() if str(field(i, "status")).lower() == "running"]
    if len(running) > 1:
        import socket
        local = set(socket.gethostbyname_ex(socket.gethostname())[2])
        running = [i for i in running if field(i, "private_ip") in local] or running
    if len(running) != 1:
        sys.exit(f"cannot tell which instance this is ({len(running)} running): pause it in the Jarvis dashboard")
    return running[0]


inst = this_machine()
mid = field(inst, "machine_id")
print("this machine:", mid, field(inst, "gpu_type"), field(inst, "status"))
if a.dry_run:
    print("dry run: key and machine id work; not pausing")
else:
    print("pausing:", client.instances.pause(mid))
