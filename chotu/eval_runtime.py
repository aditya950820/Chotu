"""Standalone GPU evaluation shutdown; the chain explicitly takes ownership."""
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv


@contextmanager
def shutdown_guard(device, no_auto_stop):
    load_dotenv()
    cuda = device.startswith("cuda") and not no_auto_stop
    try:
        if cuda and not (os.getenv("JL_MACHINE_ID") and os.getenv("JL_API_KEY")):
            raise ValueError("CUDA eval requires Jarvis credentials or --no-auto-stop with an external owner")
        yield
    finally:
        if cuda:
            subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/stop_machine.py")], check=True)
