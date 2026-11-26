"""Put back the file times a training checkpoint recorded, after a restore onto a new disk.

    python tools/match_manifest.py --storage /home/chotu-data --run chat

train.py refuses to resume when its size + modification-time manifest of the input files differs
from the newest checkpoint's. A restore (HF backup download, rsync) brings back the same bytes but
new times, so the run would refuse to continue on a new machine. This script only changes
modification times, and only for files whose size matches the checkpoint's record exactly; any
missing or different-size file is reported and the script exits 1 (then do NOT resume).
"""
import argparse
import os
from pathlib import Path
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage", default=os.getenv("CHOTU_STORAGE"))
    ap.add_argument("--run", default="chat", help="run folder under <storage>/runs")
    a = ap.parse_args()
    checkpoints = sorted(Path(a.storage, "runs", a.run).glob("checkpoint-*.pt"))
    if not checkpoints:
        print(f"no checkpoint in runs/{a.run}; nothing to match")
        return
    signature = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)["signature"]
    entries = signature.get("shards", []) + signature.get("init_manifest", [])
    fixed, problems = 0, []
    for path, size, mtime in entries:
        p = Path(path)
        if not p.exists():
            problems.append(f"missing: {p}")
        elif p.stat().st_size != size:
            problems.append(f"size differs: {p} ({p.stat().st_size} vs recorded {size})")
        else:
            os.utime(p, ns=(mtime, mtime))
            fixed += 1
    print(f"{checkpoints[-1].name}: {fixed}/{len(entries)} file times matched to the checkpoint's record")
    for line in problems:
        print("PROBLEM " + line)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
