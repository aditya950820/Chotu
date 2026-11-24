"""Keep a private Hugging Face copy of everything the run writes, so nothing is lost even if a
Spot interruption deletes the instance.

    python scripts/backup.py --check       # create/verify the PRIVATE repo (gpu_up.sh, before spending)
    python scripts/backup.py --loop        # logs/reports/metrics every 10 min, checkpoints every 60 min
    python scripts/backup.py --once        # one full backup now (the chain runs this before pausing)
    python scripts/backup.py --restore     # new instance: put the backup back before anything starts
    python scripts/backup.py --apply-shard-mtimes   # after re-downloading shards (resume signature)

Repo: $CHOTU_BACKUP_REPO, default <your HF user>/chotu-run-backup, always private (checked).
Backed up: every file under the storage volume except data that is re-downloadable (shards/, chat/,
cache/, hf-home/) and the dashboard key. Checkpoints: only the newest per run of a stage that is not
finished yet (older remote ones are deleted and history squashed, so HF storage stays small).
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

SKIP_PARTS = {"shards", "chat", "cache", "hf-home", "backup-staging", "backup-restore", "__pycache__", ".cache"}
SKIP_FILES = {"dashboard/key", "dashboard/backup-state.json"}
STAGE_OF_RUN = {"speed": "speed", "lr-0.0015": "lr_test", "lr-0.003": "lr_test", "pretrain": "pretrain", "chat": "chat_tune"}
BIG = 50 * 2**20


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def collect(storage):
    """-> (small files, big files, wanted checkpoints, symlinks, shard mtimes); paths relative to storage."""
    small, big, checkpoints, links, mtimes = [], [], [], {}, {}
    for folder, dirs, files in os.walk(storage):
        rel_dir = Path(folder).relative_to(storage)
        for d in list(dirs):
            if (Path(folder) / d).is_symlink():
                links[str(rel_dir / d)] = os.readlink(Path(folder) / d)
                dirs.remove(d)
            elif d in SKIP_PARTS or d.startswith("sweep-"):
                dirs.remove(d)
        for name in files:
            path, rel = Path(folder) / name, rel_dir / name
            if path.is_symlink():
                links[str(rel)] = os.readlink(path)
                continue
            if str(rel) in SKIP_FILES or name.endswith((".tmp", ".env")) or name == ".env":
                continue
            size = path.stat().st_size
            if name.endswith(".pt"):
                continue  # chosen below: only the newest checkpoint of unfinished stages
            if rel.parts[0] == "rehearsal" and size > 5 * 2**20:
                continue  # rehearsal: logs and reports are the point, not its tiny models
            (big if size >= BIG else small).append(str(rel))
    for run, stage in STAGE_OF_RUN.items():
        run_dir = storage / "runs" / run
        if not (storage / "stages" / f"{stage}.done.json").exists() and run_dir.exists():
            latest = sorted(run_dir.glob("checkpoint-*.pt"))
            if latest:
                checkpoints.append(str(latest[-1].relative_to(storage)))
    for shard in sorted((storage / "shards").glob("*.bin")) if (storage / "shards").exists() else []:
        mtimes[str(shard.relative_to(storage))] = shard.stat().st_mtime_ns
    return sorted(small), sorted(big), checkpoints, links, mtimes


class Backup:
    def __init__(self, storage, repo=None, api=None):
        from huggingface_hub import HfApi
        self.storage = Path(storage).expanduser().resolve()
        self.api = api or HfApi()
        self.repo = repo or os.getenv("CHOTU_BACKUP_REPO") or f"{self.api.whoami()['name']}/chotu-run-backup"
        self.state_path = self.storage / "dashboard" / "backup-state.json"
        self.status_path = self.storage / "dashboard" / "backup.json"

    def status(self, **values):
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        current = json.loads(self.status_path.read_text()) if self.status_path.exists() else {}
        current.update(repo=self.repo, **values)
        self.status_path.write_text(json.dumps(current))

    def check(self):
        self.api.create_repo(repo_id=self.repo, repo_type="model", private=True, exist_ok=True)
        if self.api.repo_info(repo_id=self.repo, repo_type="model").private is not True:
            raise SystemExit(f"refusing to back up: {self.repo} is not private")
        self.status(checked=now())
        print(f"backup repo OK (private): {self.repo}")

    def run(self, checkpoints=True):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        self.check()
        small, big, wanted, links, mtimes = collect(self.storage)
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        staging = self.storage / "backup-staging" / str(os.getpid())  # the loop and a final --once may overlap
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        manifest = dict(time=now(), symlinks=links, shard_mtimes_ns=mtimes, checkpoints=wanted)
        (staging / "backup-manifest.json").write_text(json.dumps(manifest, indent=1))
        ops, sent = [CommitOperationAdd("backup-manifest.json", str(staging / "backup-manifest.json"))], {}
        files = small + big + (wanted if checkpoints else [])
        for rel in files:
            path = self.storage / rel
            try:
                stat = path.stat()
            except OSError:
                continue  # rotated away since the scan
            key = [stat.st_size, stat.st_mtime_ns]
            if state.get(rel) == key:
                continue
            # A hard link pins the file even if training rotates it away mid-upload (no extra disk).
            copy = staging / rel
            copy.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, copy)
            except OSError:
                shutil.copy2(path, copy)
            ops.append(CommitOperationAdd("storage/" + rel, str(copy)))
            sent[rel] = key
        remote = set(self.api.list_repo_files(repo_id=self.repo, repo_type="model"))
        stale = [f for f in remote if f.endswith(".pt") and f.removeprefix("storage/") not in wanted]
        if checkpoints:
            ops += [CommitOperationDelete(f) for f in stale]
        message = f"backup {now()}: {len(sent)} files" + (f", -{len(stale)} old checkpoints" if checkpoints and stale else "")
        self.api.create_commit(repo_id=self.repo, repo_type="model", operations=ops, commit_message=message)
        state.update(sent)
        for rel in [k for k in state if not (self.storage / k).exists()]:
            state.pop(rel)
        self.state_path.write_text(json.dumps(state))
        shutil.rmtree(staging, ignore_errors=True)
        values = dict(logs_time=now(), error="")
        if checkpoints:
            values.update(checkpoint_time=now(), checkpoint=", ".join(wanted) or "none needed")
            try:  # drop old checkpoint blobs from history so HF storage stays ~one checkpoint
                self.api.super_squash_history(repo_id=self.repo, repo_type="model")
            except Exception as error:
                print(f"squash skipped: {error}", flush=True)
        self.status(**values)
        print(message, flush=True)

    def loop(self, every_min, checkpoint_min):
        last_checkpoint = 0.0
        while True:
            due = time.monotonic() - last_checkpoint >= checkpoint_min * 60
            try:
                self.run(checkpoints=due)
                if due:
                    last_checkpoint = time.monotonic()
            except Exception as error:
                print(f"backup failed, retrying next round: {type(error).__name__}: {error}", flush=True)
                try:
                    self.status(error=f"{now()}: {type(error).__name__}: {error}"[:300])
                except OSError:
                    pass
            time.sleep(every_min * 60)

    def restore(self):
        from huggingface_hub import snapshot_download
        if any((self.storage / "stages").glob("*.done.json")) or any((self.storage / "runs").glob("*/checkpoint-*.pt")):
            print("local run state present; nothing restored")
            return
        if not any(f == "backup-manifest.json" for f in self.api.list_repo_files(repo_id=self.repo, repo_type="model")):
            print("no backup yet; fresh start")
            return
        target = self.storage / "backup-restore"
        shutil.rmtree(target, ignore_errors=True)
        snapshot_download(self.repo, repo_type="model", local_dir=target, max_workers=16)
        manifest = json.loads((target / "backup-manifest.json").read_text())
        moved = 0
        for path in sorted((target / "storage").rglob("*")) if (target / "storage").exists() else []:
            if path.is_file():
                dest = self.storage / path.relative_to(target / "storage")
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, dest)
                moved += 1
        for rel, link in manifest["symlinks"].items():
            dest = self.storage / rel
            if not dest.exists() and not dest.is_symlink():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(link)
        (self.storage / "dashboard").mkdir(parents=True, exist_ok=True)
        (self.storage / "dashboard" / "restored-manifest.json").write_text(json.dumps(manifest))
        shutil.rmtree(target, ignore_errors=True)
        print(f"restored {moved} files and {len(manifest['symlinks'])} links from {self.repo} (backup {manifest['time']})")


def apply_shard_mtimes(storage):
    """Training resume checks shard size + mtime; re-downloaded shards get the original mtimes back."""
    path = Path(storage).expanduser() / "dashboard" / "restored-manifest.json"
    if not path.exists():
        return
    for rel, mtime in json.loads(path.read_text())["shard_mtimes_ns"].items():
        shard = Path(storage).expanduser() / rel
        if shard.exists():
            os.utime(shard, ns=(mtime, mtime))
    print("shard mtimes restored for resume")


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage", default=os.getenv("CHOTU_STORAGE"))
    ap.add_argument("--repo")
    mode = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--check", "--loop", "--once", "--restore", "--apply-shard-mtimes"):
        mode.add_argument(flag, action="store_true")
    ap.add_argument("--every-min", type=float, default=10)
    ap.add_argument("--checkpoint-min", type=float, default=60)
    a = ap.parse_args()
    if not a.storage:
        sys.exit("pass --storage or set CHOTU_STORAGE")
    if a.apply_shard_mtimes:
        return apply_shard_mtimes(a.storage)
    backup = Backup(a.storage, a.repo)
    if a.check:
        backup.check()
    elif a.restore:
        backup.restore()
    elif a.once:
        backup.run(checkpoints=True)
    else:
        backup.loop(a.every_min, a.checkpoint_min)


if __name__ == "__main__":
    main()
