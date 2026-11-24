import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import backup  # noqa: E402
import dashboard  # noqa: E402


class FakeHub:
    """In-memory private repo: files are copied into a folder on commit."""
    def __init__(self, root, private=True):
        self.root, self.private, self.commits = Path(root), private, []
        self.root.mkdir(parents=True, exist_ok=True)

    def whoami(self):
        return {"name": "someone"}

    def create_repo(self, **kw):
        assert kw["private"] is True

    def repo_info(self, **kw):
        return SimpleNamespace(private=self.private)

    def list_repo_files(self, **kw):
        return [str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()]

    def create_commit(self, operations, **kw):
        self.commits.append([type(op).__name__ + ":" + op.path_in_repo for op in operations])
        for op in operations:
            dest = self.root / op.path_in_repo
            if type(op).__name__ == "CommitOperationDelete":
                dest.unlink()
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(op.path_or_fileobj, dest)

    def super_squash_history(self, **kw):
        pass


def make_storage(root):
    (root / "shards").mkdir(parents=True)
    (root / "shards/train_000.bin").write_bytes(b"x" * 10)
    os.utime(root / "shards/train_000.bin", ns=(1_000_000_000, 1_000_000_000))
    (root / "chat").mkdir()
    (root / "chat/train.ids").write_bytes(b"c")
    run = root / "runs/pretrain"
    (run / "hf/final-abc").mkdir(parents=True)
    (run / "hf/final-abc/config.json").write_text("{}")
    (run / "final").symlink_to("hf/final-abc", target_is_directory=True)
    for step in (100, 200):
        (run / f"checkpoint-{step:08d}.pt").write_bytes(b"w" * step)
    (run / "plan.json").write_text(json.dumps({"stage": "pretrain", "steps": 1000, "batch_tokens": 524288}))
    rows = [dict(step=i, loss=10 - i / 50, lr=3e-3, grad_norm=0.4, tokens_per_second=2e5, mfu=0.3,
                 data_wait_frac=0.01, seq_len=2048, val_bpb=1.1 if i == 200 else None) for i in range(1, 201)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "stages").mkdir()
    (root / "stages/preflight.done.json").write_text(json.dumps({"seconds": 12}))
    (root / "dashboard").mkdir()
    (root / "dashboard/key").write_text("secret-key")
    (root / "logs").mkdir()
    (root / "logs/chain.jsonl").write_text(json.dumps({"stage": "pretrain", "state": "progress"}) + "\n")
    (root / "status.json").write_text(json.dumps({"stage": "pretrain", "state": "progress"}))


def test_backup_skips_data_and_key_keeps_newest_checkpoint_then_restores(tmp_path, monkeypatch):
    src = tmp_path / "old-instance"
    make_storage(src)
    hub = FakeHub(tmp_path / "hub")
    b = backup.Backup(src, api=hub)
    b.run(checkpoints=True)
    files = set(hub.list_repo_files())
    assert "storage/runs/pretrain/checkpoint-00000200.pt" in files
    assert "storage/runs/pretrain/checkpoint-00000100.pt" not in files
    assert "storage/runs/pretrain/metrics.jsonl" in files and "storage/runs/pretrain/hf/final-abc/config.json" in files
    assert not any(f.startswith(("storage/shards", "storage/chat")) or f.endswith("dashboard/key") for f in files)

    # next hour: a newer checkpoint replaces the old one remotely; unchanged files are not resent
    (src / "runs/pretrain/checkpoint-00000100.pt").unlink()
    (src / "runs/pretrain/checkpoint-00000300.pt").write_bytes(b"w" * 300)
    b.run(checkpoints=True)
    assert "CommitOperationDelete:storage/runs/pretrain/checkpoint-00000200.pt" in hub.commits[-1]
    assert not any("metrics.jsonl" in op for op in hub.commits[-1])
    # a finished stage needs no checkpoint backup at all
    (src / "stages/pretrain.done.json").write_text("{}")
    b.run(checkpoints=True)
    assert not any(f.endswith(".pt") for f in hub.list_repo_files())

    # brand-new instance: restore, then shards are re-downloaded with new mtimes
    dst = tmp_path / "new-instance"
    dst.mkdir()
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda repo, local_dir, **kw: shutil.copytree(hub.root, local_dir, dirs_exist_ok=True))
    backup.Backup(dst, api=hub).restore()
    assert json.loads((dst / "runs/pretrain/plan.json").read_text())["steps"] == 1000
    assert (dst / "runs/pretrain/final").is_symlink() and (dst / "runs/pretrain/final/config.json").exists()
    assert (dst / "stages/preflight.done.json").exists()
    (dst / "shards").mkdir()
    (dst / "shards/train_000.bin").write_bytes(b"x" * 10)
    backup.apply_shard_mtimes(dst)
    assert (dst / "shards/train_000.bin").stat().st_mtime_ns == 1_000_000_000
    backup.Backup(dst, api=hub).restore()  # second call is a no-op: local state present


def test_backup_refuses_public_repo(tmp_path):
    make_storage(tmp_path / "s")
    with pytest.raises(SystemExit, match="not private"):
        backup.Backup(tmp_path / "s", api=FakeHub(tmp_path / "hub", private=False)).check()


def test_dashboard_page_shows_live_numbers(tmp_path):
    make_storage(tmp_path)
    (tmp_path / "logs/system.jsonl").write_text(
        "".join(json.dumps(dict(time=i * 30, gpu_util=97, gpu_mem_gib=60, disk_free_gib=50)) + "\n" for i in range(120)))
    html = dashboard.page(tmp_path, rate=217, sample_seconds=30)
    assert "pretrain step" in html and "200 / 1,000" in html and "20.0%" in html
    assert "1h 00m" in html and "₹217" in html          # 120 samples × 30 s of billed uptime
    assert "200,000" in html and "MFU 30.0%" in html and "<svg" in html
    assert "secret-key" not in html


def test_dashboard_without_any_files(tmp_path):
    html = dashboard.page(tmp_path, rate=217)
    assert "no data yet" in html and "setup" in html
