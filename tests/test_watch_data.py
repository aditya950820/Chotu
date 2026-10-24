"""The live dashboard reads progress correctly from the shard folder (no W&B needed)."""
import importlib.util
import json
import os

spec = importlib.util.spec_from_file_location("watch", os.path.join(os.path.dirname(__file__), "..", "scripts", "watch_data.py"))
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def test_snapshot_counts_finished_and_in_progress(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "train_000.bin").write_bytes(b"\0" * 2000)       # 1,000 tokens, finished
    (shards / "000.done").write_text(json.dumps({"docs": 1}))
    (shards / "train_001.bin.tmp").write_bytes(b"\0" * 600)     # 300 tokens, in progress
    s = watch.snapshot(str(tmp_path))
    assert s == {"files_done": 1, "tokens": 1300, "report_written": False}


def test_rate_and_eta():
    history = [(0, 0), (100, 1_000_000)]                      # 10k tokens/s
    rate, eta = watch.rate_and_eta(history, 1_000_000)
    assert rate == 10_000
    assert abs(eta - (10e9 - 1e6) / 10_000 / 3600) < 1e-9
    assert watch.rate_and_eta([(0, 0)], 0) == (0.0, None)


def test_size_of_vanished_file_is_zero(tmp_path):
    assert watch._size(str(tmp_path / "renamed_away.bin.tmp")) == 0
