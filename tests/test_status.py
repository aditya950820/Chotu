import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_status_shows_progress_and_time_left(tmp_path):
    (tmp_path / "stages").mkdir()
    (tmp_path / "stages/preflight.done.json").write_text("{}")
    run = tmp_path / "runs/pretrain"
    run.mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({"stage": "pretrain", "steps": 1000, "batch_tokens": 524288}))
    rows = [dict(step=i, loss=10.8 - i / 100, lr=3e-3, grad_norm=0.5, tokens_per_second=200000.0, mfu=0.3,
                 data_wait_frac=0.01, seq_len=2048, val_bpb=1.2 if i == 100 else None, best_val_bpb=1.2)
            for i in range(1, 101)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = subprocess.run([sys.executable, str(ROOT / "scripts/status.py"), "--storage", str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    assert "Stages done 1/12" in out and "[x]preflight" in out
    assert "step 100 / 1,000 (10.0%)" in out
    assert "200,000 tok/s" in out and "MFU 30.0%" in out and "val_bpb 1.2000" in out
    assert "time left for this run ≈ 39 min" in out   # 900 steps × 524,288 / 200k tok/s


def test_status_without_runs(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "scripts/status.py"), "--storage", str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    assert "Stages done 0/12" in out and "No training run yet." in out
