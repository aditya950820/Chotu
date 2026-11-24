import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("run_chain", Path(__file__).resolve().parents[1] / "scripts/run_chain.py")
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    args = chain.parser().parse_args(["--storage", str(tmp_path), "--rehearsal", "--device", "cpu", "--skip-upload"])
    return chain.Chain(args)


def fake_stages(monkeypatch, runner, seen, failure=None):
    for name in chain.STAGES:
        def stage(name=name):
            seen.append(name)
            if name == "pretrain" and failure:
                raise failure
            return {}
        monkeypatch.setattr(runner, name, stage)


def test_order_done_skip_and_shutdown(runner, monkeypatch):
    seen = []
    fake_stages(monkeypatch, runner, seen)
    pause = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(chain.subprocess, "run", pause)
    before = signal.getsignal(signal.SIGTERM)
    assert runner.run() == 0
    assert seen == list(chain.STAGES)
    assert len(list(runner.markers.glob("*.done.json"))) == len(chain.STAGES)
    assert "stop_machine.py" in pause.call_args.args[0][1]
    assert signal.getsignal(signal.SIGTERM) == before
    assert runner.run() == 0
    assert seen == list(chain.STAGES)
    assert pause.call_count == 2
    events = chain.metrics(runner.logs / "chain.jsonl")
    assert sum(e["state"] == "skip_done" for e in events) == len(chain.STAGES)


@pytest.mark.parametrize("failure,code", [
    (chain.StageError(20, "NaN"), 20), (chain.StageError(1, "crash"), 1),
    (ValueError("exception"), 1), (chain.ChainSignal(signal.SIGINT), 130),
])
def test_shutdown_on_failure_exception_and_ctrl_c(runner, monkeypatch, failure, code):
    seen = []
    fake_stages(monkeypatch, runner, seen, failure)
    pause = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(chain.subprocess, "run", pause)
    assert runner.run() == code
    assert seen[-1] == "pretrain" and "base_evals" not in seen
    assert not (runner.markers / "pretrain.done.json").exists()
    pause.assert_called_once()


def test_no_stop_and_failed_pause(runner, monkeypatch, capsys):
    fake_stages(monkeypatch, runner, [])
    pause = MagicMock(return_value=SimpleNamespace(returncode=1))
    monkeypatch.setattr(chain.subprocess, "run", pause)
    runner.a.no_stop = True
    assert runner.run() == 0
    pause.assert_not_called()
    runner.a.no_stop = False
    assert runner.run() == 25
    assert "PAUSE FAILED" in capsys.readouterr().out


def test_pause_survives_broken_log_volume(runner, monkeypatch):
    runner.setup()
    monkeypatch.setattr(Path, "open", MagicMock(side_effect=OSError("full volume")))
    pause = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(chain.subprocess, "run", pause)
    assert runner.stop()
    pause.assert_called_once()


def test_rehearsal_wandb_project_is_separate(runner, monkeypatch):
    import sys
    monkeypatch.setenv("WANDB_API_KEY", "mock-key")
    monkeypatch.setenv("WANDB_PROJECT", "chotu")
    wandb = SimpleNamespace(init=MagicMock())
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    runner.setup()
    assert wandb.init.call_args.kwargs["project"] == "chotu-rehearsal"
    assert chain.os.environ["WANDB_PROJECT"] == "chotu-rehearsal"


def test_setup_exception_still_pauses(runner, monkeypatch):
    monkeypatch.setattr(runner, "setup", MagicMock(side_effect=RuntimeError("setup failed")))
    pause = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(chain.subprocess, "run", pause)
    assert runner.run() == 1
    pause.assert_called_once()


@pytest.mark.parametrize("codes,training,expected_calls,expected_error", [
    ([1, 0], True, 2, None), ([-9, 0], True, 2, None), ([1, 1, 1], True, 3, 1),
    ([20, 0], True, 1, 20), ([21, 0], True, 1, 21), ([22, 0], True, 1, 22),
    ([23, 0], True, 1, 23), ([24, 0], True, 1, 24), ([1, 0], False, 1, 1),
])
def test_retry_policy(runner, monkeypatch, codes, training, expected_calls, expected_error):
    runner.setup()
    popen = MagicMock()
    def child(*args, **kwargs):
        code = codes.pop(0)
        return SimpleNamespace(pid=123, stdout=iter(['{"step":1,"loss":3}\n']), wait=lambda: code)
    popen.side_effect = child
    monkeypatch.setattr(chain.subprocess, "Popen", popen)
    if expected_error:
        with pytest.raises(chain.StageError) as error:
            runner.execute(["fake"], training=training)
        assert error.value.code == expected_error
    else:
        runner.execute(["fake"], training=training)
    assert popen.call_count == expected_calls


def test_signal_and_child_termination(runner, monkeypatch):
    with pytest.raises(chain.ChainSignal) as stopped:
        runner.on_signal(signal.SIGTERM, None)
    assert stopped.value.number == signal.SIGTERM
    child = MagicMock(pid=123)
    child.poll.return_value = None
    child.wait.side_effect = [subprocess.TimeoutExpired("fake", 60), 0]
    runner.child = child
    kill = MagicMock()
    monkeypatch.setattr(chain.os, "killpg", kill)
    runner.terminate_child()
    assert [c.args for c in kill.call_args_list] == [(123, signal.SIGTERM), (123, signal.SIGKILL)]
    assert runner.child is None


def test_manual_controls_and_dry_run(tmp_path, capsys):
    args = chain.parser().parse_args(["--storage", str(tmp_path / "not-created"), "--dry-run"])
    chain.dry_plan(args)
    output = capsys.readouterr().out
    for i, name in enumerate(chain.STAGES, 1):
        assert f"{i:02d}. {name}" in output
    assert "74000" not in output  # formatted, readable throughput range
    assert "74,000–150,000" in output and "330" in output
    assert not (tmp_path / "not-created").exists()
    args.from_stage = "pretrain"
    assert chain.selected_stages(args)[0] == "pretrain"
    args.only_stage = "export_gguf"
    assert chain.selected_stages(args) == ("export_gguf",)


def test_dry_run_saved_speed(runner, capsys):
    runner.reports.mkdir(parents=True)
    chain.atomic_json(runner.reports / "speed.json", {"tokens_per_second": 100000})
    chain.dry_plan(runner.a)
    assert "100,000–100,000 tok/s (saved speed check)" in capsys.readouterr().out


def test_configuration_and_missing_outputs_guard(runner, monkeypatch):
    seen = []
    fake_stages(monkeypatch, runner, seen)
    monkeypatch.setattr(runner, "stop", lambda: True)
    runner.a.no_stop = True
    assert runner.run() == 0
    marker = runner.markers / "shards.done.json"
    data = chain.read_json(marker)
    data["result"]["outputs"] = [str(runner.storage / "missing.bin")]
    chain.atomic_json(marker, data)
    assert runner.run() == 1
    runner.a.rehearsal_tokens *= 2
    with pytest.raises(ValueError, match="settings/code changed"):
        runner.setup()


def test_disk_calculation():
    budget = chain.disk_budget(20_210_000_000, 180925056)
    assert budget["parts"]["checkpoints"] == 180925056 * 12 * 2 * 5
    assert budget["required_bytes"] > sum(budget["parts"].values())
    assert 70 * chain.GIB < budget["required_bytes"] < 100 * chain.GIB


def test_shards_counts_hashes_and_corruption(runner, monkeypatch):
    import huggingface_hub
    runner.setup()
    root = runner.storage / "shards"
    root.mkdir()
    np.asarray([1, 2, 0, 3, 0], dtype=np.uint16).tofile(root / "train_000.bin")
    np.asarray([4, 0], dtype=np.uint16).tofile(root / "val_000.bin")
    chain.atomic_json(root / "000.done", {"train_tokens": 5, "val_tokens": 2, "docs": 3, "val_docs": 1})
    files = {"shards/" + p.name: {"size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.suffix == ".bin" else None} for p in root.iterdir()}
    monkeypatch.setattr(runner, "shard_manifest", lambda: {"revision": "fixture", "files": files})
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name, **kwargs: runner.storage / name)
    assert runner.shards()["tokens"] == {"train": 5, "val": 2}
    (root / "val_000.bin").write_bytes(b"1234")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runner.shards()


def test_shards_allow_literal_endoftext_but_not_missing_docs(runner, monkeypatch):
    import huggingface_hub
    runner.setup()
    root = runner.storage / "shards"
    root.mkdir()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name, **kwargs: runner.storage / name)

    def check(train, docs):
        np.asarray(train, dtype=np.uint16).tofile(root / "train_000.bin")
        np.asarray([4, 0], dtype=np.uint16).tofile(root / "val_000.bin")
        chain.atomic_json(root / "000.done", {"train_tokens": len(train), "val_tokens": 2, "docs": docs, "val_docs": 1})
        files = {"shards/" + p.name: {"size": p.stat().st_size, "sha256": None} for p in root.iterdir()}
        monkeypatch.setattr(runner, "shard_manifest", lambda: {"revision": "fixture", "files": files})
        return runner.shards()

    # one document whose text contained "<|endoftext|>": 3 separators for 2 train documents
    result = check([1, 0, 2, 0, 3, 0], docs=3)
    assert result["docs"] == {"train": 2, "val": 1}
    assert result["literal_endoftext_in_text"] == {"shards/train_000.bin": 1}
    with pytest.raises(ValueError, match="document count mismatch"):
        check([1, 0, 2, 3, 0], docs=4)      # fewer separators than documents
    with pytest.raises(ValueError, match="document count mismatch"):
        check([1, 0, 2, 0, 3], docs=3)      # shard does not end with a separator


def test_training_and_eval_shutdown_ownership(runner, monkeypatch):
    command = runner.train("test", "--stage", "pretrain", "--lr", ".003")
    assert "--no-auto-stop" in command and command[command.index("--model") + 1] == "tiny"
    calls = []
    monkeypatch.setattr(runner, "execute", lambda cmd: calls.append(cmd))
    runner.eval_lm("model", "test")
    assert "--no-auto-stop" in calls[0]


def test_lr_frozen_trials_and_pick(runner, monkeypatch):
    runner.setup()
    runner.a.rehearsal = False
    calls = []
    def execute(command, training):
        calls.append(command)
        name = command[command.index("--name") + 1]
        path = runner.storage / "runs" / name
        path.mkdir(parents=True)
        (path / "metrics.jsonl").write_text(json.dumps({"val_bpb": 2 if "0.0015" in name else 1.8}) + "\n")
    monkeypatch.setattr(runner, "execute", execute)
    assert runner.lr_test()["lr"] == .003
    assert len(calls) == 2
    for command in calls:
        assert command[command.index("--total-tokens") + 1] == "5e8"
        assert command[command.index("--warmup-steps") + 1] == "100"
        assert command[command.index("--decay-frac") + 1] == ".15"


def test_rehearsal_upload_is_separate_or_explicit_skip(runner, monkeypatch):
    runner.setup()
    calls = []
    monkeypatch.setattr(runner, "execute", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(runner, "write_report", lambda status: None)
    assert runner.upload()["skipped"]
    assert calls == []
    runner.a.skip_upload = False
    runner.repo = "owner/chotu"
    assert runner.upload()["repo"] == "owner/chotu-rehearsal"
    assert "owner/chotu-rehearsal" in calls[0]
