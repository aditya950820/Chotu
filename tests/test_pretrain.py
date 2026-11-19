import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from chotu.config import chotu_config
from chotu.data import PretrainBatches, ShardStream

spec = importlib.util.spec_from_file_location("train", Path(__file__).resolve().parents[1] / "scripts/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


@pytest.fixture
def shards(tmp_path):
    directory = tmp_path / "shards"
    directory.mkdir()
    for split in ("train", "val"):
        for index in range(2):
            doc = np.concatenate([np.full(10000 + index, 3 + index, dtype=np.uint16), [0]]).astype(np.uint16)
            np.tile(doc, 5).tofile(directory / f"{split}_{index:03d}.bin")
    return tmp_path


def test_stream_and_batches(shards):
    stream = ShardStream(shards, seed=7)
    again = ShardStream(shards, seed=7)
    assert stream.paths == again.paths
    boundary = stream.ends[0]
    assert stream.read(boundary - 1, 3).tolist() == [0, int(stream.arrays[1][0]), int(stream.arrays[1][1])]
    assert stream.read(0, 0).size == 0
    batches = PretrainBatches(stream, 65536, 8192, 0.5)
    other = PretrainBatches(again, 65536, 8192, 0.5)
    all_targets = []
    for step in range(batches.steps):
        for (x, y), (xx, yy) in zip(batches.batch(step, 8192), other.batch(step, 8192)):
            assert torch.equal(x, xx) and torch.equal(y, yy)
        if step < batches.switch_step:
            all_targets.extend(range(step * 8192 + 1, (step + 1) * 8192 + 1))
    for start in batches.long_offsets:
        all_targets.extend(range(start + 1, start + 8193))
    assert len(all_targets) == len(set(all_targets))
    assert batches.tail_stats["docs_ge_8192"] > 0
    assert batches.tail_stats["preferred_starts"] > 0
    for start in batches.long_offsets:
        assert start >= batches.switch_step * 8192
        assert start == 0 or stream.read(start - 1, 1)[0] == 0
    with pytest.raises(ValueError, match="outside stream"):
        stream.read(-1, 3)
    with pytest.raises(ValueError, match="not enough"):
        PretrainBatches(stream, 10, 8192)


def test_document_tail_and_empty(tmp_path, shards):
    stream = ShardStream(shards)
    docs = list(stream.documents(5000, 35000))
    assert docs and docs[0][0] >= 5000
    assert all(stream.read(end - 1, 1)[0] == 0 for _, end in docs)
    assert list(stream.documents(docs[0][0], docs[0][1])) == [docs[0]]
    with pytest.raises(ValueError, match="no tokens"):
        ShardStream(tmp_path / "missing")


def test_wsd():
    values = [train.wsd_lr(s, 10, 1, 2, 0.3) for s in range(10)]
    assert values == [0.5, 1, 1, 1, 1, 1, 1, 1, 0.5, 0]


def test_bpb():
    class Tokenizer:
        def decode(self, row, **kwargs):
            return "é" if row == [1] else "abc"
    assert train.bits_per_byte(5 * math.log(2), [[1], [2]], Tokenizer()) == pytest.approx(1)
    with pytest.raises(ValueError, match="zero bytes"):
        train.bits_per_byte(1, [], Tokenizer())


def test_output_must_stay_in_storage(shards):
    args = train.parser().parse_args(["--stage", "pretrain", "--storage", str(shards),
                                     "--lr", "0.003", "--name", "../../escape"])
    with pytest.raises(ValueError, match="inside --storage"):
        train.run(args)


def test_nan_in_training_stops(shards, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(train, "tiny_config", lambda: chotu_config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    original = train.F.cross_entropy
    monkeypatch.setattr(train.F, "cross_entropy", lambda *a, **kw: original(*a, **kw) * float("nan"))
    args = train.parser().parse_args(["--stage", "pretrain", "--model", "tiny",
        "--storage", str(shards), "--lr", "0.003", "--batch-tokens", "8192",
        "--micro-tokens", "8192", "--total-tokens", "16384", "--long-frac", "0",
        "--kill-rules", "--device", "cpu"])
    with pytest.raises(train.TrainingStop) as stopped:
        train.run(args)
    assert stopped.value.code == 20
    assert not list((shards / "runs" / "pretrain").glob("checkpoint-*.pt"))


def test_shutdown_after_rule_stop(monkeypatch):
    monkeypatch.setattr(train, "load_dotenv", lambda: None)
    monkeypatch.setenv("JL_MACHINE_ID", "test-machine")
    monkeypatch.setenv("JL_API_KEY", "test-only-not-a-key")
    monkeypatch.setattr(train.sys, "argv", ["train.py", "--stage", "pretrain", "--storage", "/unused",
                                          "--lr", "0.003", "--device", "cuda"])

    def stop(args):
        raise train.TrainingStop(22, "slow")

    monkeypatch.setattr(train, "run", stop)
    calls = []

    def pause(args):
        calls.append(args)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(train.subprocess, "run", pause)
    assert train.main() == 22
    assert calls and calls[0][-1].endswith("stop_machine.py")


def test_optional_telemetry(capsys):
    def unavailable():
        raise ConnectionError("offline")
    assert train.wandb_call(unavailable) is None
    assert "W&B unavailable: ConnectionError" in capsys.readouterr().out


@pytest.mark.parametrize("loss", [float("nan"), float("inf")])
def test_nonfinite_rule(loss):
    with pytest.raises(train.TrainingStop) as stopped:
        train.KillRules().check(1, loss, 100000)
    assert stopped.value.code == 20


def test_slow_loss_and_throughput_rules():
    with pytest.raises(train.TrainingStop) as stopped:
        rules = train.KillRules()
        for step in range(1, 301):
            rules.check(step, 7, 100000)
    assert stopped.value.code == 21
    with pytest.raises(train.TrainingStop) as stopped:
        rules = train.KillRules()
        for step in range(1, 51):
            rules.check(step, 5, 1)
    assert stopped.value.code == 22


def test_spike_and_recovery():
    rules = train.KillRules(0)
    rules.check(1, 2, 1)
    rules.check(2, 4, 1)
    rules.check(3, 2.5, 1)
    assert rules.spike_since is None
    rules.check(4, 4, 1)
    with pytest.raises(train.TrainingStop) as stopped:
        for step in range(5, 205):
            rules.check(step, 4, 1)
    assert stopped.value.code == 23


@pytest.mark.parametrize("interruption", [2, 3])
def test_exact_resume_and_stage_switch(shards, monkeypatch, interruption):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(train, "tiny_config", lambda: chotu_config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    args = ["--stage", "pretrain", "--model", "tiny", "--storage", str(shards),
            "--device", "cpu", "--total-tokens", "40960", "--batch-tokens", "8192",
            "--micro-tokens", "8192", "--long-frac", "0.6", "--warmup-steps", "1",
            "--lr", "0.003", "--ckpt-every", "1", "--eval-every", "1", "--val-tokens", "2048"]
    try:
        train.run(train.parser().parse_args(args + ["--name", "straight"]))
        save = train.save_checkpoint

        def crash(*params):
            save(*params)
            if params[3] == interruption:
                raise RuntimeError("simulated crash")

        monkeypatch.setattr(train, "save_checkpoint", crash)
        with pytest.raises(RuntimeError, match="simulated crash"):
            train.run(train.parser().parse_args(args + ["--name", "resumed"]))
        monkeypatch.setattr(train, "save_checkpoint", save)
        train.run(train.parser().parse_args(args + ["--name", "resumed"]))
        checkpoints = []
        logs = []
        for name in ("straight", "resumed"):
            out = shards / "runs" / name
            paths = sorted(out.glob("checkpoint-*.pt"))
            assert len(paths) == 2 and (out / "done.json").exists()
            checkpoints.append(torch.load(paths[-1], weights_only=False))
            logs.append([json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()])
        assert [r["loss"] for r in logs[0]] == [r["loss"] for r in logs[1]]
        for key in checkpoints[0]["model"]:
            assert torch.equal(checkpoints[0]["model"][key], checkpoints[1]["model"][key]), key
        assert logs[0][0]["seq_len"] == 2048 and logs[0][0]["theta"] == 10000
        assert logs[0][-1]["seq_len"] == 8192 and logs[0][-1]["theta"] == 500000
        # Completed commands are no-ops, but changed commands must not reuse checkpoints.
        train.run(train.parser().parse_args(args + ["--name", "resumed"]))
        with pytest.raises(ValueError, match="differs"):
            train.run(train.parser().parse_args(args + ["--name", "resumed", "--lr", "0.01"]))
    finally:
        torch.set_num_threads(old_threads)
