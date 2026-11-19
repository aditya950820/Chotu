import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

from chotu.chat_data import ChatBatches, ChatExamples, ConversionError, chat_split, conversation_key, convert_chat
from chotu.chat_format import parse_tool_calls, render_conversation, tokenize_conversation
from chotu.config import TOKENIZER_ID, chotu_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((Path(__file__).with_name("chat_fixtures.json")).read_text())


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_script("train")
prep = load_script("prepare_chat")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def test_real_source_conversion():
    messages, tools, calls = convert_chat(FIXTURES["apigen"], "apigen")
    assert calls == [{"name": "use_cuda", "arguments": {}}]
    assert tools[0]["function"]["name"] == "use_cuda"
    assert "output MUST" not in messages[0]["content"]
    assert parse_tool_calls(render_conversation(messages, tools)) == (calls, [])
    assert not any(m["role"] == "tool" for m in messages)
    messages, tools, calls = convert_chat(FIXTURES["smol"], "smol")
    assert messages == FIXTURES["smol"]["messages"] and tools == calls == []
    key = conversation_key(messages, tools)
    assert key == conversation_key(messages, tools)
    keys = [f"{index:064x}" for index in range(100)]
    assert all(chat_split(key) == "val" for key in keys)
    assert chat_split("f" * 64) == "train"


@pytest.mark.parametrize("content,reason", [
    ('<tool_call>{oops}</tool_call>', "invalid_call_json"),
    ('<tool_call>{}</tool_call>', "invalid_call_shape"),
    ('<tool_call>[{"name":"use_cuda","arguments":"{}"}]</tool_call>', "invalid_call_shape"),
    ('<tool_call>[{"name":"unknown","arguments":{}}]</tool_call>', "unknown_tool"),
    ('prefix <tool_call>[]</tool_call>', "mixed_call_text"),
])
def test_bad_conversion(content, reason):
    row = copy.deepcopy(FIXTURES["apigen"])
    row["messages"][-1]["content"] = content
    with pytest.raises(ConversionError, match=reason):
        convert_chat(row, "apigen")


def test_refusal_and_tool_result():
    row = copy.deepcopy(FIXTURES["apigen"])
    row["messages"][-1]["content"] = "No applicable tool."
    assert convert_chat(row, "apigen")[2] == []
    row = {"messages": [{"role": "tool", "content": "result"}, {"role": "assistant", "content": "Done"}]}
    assert convert_chat(row, "smol")[0] == row["messages"]
    with pytest.raises(ConversionError, match="empty_messages"):
        convert_chat({}, "smol")


def test_prepared_masks_and_report(tmp_path, tokenizer):
    rows = {name: [row] for name, row in FIXTURES.items()}
    # The same live row twice exercises explicit duplicate accounting.
    rows["smol"].append(FIXTURES["smol"])
    prep.prepare(tmp_path, rows, tokenizer, {})
    counters = {"train": 0, "val": 0}
    for source, row in FIXTURES.items():
        messages, tools, _ = convert_chat(row, source)
        encoded = tokenize_conversation(messages, tokenizer, tools)
        split = chat_split(conversation_key(messages, tools))
        examples = ChatExamples(tmp_path, split)
        x, y, attention = examples.batch([counters[split]])
        assert x[0].tolist() == encoded["input_ids"][:-1]
        assert y[0].tolist() == encoded["labels"][1:]
        assert attention.all()
        counters[split] += 1
    marker = json.loads((tmp_path / "chat/.done").read_text())
    assert marker["sources"]["smol"]["dropped_by_reason"] == {"duplicate": 1}
    assert "⟦LOSS⟧" in (tmp_path / "reports/chat_report.md").read_text()
    assert marker["sources"]["apigen"]["gold_calls"] == marker["sources"]["apigen"]["valid_gold_calls"] == 1


@pytest.fixture
def small_chat(tmp_path):
    root = tmp_path / "chat"
    root.mkdir()
    for split in ("train", "val"):
        with (root / f"{split}.ids").open("wb") as ids, (root / f"{split}.mask").open("wb") as mask, (root / f"{split}.offsets").open("wb") as offsets:
            np.asarray([0], dtype=np.int64).tofile(offsets)
            offset = 0
            for i in range(8 if split == "train" else 2):
                n = 32 + 8 * i
                values = [3 + i % 3] * n
                labels = [-100] * (n // 2) + values[n // 2:]
                offset = prep.write_example([ids, mask, offsets], {"input_ids": values, "labels": labels}, offset)
    return tmp_path


def test_length_bucketing_and_padding(small_chat):
    examples = ChatExamples(small_chat, "train")
    groups = examples.microbatches(256, seed=42)
    assert sorted(i for group in groups for i in group) == list(range(8))
    assert groups == examples.microbatches(256, seed=42)
    for indices in groups:
        assert len(indices) * max(examples.lengths[i] for i in indices) <= 256
        assert max(examples.lengths[i] for i in indices) - min(examples.lengths[i] for i in indices) <= 24
        x, y, attention = examples.batch(indices)
        assert torch.all(y[attention == 0] == -100)
    batches = ChatBatches(examples, 2, 256, 256, 42)
    assert len(batches.epoch_ends) == 2 and batches.steps in batches.epoch_ends
    with pytest.raises(ValueError, match="smaller"):
        examples.microbatches(16)
    with pytest.raises(ValueError, match="1–2 epochs"):
        ChatBatches(examples, 3, 256, 256, 42)


def test_defaults_and_no_auto_stop(monkeypatch):
    args = train.parser().parse_args(["--stage", "pretrain", "--storage", "/unused", "--lr", ".003"])
    assert args.val_tokens == 1048576
    assert [train.chat_lr(s, 6, 1, 2) for s in range(6)] == [0.5, 1, 1, 2 / 3, 1 / 3, 0]
    monkeypatch.setattr(train, "load_dotenv", lambda: None)
    monkeypatch.setattr(train.sys, "argv", ["train.py", "--stage", "chat", "--storage", "/unused", "--device", "cuda", "--no-auto-stop"])
    monkeypatch.setattr(train, "run", lambda args: None)
    monkeypatch.setattr(train.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not pause"))
    assert train.main() == 0


def test_chat_resume_and_best(small_chat, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        init = small_chat / "init"
        model = Qwen3ForCausalLM(chotu_config(vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8))
        model.save_pretrained(init)
        args = ["--stage", "chat", "--model", "tiny", "--storage", str(small_chat), "--init", str(init),
                "--batch-tokens", "256", "--micro-tokens", "256", "--warmup-steps", "1",
                "--max-steps", "4", "--eval-every", "1", "--ckpt-every", "1", "--device", "cpu"]
        export = train.export_model
        exported = []

        def capture(out, model, tokenizer, name):
            exported.append(copy.deepcopy(model.state_dict()))
            export(out, model, tokenizer, name)

        monkeypatch.setattr(train, "export_model", capture)
        train.run(train.parser().parse_args(args + ["--name", "straight"]))
        save = train.save_checkpoint

        def crash(*params):
            save(*params)
            if params[3] == 2:
                raise RuntimeError("simulated crash")

        monkeypatch.setattr(train, "save_checkpoint", crash)
        with pytest.raises(RuntimeError, match="simulated crash"):
            train.run(train.parser().parse_args(args + ["--name", "resumed"]))
        monkeypatch.setattr(train, "save_checkpoint", save)
        train.run(train.parser().parse_args(args + ["--name", "resumed"]))
        logs, states = [], []
        for name in ("straight", "resumed"):
            out = small_chat / "runs" / name
            logs.append([json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()])
            states.append(torch.load(sorted(out.glob("checkpoint-*.pt"))[-1], weights_only=False))
            loaded = AutoModelForCausalLM.from_pretrained(out / "best")
            assert loaded.config.rope_parameters["rope_theta"] == 500000
            assert (out / "best/tokenizer.json").exists()
            assert states[-1]["best"] == min(row["val_loss"] for row in logs[-1])
        assert [r["loss"] for r in logs[0]] == [r["loss"] for r in logs[1]]
        for key in states[0]["model"]:
            assert torch.equal(states[0]["model"][key], states[1]["model"][key]), key
        # Best export is selected by held-out loss, including a regression.
        metrics = iter([3.0, 1.0, 2.0, 2.5])
        monkeypatch.setattr(train, "validate_chat", lambda *a: next(metrics))
        exported.clear()
        train.run(train.parser().parse_args(args + ["--name", "selection"]))
        assert len(exported) == 2
        best = AutoModelForCausalLM.from_pretrained(small_chat / "runs/selection/best")
        for key in best.state_dict():
            assert torch.equal(best.state_dict()[key], exported[1][key])
        marker = json.loads((small_chat / "runs/selection/done.json").read_text())
        assert marker["best_val_loss"] == 1.0
        # Even with a large eval interval, both complete epochs are evaluated.
        monkeypatch.setattr(train, "validate_chat", lambda *a: 1.0)
        train.run(train.parser().parse_args(args + ["--name", "epochs", "--max-steps", "100", "--eval-every", "100"]))
        epoch_logs = [json.loads(line) for line in (small_chat / "runs/epochs/metrics.jsonl").read_text().splitlines()]
        evaluated = [row for row in epoch_logs if row["val_loss"] is not None]
        assert [row["epoch"] for row in evaluated] == [1, 2]
    finally:
        torch.set_num_threads(old_threads)
