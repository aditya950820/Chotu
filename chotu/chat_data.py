"""Dataset normalization and padded, length-bucketed SFT batches."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch


class ConversionError(ValueError):
    pass


def convert_chat(row, source):
    messages = [dict(m) for m in row.get("messages", [])]
    if not messages:
        raise ConversionError("empty_messages")
    tools = []
    if source == "apigen":
        if messages[0].get("role") != "system":
            raise ConversionError("missing_system")
        found = re.search(r"<tools>(.*?)</tools>", messages[0].get("content", ""), re.DOTALL)
        if not found:
            raise ConversionError("missing_tools")
        try:
            tools = json.loads(found.group(1))
        except ValueError:
            raise ConversionError("invalid_tools_json") from None
        if not isinstance(tools, list):
            raise ConversionError("invalid_tools_shape")
        # The source system requires a list inside one tool_call block. Replace
        # that incompatible formatting instruction, retaining the user request.
        messages[0]["content"] = "Use the available tools when appropriate. If none apply or parameters are missing, explain why."
    names = set()
    for tool in tools:
        function = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ConversionError("invalid_tool_schema")
        names.add(function["name"])
    calls = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            raise ConversionError("invalid_message")
        if source == "apigen" and role == "assistant" and "<tool_call>" in content:
            match = re.fullmatch(r"\s*<tool_call>(.*?)</tool_call>\s*", content, re.DOTALL)
            if not match:
                raise ConversionError("mixed_call_text")
            try:
                parsed = json.loads(match.group(1))
            except ValueError:
                raise ConversionError("invalid_call_json") from None
            if not isinstance(parsed, list):
                raise ConversionError("invalid_call_shape")
            normalized = []
            for call in parsed:
                if (not isinstance(call, dict) or not isinstance(call.get("name"), str)
                        or not isinstance(call.get("arguments"), dict)):
                    raise ConversionError("invalid_call_shape")
                if call["name"] not in names:
                    raise ConversionError("unknown_tool")
                normalized.append({"name": call["name"], "arguments": call["arguments"]})
            message["content"] = ""
            message["tool_calls"] = normalized
            calls.extend(normalized)
    if not any(m["role"] == "assistant" for m in messages):
        raise ConversionError("no_assistant")
    return messages, tools, calls


def conversation_key(messages, tools):
    canonical = json.dumps([messages, tools], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def chat_split(key):
    return "val" if int(key[:8], 16) / 2**32 < 0.01 else "train"


class ChatExamples:
    def __init__(self, storage, split):
        root = Path(storage) / "chat"
        self.paths = [root / f"{split}.{suffix}" for suffix in ("ids", "mask", "offsets")]
        self.ids = np.memmap(self.paths[0], dtype=np.uint16, mode="r") if self.paths[0].stat().st_size else np.empty(0, dtype=np.uint16)
        self.mask = np.memmap(self.paths[1], dtype=np.uint8, mode="r") if self.paths[1].stat().st_size else np.empty(0, dtype=np.uint8)
        self.offsets = np.memmap(self.paths[2], dtype=np.int64, mode="r")
        if (len(self.ids) != len(self.mask) or self.offsets[0] != 0
                or self.offsets[-1] != len(self.ids) or np.any(np.diff(self.offsets) < 2)):
            raise ValueError("invalid chat arrays")
        self.lengths = np.diff(self.offsets)

    def __len__(self):
        return len(self.lengths)

    def batch(self, indices):
        length = max(int(self.lengths[i]) for i in indices)
        ids = torch.zeros((len(indices), length), dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids)
        for row, index in enumerate(indices):
            a, b = self.offsets[index:index + 2]
            n = int(b - a)
            values = torch.from_numpy(self.ids[a:b].astype(np.int64))
            mask = torch.from_numpy(self.mask[a:b].astype(bool))
            ids[row, :n] = values
            labels[row, :n] = torch.where(mask, values, -100)
            attention[row, :n] = 1
        return ids[:, :-1], labels[:, 1:], attention[:, :-1]

    def microbatches(self, token_budget, seed=None):
        if len(self) == 0:
            raise ValueError("empty chat split")
        if self.lengths.max() > token_budget:
            raise ValueError("micro token budget smaller than longest conversation")
        order = np.argsort(self.lengths, kind="stable")
        batches, batch = [], []
        for index in order:
            if batch and (len(batch) + 1) * self.lengths[index] > token_budget:
                batches.append(batch)
                batch = []
            batch.append(int(index))
        if batch:
            batches.append(batch)
        if seed is not None:
            np.random.default_rng(seed).shuffle(batches)
        return batches


class ChatBatches:
    def __init__(self, examples, epochs, batch_tokens, micro_tokens, seed, max_steps=None):
        if not 1 <= epochs <= 2 or micro_tokens <= 0 or batch_tokens < micro_tokens:
            raise ValueError("chat requires 1–2 epochs and batch >= positive micro tokens")
        self.examples, self.updates, self.epoch_ends = examples, [], set()
        for epoch in range(epochs):
            update, used = [], 0
            for batch in examples.microbatches(micro_tokens, seed + epoch):
                padded = len(batch) * max(examples.lengths[i] for i in batch)
                if update and used + padded > batch_tokens:
                    self.updates.append((epoch, update))
                    update, used = [], 0
                update.append(batch)
                used += padded
            if update:
                self.updates.append((epoch, update))
            self.epoch_ends.add(len(self.updates))
        if max_steps is not None:
            if max_steps < 1:
                raise ValueError("max steps must be positive")
            self.updates = self.updates[:max_steps]
        self.steps = len(self.updates)

    def batch(self, step, micro_tokens):
        for indices in self.updates[step][1]:
            yield self.examples.batch(indices)
