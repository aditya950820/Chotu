"""Memory-mapped, single-pass token streams. Offsets address input tokens;
targets are the following tokens, with no target repeated within a run.
"""
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import torch


class ShardStream:
    def __init__(self, storage, split="train", seed=0):
        paths = sorted((Path(storage) / "shards").glob(f"{split}_*.bin"))
        if split == "train":
            np.random.default_rng(seed).shuffle(paths)
        self.paths = paths
        self.arrays = []
        for path in paths:
            if path.stat().st_size % 2:
                raise ValueError(f"odd-sized uint16 shard: {path}")
            if path.stat().st_size:
                self.arrays.append(np.memmap(path, dtype=np.uint16, mode="r"))
        self.ends = np.cumsum([len(a) for a in self.arrays]).tolist()
        self.size = self.ends[-1] if self.ends else 0
        if not self.size:
            raise ValueError(f"no tokens in {split} shards")

    def read(self, start, length):
        if start < 0 or length < 0 or start + length > self.size:
            raise ValueError("read outside stream")
        if not length:
            return np.empty(0, dtype=np.int64)
        pieces = []
        while length:
            index = bisect_right(self.ends, start)
            base = self.ends[index - 1] if index else 0
            count = min(length, self.ends[index] - start)
            pieces.append(self.arrays[index][start - base:start - base + count])
            start += count
            length -= count
        return np.concatenate(pieces).astype(np.int64)

    def documents(self, start, end):
        """Yield complete documents only, including their EOT, bounded-memory scan."""
        previous = start if start == 0 or self.read(start - 1, 1)[0] == 0 else 0
        for pos in range(start, end, 1_000_000):
            chunk = self.read(pos, min(1_000_000, end - pos))
            for eot in np.flatnonzero(chunk == 0):
                stop = pos + int(eot) + 1
                if previous >= start:
                    yield previous, stop
                previous = stop

    def batch(self, offsets, length):
        tokens = torch.from_numpy(np.stack([self.read(int(s), length + 1) for s in offsets]))
        return tokens[:, :-1], tokens[:, 1:]


class PretrainBatches:
    """A fixed shuffled shard order; batch(step) has no mutable cursor.

    Long windows start at documents, preferring a >=4096-token document when
    one starts within 4096 tokens of the next available start. Never overlap.
    """
    def __init__(self, stream, total_tokens, batch_tokens, long_frac=0.08,
                 short_length=2048, long_length=8192):
        if (batch_tokens <= 0 or min(short_length, long_length) <= 0
                or not 0 <= long_frac <= 1 or batch_tokens % short_length or batch_tokens % long_length):
            raise ValueError("invalid long fraction or batch not divisible by sequence lengths")
        self.stream, self.batch_tokens = stream, batch_tokens
        self.short_length, self.long_length = short_length, long_length
        budget = min(total_tokens, stream.size - 1)
        planned = budget // batch_tokens
        if planned < 1:
            raise ValueError("not enough tokens for one batch")
        long_steps = max(1, round(planned * long_frac)) if long_frac else 0
        self.switch_step = planned - long_steps
        tail = self.switch_step * batch_tokens
        docs = list(stream.documents(tail, min(stream.size, budget + 1)))
        self.tail_stats = {"tail_tokens": int(budget - tail), "complete_docs": len(docs)}
        for threshold in (2048, 8192):
            lengths = [b - a for a, b in docs if b - a >= threshold]
            self.tail_stats[f"docs_ge_{threshold}"] = len(lengths)
            self.tail_stats[f"tokens_ge_{threshold}"] = sum(lengths)
        starts, cursor = [], tail
        preferred = [(a, b) for a, b in docs if b - a >= 4096]
        doc_starts = [a for a, _ in docs]
        preferred_starts = [a for a, _ in preferred]
        while True:
            index = bisect_left(doc_starts, cursor)
            if index == len(doc_starts):
                break
            start = doc_starts[index]
            index = bisect_left(preferred_starts, start)
            if index < len(preferred_starts):
                candidate = preferred_starts[index]
                if candidate - start <= 4096 and candidate + long_length <= budget:
                    start = candidate
            if start + long_length > budget:
                break
            starts.append(start)
            cursor = start + long_length
        per_batch = batch_tokens // long_length
        self.long_offsets = starts[:long_steps * per_batch]
        actual_long_steps = len(self.long_offsets) // per_batch
        self.long_offsets = self.long_offsets[:actual_long_steps * per_batch]
        self.steps = self.switch_step + actual_long_steps
        preferred_set = {a for a, _ in preferred}
        self.tail_stats.update(planned_steps=planned, actual_steps=self.steps,
                               unused_tokens=int(budget - self.steps * batch_tokens),
                               preferred_starts=sum(s in preferred_set for s in self.long_offsets))

    def batch(self, step, micro_tokens):
        if not 0 <= step < self.steps:
            raise IndexError(step)
        length = self.short_length if step < self.switch_step else self.long_length
        if micro_tokens <= 0 or micro_tokens % length or self.batch_tokens % micro_tokens:
            raise ValueError("micro tokens must divide batch and contain whole sequences")
        if step < self.switch_step:
            offsets = range(step * self.batch_tokens, (step + 1) * self.batch_tokens, length)
        else:
            count = self.batch_tokens // length
            begin = (step - self.switch_step) * count
            offsets = self.long_offsets[begin:begin + count]
        offsets = list(offsets)
        for begin in range(0, len(offsets), micro_tokens // length):
            yield self.stream.batch(offsets[begin:begin + micro_tokens // length], length)
