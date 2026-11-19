"""Stream real chat sources into flat IDs/masks/offsets on persistent storage."""
import argparse
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import sys
import time

from datasets import load_dataset
from huggingface_hub import HfApi, HfFileSystem
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chotu.chat_data import ConversionError, chat_split, conversation_key, convert_chat
from chotu.chat_format import parse_tool_calls, render_conversation, tokenize_conversation
from chotu.config import TOKENIZER_ID

SOURCES = {"smol": ("HuggingFaceTB/smol-smoltalk", "default"),
           "apigen": ("HuggingFaceTB/smoltalk", "apigen-80k")}


def write_example(files, encoded, offset):
    ids = np.asarray(encoded["input_ids"], dtype=np.uint16)
    mask = np.asarray(encoded["labels"]) != -100
    ids.tofile(files[0])
    mask.astype(np.uint8).tofile(files[1])
    offset += len(ids)
    np.asarray([offset], dtype=np.int64).tofile(files[2])
    return offset


def marked_sample(encoded, tokenizer):
    ids, labels = encoded["input_ids"], encoded["labels"]
    parts, start = [], 0
    for end in range(1, len(ids) + 1):
        if end == len(ids) or (labels[end] == -100) != (labels[start] == -100):
            text = tokenizer.decode(ids[start:end], clean_up_tokenization_spaces=False)
            parts.append(text if labels[start] == -100 else "⟦LOSS⟧" + text + "⟦/LOSS⟧")
            start = end
    return "".join(parts)


def prepare(storage, rows_by_source, tokenizer, metadata, limit=None):
    root = Path(storage) / "chat"
    root.mkdir(parents=True, exist_ok=True)
    files, offsets = {}, {"train": 0, "val": 0}
    counts, samples = {}, {}
    seen = set()
    clock = time.monotonic()
    try:
        for split in offsets:
            files[split] = [(root / f"{split}.{suffix}.tmp").open("wb") for suffix in ("ids", "mask", "offsets")]
            np.asarray([0], dtype=np.int64).tofile(files[split][2])
        with (root / "apigen-heldout.jsonl.tmp").open("w") as heldout:
            for source, rows in rows_by_source.items():
                stats = dict(raw=0, converted=0, kept=0, over_length=0, dropped_by_reason=Counter(),
                             train=0, val=0, assistant_tokens=0, gold_calls=0, valid_gold_calls=0)
                lengths, samples[source] = [], []
                start_time = time.monotonic()
                for row in islice(rows, limit):
                    stats["raw"] += 1
                    try:
                        messages, tools, calls = convert_chat(row, source)
                        stats["converted"] += 1
                        key = conversation_key(messages, tools)
                        if key in seen:
                            raise ConversionError("duplicate")
                        encoded = tokenize_conversation(messages, tokenizer, tools)
                        if encoded is None:
                            stats["over_length"] += 1
                            continue
                        if not any(label != -100 for label in encoded["labels"][1:]):
                            raise ConversionError("no_supervised_targets")
                        parsed, errors = parse_tool_calls(render_conversation([
                            {"role": "assistant", "tool_calls": calls}]))
                        if errors or parsed != calls:
                            raise ConversionError("gold_roundtrip")
                        stats["gold_calls"] += len(calls)
                        stats["valid_gold_calls"] += len(parsed)
                        split = chat_split(key)
                        offsets[split] = write_example(files[split], encoded, offsets[split])
                        seen.add(key)
                        stats[split] += 1
                        stats["kept"] += 1
                        stats["assistant_tokens"] += sum(label != -100 for label in encoded["labels"])
                        lengths.append(len(encoded["input_ids"]))
                        if len(samples[source]) < 5:
                            samples[source].append(marked_sample(encoded, tokenizer))
                        if source == "apigen" and split == "val":
                            heldout.write(json.dumps(dict(id=key, raw=row,
                                prompt_messages=messages[:-1], tools=tools, gold_calls=calls,
                                gold_text=messages[-1].get("content", "")), ensure_ascii=False) + "\n")
                    except ConversionError as error:
                        stats["dropped_by_reason"][str(error)] += 1
                    if stats["raw"] % 250 == 0:
                        print(f"{source}: {stats['raw']} raw, {stats['kept']} kept", flush=True)
                stats["seconds"] = time.monotonic() - start_time
                stats["length_percentiles"] = dict(zip(["p0", "p50", "p90", "p95", "p99", "p100"],
                    np.percentile(lengths, [0, 50, 90, 95, 99, 100]).tolist())) if lengths else {}
                stats["estimated_full_seconds"] = stats["seconds"] * metadata.get(source, {}).get("rows", stats["raw"]) / max(1, stats["raw"])
                counts[source] = stats
    finally:
        for handles in files.values():
            for handle in handles:
                handle.close()
    total_conversations = sum(s["kept"] for s in counts.values())
    total_assistant = sum(s["assistant_tokens"] for s in counts.values())
    for stats in counts.values():
        stats["conversation_share"] = stats["kept"] / max(1, total_conversations)
        stats["assistant_token_share"] = stats["assistant_tokens"] / max(1, total_assistant)
    report = ["# Chat preparation report", "", "Live data only. ⟦LOSS⟧ marks supervised tokens, including assistant end markers.",
              "Hash-held-out 1% per source; duplicates removed across sources. No resampling/reweighting.",
              "APIGen embeds tools in the system text and a JSON call list in one tag. Convert to one tag per call.",
              "Raw APIGen has system/user/assistant only: no tool-result turns to invent.", "", "## Counts and timings", "", "```json",
              json.dumps(dict(metadata=metadata, sources=counts, total_seconds=time.monotonic() - clock), indent=2), "```"]
    gold = sum(s["gold_calls"] for s in counts.values())
    valid = sum(s["valid_gold_calls"] for s in counts.values())
    report += ["", f"Gold-call JSON roundtrip validity: {valid}/{gold} ({valid / gold:.2%})." if gold else "No gold calls in this selection."]
    for source, examples in samples.items():
        report += ["", f"## {source}: five rendered samples (or all if fewer)"]
        for index, sample in enumerate(examples, 1):
            report += ["", f"### {index}", "", "```text", sample, "```"]
    reports = Path(storage) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "chat_report.md").write_text("\n".join(report) + "\n")
    for path in root.glob("*.tmp"):
        os.replace(path, path.with_suffix(""))
    marker = dict(limit=limit, tokenizer=TOKENIZER_ID, metadata=metadata, sources=counts)
    (root / ".done.tmp").write_text(json.dumps(marker, indent=2))
    os.replace(root / ".done.tmp", root / ".done")
    print(json.dumps(marker, indent=2), flush=True)
    return marker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--storage", required=True)
    ap.add_argument("--limit", type=int, help="first N train rows per source; use separate smoke storage")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("limit must be positive")
    storage = Path(args.storage).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    marker = storage / "chat/.done"
    if marker.exists():
        if json.loads(marker.read_text())["limit"] != args.limit:
            ap.error("existing preparation has a different limit; use a fresh storage directory")
        print("Chat preparation already complete")
        return
    metadata, rows = {}, {}
    for source, (repo, config) in SOURCES.items():
        sha = HfApi().dataset_info(repo).sha
        prefix = "data/apigen-80k/" if source == "apigen" else "data/"
        paths = [f for f in HfApi().list_repo_tree(repo, recursive=True, repo_type="dataset", revision=sha)
                 if f.path.startswith(prefix + "train-") and f.path.endswith(".parquet")]
        full_rows = sum(pq.ParquetFile(HfFileSystem().open(f"datasets/{repo}@{sha}/{f.path}", "rb")).metadata.num_rows for f in paths)
        metadata[source] = dict(repo=repo, config=config, revision=sha, rows=full_rows, parquet_bytes=sum(f.size for f in paths))
        rows[source] = load_dataset(repo, config, split="train", streaming=True, revision=sha,
                                    cache_dir=str(storage / "chat-cache"))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, cache_dir=storage / "tokenizer-cache")
    tokenizer.model_max_length = 10**9
    prepare(storage, rows, tokenizer, metadata, args.limit)


if __name__ == "__main__":
    main()
