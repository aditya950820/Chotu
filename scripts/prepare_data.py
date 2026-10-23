"""Session 2: FineWeb-Edu sample-10BT -> uint16 token shards + a report.

Works on Colab (storage = Google Drive) or any machine (storage = persistent volume).
Resumable: each of the 14 source files becomes one train shard + one val shard, with a
.done marker. Rerun after a disconnect and finished files are skipped.

    python scripts/prepare_data.py                     # full run (~10B tokens)
    python scripts/prepare_data.py --max-docs 2000     # quick test, reads a few MB only

Storage comes from --storage or $CHOTU_STORAGE. Output:
    <storage>/shards/train_000.bin, val_000.bin, ...   uint16 token ids, <|endoftext|> after every doc
    <storage>/shards/000.done                          JSON counts for that file
    <storage>/reports/data_report.md                   what you read at the gate
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chotu.config import TOKENIZER_ID  # noqa: E402

REPO = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample/10BT"
EOT = 0  # <|endoftext|>
VAL_FRAC = 0.005
SHARD_BYTES_PER_TOKEN = 2  # uint16

_tok = None


def _init():
    global _tok
    _tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    _tok.model_max_length = 10**9  # we tokenize whole documents, not model inputs


def _encode(texts):
    ids = _tok(texts, add_special_tokens=False)["input_ids"]
    return [np.array(x + [EOT], dtype=np.uint16) for x in ids]


def is_val(doc_id: str) -> bool:
    """Deterministic 0.5% split by document id, so reruns give the same split."""
    h = int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < VAL_FRAC


def tokenizer_checks(tok, sample_texts):
    """The Session 2 tokenizer gate. Returns list of (name, ok, detail)."""
    out = []
    bad = sum(tok.decode(tok(t, add_special_tokens=False)["input_ids"]) != t for t in sample_texts)
    out.append(("round trip", bad == 0, f"{len(sample_texts) - bad}/{len(sample_texts)} documents identical after encode-decode"))
    d = tok.convert_ids_to_tokens(tok("2048", add_special_tokens=False)["input_ids"])
    out.append(("digits split", d == ["2", "0", "4", "8"], f"2048 -> {d}"))
    for s in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        out.append((f"{s} single token", len(ids) == 1, f"ids={ids}"))
    out.append(("end-of-text id is 0", tok.convert_tokens_to_ids("<|endoftext|>") == EOT, ""))
    out.append(("vocab fits uint16", len(tok) <= 65_536, f"vocab={len(tok):,}"))
    return out


def iter_batches(path, max_docs, local_dir):
    """Yield (ids, texts) batches. Test mode reads only the first row group remotely."""
    if max_docs:
        f = HfFileSystem().open(f"datasets/{REPO}/{path}", "rb")
        pf = pq.ParquetFile(f)
        tbl = pf.read_row_group(0, columns=["id", "text"]).slice(0, max_docs)
        yield tbl["id"].to_pylist(), tbl["text"].to_pylist()
        return
    local = hf_hub_download(REPO, path, repo_type="dataset", local_dir=local_dir)
    pf = pq.ParquetFile(local)
    for b in pf.iter_batches(batch_size=8192, columns=["id", "text"]):
        yield b.column(0).to_pylist(), b.column(1).to_pylist()
    os.remove(local)


def process_file(i, path, shard_dir, pool, max_docs, local_dir):
    counts = {"file": path, "docs": 0, "train_tokens": 0, "val_tokens": 0, "val_docs": 0}
    tmp = {k: os.path.join(shard_dir, f"{k}_{i:03d}.bin.tmp") for k in ("train", "val")}
    fh = {k: open(v, "wb") for k, v in tmp.items()}
    for ids, texts in iter_batches(path, max_docs, local_dir):
        chunks = [texts[j:j + 512] for j in range(0, len(texts), 512)]
        enc = [a for part in pool.map(_encode, chunks) for a in part]
        for doc_id, arr in zip(ids, enc):
            split = "val" if is_val(doc_id) else "train"
            arr.tofile(fh[split])
            counts[f"{split}_tokens"] += len(arr)
            counts["docs"] += 1
            counts["val_docs"] += split == "val"
    for k in fh:
        fh[k].close()
        os.replace(tmp[k], tmp[k][:-4])  # atomic: a shard exists only when complete
    with open(os.path.join(shard_dir, f"{i:03d}.done"), "w") as f:
        json.dump(counts, f)
    return counts


def write_report(storage, tok, checks, all_counts, seconds, test_mode):
    shard_dir = os.path.join(storage, "shards")
    train = sum(c["train_tokens"] for c in all_counts)
    val = sum(c["val_tokens"] for c in all_counts)
    docs = sum(c["docs"] for c in all_counts)
    disk = sum(os.path.getsize(os.path.join(shard_dir, f)) for f in os.listdir(shard_dir) if f.endswith(".bin"))
    first = os.path.join(shard_dir, "train_000.bin")
    arr = np.memmap(first, dtype=np.uint16, mode="r")
    rng = random.Random(0)
    samples = []
    for _ in range(10):
        s = rng.randrange(0, max(1, len(arr) - 300))
        samples.append(tok.decode(arr[s:s + 200].tolist()).replace("<|endoftext|>", "\n⟨END OF DOCUMENT⟩\n"))
    lines = [f"# Data report{' (TEST MODE)' if test_mode else ''}", "",
             f"- Files done: {len(all_counts)}/14",
             f"- Documents: {docs:,}",
             f"- Train tokens: {train:,} ({train / 1e9:.2f}B)",
             f"- Validation tokens: {val:,} ({val / max(1, train + val):.2%} of total)",
             f"- Shard size on disk: {disk / 1e9:.2f} GB",
             f"- Time this run: {seconds / 60:.1f} min", "", "## Tokenizer checks", ""]
    lines += [f"- [{'OK' if ok else 'FAIL'}] {n}: {d}" for n, ok, d in checks]
    lines += ["", "## 10 random samples (200 tokens each, read these)", ""]
    for k, s in enumerate(samples, 1):
        lines += [f"### Sample {k}", "", "```", s.strip(), "```", ""]
    os.makedirs(os.path.join(storage, "reports"), exist_ok=True)
    p = os.path.join(storage, "reports", "data_report.md")
    open(p, "w").write("\n".join(lines))
    return p, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", default=os.getenv("CHOTU_STORAGE"))
    ap.add_argument("--max-docs", type=int, default=0, help="test mode: docs per file")
    ap.add_argument("--max-files", type=int, default=14)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--local-dir", default="/tmp/fineweb", help="fast local disk for the 2 GB downloads")
    a = ap.parse_args()
    if not a.storage:
        sys.exit("Set --storage or $CHOTU_STORAGE (Drive folder on Colab, volume path elsewhere).")
    test = a.max_docs > 0
    if test:
        a.storage = os.path.join(a.storage, "test")
    shard_dir = os.path.join(a.storage, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tok.model_max_length = 10**9
    files = sorted(f.path for f in HfApi().list_repo_tree(REPO, path_in_repo=SUBSET, repo_type="dataset")
                   if f.path.endswith(".parquet"))[: a.max_files]
    print(f"{len(files)} source files, storage={a.storage}, workers={a.workers}")

    _, sample_texts = next(iter_batches(files[0], 200 if not test else min(200, a.max_docs), a.local_dir)) \
        if test else (None, pq.ParquetFile(HfFileSystem().open(f"datasets/{REPO}/{files[0]}", "rb"))
                      .read_row_group(0, columns=["text"]).slice(0, 200)["text"].to_pylist())
    checks = tokenizer_checks(tok, sample_texts)
    for n, ok, d in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {n}: {d}")
    if not all(ok for _, ok, _ in checks):
        sys.exit("Tokenizer checks failed: stop and report, don't work around it.")

    t0 = time.time()
    all_counts = []
    with Pool(a.workers, initializer=_init) as pool:
        for i, path in enumerate(files):
            done = os.path.join(shard_dir, f"{i:03d}.done")
            if os.path.exists(done):
                all_counts.append(json.load(open(done)))
                print(f"[{i + 1}/{len(files)}] {path}: already done, skipping")
                continue
            free = shutil.disk_usage(a.storage).free
            if not test and free < 4e9:
                sys.exit(f"Only {free / 1e9:.1f} GB free in storage; each file needs ~1.5 GB of shards. Free up space and rerun.")
            ts = time.time()
            c = process_file(i, path, shard_dir, pool, a.max_docs, a.local_dir)
            all_counts.append(c)
            tps = (c["train_tokens"] + c["val_tokens"]) / max(1e-9, time.time() - ts)
            print(f"[{i + 1}/{len(files)}] {path}: {c['docs']:,} docs, {c['train_tokens'] / 1e6:,.0f}M train tokens, {tps / 1e6:.2f}M tok/s")

    p, train = write_report(a.storage, tok, checks, all_counts, time.time() - t0, test)
    print(f"\nReport: {p}\nTrain tokens: {train / 1e9:.2f}B")


if __name__ == "__main__":
    main()
