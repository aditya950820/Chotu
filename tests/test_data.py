"""Shard format: uint16 ids, one <|endoftext|> (0) after every document, decodes back to the text."""
import importlib.util
import os

import numpy as np

spec = importlib.util.spec_from_file_location("prep", os.path.join(os.path.dirname(__file__), "..", "scripts", "prepare_data.py"))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_shard_roundtrip(tmp_path):
    prep._init()
    docs = ["The cat sat on the mat.", "Photosynthesis needs light.", "2048 is a number."]
    arrs = prep._encode(docs)
    path = tmp_path / "s.bin"
    with open(path, "wb") as f:
        for a in arrs:
            a.tofile(f)
    ids = np.fromfile(path, dtype=np.uint16)
    assert (ids == prep.EOT).sum() == len(docs)
    parts = np.split(ids, np.where(ids == prep.EOT)[0] + 1)[:-1]
    assert [prep._tok.decode(p[:-1].tolist()) for p in parts] == docs


def test_val_split_is_deterministic_and_small():
    ids = [f"<urn:uuid:{i}>" for i in range(200_000)]
    frac = sum(map(prep.is_val, ids)) / len(ids)
    assert 0.004 < frac < 0.006
    assert prep.is_val(ids[123]) == prep.is_val(ids[123])
