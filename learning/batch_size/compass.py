"""Lesson 4 experiment: how much do small groups of windows agree on which way to nudge the knobs?

For a model (random start, or a saved checkpoint), compute the nudge direction (gradient) suggested by
groups of 1, 2, 4 ... 256 windows, and compare each with a reference direction from 768 other windows.
Agreement is reported as an angle: 0 degrees = same direction as the reference, 90 = unrelated.

    python learning/batch_size/compass.py <data_dir> <out.json> [--ckpt path] [--groups 6]
"""
import argparse, json, math, sys, time, pathlib
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from chotu.config import tiny_config
from transformers import Qwen3ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--ckpt", default="")
ap.add_argument("--groups", type=int, default=6); ap.add_argument("--ref", type=int, default=768); ap.add_argument("--seq", type=int, default=256)
a = ap.parse_args()
torch.manual_seed(0); torch.set_num_threads(2)
model = Qwen3ForCausalLM(tiny_config())
if a.ckpt: model.load_state_dict(torch.load(a.ckpt))
model.eval()  # no dropout anyway; gradients still flow
train = np.memmap(f"{a.data}/shards/train_000.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(777)  # windows the training run never used (different seeds)
params = [p for p in model.parameters()]

def direction(n):
    """Average nudge direction suggested by n fresh windows, as one flat vector."""
    for p in params: p.grad = None
    starts = rng.integers(0, len(train) - a.seq - 1, n); tot = 0.0
    for i in range(0, n, 8):
        w = torch.tensor(np.stack([train[s:s + a.seq + 1] for s in starts[i:i + 8]]).astype(np.int64))
        loss = torch.nn.functional.cross_entropy(model(w[:, :-1]).logits.flatten(0, 1), w[:, 1:].flatten(), reduction="sum")
        (loss / (n * a.seq)).backward(); tot += loss.item()
    return torch.cat([p.grad.flatten() for p in params]).clone(), tot / (n * a.seq)

t0 = time.time()
ref, ref_loss = direction(a.ref); refn = ref.norm()
out = {"ckpt": a.ckpt or "random start", "ref_windows": a.ref, "seq": a.seq, "loss": ref_loss, "sizes": {}}
for b in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    angles = []
    for g in range(a.groups):
        d, _ = direction(b)
        cos = (d @ ref / (d.norm() * refn)).item()
        angles.append(math.degrees(math.acos(max(-1, min(1, cos)))))
    out["sizes"][b] = angles
    print(f"{b:>3} windows: angles {[round(x) for x in angles]}  {time.time()-t0:.0f}s", flush=True)
out["seconds"] = time.time() - t0
json.dump(out, open(a.out, "w"), indent=1)
