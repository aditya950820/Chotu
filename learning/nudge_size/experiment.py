"""Lesson 3 experiment: the same tiny Chotu twin, the same batches, different nudge sizes (learning rates).

Real training on CPU: tiny_config (7,079,552 knobs), real train shard for practice, real val shard for the exam.
Writes one JSON per run with per-step loss, gradient size, update size and periodic validation loss.

    python learning/nudge_size/experiment.py <data_dir> <out_dir> <name> <lr> [--steps N] [--warmup N] [--decay FRAC]
"""
import argparse, json, math, sys, time, pathlib
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from chotu.config import tiny_config
from transformers import Qwen3ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("name"); ap.add_argument("lr", type=float)
ap.add_argument("--steps", type=int, default=300); ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--decay", type=float, default=0.0, help="fraction of steps at the end where the nudge shrinks to 0")
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seq", type=int, default=256)
ap.add_argument("--eval-every-windows", type=int, default=0, help="exam every N windows read (0 = every 25 steps)")
ap.add_argument("--micro", type=int, default=0, help="read the batch in rounds of this many windows (gradient accumulation); 0 = all at once")
ap.add_argument("--save", default="", help="save the trained knobs here at the end")
a = ap.parse_args()
torch.manual_seed(0); torch.set_num_threads(2)
train = np.memmap(f"{a.data}/shards/train_000.bin", dtype=np.uint16, mode="r")
val = np.memmap(f"{a.data}/shards/val_013.bin", dtype=np.uint16, mode="r")

def batch(arr, step, n):  # the same windows for every run: position depends only on the step number
    rng = np.random.default_rng(1000 + step)
    starts = rng.integers(0, len(arr) - a.seq - 1, n)
    w = torch.tensor(np.stack([arr[s:s + a.seq + 1] for s in starts]).astype(np.int64))
    return w[:, :-1], w[:, 1:]

vx, vy = batch(val, -1, 32)  # fixed exam: 32 windows the model never trains on
model = Qwen3ForCausalLM(tiny_config())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)

def lr_at(s):  # warm up, hold, then (optionally) slow down: the same shape as Chotu's plan
    if s < a.warmup: return a.lr * (s + 1) / a.warmup
    d0 = int(a.steps * (1 - a.decay))
    if a.decay and s >= d0: return a.lr * (a.steps - s) / (a.steps - d0)
    return a.lr

def val_loss():
    model.eval()
    with torch.no_grad():
        l = sum(torch.nn.functional.cross_entropy(model(vx[i:i+8]).logits.flatten(0, 1), vy[i:i+8].flatten()).item() for i in range(0, 32, 8)) / 4
    model.train(); return l

log = {"name": a.name, "lr": a.lr, "warmup": a.warmup, "decay": a.decay, "steps": a.steps,
       "tokens_per_step": a.batch * a.seq, "params": sum(p.numel() for p in model.parameters()),
       "loss": [], "grad": [], "update": [], "lr_sched": [], "val": [[0, val_loss()]]}
t0 = time.time()
for s in range(a.steps):
    for g in opt.param_groups: g["lr"] = lr_at(s)
    x, y = batch(train, s, a.batch)
    opt.zero_grad(); m = a.micro or a.batch; loss = 0.0
    for i in range(0, a.batch, m):  # rounds of m windows; adding up their nudges = one big batch
        part = torch.nn.functional.cross_entropy(model(x[i:i+m]).logits.flatten(0, 1), y[i:i+m].flatten()) * len(x[i:i+m]) / a.batch
        part.backward(); loss = loss + part.detach()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    before = [p.detach().clone() for p in model.parameters()]
    opt.step()
    with torch.no_grad():  # how far the knobs moved this step, relative to their size
        num = sum((p - b).pow(2).sum() for p, b in zip(model.parameters(), before)).sqrt()
        den = sum(b.pow(2).sum() for b in before).sqrt()
    l = loss.item()
    log["loss"].append(l if math.isfinite(l) else None); log["grad"].append(gn if math.isfinite(gn) else None)
    log["update"].append((num / den).item() if math.isfinite(l) else None); log["lr_sched"].append(lr_at(s))
    ew = a.eval_every_windows
    if ((s + 1) * a.batch % ew == 0) if ew else ((s + 1) % 25 == 0 or s + 1 == a.steps):
        log["val"].append([s + 1, val_loss() if math.isfinite(l) else None])
        print(f"{a.name} step {s+1} loss {l:.3f} grad {gn:.2f} {time.time()-t0:.0f}s", flush=True)
    if not math.isfinite(l):
        log["blew_up_at"] = s + 1; print(a.name, "NaN at step", s + 1); break
log["seconds"] = time.time() - t0; log["batch"] = a.batch
if a.save: torch.save(model.state_dict(), a.save)
json.dump(log, open(f"{a.out}/{a.name}.json", "w"))
