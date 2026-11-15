"""Weight-space editing: improve the 7M twin by changing its weights directly, with no gradients.

Every edit is judged on a TUNING set (windows from val_001), and only the final choices are scored once on the
LOCKED EXAM (the same 96 windows from val_000 used in learning/twin_5x/analyze.py), so the exam is never tuned on.
Edits tried:
  1 blend      mix the 140M and 700M twins, and step past the 700M twin in the "more reading" direction
  2 stretch    move the 700M twin further from (or back towards) its random start
  3 swap       word list from one twin, thinking layers from the other
  4 confidence scale the final norm (the same as making every guess sharper or softer)
  5 simplify   drop each weight matrix's weakest directions (truncated SVD), keep only changes that help
  6 volume     scale how loudly each layer writes into the running signal (attention out, MLP out)
  7 snapshots  average the 140M run's last saved checkpoints
then combines the edits that helped and scores the result on the locked exam.

    python learning/weight_edit/edit.py --dir <folder with twin-day/... and shards/val_00*.bin>
"""
import argparse, copy, json, math, pathlib, sys, time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chotu.config import tiny_config  # noqa: E402
from transformers import Qwen3ForCausalLM  # noqa: E402

BYTES_PER_TOKEN, SEQ = 4.54, 2048
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--tune-windows", type=int, default=16)
ap.add_argument("--exam-windows", type=int, default=96)
ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("results.json")))
a = ap.parse_args()
D = pathlib.Path(a.dir)
t0 = time.time()
LOG = []


def log(*m):
    print(f"[{time.time() - t0:6.0f}s]", *m, flush=True)


def windows(path, n, seed):
    sh = np.memmap(path, dtype=np.uint16, mode="r")
    st = np.sort(np.random.default_rng(seed).choice(len(sh) - SEQ - 1, n, replace=False))
    return torch.from_numpy(np.stack([sh[s:s + SEQ + 1] for s in st]).astype(np.int64))


TUNE = windows(D / "shards/val_001.bin", a.tune_windows, 2024)
EXAM = windows(D / "shards/val_000.bin", a.exam_windows, 999)  # exactly the analysis exam


def load(path=None):
    torch.manual_seed(1234)
    m = Qwen3ForCausalLM(tiny_config())
    if path:
        sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
        m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})
    return m.eval()


@torch.inference_mode()
def score(m, W):
    tot = 0.0
    for i in range(0, len(W), 4):
        x = W[i:i + 4]
        tot += F.cross_entropy(m(x[:, :-1]).logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="sum").item()
    return tot / (len(W) * SEQ) / math.log(2) / BYTES_PER_TOKEN  # bits per byte


def sd(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def with_sd(state):
    m = load()
    m.load_state_dict(state)
    return m


def tune(label, state, group, **info):
    s = score(with_sd(state), TUNE)
    LOG.append(dict(group=group, label=label, tune=round(s, 5), **info))
    log(f"{group:10s} {label:42s} tune {s:.4f}")
    return s


init, A, B = sd(load()), sd(load(D / "twin-day/main/ckpt/step_002136.pt")), sd(load(D / "twin-day/main-5x/ckpt/step_010681.pt"))
keys = list(B)
base = {"140M": tune("140M twin as trained", A, "baseline"), "700M": tune("700M twin as trained", B, "baseline")}
mix = lambda P, Q, t: {k: (1 - t) * P[k] + t * Q[k] for k in keys}

# 1 blend: t=0 is the 140M twin, t=1 the 700M twin, t>1 steps past it
for t in [0.25, 0.5, 0.75, 1.25, 1.5]:
    tune(f"blend t={t}", mix(A, B, t), "blend", t=t)

# 2 stretch: the 700M twin's distance from its random start, times f
for f_ in [0.9, 0.95, 1.05, 1.1, 1.2]:
    tune(f"stretch x{f_}", {k: init[k] + f_ * (B[k] - init[k]) for k in keys}, "stretch", f=f_)

# 3 swap parts
is_emb = lambda k: "embed_tokens" in k
tune("word list 140M + layers 700M", {k: (A if is_emb(k) else B)[k] for k in keys}, "swap")
tune("word list 700M + layers 140M", {k: (B if is_emb(k) else A)[k] for k in keys}, "swap")

# 4 confidence: final norm scale (logits are linear in it, so this is exactly a temperature)
best_conf, best_c = base["700M"], 1.0
for c in [0.9, 0.95, 1.05, 1.1]:
    st = dict(B); st["model.norm.weight"] = B["model.norm.weight"] * c
    s = tune(f"confidence x{c}", st, "confidence", c=c)
    if s < best_conf:
        best_conf, best_c = s, c

# 7 snapshots: average the 140M run's last checkpoints (all in its slow-down phase)
snaps = [sd(load(D / f"twin-day/main/ckpt/step_{s:06d}.pt")) for s in (1750, 2000, 2136)]
tune("140M: average of steps 1750, 2000, 2136", {k: sum(s[k] for s in snaps) / 3 for k in keys}, "snapshots")
tune("140M: average of steps 2000, 2136", {k: (snaps[1][k] + snaps[2][k]) / 2 for k in keys}, "snapshots")

# 5 simplify: per matrix, keep only the strongest directions; greedy, keep a change only if the tune score improves
cur, cur_s = dict(B), base["700M"]
mats = [k for k in keys if B[k].dim() == 2 and "embed" not in k]
simplify_kept = []
for k in mats:
    U, S, Vh = torch.linalg.svd(B[k].float(), full_matrices=False)
    best = None
    for keep in [0.9, 0.75]:
        r = max(1, int(len(S) * keep))
        st = dict(cur); st[k] = (U[:, :r] * S[:r]) @ Vh[:r]
        s = tune(f"simplify {k.replace('model.layers.', 'L').replace('.weight', '')} keep {int(keep * 100)}%", st, "simplify", matrix=k, keep=keep)
        if s < cur_s - 1e-4 and (best is None or s < best[0]):
            best = (s, st, keep)
    if best:
        cur_s, cur = best[0], best[1]
        simplify_kept.append(dict(matrix=k, keep=best[2], tune=round(cur_s, 5)))
log("simplify kept", len(simplify_kept), "changes; tune", round(cur_s, 4))
simplified = cur

# 6 volume: scale each layer's attention-out and MLP-out (coordinate search, two rounds), starting from the simplified twin
vol, vol_s = dict(simplified), cur_s
vol_kept = []
for rnd in (1, 2):
    for l in range(4):
        for part in ("self_attn.o_proj", "mlp.down_proj"):
            k = f"model.layers.{l}.{part}.weight"
            for f_ in (0.9, 1.1):
                st = dict(vol); st[k] = vol[k] * f_
                s = tune(f"volume round {rnd} L{l} {part.split('.')[0]} x{f_}", st, "volume", layer=l, part=part, f=f_, round=rnd)
                if s < vol_s - 1e-4:
                    vol, vol_s = st, s
                    vol_kept.append(dict(layer=l, part=part, f=f_, round=rnd, tune=round(s, 5)))
log("volume kept", len(vol_kept), "changes; tune", round(vol_s, 4))

# combine: simplified + volume, then re-check the confidence dial on top
final, final_s, final_c = vol, vol_s, 1.0
for c in [0.95, 0.97, 1.03, 1.05]:
    st = dict(vol); st["model.norm.weight"] = vol["model.norm.weight"] * c
    s = tune(f"combined + confidence x{c}", st, "combined", c=c)
    if s < final_s - 1e-4:
        final, final_s, final_c = st, s, c

# the only exam scores: baselines, each single best edit, and the combination
exam = {}
for label, st in [("140M twin as trained", A), ("700M twin as trained", B), ("700M + simplify", simplified), ("700M + simplify + volume", vol),
                  ("700M + all edits", final), ("blend t=1.25 (past 700M)", mix(A, B, 1.25))]:
    exam[label] = round(score(with_sd(st), EXAM), 5)
    log(f"EXAM {label:32s} {exam[label]:.4f}")
torch.save({"model": final, "edits": dict(simplify=simplify_kept, volume=vol_kept, confidence=final_c)}, D / "twin700_edited.pt")
json.dump(dict(meta=dict(tune_windows=a.tune_windows, exam_windows=a.exam_windows, tune_tokens=a.tune_windows * SEQ, exam_tokens=a.exam_windows * SEQ,
                         minutes=round((time.time() - t0) / 60, 1)),
               baseline=base, trials=LOG, simplify_kept=simplify_kept, volume_kept=vol_kept, final_confidence=final_c,
               tune_final=round(final_s, 5), exam=exam), open(a.out, "w"), indent=1)
log("wrote", a.out)
