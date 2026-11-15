"""Group tokens that behave alike into classes, with no gradients (Brown et al. 1992 idea, done with linear algebra):
  1. for the most common tokens, count which tokens come just before and just after them (training text only)
  2. turn counts into PPMI (how much more often than chance two tokens meet) and compress with a truncated SVD
  3. group the resulting vectors with k-means (Lloyd's algorithm: assign to nearest centre, move centres, repeat)
Rarer tokens are put in the class of the common token their own neighbour counts most resemble.

    python learning/no_gradient/clusters.py --shard <train_000.bin> --classes 512 --out classes.npy
"""
import argparse, time

import numpy as np

V = 49_152
ap = argparse.ArgumentParser()
ap.add_argument("--shard", required=True)
ap.add_argument("--tokens", type=float, default=200e6)
ap.add_argument("--classes", type=int, default=512)
ap.add_argument("--common", type=int, default=16_000, help="tokens clustered directly")
ap.add_argument("--context", type=int, default=3_000, help="neighbour tokens used as features")
ap.add_argument("--dims", type=int, default=160)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()
log = lambda *m: print(f"[{time.time() - t0:5.0f}s]", *m, flush=True)

t = np.asarray(np.memmap(a.shard, dtype=np.uint16, mode="r")[:int(a.tokens)]).astype(np.int64)
freq = np.bincount(t, minlength=V)
order = np.argsort(-freq)
ctx_ids = order[:a.context]
ctx_index = np.full(V, -1); ctx_index[ctx_ids] = np.arange(a.context)
# neighbour counts for every token: left neighbour features and right neighbour features
F = np.zeros((V, 2 * a.context), np.float32)
for side, (src, nb) in enumerate([(t[1:], t[:-1]), (t[:-1], t[1:])]):  # (token, its left / right neighbour)
    ok = ctx_index[nb] >= 0
    np.add.at(F, (src[ok], side * a.context + ctx_index[nb[ok]]), 1)
log("neighbour counts done")
# PPMI on the common tokens
common = order[:a.common]
X = F[common].astype(np.float64)
tot = X.sum()
pr, pc = X.sum(1, keepdims=True) / tot, X.sum(0, keepdims=True) / tot
with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log(np.maximum(X / tot, 1e-300) / (pr * pc))
ppmi = np.where(X > 0, np.maximum(pmi, 0), 0)
U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
E = U[:, :a.dims] * np.sqrt(S[:a.dims])
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
log("SVD done")
# k-means (Lloyd), seeded by k-means++
rng = np.random.default_rng(0)
cent = [E[rng.integers(len(E))]]
d2 = ((E - cent[0]) ** 2).sum(1)
for _ in range(a.classes - 1):
    cent.append(E[rng.choice(len(E), p=d2 / d2.sum())])
    d2 = np.minimum(d2, ((E - cent[-1]) ** 2).sum(1))
C = np.stack(cent)
for it in range(40):
    lab = (E @ C.T - 0.5 * (C ** 2).sum(1)).argmax(1)  # nearest centre
    newC = np.stack([E[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(a.classes)])
    moved = np.abs(newC - C).max()
    C = newC
    if moved < 1e-4:
        break
log(f"k-means done after {it + 1} rounds")
cls = np.full(V, -1, np.int64)
cls[common] = lab
# rarer tokens: nearest common token by their own (raw, normalised) neighbour counts; unseen tokens: one extra class
rare = order[a.common:]
seen_rare = rare[freq[rare] > 0]
if len(seen_rare):
    Xr = F[seen_rare].astype(np.float64)
    Xr /= np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-9
    Xc = F[common].astype(np.float64)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9
    for i in range(0, len(seen_rare), 4096):
        cls[seen_rare[i:i + 4096]] = lab[(Xr[i:i + 4096] @ Xc.T).argmax(1)]
cls[cls < 0] = a.classes
np.save(a.out, cls.astype(np.int32))
sizes = np.bincount(cls, minlength=a.classes + 1)
log(f"saved {a.out}: {a.classes} classes (+1 for unseen), sizes min {sizes[:-1].min()} median {int(np.median(sizes[:-1]))} max {sizes.max()}")
