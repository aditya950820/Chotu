"""Hypothesis A: multiply the counting model by a 'soft context' part whose weights are COMPUTED from counts.

Soft context (one network layer, written from statistics, no training):
  for each distance band b (1, 2, 3, 4, 5-8, 9-32, 33-256 tokens back), the smoothed log odds
      L_b[x, w] = log( (c_b(x, w) + k p(w)) / ((c_b(x) + k) p(w)) )
  of target w given token x somewhere in that band, compressed by a randomized SVD:  L_b ~ U_b V_b^T.
  At a position, h_b = average of U_b over the context tokens in band b; score s(w) = sum_b beta_b h_b . V_b[w].
Combination (product of experts, Rosenfeld 1996 style, exact normalisation over all 49,152 tokens):
      p(w) = p_mix(w) exp(s(w)) / Z,   p_mix = EM mix of Kneser-Ney (orders 1-4), document memory and copy
The betas are picked by trying values on tuning windows; the exam windows are scored once.

    python learning/no_gradient/poe.py --dir <folder with shards/> --train-tokens 300e6
"""
import argparse, json, math, pathlib, sys, time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import kn  # noqa: E402

V, SEQ = kn.V, kn.SEQ
BANDS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 32), (33, 256)]
# wide bands: every STEP-th distance on the first TOK tokens (far context is about topic; this keeps ~1B pairs per band)
BAND_STEP = {(5, 8): 1, (9, 32): 2, (33, 256): 8}
BAND_TOKENS = {(9, 32): 100e6, (33, 256): 40e6}
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--train-tokens", type=float, default=300e6)
ap.add_argument("--ctx-vocab", type=int, default=8192, help="context tokens with their own row (the rest share one)")
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--kappa", type=float, default=50.0, help="smoothing towards 'no effect'")
ap.add_argument("--tune-windows", type=int, default=8)
ap.add_argument("--exam-windows", type=int, default=24)
ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("poe_results.json")))
a = ap.parse_args()
D_ = pathlib.Path(a.dir)
t0 = time.time()
log = lambda *m: print(f"[{time.time() - t0:6.0f}s]", *m, flush=True)
N = 4  # Kneser-Ney orders with exactly stored follower lists

train = D_ / "shards/train_000.bin"
T = np.asarray(np.memmap(train, dtype=np.uint16, mode="r")[:int(a.train_tokens)]).astype(np.int64)
freq = np.bincount(T, minlength=V).astype(np.float64)
p_uni = (freq + 0.5) / (freq + 0.5).sum()
order = np.argsort(-freq)
row = np.full(V, a.ctx_vocab, np.int64)  # rare context tokens share the last row
row[order[:a.ctx_vocab]] = np.arange(a.ctx_vocab)
R = a.ctx_vocab + 1

# ---------------------------------------------------------------- 1. soft-context tables from counts
rng = np.random.default_rng(0)
U, Vt = [], []
for lo, hi in BANDS:
    C = np.zeros(R * V, np.float32)
    n_tok = int(min(len(T), BAND_TOKENS.get((lo, hi), len(T))))
    for d in range(lo, hi + 1, BAND_STEP.get((lo, hi), 1)):  # (context token d back, target) pairs in the band
        for s in range(0, n_tok - d, 50_000_000):
            e = min(n_tok - d, s + 50_000_000)
            C += np.bincount(row[T[s:e]] * V + T[s + d:e + d], minlength=R * V).astype(np.float32)
    C = C.reshape(R, V)
    cx = C.sum(1, keepdims=True)
    L = np.log((C + a.kappa * p_uni[None, :].astype(np.float32)) / ((cx + a.kappa) * p_uni[None, :].astype(np.float32)))
    del C
    # randomized SVD (Halko et al. 2011): L ~ Q B, B = small, then exact SVD of B
    Om = rng.standard_normal((V, a.rank + 16)).astype(np.float32)
    Y = L @ Om
    for _ in range(2):  # power iterations sharpen the top directions
        Y = L @ (L.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ L
    Ub, Sb, Vb = np.linalg.svd(B, full_matrices=False)
    U.append((Q @ Ub[:, :a.rank]) * Sb[:a.rank])  # rows: context token -> vector
    Vt.append(Vb[:a.rank].T.copy())               # rows: target token -> vector
    kept = float((Sb[:a.rank] ** 2).sum() / (L ** 2).sum())
    del L, Y, B, Q
    log(f"band {lo}-{hi}: rank {a.rank} keeps {kept:.1%} of the table")
Vcat = np.concatenate(Vt, 1).astype(np.float32)  # V x (bands * rank)

# ---------------------------------------------------------------- 2. counting model (KN 1-4), document memory, copy
tune = kn.Positions(kn.windows(D_ / "shards/val_001.bin", a.tune_windows, 2024), N)
exam = kn.Positions(kn.windows(D_ / "shards/val_000.bin", 96, 999)[:a.exam_windows], N)  # first exam windows
sub = D_ / "shards/_poe_train.bin"
T.astype(np.uint16).tofile(sub)
cont, top, _ = kn.count([sub], [tune, exam], N, log=lambda *m: None)
S = kn.finalize(cont, top, N)
del cont, top
Dk = {k: kn.standard_discounts(S[k]["coc"]) for k in S}
p1 = np.full(V, 1.0 / V)  # dense order-1 (continuation unigram) distribution
st, (d1, d2, d3) = S[1], Dk[1]
c1 = np.zeros(V); c1[(st["hw"] & np.uint64(0xFFFF)).astype(np.int64)] = st["c"]
tot1 = st["tot"][0]
dd = np.where(c1 == 0, 0, np.where(c1 == 1, d1, np.where(c1 == 2, d2, d3)))
p1 = np.maximum(c1 - dd, 0) / tot1 + (d1 * st["n1"][0] + d2 * st["n2"][0] + d3 * st["n3"][0]) / tot1 / V
log("counting model ready")


def kn_sparse(pos, q):
    """p_KN(.) at position q, exactly, as  G * p1  +  sum over seen orders k of  scale_k * max(c - D, 0) / tot  on that
    context's follower list. Returns G and the list locations (k, lo, hi, scale, tot) - no copies of the lists."""
    G, segs = 1.0, []
    for k in range(2, N + 1):
        if pos.avail[q] < k - 1:
            break
        stk, (e1, e2, e3) = S[k], Dk[k]
        h = pos.ctx[k][q]
        i = np.searchsorted(stk["h"], h)
        if i >= len(stk["h"]) or stk["h"][i] != h:
            continue
        tot = stk["tot"][i]
        gam = (e1 * stk["n1"][i] + e2 * stk["n2"][i] + e3 * stk["n3"][i]) / tot
        base = np.uint64(int(h) << 16)
        lo_ = int(np.searchsorted(stk["hw"], base, side="left"))
        hi_ = int(np.searchsorted(stk["hw"], base + np.uint64(0xFFFF), side="right"))
        G *= gam
        segs = [(kk, l, hh, sc * gam, tt) for kk, l, hh, sc, tt in segs]
        segs.append((k, lo_, hi_, 1.0, tot))
    return G, segs


def seg_dot(segs, r_):
    """sum over the sparse part of p_KN(w) * r_(w)"""
    z = 0.0
    for k, lo_, hi_, sc, tot in segs:
        e1, e2, e3 = Dk[k]
        cs = S[k]["c"][lo_:hi_].astype(np.float64)
        ws = (S[k]["hw"][lo_:hi_] & np.uint64(0xFFFF)).astype(np.int64)
        dk = np.where(cs == 1, e1, np.where(cs == 2, e2, e3))
        z += sc * (np.maximum(cs - dk, 0) / tot * r_[ws]).sum()
    return z


def doc_sparse(W):
    """Per position: document-memory distribution and copy distribution as sparse (ids, probs), plus repeat length."""
    out = []
    for w in W:
        w = w.tolist()
        counts, n_doc, follow, start = {}, 0, {}, 0
        for j in range(1, len(w)):
            if w[j - 1] == 0:
                counts, n_doc, follow, start = {}, 0, {}, j
            cache = (np.array(list(counts)), np.array(list(counts.values()), float) / n_doc) if n_doc else (np.zeros(0, int), np.zeros(0))
            copy, best = (np.zeros(0, int), np.zeros(0)), 0
            for L in range(min(16, j - start), 0, -1):
                f = follow.get((L, tuple(w[j - L:j])))
                if f:
                    tot = sum(f.values())
                    copy, best = (np.array(list(f)), np.array(list(f.values()), float) / tot), L
                    break
            out.append((cache, copy, best, n_doc))
            x = w[j]
            for L in range(1, min(16, j - start) + 1):
                d = follow.setdefault((L, tuple(w[j - L:j])), {})
                d[x] = d.get(x, 0) + 1
            counts[x] = counts.get(x, 0) + 1
            n_doc += 1
    return out


def context_vectors(pos):
    """h_b for every position: average soft-context vector of the tokens in each band (inside the window)."""
    H = np.zeros((len(pos.target), len(BANDS) * a.rank), np.float32)
    for wi, w in enumerate(pos.W):
        r = row[w]
        for b, (lo, hi) in enumerate(BANDS):
            Ub = U[b][r]  # (SEQ+1) x rank
            cs = np.vstack([np.zeros((1, a.rank), np.float32), np.cumsum(Ub, 0)])
            j = np.arange(1, SEQ + 1)  # target index: context tokens j-hi .. j-lo
            top_, bot = j - lo + 1, np.maximum(j - hi, 0)
            n = np.maximum(top_ - bot, 0)
            v = (cs[np.maximum(top_, 0)] - cs[bot]) / np.maximum(n, 1)[:, None]
            v[n == 0] = 0
            H[wi * SEQ:(wi + 1) * SEQ, b * a.rank:(b + 1) * a.rank] = v
    return H


def prepare(pos):
    docs = doc_sparse(pos.W)
    kns = [kn_sparse(pos, q) for q in range(len(pos.target))]
    tgt = pos.target
    pk = kn.kn_prob(pos, S, Dk, N)
    for q in np.random.default_rng(1).choice(len(tgt), 200, replace=False):  # the sparse form must give the same numbers
        G, segs = kns[q]
        onehot = np.zeros(V); onehot[tgt[q]] = 1.0
        assert abs(G * p1[tgt[q]] + seg_dot(segs, onehot) - pk[q]) < 1e-9 * max(pk[q], 1e-12) + 1e-15, q
    pc = np.array([v[i == t].sum() if len(i) else 0.0 for ((i, v), _, _, _), t in zip(docs, tgt)])
    pp = np.array([v[i == t].sum() if len(i) else 0.0 for (_, (i, v), _, _), t in zip(docs, tgt)])
    pc = 0.9 * pc + 0.1 / V
    mlen = np.array([d[2] for d in docs]); hist = np.array([d[3] for d in docs])
    deep = kn.deepest(pos, S, N)
    b = kn.buckets(mlen, hist, deep)
    return dict(kns=kns, docs=docs, P=np.stack([pk, pc, pp], 1), b=b, H=context_vectors(pos), tgt=tgt)


log("preparing tuning positions")
TU = prepare(tune)
log("preparing exam positions")
EX = prepare(exam)
lam = kn.em_weights(TU["P"], TU["b"], 60)


def score(X, beta, chunk=2048):
    """Bits per byte of p(w) = p_mix(w) exp(s(w)) / Z at every position of X (exact Z over the vocabulary)."""
    bscale = np.repeat(np.asarray(beta, np.float32), a.rank)
    nll = 0.0
    n = len(X["tgt"])
    lw = lam[X["b"]]
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        Sc = (X["H"][s0:s1] * bscale) @ Vcat.T  # chunk x V scores
        Sc -= Sc.max(1, keepdims=True)
        E = np.exp(Sc)
        Zuni = E @ p1  # sum_w p1(w) e^s(w) for every position
        for q in range(s0, s1):
            r_ = E[q - s0]
            G, segs = X["kns"][q]
            zk = G * Zuni[q - s0] + seg_dot(segs, r_)
            (ci, cv), (pi, pv), _, _ = X["docs"][q]
            zc = 0.9 * ((cv * r_[ci]).sum() if len(ci) else 0.0) + 0.1 / V * r_.sum()
            zp = (pv * r_[pi]).sum() if len(pi) else 0.0
            Z = lw[q, 0] * zk + lw[q, 1] * zc + lw[q, 2] * zp
            t = X["tgt"][q]
            pm = lw[q, 0] * X["P"][q, 0] + lw[q, 1] * X["P"][q, 1] + lw[q, 2] * X["P"][q, 2]
            nll -= math.log(max(pm * r_[t] / Z, 1e-30))
    return nll / n / math.log(2) / kn.BYTES_PER_TOKEN


base_tune, base_exam = score(TU, [0] * len(BANDS)), score(EX, [0] * len(BANDS))
log(f"without soft context: tune {base_tune:.4f}  exam {base_exam:.4f}")
beta = [0.0] * len(BANDS)
best = base_tune
for sweep in range(2):
    for b in range(len(BANDS)):
        for v in ([0.03, 0.1, 0.3] if sweep == 0 else ([beta[b] * 0.6, beta[b] * 1.5] if beta[b] else [0.01])):
            trial = list(beta); trial[b] = v
            s = score(TU, trial)
            log(f"   try band {BANDS[b]} beta {v:.3f}: tune {s:.4f}")
            if s < best - 1e-4:
                best, beta = s, trial
        log(f"sweep {sweep + 1} band {BANDS[b]}: beta {beta[b]:.3f}  tune {best:.4f}")
final_exam = score(EX, beta)
log(f"WITH soft context: tune {best:.4f}  exam {final_exam:.4f}  (betas {[round(x, 3) for x in beta]})")
json.dump(dict(train_tokens=int(a.train_tokens), kn_orders=N, rank=a.rank, ctx_vocab=a.ctx_vocab, kappa=a.kappa, bands=BANDS,
               tune_positions=len(TU["tgt"]), exam_positions=len(EX["tgt"]), base=dict(tune=base_tune, exam=base_exam),
               soft=dict(tune=best, exam=final_exam), betas=beta), open(a.out, "w"), indent=1)
sub.unlink()
