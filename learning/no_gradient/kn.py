"""A language model built with no gradient descent: counting, smoothing, and EM mixing only.

Parts (all classic, see learning/no_gradient/README.md for the papers):
  1. Interpolated modified Kneser-Ney n-gram model, orders 1..N (Kneser & Ney 1995; Chen & Goodman 1998).
     Built by counting n-grams in our training shards. Lower orders use "continuation counts": how many different
     words come before an n-gram, not how often it occurs.
  2. Document memory (cache): how often each token already appeared in the current document (Kuhn & De Mori 1990).
  3. Copy model: find the longest recent repeat of the last few tokens in this document and predict what followed it
     (the "copying" skill the twin never learned; an exact, count-based cousin of an induction head).
  4. Mixing weights for the three parts by EM (Jelinek & Mercer 1980; Dempster et al. 1977), separately per situation.
Settings (discounts, mixing weights) are fitted on a TUNING set from val_001; the LOCKED EXAM (the same 96 windows from
val_000 used for the twins) is scored once at the end. Nothing in this file computes a gradient.

Counting only what is asked: the model is defined on the whole training text, but we only store counts for contexts
that occur at the tuning and exam positions. That changes no score; it is exactly what the full model outputs there.

    python learning/no_gradient/kn.py --dir <folder with shards/> --train train_000 [train_001 ...] --order 6
"""
import argparse, json, math, pathlib, time

import numpy as np

BYTES_PER_TOKEN, SEQ, V = 4.54, 2048, 49_152
P = np.uint64(0x9E3779B97F4A7C15)  # hash multiplier for n-grams longer than 4 tokens
U16 = np.uint64(16)


def mask(L):  # low bits holding L exact 16-bit tokens
    return np.uint64((1 << (16 * L)) - 1) if L < 4 else np.uint64(0xFFFFFFFFFFFFFFFF)


def ngram_keys(t, N):
    """K[L][i] = key of the L tokens ending at position i (L = 0..N). Up to 4 tokens the key is exact (16 bits each,
    newest in the low bits); longer n-grams are hashed. Entries with i < L-1 are meaningless and never used."""
    t = t.astype(np.uint64)
    K = [np.zeros(len(t), np.uint64)]
    cur = K[0]
    with np.errstate(over="ignore"):
        for L in range(1, N + 1):
            prev = np.empty_like(cur)
            prev[0] = 0
            prev[1:] = cur[:-1]  # key of the L-1 tokens ending one step earlier
            cur = (prev << U16) | t if L <= 4 else prev * P + t + np.uint64(1)
            K.append(cur)
    return K


def member(x, sorted_set):
    if len(sorted_set) == 0:
        return np.zeros(len(x), bool)
    i = np.searchsorted(sorted_set, x)
    i[i == len(sorted_set)] = 0
    return sorted_set[i] == x


def lookup(x, keys, vals, default=0):
    """vals[keys == x] for each x, or default when x is not a key (keys sorted)."""
    out = np.full(len(x), default, dtype=vals.dtype)
    if len(keys) == 0:
        return out
    i = np.searchsorted(keys, x)
    i[i == len(keys)] = 0
    hit = keys[i] == x
    out[hit] = vals[i[hit]]
    return out


# ---------------------------------------------------------------- evaluation windows (same as the twins' exams)
def windows(path, n, seed):
    sh = np.memmap(path, dtype=np.uint16, mode="r")
    st = np.sort(np.random.default_rng(seed).choice(len(sh) - SEQ - 1, n, replace=False))
    return np.stack([sh[s:s + SEQ + 1] for s in st]).astype(np.int64)


class Positions:
    """Every target position of a set of windows: target token, and its n-gram keys inside the window."""

    def __init__(self, W, N):
        self.W, self.N = W, N
        Ks = [ngram_keys(w, N) for w in W]
        j = np.arange(1, SEQ + 1)  # target index inside a window; it has j tokens of context
        self.avail = np.tile(np.minimum(j, N - 1), len(W))  # usable context length
        self.target = W[:, 1:].ravel()
        # ctx[k] = key of the k-1 tokens before the target; full[k] = key of those plus the target
        self.ctx = {k: np.concatenate([K[k - 1][j - 1] for K in Ks]) for k in range(1, N + 1)}
        self.full = {k: np.concatenate([K[k][j] for K in Ks]) for k in range(1, N + 1)}


# ---------------------------------------------------------------- counting
class Store:
    """Distinct n-gram types (with payload columns), merged every few chunks to keep memory bounded."""

    def __init__(self, with_counts=False):
        self.keys, self.cols, self.cnt, self.buf, self.with_counts = None, None, None, [], with_counts

    def add(self, keys, cols):
        if self.with_counts:
            u, idx, c = np.unique(keys, return_index=True, return_counts=True)
            self.buf.append((u, [col[idx] for col in cols], c.astype(np.int64)))
        else:
            u, idx = np.unique(keys, return_index=True)
            self.buf.append((u, [col[idx] for col in cols], None))
        if sum(len(b[0]) for b in self.buf) > 60_000_000:
            self.merge()

    def merge(self):
        parts = self.buf + ([(self.keys, self.cols, self.cnt)] if self.keys is not None else [])
        self.buf = []
        if not parts:
            return
        keys = np.concatenate([p[0] for p in parts])
        cols = [np.concatenate([p[1][c] for p in parts]) for c in range(len(parts[0][1]))]
        if self.with_counts:
            u, idx, inv = np.unique(keys, return_index=True, return_inverse=True)
            self.cnt = np.bincount(inv, weights=np.concatenate([p[2] for p in parts])).astype(np.int64)
        else:
            u, idx = np.unique(keys, return_index=True)
        self.keys, self.cols = u, [c[idx] for c in cols]


def count(train_paths, pos_sets, N, chunk=20_000_000, log=print):
    """One pass over the training shards. For each order k < N: distinct (k+1)-gram types x·h·w whose middle h is a
    context we need (continuation counts). For order N: counts of N-grams whose context we need."""
    need = {k: np.unique(np.concatenate([p.ctx[k] for p in pos_sets])) for k in range(2, N + 1)}
    cont = {k: Store() for k in range(1, N)}
    top = Store(with_counts=True)
    seen = 0
    for path in train_paths:
        sh = np.memmap(path, dtype=np.uint16, mode="r")
        for s in range(0, len(sh), chunk):
            s0, e = max(0, s - N), min(len(sh), s + chunk)
            K = ngram_keys(np.asarray(sh[s0:e]), N)
            i = np.arange(s - s0, e - s0)  # targets of this chunk (the first N tokens are only history)
            for k in range(1, N):
                ii = i[i >= k]  # needs a token x before the context
                if k >= 2:
                    ii = ii[member(K[k - 1][ii - 1], need[k])]
                cols = [] if k + 1 <= 4 else [K[k][ii], K[k - 1][ii - 1]]  # exact keys hold hw and h; hashes don't
                cont[k].add(K[k + 1][ii], cols)
            ii = i[i >= N - 1]
            ii = ii[member(K[N - 1][ii - 1], need[N])]
            top.add(K[N][ii], [K[N - 1][ii - 1]])
            seen += e - s
            log(f"counted {seen / 1e6:,.0f}M tokens")
    for st in list(cont.values()) + [top]:
        st.merge()
    return cont, top, seen


def finalize(cont, top, N):
    """Turn the stored types into per-context statistics: count of (h, w) and, per context h, its total and how many
    w were seen once, twice, three or more times. Also count-of-counts for the standard discount estimate."""
    S = {}
    for k in range(1, N + 1):
        if k < N:
            st = cont[k]
            if k + 1 <= 4:
                hw, h = st.keys & mask(k), (st.keys >> U16) & mask(k - 1)
            else:
                hw, h = st.cols
            uhw, idx, c = np.unique(hw, return_index=True, return_counts=True)
            hh = h[idx]
        else:
            uhw, c, hh = top.keys, top.cnt, top.cols[0]
        uh, hinv = np.unique(hh, return_inverse=True)
        S[k] = dict(hw=uhw, c=c.astype(np.int64), h=uh, tot=np.bincount(hinv, weights=c),
                    n1=np.bincount(hinv, weights=c == 1), n2=np.bincount(hinv, weights=c == 2), n3=np.bincount(hinv, weights=c >= 3),
                    coc=[int((c == r).sum()) for r in (1, 2, 3, 4)], types=len(uhw))
    return S


def standard_discounts(coc):
    n1, n2, n3, n4 = [max(x, 1) for x in coc]
    Y = n1 / (n1 + 2 * n2)
    return [max(0.05, 1 - 2 * Y * n2 / n1), max(0.05, 2 - 3 * Y * n3 / n2), max(0.05, 3 - 4 * Y * n4 / n3)]


def kn_prob(pos, S, D, N):
    """Interpolated modified Kneser-Ney probability of every target, built up from order 1 to order N."""
    p = np.full(len(pos.target), 1.0 / V)
    for k in range(1, N + 1):
        st, (d1, d2, d3) = S[k], D[k]
        ok = pos.avail >= k - 1
        h, hw = pos.ctx[k][ok], pos.full[k][ok]
        tot = lookup(h, st["h"], st["tot"], 0.0)
        c = lookup(hw, st["hw"], st["c"], 0).astype(np.float64)
        n1, n2, n3 = (lookup(h, st["h"], st[n], 0.0) for n in ("n1", "n2", "n3"))
        d = np.where(c == 0, 0, np.where(c == 1, d1, np.where(c == 2, d2, d3)))
        seen = tot > 0
        safe = np.where(seen, tot, 1)
        new = np.maximum(c - d, 0) / safe + (d1 * n1 + d2 * n2 + d3 * n3) / safe * p[ok]
        p[ok] = np.where(seen, new, p[ok])
    return p


def deepest(pos, S, N):
    """Longest order whose context was seen in training (a confidence signal for the mixer)."""
    out = np.zeros(len(pos.target), np.int64)
    for k in range(2, N + 1):
        ok = pos.avail >= k - 1
        seen = np.zeros(len(pos.target), bool)
        seen[ok] = lookup(pos.ctx[k][ok], S[k]["h"], S[k]["tot"], 0.0) > 0
        out[seen] = k
    return out


# ---------------------------------------------------------------- skip-grams (Goodman 2001)
def pattern_keys(t, offs):
    """Context made of the tokens at the given distances before position i (e.g. (2, 3) skips the token just before)."""
    t = t.astype(np.uint64)
    ctx = np.zeros(len(t), np.uint64)
    for o in offs:
        sh = np.zeros(len(t), np.uint64)
        sh[o:] = t[:-o]
        ctx = (ctx << U16) | sh
    return ctx, (ctx << U16) | t


def count_patterns(train_paths, pos_sets, patterns, chunk=20_000_000, log=print):
    """Counts of (skip-context, next token) for contexts that occur at the tuning and exam positions."""
    out = {}
    for offs in patterns:
        need = np.unique(np.concatenate([p.pat[offs][0] for p in pos_sets]))
        st = Store(with_counts=True)
        for path in train_paths:
            sh = np.memmap(path, dtype=np.uint16, mode="r")
            for s in range(0, len(sh), chunk):
                s0, e = max(0, s - max(offs)), min(len(sh), s + chunk)
                ctx, full = pattern_keys(np.asarray(sh[s0:e]), offs)
                i = np.arange(s - s0, e - s0)
                i = i[i >= max(offs)]
                i = i[member(ctx[i], need)]
                st.add(full[i], [ctx[i]])
        st.merge()
        uh, hinv = np.unique(st.cols[0], return_inverse=True)
        c = st.cnt
        out[offs] = dict(hw=st.keys, c=c, h=uh, tot=np.bincount(hinv, weights=c), n1=np.bincount(hinv, weights=c == 1),
                         n2=np.bincount(hinv, weights=c == 2), n3=np.bincount(hinv, weights=c >= 3), coc=[int((c == r).sum()) for r in (1, 2, 3, 4)])
        log(f"skip pattern {offs}: {len(uh):,} contexts, {len(c):,} types")
    return out


def add_patterns(pos, patterns):
    pos.pat = {}
    for offs in patterns:
        cs, fs, ok = [], [], []
        for w in pos.W:
            ctx, full = pattern_keys(w, offs)
            cs.append(ctx[1:]); fs.append(full[1:])
            ok.append(np.arange(1, SEQ + 1) >= max(offs))
        pos.pat[offs] = (np.concatenate(cs), np.concatenate(fs), np.concatenate(ok))


def skip_prob(pos, st, D, offs, base):
    """Modified absolute discounting on the skip context, falling back to `base` (the KN unigram)."""
    ctx, full, ok = pos.pat[offs]
    d1, d2, d3 = D
    tot = lookup(ctx, st["h"], st["tot"], 0.0)
    c = lookup(full, st["hw"], st["c"], 0).astype(np.float64)
    n1, n2, n3 = (lookup(ctx, st["h"], st[n], 0.0) for n in ("n1", "n2", "n3"))
    d = np.where(c == 0, 0, np.where(c == 1, d1, np.where(c == 2, d2, d3)))
    seen = (tot > 0) & ok
    safe = np.where(seen, tot, 1)
    return np.where(seen, np.maximum(c - d, 0) / safe + (d1 * n1 + d2 * n2 + d3 * n3) / safe * base, base)


# ---------------------------------------------------------------- document memory
def doc_memory(W, max_copy=16):
    """For each target: (a) share of the current document's earlier tokens equal to it (unigram cache, with a small
    floor so unseen tokens get a little), (b) copy model: among earlier places in this document where the longest
    possible stretch of the most recent tokens appeared, the share followed by this target, and (c) that stretch's length."""
    cache, copy, mlen, hist = [], [], [], []
    for w in W:
        w = w.tolist()
        counts, n_doc, start = {}, 0, 0
        follow = {}  # (L, tuple of last L tokens) -> {next token: count}
        for j in range(1, len(w)):
            if w[j - 1] == 0:  # end of text: a new document starts, forget the old one
                counts, n_doc, follow, start = {}, 0, {}, j
            x = w[j]
            counts_x = counts.get(x, 0)
            cache.append(counts_x / n_doc if n_doc else 0.0)
            hist.append(n_doc)
            best, pc = 0, 0.0
            for L in range(min(max_copy, j - start), 0, -1):
                f = follow.get((L, tuple(w[j - L:j])))
                if f:
                    best, pc = L, f.get(x, 0) / sum(f.values())
                    break
            copy.append(pc)
            mlen.append(best)
            # learn from this token: the stretches ending just before it are followed by it
            for L in range(1, min(max_copy, j - start) + 1):
                d = follow.setdefault((L, tuple(w[j - L:j])), {})
                d[x] = d.get(x, 0) + 1
            counts[x] = counts_x + 1
            n_doc += 1
    return np.array(cache), np.array(copy), np.array(mlen), np.array(hist)


def buckets(mlen, hist, deep):
    """Situations that get their own mixing weights: copy-match length x document-history length x n-gram depth."""
    m = np.digitize(mlen, [1, 2, 4, 8])        # 0, 1, 2-3, 4-7, 8+
    h = np.digitize(hist, [1, 50, 400])        # none, short, medium, long history
    d = np.digitize(deep, [3, 5])              # shallow, medium, deep n-gram context
    return (m * 4 + h) * 3 + d


def em_weights(P, b, nb, iters=60):
    """EM for mixture weights, one set per bucket: weights = average responsibility of each part (no gradients)."""
    lam = np.full((nb, P.shape[1]), 1.0 / P.shape[1])
    for _ in range(iters):
        r = lam[b] * P
        r /= r.sum(1, keepdims=True)
        for q in range(nb):
            sel = b == q
            if sel.sum() >= 50:
                lam[q] = r[sel].mean(0)
    return lam


def mix(P, lam, b):
    return (lam[b] * P).sum(1)


def bpb(p):
    return float(-np.log(np.maximum(p, 1e-12)).mean() / math.log(2) / BYTES_PER_TOKEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--train", nargs="+", default=["train_000"])
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--tune-windows", type=int, default=16)
    ap.add_argument("--exam-windows", type=int, default=96)
    ap.add_argument("--max-tokens", type=float, default=0, help="use only the first this-many training tokens (0 = all)")
    ap.add_argument("--skip", nargs="*", default=[], help="skip-gram patterns as comma-separated distances, e.g. 2,3 1,3 2")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("results.json")))
    a = ap.parse_args()
    patterns = [tuple(int(x) for x in p.split(",")) for p in a.skip]
    D_, N, t0 = pathlib.Path(a.dir), a.order, time.time()
    log = lambda *m: print(f"[{time.time() - t0:6.0f}s]", *m, flush=True)

    tune = Positions(windows(D_ / "shards/val_001.bin", a.tune_windows, 2024), N)
    exam = Positions(windows(D_ / "shards/val_000.bin", a.exam_windows, 999), N)
    paths = [D_ / f"shards/{t}.bin" for t in a.train]
    if a.max_tokens:  # a prefix of the first shard, for quick runs
        sub = D_ / f"shards/_prefix_{int(a.max_tokens)}.bin"
        np.asarray(np.memmap(paths[0], dtype=np.uint16, mode="r")[:int(a.max_tokens)]).tofile(sub)
        paths = [sub]
    cont, top, seen = count(paths, [tune, exam], N, log=log)
    S = finalize(cont, top, N)
    del cont, top
    log("types per order:", {k: S[k]["types"] for k in S})

    # 1. Kneser-Ney: start from the standard discounts, then improve them by trying nearby values on the tuning set
    D = {k: standard_discounts(S[k]["coc"]) for k in S}
    base_tune = bpb(kn_prob(tune, S, D, N))
    log("KN standard discounts", {k: [round(x, 3) for x in D[k]] for k in D}, "tune", round(base_tune, 4))
    best = base_tune
    for sweep in range(2):
        for k in range(1, N + 1):
            for j in range(3):
                for f in (0.8, 0.9, 1.1, 1.2):
                    trial = {kk: list(v) for kk, v in D.items()}
                    trial[k][j] = min(j + 1 - 0.01, D[k][j] * f)
                    s = bpb(kn_prob(tune, S, trial, N))
                    if s < best - 1e-5:
                        best, D = s, trial
        log(f"discount search sweep {sweep + 1}: tune {best:.4f}")

    # skip-grams: extra parts for the mix
    skipS = {}
    if patterns:
        for pos in (tune, exam):
            add_patterns(pos, patterns)
        skipS = count_patterns(paths, [tune, exam], patterns, log=log)

    # 2-4. document memory and EM mixing, per situation
    parts = {}
    for name, pos in (("tune", tune), ("exam", exam)):
        kn = kn_prob(pos, S, D, N)
        cache, copy, mlen, hist = doc_memory(pos.W)
        cache = 0.9 * cache + 0.1 / V  # never exactly zero
        base = kn_prob(pos, S, D, 1)
        skips = [skip_prob(pos, skipS[o], standard_discounts(skipS[o]["coc"]), o, base) for o in patterns]
        parts[name] = dict(P=np.stack([kn, cache, copy] + skips, 1), b=buckets(mlen, hist, deepest(pos, S, N)), mlen=mlen, kn=kn)
    nb = 5 * 4 * 3
    cols = {"kn_cache": [0, 1], "kn_cache_copy": [0, 1, 2], "kn_skip": [0] + list(range(3, 3 + len(patterns))), "full": list(range(3 + len(patterns)))}
    lams = {k: em_weights(parts["tune"]["P"][:, c], parts["tune"]["b"], nb) for k, c in cols.items()}
    lam = lams["full"]
    res = {}
    for name in ("tune", "exam"):
        pt = parts[name]
        res[name] = dict(kn=round(bpb(pt["kn"]), 4), **{k: round(bpb(mix(pt["P"][:, c], lams[k], pt["b"])), 4) for k, c in cols.items()})
        log(name, res[name])
    full_exam = mix(parts["exam"]["P"], lam, parts["exam"]["b"])
    by_copy = [dict(match=lab, share=round(float(sel.mean()), 4), kn=round(bpb(parts["exam"]["kn"][sel]), 4), full=round(bpb(full_exam[sel]), 4))
               for lab, sel in [("no repeat found", parts["exam"]["mlen"] == 0), ("repeat of 1-3 tokens", (parts["exam"]["mlen"] >= 1) & (parts["exam"]["mlen"] < 4)),
                                ("repeat of 4+ tokens", parts["exam"]["mlen"] >= 4)]]
    json.dump(dict(meta=dict(order=N, skip=[list(o) for o in patterns], train=[str(p.name) for p in paths], train_tokens=int(seen), minutes=round((time.time() - t0) / 60, 1),
                             tune_tokens=len(tune.target), exam_tokens=len(exam.target)),
                   types={k: S[k]["types"] for k in S}, discounts=D, tune=res["tune"], exam=res["exam"], by_copy=by_copy,
                   weights=lam.round(4).tolist()), open(a.out, "w"), indent=1)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
