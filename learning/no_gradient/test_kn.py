"""Checks kn.py against a slow, obviously-correct Kneser-Ney written with Python dicts, and that probabilities sum to 1.
    python learning/no_gradient/test_kn.py <folder with shards/>"""
import sys, pathlib, collections
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import kn

D_ = pathlib.Path(sys.argv[1])
corpus = np.asarray(np.memmap(D_ / "shards/train_000.bin", dtype=np.uint16, mode="r")[:400_000]).astype(np.int64)
tmp = D_ / "shards/_test_corpus.bin"; corpus.astype(np.uint16).tofile(tmp)
for N in (3, 6):
    pos = kn.Positions(kn.windows(D_ / "shards/val_001.bin", 2, 5), N)
    cont, top, _ = kn.count([tmp], [pos], N, chunk=150_000, log=lambda *a: None)  # small chunks: tests the chunk overlap too
    S = kn.finalize(cont, top, N)
    D = {k: kn.standard_discounts(S[k]["coc"]) for k in S}
    fast = kn.kn_prob(pos, S, D, N)
    # slow reference
    c = corpus.tolist()
    raw = collections.Counter(tuple(c[i - N + 1:i + 1]) for i in range(N - 1, len(c)))
    left = {k: collections.defaultdict(set) for k in range(1, N)}  # (h..., w) -> distinct x before it
    for k in range(1, N):
        for i in range(k, len(c)):
            left[k][tuple(c[i - k + 1:i + 1])].add(c[i - k])
    def stats(k, h):
        if k == N:
            items = {g[-1]: n for g, n in raw.items() if g[:-1] == h}
        else:
            items = {g[-1]: len(xs) for g, xs in left[k].items() if g[:-1] == h}
        return items
    cache = {}
    def p_slow(hist, w, k):
        if k == 0:
            return 1.0 / kn.V
        h = tuple(hist[len(hist) - (k - 1):]) if k > 1 else ()
        if (k, h) not in cache:
            cache[(k, h)] = stats(k, h)
        it = cache[(k, h)]
        lower = p_slow(hist, w, k - 1)
        tot = sum(it.values())
        if tot == 0:
            return lower
        d1, d2, d3 = D[k]
        disc = lambda n: 0 if n == 0 else d1 if n == 1 else d2 if n == 2 else d3
        n = it.get(w, 0)
        gamma = sum(disc(v) for v in it.values()) / tot
        return max(n - disc(n), 0) / tot + gamma * lower
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pos.target), 40, replace=False)
    worst = 0
    for q in idx:
        win, j = divmod(q, kn.SEQ)
        j += 1
        hist = pos.W[win][:j].tolist()[-(N - 1):] if N > 1 else []
        k_top = min(N, j + 1)
        ps = p_slow(hist, int(pos.target[q]), k_top)
        worst = max(worst, abs(ps - fast[q]) / ps)
    # probabilities over the whole vocabulary add up to 1, for two contexts
    sums = []
    for q in idx[:2]:
        win, j = divmod(q, kn.SEQ); j += 1
        hist = pos.W[win][:j].tolist()[-(N - 1):]
        sums.append(sum(p_slow(hist, w, min(N, j + 1)) for w in range(kn.V)))
    print(f"N={N}: fast vs slow worst relative difference {worst:.2e}; sums over vocabulary {[round(s, 6) for s in sums]}")
    assert worst < 1e-6 and all(abs(s - 1) < 1e-6 for s in sums)
tmp.unlink()
print("OK")
