"""Study better document memory and mixing, using the parts saved by kn.py (no recounting, no gradients).

New parts tried (all computed from the window text itself):
  recency   document memory where recent tokens count more (exponential decay, several half-lives)
  pair      two-word memory: how often this token followed the previous token earlier in the document
  copy@L    copy predictions for separate repeat lengths (1, 2-3, 4-7, 8+), mixed instead of taking only the longest
Mixing weights: EM per situation, fitted on the tuning set only; the exam is scored once per variant.

    python learning/no_gradient/mix.py <kn_run.parts.npz> [--out results_mix.json]
"""
import argparse, json, math

import numpy as np

V, BYTES_PER_TOKEN = 49_152, 4.54


def bpb(p):
    return float(-np.log(np.maximum(p, 1e-12)).mean() / math.log(2) / BYTES_PER_TOKEN)


def memory_parts(W, halflives=(30, 300), lens=((1, 1), (2, 3), (4, 7), (8, 32))):
    """Per target: decayed document-memory shares, two-word memory share, and copy shares per repeat-length band."""
    out = {f"recency{h}": [] for h in halflives}
    out.update(pair=[], **{f"copy{a}-{b}": [] for a, b in lens}, longest=[])
    for w in W:
        w = w.tolist()
        dec = {h: ({}, [0.0]) for h in halflives}  # token -> decayed weight, total weight
        pair, follow, start = {}, {}, 0
        for j in range(1, len(w)):
            if w[j - 1] == 0:  # new document
                dec = {h: ({}, [0.0]) for h in halflives}
                pair, follow, start = {}, {}, j
            x, prev = w[j], w[j - 1]
            for h, (cnt, tot) in dec.items():  # decay is applied lazily: weights stored as exp(+t/tau)
                tau = h / math.log(2)
                out[f"recency{h}"].append(cnt.get(x, 0.0) / tot[0] if tot[0] else 0.0)
                g = math.exp((j - start) / tau)
                cnt[x] = cnt.get(x, 0.0) + g
                tot[0] += g
                if tot[0] > 1e200:  # rescale to stay finite
                    for k in cnt:
                        cnt[k] /= 1e200
                    tot[0] /= 1e200
            pd = pair.get(prev)
            out["pair"].append(pd.get(x, 0) / sum(pd.values()) if pd else 0.0)
            best = {}
            longest = 0
            for L in range(min(32, j - start), 0, -1):
                f = follow.get((L, tuple(w[j - L:j])))
                if f:
                    longest = max(longest, L)
                    for a, b in lens:
                        if a <= L <= b and (a, b) not in best:
                            best[(a, b)] = f.get(x, 0) / sum(f.values())
            for a, b in lens:
                out[f"copy{a}-{b}"].append(best.get((a, b), 0.0))
            out["longest"].append(longest)
            pair.setdefault(prev, {})
            pair[prev][x] = pair[prev].get(x, 0) + 1
            for L in range(1, min(32, j - start) + 1):
                d = follow.setdefault((L, tuple(w[j - L:j])), {})
                d[x] = d.get(x, 0) + 1
    return {k: np.array(v) for k, v in out.items()}


def em(P, b, nb, iters=80, min_count=50):
    lam = np.full((nb, P.shape[1]), 1.0 / P.shape[1])
    for _ in range(iters):
        r = lam[b] * P
        r /= r.sum(1, keepdims=True)
        for q in range(nb):
            sel = b == q
            if sel.sum() >= min_count:
                lam[q] = r[sel].mean(0)
    return lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="results_mix.json")
    a = ap.parse_args()
    z = np.load(a.npz)
    data = {}
    for s in ("tune", "exam"):
        m = memory_parts(z[f"{s}_W"])
        hist = np.concatenate([np.minimum(np.arange(1, 2049), 10 ** 9) for _ in z[f"{s}_W"]])  # position in window
        data[s] = dict(kn=z[f"{s}_P"][:, 0], cache=z[f"{s}_P"][:, 1], copy=z[f"{s}_P"][:, 2], deep=z[f"{s}_deep"], b0=z[f"{s}_b"], pos=hist, **m)
        print(s, "parts ready", flush=True)

    def bucket(d, scheme):
        m = np.digitize(d["longest"], [1, 2, 4, 8, 16])
        dp = np.digitize(d["deep"], [3, 4, 5, 6])
        ps = np.digitize(d["pos"], [16, 128, 512])
        if scheme == "base":
            return d["b0"], int(d["b0"].max()) + 1
        if scheme == "fine":
            return (m * 5 + dp) * 4 + ps, 6 * 5 * 4
        if scheme == "copy_x_depth":
            return m * 5 + dp, 6 * 5
        raise ValueError(scheme)

    floor = lambda x: 0.9 * x + 0.1 / V
    variants = {
        "saved mix (kn, cache, copy)": (["kn", "cache", "copy"], "base"),
        "finer situations": (["kn", "cache", "copy"], "fine"),
        "+ recency memory": (["kn", "cache", "copy", "recency30", "recency300"], "fine"),
        "+ two-word memory": (["kn", "cache", "copy", "recency30", "recency300", "pair"], "fine"),
        "+ copy by repeat length": (["kn", "cache", "recency30", "recency300", "pair", "copy1-1", "copy2-3", "copy4-7", "copy8-32"], "fine"),
        "+ copy by repeat length (copy x depth situations)": (["kn", "cache", "recency30", "recency300", "pair", "copy1-1", "copy2-3", "copy4-7", "copy8-32"], "copy_x_depth"),
    }
    res = []
    for name, (cols, scheme) in variants.items():
        P = {s: np.stack([floor(data[s][c]) if c in ("cache", "recency30", "recency300") else data[s][c] for c in cols], 1) for s in data}
        bt, nb = bucket(data["tune"], scheme)
        be, _ = bucket(data["exam"], scheme)
        lam = em(P["tune"], bt, nb)
        r = dict(variant=name, parts=cols, situations=nb, tune=round(bpb((lam[bt] * P["tune"]).sum(1)), 4), exam=round(bpb((lam[be] * P["exam"]).sum(1)), 4))
        res.append(r)
        print(r, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
