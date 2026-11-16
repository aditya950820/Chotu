"""A no-gradient model that WRITES: counts from a suffix array over training text, plus document memory and copying.

Same ideas as kn.py (the 1.27 model), in a form that can answer any context:
  - suffix array over the first --tokens training tokens (Manber & Myers 1990; as in infini-gram, Liu et al. 2024):
    every place a context occurred is found by binary search, and the tokens that followed are counted on the spot
  - smoothing: interpolated absolute discounting from the longest seen context (up to --order - 1 tokens) down to
    single words (Ney et al. 1994), shorter contexts fill in for longer ones
  - document memory (tokens already used in the prompt and answer) and copying (what followed the longest recent repeat)
  - decoding: repetition penalty, temperature, top-k, as in the twin playground
No gradients anywhere.

    python learning/no_gradient/gen.py --shard <train_000.bin> --tokens 300e6 [--twin <step_010681.pt>]
"""
import argparse, pathlib, sys, time

import numpy as np

V = 49_152
ap = argparse.ArgumentParser(description="See the module docstring.")
ap.add_argument("--shard", required=True)
ap.add_argument("--tokens", type=float, default=300e6)
ap.add_argument("--order", type=int, default=13, help="longest n-gram (context up to order-1 tokens)")
ap.add_argument("--cap", type=int, default=200_000, help="count at most this many occurrences of a context (sampled)")
ap.add_argument("--twin", default="", help="700M twin checkpoint, to print its answers alongside")
ap.add_argument("--prompts", nargs="*", default=["The capital of France is", "Photosynthesis is the process by which",
                                                 "Once upon a time, there was a", "The most important thing about exercise is",
                                                 "In 1969, the first", "To make a cup of tea, you", "Water boils at",
                                                 "The best way to learn a new language is"])
ap.add_argument("--length", type=int, default=50)
ap.add_argument("--out", default="")
a = ap.parse_args() if __name__ == "__main__" else ap.parse_args(["--shard", ""])
t0 = time.time()
log = lambda *m: print(f"[{time.time() - t0:5.0f}s]", *m, flush=True)

# ---------------------------------------------------------------- the index
T = B = sa = uni = None
rng = np.random.default_rng(0)


def build(shard, tokens):
    """Load the training tokens and build the suffix array (about 2 minutes for 300M tokens)."""
    global T, B, sa, uni
    import pydivsufsort
    T = np.array(np.memmap(shard, dtype=np.uint16, mode="r")[:int(tokens)])
    B = T.astype(">u2").tobytes()  # big-endian, so byte order = token order
    s_ = pydivsufsort.divsufsort(np.frombuffer(bytearray(B), np.uint8))
    sa = (s_[s_ % 2 == 0] // 2).astype(np.int64)  # keep suffixes that start on a token
    del s_
    uni = np.bincount(T, minlength=V).astype(np.float64) + 0.5
    uni /= uni.sum()
    log(f"suffix array over {len(T):,} tokens")


def span(ctx):
    """SA rows whose suffix starts with ctx (binary search, comparing bytes)."""
    if not ctx:
        return 0, len(sa)
    key = np.array(ctx, ">u2").tobytes()
    L = len(key)
    lo, hi = 0, len(sa)
    while lo < hi:
        mid = (lo + hi) // 2
        p = 2 * int(sa[mid])
        if B[p:p + L] < key:
            lo = mid + 1
        else:
            hi = mid
    start, hi = lo, len(sa)
    while lo < hi:
        mid = (lo + hi) // 2
        p = 2 * int(sa[mid])
        if B[p:p + L] <= key:
            lo = mid + 1
        else:
            hi = mid
    return start, lo


def followers(ctx):
    s, e = span(ctx)
    n = e - s
    if n == 0:
        return None, 0
    rows = sa[s:e] if n <= a.cap else sa[s + rng.integers(0, n, a.cap)]
    nxt = rows + len(ctx)
    nxt = nxt[nxt < len(T)]
    c = np.bincount(T[nxt], minlength=V).astype(np.float64)
    return c * (n / max(len(nxt), 1)), n


def p_ngram(hist, D=0.75):
    """Interpolated absolute discounting, built from the empty context up to the longest one seen in training."""
    p = uni.copy()
    deepest = 0
    for L in range(1, a.order):
        if L > len(hist):
            break
        c, n = followers(hist[-L:])
        if n == 0:
            break  # longer contexts can't have been seen either
        C, u = c.sum(), (c > 0).sum()
        p = np.maximum(c - D, 0) / C + D * u / C * p
        deepest = L
    return p, deepest


def p_document(hist):
    """Document memory and copy distributions from the prompt + answer so far."""
    cache = np.bincount(np.array(hist), minlength=V).astype(np.float64)
    cache /= cache.sum()
    copy, best = None, 0
    for L in range(min(16, len(hist) - 1), 0, -1):
        tail = hist[-L:]
        nxt = [hist[i + L] for i in range(len(hist) - L) if hist[i:i + L] == tail]
        if nxt:
            copy = np.bincount(np.array(nxt), minlength=V).astype(np.float64)
            copy /= copy.sum()
            best = L
            break
    return cache, copy, best


def next_dist(hist, info=False):
    pn, deep = p_ngram(hist)
    cache, copy, rep = p_document(hist)
    w_copy = 0.0 if copy is None else (0.05 if rep < 4 else 0.3)  # trust a long repeat more (as EM found)
    p = (0.93 - w_copy) * pn + 0.07 * cache + (w_copy * copy if copy is not None else 0)
    p = p / p.sum()
    return (p, dict(deep=deep, repeat=rep)) if info else p


def write(prompt_ids, n, temp=0.8, rep=1.2, top_k=40, seed=1, details=None):
    """Continue prompt_ids by n tokens. If `details` is a list, one dict per token is appended to it: the probability
    the model gave the chosen token, how many tokens of context it found in training, and the document repeat length."""
    r = np.random.default_rng(seed)
    hist = list(prompt_ids)
    for _ in range(n):
        p, inf = next_dist(hist, info=True)
        p0 = p.copy()
        used = np.unique(hist)
        p[used] /= rep  # repetition penalty on probabilities
        if temp == 0:
            w = int(p.argmax())
        else:
            top = np.argpartition(-p, top_k)[:top_k]
            q = np.log(p[top]) / temp
            q = np.exp(q - q.max())
            w = int(r.choice(top, p=q / q.sum()))
        hist.append(w)
        if details is not None:
            details.append(dict(p=float(p0[w]), **inf))
        if w == 0:
            break
    return hist[len(prompt_ids):]


if __name__ == "__main__":
    build(a.shard, a.tokens)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from transformers import AutoTokenizer  # noqa: E402
    from chotu.config import TOKENIZER_ID  # noqa: E402
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    twin = None
    if a.twin:
        import torch
        from transformers import Qwen3ForCausalLM
        from chotu.config import tiny_config
        torch.manual_seed(1234)
        twin = Qwen3ForCausalLM(tiny_config())
        sd = torch.load(a.twin, map_location="cpu", weights_only=False)["model"]
        twin.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})
        twin.eval()


    def twin_write(ids, n, temp=0.8, rep=1.2, seed=1):
        import torch
        g = torch.Generator().manual_seed(seed)
        x = torch.tensor([ids])
        with torch.inference_mode():
            for _ in range(n):
                lg = twin(x).logits[0, -1].float()
                used = x[0].unique()
                lg[used] = torch.where(lg[used] > 0, lg[used] / rep, lg[used] * rep)
                top = lg.topk(40)
                w = top.indices[torch.multinomial(torch.softmax(top.values / temp, -1), 1, generator=g)[0]]
                x = torch.cat([x, w.view(1, 1)], 1)
        return x[0, len(ids):].tolist()


    lines = []
    for pr in a.prompts:
        ids = tok(pr).input_ids
        t1 = time.time()
        ng = tok.decode(write(ids, a.length))
        secs = time.time() - t1
        block = f"\n### {pr} …\n- **No-gradient model** ({secs:.0f} s): {pr}{ng}"
        if twin is not None:
            block += f"\n- **700M twin**: {pr}{tok.decode(twin_write(ids, a.length))}"
        print(block, flush=True)
        lines.append(block)
    if a.out:
        pathlib.Path(a.out).write_text("# No-gradient model vs 700M twin: same prompts, same sampling (temperature 0.8, top-40, repetition penalty 1.2)\n" + "\n".join(lines) + "\n")
