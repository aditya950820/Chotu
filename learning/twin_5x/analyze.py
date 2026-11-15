"""Twin 140M vs twin 700M: what did 5x more reading change, and why? (live data only)

Compares, on the same locked held-out windows from our val shards:
    untrained   the twin's random start (seed 1234, the same start both runs began from)
    unigram     a "common words only" guesser (token counts from a different val shard)
    twin-140M   2026-11-07, 2,136 steps (HF magnateadi/chotu-twin-day, twin-day/main)
    twin-700M   2026-11-11, 10,681 steps (twin-day/main-5x)
and writes learning/twin_5x/results.json for build.py.

    python learning/twin_5x/analyze.py --dir <folder with twin-day/... and shards/val_00*.bin>
"""
import argparse, json, math, pathlib, sys, time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chotu.config import TOKENIZER_ID, tiny_config  # noqa: E402
from transformers import AutoTokenizer, Qwen3ForCausalLM  # noqa: E402

BYTES_PER_TOKEN = 4.54
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--windows", type=int, default=96, help="held-out windows of 2,048 tokens")
ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("results.json")))
a = ap.parse_args()
D = pathlib.Path(a.dir)
torch.set_num_threads(max(1, torch.get_num_threads()))
t_start = time.time()
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
V = 49_152
SEQ = 2048


def log(*m):
    print(f"[{time.time() - t_start:6.0f}s]", *m, flush=True)


# ---------- the locked exam: windows from val_000; the unigram guesser learns from val_001 ----------
exam_shard = np.memmap(D / "shards/val_000.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(999)
starts = np.sort(rng.choice(len(exam_shard) - SEQ - 1, a.windows, replace=False))
W = torch.from_numpy(np.stack([exam_shard[s:s + SEQ + 1] for s in starts]).astype(np.int64))
X, Y = W[:, :-1], W[:, 1:]
counts = np.bincount(np.memmap(D / "shards/val_001.bin", dtype=np.uint16, mode="r"), minlength=V).astype(np.float64)
uni_logp = np.log((counts + 0.5) / (counts + 0.5).sum())  # add-half smoothing so unseen tokens are not infinite
rank = np.empty(V, dtype=np.int64)
rank[np.argsort(-counts)] = np.arange(V)  # 0 = most common token
log(f"exam: {a.windows} windows x {SEQ} = {Y.numel():,} tokens; unigram counts from {int(counts.sum()):,} tokens")


# ---------- the players ----------
def load(path):
    torch.manual_seed(1234)  # train.py's seed: the random start both runs began from
    m = Qwen3ForCausalLM(tiny_config())
    if path:
        sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
        m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})
    return m.eval()


models = {"untrained": load(None),
          "twin-140M": load(D / "twin-day/main/ckpt/step_002136.pt"),
          "twin-700M": load(D / "twin-day/main-5x/ckpt/step_010681.pt")}


@torch.no_grad()
def token_losses(m):
    """Surprise (nats) on every exam token, plus whether the top guess was right and how sure it was."""
    loss, top1, conf = [], [], []
    for i in range(0, len(X), 4):
        lg = m(X[i:i + 4]).logits.float()
        lp = lg.log_softmax(-1)
        loss.append(-lp.gather(-1, Y[i:i + 4, :, None])[..., 0])
        top1.append(lg.argmax(-1) == Y[i:i + 4])
        conf.append(lp.max(-1).values.exp())
    return torch.cat(loss).numpy(), torch.cat(top1).numpy(), torch.cat(conf).numpy()


L, TOP1, CONF = {}, {}, {}
for name, m in models.items():
    L[name], TOP1[name], CONF[name] = token_losses(m)
    log(name, "mean surprise", round(float(L[name].mean()), 4))
L["unigram"] = -uni_logp[Y.numpy()]
bpb = lambda x: float(x.mean() / math.log(2) / BYTES_PER_TOKEN)
players = ["untrained", "unigram", "twin-140M", "twin-700M"]
R = {"meta": dict(windows=a.windows, tokens=int(Y.numel()), seq=SEQ, params=sum(p.numel() for p in models["twin-700M"].parameters()),
                  made=time.strftime("%Y-%m-%d %H:%M"))}
R["overall"] = [dict(player=p, loss=round(float(L[p].mean()), 4), bpb=round(bpb(L[p]), 4),
                     top1=round(float(TOP1[p].mean()), 4) if p in TOP1 else None) for p in players]

# ---------- 1. where in the window: does it use what it read earlier? ----------
pos = np.arange(SEQ)
edges = [0, 1, 4, 16, 64, 256, 1024, 2048]
R["by_position"] = [dict(pos=f"{lo}-{hi - 1}" if hi - lo > 1 else str(lo), lo=lo, hi=hi,
                         **{p: round(float(L[p][:, lo:hi].mean()), 4) for p in players})
                    for lo, hi in zip(edges[:-1], edges[1:])]

# ---------- 2. common vs rare words ----------
yr = rank[Y.numpy()]
fb = [(0, 10, "top 10"), (10, 100, "10-100"), (100, 1000, "100-1k"), (1000, 10000, "1k-10k"), (10000, V, "rarer than 10k")]
R["by_frequency"] = []
for lo, hi, lab in fb:
    sel = (yr >= lo) & (yr < hi)
    R["by_frequency"].append(dict(bucket=lab, share=round(float(sel.mean()), 4), **{p: round(float(L[p][sel].mean()), 4) for p in players}))

# ---------- 3. kinds of tokens ----------
pieces = tok.convert_ids_to_tokens(list(range(V)))
kind = np.empty(V, dtype=object)
for i, s in enumerate(pieces):
    t = s.replace("Ġ", " ").replace("Ċ", "\n")
    if t.strip() == "":
        kind[i] = "space / newline"
    elif t.strip().isdigit():
        kind[i] = "digit"
    elif all(not c.isalnum() for c in t.strip()):
        kind[i] = "punctuation"
    elif s.startswith("Ġ"):
        kind[i] = "word start"
    else:
        kind[i] = "word piece (middle/end)"
yk = kind[Y.numpy()]
R["by_kind"] = []
for k in ["word start", "word piece (middle/end)", "punctuation", "digit", "space / newline"]:
    sel = yk == k
    R["by_kind"].append(dict(kind=k, share=round(float(sel.mean()), 4), **{p: round(float(L[p][sel].mean()), 4) for p in players}))

# ---------- 4. token by token: who got better, who got worse ----------
d = L["twin-140M"] - L["twin-700M"]  # positive = the 700M twin is less surprised
R["delta"] = dict(mean=round(float(d.mean()), 4), better=round(float((d > 0.1).mean()), 4), worse=round(float((d < -0.1).mean()), 4),
                  same=round(float((np.abs(d) <= 0.1).mean()), 4),
                  hist=np.histogram(np.clip(d, -4, 4), bins=40, range=(-4, 4))[0].tolist())
ids = Y.numpy().ravel()
df = d.ravel()
per = {}
for t_ in np.unique(ids):
    sel = ids == t_
    if sel.sum() >= 60:
        per[int(t_)] = (float(df[sel].mean()), int(sel.sum()), float(L["twin-700M"].ravel()[sel].mean()))
best = sorted(per.items(), key=lambda kv: -kv[1][0])[:12]
worst = sorted(per.items(), key=lambda kv: kv[1][0])[:6]
fmt = lambda t_: tok.decode([t_]).replace("\n", "\\n")
R["tokens_gained"] = [dict(token=fmt(t_), gain=round(g, 3), n=n, loss_700=round(l7, 3)) for t_, (g, n, l7) in best]
R["tokens_lost"] = [dict(token=fmt(t_), gain=round(g, 3), n=n, loss_700=round(l7, 3)) for t_, (g, n, l7) in worst]

# ---------- 5. one real passage, token by token ----------
w0 = 0  # the first exam window, not cherry-picked
s0 = 400
R["passage"] = [dict(t=tok.decode([int(Y[w0, j])]), a=round(float(L["twin-140M"][w0, j]), 2), b=round(float(L["twin-700M"][w0, j]), 2))
                for j in range(s0, s0 + 90)]

# ---------- 6. does it reuse what is already on the page? ----------
seen = np.zeros_like(Y.numpy(), dtype=bool)
for i in range(len(Y)):
    row, on_page = W[i].tolist(), set()
    on_page.add(row[0])
    for j in range(SEQ):  # target j is row[j + 1]: has it already appeared in the window?
        seen[i, j] = row[j + 1] in on_page
        on_page.add(row[j + 1])
R["repeat_vs_new"] = [dict(case=c, share=round(float(sel.mean()), 4), **{p: round(float(L[p][sel].mean()), 4) for p in players})
                      for c, sel in [("token already appeared earlier in the window", seen), ("first time in the window", ~seen)]]


@torch.no_grad()
def copying(m, n=128, reps=8):
    """A random token string shown twice: a model that can look back finds the repeat easy."""
    g = torch.Generator().manual_seed(7)
    s = torch.randint(1000, 40_000, (reps, n), generator=g)
    x = torch.cat([s, s], 1)
    lp = m(x[:, :-1]).logits.float().log_softmax(-1)
    ce = -lp.gather(-1, x[:, 1:, None])[..., 0]
    return float(ce[:, :n - 1].mean()), float(ce[:, n:].mean())


R["copying"] = []
for p in ["untrained", "twin-140M", "twin-700M"]:
    f_, s_ = copying(models[p])
    R["copying"].append(dict(player=p, first=round(f_, 3), repeat=round(s_, 3), blind=round(math.log(V), 3)))

# ---------- 7. confidence: when it is sure, is it right? ----------
R["calibration"] = []
for p in ["twin-140M", "twin-700M"]:
    rows = []
    for lo, hi in [(0, .1), (.1, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        sel = (CONF[p] >= lo) & (CONF[p] < hi)
        if sel.sum():
            rows.append(dict(band=f"{int(lo * 100)}-{min(100, int(hi * 100))}%", share=round(float(sel.mean()), 4),
                             said=round(float(CONF[p][sel].mean()), 3), right=round(float(TOP1[p][sel].mean()), 3)))
    R["calibration"].append(dict(player=p, rows=rows))


# ---------- 8. writing: same prompts, and how much it loops ----------
@torch.no_grad()
def write(m, ids, n=60, temp=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([ids])
    for _ in range(n):
        lg = m(x[:, -SEQ:]).logits[0, -1].float()
        nxt = lg.argmax() if temp == 0 else torch.multinomial((lg / temp).softmax(-1), 1, generator=g)[0]
        x = torch.cat([x, nxt.view(1, 1)], 1)
    return x[0, len(ids):].tolist()


def loopiness(ids, n=4):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / max(1, len(grams))  # share of 4-token phrases already said before


PROMPTS = ["The capital of France is", "Photosynthesis is the process by which", "Once upon a time, there was a",
           "The most important thing about exercise is", "In 1969, the first"]
R["writing"] = []
for pr in PROMPTS:
    ids = tok(pr).input_ids
    R["writing"].append(dict(prompt=pr, **{p: dict(greedy=tok.decode(write(models[p], ids, 50)), sampled=tok.decode(write(models[p], ids, 50, 0.8, 1)))
                                          for p in ["twin-140M", "twin-700M"]}))
loops = {p: [] for p in ["twin-140M", "twin-700M"]}
for i in range(24):  # 24 real held-out openings, greedy continuation of 80 tokens
    ids = W[i, :48].tolist()
    for p in loops:
        loops[p].append(loopiness(write(models[p], ids, 80)))
R["loopiness"] = [dict(player=p, mean=round(float(np.mean(v)), 3), runs=len(v)) for p, v in loops.items()]
log("writing done")

# ---------- 9. what changed inside: distance travelled from the shared random start ----------
w_init, w_a, w_b = (dict(models[k].named_parameters()) for k in ("untrained", "twin-140M", "twin-700M"))


def group(n):
    if "embed_tokens" in n:
        return "embeddings (shared with output)"
    if "norm" in n:
        return "norms"
    if "self_attn" in n:
        return "attention"
    return "MLP" if "mlp" in n else "other"


moved = {}
for n, p0 in w_init.items():
    gk = group(n)
    layer = n.split("layers.")[1].split(".")[0] if "layers." in n else "-"
    key = (gk, layer)
    e = moved.setdefault(key, [0.0, 0.0, 0.0, 0.0])
    e[0] += float((w_a[n] - p0).pow(2).sum())
    e[1] += float((w_b[n] - p0).pow(2).sum())
    e[2] += float(p0.pow(2).sum())
    e[3] += float((w_b[n] - w_a[n]).pow(2).sum())
R["weights"] = [dict(group=g_, layer=l_, moved_140=round(math.sqrt(v[0] / v[2]), 4), moved_700=round(math.sqrt(v[1] / v[2]), 4),
                     a_to_b=round(math.sqrt(v[3] / v[2]), 4)) for (g_, l_), v in sorted(moved.items(), key=lambda kv: (kv[0][1], kv[0][0]))]
fa = torch.cat([(w_a[n] - w_init[n]).flatten() for n in w_init])
fb_ = torch.cat([(w_b[n] - w_init[n]).flatten() for n in w_init])
R["weights_summary"] = dict(cosine_of_moves=round(float(F.cosine_similarity(fa, fb_, 0)), 4),
                            moved_ratio=round(float(fb_.norm() / fa.norm()), 3))

# ---------- 10. the two training runs over time ----------
def curve(path):
    recs = [json.loads(x) for x in open(path)]
    ev = [dict(step=r["step"], val_bpb=r["val_bpb"]) for r in recs if r["t"] == "eval"]
    st = [r for r in recs if r["t"] == "step"]
    every = max(1, len(st) // 300)
    return dict(evals=ev, steps=[dict(step=r["step"], loss=r["loss"], lr=r["lr"]) for r in st[::every]] + [dict(step=st[-1]["step"], loss=st[-1]["loss"], lr=st[-1]["lr"])],
                batch=65_536)


R["curves"] = {"twin-140M": curve(D / "twin-day/main/metrics.jsonl"), "twin-700M": curve(D / "twin-day/main-5x/metrics.jsonl")}
json.dump(R, open(a.out, "w"), indent=1)
log("wrote", a.out)
