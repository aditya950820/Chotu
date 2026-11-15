"""Weight-space programming on the 700M twin: remove its "repetition direction" from MLP weights, no gradients.

Follows the method described for Qwen3.5-0.8B (Takehi et al., Nov 2026, "weight-space programming"):
  1. the twin writes greedy continuations of held-out openings; each position is labelled LOOPING (the 4-token
     phrase it just wrote already appeared earlier in its answer) or FRESH
  2. per layer, the repetition direction u = mean residual activation when looping - mean when fresh (normalised)
  3. find the MLP output columns (down_proj) that write most strongly along u, and remove a scaled share of that:
        W_new = W - s * outer(u, mask * (u @ W))
  4. pick layers, column count and strength s on TUNING openings (val_001) by looping, keeping the exam score honest;
     report looping, exam bits per byte and samples on held-out openings (val_000) once.

    python learning/weight_edit/unloop.py --dir <folder with twin-day/... and shards/val_00*.bin>
"""
import argparse, json, math, pathlib, sys, time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chotu.config import TOKENIZER_ID, tiny_config  # noqa: E402
from transformers import AutoTokenizer, Qwen3ForCausalLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("unloop_results.json")))
a = ap.parse_args()
D = pathlib.Path(a.dir)
t0 = time.time()
log = lambda *m: print(f"[{time.time() - t0:5.0f}s]", *m, flush=True)
SEQ, BPT = 2048, 4.54
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)


def load():
    torch.manual_seed(1234)
    m = Qwen3ForCausalLM(tiny_config())
    sd = torch.load(D / "twin-day/main-5x/ckpt/step_010681.pt", map_location="cpu", weights_only=False)["model"]
    m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})
    return m.eval()


def openings(path, n, seed, length=48):
    sh = np.memmap(path, dtype=np.uint16, mode="r")
    out, rng = [], np.random.default_rng(seed)
    while len(out) < n:
        s = int(rng.integers(0, len(sh) - length))
        w = sh[s:s + length].astype(np.int64)
        if 0 not in w:  # inside one document
            out.append(w.tolist())
    return out


@torch.inference_mode()
def write(m, ids, n=80):
    x = torch.tensor([ids])
    for _ in range(n):
        x = torch.cat([x, m(x).logits[:, -1].argmax(-1, keepdim=True)], 1)
    return x[0].tolist()


def loop_labels(seq, start):
    """For each generated position: is the 4-token phrase ending here already in the answer so far?"""
    lab, seen = [], set()
    for j in range(start, len(seq)):
        g = tuple(seq[j - 3:j + 1])
        lab.append(g in seen)
        seen.add(g)
    return lab


def loopiness(m, opens):
    vals = []
    for ids in opens:
        out = write(m, ids)
        vals.append(float(np.mean(loop_labels(out, len(ids)))))
    return float(np.mean(vals)), vals


@torch.inference_mode()
def exam_bpb(m, n=24):
    sh = np.memmap(D / "shards/val_000.bin", dtype=np.uint16, mode="r")
    st = np.sort(np.random.default_rng(999).choice(len(sh) - SEQ - 1, 96, replace=False))[:n]  # first 24 exam windows
    W = torch.from_numpy(np.stack([sh[s:s + SEQ + 1] for s in st]).astype(np.int64))
    tot = 0.0
    for i in range(0, n, 4):
        x = W[i:i + 4]
        tot += F.cross_entropy(m(x[:, :-1]).logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="sum").item()
    return tot / (n * SEQ) / math.log(2) / BPT


# ---------- 1-2. find the repetition direction from the twin's own writing on tuning openings ----------
m = load()
tune_open = openings(D / "shards/val_001.bin", 32, 1)
held_open = openings(D / "shards/val_000.bin", 32, 2)
acts = {l: {"loop": [], "fresh": []} for l in range(4)}
with torch.inference_mode():
    for ids in tune_open:
        seq = write(m, ids)
        lab = loop_labels(seq, len(ids))
        hs = m(torch.tensor([seq]), output_hidden_states=True).hidden_states  # input to each layer + final
        for l in range(4):
            h = hs[l + 1][0, len(ids):len(seq)]  # residual stream after layer l, at generated positions
            for v, isloop in zip(h, lab):
                acts[l]["loop" if isloop else "fresh"].append(v.numpy())
U = {}
for l in range(4):
    lo, fr = np.array(acts[l]["loop"]), np.array(acts[l]["fresh"])
    u = lo.mean(0) - fr.mean(0)
    U[l] = torch.tensor(u / np.linalg.norm(u), dtype=torch.float32)
    log(f"layer {l}: {len(lo)} looping vs {len(fr)} fresh positions; direction separation {np.linalg.norm(lo.mean(0) - fr.mean(0)):.3f}")

base = dict(tune_loop=loopiness(m, tune_open)[0], held_loop=loopiness(m, held_open)[0], exam=exam_bpb(m))
log("before:", {k: round(v, 4) for k, v in base.items()})


def edited(layers, k, s):
    e = load()
    with torch.no_grad():
        for l in layers:
            W = e.model.layers[l].mlp.down_proj.weight  # [hidden 128, ffn 384]: column j = what neuron j writes
            proj = U[l] @ W  # how strongly each column writes along the direction
            mask = torch.zeros_like(proj)
            mask[proj.topk(k).indices] = 1.0  # columns that push towards looping
            W -= s * torch.outer(U[l], mask * proj)
    return e


# ---------- 3. choose layers, column count and strength on tuning openings ----------
trials = []
best = None
for layers in ([1], [2], [3], [2, 3], [1, 2, 3], [0, 1, 2, 3]):
    for k in (16, 64):
        for s in (2.0, 6.0):
            e = edited(layers, k, s)
            tl = loopiness(e, tune_open)[0]
            ex = exam_bpb(e)
            trials.append(dict(layers=layers, columns=k, strength=s, tune_loop=round(tl, 4), exam=round(ex, 4)))
            log(trials[-1])
            ok = ex <= base["exam"] * 1.01  # allow at most 1% worse exam score
            if ok and (best is None or tl < best["tune_loop"]):
                best = trials[-1]
log("chosen on tuning openings:", best)

# ---------- 4. held-out check ----------
e = edited(best["layers"], best["columns"], best["strength"]) if best else m
after = dict(held_loop=loopiness(e, held_open)[0], exam=exam_bpb(e))
samples = []
for p in ["The capital of France is", "Photosynthesis is the process by which", "The most important thing about exercise is", "In 1969, the first"]:
    ids = tok(p).input_ids
    samples.append(dict(prompt=p, before=tok.decode(write(m, ids, 50)[len(ids):]), after=tok.decode(write(e, ids, 50)[len(ids):])))
log("held-out after:", {k: round(v, 4) for k, v in after.items()})
json.dump(dict(base={k: round(v, 4) for k, v in base.items()}, chosen=best, held_after={k: round(v, 4) for k, v in after.items()},
               trials=trials, samples=samples), open(a.out, "w"), indent=1)
for s_ in samples:
    print("\n", s_["prompt"], "\n  before:", repr(s_["before"][:150]), "\n  after: ", repr(s_["after"][:150]))
