"""Run the best no-gradient model again on the locked exam and show it on real text, next to the 700M twin.

    python learning/no_gradient/show.py <kn.py run>.parts.npz --dir <folder with twin-day/... checkpoints>

1. rebuilds the final model from the parts kn.py saved (counting model, document memory, two-word memory, copy at four
   repeat lengths), refits the EM mixing weights on the tuning windows and scores the exam windows
2. prints one real exam passage token by token: the probability each model gave the token that actually came next
"""
import argparse, math, pathlib, sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import mix as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--dir", required=True)
ap.add_argument("--window", type=int, default=0)
ap.add_argument("--start", type=int, default=600)
ap.add_argument("--length", type=int, default=70)
a = ap.parse_args()
z = np.load(a.npz)
V = 49_152
bpb = lambda p: float(-np.log(np.maximum(p, 1e-12)).mean() / math.log(2) / 4.54)

# ---------- 1. the final model: same parts and situations as the best variant in mix.py ----------
cols = ["kn", "cache", "recency30", "recency300", "pair", "copy1-1", "copy2-3", "copy4-7", "copy8-32"]
data, P, B = {}, {}, {}
for s in ("tune", "exam"):
    m = M.memory_parts(z[f"{s}_W"])
    d = dict(kn=z[f"{s}_P"][:, 0], cache=z[f"{s}_P"][:, 1], **m)
    floor = lambda x: 0.9 * x + 0.1 / V
    P[s] = np.stack([floor(d[c]) if c in ("cache", "recency30", "recency300") else d[c] for c in cols], 1)
    B[s] = np.digitize(d["longest"], [1, 2, 4, 8, 16]) * 5 + np.digitize(z[f"{s}_deep"], [3, 4, 5, 6])
    data[s] = d
lam = M.em(P["tune"], B["tune"], 30)
p_exam = (lam[B["exam"]] * P["exam"]).sum(1)
p_kn = data["exam"]["kn"]
print(f"locked exam ({len(p_exam):,} tokens):  counting only {bpb(p_kn):.4f}   full no-gradient model {bpb(p_exam):.4f}")
print("mixing weights, situation 'no repeat, deep context':", dict(zip(cols, lam[0 * 5 + 4].round(3))))
print("mixing weights, situation 'repeat of 8+ tokens':    ", dict(zip(cols, lam[4 * 5 + 4].round(3))))

# ---------- 2. one real passage, token by token, against the 700M twin ----------
import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen3ForCausalLM  # noqa: E402
from chotu.config import TOKENIZER_ID, tiny_config  # noqa: E402
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
torch.manual_seed(1234)
twin = Qwen3ForCausalLM(tiny_config())
sd = torch.load(pathlib.Path(a.dir) / "twin-day/main-5x/ckpt/step_010681.pt", map_location="cpu", weights_only=False)["model"]
twin.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})
twin.eval()
W = z["exam_W"][a.window]
with torch.inference_mode():
    lp = twin(torch.tensor(W[None, :-1])).logits[0].float().log_softmax(-1)
p_twin = lp.gather(-1, torch.tensor(W[1:, None]))[:, 0].exp().numpy()
off = a.window * 2048
print(f"\nexam window {a.window}, tokens {a.start}-{a.start + a.length}: probability each model gave the real next token")
print(f"(this window: no-gradient {bpb(p_exam[off:off + 2048]):.3f}, twin {bpb(p_twin):.3f} bits per byte)\n")
print(f"{'next token':>18s}  {'no-grad':>8s}  {'twin':>8s}   repeat  winner")
print("context: " + repr(tok.decode(W[max(0, a.start - 40):a.start + 1].tolist()))[:150])
for j in range(a.start, a.start + a.length):
    t = int(W[j + 1])
    pn, pt = p_exam[off + j], p_twin[j]
    rep = int(data["exam"]["longest"][off + j])
    win = "no-grad" if pn > pt * 1.25 else "twin" if pt > pn * 1.25 else "-"
    print(f"{tok.decode([t])!r:>18s}  {pn:8.1%}  {pt:8.1%}   {rep:6d}  {win}")
