import sys, json, numpy as np, torch
sys.path.insert(0, "/home/claude/chotu")
from chotu.config import chotu_config, TOKENIZER_ID
from transformers import Qwen3ForCausalLM, AutoModelForCausalLM, AutoTokenizer
S = sys.argv[1]; torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
ev = np.memmap(f"{S}/data/shards/val_013.bin", dtype=np.uint16, mode="r")
eot = np.where(ev[:3_000_000] == 0)[0]
freq = np.load(f"{S}/freq.npy")
# (doc index in val_013, phrase where play starts, tokens to play, title)
ROUNDS = [(28, " these truths to be", 16, "A famous document"),
          (248, " is a force exerted", 16, "A science textbook"),
          (611, " was five times stronger", 25, "An inventor's story"),
          (593, " are the Bengal", 18, "An animal fact page"),
          (33, " became the first non", 20, "A history article")]
torch.manual_seed(0); chotu = Qwen3ForCausalLM(chotu_config()).eval()
smol = AutoModelForCausalLM.from_pretrained(TOKENIZER_ID, dtype=torch.float32).eval()
show = lambda i: tok.decode([int(i)])
def probs(m, ids): return torch.softmax(m(torch.tensor([ids])).logits[0].float(), -1).numpy()
out = []
for di, phrase, n, title in ROUNDS:
    a, b = eot[di] + 1, eot[di + 1]; doc = [int(t) for t in ev[a:b]]
    text = tok.decode(doc); ci = text.index(phrase)
    start = next(i for i in range(len(doc)) if len(tok.decode(doc[:i])) >= ci)
    ids = doc[:start + n]; pc, ps = probs(chotu, ids), probs(smol, ids)
    steps = []
    for p in range(start, start + n):
        actual = ids[p]; ctxpos = p - 1  # prediction made after reading token p-1
        top = list(np.argsort(-ps[ctxpos])[:6]); cards = [actual] + [t for t in top if t != actual][:5]
        rng = np.random.default_rng(p); rng.shuffle(cards)
        def player(pr): t3 = np.argsort(-pr)[:3]; return {"p": float(pr[actual]), "top": [[show(t), float(pr[t])] for t in t3]}
        steps.append({"actual": show(actual), "id": actual, "cards": [show(c) for c in cards], "card_ids": [int(c) for c in cards],
                      "untrained": player(pc[ctxpos]), "freq": player(freq), "smol": player(ps[ctxpos])})
    out.append({"title": title, "context": tok.decode(doc[:start]), "steps": steps, "after": tok.decode(doc[start + n:start + n + 25])})
    print(title, "|", "".join(s["actual"] for s in steps), "|", np.mean([-np.log(s["smol"]["p"]) for s in steps]).round(2),
          np.mean([-np.log(s["freq"]["p"]) for s in steps]).round(2), np.mean([-np.log(s["untrained"]["p"]) for s in steps]).round(2))
json.dump(out, open(f"{S}/rounds.json", "w"))
