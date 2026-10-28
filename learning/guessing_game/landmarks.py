import sys, glob, json, math, numpy as np, torch
sys.path.insert(0, "/home/claude/chotu")
from chotu.config import chotu_config, TOKENIZER_ID, EXPECTED_PARAMS
from transformers import Qwen3ForCausalLM, AutoModelForCausalLM
S = sys.argv[1]; V = 49152; T = 2048; NWIN = int(sys.argv[2])
fs = sorted(glob.glob(f"{S}/data/shards/val_*.bin"))
# 1. frequency-only player: counts from 13 val shards
counts = np.ones(V, dtype=np.float64)  # +1 so no card has zero chips
for f in fs[:-1]:
    counts += np.bincount(np.memmap(f, dtype=np.uint16, mode="r"), minlength=V)
freq = counts / counts.sum(); np.save(f"{S}/freq.npy", freq)
print("counted tokens:", int(counts.sum() - V))
# 2. evaluation text: NWIN windows of 2,048 from the 14th shard (never counted)
ev = np.memmap(fs[-1], dtype=np.uint16, mode="r")
starts = np.linspace(0, len(ev) - T - 1, NWIN).astype(int)
wins = torch.tensor(np.stack([ev[s:s + T + 1] for s in starts]).astype(np.int64))
x, y = wins[:, :-1], wins[:, 1:]
res = {"uniform": math.log(V), "tokens_scored": int(y.numel())}
res["frequency_only"] = float(-np.log(freq[y.numpy()]).mean())
torch.set_grad_enabled(False)
def avg_loss(m):
    tot = 0.0
    for i in range(NWIN):
        lg = m(x[i:i+1]).logits.float()
        tot += torch.nn.functional.cross_entropy(lg[0], y[i], reduction="sum").item()
    return tot / y.numel()
torch.manual_seed(0)
chotu = Qwen3ForCausalLM(chotu_config()).eval()
n = sum(p.numel() for p in chotu.parameters()); assert n == EXPECTED_PARAMS, n
res["untrained_chotu"] = avg_loss(chotu); del chotu
smol = AutoModelForCausalLM.from_pretrained(TOKENIZER_ID, torch_dtype=torch.float32).eval()
res["smollm2_135m"] = avg_loss(smol)
print(json.dumps(res, indent=1)); json.dump(res, open(f"{S}/landmarks.json", "w"), indent=1)
