"""Find where the 181M model's first-step gradient turns into NaN on the GPU (seen on the H200, 2026-11-11).

On the CPU in plain fp32 the first step is fine (gradient 2.29), so this tries one step on the GPU in three ways
and names the first weights whose gradient is not a number:
    1. fp32, no compile      2. bf16, no compile      3. bf16 + torch.compile (what train.py does)

    python scripts/diag_nan.py --data ~/chotu-data
"""
import argparse, glob, math, pathlib, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from chotu.config import chotu_config  # noqa: E402
from transformers import Qwen3ForCausalLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--windows", type=int, default=8, help="windows per pass (train.py uses 8)")
a = ap.parse_args()

shard = np.memmap(sorted(glob.glob(str(pathlib.Path(a.data).expanduser() / "shards" / "train_*.bin")))[0], dtype=np.uint16, mode="r")
rng = np.random.default_rng(0)
w = np.stack([shard[s:s + 2049] for s in rng.integers(0, len(shard) - 2050, a.windows)]).astype(np.int64)
x, y = torch.from_numpy(w[:, :-1]).cuda(), torch.from_numpy(w[:, 1:]).cuda()
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)


def one_step(label, bf16, compile_):
    torch.manual_seed(1234)
    cfg = chotu_config()
    cfg._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(cfg).cuda()
    fwd = torch.compile(model) if compile_ else model
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
    with ctx:
        logits = fwd(x).logits
    finite_logits = torch.isfinite(logits).all().item()
    loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
    loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    print(f"\n[{label}] loss {loss.item():.3f} | logits finite {finite_logits} | grad norm {norm:.3f} | "
          f"weights with NaN/inf grads: {len(bad)}")
    for n in bad[:8]:
        print("   ", n)
    del model, fwd, logits, loss
    torch.cuda.empty_cache()
    return math.isfinite(norm)


ok = {}
for label, bf16, comp in [("1 fp32", False, False), ("2 bf16", True, False), ("3 bf16+compile", True, True)]:
    try:
        ok[label] = one_step(label, bf16, comp)
    except Exception as e:
        print(f"\n[{label}] crashed: {type(e).__name__}: {str(e)[:300]}")
        ok[label] = None
print("\nSUMMARY:", ok)
