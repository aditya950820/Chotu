"""GPU pre-check (about 1 minute): is the fused Liger loss correct on THIS machine's stack?

    python scripts/check_loss.py --storage /home/chotu-data

On the real 181M model and one real 2,048-token-window microbatch from the first train shard,
compares loss and every parameter gradient of:
  reference  plain cross-entropy in fp32 logits (the slow, obviously-correct path)
  liger      LigerFusedLinearCrossEntropyLoss, exactly as train.py calls it
  liger_bf16 the same, with hidden states and lm_head weight cast to bf16 together
Writes <storage>/reports/loss-check.json and exits 0 only if a fused variant matches the reference
(finite, loss within 1e-2, gradient cosine >= 0.99). train.py reads the verdict and uses that variant,
or the chunked fallback if none passes, so a broken kernel can never start a run again.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chotu.config import PRETRAINING_WINDOW, chotu_config  # noqa: E402
from transformers import Qwen3ForCausalLM  # noqa: E402


def grads(model):
    return torch.cat([p.grad.float().flatten() for p in model.parameters() if p.grad is not None])


def run(model, x, y, variant, compiled):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = compiled(x, use_cache=False).last_hidden_state
        flat, target = hidden.reshape(-1, hidden.shape[-1]), y.reshape(-1)
        if variant == "reference":
            loss = F.cross_entropy(model.lm_head(flat).float(), target)
        else:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            weight = model.lm_head.weight
            if variant == "liger_bf16":
                flat, weight = flat.to(torch.bfloat16), weight.to(torch.bfloat16)
            loss = LigerFusedLinearCrossEntropyLoss()(weight, flat, target)
    loss.backward()
    torch.cuda.synchronize()
    return float(loss.detach()), grads(model), str(hidden.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage", default=os.getenv("CHOTU_STORAGE"))
    ap.add_argument("--rows", type=int, default=8, help="2,048-token windows in the microbatch")
    ap.add_argument("--compile", action="store_true", help="also run under torch.compile like training")
    a = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    shard = np.memmap(sorted(Path(a.storage, "shards").glob("train_*.bin"))[0], dtype=np.uint16, mode="r")
    tokens = torch.from_numpy(shard[: a.rows * (PRETRAINING_WINDOW + 1)].astype(np.int64)).view(a.rows, -1)
    x, y = tokens[:, :-1].to(device), tokens[:, 1:].to(device)
    model = Qwen3ForCausalLM(chotu_config()).to(device)
    model.train()
    forward = torch.compile(model.model) if a.compile else model.model
    ref_loss, ref_grad, dtype = run(model, x, y, "reference", forward)
    results = dict(hidden_dtype=dtype, reference=dict(loss=ref_loss, grad_norm=float(ref_grad.norm()),
                                                      finite=bool(torch.isfinite(ref_grad).all())))
    choice = None
    for variant in ("liger", "liger_bf16"):
        try:
            loss, grad, _ = run(model, x, y, variant, forward)
            finite = bool(torch.isfinite(grad).all()) and math.isfinite(loss)
            cosine = float(F.cosine_similarity(grad, ref_grad, dim=0)) if finite else None
            ok = bool(finite and abs(loss - ref_loss) < 1e-2 and cosine >= 0.99)
            results[variant] = dict(loss=loss, grad_norm=float(grad.norm()), finite=finite,
                                    nonfinite_grads=int((~torch.isfinite(grad)).sum()), cosine=cosine, ok=ok)
        except Exception as error:
            results[variant] = dict(ok=False, error=f"{type(error).__name__}: {error}"[:400])
        choice = choice or (variant if results[variant]["ok"] else None)
    results.update(fused_ce=choice or "off", torch=torch.__version__, compile=a.compile)
    try:
        import liger_kernel
        results["liger_kernel"] = getattr(liger_kernel, "__version__", "?")
    except ImportError:
        results["liger_kernel"] = None
    path = Path(a.storage, "reports", "loss-check.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if not results["reference"]["finite"]:
        sys.exit("reference gradients are not finite: the problem is not the loss kernel")
    sys.exit(0 if choice else 3)


if __name__ == "__main__":
    main()
