"""Export the two twins and the tokenizer into small files the Talk to the Twins page loads in the browser.

    python learning/twin_5x/playground/export.py --dir <folder with twin-day/...ckpt> --out <folder>

Writes tokenizer.json (vocab + merges only), twin140.bin and twin700.bin (all weights as float16, in the order of
manifest.json), and check.json (PyTorch's own next-token logits for a test sentence, so the page's maths can be
verified). Weights and data never go in git; these files only go to the published page.
"""
import argparse, base64, json, pathlib, sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chotu.config import TOKENIZER_ID, tiny_config  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer, Qwen3ForCausalLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
D, O = pathlib.Path(a.dir), pathlib.Path(a.out)
O.mkdir(parents=True, exist_ok=True)

t = json.load(open(hf_hub_download(TOKENIZER_ID, "tokenizer.json")))
vocab = [None] * len(t["model"]["vocab"])
for s, i in t["model"]["vocab"].items():
    vocab[i] = s
json.dump({"vocab": vocab, "merges": t["model"]["merges"]}, open(O / "tokenizer.json", "w"), separators=(",", ":"))

cfg = tiny_config()
spec = dict(hidden=cfg.hidden_size, layers=cfg.num_hidden_layers, heads=cfg.num_attention_heads, kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, ffn=cfg.intermediate_size, vocab=cfg.vocab_size, rope_theta=cfg.rope_parameters["rope_theta"], eps=cfg.rms_norm_eps)
names = ["model.embed_tokens.weight"]
for i in range(cfg.num_hidden_layers):
    p = f"model.layers.{i}."
    names += [p + n for n in ("input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                              "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.o_proj.weight",
                              "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")]
names.append("model.norm.weight")
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
test = "Photosynthesis is the process by which plants turn light into energy. In 1969, the first"
check = {"text": test, "ids": tok(test).input_ids, "models": {}}
for key, f in [("twin140", "twin-day/main/ckpt/step_002136.pt"), ("twin700", "twin-day/main-5x/ckpt/step_010681.pt")]:
    sd = {k.removeprefix("_orig_mod."): v for k, v in torch.load(D / f, map_location="cpu", weights_only=False)["model"].items()}
    m = Qwen3ForCausalLM(cfg)
    m.load_state_dict(sd)
    m.eval()
    shapes, chunks = [], []
    for n in names:
        w = m.state_dict()[n].float().numpy()
        shapes.append([n, list(w.shape)])
        chunks.append(w.astype(np.float16).ravel())
    raw = np.concatenate(chunks).tobytes()
    (O / f"{key}.bin").write_bytes(raw)
    h = len(raw) // 2 - (len(raw) // 2) % 6  # two halves for the page host, each valid base64 on its own
    for i, part in enumerate((raw[:h], raw[h:]), 1):
        (O / f"{key}.{i}.txt").write_text(base64.b64encode(part).decode())
    with torch.no_grad():  # the reference: PyTorch with the same float16-rounded weights the page uses
        for n in names:
            m.state_dict()[n].copy_(m.state_dict()[n].half().float())
        lg = m(torch.tensor([check["ids"]])).logits[0].float()
    check["models"][key] = {"last_top5": lg[-1].topk(5).indices.tolist(), "last_logits_top5": [round(x, 4) for x in lg[-1].topk(5).values.tolist()],
                            "logit_sum_abs": [round(float(r.abs().sum()), 2) for r in lg]}
    print(key, (O / f"{key}.bin").stat().st_size // 1_000_000, "MB")
json.dump({"spec": spec, "tensors": shapes}, open(O / "manifest.json", "w"))
json.dump(check, open(O / "check.json", "w"))
print("tokenizer", (O / "tokenizer.json").stat().st_size // 1000, "KB")
