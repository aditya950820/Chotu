"""Talk to the twin-day model (7M twin, 2026-11-07): give it the start of a sentence, it continues it.

    python scripts/talk.py                 # type prompts
    python scripts/talk.py "Once upon a time"

Options: --ckpt <file.pt>  --tokens 60  --temp 0.8 (0 = always the top guess)
"""
import argparse
import pathlib
import sys

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from chotu.config import TOKENIZER_ID, tiny_config  # noqa: E402

HERE = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="*")
ap.add_argument("--ckpt", default="", help="local .pt; default: final twin-day checkpoint from HF")
ap.add_argument("--tokens", type=int, default=60)
ap.add_argument("--temp", type=float, default=0.8)
a = ap.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
if not a.ckpt:  # private HF repo; needs HF_TOKEN in .env
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    load_dotenv(HERE.parent / ".env")
    a.ckpt = hf_hub_download("magnateadi/chotu-twin-day", "twin-day/main/ckpt/step_002136.pt")
state = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
model = Qwen3ForCausalLM(tiny_config())
model.load_state_dict(state)
model.to(dev).eval()
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)


@torch.no_grad()
def continue_text(prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    for _ in range(a.tokens):
        logits = model(ids[:, -2048:]).logits[:, -1].float()
        if a.temp <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(torch.softmax(logits / a.temp, -1), 1)
        ids = torch.cat([ids, nxt], 1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0])


if a.prompt:
    print(continue_text(" ".join(a.prompt)))
else:
    print("Chotu twin (7M). Type the start of a sentence; empty line quits.")
    while (p := input("> ").strip()):
        print(continue_text(p), "\n")
