"""Talk to the 7M twins: give them the start of a sentence, they continue it.

    python scripts/talk.py                              # type prompts; both twins answer side by side
    python scripts/talk.py "Once upon a time"
    python scripts/talk.py --twin 700 --temp 0 "The capital of France is"

Twins (private HF repo magnateadi/chotu-twin-day, needs HF_TOKEN in .env):
    140   2026-11-07, read 140M tokens (twin-day/main, step 2,136)
    700   2026-11-11, read 700M tokens (twin-day/main-5x, step 10,681)
Options: --twin 140|700|both  --ckpt <file.pt> (instead of HF)  --tokens 60  --temp 0.8 (0 = always the top guess)
         --rep 1.2 (repetition penalty: above 1 discourages words already used; 1 = off)  --seed 7
The same page runs in the browser: "Talk to the Twins" (learning/twin_5x/playground/).
"""
import argparse
import pathlib
import sys

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from chotu.config import TOKENIZER_ID, tiny_config  # noqa: E402

HERE = pathlib.Path(__file__).parent
CKPTS = {"140": "twin-day/main/ckpt/step_002136.pt", "700": "twin-day/main-5x/ckpt/step_010681.pt"}
ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="*")
ap.add_argument("--twin", choices=["140", "700", "both"], default="both")
ap.add_argument("--ckpt", default="", help="local .pt instead of the HF checkpoint (with --twin 140 or 700)")
ap.add_argument("--tokens", type=int, default=60)
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--rep", type=float, default=1.2, help="repetition penalty (1 = off)")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
names = ["140", "700"] if a.twin == "both" else [a.twin]


def load(name):
    path = a.ckpt
    if not path:
        from dotenv import load_dotenv
        from huggingface_hub import hf_hub_download
        load_dotenv(HERE.parent / ".env")
        path = hf_hub_download("magnateadi/chotu-twin-day", CKPTS[name])
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    m = Qwen3ForCausalLM(tiny_config())
    m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    return m.to(dev).eval()


models = {n: load(n) for n in names}
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)


@torch.no_grad()
def continue_text(model, prompt):
    g = torch.Generator().manual_seed(a.seed)  # same seed for both twins, so differences come from what they learned
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    start = ids.shape[1]
    for _ in range(a.tokens):
        logits = model(ids[:, -2048:]).logits[0, -1].float().cpu()
        if a.rep != 1:  # repetition penalty: make tokens it already used less likely
            used = ids[0].unique().cpu()
            logits[used] = torch.where(logits[used] > 0, logits[used] / a.rep, logits[used] * a.rep)
        if a.temp <= 0:
            nxt = logits.argmax()
        else:
            top = logits.topk(40)
            nxt = top.indices[torch.multinomial(torch.softmax(top.values / a.temp, -1), 1, generator=g)[0]]
        ids = torch.cat([ids, nxt.view(1, 1).to(dev)], 1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0, start:])


def answer(prompt):
    for n, m in models.items():
        print(f"\n[{n}M twin] {prompt}{continue_text(m, prompt)}")
    print()


if a.prompt:
    answer(" ".join(a.prompt))
else:
    print(f"Chotu twins ({' and '.join(n + 'M' for n in names)} tokens). Type the start of a sentence; empty line quits.")
    while (p := input("> ").strip()):
        answer(p)
