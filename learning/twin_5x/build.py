"""Build the "Five Times the Reading" page from learning/twin_5x/results.json (made by analyze.py).

    python learning/twin_5x/build.py <out.html>
"""
import json, pathlib, sys

here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here.parents[1]))
from chotu.config import chotu_config, tiny_config  # noqa: E402
from transformers import Qwen3ForCausalLM  # noqa: E402

R = json.load(open(here / "results.json"))
split = []  # how each model's knobs divide between the word list (embeddings) and the thinking layers
for name, cfg in [("twin (7M)", tiny_config()), ("Chotu (181M)", chotu_config())]:
    m = Qwen3ForCausalLM(cfg)
    total = sum(p.numel() for p in m.parameters())
    emb = m.model.embed_tokens.weight.numel()
    split.append(dict(model=name, total=total, embeddings=emb, layers=total - emb))
R["split"] = split


def rnd(o):
    if isinstance(o, float):
        return float(f"{o:.4g}")
    if isinstance(o, list):
        return [rnd(x) for x in o]
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    return o


page = (here / "template.html").read_text().replace("__DATA__", json.dumps(rnd(R)))
pathlib.Path(sys.argv[1]).write_text(page)
print("wrote", sys.argv[1], len(page), "bytes")
