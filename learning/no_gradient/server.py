"""Side by side, live: the no-gradient model (suffix-array counts + document memory + copying) and the 700M twin.

    python learning/no_gradient/server.py --shard <train_000.bin> --twin <step_010681.pt> [--tokens 300e6] [--port 8800]
then open http://localhost:8800. Only the standard library serves the page; both models stay loaded in memory.
"""
import argparse, json, pathlib, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import gen  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", required=True)
ap.add_argument("--twin", required=True)
ap.add_argument("--tokens", type=float, default=300e6)
ap.add_argument("--port", type=int, default=8800)
a = ap.parse_args()

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen3ForCausalLM  # noqa: E402
from chotu.config import TOKENIZER_ID, tiny_config  # noqa: E402

gen.build(a.shard, a.tokens)
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
torch.manual_seed(1234)
twin = Qwen3ForCausalLM(tiny_config())
sd = torch.load(a.twin, map_location="cpu", weights_only=False)["model"]
twin.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})
twin.eval()
LOCK = threading.Lock()
PAGE = (HERE / "server_page.html").read_text()


def twin_write(ids, n, temp, rep, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([ids])
    out = []
    with torch.inference_mode():
        for _ in range(n):
            lg = twin(x[:, -2048:]).logits[0, -1].float()
            p0 = torch.softmax(lg, -1)
            used = x[0].unique()
            lg[used] = torch.where(lg[used] > 0, lg[used] / rep, lg[used] * rep)
            if temp == 0:
                w = lg.argmax()
            else:
                top = lg.topk(40)
                w = top.indices[torch.multinomial(torch.softmax(top.values / temp, -1), 1, generator=g)[0]]
            out.append(dict(t=tok.decode([int(w)]), p=float(p0[w])))
            x = torch.cat([x, w.view(1, 1)], 1)
            if int(w) == 0:
                break
    return out


def guesses(ids):
    p = gen.next_dist(list(ids))
    top = np.argsort(-p)[:8]
    ng = [dict(t=tok.decode([int(i)]), p=float(p[i])) for i in top]
    with torch.inference_mode():
        pt = torch.softmax(twin(torch.tensor([ids])).logits[0, -1].float(), -1)
    tw = [dict(t=tok.decode([int(i)]), p=float(pt[i])) for i in pt.topk(8).indices]
    return ng, tw


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, ctype):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, PAGE.replace("__TOKENS__", f"{len(gen.T) / 1e6:,.0f}M"), "text/html; charset=utf-8")
        else:
            self.send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            prompt = str(req.get("prompt", ""))[:2000]
            ids = tok(prompt).input_ids
            if not ids:
                return self.send(400, json.dumps(dict(error="Type a prompt first.")), "application/json")
            n = max(5, min(120, int(req.get("length", 50))))
            temp = max(0.0, min(1.5, float(req.get("temp", 0.8))))
            rep = max(1.0, min(2.0, float(req.get("rep", 1.2))))
            seed = int(req.get("seed", 1))
            with LOCK:  # one generation at a time: both models share this machine's CPU
                if self.path == "/api/write":
                    det = []
                    toks = gen.write(ids, n, temp=temp, rep=rep, seed=seed, details=det)
                    ng = [dict(t=tok.decode([t]), **d) for t, d in zip(toks, det)]
                    body = dict(nograd=ng, twin=twin_write(ids, n, temp, rep, seed))
                elif self.path == "/api/next":
                    ng, tw = guesses(ids)
                    body = dict(nograd=ng, twin=tw)
                else:
                    return self.send(404, "{}", "application/json")
            self.send(200, json.dumps(body), "application/json")
        except Exception as e:  # show the problem on the page instead of hanging
            self.send(500, json.dumps(dict(error=f"{type(e).__name__}: {e}")), "application/json")


print(f"ready: http://localhost:{a.port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
