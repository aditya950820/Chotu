"""Try a Chotu checkpoint in the browser, on the Mac, for free (CPU).

    .venv/bin/python tools/playground.py                      # newest model in ~/chotu-models
    .venv/bin/python tools/playground.py --model <hf folder>  # any exported HF folder
    open http://127.0.0.1:7860

Two modes:
  Continue  the base model's real skill: it continues whatever text you start.
  Q&A       wraps your question in a few example Question/Answer pairs and stops at the end of
            the answer line, so a base model (not chat-tuned yet) gives one short answer.
Standard library server only; the page streams tokens as they are generated.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

QA_PREFIX = (
    "Question: What is the boiling point of water at sea level?\n"
    "Answer: Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.\n\n"
    "Question: Who wrote Romeo and Juliet?\n"
    "Answer: William Shakespeare wrote Romeo and Juliet.\n\n"
    "Question: {q}\nAnswer:"
)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chotu Playground</title><style>
:root{--bg:#fafaf7;--card:#fff;--ink:#1d1d1b;--mute:#77756f;--line:#e4e2dc;--accent:#2f6fde;--gen:#0f5132;--genbg:#e8f4ec}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--mute:#9a988f;--line:#33322e;--accent:#7aa7ff;--gen:#9be3b4;--genbg:#183324}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mute);margin:0 0 18px;font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.modes{display:flex;gap:8px;margin-bottom:10px}.modes button{flex:0 0 auto}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:7px 14px;cursor:pointer}
button.on{border-color:var(--accent);color:var(--accent);font-weight:600}button.go{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
textarea{width:100%;min-height:110px;font:inherit;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);resize:vertical}
.row{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin-top:10px;font-size:14px;color:var(--mute)}
.row label{display:flex;gap:6px;align-items:center}input[type=range]{width:110px}
#out{white-space:pre-wrap;min-height:60px}#out .p{color:var(--mute)}#out .g{background:var(--genbg);color:var(--gen);border-radius:3px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.chips button{font-size:13px;padding:4px 10px;color:var(--mute)}
#stat{font-size:13px;color:var(--mute);margin-top:8px}.note{font-size:13px;color:var(--mute)}
</style></head><body><main>
<h1>Chotu playground</h1><p class="sub">MODEL_NAME · base model (pretrained, not chat-tuned) · running on this Mac's CPU</p>
<div class="card"><div class="modes"><button id="mc" class="on">Continue text</button><button id="mq">Q&amp;A</button></div>
<textarea id="t" placeholder="Start a passage and Chotu continues it…"></textarea>
<div class="chips" id="chips"></div>
<div class="row"><label>Temperature <input id="temp" type="range" min="0" max="1.2" step="0.05" value="0.7"><span id="tv">0.70</span></label>
<label>Max tokens <input id="mx" type="range" min="10" max="400" step="10" value="120"><span id="mv">120</span></label>
<button class="go" id="go">Generate</button><button id="stop">Stop</button></div></div>
<div class="card"><div id="out"><span class="note">Output appears here. Grey is what you wrote, green is Chotu.</span></div><div id="stat"></div></div>
<p class="note">Tip: a base model finishes text; it does not follow instructions. Write the start of a passage, or use Q&amp;A mode for one short answer.</p>
</main><script>
let mode="continue",ctl=null;const $=id=>document.getElementById(id);
const ex={continue:["The water cycle is the process by which","In 1969, the first humans to walk on the Moon","A good way to learn programming is","The main causes of the French Revolution were"],
qa:["What is the capital of Japan?","Why is the sky blue?","What do plants need to grow?","How many legs does a spider have?"]};
function chips(){$("chips").innerHTML="";ex[mode].forEach(s=>{const b=document.createElement("button");b.textContent=s;b.onclick=()=>{$("t").value=s;run()};$("chips").appendChild(b)})}
function setMode(m){mode=m;$("mc").classList.toggle("on",m=="continue");$("mq").classList.toggle("on",m=="qa");
$("t").placeholder=m=="qa"?"Ask a short factual question…":"Start a passage and Chotu continues it…";if(m=="qa"){$("mx").value=60;$("mv").textContent=60;$("temp").value=0.3;$("tv").textContent="0.30"}chips()}
$("mc").onclick=()=>setMode("continue");$("mq").onclick=()=>setMode("qa");
$("temp").oninput=e=>$("tv").textContent=(+e.target.value).toFixed(2);$("mx").oninput=e=>$("mv").textContent=e.target.value;
$("stop").onclick=()=>ctl&&ctl.abort();
$("t").addEventListener("keydown",e=>{if(e.key=="Enter"&&(e.metaKey||e.ctrlKey))run()});$("go").onclick=run;
async function run(){const text=$("t").value.trim();if(!text)return;if(ctl)ctl.abort();ctl=new AbortController();
const out=$("out");out.innerHTML="";const p=document.createElement("span");p.className="p";p.textContent=mode=="qa"?"Q: "+text+"\\nA:":text;
const g=document.createElement("span");g.className="g";out.append(p,g);$("stat").textContent="thinking…";const t0=performance.now();let n=0;
try{const r=await fetch("api/generate",{method:"POST",signal:ctl.signal,body:JSON.stringify({text,mode,temperature:+$("temp").value,max_new_tokens:+$("mx").value})});
const rd=r.body.getReader(),dec=new TextDecoder();let buf="";
for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;
while((i=buf.indexOf("\\n"))>=0){const line=buf.slice(0,i);buf=buf.slice(i+1);if(!line)continue;const m=JSON.parse(line);
if(m.t!==undefined){g.textContent+=m.t;n++}if(m.done)$("stat").textContent=`${m.tokens} tokens · ${m.tok_s.toFixed(0)} tok/s · ${m.seconds.toFixed(1)} s`;}}}
catch(e){$("stat").textContent=e.name=="AbortError"?"stopped":"error: "+e}}
chips();
</script></body></html>"""


def newest_model():
    found = sorted(Path("~/chotu-models").expanduser().rglob("model.safetensors"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit("no model found under ~/chotu-models; pass --model <hf folder>")
    return found[-1].parent


class Playground:
    def __init__(self, model_dir):
        self.name = Path(model_dir).name
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
        self.lock = threading.Lock()  # one generation at a time on the CPU

    def stream(self, text, mode, temperature, max_new_tokens):
        prompt = QA_PREFIX.format(q=text) if mode == "qa" else text
        ids = self.tok(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**ids, max_new_tokens=max(1, min(int(max_new_tokens), 512)), streamer=streamer,
                      repetition_penalty=1.1, pad_token_id=self.tok.eos_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        if mode == "qa":
            kwargs.update(stop_strings=["\n"], tokenizer=self.tok)
        with self.lock:
            start = time.time()
            worker = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
            worker.start()
            shown = ""
            for piece in streamer:
                if mode == "qa" and "\n" in piece:
                    piece = piece.split("\n")[0]
                    if piece:
                        shown += piece
                        yield {"t": piece}
                    break
                shown += piece
                yield {"t": piece}
            worker.join()
            seconds = time.time() - start
            tokens = len(self.tok(shown)["input_ids"])
            yield {"done": True, "tokens": tokens, "seconds": seconds, "tok_s": tokens / max(seconds, 1e-6)}


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = PAGE.replace("MODEL_NAME", app.name).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.rstrip("/") != "/api/generate":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                for msg in app.stream(req.get("text", ""), req.get("mode", "continue"),
                                      float(req.get("temperature", 0.7)), req.get("max_new_tokens", 120)):
                    self.wfile.write((json.dumps(msg) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page pressed Stop

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="exported HF folder (default: newest under ~/chotu-models)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    model_dir = a.model or newest_model()
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    app = Playground(model_dir)
    print(f"Chotu playground: http://127.0.0.1:{a.port}  (model {model_dir})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(app)).serve_forever()


if __name__ == "__main__":
    main()
