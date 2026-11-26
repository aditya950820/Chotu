"""Compare Chotu before and after chat tuning, side by side, on the Mac's CPU (free).

    .venv/bin/python tools/playground.py
    open http://127.0.0.1:7860

    --base <hf folder>        pretrained model   (default: newest runs/pretrain/hf/* under ~/chotu-models)
    --chat <hf folder | .pt>  chat-tuned weights (default: newest runs/chat checkpoint or hf folder there)

Modes (both models get the same question; each gets the prompt it can use, shown on the page):
  Chat      chat model: ChatML turn. Base model: two example Q/A pairs, then the question.
  Continue  both continue raw text.
  Tools     both see a web_search and a get_weather tool. A valid tool call is really executed
            (web_search = Wikipedia search API, get_weather = Open-Meteo; no keys, no account),
            then the chat model answers from the results in a second turn. Only the search words /
            city name leave this Mac. The base model's raw output is parsed the same way.
Standard-library server; the page streams tokens as they are generated.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chotu.chat_format import parse_tool_calls, render_conversation  # noqa: E402

AGENT = "ChotuPlayground/0.2 (local learning tool)"
QA_PREFIX = (
    "Question: What is the boiling point of water at sea level?\n"
    "Answer: Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.\n\n"
    "Question: Who wrote Romeo and Juliet?\n"
    "Answer: William Shakespeare wrote Romeo and Juliet.\n\n"
    "Question: {q}\nAnswer:"
)
TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web and return the top results",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
]
WEATHER_CODES = {0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
                 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
                 65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
                 81: "rain showers", 82: "violent rain showers", 95: "thunderstorm", 96: "thunderstorm", 99: "thunderstorm"}


# ---------------------------------------------------------------- tools (really executed)

def fetch_json(url):
    with urlopen(Request(url, headers={"User-Agent": AGENT}), timeout=15) as response:
        return json.loads(response.read())


def web_search(query, n=3):
    """Wikipedia full-text search, then the intro of each hit."""
    api = "https://en.wikipedia.org/w/api.php?"
    hits = fetch_json(api + urlencode(dict(action="query", list="search", srsearch=query, srlimit=n,
                                           format="json")))["query"]["search"]
    if not hits:
        return []
    pages = fetch_json(api + urlencode(dict(action="query", prop="extracts", exintro=1, explaintext=1,
                                            titles="|".join(h["title"] for h in hits), format="json")))["query"]["pages"]
    intro = {p["title"]: p.get("extract", "") for p in pages.values()}
    return [dict(title=h["title"], url="https://en.wikipedia.org/wiki/" + h["title"].replace(" ", "_"),
                 text=re.sub(r"\s+", " ", intro.get(h["title"], ""))[:900]) for h in hits]


def get_weather(city, unit="celsius"):
    unit = unit if unit in ("celsius", "fahrenheit") else "celsius"
    place = fetch_json("https://geocoding-api.open-meteo.com/v1/search?" + urlencode(dict(name=city, count=1)))
    if not place.get("results"):
        return {"error": f"no place called {city!r}"}
    p = place["results"][0]
    now = fetch_json("https://api.open-meteo.com/v1/forecast?" + urlencode(dict(
        latitude=p["latitude"], longitude=p["longitude"], temperature_unit=unit,
        current="temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m")))["current"]
    return dict(place=f"{p['name']}, {p.get('country', '')}", time=now["time"],
                temperature=f"{now['temperature_2m']} °{'C' if unit == 'celsius' else 'F'}",
                humidity=f"{now['relative_humidity_2m']}%", wind=f"{now['wind_speed_10m']} km/h",
                sky=WEATHER_CODES.get(now["weather_code"], f"code {now['weather_code']}"))


def run_tool(call):
    args = call["arguments"]
    try:
        if call["name"] == "web_search" and isinstance(args.get("query"), str):
            return {"results": web_search(args["query"])}
        if call["name"] == "get_weather" and isinstance(args.get("city"), str):
            return get_weather(args["city"], args.get("unit", "celsius"))
        return {"error": f"unknown tool or bad arguments: {call['name']}"}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"[:300]}


def results_as_text(calls, results):
    lines = []
    for call, result in zip(calls, results):
        if call["name"] == "web_search":
            lines.append(f'Search results for "{call["arguments"].get("query")}":')
            for i, r in enumerate(result.get("results", []), 1):
                lines.append(f"[{i}] {r['title']}: {r['text']}")
            if result.get("error") or not result.get("results"):
                lines.append(result.get("error", "no results"))
        else:
            lines.append(f"Weather result: {json.dumps(result, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- models

def newest(pattern, root="~/chotu-models"):
    found = sorted(Path(root).expanduser().rglob(pattern), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


class Models:
    def __init__(self, base_dir, chat_path):
        self.tok = AutoTokenizer.from_pretrained(base_dir)
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.models, self.locks, self.labels = {}, {}, {}
        self.models["base"] = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).eval()
        self.labels["base"] = f"pretrained only ({Path(base_dir).name})"
        if chat_path:
            chat_path = Path(chat_path)
            if chat_path.suffix == ".pt":  # raw training checkpoint: load just the weights
                model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
                saved = torch.load(chat_path, map_location="cpu", weights_only=False)
                model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in saved["model"].items()})
                self.labels["chat"] = f"chat-tuned, step {saved.get('step', '?'):,}"
                del saved
            else:
                model = AutoModelForCausalLM.from_pretrained(chat_path, dtype=torch.float32)
                self.labels["chat"] = f"chat-tuned ({chat_path.name})"
            self.models["chat"] = model.eval()
        self.locks = {k: threading.Lock() for k in self.models}

    def generate(self, key, prompt, max_new, temperature, stops=None, chat=False):
        """Yields text pieces; the last item is a stats dict."""
        ids = self.tok(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**ids, max_new_tokens=max(1, min(int(max_new), 512)), streamer=streamer,
                      repetition_penalty=1.1, pad_token_id=self.end)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        if chat:
            kwargs["eos_token_id"] = self.end
        if stops:
            kwargs.update(stop_strings=stops, tokenizer=self.tok)
        with self.locks[key]:
            start, text = time.time(), ""
            worker = threading.Thread(target=self.models[key].generate, kwargs=kwargs, daemon=True)
            worker.start()
            for piece in streamer:
                text += piece
                yield piece
            worker.join()
            seconds = time.time() - start
            tokens = len(self.tok(text)["input_ids"])
            yield {"tokens": tokens, "seconds": seconds, "prompt_tokens": ids["input_ids"].shape[1]}


def trim_stops(text, stops):
    cut = min([text.find(s) for s in stops if s in text] or [len(text)])
    return text[:cut]


def events(models, key, mode, text, temperature, max_new):
    """One model's whole turn as page events."""
    send = lambda **kw: kw  # noqa: E731
    if mode == "continue":
        prompt, stops, chat = text, None, False
    elif key == "base" and mode == "chat":
        prompt, stops, chat = QA_PREFIX.format(q=text), ["\n\nQuestion:", "\nQuestion:"], False
    else:
        tools = TOOLS if mode == "tools" else None
        prompt = render_conversation([{"role": "user", "content": text}], tools) + "<|im_start|>assistant\n"
        stops, chat = (["<|im_end|>", "<|im_start|>"] if key == "base" else None), True
    yield send(ev="prompt", text=prompt)
    shown, stats = "", None
    for piece in models.generate(key, prompt, max_new, temperature, stops, chat):
        if isinstance(piece, dict):
            stats = piece
            continue
        shown += piece
        yield send(ev="text", t=piece)
    if stops:
        shown = trim_stops(shown, stops)
        yield send(ev="final", text=shown)  # drop the stop marker that was already streamed
    yield send(ev="stats", **stats)
    if mode != "tools":
        return
    calls, errors = parse_tool_calls(shown)
    yield send(ev="parsed", calls=calls, errors=errors[:3])
    if not calls:
        return
    results = []
    for call in calls[:3]:
        result = run_tool(call)
        results.append(result)
        yield send(ev="result", call=call, result=result)
    if key != "chat":
        return
    # Our tool training data never shows results coming back, so hand them over as a plain
    # user turn and ask for an answer from them (summarising is something it was trained on).
    answer_prompt = render_conversation([{"role": "user", "content":
        results_as_text(calls[:3], results) + f"\nUsing only the results above, answer this: {text}"}]) + "<|im_start|>assistant\n"
    yield send(ev="answer_start", prompt=answer_prompt)
    for piece in models.generate(key, answer_prompt, max(max_new, 160), temperature, None, True):
        if isinstance(piece, dict):
            yield send(ev="answer_stats", **piece)
        else:
            yield send(ev="answer", t=piece)


# ---------------------------------------------------------------- page

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chotu Playground</title><style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mute:#6f6d66;--line:#e3e1da;--accent:#2f6fde;--base:#8a5a00;--basebg:#fbf0dc;--chat:#0f5c36;--chatbg:#e5f3ea;--code:#f1efe9;--warn:#a33}
@media (prefers-color-scheme:dark){:root{--bg:#151513;--card:#1e1e1c;--ink:#ecebe6;--mute:#9b998f;--line:#34332f;--accent:#7aa7ff;--base:#f0c070;--basebg:#3a2c12;--chat:#9be3b4;--chatbg:#173223;--code:#262522;--warn:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:22px 16px 60px}h1{font-size:22px;margin:0 0 2px}.sub{color:var(--mute);margin:0 0 16px;font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.modes{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:7px 14px;cursor:pointer}
button.on{border-color:var(--accent);color:var(--accent);font-weight:600}button.go{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.explain{font-size:13.5px;color:var(--mute);margin:4px 0 10px}
textarea{width:100%;min-height:84px;font:inherit;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);resize:vertical}
.row{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin-top:10px;font-size:14px;color:var(--mute)}
.row label{display:flex;gap:6px;align-items:center}input[type=range]{width:110px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.chips button{font-size:13px;padding:4px 10px;color:var(--mute)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media (max-width:820px){.cols{grid-template-columns:1fr}}
.col h2{font-size:15px;margin:0 0 2px}.col .tag{font-size:12.5px;color:var(--mute);margin-bottom:10px}
.base h2{color:var(--base)}.chat h2{color:var(--chat)}
.out{white-space:pre-wrap;word-wrap:break-word;min-height:48px}.base .g{background:var(--basebg)}.chat .g{background:var(--chatbg)}
.g{border-radius:3px}.stat{font-size:12.5px;color:var(--mute);margin-top:6px}
details{margin-top:10px;font-size:12.5px;color:var(--mute)}details pre{white-space:pre-wrap;background:var(--code);padding:8px;border-radius:6px;max-height:260px;overflow:auto;color:var(--ink)}
.step{margin-top:12px;padding-top:10px;border-top:1px dashed var(--line)}.step b{font-size:13px}
.call{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:var(--code);padding:6px 8px;border-radius:6px;margin-top:4px;white-space:pre-wrap;word-break:break-all}
.res{font-size:13px;margin-top:4px}.res a{color:var(--accent)}.bad{color:var(--warn);font-size:13px}.ok{color:var(--chat);font-size:13px}
.note{font-size:13px;color:var(--mute)}
</style></head><body><main>
<h1>Chotu: before and after chat tuning</h1><p class="sub">Same question, two versions of our 181M model, running on this Mac's CPU.</p>
<div class="card"><div class="modes"><button data-m="chat" class="on">Chat</button><button data-m="tools">Tools + web search</button><button data-m="continue">Continue text</button></div>
<div class="explain" id="explain"></div>
<textarea id="t"></textarea><div class="chips" id="chips"></div>
<div class="row"><label>Temperature <input id="temp" type="range" min="0" max="1.2" step="0.05" value="0.3"><span id="tv">0.30</span></label>
<label>Max tokens <input id="mx" type="range" min="20" max="400" step="10" value="160"><span id="mv">160</span></label>
<button class="go" id="go">Ask both</button><button id="stop">Stop</button></div></div>
<div class="cols">
 <div class="card col base"><h2>Before: pretrained only</h2><div class="tag">BASE_LABEL</div><div class="out" id="o-base"><span class="note">waiting…</span></div><div class="stat" id="s-base"></div><div id="x-base"></div></div>
 <div class="card col chat"><h2>After: chat-tuned</h2><div class="tag">CHAT_LABEL</div><div class="out" id="o-chat"><span class="note">waiting…</span></div><div class="stat" id="s-chat"></div><div id="x-chat"></div></div>
</div>
<p class="note">Highlighted text is what the model wrote. "Exact prompt" shows the tokens each model really saw. Tools really run: web_search queries Wikipedia, get_weather queries Open-Meteo; only your search words or city leave this Mac.</p>
</main><script>
const $=id=>document.getElementById(id);let mode="chat",ctls=[];
const EX={chat:["What is the capital of Japan?","Give me three tips for studying better.","Why is the sky blue?","Write a two-line poem about the monsoon."],
tools:["Search the web for ISRO's Gaganyaan mission.","What's the weather in Noida right now?","Who founded the Indian Space Research Organisation?","Is it raining in Mumbai, in celsius?"],
continue:["The water cycle is the process by which","In 1969, the first humans to walk on the Moon","A good way to learn programming is"]};
const EXPLAIN={chat:"Before: the base model only continues text, so it gets two example Q&A pairs first (its best shot). After: the chat model gets a real chat turn and stops on its own. Watch which one answers and stops.",
tools:"Both see the same two tools in the chat format. A valid tool call is really executed; the chat model then answers from the results in a second turn. The base model never saw tool calls in training. What does it write instead? Temperature starts at 0 (always the most likely token, like our tool evals); raise it and watch tool choice get less reliable.",
continue:"Both models get the raw text and just continue it. Did chat tuning keep the base model's writing ability, or change its style?"};
function setMode(m){mode=m;document.querySelectorAll(".modes button").forEach(b=>b.classList.toggle("on",b.dataset.m==m));$("explain").textContent=EXPLAIN[m];
$("t").placeholder=m=="continue"?"Start a passage…":"Ask something…";$("chips").innerHTML="";
EX[m].forEach(s=>{const b=document.createElement("button");b.textContent=s;b.onclick=()=>{$("t").value=s;run()};$("chips").appendChild(b)});
const t=m=="continue"?0.7:m=="tools"?0:0.3;$("temp").value=t;$("tv").textContent=t.toFixed(2)}
document.querySelectorAll(".modes button").forEach(b=>b.onclick=()=>setMode(b.dataset.m));
$("temp").oninput=e=>$("tv").textContent=(+e.target.value).toFixed(2);$("mx").oninput=e=>$("mv").textContent=e.target.value;
$("stop").onclick=()=>ctls.forEach(c=>c.abort());$("go").onclick=run;$("t").addEventListener("keydown",e=>{if(e.key=="Enter"&&(e.metaKey||e.ctrlKey))run()});
const el=(tag,cls,txt)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(txt!==undefined)e.textContent=txt;return e};
function details(label,text){const d=el("details");d.append(el("summary",null,label),el("pre",null,text));return d}
function stat(s){return `${s.tokens} tokens · ${(s.tokens/Math.max(s.seconds,1e-6)).toFixed(0)} tok/s · ${s.seconds.toFixed(1)} s · prompt ${s.prompt_tokens} tokens`}
async function one(key,text,ctl){const out=$("o-"+key),x=$("x-"+key);out.innerHTML="";x.innerHTML="";$("s-"+key).textContent="thinking…";
let g=el("span","g");out.append(g);let ans=null;
try{const r=await fetch("api/generate",{method:"POST",signal:ctl.signal,body:JSON.stringify({model:key,mode,text,temperature:+$("temp").value,max_new_tokens:+$("mx").value})});
if(!r.ok){$("s-"+key).textContent=await r.text();return}
const rd=r.body.getReader(),dec=new TextDecoder();let buf="";
for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;
while((i=buf.indexOf("\n"))>=0){const line=buf.slice(0,i);buf=buf.slice(i+1);if(!line)continue;const m=JSON.parse(line);
if(m.ev=="prompt")x.append(details("Exact prompt",m.text));
else if(m.ev=="text")g.textContent+=m.t;
else if(m.ev=="final")g.textContent=m.text;
else if(m.ev=="stats")$("s-"+key).textContent=stat(m);
else if(m.ev=="parsed"){const s=el("div","step");s.append(el("b",null,"1. Tool call?"));
 if(m.calls.length)m.calls.forEach(c=>s.append(el("div","call",JSON.stringify(c))));else s.append(el("div","bad","No valid tool call in this output"+(m.errors.length?" ("+m.errors[0]+")":"")+"."));x.append(s);
 if(m.calls.length){const s2=el("div","step");s2.id="run-"+key;s2.append(el("b",null,"2. Ran it for real"));x.append(s2)}}
else if(m.ev=="result"){const s2=$("run-"+key);const r=m.result;
 if(r.results){r.results.forEach(h=>{const d=el("div","res");const a=el("a",null,h.title);a.href=h.url;a.target="_blank";d.append(a,document.createTextNode(": "+h.text.slice(0,220)+"…"));s2.append(d)});if(!r.results.length)s2.append(el("div","bad","no results"))}
 else if(r.error)s2.append(el("div","bad",r.error));else s2.append(el("div","res",Object.entries(r).map(([k,v])=>k+": "+v).join(" · ")))}
else if(m.ev=="answer_start"){const s3=el("div","step");s3.append(el("b",null,"3. Chotu's answer from the results"));ans=el("div","out");const ag=el("span","g");ans.append(ag);ans.g=ag;s3.append(ans,details("Exact prompt for this turn",m.prompt));x.append(s3)}
else if(m.ev=="answer")ans.g.textContent+=m.t;
else if(m.ev=="answer_stats")x.append(el("div","stat","answer: "+stat(m)));
else if(m.ev=="error")$("s-"+key).textContent="error: "+m.message}}}
catch(e){$("s-"+key).textContent=e.name=="AbortError"?"stopped":"error: "+e}}
function run(){const text=$("t").value.trim();if(!text)return;ctls.forEach(c=>c.abort());ctls=[new AbortController(),new AbortController()];
one("base",text,ctls[0]);one("chat",text,ctls[1])}
setMode("chat");
</script></body></html>"""


def handler(models):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = (PAGE.replace("BASE_LABEL", models.labels["base"])
                    .replace("CHAT_LABEL", models.labels.get("chat", "no chat weights found"))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.rstrip("/") != "/api/generate":
                return self.send_error(404)
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            key, mode = req.get("model"), req.get("mode")
            if key not in models.models or mode not in ("chat", "tools", "continue"):
                return self.send_error(400, "unknown model or mode")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                for event in events(models, key, mode, str(req.get("text", "")),
                                    float(req.get("temperature", 0.3)), req.get("max_new_tokens", 160)):
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # Stop pressed
            except Exception as error:
                self.wfile.write((json.dumps({"ev": "error", "message": f"{type(error).__name__}: {error}"}) + "\n").encode())

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, help="pretrained HF folder")
    ap.add_argument("--chat", type=Path, help="chat-tuned HF folder or training checkpoint .pt")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    base = a.base
    if base is None:
        found = newest("runs/pretrain/hf/*/model.safetensors")
        if not found:
            raise SystemExit("no base model under ~/chotu-models; pass --base <hf folder>")
        base = found.parent
    chat = a.chat
    if chat is None:
        hf = newest("runs/chat/hf/*/model.safetensors")
        chat = hf.parent if hf else newest("runs/chat/checkpoint-*.pt")
    models = Models(base, chat)
    print(f"Chotu playground: http://127.0.0.1:{a.port}\n  base: {base}\n  chat: {chat}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(models)).serve_forever()


if __name__ == "__main__":
    main()
