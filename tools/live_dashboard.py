"""Interactive live dashboard, served from the GPU machine (works with the Mac off).

    python tools/live_dashboard.py --storage /home/chotu-data      # gpu_up.sh starts it on port 6006

Same link and key as before (<Jarvis URL>/<key>/). The original page stays at <key>/classic.
Lives in tools/ on purpose: scripts/*.py is part of the training code fingerprint, so a dashboard
change there would block resuming the run. This file only READS the run's files.

Charts (uPlot): hover for values, drag to zoom, double-click to reset, click a legend entry to hide
it, and the expand button opens any chart full screen with every data point, range buttons, log
scale and smoothing. Data refreshes every 30 s without losing your zoom.

Validation loss in nats/token is derived exactly from bits per byte:
nats = bpb * ln 2 * bytes_per_token of the fixed validation set (reports/val-bytes.json).
"""
import argparse
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import statistics
import threading
import time
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("classic_dashboard", ROOT / "scripts" / "dashboard.py")
classic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classic)

RUNS = classic.RUNS
STAGES = classic.STAGES
POST_PRETRAIN_HOURS = 2.6  # chat tuning (~1.6 h) + evals, export, upload (~1 h); an estimate
_cache = {}


def cached_rows(path):
    """metrics.jsonl parsed once per file change (pretraining writes ~19k rows)."""
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return []
    hit = _cache.get(path)
    if hit and hit[0] == stamp:
        return hit[1]
    rows = classic.read_jsonl(path)
    _cache[path] = (stamp, rows)
    return rows


def env_value(name):
    if os.getenv(name):
        return os.getenv(name)
    try:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


class Balance:
    """Jarvis balance, refreshed at most every 5 minutes, never blocking a page."""
    def __init__(self):
        self.value, self.time, self.lock = None, 0.0, threading.Lock()

    def get(self):
        if time.time() - self.time > 300 and self.lock.acquire(blocking=False):
            threading.Thread(target=self.refresh, daemon=True).start()
        return self.value

    def refresh(self):
        try:
            from jarvislabs import Client
            result = Client(api_key=env_value("JL_API_KEY")).account.balance()
            self.value = float(result["balance"] if isinstance(result, dict) else result.balance)
        except Exception:
            pass
        finally:
            self.time = time.time()
            self.lock.release()


BALANCE = Balance()


def head_lines(path, size=200_000):
    """First lines of a log (COMMAND and TAIL are printed once, at the start of a run)."""
    try:
        with open(path, "rb") as f:
            return f.read(size).decode(errors="replace").splitlines()
    except OSError:
        return []


def run_settings(storage, run):
    """Schedule settings for a run: train.py defaults, overridden by its logged COMMAND line."""
    settings = dict(warmup=700, decay_frac=0.2, long_frac=0.08, batch=524288)
    if run.startswith("lr-"):
        settings.update(warmup=100, decay_frac=0.15, long_frac=0.0)
    if run in ("speed",):
        settings.update(long_frac=0.0)
    if run == "chat":
        settings.update(warmup=20, decay_frac=None, long_frac=0.0, batch=32768)
    stage_log = {"pretrain": "pretrain", "chat": "chat_tune", "speed": "speed"}.get(run, "lr_test")
    for line in head_lines(storage / "logs" / f"{stage_log}.log"):
        if line.startswith("COMMAND") and f"--name {run} " in line + " ":
            for flag, key, cast in (("--warmup-steps", "warmup", int), ("--decay-frac", "decay_frac", float),
                                    ("--long-frac", "long_frac", float), ("--batch-tokens", "batch", int)):
                match = re.search(rf"{flag} (\S+)", line)
                if match:
                    settings[key] = cast(match.group(1))
            break
    return settings


def phases(storage, run, steps):
    if not steps:
        return {}
    s = run_settings(storage, run)
    out = dict(warmup_end=min(s["warmup"], steps))
    if s["decay_frac"]:
        out["decay_start"] = steps - max(1, math.ceil(steps * s["decay_frac"])) + 1
    if run == "pretrain" and s["long_frac"]:
        log = storage / "logs" / "pretrain.log"
        for line in head_lines(log):
            if line.startswith("TAIL "):
                tail = json.loads(line[5:])
                out["switch_8k"] = tail["planned_steps"] - tail["tail_tokens"] // s["batch"] + 1
                break
        for line in classic.tail(log, 400):  # once it happens, the logged switch is authoritative
            if line.startswith("SWITCH step="):
                out["switch_8k"] = int(line.split("=")[1].split()[0]) + 1
    return out


def series(storage, full):
    bytes_per_token = (classic.read_json(storage / "reports" / "val-bytes.json") or {}).get("bytes_per_token")
    to_nats = math.log(2) * bytes_per_token if bytes_per_token else None
    out = {}
    for run in RUNS:
        rows = cached_rows(storage / "runs" / run / "metrics.jsonl")
        if not rows:
            continue
        batch = run_settings(storage, run)["batch"]
        evals = [r for r in rows if r.get("val_bpb") is not None or r.get("val_loss") is not None]
        stride = 1 if full else max(1, len(rows) // 1500)
        keep = {id(r) for r in rows[::stride]} | {id(r) for r in evals} | {id(rows[-1])}
        picked = [r for r in rows if id(r) in keep]

        def col(key, scale=1.0):
            return [None if r.get(key) is None else r[key] * scale for r in picked]
        val_nats = [r["val_loss"] if r.get("val_loss") is not None
                    else (r["val_bpb"] * to_nats if r.get("val_bpb") is not None and to_nats else None) for r in picked]
        out[run] = dict(step=[r["step"] for r in picked], tokens_b=[r["step"] * batch / 1e9 for r in picked],
                        loss=col("loss"), val_loss=val_nats, val_bpb=col("val_bpb"), lr=col("lr"),
                        grad_norm=col("grad_norm"), tok_s=col("tokens_per_second"), mfu=col("mfu", 100),
                        seq_len=col("seq_len"), total_rows=len(rows))
    system = classic.read_jsonl(storage / "logs" / "system.jsonl")
    if system:
        stride = 1 if full else max(1, len(system) // 1500)
        rows = system[::stride]
        t0 = system[0]["time"]
        out["_system"] = dict(hours=[(r["time"] - t0) / 3600 for r in rows],
                              gpu_util=[r.get("gpu_util") for r in rows], gpu_mem=[r.get("gpu_mem_gib") for r in rows],
                              gpu_watts=[r.get("gpu_watts") for r in rows], disk_free=[r.get("disk_free_gib") for r in rows])
    return out


def summary(storage, rate, sample_seconds):
    now = time.time()
    status = classic.read_json(storage / "status.json") or {}
    system_rows = sum(1 for _ in open(storage / "logs" / "system.jsonl")) if (storage / "logs" / "system.jsonl").exists() else 0
    uptime = system_rows * sample_seconds
    stages = []
    for name in STAGES:
        marker = classic.read_json(storage / "stages" / f"{name}.done.json")
        stages.append(dict(name=name, done=bool(marker), seconds=marker and marker.get("seconds"),
                           live=status.get("stage") == name and not marker))
    runs = {}
    current = None
    for run in RUNS:
        rows = cached_rows(storage / "runs" / run / "metrics.jsonl")
        if not rows:
            continue
        plan = classic.read_json(storage / "runs" / run / "plan.json") or {}
        done = (storage / "runs" / run / "done.json").exists()
        recent = rows[-50:]
        speed = statistics.median(r["tokens_per_second"] for r in recent)
        total = plan.get("steps")
        left = (total - rows[-1]["step"]) * plan.get("batch_tokens", 524288) / speed if total and not done else 0
        checkpoints = sorted((storage / "runs" / run).glob("checkpoint-*.pt"))
        runs[run] = dict(step=rows[-1]["step"], total=total, done=done, speed=speed,
                         mfu=statistics.median(r["mfu"] for r in recent) * 100, loss=rows[-1]["loss"],
                         left_seconds=left, phases=phases(storage, run, total),
                         stalled_seconds=None if done else now - (storage / "runs" / run / "metrics.jsonl").stat().st_mtime,
                         last_checkpoint=checkpoints[-1].name if checkpoints else None,
                         last_checkpoint_age=now - checkpoints[-1].stat().st_mtime if checkpoints else None)
        if not done:
            current = run
    # Remaining money: this run, pretraining if it has not started, then the post-pretraining stages.
    remaining = runs[current]["left_seconds"] / 3600 if current else 0.0
    if "pretrain" not in runs and not (storage / "stages" / "pretrain.done.json").exists():
        speed = (classic.read_json(storage / "reports" / "speed.json") or {}).get("tokens_per_second") or 190000
        remaining += 18868 * 524288 / speed / 3600
    if not (storage / "stages" / "final_report.done.json").exists() and current != "chat":
        remaining += POST_PRETRAIN_HOURS
    events = classic.read_jsonl(storage / "logs" / "chain.jsonl", keep=5000)
    done_times = [e.get("time", "") for e in events if e.get("state") == "done"]
    alerts = [dict(time=e.get("time", "")[:19].replace("T", " "), message=e.get("message", ""),
                   resolved=any(t > e.get("time", "") for t in done_times))  # the chain moved on after it
              for e in events if e.get("state") == "ALERT"]
    if current and runs[current]["stalled_seconds"] and runs[current]["stalled_seconds"] > 600:
        alerts.append(dict(time="now", resolved=False, message=f"no new training step for {classic.duration(runs[current]['stalled_seconds'])}"))
    lr_marker = (classic.read_json(storage / "stages" / "lr_test.done.json") or {}).get("result")
    sweep = classic.read_json(storage / "reports" / "speed-sweep.json")
    evals = {p.stem: p.read_text()[:6000] for p in sorted((storage / "reports").glob("*-lm.md"))} if (storage / "reports").exists() else {}
    for name in ("final-tools.md", "final_report.md"):
        if (storage / "reports" / name).exists():
            evals[name] = (storage / "reports" / name).read_text()[:6000]
    log_name = f'logs/{status.get("stage")}.log' if status.get("stage") else "gpu_up.log"
    return dict(
        server_time=now, status=status, stages=stages, runs=runs, current=current,
        uptime_seconds=uptime, spent=uptime / 3600 * rate, rate=rate, balance=BALANCE.get(),
        remaining_hours=remaining, remaining_cost=remaining * rate,
        alerts=alerts[-6:], lr_test=lr_marker, sweep=sweep,
        val_bytes_per_token=(classic.read_json(storage / "reports" / "val-bytes.json") or {}).get("bytes_per_token"),
        backup=classic.read_json(storage / "dashboard" / "backup.json") or {},
        rehearsal=classic.read_json(storage / "rehearsal" / "status.json"),
        evals=evals,
        logs={"Setup log (gpu_up.log)": "\n".join(classic.tail(storage / "gpu_up.log", 20)),
              f"Current stage log ({log_name})": "\n".join(l for l in classic.tail(storage / log_name, 40)
                                                          if "Warning" not in l)[-6000:]})


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Chotu run</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.min.css">
<style>
:root{--bg:#f6f7fb;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--ok:#059669;--live:#2563eb;--bad:#dc2626;--warn:#d97706}
@media (prefers-color-scheme: dark){:root{--bg:#0b1120;--card:#111827;--ink:#e5e7eb;--muted:#94a3b8;--line:#1f2937}}
*{box-sizing:border-box} body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif}
h1{font-size:19px;margin:0 0 4px} h3{font-size:14px;margin:0} .muted,small{color:var(--muted)} a{color:var(--live)}
.top{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:10px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-bottom:10px}
.kpi,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.kpi b{display:block;font-size:19px;font-variant-numeric:tabular-nums} .kpi span,.kpi small{display:block;font-size:12px;color:var(--muted)}
.kpi.bad{border-color:var(--bad)} .kpi.bad b{color:var(--bad)} .kpi.warn{border-color:var(--warn)}
.bar{position:relative;height:22px;background:var(--line);border-radius:6px;margin:4px 0 12px;overflow:hidden}
.bar .fill{position:absolute;inset:0 auto 0 0;background:var(--live);opacity:.85}
.bar .mark{position:absolute;top:0;bottom:0;width:2px;background:var(--ink);opacity:.6}
.bar .lbl{position:absolute;top:3px;font-size:11px;color:var(--ink);white-space:nowrap;padding-left:4px}
.stages{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:6px;margin-bottom:12px}
.stage{border:1px solid var(--line);border-radius:8px;padding:5px 7px;font-size:12px;background:var(--card)}
.stage small{display:block} .stage.done{border-color:var(--ok)} .stage.done b{color:var(--ok)} .stage.live{border:2px solid var(--live)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:10px;margin-bottom:10px}
@media (max-width:480px){.grid{grid-template-columns:1fr}}
.card{margin-bottom:10px;min-width:0} .head{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:4px}
button{font:inherit;font-size:12px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;padding:3px 8px;cursor:pointer}
button.on{background:var(--live);color:#fff;border-color:var(--live)}
.alert{border-color:var(--bad)} .alert h3{color:var(--bad)}
pre{white-space:pre-wrap;word-break:break-word;font-size:11px;margin:0;max-height:280px;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums} td,th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left}
.win{color:var(--ok);font-weight:600}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;z-index:10;padding:16px}
.modal.open{display:block} .modal .box{background:var(--card);border-radius:12px;height:100%;padding:12px;display:flex;flex-direction:column}
.controls{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0} .controls label{font-size:12px;color:var(--muted)}
.plot{width:100%;overflow:hidden;min-height:250px} .u-legend{font-size:12px} .u-legend .u-marker{border-radius:2px} .uplot{width:100%}
.hint{font-size:11px;color:var(--muted)}
</style></head><body>
<div class="top"><div><h1>Chotu 181M · live run</h1><small id="updated">loading…</small></div>
<div><button id="xmode">x-axis: steps</button> <a href="classic">classic view</a></div></div>
<div id="nolib" class="card alert" style="display:none"><h3>Charts library did not load</h3><p>Your browser could not reach cdn.jsdelivr.net. Numbers below still update; or open the <a href="classic">classic view</a>.</p></div>
<div class="kpis" id="kpis"></div>
<div id="phasebar"></div>
<div class="stages" id="stages"></div>
<div id="alerts"></div>
<p class="hint">Charts: hover for values · drag to zoom · double-click to reset · click a legend entry to hide it · ⤢ opens full screen with every data point.</p>
<div class="grid" id="charts"></div>
<div class="grid"><div class="card" id="lrcard"></div><div class="card" id="sweepcard"></div></div>
<div class="grid"><div class="card" id="backupcard"></div><div class="card" id="evalcard"></div></div>
<div id="logs"></div>
<div class="modal" id="modal"><div class="box">
 <div class="head"><h3 id="mtitle"></h3><button id="mclose">Close ✕</button></div>
 <div class="controls"><label>Range</label><button data-r="all" class="on">All</button><button data-r="5000">Last 5k steps</button><button data-r="1000">Last 1k</button><button data-r="200">Last 200</button>
 <label style="margin-left:10px">Smoothing</label><input id="smooth" type="range" min="0" max="0.99" step="0.01" value="0"><span id="smoothv" class="hint">off</span>
 <button id="logy">log y</button></div>
 <div id="mchart" style="flex:1;min-height:0"></div><div class="hint">Full resolution. Drag to zoom, double-click to reset.</div>
</div></div>
<script src="https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.iife.min.js"></script>
<script>
const COLORS=["#2563eb","#dc2626","#059669","#d97706","#7c3aed","#0891b2","#db2777"];
let S=null, D=null, XTOK=false, charts={}, modal=null, modalDef=null, range="all", logy=false;
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const num=(v,d=0)=>v==null||isNaN(v)?"–":Number(v).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d});
const dur=s=>{if(s==null)return"–";s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}h ${String(m).padStart(2,"0")}m`:`${m}m ${String(s%60).padStart(2,"0")}s`};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// chart definitions: each returns {x:[...], ys:[{label,data,points?,dash?}], ylabel}
function xOf(r){return XTOK?r.tokens_b:r.step}
const DEFS=[
 {id:"loss",title:"Loss (nats/token): training vs validation",runs:()=>mainRuns(),build:(run,r)=>[{label:`${run} train`,data:r.loss},{label:`${run} validation`,data:r.val_loss,points:true}],phases:true},
 {id:"lrtest",title:"LR test: 1.5e-3 vs 3e-3 (loss, nats/token)",runs:()=>["lr-0.0015","lr-0.003"].filter(k=>D[k]),build:(run,r)=>[{label:`${run} train`,data:r.loss},{label:`${run} val`,data:r.val_loss,points:true}]},
 {id:"bpb",title:"Validation bits per byte (all runs)",runs:()=>Object.keys(D).filter(k=>k!=="_system"&&k!=="speed"),build:(run,r)=>[{label:run,data:r.val_bpb,points:true}]},
 {id:"tok",title:"Tokens per second",runs:()=>mainRuns(),build:(run,r)=>[{label:run,data:r.tok_s}],phases:true},
 {id:"mfu",title:"MFU %",runs:()=>mainRuns(),build:(run,r)=>[{label:run,data:r.mfu}]},
 {id:"lr",title:"Learning rate (schedule)",runs:()=>mainRuns(),build:(run,r)=>[{label:run,data:r.lr}],phases:true},
 {id:"grad",title:"Gradient norm (clipped at 1.0 for the update)",runs:()=>mainRuns(),build:(run,r)=>[{label:run,data:r.grad_norm}]},
 {id:"gpu",title:"GPU busy % (x = machine hours)",system:"gpu_util"},
 {id:"mem",title:"GPU memory GiB",system:"gpu_mem"},
 {id:"watts",title:"GPU power W",system:"gpu_watts"},
 {id:"disk",title:"Disk free GiB",system:"disk_free"},
];
function mainRuns(){const c=S&&S.current;if(c==="chat")return["chat"];if(D.pretrain)return["pretrain"];return c&&D[c]?[c]:Object.keys(D).filter(k=>k!=="_system").slice(-1)}

// Align several runs (different x) into one uPlot table: union of x values.
function table(def,data,lim){
 if(def.system){const s=data._system;if(!s)return null;return{x:s.hours,ys:[{label:def.system,data:s[def.system]}]};}
 const runs=def.runs().filter(k=>data[k]);if(!runs.length)return null;
 const xs=new Set();runs.forEach(k=>xOf(data[k]).forEach(v=>xs.add(v)));let X=[...xs].sort((a,b)=>a-b);
 if(lim&&lim!=="all"&&!XTOK){const maxStep=Math.max(...runs.map(k=>data[k].step.at(-1)));X=X.filter(v=>v>maxStep-lim);}
 const ys=[];runs.forEach(k=>{const r=data[k],idx=new Map(xOf(r).map((v,i)=>[v,i]));def.build(k,r).forEach(s=>{if(!s.data.some(v=>v!=null))return;ys.push({...s,data:X.map(v=>{const i=idx.get(v);return i==null?null:s.data[i]})})})});
 return ys.length?{x:X,ys}:null;
}
function ema(arr,a){if(!a)return arr;let m=null;return arr.map(v=>{if(v==null)return null;m=m==null?v:a*m+(1-a)*v;return m})}
function phaseHook(def){return{draw:[u=>{if(!def.phases||XTOK||!S)return;const run=mainRuns()[0],p=S.runs[run]&&S.runs[run].phases;if(!p)return;const ctx=u.ctx;
 [["warmup_end","warmup ends"],["decay_start","decay starts"],["switch_8k","8k context"]].forEach(([k,l])=>{if(!p[k])return;const x=u.valToPos(p[k],"x",true);if(x<u.bbox.left||x>u.bbox.left+u.bbox.width)return;
 ctx.save();ctx.strokeStyle=css("--muted");ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x,u.bbox.top);ctx.lineTo(x,u.bbox.top+u.bbox.height);ctx.stroke();
 ctx.fillStyle=css("--muted");ctx.font=`${11*devicePixelRatio}px system-ui`;ctx.fillText(l,x+4,u.bbox.top+12*devicePixelRatio);ctx.restore();})}]}}
function opts(def,t,w,h,big){const axis={stroke:css("--muted"),grid:{stroke:css("--line")},ticks:{stroke:css("--line")}};
 return{width:w,height:h,cursor:{drag:{x:true,y:false}},scales:{x:{time:false},y:{distr:big&&logy?3:1}},legend:{live:true},
 axes:[{...axis,label:def.system?"hours":XTOK?"tokens seen (billions)":"step"},{...axis,size:60}],hooks:phaseHook(def),
 series:[{label:def.system?"h":XTOK?"B tok":"step"},...t.ys.map((s,i)=>({label:s.label,stroke:COLORS[i%COLORS.length],width:s.points?0:1.5,
 points:{show:!!s.points,size:6,fill:COLORS[i%COLORS.length]},spanGaps:!s.points,value:(u,v)=>v==null?"–":Math.abs(v)>=1000?num(v,0):Math.abs(v)<0.01&&v!==0?v.toExponential(2):num(v,4)}))]}}
function drawCharts(){
 if(!window.uPlot){$("nolib").style.display="block";return}
 const host=$("charts");
 DEFS.forEach(def=>{const t=table(def,D,"all");let card=$("c_"+def.id);
  if(!t){if(card)card.remove();if(charts[def.id]){charts[def.id].destroy();delete charts[def.id]}return}
  if(!card){card=document.createElement("div");card.className="card";card.id="c_"+def.id;card.innerHTML=`<div class="head"><h3>${esc(def.title)}</h3><button title="expand">⤢ expand</button></div><div class="plot"></div>`;host.appendChild(card);card.querySelector("button").onclick=()=>openModal(def)}
  const plot=card.querySelector(".plot"),w=Math.max(200,Math.floor(plot.getBoundingClientRect().width)),data=[t.x,...t.ys.map(s=>s.data)];const ch=charts[def.id];
  if(ch&&ch.series.length===data.length&&ch._xtok===XTOK){ch.setData(data,false)}else{if(ch)ch.destroy();plot.innerHTML="";charts[def.id]=new uPlot(opts(def,t,w,240,false),data,plot);charts[def.id]._xtok=XTOK;
   if(!plot._ro){plot._ro=new ResizeObserver(()=>{const c=charts[def.id],nw=Math.floor(plot.getBoundingClientRect().width);if(c&&nw>0&&Math.abs(nw-c.width)>2)c.setSize({width:nw,height:240})});plot._ro.observe(plot)}}
 });
}
let FULL=null;
async function openModal(def){modalDef=def;$("mtitle").textContent=def.title+" — full resolution (loading…)";$("modal").classList.add("open");
 FULL=await (await fetch("api/series?full=1")).json();$("mtitle").textContent=def.title+" — full resolution";drawModal()}
function drawModal(){if(!modalDef||!FULL)return;const t=table(modalDef,FULL,range);if(!t)return;
 const a=+$("smooth").value;$("smoothv").textContent=a?a.toFixed(2):"off";t.ys=t.ys.map(s=>s.points?s:{...s,data:ema(s.data,a)});
 const host=$("mchart");if(modal)modal.destroy();host.innerHTML="";modal=new uPlot(opts(modalDef,t,host.clientWidth,Math.max(260,host.clientHeight-70),true),[t.x,...t.ys.map(s=>s.data)],host)}
$("mclose").onclick=()=>{$("modal").classList.remove("open");if(modal){modal.destroy();modal=null}modalDef=null;FULL=null};
document.querySelectorAll("[data-r]").forEach(b=>b.onclick=()=>{document.querySelectorAll("[data-r]").forEach(x=>x.classList.remove("on"));b.classList.add("on");range=b.dataset.r==="all"?"all":+b.dataset.r;drawModal()});
$("smooth").oninput=()=>drawModal();$("logy").onclick=()=>{logy=!logy;$("logy").classList.toggle("on",logy);drawModal()};
$("xmode").onclick=()=>{XTOK=!XTOK;$("xmode").textContent="x-axis: "+(XTOK?"tokens":"steps");drawCharts()};
window.addEventListener("resize",()=>{if(modal){const h=$("mchart");modal.setSize({width:h.clientWidth,height:Math.max(260,h.clientHeight-70)})}});

function render(){
 const c=S.current,r=c&&S.runs[c],k=[];
 k.push(`<div class="kpi"><span>Stage</span><b>${esc(S.status.stage||"setup")}</b><small>${esc(S.status.state||"")}</small></div>`);
 if(r){const finish=new Date((S.server_time+r.left_seconds)*1000);
  k.push(`<div class="kpi"><span>${esc(c)} step</span><b>${num(r.step)}${r.total?" / "+num(r.total):""}</b><small>${r.total?(100*r.step/r.total).toFixed(1)+"%":""} · loss ${num(r.loss,3)}</small></div>`);
  k.push(`<div class="kpi"><span>This run finishes (your time)</span><b>${finish.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}</b><small>${finish.toLocaleDateString([], {weekday:"short",day:"numeric",month:"short"})} · in ${dur(r.left_seconds)}</small></div>`);
  k.push(`<div class="kpi"><span>Speed</span><b>${num(r.speed)}</b><small>tok/s · MFU ${num(r.mfu,1)}%</small></div>`);}
 const all=new Date((S.server_time+S.remaining_hours*3600)*1000);
 k.push(`<div class="kpi"><span>Whole chain done ≈</span><b>${all.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}</b><small>${all.toLocaleDateString([], {weekday:"short",day:"numeric",month:"short"})} · estimate</small></div>`);
 k.push(`<div class="kpi"><span>Spent (billed uptime)</span><b>₹${num(S.spent)}</b><small>${dur(S.uptime_seconds)} at ₹${num(S.rate)}/h</small></div>`);
 const short=S.balance!=null&&S.balance<S.remaining_cost;
 k.push(`<div class="kpi ${short?"bad":""}"><span>Jarvis balance</span><b>${S.balance==null?"–":"₹"+num(S.balance)}</b><small>${short?"TOP UP: need ≈ ₹"+num(S.remaining_cost-S.balance)+" more":"needs ≈ ₹"+num(S.remaining_cost)+" to finish"}</small></div>`);
 $("kpis").innerHTML=k.join("");
 // phase bar for the main run
 const m=S.runs.pretrain||r;let pb="";
 if(m&&m.total){const p=m.phases||{},pct=v=>100*v/m.total;pb=`<div class="bar"><div class="fill" style="width:${pct(m.step)}%"></div>`+
  [["warmup_end","warmup"],["decay_start","decay"],["switch_8k","8k"]].filter(([k])=>p[k]).map(([k,l])=>`<div class="mark" style="left:${pct(p[k])}%"></div><div class="lbl" style="left:${pct(p[k])}%">${l} @${num(p[k])}</div>`).join("")+`</div>`}
 $("phasebar").innerHTML=pb;
 $("stages").innerHTML=S.stages.map(s=>`<div class="stage ${s.done?"done":s.live?"live":""}"><b>${s.name}</b><small>${s.done?dur(s.seconds):s.live?"running":""}</small></div>`).join("");
 const act=S.alerts.filter(a=>!a.resolved),old=S.alerts.filter(a=>a.resolved);
 $("alerts").innerHTML=(act.length?`<div class="card alert"><h3>Alerts</h3>${act.map(a=>`<p>${esc(a.time)} · ${esc(a.message)}</p>`).join("")}</div>`:"")+
  (old.length?`<details class="card"><summary class="muted">${old.length} earlier alert(s), resolved: the chain continued after them</summary>${old.map(a=>`<p class="muted">${esc(a.time)} · ${esc(a.message)}</p>`).join("")}</details>`:"");
 // LR test table
 const lr=S.lr_test;let h=`<h3>LR test</h3>`;
 if(lr){h+=`<table><tr><th>learning rate</th><th>final val BPB</th><th>final val loss (nats)</th></tr>`+lr.trials.map(t=>`<tr class="${t.lr===lr.lr?"win":""}"><td>${t.lr}${t.lr===lr.lr?" ✓ chosen":""}</td><td>${num(t.final_val_bpb,4)}</td><td>${S.val_bytes_per_token?num(t.final_val_bpb*Math.LN2*S.val_bytes_per_token,4):"–"}</td></tr>`).join("")+`</table><p class="hint">Rule: ${esc(lr.criterion)}</p>`}
 else h+=`<p class="muted">not finished yet</p>`;
 $("lrcard").innerHTML=h;
 const sw=S.sweep;$("sweepcard").innerHTML=`<h3>Speed sweep</h3>`+(sw?`<table><tr><th>micro</th><th>compile</th><th>tok/s</th><th>MFU</th></tr>`+sw.results.map(x=>`<tr><td>${num(x.micro_tokens)}</td><td>${esc(x.compile_mode)}</td>${x.ok?`<td>${num(x.tokens_per_second)}</td><td>${num(x.mfu*100,1)}%</td>`:`<td colspan="2">failed</td>`}</tr>`).join("")+`</table>`:`<p class="muted">not run</p>`);
 const b=S.backup;const ck=m&&m.last_checkpoint;
 $("backupcard").innerHTML=`<h3>Safety: checkpoints and backups</h3><table>
 <tr><td>Last checkpoint (${esc(S.current||"–")})</td><td>${esc(ck||"–")}${m&&m.last_checkpoint_age!=null?" · "+dur(m.last_checkpoint_age)+" ago":""}</td></tr>
 <tr><td>Last logs backup to HF</td><td>${esc(b.logs_time||"not yet")}</td></tr>
 <tr><td>Last checkpoint backup to HF</td><td>${esc(b.checkpoint_time||"not yet")} ${esc(b.checkpoint||"")}</td></tr>
 <tr><td>Backup repo</td><td>${esc(b.repo||"–")} (private)</td></tr>
 <tr><td>CPU rehearsal</td><td>${esc(S.rehearsal?S.rehearsal.stage+" · "+S.rehearsal.state:"not started")}</td></tr>
 ${b.error?`<tr><td>Backup error</td><td>${esc(b.error)}</td></tr>`:""}</table>`;
 const ev=Object.entries(S.evals);$("evalcard").innerHTML=`<h3>Evaluations</h3>`+(ev.length?ev.map(([k,v])=>`<details><summary>${esc(k)}</summary><pre>${esc(v)}</pre></details>`).join(""):`<p class="muted">appear after pretraining (base model vs GPT-2 / SmolLM2) and after chat tuning</p>`);
 $("logs").innerHTML=Object.entries(S.logs).map(([k,v])=>`<div class="card"><h3>${esc(k)}</h3><pre>${esc(v)}</pre></div>`).join("");
 $("updated").textContent=`updated ${new Date(S.server_time*1000).toLocaleTimeString()} (your time) · refreshes every 30 s · validation nats = BPB × ln2 × ${S.val_bytes_per_token?num(S.val_bytes_per_token,3):"?"} bytes/token`;
}
async function refresh(){try{const [s,d]=await Promise.all([fetch("api/summary").then(r=>r.json()),fetch("api/series").then(r=>r.json())]);S=s;D=d;render();drawCharts()}catch(e){$("updated").textContent="update failed, retrying: "+e}}
refresh();setInterval(refresh,30000);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage", default=os.getenv("CHOTU_STORAGE"))
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--rupees-per-hour", type=float, default=float(os.getenv("RUPEES_PER_HOUR", "217")))
    ap.add_argument("--sample-seconds", type=int, default=30)
    a = ap.parse_args()
    storage = Path(a.storage).expanduser()
    folder = storage / "dashboard"
    folder.mkdir(parents=True, exist_ok=True)
    key_file = folder / "key"
    if not key_file.exists():
        key_file.write_text(secrets.token_urlsafe(18))
        key_file.chmod(0o600)
    key = key_file.read_text().strip()
    threading.Thread(target=classic.recorder, args=(storage, a.sample_seconds), daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            path, query = url.path.rstrip("/"), parse_qs(url.query)
            try:
                if path == f"/{key}":
                    body, kind = PAGE.encode(), "text/html; charset=utf-8"
                elif path == f"/{key}/classic":
                    body, kind = classic.page(storage, a.rupees_per_hour, a.sample_seconds).encode(), "text/html; charset=utf-8"
                elif path == f"/{key}/api/summary":
                    body, kind = json.dumps(summary(storage, a.rupees_per_hour, a.sample_seconds), default=str).encode(), "application/json"
                elif path == f"/{key}/api/series":
                    body, kind = json.dumps(series(storage, query.get("full") == ["1"])).encode(), "application/json"
                elif path == f"/{key}/status.json":
                    body, kind = json.dumps(classic.read_json(storage / "status.json") or {}).encode(), "application/json"
                else:
                    self.send_error(404)
                    return
            except Exception as error:  # a bad file must never take the dashboard down
                body, kind = json.dumps({"error": f"{type(error).__name__}: {error}"}).encode(), "application/json"
                self.send_response(500)
            else:
                self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    print(f"live dashboard on port {a.port}: <Jarvis URL>/{key}/  (classic: /{key}/classic)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
