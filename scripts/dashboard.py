"""Live web dashboard served from the GPU machine itself, so it keeps updating with the Mac off.

    python scripts/dashboard.py --storage /home/chotu-data        # gpu_up.sh starts it at second one

Jarvis publishes port 6006 through Cloudflare: Jarvis dashboard -> your instance -> "API" button
gives the base URL; open <base URL>/<key>/ where <key> is in <storage>/dashboard/key (launch.sh
prints the full path). Requests without the key get 404, because the port is public.

Standard library only (it starts before any package is installed). Every page view reads the live
files the chain writes; nothing is cached or made up. A background thread also records GPU, CPU,
memory and disk every 30 s to <storage>/logs/system.jsonl, so system history is logged too.
"""
import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time

STAGES = ("preflight", "shards", "chat_data", "speed", "lr_test", "pretrain",
          "base_evals", "chat_tune", "final_evals", "export_gguf", "upload", "final_report")
RUNS = ("speed", "lr-0.0015", "lr-0.003", "pretrain", "chat")
COLORS = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed")


def read_jsonl(path, keep=None):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-keep:] if keep else lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def tail(path, n=15):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            return f.read().decode(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def system_sample(storage):
    row = dict(time=time.time())
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        util, used, total, power, temp = [float(x) for x in out.splitlines()[0].split(",")]
        row.update(gpu_util=util, gpu_mem_gib=used / 1024, gpu_mem_total_gib=total / 1024, gpu_watts=power, gpu_temp_c=temp)
    except Exception:
        pass
    try:
        row["load1"] = os.getloadavg()[0]
        row["cpus"] = os.cpu_count()
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        row["ram_used_gib"] = (int(mem["MemTotal"].split()[0]) - int(mem["MemAvailable"].split()[0])) / 2**20
        row["ram_total_gib"] = int(mem["MemTotal"].split()[0]) / 2**20
    except Exception:
        pass
    try:
        row["disk_free_gib"] = shutil.disk_usage(storage).free / 2**30
    except OSError:
        pass
    return row


def recorder(storage, every):
    path = storage / "logs" / "system.jsonl"
    while True:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(system_sample(storage)) + "\n")
        except OSError:
            pass
        time.sleep(every)


def downsample(points, limit=500):
    step = max(1, len(points) // limit)
    return points[::step] + ([points[-1]] if points and len(points) % step else [])


def chart(title, series, unit="", log_y=False):
    """series: list of (label, [(x, y), ...]). Inline SVG, no scripts."""
    series = [(label, [(x, y) for x, y in pts if y is not None]) for label, pts in series]
    series = [(label, downsample(pts)) for label, pts in series if pts]
    if not series:
        return f'<div class="card chart"><h3>{html.escape(title)}</h3><p class="muted">no data yet</p></div>'
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    if y0 == y1:
        y0, y1 = y0 - 1, y1 + 1
    w, h, pad = 520, 200, 40

    def sx(x):
        return pad + (x - x0) / ((x1 - x0) or 1) * (w - pad - 10)

    def sy(y):
        return h - 22 - (y - y0) / (y1 - y0) * (h - 34)
    lines = []
    for i, (label, pts) in enumerate(series):
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        lines.append(f'<polyline fill="none" stroke="{COLORS[i % len(COLORS)]}" stroke-width="1.6" points="{path}"/>')
    legend = " ".join(f'<span style="color:{COLORS[i % len(COLORS)]}">■ {html.escape(l)} '
                      f'<b>{fmt(pts[-1][1])}{unit}</b></span>' for i, (l, pts) in enumerate(series))
    return (f'<div class="card chart"><h3>{html.escape(title)}</h3><div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">'
            f'<line x1="{pad}" y1="{h - 22}" x2="{w - 10}" y2="{h - 22}" class="axis"/>'
            f'<line x1="{pad}" y1="10" x2="{pad}" y2="{h - 22}" class="axis"/>'
            f'<text x="{pad - 4}" y="16" text-anchor="end">{fmt(y1)}</text>'
            f'<text x="{pad - 4}" y="{h - 24}" text-anchor="end">{fmt(y0)}</text>'
            f'<text x="{pad}" y="{h - 6}">{fmt(x0)}</text>'
            f'<text x="{w - 10}" y="{h - 6}" text-anchor="end">{fmt(x1)}</text>'
            + "".join(lines) + "</svg></div>")


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        if v != 0 and abs(v) < 0.01:
            return f"{v:.3g}"
        return f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:,.0f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def duration(seconds):
    if seconds is None:
        return "–"
    seconds = int(seconds)
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m {seconds % 60:02d}s"


def page(storage, rate, sample_seconds=30):
    now = time.time()
    status = read_json(storage / "status.json") or {}
    system = read_jsonl(storage / "logs" / "system.jsonl")
    # system.jsonl only grows while the machine runs, so this is billed uptime (pauses excluded)
    elapsed = len(system) * sample_seconds if system else None
    parts = []

    # --- headline -------------------------------------------------------------------------------
    alerts = [e for e in read_jsonl(storage / "logs" / "chain.jsonl", keep=5000) if e.get("state") == "ALERT"]
    state = status.get("state", "setting up")
    head = (f'<div class="kpis"><div class="kpi"><span>Stage</span><b>{html.escape(str(status.get("stage", "setup")))}</b>'
            f'<small>{html.escape(str(state))}</small></div>'
            f'<div class="kpi"><span>Machine up</span><b>{duration(elapsed)}</b><small>billed uptime, pauses excluded</small></div>'
            f'<div class="kpi"><span>Spent so far</span><b>₹{(elapsed or 0) / 3600 * rate:,.0f}</b><small>at ₹{rate:.0f}/h GPU</small></div>')

    # --- current run + ETA ---------------------------------------------------------------------
    runs = {name: read_jsonl(storage / "runs" / name / "metrics.jsonl") for name in RUNS}
    live = [n for n in RUNS if runs[n] and not (storage / "runs" / n / "done.json").exists()]
    current = live[-1] if live else None
    if current:
        rows, plan = runs[current], read_json(storage / "runs" / current / "plan.json") or {}
        last, recent = rows[-1], rows[-50:]
        speed = sorted(r["tokens_per_second"] for r in recent)[len(recent) // 2]
        total = plan.get("steps")
        left = (total - last["step"]) * plan.get("batch_tokens", 524288) / speed if total else None
        pct = last["step"] / total if total else 0
        head += (f'<div class="kpi"><span>{html.escape(current)} step</span><b>{last["step"]:,}'
                 f'{" / " + format(total, ",") if total else ""}</b><small>{pct:.1%}</small></div>'
                 f'<div class="kpi"><span>Time left (this run)</span><b>{duration(left)}</b>'
                 f'<small>≈ ₹{(left or 0) / 3600 * rate:,.0f}</small></div>'
                 f'<div class="kpi"><span>Speed</span><b>{speed:,.0f}</b><small>tok/s · MFU {last["mfu"]:.1%}</small></div>')
        age = now - (storage / "runs" / current / "metrics.jsonl").stat().st_mtime
        if age > 600:
            alerts.append(dict(time="now", message=f"no new training step for {duration(age)}"))
        parts.append(f'<div class="bar"><div style="width:{pct * 100:.2f}%"></div></div>')
    head += "</div>"
    parts.insert(0, head)
    if alerts:
        parts.append('<div class="card alert"><h3>Alerts</h3>' + "".join(
            f'<p>{html.escape(str(a.get("time", "")))[:19]} · {html.escape(str(a.get("message", a)))}</p>' for a in alerts[-5:]) + "</div>")

    # --- stages ------------------------------------------------------------------------------------
    cells = []
    for name in STAGES:
        marker = read_json(storage / "stages" / f"{name}.done.json")
        cls = "done" if marker else ("live" if status.get("stage") == name else "")
        cells.append(f'<div class="stage {cls}"><b>{name}</b><small>{duration(marker["seconds"]) if marker else ("running" if cls else "")}</small></div>')
    parts.append('<div class="stages">' + "".join(cells) + "</div>")

    # --- charts ------------------------------------------------------------------------------------
    show = [n for n in RUNS if runs[n] and n != "speed"]

    def s(key, names=show):
        return [(n, [(r["step"], r.get(key)) for r in runs[n]]) for n in names]
    main = [current] if current else show[-1:]
    charts = [chart("Training loss", s("loss")),
              chart("Validation bits per byte (pretraining)", s("val_bpb", [n for n in show if n != "chat"])),
              chart("Tokens per second", s("tokens_per_second", main)),
              chart("MFU", [(n, [(x, y * 100 if y is not None else None) for x, y in pts]) for n, pts in s("mfu", main)], "%"),
              chart("Learning rate", s("lr", main)),
              chart("Gradient norm", s("grad_norm", main))]
    if runs["chat"]:
        charts.append(chart("Chat held-out loss", s("val_loss", ["chat"])))
    t0 = system[0]["time"] if system else 0
    hrs = [((r["time"] - t0) / 3600) for r in system]
    charts.append(chart("GPU busy % (x = hours)", [("GPU", list(zip(hrs, [r.get("gpu_util") for r in system])))], "%"))
    charts.append(chart("GPU memory GiB", [("used", list(zip(hrs, [r.get("gpu_mem_gib") for r in system])))]))
    charts.append(chart("Disk free GiB", [("free", list(zip(hrs, [r.get("disk_free_gib") for r in system])))]))
    parts.append('<div class="grid">' + "".join(charts) + "</div>")

    # --- tables: sweep, LR test, evals ----------------------------------------------------------
    sweep = read_json(storage / "reports" / "speed-sweep.json")
    if sweep:
        rows = "".join(f'<tr><td>{r["micro_tokens"]:,}</td><td>{r["compile_mode"]}</td>'
                       + (f'<td>{r["tokens_per_second"]:,.0f}</td><td>{r["mfu"]:.1%}</td><td>{r["data_wait_frac"]:.1%}</td>'
                          if r["ok"] else '<td colspan="3">failed (out of memory)</td>') + "</tr>" for r in sweep["results"])
        parts.append(f'<div class="card"><h3>Speed sweep</h3><table><tr><th>micro</th><th>compile</th><th>tok/s</th>'
                     f'<th>MFU</th><th>data wait</th></tr>{rows}</table></div>')
    reports = sorted(p for p in (storage / "reports").glob("*.json")) if (storage / "reports").exists() else []
    if reports:
        parts.append('<div class="card"><h3>Reports saved</h3><p class="muted">' + " · ".join(
            html.escape(p.name) for p in reports) + "</p></div>")

    # --- background jobs + logs ------------------------------------------------------------------
    backup = read_json(storage / "dashboard" / "backup.json") or {}
    rehearsal = storage / "rehearsal" / "status.json"
    parts.append('<div class="grid">'
                 f'<div class="card"><h3>Backup to private HF</h3><p>last logs backup: {html.escape(str(backup.get("logs_time", "not yet")))}<br>'
                 f'last checkpoint backup: {html.escape(str(backup.get("checkpoint_time", "not yet")))} {html.escape(str(backup.get("checkpoint", "")))}<br>'
                 f'repo: {html.escape(str(backup.get("repo", "–")))}<br>{html.escape(str(backup.get("error", "")))}</p></div>'
                 f'<div class="card"><h3>CPU rehearsal (early warning)</h3><p>{html.escape(json.dumps(read_json(rehearsal)) if rehearsal.exists() else "not started")}</p></div>'
                 "</div>")
    log_name = f'logs/{status.get("stage")}.log' if status.get("stage") else "gpu_up.log"
    for title, path in (("Setup log (gpu_up.log)", storage / "gpu_up.log"), (f"Current stage log ({log_name})", storage / log_name)):
        parts.append(f'<div class="card"><h3>{html.escape(title)}</h3><pre>{html.escape(chr(10).join(tail(path)))}</pre></div>')

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="30">
<title>Chotu run</title><style>
:root{{--bg:#f8fafc;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--ok:#059669;--live:#2563eb;--bad:#dc2626}}
@media (prefers-color-scheme: dark){{:root{{--bg:#0b1120;--card:#111827;--ink:#e5e7eb;--muted:#94a3b8;--line:#1f2937}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}}
h1{{font-size:20px;margin:0 0 12px}} h3{{font-size:14px;margin:0 0 6px}} .muted,small,span{{color:var(--muted)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-bottom:10px}}
.kpi,.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}}
.kpi b{{display:block;font-size:20px}} .kpi span,.kpi small{{display:block;font-size:12px}}
.bar{{height:8px;background:var(--line);border-radius:4px;margin:6px 0 12px}} .bar div{{height:100%;background:var(--live);border-radius:4px}}
.stages{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:6px;margin-bottom:12px}}
.stage{{border:1px solid var(--line);border-radius:8px;padding:6px;font-size:12px;background:var(--card)}}
.stage small{{display:block}} .stage.done{{border-color:var(--ok)}} .stage.done b{{color:var(--ok)}} .stage.live{{border-color:var(--live);border-width:2px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px;margin-bottom:10px}}
svg{{width:100%;height:auto}} svg text{{font-size:10px;fill:var(--muted)}} .axis{{stroke:var(--line)}}
.legend{{font-size:12px;display:flex;gap:10px;flex-wrap:wrap}} .alert{{border-color:var(--bad)}} .alert h3{{color:var(--bad)}}
pre{{white-space:pre-wrap;word-break:break-all;font-size:11px;margin:0;max-height:260px;overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:12px}} td,th{{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left}}
.card{{margin-bottom:10px;overflow:hidden}}</style></head><body>
<h1>Chotu 181M · live run <small>updated {time.strftime("%H:%M:%S UTC", time.gmtime(now))} · refreshes every 30 s</small></h1>
{"".join(parts)}</body></html>"""


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
    threading.Thread(target=recorder, args=(storage, a.sample_seconds), daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == f"/{key}":
                body, kind = page(storage, a.rupees_per_hour, a.sample_seconds).encode(), "text/html; charset=utf-8"
            elif path == f"/{key}/status.json":
                body, kind = json.dumps(read_json(storage / "status.json") or {}).encode(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    print(f"dashboard on port {a.port}: open <Jarvis API URL>/{key}/", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
