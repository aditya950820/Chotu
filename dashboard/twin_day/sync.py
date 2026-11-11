"""Pull twin-day logs from GitHub (branch live-runs, then twin-day) and write twin.json for the dashboard.
One long table; the `k` column says what each row is: live, run, step, eval, check, alert, sample.
    python sync.py <out.json>
"""
import json, math, os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).parent
CLONE = HERE / "live"
URL = "https://github.com/aditya950820/Chotu.git"
out_path = pathlib.Path(sys.argv[1])
START = math.log(49152)


def git(*a):
    return subprocess.run(["git", *a], cwd=CLONE, capture_output=True, text=True)


if not (CLONE / ".git").exists():
    CLONE.mkdir(parents=True, exist_ok=True)
    git("init", "-q"); git("remote", "add", "origin", URL)
have_live = git("fetch", "-q", "--depth", "1", "origin", "live-runs").returncode == 0
commit_time = None
root = None
if have_live:
    git("checkout", "-q", "-f", "FETCH_HEAD")
    commit_time = int(git("log", "-1", "--format=%ct").stdout.strip() or 0)
    root = CLONE / "runs" / "twin-day"
report_pushed = subprocess.run(["git", "ls-remote", URL, "twin-day"], capture_output=True, text=True).stdout.strip() != "" and \
    subprocess.run(["git", "-C", str(CLONE), "fetch", "-q", "--depth", "1", "origin", "twin-day"]).returncode == 0 and \
    "reports/twin_day.md" in subprocess.run(["git", "-C", str(CLONE), "ls-tree", "-r", "--name-only", "FETCH_HEAD"], capture_output=True, text=True).stdout

RUN_ORDER = ["overfit", "drill-ref", "drill", "main", "speed-181m", "main-5x"]
rows, status, runs = [], {}, {}
if root and root.exists():
    if (root / "status.jsonl").exists():
        for line in open(root / "status.jsonl"):
            s = json.loads(line)
            status[s["id"]] = s
            status.setdefault("_first", s)
    for d in sorted(root.iterdir()):
        if (d / "metrics.jsonl").exists():
            runs[d.name] = [json.loads(line) for line in open(d / "metrics.jsonl") if line.strip()]
# the 140M main run is the baseline for the 5x run; on a fresh machine it is not in live-runs, so take it from twin-day
if "main" not in runs and subprocess.run(["git", "-C", str(CLONE), "fetch", "-q", "--depth", "1", "origin", "twin-day"]).returncode == 0:
    txt = git("show", "FETCH_HEAD:logs/twin-day/main/metrics.jsonl").stdout
    if txt.strip():
        runs["main"] = [json.loads(line) for line in txt.splitlines() if line.strip()]

LABELS = {"setup": "Setup and size test", "data": "Download shards", "overfit": "Overfit one batch", "firedrill": "Fire drill",
          "main": "Main twin run", "speed": "181M speed test", "main5x": "5x twin run (700M tokens)",
          "upload": "Checkpoint to HF", "end": "Report and push"}
if root and (root / "status.jsonl").exists():
    seen = {}
    for line in open(root / "status.jsonl"):
        r = json.loads(line)
        st_ = seen.setdefault(r["id"], dict(k="stage", order=len(seen), id=r["id"], label=LABELS.get(r["id"], r["id"]), start=r["time"]))
        st_.update(status=r["status"], note=r.get("note", ""), last=r["time"])
    for st_ in seen.values():
        st_["seconds"] = st_["last"] - st_["start"] if st_["status"] != "running" else round(time.time()) - st_["start"]
        rows.append(st_)
steps_by_run = {}
for name in sorted(runs, key=lambda n: RUN_ORDER.index(n) if n in RUN_ORDER else 99):
    recs = runs[name]
    meta = next((r for r in recs if r["t"] == "meta"), {})
    steps = [r for r in recs if r["t"] == "step"]
    steps_by_run[name] = steps
    evals = [r for r in recs if r["t"] == "eval"]
    end = next((r for r in reversed(recs) if r["t"] == "end"), None)
    every = max(1, len(steps) // 500)  # keep charts light
    for i, r in enumerate(steps):
        if i % every == 0 or i == len(steps) - 1:
            rows.append(dict(k="step", run=name, step=r["step"], loss=r["loss"], lr=r["lr"], grad=r["grad"],
                             tok_s=r["tok_s"], mfu=r.get("mfu"), elapsed=r["elapsed"]))
    for r in evals:
        rows.append(dict(k="eval", run=name, step=r["step"], val_loss=r["val_loss"], val_bpb=r["val_bpb"]))
    for r in recs:
        if r["t"] == "alert":
            rows.append(dict(k="alert", run=name, step=r["step"], kind=r["kind"], msg=r["msg"]))
        if r["t"] == "sample":
            rows.append(dict(k="sample", run=name, step=r["step"], prompt=r["prompt"], text=r["text"]))
    last = steps[-1] if steps else {}
    state = "running"
    if end:
        state = "crashed (fire drill)" if "crash" in end["reason"] else ("finished" if end["reason"] == "finished" else end["reason"])
    half = [r["tok_s"] for r in steps[len(steps) // 2:]] or [0]
    half.sort()
    run_steps = meta.get("run_steps") or 0
    summary = end or {}
    rows.append(dict(k="run", run=name, kind=meta.get("kind"), model=meta.get("model"), gpu=meta.get("gpu"), state=state,
                     step=last.get("step", 0), run_steps=run_steps,
                     progress=round(last.get("step", 0) / run_steps, 4) if run_steps else 0,
                     tokens=last.get("tokens", 0), loss=last.get("loss"), val_bpb=evals[-1]["val_bpb"] if evals else None,
                     tok_s=half[len(half) // 2], mfu=summary.get("mfu") or last.get("mfu"),
                     minutes=round(last.get("elapsed", 0) / 60, 1), cost_inr=last.get("cost"),
                     full_run_hours=summary.get("full_run_hours"), full_run_cost_inr=summary.get("full_run_cost_inr"),
                     copy_first=summary.get("copy_first"), copy_second=summary.get("copy_second"),
                     updated=recs[-1].get("time")))

# the run that is moving now (or the last one)
cur = None
for name in RUN_ORDER[::-1]:
    if name in runs and steps_by_run.get(name):
        cur = name if cur is None else cur
active = next((n for n in RUN_ORDER if status.get({"overfit": "overfit", "drill-ref": "firedrill", "drill": "firedrill",
                                                   "main": "main", "speed-181m": "speed", "main-5x": "main5x"}[n], {}).get("status") == "running"
               and n in runs), None)
cur = active or cur
price = next((r.get("gpu_price") for rs in runs.values() for r in rs if r["t"] == "meta"), None)
gpu = next((r.get("gpu") for rs in runs.values() for r in rs if r["t"] == "meta"), None)
first_t = status.get("_first", {}).get("time")
last_t = max([s["time"] for k, s in status.items() if k != "_first"] + [commit_time or 0]) if status else None
script_done = "end" in status
machine_min = round((last_t - first_t) / 60, 1) if first_t and last_t else None
cr = next((r for r in rows if r["k"] == "run" and r["run"] == cur), {})
main_evals = [r for r in rows if r["k"] == "eval" and r["run"] == (cur or "")]
rows.append(dict(k="live", synced=round(time.time()), pushed=commit_time, run=cur, state="done" if script_done else (cr.get("state") or "waiting"),
                 step=cr.get("step"), run_steps=cr.get("run_steps"), progress=cr.get("progress"), loss=cr.get("loss"),
                 val_bpb=main_evals[-1]["val_bpb"] if main_evals else None, tok_s=cr.get("tok_s"), mfu=cr.get("mfu"),
                 machine_min=machine_min, cost_so_far=round(machine_min / 60 * price, 1) if machine_min and price else None,
                 gpu=gpu, price=price))


# ---------- checklist: today's gates come from the run itself ----------
def st(sid):
    s = status.get(sid)
    if not s:
        return "todo", ""
    return {"pass": "done", "fail": "failed", "running": "running", "info": "done"}[s["status"]], s.get("note", "")


main = steps_by_run.get("main", [])
mevals = [r for r in rows if r["k"] == "eval" and r["run"] == "main"]
mrun = next((r for r in rows if r["k"] == "run" and r["run"] == "main"), {})
srun = next((r for r in rows if r["k"] == "run" and r["run"] == "speed-181m"), {})
s300 = next((r for r in main if r["step"] == 300), None)
checks = []


def add(phase, item, stat, evidence=""):
    checks.append(dict(k="check", order=len(checks), phase=phase, item=item, status=stat, evidence=evidence))


T = "1 · Today: twin day"
add(T, "Rent a GPU (Jarvislabs) and add your SSH key", "done", "Instance at 151.185.52.36")
add(T, "Persistent storage checked (data survives a pause)", "todo", "Check in the Jarvis dashboard")
add(T, "Scripts on GitHub (branch twin-day)", "done", "train.py, scripts/twin_day.sh, scripts/twin_report.py")
add(T, "Setup: CUDA, packages, size test 180,925,056", *st("setup"))
add(T, "Data: 1 train + 1 val shard from the private HF dataset", *st("data"))
if main:
    l0 = main[0]["loss"]
    add(T, "First loss near 10.8", "done" if abs(l0 - START) < 0.6 else "failed", f"first loss {l0:.2f}")
else:
    add(T, "First loss near 10.8", "todo")
ov = steps_by_run.get("overfit", [])
if st("overfit")[0] in ("done", "failed") and ov:
    ok_ = ov[-1]["loss"] < 2.0 and ov[0]["loss"] > 9
    add(T, "Overfit one batch: loss collapses on a batch it sees every step", "done" if ok_ else "failed",
        f"{ov[0]['loss']:.2f} → {ov[-1]['loss']:.2f} in {ov[-1]['step']} steps" + (" (the script's own < 1.0 gate was too strict for 7M)" if ok_ and ov[-1]["loss"] >= 1 else ""))
else:
    add(T, "Overfit one batch: loss collapses on a batch it sees every step", *st("overfit"))
add(T, "Fire drill: crash at step 70, resume, lines match", *st("firedrill"))
add(T, "Main twin run: 140M tokens", *st("main"))
if s300:
    add(T, "Loss below 6 by step 300", "done" if s300["loss"] < 6 else "failed", f"step 300 loss {s300['loss']:.2f}")
else:
    add(T, "Loss below 6 by step 300", "running" if main else "todo")
if len(mevals) > 1:
    add(T, "Exam score falls with practice", "done" if mevals[-1]["val_bpb"] < mevals[0]["val_bpb"] else "failed",
        f"{mevals[0]['val_bpb']:.2f} → {mevals[-1]['val_bpb']:.2f} bits per byte")
else:
    add(T, "Exam score falls with practice", "running" if main else "todo")
if mrun.get("copy_first") is not None:
    ok = mrun["copy_second"] < mrun["copy_first"] - 1
    add(T, "Copying test: the repeat is much easier", "done" if ok else "failed",
        f"surprise {mrun['copy_first']:.2f} → {mrun['copy_second']:.2f}")
else:
    add(T, "Copying test: the repeat is much easier", "todo")
add(T, "181M speed test (30 steps at the real batch)", *st("speed"))
if srun.get("mfu"):
    add(T, "181M MFU at least 10% (kill rule)", "done" if srun["mfu"] >= 0.10 else "failed", f"MFU {srun['mfu']:.1%}")
else:
    add(T, "181M MFU at least 10% (kill rule)", "todo")
add(T, "Report and logs pushed to GitHub", "done" if report_pushed else "todo", "reports/twin_day.md on branch twin-day" if report_pushed else "")
add(T, "Machine stopped", "todo", "Stop it in the Jarvis dashboard")
add(T, "Old Jarvis instance from 24 Oct paused or deleted", "todo", "PROGRESS.md still lists it as running")
add(T, "PROGRESS.md log entry for today", "todo", "")

# the 5x day: does reading 5x more (700M tokens) make the same 7M twin better?
if "main5x" in status or "main-5x" in runs:
    F = "1b · 5x twin run"
    frun = next((r for r in rows if r["k"] == "run" and r["run"] == "main-5x"), {})
    fev = [r for r in rows if r["k"] == "eval" and r["run"] == "main-5x"]
    add(F, "181M speed test passes with 8 windows per pass (ran out of memory at 32)", *st("speed"))
    add(F, "4 extra train shards downloaded while the GPU worked", "done" if status.get("data", {}).get("status") == "info" and st("main5x")[0] != "todo" else st("data")[0],
        "5 train shards, so repeats stay at the 140M run's share")
    add(F, "5x run: 700M tokens, same model, recipe and exam", *st("main5x"))
    base = mevals[-1]["val_bpb"] if mevals else 1.3881
    if frun.get("state") == "finished" and fev:
        add(F, "Exam better than the 140M run", "done" if fev[-1]["val_bpb"] < base else "failed",
            f"{fev[-1]['val_bpb']:.4f} vs {base:.4f} bits per byte ({(fev[-1]['val_bpb'] - base) / base:+.1%})")
    else:
        add(F, "Exam better than the 140M run", "running" if fev else "todo", f"140M run: {base:.4f} bits per byte")
    add(F, "Final checkpoint on HF (magnateadi/chotu-twin-day, twin-day/main-5x)", *st("upload"))
    add(F, "Machine paused", "done" if "end" in status and st("main5x")[0] != "running" else "todo", "the script pauses it when JL_API_KEY and JL_MACHINE_ID are in .env")

fd = st("firedrill")[0]
P = [("2 · Before the real run", [
    ("Training and exam shards kept separate", "done", "0.5% held out in Session 2"),
    ("Untrained Chotu scores near 10.8", "done", "10.91 in the Guessing Game"),
    ("Tiny twin trains and its loss falls", "done" if st("main")[0] == "done" else "todo", ""),
    ("Checkpoint fire drill passed", fd if fd in ("done", "failed") else "todo", ""),
    ("Config confirmed: batch 524,288, 19,181 steps, warm-up ~700, slow-down last 20%", "todo", ""),
    ("Checkpoints every 30–60 min to storage outside the GPU machine", "todo", ""),
    ("Dashboard follows loss, exam, gradient, stride, speed, cost", "done", "This page"),
    ("Budget and kill point agreed", "todo", ""),
    ("Report card ready: prompts, blanks, copying test, word map", "todo", ""),
    ("GPU booked with capacity for about a day and a half", "todo", "Portal H200 at ₹140/h is the cheapest")]),
    ("3 · LR test", [
        ("Run 1.5e-3 and 3e-3 on 0.5B tokens each", "todo", ""),
        ("Lower held-out bits per byte wins", "todo", ""),
        ("Result and reason in the run log", "todo", "")]),
    ("4 · Launch: first 10 minutes", [
        ("No errors, GPU memory well used, first loss near 10.8", "todo", ""),
        ("Tokens per second steady and close to the speed test", "todo", ""),
        ("Projected hours and cost within budget", "todo", ""),
        ("Stride chart climbs through warm-up", "todo", ""),
        ("Step 300: loss below about 6", "todo", ""),
        ("First checkpoint saved and loads", "todo", ""),
        ("Decision: keep, fix and restart, or kill", "todo", "")]),
    ("5 · During the run", [
        ("Loss trending down; flat with small wiggles is fine", "todo", ""),
        ("Exam dots close to the practice line", "todo", ""),
        ("No spikes, no NaN, gradient not always at the clip", "todo", ""),
        ("Speed steady (short dips are exams and saves)", "todo", ""),
        ("New checkpoints arriving; last few kept", "todo", ""),
        ("Last 20%: stride shrinks, loss takes one more drop", "todo", "")]),
    ("6 · If the machine dies", [
        ("Resume from the newest good checkpoint, not from scratch", "todo", ""),
        ("Loss continues where it left off, no warm-up restart", "todo", ""),
        ("Log what happened and the time lost", "todo", "")]),
    ("7 · At the end and after", [
        ("Final checkpoint copied to safe storage", "todo", ""),
        ("Final exam score and bits per byte recorded", "todo", ""),
        ("GPU shut down straight away", "todo", ""),
        ("Report card on checkpoints across the run", "todo", ""),
        ("Compare with SmolLM2 and GPT-2 small", "todo", ""),
        ("Write-up saved to the repo", "todo", "")])]
for phase, items in P:
    for item, s_, ev in items:
        add(phase, item, s_, ev)
rows.extend(checks)
json.dump(rows, open(out_path, "w"))
live = rows[-len(checks) - 1] if False else next(r for r in rows if r["k"] == "live")
print(json.dumps({k: live.get(k) for k in ("run", "state", "step", "run_steps", "loss", "machine_min", "cost_so_far", "pushed")}),
      "rows", len(rows), "report_pushed", report_pushed)
