"""Turn a twin-day run folder into a short markdown report (written to stdout).

    python scripts/twin_report.py ~/chotu-runs/twin-day > reports/twin_day.md
"""
import json, pathlib, sys

root = pathlib.Path(sys.argv[1]).expanduser()
saved = pathlib.Path(__file__).resolve().parent.parent / "logs" / "twin-day"  # earlier runs kept in git


def find(run, name):  # this machine's run folder first, then the copy in git (e.g. the 140M run on a fresh machine)
    for base in (root, saved):
        if (base / run / name).exists():
            return base / run / name
    return None


def rows(run, kind=None):
    p = find(run, "metrics.jsonl")
    if not p:
        return []
    out = [json.loads(line) for line in open(p)]
    return [r for r in out if kind is None or r["t"] == kind]


def summary(run):
    p = find(run, "summary.json")
    return json.load(open(p)) if p else {}


status = [json.loads(line) for line in open(root / "status.jsonl")] if (root / "status.jsonl").exists() else []
latest = {}
for s in status:
    latest[s["id"]] = s

print("# Twin day: Session 3 rehearsal\n")
print("| Step | Result | Note |\n| --- | --- | --- |")
for sid, s in latest.items():
    print(f"| {sid} | {s['status']} | {s.get('note', '')} |")

m = summary("main")
if m:
    print("\n## Main twin run\n")
    print(f"- {m['tokens']:,} tokens in {m['minutes']} min on {m['gpu']}, about ₹{m['cost_inr']}")
    print(f"- Final training loss {m['final_loss']}, held-out {m['val_bpb']} bits per byte")
    print(f"- Speed {m['tok_s']:,} tokens/s" + (f", MFU {m['mfu']:.1%}" if m.get("mfu") else ""))
    if m.get("copy_first") is not None:
        print(f"- Copying test: surprise {m['copy_first']:.2f} on the first copy, {m['copy_second']:.2f} on the repeat")
    for r in rows("main", "sample")[-3:]:
        print(f"- Sample at step {r['step']}: *{r['prompt']}* → {r['text']!r}")

f = summary("main-5x")
if f:
    print("\n## 5x twin run: does reading 5x more help?\n")
    print("Same 7M model, recipe and exam (64 locked held-out windows); only the amount of text read changes. "
          "Lower bits per byte is better.\n")
    print("| | 140M run | 700M run (5x) |\n| --- | --- | --- |")
    for label, key, fmt in [("Tokens read", "tokens", "{:,}"), ("Steps", "steps", "{:,}"),
                            ("Held-out bits per byte", "val_bpb", "{}"), ("Final training loss", "final_loss", "{}"),
                            ("Copying test, repeat surprise", "copy_second", "{:.2f}"),
                            ("Minutes", "minutes", "{}"), ("Cost ₹", "cost_inr", "{}")]:
        cell = lambda d: fmt.format(d[key]) if d.get(key) is not None else "–"
        print(f"| {label} | {cell(m)} | {cell(f)} |")
    if m.get("val_bpb") and f.get("val_bpb"):
        gain = (m["val_bpb"] - f["val_bpb"]) / m["val_bpb"]
        print(f"\nHeld-out bits per byte changed by {-gain:+.1%} with 5x the tokens.")
    # the 5x run's exam when it had read the same 140M tokens (still at full stride then, so not a like-for-like finish)
    ev = rows("main-5x", "eval")
    if ev and m.get("steps"):
        e = min(ev, key=lambda r: abs(r["step"] - m["steps"]))
        print(f"At step {e['step']} (about the 140M mark, before its slow-down) the 5x run scored {e['val_bpb']} bits per byte.")
    for r in rows("main-5x", "sample")[-3:]:
        print(f"- Sample at step {r['step']}: *{r['prompt']}* → {r['text']!r}")

s = summary("speed-181m")
if s:
    print("\n## Speed test, real 181M model\n")
    print(f"- {s['tok_s']:,} tokens/s on {s['gpu']}" + (f", MFU {s['mfu']:.1%}" if s.get("mfu") else ""))
    if s.get("full_run_hours"):
        print(f"- Full 10.06B-token run at this speed: {s['full_run_hours']} hours, about ₹{s['full_run_cost_inr']:,} at this machine's price")

alerts = [r for run in ("overfit", "drill-ref", "drill", "main", "main-5x", "speed-181m") for r in rows(run, "alert")]
if alerts:
    print("\n## Alerts\n")
    for r in alerts:
        print(f"- step {r['step']} [{r['kind']}]: {r['msg']}")
