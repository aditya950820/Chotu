"""Turn a twin-day run folder into a short markdown report (written to stdout).

    python scripts/twin_report.py ~/chotu-runs/twin-day > reports/twin_day.md
"""
import json, pathlib, sys

root = pathlib.Path(sys.argv[1]).expanduser()


def rows(run, kind=None):
    p = root / run / "metrics.jsonl"
    if not p.exists():
        return []
    out = [json.loads(line) for line in open(p)]
    return [r for r in out if kind is None or r["t"] == kind]


def summary(run):
    p = root / run / "summary.json"
    return json.load(open(p)) if p.exists() else {}


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

s = summary("speed-181m")
if s:
    print("\n## Speed test, real 181M model\n")
    print(f"- {s['tok_s']:,} tokens/s on {s['gpu']}" + (f", MFU {s['mfu']:.1%}" if s.get("mfu") else ""))
    if s.get("full_run_hours"):
        print(f"- Full 10.06B-token run at this speed: {s['full_run_hours']} hours, about ₹{s['full_run_cost_inr']:,} at this machine's price")

alerts = [r for run in ("overfit", "drill-ref", "drill", "main", "speed-181m") for r in rows(run, "alert")]
if alerts:
    print("\n## Alerts\n")
    for r in alerts:
        print(f"- step {r['step']} [{r['kind']}]: {r['msg']}")
