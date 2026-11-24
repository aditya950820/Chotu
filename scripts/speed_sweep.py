"""15-minute speed check: try a few speed-only settings on the real model, print MFU, pause the GPU.

    python scripts/speed_sweep.py --storage /home/jl/chotu            # sweep, then pause
    python scripts/speed_sweep.py --storage ... --no-stop             # keep the machine on

Each config trains the full 181M model for --steps updates on real shards (outputs go to
<storage>/runs/sweep-*/, deleted afterwards). Steps 1-5 include compile warm-up and are not
measured. Nothing here changes the training recipe: batch, LR and data stay frozen;
only microbatch size (65k/131k/262k, until out of memory) and compile mode vary. Writes <storage>/reports/speed-sweep.json.
"""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

TRAIN = Path(__file__).with_name("train.py")
# Microbatch sizes tried smallest first; stop at the first that runs out of memory
# (H200: 65k ≈ 45 GB, 131k ≈ 90 GB of activations). The best size then tries autotuned compile.
MICRO_SIZES = [65536, 131072, 262144]  # H200 141 GB; 262k usually runs out of memory


def run_config(a, micro, mode):
    name = f"sweep-{micro}-{mode}"
    out = Path(a.storage) / "runs" / name
    shutil.rmtree(out, ignore_errors=True)
    command = [sys.executable, str(TRAIN), "--stage", "pretrain", "--storage", a.storage, "--name", name,
               "--lr", "0.003", "--total-tokens", str(a.steps * 524288), "--batch-tokens", "524288",
               "--micro-tokens", str(micro), "--compile-mode", mode, "--long-frac", "0",
               "--warmup-steps", "5", "--ckpt-every", "100000", "--eval-every", "100000",
               "--val-tokens", "8192", "--no-auto-stop"]
    started = time.monotonic()
    code = subprocess.run(command).returncode
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()] \
        if (out / "metrics.jsonl").exists() else []
    measured = [r for r in rows if r["step"] > 5]
    shutil.rmtree(out, ignore_errors=True)  # a 2 GB checkpoint per config is not worth keeping
    if code or not measured:
        return dict(micro_tokens=micro, compile_mode=mode, ok=False, exit=code,
                    note="failed (out of memory is the usual reason at larger microbatches)")
    return dict(micro_tokens=micro, compile_mode=mode, ok=True,
                tokens_per_second=statistics.median(r["tokens_per_second"] for r in measured),
                mfu=statistics.median(r["mfu"] for r in measured),
                data_wait_frac=statistics.median(r.get("data_wait_frac", 0) for r in measured),
                final_loss=measured[-1]["loss"], wall_seconds=round(time.monotonic() - started))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--no-stop", action="store_true", help="do not pause the machine at the end")
    ap.add_argument("--quick", action="store_true", help="skip the slower autotuned-compile trial")
    a = ap.parse_args()
    results = []
    try:
        for micro in MICRO_SIZES:
            results.append(run_config(a, micro, "default"))
            if not results[-1]["ok"] and micro == MICRO_SIZES[0]:
                # The smallest size cannot be out of memory on an H200: retry once so a one-off
                # failure (e.g. a transient non-finite step) is not mistaken for a slow machine.
                print("SWEEP retrying the smallest microbatch once", flush=True)
                results[-1] = run_config(a, micro, "default")
            print("SWEEP " + json.dumps(results[-1]), flush=True)
            if not results[-1]["ok"]:
                break
        good = [r for r in results if r["ok"]]
        if good and not a.quick:
            top = max(good, key=lambda r: r["tokens_per_second"])
            results.append(run_config(a, top["micro_tokens"], "max-autotune-no-cudagraphs"))
            print("SWEEP " + json.dumps(results[-1]), flush=True)
            good = [r for r in results if r["ok"]]
        best = max(good, key=lambda r: r["tokens_per_second"]) if good else None
        report = dict(results=results, best=best,
                      hours_for_10B=round(10e9 / best["tokens_per_second"] / 3600, 1) if best else None)
        path = Path(a.storage) / "reports" / "speed-sweep.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"\n{'micro':>7} {'compile':<28} {'tok/s':>9} {'MFU':>6} {'data wait':>9}")
        for r in results:
            if r["ok"]:
                print(f"{r['micro_tokens']:>7} {r['compile_mode']:<28} {r['tokens_per_second']:>9,.0f} "
                      f"{r['mfu']:>6.1%} {r['data_wait_frac']:>9.1%}")
            else:
                print(f"{r['micro_tokens']:>7} {r['compile_mode']:<28} {'failed':>9}")
        if best:
            print(f"\nBest: micro {best['micro_tokens']}, compile {best['compile_mode']} -> "
                  f"10B tokens in ~{report['hours_for_10B']} h. Report: {path}")
    finally:
        if not a.no_stop:
            code = subprocess.run([sys.executable, str(Path(__file__).with_name("stop_machine.py"))]).returncode
            if code:
                print("MACHINE PAUSE FAILED: pause it manually now!", flush=True)


if __name__ == "__main__":
    main()
