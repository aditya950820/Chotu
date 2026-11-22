"""15-minute speed check: try a few speed-only settings on the real model, print MFU, pause the GPU.

    python scripts/speed_sweep.py --storage /home/jl/chotu            # sweep, then pause
    python scripts/speed_sweep.py --storage ... --no-stop             # keep the machine on

Configs vary microbatch size, compile mode and attention kernel (flash / cuDNN).
Each config trains the full 181M model for --steps updates on real shards (outputs go to
<storage>/runs/sweep-*/, deleted afterwards). Steps 1-5 include compile warm-up and are not
measured. Nothing here changes the training recipe: batch, LR and data stay frozen;
only speed settings vary. Writes <storage>/reports/speed-sweep.json.
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
CONFIGS = [  # (microbatch tokens, compile mode, attention kernel)
    (65536, "default", "auto"),
    (65536, "default", "flash"),
    (65536, "default", "cudnn"),
    (131072, "default", "flash"),
    (65536, "max-autotune-no-cudagraphs", "flash"),
]


def failure_reason(log):
    """Last error-looking line of the child's output, so a failure is not just "failed"."""
    lines = [line for line in log.splitlines() if "Error" in line or "error" in line or "STOP" in line]
    return (lines[-1] if lines else (log.splitlines() or ["no output"])[-1])[-300:]


def run_config(a, micro, mode, attention):
    name = f"sweep-{micro}-{mode}-{attention}"
    out = Path(a.storage) / "runs" / name
    shutil.rmtree(out, ignore_errors=True)
    command = [sys.executable, str(TRAIN), "--stage", "pretrain", "--storage", a.storage, "--name", name,
               "--lr", "0.003", "--total-tokens", str(a.steps * 524288), "--batch-tokens", "524288",
               "--micro-tokens", str(micro), "--compile-mode", mode, "--attention", attention, "--long-frac", "0",
               "--warmup-steps", "5", "--ckpt-every", "100000", "--eval-every", "100000",
               "--val-tokens", "8192", "--no-auto-stop"]
    started = time.monotonic()
    log_path = Path(a.storage) / "reports" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:  # keep the child's output for reading afterwards
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = child.wait()
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()] \
        if (out / "metrics.jsonl").exists() else []
    measured = [r for r in rows if r["step"] > 5]
    shutil.rmtree(out, ignore_errors=True)  # a 2 GB checkpoint per config is not worth keeping
    if code or not measured:
        return dict(micro_tokens=micro, compile_mode=mode, attention=attention, ok=False, exit=code,
                    reason=failure_reason(log_path.read_text()), log=str(log_path))
    return dict(micro_tokens=micro, compile_mode=mode, attention=attention, ok=True,
                tokens_per_second=statistics.median(r["tokens_per_second"] for r in measured),
                mfu=statistics.median(r["mfu"] for r in measured),
                data_wait_frac=statistics.median(r.get("data_wait_frac", 0) for r in measured),
                gpu_mem_gb=max((r.get("gpu_mem_gb") or 0) for r in measured),
                final_loss=measured[-1]["loss"], wall_seconds=round(time.monotonic() - started))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--no-stop", action="store_true", help="do not pause the machine at the end")
    a = ap.parse_args()
    results = []
    try:
        for micro, mode, attention in CONFIGS:
            results.append(run_config(a, micro, mode, attention))
            print("SWEEP " + json.dumps(results[-1]), flush=True)
        good = [r for r in results if r["ok"]]
        best = max(good, key=lambda r: r["tokens_per_second"]) if good else None
        report = dict(results=results, best=best,
                      hours_for_10B=round(10e9 / best["tokens_per_second"] / 3600, 1) if best else None)
        path = Path(a.storage) / "reports" / "speed-sweep.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"\n{'micro':>7} {'compile':<28} {'attn':<6} {'tok/s':>9} {'MFU':>6} {'data wait':>9} {'GB':>5}")
        for r in results:
            if r["ok"]:
                print(f"{r['micro_tokens']:>7} {r['compile_mode']:<28} {r['attention']:<6} "
                      f"{r['tokens_per_second']:>9,.0f} {r['mfu']:>6.1%} {r['data_wait_frac']:>9.1%} "
                      f"{r['gpu_mem_gb']:>5.1f}")
            else:
                print(f"{r['micro_tokens']:>7} {r['compile_mode']:<28} {r['attention']:<6} failed: {r['reason']}")
        if best:
            print(f"\nBest: micro {best['micro_tokens']}, compile {best['compile_mode']}, "
                  f"attention {best['attention']} -> "
                  f"10B tokens in ~{report['hours_for_10B']} h. Report: {path}")
    finally:
        if not a.no_stop:
            code = subprocess.run([sys.executable, str(Path(__file__).with_name("stop_machine.py"))]).returncode
            if code:
                print("MACHINE PAUSE FAILED: pause it manually now!", flush=True)


if __name__ == "__main__":
    main()
