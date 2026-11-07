"""Chotu pretraining: one readable training script for the twin (7M) and the real model (181M).

Frozen recipe (CLAUDE.md): AdamW (0.9, 0.95), weight decay 0.1, gradient clipping 1.0, bf16,
warm up / hold / slow down (decay over the final 20%), SDPA attention, torch.compile, fused AdamW.

Every run writes <out>/metrics.jsonl, one JSON object per line:
    {"t": "meta"}    run settings, machine, price            {"t": "step"}  loss, lr, grad, speed, MFU, cost
    {"t": "eval"}    held-out loss and bits-per-byte         {"t": "ckpt"}  a checkpoint was saved
    {"t": "sample"}  what the model writes from a prompt     {"t": "alert"} a kill rule or warning fired
    {"t": "copy"}    the copying test                        {"t": "end"}   why the run stopped
plus <out>/summary.json. With --live-push the logs are pushed to the `live-runs` branch every N seconds,
so the shared dashboard can follow the run.

Examples (from the repo root):
    python train.py --model twin --data ~/chotu-data --out ~/chotu-runs/twin-main --tokens 140e6 --compile
    python train.py --model twin ... --resume                 # continue from the newest checkpoint
    python train.py --model chotu ... --max-steps 30          # speed test of the real model
"""
import argparse, glob, json, math, os, pathlib, shutil, subprocess, sys, threading, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from chotu.config import TOKENIZER_ID, chotu_config, tiny_config  # noqa: E402
from transformers import Qwen3ForCausalLM  # noqa: E402

BYTES_PER_TOKEN = 4.54  # measured on our shards in notebooks/eda_data.ipynb; turns loss into bits-per-byte
START_LOSS = math.log(49_152)  # 10.80: blind guessing over the whole deck
PEAK_TFLOPS = {"H200": 989, "H100": 989, "H800": 989, "A100": 312, "L40S": 362, "L4": 121, "A10": 125,
               "B200": 2250, "B300": 2250, "RTX PRO 6000": 500}  # dense bf16, approximate spec sheets
PROMPTS = ["The capital of France is", "Photosynthesis is the process by which", "Once upon a time, there was a"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", choices=["twin", "chotu"], default="twin")
ap.add_argument("--data", required=True, help="folder holding shards/train_*.bin and shards/val_*.bin")
ap.add_argument("--out", required=True, help="run folder on the persistent volume (logs + checkpoints)")
ap.add_argument("--name", default="", help="run name on the dashboard (default: the out folder's name)")
ap.add_argument("--kind", default="train", help="what this run is for: train, overfit, firedrill-ref, firedrill, speed")
ap.add_argument("--tokens", type=float, default=140e6, help="total tokens to read (twin: 140M = 20 per knob)")
ap.add_argument("--batch-tokens", type=int, default=0, help="tokens per nudge (default: twin 65,536, chotu 524,288)")
ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--micro", type=int, default=0, help="windows per forward pass (gradient accumulation); default 8 twin / 32 chotu")
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--warmup", type=int, default=0, help="warm-up steps (default: 5%% of steps, at most 700)")
ap.add_argument("--decay", type=float, default=0.2, help="fraction of steps at the end where the stride shrinks to 0")
ap.add_argument("--max-steps", type=int, default=0, help="stop early after this many steps (speed test)")
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--eval-windows", type=int, default=64)
ap.add_argument("--ckpt-every", type=int, default=250)
ap.add_argument("--keep", type=int, default=3, help="checkpoints to keep (newest)")
ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --out")
ap.add_argument("--crash-at", type=int, default=0, help="fire drill: die abruptly right after this step")
ap.add_argument("--overfit", action="store_true", help="read the same batch every step (the loop must memorise it)")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--samples", action="store_true", help="write from fixed prompts at every exam")
ap.add_argument("--strict", action="store_true", help="enforce the kill rules (stop the run), not just alert")
ap.add_argument("--gpu-price", type=float, default=float(os.getenv("GPU_PRICE", "140")), help="rupees per GPU-hour")
ap.add_argument("--live-push", type=int, default=0, help="push logs to the live-runs branch every N seconds (0 = off)")
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()

# ---------- machine ----------
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if dev == "cuda" else dev
peak = next((v for k, v in PEAK_TFLOPS.items() if k in gpu_name.upper().replace("NVIDIA ", "")), 0) if dev == "cuda" else 0
amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
if dev == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

out = pathlib.Path(a.out).expanduser()
(out / "ckpt").mkdir(parents=True, exist_ok=True)
name = a.name or out.name
LOG = out / "metrics.jsonl"


def log(**rec):
    rec.setdefault("time", round(time.time(), 1))
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------- data: windows chosen by step number, so a resumed run reads exactly what it would have read ----------
def load(split):
    files = sorted(glob.glob(str(pathlib.Path(a.data).expanduser() / "shards" / f"{split}_*.bin")))
    if not files:
        sys.exit(f"No {split} shards in {a.data}/shards")
    return [np.memmap(f, dtype=np.uint16, mode="r") for f in files]


train_shards, val_shards = load("train"), load("val")


def windows(shards, seed, n):
    rng = np.random.default_rng(seed)
    out_ = []
    for _ in range(n):
        arr = shards[rng.integers(len(shards))]
        s = rng.integers(0, len(arr) - a.seq - 1)
        out_.append(arr[s:s + a.seq + 1])
    w = torch.from_numpy(np.stack(out_).astype(np.int64))
    return w[:, :-1], w[:, 1:]


# ---------- model ----------
cfg = tiny_config() if a.model == "twin" else chotu_config()
cfg.max_position_embeddings = max(cfg.max_position_embeddings, a.seq)
cfg._attn_implementation = "sdpa"
torch.manual_seed(a.seed)
model = Qwen3ForCausalLM(cfg).to(dev)
n_params = sum(p.numel() for p in model.parameters())
flops_per_token = 6 * n_params + 12 * cfg.num_hidden_layers * cfg.hidden_size * a.seq  # knobs + attention
fwd = torch.compile(model) if a.compile else model

batch_tokens = a.batch_tokens or (65_536 if a.model == "twin" else 524_288)
bsz = max(1, batch_tokens // a.seq)
micro = min(a.micro or (8 if a.model == "twin" else 32), bsz)
total_steps = max(1, int(a.tokens // (bsz * a.seq)))
warmup = a.warmup or min(700, max(10, total_steps // 20))
decay_start = int(total_steps * (1 - a.decay))
run_steps = min(total_steps, a.max_steps) if a.max_steps else total_steps

decay, no_decay = [p for p in model.parameters() if p.dim() >= 2], [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=(dev == "cuda"))


def lr_at(s):  # warm up, hold, slow down: the same shape as Chotu's plan
    if s < warmup:
        return a.lr * (s + 1) / warmup
    if s >= decay_start:
        return a.lr * max(0.0, (total_steps - s) / max(1, total_steps - decay_start))
    return a.lr


val_x, val_y = windows(val_shards, 999_999, a.eval_windows)  # the locked exam: same windows every time


@torch.no_grad()
def exam():
    model.eval()
    tot = 0.0
    for i in range(0, len(val_x), micro):
        with amp:
            logits = fwd(val_x[i:i + micro].to(dev)).logits
        tot += F.cross_entropy(logits.float().flatten(0, 1), val_y[i:i + micro].to(dev).flatten(), reduction="sum").item()
    model.train()
    loss = tot / val_y.numel()
    return loss, loss / math.log(2) / BYTES_PER_TOKEN


tok = None
if a.samples:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:  # samples are a bonus; never stop training for them
        print("tokenizer unavailable, skipping samples:", e)


@torch.no_grad()
def write_samples(step):
    if tok is None:
        return
    model.eval()
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        for _ in range(40):  # greedy, one token at a time: what Chotu bets on most
            with amp:
                nxt = model(ids[:, -a.seq:]).logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
        log(t="sample", step=step, prompt=p, text=tok.decode(ids[0, -40:]))
    model.train()


@torch.no_grad()
def copying_test():  # a random sequence shown twice: a model that learned to look back finds the repeat easy
    model.eval()
    g = torch.Generator().manual_seed(7)
    seq = torch.randint(1000, 40_000, (8, 128), generator=g)
    x = torch.cat([seq, seq], 1).to(dev)
    with amp:
        logits = model(x[:, :-1]).logits.float()
    ce = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(8, -1)
    first, second = ce[:, :127].mean().item(), ce[:, 128:].mean().item()
    model.train()
    log(t="copy", first=round(first, 3), second=round(second, 3))
    return first, second


# ---------- checkpoints: save to a temp file, then swap in; keep the newest few ----------
def save(step, extra):
    path = out / "ckpt" / f"step_{step:06d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, **extra}, tmp)
    os.replace(tmp, path)
    for old in sorted((out / "ckpt").glob("step_*.pt"))[:-a.keep]:
        old.unlink()
    log(t="ckpt", step=step, path=str(path), mb=round(path.stat().st_size / 1e6, 1))


# ---------- live push of the logs (never blocks or breaks training) ----------
class LivePush:
    def __init__(self, every):
        self.every, self.stop_ = every, threading.Event()
        self.repo = pathlib.Path(__file__).resolve().parent
        self.dir = pathlib.Path("~/chotu-live").expanduser()
        self.warned = False
        if every:
            threading.Thread(target=self.loop, daemon=True).start()

    def git(self, *args, cwd=None):
        return subprocess.run(["git", *args], cwd=cwd or self.dir, capture_output=True, text=True, timeout=60)

    def push(self):
        try:
            if not (self.dir / ".git").exists():
                self.dir.mkdir(parents=True, exist_ok=True)
                url = self.git("remote", "get-url", "origin", cwd=self.repo).stdout.strip()
                self.git("init", "-q", "-b", "live-runs")
                self.git("remote", "add", "origin", url)
                for k in ("user.name", "user.email"):
                    v = self.git("config", k, cwd=self.repo).stdout.strip()
                    if v:
                        self.git("config", k, v)
                fetched = self.git("fetch", "-q", "--depth", "1", "origin", "live-runs")
                if fetched.returncode == 0:
                    self.git("reset", "-q", "--soft", "FETCH_HEAD")
            dest = self.dir / "runs" / name
            dest.mkdir(parents=True, exist_ok=True)
            for f in ("metrics.jsonl", "summary.json"):
                if (out / f).exists():
                    shutil.copy(out / f, dest / f)
            self.git("add", "-A")
            has_head = self.git("rev-parse", "-q", "--verify", "HEAD").returncode == 0
            self.git("commit", "-q", "--amend", "--no-edit") if has_head else self.git("commit", "-q", "-m", "Live run logs")
            r = self.git("push", "-q", "-f", "origin", "live-runs")
            if r.returncode and not self.warned:
                self.warned = True
                print("live push failed (training continues):", r.stderr.strip()[:300], flush=True)
        except Exception as e:  # noqa: BLE001
            if not self.warned:
                self.warned = True
                print("live push failed (training continues):", e, flush=True)

    def loop(self):
        while not self.stop_.wait(self.every):
            self.push()

    def finish(self):
        if self.every:
            self.stop_.set()
            self.push()


live = LivePush(a.live_push)

# ---------- start or resume ----------
step0, tokens_seen, spent_s = 0, 0, 0.0
ckpts = sorted((out / "ckpt").glob("step_*.pt"))
if a.resume and ckpts:
    ck = torch.load(ckpts[-1], map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    step0, tokens_seen, spent_s = ck["step"], ck["tokens_seen"], ck["spent_s"]
    log(t="resume", step=step0, path=str(ckpts[-1]))
    print(f"resumed from {ckpts[-1].name} (step {step0})", flush=True)
elif not a.resume and LOG.exists():
    LOG.unlink()

log(t="meta", name=name, kind=a.kind, model=a.model, params=n_params, device=dev, gpu=gpu_name, peak_tflops=peak,
    gpu_price=a.gpu_price, steps=total_steps, run_steps=run_steps, batch_tokens=bsz * a.seq, seq=a.seq, micro=micro,
    lr=a.lr, warmup=warmup, decay_start=decay_start, tokens=total_steps * bsz * a.seq, compile=a.compile,
    overfit=a.overfit, crash_at=a.crash_at, resumed=step0 > 0, torch=torch.__version__)
print(f"{name}: {a.model} {n_params:,} knobs on {gpu_name} | {run_steps} of {total_steps} steps x {bsz * a.seq:,} tokens "
      f"| lr {a.lr} warmup {warmup} decay from {decay_start}", flush=True)

if step0 == 0:
    vl, vb = exam()
    log(t="eval", step=0, val_loss=round(vl, 4), val_bpb=round(vb, 4))

# ---------- the loop ----------
t_start = time.time()
ema, end_reason, alerts = None, "finished", set()


def alert(kind, msg, step, stop=False):
    if kind not in alerts:
        alerts.add(kind)
        log(t="alert", kind=kind, msg=msg, step=step)
        print(f"ALERT [{kind}] step {step}: {msg}", flush=True)
    return stop


fixed = windows(train_shards, a.seed, bsz) if a.overfit else None
step = step0
try:
    for step in range(step0 + 1, run_steps + 1):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)
        x, y = fixed or windows(train_shards, a.seed * 1_000_003 + step, bsz)
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=dev)
        for i in range(0, bsz, micro):
            xb, yb = x[i:i + micro].to(dev, non_blocking=True), y[i:i + micro].to(dev, non_blocking=True)
            with amp:
                logits = fwd(xb).logits
            part = F.cross_entropy(logits.float().flatten(0, 1), yb.flatten()) * len(xb) / bsz
            part.backward()
            loss += part.detach()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        lv = loss.item()  # waits for the GPU, so the timing below is real
        dt = time.time() - t0
        tokens_seen += bsz * a.seq
        spent_s += dt
        tok_s = bsz * a.seq / dt
        mfu = tok_s * flops_per_token / (peak * 1e12) if peak else None
        elapsed = time.time() - t_start
        log(t="step", step=step, tokens=tokens_seen, loss=round(lv, 4), lr=lr_at(step - 1), grad=round(grad, 4),
            tok_s=round(tok_s), mfu=round(mfu, 4) if mfu else None, dt=round(dt, 3), elapsed=round(elapsed, 1),
            cost=round(elapsed / 3600 * a.gpu_price, 2))
        if step % 10 == 0 or step == step0 + 1:
            print(f"step {step}/{run_steps} loss {lv:.3f} grad {grad:.2f} lr {lr_at(step - 1):.2e} "
                  f"{tok_s:,.0f} tok/s" + (f" MFU {mfu:.1%}" if mfu else "") + f" {elapsed:.0f}s", flush=True)

        # kill rules (Lessons 2 and 6)
        if not math.isfinite(lv) or not math.isfinite(grad):
            alert("nan", "loss or gradient is not a number: stop at once", step)
            end_reason = "nan"
            break
        if step == 1 and abs(lv - START_LOSS) > 0.6 and not a.overfit:
            alert("start", f"first loss {lv:.2f}, expected about {START_LOSS:.1f}: setup looks wrong", step)
        if step == 300 and lv > 6 and not a.overfit:
            if alert("step300", f"loss {lv:.2f} is not below 6 by step 300", step, a.strict):
                end_reason = "killed: step 300 rule"
                break
        if step == 20 and mfu is not None and mfu < 0.10:
            msg = f"MFU {mfu:.1%} is under 10%"
            if a.model == "twin":
                msg += " (expected for the 7M twin: too small to keep a big GPU busy; the rule is for the 181M run)"
            if alert("mfu", msg, step, a.strict and a.model == "chotu"):
                end_reason = "killed: MFU rule"
                break
        if ema is not None and step > warmup and lv > 1.5 * ema:
            alert(f"spike{step // 100}", f"loss spike: {lv:.2f} vs recent {ema:.2f}", step)
        ema = lv if ema is None else 0.95 * ema + 0.05 * lv

        if step % a.eval_every == 0 or step == run_steps:
            vl, vb = exam()
            log(t="eval", step=step, val_loss=round(vl, 4), val_bpb=round(vb, 4))
            print(f"  exam at step {step}: val loss {vl:.3f} = {vb:.3f} bits per byte", flush=True)
            if a.samples:
                write_samples(step)
        if step % a.ckpt_every == 0 or step == run_steps:
            save(step, {"tokens_seen": tokens_seen, "spent_s": spent_s})
        if a.crash_at and step == a.crash_at:
            log(t="end", step=step, reason="simulated crash (fire drill)")
            live.finish()
            print(f"FIRE DRILL: machine 'dies' after step {step}", flush=True)
            os._exit(3)
except KeyboardInterrupt:
    end_reason = "stopped by hand"

first, second = copying_test() if a.kind in ("train",) and step >= run_steps else (None, None)
elapsed = time.time() - t_start
steps_log = [json.loads(line) for line in open(LOG) if '"t": "step"' in line]
speed = [r["tok_s"] for r in steps_log[len(steps_log) // 2:]] or [0]  # second half: past compile and warm-up
evals = [json.loads(line) for line in open(LOG) if '"t": "eval"' in line]
summary = dict(name=name, kind=a.kind, model=a.model, params=n_params, gpu=gpu_name, reason=end_reason, step=step,
               steps=total_steps, run_steps=run_steps, tokens=tokens_seen, minutes=round(elapsed / 60, 2),
               cost_inr=round(elapsed / 3600 * a.gpu_price, 1), tok_s=round(float(np.median(speed))),
               final_loss=steps_log[-1]["loss"] if steps_log else None,
               val_bpb=evals[-1]["val_bpb"] if evals else None, copy_first=first, copy_second=second)
if peak:
    summary["mfu"] = round(summary["tok_s"] * flops_per_token / (peak * 1e12), 4)
if a.model == "chotu" and summary["tok_s"]:  # what the whole 10B-token run would take at this speed
    hours = 10.06e9 / summary["tok_s"] / 3600
    summary.update(full_run_hours=round(hours, 1), full_run_cost_inr=round(hours * a.gpu_price))
json.dump(summary, open(out / "summary.json", "w"), indent=1)
log(t="end", step=step, reason=end_reason, **{k: v for k, v in summary.items() if k not in ("name", "kind", "model", "step")})
live.finish()
print(json.dumps(summary, indent=1), flush=True)
