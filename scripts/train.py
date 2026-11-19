"""Readable pretraining loop. See docs/PRETRAINING.md for usage and exit codes."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dotenv import load_dotenv
from transformers import AutoTokenizer, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chotu.config import (EXPECTED_PARAMS, FINAL_CONTEXT, PRETRAINING_WINDOW,
                          TOKENIZER_ID, chotu_config, long_context_config, tiny_config)
from chotu.data import PretrainBatches, ShardStream


class TrainingStop(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def wsd_lr(step, steps, peak, warmup, decay_frac):
    """Zero-indexed update; linear warmup, stable peak, linear decay to zero."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    decay = max(1, math.ceil(steps * decay_frac))
    if step >= steps - decay:
        return peak * max(0, steps - 1 - step) / max(1, decay - 1)
    return peak


class KillRules:
    def __init__(self, min_tok_s=74000):
        self.min_tok_s = min_tok_s
        self.losses, self.speeds = [], []
        self.spike_since = None
        self.spike_floor = None
        self.best_loss = float("inf")

    def check(self, step, loss, tokens_per_second):
        if not math.isfinite(loss):
            raise TrainingStop(20, "non-finite loss")
        self.losses.append(loss)
        self.best_loss = min(self.best_loss, loss)
        if step == 300 and self.best_loss >= 6:
            raise TrainingStop(21, "loss never below 6 by step 300")
        if 20 <= step <= 50:
            self.speeds.append(tokens_per_second)
        if step == 50 and sum(self.speeds) / len(self.speeds) < self.min_tok_s:
            raise TrainingStop(22, "throughput below minimum over steps 20–50")
        floor = min(self.losses[-201:-1], default=loss)
        if self.spike_since is None and loss > 1.5 * floor:
            self.spike_since, self.spike_floor = step, floor
        if self.spike_since is not None:
            if loss <= 1.5 * self.spike_floor:
                self.spike_since = self.spike_floor = None
            elif step - self.spike_since >= 200:
                raise TrainingStop(23, "loss spike did not recover within 200 steps")
        self.losses = self.losses[-200:]


def switch_context(model):
    cfg = long_context_config(**{k: getattr(model.config, k) for k in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "vocab_size")})
    cfg._attn_implementation = "sdpa"
    model.config = cfg
    model.model.config = cfg
    model.model.rotary_emb = Qwen3RotaryEmbedding(cfg).to(next(model.parameters()).device)


def bits_per_byte(nats, target_rows, tokenizer):
    byte_count = sum(len(tokenizer.decode(row, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False).encode("utf-8"))
                     for row in target_rows)
    if not byte_count:
        raise ValueError("validation targets decode to zero bytes")
    return nats / math.log(2) / byte_count


def wandb_call(function, **kwargs):
    """Telemetry must never prevent checkpointing, shutdown, or offline use."""
    try:
        return function(**kwargs)
    except Exception as error:
        print(f"W&B unavailable: {type(error).__name__}", flush=True)


@torch.no_grad()
def validate(model, stream, tokenizer, device, tokens=8192):
    model.eval()
    nats, rows = 0.0, []
    # Fixed 2k windows at every stage for comparable validation scores.
    length = min(PRETRAINING_WINDOW, stream.size - 1, tokens)
    if length < 1:
        raise ValueError("validation needs at least two tokens")
    for offset in range(0, min(tokens, stream.size - 1) - length + 1, length):
        x, y = stream.batch([offset], length)
        with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
            hidden = model.model(x.to(device), use_cache=False).last_hidden_state
            for chunk, target in zip(hidden.reshape(-1, hidden.shape[-1]).split(256),
                                     y.to(device).reshape(-1).split(256)):
                nats += F.cross_entropy(model.lm_head(chunk).float(), target, reduction="sum").item()
        rows.extend(y.tolist())
    model.train()
    return bits_per_byte(nats, rows, tokenizer)


def save_checkpoint(out, model, optimizer, step, stage, rules, best, signature):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                 stage=stage, config=model.config.to_dict(), rules=rules.__dict__, best=best,
                 signature=signature, rng=torch.get_rng_state(), python_rng=random.getstate(),
                 numpy_rng=np.random.get_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    path = out / f"checkpoint-{step:08d}.pt"
    torch.save(state, path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)
    for old in sorted(out.glob("checkpoint-*.pt"))[:-2]:
        old.unlink()


def run(a):
    storage = Path(a.storage).resolve()
    out = (Path(a.out) if a.out else storage / "runs" / a.name).resolve()
    if not out.is_relative_to(storage):
        raise ValueError("--out must be inside --storage")
    out.mkdir(parents=True, exist_ok=True)
    if a.batch_tokens <= 0 or a.micro_tokens <= 0 or a.batch_tokens % a.micro_tokens:
        raise ValueError("micro tokens must divide positive batch tokens")
    if a.micro_tokens % FINAL_CONTEXT:
        raise ValueError("micro tokens must be divisible by 8192")
    if a.lr <= 0 or a.warmup_steps < 0 or not 0 < a.decay_frac <= 1:
        raise ValueError("invalid LR schedule")
    if min(a.ckpt_every, a.eval_every, a.val_tokens) < 1:
        raise ValueError("checkpoint/eval intervals and validation tokens must be positive")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stream = ShardStream(storage, seed=a.seed)
    batches = PretrainBatches(stream, int(a.total_tokens) if a.total_tokens else stream.size - 1,
                              a.batch_tokens, a.long_frac)
    print("TAIL " + json.dumps(batches.tail_stats), flush=True)
    if a.long_frac and batches.steps <= batches.switch_step:
        raise ValueError("tail cannot supply one long-context batch; use more tokens or smaller batch")
    signature = {k: v for k, v in vars(a).items() if k not in {
        "out", "name", "ckpt_every", "eval_every", "no_compile", "device"}}
    val = ShardStream(storage, "val")
    signature["shards"] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                           for p in stream.paths + val.paths]
    # JSON normalizes tuples in the persisted signature.
    signature = json.loads(json.dumps(signature))
    checkpoints = sorted(out.glob("checkpoint-*.pt"))
    saved = torch.load(checkpoints[-1], map_location="cpu", weights_only=False) if checkpoints else None
    if saved and saved["signature"] != signature:
        raise ValueError("resume command or shard manifest differs from checkpoint")
    if (out / "done.json").exists():
        if not saved:
            raise ValueError("done marker without checkpoint")
        print("Already complete", flush=True)
        return
    cfg = tiny_config() if a.model == "tiny" else chotu_config()
    cfg._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    if a.model == "chotu" and params != EXPECTED_PARAMS:
        raise RuntimeError(f"size gate failed: {params}")
    # Matrices decay, except the tied embedding; vectors (including norms) do not.
    groups = [{"params": [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed_tokens" not in n], "weight_decay": 0.1},
              {"params": [p for n, p in model.named_parameters() if p.ndim < 2 or "embed_tokens" in n], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    rules, best, start, stage = KillRules(a.min_tok_s), float("inf"), 0, "short"
    if saved:
        stage, start, best = saved["stage"], saved["step"], saved["best"]
        if stage == "long":
            switch_context(model)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rules.__dict__.update(saved["rules"])
        torch.set_rng_state(saved["rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"RESUME step={start} stage={stage}", flush=True)
        # Remove uncommitted metrics from a crash after the last checkpoint.
        log = out / "metrics.jsonl"
        if log.exists():
            kept = []
            for line in log.read_text().splitlines():
                try:
                    if json.loads(line)["step"] <= start:
                        kept.append(line)
                except (ValueError, KeyError):
                    pass
            log.write_text("".join(line + "\n" for line in kept))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    wandb_run = None
    if os.getenv("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(project=os.getenv("WANDB_PROJECT", "chotu"),
                                  entity=os.getenv("WANDB_ENTITY"), config=vars(a),
                                  id=out.name, resume="allow", dir=str(out))
        except Exception as error:
            print(f"W&B unavailable: {type(error).__name__}", flush=True)
    fused_ce = None
    if device.type == "cuda":
        try:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            fused_ce = LigerFusedLinearCrossEntropyLoss()
        except ImportError:
            pass
    forward = torch.compile(model.model) if device.type == "cuda" and not a.no_compile else model.model
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        model.train()
        for step in range(start, batches.steps):
            if step >= batches.switch_step and stage != "long":
                switch_context(model)
                stage = "long"
                forward = torch.compile(model.model) if device.type == "cuda" and not a.no_compile else model.model
                print(f"SWITCH step={step} seq_len=8192 theta={model.config.rope_theta}", flush=True)
            length = FINAL_CONTEXT if stage == "long" else PRETRAINING_WINDOW
            lr = wsd_lr(step, batches.steps, a.lr, a.warmup_steps, a.decay_frac)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            clock, loss_sum = time.monotonic(), 0.0
            for x, y in batches.batch(step, a.micro_tokens):
                x, y = x.to(device), y.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
                    hidden = forward(x, use_cache=False).last_hidden_state
                    if fused_ce is not None:
                        loss = fused_ce(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), y.reshape(-1))
                    else:
                        # Chunk the vocabulary projection to bound CPU/plain-CE memory.
                        flat, targets = hidden.reshape(-1, hidden.shape[-1]), y.reshape(-1)
                        def chunk_loss(chunk, target):
                            return F.cross_entropy(model.lm_head(chunk).float(), target, reduction="sum")
                        loss = sum(checkpoint(chunk_loss, chunk, target, use_reentrant=False)
                                   for chunk, target in zip(flat.split(256), targets.split(256))) / y.numel()
                    if not torch.isfinite(loss):
                        raise TrainingStop(20 if a.kill_rules else 1, "non-finite loss")
                    (loss * a.micro_tokens / a.batch_tokens).backward()
                    loss_sum += loss.item() * a.micro_tokens / a.batch_tokens
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            if not math.isfinite(grad_norm):
                raise TrainingStop(20 if a.kill_rules else 1, "non-finite gradient")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            speed = a.batch_tokens / (time.monotonic() - clock)
            if a.kill_rules:
                rules.check(step + 1, loss_sum, speed)
            record = dict(step=step + 1, loss=loss_sum, lr=lr, grad_norm=grad_norm,
                          tokens_per_second=speed, mfu=6 * params * speed / 989e12,
                          seq_len=length, theta=model.config.rope_theta, val_bpb=None)
            if (step + 1) % a.eval_every == 0 or step + 1 == batches.steps:
                record["val_bpb"] = validate(model, val, tokenizer, device, a.val_tokens)
                best = min(best, record["val_bpb"])
            record["best_val_bpb"] = best if math.isfinite(best) else None
            with (out / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if wandb_run:
                wandb_call(wandb_run.log, data=record, step=step + 1)
            if (step + 1) % a.ckpt_every == 0 or step + 1 == batches.steps or stop_requested:
                save_checkpoint(out, model, optimizer, step + 1, stage, rules, best, signature)
            if stop_requested:
                raise TrainingStop(24, "signal received; checkpoint saved")
        marker = out / "done.tmp"
        marker.write_text(json.dumps(dict(step=batches.steps, best_val_bpb=best)))
        os.replace(marker, out / "done.json")
        print(f"DONE steps={batches.steps} best_val_bpb={best}", flush=True)
    except TrainingStop as error:
        print(f"STOP code={error.code}: {error}", flush=True)
        if wandb_run:
            import wandb
            wandb_call(wandb.alert, title=f"Chotu stopped ({error.code})", text=str(error))
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if wandb_run:
            wandb_call(wandb_run.finish)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["pretrain"], required=True)
    ap.add_argument("--model", choices=["chotu", "tiny"], default="chotu")
    ap.add_argument("--storage", required=True)
    ap.add_argument("--out")
    ap.add_argument("--name", default="pretrain")
    ap.add_argument("--total-tokens", type=float)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--batch-tokens", type=int, default=524288)
    ap.add_argument("--micro-tokens", type=int, default=65536)
    ap.add_argument("--long-frac", type=float, default=0.08)
    ap.add_argument("--warmup-steps", type=int, default=700)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--kill-rules", action="store_true")
    ap.add_argument("--min-tok-s", type=float, default=74000)
    ap.add_argument("--device", choices=["cpu", "cuda"])
    return ap


def main():
    load_dotenv()
    a = parser().parse_args()
    cuda = a.device == "cuda" or (a.device is None and torch.cuda.is_available())
    code = 0
    try:
        if cuda and not (os.getenv("JL_MACHINE_ID") and os.getenv("JL_API_KEY")):
            raise ValueError("CUDA jobs require JL_MACHINE_ID / JL_API_KEY and jarvislabs for auto-stop")
        if a.stage == "pretrain":
            run(a)
    except TrainingStop as error:
        code = error.code
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        if cuda:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("stop_machine.py"))])
            if result.returncode:
                print(f"STOP code=25: machine pause failed (training exit={code}); pause manually!", flush=True)
                code = 25
    return code


if __name__ == "__main__":
    sys.exit(main())
