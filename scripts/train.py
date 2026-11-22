"""Readable pretraining/SFT loop. See docs/PRETRAINING.md and docs/CHAT_TUNING.md."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import queue
import random
import signal
import shutil
import subprocess
import sys
import time
import traceback
import tempfile
import threading

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dotenv import load_dotenv
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chotu.config import (EXPECTED_PARAMS, FINAL_CONTEXT, PRETRAINING_WINDOW,
                          TOKENIZER_ID, chotu_config, long_context_config, tiny_config)
from chotu.data import PretrainBatches, ShardStream
from chotu.chat_data import ChatBatches, ChatExamples


# Dense bf16 peak TFLOPS (no sparsity), matched by substring of the CUDA device name.
# Unsure about a card? Pass --peak-tflops from its datasheet.
PEAK_TFLOPS = [("H100 PCIe", 756), ("H100", 989), ("H200", 989), ("B200", 2250),
               ("A100", 312), ("A30", 165), ("L40S", 362), ("L4", 121),
               ("RTX PRO 6000", 504)]
LOSS_CHUNK = 8192  # tokens per lm_head chunk when Liger is missing (bounds logits memory)
H100_MIN_TOK_S = 74_000  # frozen kill rule: "MFU under 10%, about 74k tokens/s on an SXM H100"
ATTENTION = ("auto", "flash", "cudnn", "efficient")


def peak_flops(device, override=None):
    """Card's dense bf16 peak; CPU/unknown cards report H100-equivalent MFU."""
    if override:
        return override * 1e12
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        for key, tflops in PEAK_TFLOPS:
            if key in name:
                return tflops * 1e12
    return 989e12


def flops_per_token(params, cfg, seq_len):
    """Training FLOPs per token: 6N for the weights plus attention scores (PaLM appendix B)."""
    return 6 * params + 12 * cfg.num_hidden_layers * cfg.num_attention_heads * cfg.head_dim * seq_len


def min_tokens_per_second(peak):
    """The frozen 74k tok/s floor is for an H100 SXM; other cards get the same fraction of their peak."""
    return H100_MIN_TOK_S * peak / 989e12


def set_attention(name, device):
    """Pick the SDPA kernel. Anything but 'auto' turns off the slow math kernel, so a silent
    fallback becomes a loud error. Speed only: the attention maths is the same."""
    if device.type != "cuda" or name == "auto":
        return
    backends = torch.backends.cuda
    backends.enable_flash_sdp(name == "flash")
    backends.enable_cudnn_sdp(name == "cudnn")
    backends.enable_mem_efficient_sdp(name == "efficient")
    backends.enable_math_sdp(False)


def probe_attention(cfg, device, seq_len):
    """Run one attention call with the model's real shapes through the fast kernels only.

    The model calls SDPA with is_causal=True and enable_gqa=True (10 query / 5 KV heads). If no
    fused kernel accepts that, PyTorch quietly uses the math kernel, which is many times slower.
    """
    if device.type != "cuda":
        return "cpu"
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(1, cfg.num_attention_heads, seq_len, cfg.head_dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(1, cfg.num_key_value_heads, seq_len, cfg.head_dim, device=device, dtype=torch.bfloat16)
    working = []
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel([backend]):
                F.scaled_dot_product_attention(q, kv, kv, is_causal=True, enable_gqa=True)
            working.append(backend.name.lower())
        except RuntimeError:
            pass
    if not working:
        print("WARNING: no fused attention kernel accepts our shapes; SDPA will use the slow math "
              "kernel. Expect very low MFU.", flush=True)
    return ",".join(working) or "math only"


def prefetch(batches, start, steps, micro_tokens, pin):
    """Read the next steps' batches on a background thread while the GPU trains.

    Yields (step, microbatches, seconds the GPU waited for data). Order is unchanged,
    so resume and determinism behave exactly as before.
    """
    ready = queue.Queue(maxsize=2)
    finished = threading.Event()

    def work():
        try:
            for step in range(start, steps):
                micro = [tuple(t.pin_memory() if pin else t for t in batch)
                         for batch in batches.batch(step, micro_tokens)]
                while not finished.is_set():
                    try:
                        ready.put((step, micro), timeout=0.5)
                        break
                    except queue.Full:
                        pass
                if finished.is_set():
                    return
        except BaseException as error:  # hand reader errors to the training thread
            ready.put(error)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        for _ in range(start, steps):
            clock = time.monotonic()
            item = ready.get()
            if isinstance(item, BaseException):
                raise item
            yield item[0], item[1], time.monotonic() - clock
    finally:
        finished.set()


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


def chat_lr(step, steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    return peak * max(0, steps - 1 - step) / max(1, steps - warmup - 1)


class KillRules:
    def __init__(self, min_tok_s=H100_MIN_TOK_S):
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
def validate(model, stream, tokenizer, device, tokens=1_048_576, batch_tokens=8192):
    model.eval()
    nats, rows = 0.0, []
    # Fixed 2k windows at every stage for comparable validation scores.
    length = min(PRETRAINING_WINDOW, stream.size - 1, tokens)
    if length < 1:
        raise ValueError("validation needs at least two tokens")
    offsets = list(range(0, min(tokens, stream.size - 1) - length + 1, length))
    batch_size = max(1, batch_tokens // length)
    for start in range(0, len(offsets), batch_size):
        x, y = stream.batch(offsets[start:start + batch_size], length)
        with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
            hidden = model.model(x.to(device), use_cache=False).last_hidden_state
            total = torch.zeros((), device=device)
            for chunk, target in zip(hidden.reshape(-1, hidden.shape[-1]).split(LOSS_CHUNK),
                                     y.to(device).reshape(-1).split(LOSS_CHUNK)):
                total += F.cross_entropy(model.lm_head(chunk).float(), target, reduction="sum")
            nats += total.item()
        rows.extend(y.tolist())
    model.train()
    return bits_per_byte(nats, rows, tokenizer)


def causal_loss(model, hidden, targets, fused_ce=None):
    flat, targets = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
    if fused_ce is not None:
        return fused_ce(model.lm_head.weight, flat, targets)

    def chunk_loss(chunk, target):
        return F.cross_entropy(model.lm_head(chunk).float(), target, reduction="sum")

    return sum(checkpoint(chunk_loss, chunk, target, use_reentrant=False)
               for chunk, target in zip(flat.split(LOSS_CHUNK), targets.split(LOSS_CHUNK))) / (targets != -100).sum()


@torch.no_grad()
def validate_chat(model, examples, device, micro_tokens):
    model.eval()
    nats, count = 0.0, 0
    for indices in examples.microbatches(micro_tokens):
        x, y, attention = examples.batch(indices)
        with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
            hidden = model.model(x.to(device), attention_mask=attention.to(device), use_cache=False).last_hidden_state
            targets = y.to(device)
            valid = (targets != -100).sum().item()
            nats += causal_loss(model, hidden, targets).item() * valid
            count += valid
    model.train()
    if not count or not math.isfinite(nats):
        raise TrainingStop(20, "invalid chat validation loss")
    return nats / count


def export_model(out, model, tokenizer, name):
    """Publish a complete HF directory via an atomic, relative symlink swap."""
    exports = out / "hf"
    exports.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=name + "-", dir=exports))
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    pointer = out / f".{name}.link"
    pointer.unlink(missing_ok=True)
    pointer.symlink_to(directory.relative_to(out), target_is_directory=True)
    os.replace(pointer, out / name)
    for old in exports.glob(name + "-*"):
        if old != directory:
            shutil.rmtree(old)


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
    chat = a.stage == "chat"
    a.lr = a.lr if a.lr is not None else (3e-4 if chat else None)
    if a.lr is None:
        raise ValueError("--lr is required for pretraining; choose it with the LR test")
    a.batch_tokens = a.batch_tokens if a.batch_tokens is not None else (32768 if chat else 524288)
    a.micro_tokens = a.micro_tokens if a.micro_tokens is not None else (8192 if chat else 65536)
    a.warmup_steps = a.warmup_steps if a.warmup_steps is not None else (20 if chat else 700)
    storage = Path(a.storage).resolve()
    out = (Path(a.out) if a.out else storage / "runs" / (a.name or a.stage)).resolve()
    if not out.is_relative_to(storage):
        raise ValueError("--out must be inside --storage")
    out.mkdir(parents=True, exist_ok=True)
    if a.batch_tokens <= 0 or a.micro_tokens <= 0 or (not chat and a.batch_tokens % a.micro_tokens):
        raise ValueError("micro tokens must divide positive batch tokens")
    if not chat and a.micro_tokens % FINAL_CONTEXT:
        raise ValueError("micro tokens must be divisible by 8192")
    if a.lr <= 0 or a.warmup_steps < 0 or not 0 < a.decay_frac <= 1:
        raise ValueError("invalid LR schedule")
    if min(a.ckpt_every, a.eval_every, a.val_tokens) < 1:
        raise ValueError("checkpoint/eval intervals and validation tokens must be positive")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if chat:
        if not a.init:
            raise ValueError("chat requires --init pretrain checkpoint or HF directory")
        examples, val = ChatExamples(storage, "train"), ChatExamples(storage, "val")
        batches = ChatBatches(examples, a.epochs, a.batch_tokens, a.micro_tokens, a.seed, a.max_steps)
        paths = examples.paths + val.paths
    else:
        stream = ShardStream(storage, seed=a.seed)
        batches = PretrainBatches(stream, int(a.total_tokens) if a.total_tokens else stream.size - 1,
                                  a.batch_tokens, a.long_frac)
        print("TAIL " + json.dumps(batches.tail_stats), flush=True)
        if a.long_frac and batches.steps <= batches.switch_step:
            raise ValueError("tail cannot supply one long-context batch; use more tokens or smaller batch")
        val = ShardStream(storage, "val")
        paths = stream.paths + val.paths
    speed_only = {"out", "name", "ckpt_every", "eval_every", "no_compile", "device", "no_auto_stop",
                  "compile_mode", "peak_tflops", "attention", "allow_unfused_ce", "min_tok_s"}
    signature = {k: v for k, v in vars(a).items() if k not in speed_only}
    if not chat:
        for key in ("init", "epochs", "max_steps"):
            signature.pop(key)
    signature["shards"] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                           for p in paths]
    if chat:
        initial = Path(a.init).resolve()
        init_paths = [initial] if initial.is_file() else sorted(p for p in initial.rglob("*") if p.is_file())
        if not init_paths:
            raise ValueError("empty/missing initialization")
        signature["init_manifest"] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in init_paths]
    # JSON normalizes tuples in the persisted signature.
    signature = json.loads(json.dumps(signature))
    checkpoints = sorted(out.glob("checkpoint-*.pt"))
    saved = torch.load(checkpoints[-1], map_location="cpu", weights_only=False) if checkpoints else None
    # Older checkpoints may still hold keys that are now speed-only (e.g. min_tok_s).
    if saved and {k: v for k, v in saved["signature"].items() if k not in speed_only} != signature:
        raise ValueError("resume command or shard manifest differs from checkpoint")
    if (out / "done.json").exists():
        if not saved:
            raise ValueError("done marker without checkpoint")
        print("Already complete", flush=True)
        return
    if chat:
        if saved:
            cfg = Qwen3Config.from_dict(saved["config"])
            cfg._attn_implementation = "sdpa"
            model = Qwen3ForCausalLM(cfg)
        elif Path(a.init).is_dir():
            model = Qwen3ForCausalLM.from_pretrained(a.init, attn_implementation="sdpa")
        else:
            initial = torch.load(a.init, map_location="cpu", weights_only=False)
            cfg = Qwen3Config.from_dict(initial["config"])
            cfg._attn_implementation = "sdpa"
            model = Qwen3ForCausalLM(cfg)
            model.load_state_dict(initial["model"])
        model.to(device)
        switch_context(model)
    else:
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
    peak = peak_flops(device, a.peak_tflops)
    rules = KillRules(a.min_tok_s if a.min_tok_s is not None else min_tokens_per_second(peak))
    best, start, stage = float("inf"), 0, "chat" if chat else "short"
    if saved:
        stage, start, best = saved["stage"], saved["step"], saved["best"]
        if stage == "long":
            switch_context(model)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        floor = rules.min_tok_s  # this run's card/override, not the one the checkpoint ran on
        rules.__dict__.update(saved["rules"])
        rules.min_tok_s = floor
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
        torch.backends.cuda.matmul.allow_tf32 = True  # any leftover fp32 matmuls use tensor cores
        torch.backends.cudnn.allow_tf32 = True
        try:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            fused_ce = LigerFusedLinearCrossEntropyLoss()
        except ImportError:
            if a.model == "chotu" and not a.allow_unfused_ce:
                raise RuntimeError("liger-kernel is missing; the frozen recipe uses fused cross-entropy. "
                                   "pip install liger-kernel==0.8.5 (or pass --allow-unfused-ce)")
            print("WARNING: liger-kernel missing; using slower chunked cross-entropy", flush=True)
        if not chat:
            set_attention(a.attention, device)

    def compiled():
        if device.type != "cuda" or a.no_compile:
            return model.model
        return torch.compile(model.model, mode=None if a.compile_mode == "default" else a.compile_mode)

    forward = compiled()
    print(f"SPEED device={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'} "
          f"peak={peak / 1e12:.0f}TFLOPS fused_ce={fused_ce is not None} "
          f"compile={'off' if forward is model.model else a.compile_mode} micro_tokens={a.micro_tokens} "
          f"attention={a.attention if not chat else 'auto'} "
          f"fused_kernels_ok={probe_attention(model.config, device, FINAL_CONTEXT if chat else PRETRAINING_WINDOW)} "
          f"min_tok_s={rules.min_tok_s:,.0f}", flush=True)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        model.train()
        clock = run_clock = time.monotonic()
        run_tokens = 0
        for step, microbatches, data_wait in prefetch(batches, start, batches.steps, a.micro_tokens,
                                                      pin=device.type == "cuda"):
            if not chat and step >= batches.switch_step and stage != "long":
                switch_context(model)
                stage = "long"
                forward = compiled()
                print(f"SWITCH step={step} seq_len=8192 theta={model.config.rope_theta}", flush=True)
            length = FINAL_CONTEXT if stage in {"long", "chat"} else PRETRAINING_WINDOW
            lr = chat_lr(step, batches.steps, a.lr, a.warmup_steps) if chat else wsd_lr(step, batches.steps, a.lr, a.warmup_steps, a.decay_frac)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            # Target counts come from CPU tensors, so nothing here waits for the GPU.
            total_targets = sum(int((batch[1] != -100).sum()) for batch in microbatches)
            token_count = sum(batch[0].numel() for batch in microbatches)
            loss_sum = torch.zeros((), device=device)
            for batch in microbatches:
                x, y = batch[:2]
                weight = int((y != -100).sum()) / total_targets
                attention = batch[2].to(device, non_blocking=True) if chat else None
                if chat:
                    length = x.shape[1]
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
                    hidden = forward(x, attention_mask=attention, use_cache=False).last_hidden_state
                    loss = causal_loss(model, hidden, y, fused_ce)
                (loss * weight).backward()
                loss_sum += loss.detach() * weight
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # The step's single GPU sync: read loss and gradient norm together.
            loss_sum, grad_norm = (float(v) for v in torch.stack([loss_sum, grad_norm.float()]).tolist())
            if not math.isfinite(loss_sum):
                raise TrainingStop(20 if a.kill_rules else 1, "non-finite loss")
            if not math.isfinite(grad_norm):
                raise TrainingStop(20 if a.kill_rules else 1, "non-finite gradient")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.monotonic()
            step_seconds, clock = now - clock, now
            speed = token_count / step_seconds
            run_tokens += token_count
            if a.kill_rules and not chat:
                rules.check(step + 1, loss_sum, speed)
            record = dict(step=step + 1, loss=loss_sum, lr=lr, grad_norm=grad_norm,
                          tokens_per_second=speed,
                          mfu=flops_per_token(params, model.config, length) * speed / peak,
                          data_wait_frac=data_wait / step_seconds,
                          # Includes eval/checkpoint time: what the bill actually sees.
                          wall_tokens_per_second=run_tokens / (now - run_clock),
                          gpu_mem_gb=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                          seq_len=length, theta=model.config.rope_theta, val_bpb=None, val_loss=None)
            epoch_end = chat and step + 1 in batches.epoch_ends
            if (step + 1) % a.eval_every == 0 or step + 1 == batches.steps or epoch_end:
                metric = validate_chat(model, val, device, a.micro_tokens) if chat else validate(model, val, tokenizer, device, a.val_tokens, a.micro_tokens)
                record["val_loss" if chat else "val_bpb"] = metric
                if metric < best:
                    best = metric
                    if chat:
                        export_model(out, model, tokenizer, "best")
            record["best_val_loss" if chat else "best_val_bpb"] = best if math.isfinite(best) else None
            if chat:
                record["epoch"] = batches.updates[step][0] + 1
            with (out / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if wandb_run:
                wandb_call(wandb_run.log, data=record, step=step + 1)
            if (step + 1) % a.ckpt_every == 0 or step + 1 == batches.steps or stop_requested:
                saving = time.monotonic()
                save_checkpoint(out, model, optimizer, step + 1, stage, rules, best, signature)
                print(f"CKPT step={step + 1} seconds={time.monotonic() - saving:.1f}", flush=True)
            if stop_requested:
                raise TrainingStop(24, "signal received; checkpoint saved")
            clock = time.monotonic()  # exclude logging/eval/checkpoint time from speed
        if not chat:
            export_model(out, model, tokenizer, "final")
        marker = out / "done.tmp"
        marker.write_text(json.dumps({"step": batches.steps,
                                     "best_val_loss" if chat else "best_val_bpb": best}))
        os.replace(marker, out / "done.json")
        print(f"WALL tokens_per_second={run_tokens / max(1e-9, time.monotonic() - run_clock):,.0f} "
              "(includes eval and checkpoint time)", flush=True)
        print(f"DONE steps={batches.steps} best_{'val_loss' if chat else 'val_bpb'}={best}", flush=True)
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
    ap.add_argument("--stage", choices=["pretrain", "chat"], required=True)
    ap.add_argument("--model", choices=["chotu", "tiny"], default="chotu")
    ap.add_argument("--storage", required=True)
    ap.add_argument("--out")
    ap.add_argument("--name")
    ap.add_argument("--init", help="pretrain .pt checkpoint or HF model directory (chat only)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, help="cap chat updates for rehearsal")
    ap.add_argument("--total-tokens", type=float)
    ap.add_argument("--lr", type=float, help="required for pretrain; chat default 3e-4")
    ap.add_argument("--batch-tokens", type=int)
    ap.add_argument("--micro-tokens", type=int)
    ap.add_argument("--long-frac", type=float, default=0.08)
    ap.add_argument("--warmup-steps", type=int)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-tokens", type=int, default=1_048_576)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "max-autotune-no-cudagraphs", "max-autotune"],
                    help="speed only; try max-autotune-no-cudagraphs in the speed sweep")
    ap.add_argument("--attention", choices=ATTENTION, default="auto",
                    help="pretraining SDPA kernel, speed only; non-auto forbids the slow math fallback")
    ap.add_argument("--allow-unfused-ce", action="store_true",
                    help="let the full model train without liger-kernel (slow; not the frozen recipe)")
    ap.add_argument("--peak-tflops", type=float, help="override the card's dense bf16 peak for MFU")
    ap.add_argument("--no-auto-stop", action="store_true", help="orchestrator owns machine shutdown")
    ap.add_argument("--kill-rules", action="store_true")
    ap.add_argument("--min-tok-s", type=float,
                    help="kill-rule floor; default 74k on an H100 SXM, scaled by peak on other cards")
    ap.add_argument("--device", choices=["cpu", "cuda"])
    return ap


def main():
    load_dotenv()
    a = parser().parse_args()
    cuda = a.device == "cuda" or (a.device is None and torch.cuda.is_available())
    code = 0
    try:
        if cuda and not a.no_auto_stop and not (os.getenv("JL_MACHINE_ID") and os.getenv("JL_API_KEY")):
            raise ValueError("CUDA jobs require JL_MACHINE_ID / JL_API_KEY and jarvislabs for auto-stop")
        run(a)
    except TrainingStop as error:
        code = error.code
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        if cuda and not a.no_auto_stop:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("stop_machine.py"))])
            if result.returncode:
                print(f"STOP code=25: machine pause failed (training exit={code}); pause manually!", flush=True)
                code = 25
    return code


if __name__ == "__main__":
    sys.exit(main())
