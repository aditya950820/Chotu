"""One-command, resumable training chain. See docs/RUNBOOK.md before renting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = "magnateadi/chotu-fineweb-edu-10bt-tokens"
STAGES = ("preflight", "shards", "chat_data", "speed", "lr_test", "pretrain",
          "base_evals", "chat_tune", "final_evals", "export_gguf", "upload", "final_report")
GIB = 1024**3


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def metrics(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def used_bytes(root):
    """Bytes of regular files under root (symlinks not followed, missing root = 0)."""
    total = 0
    for folder, _, files in os.walk(root):
        for name in files:
            path = Path(folder, name)
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    return total


def disk_budget(shard_bytes, params):
    checkpoint = params * 12  # fp32 weights and two Adam moments
    parts = dict(shards=shard_bytes, checkpoints=checkpoint * 2 * 5,
                 hf_exports=params * 4 * 5, atomic_temporaries=checkpoint + params * 4,
                 chat_data=6 * GIB, baselines_and_eval_data=4 * GIB,
                 llama_build=4 * GIB, download_cache=8 * GIB)
    return dict(parts=parts, required_bytes=math.ceil(sum(parts.values()) * 1.2))


def selected_stages(args):
    if args.only_stage:
        return (args.only_stage,)
    return STAGES[STAGES.index(args.from_stage):] if args.from_stage else STAGES


def dry_plan(args):
    storage = Path(args.storage).expanduser().resolve()
    if args.rehearsal:
        storage /= "rehearsal"
    speed_file = storage / "reports/speed.json"
    rates = [74000, 150000]
    if speed_file.exists():
        rates = [read_json(speed_file)["tokens_per_second"]] * 2
    train_tokens = args.rehearsal_tokens if args.rehearsal else 10_000_000_000
    lr_tokens = 65536 if args.rehearsal else 500_000_000
    speed_tokens = (4 * 8192) if args.rehearsal else (50 * 524288)
    hours = sorted((train_tokens + 2 * lr_tokens + speed_tokens) / rate / 3600 for rate in rates)
    # Extra-context attention, SFT/evals, compile, networking and export are not
    # captured by a short-window throughput measurement.
    estimates = dict(short_window_compute_hours=hours,
                     planning_hours=[hours[0] + 2, hours[1] * 1.5 + 4])
    estimates["planning_rupees_at_330_per_hour"] = [round(h * 330) for h in estimates["planning_hours"]]
    print(f"DRY RUN: no installs, downloads, uploads, markers or pause calls\nStorage: {storage}")
    for index, stage in enumerate(selected_stages(args), 1):
        print(f"{index:02d}. {stage}" + (" (explicitly skipped)" if stage == "upload" and args.skip_upload else ""))
    print(f"Tokens: main={train_tokens:,}; LR=2 x {lr_tokens:,}; speed={speed_tokens:,}")
    print(f"Throughput: {min(rates):,.0f}–{max(rates):,.0f} tok/s " + ("(saved speed check)" if speed_file.exists() else "(H100 estimate)"))
    print(json.dumps(estimates, indent=2))
    print("Costs exclude storage, retries and pauses; long context can be slower. Not a spending guarantee.")


class StageError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class ChainSignal(Exception):
    def __init__(self, number):
        self.number = number


class Chain:
    def __init__(self, args):
        self.a = args
        self.storage = Path(args.storage).expanduser().resolve()
        if args.rehearsal:
            self.storage /= "rehearsal"
        self.cache = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else self.storage / "cache"
        self.logs, self.reports = self.storage / "logs", self.storage / "reports"
        self.markers = self.storage / "stages"
        self.child, self.stage, self.wandb = None, "startup", None
        self.status_path = self.storage / "status.json"
        self.repo = args.hf_repo

    def emit(self, state, **details):
        event = dict(time=datetime.now(timezone.utc).isoformat(), stage=self.stage, state=state, **details)
        try:
            print("STATUS " + json.dumps(event, default=str), flush=True)
        except OSError:
            pass
        try:
            if self.logs.exists():
                with (self.logs / "chain.jsonl").open("a") as file:
                    file.write(json.dumps(event, default=str) + "\n")
                atomic_json(self.status_path, event)
        except OSError:
            pass
        if self.wandb:
            try:
                self.wandb.log({"stage": self.stage, "state": state, **details})
            except Exception:
                pass

    def alert(self, message):
        self.emit("ALERT", message=message)
        if self.wandb:
            try:
                import wandb
                wandb.alert(title="Chotu chain stopped", text=message)
            except Exception:
                pass

    def setup(self):
        for path in (self.logs, self.reports, self.markers, self.cache):
            path.mkdir(parents=True, exist_ok=True)
        if self.a.retries < 0 or self.a.build_jobs < 1 or self.a.rehearsal_tokens < 65536:
            raise ValueError("retries >=0, build jobs >=1 and rehearsal tokens >=65536 required")
        if self.a.from_stage and self.a.only_stage:
            raise ValueError("choose --from-stage or --only-stage, not both")
        if self.a.device == "cpu" and not self.a.rehearsal:
            raise ValueError("CPU is only supported for --rehearsal")
        if self.a.shards_subset and not self.a.rehearsal:
            raise ValueError("--shards-subset is rehearsal-only")
        signature = {k: v for k, v in vars(self.a).items() if k not in {
            "from_stage", "only_stage", "dry_run", "no_stop", "retries"}}
        scripts = [ROOT / "requirements.txt", *sorted((ROOT / "scripts").glob("*.py")),
                   *sorted((ROOT / "chotu").glob("*.py"))]
        signature["code"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}
        config = self.storage / "chain-config.json"
        if config.exists() and read_json(config) != signature:
            raise ValueError("chain settings/code changed; use a fresh storage directory (or restore original code/settings)")
        atomic_json(config, signature)
        os.environ["HF_HOME"] = str(self.cache / "huggingface")
        os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
        if self.a.device == "cpu":
            os.environ.setdefault("OMP_NUM_THREADS", "2")
            os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if os.getenv("WANDB_API_KEY"):
            project = os.getenv("WANDB_PROJECT") or "chotu"
            if self.a.rehearsal and not project.endswith("-rehearsal"):
                project += "-rehearsal"
            os.environ["WANDB_PROJECT"] = project
            try:
                import wandb
                identity = hashlib.sha256(str(self.storage).encode()).hexdigest()[:12]
                self.wandb = wandb.init(project=os.getenv("WANDB_PROJECT", "chotu"),
                    entity=os.getenv("WANDB_ENTITY"), id="chain-" + identity, resume="allow",
                    job_type="chain", dir=str(self.logs), config=vars(self.a))
            except Exception as error:
                self.emit("telemetry_unavailable", error=type(error).__name__)

    def command(self, script, *args):
        return [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]

    def execute(self, command, training=False):
        log = self.logs / f"{self.stage}.log"
        for attempt in range(self.a.retries + 1 if training else 1):
            self.emit("running", attempt=attempt + 1, log=str(log))
            with log.open("a") as output:
                output.write("\nCOMMAND " + " ".join(command) + "\n")
                output.flush()
                self.child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, start_new_session=True)
                atomic_json(self.storage / "child.json", dict(pid=self.child.pid, stage=self.stage, command=command))
                for line in self.child.stdout:
                    output.write(line)
                    output.flush()
                    if training and line.startswith("{"):
                        try:
                            row = json.loads(line)
                            if "step" in row:
                                self.emit("progress", **{k: v for k, v in row.items() if k in {
                                    "step", "loss", "tokens_per_second", "mfu", "val_bpb", "val_loss"}})
                        except ValueError:
                            pass
                code = self.child.wait()
                self.child = None
            if code == 0:
                return
            normalized = 1 if code == -signal.SIGKILL else code
            if training and normalized == 1 and attempt < self.a.retries:
                self.emit("retry", exit_code=code, message="resume latest durable training checkpoint")
                continue
            raise StageError(normalized if normalized > 0 else 1, f"{self.stage} child exited {code}; see {log}")

    def terminate_child(self):
        if self.child and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()
        self.child = None

    def on_signal(self, number, frame):
        print(f"SIGNAL {number}: checkpoint/terminate child, then pause", flush=True)
        raise ChainSignal(number)

    def stop(self):
        if self.a.no_stop:
            self.emit("pause_skipped", message="--no-stop: local/external shutdown owner")
            return True
        if not self.a.rehearsal:
            # Last full copy (logs, reports, newest checkpoint) to the private HF backup before pausing.
            self.emit("final_backup")
            try:
                with (self.logs / "backup.log").open("a") as log:
                    subprocess.run(self.command("backup.py", "--once", "--storage", self.storage), cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            except Exception as error:  # never let a backup problem block the pause
                print(f"final backup failed: {error}", flush=True)
        self.emit("pausing", message="Jarvis shutdown owner")
        try:
            with (self.logs / "stop.log").open("a") as log:
                result = subprocess.run(self.command("stop_machine.py"), cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=120)
                if result.returncode == 0:
                    return True
        except OSError:
            # A full/broken log volume must not prevent the pause API call.
            try:
                if subprocess.run(self.command("stop_machine.py"), cwd=ROOT, timeout=120).returncode == 0:
                    return True
            except Exception:
                pass
        except Exception:
            pass
        self.alert("PAUSE FAILED: PAUSE THE JARVIS INSTANCE MANUALLY NOW; see logs/stop.log")
        return False

    def shard_manifest(self):
        path = self.storage / "shard-manifest.json"
        if path.exists():
            return read_json(path)
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(DATASET, files_metadata=True)
        files = {f.rfilename: dict(size=f.size, sha256=f.lfs.sha256 if f.lfs else None)
                 for f in info.siblings if f.rfilename.startswith("shards/") and
                 (f.rfilename.endswith(".bin") or f.rfilename.endswith(".done"))}
        expected = {f"shards/{i:03d}.done" for i in range(14)}
        expected |= {f"shards/val_{i:03d}.bin" for i in range(14)}
        expected |= {f"shards/train_{i:03d}.bin" for i in ([13] if self.a.shards_subset else range(14))}
        if expected - files.keys():
            raise ValueError(f"dataset layout changed; missing {sorted(expected - files.keys())}")
        manifest = dict(repo=DATASET, revision=info.sha, files={k: files[k] for k in sorted(expected)})
        atomic_json(path, manifest)
        return manifest

    def preflight(self):
        command = self.command("check_setup.py", "--size-only")
        if self.a.device == "cpu":
            command += ["--local-only"]
            self.emit("local_preflight", message="skip live W&B/Jarvis and nvidia-smi; real HF downloads still authenticate")
        self.execute(command)
        gpu = None
        if self.a.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise ValueError("CUDA torch is unavailable; use the template's CUDA torch")
            self.execute(["nvidia-smi"])
            result = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True)
            gpu = result.strip()
            memory = float(result.splitlines()[0].rsplit(",", 1)[1])
            if memory < (4000 if self.a.rehearsal else 75000):
                raise ValueError(f"insufficient GPU memory: {gpu}")
        if not self.a.no_stop:
            self.execute(self.command("stop_machine.py", "--dry-run"))
        manifest = self.shard_manifest()
        shard_bytes = sum(f["size"] for name, f in manifest["files"].items() if name.endswith(".bin"))
        params = 180925056
        if self.a.rehearsal:
            import torch
            from transformers import Qwen3ForCausalLM
            from chotu.config import tiny_config
            with torch.device("meta"):
                params = sum(p.numel() for p in Qwen3ForCausalLM(tiny_config()).parameters())
        budget = disk_budget(shard_bytes, params)
        free = shutil.disk_usage(self.storage).free
        # Files already on the volume (shards, chat data, builds, checkpoints after a resume) are
        # part of the budget, so count them as available rather than demanding it all free again.
        present = used_bytes(self.storage) + (0 if self.cache.is_relative_to(self.storage) else used_bytes(self.cache))
        atomic_json(self.reports / "preflight.json", dict(gpu=gpu, free_bytes=free, present_bytes=present, **budget))
        if free + present < budget["required_bytes"]:
            raise ValueError(f"need {budget['required_bytes'] / GIB:.1f} GiB (free + already present); "
                             f"have {free / GIB:.1f} free + {present / GIB:.1f} present")
        return dict(gpu=gpu, free_gib=free / GIB, present_gib=present / GIB, need_gib=budget["required_bytes"] / GIB)

    def shards(self):
        from huggingface_hub import hf_hub_download
        import numpy as np
        manifest = self.shard_manifest()
        for name, expected in manifest["files"].items():
            self.emit("download", file=name, bytes=expected["size"])
            path = Path(hf_hub_download(DATASET, name, repo_type="dataset", revision=manifest["revision"],
                                       local_dir=self.storage))
            if path.stat().st_size != expected["size"]:
                raise ValueError(f"shard size mismatch: {name}")
            if expected["sha256"]:
                with path.open("rb") as file:
                    digest = hashlib.file_digest(file, "sha256").hexdigest()
                if digest != expected["sha256"]:
                    raise ValueError(f"shard SHA256 mismatch: {name}; remove corrupted local file and retry")
        tokens, docs, literal_eot = {}, {}, {}
        for name in manifest["files"]:
            if not name.endswith(".bin"):
                continue
            path = self.storage / name
            split, index = path.stem.split("_")
            counts = read_json(self.storage / "shards" / f"{index}.done")
            if path.stat().st_size != 2 * counts[f"{split}_tokens"]:
                raise ValueError(f".done token count mismatch: {name}")
            array = np.memmap(path, mode="r", dtype=np.uint16)
            n_docs = sum(int(np.count_nonzero(array[i:i + 1_000_000] == 0)) for i in range(0, len(array), 1_000_000))
            expected_docs = counts["val_docs"] if split == "val" else counts["docs"] - counts["val_docs"]
            # A few web pages contain the literal text "<|endoftext|>", which the tokenizer turns into the
            # end-of-text token, so a shard may hold slightly MORE separators than documents. Fewer, a
            # missing final separator or more than 0.1% extra would mean a broken shard.
            extra = n_docs - expected_docs
            if extra < 0 or extra > max(10, expected_docs // 1000) or (len(array) and array[-1] != 0):
                raise ValueError(f".done document count mismatch: {name} ({n_docs} separators, {expected_docs} documents)")
            literal_eot[name] = extra
            tokens[split] = tokens.get(split, 0) + len(array)
            docs[split] = docs.get(split, 0) + expected_docs
        result = dict(revision=manifest["revision"], tokens=tokens, docs=docs,
                      literal_endoftext_in_text={k: v for k, v in literal_eot.items() if v},
                      outputs=[str(self.storage / name) for name in manifest["files"]],
                      verified_sizes={str(self.storage / name): info["size"] for name, info in manifest["files"].items()})
        atomic_json(self.reports / "shards.json", result)
        return result

    def chat_data(self):
        command = self.command("prepare_chat.py", "--storage", self.storage)
        if self.a.rehearsal:
            command += ["--limit", "2000"]
        self.execute(command)
        return dict(outputs=[str(self.storage / "chat/.done"), str(self.reports / "chat_report.md")])

    def train(self, name, *extra):
        # Pretraining uses the fastest speed-only settings found by scripts/speed_sweep.py,
        # if it was run on this storage. Later flags win in argparse, so this overrides.
        sweep = self.storage / "reports/speed-sweep.json"
        if not self.a.rehearsal and "chat" not in extra and sweep.exists():
            best = read_json(sweep).get("best")
            if best:
                extra = (*extra, "--micro-tokens", best["micro_tokens"], "--compile-mode", best["compile_mode"])
        return self.command("train.py", "--model", "tiny" if self.a.rehearsal else "chotu",
            "--storage", self.storage, "--name", name, "--device", self.a.device,
            "--no-auto-stop", *extra)

    def speed(self):
        steps, batch = (4, 8192) if self.a.rehearsal else (50, 524288)
        self.execute(self.train("speed", "--stage", "pretrain", "--lr", ".003", "--total-tokens", steps * batch,
            "--batch-tokens", batch, "--micro-tokens", 8192 if self.a.rehearsal else 65536,
            "--long-frac", "0", "--warmup-steps", "1" if self.a.rehearsal else "5",
            "--ckpt-every", "5", "--eval-every", str(steps),
            "--val-tokens", "2048" if self.a.rehearsal else "1048576", *([] if self.a.rehearsal else ["--kill-rules"])), training=True)
        rows = metrics(self.storage / "runs/speed/metrics.jsonl")
        measured = [r for r in rows if 20 <= r["step"] <= 50] if not self.a.rehearsal else rows[1:]
        report = dict(tokens_per_second=statistics.mean(r["tokens_per_second"] for r in measured),
                      mfu=statistics.mean(r["mfu"] for r in measured), measured_steps=[r["step"] for r in measured])
        atomic_json(self.reports / "speed.json", report)
        return report

    def lr_test(self):
        trials = []
        for lr in (.0015, .003):
            name = f"lr-{lr}"
            command = self.train(name, "--stage", "pretrain", "--lr", lr,
                "--total-tokens", "65536" if self.a.rehearsal else "5e8", "--warmup-steps", "1" if self.a.rehearsal else "100",
                "--decay-frac", ".15", "--long-frac", "0", "--ckpt-every", "5" if self.a.rehearsal else "100",
                "--eval-every", "8" if self.a.rehearsal else "100")
            if self.a.rehearsal:
                command += ["--batch-tokens", "8192", "--micro-tokens", "8192", "--val-tokens", "2048"]
            self.execute(command, training=True)
            rows = metrics(self.storage / "runs" / name / "metrics.jsonl")
            trials.append(dict(lr=lr, final_val_bpb=rows[-1]["val_bpb"]))
        picked = min(trials, key=lambda row: row["final_val_bpb"])
        result = dict(lr=picked["lr"], trials=trials, criterion="lower final held-out BPB; ties choose 1.5e-3")
        atomic_json(self.reports / "lr-choice.json", result)
        return result

    def pretrain(self):
        lr = read_json(self.reports / "lr-choice.json")["lr"]
        command = self.train("pretrain", "--stage", "pretrain", "--lr", lr)
        if self.a.rehearsal:
            command += ["--total-tokens", str(self.a.rehearsal_tokens), "--batch-tokens", "8192", "--micro-tokens", "8192",
                        "--long-frac", ".25", "--warmup-steps", "3", "--ckpt-every", "5", "--eval-every", "20", "--val-tokens", "2048"]
        else:
            command += ["--kill-rules"]
        self.execute(command, training=True)
        path = self.storage / "runs/pretrain/final"
        return dict(outputs=[str(path / "config.json"), str(path / "model.safetensors")],
                    **read_json(self.storage / "runs/pretrain/done.json"))

    def eval_lm(self, model, name):
        command = self.command("eval_lm.py", "--model", model, "--out", self.reports / f"{name}-lm.json",
            "--cache-dir", self.cache / "eval-results", "--bpb", self.storage,
            "--device", self.a.device, "--no-auto-stop")
        if self.a.rehearsal:
            command += ["--limit", "2", "--tasks", "hellaswag", "arc_easy", "--val-tokens", "2048"]
        self.execute(command)

    def base_evals(self):
        self.eval_lm(self.storage / "runs/pretrain/final", "base")
        return dict(outputs=[str(self.reports / "base-lm.json")])

    def chat_tune(self):
        command = self.train("chat", "--stage", "chat", "--init", self.storage / "runs/pretrain/final")
        if self.a.rehearsal:
            command += ["--max-steps", "8", "--batch-tokens", "8192", "--micro-tokens", "8192",
                        "--warmup-steps", "2", "--eval-every", "4", "--ckpt-every", "2"]
        self.execute(command, training=True)
        path = self.storage / "runs/chat/best"
        return dict(outputs=[str(path / "config.json"), str(path / "model.safetensors")],
                    **read_json(self.storage / "runs/chat/done.json"))

    def final_evals(self):
        self.eval_lm(self.storage / "runs/chat/best", "final")
        command = self.command("eval_tools.py", "--model", self.storage / "runs/chat/best",
            "--apigen", self.storage / "chat/apigen-heldout.jsonl", "--out", self.reports / "final-tools.json",
            "--cache-dir", self.cache / "bfcl", "--device", self.a.device, "--no-auto-stop")
        if self.a.rehearsal:
            command += ["--limit", "2", "--max-new-tokens", "32"]
        self.execute(command)
        return dict(outputs=[str(self.reports / "final-lm.json"), str(self.reports / "final-tools.json")])

    def export_gguf(self):
        self.execute(self.command("export_gguf.py", "--model", self.storage / "runs/chat/best",
            "--out", self.storage / "export", "--cache-dir", self.cache, "--jobs", self.a.build_jobs))
        return dict(outputs=[str(self.storage / "export/chotu-q8_0.gguf"), str(self.storage / "export/export-report.json")])

    def upload(self):
        if self.a.skip_upload:
            return dict(skipped=True, reason="--skip-upload: no repo created or uploaded")
        if not self.repo:
            from huggingface_hub import HfApi
            self.repo = HfApi().whoami()["name"] + "/chotu-v1"
        if self.a.rehearsal and not self.repo.endswith("-rehearsal"):
            self.repo += "-rehearsal"
        # Publish the report before upload too; after success its local copy gains
        # the confirmed upload stage and final timing information.
        self.write_report("upload pending")
        self.execute(self.command("upload.py", "--repo", self.repo, "--paths", self.storage / "runs/chat/best",
            self.storage / "export/chotu-q8_0.gguf", self.storage / "export/Modelfile", self.reports))
        return dict(repo=self.repo, private=True)

    def write_report(self, status):
        lines = ["# Chotu chain report", "", f"Mode: {'REHEARSAL (not a quality gate)' if self.a.rehearsal else 'full run'}",
                 f"Status: {status}", f"Storage: `{self.storage}`", "", "## Stages", ""]
        for stage in STAGES:
            path = self.markers / f"{stage}.done.json"
            if path.exists():
                data = read_json(path)
                lines.append(f"- {stage}: done ({data['seconds']:.1f}s)" + (" [explicit skip]" if data["result"].get("skipped") else ""))
            elif stage == "final_report" and status == "complete":
                lines.append("- final_report: complete")
            else:
                lines.append(f"- {stage}: not marked done")
        for title, path in [("Speed / MFU", self.reports / "speed.json"), ("Learning-rate choice", self.reports / "lr-choice.json"),
                            ("Shards", self.reports / "shards.json")]:
            if path.exists():
                data = read_json(path)
                data = {k: v for k, v in data.items() if k not in {"outputs", "verified_sizes"}}
                lines += ["", f"## {title}", "```json", json.dumps(data, indent=2), "```"]
        lines += ["", "## Training", ""]
        for name in ("pretrain", "chat"):
            path = self.storage / "runs" / name
            if (path / "metrics.jsonl").exists():
                rows = metrics(path / "metrics.jsonl")
                summary = dict(steps=len(rows), first_loss=rows[0]["loss"], final_loss=rows[-1]["loss"],
                               theta=rows[-1]["theta"], max_sequence_length=max(row["seq_len"] for row in rows))
                if (path / "done.json").exists():
                    summary.update(read_json(path / "done.json"))
                lines += [f"### {name}", "```json", json.dumps(summary, indent=2), "```"]
        for name in ("base", "final"):
            path = self.reports / f"{name}-lm.md"
            if path.exists():
                lines += ["", f"## {name.title()} LM evaluation", path.read_text()]
        for title, path in [("Tool evaluation", self.reports / "final-tools.json"),
                            ("GGUF", self.storage / "export/export-report.json")]:
            if path.exists():
                data = read_json(path)
                if title == "Tool evaluation":
                    data = dict(gate=data["gate"], categories={k: {a: b for a, b in v.items() if a != "failures"} for k, v in data["results"].items()})
                lines += ["", f"## {title}", "```json", json.dumps(data, indent=2), "```"]
        upload_marker = self.markers / "upload.done.json"
        lines += ["", "## Upload / shutdown", json.dumps(read_json(upload_marker)["result"]) if upload_marker.exists() else status,
                  "Shutdown owner: orchestrator; pause is attempted after this report is written." if not self.a.no_stop else "--no-stop: local run; no machine pause attempted.",
                  "", "Reports are plumbing-only under rehearsal. Gate failures are reported, not fatal.",
                  "See logs/chain.jsonl and per-stage logs. Paid-resource pause result is recorded in logs/status after finalization."]
        (self.reports / "final_report.md").write_text("\n".join(lines) + "\n")

    def final_report(self):
        self.write_report("complete")
        if not self.a.skip_upload:
            repo = read_json(self.markers / "upload.done.json")["result"]["repo"]
            self.execute(self.command("upload.py", "--repo", repo, "--paths", self.reports))
        return dict(outputs=[str(self.reports / "final_report.md")])

    def run(self):
        handlers = {s: signal.signal(s, self.on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            self.setup()
            for stage in selected_stages(self.a):
                self.stage = stage
                marker = self.markers / f"{stage}.done.json"
                if marker.exists():
                    result = read_json(marker)["result"]
                    if any(not Path(p).exists() for p in result.get("outputs", [])):
                        raise ValueError(f"{stage} marked done but outputs missing; restore artifacts before resuming")
                    if any(Path(p).stat().st_size != size for p, size in result.get("verified_sizes", {}).items()):
                        raise ValueError(f"{stage} artifact sizes changed; restore verified data before resuming")
                    self.emit("skip_done")
                    continue
                self.emit("start")
                clock = time.monotonic()
                result = getattr(self, stage)() or {}
                elapsed = time.monotonic() - clock
                atomic_json(marker, dict(stage=stage, seconds=elapsed, result=result))
                self.emit("done", seconds=round(elapsed, 2))
        except ChainSignal as error:
            code = 128 + error.number
            self.alert(f"signal {error.number}; interrupted stage can resume")
        except StageError as error:
            code = error.code
            self.alert(str(error) + ("; kill-rule exits are never retried" if 20 <= code <= 23 else ""))
        except Exception as error:
            code = 1
            self.alert(f"{type(error).__name__}: {error}")
        finally:
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                self.terminate_child()
            finally:
                # Ensure a setup exception still has a durable pause log.
                try:
                    self.logs.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
                if not self.stop():
                    code = 25
                if self.wandb:
                    try:
                        self.wandb.finish()
                    except Exception:
                        pass
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        self.emit("finished", exit_code=code)
        return code


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--storage", default=os.getenv("CHOTU_STORAGE", "~/chotu-data"))
    ap.add_argument("--cache-dir")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--rehearsal-tokens", type=int, default=524288)
    ap.add_argument("--shards-subset", action="store_true", help="rehearsal: train_013 plus all validation shards")
    ap.add_argument("--hf-repo")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--no-stop", action="store_true", help="local/external shutdown owner only")
    ap.add_argument("--retries", type=int, default=2, help="extra attempts for training crashes only")
    ap.add_argument("--build-jobs", type=int, default=2)
    ap.add_argument("--from-stage", choices=STAGES)
    ap.add_argument("--only-stage", choices=STAGES)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main():
    a = parser().parse_args()
    if a.dry_run:
        dry_plan(a)
        return 0
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return Chain(a).run()


if __name__ == "__main__":
    sys.exit(main())
