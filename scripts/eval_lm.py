"""Comparable zero-shot lm-eval scores, cached baselines and optional fixed-set BPB."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chotu.eval_runtime import shutdown_guard

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]
BASELINES = ["gpt2", "HuggingFaceTB/SmolLM2-135M"]


def model_identity(model):
    path = Path(model)
    if path.is_dir():
        files = [(str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(path.rglob("*")) if p.is_file()]
        return {"path": str(path.resolve()), "manifest": files}
    return {"hub": model, "revision": HfApi().model_info(model).sha}


def evaluate_one(model, settings, cache):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    identity = model_identity(model)
    signature = {"identity": identity, "settings": settings,
                 "lm_eval": importlib.metadata.version("lm_eval")}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = cache / f"{key}.json"
    if path.exists():
        print(f"CACHED {model}: {path}", flush=True)
        return json.loads(path.read_text())
    lm = HFLM(pretrained=model, revision=identity.get("revision", "main"),
              device=settings["device"], dtype="float32", batch_size=settings["batch_size"],
              max_length=1024, trust_remote_code=False)
    result = simple_evaluate(model=lm, tasks=settings["tasks"], num_fewshot=0,
        limit=settings["limit"], apply_chat_template=False, log_samples=False,
        bootstrap_iters=0, random_seed=1234, numpy_random_seed=1234,
        torch_random_seed=1234, fewshot_random_seed=1234)
    report = dict(model=model, signature=signature, results=result["results"],
                  task_versions=result.get("versions"), configs=result.get("configs"))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str))
    temporary.replace(path)
    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def comparison(reports, tasks, limited):
    rows = ["# Zero-shot comparison", "", "Identical raw-prompt settings: fp32, 1,024 context, no chat template; seed 1234.",
            "Scores are acc / acc_norm. N/A means the task does not define that metric.", "",
            "| Model | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for report in reports:
        scores = []
        for task in tasks:
            values = report["results"][task]
            scores.append(" / ".join(f"{values[k]:.4f}" if k in values else "N/A" for k in ("acc,none", "acc_norm,none")))
        rows.append("| " + report["model"] + " | " + " | ".join(scores) + " |")
    target = reports[0]
    gpt2 = next((r for r in reports if r["model"] == "gpt2"), None)
    needed = ("hellaswag", "arc_easy")
    complete = gpt2 is not None and all(t in target["results"] and t in gpt2["results"] for t in needed)
    beats = complete and all(target["results"][t]["acc_norm,none"] > gpt2["results"][t]["acc_norm,none"] for t in needed)
    verdict = {"metric": "acc_norm", "tasks": list(needed), "beats_gpt2": bool(beats) if complete else None,
               "status": "smoke_only" if limited else ("pass" if beats else "fail") if complete else "incomplete"}
    rows += ["", "Success bar (strictly greater GPT-2 acc_norm on both HellaSwag and ARC-Easy):", json.dumps(verdict)]
    return "\n".join(rows) + "\n", verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--cache-dir")
    ap.add_argument("--skip-baselines", action="store_true", help="rehearsal only; no success verdict")
    ap.add_argument("--bpb", metavar="STORAGE", help="fixed validation shards; same train.py metric")
    ap.add_argument("--val-tokens", type=int, default=1_048_576)
    ap.add_argument("--no-auto-stop", action="store_true")
    a = ap.parse_args()
    if (a.limit is not None and a.limit < 1) or a.batch_size < 1 or a.val_tokens < 1:
        ap.error("limits and batch size must be positive")
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir) if a.cache_dir else out.parent / "eval-cache"
    cache.mkdir(parents=True, exist_ok=True)
    settings = dict(tasks=a.tasks, limit=a.limit, device=a.device, batch_size=a.batch_size,
                    num_fewshot=0, max_length=1024, dtype="float32", apply_chat_template=False)
    with shutdown_guard(a.device, a.no_auto_stop):
        models = list(dict.fromkeys([a.model, *([] if a.skip_baselines else BASELINES)]))
        reports = [evaluate_one(model, settings, cache) for model in models]
        table, verdict = comparison(reports, a.tasks, a.limit is not None)
        result = dict(settings=settings, reports=reports, gate=verdict)
        if a.bpb:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from chotu.data import ShardStream
            from train import validate
            model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="sdpa").to(a.device)
            tokenizer = AutoTokenizer.from_pretrained(a.model)
            result["val_bpb"] = validate(model, ShardStream(a.bpb, "val"), tokenizer,
                                         torch.device(a.device), a.val_tokens)
            table += f"\nTarget fixed-set validation BPB: {result['val_bpb']:.8f} ({a.val_tokens} target-token budget).\n"
        out.write_text(json.dumps(result, indent=2, default=str) + "\n")
        out.with_suffix(".md").write_text(table)
        print(table)


if __name__ == "__main__":
    main()
