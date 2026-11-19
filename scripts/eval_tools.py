"""Greedy tool evaluation; BFCL scoring is a re-implementation, not its official harness."""
import argparse
from itertools import islice
import json
from pathlib import Path
import sys

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chotu.chat_format import render_conversation
from chotu.eval_runtime import shutdown_guard
from chotu.tool_eval import BFCL_COMMIT, BFCL_FILES, apigen_score, bfcl_score, bfcl_tools


def jsonl(path):
    with Path(path).open() as file:
        return [json.loads(line) for line in file if line.strip()]


def load_bfcl(cache, category):
    root = Path(cache) / BFCL_COMMIT
    root.mkdir(parents=True, exist_ok=True)
    filename = f"BFCL_v4_{BFCL_FILES[category]}.json"
    paths = []
    for prefix in ("", "possible_answer/"):
        path = root / ("answers-" if prefix else "questions-") / filename
        if not path.exists():
            url = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_COMMIT}/berkeley-function-call-leaderboard/bfcl_eval/data/{prefix}{filename}"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(response.text)
            temporary.replace(path)
        paths.append(path)
    answers = {row["id"]: row["ground_truth"] for row in jsonl(paths[1])}
    return [(row, answers[row["id"]]) for row in jsonl(paths[0])]


@torch.inference_mode()
def generate(model, tokenizer, messages, tools, max_new_tokens):
    prompt = render_conversation(messages, tools) + "<|im_start|>assistant\n"
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    if encoded.input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        return "", "over_context"
    stop = tokenizer.convert_tokens_to_ids("<|im_end|>")
    output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                            eos_token_id=stop, pad_token_id=tokenizer.eos_token_id,
                            use_cache=True)
    ids = output[0, encoded.input_ids.shape[1]:].tolist()
    if stop in ids:
        ids = ids[:ids.index(stop)]
    return tokenizer.decode(ids, skip_special_tokens=False), None


def summarize(rows):
    keys = [k for k in ("accuracy", "json_valid", "exact_match", "name_accuracy") if rows and k in rows[0]["score"]]
    result = {"count": len(rows), "failures": [r for r in rows if not r["score"].get("accuracy", r["score"].get("exact_match"))][:5]}
    for key in keys:
        values = [r["score"][key] for r in rows if r["score"][key] is not None]
        result[key] = sum(values) / len(values) if values else None
        if key == "json_valid":
            result["json_count"] = len(values)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--apigen", help="storage/chat/apigen-heldout.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-dir")
    ap.add_argument("--categories", nargs="+", choices=BFCL_FILES, default=list(BFCL_FILES))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--no-auto-stop", action="store_true")
    a = ap.parse_args()
    if (a.limit is not None and a.limit < 1) or a.max_new_tokens < 1:
        ap.error("limits must be positive")
    location = Path(a.model).resolve()
    if not location.is_dir():
        ap.error("--model must be a local HF directory")
    out = Path(a.out).resolve() if a.out else location.parent / "tool-eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    apigen = Path(a.apigen) if a.apigen else next((p / "chat/apigen-heldout.jsonl" for p in location.parents if (p / "chat/apigen-heldout.jsonl").exists()), None)
    if apigen is None or not apigen.is_file():
        ap.error("provide --apigen held-out JSONL")
    with shutdown_guard(a.device, a.no_auto_stop):
        model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="sdpa").to(a.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(a.model)
        result = {"model": str(location), "bfcl_commit": BFCL_COMMIT, "limit": a.limit,
                  "checker": "reimplementation, Python simple/multiple/parallel; not official harness", "results": {}}
        records = {}
        for category in ["apigen", *a.categories]:
            rows = []
            data = jsonl(apigen) if category == "apigen" else load_bfcl(a.cache_dir or out.parent / "bfcl-cache", category)
            for item in islice(data, a.limit):
                if category == "apigen":
                    row, gold = item, item["gold_calls"]
                    messages, tools = row["prompt_messages"], row["tools"]
                else:
                    row, gold = item
                    if len(row["question"]) != 1:
                        raise ValueError("multi-turn BFCL is not supported")
                    messages, tools = row["question"][0], bfcl_tools(row["function"])
                text, error = generate(model, tokenizer, messages, tools, a.max_new_tokens)
                score = apigen_score(text, gold) if category == "apigen" else bfcl_score(text, gold, row["function"])
                rows.append(dict(id=row["id"], output=text, error=error, gold=gold, score=score))
                print(json.dumps(dict(category=category, index=len(rows), score=score)), flush=True)
            if not rows:
                raise ValueError(f"empty evaluation category: {category}")
            result["results"][category] = summarize(rows)
            records[category] = rows
        simple = result["results"].get("simple", {})
        evaluated = list(result["results"].values())
        json_count = sum(v["json_count"] for v in evaluated)
        valid = sum((v["json_valid"] or 0) * v["json_count"] for v in evaluated) / json_count if json_count else None
        result["gate"] = dict(simple_target=.5, json_target=.95, overall_json_valid=valid,
                              passed=simple.get("accuracy", -1) >= .5 and all(v["json_valid"] >= .95 for v in evaluated if v["json_count"]),
                              provisional=a.limit is not None or set(a.categories) != set(BFCL_FILES), missing_simple=not bool(simple))
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        out.with_suffix(".samples.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
