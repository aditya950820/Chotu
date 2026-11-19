"""Pinned llama.cpp Q8_0 conversion and real CPU load/generation sanity check."""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

LLAMA_COMMIT = "9a080f44b503c64f66c84a6274b9d9ee5d5b09d5"


def modelfile(filename="chotu-q8_0.gguf"):
    return '''FROM ./''' + filename + '''
TEMPLATE """{{ if or .System .Tools }}<|im_start|>system
{{ .System }}{{ if .Tools }}
Available tools (JSON): <tools>{{ .Tools }}</tools>
Use <tool_call>{"name": "...", "arguments": {...}}</tool_call> for tool calls.{{ end }}<|im_end|>
{{ end }}{{ range .Messages }}{{ if eq .Role "tool" }}<|im_start|>tool
<tool_response>{{ .Content }}</tool_response><|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}{{ range .ToolCalls }}<tool_call>{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}</tool_call>{{ end }}<|im_end|>
{{ else if or (eq .Role "user") (and (eq .Role "system") (not $.System)) }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 8192
'''


def run(args, **kwargs):
    print("+ " + " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def llama_source(cache):
    source = cache / "llama.cpp"
    if not source.exists():
        source.mkdir()
        run(["git", "init", source])
        run(["git", "-C", source, "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp.git"])
        run(["git", "-C", source, "fetch", "--depth", "1", "origin", LLAMA_COMMIT])
        run(["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"])
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if head != LLAMA_COMMIT or dirty:
        raise ValueError(f"llama.cpp must be clean at {LLAMA_COMMIT}; use a fresh cache")
    return source


def check_metadata(path, source, config):
    sys.path.insert(0, str(source / "gguf-py"))
    gguf = importlib.import_module("gguf")
    reader = gguf.GGUFReader(str(path))
    fields = reader.fields
    def value(name):
        field = fields[name]
        return field.contents()
    arch = value("general.architecture")
    tokens = value("tokenizer.ggml.tokens")
    metadata = {"architecture": arch, "context_length": value(f"{arch}.context_length"),
                "tokenizer_model": value("tokenizer.ggml.model"), "vocab_size": len(tokens),
                "file_type": value("general.file_type"),
                "eos_token_id": value("tokenizer.ggml.eos_token_id")}
    if metadata["context_length"] != 8192 or config["max_position_embeddings"] != 8192:
        raise ValueError("GGUF context length must be 8192")
    if metadata["vocab_size"] != config["vocab_size"] or not metadata["tokenizer_model"]:
        raise ValueError("GGUF tokenizer metadata differs from HF config")
    if int(metadata["file_type"]) != int(gguf.LlamaFileType.MOSTLY_Q8_0):
        raise ValueError("expected Q8_0 GGUF")
    if "<|im_start|>" not in tokens or "<|im_end|>" not in tokens:
        raise ValueError("missing ChatML tokenizer markers")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["_export_model_path"])
    for marker in ("<|im_start|>", "<|im_end|>"):
        if tokens[tokenizer.convert_tokens_to_ids(marker)] != marker:
            raise ValueError("ChatML token IDs changed in GGUF")
    if metadata["eos_token_id"] != tokenizer.eos_token_id:
        raise ValueError("EOS token ID changed in GGUF")
    metadata["rope_theta"] = value(f"{arch}.rope.freq_base")
    if metadata["rope_theta"] != config.get("rope_theta", config.get("rope_parameters", {}).get("rope_theta")):
        raise ValueError("RoPE theta changed in GGUF")
    return metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", help="persistent build cache; default OUT/cache")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    if a.jobs < 1:
        ap.error("jobs must be positive")
    model, out = Path(a.model).resolve(), Path(a.out).resolve()
    if not model.is_dir():
        ap.error("model must be a local HF directory")
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir).resolve() if a.cache_dir else out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    source = llama_source(cache)
    config = json.loads((model / "config.json").read_text())
    config["_export_model_path"] = str(model)
    target = out / "chotu-q8_0.gguf"
    temporary = out / "chotu-q8_0.tmp.gguf"
    env = dict(os.environ, PYTHONPATH=str(source / "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run([sys.executable, source / "convert_hf_to_gguf.py", model, "--outfile", temporary, "--outtype", "q8_0"], env=env)
    metadata = check_metadata(temporary, source, config)
    temporary.replace(target)
    (out / "Modelfile").write_text(modelfile(target.name))
    build = source / "build"
    run(["cmake", "-S", source, "-B", build, "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF",
         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_APP=OFF", "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", build, "--target", "llama-simple", "-j", a.jobs])
    result = run([build / "bin/llama-simple", "-m", target, "-n", "8", "-ngl", "0",
                  "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"],
                 capture_output=True, text=True, timeout=180)
    (out / "llama-generation.log").write_text(result.stdout + "\n" + result.stderr)
    decoded = re.search(r"decoded (\d+) tokens", result.stderr)
    if not decoded or int(decoded.group(1)) < 1:
        raise ValueError("llama.cpp did not decode any generation tokens")
    report = {"llama_cpp_commit": LLAMA_COMMIT, "model": str(model), "gguf": str(target),
              "metadata": metadata, "generation_stdout": result.stdout, "generated_tokens": int(decoded.group(1)),
              "generation_exit": result.returncode}
    (out / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
