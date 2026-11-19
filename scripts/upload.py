"""Publish explicitly selected artifacts to a private HF model repo. Never run in smoke."""
import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RepositoryNotFoundError


CARD = """---
language: en
tags:
- chotu
- text-generation
---
# Chotu (evaluation pending)

From-scratch English Qwen3-layout learning project. See the accompanying
evaluation reports for the exact model/configuration and measured scores.
This is not a production model. Tool calls are untrusted suggestions: validate
schemas and arguments, and require authorization before any execution.

Training data: FineWeb-Edu sample-10BT; chat: smol-smoltalk and smoltalk APIGen.
Tokenizer: HuggingFaceTB/SmolLM2-135M, unchanged. Proposed context: 8,192.
GGUF Q8_0: use the accompanying Ollama Modelfile (ChatML, num_ctx 8192).

TODO: fill in training checkpoint, compute, licenses, intended use, limitations,
and full evaluation results before considering any public release.
"""


def upload(repo, paths, api=None):
    api = api or HfApi()
    artifacts = [Path(p).resolve() for p in paths]
    if not artifacts or any(not p.exists() for p in artifacts):
        raise ValueError("provide existing artifact paths")
    # Collect everything before any remote mutation; no implicit directory crawl.
    files = {}
    for path in artifacts:
        if path.is_dir():
            is_model = (path / "config.json").is_file()
            prefix = "" if is_model else path.name + "/"
            entries = [(prefix + p.relative_to(path).as_posix(), p) for p in sorted(path.rglob("*")) if p.is_file()]
        else:
            entries = [(path.name, path)]
        for name, file in entries:
            if any(part.startswith(".") for part in Path(name).parts) or file.suffix in {".env", ".pem", ".key"}:
                raise ValueError(f"refusing hidden/credential artifact: {name}")
            if name in files:
                raise ValueError(f"duplicate destination: {name}")
            files[name] = file
    if not files:
        raise ValueError("no artifact files")
    try:
        info = api.repo_info(repo_id=repo, repo_type="model")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        info = api.repo_info(repo_id=repo, repo_type="model")
    if info.private is not True:
        raise ValueError("refusing upload: HF repository must be private")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(file)) for name, file in files.items()]
    if "README.md" not in files:
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=CARD.encode()))
    return api.create_commit(repo_id=repo, repo_type="model", operations=operations,
                             commit_message="Upload Chotu model and evaluation artifacts")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--paths", nargs="+", required=True)
    a = ap.parse_args()
    print(upload(a.repo, a.paths))


if __name__ == "__main__":
    main()
