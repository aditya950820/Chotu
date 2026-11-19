# Evaluation, export and private upload (chain step 3)

These scripts do not establish quality from a tiny smoke. Full evaluations
are required before deciding whether the model clears the proposed gates.
All paths below are on persistent storage. Nothing is uploaded automatically.

## Language-model evaluation

```sh
python scripts/eval_lm.py --model /persistent/chotu/runs/chat/best \
  --out /persistent/chotu/reports/chat-lm.json \
  --bpb /persistent/chotu
```

Pinned `lm_eval[hf]==0.4.13`: zero-shot HellaSwag, ARC-Easy, ARC-Challenge,
PIQA and WinoGrande. Raw prompts (no model-specific chat templates), seed
1234, fp32, batch size 1, context capped at **1,024 for every model**, matching
GPT-2's capacity. Output includes acc and acc_norm where the task defines
them; WinoGrande has acc only, so acc_norm is N/A, not fabricated.

Also evaluates `gpt2` and `HuggingFaceTB/SmolLM2-135M`. Results are cached
under `OUT_PARENT/eval-cache` by immutable hub revision (or local file
manifest), settings and harness version. Changing tasks/limit invalidates
the cache, so a smoke cannot satisfy a full evaluation. JSON contains task
versions/configs; a sibling Markdown file contains the comparison table.
Success means **strictly greater acc_norm than GPT-2 on both HellaSwag and
ARC-Easy**. Limited runs are labelled smoke-only, never a quality pass.
`--skip-baselines` is available for isolated rehearsal; it cannot establish
the success bar. `--tasks hellaswag arc_easy --limit 2` is a small smoke.

`--bpb STORAGE` reuses the trainer's exact decoded-target UTF-8 byte metric
on the first fixed validation targets (default 1,048,576), independent of
the harness task limit. Use `--val-tokens 2048` for smoke. Run the same command
with `--model .../pretrain/final` and the same shards/token budget to compare
base and chat BPB. BPB is for the requested target, not rerun for both baselines.

CPU is the default. Standalone `--device cuda` pauses Jarvis in `finally`;
the later orchestrator must pass `--no-auto-stop` and own shutdown itself.

## Tool evaluation

```sh
python scripts/eval_tools.py --model /persistent/chotu/runs/chat/best \
  --apigen /persistent/chotu/chat/apigen-heldout.jsonl \
  --out /persistent/chotu/reports/chat-tools.json
```

Greedy generation, at most 256 new tokens, our literal ChatML/tools format,
stop at `<|im_end|>`. No Python/tool execution. Over-context inputs are scored
as failures, not silently truncated or dropped. Optional `--limit N` applies
separately to APIGen and each selected BFCL category. `--categories simple`
supports the small smoke; default is all three categories.

- APIGen: unordered **multiset** exact match on names + arguments (duplicate
  calls still count), and exact multiset name accuracy. Gold refusals can
  correctly predict no calls; their JSON metric is N/A and excluded from the
  JSON denominator. Empty model output is not a correct refusal.
- BFCL: official Gorilla repository, pinned commit
  `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, under
  `berkeley-function-call-leaderboard/bfcl_eval/data/`. Use
  `BFCL_v4_simple_python.json` (400), `BFCL_v4_multiple.json` (200),
  `BFCL_v4_parallel.json` (200), joined by ID with their `possible_answer/`
  files. Downloads are cached by commit. Function docs are converted from
  Python-style dict/float/tuple schemas to JSON object/number/array schemas.
- Checker is an explicit **re-implementation, not the official BFCL harness**:
  our JSON tags instead of Python AST parsing; allowed values per argument,
  `""` allowing omission only when not schema-required, strict types (bool is
  not int; int may promote to float), BFCL string normalization, nested
  dictionaries and one-to-one unordered parallel matching. Symbolic values
  explicitly listed by BFCL can match as variables. No Java/JavaScript,
  execution, live, multi-turn or agentic categories are claimed.
- JSON validity means nonempty well-formed calls, finite JSON values, and
  no surrounding prose. This is deliberately stricter than merely finding
  one parseable tag in otherwise malformed output.

Report per-category accuracy/validity with denominators, five failures per
category, and all generated outputs in a sibling `.samples.json`. Gate:
simple accuracy >=50% and JSON validity >=95% for each measured category
with JSON targets. Limited/subset runs are provisional. A failing gate is
reported, not an exception or nonzero exit.

## GGUF and Ollama

```sh
python scripts/export_gguf.py --model /persistent/chotu/runs/chat/best \
  --out /persistent/chotu/export --cache-dir /persistent/chotu/cache
```

Clone/build llama.cpp at pinned commit
`9a080f44b503c64f66c84a6274b9d9ee5d5b09d5`. Require an existing checkout
to be clean and at exactly that commit; do not overwrite other checkouts.
Use the checkout's bundled `gguf-py`, not an unpinned installed gguf module.
`sentencepiece==0.2.3` and `cmake==4.4.5` are pinned in our requirements;
a C++ compiler and git are also required. Do not install llama.cpp's entire
converter requirements over the project's frozen torch/transformers pins.

Convert directly to Q8_0, check file type, 8,192 context, vocabulary, unchanged
ChatML token IDs/EOS and RoPE theta. Write `Modelfile` with explicit system,
tool definitions, assistant calls and wrapped tool responses, stop on
`<|im_end|>`, `num_ctx 8192`. Build `llama-simple` on CPU, require actual
decoded generation tokens, and save metadata and generation logs. This pinned
revision puts `llama-cli` behind the server option; the smaller documented
`llama-simple` equivalent avoids that build. Generation is a loadability
check, not a numerical parity or quality claim. Ollama itself is not invoked.

## Private upload (explicit, network-mutating)

```sh
python scripts/upload.py --repo OWNER/PRIVATE_MODEL --paths \
  /persistent/chotu/runs/chat/best \
  /persistent/chotu/export/chotu-q8_0.gguf \
  /persistent/chotu/export/Modelfile /persistent/chotu/reports
```

**Do not run in tests/smoke.** Needs a write-capable token and explicit human
authorization. Validate all local paths first. Refuse existing public or
unknown-privacy repos. If missing, request `private=True` and recheck privacy
before a single commit of all artifacts. The HF model directory goes at repo
root (ready for `from_pretrained`); reports retain their directory. Supply
individual export files, not the build/cache directory. Hidden/credential
files and destination collisions are rejected. A model-card stub is added
unless a root README is supplied; complete licenses/limitations/results before
considering public release. No attempt is made to change repository visibility.

Tests mock **all** HfApi mutations. Read-token access does not establish upload
permission, and no real upload or repository creation was attempted here.
