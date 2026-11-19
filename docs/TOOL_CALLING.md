# 8k context + tool calling — proposal and report

Proposed 2026-11-19 by Adi; needs partner's agreement. No training or capability claim yet.

## Format

Keep SmolLM2's tokenizer unchanged. Tags below are ordinary text; only the existing ChatML markers are special tokens. Tools are a JSON list in the first system turn, after a short instruction. Messages use `role`, text `content`, and optional assistant `tool_calls`: a list of `{name, arguments}` objects. Multiple calls in one assistant turn mean parallel requests. Tool results follow in call order, one tool turn per result; callers must preserve this order.

```text
<|im_start|>system
Call tools with <tool_call>{"name": ..., "arguments": {...}}</tool_call> when needed.
<tools>
[{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]
</tools><|im_end|>
<|im_start|>user
Weather in Paris?<|im_end|>
<|im_start|>assistant
<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call><|im_end|>
<|im_start|>tool
<tool_response>{"city":"Paris","temperature":12}</tool_response><|im_end|>
<|im_start|>assistant
It is 12°C in Paris.<|im_end|>
```

Every turn ends in a newline. `chotu/chat_format.py` renders text, parses calls into `(calls, errors)`, and tokenizes with assistant-body-only labels including `<|im_end|>`. Headers, trailing newlines, system/user/tool turns get -100. Over-8,192 examples return `None`, never truncate. Parsing checks JSON shape, not schema compliance; no tool execution is implemented. A future runner must validate against allowed schemas and obtain permission before side effects. Tool output is untrusted data, not instructions.

## Data and long context

Start rehearsal with a proposed 80/20 conversation mix: smol-smoltalk / function-calling conversations (smoltalk `apigen-80k` or Salesforce xLAM-60k). Check licenses, actual schemas, parallel coverage and contamination before selecting a source. Normalize into this format, deduplicate and hold out conversations before mixing. Report both conversation and assistant-token shares; tune the mix on held-out chat and tool scores, not training loss alone.

Pretrain at 2,048 with theta 10,000 (`PRETRAINING_WINDOW`); reserve the final ~5–10% of the same token budget for 8,192 with theta 500,000 (`long_context_config()`). The base config declares 8,192 capacity but keeps stage-1 theta; a future training loader must explicitly select 2,048 windows. Select long FineWeb-Edu documents without replaying held-out documents or already consumed tokens. Existing packed shards alone do not preserve the document selection needed here: plan document-aware preparation before training. EDA: 9.4% of documents exceed 2,048 and contain 41.5% of tokens; this does not establish the supply above 8,192. Measure that supply first.

Full 8k pretraining is estimated ~2–3x slower; measure throughput in rehearsal. Parameters remain 180,925,056. Batch-1 bf16/fp16 KV cache at 8k is ~335 MB (32 layers, 5 KV heads, 64 head dimension, K+V), in addition to ~190 MB Q8_0 weights and runtime buffers. GGUF / Ollama needs `num_ctx 8192`; test recall across positions before claiming usable 8k context.

## Evals and proposed bar

Keep existing lm-evaluation-harness and bits-per-byte gates. Separately pin a BFCL version and run its simple, multiple and parallel subsets with supplied schemas; report per-subset accuracy and JSON-validity rate. Proposed initial gate: BFCL simple accuracy ≥50% and JSON validity ≥95% on held-out calls. These are targets, not measured results; review them after the tiny rehearsal. Report parallel/multiple scores even if poor. Aim for strong single and parallel calls against given schemas, not reliable long multi-step agents. Check chat regressions and short/long-context validation separately.

## Sessions 3–5 and report

- Session 3: rehearse the 2k→8k stage change, checkpoint/resume and throughput on the tiny twin; retain all stop-machine gates.
- Session 4: normalize tool data, inspect assistant masks, rehearse chat/tool tuning, BFCL and JSON checks; test GGUF tool tags and 8k runtime settings.
- Session 5: only after agreement and rehearsal, add the long-context tail before mixed chat tuning; run both eval families, export and stop the machine as before.

This change supplies config, format code and automated tests only. It adds no training scripts, dataset downloads or tool executor. CPU 8k forward checks model execution, not trained long-context quality. Test results are recorded in `PROGRESS.md`.
