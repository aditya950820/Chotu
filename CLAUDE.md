# Chotu — a small general-purpose LLM built from scratch

A learning project by two people: an English chat model of about 181M parameters that runs locally on laptop CPUs. Claude builds and runs the code; the humans make decisions, check each gate and learn. The full plan lives in the "Small LLM Build Checklist" doc.

## Rules for Claude

- **Never change a frozen decision below without asking first.** If a change is agreed, record it in the decision log with the reason.
- The model size test (180.9M parameters) must always pass. If it fails, stop and report it; don't work around it.
- Every script that runs on a rented GPU ends by stopping the machine, even when a step fails.
- Write tests and a short report for each piece of work, not just code. The humans check the report.
- Keep code readable: one clear training script the humans can study, not a framework.
- Save checkpoints and data only to the persistent volume, never to the machine's own disk.
- At the end of every session, update the progress log below.
- When teaching, be Socratic: one question at a time, wait for the answer, use a diagram or analogy, no maths.

## Frozen decisions

| Decision | Choice |
| --- | --- |
| Architecture | Decoder-only Hugging Face Qwen3 layout: 32 layers, 640 hidden, 10 query / 5 KV heads, head_dim 64, SwiGLU MLP 1,792, RMSNorm, RoPE (theta 10,000), QK-norm, tied embeddings, no biases |
| Config gotcha | Set every Qwen3 field explicitly. The defaults (head_dim 128, untied embeddings) silently change the model |
| Size | 180.9M parameters, asserted by a test |
| Context | 2,048 tokens |
| Tokenizer | SmolLM2's 49,152-token BPE (`HuggingFaceTB/SmolLM2-135M`), used as-is. It splits digits one by one and has `<\|im_start\|>` / `<\|im_end\|>` |
| Pretraining data | FineWeb-Edu `sample-10BT` only, one pass (about 10B tokens), tokenized to uint16 shards with end-of-text between documents, 0.5% held out for validation |
| Training recipe | AdamW (0.9, 0.95), weight decay 0.1, gradient clipping 1.0, bf16, 524,288-token batch, warmup-stable-decay (~700 warmup steps, decay over the final 20%) |
| Learning rate | Picked by a test of 1.5e-3 vs 3e-3 (0.5B tokens each, 100-step warmup, 15% decay, judged by validation bits-per-byte) |
| Speed | torch.compile, FlashAttention via SDPA, fused AdamW, fused cross-entropy (e.g. Liger kernels) |
| Chat tuning | `HuggingFaceTB/smol-smoltalk`, ChatML template, loss on assistant tokens only (`<\|im_end\|>` included), drop conversations over 2,048 tokens, up to 2 epochs, keep the best held-out checkpoint |
| Evals | lm-evaluation-harness: HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande, plus validation bits-per-byte |
| Success bar | Beat GPT-2 124M on HellaSwag and ARC-Easy, scored on our own harness |
| Local file | GGUF Q8_0 (about 190 MB), run with llama.cpp / Ollama using an explicit ChatML template and a stop on `<\|im_end\|>` |
| Target tasks | Short Q&A, summaries, rewriting, simple instructions, chat |
| Not in v1 | Multilingual, long context, tool use, retrieval, own tokenizer, DPO, 4-bit files |

## Kill rules for the H100 run

- Stop if MFU is under 10% (about 74k tokens/s on an SXM H100): that means a setup bug. 15–25% is normal.
- Stop and alert on NaN, or if loss is not below about 6 by step 300. Loss should start near 10.8.
- Auto-resume from the last checkpoint after a crash; alert on a loss spike that doesn't recover.

## Sessions

1. Set up: accounts, this repo, model config with the size test, 20 test prompts.
2. Prepare data: download sample-10BT, tokenizer checks, shards, report.
3. Rehearse pretraining on a ~10M model on a cheap GPU: tiny run, overfit one batch, kill-and-resume, alert test, GPT-2 124M and SmolLM2-135M baselines.
4. Rehearse chat tuning and export on the tiny model, then dry-run the whole chain (thresholds off) and confirm the machine stops itself.
5. Launch the H100 chain: preflight → speed check → LR test → main run → evals → chat tuning → evals → export → upload to a private Hugging Face repo → stop.
6. Mid-run check-in.
7. Results, run on laptops, model card, retro, delete paid resources.

## Decision log

- 2026-10-23: Plan frozen as above.

## Progress log

- 2026-10-23: Repo created with this CLAUDE.md. Next: Session 1.
