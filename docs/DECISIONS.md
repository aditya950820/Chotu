# Decision log

Every change to a frozen decision (see CLAUDE.md) gets an entry here first, agreed by both people.
Newest at the top. Format: date · who · decision · why · what it replaces.

### 2026-11-22 · pranay · PROPOSED: speed-review follow-ups (needs Adi)
- Decision 1: pin `liger-kernel==0.8.5` (installed by `run_all.sh`) and make it required for the full model. It is the frozen "fused cross-entropy"; missing it was a main cause of the 5% MFU test.
- Decision 2: the kill rule "MFU under 10% (about 74k tok/s on an SXM H100)" keeps 74k on an H100 SXM but scales with peak on other cards (A100 ≈ 23k, H100 PCIe ≈ 57k). Without this, a healthy A100 fallback run could be killed at step 50.
- Decision 3: pranay chose **H200 Spot on Jarvis Labs** (₹217/h, ₹3,600 budget) for the main run on 2026-11-22. This replaces the agreed H100 SXM entry below. Go/no-go is about 200–210k tok/s at 2k context, not 180k (see `docs/SPEED.md`). Keep `--ckpt-every 100`, so a preemption costs at most 100 steps; auto-resume handles the restart. The kill-rule floor stays 74k tok/s, because the H200 has the same compute peak as the H100 SXM.
- Why: see the 2026-11-22 review section in `docs/SPEED.md`.
- Replaces: unpinned/optional Liger; a fixed 74k floor on every card; H100 SXM (on-demand) as the Session 5 GPU.
- Status: PROPOSED 2026-11-22 by pranay, who has chosen H200 Spot; Adi to agree or object to all three.

### 2026-11-19 · Adi + partner · AGREED: reproducible eval/export tooling
- Decision: pin `lm_eval[hf]==0.4.13`, `sentencepiece==0.2.3`, `cmake==4.4.5`; use revision-pinned Gorilla BFCL Python simple/multiple/parallel data and llama.cpp (commits in `PROGRESS.md`). Use the checkout's bundled gguf Python module. All LM models get identical zero-shot raw prompts, fp32 and 1,024 context (GPT-2 capacity); compare acc/acc_norm, success on strictly greater GPT-2 acc_norm for HellaSwag and ARC-Easy.
- Why: make step-3 reports/export repeatable, cache baselines separately by model/settings/revision, and keep uploads private and explicitly invoked. BFCL JSON scorer is a re-implementation, not official-harness equivalence; CPU smoke is not a quality result.
- Replaces: deferred lm-eval/gguf tooling requirements; architecture/tokenizer remain unchanged. Context/tool decisions are agreed separately below.
- Status: AGREED 2026-11-19; Aditya confirmed partner agreement. Full evaluation, Ollama runtime and real private upload remain gates for later work.

### 2026-11-19 · Adi + partner · AGREED: chat preparation and SFT recipe
- Decision: pin `datasets==5.2.0` for revision-pinned streaming preparation of all smol-smoltalk train plus smoltalk `apigen-80k` train. Hash-hold out 1% of each, normalize APIGen's JSON call lists into our per-call tags, and preserve actual assistant-only loss masks in memmapped arrays. No tool results are fabricated where the source has none.
- Why: make the PR #2 tool-calling decision runnable with inspectable real data. Start with the natural source mix (report conversation and assistant-token shares), at most 2 epochs, peak SFT LR 3e-4, 20-update warmup then linear decay, 32,768 padded-token updates / 8,192-token microbatches. Select best held-out assistant loss; still evaluate BFCL/chat regressions before accepting quality.
- Replaces: no executable chat preparation/SFT recipe; `datasets` was deferred in requirements. Frozen architecture/tokenizer/parameter count stay unchanged.
- Status: AGREED 2026-11-19; Aditya confirmed partner agreement. CPU smoke is a plumbing check, not validation of this recipe's quality.

### 2026-11-19 · Adi + partner · AGREED: context 8,192
- Decision: pretrain at 2,048 (RoPE theta 10,000), then use the final ~5–10% of tokens at 8,192 with RoPE theta 500,000 and long FineWeb-Edu documents. Chat tuning drops conversations over 8,192. Export GGUF / Ollama with `num_ctx 8192`.
- Why: our EDA found 9.4% of docs exceed 2,048 but hold 41.5% of tokens. A short long-context stage avoids full 8k pretraining's estimated ~2–3x slowdown. Parameter count stays 180,925,056; bf16/fp16 KV cache is ~335 MB at 8k (batch 1), separate from weights.
- Replaces: fixed 2,048-token context; long context excluded from v1.
- Status: AGREED 2026-11-19; Aditya confirmed partner agreement. Config and format code are not evidence of trained 8k capability.

### 2026-11-19 · Adi + partner · AGREED: tool calling in v1
- Decision: mix function-calling conversations (e.g. smoltalk `apigen-80k` / Salesforce xLAM-60k) into chat tuning. Use Hermes-style plain-text `<tool_call>` tags, parallel calls, tool responses, and assistant-only loss. Keep the tokenizer unchanged.
- Why: calling against given schemas is a useful, bounded capability for a small local model. Aim for strong single/parallel calls; expect weak long multi-step agents. Evaluate BFCL simple/multiple/parallel and JSON-validity rate outside lm-evaluation-harness.
- Replaces: tool use excluded from v1; chat tuning on smol-smoltalk alone.
- Status: AGREED 2026-11-19; Aditya confirmed partner agreement. Data mix, licenses and evaluation targets need rehearsal before the H100 run. See `TOOL_CALLING.md`.

### 2026-11-19 · Adi + partner · AGREED: Jarvis Labs as the GPU provider
- Decision: use Jarvis Labs for rented GPUs (cheap GPU for Sessions 2–4 if needed, H100 SXM for Session 5). See `docs/JARVIS.md`.
- Why: a machine is already set up there; H100 SXM ₹330/h (about $3.8) is within the plan's $3.25–4 range; pause keeps storage.
- Changes: auto-stop = `scripts/stop_machine.py` (pause via Jarvis SDK). Data prep can run there so shards are already beside the GPU.
- Status: AGREED 2026-11-19; Aditya confirmed partner agreement. Originally proposed 2026-10-24.

### 2026-11-19 · Adi + partner · AGREED: data prep (Session 2) on Colab too
- Decision: skip renting a GPU provider for now; run Session 2 on Colab, with shards on Google Drive `/MyDrive/chotu/shards` (~20 GB, needs Google One storage).
- Why: start now without a second provider.
- Consequence: before Session 5 the shards must move to the H100 machine's volume (upload to a private HF dataset repo, or re-run the script there, about 1–3 h). The provider choice is postponed, not cancelled.
- Status: AGREED 2026-11-19; Aditya confirmed partner agreement. Originally proposed 2026-10-23; Colab remains an alternative to the agreed Jarvis provider.

### 2026-11-19 · Adi + partner · AGREED: rehearsals on Google Colab
- Decision: Sessions 3–4 run on Colab (L4/A100) instead of a rented cheap GPU; Session 2 data and Session 5 H100 run stay on rented machines. Storage for Colab work: Google Drive `/MyDrive/chotu`.
- Why: Adi already uses Colab; no extra provider for rehearsals. Colab can't host the long unattended run (time limits, disconnects).
- Note: the auto-stop and checkpoints-on-the-volume tests must still be rehearsed once on the rented provider before Session 5.
- Status: AGREED 2026-11-19; Aditya confirmed partner agreement. Originally proposed 2026-10-23; the actual rented-provider pause rehearsal remains required.

### 2026-10-23 · Adi + partner · Plan frozen
- Decision: architecture, tokenizer, data, recipe, success bar as listed in CLAUDE.md.
- Why: validated against sources and an independent review; reduced to 7 sessions for limited time.
- Replaces: —
