# Decision log

Every change to a frozen decision (see CLAUDE.md) gets an entry here first, agreed by both people.
Newest at the top. Format: date · who · decision · why · what it replaces.

### 2026-11-19 · Adi · PROPOSED: chat preparation and SFT recipe
- Decision: pin `datasets==5.2.0` for revision-pinned streaming preparation of all smol-smoltalk train plus smoltalk `apigen-80k` train. Hash-hold out 1% of each, normalize APIGen's JSON call lists into our per-call tags, and preserve actual assistant-only loss masks in memmapped arrays. No tool results are fabricated where the source has none.
- Why: make the PR #2 tool-calling proposal runnable with inspectable real data. Start with the natural source mix (report conversation and assistant-token shares), at most 2 epochs, peak SFT LR 3e-4, 20-update warmup then linear decay, 32,768 padded-token updates / 8,192-token microbatches. Select best held-out assistant loss; still evaluate BFCL/chat regressions before accepting quality.
- Replaces: no executable chat preparation/SFT recipe; `datasets` was deferred in requirements. Frozen architecture/tokenizer/parameter count stay unchanged.
- Status: needs partner's agreement; CPU smoke is a plumbing check, not validation of this recipe's quality.

### 2026-11-19 · Adi · PROPOSED: context 8,192
- Decision: pretrain at 2,048 (RoPE theta 10,000), then use the final ~5–10% of tokens at 8,192 with RoPE theta 500,000 and long FineWeb-Edu documents. Chat tuning drops conversations over 8,192. Export GGUF / Ollama with `num_ctx 8192`.
- Why: our EDA found 9.4% of docs exceed 2,048 but hold 41.5% of tokens. A short long-context stage avoids full 8k pretraining's estimated ~2–3x slowdown. Parameter count stays 180,925,056; bf16/fp16 KV cache is ~335 MB at 8k (batch 1), separate from weights.
- Replaces: fixed 2,048-token context; long context excluded from v1.
- Status: needs partner's agreement. Config and format code are a reviewable proposal, not evidence of trained 8k capability.

### 2026-11-19 · Adi · PROPOSED: tool calling in v1
- Decision: mix function-calling conversations (e.g. smoltalk `apigen-80k` / Salesforce xLAM-60k) into chat tuning. Use Hermes-style plain-text `<tool_call>` tags, parallel calls, tool responses, and assistant-only loss. Keep the tokenizer unchanged.
- Why: calling against given schemas is a useful, bounded capability for a small local model. Aim for strong single/parallel calls; expect weak long multi-step agents. Evaluate BFCL simple/multiple/parallel and JSON-validity rate outside lm-evaluation-harness.
- Replaces: tool use excluded from v1; chat tuning on smol-smoltalk alone.
- Status: needs partner's agreement. Data mix, licenses and evaluation targets need rehearsal before the H100 run. See `TOOL_CALLING.md`.

### 2026-10-24 · Adi · PROPOSED: Jarvis Labs as the GPU provider
- Decision: use Jarvis Labs for rented GPUs (cheap GPU for Sessions 2–4 if needed, H100 SXM for Session 5). See `docs/JARVIS.md`.
- Why: a machine is already set up there; H100 SXM ₹330/h (about $3.8) is within the plan's $3.25–4 range; pause keeps storage.
- Changes: auto-stop = `scripts/stop_machine.py` (pause via Jarvis SDK). Data prep can run there so shards are already beside the GPU.
- Status: needs partner's agreement.

### 2026-10-23 · Adi · PROPOSED: data prep (Session 2) on Colab too
- Decision: skip renting a GPU provider for now; run Session 2 on Colab, with shards on Google Drive `/MyDrive/chotu/shards` (~20 GB, needs Google One storage).
- Why: start now without a second provider.
- Consequence: before Session 5 the shards must move to the H100 machine's volume (upload to a private HF dataset repo, or re-run the script there, about 1–3 h). The provider choice is postponed, not cancelled.
- Status: needs partner's agreement.

### 2026-10-23 · Adi · PROPOSED: rehearsals on Google Colab
- Decision: Sessions 3–4 run on Colab (L4/A100) instead of a rented cheap GPU; Session 2 data and Session 5 H100 run stay on rented machines. Storage for Colab work: Google Drive `/MyDrive/chotu`.
- Why: Adi already uses Colab; no extra provider for rehearsals. Colab can't host the long unattended run (time limits, disconnects).
- Note: the auto-stop and checkpoints-on-the-volume tests must still be rehearsed once on the rented provider before Session 5.
- Status: needs partner's agreement.

### 2026-10-23 · Adi + partner · Plan frozen
- Decision: architecture, tokenizer, data, recipe, success bar as listed in CLAUDE.md.
- Why: validated against sources and an independent review; reduced to 7 sessions for limited time.
- Replaces: —
