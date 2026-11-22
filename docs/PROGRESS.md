# Progress and handover

The single place that says where the project stands. Read it before starting; update it before you stop.
Newest entries go at the top of the log. Never rewrite someone else's entry; add a new one.

## Now

- **Current session:** 8k context, tool calling and all logged proposals agreed by both people on 2026-11-19 (Aditya confirmed). All four chain steps CPU-verified; parser hardening is on PR #3. PR #2 → main and PR #3 → PR #2 are still open at the live status check; neither is recorded as merged. Full evaluation, sample-reading gate and actual GPU/pause rehearsal remain.
- **Who's on what:**
  - Adi: parser fixes and agreed-decision documentation on PR #3; next: merge PR #2 into main, then integrate PR #3 into main, followed by cheap-GPU rehearsal using RUNBOOK.md
  - Partner (pranay): speed review PR (attention kernel choice, required Liger pin, card-scaled kill floor, sweep upgrades); needs Adi's agreement on the DECISIONS proposal
- **Blocked on:** speed-review PR #4 merges only after Adi agrees to the 2026-11-22 PROPOSED entry in `DECISIONS.md`
- **Paid resources running:** Jarvis Labs instance (GPU type: TODO), started 2026-10-24. Pause it when not in use. Status inherited; not checked this session. No paid resources started here.

## Next up

1. Session 1, remaining human steps:
   - [x] Hugging Face token and W&B key work (checked by `scripts/check_setup.py`)
   - [x] Model config + size test: 180,925,056 params, tests pass
   - [ ] Both: rewrite `prompts/test_prompts.md` in your own words
   - [ ] Provider: Jarvis Labs agreed (`docs/JARVIS.md`); run its first-time setup and `stop_machine.py --dry-run`
   - [ ] Partner: get collaborator access, clone, make own `.env`, run `python scripts/check_setup.py`
   - [ ] Partner: set up Colab per `docs/COLAB.md` and run `notebooks/colab_setup.ipynb` (Adi: done, 4/4 checks OK)
   - [x] Partner: Colab decisions agreed 2026-11-19; Aditya confirmed
2. Session 2 gate: both read the 10 samples in `reports/data_report.md` (in the private HF dataset `magnateadi/chotu-fineweb-edu-10bt-tokens`). Don't start another data run; the shards are done. Then Session 3.
3. Follow `RUNBOOK.md` on a cheap GPU: verify the persistent mount, CUDA throughput, real W&B, private upload permissions and actual automatic pause. The 8k/tool decisions are agreed; inspect rehearsal reports before the full H100 chain. CPU plumbing success is not a quality or GPU-performance gate.

## Open questions

- Pending approval (Adi): 2026-11-22 PROPOSED entry in `DECISIONS.md` (Liger pin, card-scaled kill floor, H100 SXM vs H200 Spot). Everything logged before it was agreed 2026-11-19.

## Log

### 2026-11-22 · pranay · Speed review: can we reach 30–35% MFU?
- Did: reviewed Adi's speed fixes. They're correct: Liger, one sync per step, prefetch and the honest MFU formula. Added speed-only changes: `--attention {auto,flash,cudnn,efficient}` (non-auto forbids the slow math fallback); a startup probe that prints which fast kernels accept our real shapes (`fused_kernels_ok=`); `liger-kernel==0.8.5` pinned and required for the full model; kill-rule tok/s floor scaled by card peak (H100 SXM unchanged at 74k); `wall_tokens_per_second`, `gpu_mem_gb` and `CKPT seconds=` logging; a 5-config sweep (auto/flash/cuDNN, 131k, max-autotune) with saved logs and failure reasons. Recipe unchanged.
- Result: estimate for H100 SXM is **25–35% MFU (155k–218k tok/s at 2k)**. 30% is realistic, 35% is the top end, and above ~40% is unlikely at 640 hidden. A100-80GB: 40–50% but only 78k–98k tok/s (28–35 h for 10B). Reasoning: `docs/SPEED.md`. Tests: 110 passed offline; the 9 failures/errors all need the HF tokenizer download or `lm_eval`, which this sandbox can't reach. With a stub tokenizer, `test_pretrain`, `test_chain` and the size gate all pass (50 passed). No GPU run.
- Next: Adi reviews the PROPOSED entry in `DECISIONS.md`. Then run `speed_sweep.py` on the card we agree on. If `fused_kernels_ok=math only`, stop and report.
- Notes for the other person: the ₹3,600 / H200 Spot go/no-go of 180k tok/s is too low. With the LR test and the slower 8k stage, it needs ~200–210k tok/s. Also, tok/s about halves after `SWITCH` to 8k; that's expected. The agreed provider entry says H100 SXM, so H200 Spot needs your OK.

<!-- Template:
### YYYY-MM-DD · Name · Session N
- Did:
- Result / numbers:
- Next:
- Notes for the other person:
-->

### 2026-11-22 · Adi · Speed fixes for the 5% MFU H100 test
- Did: fixed six speed problems in `train.py` (main one: liger-kernel was never installed on the GPU, so the loss ran as 256 tiny checkpointed chunks per microbatch; also per-microbatch GPU syncs, no data prefetch, MFU formula missing attention). Added `scripts/speed_sweep.py` (15-min GPU check, pauses machine) and `docs/SPEED.md`. Chain auto-uses the sweep's fastest settings. Recipe unchanged.
- Result: `pytest -q` 115 passed, size gate OK; 1 local-only failure (lm_eval not installed on the Mac). Real GPU speed not measured yet.
- Next: rent H200 Spot ~15 min, run `speed_sweep.py`; go for full 10B on the ₹3,600 budget only if ≥ ~180k tok/s.
- Notes for the other person: MFU numbers now count attention and use the card's own peak, so they are not comparable to older logs (old 5% ≈ 7% new). PROGRESS lists a Jarvis instance started 2026-10-24 as possibly running; please check it isn't billing.

### 2026-11-19 · Adi · Decisions agreed and tool parser hardened
- Agreement: Aditya confirmed partner agreement on 2026-11-19. All seven formerly proposed decision entries are now AGREED, including 8k context, tool calling, SFT/dependency pins, eval/export pins and provider/Colab choices. Updated frozen values and current guides; historical log entries retain their original proposal status as history.
- Parser: decode JSON before checking closing tags, preserving literal opening/closing tags inside argument strings. Reject nested NaN/Infinity/-Infinity on parsing and rendering, including tools JSON. Catch excessive-nesting RecursionError and retain later valid calls. Tool scoring uses the same JSON-aware boundaries instead of a delimiter regex; APIGen list-in-one-tag normalization remains intact.
- Merge status: live GitHub checks show PR #2 open against main and PR #3 open against the PR #2 branch. Neither has merged into main yet; no merge performed here. Merge PR #2 first, then integrate PR #3 into main after review. This corrects the requested merged-status wording to match observed state.
- Verification: Python 3.13 `python -m pytest -q` → **114 passed in 13.86s**; focused chat-format/evaluation/chat-training tests → **64 passed in 7.00s**. Exact **180,925,056** model-size gate passes. New regressions cover literal tags/escaped strings, APIGen source-list conversion, nested nonfinite values, deep nesting recovery, missing closers and strict scorer text handling. The first nesting fixture did not trigger Python 3.13's C decoder guard because it accepts nesting beyond the Python recursion limit; the corrected fixture uses 100,000 levels and passes. Grep finds no remaining uppercase PROPOSED entries; historical proposal wording is retained only as history.
- Paid resources: none started; inherited Jarvis status above remains unverified. No GPU, upload or pause calls needed for this patch.

### 2026-11-19 · Adi · One-command chain, step 4
- Did: `run_all.sh` preserves the template torch build using a pip constraint, installs requirements/Jarvis SDK and transfers shutdown ownership to readable `run_chain.py`. Stages: preflight, verified resumable shards, chat prep, speed, frozen LR trial, pretrain, base evals/BPB, chat, final evals/tools/BPB, Q8 export, private upload and final report. Atomic stage markers skip done work; code/settings signatures and output checks guard resume. Training exit 1/SIGKILL retries up to two times; exits 20–23 do not retry. Every shutdown-capable child gets `--no-auto-stop`; the chain owns pause in `finally` with SIGINT/SIGTERM handling and a local `--no-stop` escape. Shutdown still attempts the API if log storage fails. Rehearsal W&B uses a separate project. Added CPU/local preflight options to `check_setup.py`, a fresh-instance RUNBOOK and links.
- Checks: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false python -m pytest -q` → **105 passed in 13.71s**; exact **180,925,056** size gate passes. Orchestration tests cover order/skip, manual controls/config guards, retry/no-retry, mocked stop on success/failure/exception/signal, pause failure/full-log-volume fallback, dry-run estimates, shard integrity, LR settings/choice, separate rehearsal upload/project and child shutdown ownership. `bash -n scripts/run_all.sh` and staged whitespace checks passed.
- Final-code CPU rehearsal: fresh `/home/ubuntu/chotu-chain-final/rehearsal`, private train_013 + all 14 val shards; SHA/size/EOT/document counts checked against revision **b7575cfca0033585475fff88dcecab40d994b5ff** and `.done`. **190,573,056 train / 50,147,762 val tokens**, **181,219 / 48,403 documents**. Shared HF/llama build caches reused; all stage outputs generated afresh. Command: `python scripts/run_chain.py --storage /home/ubuntu/chotu-chain-final --cache-dir /home/ubuntu/chotu-smoke --rehearsal --rehearsal-tokens 262144 --no-stop --skip-upload --device cpu --shards-subset --retries 0`.
- Resume proof: SIGKILLed pretrain child at update 12 with retries explicitly off; chain exited 1 without writing that stage marker. Same command skipped the five completed stages, dispatched `RESUME step=10 stage=short`; replay losses 11/12 identical (**8.140087127685547 / 8.01860523223877**). `SWITCH step=24 seq_len=8192 theta=500000.0`; **31 unique updates**, loss **10.836890 → 7.554592**, best BPB **2.43219204**. Document-aligned packing left 8,192 tokens unused (31/32 planned updates), as already documented for pretraining.
- Pipeline results: tiny speed **1,573 tok/s**, H100-equivalent MFU **0.00006756**; toy LR trials chose **0.003** (BPB **2.788737** vs **3.032097**). Eight chat updates: loss **8.322084 → 7.797956**, best val loss **7.797294**. Base/final BPB **2.432192 / 2.437280**. HellaSwag/ARC-Easy limit 2: base and final acc/acc_norm **0/1.0**, **0.5/0.5**; GPT-2 and SmolLM2 **0/0.5**, **0/0.5**; strict GPT-2 gate not met, smoke-only. APIGen limit 2 exact/name **50%** (one gold no-call), JSON **0%**; BFCL simple/multiple/parallel limit 2 accuracy/JSON **0%**; proposed tool gate failed nonfatally. Q8_0 checks context **8192**, vocabulary **49152**, theta **500000**, ChatML metadata; llama.cpp decoded **8** tokens, exit **0**. All 12 stage markers exist; upload is explicitly skipped and no HF repo was created/uploaded.
- Outputs: `reports/final_report.md`, per-stage `logs/`, `runs/` and `export/` under the fresh rehearsal directory above. Launcher evidence: `/home/ubuntu/chotu-smoke/chain-final-first.log`, `chain-final-resume.log`, `chain-dry-run.txt`, `step4-pytest.log`. Earlier development rehearsal also exercised actual SIGTERM handling (exit 143); final-code signal/pause paths are mocked in tests.
- Full H100 dry plan: speed 26,214,400 + LR 2×500M + main ~10B tokens; at 74k–150k tok/s **20.42–41.39** short-window compute hours, conservative planning **22.42–66.08 hours / ₹7,398–₹21,808** at ₹330/h, excluding storage/retries; not a spending guarantee. Preflight computes disk from shard metadata, five runs' checkpoints/exports, scratch/data/build/cache and 20% margin.
- Unverified / next: no GPU, full-data/full-size run, live W&B/Jarvis pause, actual private upload or Ollama runtime; installation wrapper syntax/dry-run verified, real CUDA-template installation not executed. Rehearsal scores are not quality results. No paid resources started; inherited Jarvis status above remains unverified. Use RUNBOOK on the cheap GPU first and confirm the machine really pauses.

### 2026-11-19 · Adi · Evals, export and upload, chain step 3
- Did: `eval_lm.py` runs the five frozen harness tasks and cached GPT-2 / SmolLM2 baselines under identical zero-shot settings, comparison/gate table and optional trainer BPB. `eval_tools.py` runs greedy ChatML calls on held-out APIGen and pinned official BFCL Python simple/multiple/parallel, reporting JSON validity, exact/allowed-value matching, failure examples and nonfatal proposed gates. `export_gguf.py` converts Q8_0, writes an explicit Ollama ChatML/tools Modelfile, checks GGUF metadata and runs real llama.cpp CPU generation. `upload.py` validates local artifacts, guards private visibility (including recheck after creation), and submits a model/card/reports commit only on explicit invocation. Usage/details: `EVALUATION_EXPORT.md`.
- Pins/source: `lm_eval[hf]==0.4.13`, `sentencepiece==0.2.3`, `cmake==4.4.5`; llama.cpp **`9a080f44b503c64f66c84a6274b9d9ee5d5b09d5`**, bundled `gguf-py` 0.19.0. BFCL official `ShishirPatil/gorilla` **`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`**, `bfcl_eval/data/BFCL_v4_{simple_python,multiple,parallel}.json` + matching `possible_answer/` (400/200/200 examples). All 800 function-doc rows convert to JSON schemas. This is a JSON-format re-implementation of the relevant AST rules, not the official BFCL harness or a claim of full leaderboard comparability.
- LM CPU smoke: step-2 tiny `best/`, HellaSwag and ARC-Easy, limit 2/task; acc/acc_norm: tiny **0/1.0**, **0.5/0.5**; GPT-2 **0/0.5**, **0/0.5**; SmolLM2-135M **0/0.5**, **0/0.5**. Strict GPT-2 acc_norm gate not met (ARC tie); explicitly smoke-only. Three model-specific cache files written. Tiny fixed-set validation BPB **2.35225417** on 2,048 target tokens. These minuscule samples have no quality significance. Settings: raw prompts, no chat template, fp32, max context 1,024 for every model, seed 1234.
- Tool CPU smoke: 10 held-out APIGen + 10 official BFCL simple, greedy max 64 new tokens. APIGen exact-call/name accuracy and JSON validity **0%**; BFCL simple accuracy/JSON validity **0%**. Proposed simple >=50% / JSON >=95% gate **failed**, nonfatal as intended. Generated outputs/failures stored with reports; no calls executed. Empty/plain text cannot count as valid JSON; gold no-call refusals are excluded from the JSON denominator and still scored for correct no-call prediction.
- Export CPU smoke: actual tiny Q8_0 file (~8.9 MiB), qwen3, context **8,192**, vocab **49,152**, tokenizer BPE model `gpt2`, EOS 0, unchanged ChatML IDs, theta **500,000**, file type 7/Q8_0. `llama-simple` decoded **8 tokens**, exit **0** (newlines from this tiny model, not useful chat). At this pin `llama-cli` needs server enabled, so built the allowed `llama-simple` equivalent. Corrected its positional-prompt invocation and verified decoded-token count rather than trusting echoed prompt text.
- Checks: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false python -m pytest -q` → **78 passed in 15.46s**, including allowed/optional/wrong-type/parallel BFCL matching (nested wrong-type values return false, not a crash), APIGen scoring, LM cache/settings/verdict, Modelfile content and fully mocked upload privacy/create/recheck guards. Existing exact parameter-size gate passes. A mock exception fixture initially needed the installed HF API's required response argument; fixed it without network calls.
- Artifacts: `/home/ubuntu/chotu-smoke/{lm-eval.json,lm-eval.md,lm-eval.log,tool-eval.json,tool-eval.samples.json,tool-eval.log,export.log}` and `export/{chotu-q8_0.gguf,Modelfile,export-report.json,llama-generation.log}`. All datasets/models/builds/cache remain outside git on persistent storage.
- Unverified / next: full five-task runs, full BFCL scores, 181M export/parity/quality, Ollama runtime, CUDA/W&B/Jarvis, real upload permissions and visibility race behavior. No HF repos created and nothing uploaded (tests use mocked HfApi); no paid resources started, inherited paid-resource status not checked. `git pull --rebase` could not run because this local branch has no upstream; tracking unchanged. Next: step-4 orchestration and independent shutdown ownership. No push requested/performed.

### 2026-11-19 · Adi · Chat preparation and tuning, chain step 2
- Did: `prepare_chat.py` streams revision-pinned train sources into uint16 IDs / uint8 assistant masks / int64 offsets; SHA-256 ~1% holdouts, counted conversion drops, raw APIGen eval JSONL and marked-token report. Shared `train.py --stage chat` adds padded length buckets, token-weighted assistant loss, two-epoch SFT, held-out selection, atomic HF `best/`, resumable checkpoints and done marker. Pretraining now exports HF `final/`; validation defaults to 1,048,576 fixed tokens evaluated within the microbatch budget. `--no-auto-stop` explicitly transfers shutdown ownership. Usage: `CHAT_TUNING.md` / `PRETRAINING.md`.
- Schema: inspected the real 83,144-row APIGen train parquet. System text contains `<tools>[{"type":"function","function":{...}}]</tools>`; assistant emits one `<tool_call>[{"name":...,"arguments":{...}}, ...]</tool_call>`. Only system/user/assistant roles, no tool results. Preserve refusals; replace incompatible source formatting instruction and render one call per tag. A raw positive-call example and real smoltalk example are captured in `tests/chat_fixtures.json`; positive fixture matches the actual parquet row exactly.
- Preparation CPU smoke: 2,000 rows/source; smol 2,000 converted, 1,984 kept (16 over 8k), 1,963 train / 21 val; APIGen 2,000 converted/kept, 1,981 train / 19 val. No other drops in this limited sample. Conversation shares 49.80% / 50.20%; assistant-token shares 89.97% / 10.03% (1,303,396 / 145,254 assistant tokens). Length p50/p90/p99/max: smol 909.5/1,360/1,773.58/8,018; APIGen 452/799.5/1,141.05/1,834. Gold JSON roundtrip **3,156/3,156 = 100%**. Report has five marked-loss samples per source, and held-out raw JSONL has 19 rows. Arrays occupy 7,942,408 bytes.
- Full source metadata: smol train **460,341 rows / 922,475,727 parquet bytes**, revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`; APIGen train **83,144 rows / 49,798,856 bytes**, revision `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`. Limited processing times 4.49s + 2.96s project to **17.2 + 2.0 ≈ 19.3 minutes** for full preparation, excluding setup/cold-download overhead. First-2k composition and network variability make this approximate; full preparation was not run. Natural full-source proportions differ from this equal-count smoke.
- Tests: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false python -m pytest -q` → **43 passed in 12.08s**, including real fixture conversion, stored masks, padding/buckets, exact chat crash/resume weights/losses, best selection through a validation regression, epoch-end evaluation and no-auto-stop. Existing size gate remains 180,925,056. An additional one-update real-shard tiny run published `final/`; `AutoModelForCausalLM` and tokenizer loaded with vocabulary 49,152.
- Chat CPU smoke: initialized tiny from step-1 checkpoint 58, SIGKILL after update 22, resumed checkpoint 20; replay losses 21–22 identical. Completed 40 unique updates, loss **8.2334 → 7.2814**; held-out loss at updates 10/20/30/40 **7.6260 / 7.5756 / 7.5532 / 7.5383**. Last checkpoints 35 and 40, done marker present. `AutoModelForCausalLM.from_pretrained(best)` and tokenizer load: context 8,192 / theta 500,000 / vocab 49,152. Logs: `chat-before-kill.log`, `chat-resume.log`. The chat done-metric key was corrected after the smoke process started; tests cover it and checkpoint-40 finalization is rechecked separately without extra training updates.
- Choices: proposed SFT peak LR 3e-4, 20-update warmup then linear decay, at most two epochs, 32,768 padded-token updates / 8,192-token microbatches; no source resampling. Best uses token-weighted held-out assistant CE. Pretraining quality/throughput/spike thresholds do not apply to SFT; nonfinite loss/gradient stops do. HF `best/` and `final/` are atomic relative symlinks under the run's `hf/`; retain targets or copy dereferencing links.
- Notes for the other person: preparation report at `/home/ubuntu/chotu-chat-smoke/reports/chat_report.md`, raw heldout at `chat/apigen-heldout.jsonl`; CPU logs under `/home/ubuntu/chotu-smoke/`. No paid resources started. Full-data SFT, CUDA/compile/Liger, live W&B/Jarvis and tool/chat quality remain unverified; proposals still need agreement. Next: step 3 evaluations, then export/orchestration. Do not push this branch yet.

### 2026-11-19 · Adi · Pretraining chain, step 1
- Did: memmapped deterministic single-pass batches, long-document tail indexing, a readable pretraining loop, WSD, exact decoded-target UTF-8 BPB, atomic/resumable checkpoints, optional W&B, kill-rule exit codes and standalone Jarvis shutdown. Usage and choices: `PRETRAINING.md`. No chat/eval/export/orchestrator work yet.
- Tests: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q` → **32 passed in 8.91s**, including exact synthetic CPU weights/loss after resume before and after the rotary/8k switch. The 180,925,056 size gate passes unchanged.
- Real data: ran public `prepare_data.py --max-docs 2000 --max-files 2` (existing first-row-group mode returned 1,000 docs/file). Then downloaded only private `train_013.bin` (381,146,112 bytes) and all 14 val shards to `/home/ubuntu/chotu-smoke`; no large train shards downloaded and no credentials saved in git.
- Final smoke (private shards): 60 planned / **58 actual** updates, batch=micro=8,192, LR .003, warmup 5, long fraction .15, decay .2. SIGKILL after update 27; automatic resume from 25 replayed updates 26/27 with identical losses. Switch after update 51: 2,048/theta 10,000 → 8,192/theta 500,000. Loss 10.83689 → 7.40003; final/best validation BPB 2.355529. Exit 0, done marker and exactly the last two checkpoints (55,58), 58 unique metric rows. Logs: `/home/ubuntu/chotu-smoke/private-before-kill.log`, `private-resume.log`; run: `runs/private-resume-60` under the same storage.
- Full train_013 tail stats (seed 42, batch 524,288, long fraction .08): 190,573,056 train tokens; short prefix 175,112,192; tail 15,460,863 tokens / 14,640 complete docs. ≥2,048: 1,390 docs / 6,161,799 tokens. ≥8,192: 138 docs / 1,949,912 tokens. Counts include EOT. Doc-start alignment/whole batches yield 357 of 363 planned updates, 3,402,239 unused tokens, 241 preferred starts. JSON: `/home/ubuntu/chotu-smoke/train-013-tail-stats.json`.
- Choices / caveats: every long window starts at a document boundary; prefer ≥4,096-token docs within a 4,096-token lookahead. No repeated targets, but alignment scraps lower the realized long fraction/token count (reported, not hidden). Decay matrices except embeddings; no decay norms/vectors. Unrecovered 200-update spikes alert and stop (23). Standalone CUDA jobs pause Jarvis in `finally`; later chain driver must explicitly take over shutdown ownership. CUDA/compile/Liger, live W&B and real Jarvis pause are untested here. Synthetic tests mock the shutdown call. No claim of trained 8k quality.
- Next: review, then step 2 chat tuning. Use the existing Python 3.13 `~/chotu-venv`. Commit locally only; no push requested. Frozen 8k/tool decisions still need partner agreement.

### 2026-11-19 · Adi · Proposal: 8k context + tool calling
- Did: proposed both frozen-decision changes; added staged context config, plain-text ChatML tool formatting/parsing, assistant-only masks, and tests. No training scripts or paid runs.
- Result / numbers: main baseline: 7 passed in 9.94s. Proposal: 18 passed in 4.71s (`PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 pytest -q --durations=3`); CPU 8k tiny forward 0.62s. Both stage configs: exactly 180,925,056 parameters.
- Environment: default Python 3.10 could not install pinned NumPy; used an isolated Python 3.13 environment with unchanged requirements via uv. New test collection needs repo root on `PYTHONPATH` (or `python -m pytest`). Transformers 5.20.2 uses `rope_parameters`; config sets that and exposes the requested theta alias.
- Next: partner review, then rehearse long-context stage and tool data/evals in Sessions 3–4. Session 2 sample-reading gate still applies.
- Notes for the other person: proposals are not approvals. Work committed locally only, no push access. Paid-resource status above is inherited and was not checked in this session.

### 2026-10-31 · Adi · Learning: lesson 4 (batch size lab)
- Did: lesson 4. `learning/batch_size/compass.py` measures how much gradient directions from 1–256 windows agree (at the random start and after 256 strides); race of 4 batch sizes on the same 2,048 windows. Page "Ask the Crowd" built by `learning/batch_size/build.py`; logs in `learning/batch_size/runs/`.
- Results: miss angle falls from ~75° (1 window) to ~15° (256); small crowds get noisier after training; race winner 8/stride (6.56), 128/stride lost with only 16 strides (7.52).
- Code: `learning/nudge_size/experiment.py` gained `--micro` (gradient accumulation; the 128-window run ran out of memory without it), `--eval-every-windows` and `--save`. This is the same accumulation the H100 run needs (32 × 8).
- Next: lesson 5 (train vs validation, bits-per-byte); Session 2 gate; choose where Session 3 runs.

### 2026-10-30 · Adi · Learning: lesson 3 (learning rate lab)
- Did: lesson 3 of the pretraining course. Wrote `learning/nudge_size/experiment.py` (tiny-twin CPU training at a chosen learning rate, logs loss, gradient size, update size, exam every 25 steps) and ran 8 runs (~1 h on 2 CPU cores). Logs in `learning/nudge_size/runs/`, page built by `build.py`.
- Results: best exam 3e-3 (6.56); U-shape on both sides; too-big strides show up as gradient clipping and wobble long before NaN. The experiment script is a first draft of the Session 3 training loop (AdamW 0.9/0.95, wd 0.1, clip 1.0, warm-up/hold/slow-down).
- Next: answer the lesson 3 questions in `docs/LEARNING.md`; lesson 4 (batch); Session 2 gate; choose where Session 3 runs.

### 2026-10-28 · Adi · Learning: pretraining (no training run)
- Did: started a "pretraining flight school" (8 lessons, Socratic) to prepare for watching the real run; lessons 1–2 done (loss = average surprise; three phases of a healthy loss curve and the sick shapes). Built two learning pages and `learning/guessing_game/` (scripts that rebuild the Guessing Game from our real val shards).
- Results: on 16,384 held-out tokens, untrained Chotu 10.91, common-words-only 7.53, SmolLM2-135M 2.77. So the "below ~6 by step 300" kill rule means Chotu must already use context, not just word frequency.
- Next: answer the open questions in `docs/LEARNING.md`; lessons 3–8; Session 2 gate (read the 10 samples); choose where Session 3 runs (Jarvis L4 or Colab).
- Notes for Pranay: ask Adi to share the "Guessing Game" page with you, or rebuild it with the commands in `learning/guessing_game/build.py`.

### 2026-10-24 · Adi · EDA notebook
- Did: `notebooks/eda_data.ipynb`, an end-to-end EDA on live data (our HF shards + 42k sampled source documents), with explainers and questions in 7 parts: totals and split, document quality/sources/years/length, text→tokens and the 2,048 window, vocabulary, junk check, train vs val, training implications.
- Results (ran end to end, 0 errors): 4.54 bytes/token (needed for bits-per-byte); median doc 630 tokens; **9.4% of docs exceed 2,048 tokens but hold 41.5% of tokens**; 474 vocab ids unseen in 300M tokens; chat markers absent (good); exact duplicates 0.03%; boilerplate lines exist (e.g. "1 Answer | Add Yours") but are rare; val matches train (median 633 vs 630 tokens); 19,181 steps for one pass; 56 tokens/param.
- Next: both read the notebook and answer the questions in `docs/LEARNING.md`; Session 2 gate (read the 10 samples), then Session 3.

### 2026-10-24 · pranay · Session 2 data done on a rented CPU VM
- Did: ran `scripts/prepare_data.py` on a rented 8 vCPU / 32 GB CPU VM (₹17/hr, no GPU) in tmux, without an HF token. Live W&B dashboard (`scripts/watch_data.py`, PR #1) plus Adi's Jarvis notebook in Jupyter, both reading the same run.
- Result: 14/14 files, 9,672,101 docs, **10,056,669,371 train tokens**, 50,147,762 val tokens (0.50%), 20.21 GB of shards, all tokenizer checks OK. Took 50.6 min (~3.3M tok/s, much faster than the ~0.1M tok/s per core estimate).
- Backup: uploaded to the **private** HF dataset `magnateadi/chotu-fineweb-edu-10bt-tokens` and checked (14 train + 14 val shards, 20.21 GB, matches local). Later machines download from there.
- Next: both read the 10 samples in `reports/data_report.md` (the gate); delete the VM; Session 3.
- Notes for Adi: your Mac run for Session 2 isn't needed any more, so you can stop it. Provider and H100 choice are still open (see thread notes: H100 ₹330/hr recommended, A100-80GB as fallback, L4 for rehearsals). The W&B key used on the VM was shared in chat, so please rotate it.

### 2026-10-24 · Adi · Live notebook for Jarvis
- Did: `notebooks/session2_jarvis_live.ipynb`: setup (keys typed in hidden fields into .env), data run in the background (won't start twice), live monitor (files, tokens, speed, ETA, CPU, disk, chart), learning cells on the newest finished shard, gate, push, pause.
- Tested: monitor and learning cells against test shards; all cells parse.

### 2026-10-24 · Adi · Jarvis Labs
- Did: `docs/JARVIS.md` (prices, pause rules, first-time setup, Session 2 on Jarvis), `scripts/stop_machine.py` (pauses the instance via the Jarvis SDK; `--dry-run`, `--list`), `.env.example` gains JL_API_KEY / JL_MACHINE_ID.
- Note: the Mac data run is still going (download ~6 MB/s). Running Session 2 on Jarvis instead puts the shards next to the GPU and avoids a 20 GB upload later.

### 2026-10-24 · Adi · X-ray 01 (learning visuals)
- Did: `chotu/xray.py` (tokenizer lens, shard microscope, token frequency, parameter map, forward-pass X-ray: attention per head, logit lens, per-layer signal) + `notebooks/xray_01_data_and_model.ipynb` with think-about-it questions. Reads real tokenizer/shards/model, never changes them; reusable on checkpoints after Session 3.
- Result: runs end to end (0 errors). Untrained model's top next-token guess ≈0.014% (chance is 0.002%); top 10 tokens cover ~24% of real text; params: MLP 60.9%, attention 21.7%, embeddings 17.4%.
- Also: Session 2 full run started natively on the Mac (12 cores, ~2 h estimate, data in `~/chotu-data`). The earlier sandbox run had died before writing data.
- Next: run Session 2 cell 3 (quick test), open X-ray 01, then cell 4 (full run). Answers to the questions → `docs/LEARNING.md`.

### 2026-10-24 · Adi · Session 2 started on Mac (Claude's sandbox)
- Did: cloned repo to `~/apps/Chotu`, Python 3.12 env in the Claude app's Linux sandbox (4 cores, 3 GB RAM, no torch needed), test mode passed (7/7 tokenizer checks OK), launched full run in background → `~/apps/chotu-data` (log: `~/apps/chotu-data/run.log`).
- Status: Mac disconnected right after launch; unknown whether the run is still going. Sandbox speed ~0.1–0.3M tok/s, so 10–25 h there; running natively on the Mac (`notebooks/session2_data_local.ipynb`) is several times faster. Resumable either way.

### 2026-10-24 · Adi · Mac option for Session 2
- Did: `scripts/setup_mac.sh` (venv, packages, Jupyter kernel, tests) and `notebooks/session2_data_local.ipynb` (shards to `~/chotu-data`, `caffeinate` keeps the Mac awake, resumable).
- Why: Colab gave only 2 CPU cores (~14 h); the Mac has more.
- Note: shards on the Mac must be uploaded to a private HF dataset before Session 5.

### 2026-10-24 · Adi · Fix Colab notebooks
- Problem: `can't open file '/content/scripts/prepare_data.py'`. Installing pinned numpy/pyarrow made Colab restart the session, which reset the folder to /content and cleared the settings.
- Fix: notebooks keep Colab's own torch/numpy/pyarrow, and every cell uses full paths (`cd /content/Chotu`, `--storage /content/drive/MyDrive/chotu`). After any restart, just rerun cell 1.

### 2026-10-23 · Adi · Session 2 code
- Did: `scripts/prepare_data.py` (tokenizer checks, uint16 shards with end-of-text per doc, deterministic 0.5% val split by doc id, one train+val shard per source file, resumable via .done markers, Drive space check, report with 10 samples), `tests/test_data.py`, `notebooks/session2_data.ipynb`.
- Result: test mode (2 files × 1,000 docs) passed: all tokenizer checks OK, 2.18M tokens, samples clean. 4/4 tests pass. Speed ~0.1M tok/s per CPU core, so the full run takes a few hours depending on Colab's cores.
- Next: run the notebook on Colab; gate = ~10B train tokens + samples read clean.
- Notes for the other person: two new proposals in DECISIONS.md (Colab for data, H100 provider postponed); please agree or object.

### 2026-10-23 · Adi · Colab setup
- Did: added `notebooks/colab_setup.ipynb` (GPU/bf16 check, keys from Colab Secrets, Drive storage, pulls latest code, runs setup check, pushes notes) and `docs/COLAB.md` (incl. optional Colab MCP for Claude on your Mac).
- Next: each of us runs the notebook once; partner reviews the Colab decision.

### 2026-10-23 · Adi · Session 1 (part)
- Did: model config (`chotu/config.py`), size test, pinned requirements, `scripts/check_setup.py`, draft prompts, `.env.example`.
- Result: 180,925,056 params (embeddings 31.46M shared, 32 layers × 4.67M); tiny rehearsal model 7.08M. HF user magnateadi (fine-grained token, read + write own repos); W&B entity adityaagrawal-student.
- Next: human steps under "Next up", then Session 2.
- Notes for the other person: keys live only in each person's local `.env`; copy `.env.example` and use your own keys. Use W&B entity adityaagrawal-student so runs land in one place (ask Adi to invite you).

### 2026-10-23 · Adi · Session 0
- Did: created repo, CLAUDE.md, shared-context docs and Claude Project setup.
- Next: Session 1.
