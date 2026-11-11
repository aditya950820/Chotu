# Progress and handover

The single place that says where the project stands. Read it before starting; update it before you stop.
Newest entries go at the top of the log. Never rewrite someone else's entry; add a new one.

## Now

- **Current session:** Session 3 · Twin day rehearsal done on a Jarvis H100 (branch `twin-day`); 181M speed test still to rerun
- **Who's on what:**
  - Adi: next H100 session: `ONLY="speed main5x" bash scripts/twin_day.sh` (181M speed test + 5x twin run, then the machine pauses itself)
  - Partner: review `reports/twin_day.md` and `train.py` on branch `twin-day`
- **Blocked on:** nothing
- **Paid resources running:**
  - Jarvis Labs H100 80GB at 151.185.52.36, started 2026-11-07 (₹330/h). **Stop it after the speed test.**
  - Jarvis Labs instance from 2026-10-24 (GPU type: TODO). Check it is paused or deleted.

## Next up

1. Session 1, remaining human steps:
   - [x] Hugging Face token and W&B key work (checked by `scripts/check_setup.py`)
   - [x] Model config + size test: 180,925,056 params, tests pass
   - [ ] Both: rewrite `prompts/test_prompts.md` in your own words
   - [ ] Provider: Jarvis Labs proposed (`docs/JARVIS.md`); run its first-time setup and `stop_machine.py --dry-run`
   - [ ] Partner: get collaborator access, clone, make own `.env`, run `python scripts/check_setup.py`
   - [ ] Partner: set up Colab per `docs/COLAB.md` and run `notebooks/colab_setup.ipynb` (Adi: done, 4/4 checks OK)
   - [ ] Partner: agree or object to the Colab proposal in `docs/DECISIONS.md`
2. Session 2 gate: both read the 10 samples in `reports/data_report.md` (in the private HF dataset `magnateadi/chotu-fineweb-edu-10bt-tokens`). Don't start another data run; the shards are done. Then Session 3.

## Open questions

- (add questions for the other person here, with your name)

## Log

<!-- Template:
### YYYY-MM-DD · Name · Session N
- Did:
- Result / numbers:
- Next:
- Notes for the other person:
-->

### 2026-11-11 · Adi · Session 3: 5x twin run prepared
- Did: `scripts/twin_day.sh` gains an opt-in step `main5x`: the same 7M twin reads 700M tokens (5x the 140M run, ~10,680 steps) with the same recipe and the same locked exam, to test whether more reading improves it. It reads from 5 train shards (5x the text, so repeats stay at the same share as the 140M run); the 4 extra shards download in the background during the speed test, so the GPU never waits. The final checkpoint and logs then go to HF `magnateadi/chotu-twin-day` (`twin-day/main-5x/`) before the machine pauses. `scripts/twin_report.py` adds a 140M vs 700M table (falls back to the 140M logs in git on a fresh machine).
- Tested: CPU smoke run on fake shards (train.py → report table works), bash syntax, size test passes. Not yet run on a GPU.
- Command on the H100 (repo root, branch twin-day, `.env` with HF_TOKEN, GH_TOKEN, JL_API_KEY, JL_MACHINE_ID): `ONLY="speed main5x" bash scripts/twin_day.sh 2>&1 | tee ~/twin_day_5x.log`
- Estimate: speed test ~2 min, 5x run ~15 min at ~909k tok/s, about ₹100–150 in total including setup.
- Notes for the other person: the comparison that counts is held-out bits per byte (140M run: 1.3881). The 5x run's score at step ~2,136 is not like-for-like (no slow-down yet).

### 2026-11-07 · Adi · Session 3: twin day files synced
- Did: all 10 twin-day checkpoints (85 MB each) uploaded to the private HF repo `magnateadi/chotu-twin-day` (covers the "upload the twin checkpoint" next step above). Added the full `logs/twin-day/twin_day.log`, the H100 package list `logs/twin-day/pip-freeze-h100.txt`, `scripts/setup_gpu.sh` (builds a Python 3.12 venv with CUDA torch on a bare VM) and `scripts/talk.py` (`python scripts/talk.py "Once upon a time"` fetches the final twin checkpoint from HF and continues the text).
- Notes for the other person: the VM pauses itself only if `JL_API_KEY` and `JL_MACHINE_ID` are in `.env`.

### 2026-11-07 · Adi · Session 3: twin day (rehearsal on a rented H100)
- Did: `train.py` (one training script for the twin and Chotu: frozen recipe, bf16, SDPA, compile, fused AdamW, checkpoints via temp file + swap, resume, exam + bits per byte, samples, copying test, kill-rule alerts, MFU and ₹ per step) and `scripts/twin_day.sh` (setup → data → overfit → fire drill → main twin run → 181M speed test → report → push). Logs went to GitHub live (`live-runs`, every 30 s); report and logs on `twin-day` (`reports/twin_day.md`, `logs/twin-day/`). Shared dashboard "Chotu Twin Day" follows them; its source is in `dashboard/`.
- Results (Jarvis H100 80GB, ₹330/h): setup 9 s; data 45 s; overfit 10.9 → 1.33 in 150 steps (loop learns; the < 1.0 gate was too strict for 7M); **fire drill passed** (resume matched the uninterrupted run); **main twin run** 140M tokens in 3.0 min at ~909k tok/s, first loss ~10.9, below 6 by step 300, exam 1.39 bits per byte; samples grammatical but repetitive; copying test weak (14.3 → 13.7, expected at 7M); MFU alerts 4–5% (expected for the twin). Untrained 181M scored 10.94 on the exam (matches 10.91 from the Guessing Game).
- Problems fixed: first two attempts used `python3` (no CUDA torch on Jarvis); now uses the machine's `python`. 181M speed test ran out of memory at 32 windows per pass; default is now 8 (same 524,288-token batch, more rounds). `ONLY="speed"` reruns single steps.
- Next: rerun the speed test (projected full-run hours and ₹); upload the twin checkpoint to a private HF repo (`magnateadi/chotu-twin`); stop the machine; GPT-2 124M and SmolLM2-135M bits-per-byte baselines; Session 4; get IndiaAI portal access (H200 ₹140/h vs Jarvis H100 ₹330/h) for the real run.
- Notes for the other person: `train.py` applies weight decay 0.1 to 2-D weights only (norms excluded), the usual practice; say if you'd rather decay everything. Checkpoints and the model are not in git (they stay on the machine / HF). The flight-school cheat sheet is now in `docs/FLIGHT_SCHOOL.md`; GPU cost estimates are in `dashboard/training_cost/`.


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
