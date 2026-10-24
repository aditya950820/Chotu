# Progress and handover

The single place that says where the project stands. Read it before starting; update it before you stop.
Newest entries go at the top of the log. Never rewrite someone else's entry; add a new one.

## Now

- **Current session:** Session 2 · Data, ready to run on Colab
- **Who's on what:**
  - Adi: —
  - Partner: —
- **Blocked on:** nothing
- **Paid resources running:** none (list every GPU instance and volume here while it exists)

## Next up

1. Session 1, remaining human steps:
   - [x] Hugging Face token and W&B key work (checked by `scripts/check_setup.py`)
   - [x] Model config + size test: 180,925,056 params, tests pass
   - [ ] Both: rewrite `prompts/test_prompts.md` in your own words
   - [ ] (Postponed to before Session 5) pick the H100 provider and volume; see DECISIONS.md
   - [ ] Partner: get collaborator access, clone, make own `.env`, run `python scripts/check_setup.py`
   - [ ] Partner: set up Colab per `docs/COLAB.md` and run `notebooks/colab_setup.ipynb` (Adi: done, 4/4 checks OK)
   - [ ] Partner: agree or object to the Colab proposal in `docs/DECISIONS.md`
2. Session 2: run on your Mac (`bash scripts/setup_mac.sh`, then `notebooks/session2_data_local.ipynb`, ~25 GB free disk) or on Colab (`notebooks/session2_data.ipynb`, ~22 GB free Drive). Read the report.

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
