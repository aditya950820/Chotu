# Progress and handover

The single place that says where the project stands. Read it before starting; update it before you stop.
Newest entries go at the top of the log. Never rewrite someone else's entry; add a new one.

## Now

- **Current session:** Session 1 · Set up (code done; human steps left)
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
   - [ ] Pick a GPU provider; load prepaid credit / set a spend limit
   - [ ] Create a 100 GB persistent volume in a region with on-demand H100s in stock
   - [ ] Partner: get collaborator access, clone, make own `.env`, run `python scripts/check_setup.py`
   - [ ] Partner: set up Colab per (Adi done 2026-10-23: 4/4 checks OK) `docs/COLAB.md` (Secrets, L4/A100 runtime, run `notebooks/colab_setup.ipynb`)
   - [ ] Partner: agree or object to the Colab proposal in `docs/DECISIONS.md`
2. Session 2: data (FineWeb-Edu sample-10BT → shards).

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
