# Decision log

Every change to a frozen decision (see CLAUDE.md) gets an entry here first, agreed by both people.
Newest at the top. Format: date · who · decision · why · what it replaces.

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
