# Decision log

Every change to a frozen decision (see CLAUDE.md) gets an entry here first, agreed by both people.
Newest at the top. Format: date · who · decision · why · what it replaces.

### 2026-10-23 · Adi · PROPOSED: rehearsals on Google Colab
- Decision: Sessions 3–4 run on Colab (L4/A100) instead of a rented cheap GPU; Session 2 data and Session 5 H100 run stay on rented machines. Storage for Colab work: Google Drive `/MyDrive/chotu`.
- Why: Adi already uses Colab; no extra provider for rehearsals. Colab can't host the long unattended run (time limits, disconnects).
- Note: the auto-stop and checkpoints-on-the-volume tests must still be rehearsed once on the rented provider before Session 5.
- Status: needs partner's agreement.

### 2026-10-23 · Adi + partner · Plan frozen
- Decision: architecture, tokenizer, data, recipe, success bar as listed in CLAUDE.md.
- Why: validated against sources and an independent review; reduced to 7 sessions for limited time.
- Replaces: —
