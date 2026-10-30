# What we learned

Shared notes so both of us learn from each other's sessions. After each Socratic session with Claude,
add 3–5 lines: the question, the answer in your own words, and anything still unclear.
Newest at the top.

<!-- Template:
### YYYY-MM-DD · Name · Topic
- Question:
- In my words:
- Still unclear:
-->

### 2026-10-30 · Adi · Lesson 3: nudge size (learning rate), with Claude
- Lab: 8 real CPU runs of the 7M tiny twin, 250 steps each, same batches, only the stride changed (`learning/nudge_size/`, page "Nudge Size Lab").
- Exam after 250 steps: 3e-4 7.00 · 1e-3 6.60 · **3e-3 6.56** · 1e-2 6.82 · 3e-2 6.99 · 1e-1 7.21 (a U-shape). 1e-1 never hit NaN; it showed gradients clipping in 114/225 steps and a mid-run exam bump.
- Surprise: skipping warm-up didn't hurt the tiny model (6.81 vs 6.99). Slowing down over the last 20% helped a little (6.52 vs 6.56).
- Open questions: what warns you a stride is too big before NaN? Should warm-up stay in Chotu's plan? Why test only 1.5e-3 vs 3e-3?
- Still unclear:

### 2026-10-28 · Adi · Pretraining flight school, lessons 1–2 (with Claude)
- Course: 8 lessons, each ending in a run-day skill (loss, curve shape, learning rate, batch, train vs val, speed/MFU, crashes, X-ray). Visuals: "Pretraining by the Numbers" and "The Guessing Game" artifacts (game code in `learning/guessing_game/`).
- Real landmarks on 16,384 held-out tokens: untrained Chotu 10.91 (blind guessing is 10.80), common-words-only 7.53, SmolLM2-135M 2.77.
- Open questions (answer in your own words): Why does knowing only which tokens are common cut surprise from 10.9 to 7.5? Should a nearly flat loss in the long middle phase worry us? 64 vs 256 windows per nudge: what do you gain and lose?
- Still unclear:
