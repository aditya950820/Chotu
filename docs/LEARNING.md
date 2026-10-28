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

### 2026-10-28 · Adi · Pretraining flight school, lessons 1–2 (with Claude)
- Course: 8 lessons, each ending in a run-day skill (loss, curve shape, learning rate, batch, train vs val, speed/MFU, crashes, X-ray). Visuals: "Pretraining by the Numbers" and "The Guessing Game" artifacts (game code in `learning/guessing_game/`).
- Real landmarks on 16,384 held-out tokens: untrained Chotu 10.91 (blind guessing is 10.80), common-words-only 7.53, SmolLM2-135M 2.77.
- Open questions (answer in your own words): Why does knowing only which tokens are common cut surprise from 10.9 to 7.5? Should a nearly flat loss in the long middle phase worry us? 64 vs 256 windows per nudge: what do you gain and lose?
- Still unclear:
