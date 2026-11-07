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

### 2026-11-07 · Adi · Lessons 5–8 and twin day, with Claude
- Lessons 5–8: practice paper vs exam and bits per byte; speed, MFU and the first 10 minutes; checkpoints and the fire drill; judging what the knobs learned (samples, copying test, word map, knob health). Everything is collected in `docs/FLIGHT_SCHOOL.md` (cheat sheet, checklist, glossary).
- Seen for real today: the cliff (loss below 6 well before step 300), exam dots falling with practice, a resumed run lying on top of the uninterrupted one, MFU of 4–5% on a model too small for an H100, and a twin that writes grammatical but repetitive English.
- Open questions: why does the 7M twin fail the copying test while bigger models pass it? Why did 32 windows per pass run out of memory for 181M but not for 7M? Why can a 7M model read 140M tokens in 3 minutes while 181M needs 15–25 hours for 10B?
- Still unclear:

### 2026-10-31 · Adi · Lesson 4: batch size (ask the crowd), with Claude
- Analogy: each window is a stranger you ask for directions; the batch is the crowd you ask before each stride; the reading budget is fixed, so a bigger crowd means fewer strides.
- Compasses (real gradients of the tiny twin): typical miss vs a 768-window reference. 1 window 75°, 8 windows 53°, 32 windows 33°, 256 windows 15° at the start. After 256 strides, small crowds disagree more (1 window 86°, 32 windows 41°).
- Race (same 2,048 windows each, LR 3e-3): exam 2/stride 6.61 · **8/stride 6.56** · 32/stride 6.95 · 128/stride 7.52 (only 16 strides).
- Open questions: why do small crowds disagree more after training? Who wins a 10× longer race? Why can Chotu afford 256? Twice-as-big GPU: change the crowd or the rounds?
- Still unclear:

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
