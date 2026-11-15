# Twin 140M vs twin 700M: what 5x more reading changed

Page: "Five Times the Reading" (claude.ai artifact, ask Adi to share). Rebuild: `python learning/twin_5x/analyze.py --dir <folder>` then `python learning/twin_5x/build.py <out.html>`.
Data: 96 locked windows of 2,048 tokens (196,608 tokens) from `val_000`; common-word counts from `val_001`. Checkpoints from HF `magnateadi/chotu-twin-day` (`twin-day/main` step 2,136 and `twin-day/main-5x` step 10,681). Both runs started from the same random weights (seed 1234) and used the same recipe; only the tokens read changed.

## Verdict

Yes, it improved, but only a little, and in a predictable way: **1.374 → 1.313 bits per byte (−4.4%)**, top guess right 29.1% → 30.7%. For comparison, random → 140M cut the score by 2.07; 140M → 700M cut it by 0.06.

| Player | Surprise | Bits per byte | Top guess right |
| --- | --- | --- | --- |
| Untrained twin | 10.83 | 3.442 | 0.1% |
| Common words only | 7.60 | 2.414 | – |
| Twin, 140M tokens | 4.32 | 1.374 | 29.1% |
| Twin, 700M tokens | 4.13 | 1.313 | 30.7% |

## Where the gain came from

- **Rare words gained most:** surprise on tokens rarer than the top 10k fell by 0.43, on the 10 most common tokens by only 0.07. Insides of words gained 0.27, word starts 0.21, digits only 0.04. The extra reading mostly taught the twin more about individual words.
- **Context use barely changed:** both twins drop fast over the first ~60 tokens of context, then flatten. The gain is about the same (0.16–0.20) at every position from 4 onwards, so the 700M twin does not use far context any better.
- **Broad, not focused:** 49% of tokens got better by >0.1, 34% got worse, 17% about the same. Average gain 0.19. Biggest gains on everyday content words (" light", " research", " students", " energy"); losses on formatting (bullets, broken characters).
- **When:** at the 140M mark the longer run was *behind* (1.434 vs 1.388) because it was still taking full-size steps. It then sat almost flat (1.419 → 1.393 from step 3,000 to 8,500) and got most of its lead in the final 20% slow-down (1.393 → 1.329).

## What did not improve

- **Copying test:** a random 128-token string shown twice. Both twins are worse than blind guessing (10.80) on both halves: 140M 14.32 → 13.74, 700M 13.57 → 13.32. No copying skill appeared.
- **Looping:** taking the top guess every time, 70% (140M) vs 69% (700M) of 4-token phrases repeat earlier ones. Samples at temperature 0.8 don't loop but ramble.
- **Confidence:** both are well calibrated (when they say 90%, they are right ~90%); the 700M twin is just confident more often.

## Why only this much

1. **The twin is mostly a word list.** 89% of its 7.08M knobs are embeddings; only 788k are thinking layers. More reading improved its knowledge of words, but the thinking part is already near what it can hold.
2. **Far past the sweet spot.** The usual rule of thumb is ~20 tokens per knob. The 140M run was already there (20); the 700M run is at 99, and at 888 per thinking knob. Each extra token teaches less.
3. **Most of the visible gain needs the slow-down.** At full stride the weights bounce around a good region; shrinking the stride lets them settle.
4. **New skills need size.** Copying ("look back and repeat") needs enough attention layers to form. More reading can't create it in 4 tiny layers.
5. **A different solution, not just further along.** From the same start, the two runs travelled about the same distance (0.98×) in nearly unrelated directions (cosine 0.11). The thinking layers moved 31% further in the 700M run.

## What this means for the 181M run

- Chotu is the opposite shape: 83% thinking layers (149M of 181M) and ~56 tokens per knob over 10B tokens. Expect much larger gains than the twin shows.
- A long flat middle at full stride is normal; the drop comes in the last 20%. Don't kill the run for a flat middle.
- Watch the copying test: on 181M it should fall well below blind guessing. If it stays above, alert.
- Plan sampling (temperature) and a repetition penalty for the chat model and the GGUF file.
