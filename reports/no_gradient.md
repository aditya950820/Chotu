# Better models without gradient descent: what worked and what didn't

All scores are bits per byte on the locked exam (96 windows × 2,048 tokens from `val_000`, 196,608 tokens); lower is better. Tuning used `val_001` only. No step in this report computes a gradient. Code: `learning/no_gradient/`, `learning/weight_edit/`.

| Model | Built by | Exam |
| --- | --- | --- |
| Twin 7M, 140M tokens | gradient descent | 1.3739 |
| Twin 7M, 700M tokens | gradient descent | 1.3129 |
| Twin 700M + best hand edit (blend, stretch, swap, confidence, simplify, volume) | weight editing | 1.3127 |
| Counting model, 50M tokens | counting + EM | 1.5238 |
| Counting model, 760M tokens (1 shard) | counting + EM | 1.3616 |
| **Counting model, 2.28B tokens (3 shards)** | **counting + EM** | **1.2721** |

## 1. Editing the trained twin by hand (`learning/weight_edit/`)
- Blending the 140M and 700M twins breaks them (50/50 mix: 2.39 on tuning text): the two runs found different, incompatible solutions (their weight changes from the shared start are almost unrelated, cosine 0.11).
- Swapping word list and layers between twins: 8.62 and 2.99. Each twin's parts only work together.
- Stretching, simplifying matrices (SVD), per-layer volume knobs: nothing kept. Confidence dial: 1.3129 → 1.3127.
- Removing the "repetition direction" from MLP weights (the method reported for Qwen3.5-0.8B, Nov 2026): every effective edit made the twin loop *more* (68% → up to 84% repeated phrases) and hurt the exam. The Qwen edit unlocked answers the model already knew; our 7M twin loops because it has no better continuation. Edits can unlock skills, not add them. Looping is fixed at decoding time instead (repetition penalty, see "Talk to the Twins").

## 2. A counting model (`learning/no_gradient/`)
Interpolated modified Kneser-Ney n-grams (orders 1–6) + document memory + copy model + two-word memory, mixed by EM per situation. Checked against a slow exact reference (differences ~1e-16; probabilities sum to 1).
- Data is the lever: 50M → 760M → 2.28B tokens gave 1.524 → 1.362 → 1.272.
- Document memory and copying add ~0.08 on top of counts (copying fixes what the twin can't do: repeats of 4+ tokens score 0.81).
- Tried with little or no gain: skip-grams (none), token classes from PPMI + SVD + k-means (−0.003 at 50M; the classes themselves are sensible: cities, colours, months, numbers), recency-weighted memory (none).
- Where it wins and loses vs the twin: it wins when the last 5 tokens were seen in training (1.14 vs 1.20) and on rare words; it loses on new word combinations and on common grammar words (the twin generalises, counts don't).

## What this does not give
Good *output*. These models predict the next token well on average but cannot keep a thread, follow instructions or reason. Gemma-level output needs a large neural model trained by gradient descent on trillions of tokens; no known method avoids that (the literature on hand-written weights, e.g. Tracr, covers small algorithms only). The counting model's value: a strong, cheap baseline the real 181M Chotu must clearly beat, and a lesson in how much of language modelling is memory.
