# A language model with no gradient descent

The question: how good can a next-token predictor get if nothing is trained by gradient descent? Everything here is counting, closed-form smoothing, a few values picked by trying them, and EM. It is scored on the same locked exam as the twins (96 windows × 2,048 tokens from `val_000`), in bits per byte.

## Parts

| Part | Idea | Source |
| --- | --- | --- |
| Kneser-Ney n-grams, orders 1–6 | Count what followed the same last 1–5 tokens in the training text; subtract a small discount from every count and give it to shorter contexts; for shorter contexts, count *how many different words came before*, not how often (so "Francisco" is rare after anything but "San") | Kneser & Ney 1995; Chen & Goodman 1998 (modified KN, three discounts per order) |
| Document memory (cache) | Words already used in this document are likely to come back | Kuhn & De Mori 1990 |
| Copy model | Find the longest stretch of recent tokens that already appeared earlier in this document and predict what followed it then. An exact, count-based version of what an "induction head" learns | cache LMs; "infini-gram" longest-match idea, Liu et al. 2024 |
| Two-word memory | How often this token followed the previous token earlier in the document | Goodman 2001 (cache variants) |
| Skip-grams (tried, no gain) | Contexts that skip the token just before | Goodman 2001 |
| Mixing by EM | Combine the parts with weights per situation (repeat length, n-gram depth, position); weights = average responsibility, iterated | Jelinek & Mercer 1980; Dempster, Laird & Rubin 1977 |

None of these computes a gradient. The discounts start from Chen & Goodman's closed-form estimate and are then nudged by trying ±10–20% on the tuning set.

## Fair play

- Tuning (discounts, mixing weights) uses 16 windows from `val_001`. The exam (`val_000`) is only scored, never tuned on.
- Like the twins, the model sees only the current 2,048-token window as context.
- Counts are stored only for contexts that occur at tuning and exam positions. The model is still defined on the whole training text; this just skips storing counts nobody asks for, and changes no score.
- `test_kn.py` checks the fast numpy version against a slow dictionary version (same probabilities to 1e-16) and that probabilities over the 49,152 tokens add up to 1. Splitting passes by target token (`--split`) and counting one order per pass (`--low-memory`) give identical probabilities (checked).

## Run

```bash
python learning/no_gradient/test_kn.py <folder with shards/>
python learning/no_gradient/kn.py --dir <folder> --train train_000 --low-memory --out run.json
python learning/no_gradient/mix.py run.parts.npz --out mix.json
```

One 760M-token shard takes ~30 min on 4 CPU cores and peaks at ~6 GB; add `--split 2` per extra shard or two.
