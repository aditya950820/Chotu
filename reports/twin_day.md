# Twin day: Session 3 rehearsal

| Step | Result | Note |
| --- | --- | --- |
| setup | pass | NVIDIA H200, 143771 MiB; 2.14.1+cu130 cuda True transformers 5.20.2; size test 180,925,056 OK |
| data | info | downloading 4 more train shards in the background (for main5x) |
| speed | pass | 7,718 tok/s, MFU 1.2%, full run 362.1 h ≈ ₹157,500 |
| main5x | pass | val 1.3292 bpb (140M run: 1.3881), 1,153,631 tok/s, 10.81 min |
| upload | pass | twin-day/main-5x on HF |
| end | info | script finished |

## Main twin run

- 139,984,896 tokens in 2.97 min on NVIDIA H100 80GB HBM3, about ₹16.3
- Final training loss 4.3197, held-out 1.3881 bits per byte
- Speed 909,337 tokens/s, MFU 5.1%
- Copying test: surprise 14.32 on the first copy, 13.74 on the repeat
- Sample at step 2136: *The capital of France is* → ' the capital of the country. The capital of the country is the capital of the country. The capital of the country is the capital of the country. The capital of the country is the capital of the'
- Sample at step 2136: *Photosynthesis is the process by which* → ' the people are able to do it.\nThe first thing is that the people are able to do it.\nThe first thing is that the people are able to do it.\nThe most important'
- Sample at step 2136: *Once upon a time, there was a* → ' lot of people who had been doing to do with the people.\nThe people who were not doing their work were not doing their work. They were not doing their work. They were not doing their'

## 5x twin run: does reading 5x more help?

Same 7M model, recipe and exam (64 locked held-out windows); only the amount of text read changes. Lower bits per byte is better.

| | 140M run | 700M run (5x) |
| --- | --- | --- |
| Tokens read | 139,984,896 | 699,990,016 |
| Steps | 2,136 | 10,681 |
| Held-out bits per byte | 1.3881 | 1.3292 |
| Final training loss | 4.3197 | 4.1749 |
| Copying test, repeat surprise | 13.74 | 13.32 |
| Minutes | 2.97 | 10.81 |
| Cost ₹ | 16.3 | 78.4 |

Held-out bits per byte changed by -4.2% with 5x the tokens.
At step 2250 (about the 140M mark, before its slow-down) the 5x run scored 1.4344 bits per byte.
- Sample at step 10681: *The capital of France is* → ' the capital of the world.\nThe capital of France is the capital of France. The capital of France is the capital of France. The capital of France is the capital of France. The capital of'
- Sample at step 10681: *Photosynthesis is the process by which* → ' the plant is produced.\nThe plant is a plant that is produced by the plant. The plant is a plant that is produced by the plant. The plant is a plant that is produced by the'
- Sample at step 10681: *Once upon a time, there was a* → ' great deal of time to be done.\nThe first time the first time was to be done, and the second time was to be done. The second time was to be done, and the second'

## Speed test, real 181M model

- 7,718 tokens/s on NVIDIA H200, MFU 1.2%
- Full 10.06B-token run at this speed: 362.1 hours, about ₹157,500 at this machine's price

## Alerts

- step 20 [mfu]: MFU 4.1% is under 10% (expected for the 7M twin: too small to keep a big GPU busy; the rule is for the 181M run)
- step 20 [mfu]: MFU 4.1% is under 10% (expected for the 7M twin: too small to keep a big GPU busy; the rule is for the 181M run)
- step 20 [mfu]: MFU 4.1% is under 10% (expected for the 7M twin: too small to keep a big GPU busy; the rule is for the 181M run)
- step 20 [mfu]: MFU 5.1% is under 10% (expected for the 7M twin: too small to keep a big GPU busy; the rule is for the 181M run)
- step 20 [mfu]: MFU 6.4% is under 10% (expected for the 7M twin: too small to keep a big GPU busy; the rule is for the 181M run)
- step 1 [nan]: loss or gradient is not a number: stop at once
