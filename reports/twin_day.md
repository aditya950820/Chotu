# Twin day: Session 3 rehearsal

| Step | Result | Note |
| --- | --- | --- |
| setup | pass | NVIDIA H100 80GB HBM3, 81559 MiB; 2.14.1+cu130 cuda True transformers 5.20.2; size test 180,925,056 OK |
| data | pass | train_000.bin val_000.bin  |
| overfit | fail | final loss 1.3319 (should be < 1.0) |
| firedrill | pass | resumed run matches, max loss gap 0.0463 |
| main | pass | val 1.3881 bpb, 909,337 tok/s, 2.97 min |
| speed | fail | speed test did not finish |
| end | info | script finished |

## Main twin run

- 139,984,896 tokens in 2.97 min on NVIDIA H100 80GB HBM3, about ₹16.3
- Final training loss 4.3197, held-out 1.3881 bits per byte
- Speed 909,337 tokens/s, MFU 5.1%
- Copying test: surprise 14.32 on the first copy, 13.74 on the repeat
- Sample at step 2136: *The capital of France is* → ' the capital of the country. The capital of the country is the capital of the country. The capital of the country is the capital of the country. The capital of the country is the capital of the'
- Sample at step 2136: *Photosynthesis is the process by which* → ' the people are able to do it.\nThe first thing is that the people are able to do it.\nThe first thing is that the people are able to do it.\nThe most important'
- Sample at step 2136: *Once upon a time, there was a* → ' lot of people who had been doing to do with the people.\nThe people who were not doing their work were not doing their work. They were not doing their work. They were not doing their'

## Alerts

- step 20 [mfu]: MFU 4.1% is under 10% (expected for the 7M twin: too small to keep a big GPU busy; the rule is for the 181M run)
- step 20 [mfu]: MFU 4.1% is under 10% (expected for the 7M twin: too small to keep a big GPU busy; the rule is for the 181M run)
- step 20 [mfu]: MFU 4.1% is under 10% (expected for the 7M twin: too small to keep a big GPU busy; the rule is for the 181M run)
- step 20 [mfu]: MFU 5.1% is under 10% (expected for the 7M twin: too small to keep a big GPU busy; the rule is for the 181M run)
