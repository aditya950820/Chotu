# Chat and tool-call tuning (chain step 2)

The 8k/tool-calling recipe remains **proposed**, pending partner agreement.
Architecture, tokenizer and parameter count are unchanged. CPU smoke checks
the pipeline, not model quality or the proposed BFCL target.

## Preparation

```sh
source ~/chotu-venv/bin/activate
python scripts/prepare_chat.py --storage /persistent/chotu
```

Read all train rows of `HuggingFaceTB/smol-smoltalk` and
`HuggingFaceTB/smoltalk`, config `apigen-80k`, using streaming `datasets==5.2.0`.
Resolve and record each repository revision before reading. Network/cache
files go under storage. `--limit N` reads N rows **per source** for rehearsal;
use a different storage directory for a full run. Completed reruns skip via
`chat/.done`; a changed limit requires fresh storage.

Real APIGen schema is `messages: list<{role: string, content: string}>`.
Tools are JSON inside the system turn's `<tools>...</tools>`; the assistant
puts a JSON **list** inside a single `<tool_call>...</tool_call>` block.
The inspected 83,144-row train source has only system/user/assistant roles,
no tool results. Normalize tools through the renderer's `tools` argument,
replace the incompatible list-format system instruction, and emit one
`{name, arguments}` tag per call. Preserve refusals/empty call lists; never
fabricate results. Actual tool turns are preserved and masked if encountered.
Drop/count malformed or unknown-tool calls, invalid messages, duplicates,
missing assistant targets and conversations exceeding 8,192 tokens.

Hold out a deterministic ~1% by SHA-256 of canonical messages plus tools.
Deduplicate across sources. No oversampling: use natural source proportions,
reporting both conversation share and assistant-token share. The limited-run
mix is **not** the full-run mix (it takes equal counts per source).

Storage format:
- `chat/{train,val}.ids`: flat uint16 token IDs.
- `chat/{train,val}.mask`: uint8 assistant-loss mask, including assistant end.
- `chat/{train,val}.offsets`: int64 boundaries, starts at zero, ends at ID count.
- `chat/apigen-heldout.jsonl`: original raw row, normalized prompt messages,
  tools, gold calls and any gold refusal text; reserved for subsequent eval.
- `reports/chat_report.md`: conversion/drop counts, percentiles, source shares,
  five rendered marked-loss samples per source, roundtrip validity and timings.
- `chat/.done`: metadata and recorded dataset revisions.

Temporary arrays are renamed before the completion marker. Interrupted
preparation starts over; it does not consume partially prepared arrays.

## Training

```sh
python scripts/train.py --stage chat --storage /persistent/chotu \
  --init /persistent/chotu/runs/pretrain/final --name chat
```

`--init` also accepts a resumable pretraining `.pt` checkpoint. Always rebuild
RoPE at theta 500,000 with 8,192 capacity. Train at most `--epochs 2` (default
two). Sort by length, form padded microbatches, shuffle microbatch order using
seed + epoch, and accumulate up to the token budget without splitting or
packing conversations. Padding and system/user/tool tokens get no loss.
The stored labels are shifted once at batching; assistant end is retained.
Accumulate by supervised-token count, not by the number of padded sequences.

Defaults: peak LR **3e-4**, **20** warmup updates then linear decay to zero;
**32,768 padded tokens/update**, **8,192 padded tokens/microbatch**. Updates
can be smaller at bucket/epoch boundaries. This conservative initial LR is
for SFT, not the pretraining LR test; quality needs held-out chat/tool evals.
Set warmup shorter for tiny runs. `--max-steps` caps a rehearsal's schedule,
so its checkpoints cannot resume into a different cap/epoch schedule.

Evaluate every `--eval-every` updates, at every epoch end and final update,
on **all** held-out examples in padded batches. Report assistant-token mean
CE, not BPB. Publish improved models as `best/` (complete HF model + original
tokenizer), using an atomic relative symlink under `hf/`; copy with symlink
dereferencing or retain `hf/`. Keep the last two atomic optimizer/RNG
checkpoints and write `done.json` on completion.

Resume, nonfinite loss/gradient stop, optional W&B, device/compile/fused CE
and standalone shutdown share the pretraining code. Pretraining-only
loss/throughput/spike gates are not applied to SFT; nonfinite checks always
apply. `--no-auto-stop` transfers CUDA shutdown responsibility to the
orchestrator. See `PRETRAINING.md` for exit codes and safety instructions.

## CPU rehearsal

```sh
python scripts/prepare_chat.py --storage /home/ubuntu/chotu-chat-smoke --limit 2000
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/train.py --stage chat \
  --model tiny --storage /home/ubuntu/chotu-chat-smoke --name chat-resume-40 \
  --init /home/ubuntu/chotu-smoke/runs/private-resume-60/checkpoint-00000058.pt \
  --max-steps 40 --batch-tokens 8192 --micro-tokens 8192 --warmup-steps 3 \
  --eval-every 10 --ckpt-every 5 --device cpu --no-auto-stop
```

SIGKILL after update 22, rerun exactly, and check resume at checkpoint 20 and
identical replay losses at 21–22. Verify `AutoModelForCausalLM.from_pretrained`
can load `runs/chat-resume-40/best`. Full-source row counts, preparation time
estimates and results are recorded in `PROGRESS.md`; full preparation and
CUDA/Jarvis/W&B still need the real-run rehearsal.
