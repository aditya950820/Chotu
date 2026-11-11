#!/usr/bin/env bash
# Twin day (Session 3 rehearsal): the whole pretraining chain on the 7M twin, then a speed test of the real
# 181M model, on one rented GPU. Everything goes to GitHub as it happens:
#   branch live-runs : run logs and step status, refreshed every 30 s (the shared dashboard follows these)
#   branch twin-day  : the report and the logs, pushed at the end
# Data and checkpoints stay on the machine's volume, never in git. Ends by pausing the machine, even after a failure
# (Jarvis Labs, needs JL_API_KEY and JL_MACHINE_ID; set KEEP_MACHINE=1 to skip).
#
# On the GPU machine, from the repo root (branch twin-day):
#   export HF_TOKEN=hf_...  GH_TOKEN=github_pat_...  GPU_PRICE=330      # or put them in .env
#   bash scripts/twin_day.sh 2>&1 | tee ~/twin_day.log
#   ONLY="speed" bash scripts/twin_day.sh ...    # setup, data, then only the listed steps (overfit firedrill main speed)
#
# 5x twin run (opt-in, only when listed in ONLY): does reading 5x more tokens (700M) make the 7M twin better?
#   ONLY="speed main5x" bash scripts/twin_day.sh 2>&1 | tee ~/twin_day_5x.log
# It reads from 5 train shards instead of 1 (5x the tokens over 5x the text, so the share of repeated text matches
# the 140M run). The 4 extra shards download in the background while the speed test runs, so the GPU never waits.
# The final checkpoint goes to the private HF repo magnateadi/chotu-twin-day before the machine pauses.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
[ -f .env ] && set -a && . ./.env && set +a
DATA=${CHOTU_DATA:-$HOME/chotu-data}
RUNS=${CHOTU_RUNS:-$HOME/chotu-runs/twin-day}
LIVE=$HOME/chotu-live
PRICE=${GPU_PRICE:-330}
STATUS=$RUNS/status.jsonl
PY=$(command -v python || command -v python3)   # Jarvis images keep CUDA torch under `python`
NAME="aditya950820"; EMAIL="202965202+aditya950820@users.noreply.github.com"
mkdir -p "$DATA/shards" "$RUNS"
[ -n "${ONLY:-}" ] || : > "$STATUS"   # a fresh status list, unless re-running single steps
want() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }   # ONLY="speed" re-runs just those steps
asked() { [[ " ${ONLY:-} " == *" $1 "* ]]; }                    # opt-in steps run only when listed in ONLY
TRAIN_SHARDS=1; asked main5x && TRAIN_SHARDS=${TRAIN_SHARDS_5X:-5}

status() {  # id, status (running/pass/fail/info), note
  $PY -c 'import json,sys,time; print(json.dumps({"id":sys.argv[1],"status":sys.argv[2],"note":sys.argv[3],"time":round(time.time())}))' "$1" "$2" "${3:-}" >> "$STATUS"
  echo ">>> [$1] $2 ${3:-}"
}

# ---------- GitHub: commits come from Adi's account ----------
git config user.name "$NAME"; git config user.email "$EMAIL"
if [ -n "${GH_TOKEN:-}" ]; then
  git config --global credential.helper store
  printf 'https://%s:%s@github.com\n' "$NAME" "$GH_TOKEN" > ~/.git-credentials && chmod 600 ~/.git-credentials
fi

push_live() {
  if [ ! -d "$LIVE/.git" ]; then
    mkdir -p "$LIVE" && git -C "$LIVE" init -q -b live-runs
    git -C "$LIVE" remote add origin "$(git -C "$REPO" remote get-url origin)"
    git -C "$LIVE" config user.name "$NAME"; git -C "$LIVE" config user.email "$EMAIL"
  fi
  mkdir -p "$LIVE/runs/twin-day"
  cp "$STATUS" "$LIVE/runs/twin-day/" 2>/dev/null
  for d in "$RUNS"/*/; do
    [ -f "$d/metrics.jsonl" ] || continue
    n=$(basename "$d"); mkdir -p "$LIVE/runs/twin-day/$n"
    cp "$d/metrics.jsonl" "$LIVE/runs/twin-day/$n/"; [ -f "$d/summary.json" ] && cp "$d/summary.json" "$LIVE/runs/twin-day/$n/"
  done
  git -C "$LIVE" add -A
  if git -C "$LIVE" rev-parse -q --verify HEAD >/dev/null; then git -C "$LIVE" commit -q --amend --no-edit
  else git -C "$LIVE" commit -q -m "Twin day live logs"; fi
  git -C "$LIVE" push -q -f origin live-runs
}
( while true; do push_live >/dev/null 2>&1; sleep 30; done ) &
LIVE_PID=$!

finish() {
  status end info "script finished"
  kill "$LIVE_PID" 2>/dev/null
  push_live || echo "live push failed"
  mkdir -p reports logs/twin-day
  $PY scripts/twin_report.py "$RUNS" > reports/twin_day.md
  cp "$STATUS" logs/twin-day/ 2>/dev/null
  for d in "$RUNS"/*/; do n=$(basename "$d"); mkdir -p "logs/twin-day/$n"; cp "$d"/metrics.jsonl "$d"/summary.json "logs/twin-day/$n/" 2>/dev/null; done
  git add reports/twin_day.md logs/twin-day
  [ -n "${DL_PID:-}" ] && kill "$DL_PID" 2>/dev/null
  git commit -q -m "Twin day results: Session 3 rehearsal on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  git pull -q --rebase origin twin-day && git push -q origin HEAD:twin-day && echo ">>> results pushed to GitHub (branch twin-day)"
  cat reports/twin_day.md
  if [ -n "${JL_API_KEY:-}" ] && [ -n "${JL_MACHINE_ID:-}" ] && [ -z "${KEEP_MACHINE:-}" ]; then
    echo ">>> pausing this Jarvis machine"; $PY scripts/stop_machine.py
  else
    echo ">>> STOP THE MACHINE NOW in the Jarvis dashboard (billing continues until you do)"
  fi
}
trap finish EXIT

# ---------- 1. setup ----------
status setup running "GPU check, packages, size test"
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
grep -vE '^torch==' requirements.txt > /tmp/req.txt
$PY -m pip install -q -r /tmp/req.txt jarvislabs pytest > /tmp/pip.log 2>&1 || { tail -5 /tmp/pip.log; status setup fail "pip install failed: $(tail -1 /tmp/pip.log)"; exit 1; }
TORCH=$($PY -c "import torch, transformers; print(torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)" 2>&1 | tail -1)
echo "python: $PY | $TORCH"
case "$TORCH" in *"cuda True"*) ;; *) status setup fail "no CUDA torch in $PY: $TORCH"; exit 1;; esac
if $PY -m pytest -q tests/test_model_size.py > /tmp/size.log 2>&1; then
  status setup pass "$GPU; $TORCH; size test 180,925,056 OK"
else
  tail -15 /tmp/size.log; status setup fail "size test failed: $(tail -1 /tmp/size.log)"; exit 1
fi

# ---------- 2. data: one train shard and one val shard from the private HF dataset ----------
# fetch FIRST LAST: train shards FIRST..LAST-1 (0 = also the first val shard). A shard is linked into shards/ only once
# fully downloaded, so train.py never sees half a file. Already-downloaded shards are skipped.
fetch() {
$PY - "$DATA" "$1" "$2" <<'EOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files
repo, data, first, last = "magnateadi/chotu-fineweb-edu-10bt-tokens", sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
files = list_repo_files(repo, repo_type="dataset", token=os.getenv("HF_TOKEN"))
shards = lambda prefix: sorted(x for x in files if os.path.basename(x).startswith(prefix) and x.endswith(".bin"))
want = shards("train_")[first:last] + (shards("val_")[:1] if first == 0 else [])
for f in want:
    p = hf_hub_download(repo, f, repo_type="dataset", local_dir=os.path.join(data, "hf"), token=os.getenv("HF_TOKEN"))
    link = os.path.join(data, "shards", os.path.basename(f))
    if not os.path.exists(link):
        os.symlink(p, link)
    print(f, round(os.path.getsize(p) / 1e9, 2), "GB", flush=True)
EOF
}
status data running "download 1 train + 1 val shard"
if fetch 0 1; then status data pass "$(ls "$DATA"/shards | tr '\n' ' ')"; else status data fail "HF download failed (check HF_TOKEN)"; exit 1; fi
if [ "$TRAIN_SHARDS" -gt 1 ]; then  # extra shards for the 5x run, in the background while the next steps use the GPU
  fetch 1 "$TRAIN_SHARDS" > "$RUNS/download.log" 2>&1 &
  DL_PID=$!
  status data info "downloading $((TRAIN_SHARDS - 1)) more train shards in the background (for main5x)"
fi

T="$PY train.py --data $DATA --gpu-price $PRICE"

# ---------- 3. overfit one batch: the loop must be able to memorise ----------
if want overfit; then
status overfit running "150 steps on one batch"
$T --model twin --out "$RUNS/overfit" --kind overfit --overfit --tokens 9830400 --eval-every 50 --ckpt-every 1000
L=$($PY -c "import json; print(json.load(open('$RUNS/overfit/summary.json'))['final_loss'])" 2>/dev/null || echo 99)
$PY -c "import sys; sys.exit(0 if float('$L') < 1.0 else 1)" && status overfit pass "final loss $L (< 1.0)" || status overfit fail "final loss $L (should be < 1.0)"
fi

# ---------- 4. fire drill: crash at step 70, resume, compare with an uninterrupted run ----------
if want firedrill; then
status firedrill running "reference run, crash at 70, resume"
D="--model twin --tokens 7864320 --eval-every 60 --ckpt-every 25 --kind firedrill"
$T $D --out "$RUNS/drill-ref" --name drill-ref
$T $D --out "$RUNS/drill" --name drill --crash-at 70
$T $D --out "$RUNS/drill" --name drill --resume
DIFF=$($PY -c "
import json
L=lambda p:{r['step']:r['loss'] for r in map(json.loads,open(p)) if r['t']=='step'}
a,b=L('$RUNS/drill-ref/metrics.jsonl'),L('$RUNS/drill/metrics.jsonl')
print(round(max(abs(a[s]-b[s]) for s in range(51,121)),4))" 2>/dev/null || echo 99)
$PY -c "import sys; sys.exit(0 if float('$DIFF') < 0.05 else 1)" && status firedrill pass "resumed run matches, max loss gap $DIFF" || status firedrill fail "resumed run differs, max loss gap $DIFF"
fi

# ---------- 5. main twin run: 140M tokens (20 per knob), compiled, exams and samples ----------
if want main; then
status main running "140M tokens, compile, samples"
$T --model twin --out "$RUNS/main" --kind train --tokens 140e6 --eval-every 100 --ckpt-every 250 --compile --samples
$PY -c "import json; s=json.load(open('$RUNS/main/summary.json')); exit(0 if s['reason']=='finished' else 1)" 2>/dev/null \
  && status main pass "$($PY -c "import json; s=json.load(open('$RUNS/main/summary.json')); print(f\"val {s['val_bpb']} bpb, {s['tok_s']:,} tok/s, {s['minutes']} min\")")" \
  || status main fail "did not finish (see alerts)"
fi

# ---------- 6. speed test of the real 181M model: 30 steps at the real batch ----------
if want speed; then
status speed running "181M, 30 steps x 524,288 tokens"
$T --model chotu --out "$RUNS/speed-181m" --name speed-181m --kind speed --tokens 10.06e9 --max-steps 30 --compile --eval-windows 16 --eval-every 1000 --ckpt-every 1000
$PY -c "import json; s=json.load(open('$RUNS/speed-181m/summary.json')); print(f\"{s['tok_s']:,} tok/s, MFU {s.get('mfu',0):.1%}, full run {s.get('full_run_hours')} h ≈ ₹{s.get('full_run_cost_inr',0):,}\")" > /tmp/speed.txt 2>/dev/null \
  && status speed pass "$(cat /tmp/speed.txt)" || status speed fail "speed test did not finish"
fi

# ---------- 7. 5x twin run (opt-in): 700M tokens, same model, same recipe, same exam as the 140M main run ----------
if asked main5x; then
status main5x running "waiting for the extra train shards"
if [ -n "${DL_PID:-}" ] && ! wait "$DL_PID"; then
  tail -5 "$RUNS/download.log"; status main5x fail "extra shard download failed (see download.log)"; exit 1
fi
DL_PID=""
N=$(ls "$DATA"/shards/train_*.bin | wc -l)
[ "$N" -ge "$TRAIN_SHARDS" ] || { status main5x fail "only $N of $TRAIN_SHARDS train shards present"; exit 1; }
status main5x running "700M tokens over $N train shards, compile, samples"
$T --model twin --out "$RUNS/main-5x" --name main-5x --kind train --tokens 700e6 --eval-every 250 --ckpt-every 1000 --compile --samples
$PY -c "import json; s=json.load(open('$RUNS/main-5x/summary.json')); exit(0 if s['reason']=='finished' else 1)" 2>/dev/null \
  && status main5x pass "$($PY -c "import json; s=json.load(open('$RUNS/main-5x/summary.json')); print(f\"val {s['val_bpb']} bpb (140M run: 1.3881), {s['tok_s']:,} tok/s, {s['minutes']} min\")")" \
  || status main5x fail "did not finish (see alerts)"

# keep the result before the machine pauses: newest checkpoint + logs to the private HF repo
status upload running "main-5x checkpoint to HF magnateadi/chotu-twin-day"
if $PY - "$RUNS/main-5x" <<'EOF'
import os, sys, pathlib
from huggingface_hub import HfApi
run = pathlib.Path(sys.argv[1])
ck = sorted((run / "ckpt").glob("step_*.pt"))[-1]
api = HfApi(token=os.getenv("HF_TOKEN"))
for f in (ck, run / "metrics.jsonl", run / "summary.json"):
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f"twin-day/main-5x/{f.relative_to(run)}", repo_id="magnateadi/chotu-twin-day")
print(ck.name)
EOF
then status upload pass "twin-day/main-5x on HF"; else status upload fail "upload failed; checkpoint is still in $RUNS/main-5x/ckpt"; fi
fi
