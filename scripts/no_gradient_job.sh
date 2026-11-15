#!/usr/bin/env bash
# The no-gradient language model on all training shards, in parallel, on a rented many-core machine.
# Started by scripts/jarvis_launch.py --job scripts/no_gradient_job.sh (the machine's .env holds the keys).
#   1. download the val shards and SHARDS train shards from the private HF dataset (in parallel)
#   2. one worker per (shard, order group): count the n-grams the exam and tuning positions need (JOBS at a time)
#   3. merge all workers' counts, fit discounts and EM mixing on the tuning set, score the locked exam (kn.py --combine)
#   4. study better document memory and mixing (mix.py)
# Steps go to the no-gradient-live branch every 60 s; results (numbers only, no data) are committed to this branch;
# the saved predictions go to the private HF repo. Ends by pausing the machine, even after a failure.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
[ -f .env ] && set -a && . ./.env && set +a
PY=${PY:-$(command -v python3)}
SHARDS=${SHARDS:-14}
TEST=${TEST:-}          # TEST=1: no pushes, uploads or pausing (for a local dry run)
MAXTOK=${MAXTOK:-}      # e.g. 2e6: count only the first tokens of each shard (dry run)
JOBS=${JOBS:-14}
W=$HOME/ng
OUT=learning/no_gradient/runs/full_${SHARDS}shards
STATUS=$W/status.jsonl
LIVE=$HOME/ng-live
NAME="aditya950820"; EMAIL="202965202+aditya950820@users.noreply.github.com"
mkdir -p "$W/shards" "$W/dump"; [ -n "$TEST" ] || mkdir -p "$OUT"
: > "$STATUS"
git config user.name "$NAME"; git config user.email "$EMAIL"

status() {  # id, status, note
  $PY -c 'import json,sys,time; print(json.dumps({"id":sys.argv[1],"status":sys.argv[2],"note":sys.argv[3],"time":round(time.time())}))' "$1" "$2" "${3:-}" >> "$STATUS"
  echo ">>> [$1] $2 ${3:-}"
}
push_live() {
  if [ ! -d "$LIVE/.git" ]; then
    mkdir -p "$LIVE" && git -C "$LIVE" init -q -b no-gradient-live
    git -C "$LIVE" remote add origin "https://$NAME:$GH_TOKEN@github.com/aditya950820/Chotu.git"
    git -C "$LIVE" config user.name "$NAME"; git -C "$LIVE" config user.email "$EMAIL"
  fi
  mkdir -p "$LIVE/runs/no-gradient"
  cp "$STATUS" "$LIVE/runs/no-gradient/status.jsonl"
  cat "$W/workers.log" "$W/run.log" 2>/dev/null | grep -v "counted" | tail -40 | sed -E 's/(github_pat_|hf_)[A-Za-z0-9_]+/***/g' > "$LIVE/runs/no-gradient/log_tail.txt"
  free -g | head -2 > "$LIVE/runs/no-gradient/machine.txt"; nproc >> "$LIVE/runs/no-gradient/machine.txt"
  git -C "$LIVE" add -A
  if git -C "$LIVE" rev-parse -q --verify HEAD >/dev/null; then git -C "$LIVE" commit -q --amend --no-edit
  else git -C "$LIVE" commit -q -m "No-gradient live status"; fi
  git -C "$LIVE" push -q -f origin no-gradient-live
}
[ -n "$TEST" ] && push_live() { :; }
( while true; do push_live >/dev/null 2>&1; sleep 60; done ) &
LIVE_PID=$!

finish() {
  status end info "job finished"
  kill "$LIVE_PID" 2>/dev/null
  push_live || true
  [ -n "$TEST" ] && { echo ">>> TEST: results in $W, nothing pushed, machine not paused"; return; }
  cp "$W"/run.json "$W"/mix.json "$OUT"/ 2>/dev/null
  sed -E 's/(github_pat_|hf_)[A-Za-z0-9_]+/***/g' "$W/run.log" > "$OUT/run.log" 2>/dev/null
  cp "$STATUS" "$OUT/" 2>/dev/null
  git add "$OUT" && git commit -q -m "No-gradient model on $SHARDS shards: results" && \
    { git pull -q --rebase origin "$(git rev-parse --abbrev-ref HEAD)"; git push -q origin HEAD; }
  $PY - "$W/run.parts.npz" <<'EOF' || true
import os, sys
from huggingface_hub import HfApi
if os.path.exists(sys.argv[1]):
    HfApi(token=os.getenv("HF_TOKEN")).upload_file(path_or_fileobj=sys.argv[1], path_in_repo="no-gradient/full_run.parts.npz", repo_id="magnateadi/chotu-twin-day")
EOF
  $PY scripts/stop_machine.py || echo ">>> STOP THE MACHINE in the Jarvis dashboard"
}
trap finish EXIT

status setup pass "$(nproc) CPUs, $(free -g | awk '/Mem/{print $2}') GB memory, $($PY -c 'import numpy; print("numpy", numpy.__version__)')"

# ---------- 1. data ----------
status data running "downloading val + $SHARDS train shards"
if $PY - "$W" "$SHARDS" <<'EOF'
import os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
W, n = sys.argv[1], int(sys.argv[2])
files = ["shards/val_000.bin", "shards/val_001.bin"] + [f"shards/train_{i:03d}.bin" for i in range(n)]
get = lambda f: hf_hub_download("magnateadi/chotu-fineweb-edu-10bt-tokens", f, repo_type="dataset", local_dir=W, token=os.getenv("HF_TOKEN"))
with ThreadPoolExecutor(8) as ex:
    for p in ex.map(get, files):
        print(p, os.path.getsize(p) // 2, flush=True)
EOF
then status data pass "$(ls "$W"/shards/train_*.bin | wc -l) train shards"; else status data fail "download failed"; exit 1; fi

# ---------- 2. workers ----------
status count running "$((SHARDS * 4)) workers, $JOBS at a time"
for f in "$W"/shards/train_*.bin; do
  for g in 1 2 3 4,5,6; do echo "$(basename "$f" .bin) $g"; done
done | xargs -P "$JOBS" -L 1 sh -c "$PY learning/no_gradient/kn.py --dir $W --order 6 --train \$0 --worker \$1 --dump $W/dump ${MAXTOK:+--max-tokens $MAXTOK} >> $W/workers.log 2>&1 && echo done >> $W/workers_done.txt"
DONE=$(wc -l < "$W/workers_done.txt" 2>/dev/null || echo 0)
[ "$DONE" -eq $((SHARDS * 4)) ] && status count pass "$DONE workers" || { status count fail "only $DONE of $((SHARDS * 4)) workers finished"; tail -20 "$W/workers.log"; exit 1; }

# ---------- 3. merge and score ----------
status combine running "merging counts, fitting discounts and mixing on the tuning set"
if $PY -u learning/no_gradient/kn.py --dir "$W" --order 6 --combine --dump "$W/dump" --out "$W/run.json" > "$W/run.log" 2>&1; then
  status combine pass "$($PY -c "import json; r=json.load(open('$W/run.json')); print('exam', r['exam'])")"
else status combine fail "$(tail -1 "$W/run.log")"; exit 1; fi

# ---------- 4. mixing study ----------
status mix running "document memory and mixing variants"
$PY -u learning/no_gradient/mix.py "$W/run.parts.npz" --out "$W/mix.json" >> "$W/run.log" 2>&1 \
  && status mix pass "$($PY -c "import json; r=json.load(open('$W/mix.json')); b=min(r, key=lambda x: x['tune']); print('best on tuning:', b['variant'], '| exam', b['exam'])")" \
  || status mix fail "$(tail -1 "$W/run.log")"
