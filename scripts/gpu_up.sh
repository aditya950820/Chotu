#!/usr/bin/env bash
# One command on a fresh Jarvis H200 (PyTorch template). Safe to rerun after a Spot interruption:
# finished work is skipped and training resumes from its last checkpoint.
#
#   bash scripts/gpu_up.sh      # usually started for you by scripts/launch.sh (on the Mac)
#
# Order, built so the GPU is never idle waiting on something that could overlap:
#   t=0   in parallel: packages → llama.cpp prebuild | first shards → rest of shards | chat data
#   ~3m   speed sweep on the real model as soon as packages + first shards are in (GPU busy)
#   ~15m  go/no-go on tokens/sec, wait for downloads, then the full chain (it pauses the machine itself)
#         + a CPU-only rehearsal of the later stages (chat/evals/export) on spare cores, for early warning
# Needs .env: HF_TOKEN JL_API_KEY JL_MACHINE_ID CHOTU_STORAGE GIT_NAME GIT_EMAIL
#             (optional: WANDB_API_KEY, MIN_TOK_S, NO_STOP=1). Progress: python scripts/status.py
set -Eeuo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
for key in HF_TOKEN JL_API_KEY JL_MACHINE_ID CHOTU_STORAGE GIT_NAME GIT_EMAIL; do
  [[ -n "${!key:-}" ]] || { echo "missing $key in .env" >&2; exit 2; }
done
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
S=$CHOTU_STORAGE
LOGS=$S/setup-logs
mkdir -p "$LOGS" "$S/reports" "$S/logs"
# Caches on the persistent volume: survive pause/resume and keep the small root disk free.
export HF_HOME=$S/hf-home TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor TRITON_CACHE_DIR=$S/cache/triton

# Live dashboard first (standard library only, so it is up within a second): port 6006.
pgrep -f "scripts/dashboard.py" >/dev/null || nohup python scripts/dashboard.py --storage "$S" > "$LOGS/dashboard.log" 2>&1 &

# Our pins need Python >= 3.11; the Jarvis PyTorch template ships 3.10. Build a 3.12 venv on the
# volume once (survives pause/resume), with the template's own torch build so CUDA matches the driver.
export UV_PYTHON_INSTALL_DIR=$S/uv-python UV_CACHE_DIR=$S/cache/uv UV_LINK_MODE=copy
if ! python -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  if [[ ! -x "$S/venv/bin/python" ]] || ! "$S/venv/bin/python" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    stamp "template Python $(python -V 2>&1 | cut -d' ' -f2) < 3.11: building Python 3.12 venv on the volume (once)"
    torch_ver=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
    cuda_tag=$(python -c 'import torch; print("cu" + torch.version.cuda.replace(".", ""))')
    command -v uv >/dev/null || python -m pip install -q uv
    rm -rf "$S/venv"
    uv venv --seed --python 3.12 "$S/venv"
    uv pip install --python "$S/venv/bin/python" "torch==$torch_ver" --index-url "https://download.pytorch.org/whl/$cuda_tag"
  fi
  source "$S/venv/bin/activate"
  stamp "using $(python -V) with torch $(python -c 'import torch; print(torch.__version__)')"
fi
MIN_TOK_S=${MIN_TOK_S:-180000}
NCPU=$(nproc)
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
handed_over=0

pause_machine() {
  [[ "${NO_STOP:-0}" == 1 ]] && { stamp "NO_STOP=1: machine left running"; return; }
  python scripts/stop_machine.py || echo 'MACHINE PAUSE FAILED: PAUSE IT MANUALLY NOW' >&2
}
# Until the chain takes over, every exit (failure or no-go) pauses the GPU.
final_backup() { timeout 900 python scripts/backup.py --once --storage "$S" >> "$LOGS/backup.log" 2>&1 || echo "final backup failed (see $LOGS/backup.log)"; }
trap 'code=$?; if (( ! handed_over )); then (( code == 0 )) || stamp "SETUP FAILED (exit $code); logs in $LOGS"; final_backup; pause_machine; fi' EXIT

stamp "checks"
python - <<'EOF'
import shutil, sys, os, torch
assert sys.version_info >= (3, 11), f"need Python >= 3.11, got {sys.version}"
assert torch.cuda.is_available(), "template torch has no CUDA"
name = torch.cuda.get_device_name(0)
print(f"GPU {name} | torch {torch.__version__} | CUDA {torch.version.cuda} | Python {sys.version.split()[0]}")
if "H200" not in name:
    print(f"WARNING: tuned for H200, got {name}")
storage = os.environ["CHOTU_STORAGE"]
free = shutil.disk_usage(storage).free / 2**30
present = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(storage) for f in fs
              if not os.path.islink(os.path.join(d, f))) / 2**30
# the chain needs ~81 GiB in total (shards, 10 checkpoints, exports, eval data, builds, 20% margin)
assert free + present >= 100, f"need >= 100 GiB on the volume (free + already used by Chotu), have {free:.0f} + {present:.0f}"
print(f"disk: {free:.0f} GiB free + {present:.0f} GiB already used by Chotu")
EOF
git config --global user.name "$GIT_NAME"     # check_setup refuses non-human identities
git config --global user.email "$GIT_EMAIL"
python -m pip install -q huggingface_hub==2.3.0 python-dotenv jarvislabs   # seconds; downloads need it now
python scripts/stop_machine.py --dry-run >/dev/null || { echo "Jarvis key / machine id wrong" >&2; exit 2; }
python scripts/backup.py --check --storage "$S" || { echo "HF token cannot create/verify the private backup repo (needs write)" >&2; exit 2; }
stamp "restore from the HF backup if this is a new instance (no-op otherwise)"
python scripts/backup.py --restore --storage "$S" 2>&1 | tee -a "$LOGS/backup.log"
pgrep -f "scripts/backup.py --loop" >/dev/null || nohup nice -n 10 python scripts/backup.py --loop --storage "$S" >> "$LOGS/backup.log" 2>&1 &

# --- parallel setup ----------------------------------------------------------------------------
stamp "packages + llama.cpp prebuild (background)"
(
  req=$(mktemp); constraint=$(mktemp)
  grep -v '^torch==' requirements.txt > "$req"
  python -c 'import torch; print("torch==" + torch.__version__)' > "$constraint"   # keep template torch
  python -m pip install -q -r "$req" -c "$constraint" jarvislabs liger-kernel
  python -c 'import liger_kernel, transformers; print("liger + transformers OK")'
  python scripts/export_gguf.py --prebuild --cache-dir "$S/cache" --jobs $(( NCPU / 2 ))
) > "$LOGS/packages.log" 2>&1 &
packages=$!
packages_ok=0
wait_packages() {
  (( packages_ok )) && return
  wait "$packages" || { tail -20 "$LOGS/packages.log"; exit 1; }
  packages_ok=1
}

stamp "shards: first train/val shard, then the rest (background)"
(
  python - <<'EOF'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
repo, storage = "magnateadi/chotu-fineweb-edu-10bt-tokens", os.environ["CHOTU_STORAGE"]
snapshot_download(repo, repo_type="dataset", local_dir=storage, max_workers=8,
                  allow_patterns=["shards/train_000.bin", "shards/val_000.bin"])
Path(storage, "setup-logs/first-shards.ready").touch()
snapshot_download(repo, repo_type="dataset", local_dir=storage, max_workers=16)
print("all shards downloaded")
EOF
) > "$LOGS/shards.log" 2>&1 &
shards=$!

stamp "chat data (background)"
(
  # launch.sh copies the Mac-prepared data (writes .done last). Fall back to preparing it here.
  for _ in $(seq 1 480); do [[ -f "$S/chat/.done" ]] && break; sleep 5; done
  if [[ -f "$S/chat/.done" ]]; then echo "chat data present"; else
    echo "chat data not copied after 40 min; preparing it here"
    while kill -0 "$packages" 2>/dev/null; do sleep 5; done   # needs the installed packages
    python scripts/prepare_chat.py --storage "$S"
  fi
) > "$LOGS/chat.log" 2>&1 &
chat=$!

# --- speed sweep as soon as possible -----------------------------------------------------------
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
sweep_ok() { python -c 'import json,sys; sys.exit(not (json.load(open(sys.argv[1])).get("best") or {}).get("tokens_per_second"))' "$S/reports/speed-sweep.json" 2>/dev/null; }
if sweep_ok && [[ "$(cat "$S/reports/speed-sweep.gpu" 2>/dev/null)" == "$gpu" ]]; then
  stamp "speed sweep already done on this GPU type; skipping"
else
  rm -f "$S/reports/speed-sweep.json"
  stamp "waiting for packages + first shards"
  wait_packages
  until [[ -f "$LOGS/first-shards.ready" ]]; do
    kill -0 "$shards" 2>/dev/null || { tail -20 "$LOGS/shards.log"; exit 1; }
    sleep 2
  done
  stamp "gradient check: fused loss vs plain loss on the real model (catches a broken GPU/kernel)"
  if ! python scripts/check_loss.py --storage "$S" > "$LOGS/loss-check.log" 2>&1; then
    tail -30 "$LOGS/loss-check.log"; echo "gradient check failed: GPU or kernel gives wrong gradients" >&2; exit 1
  fi
  grep -E '"(fused_ce|cosine)"' "$LOGS/loss-check.log" || true
  stamp "speed sweep (real 181M model)"
  python scripts/speed_sweep.py --storage "$S" --no-stop 2>&1 | tee "$LOGS/sweep.log" | grep -E '^(SWEEP|Best|SPEED| *[0-9])' || true
  [[ -f "$S/reports/speed-sweep.json" ]] || { echo "sweep wrote no report" >&2; exit 1; }
  echo "$gpu" > "$S/reports/speed-sweep.gpu"
fi
python - "$S/reports/speed-sweep.json" "$MIN_TOK_S" <<'EOF'
import json, sys
best = json.load(open(sys.argv[1]))["best"]
if not best or best["tokens_per_second"] < float(sys.argv[2]):
    sys.exit(f"NO-GO: best {best and round(best['tokens_per_second'])} tok/s < {sys.argv[2]}. Pausing; nothing else spent.")
print(f"GO: {best['tokens_per_second']:,.0f} tok/s, MFU {best['mfu']:.1%}, micro {best['micro_tokens']}, "
      f"compile {best['compile_mode']} -> ~{10e9 / best['tokens_per_second'] / 3600:.1f} h for 10B tokens")
EOF

stamp "waiting for remaining downloads, packages and chat data"
wait_packages
wait "$shards" || { tail -20 "$LOGS/shards.log"; exit 1; }
python scripts/backup.py --apply-shard-mtimes --storage "$S"   # only acts after a restore
wait "$chat" || { tail -20 "$LOGS/chat.log"; exit 1; }
[[ -f "$S/chat/.done" ]] || { echo "chat data missing" >&2; exit 1; }

# --- early warning: CPU rehearsal of every later stage, on spare cores, while the GPU trains ----
if [[ ! -f "$S/rehearsal/stages/final_report.done.json" ]]; then
  stamp "CPU rehearsal of chat/evals/export in background -> $LOGS/rehearsal.log"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup nice -n 19 python scripts/run_chain.py --storage "$S" \
    --rehearsal --device cpu --shards-subset --skip-upload --no-stop --retries 0 \
    > "$LOGS/rehearsal.log" 2>&1 &
fi

stamp "handing over to the full chain (it pauses the machine when done or on failure)"
handed_over=1
exec bash scripts/run_all.sh --storage "$S" --build-jobs "$NCPU"
