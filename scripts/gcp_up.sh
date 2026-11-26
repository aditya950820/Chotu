#!/usr/bin/env bash
# On a GCP GPU VM (Deep Learning VM image, NVIDIA driver 580). Continues the chain from where the
# private HF backup left off, then takes a final backup and powers the VM off (GPU billing stops,
# the disk is kept), even when a step fails. Safe to rerun: after a Spot preemption the VM's
# startup script runs it again and training resumes from its last checkpoint.
#
#   bash scripts/gcp_up.sh      # started for you by scripts/launch_gcp.sh (on the Mac)
#
# Differences from scripts/gpu_up.sh (Jarvis): no Jarvis pause (the chain runs with --no-stop and
# this script owns shutdown), no speed sweep or tokens/s go/no-go (those gate pretraining, which
# is done), and file times are put back after a restore so the chat checkpoint accepts its inputs.
# Needs .env: HF_TOKEN CHOTU_STORAGE GIT_NAME GIT_EMAIL (optional: NO_STOP=1 keeps the VM on).
set -Eeuo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
for key in HF_TOKEN CHOTU_STORAGE GIT_NAME GIT_EMAIL; do
  [[ -n "${!key:-}" ]] || { echo "missing $key in .env" >&2; exit 2; }
done
step="start"
stamp() { step="$*"; echo "[$(date +%H:%M:%S)] $*"; }
S=$CHOTU_STORAGE
LOGS=$S/setup-logs
mkdir -p "$LOGS" "$S/reports" "$S/logs" "$S/dashboard"
export HF_HOME=$S/hf-home TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor TRITON_CACHE_DIR=$S/cache/triton
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
BUILD_JOBS=28   # recorded in chain-config.json on the first machine; must match or the chain refuses to resume
TORCH=2.11.0    # the same torch build the run used on Jarvis (2.11.0+cu130)

finish() {
  code=$?
  trap - EXIT
  (( code == 0 )) || echo "GCP RUN STOPPED (exit $code) during: $step; logs in $LOGS"
  date -u +"%Y-%m-%dT%H:%M:%SZ exit=$code step=$step" >> "$S/dashboard/gcp-exits.log"
  if [[ -x "$S/venv/bin/python" ]]; then
    timeout 1800 "$S/venv/bin/python" scripts/backup.py --once --storage "$S" >> "$LOGS/backup.log" 2>&1 \
      || echo "final backup failed (see $LOGS/backup.log)"
  fi
  if [[ "${NO_STOP:-0}" == 1 ]]; then
    stamp "NO_STOP=1: not powering off (on Vertex AI the job ending releases the machine)"
  else
    stamp "powering the VM off (GPU billing stops; disk kept)"
    sudo shutdown -h now
  fi
  exit "$code"
}
trap finish EXIT

# Live dashboard first (standard library only): port 6006, key-protected.
pgrep -f "dashboard.py --storage" >/dev/null || nohup python3 tools/live_dashboard.py --storage "$S" > "$LOGS/dashboard.log" 2>&1 &

stamp "waiting for the NVIDIA driver"
for _ in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

stamp "Python 3.12 venv with torch $TORCH on the disk (once)"
export UV_PYTHON_INSTALL_DIR=$S/uv-python UV_CACHE_DIR=$S/cache/uv UV_LINK_MODE=copy PATH=$HOME/.local/bin:$PATH
if ! "$S/venv/bin/python" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
  # pip --user works on the GCP image; Ubuntu 24.04 (Nebius) refuses it, so fall back to uv's own installer.
  command -v uv >/dev/null || python3 -m pip install -q --user uv 2>/dev/null ||
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
  # The torch build must not need a newer CUDA than the host driver supports.
  cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$')
  tag=$(python3 -c 'import sys; v=tuple(map(int, sys.argv[1].split("."))); print("cu130" if v >= (13,0) else "cu128" if v >= (12,8) else "cu126")' "$cuda")
  stamp "driver supports CUDA $cuda: torch $TORCH+$tag"
  rm -rf "$S/venv"
  uv venv --seed --python 3.12 "$S/venv"
  uv pip install --python "$S/venv/bin/python" "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$tag"
fi
source "$S/venv/bin/activate"
python -m pip install -q huggingface_hub==2.3.0 python-dotenv
python - <<'EOF'
import os, shutil, sys, torch
assert torch.cuda.is_available(), "torch has no CUDA"
print(f"GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda} | Python {sys.version.split()[0]}")
storage = os.environ["CHOTU_STORAGE"]
free = shutil.disk_usage(storage).free / 2**30
assert free >= 60, f"need >= 60 GiB free on the disk, have {free:.0f}"
EOF
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

stamp "restore from the HF backup if this disk is new (no-op otherwise)"
python scripts/backup.py --check --storage "$S"
python scripts/backup.py --restore --storage "$S" 2>&1 | tee -a "$LOGS/backup.log"
# Newest checkpoint to HF every BACKUP_CHECKPOINT_MIN minutes (Vertex sets 20: its disk vanishes with the job).
pgrep -f "scripts/backup.py --loop" >/dev/null || nohup nice -n 10 python scripts/backup.py --loop --storage "$S" \
  --checkpoint-min "${BACKUP_CHECKPOINT_MIN:-60}" >> "$LOGS/backup.log" 2>&1 &

stamp "packages + llama.cpp prebuild (background)"
(
  req=$(mktemp); constraint=$(mktemp)
  grep -v '^torch==' requirements.txt > "$req"
  python -c 'import torch; print("torch==" + torch.__version__)' > "$constraint"
  python -m pip install -q -r "$req" -c "$constraint" jarvislabs liger-kernel
  python -c 'import liger_kernel, transformers; print("liger + transformers OK")'
  python scripts/export_gguf.py --prebuild --cache-dir "$S/cache" --jobs $(( $(nproc) / 2 ))
) > "$LOGS/packages.log" 2>&1 &
packages=$!

stamp "shards (background; the finished shards stage checks them, and the final evals use val)"
(
  python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download("magnateadi/chotu-fineweb-edu-10bt-tokens", repo_type="dataset",
                  local_dir=os.environ["CHOTU_STORAGE"], max_workers=16)
print("all shards downloaded")
EOF
) > "$LOGS/shards.log" 2>&1 &
shards=$!

stamp "waiting for packages, shards and the chat data copied from the Mac"
wait "$packages" || { tail -20 "$LOGS/packages.log"; exit 1; }
wait "$shards" || { tail -20 "$LOGS/shards.log"; exit 1; }
python scripts/backup.py --apply-shard-mtimes --storage "$S"
for _ in $(seq 1 360); do [[ -f "$S/chat/.done" ]] && break; sleep 5; done
[[ -f "$S/chat/.done" ]] || { echo "chat data missing: run scripts/launch_gcp.sh on the Mac" >&2; exit 1; }

stamp "put back the file times the chat checkpoint recorded"
python tools/match_manifest.py --storage "$S" --run chat

stamp "fused-loss check on this GPU (the H200 result is kept as loss-check.H200.json)"
[[ -f "$S/reports/loss-check.H200.json" ]] || cp "$S/reports/loss-check.json" "$S/reports/loss-check.H200.json" 2>/dev/null || true
python scripts/check_loss.py --storage "$S" > "$LOGS/loss-check.log" 2>&1 || echo "fused loss failed its check; training uses the safe fallback"

stamp "chain: resumes chat tuning, then final evals, export, upload (VM powers off afterwards)"
python scripts/run_chain.py --storage "$S" --build-jobs "$BUILD_JOBS" --no-stop
stamp "chain finished"
