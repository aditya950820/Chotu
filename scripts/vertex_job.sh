#!/usr/bin/env bash
# Inside the Vertex AI training container. Unpacks the code and chat data that scripts/launch_vertex.sh
# put in the private bucket (mounted by Vertex at /gcs/<bucket>), then runs scripts/gcp_up.sh, which
# restores the HF backup and continues the chain. When this script exits, the job ends and Vertex
# releases the GPU, so there is nothing to power off.
# Needs env (set in the job spec): BUCKET HF_TOKEN GIT_NAME GIT_EMAIL
set -Eeuo pipefail
B=/gcs/$BUCKET
S=/home/chotu-data     # must stay /home/chotu-data: the run's recorded settings name this path
R=/home/Chotu
echo "[$(date +%H:%M:%S)] Vertex job on $(hostname): $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for _ in $(seq 1 30); do [[ -f "$B/code.tgz" ]] && break; sleep 5; done   # the bucket mount can take a moment
[[ -f "$B/code.tgz" ]] || { echo "bucket $BUCKET not mounted at /gcs" >&2; exit 2; }
mkdir -p "$R" "$S/chat" "$S/reports"
tar -xzf "$B/code.tgz" -C "$R"
cd "$R"
( umask 077; printf 'HF_TOKEN=%s\nCHOTU_STORAGE=%s\nGIT_NAME=%s\nGIT_EMAIL=%s\n' "$HF_TOKEN" "$S" "$GIT_NAME" "$GIT_EMAIL" > .env )
if [[ ! -f "$S/chat/.done" ]]; then
  echo "[$(date +%H:%M:%S)] copying chat data from the bucket"
  for f in "$B"/chat/*; do cp "$f" "$S/chat/"; done
  cp "$B/chat_report.md" "$S/reports/chat_report.md"
  cp "$B/chat/.done" "$S/chat/.done"
fi
NO_STOP=1 BACKUP_CHECKPOINT_MIN=20 bash scripts/gcp_up.sh
