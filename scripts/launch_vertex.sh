#!/usr/bin/env bash
# Mac: run the rest of the chain as a Vertex AI custom training job on one Spot A100 40GB.
# (New GCP accounts get no Compute Engine GPU quota, but Vertex AI gives 1 Spot A100 per region.)
#
#   bash scripts/launch_vertex.sh            # upload code + chat data, submit the job
#   bash scripts/launch_vertex.sh --resubmit # code/data already uploaded: submit again (after a preemption)
#
# The job restores the private HF backup, resumes from the newest checkpoint and ends by itself
# (GPU released) when the chain finishes or fails. Hard cap: 12 h. Uses the gcloud configuration
# "chotu". The HF token goes into the job's environment, visible only to people on this project.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export CLOUDSDK_ACTIVE_CONFIG_NAME=${CLOUDSDK_ACTIVE_CONFIG_NAME:-chotu}
export PATH=/opt/homebrew/share/google-cloud-sdk/bin:$PATH
REGION=${REGION:-us-central1}
IMAGE=${IMAGE:-us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest}
MAX_HOURS=${MAX_HOURS:-12}
chat=${CHAT_DIR:-$HOME/chotu-chat}
project=$(gcloud config get-value project 2>/dev/null)
number=$(gcloud projects describe "$project" --format="value(projectNumber)")
BUCKET=${BUCKET:-chotu-run-$number}

set -a; source .env; set +a
[[ -n "${HF_TOKEN:-}" ]] || { echo "add HF_TOKEN to your Mac .env first" >&2; exit 2; }
name=$(git config user.name); email=$(git config user.email)
[[ -n "$name" && -n "$email" ]] || { echo "set git user.name/user.email on the Mac" >&2; exit 2; }
echo "== project $project, region $REGION, bucket gs://$BUCKET"

if [[ "${1:-}" != --resubmit ]]; then
  [[ -f "$chat/chat/.done" && -f "$chat/reports/chat_report.md" ]] || { echo "chat data not ready in $chat" >&2; exit 2; }
  gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 || gcloud storage buckets create "gs://$BUCKET" \
    --location "$REGION" --uniform-bucket-level-access --public-access-prevention
  echo "== code ($(git rev-parse --short HEAD)$(git diff --quiet || echo ', with local edits'))"
  tgz=$(mktemp).tgz
  git ls-files -z | tar --null -czf "$tgz" -T -
  gcloud storage cp "$tgz" "gs://$BUCKET/code.tgz"
  echo "== chat data (1.2 GB, once; .done goes last)"
  gcloud storage rsync "$chat/chat" "gs://$BUCKET/chat" --exclude '^\.done$'
  gcloud storage cp "$chat/reports/chat_report.md" "gs://$BUCKET/chat_report.md"
  gcloud storage cp "$chat/chat/.done" "gs://$BUCKET/chat/.done"
fi

spec=$(mktemp).yaml
cat > "$spec" <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 250
    containerSpec:
      imageUri: $IMAGE
      command: ["bash", "-c", "tar -xzf /gcs/$BUCKET/code.tgz -C /tmp scripts/vertex_job.sh && bash /tmp/scripts/vertex_job.sh"]
      env:
        - {name: BUCKET, value: "$BUCKET"}
        - {name: HF_TOKEN, value: "$HF_TOKEN"}
        - {name: GIT_NAME, value: "$name"}
        - {name: GIT_EMAIL, value: "$email"}
scheduling:
  strategy: SPOT
  timeout: $(( MAX_HOURS * 3600 ))s
EOF
echo "== submitting Spot A100 job"
job=$(gcloud ai custom-jobs create --region "$REGION" --display-name "chotu-chain-$(date +%m%d-%H%M)" \
  --config "$spec" --format="value(name)")
rm -f "$spec"
echo "$job" > "$HOME/.chotu-vertex-job"
echo "== job: $job"
echo "Logs:   gcloud ai custom-jobs stream-logs ${job##*/} --region $REGION"
echo "State:  gcloud ai custom-jobs describe ${job##*/} --region $REGION --format='value(state)'"
