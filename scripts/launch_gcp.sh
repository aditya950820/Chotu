#!/usr/bin/env bash
# Mac: create (or restart) a Spot GPU VM on GCP and start scripts/gcp_up.sh on it.
# The VM powers itself off when the chain ends; tools/gcp_watchdog.sh restarts it after a Spot preemption.
#
#   bash scripts/launch_gcp.sh
#   CANDIDATES="a2-highgpu-1g:us-central1-c" bash scripts/launch_gcp.sh    # force one machine/zone
#
# Uses the gcloud configuration "chotu" (your own account + project). SSH goes through IAP over HTTPS,
# so it also works on networks that block port 22. Needs on the Mac: .env with HF_TOKEN, git user.name
# and user.email, and the chat data in ~/chotu-chat (copied up so it matches the chat checkpoint).
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export CLOUDSDK_ACTIVE_CONFIG_NAME=${CLOUDSDK_ACTIVE_CONFIG_NAME:-chotu}
export PATH=/opt/homebrew/share/google-cloud-sdk/bin:$PATH
VM=${VM:-chotu-gpu}
# Fastest first: 1x H100 (Spot), then 1x A100 40GB (Spot), in the zones of the region with quota.
CANDIDATES=${CANDIDATES:-"a3-highgpu-1g:us-central1-a a3-highgpu-1g:us-central1-b a3-highgpu-1g:us-central1-c \
a2-highgpu-1g:us-central1-a a2-highgpu-1g:us-central1-b a2-highgpu-1g:us-central1-c a2-highgpu-1g:us-central1-f"}
IMAGE_FAMILY=${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}
DISK_GB=${DISK_GB:-200}
MAX_HOURS=${MAX_HOURS:-12}   # hard cost cap: GCP stops the VM after this many hours, whatever happens
remote_repo=/home/Chotu
storage=/home/chotu-data      # must stay /home/chotu-data: the run's recorded settings name this path
chat=${CHAT_DIR:-$HOME/chotu-chat}
user=$(whoami | tr '[:upper:]' '[:lower:]')

set -a; source .env; set +a
[[ -n "${HF_TOKEN:-}" ]] || { echo "add HF_TOKEN to your Mac .env first" >&2; exit 2; }
[[ -f "$chat/chat/.done" && -f "$chat/reports/chat_report.md" ]] || { echo "chat data not ready in $chat" >&2; exit 2; }
name=$(git config user.name); email=$(git config user.email)
[[ -n "$name" && -n "$email" ]] || { echo "set git user.name/user.email on the Mac" >&2; exit 2; }
project=$(gcloud config get-value project 2>/dev/null)
echo "== project $project (gcloud configuration $CLOUDSDK_ACTIVE_CONFIG_NAME)"

echo "== APIs and firewall (once)"
gcloud services enable compute.googleapis.com iap.googleapis.com --quiet
gcloud compute firewall-rules describe chotu-iap-ssh >/dev/null 2>&1 || gcloud compute firewall-rules create chotu-iap-ssh \
  --network default --allow tcp:22 --source-ranges 35.235.240.0/20 --target-tags chotu --quiet
gcloud compute firewall-rules describe chotu-dashboard >/dev/null 2>&1 || gcloud compute firewall-rules create chotu-dashboard \
  --network default --allow tcp:6006 --source-ranges 0.0.0.0/0 --target-tags chotu --quiet   # page needs the secret key

startup=$(mktemp)
cat > "$startup" <<'EOF'
#!/bin/bash
# Chotu: after a (re)boot, e.g. a Spot restart, resume the run as the user who launched it.
[[ -f /home/Chotu/scripts/gcp_up.sh && -f /home/Chotu/.env ]] || exit 0
u=$(stat -c %U /home/Chotu)
sudo -u "$u" -H bash -lc 'cd /home/Chotu && (tmux has-session -t chotu 2>/dev/null || tmux new -d -s chotu "bash scripts/gcp_up.sh 2>&1 | tee -a /home/chotu-data/gcp_up.log; exec bash")'
EOF

zone=$(gcloud compute instances list --filter="name=$VM" --format="value(zone.basename())")
if [[ -n "$zone" ]]; then
  status=$(gcloud compute instances describe "$VM" --zone "$zone" --format="value(status)")
  echo "== $VM already exists in $zone ($status)"
  [[ "$status" == RUNNING ]] || gcloud compute instances start "$VM" --zone "$zone" --quiet
else
  for c in $CANDIDATES; do
    machine=${c%%:*}; z=${c##*:}
    echo "== trying Spot $machine in $z"
    nic=(); [[ "$machine" == a3-* ]] && nic=(--network-interface=nic-type=GVNIC)
    if gcloud compute instances create "$VM" --zone "$z" --machine-type "$machine" "${nic[@]}" \
        --image-project deeplearning-platform-release --image-family "$IMAGE_FAMILY" \
        --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
        --provisioning-model SPOT --instance-termination-action STOP --max-run-duration "${MAX_HOURS}h" \
        --maintenance-policy TERMINATE --tags chotu \
        --metadata install-nvidia-driver=True --metadata-from-file startup-script="$startup" --quiet; then
      zone=$z; break
    fi
  done
  [[ -n "$zone" ]] || { echo "no Spot GPU available in any candidate zone right now; try again later" >&2; exit 3; }
fi
echo "$VM $zone" > "$HOME/.chotu-gcp-vm"

# ssh/rsync through IAP with the Mac's gcloud key
cfg=$(mktemp)
cat > "$cfg" <<EOF
Host chotu-gcp
  HostName $VM
  User $user
  IdentityFile ~/.ssh/google_compute_engine
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
  ServerAliveInterval 30
  ProxyCommand env CLOUDSDK_ACTIVE_CONFIG_NAME=$CLOUDSDK_ACTIVE_CONFIG_NAME $(command -v gcloud) compute start-iap-tunnel $VM 22 --listen-on-stdin --zone $zone --verbosity=warning
EOF
echo "== waiting for SSH (adds the Mac's key to the VM on first use)"
for _ in $(seq 1 40); do
  gcloud compute ssh "$VM" --zone "$zone" --tunnel-through-iap --quiet --command true >/dev/null 2>&1 && break
  sleep 10
done
s=(ssh -F "$cfg" chotu-gcp)
"${s[@]}" true

echo "== folders and tools"
"${s[@]}" "sudo mkdir -p $remote_repo $storage/chat $storage/reports && sudo chown $user:$user $remote_repo $storage $storage/chat $storage/reports && (command -v tmux >/dev/null && command -v rsync >/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq tmux rsync) >/dev/null)"

echo "== code ($(git rev-parse --short HEAD)$(git diff --quiet || echo ', with local edits'))"
git ls-files -z | rsync -a --from0 --files-from=- -e "ssh -F $cfg" ./ "chotu-gcp:$remote_repo/"

echo "== .env (values not shown; no Jarvis keys)"
{
  echo "HF_TOKEN=$HF_TOKEN"
  echo "CHOTU_STORAGE=$storage"
  echo "GIT_NAME=$name"
  echo "GIT_EMAIL=$email"
} | "${s[@]}" "umask 077; cat > $remote_repo/.env"

echo "== chat data (.done goes last so the VM never sees a half copy)"
rsync -a --partial --exclude .done -e "ssh -F $cfg" "$chat/chat/" "chotu-gcp:$storage/chat/"
rsync -a -e "ssh -F $cfg" "$chat/reports/chat_report.md" "chotu-gcp:$storage/reports/"
rsync -a -e "ssh -F $cfg" "$chat/chat/.done" "chotu-gcp:$storage/chat/"

echo "== start gcp_up.sh in tmux session 'chotu'"
"${s[@]}" "cd $remote_repo && (tmux has-session -t chotu 2>/dev/null && echo 'already running') || tmux new -d -s chotu 'bash scripts/gcp_up.sh 2>&1 | tee -a $storage/gcp_up.log; exec bash'"

sleep 5
ip=$(gcloud compute instances describe "$VM" --zone "$zone" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
key=$("${s[@]}" "cat $storage/dashboard/key 2>/dev/null" || true)
echo "== DASHBOARD (works with the Mac off): http://$ip:6006/${key:-<key appears in $storage/dashboard/key>}/"
echo "Watch:  gcloud compute ssh $VM --zone $zone --tunnel-through-iap -- tmux attach -t chotu"
echo "Spot restarts: run tools/gcp_watchdog.sh on the Mac"
