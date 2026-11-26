#!/usr/bin/env bash
# Mac: finish the run on one Nebius Spot (preemptible) GPU VM, then the VM powers itself off.
#
#   bash scripts/launch_nebius.sh plan     # write and print the VM spec (creates nothing)
#   bash scripts/launch_nebius.sh create   # create the VM from that spec (costs money; run once)
#   bash scripts/launch_nebius.sh setup    # copy code + .env + chat data, start scripts/gcp_up.sh
#
# The VM runs scripts/gcp_up.sh (restore the HF backup, put back recorded file times, resume the chain,
# final backup, shutdown). SSH listens on 22 and 443, so setup also works on networks that block 22.
# After a Spot preemption the VM is stopped; tools/nebius_watchdog.sh starts it again and an
# @reboot job resumes the run. Needs: nebius CLI profile, .env with HF_TOKEN, ~/chotu-chat.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export AI_AGENT=${AI_AGENT:-claude-code}
STATE=$HOME/.chotu-nebius; mkdir -p "$STATE"
PROJECT=${PROJECT:-$(nebius config get parent-id)}
NAME=${NAME:-chotu-h100}
PLATFORM=${PLATFORM:-gpu-h100-sxm}
PRESET=${PRESET:-1gpu-16vcpu-200gb}
IMAGE_FAMILY=${IMAGE_FAMILY:-ubuntu24.04-cuda13.0}   # NVIDIA 580 + CUDA 13: the run's torch 2.11+cu130
DISK_GB=${DISK_GB:-250}
PUBKEY=${PUBKEY:-$HOME/.ssh/id_ed25519.pub}
user=$(whoami | tr '[:upper:]' '[:lower:]')
storage=/home/chotu-data   # must stay /home/chotu-data: the run's recorded settings name this path
remote_repo=/home/Chotu
chat=${CHAT_DIR:-$HOME/chotu-chat}

plan() {
  subnet=$(nebius vpc subnet list --parent-id "$PROJECT" --format json | jq -r '.items[0].metadata.id')
  region=$(nebius iam project get --id "$PROJECT" --format json 2>/dev/null | jq -r '.status.region // empty')
  image=$(nebius compute image list-public --region "${region:-eu-north1}" --format json |
    jq -r --arg f "$IMAGE_FAMILY" '[.items[] | select(.spec.image_family==$f and .status.state=="READY")] | sort_by(.metadata.name) | last | .metadata.id')
  [[ -f "$PUBKEY" ]] || { echo "no SSH public key at $PUBKEY" >&2; exit 2; }
  cat > "$STATE/cloud-init.yaml" <<EOF
#cloud-config
users:
  - name: $user
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat "$PUBKEY")
packages: [tmux, rsync, jq]
write_files:
  - path: /etc/ssh/sshd_config.d/99-chotu-ports.conf
    content: |
      Port 22
      Port 443
  - path: /etc/cron.d/chotu-resume
    content: |
      @reboot $user sleep 30; [ -f $remote_repo/.env ] && cd $remote_repo && tmux new -d -s chotu "bash scripts/gcp_up.sh 2>&1 | tee -a $storage/gcp_up.log; exec bash"
runcmd:
  - mkdir -p $remote_repo $storage && chown $user:$user $remote_repo $storage
  - systemctl daemon-reload && (systemctl restart ssh.socket || true) && systemctl restart ssh
EOF
  cat > "$STATE/create.sh" <<EOF
nebius compute instance create --format json \\
  --parent-id $PROJECT --name $NAME \\
  --resources-platform $PLATFORM --resources-preset $PRESET \\
  --boot-disk-attach-mode read_write \\
  --boot-disk-managed-disk-name $NAME-boot --boot-disk-managed-disk-type network_ssd \\
  --boot-disk-managed-disk-size-gibibytes $DISK_GB --boot-disk-managed-disk-source-image-id $image \\
  --network-interfaces '[{"name":"eth0","subnet_id":"$subnet","ip_address":{},"public_ip_address":{}}]' \\
  --follows-spot-price --preemptible-on-preemption STOP \\
  --cloud-init-user-data "\$(cat $STATE/cloud-init.yaml)"
EOF
  echo "== cloud-init ($STATE/cloud-init.yaml):"; cat "$STATE/cloud-init.yaml"
  echo "== command ($STATE/create.sh):"; cat "$STATE/create.sh"
}

create() {
  [[ -f "$STATE/create.sh" ]] || { echo "run 'plan' first" >&2; exit 2; }
  bash "$STATE/create.sh" > "$STATE/created.json"
  jq -r '.resource_id // .metadata.id // empty' "$STATE/created.json" > "$STATE/id" || true
  [[ -s "$STATE/id" ]] || nebius compute instance get-by-name --parent-id "$PROJECT" --name "$NAME" --format json | jq -r '.metadata.id' > "$STATE/id"
  echo "== created $(cat "$STATE/id")"
}

ip() {
  nebius compute instance get --id "$(cat "$STATE/id")" --format json |
    jq -r '.status.network_interfaces[0].public_ip_address.address // empty' | cut -d/ -f1
}

setup() {
  addr=$(ip); [[ -n "$addr" ]] || { echo "no public IP yet" >&2; exit 2; }
  echo "$addr" > "$STATE/ip"
  set -a; source .env; set +a
  [[ -n "${HF_TOKEN:-}" ]] || { echo "add HF_TOKEN to your Mac .env first" >&2; exit 2; }
  name=$(git config user.name); email=$(git config user.email)
  port=443; nc -z -G 5 "$addr" 22 2>/dev/null && port=22
  s=(ssh -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 "$user@$addr")
  echo "== waiting for SSH on $addr:$port (cloud-init installs packages first)"
  for _ in $(seq 1 60); do "${s[@]}" -o ConnectTimeout=8 'test -f /var/lib/cloud/instance/boot-finished' 2>/dev/null && break; sleep 10; done
  "${s[@]}" "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; mkdir -p $storage/chat $storage/reports"
  echo "== code ($(git rev-parse --short HEAD)$(git diff --quiet || echo ', with local edits'))"
  git ls-files -z | rsync -a --from0 --files-from=- -e "ssh -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" ./ "$user@$addr:$remote_repo/"
  echo "== .env (values not shown; no Jarvis keys)"
  printf 'HF_TOKEN=%s\nCHOTU_STORAGE=%s\nGIT_NAME=%s\nGIT_EMAIL=%s\n' "$HF_TOKEN" "$storage" "$name" "$email" | "${s[@]}" "umask 077; cat > $remote_repo/.env"
  echo "== chat data (.done goes last)"
  r=(rsync -a --partial -e "ssh -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR")
  "${r[@]}" --exclude .done "$chat/chat/" "$user@$addr:$storage/chat/"
  "${r[@]}" "$chat/reports/chat_report.md" "$user@$addr:$storage/reports/"
  "${r[@]}" "$chat/chat/.done" "$user@$addr:$storage/chat/"
  echo "== start gcp_up.sh in tmux session 'chotu'"
  "${s[@]}" "cd $remote_repo && (tmux has-session -t chotu 2>/dev/null && echo 'already running') || tmux new -d -s chotu 'bash scripts/gcp_up.sh 2>&1 | tee -a $storage/gcp_up.log; exec bash'"
  sleep 5
  key=$("${s[@]}" "cat $storage/dashboard/key 2>/dev/null" || true)
  echo "== DASHBOARD: http://$addr:6006/${key:-<key in $storage/dashboard/key>}/"
  echo "Watch: ssh -p $port $user@$addr -t tmux attach -t chotu"
}

case "${1:-}" in
  plan) plan ;; create) create ;; setup) setup ;; ip) ip ;;
  *) echo "usage: $0 plan|create|setup|ip" >&2; exit 2 ;;
esac
