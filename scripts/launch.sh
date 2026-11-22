#!/usr/bin/env bash
# THE one command, run on the Mac once the H200 is up:
#
#   bash scripts/launch.sh <user@host> <ssh-port> <jarvis-machine-id>
#
# Copies the code (exactly this checkout's tracked files, no GitHub login needed), writes the
# machine's .env from your Mac .env without printing secrets, starts scripts/gpu_up.sh in tmux,
# and copies the Mac-prepared chat data in parallel. Rerun the same command after a Spot
# interruption: everything resumes.
# Watch:  ssh -p <port> <user@host> -t tmux attach -t chotu     (detach: Ctrl-B then D)
set -Eeuo pipefail
cd "$(dirname "$0")/.."
[[ $# -eq 3 ]] || { sed -n 2,12p "$0"; exit 2; }
target=$1; port=$2; machine=$3
storage=${REMOTE_STORAGE:-/home/chotu-data}
remote_repo=${REMOTE_REPO:-/home/Chotu}
chat=${CHAT_DIR:-$HOME/chotu-chat}
ssh_cmd=(ssh -p "$port" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$target")

set -a; source .env; set +a
for key in HF_TOKEN WANDB_API_KEY JL_API_KEY; do
  [[ -n "${!key:-}" ]] || { echo "add $key to your Mac .env first" >&2; exit 2; }
done
[[ -f "$chat/chat/.done" && -f "$chat/reports/chat_report.md" ]] || { echo "chat data not ready in $chat" >&2; exit 2; }
name=$(git config user.name); email=$(git config user.email)
[[ -n "$name" && -n "$email" ]] || { echo "set git user.name/user.email on the Mac" >&2; exit 2; }

echo "== remote tools"
"${ssh_cmd[@]}" "command -v rsync >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync tmux) >/dev/null; mkdir -p '$remote_repo' '$storage/chat' '$storage/reports'"

echo "== code ($(git rev-parse --short HEAD)$(git diff --quiet || echo ', with local edits'))"
git ls-files -z | rsync -a --from0 --files-from=- -e "ssh -p $port" ./ "$target:$remote_repo/"

echo "== .env (values not shown)"
{
  grep -v -E '^(JL_MACHINE_ID|CHOTU_STORAGE|GIT_NAME|GIT_EMAIL)=' .env
  echo "JL_MACHINE_ID=$machine"
  echo "CHOTU_STORAGE=$storage"
  echo "GIT_NAME=$name"
  echo "GIT_EMAIL=$email"
} | "${ssh_cmd[@]}" "umask 077; cat > '$remote_repo/.env'"

echo "== chat data copy (background; .done goes last so the machine never sees a half copy)"
(
  rsync -a --partial --exclude .done -e "ssh -p $port" "$chat/chat/" "$target:$storage/chat/"
  rsync -a -e "ssh -p $port" "$chat/reports/chat_report.md" "$target:$storage/reports/"
  rsync -a -e "ssh -p $port" "$chat/chat/.done" "$target:$storage/chat/"
) > /tmp/chotu-chat-copy.log 2>&1 &
copy=$!

echo "== start gpu_up.sh in tmux session 'chotu'"
"${ssh_cmd[@]}" "cd '$remote_repo' && (tmux has-session -t chotu 2>/dev/null && echo 'already running; attach to watch') || tmux new -d -s chotu 'bash scripts/gpu_up.sh 2>&1 | tee -a $storage/gpu_up.log; exec bash'"

wait "$copy" && echo "== chat data copied" || { echo "chat copy failed (see /tmp/chotu-chat-copy.log); the machine will prepare it itself after 40 min"; }
echo "Watch: ssh -p $port $target -t tmux attach -t chotu"
