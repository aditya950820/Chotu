#!/usr/bin/env bash
# Mac: start the Nebius Spot VM again after a preemption (Nebius stops it), so the run resumes from
# its last checkpoint (an @reboot job on the VM starts scripts/gcp_up.sh again). Never restarts after
# a deliberate stop: scripts/gcp_up.sh appends to dashboard/gcp-exits.log and backs it up to HF just
# before it powers off, so a STOPPED VM with a newer exit line than the last time we saw it RUNNING
# stopped on purpose (run finished or failed).
#
#   caffeinate -i bash tools/nebius_watchdog.sh     # Ctrl-C to quit; touch ~/.chotu-nebius/hold to pause it
set -uo pipefail
cd "$(dirname "$0")/.."
export AI_AGENT=${AI_AGENT:-claude-code}
STATE=$HOME/.chotu-nebius
ID=$(cat "$STATE/id" 2>/dev/null) || { echo "no $STATE/id: run scripts/launch_nebius.sh create first" >&2; exit 2; }
EVERY=${EVERY:-60}; MAX_RESTARTS=${MAX_RESTARTS:-6}; restarts=0
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
state() { nebius compute instance get --id "$ID" --format json 2>/dev/null | jq -r '.status.state // "GONE"'; }
last_exit() {  # epoch seconds of the newest deliberate exit recorded in the HF backup (0 if none)
  .venv/bin/python - <<'EOF' 2>/dev/null || echo 0
import datetime, os
from dotenv import load_dotenv; load_dotenv(".env")
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download("magnateadi/chotu-run-backup", "storage/dashboard/gcp-exits.log", force_download=True)
    t = [l for l in open(p).read().splitlines() if l.strip()][-1].split()[0]
    print(int(datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()))
except Exception:
    print(0)
EOF
}
log "watching $ID every ${EVERY}s (hold file: $STATE/hold)"
seen_running=$(date +%s); last=""
while true; do
  s=$(state)
  [[ "$s" != "$last" ]] && log "VM is $s" && last=$s
  [[ "$s" == RUNNING || "$s" == STARTING ]] && seen_running=$(date +%s)
  if [[ "$s" == GONE ]]; then log "VM no longer exists; exiting"; exit 0; fi
  if [[ "$s" == STOPPED && ! -f "$STATE/hold" ]]; then
    if (( $(last_exit) >= seen_running - 300 )); then
      log "stopped on purpose (gcp_up.sh recorded an exit): not restarting; exiting"
      osascript -e 'display notification "Nebius VM stopped (finished or failed on purpose)" with title "Chotu"' 2>/dev/null || true
      exit 0
    fi
    (( restarts >= MAX_RESTARTS )) && { log "preempted again after $MAX_RESTARTS restarts; stopping"; exit 1; }
    log "preempted: starting the VM again (the run resumes from its last checkpoint)"
    if nebius compute instance start --id "$ID" --format json >/dev/null; then
      restarts=$(( restarts + 1 )); seen_running=$(date +%s)
      osascript -e 'display notification "Nebius Spot VM restarted after preemption" with title "Chotu"' 2>/dev/null || true
    else
      log "start failed (no Spot capacity?); retrying"
    fi
  fi
  sleep "$EVERY"
done
