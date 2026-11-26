#!/usr/bin/env bash
# Mac: restart the GCP Spot VM after a preemption, so the run resumes from its last checkpoint
# (the VM's startup script starts scripts/gcp_up.sh again). Never restarts after a deliberate stop:
# the run finishing or failing (gcp_up.sh powers off), the max-run-duration cap, or a manual stop.
#
#   caffeinate -i bash tools/gcp_watchdog.sh        # Ctrl-C to quit; touch ~/.chotu-gcp-hold to pause it
set -uo pipefail
export CLOUDSDK_ACTIVE_CONFIG_NAME=${CLOUDSDK_ACTIVE_CONFIG_NAME:-chotu}
export PATH=/opt/homebrew/share/google-cloud-sdk/bin:$PATH
read -r VM ZONE < "$HOME/.chotu-gcp-vm" || { echo "no ~/.chotu-gcp-vm: run scripts/launch_gcp.sh first" >&2; exit 2; }
HOLD=$HOME/.chotu-gcp-hold
EVERY=${EVERY:-60}
MAX_RESTARTS=${MAX_RESTARTS:-6}
restarts=0
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
newest_op() {  # newest insertTime of an operation type on this VM (epoch seconds, 0 if none)
  gcloud compute operations list --zones "$ZONE" --sort-by=~insertTime --limit 1 --format="value(insertTime)" \
    --filter="targetLink~/instances/$VM\$ AND operationType=$1" 2>/dev/null |
    python3 -c 'import sys; from datetime import datetime; t=sys.stdin.read().strip(); print(int(datetime.fromisoformat(t).timestamp()) if t else 0)'
}
log "watching $VM in $ZONE every ${EVERY}s (hold file: $HOLD)"
last=""
while true; do
  status=$(gcloud compute instances describe "$VM" --zone "$ZONE" --format="value(status)" 2>/dev/null || echo GONE)
  [[ "$status" != "$last" ]] && log "VM is $status" && last=$status
  if [[ "$status" == GONE ]]; then log "VM no longer exists; exiting"; exit 0; fi
  if [[ "$status" == TERMINATED ]]; then
    if [[ -f "$HOLD" ]]; then sleep "$EVERY"; continue; fi
    preempted=$(newest_op compute.instances.preempted)
    started=$(( $(newest_op start) > $(newest_op insert) ? $(newest_op start) : $(newest_op insert) ))
    if (( preempted > started )); then
      if (( restarts >= MAX_RESTARTS )); then log "preempted again but $MAX_RESTARTS restarts already; stopping"; exit 1; fi
      log "Spot preemption detected: starting the VM again (the run resumes from its last checkpoint)"
      if gcloud compute instances start "$VM" --zone "$ZONE" --quiet; then
        restarts=$(( restarts + 1 ))
        osascript -e 'display notification "Spot VM restarted after preemption" with title "Chotu"' 2>/dev/null || true
      else
        log "no Spot capacity right now; retrying"
      fi
    else
      log "VM stopped on purpose (run finished/failed, time cap, or manual): not restarting; exiting"
      osascript -e 'display notification "GCP VM stopped (finished or deliberate)" with title "Chotu"' 2>/dev/null || true
      exit 0
    fi
  fi
  sleep "$EVERY"
done
