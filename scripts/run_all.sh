#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
no_stop=0
dry_run=0
for arg in "$@"; do
  [[ "$arg" != --no-stop ]] || no_stop=1
  [[ "$arg" != --dry-run ]] || dry_run=1
done
if (( dry_run )); then
  exec python scripts/run_chain.py "$@"
fi
owned=0
req=""
constraint=""
cleanup() {
  code=$?
  trap - EXIT
  [[ -z "$req" ]] || rm -f "$req"
  [[ -z "$constraint" ]] || rm -f "$constraint"
  if (( ! owned && ! no_stop )); then
    python scripts/stop_machine.py || { echo 'PAUSE FAILED AFTER SETUP ERROR: PAUSE MANUALLY NOW' >&2; code=25; }
  fi
  exit "$code"
}
trap cleanup EXIT
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade
python -m pip install python-dotenv jarvislabs
req=$(mktemp)
constraint=$(mktemp)
python -c 'from pathlib import Path; print("\n".join(x for x in Path("requirements.txt").read_text().splitlines() if not x.startswith("torch==")))' > "$req"
python -c 'import torch; print("torch==" + torch.__version__)' > "$constraint"
# liger-kernel = the frozen "fused cross-entropy"; without it the loss path is several times slower.
python -m pip install -r "$req" -c "$constraint" jarvislabs liger-kernel==0.8.5
owned=1
python scripts/run_chain.py "$@"
