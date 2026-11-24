#!/usr/bin/env bash
# Put keys into the gitignored .env without them appearing on screen or in shell history.
#   bash scripts/set_keys.sh                 # asks for HF_TOKEN and JL_API_KEY
#   bash scripts/set_keys.sh WANDB_API_KEY   # or any keys you name
set -euo pipefail
cd "$(dirname "$0")/.."
touch .env && chmod 600 .env
keys=("$@"); (( ${#keys[@]} )) || keys=(HF_TOKEN JL_API_KEY)
for k in "${keys[@]}"; do
  read -rsp "$k (typing is hidden, Enter when done): " v; echo
  [[ -n "$v" ]] || { echo "  empty, $k left unchanged"; continue; }
  grep -v "^$k=" .env > .env.tmp || true
  echo "$k=$v" >> .env.tmp && mv .env.tmp .env && chmod 600 .env
  echo "  $k saved (${#v} characters)"
done
