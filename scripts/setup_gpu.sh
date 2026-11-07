#!/usr/bin/env bash
# One-time Python setup on a bare Jarvis Labs GPU VM (Ubuntu 22.04, Python 3.10, no pip, no torch).
# Our pins need Python 3.12, so this builds ~/venv312 with uv, the CUDA build of torch 2.14.1 and requirements.txt.
# Found on twin day (2026-11-07): the VM shipped without torch, so setup failed until this ran.
#
#   bash scripts/setup_gpu.sh
#   echo 'PATH=/home/ubuntu/venv312/bin:$PATH' >> .env     # twin_day.sh sources .env, so it then uses this Python
set -euo pipefail
cd "$(dirname "$0")/.."
CU=${CU:-cu130}   # match the driver: nvidia-smi shows "CUDA Version: 13.0" on the H100 VMs
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >/dev/null
python3 -m venv ~/venv && ~/venv/bin/pip install -q -U pip uv
~/venv/bin/uv venv -q --python 3.12 ~/venv312
export VIRTUAL_ENV=~/venv312
~/venv/bin/uv pip install -q torch==2.14.1 --index-url "https://download.pytorch.org/whl/$CU"
grep -vE '^torch==' requirements.txt > /tmp/req.txt
~/venv/bin/uv pip install -q -r /tmp/req.txt jarvislabs pip
~/venv312/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__, 'cuda', torch.cuda.is_available())"
