#!/usr/bin/env bash
# One-time Mac setup: run from the repo root with  bash scripts/setup_mac.sh
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt jupyterlab ipykernel
python -m ipykernel install --user --name chotu --display-name "Python (chotu)"
[ -f .env ] || cp .env.example .env
python -m pytest -q tests
echo "Done. Start Jupyter with:  source .venv/bin/activate && jupyter lab notebooks/session2_data_local.ipynb"
