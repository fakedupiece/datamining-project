#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPOSITORY_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing .venv. Run: python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPOSITORY_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${REPOSITORY_ROOT}/.cache"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cd "${REPOSITORY_ROOT}"
"${PYTHON_BIN}" -m src.download_data --config configs/experiment.yaml
"${PYTHON_BIN}" -m src.data_audit --config configs/experiment.yaml
"${PYTHON_BIN}" -m src.preprocess --config configs/experiment.yaml
"${PYTHON_BIN}" -m src.eda --config configs/experiment.yaml
"${PYTHON_BIN}" -m src.clustering_tendency --config configs/experiment.yaml
"${PYTHON_BIN}" -m pytest -q
