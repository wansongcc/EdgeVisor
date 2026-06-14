#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "0,1,2" ]]; then
  echo "Refusing to run: CUDA_VISIBLE_DEVICES must be 0,1,2 to avoid GPU3" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/home/byh/B01/agent_langgraph_venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

args=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  args+=(--smoke)
fi
if [[ -n "${EPISODE:-}" ]]; then
  args+=(--episode "${EPISODE}")
fi
if [[ -n "${OUT_ROOT:-}" ]]; then
  args+=(--out-root "${OUT_ROOT}")
fi
if [[ -n "${REPEATS:-}" ]]; then
  args+=(--repeats "${REPEATS}")
fi
if [[ -n "${VARIANTS:-}" ]]; then
  args+=(--variants "${VARIANTS}")
fi
if [[ -n "${FLUCTUATIONS:-}" ]]; then
  args+=(--fluctuations "${FLUCTUATIONS}")
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_real_ablation_suite.py" "${args[@]}"
