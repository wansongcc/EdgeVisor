#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "0,1,2" ]]; then
  echo "Refusing to run: CUDA_VISIBLE_DEVICES must be 0,1,2 to avoid GPU3" >&2
  exit 2
fi

EPISODE="${EPISODE:-${PROJECT_ROOT}/agent_bench/episodes/agentic_ablation_episode.json}"
OUT_ROOT="${OUT_ROOT:-/home/byh/B01/agentic_ablation_results/$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
BACKEND="${BACKEND:-edgevisor_ablation}"
CTX="${CTX:-2048}"
EDGE_WORKER_GPUS="${EDGE_WORKER_GPUS:-1,2}"
EDGE_RATIOS="${EDGE_RATIOS:-1:1:1}"
EDGE_STEPS="${EDGE_STEPS:-256}"
PYTHON_BIN="${PYTHON_BIN:-/home/byh/B01/agent_langgraph_venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

mkdir -p "${OUT_ROOT}"

variants=(
  "full:enabled:enabled:enabled:enabled"
  "shadow_disabled_transfer:disabled_transfer:enabled:enabled:enabled"
  "shadow_disabled_recompute:disabled_recompute:enabled:enabled:enabled"
  "pointer_operator_rebuild:enabled:operator_rebuild:enabled:enabled"
  "pointer_weight_rematerialize:enabled:weight_rematerialize:enabled:enabled"
  "jit_static:enabled:enabled:static:enabled"
  "jit_greedy:enabled:enabled:greedy:enabled"
  "vg_flat:enabled:enabled:enabled:flat"
  "vg_random:enabled:enabled:enabled:random"
)

fluctuations=(compute network mixed)

if [[ "${SMOKE:-0}" == "1" ]]; then
  REPEATS="${SMOKE_REPEATS:-1}"
  variants=(
    "full:enabled:enabled:enabled:enabled"
    "shadow_disabled_transfer:disabled_transfer:enabled:enabled:enabled"
    "shadow_disabled_recompute:disabled_recompute:enabled:enabled:enabled"
    "pointer_operator_rebuild:enabled:operator_rebuild:enabled:enabled"
    "jit_static:enabled:enabled:static:enabled"
    "vg_flat:enabled:enabled:enabled:flat"
  )
  fluctuations=(mixed)
fi

manifest="${OUT_ROOT}/manifest.jsonl"
: > "${manifest}"

for variant in "${variants[@]}"; do
  IFS=: read -r name shadow pointer jit vg <<< "${variant}"
  for fluct in "${fluctuations[@]}"; do
    for repeat in $(seq 1 "${REPEATS}"); do
      run_root="${OUT_ROOT}/${name}/${fluct}/rep_${repeat}"
      mkdir -p "${run_root}"
      episode_copy="${run_root}/episode.json"
      "${PYTHON_BIN}" - "$EPISODE" "$episode_copy" "$fluct" <<'PY'
import json
import sys
src, dst, fluct = sys.argv[1:4]
episode = json.load(open(src, encoding="utf-8"))
plan = dict(episode.get("edgevisor_dynamic_plan", {}))
plan["fluctuation_type"] = fluct
if fluct == "compute":
    plan["simulated_stall_ms"] = 8.0
elif fluct == "network":
    plan["simulated_stall_ms"] = 12.0
else:
    plan["simulated_stall_ms"] = 16.0
plan["simulated_recovery_latency_ms"] = plan["simulated_stall_ms"]
episode["edgevisor_dynamic_plan"] = plan
json.dump(episode, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
PY
      echo "{\"variant\":\"${name}\",\"fluctuation\":\"${fluct}\",\"repeat\":${repeat},\"run_root\":\"${run_root}\"}" >> "${manifest}"
      "${PYTHON_BIN}" -m agent_bench.run_loop_episode \
        --backend "${BACKEND}" \
        --episode "${episode_copy}" \
        --out-root "${run_root}" \
        --cuda-visible "${CUDA_VISIBLE_DEVICES}" \
        --edge-worker-gpus "${EDGE_WORKER_GPUS}" \
        --edge-ratios "${EDGE_RATIOS}" \
        --edge-steps "${EDGE_STEPS}" \
        --ctx "${CTX}" \
        --shadow-kv-mode "${shadow}" \
        --pointer-swizzling-mode "${pointer}" \
        --jit-mode "${jit}" \
        --vg-mode "${vg}" \
        --experiment-id "${name}_${fluct}_rep${repeat}" \
        | tee "${run_root}/run_stdout.json"
    done
  done
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/summarize_ablation_results.py" "${OUT_ROOT}"
echo "Ablation suite complete: ${OUT_ROOT}"
