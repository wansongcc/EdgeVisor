#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../config/env.sh
source "${PROJECT_ROOT}/config/env.sh"
BASE="${EDGEVISOR_ENGINE_DIR}"
MODEL3="${EDGEVISOR_MODEL3}"
TOK="${EDGEVISOR_TOKENIZER}"
MODE="${1:-static}"
DYN_STEPS="${DYN_STEPS:-8}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${EDGEVISOR_LOG_ROOT}/gpu/gpu_patch_${MODE}_${STAMP}"
mkdir -p "${LOG_DIR}"
cd "${BASE}" || exit 2

PORT1=19101
PORT2=19102
SOCK=/tmp/dllama_gpu_patch_plan.sock

cleanup() {
  if [[ -n "${P1:-}" ]]; then kill "${P1}" 2>/dev/null || true; fi
  if [[ -n "${P2:-}" ]]; then kill "${P2}" 2>/dev/null || true; fi
  if [[ -n "${ROOT_PID:-}" ]]; then kill "${ROOT_PID}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  rm -f "${SOCK}"
}
trap cleanup EXIT

./dllama worker --port "${PORT1}" --nthreads 1 --gpu-index 1 >"${LOG_DIR}/worker_gpu1.log" 2>&1 &
P1=$!
./dllama worker --port "${PORT2}" --nthreads 1 --gpu-index 2 >"${LOG_DIR}/worker_gpu2.log" 2>&1 &
P2=$!
sleep 4

common_args=(
  inference
  --prompt "Hi"
  --steps 16
  --model "${MODEL3}"
  --tokenizer "${TOK}"
  --buffer-float-type q80
  --nthreads 1
  --max-seq-len 512
  --gpu-index 0
  --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}"
  --ratios "2:3:3"
)

if [[ "${MODE}" == "dynamic" ]]; then
  rm -f "${SOCK}"
  DLLAMA_PLAN_CTRL_SOCKET="${SOCK}" ./dllama "${common_args[@]}" \
    --steps "${DYN_STEPS}" \
    --enable-plan-barrier \
    --enable-kv-redundancy-during-migration 1 \
    --kv-redundancy 2 >"${LOG_DIR}/root_gpu0.log" 2>&1 &
  ROOT_PID=$!

  for i in $(seq 1 80); do
    if [[ -S "${SOCK}" ]]; then break; fi
    sleep 0.25
  done
  python3 examples/plan-uds-client.py "${SOCK}" ping >"${LOG_DIR}/uds_ping.json" 2>&1 || true
  python3 examples/plan-uds-client.py "${SOCK}" set_plan \
    --seq 101 \
    --mode next_barrier \
    --stage 0 \
    --from 1 \
    --to 2 \
    --kind 1 \
    --heads 1 \
    --ffn 0 >"${LOG_DIR}/uds_set_plan.json" 2>&1 || true
  wait "${ROOT_PID}"
  RC=$?
else
  ./dllama "${common_args[@]}" >"${LOG_DIR}/root_gpu0.log" 2>&1
  RC=$?
fi

echo "LOG_DIR=${LOG_DIR}"
echo "RC=${RC}"
echo "--- root tail ---"
tail -n 80 "${LOG_DIR}/root_gpu0.log" || true
echo "--- worker1 tail ---"
tail -n 20 "${LOG_DIR}/worker_gpu1.log" || true
echo "--- worker2 tail ---"
tail -n 20 "${LOG_DIR}/worker_gpu2.log" || true
exit "${RC}"
