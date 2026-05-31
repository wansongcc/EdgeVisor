#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../config/env.sh
source "${PROJECT_ROOT}/config/env.sh"
BASE="${EDGEVISOR_ENGINE_DIR}"
MODEL3="${EDGEVISOR_MODEL3}"
TOK="${EDGEVISOR_TOKENIZER}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${EDGEVISOR_LOG_ROOT}/gpu/gpu_pp_migration_${STAMP}"
mkdir -p "${LOG_DIR}"
cd "${BASE}" || exit 2

PORT1=19301
PORT2=19302
SOCK=/tmp/dllama_gpu_pp_migration.sock

cleanup() {
  if [[ -n "${P1:-}" ]]; then kill "${P1}" 2>/dev/null || true; fi
  if [[ -n "${P2:-}" ]]; then kill "${P2}" 2>/dev/null || true; fi
  if [[ -n "${ROOT_PID:-}" ]]; then kill "${ROOT_PID}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  rm -f "${SOCK}"
}
trap cleanup EXIT

rm -f "${SOCK}"

./dllama worker --port "${PORT1}" --nthreads 1 --gpu-index 1 >"${LOG_DIR}/worker_gpu1.log" 2>&1 &
P1=$!
./dllama worker --port "${PORT2}" --nthreads 1 --gpu-index 2 >"${LOG_DIR}/worker_gpu2.log" 2>&1 &
P2=$!
sleep 4

DLLAMA_PLAN_CTRL_SOCKET="${SOCK}" ./dllama inference \
  --prompt "Hi" \
  --steps 32 \
  --model "${MODEL3}" \
  --tokenizer "${TOK}" \
  --buffer-float-type q80 \
  --nthreads 1 \
  --max-seq-len 512 \
  --gpu-index 0 \
  --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}" \
  --ratios "1@8*1@10*1@10" \
  --enable-pp-migration \
  --enable-kv-redundancy-during-migration 1 \
  --kv-redundancy 2 >"${LOG_DIR}/root_gpu0.log" 2>&1 &
ROOT_PID=$!

for i in $(seq 1 100); do
  if [[ -S "${SOCK}" ]]; then break; fi
  sleep 0.25
done

python3 examples/plan-uds-client.py "${SOCK}" ping >"${LOG_DIR}/uds_ping.json" 2>&1 || true
python3 examples/plan-uds-client.py "${SOCK}" set_pp_migration \
  --seq 301 \
  --mode next_barrier \
  --from 0 \
  --to 1 \
  --layer-count 1 \
  --trigger-pos 0 >"${LOG_DIR}/uds_set_pp_migration.json" 2>&1 || true

wait "${ROOT_PID}"
RC=$?

echo "LOG_DIR=${LOG_DIR}"
echo "RC=${RC}"
echo "--- root evidence ---"
grep -E "kv-migrate|kv-export-gpu|kv-write-gpu|worker-switch|layer-gate|pp migration|set_pp_migration|plan" "${LOG_DIR}/root_gpu0.log" | tail -n 160 || true
echo "--- worker1 evidence ---"
grep -E "kv-migrate|kv-export-gpu|kv-write-gpu|worker-switch|layer-gate|pp migration|plan" "${LOG_DIR}/worker_gpu1.log" | tail -n 160 || true
echo "--- worker2 evidence ---"
grep -E "kv-migrate|kv-export-gpu|kv-write-gpu|worker-switch|layer-gate|pp migration|plan" "${LOG_DIR}/worker_gpu2.log" | tail -n 80 || true
echo "--- root tail ---"
tail -n 80 "${LOG_DIR}/root_gpu0.log" || true
exit "${RC}"
