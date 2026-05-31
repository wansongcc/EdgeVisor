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
LOG_DIR="${EDGEVISOR_LOG_ROOT}/semantic/cpu_dynamic_heads_${STAMP}"
mkdir -p "${LOG_DIR}"
cd "${BASE}" || exit 2
PORT1=19711
PORT2=19712
SOCK=/tmp/dllama_semantic_cpu_heads.sock
PROMPT="Write a comma-separated list of the numbers from 1 to 20."
cleanup() {
  if [[ -n "${P1:-}" ]]; then kill "${P1}" 2>/dev/null || true; fi
  if [[ -n "${P2:-}" ]]; then kill "${P2}" 2>/dev/null || true; fi
  if [[ -n "${ROOT_PID:-}" ]]; then kill "${ROOT_PID}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  rm -f "${SOCK}"
}
trap cleanup EXIT
rm -f "${SOCK}"
./dllama worker --port "${PORT1}" --nthreads 1 >"${LOG_DIR}/worker_cpu1.log" 2>&1 &
P1=$!
./dllama worker --port "${PORT2}" --nthreads 1 >"${LOG_DIR}/worker_cpu2.log" 2>&1 &
P2=$!
sleep 4
DLLAMA_PLAN_CTRL_SOCKET="${SOCK}" ./dllama inference \
  --prompt "${PROMPT}" \
  --steps 64 \
  --model "${MODEL3}" \
  --tokenizer "${TOK}" \
  --buffer-float-type q80 \
  --nthreads 1 \
  --max-seq-len 512 \
  --temperature 0 \
  --seed 1 \
  --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}" \
  --ratios "2:3:3" \
  --enable-stage-full-weights \
  --enable-plan-barrier \
  --enable-kv-redundancy-during-migration 1 \
  --kv-redundancy 2 >"${LOG_DIR}/root_cpu0.log" 2>&1 &
ROOT_PID=$!
for i in $(seq 1 100); do
  if [[ -S "${SOCK}" ]]; then break; fi
  sleep 0.25
done
python3 examples/plan-uds-client.py "${SOCK}" ping >"${LOG_DIR}/uds_ping.json" 2>&1 || true
python3 examples/plan-uds-client.py "${SOCK}" set_plan \
  --seq 601 \
  --mode next_barrier \
  --stage 0 \
  --from 1 \
  --to 2 \
  --kind 1 \
  --heads 1 \
  --ffn 0 >"${LOG_DIR}/uds_set_plan.json" 2>&1 || true
wait "${ROOT_PID}"
RC=$?
echo "LOG_DIR=${LOG_DIR}"
echo "RC=${RC}"
echo "--- migration evidence ---"
grep -E "plan|apply|emit|Pred|EOS|Critical|error" "${LOG_DIR}/root_cpu0.log" | tail -n 180 || true
echo "--- generated text ---"
grep -E "Pred|EOS" "${LOG_DIR}/root_cpu0.log" || true
exit "${RC}"
