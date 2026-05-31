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
LOG_DIR="${EDGEVISOR_LOG_ROOT}/semantic/gpu_full_static_2plus2_${STAMP}"
mkdir -p "${LOG_DIR}"
cd "${BASE}" || exit 2
PORT1=19731
PORT2=19732
cleanup() {
  if [[ -n "${P1:-}" ]]; then kill "${P1}" 2>/dev/null || true; fi
  if [[ -n "${P2:-}" ]]; then kill "${P2}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
}
trap cleanup EXIT
./dllama worker --port "${PORT1}" --nthreads 1 --gpu-index 1 >"${LOG_DIR}/worker_gpu1.log" 2>&1 &
P1=$!
./dllama worker --port "${PORT2}" --nthreads 1 --gpu-index 2 >"${LOG_DIR}/worker_gpu2.log" 2>&1 &
P2=$!
sleep 4
./dllama inference \
  --prompt "What is 2+2? Answer with only the number." \
  --steps 32 \
  --model "${MODEL3}" \
  --tokenizer "${TOK}" \
  --buffer-float-type q80 \
  --nthreads 1 \
  --max-seq-len 512 \
  --temperature 0 \
  --seed 1 \
  --gpu-index 0 \
  --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}" \
  --ratios "2:2:4" \
  --enable-stage-full-weights >"${LOG_DIR}/root_gpu0.log" 2>&1
RC=$?
echo "LOG_DIR=${LOG_DIR}"
echo "RC=${RC}"
grep -E "Stage full|Root Logits|Pred|EOS|Critical|error|FAIL" "${LOG_DIR}/root_gpu0.log" | tail -n 120 || true
exit "${RC}"
