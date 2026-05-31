#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../config/env.sh
source "${PROJECT_ROOT}/config/env.sh"

STAMP="${EDGEVISOR_TEST_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${EDGEVISOR_TEST_OUT_BASE:-${EDGEVISOR_LOG_ROOT}/benchmark_docs_${STAMP}}"
mkdir -p "${OUT_BASE}"
cd "${EDGEVISOR_ENGINE_DIR}"

echo "OUT_BASE=${OUT_BASE}"

cleanup_pids=()
cleanup() {
  for pid in "${cleanup_pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

wait_for_socket() {
  local sock="$1"
  local i
  for i in $(seq 1 100); do
    [[ -S "${sock}" ]] && return 0
    sleep 0.25
  done
  return 1
}

run_cpu_single() {
  local log_dir="${OUT_BASE}/01_cpu_single"
  mkdir -p "${log_dir}"
  ./dllama inference \
    --prompt "What is 2+2? Answer with only the number." \
    --steps 32 \
    --model "${EDGEVISOR_MODEL3}" \
    --tokenizer "${EDGEVISOR_TOKENIZER}" \
    --buffer-float-type q80 \
    --nthreads 2 \
    --max-seq-len 512 \
    --temperature 0 \
    --seed 1 \
    --benchmark >"${log_dir}/root.log" 2>&1
  echo $? >"${log_dir}/rc.txt"
  echo "01_CPU_SINGLE_LOG=${log_dir} RC=$(cat "${log_dir}/rc.txt")"
}

run_gpu_single() {
  local log_dir="${OUT_BASE}/02_gpu_single"
  mkdir -p "${log_dir}"
  ./dllama inference \
    --prompt "What is 2+2? Answer with only the number." \
    --steps 32 \
    --model "${EDGEVISOR_MODEL3}" \
    --tokenizer "${EDGEVISOR_TOKENIZER}" \
    --buffer-float-type q80 \
    --nthreads 1 \
    --max-seq-len 512 \
    --temperature 0 \
    --seed 1 \
    --gpu-index 0 \
    --benchmark >"${log_dir}/root.log" 2>&1
  echo $? >"${log_dir}/rc.txt"
  echo "02_GPU_SINGLE_LOG=${log_dir} RC=$(cat "${log_dir}/rc.txt")"
}

run_cpu_static() {
  local log_dir="${OUT_BASE}/03_cpu_uneven_static"
  local p1="${EDGEVISOR_CPU_STATIC_PORT1:-19601}"
  local p2="${EDGEVISOR_CPU_STATIC_PORT2:-19602}"
  mkdir -p "${log_dir}"
  ./dllama worker --port "${p1}" --nthreads 2 >"${log_dir}/worker1.log" 2>&1 & local w1=$!
  ./dllama worker --port "${p2}" --nthreads 2 >"${log_dir}/worker2.log" 2>&1 & local w2=$!
  cleanup_pids+=("${w1}" "${w2}")
  sleep 4
  set +e
  ./dllama inference \
    --prompt "What is 2+2? Answer with only the number." \
    --steps 32 \
    --model "${EDGEVISOR_MODEL3}" \
    --tokenizer "${EDGEVISOR_TOKENIZER}" \
    --buffer-float-type q80 \
    --nthreads 2 \
    --max-seq-len 512 \
    --temperature 0 \
    --seed 1 \
    --workers "127.0.0.1:${p1}" "127.0.0.1:${p2}" \
    --benchmark \
    --ratios "2:3:3" >"${log_dir}/root.log" 2>&1
  local rc=$?
  set -e
  kill "${w1}" "${w2}" 2>/dev/null || true
  wait 2>/dev/null || true
  cleanup_pids=()
  echo "${rc}" >"${log_dir}/rc.txt"
  echo "03_CPU_UNEVEN_STATIC_LOG=${log_dir} RC=${rc}"
  return "${rc}"
}

run_gpu_static() {
  local log_dir="${OUT_BASE}/04_gpu_uneven_static"
  local p1="${EDGEVISOR_GPU_STATIC_PORT1:-19501}"
  local p2="${EDGEVISOR_GPU_STATIC_PORT2:-19502}"
  mkdir -p "${log_dir}"
  ./dllama worker --port "${p1}" --nthreads 1 --gpu-index 1 >"${log_dir}/worker1.log" 2>&1 & local w1=$!
  ./dllama worker --port "${p2}" --nthreads 1 --gpu-index 2 >"${log_dir}/worker2.log" 2>&1 & local w2=$!
  cleanup_pids+=("${w1}" "${w2}")
  sleep 4
  set +e
  ./dllama inference \
    --prompt "What is 2+2? Answer with only the number." \
    --steps 64 \
    --model "${EDGEVISOR_MODEL3}" \
    --tokenizer "${EDGEVISOR_TOKENIZER}" \
    --buffer-float-type q80 \
    --nthreads 1 \
    --max-seq-len 512 \
    --temperature 0 \
    --seed 1 \
    --gpu-index 0 \
    --workers "127.0.0.1:${p1}" "127.0.0.1:${p2}" \
    --benchmark \
    --ratios "2:3:3" >"${log_dir}/root.log" 2>&1
  local rc=$?
  set -e
  kill "${w1}" "${w2}" 2>/dev/null || true
  wait 2>/dev/null || true
  cleanup_pids=()
  echo "${rc}" >"${log_dir}/rc.txt"
  echo "04_GPU_UNEVEN_STATIC_LOG=${log_dir} RC=${rc}"
  return "${rc}"
}

run_cpu_dynamic() {
  local log_dir="${OUT_BASE}/05_cpu_uneven_dynamic"
  local p1="${EDGEVISOR_CPU_DYNAMIC_PORT1:-19711}"
  local p2="${EDGEVISOR_CPU_DYNAMIC_PORT2:-19712}"
  local sock="${EDGEVISOR_CPU_DYNAMIC_SOCKET:-/tmp/dllama_bench_cpu_heads.sock}"
  mkdir -p "${log_dir}"
  rm -f "${sock}"
  ./dllama worker --port "${p1}" --nthreads 1 >"${log_dir}/worker_cpu1.log" 2>&1 & local w1=$!
  ./dllama worker --port "${p2}" --nthreads 1 >"${log_dir}/worker_cpu2.log" 2>&1 & local w2=$!
  cleanup_pids+=("${w1}" "${w2}")
  sleep 4
  DLLAMA_PLAN_CTRL_SOCKET="${sock}" ./dllama inference \
    --prompt "Write a comma-separated list of the numbers from 1 to 20." \
    --steps 64 \
    --model "${EDGEVISOR_MODEL3}" \
    --tokenizer "${EDGEVISOR_TOKENIZER}" \
    --buffer-float-type q80 \
    --nthreads 1 \
    --max-seq-len 512 \
    --temperature 0 \
    --seed 1 \
    --workers "127.0.0.1:${p1}" "127.0.0.1:${p2}" \
    --benchmark \
    --ratios "2:3:3" \
    --enable-stage-full-weights \
    --enable-plan-barrier \
    --enable-kv-redundancy-during-migration 1 \
    --kv-redundancy 2 >"${log_dir}/root_cpu0.log" 2>&1 & local root_pid=$!
  cleanup_pids+=("${root_pid}")
  wait_for_socket "${sock}"
  python3 examples/plan-uds-client.py "${sock}" ping >"${log_dir}/uds_ping.json" 2>&1 || true
  python3 examples/plan-uds-client.py "${sock}" set_plan \
    --seq 601 \
    --mode next_barrier \
    --stage 0 \
    --from 1 \
    --to 2 \
    --kind 1 \
    --heads 1 \
    --ffn 0 >"${log_dir}/uds_set_plan.json" 2>&1 || true
  set +e
  wait "${root_pid}"
  local rc=$?
  set -e
  kill "${w1}" "${w2}" 2>/dev/null || true
  wait 2>/dev/null || true
  rm -f "${sock}"
  cleanup_pids=()
  echo "${rc}" >"${log_dir}/rc.txt"
  echo "05_CPU_UNEVEN_DYNAMIC_LOG=${log_dir} RC=${rc}"
  return "${rc}"
}

run_gpu_dynamic() {
  local log_dir="${OUT_BASE}/06_gpu_uneven_dynamic"
  local p1="${EDGEVISOR_GPU_DYNAMIC_PORT1:-19701}"
  local p2="${EDGEVISOR_GPU_DYNAMIC_PORT2:-19702}"
  local sock="${EDGEVISOR_GPU_DYNAMIC_SOCKET:-/tmp/dllama_bench_gpu_heads.sock}"
  mkdir -p "${log_dir}"
  rm -f "${sock}"
  ./dllama worker --port "${p1}" --nthreads 1 --gpu-index 1 >"${log_dir}/worker_gpu1.log" 2>&1 & local w1=$!
  ./dllama worker --port "${p2}" --nthreads 1 --gpu-index 2 >"${log_dir}/worker_gpu2.log" 2>&1 & local w2=$!
  cleanup_pids+=("${w1}" "${w2}")
  sleep 4
  DLLAMA_PLAN_CTRL_SOCKET="${sock}" ./dllama inference \
    --prompt "Write a comma-separated list of the numbers from 1 to 20." \
    --steps 96 \
    --model "${EDGEVISOR_MODEL3}" \
    --tokenizer "${EDGEVISOR_TOKENIZER}" \
    --buffer-float-type q80 \
    --nthreads 1 \
    --max-seq-len 512 \
    --temperature 0 \
    --seed 1 \
    --gpu-index 0 \
    --workers "127.0.0.1:${p1}" "127.0.0.1:${p2}" \
    --benchmark \
    --ratios "2:3:3" \
    --enable-stage-full-weights \
    --enable-plan-barrier \
    --enable-kv-redundancy-during-migration 1 \
    --kv-redundancy 2 >"${log_dir}/root_gpu0.log" 2>&1 & local root_pid=$!
  cleanup_pids+=("${root_pid}")
  wait_for_socket "${sock}"
  python3 examples/plan-uds-client.py "${sock}" ping >"${log_dir}/uds_ping.json" 2>&1 || true
  python3 examples/plan-uds-client.py "${sock}" set_plan \
    --seq 501 \
    --mode next_barrier \
    --stage 0 \
    --from 1 \
    --to 2 \
    --kind 1 \
    --heads 1 \
    --ffn 0 >"${log_dir}/uds_set_plan.json" 2>&1 || true
  set +e
  wait "${root_pid}"
  local rc=$?
  set -e
  kill "${w1}" "${w2}" 2>/dev/null || true
  wait 2>/dev/null || true
  rm -f "${sock}"
  cleanup_pids=()
  echo "${rc}" >"${log_dir}/rc.txt"
  echo "06_GPU_UNEVEN_DYNAMIC_LOG=${log_dir} RC=${rc}"
  return "${rc}"
}

run_cpu_single
run_gpu_single
run_cpu_static
run_gpu_static
run_cpu_dynamic
run_gpu_dynamic

echo "DONE_OUT_BASE=${OUT_BASE}"
