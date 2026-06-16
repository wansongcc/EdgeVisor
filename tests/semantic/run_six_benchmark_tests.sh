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

EDGEVISOR_FORCE_DISTRIBUTED="${EDGEVISOR_FORCE_DISTRIBUTED:-0}"
EDGEVISOR_PARALLEL_WARMUP_MODE="${EDGEVISOR_PARALLEL_WARMUP_MODE:-pp}"
EDGEVISOR_GPU_PP_RATIO_CANDIDATES="${EDGEVISOR_GPU_PP_RATIO_CANDIDATES:-1@9*1@9*1@10 1@7*1@10*1@11 1@11*1@10*1@7}"
EDGEVISOR_GPU_HYBRID_RATIO_CANDIDATES="${EDGEVISOR_GPU_HYBRID_RATIO_CANDIDATES:-1:1:1 2:3:2 4:3:2 1@14*1:1@14 1:1@14*1@14 1@9*1@9*1@10}"
EDGEVISOR_GPU_RATIO_CANDIDATES="${EDGEVISOR_GPU_RATIO_CANDIDATES:-}"
EDGEVISOR_GPU_WARMUP_STEPS="${EDGEVISOR_GPU_WARMUP_STEPS:-48}"
EDGEVISOR_SINGLE_GPU_MEMORY_HEADROOM_MB="${EDGEVISOR_SINGLE_GPU_MEMORY_HEADROOM_MB:-2048}"
EDGEVISOR_SELECTED_GPU_RATIOS="${EDGEVISOR_SELECTED_GPU_RATIOS:-}"

model_size_mb() {
  local path="$1"
  python3 - "$path" <<'PY'
import os
import sys
print((os.path.getsize(sys.argv[1]) + 1024 * 1024 - 1) // (1024 * 1024))
PY
}

gpu_total_memory_mb() {
  local gpu_index="${1:-0}"
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${gpu_index}" 2>/dev/null | awk 'NR==1 {print int($1)}'
}

gpu_model_fits_single() {
  local model_mb gpu_mb required_mb
  model_mb="$(model_size_mb "${EDGEVISOR_MODEL3}")"
  gpu_mb="$(gpu_total_memory_mb 0 || true)"
  [[ -n "${gpu_mb}" && "${gpu_mb}" -gt 0 ]] || return 1
  required_mb=$((model_mb + EDGEVISOR_SINGLE_GPU_MEMORY_HEADROOM_MB))
  [[ "${required_mb}" -lt "${gpu_mb}" ]]
}

prediction_ms_per_tok() {
  local log="$1"
  sed -n '/^Prediction$/,/^$/ s/.*(\([0-9.][0-9.]*\) ms\/tok).*/\1/p' "${log}" | head -n 1
}

run_gpu_distributed_inference() {
  local log_file="$1"
  local ratios="$2"
  local p1="$3"
  local p2="$4"
  local prompt="$5"
  local steps="$6"
  shift 6
  DLLAMA_LAYER_PROF_PATH="${log_file}.layer_prof.bin" ./dllama inference \
    --prompt "${prompt}" \
    --steps "${steps}" \
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
    --ratios "${ratios}" \
    "$@" >"${log_file}" 2>&1
}

select_gpu_ratios() {
  if [[ -n "${EDGEVISOR_SELECTED_GPU_RATIOS}" ]]; then
    printf '%s\n' "${EDGEVISOR_SELECTED_GPU_RATIOS}"
    return 0
  fi

  local candidates
  if [[ -n "${EDGEVISOR_GPU_RATIO_CANDIDATES}" ]]; then
    candidates="${EDGEVISOR_GPU_RATIO_CANDIDATES}"
  elif [[ "${EDGEVISOR_PARALLEL_WARMUP_MODE}" == "hybrid" ]]; then
    candidates="${EDGEVISOR_GPU_HYBRID_RATIO_CANDIDATES}"
  else
    candidates="${EDGEVISOR_GPU_PP_RATIO_CANDIDATES}"
  fi

  local log_dir="${OUT_BASE}/00_gpu_ratio_warmup"
  mkdir -p "${log_dir}"
  local best_ratio=""
  local best_ms=""
  local i=0
  local ratios
  echo "GPU_RATIO_WARMUP_MODE=${EDGEVISOR_PARALLEL_WARMUP_MODE} candidates=${candidates}" >&2
  set -f
  for ratios in ${candidates}; do
    i=$((i + 1))
    local p1=$((19800 + i * 10 + 1))
    local p2=$((19800 + i * 10 + 2))
    local log_file="${log_dir}/ratio_${i}_$(printf '%s' "${ratios}" | tr ':,*@' '____').log"
    ./dllama worker --port "${p1}" --nthreads 1 --gpu-index 1 >"${log_file}.worker1" 2>&1 & local w1=$!
    ./dllama worker --port "${p2}" --nthreads 1 --gpu-index 2 >"${log_file}.worker2" 2>&1 & local w2=$!
    cleanup_pids+=("${w1}" "${w2}")
    sleep 4
    set +e
    run_gpu_distributed_inference "${log_file}" "${ratios}" "${p1}" "${p2}" \
      "Write a comma-separated list of the numbers from 1 to 20." \
      "${EDGEVISOR_GPU_WARMUP_STEPS}"
    local rc=$?
    set -e
    kill "${w1}" "${w2}" 2>/dev/null || true
    wait 2>/dev/null || true
    cleanup_pids=()
    if [[ "${rc}" -ne 0 ]]; then
      echo "GPU_RATIO_WARMUP ratio=${ratios} rc=${rc} log=${log_file}" >&2
      continue
    fi
    local ms
    ms="$(prediction_ms_per_tok "${log_file}" || true)"
    if [[ -z "${ms}" ]]; then
      echo "GPU_RATIO_WARMUP ratio=${ratios} no_prediction_ms log=${log_file}" >&2
      continue
    fi
    echo "GPU_RATIO_WARMUP ratio=${ratios} prediction_ms_per_tok=${ms} log=${log_file}" >&2
    if [[ -z "${best_ms}" ]] || awk -v a="${ms}" -v b="${best_ms}" 'BEGIN {exit !(a < b)}'; then
      best_ms="${ms}"
      best_ratio="${ratios}"
    fi
  done
  set +f

  if [[ -z "${best_ratio}" ]]; then
    best_ratio="2:3:2"
  fi
  EDGEVISOR_SELECTED_GPU_RATIOS="${best_ratio}"
  echo "GPU_RATIO_SELECTED=${best_ratio} prediction_ms_per_tok=${best_ms:-unknown}" >&2
  printf '%s\n' "${best_ratio}"
}

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
  DLLAMA_LAYER_PROF_PATH="${log_dir}/layer_prof.bin" ./dllama inference \
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
  DLLAMA_LAYER_PROF_PATH="${log_dir}/layer_prof.bin" ./dllama inference \
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
  DLLAMA_LAYER_PROF_PATH="${log_dir}/layer_prof.bin" ./dllama inference \
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

  if [[ "${EDGEVISOR_FORCE_DISTRIBUTED}" != "1" ]] && gpu_model_fits_single; then
    echo "04_GPU_STATIC_MODE=single_gpu model_fits_gpu0 force_distributed=${EDGEVISOR_FORCE_DISTRIBUTED}" | tee "${log_dir}/selection.txt"
    DLLAMA_LAYER_PROF_PATH="${log_dir}/layer_prof.bin" ./dllama inference \
      --prompt "Write a comma-separated list of the numbers from 1 to 80." \
      --steps 96 \
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
    echo "04_GPU_STATIC_SINGLE_LOG=${log_dir} RC=$(cat "${log_dir}/rc.txt")"
    return 0
  fi

  local ratios
  ratios="$(select_gpu_ratios)"
  echo "04_GPU_STATIC_MODE=distributed ratios=${ratios} force_distributed=${EDGEVISOR_FORCE_DISTRIBUTED}" | tee "${log_dir}/selection.txt"
  ./dllama worker --port "${p1}" --nthreads 1 --gpu-index 1 >"${log_dir}/worker1.log" 2>&1 & local w1=$!
  ./dllama worker --port "${p2}" --nthreads 1 --gpu-index 2 >"${log_dir}/worker2.log" 2>&1 & local w2=$!
  cleanup_pids+=("${w1}" "${w2}")
  sleep 4
  set +e
  run_gpu_distributed_inference "${log_dir}/root.log" "${ratios}" "${p1}" "${p2}" \
    "Write a comma-separated list of the numbers from 1 to 80." \
    96
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
  DLLAMA_LAYER_PROF_PATH="${log_dir}/layer_prof.bin" DLLAMA_PLAN_CTRL_SOCKET="${sock}" ./dllama inference \
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
  local ratios
  ratios="$(select_gpu_ratios)"
  echo "06_GPU_DYNAMIC_RATIOS=${ratios}" | tee "${log_dir}/selection.txt"
  ./dllama worker --port "${p1}" --nthreads 1 --gpu-index 1 >"${log_dir}/worker_gpu1.log" 2>&1 & local w1=$!
  ./dllama worker --port "${p2}" --nthreads 1 --gpu-index 2 >"${log_dir}/worker_gpu2.log" 2>&1 & local w2=$!
  cleanup_pids+=("${w1}" "${w2}")
  sleep 4
  DLLAMA_LAYER_PROF_PATH="${log_dir}/layer_prof.bin" DLLAMA_PLAN_CTRL_SOCKET="${sock}" ./dllama inference \
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
    --ratios "${ratios}" \
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
