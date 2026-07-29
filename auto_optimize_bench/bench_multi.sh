#!/usr/bin/env bash
# Multi-run benchmark for stability.
set -u
LABEL="${1:-test}"
RUNS="${2:-3}"
ROOT="/home/byh/B01/EdgeVisor"
ENGINE="${ROOT}/EdgeVisor"
MODEL3="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOK="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
export EDGEVISOR_MODEL3="${MODEL3}"
export EDGEVISOR_TOKENIZER="${TOK}"
export EDGEVISOR_LOG_ROOT="${ROOT}/runtime_logs"
export CPATH="${ROOT}/tools/vulkan_deps/root/usr/include${CPATH:+:${CPATH}}"
export PATH="${ROOT}/tools/vulkan_deps/root/usr/bin:${PATH}"
export LIBRARY_PATH="${ROOT}/tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${ROOT}/tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
SUMMARY_DIR="${ROOT}/auto_optimize_bench/summary"
mkdir -p "${SUMMARY_DIR}"
SUMMARY="${SUMMARY_DIR}/${LABEL}_multi.txt"

run_one() {
  local NAME="$1" PORT1="$2" PORT2="$3" RATIOS="$4" STEPS="$5" PROMPT="$6" EXTRA="$7" SOCK="${8:-}"
  local LOG="/tmp/bench_${NAME}_$$.log"
  cleanup() {
    if [[ -n "${P1:-}" ]]; then kill "${P1}" 2>/dev/null || true; fi
    if [[ -n "${P2:-}" ]]; then kill "${P2}" 2>/dev/null || true; fi
    if [[ -n "${ROOT_PID:-}" ]]; then kill "${ROOT_PID}" 2>/dev/null || true; fi
    wait 2>/dev/null || true
    if [[ -n "${SOCK}" ]]; then rm -f "${SOCK}"; fi
  }
  trap cleanup RETURN
  cd "${ENGINE}"
  ./dllama worker --port "${PORT1}" --nthreads 1 --gpu-index 1 >"${LOG}.w1" 2>&1 & P1=$!
  ./dllama worker --port "${PORT2}" --nthreads 1 --gpu-index 2 >"${LOG}.w2" 2>&1 & P2=$!
  sleep 2
  if [[ -n "${SOCK}" ]]; then
    rm -f "${SOCK}"
    DLLAMA_PLAN_CTRL_SOCKET="${SOCK}" ./dllama inference \
      --prompt "${PROMPT}" --steps "${STEPS}" --model "${MODEL3}" --tokenizer "${TOK}" \
      --buffer-float-type q80 --nthreads 1 --max-seq-len 512 --temperature 0 --seed 1 \
      --gpu-index 0 --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}" \
      --ratios "${RATIOS}" ${EXTRA} >"${LOG}" 2>&1 &
  else
    ./dllama inference \
      --prompt "${PROMPT}" --steps "${STEPS}" --model "${MODEL3}" --tokenizer "${TOK}" \
      --buffer-float-type q80 --nthreads 1 --max-seq-len 512 --temperature 0 --seed 1 \
      --gpu-index 0 --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}" \
      --ratios "${RATIOS}" ${EXTRA} >"${LOG}" 2>&1 &
  fi
  ROOT_PID=$!
  if [[ -n "${SOCK}" ]]; then
    for i in $(seq 1 100); do
      if [[ -S "${SOCK}" ]]; then break; fi
      sleep 0.25
    done
  fi
  wait "${ROOT_PID}"
  local EVAL_MS=$(grep -A4 "Evaluation (root wall-clock)" "${LOG}" | grep "ms/tok" | sed -E "s/.*\(([0-9.]+) ms\/tok\).*/\1/")
  local PRED_MS=$(grep -A4 "Prediction (root wall-clock)" "${LOG}" | grep "ms/tok" | sed -E "s/.*\(([0-9.]+) ms\/tok\).*/\1/")
  local EVAL_N=$(grep -A4 "Evaluation (root wall-clock)" "${LOG}" | grep -E "nTokens" | awk "{print \$2}")
  local PRED_N=$(grep -A4 "Prediction (root wall-clock)" "${LOG}" | grep -E "nTokens" | awk "{print \$2}")
  rm -f "${LOG}" "${LOG}.w1" "${LOG}.w2"
  echo "${EVAL_MS:-NA} ${PRED_MS:-NA} ${EVAL_N:-NA} ${PRED_N:-NA}"
}

agg() {
  local NAME="$1"; shift
  local eval_sum=0; local pred_sum=0; local eval_n=0; local pred_n=0
  for line in "$@"; do
    read e p en pn <<<"${line}"
    if [[ "${e}" != "NA" && "${e}" != "0" ]]; then eval_sum=$(echo "${eval_sum} + ${e}" | bc -l); eval_n=$((eval_n+1)); fi
    if [[ "${p}" != "NA" && "${p}" != "0" ]]; then pred_sum=$(echo "${pred_sum} + ${p}" | bc -l); pred_n=$((pred_n+1)); fi
  done
  local eval_avg="NA"; local pred_avg="NA"
  if [[ ${eval_n} -gt 0 ]]; then eval_avg=$(echo "scale=3; ${eval_sum} / ${eval_n}" | bc -l); fi
  if [[ ${pred_n} -gt 0 ]]; then pred_avg=$(echo "scale=3; ${pred_sum} / ${pred_n}" | bc -l); fi
  echo "${NAME}: ttft_avg=${eval_avg} tpot_avg=${pred_avg} eval_n=${eval_n} pred_n=${pred_n}"
}

echo "label=${LABEL} runs=${RUNS}" > "${SUMMARY}"

# Test 1: static_uneven
declare -a STATIC=()
for i in $(seq 1 ${RUNS}); do
  PORT1=$((20000 + i*10)); PORT2=$((20001 + i*10))
  R=$(run_one "static" ${PORT1} ${PORT2} "2:3:3" 128 "Write a clear explanation of distributed LLM inference and tensor parallelism in three short paragraphs." "" "")
  STATIC+=("${R}")
  echo "static_run${i}: ${R}" >> "${SUMMARY}"
done
agg "static_uneven" "${STATIC[@]}" >> "${SUMMARY}"

# Test 2: dynamic_heads
declare -a DYNAMIC=()
for i in $(seq 1 ${RUNS}); do
  PORT1=$((20100 + i*10)); PORT2=$((20101 + i*10))
  SOCK=/tmp/dllama_bench_dyn_${i}.sock
  R=$(run_one "dynamic" ${PORT1} ${PORT2} "2:3:3" 96 "Write a comma-separated list of the numbers from 1 to 20." \
    "--enable-stage-full-weights --enable-plan-barrier --enable-kv-redundancy-during-migration 1 --kv-redundancy 2" "${SOCK}")
  python3 "${ROOT}/EdgeVisor/examples/plan-uds-client.py" "${SOCK}" set_plan --seq 501 --mode next_barrier --stage 0 --from 1 --to 2 --kind 1 --heads 1 --ffn 0 >/dev/null 2>&1 || true
  wait "${ROOT_PID:-}" 2>/dev/null || true
  DYNAMIC+=("${R}")
  echo "dynamic_run${i}: ${R}" >> "${SUMMARY}"
done
agg "dynamic_heads" "${DYNAMIC[@]}" >> "${SUMMARY}"

# Test 3: pp_migration
declare -a PP=()
for i in $(seq 1 ${RUNS}); do
  PORT1=$((20200 + i*10)); PORT2=$((20201 + i*10))
  SOCK=/tmp/dllama_bench_pp_${i}.sock
  R=$(run_one "pp" ${PORT1} ${PORT2} "1@8*1@10*1@10" 64 "Tell me a long story about distributed inference systems with multiple GPU nodes." \
    "--enable-pp-migration --enable-kv-redundancy-during-migration 1 --kv-redundancy 2" "${SOCK}")
  python3 "${ROOT}/EdgeVisor/examples/plan-uds-client.py" "${SOCK}" set_pp_migration --seq 301 --mode next_barrier --from 0 --to 1 --layer-count 1 --trigger-pos 0 >/dev/null 2>&1 || true
  wait "${ROOT_PID:-}" 2>/dev/null || true
  PP+=("${R}")
  echo "pp_run${i}: ${R}" >> "${SUMMARY}"
done
agg "pp_migration" "${PP[@]}" >> "${SUMMARY}"

# Test 4: agentic_proxy (no migration)
declare -a AGENTIC=()
for i in $(seq 1 ${RUNS}); do
  PORT1=$((20300 + i*10)); PORT2=$((20301 + i*10))
  R=$(run_one "agentic" ${PORT1} ${PORT2} "2:3:3" 256 "You are an agent. Solve x*x=49 and show steps. Then explain." "" "")
  AGENTIC+=("${R}")
  echo "agentic_run${i}: ${R}" >> "${SUMMARY}"
done
agg "agentic_proxy" "${AGENTIC[@]}" >> "${SUMMARY}"

cat "${SUMMARY}"
echo ""
echo "SUMMARY=${SUMMARY}"
