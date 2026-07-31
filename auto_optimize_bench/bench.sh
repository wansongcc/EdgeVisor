#!/usr/bin/env bash
# Auto-optimization benchmarking harness for EdgeVisor.
set -u
LABEL="${1:-baseline}"
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
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${ROOT}/auto_optimize_bench/logs/${LABEL}_${STAMP}"
SUMMARY_DIR="${ROOT}/auto_optimize_bench/summary"
mkdir -p "${LOG_ROOT}" "${SUMMARY_DIR}"
echo "LABEL=${LABEL}" | tee "${LOG_ROOT}/meta.txt"
echo "STAMP=${STAMP}" | tee -a "${LOG_ROOT}/meta.txt"
cd "${ENGINE}" || exit 2
G0=0; G1=1; G2=2

run_test() {
  local NAME="$1" PORT1="$2" PORT2="$3" RATIOS="$4" STEPS="$5" PROMPT="$6" EXTRA="$7" SOCK="${8:-}" UDS_CMD="${9:-}"
  local TEST_LOG="${LOG_ROOT}/${NAME}"
  mkdir -p "${TEST_LOG}"
  unset P1 P2 ROOT_PID
  cleanup() {
    if [[ -n "${P1:-}" ]]; then kill "${P1}" 2>/dev/null || true; fi
    if [[ -n "${P2:-}" ]]; then kill "${P2}" 2>/dev/null || true; fi
    if [[ -n "${ROOT_PID:-}" ]]; then kill "${ROOT_PID}" 2>/dev/null || true; fi
    wait 2>/dev/null || true
    if [[ -n "${SOCK}" ]]; then rm -f "${SOCK}"; fi
  }
  trap cleanup RETURN
  ./dllama worker --port "${PORT1}" --nthreads 1 --gpu-index "${G1}" >"${TEST_LOG}/worker1.log" 2>&1 &
  P1=$!
  ./dllama worker --port "${PORT2}" --nthreads 1 --gpu-index "${G2}" >"${TEST_LOG}/worker2.log" 2>&1 &
  P2=$!
  sleep 3
  if [[ -n "${SOCK}" ]]; then
    rm -f "${SOCK}"
    DLLAMA_PLAN_CTRL_SOCKET="${SOCK}" ./dllama inference \
      --prompt "${PROMPT}" --steps "${STEPS}" --model "${MODEL3}" --tokenizer "${TOK}" \
      --buffer-float-type q80 --nthreads 1 --max-seq-len 512 --temperature 0 --seed 1 \
      --gpu-index "${G0}" --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}" \
      --ratios "${RATIOS}" ${EXTRA} >"${TEST_LOG}/root.log" 2>&1 &
  else
    ./dllama inference \
      --prompt "${PROMPT}" --steps "${STEPS}" --model "${MODEL3}" --tokenizer "${TOK}" \
      --buffer-float-type q80 --nthreads 1 --max-seq-len 512 --temperature 0 --seed 1 \
      --gpu-index "${G0}" --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}" \
      --ratios "${RATIOS}" ${EXTRA} >"${TEST_LOG}/root.log" 2>&1 &
  fi
  ROOT_PID=$!
  if [[ -n "${SOCK}" ]]; then
    for i in $(seq 1 100); do
      if [[ -S "${SOCK}" ]]; then break; fi
      sleep 0.25
    done
  fi
  if [[ -n "${UDS_CMD}" ]]; then
    # Issue the UDS control command while the root process is still alive.
    # The socket is created by the root and dies with it, so a command sent
    # after `wait` below can only hit a dead socket (swallowed by `|| true`),
    # silently measuring plain inference instead of the intended migration.
    bash -c "${UDS_CMD}"
  fi
  wait "${ROOT_PID}"
  local RC=$?
  echo "test=${NAME} rc=${RC}" | tee -a "${LOG_ROOT}/meta.txt"
}

run_test "static_uneven" 19501 19502 "2:3:3" 128 "Write a clear explanation of distributed LLM inference and tensor parallelism in three short paragraphs." "" ""
SOCK=/tmp/dllama_bench_heads.sock
run_test "dynamic_heads" 19701 19702 "2:3:3" 96 \
  "Write a comma-separated list of the numbers from 1 to 20." \
  "--enable-stage-full-weights --enable-plan-barrier --enable-kv-redundancy-during-migration 1 --kv-redundancy 2" "${SOCK}" \
  "python3 \"${ROOT}/EdgeVisor/examples/plan-uds-client.py\" \"${SOCK}\" set_plan --seq 501 --mode next_barrier --stage 0 --from 1 --to 2 --kind 1 --heads 1 --ffn 0 >\"${LOG_ROOT}/dynamic_heads/uds_set_plan.json\" 2>&1 || true"
SOCK=/tmp/dllama_bench_pp.sock
run_test "pp_migration" 19301 19302 "1@8*1@10*1@10" 32 "Hi" \
  "--enable-pp-migration --enable-kv-redundancy-during-migration 1 --kv-redundancy 2" "${SOCK}" \
  "python3 \"${ROOT}/EdgeVisor/examples/plan-uds-client.py\" \"${SOCK}\" set_pp_migration --seq 301 --mode next_barrier --from 0 --to 1 --layer-count 1 --trigger-pos 0 >\"${LOG_ROOT}/pp_migration/uds_set_pp_migration.json\" 2>&1 || true"
SOCK=""
run_test "agentic_proxy" 19511 19512 "2:3:3" 256 "You are an agent. Solve: x*x=49. Show steps then answer." "" ""

SUMMARY="${SUMMARY_DIR}/${LABEL}_${STAMP}.txt"
echo "label=${LABEL} stamp=${STAMP}" > "${SUMMARY}"
echo "================================================" >> "${SUMMARY}"

parse_metrics() {
  local LOG="$1" NAME="$2"
  if [[ ! -f "${LOG}" ]]; then echo "${NAME}: NO LOG" >> "${SUMMARY}"; return; fi
  # Wall-clock eval (TTFT proxy: per-token time during prompt eval)
  local EVAL_MS_PER_TOK=$(grep -A4 "Evaluation (root wall-clock)" "${LOG}" | grep "ms/tok" | sed -E "s/.*\(([0-9.]+) ms\/tok\).*/\1/")
  local EVAL_TOKENS=$(grep -A4 "Evaluation (root wall-clock)" "${LOG}" | grep -E "nTokens" | awk "{print \$2}")
  local EVAL_TOKENS_S=$(grep -A2 "Evaluation (root wall-clock)" "${LOG}" | grep -E "tokens/s:" | sed -E "s/.*tokens\/s:[ ]*([0-9.]+).*/\1/")
  # Wall-clock pred (TPOT proxy)
  local PRED_MS_PER_TOK=$(grep -A4 "Prediction (root wall-clock)" "${LOG}" | grep "ms/tok" | sed -E "s/.*\(([0-9.]+) ms\/tok\).*/\1/")
  local PRED_TOKENS=$(grep -A4 "Prediction (root wall-clock)" "${LOG}" | grep -E "nTokens" | awk "{print \$2}")
  local PRED_TOKENS_S=$(grep -A2 "Prediction (root wall-clock)" "${LOG}" | grep -E "tokens/s:" | sed -E "s/.*tokens\/s:[ ]*([0-9.]+).*/\1/")
  echo "${NAME}: ttft_eval_ms_per_tok=${EVAL_MS_PER_TOK:-NA} ttft_tokens=${EVAL_TOKENS:-NA} ttft_tokens_per_s=${EVAL_TOKENS_S:-NA} | tpot_pred_ms_per_tok=${PRED_MS_PER_TOK:-NA} pred_tokens=${PRED_TOKENS:-NA} pred_tokens_per_s=${PRED_TOKENS_S:-NA}" >> "${SUMMARY}"
}

for t in static_uneven dynamic_heads pp_migration agentic_proxy; do
  parse_metrics "${LOG_ROOT}/${t}/root.log" "${t}"
done
cat "${SUMMARY}"
echo ""
echo "LOG_ROOT=${LOG_ROOT}"
echo "SUMMARY=${SUMMARY}"
