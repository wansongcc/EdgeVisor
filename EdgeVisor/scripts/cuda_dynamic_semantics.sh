#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ -z "${EDGEVISOR_GPU_INDEX:-}" ]; then
    echo "EDGEVISOR_GPU_INDEX is required; do not hard-code GPU IDs in this script." >&2
    exit 2
fi
GPU_INDEX=$EDGEVISOR_GPU_INDEX
LOG_DIR=${EDGEVISOR_LOG_DIR:-"runtime_logs/cuda_dynamic_semantics_$(date +%Y%m%d_%H%M%S)"}
PROMPT=${EDGEVISOR_PROMPT:-"The capital of France is"}
STEPS=${EDGEVISOR_STEPS:-64}
MAX_SEQ_LEN=${EDGEVISOR_MAX_SEQ_LEN:-2048}
NTHREADS=${EDGEVISOR_NTHREADS:-2}
BUFFER_FLOAT_TYPE=${EDGEVISOR_BUFFER_FLOAT_TYPE:-q80}
PLAN_SOCKET=${EDGEVISOR_PLAN_SOCKET:-"$LOG_DIR/dllama_plan.sock"}
DYNAMIC_RATIOS=${EDGEVISOR_DYNAMIC_RATIOS:-"1:1"}

mkdir -p "$LOG_DIR"
rm -f "$PLAN_SOCKET"

run_logged() {
    name=$1
    shift
    log="$LOG_DIR/$name.log"
    printf '==> %s\n' "$*" > "$log"
    "$@" >> "$log" 2>&1
    status=$?
    cat "$log"
    return "$status"
}

run_logged 00_clean make clean
if [ -n "${CUDA_ARCHS:-}" ]; then
    run_logged 01_build make -j"${EDGEVISOR_MAKE_JOBS:-4}" DLLAMA_CUDA=1 CUDA_ARCHS="$CUDA_ARCHS" dllama nn-cuda-test
else
    run_logged 01_build make -j"${EDGEVISOR_MAKE_JOBS:-4}" DLLAMA_CUDA=1 dllama nn-cuda-test
fi
run_logged 02_nn_cuda_test_pr11_pr12 ./nn-cuda-test --gpu-index "$GPU_INDEX"

if [ "${EDGEVISOR_RUN_L5:-0}" = "1" ]; then
    if [ -z "${EDGEVISOR_MODEL3:-}" ] || [ -z "${EDGEVISOR_TOKENIZER:-}" ] || [ -z "${EDGEVISOR_WORKERS:-}" ]; then
        printf 'SKIP L5 dynamic model run: require EDGEVISOR_MODEL3, EDGEVISOR_TOKENIZER, EDGEVISOR_WORKERS.\n' | tee "$LOG_DIR/03_l5_dynamic.skip.log"
        exit 0
    fi

    export DLLAMA_PLAN_CTRL_SOCKET="$PLAN_SOCKET"
    export DLLAMA_BUBBLE_SHADOW_KV="${DLLAMA_BUBBLE_SHADOW_KV:-1}"
    export DLLAMA_BUBBLE_SHADOW_KV_LOG="${DLLAMA_BUBBLE_SHADOW_KV_LOG:-1}"

    # EDGEVISOR_WORKERS is intentionally expanded into multiple host:port arguments.
    # shellcheck disable=SC2086
    run_logged 03_l5_dynamic_head_ffn_pp_bubble \
        ./dllama inference \
        --backend cuda \
        --gpu-index "$GPU_INDEX" \
        --model "$EDGEVISOR_MODEL3" \
        --tokenizer "$EDGEVISOR_TOKENIZER" \
        --buffer-float-type "$BUFFER_FLOAT_TYPE" \
        --temperature 0 \
        --seed 1 \
        --steps "$STEPS" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --nthreads "$NTHREADS" \
        --prompt "$PROMPT" \
        --workers $EDGEVISOR_WORKERS \
        --ratios "$DYNAMIC_RATIOS" \
        --enable-plan-barrier \
        --enable-pp-migration \
        --enable-kv-aggregate \
        --runtime-active-seg-enabled \
        --runtime-redundant-seg-enabled \
        --benchmark
else
    printf 'SKIP L5 dynamic model run: set EDGEVISOR_RUN_L5=1 plus model/tokenizer/workers env.\n' | tee "$LOG_DIR/03_l5_dynamic.skip.log"
fi

printf 'CUDA dynamic semantics logs: %s\n' "$LOG_DIR"
