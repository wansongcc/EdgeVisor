#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ -z "${EDGEVISOR_GPU_INDEX:-}" ]; then
    echo "EDGEVISOR_GPU_INDEX is required; do not hard-code GPU IDs in this script." >&2
    exit 2
fi
GPU_INDEX=$EDGEVISOR_GPU_INDEX
LOG_DIR=${EDGEVISOR_LOG_DIR:-"runtime_logs/cuda_static_semantics_$(date +%Y%m%d_%H%M%S)"}
PROMPT=${EDGEVISOR_PROMPT:-"The capital of France is"}
STEPS=${EDGEVISOR_STEPS:-64}
MAX_SEQ_LEN=${EDGEVISOR_MAX_SEQ_LEN:-2048}
NTHREADS=${EDGEVISOR_NTHREADS:-2}
BUFFER_FLOAT_TYPE=${EDGEVISOR_BUFFER_FLOAT_TYPE:-q80}
TP_RATIOS=${EDGEVISOR_TP_RATIOS:-"2:3:3"}
PP_RATIOS=${EDGEVISOR_PP_RATIOS:-"1@8*1@10*1@10"}
TPPP_RATIOS=${EDGEVISOR_TPPP_RATIOS:-"2:3:3@8*1@10*1@10"}

mkdir -p "$LOG_DIR"

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
run_logged 02_nn_cuda_test ./nn-cuda-test --gpu-index "$GPU_INDEX"

if [ -n "${EDGEVISOR_MODEL3:-}" ] && [ -n "${EDGEVISOR_TOKENIZER:-}" ]; then
    run_logged 03_single_gpu_cuda_model \
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
        --benchmark
else
    printf 'SKIP single GPU model run: set EDGEVISOR_MODEL3 and EDGEVISOR_TOKENIZER.\n' | tee "$LOG_DIR/03_single_gpu_cuda_model.skip.log"
fi

if [ "${EDGEVISOR_RUN_L4:-0}" = "1" ]; then
    if [ -z "${EDGEVISOR_MODEL3:-}" ] || [ -z "${EDGEVISOR_TOKENIZER:-}" ] || [ -z "${EDGEVISOR_WORKERS:-}" ]; then
        printf 'SKIP L4 static distributed runs: require EDGEVISOR_MODEL3, EDGEVISOR_TOKENIZER, EDGEVISOR_WORKERS.\n' | tee "$LOG_DIR/04_l4_static.skip.log"
        exit 0
    fi

    # EDGEVISOR_WORKERS is intentionally expanded into multiple host:port arguments.
    # Example: EDGEVISOR_WORKERS="10.0.0.2:9999 10.0.0.3:9999".
    # shellcheck disable=SC2086
    run_logged 04_l4_tp_uneven \
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
        --ratios "$TP_RATIOS" \
        --benchmark

    # shellcheck disable=SC2086
    run_logged 05_l4_pp_three_stage \
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
        --ratios "$PP_RATIOS" \
        --benchmark

    # shellcheck disable=SC2086
    run_logged 06_l4_tppp \
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
        --ratios "$TPPP_RATIOS" \
        --benchmark
else
    printf 'SKIP L4 static distributed runs: set EDGEVISOR_RUN_L4=1 and EDGEVISOR_WORKERS.\n' | tee "$LOG_DIR/04_l4_static.skip.log"
fi

printf 'CUDA static semantics logs: %s\n' "$LOG_DIR"
