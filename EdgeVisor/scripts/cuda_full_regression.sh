#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ -z "${EDGEVISOR_GPU_INDEX:-}" ]; then
    echo "EDGEVISOR_GPU_INDEX is required; do not hard-code GPU IDs in this script." >&2
    exit 2
fi
GPU_INDEX=$EDGEVISOR_GPU_INDEX
LOG_DIR=${EDGEVISOR_LOG_DIR:-"runtime_logs/cuda_full_regression_$(date +%Y%m%d_%H%M%S)"}
SANITIZER=${EDGEVISOR_COMPUTE_SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}

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

run_logged 00_clean_cpu make clean
run_logged 01_cpu_build make -j"${EDGEVISOR_MAKE_JOBS:-4}" dllama nn-cpu-test nn-cpu-ops-test tokenizer-test nn-slice-test
run_logged 02_nn_cpu_test ./nn-cpu-test
run_logged 03_nn_cpu_ops_test ./nn-cpu-ops-test
run_logged 04_tokenizer_test ./tokenizer-test
run_logged 05_nn_slice_test ./nn-slice-test

run_logged 06_clean_cuda make clean
if [ -n "${CUDA_ARCHS:-}" ]; then
    run_logged 07_cuda_build make -j"${EDGEVISOR_MAKE_JOBS:-4}" DLLAMA_CUDA=1 CUDA_ARCHS="$CUDA_ARCHS" dllama nn-cuda-test
else
    run_logged 07_cuda_build make -j"${EDGEVISOR_MAKE_JOBS:-4}" DLLAMA_CUDA=1 dllama nn-cuda-test
fi
run_logged 08_nn_cuda_test ./nn-cuda-test --gpu-index "$GPU_INDEX"

if [ "${EDGEVISOR_RUN_SANITIZER:-0}" = "1" ]; then
    if [ ! -x "$SANITIZER" ]; then
        printf 'SKIP Compute Sanitizer: not executable at %s\n' "$SANITIZER" | tee "$LOG_DIR/09_compute_sanitizer.skip.log"
    else
        run_logged 09_compute_sanitizer_memcheck "$SANITIZER" --tool memcheck ./nn-cuda-test --gpu-index "$GPU_INDEX"
        run_logged 10_compute_sanitizer_racecheck "$SANITIZER" --tool racecheck ./nn-cuda-test --gpu-index "$GPU_INDEX"
    fi
else
    printf 'SKIP Compute Sanitizer: set EDGEVISOR_RUN_SANITIZER=1.\n' | tee "$LOG_DIR/09_compute_sanitizer.skip.log"
fi

if [ "${EDGEVISOR_RUN_STATIC_SEMANTICS:-0}" = "1" ]; then
    EDGEVISOR_LOG_DIR="$LOG_DIR/static" scripts/cuda_static_semantics.sh
else
    printf 'SKIP static model semantics: set EDGEVISOR_RUN_STATIC_SEMANTICS=1.\n' | tee "$LOG_DIR/11_static_semantics.skip.log"
fi

if [ "${EDGEVISOR_RUN_DYNAMIC_SEMANTICS:-0}" = "1" ]; then
    EDGEVISOR_LOG_DIR="$LOG_DIR/dynamic" scripts/cuda_dynamic_semantics.sh
else
    printf 'SKIP dynamic model semantics: set EDGEVISOR_RUN_DYNAMIC_SEMANTICS=1.\n' | tee "$LOG_DIR/12_dynamic_semantics.skip.log"
fi

printf 'CUDA full regression logs: %s\n' "$LOG_DIR"
