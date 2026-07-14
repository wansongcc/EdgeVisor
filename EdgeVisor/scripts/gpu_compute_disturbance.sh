#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  gpu_compute_disturbance.sh start [options]
  gpu_compute_disturbance.sh status [options]
  gpu_compute_disturbance.sh stop [options]

Starts or stops the recommended Vulkan GPU compute disturbance workload:
1xQwen3-14B + 1xQwen3-8B on the local GPU. The script only stops process
groups recorded in its own pid files, so it does not kill EdgeVisor workers.

Options:
  --duration SEC          Disturbance loop duration for each workload (default: 240)
  --preheat SEC           Sleep after start before returning (default: 60)
  --bin-dir PATH          Directory containing ./dllama (default: /home/jetson/cc/EdgeVisor/EdgeVisor)
  --models-src PATH       Source model directory for /tmp symlink (default: /home/jetson/cc/models)
  --model-link PATH       Model symlink used by commands (default: /tmp/edgevisor_models)
  --log-dir PATH          Log/pid directory (default: /tmp/edgevisor_disturbance)
  --gpu-index INDEX       Vulkan GPU index (default: 0)
  --buffer-float-type T   dllama buffer float type (default: q80)
  --max-seq-len N         Disturbance max sequence length (default: 512)
  --steps N               Disturbance generation steps (default: 512)
  --prompt TEXT           Disturbance prompt
  -h, --help              Show this help
EOF
}

ACTION="${1:-}"
if [[ -z "${ACTION}" || "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

DURATION=240
PREHEAT=60
BIN_DIR="/home/jetson/cc/EdgeVisor/EdgeVisor"
MODELS_SRC="/home/jetson/cc/models"
MODEL_LINK="/tmp/edgevisor_models"
LOG_DIR="/tmp/edgevisor_disturbance"
GPU_INDEX=0
BUFFER_FLOAT_TYPE="q80"
MAX_SEQ_LEN=512
STEPS=512
PROMPT="Continue this sequence for a long time: alpha beta gamma delta epsilon"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --preheat)
      PREHEAT="$2"
      shift 2
      ;;
    --bin-dir)
      BIN_DIR="$2"
      shift 2
      ;;
    --models-src)
      MODELS_SRC="$2"
      shift 2
      ;;
    --model-link)
      MODEL_LINK="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --gpu-index)
      GPU_INDEX="$2"
      shift 2
      ;;
    --buffer-float-type)
      BUFFER_FLOAT_TYPE="$2"
      shift 2
      ;;
    --max-seq-len)
      MAX_SEQ_LEN="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PID_14B="${LOG_DIR}/gpu_disturb_14b.pid"
PID_8B="${LOG_DIR}/gpu_disturb_8b.pid"
LOG_14B="${LOG_DIR}/gpu_disturb_14b.log"
LOG_8B="${LOG_DIR}/gpu_disturb_8b.log"

ensure_paths() {
  mkdir -p "${LOG_DIR}"
  if [[ ! -d "${BIN_DIR}" ]]; then
    echo "Missing dllama directory: ${BIN_DIR}" >&2
    exit 1
  fi
  if [[ -d "${MODELS_SRC}" ]]; then
    ln -sfn "${MODELS_SRC}" "${MODEL_LINK}"
  elif [[ ! -e "${MODEL_LINK}" ]]; then
    echo "Missing model source ${MODELS_SRC} and model link ${MODEL_LINK}" >&2
    exit 1
  fi
}

stop_pid_file() {
  local pid_file="$1"
  local pgid
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  pgid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ -n "${pgid}" ]]; then
    kill -- "-${pgid}" 2>/dev/null || true
  fi
  rm -f "${pid_file}"
}

stop_all() {
  stop_pid_file "${PID_14B}"
  stop_pid_file "${PID_8B}"
}

is_running() {
  local pid_file="$1"
  local pgid
  [[ -f "${pid_file}" ]] || return 1
  pgid="$(cat "${pid_file}" 2>/dev/null || true)"
  [[ -n "${pgid}" ]] || return 1
  kill -0 "-${pgid}" 2>/dev/null
}

status_one() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local pgid=""
  if [[ -f "${pid_file}" ]]; then
    pgid="$(cat "${pid_file}" 2>/dev/null || true)"
  fi
  if [[ -n "${pgid}" ]] && kill -0 "-${pgid}" 2>/dev/null; then
    echo "${name}: running pgid=${pgid} log=${log_file}"
  else
    echo "${name}: stopped log=${log_file}"
  fi
}

start_workload() {
  local name="$1"
  local model="$2"
  local tokenizer="$3"
  local log_file="$4"
  local pid_file="$5"

  if is_running "${pid_file}"; then
    echo "${name}: already running pgid=$(cat "${pid_file}")"
    return 0
  fi

  rm -f "${pid_file}"
  (
    cd "${BIN_DIR}"
    DISTURB_DURATION="${DURATION}" \
    DISTURB_MODEL="${model}" \
    DISTURB_TOKENIZER="${tokenizer}" \
    DISTURB_GPU_INDEX="${GPU_INDEX}" \
    DISTURB_BUFFER_FLOAT_TYPE="${BUFFER_FLOAT_TYPE}" \
    DISTURB_MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
    DISTURB_STEPS="${STEPS}" \
    DISTURB_PROMPT="${PROMPT}" \
    setsid bash -lc '
      end=$((SECONDS + DISTURB_DURATION))
      while [ "$SECONDS" -lt "$end" ]; do
        ./dllama inference \
          --model "$DISTURB_MODEL" \
          --tokenizer "$DISTURB_TOKENIZER" \
          --backend vulkan \
          --gpu-index "$DISTURB_GPU_INDEX" \
          --buffer-float-type "$DISTURB_BUFFER_FLOAT_TYPE" \
          --benchmark \
          --nthreads 1 \
          --max-seq-len "$DISTURB_MAX_SEQ_LEN" \
          --prompt "$DISTURB_PROMPT" \
          --steps "$DISTURB_STEPS"
      done
    ' > "${log_file}" 2>&1 &
    printf "%s\n" "$!" > "${pid_file}"
  )
  echo "${name}: started pgid=$(cat "${pid_file}") log=${log_file}"
}

case "${ACTION}" in
  start)
    ensure_paths
    stop_all
    start_workload \
      "qwen3-14b" \
      "${MODEL_LINK}/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m" \
      "${MODEL_LINK}/qwen3_14b_q40/dllama_tokenizer_qwen3_14b_q40.t" \
      "${LOG_14B}" \
      "${PID_14B}"
    start_workload \
      "qwen3-8b" \
      "${MODEL_LINK}/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m" \
      "${MODEL_LINK}/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t" \
      "${LOG_8B}" \
      "${PID_8B}"
    if [[ "${PREHEAT}" != "0" ]]; then
      echo "preheating ${PREHEAT}s before returning"
      sleep "${PREHEAT}"
    fi
    status_one "qwen3-14b" "${PID_14B}" "${LOG_14B}"
    status_one "qwen3-8b" "${PID_8B}" "${LOG_8B}"
    ;;
  status)
    status_one "qwen3-14b" "${PID_14B}" "${LOG_14B}"
    status_one "qwen3-8b" "${PID_8B}" "${LOG_8B}"
    ;;
  stop)
    stop_all
    status_one "qwen3-14b" "${PID_14B}" "${LOG_14B}"
    status_one "qwen3-8b" "${PID_8B}" "${LOG_8B}"
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
