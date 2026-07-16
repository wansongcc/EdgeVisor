#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  jetson_gpu_freq_inject.sh --level PERCENT [--duration SEC] [--sysfs-root PATH]
  jetson_gpu_freq_inject.sh --list [--sysfs-root PATH]

Temporarily locks the GPU frequency on Jetson Orin Nano or Orin NX, then
restores the previous GPU frequency limits and 3D scaling state.

Options:
  --level PERCENT    Disturbance level from 0 (lowest) to 100 (highest)
  --duration SEC     Lock duration; positive integer or decimal (default: 20)
  --list             Show available/current GPU frequencies without modifying them
  --sysfs-root PATH  Alternate sysfs root, primarily for testing (default: /sys)
  -h, --help         Show this help
EOF
}

die() {
  echo "jetson_gpu_freq_inject: $*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local count="$2"
  (( count >= 2 )) || die "${option} requires a value"
}

try_read_uint() {
  local file="$1"
  local value
  [[ -r "${file}" ]] || return 1
  value="$(tr -d '[:space:]' < "${file}")" || return 1
  [[ "${value}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${value}"
}

read_uint() {
  local file="$1"
  local value
  value="$(try_read_uint "${file}")" || \
    die "missing, unreadable, or invalid unsigned integer in sysfs file: ${file}"
  printf '%s\n' "${value}"
}

seconds_to_ms() {
  local seconds="$1"
  local whole fraction milliseconds
  [[ "${seconds}" =~ ^([0-9]+)(\.([0-9]+))?$ ]] || return 1
  whole="${BASH_REMATCH[1]}"
  fraction="${BASH_REMATCH[3]:-}"
  fraction="${fraction:0:3}"
  while (( ${#fraction} < 3 )); do
    fraction+="0"
  done
  [[ -n "${fraction}" ]] || fraction="000"
  milliseconds=$((10#${whole} * 1000 + 10#${fraction}))
  (( milliseconds > 0 )) || return 1
  printf '%s\n' "${milliseconds}"
}

discover_paths() {
  local prefix="${SYSFS_ROOT%/}"
  PLATFORM_DIR="${prefix}/devices/platform/17000000.gpu"
  GPU_DIR="${PLATFORM_DIR}/devfreq/17000000.gpu"
  AVAILABLE_FILE="${GPU_DIR}/available_frequencies"
  MIN_FILE="${GPU_DIR}/min_freq"
  MAX_FILE="${GPU_DIR}/max_freq"
  CUR_FILE="${GPU_DIR}/cur_freq"
  SCALING_FILE="${PLATFORM_DIR}/enable_3d_scaling"

  [[ -d "${GPU_DIR}" ]] || die "Jetson Orin GPU devfreq directory not found: ${GPU_DIR}"
  [[ -r "${AVAILABLE_FILE}" ]] || die "missing available GPU frequencies: ${AVAILABLE_FILE}"
  read_uint "${MIN_FILE}" >/dev/null
  read_uint "${MAX_FILE}" >/dev/null
  read_uint "${CUR_FILE}" >/dev/null
}

load_frequencies() {
  local value
  local -a unsorted=()
  FREQUENCIES=()

  for value in $(<"${AVAILABLE_FILE}"); do
    [[ "${value}" =~ ^[0-9]+$ ]] || die "invalid frequency in ${AVAILABLE_FILE}: ${value}"
    if (( value <= ORIG_MAX )); then
      unsorted+=("${value}")
    fi
  done
  (( ${#unsorted[@]} > 0 )) || die "no available GPU frequency is allowed by current max_freq=${ORIG_MAX}"
  mapfile -t FREQUENCIES < <(printf '%s\n' "${unsorted[@]}" | sort -n -u)
}

select_frequency() {
  local level="$1"
  local lowest highest nominal best best_distance frequency distance
  lowest="${FREQUENCIES[0]}"
  highest="${FREQUENCIES[${#FREQUENCIES[@]} - 1]}"
  nominal=$((lowest + (highest - lowest) * level / 100))
  best="${lowest}"
  best_distance=$((nominal - lowest))

  for frequency in "${FREQUENCIES[@]}"; do
    distance=$((frequency - nominal))
    (( distance < 0 )) && distance=$((-distance))
    if (( distance < best_distance )); then
      best="${frequency}"
      best_distance="${distance}"
    fi
  done
  printf '%s\n' "${best}"
}

write_value() {
  local file="$1"
  local value="$2"
  printf '%s\n' "${value}" > "${file}" || return 1
  if [[ -n "${JETSON_GPU_WRITE_LOG:-}" ]]; then
    printf '%s=%s\n' "${file##*/}" "${value}" >> "${JETSON_GPU_WRITE_LOG}" || return 1
  fi
}

set_bounds() {
  local desired_min="$1"
  local desired_max="$2"
  local current_min current_max
  current_min="$(try_read_uint "${MIN_FILE}")" || return 1
  current_max="$(try_read_uint "${MAX_FILE}")" || return 1

  if (( desired_max < current_min )); then
    write_value "${MIN_FILE}" "${desired_min}" || return 1
    write_value "${MAX_FILE}" "${desired_max}" || return 1
  elif (( desired_min > current_max )); then
    write_value "${MAX_FILE}" "${desired_max}" || return 1
    write_value "${MIN_FILE}" "${desired_min}" || return 1
  else
    write_value "${MIN_FILE}" "${desired_min}" || return 1
    write_value "${MAX_FILE}" "${desired_max}" || return 1
  fi
}

restore_state() {
  local failed=0
  set +e
  set_bounds "${ORIG_MIN}" "${ORIG_MAX}" || failed=1
  if (( SCALING_PRESENT )); then
    write_value "${SCALING_FILE}" "${ORIG_SCALING}" || failed=1
  fi
  RESTORE_ACTIVE=0

  if (( failed )); then
    echo "jetson_gpu_freq_inject: automatic restoration failed" >&2
    echo "Manual recovery:" >&2
    echo "  echo ${ORIG_MAX} | sudo tee ${MAX_FILE}" >&2
    echo "  echo ${ORIG_MIN} | sudo tee ${MIN_FILE}" >&2
    if (( SCALING_PRESENT )); then
      echo "  echo ${ORIG_SCALING} | sudo tee ${SCALING_FILE}" >&2
    fi
    return 1
  fi
  echo "restored min_freq_hz=${ORIG_MIN} max_freq_hz=${ORIG_MAX} enable_3d_scaling=${ORIG_SCALING:-unavailable}"
  return 0
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if (( RESTORE_ACTIVE )); then
    if ! restore_state; then
      status=1
    fi
  fi
  exit "${status}"
}

run_interval() {
  local duration_ms="$1"
  local start_ms end_ms now_ms remaining_ms sleep_ms sleep_seconds current

  if [[ "${JETSON_GPU_TEST_NO_SLEEP:-0}" == "1" ]]; then
    current="$(read_uint "${CUR_FILE}")"
    echo "remaining_ms=${duration_ms} cur_freq_hz=${current}"
    return 0
  fi

  start_ms="$(date +%s%3N)"
  end_ms=$((start_ms + duration_ms))
  while true; do
    now_ms="$(date +%s%3N)"
    remaining_ms=$((end_ms - now_ms))
    (( remaining_ms > 0 )) || break
    current="$(read_uint "${CUR_FILE}")"
    echo "remaining_ms=${remaining_ms} cur_freq_hz=${current}"
    sleep_ms="${remaining_ms}"
    (( sleep_ms > 1000 )) && sleep_ms=1000
    printf -v sleep_seconds '%d.%03d' "$((sleep_ms / 1000))" "$((sleep_ms % 1000))"
    sleep "${sleep_seconds}"
  done
}

LEVEL=""
DURATION="20"
LIST_ONLY=0
SYSFS_ROOT="/sys"

while (( $# > 0 )); do
  case "$1" in
    --level)
      require_value "$1" "$#"
      LEVEL="$2"
      shift 2
      ;;
    --duration)
      require_value "$1" "$#"
      DURATION="$2"
      shift 2
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --sysfs-root)
      require_value "$1" "$#"
      SYSFS_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

discover_paths
ORIG_MIN="$(read_uint "${MIN_FILE}")"
ORIG_MAX="$(read_uint "${MAX_FILE}")"
load_frequencies

if (( LIST_ONLY )); then
  printf 'available_frequencies_hz='
  printf '%s ' "${FREQUENCIES[@]}"
  printf '\nmin_freq_hz=%s\nmax_freq_hz=%s\ncurrent_freq_hz=%s\n' \
    "${ORIG_MIN}" "${ORIG_MAX}" "$(read_uint "${CUR_FILE}")"
  exit 0
fi

[[ -n "${LEVEL}" ]] || die "--level is required unless --list is used"
[[ "${LEVEL}" =~ ^[0-9]+$ ]] || die "--level must be an integer from 0 to 100"
LEVEL_VALUE=$((10#${LEVEL}))
(( LEVEL_VALUE >= 0 && LEVEL_VALUE <= 100 )) || die "--level must be an integer from 0 to 100"
DURATION_MS="$(seconds_to_ms "${DURATION}")" || die "--duration must be a positive integer or decimal number of seconds"

EFFECTIVE_UID="${JETSON_GPU_TEST_EUID:-${EUID}}"
[[ "${EFFECTIVE_UID}" =~ ^[0-9]+$ ]] || die "invalid effective UID"
(( EFFECTIVE_UID == 0 )) || die "root privileges are required; run with sudo"
command -v flock >/dev/null 2>&1 || die "flock is required (install util-linux)"

LOCK_FILE="${JETSON_GPU_LOCK_FILE:-/tmp/jetson_gpu_freq_inject.lock}"
exec 9>"${LOCK_FILE}"
flock -n 9 || die "another GPU frequency injection is already running"

SCALING_PRESENT=0
ORIG_SCALING=""
if [[ -e "${SCALING_FILE}" ]]; then
  ORIG_SCALING="$(read_uint "${SCALING_FILE}")"
  SCALING_PRESENT=1
fi

TARGET_FREQ="$(select_frequency "${LEVEL_VALUE}")"
RESTORE_ACTIVE=1
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set_bounds "${TARGET_FREQ}" "${TARGET_FREQ}"
if (( SCALING_PRESENT )); then
  write_value "${SCALING_FILE}" 0
fi

LOCKED_MIN="$(read_uint "${MIN_FILE}")"
LOCKED_MAX="$(read_uint "${MAX_FILE}")"
[[ "${LOCKED_MIN}" == "${TARGET_FREQ}" && "${LOCKED_MAX}" == "${TARGET_FREQ}" ]] || \
  die "failed to verify GPU frequency lock: min=${LOCKED_MIN} max=${LOCKED_MAX} target=${TARGET_FREQ}"

echo "target_level=${LEVEL_VALUE} target_freq_hz=${TARGET_FREQ} duration_sec=${DURATION}"
echo "warning: thermal, electrical, or power-mode limits may still throttle the observed GPU clock"
run_interval "${DURATION_MS}"
echo "GPU frequency disturbance complete"
