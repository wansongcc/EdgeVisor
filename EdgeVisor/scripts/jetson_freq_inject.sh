#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  jetson_freq_inject.sh --level PERCENT [--target TARGETS] [--duration SEC] [--sysfs-root PATH]
  jetson_freq_inject.sh --list [--sysfs-root PATH]

Temporarily locks the GPU, CPU, and/or EMC frequency on Jetson Orin Nano or
Orin NX with JetPack 5 or 6, then restores the previous state afterward.

Options:
  --level PERCENT    Disturbance level from 0 (lowest) to 100 (highest)
  --target TARGETS   Comma-separated subsystems: gpu, cpu, emc, or all (default: all)
  --duration SEC     Lock duration; positive integer or decimal (default: 20)
  --list             Show available/current frequencies without modifying them
  --sysfs-root PATH  Alternate sysfs root, primarily for testing (default: /sys)
  -h, --help         Show this help
EOF
}

die() {
  echo "jetson_freq_inject: $*" >&2
  exit 1
}

warn() {
  echo "jetson_freq_inject: warning: $*" >&2
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

try_read_text() {
  local file="$1"
  [[ -r "${file}" ]] || return 1
  tr -d '\n' < "${file}"
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

write_value() {
  local file="$1"
  local value="$2"
  printf '%s\n' "${value}" > "${file}" || return 1
  if [[ -n "${JETSON_FREQ_WRITE_LOG:-}" ]]; then
    printf '%s=%s\n' "${file##*/}" "${value}" >> "${JETSON_FREQ_WRITE_LOG}" || return 1
  fi
}

# ---------------------------------------------------------------------------
# Frequency selection: map level 0-100 to the nearest available frequency.
# Accepts a space-separated sorted frequency list via $FREQUENCIES array.
# ---------------------------------------------------------------------------
select_frequency() {
  local level="$1"
  shift
  local -a freqs=("$@")
  local lowest highest nominal best best_distance frequency distance
  lowest="${freqs[0]}"
  highest="${freqs[${#freqs[@]} - 1]}"
  nominal=$((lowest + (highest - lowest) * level / 100))
  best="${lowest}"
  best_distance=$((nominal - lowest))

  for frequency in "${freqs[@]}"; do
    distance=$((frequency - nominal))
    (( distance < 0 )) && distance=$((-distance))
    if (( distance < best_distance )); then
      best="${frequency}"
      best_distance="${distance}"
    fi
  done
  printf '%s\n' "${best}"
}

# ---------------------------------------------------------------------------
# GPU subsystem
# ---------------------------------------------------------------------------
GPU_PRESENT=0
GPU_DIR=""
GPU_PLATFORM_DIR=""
GPU_MIN_FILE=""
GPU_MAX_FILE=""
GPU_CUR_FILE=""
GPU_AVAILABLE_FILE=""
GPU_SCALING_FILE=""
GPU_ORIG_MIN=""
GPU_ORIG_MAX=""
GPU_ORIG_SCALING=""
GPU_SCALING_PRESENT=0
declare -a GPU_FREQUENCIES=()

gpu_discover() {
  local prefix="${SYSFS_ROOT%/}"
  local r36_platform="${prefix}/devices/platform/17000000.gpu"
  local r35_platform="${prefix}/devices/17000000.ga10b"
  local gpu_name

  if [[ -d "${r36_platform}/devfreq/17000000.gpu" ]]; then
    GPU_PLATFORM_DIR="${r36_platform}"
    gpu_name="17000000.gpu"
  elif [[ -d "${r35_platform}/devfreq/17000000.ga10b" ]]; then
    GPU_PLATFORM_DIR="${r35_platform}"
    gpu_name="17000000.ga10b"
  else
    return 1
  fi

  GPU_DIR="${GPU_PLATFORM_DIR}/devfreq/${gpu_name}"
  GPU_AVAILABLE_FILE="${GPU_DIR}/available_frequencies"
  GPU_MIN_FILE="${GPU_DIR}/min_freq"
  GPU_MAX_FILE="${GPU_DIR}/max_freq"
  GPU_CUR_FILE="${GPU_DIR}/cur_freq"
  GPU_SCALING_FILE="${GPU_PLATFORM_DIR}/enable_3d_scaling"

  [[ -r "${GPU_AVAILABLE_FILE}" ]] || return 1
  read_uint "${GPU_MIN_FILE}" >/dev/null || return 1
  read_uint "${GPU_MAX_FILE}" >/dev/null || return 1
  GPU_PRESENT=1
  return 0
}

gpu_load_frequencies() {
  local orig_max value
  orig_max="$(read_uint "${GPU_MAX_FILE}")"
  local -a unsorted=()
  GPU_FREQUENCIES=()

  for value in $(<"${GPU_AVAILABLE_FILE}"); do
    [[ "${value}" =~ ^[0-9]+$ ]] || die "invalid frequency in ${GPU_AVAILABLE_FILE}: ${value}"
    if (( value <= orig_max )); then
      unsorted+=("${value}")
    fi
  done
  (( ${#unsorted[@]} > 0 )) || die "no available GPU frequency is allowed by current max_freq=${orig_max}"
  mapfile -t GPU_FREQUENCIES < <(printf '%s\n' "${unsorted[@]}" | sort -n -u)
}

gpu_save_state() {
  GPU_ORIG_MIN="$(read_uint "${GPU_MIN_FILE}")"
  GPU_ORIG_MAX="$(read_uint "${GPU_MAX_FILE}")"
  GPU_SCALING_PRESENT=0
  GPU_ORIG_SCALING=""
  if [[ -e "${GPU_SCALING_FILE}" ]]; then
    GPU_ORIG_SCALING="$(read_uint "${GPU_SCALING_FILE}")"
    GPU_SCALING_PRESENT=1
  fi
}

gpu_set_bounds() {
  local desired_min="$1"
  local desired_max="$2"
  local current_min current_max
  current_min="$(try_read_uint "${GPU_MIN_FILE}")" || return 1
  current_max="$(try_read_uint "${GPU_MAX_FILE}")" || return 1

  if (( desired_max < current_min )); then
    write_value "${GPU_MIN_FILE}" "${desired_min}" || return 1
    write_value "${GPU_MAX_FILE}" "${desired_max}" || return 1
  elif (( desired_min > current_max )); then
    write_value "${GPU_MAX_FILE}" "${desired_max}" || return 1
    write_value "${GPU_MIN_FILE}" "${desired_min}" || return 1
  else
    write_value "${GPU_MIN_FILE}" "${desired_min}" || return 1
    write_value "${GPU_MAX_FILE}" "${desired_max}" || return 1
  fi
}

gpu_inject() {
  local target="$1"
  gpu_set_bounds "${target}" "${target}" || return 1
  if (( GPU_SCALING_PRESENT )); then
    write_value "${GPU_SCALING_FILE}" 0 || return 1
  fi
  local locked_min locked_max
  locked_min="$(read_uint "${GPU_MIN_FILE}")"
  locked_max="$(read_uint "${GPU_MAX_FILE}")"
  [[ "${locked_min}" == "${target}" && "${locked_max}" == "${target}" ]] || \
    die "failed to verify GPU frequency lock: min=${locked_min} max=${locked_max} target=${target}"
}

gpu_restore_state() {
  local failed=0
  gpu_set_bounds "${GPU_ORIG_MIN}" "${GPU_ORIG_MAX}" || failed=1
  if (( GPU_SCALING_PRESENT )); then
    write_value "${GPU_SCALING_FILE}" "${GPU_ORIG_SCALING}" || failed=1
  fi
  if (( failed )); then
    echo "  GPU manual recovery:" >&2
    echo "    echo ${GPU_ORIG_MAX} | sudo tee ${GPU_MAX_FILE}" >&2
    echo "    echo ${GPU_ORIG_MIN} | sudo tee ${GPU_MIN_FILE}" >&2
    if (( GPU_SCALING_PRESENT )); then
      echo "    echo ${GPU_ORIG_SCALING} | sudo tee ${GPU_SCALING_FILE}" >&2
    fi
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# CPU subsystem
# ---------------------------------------------------------------------------
CPU_PRESENT=0
declare -a CPU_DIRS=()
declare -a CPU_ORIG_GOVERNORS=()
declare -a CPU_ORIG_MINS=()
declare -a CPU_ORIG_MAXES=()
declare -a CPU_FREQUENCIES=()

cpu_discover() {
  local prefix="${SYSFS_ROOT%/}"
  local cpu_base="${prefix}/devices/system/cpu"
  local dir
  CPU_DIRS=()

  for dir in "${cpu_base}"/cpu[0-9]*/cpufreq; do
    [[ -d "${dir}" ]] || continue
    if [[ -r "${dir}/scaling_min_freq" && -r "${dir}/scaling_max_freq" ]]; then
      CPU_DIRS+=("${dir}")
    fi
  done

  (( ${#CPU_DIRS[@]} > 0 )) || return 1
  CPU_PRESENT=1
  return 0
}

cpu_load_frequencies() {
  local first_dir="${CPU_DIRS[0]}"
  local -a unsorted=()
  CPU_FREQUENCIES=()

  if [[ -r "${first_dir}/scaling_available_frequencies" ]]; then
    local value
    for value in $(<"${first_dir}/scaling_available_frequencies"); do
      [[ "${value}" =~ ^[0-9]+$ ]] || continue
      unsorted+=("${value}")
    done
  fi

  if (( ${#unsorted[@]} == 0 )); then
    local info_min info_max
    info_min="$(try_read_uint "${first_dir}/cpuinfo_min_freq")" || return 1
    info_max="$(try_read_uint "${first_dir}/cpuinfo_max_freq")" || return 1
    unsorted=("${info_min}" "${info_max}")
  fi

  (( ${#unsorted[@]} > 0 )) || return 1
  mapfile -t CPU_FREQUENCIES < <(printf '%s\n' "${unsorted[@]}" | sort -n -u)
  return 0
}

cpu_save_state() {
  local i=0
  CPU_ORIG_GOVERNORS=()
  CPU_ORIG_MINS=()
  CPU_ORIG_MAXES=()
  local dir
  for dir in "${CPU_DIRS[@]}"; do
    local gov
    gov="$(try_read_text "${dir}/scaling_governor")" || gov=""
    CPU_ORIG_GOVERNORS[i]="${gov}"
    CPU_ORIG_MINS[i]="$(read_uint "${dir}/scaling_min_freq")"
    CPU_ORIG_MAXES[i]="$(read_uint "${dir}/scaling_max_freq")"
    (( i += 1 ))
  done
}

cpu_inject() {
  local target="$1"
  local i=0
  local dir
  for dir in "${CPU_DIRS[@]}"; do
    local gov_file="${dir}/scaling_governor"
    if [[ -w "${gov_file}" ]]; then
      if [[ -r "${dir}/scaling_available_governors" ]] && \
         grep -q 'userspace' "${dir}/scaling_available_governors" 2>/dev/null; then
        write_value "${gov_file}" "userspace" || return 1
      fi
    fi
    local cur_min cur_max
    cur_min="$(try_read_uint "${dir}/scaling_min_freq")" || return 1
    cur_max="$(try_read_uint "${dir}/scaling_max_freq")" || return 1
    if (( target < cur_min )); then
      write_value "${dir}/scaling_min_freq" "${target}" || return 1
      write_value "${dir}/scaling_max_freq" "${target}" || return 1
    elif (( target > cur_max )); then
      write_value "${dir}/scaling_max_freq" "${target}" || return 1
      write_value "${dir}/scaling_min_freq" "${target}" || return 1
    else
      write_value "${dir}/scaling_min_freq" "${target}" || return 1
      write_value "${dir}/scaling_max_freq" "${target}" || return 1
    fi
    (( i += 1 ))
  done
}

cpu_restore_state() {
  local failed=0
  local i=0
  local dir
  for dir in "${CPU_DIRS[@]}"; do
    local orig_min="${CPU_ORIG_MINS[i]}"
    local orig_max="${CPU_ORIG_MAXES[i]}"
    local cur_min cur_max
    cur_min="$(try_read_uint "${dir}/scaling_min_freq")" || { (( failed=1 )); (( i+=1 )); continue; }
    cur_max="$(try_read_uint "${dir}/scaling_max_freq")" || { (( failed=1 )); (( i+=1 )); continue; }

    if (( orig_max < cur_min )); then
      write_value "${dir}/scaling_min_freq" "${orig_min}" || failed=1
      write_value "${dir}/scaling_max_freq" "${orig_max}" || failed=1
    elif (( orig_min > cur_max )); then
      write_value "${dir}/scaling_max_freq" "${orig_max}" || failed=1
      write_value "${dir}/scaling_min_freq" "${orig_min}" || failed=1
    else
      write_value "${dir}/scaling_min_freq" "${orig_min}" || failed=1
      write_value "${dir}/scaling_max_freq" "${orig_max}" || failed=1
    fi

    if [[ -n "${CPU_ORIG_GOVERNORS[i]}" && -w "${dir}/scaling_governor" ]]; then
      write_value "${dir}/scaling_governor" "${CPU_ORIG_GOVERNORS[i]}" || failed=1
    fi
    (( i += 1 ))
  done

  if (( failed )); then
    echo "  CPU manual recovery:" >&2
    local j=0
    for dir in "${CPU_DIRS[@]}"; do
      echo "    echo ${CPU_ORIG_MAXES[j]} | sudo tee ${dir}/scaling_max_freq" >&2
      echo "    echo ${CPU_ORIG_MINS[j]} | sudo tee ${dir}/scaling_min_freq" >&2
      if [[ -n "${CPU_ORIG_GOVERNORS[j]}" ]]; then
        echo "    echo ${CPU_ORIG_GOVERNORS[j]} | sudo tee ${dir}/scaling_governor" >&2
      fi
      (( j += 1 ))
    done
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# EMC subsystem (best-effort)
# ---------------------------------------------------------------------------
EMC_PRESENT=0
EMC_MODE=""          # "devfreq" or "debugfs"
EMC_DIR=""
EMC_MIN_FILE=""
EMC_MAX_FILE=""
EMC_CUR_FILE=""
EMC_ORIG_MIN=""
EMC_ORIG_MAX=""
EMC_SINGLE_RATE=0
declare -a EMC_FREQUENCIES=()

emc_discover() {
  local prefix="${SYSFS_ROOT%/}"
  EMC_SINGLE_RATE=0

  # Try debugfs BPMP interface first (Orin JetPack 5/6)
  local bpmp_emc="${prefix}/kernel/debug/bpmp/debug/clk/emc"
  if [[ -d "${bpmp_emc}" && -r "${bpmp_emc}/rate" && -r "${bpmp_emc}/min_rate" && -r "${bpmp_emc}/max_rate" ]]; then
    EMC_MODE="debugfs"
    EMC_DIR="${bpmp_emc}"
    EMC_CUR_FILE="${bpmp_emc}/rate"
    EMC_MIN_FILE="${bpmp_emc}/min_rate"
    EMC_MAX_FILE="${bpmp_emc}/max_rate"
    EMC_PRESENT=1
    return 0
  fi

  # Try devfreq tegra_mc interface
  local mc_dir
  for mc_dir in "${prefix}"/devices/platform/tegra*-mc/devfreq "${prefix}"/devices/platform/tegra*-mc/devfreq/tegra*-mc; do
    if [[ -d "${mc_dir}" && -r "${mc_dir}/min_freq" && -r "${mc_dir}/max_freq" ]]; then
      EMC_MODE="devfreq"
      EMC_DIR="${mc_dir}"
      EMC_CUR_FILE="${mc_dir}/cur_freq"
      EMC_MIN_FILE="${mc_dir}/min_freq"
      EMC_MAX_FILE="${mc_dir}/max_freq"
      EMC_PRESENT=1
      return 0
    fi
  done

  local bwmgr="${prefix}/kernel/debug/tegra_bwmgr"
  # Try debugfs bwmgr interface (older JetPack)
  if [[ -d "${bwmgr}/emc" && -r "${bwmgr}/emc/rate" && -r "${bwmgr}/emc/min_rate" && -r "${bwmgr}/emc/max_rate" ]]; then
    EMC_MODE="debugfs"
    EMC_DIR="${bwmgr}/emc"
    EMC_CUR_FILE="${bwmgr}/emc/rate"
    EMC_MIN_FILE="${bwmgr}/emc/min_rate"
    EMC_MAX_FILE="${bwmgr}/emc/max_rate"
    EMC_PRESENT=1
    return 0
  fi
  if [[ -d "${bwmgr}" && -r "${bwmgr}/emc" ]]; then
    EMC_MODE="bwmgr"
    EMC_DIR="${bwmgr}"
    EMC_CUR_FILE="${bwmgr}/emc"
    EMC_MIN_FILE=""
    if [[ -r "${bwmgr}/emc_max" ]]; then
      EMC_MAX_FILE="${bwmgr}/emc_max"
    else
      # Older JetPack 5 exposes one writable EMC rate node only.
      EMC_MAX_FILE="${bwmgr}/emc"
      EMC_SINGLE_RATE=1
    fi
    EMC_PRESENT=1
    return 0
  fi

  return 1
}

emc_load_frequencies() {
  EMC_FREQUENCIES=()
  if [[ "${EMC_MODE}" == "devfreq" && -r "${EMC_DIR}/available_frequencies" ]]; then
    local value
    local -a unsorted=()
    for value in $(<"${EMC_DIR}/available_frequencies"); do
      [[ "${value}" =~ ^[0-9]+$ ]] || continue
      unsorted+=("${value}")
    done
    if (( ${#unsorted[@]} > 0 )); then
      mapfile -t EMC_FREQUENCIES < <(printf '%s\n' "${unsorted[@]}" | sort -n -u)
      return 0
    fi
  fi

  # For debugfs/bwmgr or when no available_frequencies, use min/max range
  local emin emax
  if [[ -n "${EMC_MIN_FILE}" ]]; then
    emin="$(try_read_uint "${EMC_MIN_FILE}")" || return 1
  else
    emin=0
  fi
  emax="$(try_read_uint "${EMC_MAX_FILE}")" || return 1
  EMC_FREQUENCIES=("${emin}" "${emax}")
  return 0
}

emc_save_state() {
  if [[ -n "${EMC_MIN_FILE}" ]]; then
    EMC_ORIG_MIN="$(read_uint "${EMC_MIN_FILE}")"
  else
    EMC_ORIG_MIN=""
  fi
  EMC_ORIG_MAX="$(read_uint "${EMC_MAX_FILE}")"
}

emc_inject() {
  local target="$1"
  if [[ "${EMC_MODE}" == "bwmgr" ]]; then
    write_value "${EMC_MAX_FILE}" "${target}" || return 1
    return 0
  fi

  if [[ -n "${EMC_MIN_FILE}" ]]; then
    local cur_min cur_max
    cur_min="$(try_read_uint "${EMC_MIN_FILE}")" || return 1
    cur_max="$(try_read_uint "${EMC_MAX_FILE}")" || return 1
    if (( target < cur_min )); then
      write_value "${EMC_MIN_FILE}" "${target}" || return 1
      write_value "${EMC_MAX_FILE}" "${target}" || return 1
    elif (( target > cur_max )); then
      write_value "${EMC_MAX_FILE}" "${target}" || return 1
      write_value "${EMC_MIN_FILE}" "${target}" || return 1
    else
      write_value "${EMC_MIN_FILE}" "${target}" || return 1
      write_value "${EMC_MAX_FILE}" "${target}" || return 1
    fi
  else
    write_value "${EMC_MAX_FILE}" "${target}" || return 1
  fi
}

emc_restore_state() {
  local failed=0
  if [[ -n "${EMC_MIN_FILE}" ]]; then
    local cur_min cur_max
    cur_min="$(try_read_uint "${EMC_MIN_FILE}")" || { failed=1; }
    cur_max="$(try_read_uint "${EMC_MAX_FILE}")" || { failed=1; }
    if (( ! failed )); then
      if (( EMC_ORIG_MAX < cur_min )); then
        write_value "${EMC_MIN_FILE}" "${EMC_ORIG_MIN}" || failed=1
        write_value "${EMC_MAX_FILE}" "${EMC_ORIG_MAX}" || failed=1
      elif (( EMC_ORIG_MIN > cur_max )); then
        write_value "${EMC_MAX_FILE}" "${EMC_ORIG_MAX}" || failed=1
        write_value "${EMC_MIN_FILE}" "${EMC_ORIG_MIN}" || failed=1
      else
        write_value "${EMC_MIN_FILE}" "${EMC_ORIG_MIN}" || failed=1
        write_value "${EMC_MAX_FILE}" "${EMC_ORIG_MAX}" || failed=1
      fi
    fi
  else
    write_value "${EMC_MAX_FILE}" "${EMC_ORIG_MAX}" || failed=1
  fi

  if (( failed )); then
    echo "  EMC manual recovery:" >&2
    if [[ -n "${EMC_MIN_FILE}" ]]; then
      echo "    echo ${EMC_ORIG_MAX} | sudo tee ${EMC_MAX_FILE}" >&2
      echo "    echo ${EMC_ORIG_MIN} | sudo tee ${EMC_MIN_FILE}" >&2
    else
      echo "    echo ${EMC_ORIG_MAX} | sudo tee ${EMC_MAX_FILE}" >&2
    fi
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Interval and restoration
# ---------------------------------------------------------------------------
RESTORE_ACTIVE=0

restore_all() {
  local failed=0
  set +e
  if (( WANT_GPU && GPU_PRESENT )); then
    gpu_restore_state || failed=1
  fi
  if (( WANT_CPU && CPU_PRESENT )); then
    cpu_restore_state || failed=1
  fi
  if (( WANT_EMC && EMC_PRESENT )); then
    emc_restore_state || failed=1
  fi
  RESTORE_ACTIVE=0
  set -e

  if (( failed )); then
    echo "jetson_freq_inject: automatic restoration failed for one or more subsystems" >&2
    return 1
  fi
  echo "restored all subsystems to original state"
  return 0
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if (( RESTORE_ACTIVE )); then
    if ! restore_all; then
      status=1
    fi
  fi
  exit "${status}"
}

run_interval() {
  local duration_ms="$1"
  local start_ms end_ms now_ms remaining_ms sleep_ms sleep_seconds

  if [[ "${JETSON_FREQ_TEST_NO_SLEEP:-0}" == "1" ]]; then
    local status_parts=""
    if (( WANT_GPU && GPU_PRESENT )); then
      status_parts+=" gpu_freq_hz=$(read_uint "${GPU_CUR_FILE}")"
    fi
    if (( WANT_CPU && CPU_PRESENT && ${#CPU_DIRS[@]} > 0 )); then
      status_parts+=" cpu_freq_khz=$(try_read_uint "${CPU_DIRS[0]}/scaling_min_freq" 2>/dev/null || echo '?')"
    fi
    if (( WANT_EMC && EMC_PRESENT )); then
      status_parts+=" emc_freq=$(try_read_uint "${EMC_CUR_FILE}" 2>/dev/null || echo '?')"
    fi
    echo "remaining_ms=${duration_ms}${status_parts}"
    return 0
  fi

  start_ms="$(date +%s%3N)"
  end_ms=$((start_ms + duration_ms))
  while true; do
    now_ms="$(date +%s%3N)"
    remaining_ms=$((end_ms - now_ms))
    (( remaining_ms > 0 )) || break
    local status_parts=""
    if (( WANT_GPU && GPU_PRESENT )); then
      status_parts+=" gpu_freq_hz=$(read_uint "${GPU_CUR_FILE}")"
    fi
    if (( WANT_CPU && CPU_PRESENT && ${#CPU_DIRS[@]} > 0 )); then
      status_parts+=" cpu_freq_khz=$(try_read_uint "${CPU_DIRS[0]}/scaling_min_freq" 2>/dev/null || echo '?')"
    fi
    if (( WANT_EMC && EMC_PRESENT )); then
      status_parts+=" emc_freq=$(try_read_uint "${EMC_CUR_FILE}" 2>/dev/null || echo '?')"
    fi
    echo "remaining_ms=${remaining_ms}${status_parts}"
    sleep_ms="${remaining_ms}"
    (( sleep_ms > 1000 )) && sleep_ms=1000
    printf -v sleep_seconds '%d.%03d' "$((sleep_ms / 1000))" "$((sleep_ms % 1000))"
    sleep "${sleep_seconds}"
  done
}

# ---------------------------------------------------------------------------
# List mode
# ---------------------------------------------------------------------------
do_list() {
  local found=0

  if gpu_discover; then
    gpu_load_frequencies
    printf 'gpu_available_frequencies_hz='
    printf '%s ' "${GPU_FREQUENCIES[@]}"
    printf '\ngpu_min_freq_hz=%s\ngpu_max_freq_hz=%s\ngpu_current_freq_hz=%s\n' \
      "$(read_uint "${GPU_MIN_FILE}")" "$(read_uint "${GPU_MAX_FILE}")" "$(read_uint "${GPU_CUR_FILE}")"
    found=1
  else
    echo "gpu: not found"
  fi

  if cpu_discover; then
    cpu_load_frequencies
    printf 'cpu_count=%d\n' "${#CPU_DIRS[@]}"
    printf 'cpu_available_frequencies_khz='
    printf '%s ' "${CPU_FREQUENCIES[@]}"
    printf '\ncpu_min_freq_khz=%s\ncpu_max_freq_khz=%s\n' \
      "$(read_uint "${CPU_DIRS[0]}/scaling_min_freq")" "$(read_uint "${CPU_DIRS[0]}/scaling_max_freq")"
    found=1
  else
    echo "cpu: not found"
  fi

  if emc_discover; then
    emc_load_frequencies
    printf 'emc_mode=%s\n' "${EMC_MODE}"
    printf 'emc_available_frequencies='
    printf '%s ' "${EMC_FREQUENCIES[@]}"
    printf '\nemc_current='
    try_read_uint "${EMC_CUR_FILE}" 2>/dev/null || echo '?'
    printf '\n'
    found=1
  else
    echo "emc: not found"
  fi

  (( found )) || die "no supported subsystem found under ${SYSFS_ROOT}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
LEVEL=""
DURATION="20"
LIST_ONLY=0
SYSFS_ROOT="/sys"
TARGET_STR="all"
WANT_GPU=0
WANT_CPU=0
WANT_EMC=0

while (( $# > 0 )); do
  case "$1" in
    --level)
      require_value "$1" "$#"
      LEVEL="$2"
      shift 2
      ;;
    --target)
      require_value "$1" "$#"
      TARGET_STR="$2"
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

# Parse target
IFS=',' read -ra TARGET_PARTS <<< "${TARGET_STR}"
for part in "${TARGET_PARTS[@]}"; do
  case "${part}" in
    gpu)  WANT_GPU=1 ;;
    cpu)  WANT_CPU=1 ;;
    emc)  WANT_EMC=1 ;;
    all)  WANT_GPU=1; WANT_CPU=1; WANT_EMC=1 ;;
    *)    die "unknown target: ${part} (valid: gpu, cpu, emc, all)" ;;
  esac
done

if (( LIST_ONLY )); then
  do_list
  exit 0
fi

[[ -n "${LEVEL}" ]] || die "--level is required unless --list is used"
[[ "${LEVEL}" =~ ^[0-9]+$ ]] || die "--level must be an integer from 0 to 100"
LEVEL_VALUE=$((10#${LEVEL}))
(( LEVEL_VALUE >= 0 && LEVEL_VALUE <= 100 )) || die "--level must be an integer from 0 to 100"
DURATION_MS="$(seconds_to_ms "${DURATION}")" || die "--duration must be a positive integer or decimal number of seconds"

EFFECTIVE_UID="${JETSON_FREQ_TEST_EUID:-${EUID}}"
[[ "${EFFECTIVE_UID}" =~ ^[0-9]+$ ]] || die "invalid effective UID"
(( EFFECTIVE_UID == 0 )) || die "root privileges are required; run with sudo"
command -v flock >/dev/null 2>&1 || die "flock is required (install util-linux)"

LOCK_FILE="${JETSON_FREQ_LOCK_FILE:-/run/lock/jetson_freq_inject.lock}"
exec 9>"${LOCK_FILE}"
flock -n 9 || die "another frequency injection is already running"

# Discover requested subsystems
ACTIVE_TARGETS=""
if (( WANT_GPU )); then
  if gpu_discover; then
    gpu_load_frequencies
    ACTIVE_TARGETS+=" gpu"
  else
    die "GPU devfreq not found under ${SYSFS_ROOT}"
  fi
fi

if (( WANT_CPU )); then
  if cpu_discover; then
    cpu_load_frequencies
    ACTIVE_TARGETS+=" cpu"
  else
    die "CPU cpufreq not found under ${SYSFS_ROOT}"
  fi
fi

if (( WANT_EMC )); then
  if emc_discover; then
    emc_load_frequencies
    ACTIVE_TARGETS+=" emc"
  else
    warn "EMC frequency control not found under ${SYSFS_ROOT}; skipping EMC"
    WANT_EMC=0
  fi
fi

[[ -n "${ACTIVE_TARGETS}" ]] || die "no requested subsystem is available"

# Select target frequencies
GPU_TARGET=""
CPU_TARGET=""
EMC_TARGET=""
if (( WANT_GPU && GPU_PRESENT )); then
  GPU_TARGET="$(select_frequency "${LEVEL_VALUE}" "${GPU_FREQUENCIES[@]}")"
fi
if (( WANT_CPU && CPU_PRESENT )); then
  CPU_TARGET="$(select_frequency "${LEVEL_VALUE}" "${CPU_FREQUENCIES[@]}")"
fi
if (( WANT_EMC && EMC_PRESENT )); then
  EMC_TARGET="$(select_frequency "${LEVEL_VALUE}" "${EMC_FREQUENCIES[@]}")"
fi

# Save state
if (( WANT_GPU && GPU_PRESENT )); then
  gpu_save_state
fi
if (( WANT_CPU && CPU_PRESENT )); then
  cpu_save_state
fi
if (( WANT_EMC && EMC_PRESENT )); then
  emc_save_state
fi

# Inject: EMC -> CPU -> GPU (bandwidth first, then compute)
RESTORE_ACTIVE=1
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if (( WANT_EMC && EMC_PRESENT )); then
  emc_inject "${EMC_TARGET}"
fi
if (( WANT_CPU && CPU_PRESENT )); then
  cpu_inject "${CPU_TARGET}"
fi
if (( WANT_GPU && GPU_PRESENT )); then
  gpu_inject "${GPU_TARGET}"
fi

# Report
echo "targets:${ACTIVE_TARGETS} level=${LEVEL_VALUE} duration_sec=${DURATION}"
if [[ -n "${GPU_TARGET}" ]]; then
  echo "gpu_target_freq_hz=${GPU_TARGET}"
fi
if [[ -n "${CPU_TARGET}" ]]; then
  echo "cpu_target_freq_khz=${CPU_TARGET} cpu_count=${#CPU_DIRS[@]}"
fi
if [[ -n "${EMC_TARGET}" ]]; then
  echo "emc_target_freq=${EMC_TARGET} emc_mode=${EMC_MODE}"
fi
echo "warning: thermal, electrical, or power-mode limits may still throttle observed clocks"

run_interval "${DURATION_MS}"
echo "frequency disturbance complete"
