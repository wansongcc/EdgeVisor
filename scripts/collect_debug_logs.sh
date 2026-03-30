#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/collect_debug_logs.sh [options]

Modes:
  --mode both|root|worker      Run both/root/worker collection. Default: both.

Optional:
  --root-cmd "<cmd>"          Root command to run (required in mode=both/root).
  --worker-cmd "<cmd>"        Worker command to run in background.
  --workspace <path>           Workspace directory. Default: parent of script.
  --pos <int>                  Position filter for layer-out trace. Default: 5.
  --layer-filter "r1|r2|..."   Layer regex for focused logs. Default: 2|3|4|5|6.
  --run-id <id>                Fixed run ID. Default: timestamp.
  --no-compare                 Disable layer-out baseline compare.
  -h, --help                   Show this help.

What this script does (one command):
  1) Creates logs/<RUN_ID>/
  2) Injects low-noise debug env vars
  3) Runs root (and optional worker) with raw logs
  4) Extracts key logs for analysis
  5) Extracts focused layer/pos logs (default layer=2..6, pos=5)
  6) Packs everything into logs/<RUN_ID>.tar.gz

Examples:
  scripts/collect_debug_logs.sh \
    --mode root \
    --root-cmd "./dllama inference --interactive --prompt 'The capital of France is' --steps 256 --model /workspace/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m --tokenizer /workspace/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t --buffer-float-type q80 --nthreads 2 --max-seq-len 2048 --enable-plan-barrier --enable-stage-full-weights --benchmark --workers rpi-trixie5:9999 --ratios 1:1"

  scripts/collect_debug_logs.sh \
    --mode both \
    --root-cmd "./dllama inference ..." \
    --worker-cmd "./dllama worker --port 9999 --model ... --ratios 1:1"

  # Two containers sharing the same mounted workspace/log directory
  # Container A (worker):
  scripts/collect_debug_logs.sh --mode worker --run-id 20260309_090000 --worker-cmd "./dllama worker --port 9999"
  # Container B (root):
  scripts/collect_debug_logs.sh --mode root --run-id 20260309_090000 --root-cmd "./dllama inference ..."
EOF
}

ROOT_CMD=""
WORKER_CMD=""
MODE="both"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POS_FILTER="5"
LAYER_FILTER="2|3|4|5|6"
RUN_ID=""
NO_COMPARE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --root-cmd)
      ROOT_CMD="$2"
      shift 2
      ;;
    --worker-cmd)
      WORKER_CMD="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --pos)
      POS_FILTER="$2"
      shift 2
      ;;
    --layer-filter)
      LAYER_FILTER="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --no-compare)
      NO_COMPARE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "both" && "$MODE" != "root" && "$MODE" != "worker" ]]; then
  echo "Error: --mode must be one of: both, root, worker" >&2
  usage
  exit 1
fi

if [[ "$MODE" == "both" || "$MODE" == "root" ]]; then
  if [[ -z "$ROOT_CMD" ]]; then
    echo "Error: --root-cmd is required in mode=$MODE." >&2
    usage
    exit 1
  fi
fi

if [[ "$MODE" == "worker" && -z "$WORKER_CMD" ]]; then
  echo "Error: --worker-cmd is required in mode=worker." >&2
  usage
  exit 1
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

LOG_DIR="$WORKSPACE/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

ROOT_RAW="$LOG_DIR/root.raw.log"
WORKER_RAW="$LOG_DIR/worker.raw.log"
ROOT_KEY="$LOG_DIR/root.key.log"
WORKER_KEY="$LOG_DIR/worker.key.log"
ROOT_FOCUS="$LOG_DIR/root.focus.log"
WORKER_FOCUS="$LOG_DIR/worker.focus.log"
META_FILE="$LOG_DIR/run.meta.txt"
ARCHIVE_PATH="$WORKSPACE/logs/$RUN_ID.tar.gz"

COMPARE_VALUE="1"
if [[ "$NO_COMPARE" == "1" ]]; then
  COMPARE_VALUE="0"
fi

write_meta() {
  {
    echo "mode=$MODE"
    echo "run_id=$RUN_ID"
    echo "workspace=$WORKSPACE"
    echo "created_at=$(date -Iseconds)"
    echo "pos_filter=$POS_FILTER"
    echo "layer_filter=$LAYER_FILTER"
    echo "compare_enabled=$COMPARE_VALUE"
    echo "root_cmd=$ROOT_CMD"
    if [[ -n "$WORKER_CMD" ]]; then
      echo "worker_cmd=$WORKER_CMD"
    fi
  } > "$META_FILE"
}

build_env_prefix() {
  cat <<EOF
export DLLAMA_MIGRATION_OP_TRACE=1;
export DLLAMA_MIGRATION_OP_TRACE_LIMIT=0;
export DLLAMA_DEBUG_LAYER_OUT_TRACE=1;
export DLLAMA_DEBUG_LAYER_OUT_TRACE_POS=$POS_FILTER;
export DLLAMA_DEBUG_LAYER_OUT_TRACE_LIMIT=500;
export DLLAMA_DEBUG_LAYER_OUT_COMPARE=$COMPARE_VALUE;
export DLLAMA_DEBUG_ROOT_LOGITS_SPLIT=1;
export DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE="${DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE:-1}";
export DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_POS="${DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_POS:-$POS_FILTER}";
export DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_LIMIT="${DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_LIMIT:-500}";
export DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_SLICES="${DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_SLICES:-2}";
export DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_SAMPLES="${DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_SAMPLES:-4}";
unset DLLAMA_DEBUG_ATT;
unset DLLAMA_DEBUG_ATT_QK;
unset DLLAMA_DEBUG_ATT_SCORES;
unset DLLAMA_DEBUG_KVCACHE_PER_HEAD;
unset DLLAMA_DEBUG_FINAL_NORM_INPUT;
unset DLLAMA_DEBUG_FINAL_LOGITS_SLICE;
EOF
}

run_with_env_and_log() {
  local cmd="$1"
  local log_file="$2"
  local env_prefix
  env_prefix="$(build_env_prefix)"

  stdbuf -oL -eL bash -lc "set -euo pipefail; cd \"$WORKSPACE\"; $env_prefix $cmd" \
    2>&1 | tee "$log_file"
}

extract_logs() {
  if [[ -f "$ROOT_RAW" ]]; then
    grep -E '\[op-trace\]|\[layer-out\]|\[layer-out-diff\]|\[root-logits-split\]|\[sync-topk\]|\[sync-topk-diff\]|\[logits-gather-slice\]' \
      "$ROOT_RAW" > "$ROOT_KEY" || true
  else
    : > "$ROOT_KEY"
  fi

  if [[ -f "$WORKER_RAW" ]]; then
    grep -E '\[op-trace\]|\[layer-out\]|\[layer-out-diff\]|\[op-trace\]\[attn-kv\]' \
      "$WORKER_RAW" > "$WORKER_KEY" || true
  else
    : > "$WORKER_KEY"
  fi

  if [[ -s "$ROOT_KEY" ]]; then
    grep -E "layer=($LAYER_FILTER)\\b.*pos=$POS_FILTER" "$ROOT_KEY" \
      | grep -E 'block_merge_add2|block_matmul_w2|layer-out-diff|final_merge_add|root-logits-split|sync-topk-diff' \
      > "$ROOT_FOCUS" || true
  else
    : > "$ROOT_FOCUS"
  fi

  if [[ -s "$WORKER_KEY" ]]; then
    grep -E "layer=($LAYER_FILTER)\\b.*pos=$POS_FILTER" "$WORKER_KEY" \
      | grep -E 'block_merge_add2|block_matmul_w2|layer-out-diff|\[op-trace\]\[attn-kv\]' \
      > "$WORKER_FOCUS" || true
  else
    : > "$WORKER_FOCUS"
  fi
}

pack_logs() {
  rm -f "$ARCHIVE_PATH"
  tar -czf "$ARCHIVE_PATH" -C "$WORKSPACE/logs" "$RUN_ID"
}

write_meta

echo "[collect] log dir: $LOG_DIR"

worker_pid=""
if [[ "$MODE" == "both" && -n "$WORKER_CMD" ]]; then
  echo "[collect] starting worker in background"
  run_with_env_and_log "$WORKER_CMD" "$WORKER_RAW" &
  worker_pid="$!"
  sleep 2
fi

cleanup() {
  if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then
    echo "[collect] stopping worker (pid=$worker_pid)"
    kill "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$MODE" == "worker" ]]; then
  echo "[collect] running worker command (foreground)"
  run_with_env_and_log "$WORKER_CMD" "$WORKER_RAW"
else
  echo "[collect] running root command"
  run_with_env_and_log "$ROOT_CMD" "$ROOT_RAW"
fi

echo "[collect] extracting key/focus logs"
extract_logs

echo "[collect] packing archive"
pack_logs

echo "[collect] done"
echo "[collect] root raw:    $ROOT_RAW"
if [[ -f "$WORKER_RAW" ]]; then
  echo "[collect] worker raw:  $WORKER_RAW"
fi
echo "[collect] root key:    $ROOT_KEY"
echo "[collect] worker key:  $WORKER_KEY"
echo "[collect] root focus:  $ROOT_FOCUS"
echo "[collect] worker focus:$WORKER_FOCUS"
echo "[collect] archive:     $ARCHIVE_PATH"
