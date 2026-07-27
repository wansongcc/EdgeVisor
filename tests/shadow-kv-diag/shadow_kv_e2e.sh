#!/usr/bin/env bash
# End-to-end PP layer migration consistency test (CPU only).
# Migrates layer 14 from stage1 (node1) to stage0 (node0) at triggerPos=32.
# usage: run_e2e.sh <tag> <baseline|transfer|recompute|shadow>
set -u
TAG="${1:?tag}"; MODE="${2:?baseline|transfer|recompute|shadow}"
TRIGGER_POS="${3:-32}"
LAYER_COUNT="${4:-1}"
PROMPT="${5:-Write a comma-separated list of the numbers from 1 to 20.}"
STEPS="${6:-48}"

ROOT="$HOME/B01/EdgeVisor"
BIN="$ROOT/EdgeVisor/dllama"
CLIENT="$ROOT/EdgeVisor/examples/plan-uds-client.py"
MODEL="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOK="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
LOGDIR="$ROOT/runtime_logs/shadow_kv_diag/$TAG"
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
SOCK="/tmp/shadow_kv_e2e_${TAG}.sock"
rm -f "$SOCK"

BASE=$(( 26000 + RANDOM % 2000 ))
P1=$BASE

BUBBLE_ENV=()
ABLATION_ENV=()
case "$MODE" in
  shadow)
    BUBBLE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1)
    ABLATION_ENV=(EDGEVISOR_SHADOW_KV_MODE=enabled)
    ;;
  bubbleonly)
    BUBBLE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1)
    ;;
  transfer) ABLATION_ENV=(EDGEVISOR_SHADOW_KV_MODE=disabled_transfer) ;;
  recompute) ABLATION_ENV=(EDGEVISOR_SHADOW_KV_MODE=disabled_recompute) ;;
  baseline) ;;
esac

env "${BUBBLE_ENV[@]}" "${ABLATION_ENV[@]}" \
  "$BIN" worker --port "$P1" --nthreads 1 >"$LOGDIR/worker1.log" 2>&1 &
W1=$!
sleep 3

env "${BUBBLE_ENV[@]}" "${ABLATION_ENV[@]}" \
  "$BIN" inference \
  --prompt "$PROMPT" \
  --steps "$STEPS" \
  --model "$MODEL" --tokenizer "$TOK" \
  --buffer-float-type q80 --nthreads 1 --max-seq-len 512 \
  --temperature 0 --seed 1 \
  --backend cpu \
  --workers "127.0.0.1:$P1" \
  --ratios "1@14*1@14" \
  --enable-plan-barrier \
  --enable-stage-full-weights \
  --enable-pp-migration \
  --runtime-redundant-boundary-layers 1 \
  --runtime-active-seg-enabled 1 \
  --runtime-redundant-seg-enabled 1 \
  --plan-ctrl-socket "$SOCK" \
  --benchmark >"$LOGDIR/root.log" 2>&1 &
R1=$!

if [ "$MODE" != "baseline" ] && [ "$MODE" != "bubbleonly" ]; then
  # wait for UDS to come up, then arm an exact-trigger migration at pos 32
  for i in $(seq 1 60); do
    [ -S "$SOCK" ] && break
    sleep 1
  done
  python3 "$CLIENT" "$SOCK" set_pp_migration \
    --seq 42 --mode exact --trigger-pos "$TRIGGER_POS" \
    --stage 0 --from 1 --to 0 --layer-count "$LAYER_COUNT" >"$LOGDIR/uds_response.json" 2>&1
fi

wait $R1
RC=$?
kill $W1 2>/dev/null
wait 2>/dev/null
echo "RC=$RC LOGDIR=$LOGDIR"
