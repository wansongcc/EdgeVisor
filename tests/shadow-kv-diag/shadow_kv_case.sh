#!/usr/bin/env bash
# Shadow KV numerical-diagnosis runner (CPU only).
# usage: shadow_kv_case.sh <tag> <2pp|3pp> <async|sync|off> <nbatches> [steps]
set -u
TAG="${1:?tag}"; TOPO="${2:?2pp|3pp}"; MODE="${3:?async|sync|off}"; NB="${4:-1}"; STEPS="${5:-8}"; PROMPT="${6:-Write a comma-separated list of the numbers from 1 to 20.}"

ROOT="$HOME/B01/EdgeVisor"
BIN="$ROOT/EdgeVisor/dllama"
MODEL="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOK="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
LOGDIR="$ROOT/runtime_logs/shadow_kv_diag/$TAG"
DUMP="$LOGDIR/dump"
rm -rf "$LOGDIR"; mkdir -p "$DUMP"

BASE=$(( 28000 + RANDOM % 2000 ))
P1=$BASE; P2=$((BASE+1))

if [ "$MODE" = "off" ]; then
  BUBBLE_ENV=()
elif [ "$MODE" = "sync" ]; then
  BUBBLE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=0 DLLAMA_BUBBLE_SHADOW_KV_LOG=1)
else
  BUBBLE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1)
fi

if [ "$TOPO" = "2pp" ]; then
  RATIOS="1@14*1@14"; WORKERS=( "127.0.0.1:$P1" ); NWORK=1
else
  RATIOS="1@10*1@9*1@9"; WORKERS=( "127.0.0.1:$P1" "127.0.0.1:$P2" ); NWORK=2
fi

# start workers (env DLLAMA_DUMP_KV_DIR needed on each node; bubble env propagates from root bootstrap)
env DLLAMA_DUMP_KV_DIR="$DUMP" "${BUBBLE_ENV[@]}" \
  "$BIN" worker --port "$P1" --nthreads 1 >"$LOGDIR/worker1.log" 2>&1 &
W1=$!
if [ "$NWORK" = "2" ]; then
  env DLLAMA_DUMP_KV_DIR="$DUMP" "${BUBBLE_ENV[@]}" \
    "$BIN" worker --port "$P2" --nthreads 1 >"$LOGDIR/worker2.log" 2>&1 &
  W2=$!
fi
sleep 3

env DLLAMA_DUMP_KV_DIR="$DUMP" DLLAMA_NBATCHES="$NB" "${BUBBLE_ENV[@]}" \
  "$BIN" inference \
  --prompt "$PROMPT" \
  --steps "$STEPS" \
  --model "$MODEL" --tokenizer "$TOK" \
  --buffer-float-type q80 --nthreads 1 --max-seq-len 512 \
  --temperature 0 --seed 1 \
  --backend cpu \
  --workers "${WORKERS[@]}" \
  --ratios "$RATIOS" \
  --enable-plan-barrier \
  --enable-stage-full-weights \
  --enable-pp-migration \
  --runtime-redundant-boundary-layers 1 \
  --runtime-active-seg-enabled 1 \
  --runtime-redundant-seg-enabled 0 \
  ${EXTRA_ARGS:-} \
  --benchmark >"$LOGDIR/root.log" 2>&1
RC=$?

kill $W1 ${W2:-} 2>/dev/null
wait 2>/dev/null
echo "RC=$RC LOGDIR=$LOGDIR"
ls "$DUMP" | wc -l
