#!/usr/bin/env bash
# batchsize matrix case: 4-GPU PP, CUDA, LSS on.
# usage: run_matrix_case.sh <tag> <batch> <off|l1|l2>
set -u
TAG="${1:?tag}"; BATCH="${2:?batch}"; MODE="${3:?off|l1|l2}"

ROOT="$HOME/B01/EdgeVisor"
BIN="$ROOT/EdgeVisor/dllama"
MODEL="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOK="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
LOGDIR="$ROOT/runtime_logs/gpu_batch_matrix/$TAG"
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"

P1=$(( 26100 + BATCH * 10 + 1 )); P2=$(( 26100 + BATCH * 10 + 2 )); P3=$(( 26100 + BATCH * 10 + 3 ))
case "$MODE" in
  l1)  BUBBLE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1) ;;
  l2)  BUBBLE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L2=1) ;;
  *)   BUBBLE_ENV=() ;;
esac

# memory guard: refuse to run when free VRAM is too small
FREE_MIN=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk '{print $2-$4}' | sort -n | head -1)
if [ "$FREE_MIN" -lt 2500 ]; then
  echo "SKIP: min free VRAM ${FREE_MIN} MiB < 2500 MiB" | tee "$LOGDIR/SKIPPED.txt"
  exit 3
fi

env "${BUBBLE_ENV[@]}" "$BIN" worker --port $P1 --nthreads 1 --gpu-index 1 --backend cuda >"$LOGDIR/worker1.log" 2>&1 &
W1=$!
env "${BUBBLE_ENV[@]}" "$BIN" worker --port $P2 --nthreads 1 --gpu-index 2 --backend cuda >"$LOGDIR/worker2.log" 2>&1 &
W2=$!
env "${BUBBLE_ENV[@]}" "$BIN" worker --port $P3 --nthreads 1 --gpu-index 3 --backend cuda >"$LOGDIR/worker3.log" 2>&1 &
W3=$!
sleep 5

PROMPT="Write a detailed story about a robot learning to paint. The robot lived in a small workshop near the river. Every morning it watched the sunrise through the dusty window and dreamed of colors. One day it found an old brush and a box of paints forgotten by its owner."

env DLLAMA_NBATCHES="$BATCH" "${BUBBLE_ENV[@]}" \
  "$BIN" inference \
  --prompt "$PROMPT" \
  --steps 80 \
  --model "$MODEL" --tokenizer "$TOK" \
  --buffer-float-type q80 --nthreads 1 --max-seq-len 512 \
  --temperature 0 --seed 1 \
  --backend cuda --gpu-index 0 \
  --workers "127.0.0.1:$P1" "127.0.0.1:$P2" "127.0.0.1:$P3" \
  --ratios "1@7*1@7*1@7*1@7" \
  --enable-plan-barrier \
  --enable-pp-migration \
  --runtime-redundant-boundary-layers 1 \
  --runtime-active-seg-enabled 1 \
  --runtime-redundant-seg-enabled 0 \
  --last-stage-sampling \
  --benchmark >"$LOGDIR/root.log" 2>&1
RC=$?

kill $W1 $W2 $W3 2>/dev/null
wait 2>/dev/null

# extract per-forward bubble stats + summary
grep -E "🔶 Pred" "$LOGDIR/root.log" | sed -E "s/.*\| pos=([0-9]+) \|.*\| /\1:/" | tr "\n" " " >"$LOGDIR/tokens.txt"
echo "RC=$RC batch=$BATCH mode=$MODE"
grep "Prediction tokens/s" "$LOGDIR/root.log" || true
grep -E "bubbleDrain/fwd|bubbleSeg=" "$LOGDIR/root.log" | head -4 || true
echo "LOGDIR=$LOGDIR"
