#!/usr/bin/env bash
# L2 catch-up verification per batchSize (chat harness, L1-disable forces debt).
# usage: run_matrix_l2_catchup.sh <tag> <batch>
set -u
TAG="${1:?tag}"; BATCH="${2:?batch}"

ROOT="$HOME/B01/EdgeVisor"
BIN="$ROOT/EdgeVisor/dllama"
CLIENT="$ROOT/EdgeVisor/examples/plan-uds-client.py"
MODEL="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOK="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
LOGDIR="$ROOT/runtime_logs/gpu_batch_matrix/$TAG"
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
SOCK="/tmp/gpu_bm_l2_${TAG}.sock"
rm -f "$SOCK"

P1=$(( 26200 + BATCH * 10 + 1 )); P2=$(( 26200 + BATCH * 10 + 2 )); P3=$(( 26200 + BATCH * 10 + 3 ))
BUBBLE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L2=1 DLLAMA_SHADOW_L1_DISABLE=1)

# CSV fields are (total, used); the old `$2-$4` computed *used* instead of
# *free*, so an empty GPU (used=0) was falsely SKIPped and a busy one passed.
FREE_MIN=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}' | sort -n | head -1)
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

( printf '\n'; echo "Write a comma-separated list of the numbers from 1 to 20."; sleep 600 ) | \
env DLLAMA_NBATCHES="$BATCH" "${BUBBLE_ENV[@]}" \
  "$BIN" chat \
  --steps 48 \
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
  --plan-ctrl-socket "$SOCK" \
  --benchmark >"$LOGDIR/root.log" 2>&1 &
R1=$!

for i in $(seq 1 420); do
  N=$(grep -c "👱 User" "$LOGDIR/root.log" 2>/dev/null || true); [ -z "$N" ] && N=0
  [ "$N" -ge 2 ] && break
  sleep 1
done

echo "=== debt before window ==="
python3 "$CLIENT" "$SOCK" shadow_debt | tee "$LOGDIR/debt_before.json" | grep -E "debtEntries|debtBytes|catchup"
echo "=== tool_window_begin ==="
python3 "$CLIENT" "$SOCK" tool_window_begin >/dev/null
sleep 4
echo "=== debt after catch-up ==="
python3 "$CLIENT" "$SOCK" shadow_debt | tee "$LOGDIR/debt_after.json" | grep -E "debtEntries|debtBytes|catchup"
python3 "$CLIENT" "$SOCK" tool_window_end >/dev/null

kill $R1 $W1 $W2 $W3 2>/dev/null
wait 2>/dev/null
echo "LOGDIR=$LOGDIR"
