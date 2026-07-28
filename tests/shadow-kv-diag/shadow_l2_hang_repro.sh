#!/usr/bin/env bash
# Debug repro for the L2 window hang: turn2 with/without tool_window_begin.
# usage: run_l2_hang_repro.sh <tag> <window|nowindow>
set -u
TAG="${1:?tag}"; MODE="${2:?window|nowindow}"

ROOT="$HOME/B01/EdgeVisor"
BIN="$ROOT/EdgeVisor/dllama"
CLIENT="$ROOT/EdgeVisor/examples/plan-uds-client.py"
MODEL="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOK="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
LOGDIR="$ROOT/runtime_logs/shadow_kv_diag/$TAG"
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
SOCK="/tmp/shadow_l2dbg_${TAG}.sock"
FIFO="/tmp/shadow_l2dbg_${TAG}.fifo"
rm -f "$SOCK" "$FIFO"; mkfifo "$FIFO"

BASE=$(( 18000 + RANDOM % 2000 ))
P1=$BASE
BUBBLE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L2=1 DLLAMA_SHADOW_L1_DISABLE=1 DLLAMA_SEG_RUNTIME_PRINT=1)

env "${BUBBLE_ENV[@]}" strace -f -tt -yy -e trace=recvfrom,sendto,read,write,poll,ppoll,nanosleep,futex -o "$LOGDIR/worker.strace" "$BIN" worker --port "$P1" --nthreads 1 >"$LOGDIR/worker1.log" 2>&1 &
W1=$!
sleep 3

( exec 3>"$FIFO"; printf '\n' >&3; python3 -c "print('Please summarize the following text. ' + 'The quick brown fox jumps over the lazy dog near the river bank. ' * 25)" >&3; sleep 1200 ) &
FHOLD=$!

env DLLAMA_NBATCHES=32 "${BUBBLE_ENV[@]}" \
  strace -f -tt -yy -e trace=recvfrom,sendto,read,write,poll,ppoll,nanosleep,futex -o "$LOGDIR/root.strace" "$BIN" chat --steps 48 \
  --model "$MODEL" --tokenizer "$TOK" \
  --buffer-float-type q80 --nthreads 1 --max-seq-len 512 \
  --temperature 0 --seed 1 --backend cpu \
  --workers "127.0.0.1:$P1" --ratios "1@14*1@14" \
  --enable-plan-barrier --enable-stage-full-weights --enable-pp-migration \
  --runtime-redundant-boundary-layers 1 --runtime-active-seg-enabled 1 --runtime-redundant-seg-enabled 0 \
  --plan-ctrl-socket "$SOCK" \
  --benchmark <"$FIFO" >"$LOGDIR/root.log" 2>&1 &
R1=$!

for i in $(seq 1 600); do
  if grep -q "Network is in blocking mode" "$LOGDIR/worker1.log" 2>/dev/null; then break; fi
  sleep 1
done
echo "turn1 idle detected"

if [ "$MODE" = "window" ]; then
  python3 "$CLIENT" "$SOCK" tool_window_begin
fi
sleep 0.15
echo "Write a comma-separated list of the numbers from 21 to 40." >"$FIFO"
echo "prompt2 sent; watching turn-2 progress for 60s..."

ROOTPID=$R1
WORKERPID=$(pgrep -f "dllama worker --port $P1" | head -1)
for i in $(seq 1 12); do
  sleep 0.15
  LAST=$(grep -oE "pos=[0-9]+" "$LOGDIR/root.log" | tail -1)
  echo "t+$((i*5))s root last $LAST"
done

echo "=== ss queues (root<->worker) ==="
ss -tnp 2>/dev/null | grep -E "$P1|dllama" | head -8
echo "=== worker strace tail ==="
tail -30 "$LOGDIR/worker.strace"
echo "=== root.log tail ==="
tail -4 "$LOGDIR/root.log"
echo "=== worker1.log tail ==="
tail -4 "$LOGDIR/worker1.log"

kill $R1 $W1 $FHOLD 2>/dev/null
wait 2>/dev/null
echo "LOGDIR=$LOGDIR"
