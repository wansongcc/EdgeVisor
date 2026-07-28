#!/usr/bin/env bash
# Shadow L2 (tool-wait catch-up) unit scenario, CPU only.
# Uses chat mode: after a turn completes, the root idles on stdin => a real
# tool-wait window. We then drive tool_window_begin/end + shadow_debt via UDS.
# usage: run_l2_case.sh <tag> [nbatches] [steps]
set -u
TAG="${1:?tag}"; NB="${2:-32}"; STEPS="${3:-48}"

ROOT="$HOME/B01/EdgeVisor"
BIN="$ROOT/EdgeVisor/dllama"
CLIENT="$ROOT/EdgeVisor/examples/plan-uds-client.py"
MODEL="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOK="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
LOGDIR="$ROOT/runtime_logs/shadow_kv_diag/$TAG"
DUMP="$LOGDIR/dump"
rm -rf "$LOGDIR"; mkdir -p "$DUMP"
SOCK="/tmp/shadow_l2_${TAG}.sock"
rm -f "$SOCK"

BASE=$(( 24000 + RANDOM % 2000 ))
P1=$BASE

BUBBLE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L2=1)

env DLLAMA_DUMP_KV_DIR="$DUMP" "${BUBBLE_ENV[@]}" \
  "$BIN" worker --port "$P1" --nthreads 1 >"$LOGDIR/worker1.log" 2>&1 &
W1=$!
sleep 3

# chat stdin: empty system prompt, then one user prompt, then hold open.
( printf '\n'; echo "Write a comma-separated list of the numbers from 1 to 20."; sleep 900 ) | \
env DLLAMA_DUMP_KV_DIR="$DUMP" DLLAMA_NBATCHES="$NB" "${BUBBLE_ENV[@]}" \
  "$BIN" chat \
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
  --runtime-redundant-seg-enabled 0 \
  --plan-ctrl-socket "$SOCK" \
  --benchmark >"$LOGDIR/root.log" 2>&1 &
R1=$!

# wait for the turn to finish (root waits for the next user input)
for i in $(seq 1 300); do
  N=$(grep -c "👱 User" "$LOGDIR/root.log" 2>/dev/null || true)
  [ -z "$N" ] && N=0
  [ "$N" -ge 2 ] && break
  sleep 1
done

echo "=== [1] shadow_debt before tool window (expect debtEntries > 0) ==="
python3 "$CLIENT" "$SOCK" shadow_debt | tee "$LOGDIR/debt_before.json"

echo "=== [2] tool_window_begin ==="
python3 "$CLIENT" "$SOCK" tool_window_begin | tee "$LOGDIR/window_begin.json"
sleep 8

echo "=== [2] shadow_debt after catch-up (expect debtEntries = 0) ==="
python3 "$CLIENT" "$SOCK" shadow_debt | tee "$LOGDIR/debt_after.json"

echo "=== [4] tool_window_end ==="
python3 "$CLIENT" "$SOCK" tool_window_end | tee "$LOGDIR/window_end.json"

sleep 2
kill $R1 $W1 2>/dev/null
wait 2>/dev/null
echo "LOGDIR=$LOGDIR"
