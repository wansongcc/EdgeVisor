#!/usr/bin/env bash
# Shadow L2 multi-window (continuous tool-wait) smoke, GPU (4xT4 CUDA).
# Drives N consecutive tool_window_begin / shadow_debt / tool_window_end
# cycles over a single chat session to verify that L2 catch-up runs in every
# window and debt returns to 0 each time.
#
# usage: shadow_l2_multi_window_gpu.sh <tag> <mode> [nbatches] [steps] [n_windows]
#   mode in: off | l1 | l2 | l2_l1off
set -u
TAG="${1:?tag}"; MODE="${2:?mode off|l1|l2|l2_l1off}"; NB="${3:-8}"; STEPS="${4:-32}"; NWIN="${5:-5}"

ROOT="$HOME/B01/EdgeVisor"
BIN="$ROOT/EdgeVisor/dllama"
CLIENT="$ROOT/EdgeVisor/examples/plan-uds-client.py"
MODEL="${SPOT_MODEL:-/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m}"
TOK="${SPOT_TOKENIZER:-/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t}"
LOGDIR="$ROOT/runtime_logs/shadow_kv_diag/$TAG"
DUMP="$LOGDIR/dump"
rm -rf "$LOGDIR"; mkdir -p "$DUMP"
SOCK="/tmp/shadow_l2_multi_gpu_${TAG}.sock"
rm -f "$SOCK"

case "$MODE" in
  off)         MODE_ENV=() ;;
  l1)          MODE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L1_DISABLE=1) ;;
  l2)          MODE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L2=1) ;;
  l2_l1off)    MODE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L2=1 DLLAMA_SHADOW_L1_DISABLE=1) ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

P1=$(( 27000 + RANDOM % 1000 ))
P2=$(( 28000 + RANDOM % 1000 ))
P3=$(( 29000 + RANDOM % 1000 ))

# 3-worker GPU pipeline (one CUDA worker per stage 1/2/3; root uses gpu-index 0).
env "${MODE_ENV[@]}" \
  "$BIN" worker --port "$P1" --nthreads 1 --gpu-index 1 --backend cuda \
  >"$LOGDIR/worker1.log" 2>&1 &
W1=$!
env "${MODE_ENV[@]}" \
  "$BIN" worker --port "$P2" --nthreads 1 --gpu-index 2 --backend cuda \
  >"$LOGDIR/worker2.log" 2>&1 &
W2=$!
env "${MODE_ENV[@]}" \
  "$BIN" worker --port "$P3" --nthreads 1 --gpu-index 3 --backend cuda \
  >"$LOGDIR/worker3.log" 2>&1 &
W3=$!
echo "workers PIDs: $W1 $W2 $W3 (ports $P1 $P2 $P3)"

# Wait for workers to finish init (look for stage-full-weights or first sync done)
echo "[$(date +%T)] waiting for worker init..."
for i in $(seq 1 300); do
  ok=1
  for f in "$LOGDIR/worker1.log" "$LOGDIR/worker2.log" "$LOGDIR/worker3.log"; do
    if ! grep -q "first sync done\|first fwd\|stage full weights\|listening for control\|awaiting root\|network loop" "$f" 2>/dev/null; then
      ok=0; break
    fi
  done
  [ "$ok" = "1" ] && break
  sleep 1
done
sleep 3
echo "[$(date +%T)] worker init done. settling 3s."

( printf '\n'; echo "Write a comma-separated list of the numbers from 1 to 20."; sleep 1800 ) | \
env DLLAMA_DUMP_KV_DIR="$DUMP" DLLAMA_NBATCHES="$NB" "${MODE_ENV[@]}" \
  "$BIN" chat \
  --steps "$STEPS" \
  --model "$MODEL" --tokenizer "$TOK" \
  --buffer-float-type q80 --nthreads 1 --max-seq-len 512 \
  --temperature 0 --seed 1 \
  --backend cuda --gpu-index 0 \
  --workers "127.0.0.1:$P1" "127.0.0.1:$P2" "127.0.0.1:$P3" \
  --ratios "1@7*1@7*1@7*1@7" \
  --enable-plan-barrier \
  --enable-stage-full-weights \
  --enable-pp-migration \
  --runtime-redundant-boundary-layers 1 \
  --runtime-active-seg-enabled 1 \
  --runtime-redundant-seg-enabled 0 \
  --last-stage-sampling \
  --plan-ctrl-socket "$SOCK" \
  --benchmark >"$LOGDIR/root.log" 2>&1 &
R1=$!
echo "root pid=$R1"

# Wait for turn 1 to finish (root prints second "👱 User" while waiting for next input).
echo "[$(date +%T)] waiting for root turn 1 to complete..."
for i in $(seq 1 300); do
  N=$(grep -c "👱 User" "$LOGDIR/root.log" 2>/dev/null || true)
  [ -z "$N" ] && N=0
  [ "$N" -ge 2 ] && break
  sleep 1
done
echo "[$(date +%T)] turn-1 ready (👱 User count=$N)"

echo "=== multi-window sweep ($NWIN cycles, mode=$MODE) ===" | tee "$LOGDIR/multi_window.log"
PASS=1
for w in $(seq 1 "$NWIN"); do
  echo "--- window $w ($(date +%T)) ---" | tee -a "$LOGDIR/multi_window.log"
  DB=$(python3 "$CLIENT" "$SOCK" shadow_debt 2>/dev/null || echo '{"ok":false}')
  echo "  pre:  $DB" | tee -a "$LOGDIR/multi_window.log"
  python3 "$CLIENT" "$SOCK" tool_window_begin >"$LOGDIR/win${w}_begin.json" 2>&1 || true
  sleep 6   # simulate tool call duration (e.g. mock_long_task)
  DA=$(python3 "$CLIENT" "$SOCK" shadow_debt 2>/dev/null || echo '{"ok":false}')
  echo "  mid:  $DA" | tee -a "$LOGDIR/multi_window.log"
  sleep 2
  python3 "$CLIENT" "$SOCK" tool_window_end >"$LOGDIR/win${w}_end.json" 2>&1 || true
  DF=$(python3 "$CLIENT" "$SOCK" shadow_debt 2>/dev/null || echo '{"ok":false}')
  echo "  post: $DF" | tee -a "$LOGDIR/multi_window.log"
done

sleep 2
echo "=== shutting down ===" | tee -a "$LOGDIR/multi_window.log"
kill $R1 $W1 $W2 $W3 2>/dev/null
wait 2>/dev/null

# Extract generated tokens.
python3 - "$LOGDIR/root.log" <<'PYEOF'
import sys, re, hashlib
log = open(sys.argv[1]).read()
m = re.search(r"👱 Assistant:(.*)", log, re.DOTALL)
text = (m.group(1) if m else log[-2000:]).strip()
print("token text length:", len(text))
print("md5:", hashlib.md5(text.encode()).hexdigest())
with open(sys.argv[1] + ".tokens.md5", "w") as f:
    f.write(hashlib.md5(text.encode()).hexdigest() + "\n")
PYEOF
echo "DONE"