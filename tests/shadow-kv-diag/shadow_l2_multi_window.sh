#!/usr/bin/env bash
# Shadow L2 multi-window (continuous tool-wait) smoke, CPU only.
# Drives 5 back-to-back tool_window_begin / shadow_debt / tool_window_end
# cycles over a single chat session to verify that L2 catch-up runs in every
# window, debt returns to 0, and there is no deadlock / token divergence.
#
# usage: shadow_l2_multi_window.sh <tag> <mode> [nbatches] [steps] [n_windows]
#   mode in: off | l1 | l2 | l2_l1off
set -u
TAG="${1:?tag}"; MODE="${2:?mode off|l1|l2|l2_l1off}"; NB="${3:-32}"; STEPS="${4:-48}"; NWIN="${5:-5}"

ROOT="$HOME/B01/EdgeVisor"
BIN="$ROOT/EdgeVisor/dllama"
CLIENT="$ROOT/EdgeVisor/examples/plan-uds-client.py"
MODEL="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOK="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
LOGDIR="$ROOT/runtime_logs/shadow_kv_diag/$TAG"
DUMP="$LOGDIR/dump"
rm -rf "$LOGDIR"; mkdir -p "$DUMP"
SOCK="/tmp/shadow_l2_multi_${TAG}.sock"
rm -f "$SOCK"

case "$MODE" in
  off)         MODE_ENV=() ;;
  l1)          MODE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L1_DISABLE=1) ;;
  l2)          MODE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L2=1) ;;
  l2_l1off)    MODE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L2=1 DLLAMA_SHADOW_L1_DISABLE=1) ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

P1=$(( 24000 + RANDOM % 2000 ))

# --- Phase 1: launch worker, wait until it fully initialises -----------------
env "${MODE_ENV[@]}" \
  "$BIN" worker --port "$P1" --nthreads 1 >"$LOGDIR/worker1.log" 2>&1 &
W1=$!
echo "worker pid=$W1 port=$P1"

# Wait for worker init-complete signal (last log line under non-L1-disabled mode).
echo "[$(date +%T)] waiting for worker init..."
for i in $(seq 1 600); do
  if grep -q "Runtime-role loading\|stage.*weights\|Connected to root\|first sync done\|first fwd\|network loop\|listening for control\|awaiting root" "$LOGDIR/worker1.log" 2>/dev/null; then
    sleep 5  # settle
    break
  fi
  sleep 1
done
echo "[$(date +%T)] worker init done (or timeout). lines=$(wc -l <"$LOGDIR/worker1.log")"

# --- Phase 2: launch root chat, hold stdin open ----------------------------
# The stdin keeper runs as a tracked background job writing to a named pipe;
# a bare `( ... ) | chat &` leaves the keeper's `sleep 1800` as an untracked
# child, so the shutdown `wait` below would block for 30 minutes.
mkfifo "$LOGDIR/chat_stdin.pipe" 2>/dev/null || true
( printf '\n'; echo "Write a comma-separated list of the numbers from 1 to 20."; sleep 1800 ) >"$LOGDIR/chat_stdin.pipe" &
STDIN_KEEPER=$!
env DLLAMA_DUMP_KV_DIR="$DUMP" DLLAMA_NBATCHES="$NB" "${MODE_ENV[@]}" \
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
  --benchmark <"$LOGDIR/chat_stdin.pipe" >"$LOGDIR/root.log" 2>&1 &
R1=$!
echo "root pid=$R1 stdin_keeper=$STDIN_KEEPER"

# Wait for turn 1 to finish (root prints second "👱 User" while waiting for next input).
echo "[$(date +%T)] waiting for root turn 1 to complete..."
for i in $(seq 1 600); do
  N=$(grep -c "👱 User" "$LOGDIR/root.log" 2>/dev/null || true)
  [ -z "$N" ] && N=0
  [ "$N" -ge 2 ] && break
  sleep 1
done
echo "[$(date +%T)] turn-1 ready (👱 User count=$N)"

# --- Phase 3: multi-window sweep --------------------------------------------
echo "=== multi-window sweep ($NWIN cycles, mode=$MODE) ===" | tee "$LOGDIR/multi_window.log"
PASS=1
for w in $(seq 1 "$NWIN"); do
  echo "--- window $w ($(date +%T)) ---" | tee -a "$LOGDIR/multi_window.log"
  DB=$(python3 "$CLIENT" "$SOCK" shadow_debt 2>/dev/null || echo {ok:false})
  echo "  pre:  $DB" | tee -a "$LOGDIR/multi_window.log"
  python3 "$CLIENT" "$SOCK" tool_window_begin >"$LOGDIR/win${w}_begin.json" 2>&1 || true
  sleep 6   # simulate tool call duration
  DA=$(python3 "$CLIENT" "$SOCK" shadow_debt 2>/dev/null || echo {ok:false})
  echo "  mid:  $DA" | tee -a "$LOGDIR/multi_window.log"
  sleep 2
  python3 "$CLIENT" "$SOCK" tool_window_end >"$LOGDIR/win${w}_end.json" 2>&1 || true
  DF=$(python3 "$CLIENT" "$SOCK" shadow_debt 2>/dev/null || echo {ok:false})
  echo "  post: $DF" | tee -a "$LOGDIR/multi_window.log"
  # For l2_l1off we expect debt to drop to 0; for others debt may be 0 throughout.
  # The client prints json.dump(resp, indent=2), so the field is `"debtEntries": N`
  # (quoted name); the old `debtEntries: ?0` pattern could never match it.
  if [ "$MODE" = "l2_l1off" ]; then
    if echo "$DF" | grep -Eq '"debtEntries": ?0'; then
      echo "  PASS: debt cleared in window $w" | tee -a "$LOGDIR/multi_window.log"
    else
      echo "  WARN: debt not zero after window $w" | tee -a "$LOGDIR/multi_window.log"
      PASS=0
    fi
  fi
done

sleep 2
echo "=== shutting down ===" | tee -a "$LOGDIR/multi_window.log"
kill $R1 $W1 "${STDIN_KEEPER:-}" 2>/dev/null
wait 2>/dev/null
rm -f "$LOGDIR/chat_stdin.pipe"

# Extract generated tokens (everything after first "👱 Assistant:" line onwards).
python3 - "$LOGDIR/root.log" <<PYEOF
import sys, re, hashlib
log = open(sys.argv[1]).read()
m = re.search(r"👱 Assistant:(.*)", log, re.DOTALL)
text = (m.group(1) if m else log[-2000:]).strip()
print("token text length:", len(text))
print("md5:", hashlib.md5(text.encode()).hexdigest())
with open(sys.argv[1] + ".tokens.md5", "w") as f:
    f.write(hashlib.md5(text.encode()).hexdigest() + "\n")
PYEOF

if [ "${PASS:-1}" -ne 1 ]; then
  echo "FAIL: not all windows cleared shadow debt (mode=$MODE)" >&2
  exit 1
fi
echo "DONE"
