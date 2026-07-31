#!/usr/bin/env bash
# Shadow L2 interrupt test (v2): large debt, interrupt catch-up mid-flight with
# a new chat turn; verify catch-up stops (partial repayment), inference resumes
# without extra stall, and a later window repays the rest. CPU only.
# usage: shadow_l2_interrupt.sh <tag> [steps]
set -u
TAG="${1:?tag}"; STEPS="${2:-10}"

ROOT="$HOME/B01/EdgeVisor"
BIN="$ROOT/EdgeVisor/dllama"
CLIENT="$ROOT/EdgeVisor/examples/plan-uds-client.py"
MODEL="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOK="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
LOGDIR="$ROOT/runtime_logs/shadow_kv_diag/$TAG"
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
SOCK="/tmp/shadow_l2int_${TAG}.sock"
FIFO="/tmp/shadow_l2int_${TAG}.fifo"
rm -f "$SOCK" "$FIFO"; mkfifo "$FIFO"

BASE=$(( 20000 + RANDOM % 2000 ))
P1=$BASE

BUBBLE_ENV=(DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_SHADOW_L2=1 DLLAMA_SHADOW_L1_DISABLE=1)

env "${BUBBLE_ENV[@]}" \
  "$BIN" worker --port "$P1" --nthreads 1 >"$LOGDIR/worker1.log" 2>&1 &
W1=$!
sleep 3

( exec 3>"$FIFO"; printf '\n' >&3; python3 -c "print('Please summarize the following text. ' + 'The quick brown fox jumps over the lazy dog near the river bank. ' * 25)" >&3; sleep 1800 ) &
FHOLD=$!

env DLLAMA_NBATCHES=32 "${BUBBLE_ENV[@]}" \
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
  --benchmark <"$FIFO" >"$LOGDIR/root.log" 2>&1 &
R1=$!

# wait for turn 1 to finish
for i in $(seq 1 900); do
  N=$(grep -c "👱 User" "$LOGDIR/root.log" 2>/dev/null || true)
  [ -z "$N" ] && N=0
  [ "$N" -ge 2 ] && break
  sleep 1
done

debt_field() { python3 "$CLIENT" "$SOCK" shadow_debt 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['root']['debtEntries'], d['root']['catchupEntries'])"; }

echo "=== debt before window ===" | tee -a "$LOGDIR/summary.txt"
debt_field | tee -a "$LOGDIR/summary.txt"

python3 "$CLIENT" "$SOCK" tool_window_begin >/dev/null
T0=$(date +%s)
sleep 0.15
echo "Write a comma-separated list of the numbers from 21 to 40." >"$FIFO"

# poll until turn-2 forwards start producing new debt (catch-up interrupted)
RESUME=-1
for i in $(seq 1 60); do
  read DEBT CATCH < <(debt_field)
  if [ "$DEBT" -gt 0 ]; then RESUME=$(( $(date +%s) - T0 )); break; fi
  sleep 0.5
done
echo "catch-up repaid before interrupt: catchupEntries=$CATCH (of ~$STEPS+1)" | tee -a "$LOGDIR/summary.txt"
echo "resume latency (interrupt -> first new debt): ${RESUME}s (one prefill forward is ~8s on this setup)" | tee -a "$LOGDIR/summary.txt"

# wait for turn 2 to finish
for i in $(seq 1 900); do
  N=$(grep -c "👱 User" "$LOGDIR/root.log" 2>/dev/null || true)
  [ -z "$N" ] && N=0
  [ "$N" -ge 3 ] && break
  sleep 1
done
echo "turn 2 finished (user-prompt count=$N)" | tee -a "$LOGDIR/summary.txt"

echo "=== debt after interrupted turn (new debt from turn 2) ===" | tee -a "$LOGDIR/summary.txt"
debt_field | tee -a "$LOGDIR/summary.txt"

python3 "$CLIENT" "$SOCK" tool_window_begin >/dev/null
sleep 15
echo "=== after second window (expect debt 0) ===" | tee -a "$LOGDIR/summary.txt"
debt_field | tee -a "$LOGDIR/summary.txt"
python3 "$CLIENT" "$SOCK" tool_window_end >/dev/null

echo "=== turn 2 output tail ===" >> "$LOGDIR/summary.txt"
tail -6 "$LOGDIR/root.log" >> "$LOGDIR/summary.txt"

kill $R1 $W1 $FHOLD 2>/dev/null
wait 2>/dev/null
cat "$LOGDIR/summary.txt"
echo "LOGDIR=$LOGDIR"
