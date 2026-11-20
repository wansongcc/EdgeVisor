#!/usr/bin/env bash
# Drop the stage that owns layers [8, 12) and bring it back.
# Neighbors both cover that stage, and the root stage is slowed so the
# faster downstream neighbor must take the overlap. A fallback that always
# leaves the overlap with the previous stage would log front=[8,11).
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-$ROOT_DIR/EdgeVisor/dllama}"
MODEL="${MODEL:-/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m}"
TOK="${TOK:-/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t}"
DIR="${DIR:-$ROOT_DIR/experiments/algo2/logs/t4_rejoin}"
mkdir -p "$DIR"
rm -f "$DIR"/*.log "$DIR"/*.txt

export DLLAMA_IGNORE_EOS=1
export DLLAMA_BUBBLE_SHADOW_KV=1
export DLLAMA_BUBBLE_SHADOW_KV_ASYNC=0
export DLLAMA_BUBBLE_SHADOW_KV_DURING_FORWARD=1
export DLLAMA_REDUNDANT_BOUNDARY_DEPTHS='3,1;1,3;1,1'
export DLLAMA_TCP_KEEPIDLE_S=2
export DLLAMA_TCP_KEEPINTVL_S=1
export DLLAMA_TCP_KEEPCNT=3
export DLLAMA_TCP_USER_TIMEOUT_MS=4000
# One root layer sleeps 80ms, so stage 0 is several times slower than stage 2.
export DLLAMA_TEST_COMPUTE_DELAY_US=80000
export DLLAMA_TEST_COMPUTE_DELAY_NODE=0
export DLLAMA_TEST_COMPUTE_DELAY_LAYER=2
export DLLAMA_TEST_COMPUTE_DELAY_DECODE_ONLY=1
export DLLAMA_TEST_COMPUTE_DELAY_AFTER_DECODE_TOKENS=4

P1=29241
P2=29242
P3=29243
w1="" w2="" w3="" root=""
cleanup() {
  for pid in "$w1" "$w2" "$w3" "$root"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

stdbuf -oL -eL "$BIN" worker --backend cuda --port "$P1" --nthreads 1 --gpu-index 1 >"$DIR/w1.log" 2>&1 &
w1=$!
stdbuf -oL -eL "$BIN" worker --backend cuda --port "$P2" --nthreads 1 --gpu-index 2 >"$DIR/w2.log" 2>&1 &
w2=$!
stdbuf -oL -eL "$BIN" worker --backend cuda --port "$P3" --nthreads 1 --gpu-index 3 >"$DIR/w3.log" 2>&1 &
w3=$!
sleep 2

stdbuf -oL -eL "$BIN" inference \
  --model "$MODEL" --tokenizer "$TOK" \
  --prompt "Hi" --steps 500 \
  --ratios '1@8*1@4*1@8*1@8' \
  --runtime-redundant-boundary-layers 3 \
  --workers "127.0.0.1:$P1" "127.0.0.1:$P2" "127.0.0.1:$P3" \
  --gpu-index 0 --backend cuda --nthreads 1 \
  --max-seq-len 512 --temperature 0 --seed 1 \
  --buffer-float-type q80 --enable-pp-migration \
  >"$DIR/root.log" 2>&1 &
root=$!

for _ in $(seq 1 180); do
  if grep -q "Pos=24" "$DIR/w2.log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$root" 2>/dev/null; then
    echo "root exited before decode pos 24" | tee "$DIR/fault.txt"
    exit 1
  fi
  sleep 1
done
if ! grep -q "Pos=24" "$DIR/w2.log"; then
  echo "timed out waiting for decode pos 24" | tee "$DIR/fault.txt"
  exit 1
fi
echo "KILL_W1_AT=$(date +%s) pid=$w1" | tee "$DIR/fault.txt"
kill -9 "$w1" || true
w1=""
sleep 2
stdbuf -oL -eL "$BIN" worker --backend cuda --port "$P1" --nthreads 1 --gpu-index 1 >"$DIR/w1-rejoin.log" 2>&1 &
w1=$!
echo "RESTART_W1_AT=$(date +%s) pid=$w1" | tee -a "$DIR/fault.txt"

wait "$root" || true
echo "ROOT_DONE=$(date +%s)" | tee -a "$DIR/fault.txt"

{
  echo "===== split / rejoin ====="
  grep -E 'failover\] split|failover\] stop|catch-up |loading weights|weights loaded|rejoined stage|rejoin' "$DIR/root.log" || true
  echo "===== root tail ====="
  tail -n 30 "$DIR/root.log" || true
} >"$DIR/summary.txt"
cat "$DIR/summary.txt"
