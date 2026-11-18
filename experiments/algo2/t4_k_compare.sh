#!/usr/bin/env bash
# Same injected sleep as t4_pair.sh. Compare no move, a forced 1-layer move,
# and the natural 6-layer move on the stage-2 to stage-3 boundary.
# Each revised run may issue only one move. Settle 4 and verify 8, so 80
# steps cover the keep decision.
set -u
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT_DIR/EdgeVisor/dllama}"
MODEL="${MODEL:-/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m}"
TOK="${TOK:-/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t}"
OUT="${OUT:-$ROOT_DIR/experiments/algo2/logs/t4_k_compare}"
STEPS="${STEPS:-80}"
CODE="$(git -C "$ROOT_DIR" rev-parse HEAD)"

mkdir -p "$OUT"
{
  echo "code=$CODE"
  echo "steps=$STEPS"
  echo "delay_us=15000 node=2 after_decode_tokens=24 decode_only=1"
  echo "ratios=1@7*1@7*1@7*1@7 split_order=gpu0,gpu1,gpu2,gpu3"
  echo "verify_tokens=8 settle_tokens=4 max_issues=1"
  echo "k1=force from 2 to 3 k=1; k6=unforced, previously selected k=6"
  nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
} >"$OUT/meta.txt"

run_one() {
  local name="$1"
  local mode="$2"
  local arm="$3"
  local DIR="$OUT/$name"
  rm -rf "$DIR"
  mkdir -p "$DIR"
  unset DLLAMA_ALGO2_FORCE_FROM || true
  unset DLLAMA_ALGO2_FORCE_TO || true
  unset DLLAMA_ALGO2_FORCE_K || true
  unset DLLAMA_ALGO2_MAX_ISSUES || true
  if [[ "$arm" == "k1" ]]; then
    export DLLAMA_ALGO2_FORCE_FROM=2
    export DLLAMA_ALGO2_FORCE_TO=3
    export DLLAMA_ALGO2_FORCE_K=1
    export DLLAMA_ALGO2_MAX_ISSUES=1
  elif [[ "$arm" == "k6" ]]; then
    export DLLAMA_ALGO2_MAX_ISSUES=1
  fi
  export DLLAMA_ALGO2_MODE="$mode"
  export DLLAMA_ALGO2_OUT="$DIR"
  export DLLAMA_TPOT_LOG="$DIR/tpot.log"
  export DLLAMA_ALGO2_METRICS_WINDOW=8
  export DLLAMA_ALGO2_DECISION_WINDOW=8
  export DLLAMA_ALGO2_DECISION_STRIDE=8
  export DLLAMA_ALGO2_MIN_DECISION_INTERVAL_MS=0
  export DLLAMA_ALGO2_SETTLE_TOKENS=4
  export DLLAMA_ALGO2_VERIFY_TOKENS=8
  export DLLAMA_ALGO2_VERIFY_MAX_TOKENS=16
  export DLLAMA_ALGO2_COOLDOWN_TOKENS=4
  export DLLAMA_ALGO2_COOLDOWN_MIN_MS=0
  export DLLAMA_IGNORE_EOS=1
  export EDGEVISOR_SHADOW_ALLOCATION=equal
  export EDGEVISOR_SHADOW_BUDGET_CAP=6
  export EDGEVISOR_SHADOW_PROFILE_SECONDS=8
  export EDGEVISOR_SHADOW_PROFILE_WARMUP=2
  export EDGEVISOR_SHADOW_PROFILE_DIR="$DIR/profile"
  export DLLAMA_BUBBLE_SHADOW_KV=1
  export DLLAMA_TEST_COMPUTE_DELAY_US=15000
  export DLLAMA_TEST_COMPUTE_DELAY_NODE=2
  export DLLAMA_TEST_COMPUTE_DELAY_DECODE_ONLY=1
  export DLLAMA_TEST_COMPUTE_DELAY_AFTER_DECODE_TOKENS=24
  unset DLLAMA_REDUNDANT_BOUNDARY_DEPTHS || true
  local w1="" w2="" w3=""
  cleanup() {
    [[ -n "$w1" ]] && kill "$w1" 2>/dev/null || true
    [[ -n "$w2" ]] && kill "$w2" 2>/dev/null || true
    [[ -n "$w3" ]] && kill "$w3" 2>/dev/null || true
    wait "$w1" "$w2" "$w3" 2>/dev/null || true
  }
  trap cleanup RETURN
  "$BIN" worker --backend cuda --port 29141 --nthreads 1 --gpu-index 1 >"$DIR/w1.log" 2>&1 &
  w1=$!
  "$BIN" worker --backend cuda --port 29142 --nthreads 1 --gpu-index 2 >"$DIR/w2.log" 2>&1 &
  w2=$!
  "$BIN" worker --backend cuda --port 29143 --nthreads 1 --gpu-index 3 >"$DIR/w3.log" 2>&1 &
  w3=$!
  sleep 2
  set +e
  timeout 240 "$BIN" inference \
    --model "$MODEL" --tokenizer "$TOK" \
    --prompt "Hi" --steps "$STEPS" \
    --ratios '1@7*1@7*1@7*1@7' \
    --workers 127.0.0.1:29141 127.0.0.1:29142 127.0.0.1:29143 \
    --gpu-index 0 --backend cuda --nthreads 1 \
    --max-seq-len 256 --temperature 0 --seed 1 \
    --buffer-float-type q80 --enable-pp-migration --enable-dynamic-tpot \
    >"$DIR/root.log" 2>&1
  local rc=$?
  set -e
  echo "$name mode=$mode arm=$arm exit=$rc" | tee "$DIR/exit.txt"
}

run_one r1_static static static
run_one r1_k6 revised_variable_step k6
run_one r1_k1 revised_variable_step k1
run_one r2_k1 revised_variable_step k1
run_one r2_static static static
run_one r2_k6 revised_variable_step k6
run_one r3_k6 revised_variable_step k6
run_one r3_k1 revised_variable_step k1
run_one r3_static static static
echo "k compare done"
