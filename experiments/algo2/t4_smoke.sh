#!/usr/bin/env bash
# Host-side T4 smoke for algorithm 2. This is not the 512-token matrix.
# Docker Hub was unreachable from this machine, so the run uses the host GPUs.
set -u
BIN="${BIN:-/home/byh/B01/EdgeVisor-wansongcc/EdgeVisor/edgevisor}"
MODEL="${MODEL:-/home/byh/B01/models/llama3.2_3b_instruct_q40/edgevisor_model_llama3.2-3b-instruct_q40.m}"
TOK="${TOK:-/home/byh/B01/models/llama3.1_instruct_q40/edgevisor_tokenizer_llama_3_1.t}"
ROOT="${ROOT:-/tmp/ev_algo2_smoke}"

run_one() {
  local name="$1"
  local mode="$2"
  local steps="$3"
  local delay="$4"
  local DIR="$ROOT/$name"
  rm -rf "$DIR"
  mkdir -p "$DIR"
  export EDGEVISOR_ALGO2_MODE="$mode"
  export EDGEVISOR_ALGO2_OUT="$DIR"
  export EDGEVISOR_TPOT_LOG="$DIR/tpot.log"
  export EDGEVISOR_ALGO2_METRICS_WINDOW=8
  export EDGEVISOR_ALGO2_DECISION_WINDOW=8
  export EDGEVISOR_ALGO2_DECISION_STRIDE=8
  export EDGEVISOR_ALGO2_MIN_DECISION_INTERVAL_MS=0
  export EDGEVISOR_ALGO2_SETTLE_TOKENS=4
  export EDGEVISOR_ALGO2_VERIFY_TOKENS=8
  export EDGEVISOR_ALGO2_VERIFY_MAX_TOKENS=16
  export EDGEVISOR_ALGO2_COOLDOWN_TOKENS=4
  export EDGEVISOR_ALGO2_COOLDOWN_MIN_MS=0
  export EDGEVISOR_IGNORE_EOS=1
  export EDGEVISOR_SHADOW_ALLOCATION=equal
  export EDGEVISOR_SHADOW_BUDGET_CAP=6
  export EDGEVISOR_SHADOW_PROFILE_SECONDS=8
  export EDGEVISOR_SHADOW_PROFILE_WARMUP=2
  export EDGEVISOR_SHADOW_PROFILE_DIR="$DIR/profile"
  export EDGEVISOR_BUBBLE_SHADOW_KV=1
  unset EDGEVISOR_REDUNDANT_BOUNDARY_DEPTHS || true
  if [[ "$delay" == "1" ]]; then
    export EDGEVISOR_TEST_COMPUTE_DELAY_US=15000
    export EDGEVISOR_TEST_COMPUTE_DELAY_NODE=2
    export EDGEVISOR_TEST_COMPUTE_DELAY_DECODE_ONLY=1
    export EDGEVISOR_TEST_COMPUTE_DELAY_AFTER_DECODE_TOKENS=24
  else
    unset EDGEVISOR_TEST_COMPUTE_DELAY_US || true
    unset EDGEVISOR_TEST_COMPUTE_DELAY_NODE || true
  fi
  local w1="" w2="" w3=""
  cleanup() {
    [[ -n "$w1" ]] && kill "$w1" 2>/dev/null || true
    [[ -n "$w2" ]] && kill "$w2" 2>/dev/null || true
    [[ -n "$w3" ]] && kill "$w3" 2>/dev/null || true
    wait "$w1" "$w2" "$w3" 2>/dev/null || true
  }
  trap cleanup RETURN
  "$BIN" worker --backend cuda --port 29121 --nthreads 1 --gpu-index 1 >"$DIR/w1.log" 2>&1 &
  w1=$!
  "$BIN" worker --backend cuda --port 29122 --nthreads 1 --gpu-index 2 >"$DIR/w2.log" 2>&1 &
  w2=$!
  "$BIN" worker --backend cuda --port 29123 --nthreads 1 --gpu-index 3 >"$DIR/w3.log" 2>&1 &
  w3=$!
  sleep 2
  set +e
  "$BIN" inference \
    --model "$MODEL" --tokenizer "$TOK" \
    --prompt "Hi" --steps "$steps" \
    --ratios '1@7*1@7*1@7*1@7' \
    --workers 127.0.0.1:29121 127.0.0.1:29122 127.0.0.1:29123 \
    --gpu-index 0 --backend cuda --nthreads 1 \
    --max-seq-len 256 --temperature 0 --seed 1 \
    --buffer-float-type q80 --enable-pp-migration --enable-dynamic-tpot \
    >"$DIR/root.log" 2>&1
  local rc=$?
  set -e
  echo "===== $name exit=$rc ====="
  grep -E 'tokens/s|Critical error|topology:|served from reservation|stallMs=' "$DIR/root.log" || true
}

run_one static_long static 48 0
run_one revised_delay revised_variable_step 120 1
