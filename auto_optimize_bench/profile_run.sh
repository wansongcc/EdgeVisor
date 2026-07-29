#!/usr/bin/env bash
set -u
ROOT="/home/byh/B01/EdgeVisor"
ENGINE="${ROOT}/EdgeVisor"
MODEL3="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOK="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
export PATH="${ROOT}/tools/vulkan_deps/root/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${ROOT}/tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CPATH="${ROOT}/tools/vulkan_deps/root/usr/include"
export LIBRARY_PATH="${ROOT}/tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu"
cd "${ENGINE}"
./dllama worker --port 19801 --nthreads 1 --gpu-index 1 >/tmp/worker1.log 2>&1 &
P1=$!
./dllama worker --port 19802 --nthreads 1 --gpu-index 2 >/tmp/worker2.log 2>&1 &
P2=$!
sleep 3
./dllama inference --prompt "Hello world, please write a long paragraph explaining distributed inference systems in detail." --steps 32 --model "${MODEL3}" --tokenizer "${TOK}" --buffer-float-type q80 --nthreads 1 --max-seq-len 512 --gpu-index 0 --workers 127.0.0.1:19801 127.0.0.1:19802 --ratios 2:3:3 --benchmark 2>&1 | tail -30
kill ${P1} ${P2} 2>/dev/null || true
wait 2>/dev/null || true
