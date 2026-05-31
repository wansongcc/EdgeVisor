#!/usr/bin/env bash
# Shared runtime configuration for EdgeVisor scripts.
# Override any value from the environment before invoking a script.

if [[ -z "${EDGEVISOR_PROJECT_ROOT:-}" ]]; then
  EDGEVISOR_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

export EDGEVISOR_PROJECT_ROOT
export EDGEVISOR_ENGINE_DIR="${EDGEVISOR_ENGINE_DIR:-${EDGEVISOR_PROJECT_ROOT}/EdgeVisor}"
export EDGEVISOR_MODEL3="${EDGEVISOR_MODEL3:-/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m}"
export EDGEVISOR_TOKENIZER="${EDGEVISOR_TOKENIZER:-/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t}"
export EDGEVISOR_LOG_ROOT="${EDGEVISOR_LOG_ROOT:-${EDGEVISOR_PROJECT_ROOT}/runtime_logs}"

export CPATH="${EDGEVISOR_PROJECT_ROOT}/tools/vulkan_deps/root/usr/include${CPATH:+:${CPATH}}"
export PATH="${EDGEVISOR_PROJECT_ROOT}/tools/vulkan_deps/root/usr/bin:${PATH}"
export LIBRARY_PATH="${EDGEVISOR_PROJECT_ROOT}/tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${EDGEVISOR_PROJECT_ROOT}/tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
