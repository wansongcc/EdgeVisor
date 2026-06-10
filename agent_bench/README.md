# Agent Bench

This directory contains a minimal LangGraph-based agent episode runner for
comparing LLM generation backends while keeping tool execution and episode
control identical.

The runtime graph is:

```text
LLM generation -> tool call -> LLM generation
```

Backends currently implemented:

- `prima`: runs `/home/byh/B01/prima_cpp_work/prima.cpp/llama-cli`
- `edgevisor`: runs EdgeVisor `dllama` with root/worker processes
- `mock`: no model, useful for validating LangGraph/runtime behavior

EdgeVisor dynamic support:

- The EdgeVisor backend sets `DLLAMA_PLAN_CTRL_SOCKET`.
- It starts root inference with `--enable-plan-barrier`.
- During the configured generation it sends `set_plan` over UDS.
- The trace records UDS ping, set_plan response, status polling, and whether
  EdgeVisor logs contain plan emit/apply markers.

Example:

```bash
/home/byh/B01/agent_langgraph_venv/bin/python -m agent_bench.run_episode --backend mock

/home/byh/B01/agent_langgraph_venv/bin/python -m agent_bench.run_episode \
  --backend prima \
  --cuda-visible 0,1,2

/home/byh/B01/agent_langgraph_venv/bin/python -m agent_bench.run_episode \
  --backend edgevisor \
  --cuda-visible 0,1,2 \
  --edge-worker-gpus 1 \
  --edge-ratios 1:1
```

All outputs are written under `/home/byh/B01/agent_bench_results` by default.
