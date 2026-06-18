# EdgeVisor GPU Version

EdgeVisor GPU Version 是在 Distributed Llama 基础上扩展的 CPU/GPU 分布式 LLM 推理实验工程。当前版本保留原有单机推理、非均匀静态张量并行、UDS 控制的动态迁移、GPU Vulkan 后端以及 PP/TP 组合实验能力，同时把项目脚本、历史产物、测试记录和维护补丁按工程化目录重新整理。

## 目录结构

```text
EdgeVisor_GPU_Version/
├── EdgeVisor/                 # 核心 C++/Vulkan 推理引擎，保留原 Distributed Llama 构建方式
│   ├── src/                   # 推理、网络同步、动态迁移、tokenizer、Vulkan/CPU backend
│   ├── examples/              # UDS client 和原始示例
│   ├── docs/                  # 引擎级使用说明
│   └── Makefile               # 核心二进制构建入口
├── config/env.sh              # 统一模型路径、tokenizer、日志目录、Vulkan 依赖环境
├── scripts/                   # 维护后的脚本入口
│   ├── semantic/              # CPU/GPU 语义回归与分布式推理脚本
│   ├── gpu/                   # GPU PP、补丁回归和调试脚本
│   └── build.sh               # 标准构建脚本
├── tests/semantic/            # 六项 benchmark 回归和测试记录生成器
├── docs/test_records/         # 本次/后续验收测试记录输出目录
├── maintenance/               # 历史补丁、临时修复脚本、调试配置归档
├── artifacts/                 # 历史日志、实验结果、docker 运行记录归档
├── tools/                     # 本地 Vulkan 依赖等辅助工具
├── build.sh                   # 兼容的一键构建入口
└── run_*.sh                   # 顶层兼容 wrapper，转发到 scripts/ 下的维护脚本
```

顶层 `run_semantic_*.sh` 和 `run_gpu_*.sh` 文件仍然保留，用作兼容入口；真实维护位置在 `scripts/semantic/` 和 `scripts/gpu/`。

## 环境约束

默认远端模型路径为：

```bash
/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m
/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t
```

GPU 测试只使用 GPU0、GPU1、GPU2。不要在命令中使用 `--gpu-index 3`，也不要终止 GPU3 上的进程。

可通过环境变量覆盖默认配置：

```bash
export EDGEVISOR_MODEL3=/path/to/dllama_model.m
export EDGEVISOR_TOKENIZER=/path/to/dllama_tokenizer.t
export EDGEVISOR_LOG_ROOT=/path/to/runtime_logs
```

## 构建

推荐使用项目根目录入口：

```bash
cd /path/to/EdgeVisor_GPU_Version
bash build.sh
```

等价于：

```bash
source config/env.sh
cd "$EDGEVISOR_ENGINE_DIR"
make DLLAMA_VULKAN=1 dllama
```

如果只需要 CPU 构建，可执行：

```bash
DLLAMA_VULKAN=0 bash build.sh
```

## 六项验收回归

完整回归包括：

1. CPU 单机测试
2. GPU 单机测试
3. CPU 非均匀静态测试
4. GPU 非均匀静态测试
5. CPU 非均匀动态迁移测试
6. GPU 非均匀动态迁移测试

执行：

```bash
cd /path/to/EdgeVisor_GPU_Version
bash run_six_benchmark_tests.sh
```

默认输出到：

```bash
runtime_logs/benchmark_docs_YYYYmmdd_HHMMSS/
```

生成测试记录：

```bash
python3 tests/semantic/generate_test_records.py \
  --logs runtime_logs/benchmark_docs_YYYYmmdd_HHMMSS \
  --out docs/test_records \
  --project-root "$PWD"
```

每个测试记录包含实际 init 输入、动态迁移 UDS 输入、UDS 返回、token 原始输出、`--benchmark` 的 `Prediction tokens/s` 和 Stage/Node profile。

## 常用兼容脚本

```bash
bash run_semantic_cpu_tp_static.sh
bash run_semantic_gpu_tp_static.sh
bash run_semantic_cpu_dynamic_heads.sh
bash run_semantic_gpu_dynamic_heads.sh
bash run_gpu_pp_migration.sh
```

这些命令会转发到 `scripts/` 下的维护脚本。日志默认仍由各脚本写入自身配置的目录；标准化六项验收建议优先使用 `run_six_benchmark_tests.sh`。

## UDS 动态迁移

UDS 客户端位于：

```bash
EdgeVisor/examples/plan-uds-client.py
```

Heads/FFN 迁移示例：

```bash
python3 EdgeVisor/examples/plan-uds-client.py SOCKET set_plan \
  --seq 501 \
  --mode next_barrier \
  --stage 0 \
  --from 1 \
  --to 2 \
  --kind 1 \
  --heads 1 \
  --ffn 0
```

`--kind 1` 表示 heads，`--kind 2` 表示 FFN，`--kind 3` 表示二者同时迁移。

PP layer 迁移示例：

```bash
python3 EdgeVisor/examples/plan-uds-client.py SOCKET set_pp_migration \
  --seq 1 \
  --mode next_barrier \
  --from 0 \
  --to 1 \
  --layer-count 1 \
  --trigger-pos 0
```

## Agentic Ablation 实验框架

`EdgeVisor_Agentlized_Ablation` 分支新增统一 ablation 配置层和 Agentic
实验入口。默认配置等价于完整 EdgeVisor：

- `shadow_kv_mode=enabled`
- `pointer_swizzling_mode=enabled`
- `jit_mode=enabled`
- `vg_mode=enabled`
- `fallback_policy=disabled_unless_necessary`

配置优先级为 `CLI > env > JSON > default`。支持的 JSON/CLI/env 字段如下：

| 字段 | CLI | Env | 模式 |
|---|---|---|---|
| `shadow_kv_mode` | `--shadow-kv-mode` | `EDGEVISOR_SHADOW_KV_MODE` | `enabled`, `disabled_transfer`, `disabled_recompute` |
| `pointer_swizzling_mode` | `--pointer-swizzling-mode` | `EDGEVISOR_POINTER_SWIZZLING_MODE` | `enabled`, `operator_rebuild`, `weight_rematerialize` |
| `jit_mode` | `--jit-mode` | `EDGEVISOR_JIT_MODE` | `enabled`, `static`, `greedy`, `oracle` |
| `vg_mode` | `--vg-mode` | `EDGEVISOR_VG_MODE` | `enabled`, `flat`, `random`, `pure_pp`, `no_elastic_vg` |
| `fallback_policy` | `--fallback-policy` | `EDGEVISOR_FALLBACK_POLICY` | 自定义字符串 |
| `ablation_log_path` | `--ablation-log-path` | `EDGEVISOR_ABLATION_LOG_PATH` | JSONL 路径 |
| `experiment_id` | `--experiment-id` | `EDGEVISOR_EXPERIMENT_ID` | 实验 ID |

直接运行示例：

```bash
./EdgeVisor/dllama inference \
  --enable-plan-barrier \
  --edgevisor-ablation-config ablation_config.json \
  --ablation-log-path logs/ablation.jsonl

./EdgeVisor/dllama inference \
  --enable-plan-barrier \
  --shadow-kv-mode disabled_transfer \
  --pointer-swizzling-mode enabled \
  --jit-mode enabled \
  --vg-mode enabled \
  --ablation-log-path logs/shadow_transfer.jsonl
```

统一 JSONL 日志会记录 `experiment_id`、`ablation_variant`、`event_id`、
`trigger_pos`、`trigger_layer`、`affected_stage`、`from_node`、`to_node`、
`selected_policy`、`t_decision_ms`、`t_state_prepare_ms`、`t_bind_ms`、
`t_command_ms`、`t_apply_ms`、`t_recover_ms`、`stall_time_ms`、
`state_transfer_bytes`、`recompute_tokens_or_layers`、`binding_update_count`、
`vg_mapping_before`、`vg_mapping_after`、`fallback_reason` 和 `apply_success`。

服务器端完整 Agentic 消融套件入口：

```bash
cd /home/byh/B01/EdgeVisor
bash scripts/run_agentic_ablation_suite.sh
```

脚本会显式约束 `CUDA_VISIBLE_DEVICES=0,1,2`，拒绝使用 GPU3。完整模式默认运行
Full EdgeVisor、Shadow KV transfer/recompute、Pointer operator rebuild/weight
rematerialize、JIT static/greedy、VG flat/random，并覆盖 compute、network、
mixed 三类波动和 3 次重复。最小 smoke test 可执行：

```bash
SMOKE=1 bash scripts/run_agentic_ablation_suite.sh
```

结果汇总：

```bash
python3 scripts/summarize_ablation_results.py /home/byh/B01/agentic_ablation_results/RUN_DIR
```

汇总工具会输出 `ablation_summary.csv`、`ablation_runs.csv`、总表 Markdown，以及
Shadow KV、Pointer Swizzling、JIT、VG 四组分表。所有数值均从真实 `trace.json`
解析，不硬编码实验结果。

`edgevisor_ablation` backend 会在 Agent episode 内部按模式改写实际运行策略：

- Shadow KV disabled 模式会把 root 启动参数切到
  `--kv-redundancy 0 --enable-kv-redundancy-during-migration 0`，并在 JSONL
  中记录 transfer/recompute fallback 或 apply reject reason。
- JIT `static` / `greedy` 会在发送 UDS `set_plan` 前改写 move list，而不是关闭
  PlanCommand。
- VG `flat` / `random` / `pure_pp` / `no_elastic_vg` 会生成合法 worker/ratio
  映射，并把 physical/logical group 与 VG mapping 写入同一条 ablation 日志。
- Pointer `operator_rebuild` / `weight_rematerialize` 保持安全 pointer refresh，
  但将替代式 rebinding / rematerialization 成本字段写入统一 JSONL。


## Bubble Shadow KV v1

Bubble Shadow KV v1 moves runtime redundant Shadow KV computation out of the primary
forward path. Enable it on the root process with:

```bash
DLLAMA_BUBBLE_SHADOW_KV=1 ./dllama inference ... \
  --runtime-redundant-boundary-layers 2 \
  --runtime-active-seg-enabled 1 \
  --runtime-redundant-seg-enabled \
  --kv-redundancy 2
```

The root bootstrap packet propagates the bubble mode to workers, so workers do not need
a separate environment variable when they are launched by EdgeVisor. Optional per-token
bubble tracing is available with `DLLAMA_BUBBLE_SHADOW_KV_LOG=1`; leave it off for
performance measurements.

Scope of this first version:

- Supports runtime redundant segment execution after each primary forward.
- Covers both intra-stage head redundancy (`--kv-redundancy 2`) and inter-stage layer
  redundancy (`--runtime-redundant-boundary-layers 2`).
- Does not implement JIT-aware dynamic Shadow KV coverage selection yet.
- Benchmark profile packets include `bubble` time separately; distributed TPOT should be
  read from the Stage/Node critical path (`per-tok total` max), not only from root
  `Prediction ms/tok`.

Validated smoke topology:

```text
ratios: 1:1:1*1:1:1*1:1
logical node GPUs: 0,1,2,0,1,2,0,1
head redundancy: 2 heads
PP boundary redundancy: 2 layers
```

Agentic runs can use the same mode through:

```bash
/home/byh/B01/agent_langgraph_venv/bin/python -m agent_bench.run_loop_episode \
  --backend edgevisor_ablation \
  --edge-virtual-pp-tp-3stage \
  --runtime-redundant-boundary-layers 2 \
  --edge-benchmark \
  --bubble-shadow-kv
```

## LinguaLinked 复现路线

LinguaLinked 复现版本独立维护在：

```text
/home/byh/B01/EdgeVisor-LinguaLinked
git@github.com:BianYanhui/EdgeVisor-LinguaLinked.git
```

该版本基于 EdgeVisor 的 PP 能力实现：

- 纯 PP：每个 stage 只有一个设备，不使用 TP。
- 相邻 stage 冗余加载 overlap layers 的权重。
- 冗余 layers 在正常 request 中保持休眠，不进行冗余 KV-cache 计算。
- 每个 request 结束后，依据上一轮各 stage 耗时，在 overlap 范围内调整下一轮 PP layer 边界。

Agentic 路线沿用本项目 `agent_bench` 的 LangGraph backend 抽象，可直接把 LLM
Generation backend 切换为 `edgevisor_lingualinked`：

```bash
cd /home/byh/B01/EdgeVisor
/home/byh/B01/agent_langgraph_venv/bin/python -m agent_bench.run_loop_episode \
  --backend edgevisor_lingualinked \
  --cuda-visible 0,1,2 \
  --lingualinked-gpus 0,1,2 \
  --lingualinked-overlap-layers 2 \
  --edge-steps 96 \
  --ctx 2048
```

## 已保留的功能范围

- Llama3 chat template 使用 `<|eot_id|>` 作为消息结束符。
- CPU/GPU 单机语义推理。
- CPU/GPU 非均匀 TP 静态推理，支持 `--ratios "2:3:3"`。
- CPU/GPU UDS 控制的 heads 动态迁移，支持 plan barrier、stage full weights 和 KV redundancy。
- GPU Vulkan q80/q40 matmul 支持在线 repartition 后输入宽度变化。
- PP stage 边界发送前合并 residual，并保证非首个 PP stage 的 Vulkan preSync position 正确上传。

## 历史产物和维护文件

历史日志和实验结果不再混放在项目顶层：

```text
artifacts/logs/
artifacts/experiments/
artifacts/docker/
```

历史补丁、一次性修复脚本和调试配置归档在：

```text
maintenance/patches/
maintenance/debug/
```

这些文件用于追溯，不作为日常运行入口。
