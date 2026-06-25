# EdgeVisor GPU Version / EdgeVisor GPU 版本

## English

EdgeVisor GPU Version is an experimental distributed LLM inference project for heterogeneous edge devices. It extends the original distributed inference workflow with GPU/Vulkan execution, uneven static tensor parallelism, pipeline parallelism, hybrid PP/TP layouts, UDS-controlled runtime migration, Shadow KV support, last-stage local sampling, and memory-oriented optimizations such as Q80-resident root token embeddings.

### Acknowledgement

EdgeVisor is developed based on [distributed-llama](https://github.com/b4rtaz/distributed-llama). We sincerely thank the distributed-llama authors and contributors for their open-source work, including the model format, tokenizer/runtime structure, distributed root-worker execution model, synchronization design, and the original CPU/Vulkan inference foundation that made this project possible.

### Repository Layout

```text
EdgeVisor_GPU_Version/
├── EdgeVisor/                 # Core C++/Vulkan inference engine
│   ├── src/                   # Inference, networking, migration, tokenizer, CPU/Vulkan backends
│   ├── examples/              # UDS client and original examples
│   ├── docs/                  # Engine-level documentation
│   └── Makefile               # Core binary build entry
├── config/env.sh              # Shared model, tokenizer, log, and Vulkan dependency settings
├── scripts/                   # Maintained script entry points
│   ├── semantic/              # CPU/GPU semantic regression scripts
│   ├── gpu/                   # GPU PP, migration, and debug scripts
│   └── build.sh               # Standard build script
├── tests/semantic/            # Six-test benchmark regression and record generator
├── docs/test_records/         # Generated acceptance/regression records
├── maintenance/               # Archived patches, temporary fixes, and debug scripts
├── artifacts/                 # Archived historical logs and experiment outputs
├── tools/                     # Local Vulkan dependencies and helper tools
├── build.sh                   # Top-level build wrapper
└── run_*.sh                   # Compatibility wrappers forwarding to scripts/
```

Top-level `run_semantic_*.sh` and `run_gpu_*.sh` files are kept as compatibility wrappers. The maintained script locations are under `scripts/semantic/` and `scripts/gpu/`.

### Main Capabilities

- CPU and Vulkan GPU inference.
- Uneven tensor parallelism for heterogeneous devices.
- Pipeline parallelism and hybrid PP/TP layouts, for example `1@14*1:1@14`.
- Local weight loading for distributed nodes.
- Runtime migration controlled through UDS plan commands.
- Shadow KV and migration-readiness mechanisms.
- Last-stage local sampling, including TP last-stage gather before sampling, to avoid sending a full vocabulary logits tensor back to the root.
- Q80-resident root token embedding to reduce root memory footprint while keeping the model file format unchanged.
- Agentic workload and ablation harnesses for multi-generation, tool-using tasks.
- Auto warmup selection: automatically probes candidate partition topologies and worker subsets, then selects the configuration that minimizes the slowest node's per-token time.

### Environment

Default model paths can be configured in `config/env.sh`. Typical local paths are:

```bash
/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m
/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t
```

You can override them with:

```bash
export EDGEVISOR_MODEL3=/path/to/dllama_model.m
export EDGEVISOR_TOKENIZER=/path/to/dllama_tokenizer.t
export EDGEVISOR_LOG_ROOT=/path/to/runtime_logs
```

On the shared GPU test server, use GPU0, GPU1, and GPU2 unless a test explicitly says otherwise. Do not use `--gpu-index 3` and do not terminate processes on GPU3.

### Build

Recommended top-level build:

```bash
cd /path/to/EdgeVisor_GPU_Version
bash build.sh
```

Equivalent GPU build:

```bash
source config/env.sh
cd "$EDGEVISOR_ENGINE_DIR"
make DLLAMA_VULKAN=1 dllama dllama-api
```

CPU-only build:

```bash
DLLAMA_VULKAN=0 bash build.sh
```

### Common Runs

Static uneven TP/PP examples:

```bash
bash run_semantic_gpu_tp_static.sh
bash run_gpu_pp_static.sh
```

Dynamic migration examples:

```bash
bash run_semantic_gpu_dynamic_heads.sh
bash run_gpu_pp_migration.sh
```

A two-stage GPU smoke configuration with Stage 0 on GPU0 and Stage 1 using TP on GPU1/GPU2 can use:

```bash
--workers 127.0.0.1:25101 127.0.0.1:25102 \
--gpu-index 0 \
--ratios '1@14*1:1@14' \
--last-stage-sampling
```

### Auto Warmup Selection

When `--ratios` is not provided, `--warmup` enables automatic topology and worker-subset selection before inference. The warmup phase probes a configurable set of candidate configurations (pure TP, pure PP, hybrid PP/TP with dynamic layer counts, and different worker subsets), runs a short profiling generation pass on each, and selects the configuration that minimizes the slowest stage/node per-token time.

```bash
# Enable auto warmup (no --ratios needed)
./dllama --model model.m --tokenizer tokenizer.t \
  --workers 127.0.0.1:25101 127.0.0.1:25102 127.0.0.1:25103 \
  --warmup --warmup-steps 16 --warmup-budget 8

# With explicit candidate override (semicolon or whitespace separated)
./dllama --model model.m --tokenizer tokenizer.t \
  --workers 127.0.0.1:25101 127.0.0.1:25102 \
  --warmup --warmup-candidates "1@14*1:1@14 1@7*1@7*1@7*1@7"
```

| Flag | Default | Description |
|---|---|---|
| `--warmup` | disabled | Enable auto topology/worker-subset selection |
| `--warmup-steps` | 16 | Generation steps per candidate probe |
| `--warmup-budget` | 8 | Maximum number of candidates to probe |
| `--warmup-candidates` | (auto) | Explicit candidate override (semicolon/whitespace separated) |

Warmup scoring uses the minimax metric: the slowest stage/node per-token total time (exec + sync + bubble). Workers must be running and reachable before starting the root with `--warmup`. The inter-probe relisten delay is controlled by the `DLLAMA_WARMUP_RELISTEN_DELAY_MS` environment variable (default 1000 ms; set to 0 to disable). If `--ratios` is also provided, warmup is skipped and the explicit ratios are used.

### Regression Tests

The six standard acceptance/regression tests are:

1. CPU single-node inference
2. GPU single-node inference
3. CPU uneven static inference
4. GPU uneven static inference
5. CPU uneven dynamic migration
6. GPU uneven dynamic migration

Run them with:

```bash
cd /path/to/EdgeVisor_GPU_Version
bash run_six_benchmark_tests.sh
```

Logs are written to:

```bash
runtime_logs/benchmark_docs_YYYYmmdd_HHMMSS/
```

Generate test records with:

```bash
python3 tests/semantic/generate_test_records.py \
  --logs runtime_logs/benchmark_docs_YYYYmmdd_HHMMSS \
  --out docs/test_records \
  --project-root "$PWD"
```

---

## 中文

EdgeVisor GPU 版本是面向异构边缘设备的分布式大模型推理实验工程。它在原有分布式推理流程的基础上，扩展了 GPU/Vulkan 执行、非均匀静态张量并行、流水线并行、PP/TP 混合切分、UDS 控制的运行时迁移、Shadow KV、最后 Stage 本地采样，以及 Q80 常驻 Root token embedding 等面向内存和通信开销的优化。

### 致谢

EdgeVisor 基于 [distributed-llama](https://github.com/b4rtaz/distributed-llama) 开发。我们衷心感谢 distributed-llama 的作者和贡献者。该项目提供了模型格式、tokenizer/runtime 结构、分布式 root-worker 执行模型、同步机制以及原始 CPU/Vulkan 推理基础，为 EdgeVisor 的后续实验和扩展奠定了重要基础。

### 目录结构

```text
EdgeVisor_GPU_Version/
├── EdgeVisor/                 # 核心 C++/Vulkan 推理引擎
│   ├── src/                   # 推理、网络同步、动态迁移、tokenizer、CPU/Vulkan backend
│   ├── examples/              # UDS client 和原始示例
│   ├── docs/                  # 引擎级使用说明
│   └── Makefile               # 核心二进制构建入口
├── config/env.sh              # 统一模型路径、tokenizer、日志目录、Vulkan 依赖环境
├── scripts/                   # 维护后的脚本入口
│   ├── semantic/              # CPU/GPU 语义回归与分布式推理脚本
│   ├── gpu/                   # GPU PP、迁移和调试脚本
│   └── build.sh               # 标准构建脚本
├── tests/semantic/            # 六项 benchmark 回归和测试记录生成器
├── docs/test_records/         # 生成的验收/回归测试记录
├── maintenance/               # 历史补丁、临时修复脚本、调试配置归档
├── artifacts/                 # 历史日志和实验结果归档
├── tools/                     # 本地 Vulkan 依赖等辅助工具
├── build.sh                   # 顶层构建 wrapper
└── run_*.sh                   # 顶层兼容 wrapper，转发到 scripts/
```

顶层 `run_semantic_*.sh` 和 `run_gpu_*.sh` 文件保留为兼容入口；真实维护位置在 `scripts/semantic/` 和 `scripts/gpu/`。

### 主要能力

- CPU 和 Vulkan GPU 推理。
- 面向异构设备的非均匀张量并行。
- 流水线并行和 PP/TP 混合切分，例如 `1@14*1:1@14`。
- 分布式节点本地加载权重。
- 通过 UDS plan command 控制运行时迁移。
- Shadow KV 和迁移就绪机制。
- 最后 Stage 本地采样；当最后 Stage 内部使用 TP 时，会先在最后 Stage 内部汇聚 logits slice，再由 Stage root 采样，避免把完整 vocab logits 回传给全局 Root。
- Root token embedding 使用 Q80 常驻，降低 Root 内存占用，同时不改变原始模型文件格式。
- 支持多轮 generation、工具调用和迁移注入的 Agentic workload 与 ablation harness。
- 自动预热选优：自动探测候选切分拓扑和 Worker 组合，选择最小化最慢节点每 token 耗时的配置。

### 环境配置

默认模型路径可在 `config/env.sh` 中配置。常用本地路径为：

```bash
/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m
/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t
```

也可以通过环境变量覆盖：

```bash
export EDGEVISOR_MODEL3=/path/to/dllama_model.m
export EDGEVISOR_TOKENIZER=/path/to/dllama_tokenizer.t
export EDGEVISOR_LOG_ROOT=/path/to/runtime_logs
```

在共享 GPU 测试服务器上，除非测试明确要求，否则只使用 GPU0、GPU1、GPU2。不要使用 `--gpu-index 3`，也不要终止 GPU3 上的进程。

### 构建

推荐使用项目根目录入口：

```bash
cd /path/to/EdgeVisor_GPU_Version
bash build.sh
```

等价 GPU 构建命令：

```bash
source config/env.sh
cd "$EDGEVISOR_ENGINE_DIR"
make DLLAMA_VULKAN=1 dllama dllama-api
```

CPU-only 构建：

```bash
DLLAMA_VULKAN=0 bash build.sh
```

### 常用运行方式

静态非均匀 TP/PP：

```bash
bash run_semantic_gpu_tp_static.sh
bash run_gpu_pp_static.sh
```

动态迁移：

```bash
bash run_semantic_gpu_dynamic_heads.sh
bash run_gpu_pp_migration.sh
```

两阶段 GPU smoke 示例：Stage 0 使用 GPU0，Stage 1 使用 GPU1/GPU2 做 TP，可使用：

```bash
--workers 127.0.0.1:25101 127.0.0.1:25102 \
--gpu-index 0 \
--ratios '1@14*1:1@14' \
--last-stage-sampling
```

### 自动预热选优

当未提供 `--ratios` 时，`--warmup` 在推理前启用自动拓扑和 Worker 子集选择。预热阶段探测一组可配置的候选配置（纯 TP、纯 PP、混合 PP/TP 及不同 Worker 子集），对每个候选运行短时间的性能采样生成，并选择最小化最慢 Stage/节点每 token 耗时的配置。

```bash
# 启用自动预热（无需 --ratios）
./dllama --model model.m --tokenizer tokenizer.t \
  --workers 127.0.0.1:25101 127.0.0.1:25102 127.0.0.1:25103 \
  --warmup --warmup-steps 16 --warmup-budget 8

# 使用显式候选覆盖（分号或空格分隔）
./dllama --model model.m --tokenizer tokenizer.t \
  --workers 127.0.0.1:25101 127.0.0.1:25102 \
  --warmup --warmup-candidates "1@14*1:1@14 1@7*1@7*1@7*1@7"
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--warmup` | 关闭 | 启用自动拓扑/Worker 子集选择 |
| `--warmup-steps` | 16 | 每个候选探针的生成步数 |
| `--warmup-budget` | 8 | 最多探测的候选数量 |
| `--warmup-candidates` | (自动) | 显式候选覆盖（分号/空格分隔） |

预热使用 minimax 指标评分：最慢 Stage/节点的每 token 总耗时（exec + sync + bubble）。Worker 必须在启动带 `--warmup` 的 Root 之前已运行并可达。探测间隔的 relisten 延迟由环境变量 `DLLAMA_WARMUP_RELISTEN_DELAY_MS` 控制（默认 1000 ms；设为 0 可关闭）。若同时提供 `--ratios`，则跳过预热直接使用显式 ratios。

### 六项验收回归

完整回归包括：

1. CPU 单机推理
2. GPU 单机推理
3. CPU 非均匀静态推理
4. GPU 非均匀静态推理
5. CPU 非均匀动态迁移
6. GPU 非均匀动态迁移

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
