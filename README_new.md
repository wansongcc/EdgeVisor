# Distributed Llama（本仓库版本）使用说明（README_new）

本文档面向本工作区当前代码状态（包含：非均匀 TP/PP 并行、按 layer 粒度的在线迁移/重分片）。

- 目标读者：想从“能跑起来”到“能做非均匀并行 + 在线迁移验证”的同学
- 适用平台：Linux/macOS/Windows（root/worker 都需能编译运行）

> 说明：本仓库还保留了上游文档，更多细节可参考：
> - docs/HOW_TO_RUN_LINUX_MACOS_WIN.md
> - docs/HOW_TO_RUN_GPU.md
> - docs/README_ENV_VARS.md
> - docs/HOW_TO_ONLINE_MIGRATION.md（本文对其做了补充与更新）

---

## 1. 快速开始（单机单进程）

### 1.1 编译

在项目根目录：

```bash
make -j dllama
make -j dllama-api
```

（可选）开启 Vulkan：

```bash
DLLAMA_VULKAN=1 make -j dllama
DLLAMA_VULKAN=1 make -j dllama-api
```

（可选）开启 Attention/KV tracing（编译期开关，见第 7 节）：

```bash
ATTN=1 make -j dllama
```

### 1.2 下载模型（推荐用 launch.py）

在 root 节点执行：

```bash
python3 launch.py  # 列出可用模型
python3 launch.py qwen3_8b_q40
```

将得到类似：

- models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m
- models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t

也可以直接运行脚本：

```bash
bash run_qwen3_8b_q40.sh
```

### 1.3 运行 chat / inference

```bash
./dllama chat \
  --model models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m \
  --tokenizer models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t \
  --buffer-float-type q80 \
  --nthreads 8 \
  --max-seq-len 4096
```

---

## 2. 分布式运行（Root + Workers）

典型拓扑（以 1 个 root + 3 个 worker 为例）：

- root：10.0.0.1
- worker1：10.0.0.2:9999
- worker2：10.0.0.3:9999
- worker3：10.0.0.4:9999

### 2.1 在所有机器编译

在 root 和所有 worker 上都执行：

```bash
git clone <your-repo-url>
cd distributed-llama
make -j dllama
```

### 2.2 启动 worker 进程

在每台 worker 机器：

```bash
./dllama worker --port 9999 --nthreads 8
```

（本机多 worker 调试）可用 examples/n-workers.sh：

```bash
W=3 T=2 bash examples/n-workers.sh start
# W=3 bash examples/n-workers.sh stop
```

### 2.3 在 root 上运行推理

```bash
./dllama inference \
  --prompt "Hello world" \
  --steps 64 \
  --model models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m \
  --tokenizer models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t \
  --buffer-float-type q80 \
  --nthreads 8 \
  --max-seq-len 4096 \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999
```

如果你希望“一次推理结束后不退出”，可以加 `--interactive`，进程会在每次推理结束后等待终端输入：

```bash
./dllama inference --interactive \
  --prompt "Hello world" \
  --steps 64 \
  --model models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m \
  --tokenizer models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t
```

---

## 3. 非均匀 TP/PP 并行（--ratios）

本仓库支持“多 Stage（PP）+ Stage 内 TP（可非均匀）”。入口参数是：

- root 的 ./dllama inference/chat/api：增加 `--ratios <spec>`

`--ratios` 的解析逻辑在 src/app.cpp 的 parseStageDefs。

### 3.1 核心概念

- 一个 Stage = 一段连续 layer 区间（PP 切分）
- Stage 内的若干 node 做 TP（张量并行），每个 node 可按比例分配计算/切分长度（非均匀 TP）
- 多个 Stage 串起来构成 PP pipeline

### 3.2 ratios 字符串格式（两种都支持）

#### 格式 A：仅按 Stage 的 TP 比例串联（推荐）

用 `*`（或 `;` / `|`）分隔 Stage；Stage 内用 `:` 或 `,` 表示每个 node 的 TP 比例。

- 2 nodes，2 stages：

```text
1*1
```

- 4 nodes，2 stages（每个 stage 2 个节点）：

```text
1:1*1:1
```

- 4 nodes，2 stages（stage0 两节点等比例，stage1 两节点 2:3 非均匀）：

```text
1:1*2:3
```

可选：给某个 stage 指定 layer 数（更可控，推荐用 @ 语法）：

```text
1:1@10*2:3@18
```

> legacy 兼容：当 stage 内比例用逗号时，允许写成 "1,1:10" 表示 layers=10（注意此语法在 ':' 比例下会歧义，因此只对逗号生效）。

#### 格式 B：先给 Stage 权重，再给每个 Stage 的 TP 比例

形式：

```text
stageWeights*tpStage0*tpStage1*...
```

例如（nNodes=4）：stage 权重 1:2；stage0 两节点 1:1；stage1 两节点 2:3

```text
1:2*1:1*2:3
```

同样可对某些 stage 指定层数：

```text
1:2*1:1@10*2:3@18
```

### 3.3 在 root 命令里使用 ratios

例：单 stage，2 nodes（root + 1 worker）做非均匀 TP：

```bash
./dllama inference ... \
  --workers 127.0.0.1:9999 \
  --ratios 2:3
```

例：两 stage，总 4 nodes（root + 3 workers），stage0：1:1，stage1：2:3

```bash
./dllama inference ... \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999 \
  --ratios 1:1*2:3
```

---

## 4. Layer 粒度在线迁移（PlanCommand + per-layer barrier/apply）

本仓库实现了“每层插入 barrier/apply 的 CPU-only hook”，让你能在推理过程中动态调整：

- headSplit（注意力 head 维度切分）
- ffnSplit（FFN hidden dim 切分）

机制简述：

- OP_PLAN_BARRIER：在某层某个 token 位置命中触发条件后，在 stage 内广播一条“计划消息”（epoch + 迁移参数）
- OP_PLAN_APPLY：各节点收到消息后更新分片 plan，并 bump epoch；之后 forward 会 refresh pointers

### 4.1 启用迁移功能（必须）

在 root 运行 inference/chat/api 时设置：

```bash
export DLLAMA_ENABLE_PLAN_BARRIER=1
```

并建议打开 stage full residency（迁移/冗余更容易正确）：

```bash
export DLLAMA_STAGE_FULL_WEIGHTS=1
```

### 4.2 启用 UDS 控制器（推荐）

让 root 启动 UDS 监听，外部通过 UNIX domain socket 下发 PlanCommand：

```bash
export DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock
```

启动后可以用客户端：

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock ping
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock status
```

### 4.3 下发单边迁移（legacy 单 move）

- mode=next_barrier：下一次 barrier 触发

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock set_plan \
  --seq 1 --mode next_barrier --stage 0 \
  --from 0 --to 1 --kind 3 --heads 1 --ffn 256
```

- mode=exact：到达指定 (trigger-layer, trigger-pos) 才触发

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock set_plan \
  --seq 2 --mode exact --stage 0 \
  --from 0 --to 1 --kind 3 --heads 1 --ffn 256 \
  --trigger-layer 5 --trigger-pos 30
```

### 4.4 下发多点定向迁移（v2 moves[]，同一时刻同时生效）

客户端用可重复的 `--move FROM,TO[,KIND[,HEADS[,FFN]]]`，会生成 PlanCommand v2 的 moves[]。

例：3 条定向迁移同时生效：0→1、2→1、3→2

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock set_plan \
  --seq 10 --mode next_barrier --stage 0 \
  --move 0,1,3,1,256 \
  --move 2,1,3,1,256 \
  --move 3,2,3,1,256
```

### 4.5 运行时约束（重要）

为避免迁移后产生计算错误，本实现对 move 有约束：

- 迁移只允许发生在同一 stage 内
- 只允许 stage 内“顺序相邻”的节点迁移（按 stage 中 nodeIndices 的顺序，距离必须为 1）
- v2 moves[] 的数量会做运行时上限：maxAllowed = min(64, 2 * stageNodes)

另外，涉及 KV/GQA 的 head 迁移时还有安全约束：

- 在 GQA lockstep 场景，headMove 会被视为“KV heads 的迁移”
- 无显式 KV cache 传输时，仅允许迁移量不超过冗余 pad（当前为 2 个 KV head；见 NN_KV_REDUNDANCY_PAD_HEADS）

### 4.6 KV redundancy during migration（默认行为）

在启用 plan barrier 时，代码默认会尽量启用 KV redundancy 计算切片（kvHeadComputeSplit），用于在小范围 KV-head 迁移时保持连续性。

你仍然可以显式关闭：

```bash
export DLLAMA_ENABLE_KV_REDUNDANCY_DURING_MIGRATION=0
```

> 注意：KV redundancy 依赖 full KV buffer（通常需要 DLLAMA_STAGE_FULL_WEIGHTS=1）。

### 4.7 如何确认迁移已发生

运行时会打印类似日志（只要启用了 DLLAMA_ENABLE_PLAN_BARRIER，并且命中触发点）：

- barrier 侧：
  - 🧭 [plan][emit] ...
- apply 侧：
  - 🧭 [plan][apply] ...
  - 失败会打印 reject 原因（non-adjacent / underflow / KV safety 等）

---

## 5. GPU（Vulkan）运行

编译：

```bash
DLLAMA_VULKAN=1 make -j dllama
```

运行时参数：

- `--gpu-index 0` 使用第 0 块 GPU
- Vulkan 后端建议 `--nthreads 1`

例：

```bash
./dllama chat ... --nthreads 1 --gpu-index 0
```

---

## 6. 常见问题（FAQ）

### 6.1 UDS socket 不存在 / ping 失败

- 确认 root 进程设置了：
  - DLLAMA_ENABLE_PLAN_BARRIER=1
  - DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock
- 如 socket 被旧进程占用：

```bash
rm -f /tmp/dllama_plan.sock
```

### 6.2 看到 emit 但看不到 apply

- 确认触发点后续确实执行到了 apply（per-layer segments 顺序是 barrier → SYNC_WITH_ROOT → apply）
- 如果 apply 拒绝，会打印 🧭 [plan][apply] reject: ...

### 6.3 迁移被拒绝：non-adjacent

说明 from/to 在该 stage 内不是相邻顺序节点。需要调整 move 的方向或 stage 的节点排列。

---

## 7. 调试/Tracing 开关（推荐组合）

### 7.1 KV range tracing（需要 ATTN=1 编译）

编译：

```bash
make clean
ATTN=1 make -j dllama
```

运行：

```bash
export kvcache_debug=1
# 或 export DLLAMA_DEBUG_KV_RANGE=1
./dllama inference ...
```

### 7.2 让 root 同步 env 到 workers

如果你希望调试 env 在 workers 也生效：

```bash
export DLLAMA_SYNC_ENV_VARS="kvcache_debug,kvcache_debug_limit"
```

---

## 8. 附：最小可复现（单机 2 进程 + 在线迁移）

终端 A（worker）：

```bash
./dllama worker --port 9999 --nthreads 4
```

终端 B（root）：

```bash
export DLLAMA_ENABLE_PLAN_BARRIER=1
export DLLAMA_STAGE_FULL_WEIGHTS=1
export DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock

./dllama inference \
  --prompt "The capital of France is" \
  --steps 128 \
  --model models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m \
  --tokenizer models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t \
  --buffer-float-type q80 \
  --nthreads 4 \
  --max-seq-len 2048 \
  --workers 127.0.0.1:9999 \
  --ratios 1:1
```

终端 C（下发迁移）：

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock set_plan \
  --seq 1 --mode next_barrier --stage 0 --from 0 --to 1 --kind 3 --heads 1 --ffn 256
```
