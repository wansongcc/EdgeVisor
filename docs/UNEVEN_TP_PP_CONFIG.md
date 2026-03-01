# 非均匀 TP/PP 启动配置指南

本文档介绍如何通过命令行参数为分布式推理中的每个设备配置工作量分配。

---

## 目录

1. [核心概念](#核心概念)
2. [配置参数概览](#配置参数概览)
3. [--ratios 参数详解](#--ratios-参数详解)
4. [--kv-redundancy 参数详解](#--kv-redundancy-参数详解)
5. [完整配置示例](#完整配置示例)
6. [代码实现细节](#代码实现细节)

---

## 核心概念

### Tensor Parallelism (TP) 和 Pipeline Parallelism (PP)

- **TP (张量并行)**: 在同一 stage 内，将计算任务按比例分配给多个节点
- **PP (流水线并行)**: 将模型的层划分为多个连续的 stage，每个 stage 负责处理一段连续的层

### Stage (阶段)

- 一个 Stage 对应模型中一段连续的层
- Stage 内部使用 TP 并行处理
- 多个 Stage 串联形成 PP pipeline

### 非均匀分配

- 允许同一 stage 内的节点以不同比例分配计算任务
- 允许不同 stage 根据权重分配层数

---

## 配置参数概览

| 参数 | 描述 | 格式示例 | 默认值 |
|------|------|----------|--------|
| `--ratios` | TP/PP 工作量分配比例 | `"1:1*2:3"` | 均匀分配 |
| `--kv-redundancy` | 每个节点的 KV 冗余头数量 | `"2"` 或 `"2,3,2,3"` | `2` |
| `--enable-plan-barrier` | 启用在线迁移屏障 | `1` 或 `0` | `0` |

---

## --ratios 参数详解

### 参数解析位置

代码实现位于: [src/app.cpp:404](../src/app.cpp#L404) 的 `parseStageDefs` 函数

### 支持的两种格式

#### 格式 A：仅按 Stage 的 TP 比例串联（推荐）

使用 `*`、`;` 或 `|` 分隔不同的 Stage；Stage 内使用 `:` 或 `,` 表示每个节点的 TP 比例。

**语法结构**:
```
tpStage0*tpStage1*tpStage2*...
```

其中每个 `tpStageK` 可以是:
- `"1:1"` - 两个节点等比例
- `"2:3"` - 两个节点 2:3 非均匀
- `"1:2:3"` - 三个节点 1:2:3 非均匀

**可选显式层数**:
- `"1:1@10"` - 该 stage 分配 10 层
- `"1,1@10"` - 逗号分隔时也可使用

**示例**:

```bash
# 2 nodes, 2 stages（每个 stage 1 个节点）
--ratios 1*1

# 4 nodes, 2 stages（每个 stage 2 个节点，等比例）
--ratios 1:1*1:1

# 4 nodes, 2 stages（stage0 等比例 1:1，stage1 非均匀 2:3）
--ratios 1:1*2:3

# 6 nodes, 3 stages（各 stage 分别有 2、2、2 个节点）
--ratios 1:1*1:2*1:2

# 显式指定层数（总共 28 层）
--ratios 1:1@10*1:1@18
```

#### 格式 B：先给 Stage 权重，再给每个 Stage 的 TP 比例

使用两级结构：先指定各 stage 的权重，然后指定每个 stage 的内部 TP 比例。

**语法结构**:
```
stageWeights*tpStage0*tpStage1*...
```

- `stageWeights`: 使用 `:` 或 `,` 分隔的权重列表，每个 stage 一个权重
- `tpStageK`: 该 stage 内的 TP 节点比例

**示例**:

```bash
# stage 权重 1:2；stage0 两节点 1:1；stage1 两节点 2:3
--ratios 1:2*1:1*2:3

# 同样可对某些 stage 指定层数
--ratios 1:2*1:1@10*2:3@18
```

### 格式自动检测

代码会自动检测使用的格式：
- 如果第一段的值数量等于总节点数，则判定为格式 A
- 如果第一段的值数量小于总节点数，则判定为格式 B

---

## --kv-redundancy 参数详解

### 参数作用

KV 冗余用于在线迁移时保持计算连续性。每个节点可以存储比自身实际需要更多的 KV 头，以便在迁移时不需传输 KV cache。

### 参数位置

代码实现位于: [src/app.cpp:362](../src/app.cpp#L362) 的 `parseKvRedundancy` 函数

### 支持的格式

```bash
# 单个值：所有节点使用相同数量的冗余头
--kv-redundancy 2

# 每节点指定：分别为每个节点设置不同的冗余头数量
--kv-redundancy 2,3,2,3

# 不指定：默认使用 2 个冗余头（与 NN_KV_REDUNDANCY_PAD_HEADS 一致）
```

### 约束

- KV 冗余仅用于 enable-plan-barrier 启用时的在线迁移
- 默认冗余量为 2 个 KV head
- 如需完全禁用迁移期 KV 冗余，使用命令行参数：
  ```bash
  --enable-kv-redundancy-during-migration 0
  ```

---

## 完整配置示例

### 示例 1：单 Stage，2 节点非均匀 TP

```bash
./dllama inference \
  --prompt "Hello world" \
  --steps 64 \
  --model models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m \
  --tokenizer models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t \
  --buffer-float-type q80 \
  --nthreads 8 \
  --workers 127.0.0.1:9999 \
  --ratios 2:3 \
  --kv-redundancy 2
```

**说明**:
- 总共 2 个节点（root + 1 worker）
- 节点 0 (root): 分配 40% 的计算任务
- 节点 1 (worker): 分配 60% 的计算任务

### 示例 2：双 Stage，4 节点非均匀分配

```bash
./dllama inference \
  --prompt "Explain quantum computing" \
  --steps 128 \
  --model models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m \
  --tokenizer models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t \
  --buffer-float-type q80 \
  --nthreads 8 \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999 \
  --ratios 1:1*2:3 \
  --kv-redundancy 2,2,2,2
```

**说明**:
- 总共 4 个节点（root + 3 workers）
- Stage 0: 2 个节点（node 0, 1），等比例 1:1
- Stage 1: 2 个节点（node 2, 3），非均匀 2:3
- 每个节点都有 2 个 KV 冗余头

### 示例 3：双 Stage，使用权重分配层数

```bash
./dllama inference \
  --prompt "Write a story" \
  --steps 256 \
  --model models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m \
  --tokenizer models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t \
  --buffer-float-type q80 \
  --nthreads 8 \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999 \
  --ratios 1:2*1:1@9*2:3@19 \
  --kv-redundancy 2
```

**说明**:
- 总共 4 个节点
- Stage 0 权重 1，Stage 1 权重 2（Stage 1 将分配约 2/3 的层）
- Stage 0 显式指定 9 层，Stage 1 显式指定 19 层
- Stage 0 内 2 节点等比例 1:1
- Stage 1 内 2 节点非均匀 2:3

### 示例 4：启用在线迁移

```bash
# 设置 UDS 控制器 socket
export DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock

# 启动推理
./dllama inference \
  --prompt "The capital of France is" \
  --steps 128 \
  --model models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m \
  --tokenizer models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t \
  --buffer-float-type q80 \
  --nthreads 8 \
  --enable-plan-barrier \
  --enable-stage-full-weights \
  --enable-kv-redundancy-during-migration 1 \
  --workers 127.0.0.1:9999 \
  --ratios 1:1 \
  --kv-redundancy 2
```

**在线迁移命令**:

```bash
# 发送迁移计划：从节点 0 向节点 1 移动 1 个 head 和 256 个 FFN 单元
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock set_plan \
  --seq 1 --mode next_barrier --stage 0 --from 0 --to 1 --kind 3 --heads 1 --ffn 256
```

---

## 代码实现细节

### 核心数据结构

#### NnStageDef ([src/nn/nn-core.hpp:55](../src/nn/nn-core.hpp#L55))

```cpp
struct NnStageDef {
    NnUint nLayers;                       // 该 Stage 负责多少层
    std::vector<float> tpRatios;           // 该 Stage 内部的 TP 比例
    std::vector<NnUint> kvRedundancyPerNode;  // 每个节点的 KV 冗余 head 数量
};
```

#### NnUnevenPartitionPlan ([src/nn/nn-core.hpp:92](../src/nn/nn-core.hpp#L92))

```cpp
struct NnUnevenPartitionPlan {
    NnUint nNodes;
    NnUint nStages;
    NnStageConfig *stages;                // Stage 数组

    NnDimSplit headSplit;                 // 注意力头切分
    NnDimSplit kvHeadSplit;               // KV 头切分
    NnDimSplit kvHeadComputeSplit;        // KV 计算范围（用于冗余）
    NnDimSplit vocabSplit;                // 词汇表切分
    NnDimSplit ffnSplit;                  // FFN 切分
    NnDimSplit dimSplit;                  // 维度切分
};
```

#### NnDimSplit ([src/nn/nn-core.hpp:45](../src/nn/nn-core.hpp#L45))

```cpp
typedef struct {
    NnUint* starts;    // 每个节点的起始位置
    NnUint* lengths;   // 每个节点的长度
} NnDimSplit;
```

### 关键函数

#### 1. parseStageDefs([src/app.cpp:404](../src/app.cpp#L404))

解析 `--ratios` 参数，生成 `NnStageDef` 向量。

**输入**:
- `ratiosStr`: 比例字符串（如 `"1:1*2:3"`）
- `nNodes`: 总节点数
- `nLayers`: 总层数

**输出**: `std::vector<NnStageDef>` - 各 stage 的定义

**功能**:
- 自动检测格式 A 或格式 B
- 支持显式层数指定（`@n` 语法）
- 自动分配未指定层数的 stage（按权重比例）
- 验证节点数匹配

#### 2. parseKvRedundancy([src/app.cpp:362](../src/app.cpp#L362))

解析 `--kv-redundancy` 参数。

**输入**:
- `redundancyStr`: 冗余字符串（如 `"2"` 或 `"2,3,2,3"`）
- `nNodes`: 总节点数

**输出**: `std::vector<NnUint>` - 每个节点的冗余头数量

**功能**:
- 支持单个值（应用到所有节点）
- 支持每节点指定
- 默认返回 2（不指定时）

#### 3. createPartitionPlan([src/nn/nn-core.hpp:713](../src/nn/nn-core.hpp#L713))

根据 stage 定义创建完整的非均匀分区计划。

**输入**:
- `stageDefs`: 各 stage 定义
- `globalNHeads`: 全局注意力头数
- `globalNKvHeads`: 全局 KV 头数
- `globalVocabSize`: 全局词汇表大小
- `globalFfnDim`: 全局 FFN 维度
- `globalDim`: 全局模型维度
- `globalKvRedundancyPerNode`: 每节点的 KV 冗余（可选）

**输出**: `NnUnevenPartitionPlan` - 完整的分区计划

**功能**:
- 计算所有维度的非均匀切分
- 应用 GQA 对齐修复
- 配置 KV 冗余计算范围
- 支持各维度有不同的对齐要求

### 启动流程

1. **参数解析** ([src/app.cpp:273](../src/app.cpp#L273))
   - 解析命令行参数
   - 提取 `--ratios` 和 `--kv-redundancy`

2. **Stage 定义生成** ([src/app.cpp:404](../src/app.cpp#L404))
   - 调用 `parseStageDefs` 生成 stage 定义
   - 调用 `parseKvRedundancy` 生成 KV 冗余配置

3. **分区计划创建** ([src/nn/nn-core.cpp](../src/nn/nn-core.cpp))
   - 调用 `createPartitionPlan` 创建完整分区计划
   - 包含所有维度的切分信息

4. **节点配置分发**
   - root 将分区计划发送给所有 workers
   - 每个节点根据计划配置本地切分

---

## 相关文档

- [在线迁移指南](HOW_TO_ONLINE_MIGRATION.md)
- [环境变量开关](README_ENV_VARS.md)
- [Linux/MacOS/Windows 运行指南](HOW_TO_RUN_LINUX_MACOS_WIN.md)
