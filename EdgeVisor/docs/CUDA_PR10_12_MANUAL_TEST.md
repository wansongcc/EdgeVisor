# CUDA PR10–PR12 手工验收指南

本文档用于手工验证 CUDA 后端 PR10、PR11、PR12 的实现效果。它覆盖本地 CPU/CUDA 回归、CUDA 算子与 runtime 单测、Compute Sanitizer、单 GPU 模型语义、三 GPU 静态 TP/PP/TP+PP、动态 Head/FFN 重分片、PP KV 迁移、runtime redundant segment 和 Bubble Shadow KV。

原则：

- 不在测试脚本中写死模型路径、GPU 编号、端口或 worker 地址。
- 所有模型路径、GPU、端口、worker 拓扑都通过环境变量传入。
- 单元测试使用程序生成的小张量，不依赖模型文件。
- 模型级测试使用 `EDGEVISOR_MODEL3`、`EDGEVISOR_TOKENIZER` 和可选 `EDGEVISOR_MOE_MODEL`。
- CUDA 构建使用 CUDA Toolkit 12.x；默认 `CUDA_ARCHS=auto`，会在本机通过 `nvidia-smi` 探测 GPU 架构，探测失败时回退到 `75`。
- 首轮不使用 NCCL、GPUDirect、FlashAttention、PageAttention 或 CUDA Graph。

---

## 1. 测试范围与通过标准

### 1.1 PR10 静态分布式 TP、PP 和混合执行

需要验证：

- CUDA Segment 输出在 executor/network 同步前已经下载到 host pipe。
- 网络同步后的 host pipe 在后续 CUDA Segment 前重新上传。
- 非均匀 TP 可运行。
- 三 stage PP 可运行。
- TP+PP 可运行。
- `--gpu-segments` CPU/CUDA 混合执行可运行。
- 现有 host pipe TCP 协议不变，不引入 NCCL。
- Vulkan 静态 TP/PP 不回退。

必须看到的关键日志：

```text
CUDA PR10 segment output visible before host sync: ok
CUDA PR10 CPU/CUDA mixed segment output: ok
CUDA PR10 static mixed executor host-pipe semantics: ok
```

模型级静态分布式通过标准：

- 固定 prompt、`temperature=0`、`seed=1`。
- 单 GPU CUDA 与分布式 CUDA top-1 token 序列一致。
- logits 不含 NaN/Inf。
- 每个模式至少生成 64 个 prediction token。

### 1.2 PR11 动态 Head/FFN 重分片

需要验证：

- CUDA 实现 `setPartitionPlan()`。
- CUDA 实现 `refreshPointers()`。
- `OP_PLAN_BARRIER` 可以 emit plan command。
- `OP_PLAN_APPLY` 可以 apply plan command。
- 更新 slice offset、logical size、Matmul view、MHA stride、Shift destination。
- plan epoch 递增，segment 在新 plan 下重新绑定。
- 迁移失败或范围越界时保持旧 plan，不产生部分更新。

必须看到的关键日志：

```text
[plan][emit][cuda]
[plan][apply][cuda]
CUDA PR11 plan barrier/apply and refreshPointers tests: ok epoch=...
```

模型级动态重分片通过标准：

- UDS `next_barrier` Head 迁移成功。
- FFN 迁移成功。
- Head+FFN 联合迁移成功。
- 日志包含 emit、apply、epoch、新旧 range。
- 迁移前后无越界、NaN、死锁、重复应用。
- 迁移后的 top-1 token 序列与等效静态最终分片运行一致。

### 1.3 PR12 PP 层迁移、KV 传输、Bubble Shadow

需要验证：

- CUDA 实现 `exportLayerKvRow()`。
- CUDA 实现 `applyTransferredKvRow()`。
- 支持完整 KV row。
- 支持指定 range 的部分 KV row。
- PP boundary migration 完成 KV export、网络传输、apply、ack、ownership switch。
- runtime primary/redundant gate 正常。
- Bubble Shadow KV 同步/异步模式都能推理。

必须看到的关键日志：

```text
[kv-export-cuda]
[kv-write-cuda]
CUDA PR12 full KV export key: ok
CUDA PR12 full KV export value: ok
CUDA PR12 partial KV export key: ok
CUDA PR12 partial KV export value: ok
CUDA PR12 partial KV apply key: ok
CUDA PR12 partial KV apply value: ok
CUDA PR12 KV export/apply tests: ok
```

模型级 PP migration 通过标准：

- PP 边界迁移完成 export、transfer、apply、ack、ownership switch。
- 目标节点 KV history 与源节点逐元素对照，误差 `<= 1e-6`。
- 重复、乱序、超范围迁移命令安全拒绝。
- Bubble Shadow 同步和异步模式均完成推理。
- 无共享 buffer race。

---

## 2. 环境准备

### 2.1 基础环境变量

在 root 节点和需要执行脚本的机器上设置：

```bash
cd /home/cc/EdgeVisor/EdgeVisor

# 本机编译通常不需要设置 CUDA_ARCHS；Makefile 会自动检测。
# 交叉编译或给其他机器打包时再显式设置，例如：export CUDA_ARCHS="75 87 89"
export EDGEVISOR_GPU_INDEX=0
export EDGEVISOR_MAKE_JOBS=4

export EDGEVISOR_LOG_DIR="$PWD/runtime_logs/manual_cuda_pr10_12_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EDGEVISOR_LOG_DIR"
```

如果是在当前机器本机编译，通常不需要设置 `CUDA_ARCHS`。如果是交叉编译、容器内没有 `nvidia-smi`、或要一次编译给多种 GPU 使用，需要显式设置，例如 `CUDA_ARCHS="75 87 89"`。

### 2.2 模型环境变量

模型级测试必须设置：

```bash
export EDGEVISOR_MODEL3=/path/to/dllama_model_3b_or_small_q40.m
export EDGEVISOR_TOKENIZER=/path/to/dllama_tokenizer.t
```

可选 MoE：

```bash
export EDGEVISOR_MOE_MODEL=/path/to/dllama_model_qwen3_moe_q40.m
```

推荐固定推理参数：

```bash
export EDGEVISOR_PROMPT="The capital of France is"
export EDGEVISOR_STEPS=96
export EDGEVISOR_MAX_SEQ_LEN=2048
export EDGEVISOR_NTHREADS=1
export EDGEVISOR_BUFFER_FLOAT_TYPE=q80
```

说明：

- `EDGEVISOR_STEPS=96` 通常能确保 prediction token 数量不少于 64，因为 prompt eval 会占用一部分 token。
- 如果程序提示 `This configuration supports max 1 threads`，把 `EDGEVISOR_NTHREADS=1`。

### 2.3 三 GPU / 三节点 worker 环境变量

PR10 L4 静态分布式验收按“三 GPU”理解为 root + 2 workers，共 3 个 node。root 使用：

```bash
export EDGEVISOR_WORKERS="worker1.example:9999 worker2.example:9999"
export EDGEVISOR_TP_RATIOS="2:3:3"
export EDGEVISOR_PP_RATIOS="1@8*1@10*1@10"
export EDGEVISOR_TPPP_RATIOS="2:3:3@8*1@10*1@10"
```

注意：

- `EDGEVISOR_WORKERS` 是传给 root 的 worker 地址列表，不包含 root 自己。
- `EDGEVISOR_PP_RATIOS` 中 `@` 后面的层数之和必须等于模型层数。`1@8*1@10*1@10` 适合 28 层模型；如果模型是 32 层，应改成例如 `1@8*1@12*1@12`。
- `EDGEVISOR_TPPP_RATIOS` 也必须保证所有 stage 层数之和等于模型层数。

### 2.4 UDS / 动态迁移环境变量

动态测试使用 UDS 控制器：

```bash
export EDGEVISOR_PLAN_SOCKET="$EDGEVISOR_LOG_DIR/dllama_plan.sock"
export DLLAMA_PLAN_CTRL_SOCKET="$EDGEVISOR_PLAN_SOCKET"
```

Bubble Shadow：

```bash
export DLLAMA_BUBBLE_SHADOW_KV=1
export DLLAMA_BUBBLE_SHADOW_KV_LOG=1
```

异步 Bubble Shadow 默认开启。要测试同步模式：

```bash
export DLLAMA_BUBBLE_SHADOW_KV_ASYNC=0
```

---

## 3. 快速验收脚本

仓库内提供 3 个脚本：

- `scripts/cuda_full_regression.sh`
- `scripts/cuda_static_semantics.sh`
- `scripts/cuda_dynamic_semantics.sh`

这些脚本都会要求显式设置 `EDGEVISOR_GPU_INDEX`。

### 3.1 本地 L0 + CUDA 单测

```bash
export EDGEVISOR_GPU_INDEX=0
export EDGEVISOR_LOG_DIR="$PWD/runtime_logs/manual_full_$(date +%Y%m%d_%H%M%S)"

scripts/cuda_full_regression.sh
```

默认会执行：

- `make clean`
- CPU build
- `nn-cpu-test`
- `nn-cpu-ops-test`
- `tokenizer-test`
- `nn-slice-test`
- `make clean`
- CUDA build
- `nn-cuda-test --gpu-index "$EDGEVISOR_GPU_INDEX"`

默认不会执行：

- Compute Sanitizer
- 模型级静态语义
- 模型级动态语义

### 3.2 本地 L0 + CUDA 单测 + Compute Sanitizer

```bash
export EDGEVISOR_GPU_INDEX=0
export EDGEVISOR_RUN_SANITIZER=1
export EDGEVISOR_COMPUTE_SANITIZER=/usr/local/cuda/bin/compute-sanitizer
export EDGEVISOR_LOG_DIR="$PWD/runtime_logs/manual_full_sanitizer_$(date +%Y%m%d_%H%M%S)"

scripts/cuda_full_regression.sh
```

通过标准：

```text
========= ERROR SUMMARY: 0 errors
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

### 3.3 单 GPU CUDA 模型 smoke

```bash
export EDGEVISOR_GPU_INDEX=0
export EDGEVISOR_MODEL3=/path/to/dllama_model_q40.m
export EDGEVISOR_TOKENIZER=/path/to/dllama_tokenizer.t
export EDGEVISOR_NTHREADS=1
export EDGEVISOR_STEPS=96
export EDGEVISOR_LOG_DIR="$PWD/runtime_logs/manual_static_single_$(date +%Y%m%d_%H%M%S)"

scripts/cuda_static_semantics.sh
```

通过标准：

- 日志中存在 CUDA device 信息。
- 生成 prediction token 数量不少于 64。
- 每次 logits 检查都是 `Valid: ✅ OK`。
- 无 `NaN`、`Inf`、`Critical error`。

### 3.4 三 GPU 静态 TP/PP/TP+PP

先在两个 worker 节点分别启动 worker。每个 worker 的端口、GPU 由环境变量指定。

worker 节点：

```bash
cd /home/cc/EdgeVisor/EdgeVisor

export EDGEVISOR_GPU_INDEX=0
export EDGEVISOR_PORT=9999
export EDGEVISOR_NTHREADS=1

make -j"$EDGEVISOR_MAKE_JOBS" DLLAMA_CUDA=1 dllama

./dllama worker \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --port "$EDGEVISOR_PORT" \
  --nthreads "$EDGEVISOR_NTHREADS"
```

root 节点：

```bash
cd /home/cc/EdgeVisor/EdgeVisor

export EDGEVISOR_GPU_INDEX=0
export EDGEVISOR_MODEL3=/path/to/dllama_model_q40.m
export EDGEVISOR_TOKENIZER=/path/to/dllama_tokenizer.t
export EDGEVISOR_WORKERS="worker1.example:9999 worker2.example:9999"
export EDGEVISOR_NTHREADS=1
export EDGEVISOR_STEPS=96
export EDGEVISOR_RUN_L4=1
export EDGEVISOR_TP_RATIOS="2:3:3"
export EDGEVISOR_PP_RATIOS="1@8*1@10*1@10"
export EDGEVISOR_TPPP_RATIOS="2:3:3@8*1@10*1@10"
export EDGEVISOR_LOG_DIR="$PWD/runtime_logs/manual_static_l4_$(date +%Y%m%d_%H%M%S)"

scripts/cuda_static_semantics.sh
```

通过标准：

- `04_l4_tp_uneven.log` 生成不少于 64 个 prediction token。
- `05_l4_pp_three_stage.log` 生成不少于 64 个 prediction token。
- `06_l4_tppp.log` 生成不少于 64 个 prediction token。
- logits 均为 `Valid: ✅ OK`。
- root/worker 无死锁、断连、NaN、Inf、OOM。
- stage 边界 position pipe、ZQ pipe、logits gather 不报错。

---

## 4. 逐项手工验收命令

如果不使用脚本，可以按以下步骤逐项执行。

### 4.1 CPU-only L0

```bash
make clean
make -j"$EDGEVISOR_MAKE_JOBS" dllama nn-cpu-test nn-cpu-ops-test tokenizer-test nn-slice-test

./nn-cpu-test | tee "$EDGEVISOR_LOG_DIR/nn_cpu_test.log"
./nn-cpu-ops-test | tee "$EDGEVISOR_LOG_DIR/nn_cpu_ops_test.log"
./tokenizer-test | tee "$EDGEVISOR_LOG_DIR/tokenizer_test.log"
./nn-slice-test | tee "$EDGEVISOR_LOG_DIR/nn_slice_test.log"
```

通过标准：

- `nn-cpu-test` exit code 为 0。
- `nn-cpu-ops-test` 中每项显示 `passed`。
- `tokenizer-test` 中每项显示 `passed`。
- `nn-slice-test` 输出 `All pointer/slice/view tests passed`。

### 4.2 CUDA build + unit

```bash
make clean
make -j"$EDGEVISOR_MAKE_JOBS" DLLAMA_CUDA=1 dllama nn-cuda-test

./nn-cuda-test --gpu-index "$EDGEVISOR_GPU_INDEX" \
  | tee "$EDGEVISOR_LOG_DIR/nn_cuda_test.log"
```

必须通过的 PR10–PR12 关键行：

```bash
grep -E "CUDA PR10|CUDA PR11|CUDA PR12|plan\\]|kv-" "$EDGEVISOR_LOG_DIR/nn_cuda_test.log"
```

预期包含：

```text
CUDA PR10 segment output visible before host sync: ok
CUDA PR10 CPU/CUDA mixed segment output: ok
CUDA PR10 static mixed executor host-pipe semantics: ok
[plan][emit][cuda]
[plan][apply][cuda]
CUDA PR11 plan barrier/apply and refreshPointers tests: ok epoch=1
[kv-export-cuda]
[kv-write-cuda]
CUDA PR12 KV export/apply tests: ok
```

### 4.3 Compute Sanitizer

```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  ./nn-cuda-test --gpu-index "$EDGEVISOR_GPU_INDEX" \
  | tee "$EDGEVISOR_LOG_DIR/compute_sanitizer_memcheck.log"

/usr/local/cuda/bin/compute-sanitizer --tool racecheck \
  ./nn-cuda-test --gpu-index "$EDGEVISOR_GPU_INDEX" \
  | tee "$EDGEVISOR_LOG_DIR/compute_sanitizer_racecheck.log"
```

通过标准：

```bash
grep -E "ERROR SUMMARY|RACECHECK SUMMARY" \
  "$EDGEVISOR_LOG_DIR/compute_sanitizer_memcheck.log" \
  "$EDGEVISOR_LOG_DIR/compute_sanitizer_racecheck.log"
```

预期：

```text
ERROR SUMMARY: 0 errors
RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

### 4.4 单 GPU CUDA 模型运行

```bash
./dllama inference \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --model "$EDGEVISOR_MODEL3" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps "$EDGEVISOR_STEPS" \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --benchmark \
  | tee "$EDGEVISOR_LOG_DIR/model_cuda_single.log"
```

通过标准：

```bash
grep -E "CUDA Device|Root Logits|Prediction|tokens/s|Critical error|NaN|Inf" \
  "$EDGEVISOR_LOG_DIR/model_cuda_single.log"
```

要求：

- 日志中出现 `CUDA Device[...]`。
- logits 行全部是 `Valid: ✅ OK`。
- prediction token 数量不少于 64。
- 无 `Critical error`。
- 无 NaN/Inf。

### 4.5 CPU/CUDA top-1 token 序列对照

CUDA：

```bash
./dllama inference \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --model "$EDGEVISOR_MODEL3" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps "$EDGEVISOR_STEPS" \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --benchmark \
  | tee "$EDGEVISOR_LOG_DIR/model_cuda_top1.log"
```

CPU：

```bash
./dllama inference \
  --backend cpu \
  --model "$EDGEVISOR_MODEL3" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps "$EDGEVISOR_STEPS" \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --benchmark \
  | tee "$EDGEVISOR_LOG_DIR/model_cpu_top1.log"
```

比较 token 文本：

```bash
python3 - "$EDGEVISOR_LOG_DIR/model_cuda_top1.log" "$EDGEVISOR_LOG_DIR/model_cpu_top1.log" <<'PY'
import sys
from pathlib import Path

def pred_tokens(path):
    out = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        if "🔶 Pred" in line and " | " in line:
            out.append(line.split(" | ", 1)[1])
    return out

cuda = pred_tokens(sys.argv[1])
cpu = pred_tokens(sys.argv[2])
n = min(len(cuda), len(cpu))
first = None
for i in range(n):
    if cuda[i] != cpu[i]:
        first = i
        break

print("cuda_prediction_tokens", len(cuda))
print("cpu_prediction_tokens", len(cpu))
print("first_diff_index", first)
if first is not None:
    print("cuda", repr(cuda[first]))
    print("cpu ", repr(cpu[first]))
    sys.exit(1)
if len(cuda) != len(cpu):
    print("length_mismatch")
    sys.exit(1)
print("top1_sequences_match")
PY
```

通过标准：

```text
top1_sequences_match
```

当前已知风险：

- 在本地 Qwen3 0.6B Q40 smoke 中，CPU/CUDA 都无 NaN/Inf，CUDA 能生成 68 个 prediction token，但 top-1 序列在 `pos=47` 首次分叉。
- 如果你复现该分叉，模型级 deterministic top-1 一致性不能判为通过；需要继续定位 CPU/CUDA 数值差异。

---

## 5. PR10 静态分布式手工测试

### 5.1 启动 workers

在 worker1：

```bash
cd /home/cc/EdgeVisor/EdgeVisor

# 本机编译通常不需要设置 CUDA_ARCHS；Makefile 会自动检测。
# 交叉编译或给其他机器打包时再显式设置，例如：export CUDA_ARCHS="75 87 89"
export EDGEVISOR_GPU_INDEX=0
export EDGEVISOR_PORT=9999
export EDGEVISOR_NTHREADS=1

make -j"$EDGEVISOR_MAKE_JOBS" DLLAMA_CUDA=1 dllama

./dllama worker \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --port "$EDGEVISOR_PORT" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  2>&1 | tee "$EDGEVISOR_LOG_DIR/worker1.log"
```

在 worker2：

```bash
cd /home/cc/EdgeVisor/EdgeVisor

# 本机编译通常不需要设置 CUDA_ARCHS；Makefile 会自动检测。
# 交叉编译或给其他机器打包时再显式设置，例如：export CUDA_ARCHS="75 87 89"
export EDGEVISOR_GPU_INDEX=0
export EDGEVISOR_PORT=9999
export EDGEVISOR_NTHREADS=1

make -j"$EDGEVISOR_MAKE_JOBS" DLLAMA_CUDA=1 dllama

./dllama worker \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --port "$EDGEVISOR_PORT" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  2>&1 | tee "$EDGEVISOR_LOG_DIR/worker2.log"
```

注意：

- worker 必须显式传 `--backend cuda`。只传 `--gpu-index` 会触发旧兼容逻辑，可能选择 Vulkan。
- worker 不需要手动传模型路径；root 会下发配置。

### 5.2 单 GPU baseline

先生成单 GPU CUDA baseline：

```bash
./dllama inference \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --model "$EDGEVISOR_MODEL3" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps "$EDGEVISOR_STEPS" \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --benchmark \
  | tee "$EDGEVISOR_LOG_DIR/baseline_cuda_single.log"
```

### 5.3 非均匀 TP：`2:3:3`

```bash
./dllama inference \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --model "$EDGEVISOR_MODEL3" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps "$EDGEVISOR_STEPS" \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --workers $EDGEVISOR_WORKERS \
  --ratios "$EDGEVISOR_TP_RATIOS" \
  --benchmark \
  | tee "$EDGEVISOR_LOG_DIR/static_tp_2_3_3.log"
```

### 5.4 三 stage PP

```bash
./dllama inference \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --model "$EDGEVISOR_MODEL3" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps "$EDGEVISOR_STEPS" \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --workers $EDGEVISOR_WORKERS \
  --ratios "$EDGEVISOR_PP_RATIOS" \
  --benchmark \
  | tee "$EDGEVISOR_LOG_DIR/static_pp_three_stage.log"
```

### 5.5 TP+PP

```bash
./dllama inference \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --model "$EDGEVISOR_MODEL3" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps "$EDGEVISOR_STEPS" \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --workers $EDGEVISOR_WORKERS \
  --ratios "$EDGEVISOR_TPPP_RATIOS" \
  --benchmark \
  | tee "$EDGEVISOR_LOG_DIR/static_tppp.log"
```

### 5.6 分布式 top-1 对照

把分布式日志与 baseline 对照：

```bash
python3 - "$EDGEVISOR_LOG_DIR/baseline_cuda_single.log" "$EDGEVISOR_LOG_DIR/static_tp_2_3_3.log" <<'PY'
import sys
from pathlib import Path

def pred_tokens(path):
    return [
        line.split(" | ", 1)[1]
        for line in Path(path).read_text(errors="ignore").splitlines()
        if "🔶 Pred" in line and " | " in line
    ]

a = pred_tokens(sys.argv[1])
b = pred_tokens(sys.argv[2])
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        print("first_diff", i, repr(x), repr(y))
        sys.exit(1)
if len(a) != len(b):
    print("length_mismatch", len(a), len(b))
    sys.exit(1)
print("match", len(a))
PY
```

对 `static_pp_three_stage.log` 和 `static_tppp.log` 重复同样比较。

通过标准：

- 输出 `match <n>`。
- `<n> >= 64`。

---

## 6. PR11 动态 Head/FFN 重分片手工测试

### 6.1 启动动态 root

先启动 workers，方式同 PR10。

root 启动：

```bash
rm -f "$EDGEVISOR_PLAN_SOCKET"

DLLAMA_PLAN_CTRL_SOCKET="$EDGEVISOR_PLAN_SOCKET" \
./dllama inference \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --model "$EDGEVISOR_MODEL3" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps "$EDGEVISOR_STEPS" \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --workers $EDGEVISOR_WORKERS \
  --ratios "$EDGEVISOR_TP_RATIOS" \
  --enable-plan-barrier \
  --benchmark \
  2>&1 | tee "$EDGEVISOR_LOG_DIR/dynamic_head_ffn_root.log"
```

### 6.2 UDS 连通性

另开一个终端：

```bash
python3 examples/plan-uds-client.py "$EDGEVISOR_PLAN_SOCKET" ping
python3 examples/plan-uds-client.py "$EDGEVISOR_PLAN_SOCKET" status
```

通过标准：

- `ping` 返回 `ok:true`。
- `status` 能看到 position/batchSize 或 plan 状态。

### 6.3 Head 迁移

```bash
python3 examples/plan-uds-client.py "$EDGEVISOR_PLAN_SOCKET" set_plan \
  --seq 1 \
  --mode next_barrier \
  --stage 0 \
  --from 0 \
  --to 1 \
  --kind 1 \
  --heads 1 \
  --ffn 0
```

### 6.4 FFN 迁移

```bash
python3 examples/plan-uds-client.py "$EDGEVISOR_PLAN_SOCKET" set_plan \
  --seq 2 \
  --mode next_barrier \
  --stage 0 \
  --from 0 \
  --to 1 \
  --kind 2 \
  --heads 0 \
  --ffn 256
```

### 6.5 Head+FFN 联合迁移

```bash
python3 examples/plan-uds-client.py "$EDGEVISOR_PLAN_SOCKET" set_plan \
  --seq 3 \
  --mode next_barrier \
  --stage 0 \
  --from 0 \
  --to 1 \
  --kind 3 \
  --heads 1 \
  --ffn 256
```

### 6.6 多 move v2 测试

如果 stage 内有 3 个 node，可以测试多 move：

```bash
python3 examples/plan-uds-client.py "$EDGEVISOR_PLAN_SOCKET" set_plan \
  --seq 4 \
  --mode next_barrier \
  --stage 0 \
  --move 0,1,3,1,256 \
  --move 2,1,3,1,256
```

### 6.7 PR11 日志判定

检查 root 和 worker 日志：

```bash
grep -E "plan\\]|epoch|emit|apply|range|NaN|Inf|Critical|deadlock|duplicate" \
  "$EDGEVISOR_LOG_DIR/dynamic_head_ffn_root.log"
```

通过标准：

- 每条命令只 apply 一次。
- 日志包含 `[plan][emit][cuda]`。
- 日志包含 `[plan][apply][cuda]`。
- epoch 递增。
- 无越界错误。
- 无 NaN/Inf。
- 推理继续生成 token。

失败判定：

- 同一个 `seq` 被重复 apply。
- 迁移范围非法但 plan 被部分更新。
- 迁移后出现 NaN/Inf。
- 迁移后 deadlock。

---

## 7. PR12 PP layer migration、KV 传输和 Bubble Shadow

### 7.1 启动 PP migration root

先启动 workers，方式同 PR10。

root：

```bash
rm -f "$EDGEVISOR_PLAN_SOCKET"

DLLAMA_PLAN_CTRL_SOCKET="$EDGEVISOR_PLAN_SOCKET" \
DLLAMA_BUBBLE_SHADOW_KV=1 \
DLLAMA_BUBBLE_SHADOW_KV_LOG=1 \
./dllama inference \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --model "$EDGEVISOR_MODEL3" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps "$EDGEVISOR_STEPS" \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --workers $EDGEVISOR_WORKERS \
  --ratios "$EDGEVISOR_PP_RATIOS" \
  --enable-plan-barrier \
  --enable-pp-migration \
  --enable-kv-aggregate \
  --runtime-active-seg-enabled \
  --runtime-redundant-seg-enabled \
  --benchmark \
  2>&1 | tee "$EDGEVISOR_LOG_DIR/pp_migration_bubble_root.log"
```

### 7.2 下发 PP migration

另开终端：

```bash
python3 examples/plan-uds-client.py "$EDGEVISOR_PLAN_SOCKET" ping

python3 examples/plan-uds-client.py "$EDGEVISOR_PLAN_SOCKET" set_pp_migration \
  --seq 10 \
  --mode exact \
  --from 0 \
  --to 1 \
  --layer-count 2 \
  --trigger-pos 32
```

如果你需要固定迁移层列表：

```bash
export DLLAMA_MIGRATION_LAYER_LIST=13,12,11,10
```

然后重新启动 root。

### 7.3 runtime primary/redundant gate

在推理过程中切换 gate：

```bash
python3 examples/plan-uds-client.py "$EDGEVISOR_PLAN_SOCKET" set_runtime_gate \
  --primary 1 \
  --redundant 1

python3 examples/plan-uds-client.py "$EDGEVISOR_PLAN_SOCKET" set_runtime_gate \
  --primary 1 \
  --redundant 0
```

通过标准：

- 推理不中断。
- 日志显示 gate 状态更新。
- 无重复执行导致的 logits 异常。

### 7.4 Bubble Shadow 同步模式

异步模式默认：

```bash
export DLLAMA_BUBBLE_SHADOW_KV=1
unset DLLAMA_BUBBLE_SHADOW_KV_ASYNC
```

同步模式：

```bash
export DLLAMA_BUBBLE_SHADOW_KV=1
export DLLAMA_BUBBLE_SHADOW_KV_ASYNC=0
```

分别重新跑 7.1 和 7.2。

通过标准：

```bash
grep -E "bubble-shadow-kv|kv-export|kv-write|KV|ack|switch|NaN|Inf|Critical" \
  "$EDGEVISOR_LOG_DIR/pp_migration_bubble_root.log"
```

预期：

- 有 `[bubble-shadow-kv]` 日志。
- 有 KV export/apply/ack/switch 相关日志。
- 无 NaN/Inf。
- 无 race、deadlock、重复 ownership switch。

---

## 8. CPU/CUDA 混合执行测试

单元测试已经覆盖 CUDA -> host sync -> CPU segment 消费：

```bash
./nn-cuda-test --gpu-index "$EDGEVISOR_GPU_INDEX" \
  | tee "$EDGEVISOR_LOG_DIR/mixed_unit.log"
```

关键日志：

```text
CUDA PR10 segment output visible before host sync: ok
CUDA PR10 CPU/CUDA mixed segment output: ok
```

模型级混合执行可使用 `--gpu-segments`。示例：

```bash
./dllama inference \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --gpu-segments 1:20 \
  --model "$EDGEVISOR_MODEL3" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps "$EDGEVISOR_STEPS" \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --benchmark \
  | tee "$EDGEVISOR_LOG_DIR/model_cpu_cuda_mixed.log"
```

通过标准：

- 日志里能看到 CUDA device 初始化。
- CPU/CUDA 分段执行完成。
- logits 无 NaN/Inf。
- 与纯 CPU top-1 序列一致。

---

## 9. Vulkan 回归

如果机器有 Vulkan SDK 和 Vulkan runtime：

```bash
make clean
make -j"$EDGEVISOR_MAKE_JOBS" DLLAMA_VULKAN=1 dllama nn-vulkan-test

./nn-vulkan-test | tee "$EDGEVISOR_LOG_DIR/nn_vulkan_test.log"
```

通过标准：

- Vulkan build 成功。
- `nn-vulkan-test` exit code 为 0。
- 现有 Vulkan 静态 TP/PP 脚本继续通过。

如果看到：

```text
fatal error: vulkan/vulkan.hpp: No such file or directory
```

说明当前机器缺少 Vulkan SDK/header。该情况不能判定 Vulkan 回归通过，需要换有 Vulkan SDK 的环境重跑。

---

## 10. Qwen3-MoE 可选模型级验证

如果设置了 `EDGEVISOR_MOE_MODEL`：

```bash
./dllama inference \
  --backend cuda \
  --gpu-index "$EDGEVISOR_GPU_INDEX" \
  --model "$EDGEVISOR_MOE_MODEL" \
  --tokenizer "$EDGEVISOR_TOKENIZER" \
  --buffer-float-type "$EDGEVISOR_BUFFER_FLOAT_TYPE" \
  --temperature 0 \
  --seed 1 \
  --steps 48 \
  --max-seq-len "$EDGEVISOR_MAX_SEQ_LEN" \
  --nthreads "$EDGEVISOR_NTHREADS" \
  --prompt "$EDGEVISOR_PROMPT" \
  --benchmark \
  | tee "$EDGEVISOR_LOG_DIR/model_cuda_moe.log"
```

通过标准：

- 至少生成 32 个 prediction token。
- logits 无 NaN/Inf。
- MoE gate active expert 集合与 CPU/Vulkan deterministic 结果一致。
- 如果 30B A3B 模型无法单卡装入，应改用多 GPU 测试；单 T4 不作为 30B A3B 单卡门禁。

---

## 11. 性能记录

每次模型运行后提取性能：

```bash
grep -E "Evaluation|Prediction|tokens/s|Stage/Node Profile|per-fwd|per-tok|bubble" \
  "$EDGEVISOR_LOG_DIR"/*.log
```

记录字段：

- Evaluation tokens/s
- Prediction tokens/s
- Prediction root wall-clock tokens/s
- Stage/Node per-fwd total
- Stage/Node per-tok total
- sync time
- bubble time
- network sent/recv
- 显存占用或 required memory

性能暂不作为 PR12 正确性合入门禁，但需要保留报告，便于后续性能优化对照。

---

## 12. 常见失败与处理

### 12.1 CUDA backend 没有启用

现象：

```text
--backend cuda requested, but this build was not compiled with DLLAMA_CUDA=1
```

处理：

```bash
make clean
make -j"$EDGEVISOR_MAKE_JOBS" DLLAMA_CUDA=1 dllama nn-cuda-test
```

### 12.2 只传 `--gpu-index` 导致走 Vulkan

兼容规则是：

- 未指定 `--backend` 且无 `--gpu-index`：CPU。
- 未指定 `--backend` 但指定 `--gpu-index`：旧行为，选择 Vulkan。
- 指定 `--backend cuda`：才选择 CUDA。

所以 CUDA 测试必须显式写：

```bash
--backend cuda --gpu-index "$EDGEVISOR_GPU_INDEX"
```

### 12.3 Jetson / Vulkan OOM

如果在 Jetson 上看到：

```text
vk::Device::allocateMemory: ErrorOutOfDeviceMemory
NvMapMemAllocInternalTagged ... error 12
```

通常不是 CUDA PR10–PR12 的问题，而是 Vulkan backend 在统一内存/CMA/大连续块分配上失败。确认命令是否漏了 `--backend cuda`。

如果确实要测 Vulkan：

- 降低模型大小或 max seq len。
- 释放其他占用显存/统一内存的进程。
- 检查 `tegrastats`、`/proc/meminfo` 中 CMA/LFB。

### 12.4 `This configuration supports max 1 threads`

处理：

```bash
export EDGEVISOR_NTHREADS=1
```

并在命令中使用：

```bash
--nthreads "$EDGEVISOR_NTHREADS"
```

### 12.5 PP ratios 层数不匹配

如果 `--ratios "1@8*1@10*1@10"` 用在 32 层模型上，会因为 8+10+10=28 不匹配。按模型层数调整：

```bash
export EDGEVISOR_PP_RATIOS="1@8*1@12*1@12"
```

### 12.6 worker 一直等待或 root 连接失败

检查：

- root 的 `EDGEVISOR_WORKERS` 是否包含所有 worker。
- worker 端口是否一致。
- 防火墙是否开放。
- worker 是否先启动。
- root/worker 是否都使用同一分支、同一 binary、同一 CUDA 构建。
- worker 是否显式传了 `--backend cuda`。

---

## 13. 最终验收表

手工测试完成后建议填写：

| 层级 | 项目 | 命令/日志 | 结果 |
|---|---|---|---|
| L0 | CPU build | `01_cpu_build.log` | Pass/Fail |
| L0 | CPU tests | `nn_cpu_test.log`, `nn_cpu_ops_test.log`, `tokenizer_test.log`, `nn_slice_test.log` | Pass/Fail |
| L2 | CUDA build | `07_cuda_build.log` | Pass/Fail |
| L2 | CUDA unit | `nn_cuda_test.log` | Pass/Fail |
| L6 | Compute Sanitizer memcheck | `compute_sanitizer_memcheck.log` | Pass/Fail |
| L6 | Compute Sanitizer racecheck | `compute_sanitizer_racecheck.log` | Pass/Fail |
| L3 | 单 GPU CUDA 模型 | `model_cuda_single.log` | Pass/Fail |
| L3 | CPU/CUDA top-1 对照 | `model_cpu_top1.log`, `model_cuda_top1.log` | Pass/Fail |
| L4 | 非均匀 TP 2:3:3 | `static_tp_2_3_3.log` | Pass/Fail |
| L4 | 三 stage PP | `static_pp_three_stage.log` | Pass/Fail |
| L4 | TP+PP | `static_tppp.log` | Pass/Fail |
| L5 | Head migration | `dynamic_head_ffn_root.log` | Pass/Fail |
| L5 | FFN migration | `dynamic_head_ffn_root.log` | Pass/Fail |
| L5 | PP layer migration | `pp_migration_bubble_root.log` | Pass/Fail |
| L5 | Bubble Shadow async | `pp_migration_bubble_root.log` | Pass/Fail |
| L5 | Bubble Shadow sync | `pp_migration_bubble_root.log` | Pass/Fail |
| Vulkan | Vulkan build/test | `nn_vulkan_test.log` | Pass/Fail/EnvSkip |

最终可以判为完整通过的条件：

- L0 全部 Pass。
- L2 全部 Pass。
- L6 memcheck/racecheck 全部 Pass。
- L3 单 GPU模型无 NaN/Inf，至少 64 prediction tokens。
- L3 CPU/CUDA deterministic top-1 一致。
- L4 三项静态分布式均与单 GPU CUDA top-1 一致。
- L5 动态迁移、KV transfer、Bubble Shadow 均无 NaN/Inf、无死锁、无重复应用。
- Vulkan 在有 Vulkan 环境的机器上回归通过。

