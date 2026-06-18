# Bubble Shadow KV 3 GPU PP 对照测试记录

## 测试目的

验证 `exp/bubble-shadow-kv` 分支中 Bubble Shadow KV 调度优化是否能在真实 3 GPU PP 场景下，把冗余 KV 相关计算更多放入 pipeline bubble / sync wait 窗口，减少对主路径的拖慢。

本次只使用真实 3 GPU PP 测试，不使用 8 节点虚拟拓扑。

## 代码版本

- 仓库：`/home/byh/B01/EdgeVisor`
- 分支：`exp/bubble-shadow-kv`
- 原始基线：`HEAD = 6a91433fe8dfc0b3a616b408381e7db8cb220370`
- 优化后源码改动文件：
  - `EdgeVisor/src/nn/nn-executor.cpp`
  - `EdgeVisor/src/nn/nn-executor.hpp`
  - `EdgeVisor/src/app.cpp`

## 构建说明

原仓库 `EdgeVisor/` 下已有部分 `.o/.d` 文件属于 `root:root`，当前 `byh` 用户无法覆盖，因此没有直接在原目录覆盖构建产物。为保证对照干净，使用两个临时构建目录：

- 原始基线构建目录：`/home/byh/B01/EdgeVisor_orig_build_tmp`
  - 来源：`git archive HEAD`
  - 构建命令：`bash build.sh`
- 优化版构建目录：`/home/byh/B01/EdgeVisor_bubble_build_tmp`
  - 来源：当前工作区源码，不包含 `.git/runtime_logs/artifacts/*.o/*.d`
  - 构建命令：`bash build.sh`

两套构建均通过，生成 `dllama` 和 `dllama-api`。

## 模型与拓扑

模型与 tokenizer：

```bash
EDGEVISOR_MODEL3=/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m
EDGEVISOR_TOKENIZER=/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t
```

3 GPU PP 拓扑：

```text
GPU0: root / stage 0 / layers 0-7
GPU1: worker / stage 1 / layers 8-17
GPU2: worker / stage 2 / layers 18-27
ratios: 1@8*1@10*1@10
```

统一推理参数：

```bash
--prompt "Hi"
--steps 16
--buffer-float-type q80
--nthreads 1
--max-seq-len 512
--gpu-index 0
--workers 127.0.0.1:PORT1 127.0.0.1:PORT2
--ratios "1@8*1@10*1@10"
--enable-plan-barrier
--enable-stage-full-weights
--enable-pp-migration
--runtime-redundant-boundary-layers 2
--runtime-active-seg-enabled 1
--runtime-redundant-seg-enabled 1
--enable-kv-redundancy-during-migration 1
--kv-redundancy 2
--benchmark
```

Bubble async 环境：

```bash
DLLAMA_BUBBLE_SHADOW_KV=1
DLLAMA_BUBBLE_SHADOW_KV_ASYNC=1
DLLAMA_BUBBLE_SHADOW_KV_LOG=1
```

## 临时测试脚本

本次使用的临时测试脚本：

```text
/tmp/run_edgevisor_bubble_compare_tmp.sh
```

该脚本只用于本次测试，没有纳入仓库。

## 对照结果

### Round 1

日志目录：

- 原始 async：`/home/byh/B01/EdgeVisor/runtime_logs/gpu/manual_bubble_verify_20260618_055321_orig_async`
- 新版 async：`/home/byh/B01/EdgeVisor/runtime_logs/gpu/manual_bubble_verify_20260618_055321_new_async`

结果：

| 版本 | Prediction tokens/s | Prediction ms/tok | 备注 |
|---|---:|---:|---|
| 原始 async | 8.94 | 111.80 | bubble shadow 仍有较明显 sync/关键路径压力 |
| 新版 async | 9.31 | 107.42 | 约 +4.1% |

关键 profile：

| 版本 | Node | per-fwd total | exec | sync | bubble |
|---|---:|---:|---:|---:|---:|
| 原始 async | 0 | 640.62 ms | 66.92 | 562.82 | 10.88 |
| 原始 async | 1 | 94.66 ms | 23.97 | 61.71 | 8.98 |
| 原始 async | 2 | 378.52 ms | 39.31 | 330.83 | 8.39 |
| 新版 async | 0 | 533.15 ms | 64.30 | 457.74 | 11.11 |
| 新版 async | 1 | 90.02 ms | 23.99 | 58.52 | 7.50 |
| 新版 async | 2 | 242.56 ms | 40.63 | 192.46 | 9.47 |

新版日志中 root 的 bubble 诊断显示大多数 token `completed=1 drain_us=0`，说明冗余段基本在 bubble/sync 窗口中完成；其中 pos=9 出现一次 `drain_us=29115`，表示该 token 有尾部补齐。

### Round 2

日志目录：

- 原始 async：`/home/byh/B01/EdgeVisor/runtime_logs/gpu/manual_bubble_verify_20260618_055401_orig_async`
- 新版 async：`/home/byh/B01/EdgeVisor/runtime_logs/gpu/manual_bubble_verify_20260618_055401_new_async`

结果：

| 版本 | Prediction tokens/s | Prediction ms/tok | 备注 |
|---|---:|---:|---|
| 原始 async | 18.48 | 54.11 | baseline |
| 新版 async | 22.97 | 43.53 | 约 +24.3% |

关键 profile：

| 版本 | Node | per-fwd total | exec | sync | bubble |
|---|---:|---:|---:|---:|---:|
| 原始 async | 0 | 504.06 ms | 24.29 | 475.41 | 4.36 |
| 原始 async | 1 | 50.33 ms | 22.43 | 20.01 | 7.89 |
| 原始 async | 2 | 213.88 ms | 23.81 | 181.32 | 8.74 |
| 新版 async | 0 | 501.93 ms | 20.65 | 477.00 | 4.28 |
| 新版 async | 1 | 38.96 ms | 14.73 | 17.54 | 6.68 |
| 新版 async | 2 | 143.28 ms | 22.27 | 112.69 | 8.32 |

新版日志中 root 每个 token 均显示 `completed=1 drain_us=0`，说明该轮中 root 侧冗余段全部在 bubble/sync 窗口内完成。

## 普通 PP Mainpath Smoke

为确认本次改动没有破坏普通 3 GPU PP 推理路径，优化版还跑了一轮关闭 Bubble Shadow KV 执行的 PP smoke。该测试不是“主路径上计算冗余 KV”的性能对照，只用于确认普通 PP mainpath 仍能正常运行。

日志目录：

```text
/home/byh/B01/EdgeVisor/runtime_logs/gpu/manual_bubble_verify_20260618_055436_new_mainpath
```

结果：

```text
Prediction tokens/s: 23.33
Prediction ms/tok: 42.87
bubbleSeg=0 bubbleOps=0
```

这里的 `bubbleSeg=0 bubbleOps=0` 表示该轮没有通过 bubble shadow executor 执行冗余段。日志中仍能看到 redundant segment 被构建、冗余权重被加载，这是因为运行参数保留了迁移/冗余能力；但在关闭 `DLLAMA_BUBBLE_SHADOW_KV` 后，这些 redundant segment 没有被调度执行。因此 `23.33 tokens/s` 只能作为普通 PP 功能回归参考，不能用来说明“mainpath 计算冗余更快”。

## 结论

本次优化后，Bubble Shadow KV 在真实 3 GPU PP 场景中可以看到正向收益：

- Round 1：`8.94 -> 9.31 tokens/s`，约 `+4.1%`
- Round 2：`18.48 -> 22.97 tokens/s`，约 `+24.3%`

优化后的关键变化是：

1. 不再让 shadow worker 跨越到下一段主路径计算。
2. 在每个 sync/bubble 窗口内分片调度 redundant segment。
3. 暂停粒度保持在完整 redundant segment 边界，避免中间 buffer 被主路径覆盖。
4. forward 末尾保留 drain 补齐，保证冗余 KV 状态完整，满足后续 heads/layers 迁移使用。

仍需注意：

- `bubble` profile 时间当前作为单独字段上报，Stage/Node summary 的 `total=exec+sync+bubble` 可能高估实际端到端关键路径，因为 async bubble 可能已经与 sync 重叠。
- 测试结果存在 GPU/系统波动，建议后续正式报告使用更多 repeat 并取均值/方差。
