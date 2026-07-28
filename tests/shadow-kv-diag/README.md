# Shadow KV (Bubble Shadow KV) 数值诊断工具

本目录是 Bubble Shadow KV 数值正确性诊断用的临时测试设施（诊断阶段产物，非回归测试）。

## 组成

- `shadow_kv_case.sh <tag> <2pp|3pp> <async|sync|off> <nBatches> [steps]`
  在单机起 root + worker(s)（CPU backend），跑 2 级或 3 级 PP 推理，
  设置 `DLLAMA_DUMP_KV_DIR` 让每个节点在每次 forward 后导出所有
  `block_shift_k/block_shift_v`（含主路径 att、takeover、shadow-kv 段）写入的
  KV cache 行（`.f32` + `.meta`），落在 `<logdir>/dump/`。
  环境变量 `DLLAMA_NBATCHES` 控制 prefill 的 batchSize。
  额外环境变量（如诊断开关 `DLLAMA_SHADOW_RIGHT_PP_CACHE=1`）会透传。
- `compare_kv_dump.py <dump_dir>`
  对每个 shadow（`red`）dump，按 (layer, kind, batch, pos) 在**另一个节点**上找
  同一层的 `main` dump（该层在相邻 stage 上是 active 层，其 K/V 为参考真值），
  输出 max/mean abs diff 和 best-matching position（用于识别"上一 token 陈旧输入"）。
- `shadow_kv_e2e.sh <tag> <baseline|transfer|recompute|shadow> [triggerPos] [layerCount] [prompt] [steps]`
  端到端迁移一致性测试：2 级 PP，经 UDS `set_pp_migration`（exact 模式）在
  指定 position 把 stage1 的边界层迁到 stage0，对比四种恢复路径的生成 token：
  - `baseline`：不迁移；
  - `transfer`：`EDGEVISOR_SHADOW_KV_MODE=disabled_transfer`（真实 KV 传输）；
  - `recompute`：`EDGEVISOR_SHADOW_KV_MODE=disabled_recompute`（重算，ground truth）；
  - `shadow`：`DLLAMA_BUBBLE_SHADOW_KV=1` + `EDGEVISOR_SHADOW_KV_MODE=enabled`（消费 red_k/red_v）。

## 运行前提

- yhs1 上运行（CPU only，`--backend cpu`，构建时 `make DLLAMA_VULKAN= DLLAMA_CUDA=`）。
- 模型/tokenizer 路径硬编码为 `/home/byh/B01/models/...`，按需修改脚本头部。
- 日志落在 `~/B01/EdgeVisor/runtime_logs/shadow_kv_diag/<tag>/`。

## 关键结论（2026-07-27，修复后 two-level-slack @ 209fddb+）

### 诊断阶段结论（修复前）

- sync 模式 + 诊断开关（右边界 shadow 改从 pp_stage_out 快照读输入）：layer 14 的
  red_k/red_v 与参考值**逐位相等**。
- 修复前默认路径（右边界 `OP_MERGE_ADD zqPipe→xBuffer`）：全部 position 数值错误
  （drain 时 xBuffer 已被 pp_stage_merge 合并过，shadow 再合并一次 → double-add）。
- 左边界 shadow 输入（`OP_CAST xPipe→xBuffer`）拿到的是上游 stage **最终输出**
  （= 上一层 output），而 shadow 层需要的是上一层 **input**，天然错一层。
- async 模式下 shadow 在 sync 窗口内执行，时序竞争导致输入可能是上一 token 的
  陈旧值（实测 best-matching position 系统性偏移 1）。

### 更正（修复阶段发现）

- 诊断报告中的"主路径污染"归因**有误**：当时 bubble-on 与 bubble-off 的输出分叉，
  实际根因是 `--runtime-redundant-seg-enabled 1` 导致 bubble-off 运行时 takeover
  （redundant）段在 boot 即处于 enabled 并在主路径上执行了边界层（`app.cpp` 中
  `DLLAMA_RUNTIME_REDUNDANT_SEG_ENABLED = redundantSegEnabled && !bubbleShadowKvEnabled()`），
  而非 shadow 改写共享 buffer。修复后干净对照（off + `--runtime-redundant-seg-enabled 0`
  vs bubble-on）：生成 token 逐 token 一致、主路径 K/V 逐位为 0。
  shadow 共享 buffer 属潜在隐患（已由私有 buffer 修复），但无未被混淆的直接证据。
- 正确 baseline 必须用 `--runtime-redundant-seg-enabled 0`（redundant 段只应在迁移时
  由 layer-gate 使能）。本目录脚本已改为默认传 0。

### 修复内容（209fddb, 1695222, 及后续）

1. 右边界 shadow 输入默认改为 pp_stage_out 快照（`OP_CAST pp_stage_out → shadow_x`），
   删除 MERGE_ADD 输入 op（诊断开关 `DLLAMA_SHADOW_RIGHT_PP_CACHE` 已移除）。
2. shadow 段使用私有工作 buffer（shadow_x/inv_rms/y/q_y/k_temp/v_temp）。
3. 右边界 shadow 仅在主路径完成 pp_send 的 pp_stage_cache 之后执行
   （executor `segmentReadyAfterStep` 门控）。
4. 左边界 shadow 默认不构建；`DLLAMA_SHADOW_LEFT_ENABLE=1` 恢复旧行为。

### 修复后回归结果

- 右边界 red_k/red_v 与参考**逐位为 0**：2pp/3pp × async/sync × batchSize 1/2/4
  （首个边界层 L14（2pp）、L10/L19（3pp），48 个 position × 所有 batch 行）。
  第二边界层（L15/L11/L20）仍因已知结构问题（多层共用同一输入）不符——不在本次修复范围。
- 左边界默认无 red dump（未构建）；`DLLAMA_SHADOW_LEFT_ENABLE=1` 后恢复构建。
- 主路径无污染：31-token prefill 记忆型 prompt，bubble-on 与干净 baseline
  token 逐 token 一致，主路径 K/V 逐位为 0。
- e2e 迁移（layer 14 → stage0，pos 34，消费右边界 red）：shadow / recompute /
  baseline 三条路径生成 token 完全一致。

详见主代理报告。日志：`~/B01/EdgeVisor/runtime_logs/shadow_kv_diag/`。

## L2：tool-wait 窗口的 Shadow KV 补算（DLLAMA_SHADOW_L2）

L1 修复后，右边界 shadow 只在 pp_send 快照（pp_stage_cache）之后的 sync 窗口
执行，算不完的部分原本全部走 forward 末尾 drain（关键路径）。L2 把 drain 挪到
tool-wait 空闲窗口：forward 末尾没算完的 shadow 不再 drain，而是把该次 forward
的输入（pp_stage_out 快照 + POS/SLT + batchSize + 进度 cursor）作为一笔**债务**
存进 stash；tool-wait 窗口内逐笔**补算**（恢复输入后跑剩余 shadow steps）。

### 开关与配置

- `DLLAMA_SHADOW_L2=1`（默认关；关时行为与 L1 完全一致：bubble + drain）。
  经 bootstrap（`LLM_BOOTSTRAP_ENABLE_SHADOW_L2`）自动下发 worker。
- `DLLAMA_SHADOW_L2_STASH_MB`（默认 512）：stash 字节上限；超限在 forward 末尾
  强制 drain 最老 entry（内存有界兜底，会打 `⚠️ [shadow-l2] stash cap exceeded`）。
- `DLLAMA_SHADOW_L1_DISABLE=1`（**仅测试旋钮**）：禁用 L1 bubble 窗口，让全部
  shadow 工作变成债务，用于验证补算链路。

### UDS ops（plan-ctrl-socket）

```bash
python3 examples/plan-uds-client.py $SOCK tool_window_begin   # 广播 LLM_CTRL_SHADOW_CATCHUP 给所有 worker（fire-and-forget），root 后台线程补算本地债务
python3 examples/plan-uds-client.py $SOCK tool_window_end     # root 侧 stop+join（下一次 forward 被调用时也会自动 stop+join）
python3 examples/plan-uds-client.py $SOCK shadow_debt         # 查询 root 债务/补算统计（worker 侧经 perf 包 bubbleStashEntries/bubbleCatchupEntries/bubbleCatchupUs 上报）
```

worker 在主循环 CONTROL_ONLY 分支收到 catch-up 包后执行补算；entry 边界对
root socket 做非阻塞 peek，发现新控制包立即中断回主循环（无 ACK、不会阻塞推理）。

### 验证命令（CPU）

```bash
# 单测场景（chat 模式构造真实 tool-wait 窗口；L1 禁用旋钮强制产生债务）
DLLAMA_SHADOW_L1_DISABLE=1 bash run_l2_case.sh l2_unit_l1off
# 中断测试（catch-up 中途来新请求，推理应立即继续）
bash run_l2_interrupt.sh l2_interrupt
# L2 开启回归（正常 L1 窗口；债务≈0，数值仍逐位为 0）
bash run_l2_case.sh l2_reg_b1 1 48
bash run_l2_case.sh l2_reg_b2 2 48
bash run_l2_case.sh l2_reg_b4 4 48
# 关键路径对照：L2 关 + L1 禁用（全部 drain）vs L2 开 + L1 禁用（debt stash）
DLLAMA_SHADOW_L1_DISABLE=1 bash run_case.sh l2off_l1disable_2pp 2pp async 1 48
DLLAMA_SHADOW_L2=1 DLLAMA_SHADOW_L1_DISABLE=1 bash run_case.sh l2on_l1disable_2pp 2pp async 1 48
```

### 验证结果（2026-07-27，two-level-slack）

- l2_unit_l1off：tool window 前 `debtEntries=60 debtBytes=1008272`；
  `tool_window_begin` 后 `debtEntries=0 catchupEntries=60 catchupUs=420667`；
  补算出的 L14 red_k/red_v 与参考值**逐位为 0**（含 prefill batch=23 与全部
  decode position）。
- 中断：见 `l2_interrupt/summary.txt`（中断→首笔新债务的 resume latency；
  数值见下方 2026-07-28 补跑）。
- L2 关等价性：`l2off_2pp_async_b1`/`l2off_2pp_sync_b1` 与 L1 修复后结果一致
  （L14 全 0，L15 为已知结构问题）。

### 验证结果（2026-07-28 补跑，two-level-slack @ 58ec57b）

- L2 开启回归（正常 L1 窗口，steps=48）：`l2_reg_b1`/`l2_reg_b2`/`l2_reg_b4`
  （batchSize 1/2/4）三者 debtEntries 均为 0（L1 窗口全部覆盖，无债务产生），
  L14 red_k/red_v 与参考值**逐位为 0**（各 164 对 = 82 position × k/v），
  L15 仍为已知结构问题（328 对 mismatch，不在本次范围）。
- 中断（`l2_interrupt`，L1 禁用）：turn 1 结束 debt=60；`tool_window_begin`
  后补算约 0.3s 内完成（catchupEntries=60），新请求到达时补算已结束
  （本轮未压到"中途打断"路径）；interrupt→首笔新债务 resume latency=15s
  （含 ~8s prefill）；turn 2 新增 debt=60，第二个 window 后 debt=0、
  累计 catchupEntries=120。详见 `l2_interrupt/summary.txt`。
- 中途打断补跑（`l2_interrupt_mid`，window_begin 后 0.15s 注入新请求）：
  补算在中途被打断（60 笔已补 42 笔），推理立即继续；turn 2 结束
  debt=78（18 遗留 + 60 新增）、catchupEntries=42；第二个 window 后
  debt=0、累计 catchupEntries=120，无债务丢失。
  注：脚本的 resume latency 用 `debtEntries>0` 判定，有遗留债务时读数
  恒为 0s，不能区分新旧债务，该指标仅在补算先于中断完成时（如
  `l2_interrupt` 的 15s）有意义。
- 关键路径对照（2pp async，L1 禁用强制全部 shadow 工作落关键路径或 stash）：
  - L2 关（`l2off_l1disable_2pp`）：root `bubbleDrain/fwd=2.78ms`，
    `complete=48/48`（每次 forward 末尾 drain，在关键路径上）；
    decode 733.4 ms/tok。
  - L2 开（`l2on_l1disable_2pp`）：root `bubbleDrain/fwd=0.00ms`
    （`segments=0 ops=0`，全部 stash 为债务，drain 移出关键路径）；
    decode 754.7 ms/tok（与 L2 关的差异在 rootWait 抖动量级，
    单机 CPU pp 同步等待主导端到端耗时）。
- L2 关默认回归：`l2off_2pp_async_b1`/`l2off_2pp_sync_b1` L14 red_k/red_v
  逐位为 0（各 96 对 = 48 position × k/v），与 L1 修复后结果一致；
  L15 为已知结构问题。

详见主代理报告。日志：`~/B01/EdgeVisor/runtime_logs/shadow_kv_diag/`。

### L2 竞态修复与复测（two-level-slack @ 8d9730c）

- 竞态（复现于 `l2dbg_v3`–`l2dbg_v9`）：root 的 catch-up 线程被下一次 forward
  中途 join 时，其收尾的 `batchSize = savedBatchSize` 可能落在 handler 已为**新**
  forward 设好 batchSize 之后 → executor 用陈旧 batchSize=1 跑 batch=23 的
  prefill（pp_send 只发 1 行，控制包声明 23 行）→ 管线死锁（worker 等 xPipe
  剩余 22 行，root 等 logits gather）。定位证据：gdb backtrace（root 卡在
  `NnNetworkNodeSynchronizer::sync → syncNodeSlices → readMany`）、双侧 strace
  （root 只 `sendto` 了 12288B=1 行）。
- 修复（`8d9730c`）：`RootLlmInference::forward()` 在 join catch-up 线程后重新
  断言 `execution->batchSize/position` 与 POS/SLT 管道内容。
- 复测：
  - `l2_int_fixed`（原挂死场景：大债务 110 entries、0.15s 后新请求打断）：
    catch-up 补算 2/110 后停止，resume latency 0s，turn 2 正常跑完（pos 463→511），
    不再死锁。
  - `l2_unit_fixed`（L1 禁用）：debt 60 → tool window 后 debt 0、
    catchupEntries=60；L14 red_k/red_v 逐位为 0。
  - `l2fix_regress_2pp`（L2 关）与 `l2fix_reg_b2`（L2 开 NB=2）：L14 全部逐位为 0。

### 基于 8d9730c（+0b5106c）的中断复测

- `l2_interrupt_racefix`（@ 8d9730c 重建 CPU 二进制）：debt=110 的窗口中
  catch-up 被新请求打断（补算 2/110 即停），resume latency 0s，turn 2 正常跑完
  （pos 463→511，Network closed），无 hang；债务无丢失（剩余 108 由后续窗口
  偿还，机制同 `l2_unit_fixed` 的 debt 60→0、补算 L14 逐位为 0）。
- `l2dbg_racefix`（hang repro，gdb/strace harness）：turn 2 越过原死锁点
  （pos 462）跑到 pos=511 正常结束。
- `l2fix2_reg_b1`（@ 0b5106c）：L14 red_k/red_v 逐位为 0（164 对），L15 仍为
  已知结构问题 —— 设备线程修复无回归。
- `0b5106c`（上 GPU 前的高危修复）：bubble shadow 线程与 L2 catch-up 线程入口
  调 `NnDevice::setCurrentThreadDevice()`（CPU 空实现；CUDA 为
  `cudaSetDevice(gpuIndex)`），避免 gpuIndex≠0 时 kernel launch 落到错误
  context。CPU 构建无 CUDA 依赖；`nn-cuda.cu` 已过 nvcc 12.8 编译检查（未上机跑）。

## GPU 冒烟（4×Tesla T4，CUDA backend，two-level-slack @ c61265d+）

### 构建

```bash
cd ~/B01/EdgeVisor/EdgeVisor && make clean && make DLLAMA_VULKAN= DLLAMA_CUDA=1 -j8 dllama dllama-api
```

`CUDA_ARCHS=auto` 自动探测为 75（sm_75，覆盖 T4）。注意：此前用 CPU-only flag
构建过的话必须 `make clean`——Makefile 不跟踪 flag 变化，旧 .o（无
-DDLLAMA_CUDA）会被误复用导致 `--backend cuda` 报 "not compiled"。
worker 进程也要显式 `--backend cuda`（BACKEND_AUTO 会落到 vulkan）。

### 冒烟命令与结果（日志 `~/B01/EdgeVisor/runtime_logs/gpu_smoke/`）

测试时 4 卡被他人 vLLM 任务占用（每卡 ~11.5/15.3GB），以下均在每卡
~1.3-2GB 的小余量内完成（3B q40，PP 分片）。

- 小规模（2 节点 PP，gpu0+gpu1）：`small_root.log`。RC=0，28.3 tok/s，
  无 CUDA error。
- (a) bubble 窗口执行（`run_gpu_l2_interrupt.sh gpu4_interrupt`）：
  `segments=2 attn=2 layers=2 ops=18 completed=1 drain_us=0`（每个 forward
  两个右边界 shadow 段全部在 L1 窗口内完成）。
- (b) L2 债务/补算（`run_gpu_l2_smoke.sh gpu4_bubble bubble`，L1-disable
  强制债务）：`debtEntries=60 debtBytes=1008272` → `tool_window_begin` →
  `debtEntries=0 catchupEntries=60 catchupUs=46106`。root(gpu0) 46ms、
  worker1(gpu1, inline catch-up) 46ms、worker2(gpu2) 558ms 各自完成 60 笔，
  证明 `setCurrentThreadDevice` 在多 gpu-index 下工作正常。
- (c) 中断：tool window 中插入新一轮对话，turn 2 正常完成，无 hang
  （`interrupt_latency.txt`，数值为正常生成耗时而非阻塞）。
- (d) e2e 数值（`run_gpu_e2e_consistency.sh`）：bubble+L2 开 vs 全关，同
  prompt/seed 生成 token 逐 token 一致（双方都于 pos=31 停）：
  `23:1 24:, 25:  26:2 27:, 28:  29:3 30:, 31:`。
- 风险点：全部进程 `--nthreads 1`（GPU 下 bubble shadow 的硬门控）；stash
  D2H 开销 60 笔 46ms（GPU 上可忽略）；无任何 CUDA error / 异常吞没迹象。

## 4×T4 batchsize 矩阵（LSS 开启，two-level-slack @ 69b4b5d）

拓扑 `1@7*1@7*1@7*1@7`（4 节点 PP，每节点一卡），3B q40，`--last-stage-sampling`
开启，全程 `--nthreads 1`。测试期间 4 卡与他人 vLLM 任务共享（~90% util，
余量 ~3.8GB/卡），端到端 tok/s 噪声大，仅作参考；设计故事以
bubbleDrain/fwd 与债务统计为准。日志 `~/B01/EdgeVisor/runtime_logs/gpu_batch_matrix/`。

### LSS 兼容性结论

LSS 在与本拓扑（单节点 stage × 4）组合下原本**完全不可用**，发现并修复三个
pre-existing bug：

1. `27ce884`：LSS 采样器 vocabSize 误按全部节点求和（单节点 stage 时 =
   nStages × vocabSize），argmax 越界读 logits pipe → 产生越 vocab 的 token
   id → root tokenizer decode assert 崩溃。修为只按末 stage 节点求和。
2. `9bd0f8c`：末 stage 对每个 batchSize==1 的 forward 都发 sampled-token 包，
   而 root 只在 decode 阶段消费；prefill 批为 1（DLLAMA_NBATCHES=1 或尾部
   chunk）时包积压错位 perf/token 序列 → root 读到 magic=0 的错位数据，全部
   回落到全零 logits pipe 采样（输出全 "!"）。修为 inference/API prefill 循环
   对 batchSize==1 的 forward 同步 drain 丢弃该包。
3. `69b4b5d`：chat 模式 prefill 循环同样的漏 drain，补齐。

修复后 12 个矩阵 case（batch 1/2/4/8 × off/L1/L2）全部 LSS 正常工作，
生成内容与无 LSS 时一致。

### 矩阵结果（per-forward 均值，ms；bubbleDrain/fwd 括注）

Stage 0（root）：

| batch | off | L1（drain） | L2（drain） |
|---|---|---|---|
| 1 | 30.01 | 50.53（0.85） | 48.84（0.00） |
| 2 | 54.66 | 47.03（0.99） | 50.76（0.00） |
| 4 | 58.92 | 60.76（0.66） | 59.72（0.00） |
| 8 | 70.02 | 62.65（1.23） | 90.81（0.00） |

Stage 1（含右边界 shadow 的中间 stage，故事主战场）：

| batch | off | L1（drain） | L2（drain） |
|---|---|---|---|
| 1 | 87.57 | 87.36（0.75） | 91.92（0.00） |
| 2 | 118.33 | 86.34（0.95） | 83.29（0.00） |
| 4 | 97.27 | 114.14（0.60） | 99.40（0.00） |
| 8 | 125.42 | 183.80（2.15） | 161.02（0.00） |

- L1 bubble 完成率 100%（各 case complete=fwd 数），但 drain 落在关键路径，
  且随 batch 增大总体变大（stage 1：0.75→2.15 ms/fwd；b8 时 stage1 比 off
  慢 47%）。
- L2 全 case `bubbleDrain/fwd=0.00`（drain 移出关键路径进 stash）；
  b8 时 stage1 比 L1 快 12%（161.0 vs 183.8 ms/fwd）。
- 端到端 wall tok/s 因共享 GPU 噪声过大（±50%），不采信。

### L2 补算验证（L1-disable 强制债务，UDS tool_window_begin）

| batch | 窗口前债务 | 窗口后 | catchupEntries | 补算耗时 |
|---|---|---|---|---|
| 1 | 82（1.0MB） | 0 | 82 | 585ms |
| 2 | 71 | 0 | 71 | 43ms |
| 4 | 65 | 0 | 65 | 21ms |
| 8 | 62 | 0 | 62 | 47ms |

### 一致性 guard

12 个矩阵 case（off/L1/L2 × batch 1/2/4/8）生成 token 的 md5 全部一致
（`b*/tokens.txt`），argmax 级无分叉。

### 复现命令

```bash
bash tests/shadow-kv-diag/run_matrix_case.sh b8_l2 8 l2       # 矩阵单 case（off|l1|l2）
bash tests/shadow-kv-diag/run_matrix_l2_catchup.sh b8_catchup 8  # L2 债务/补算
```

## 全量回归（two-level-slack @ 6930c22，默认配置 + 新开关组合）

构建：CPU-only（`make DLLAMA_VULKAN= DLLAMA_CUDA= -j8`）用于 CPU 用例；
Vulkan+CUDA 双后端（`make clean && make DLLAMA_VULKAN=1 DLLAMA_CUDA=1 -j8`）用于
GPU 用例（semantic GPU 脚本依赖 BACKEND_AUTO→vulkan；注意 Makefile 不跟踪 flag
变化，改 flag 必须 `make clean`）。main 对照用 `git worktree add /tmp/edgevisor_main main`。
模型 env 覆盖：`EDGEVISOR_MODEL3/EDGEVISOR_TOKENIZER` → `/home/byh/B01/models/`。
GPU 与 cyx vLLM 共享（~3.8GB/卡余量）。日志 `~/B01/EdgeVisor/runtime_logs/full_regression/`。

### 回归矩阵

| 用例 | 结果 | 判定 |
|---|---|---|
| scripts/semantic CPU：tp_static / even2_tp_static / uneven2_tp_static / uneven2_full / uneven2_linebuf / dynamic_heads | RC=0（6/8） | ✅ 无回归 |
| scripts/semantic CPU：even_tp_static | RC=134（`sliceKvCache: kvDim(1024)%3≠0` assert） | pre-existing（main 同败 RC=134；模型 kvDim 与 3 节点拓扑不匹配） |
| scripts/semantic CPU：uneven2_tp_static_f32 | RC=1（`Unsupported CPU op MATMUL F32_Q40_F32`） | pre-existing（main 同败 RC=1；F32 激活×Q40 权重 CPU 无内核，脚本需 f32 模型） |
| scripts/semantic GPU：tp_static / full_static / full_static_224 / dynamic_heads | RC=0（4/5） | ✅ 无回归 |
| scripts/semantic GPU：tp_static_f32 | RC=1（`Unsupported shader: MATMUL/F32_Q40_F32`） | pre-existing（main 同败 RC=1；vulkan 无该 shader，仓库从未存在） |
| scripts/gpu：pp_static / pp_migration / patch_regression / patch_gdb_dynamic | RC=0（4/4） | ✅ 无回归 |
| 组合：semantic GPU（2:3:3）+ `--last-stage-sampling` | RC=0，回答 "4"+EOS 正确 | ✅ |
| 组合：run_gpu_pp_migration + `DLLAMA_BUBBLE_SHADOW_KV=1` + `DLLAMA_SHADOW_L2=1` | RC=0 | ✅ |
| 组合：CB（`--continuous-batching --max-active-seqs 2`，CPU，默认） | RC=0，干净退出 | ✅ |
| 组合：CB + bubble + L2 | RC=0，0 错误行，bubbleDrain=0 | ✅ 不炸（CB 槽位级 shadow KV 语义未深验证，见遗留） |
| tests/semantic 六测 benchmark（缩减候选 EDGEVISOR_GPU_PP/HYBRID_RATIO_CANDIDATES） | 总体 RC=0（06_GPU_UNEVEN_DYNAMIC RC=0） | ✅；强制候选 "1@14*1@14" 因 harness 的 ratio 语法期望不同报 "Stage-weights segment must not specify layers"（脚本自动回退 2:3:2 完成；非产品回归） |

### pre-existing 问题（与 two-level-slack 无关）

1. `even_tp_static`：kvDim=1024 不可被 3 节点整除（even 路径硬 assert；main 同败）。
2. `uneven2_tp_static_f32`（CPU）与 `gpu_tp_static_f32`（Vulkan）：F32 激活 × Q40
   权重无内核/shader（main 同败）；这两个脚本需要 f32 模型才能跑。
3. Makefile 不跟踪编译 flag 变化：切换 `DLLAMA_VULKAN/DLLAMA_CUDA` 必须
   `make clean`，否则旧 .o 复用导致后端宏检查失败（多次踩坑，建议后续修）。
4. semantic GPU 脚本依赖 BACKEND_AUTO→vulkan；纯 CUDA 构建下全部报
   "not compiled with DLLAMA_VULKAN"（构建配置问题，非代码回归）。

### 未覆盖

- 六测 benchmark 全量候选（3 PP + 6 hybrid 拓扑）：共享 GPU 时间预算不足，
  以缩减候选跑通总体 RC=0（结果见 six_benchmark.out /
  benchmark_docs_20260728_171214/）。
- CB 槽位级 shadow KV 正确性（多槽 KV 与 red_k/red_v 的 slot 语义）：未深验证，
  建议单列一阶段。
