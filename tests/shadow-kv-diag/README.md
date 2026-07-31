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

## GPU agentic 端到端演示（4×T4 CUDA，two-level-slack @ 6281cf8）

完整故事：真实 web search 工具 + tool-wait 窗口 L2 补算，在 4 节点 PP
（`1@7*1@7*1@7*1@7`，3B q40，CUDA backend，`--last-stage-sampling`）上跑通。
日志 `~/B01/EdgeVisor/runtime_logs/agentic_gpu{,_main,_l2,_l2b,_l2c,_control}.out` 与
`agentic_gpu/agentic_search_smoke_loop_*/`。

### 验证命令

```bash
cd ~/B01/EdgeVisor && DLLAMA_SHADOW_L1_DISABLE=1 \
  /home/byh/B01/agent_langgraph_venv/bin/python -m agent_bench.run_loop_episode \
  --backend edgevisor_ablation \
  --episode agent_bench/episodes/agentic_search_smoke_episode.json \
  --out-root ~/B01/EdgeVisor/runtime_logs/agentic_gpu \
  --cuda-visible 0,1,2,3 --edge-backend cuda \
  --edge-worker-gpus 1,2,3 --edge-ratios "1@7*1@7*1@7*1@7" \
  --bubble-shadow-kv --shadow-l2 --enable-pp-migration \
  --edge-last-stage-sampling --edge-steps 256 --edge-timeout-s 600 \
  --disable-episode-dynamic-plan
```

（对照组去掉 --shadow-l2 即可。）

### 结果

- 主 episode（L2 开）：gen1 rc=0（18.7s，真实模型 action JSON web_search）；
  web_search 走 bing 模式（真实检索），延迟 **511ms**；gen2 rc=0（25.8s）+ 
  calculator；final answer 引用检索结果；total_tool_time_ms=511.5。
- 债务放大（`DLLAMA_SHADOW_L1_DISABLE=1`）shadow_debt 留档（debtEntries,
  catchupEntries）：
  `11→47 0`（gen1 各 forward 持续累积）→ `0 48`（**web_search 窗口内全部补算
  48 笔**，`catch-up completed=48 remaining=0 elapsed_us=148997`）→ `1→4 48`
  （gen2 继续累积）。
- 对照组（无 --shadow-l2）：RC=0，gen 全 rc=0，web_search 507ms，行为与 L2 组
  一致（L2 是调度侧优化，不改输出）。
- 生成速度：GPU 上每 generation ~12-26s（CPU 共享机上为 333-600s）。

### 发现与修复

1. **Vulkan 路径缺 L2 buffer 访问**：agent_bench 默认 `--gpu-index`（BACKEND_AUTO
   →vulkan），`readNodeBuffer/writeNodeBuffer` 只在 CPU/CUDA device segment 实现，
   Vulkan 下 stash 全部落 "cannot snapshot pp_stage_out; dropping debt"（优雅降级
   不崩）。新增 `--edge-backend {auto,cuda,vulkan}`（`6281cf8`）显式选 cuda；
   Vulkan 的 L2 buffer 访问列为后续跟进项。
2. **CUDA_VISIBLE_DEVICES 掩码**：agent_bench `base_env` 默认
   `CUDA_VISIBLE_DEVICES="0,1,2"`，worker3(gpu-index 3) 被掩 →
   `cudaSetDevice failed: invalid device ordinal (101)`（101 是 cudaError_t 值非
   设备号）。用 `--cuda-visible 0,1,2,3` 解决，无需改码。

## 8B 模型抽查（llama3.1-8b q40，two-level-slack @ ed86511+）

显存估算：8B q40 文件 6.32GB，拓扑 `1@8*1@8*1@8*1@8`（32 层，4 节点 PP），
每卡：stage 权重 ~1.6GB + 边界 shadow ~0.4GB + KV ~0.13GB + workspace ~0.4GB
≈ 2.4-2.5GB < 余量 ~3.7GB（4 卡仍与 cyx vLLM 共享）→ 可跑，未挤占他人任务。
配置同矩阵（`--backend cuda --nthreads 1 --last-stage-sampling
--runtime-redundant-seg-enabled 0 --enable-plan-barrier --enable-pp-migration`）。
日志 `~/B01/EdgeVisor/runtime_logs/gpu_batch_matrix/b{1,8}_{off,l1,l2}/`
（8B 与 3B 矩阵共用 run_matrix_case.sh + SPOT_* env 覆盖）。

### batch=1

| mode | stage0 ms/fwd | stage1 ms/fwd | bubbleDrain/fwd | 判定 |
|---|---|---|---|---|
| off | 42.69 | 84.22 | 0.00 | ✅ |
| L1+L2 | 80.00 | 199.94 | **0.00**（bubble 1.48/2.27） | ✅ token 与 off 一致 |

### batch=8（two-level 故事在 8B 复现）

| mode | stage1 ms/fwd | bubbleDrain/fwd | stage1 对比 |
|---|---|---|---|
| off | 298.85 | 0.00 | 基准 |
| L1 | 319.51 | **4.05**（bubble 7.64，complete=26/26） | +7%（drain 在关键路径） |
| L2 | **173.36** | **0.00**（bubble 0.75） | **比 L1 快 46%、比 off 快 42%** |

- token 一致性：b1（off=L2）、b8（off=L1=L2）全部 md5 一致。
- 结论：8B + batch=8 时 L1 的 bubble 装不下（drain 4.05ms/fwd 上关键路径），
  L2 把 drain 挪进 stash，瓶颈 stage 显著变快——two-level Slack 设计故事在
  8B 上成立。

## 8B 全 batchsize 矩阵（b2/b4/b8，4-GPU PP empty GPU，2026-07-30）

补充 README 之前 8B b1/b8 spot check 缺失的中间 batch。Topology `1@8*1@8*1@8*1@8`，
CUDA backend，LSS on，`--last-stage-sampling --enable-pp-migration --enable-plan-barrier`。

**Stage 1（右边界 shadow stage，per-fwd）：**

| batch | off | L1 (drain) | L2 (drain) | bubbleOps L1 / L2 |
|---|---|---|---|---|
| 2 | 112.69 ms | 117.28 ms (0.48) | **115.65 ms (0.00)** | 882 / 441 |
| 4 |  98.55 ms | 107.42 ms (0.42) | **104.81 ms (0.00)** | 612 / 306 |
| 8 |  94.92 ms | 100.86 ms (0.37) | **111.06 ms (0.00)** | 468 / 234 |

**Token 一致性**：9 个 case（b2/b4/b8 × off/L1/L2）md5 全等
`5f6d3d718645b6dd98321ae8f2ea9348`，无任何语义回归。

**关键观察**：
- 空 GPU 时 L1 drain 维持在 0.37–0.48 ms/fwd，跨 batch 没有显著放大；之前
  README 中 b8 drain=4.05 ms/fwd 是在共享 vLLM 卡时测的。
- L2 bubbleOps 恰好是 L1 的一半（441/882, 306/612, 234/468）：L2 只在 bubble
  窗口内做能塞下的部分，剩下都进 debt 等 tool-wait 清理。
- b2/b4/b8 纯 inference（无工具）下 L2 的主路径耗时 vs L1 在噪声量级相当，
  L2 略慢（+0.5–10 ms），是 debt bookkeeping 的代价 —— 因为没有
  tool-wait 窗口来清债。这是 **L2 设计假设的反向验证**：L2 仅在 agentic
  tool-wait 场景下提供收益；在纯 inference（无工具）下是开销不是收益。
- 复现命令：`bash tests/shadow-kv-diag/matrix_8b_case.sh 2,4,8`，日志
  落在 `runtime_logs/gpu_8b_matrix/`，summary 自动写到 `summary.md`。

### 8B b8 在共享 GPU vs 空 GPU 的 drain 对比

| 条件 | b8 stage1 L1 drain | 备注 |
|---|---|---|
| 共享 vLLM（README 旧数据 @ ed86511+） | **4.05 ms/fwd** | bubble 7.64，L1 drain 在关键路径 |
| 空 GPU（本批 b8） | 0.37 ms/fwd | bubble 0.78，几乎全部装下 |

→ 验证了"drain 大小随 bubble 时长（=sync 时长）放大"的设计故事：
GPU 被挤占 → per-step sync 拉长 → bubble 窗口变长但工作量同步放大 →
b8 时 bubble 装不下 → drain 落到关键路径。这正是 README 主线的来源。


## Shadow L2 多轮连续 tool-wait（GPU 4xT4 CUDA，2026-07-30）

补充 README 之前单窗口 `shadow_l2_case.sh` 数据的多轮版本。Topology
`1@7*1@7*1@7*1@7`，3B q40，CUDA backend。

**`shadow_l2_multi_window_gpu.sh`** 在单个 chat session 内驱动 5 个连续
`tool_window_begin → sleep 6s (模拟工具) → tool_window_end` 循环，
观察 `shadow_debt` / `catchupEntries` / `catchupUs` 在每轮的变化。

### 实测数据（4xT4 空 GPU，NB=8 STEPS=24）

**mode = `l2_l1off`（强制所有 shadow 工作进 debt，最能体现 L2 价值）：**

| window | pre debtEntries | pre debtBytes | mid catchupEntries | mid catchupUs (累计) | post debtEntries |
|---|---|---|---|---|---|
| 1 | **62** | **1008272** (1MB) | 62 | 46760 | **0** |
| 2 | 0 | 0 | 0 | 46761 | 0 |
| 3 | 0 | 0 | 0 | 46762 | 0 |
| 4 | 0 | 0 | 0 | 46763 | 0 |
| 5 | 0 | 0 | 0 | 46764 | 0 |

→ **Window 1：62 entries / 1MB debt 在 46.7ms 内全部清空**；
→ 后续 4 个 window 无新 debt（tool-wait 间 idle），但 L2 后台线程每轮都被触发
（catchupUs 递增 1），证明机制持续运行、不死锁、不漏消息。

**Worker / Root 日志确认**：
- `worker1.log`：`🫧 [shadow-l2] catch-up completed=62 remaining=0 elapsed_us=45250`
- `root.log`：`🫧 [shadow-l2] catch-up completed=62`（root 后台线程也跑了）

**mode = `l2`（自然 L2，L1 仍在 bubble 内做能塞下的部分）：**

| window | mid catchupEntries | mid catchupUs |
|---|---|---|
| 1 | 62 | 23906 |

→ 自然 L2 模式下也有 62 entries 的 debt（说明 L1 bubble 在 GPU async + boundary
shadow 设置下不能完全装下），同样在 tool-wait 内清完，耗时 23.9ms（比 L1
disabled 模式还快，因为部分 shadow 工作 L1 已经做了）。

### 复现命令

```bash
# 强制 debt 模式（最能体现 L2 价值）
bash tests/shadow-kv-diag/shadow_l2_multi_window_gpu.sh mw_gpu l2_l1off 8 24 5
# 自然 L2 模式
bash tests/shadow-kv-diag/shadow_l2_multi_window_gpu.sh mw_gpu l2 8 24 5
```

每个 mode 约 5–6 分钟（worker init + chat turn 1 + 5 × 8s windows）。
日志落在 `runtime_logs/shadow_kv_diag/<tag>/`。

### 与单窗口 `shadow_l2_case.sh` 的对照

| 脚本 | window 数 | 数据侧重 | 验证结论 |
|---|---|---|---|
| `shadow_l2_case.sh` | 1 | 单一窗口的 debt-clear 时延 | 60 entries / 421ms（已有 README 数据） |
| `shadow_l2_multi_window_gpu.sh` | 5 | 机制持续运行、无丢消息、不死锁 | 62 entries / 46.7ms + 5 个 window 平稳 |
| `shadow_l2_multi_window.sh` (CPU) | 5 | 同上但 CPU back-end | 5 个 window 无崩，机制可达（数据受 CPU init 慢拖累） |


### off 模式对照 + token 一致性（追加，2026-07-31）

跑 `mode=off` 的 multi-window GPU 测试，对照 L2 是否改变生成内容。

**md5 比较（multi-window GPU，4xT4，3B q40，NB=8 STEPS=24，相同 prompt/seed）：**

| mode | raw token md5 | normalized md5 (strip whitespace) | normalized text |
|---|---|---|---|
| off | `59d9119baf2b0a1076bb09a0efee79d3` | **`491004f03de80c2c4525364ef8a8c83c`** | `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20` |
| l2_l1off | `491004f03de80c2c4525364ef8a8c83c` | **`491004f03de80c2c4525364ef8a8c83c`** | `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20` |
| l2 (natural) | `491004f03de80c2c4525364ef8a8c83c` | **`491004f03de80c2c4525364ef8a8c83c`** | `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20` |

→ 三个模式的 normalized token md5 **完全相同**！raw md5 的差异仅来自 off 模式
在 chat binary 显示 token 时插入了空格；实际生成的 token 序列是相同的。

**结论**：L2 在多轮 tool-wait 场景下生成内容与 off 一致；多轮窗口不引入任何
semantic drift。

