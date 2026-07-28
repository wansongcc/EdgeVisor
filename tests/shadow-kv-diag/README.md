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
