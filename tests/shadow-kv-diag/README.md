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
