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

## 关键结论（2026-07-27，two-level-slack @ 13c502d）

- sync 模式 + `DLLAMA_SHADOW_RIGHT_PP_CACHE=1`（右边界 shadow 改从 pp_stage_out
  快照读输入）：layer 14 的 red_k/red_v 与参考值**逐位相等**（48 个 position ×
  batchSize 1/2/4 全部 maxAbsDiff = 0）。
- 默认路径（右边界 `OP_MERGE_ADD zqPipe→xBuffer`）：全部 position 数值错误
  （drain 时 xBuffer 已被 pp_stage_merge 合并过，shadow 再合并一次 → double-add）。
- 左边界 shadow 输入（`OP_CAST xPipe→xBuffer`）拿到的是上游 stage **最终输出**
  （= 上一层 output），而 shadow 层需要的是上一层 **input**，天然错一层。
- async 模式下 shadow 在 sync 窗口内执行，时序竞争导致输入可能是上一 token 的
  陈旧值（实测 best-matching position 系统性偏移 1）。
- **主路径污染（严重）**：shadow 段与主路径共享 xBuffer 等工作缓冲。右边界段的
  `OP_MERGE_ADD zqPipe→xBuffer` 若在末层 ff sync 与 pp_stage_merge 之间的窗口
  执行，主路径会 double-add 末层 FFN 输出，污染 stage 输出。实测：无任何迁移、
  仅 `DLLAMA_BUBBLE_SHADOW_KV=1`，31-token prefill 下生成序列从首个 decode token
  起与 baseline 完全分叉；主路径 K/V 对比显示 node0 各层 K 在 prefill 内与
  baseline 一致、node1 全部层从 pos 0 起偏差（stage 输出在 pp_send 前被污染）。
- 端到端迁移消费链：red_k/red_v 经 takeover MHA 消费（llm.cpp:1662-1682）；
  计数 prompt 下消费错误 red 状态未使 argmax 翻转（被机械序列掩盖）。

详见主代理诊断报告。日志：`~/B01/EdgeVisor/runtime_logs/shadow_kv_diag/`。
