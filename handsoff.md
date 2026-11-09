# handsoff

本段到下面的分隔线为止是固定说明。后来的 Agent 不要改写、不要删、不要把当次进度写进这一段。每次交接只整段替换分隔线以下的内容。

## 这份文档是干什么的

同一个人会在两个 Agent 之间交替做这个仓库的开发。一边做完一段工作，就把分隔线下面更新掉；另一边先读这份文件，然后接着做，不用再口头补一大段上下文。

## 接手时

1. 先读分隔线下面的全文，再看它点名的文件和命令，不要凭记忆补实验设计。
2. 用户说「继续」时，按「下一步」做。没写进下一步的长实验不要自己开。
3. 改代码、跑实验、只查了状态，都算一段工作。离开前更新分隔线下面。

## 交接时怎么写

替换分隔线以下的全部内容，保留分隔线本身。写清这些，有代码 diff 就写，没有也要写：

- 现在在做哪件事，做到哪一步。
- 机器上还留着什么：进程、端口、GPU、容器、没跑完的队列。
- 已经定下来的结论和约束。不要只贴 git diff。
- 卡住的原因，以及下一次打开终端应该先跑的命令或先改的函数。
- 相关 commit、关键路径、明确不要提交或不要推送的文件。

不要把对话原文、完整 diff 或论文 PDF 贴进来。事实写短，路径写全。

---

## 可变交接

更新时间：2026-11-09 17:44 CST（机器时间 09:44 UTC）。分支 `exp/multifault-weights`，已推到 GitHub `wansongcc/EdgeVisor`。

### 现在在做哪件事

跑 8B 的 E1 主矩阵（机制级对比，六臂）。**正在跑，不要停，也不要再开一份。**

- 驱动脚本：`/tmp/ev_matrix_v2/run_8b.sh`，pid `3057489`，启动命令 `setsid -f bash run_8b.sh`
- 实验进程：pid `3113960`，`python3 -u /home/byh/B01/baseline/_harness/e1_v2.py --model-key 8b ...`
- 日志：`/tmp/ev_matrix_v2/matrix.log`
- 进度：8b 4× 已完成触发点 1024 / 2048 / 4096；`8b_s4_t8192` 正在跑（bounded 已完成，transfer 进行中，还剩 pipelive / ev_hit / ev_miss）
- 队列剩余：8192、16384、再两档 6×（1024、8192），最后单独补 replay 六个组
- 队列里的组是按 `run_8b.sh` 顺序自动跑的，不需要人介入

### 已经定下来的结论

讨论方向和结论比代码更重要，先读这一段。

**1. 之前那一整批 E1 数据作废**，原因有三个，都已修：权重没算进迁移时间；`state_transfer_bytes` 双计（`sourceBytes + targetBytes` 是同一次拷贝，报 270 MB 实际过线 135 MB）；故障只注入一次、只迁一层。旧结果在 `/tmp/ev_e1_deprecated/`。

**2. bounded 没有「恢复时间」。** 它不做任何恢复动作，只有 episode 完成时间可比。主指标分两个数报：触发到新路径第一个 token（含在飞 token），以及其中迁移占的 `stall_ms`。用 bounded 的「下个 token 间隔」当恢复时间是概念错误。

**3. 故障必须持续到 episode 结束，且按重平衡搬够层。** 短窗口 + 只搬 1 层时，六臂 tok/s 全都一样（约 3.6），迁移付了成本却拿不到吞吐回报。现在用 `pos-pos:slowdown@persist+rebalance`，层数按 `S*(s-1)/(s+1)` 自动算（8B 4×→6 层，6×→7 层）。

**4. 降速惩罚必须按 primary 层数缩放，不能按节点固定加时。** 固定加时的话搬走 6 层后仍付满额惩罚，稳态只从 485→454 ms；按层缩放后 485→187 ms。`countExecutedLayers()` 只数 `SEG_ROLE_PRIMARY`，冗余段不能算进去，否则 EV 的准备会把自己拖慢。

**5. EV 的预留在气泡路径上，不在内联路径上。** 这是最容易踩的坑：`--runtime-redundant-seg-enabled` 把冗余段编进 forward，每 token 多 20 ms，会得出「EV 亏本 22 倍」的错误结论；`DLLAMA_BUBBLE_SHADOW_KV=1` 才是把冗余计算放进空隙的做法。用气泡后 3B 单线程 EV 从 8.63 回到 10.90 前置 tok/s。

**6. 这套 PP 拓扑下单线程更快。** 一 stage 一节点，多余线程只增加同步开销。所有臂统一 `--threads 1`，否则对比不公平。旧的多线程 8B/3B 数据与新数据不可比。

**7. EV 的预留是预算约束下的预测，不是全量铺。** 全铺 48 层时前置慢 2.09 倍。现在 `--ev-budget-boundaries 1`、深度=重平衡层数，并让先验 `DLLAMA_SCENARIO_PROBABILITIES` 指向故障真正发生的边界（stage 2→1 是边界 1）。`ev_hit` 与 `ev_miss` 的区别就是预测对错，两者的图结构都要 provision，否则迁移命令会被 `target stage lacks provisioned layer` 拒掉。

**8. ReMP / DynaPipe / Llumnix 不进主表。** 这四个目录里只有 `.gitkeep`，从没复现过；现有六臂是机制移植，不是系统复现。ReMP 在固定 2→1 路线上与已有两臂重叠（影子关=transfer，影子开=ev_hit）。`dynapipe_ab.py` 依赖的 `DLLAMA_TPOT_ACCOUNT_TAIL_SAMPLE` 在源码里不存在，跑不通。

### 已出的结果（8B，4×，单线程，触发点 1024/2048/4096）

episode 完成时间，从触发起算：

| 触发点 | bounded | transfer | pipelive | ev_hit | ev_miss | replay |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 325.9 s | 250.5 s | 248.5 s | **199.1 s** | 248.4 s | 477.9 s |
| 2048 | 568.0 s | 484.5 s | 484.4 s | **441.1 s** | 487.5 s | 未跑 |
| 4096 | 1163.3 s | 1063.2 s | 1056.9 s | **926.7 s** | 1056.7 s | 未跑 |

`stall_ms`：EV 0–1 ms，transfer 1509 / 2426 / 4247 ms。KV：101 / 202 / 405 MB。

未解释现象：EV 的前置吞吐稳定高 15–30%（6.92 / 5.37 / 4.77 vs 其他臂 5.26 / 4.84 / 4.14）。怀疑是气泡填充利用了 stage 间空隙，但也可能是其他臂单线程没跑满。**这个差异要么在论文里解释清楚，要么归一化掉，不能当作 EV 的收益直接报。**

### 机器上还留着什么

- GPU 0–3 被当前实验占用（7281 / 9881 / 9881 / 7207 MiB），四张都在用
- 端口 9992–9994 在实验结束后会释放
- 不要动 `/tmp/ev_e1_deprecated/`（作废数据，留对照）
- 已归档的可复用数据：`/tmp/ev_st3b/`（3B 单线程六臂，触发点 600，已验证）、`/tmp/ev_matrix_v2/8b_s4_t1024`（含 replay）

### 下一步

用户说「继续」时：

1. 先确认 `run_8b.sh` 是否还在跑（`ps -eo pid,cmd | grep run_8b`），跑着就别动。
2. 跑完后汇总 8B 完整表，然后问用户 3B 是否按新配置（单线程 + 气泡 EV）全量重跑，14B 是否开。
3. 之后才是命中率扫描（`--ev-budget-boundaries` × `DLLAMA_SCENARIO_PROBABILITIES` 组合），横轴预算、纵轴加权恢复时间。

没说继续不要开 3B/14B 长实验。

### 约束

- 不要合到 `main`，不要强推
- 不要提交 `/tmp` 下的结果目录、论文 PDF、`baseline/` 仓库的东西（那个仓库单独的 remote 是 `BianYanhui/EdgeVisor_baseline_reproduc`）
- 不要提交 `._*`
- `baseline/_harness/e1_v2.py` 改动在 `/home/byh/B01/baseline` 仓库里，目前**未提交**（`E1v2-report.md` 与五个 harness 脚本都是 untracked）。这次的 `handsoff.md` 只提交到 EdgeVisor 仓库，不要顺手把 baseline 仓库的东西带上，除非用户要求

### 相关 commit

- `ed7e695` Charge weight residency and make faults repeatable —— 本次代码改动，7 个文件：`app.cpp`（权重下发、按所有权选层、影子接管语义）、`app.hpp`（`weightResidencyPreallocated`、`sendWeightLoadRequest`）、`dllama.cpp`（多锚点 mark、逐 token 时间戳）、`llm.cpp`/`llm.hpp`（按层权重加载、`predictLlmLayerWeightBytes`）、`nn-executor.cpp`/`.hpp`（故障调度、按 primary 层校准、`countExecutedLayers`）
- 前一个提交 `13c7e5e`，其上四个基提交都已在 origin
- 关键文件：`EdgeVisor/src/nn/nn-executor.cpp`（`faultScheduleConfig`、`countExecutedLayers`、`forward` 里的惩罚施加点）、`EdgeVisor/src/llm.cpp`（`buildRuntimeStageLayerPlan` 的准入与预算）、`EdgeVisor/src/app.cpp`（`sendWeightLoadRequest`、`resolvePpMigrationLayersByOwnership`、`sendPendingLayerSwitchControlOnly`）
