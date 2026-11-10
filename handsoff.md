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

更新时间：2026-11-10 13:50 CST。分支 `exp/multifault-5dev-lab`（从 `exp/multifault-weights` @ `81cb8b7` 拉出，已推送 GitHub `wansongcc/EdgeVisor`）。

> **重要：上一段交接描述的 8B 矩阵不是这台机器上的。** `/tmp/ev_matrix_v2`、`/tmp/ev_st3b`、`/tmp/ev_e1_deprecated`、`/home/byh` 在本机都不存在，也没有 `run_8b.sh` / `e1_v2.py` 进程。那份矩阵在 `wansongcc/EdgeVisor` 的 `exp/multifault-weights` 分支上另一台机器里跑（`/home/byh/B01`）。本机是 `yahboom`（Orin NX 16GB），不是那台。

### 现在在做哪件事

在**五台物理设备**上把 EdgeVisor 跑通，并验证 bubble shadow KV 与 PP 迁移；有线（千兆交换机直连 `192.168.137.x`）和无线（`kb310` WiFi，`192.168.8.x`）各跑一遍。

跑通目标已完成，且已压到迁移的两条新路径：**多层单跳**与**重复迁移向内走边界**。目前正在收尾。

### 本轮实测（2026-11-10）

模型统一 Qwen3-14B Q40（`10982707336` 字节，五台一致），`--nthreads 1`，`--ratios "1@8*1@8*1@8*1@8*1@8"`。

| 场景 | 传输 | tokens/s | stall_ms | 说明 |
| --- | --- | ---: | ---: | --- |
| 五机 PP 基线 | 有线 | 1.83 | — | 输出正确 |
| 五机 PP + bubble shadow KV | 有线 | 1.82 | — | root 每 token 2 个影子段，`completed=1` |
| 五机 PP + 迁移 + shadow | 有线 | 1.79 | 138.2 | 1 层，`exported=21 queued=21 status=ok` |
| 五机 PP 基线 | 无线 | 1.66 | — | 比有线慢 9% |
| 五机 PP + 迁移 + shadow | 无线 | 1.40 | 188.8 | 比有线迁移慢 37% |
| **2 层单跳迁移** | 有线 | — | 246.3 | `exported=42 queued=42 status=ok` |
| **重复迁移（7 后 6）** | 有线 | — | 145.4 / 174.0 | **ownership 回退首次验证通过**，见下 |
| 单机 / 两机 PP 冒烟 | — | 13.33 / 6.57 | — | 3B 模型，输出 `...Paris!` |

延迟基线：有线 0.26–1.81 ms（nx1↔nx2 0.36 典型，ROG 1.81 略高）；无线 7–35 ms 且抖动大（mdev 1.1–38 ms）。无线迁移 stall 涨到 188.8 ms 与抖动同量级，不是迁移逻辑变慢。

### 重复迁移的验证结果（本轮最重要）

`ed7e695` 里的 `resolvePpMigrationLayersByOwnership` 之前没有真机证据。本轮用 `scripts/uds_migrate.py` 连发两条同方向命令测得：

```
cacheSeq=4  route=0->1  layers=[7]   stallMs=145.410  stateBytes=839288    status=ok
cacheSeq=6  route=0->1  layers=[6]   stallMs=173.977  stateBytes=2320328   status=ok
```

第一次搬走 7 之后，源 stage 不再拥有 7。第二次命令仍然写 triggerLayer=7，但代码没有重试 7，而是沿边界向内取了 6。这正是 commit message 里说的「successive moves walk the boundary inward」，**已确认在真机上成立**。

`stateBytes` 从 839288 涨到 2320328，因为第二次触发时上下文已经更长，KV 量更大，符合预期。

### 手工 plan controller 的两个坑

**1. `mode=exact` 的 `triggerPos` 必须还在流水线前方。** 命令发出时如果 `controlPacket.position` 已越过 `triggerPos`，命令会被接受（`ok:true`）但停在 `auto collect armed` 不再前进，永远等不到 KV ack，也不会报错。**用 `mode=next_barrier`**，它把 `triggerLayer` 当 `firstLayer` 用、不设 `triggerPos`，下一个 barrier 就触发。

**2. `set_pp_migration` 会被 JIT gate 拒。** 默认 `jitMode=enabled` 时必须有 `predictedBenefitMs`、`predictedCostMs`、`survivalProbability` 三个在线估计，否则回 `jit_missing_or_invalid_online_estimates`。`scripts/uds_migrate.py` 默认带 `--jit-estimates 500,50,0.9`。

**3. `DLLAMA_ASYNC_KV_COLLECT_SUBMIT=0` 只有 `--auto` 分支会设。** 走 `--plan-ctrl-socket` 手工路径时要自己加，否则异步 KV 收集不提交。

### 设备清单（本轮有效）

| 名字 | 用户@有线 IP | 无线 IP | 架构 | 显存 | nproc | 后端 |
| --- | --- | --- | --- | ---: | ---: | --- |
| nx1（本机）| 192.168.137.13 | 192.168.8.46 | aarch64 | 15655 MB | 8 | CUDA (Orin NX 16GB) |
| nx2 | jetson@192.168.137.15 | 192.168.8.23 | aarch64 | 15655 MB | 8 | CUDA (Orin NX 16GB) |
| nano1 | jetson@192.168.137.18 | 192.168.8.53 | aarch64 | **7619 MB** | 6 | CUDA (Orin Nano 8GB) |
| nano2 | jetson@192.168.137.16 | 192.168.8.45 | aarch64 | **7619 MB** | 6 | CUDA (Orin Nano 8GB) |
| rog | cc@192.168.137.20 | 192.168.8.49 | x86_64 | 6144 MB | 16 | **Vulkan only**（没装 CUDA toolkit）|

免密 SSH 全部可用（`~/.ssh/id_ed25519`）；`sshpass` 本机没装，密码方式用不上，也不需要。

### 必踩的坑（都已绕过，下次直接用）

**1. `--enable-pp-migration` 会隐式打开 `--enable-stage-full-weights`。** Nano 的 7.6 GB 装不下 stage 全量权重，会在第二次加载时 `cudaMalloc ... out of memory (2)` 直接崩，root 侧只看到 `Socket closed`。必须显式 `--enable-stage-full-weights 0`，此时迁移覆盖限于 `--runtime-redundant-boundary-layers`。

**2. worker 必须和 root 带同一套冗余段环境变量。** 冗余段的图是在 worker 启动时按 env 建的；两边不一致时 worker 报 `Cannot locate op name='block_matmul_q' index=N`，root 报 `PP peer offline without ready shadow cache`。重启 worker 时要带：
`env DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_BUBBLE_SHADOW_KV_LOG=1 DLLAMA_RUNTIME_REDUNDANT_BOUNDARY_LAYERS=2 ./dllama worker ...`

**3. 冗余深度 2 是 Nano 的上限，3 会 OOM。** 实测 `1@8` 五机 + Qwen3-14B：`DLLAMA_RUNTIME_REDUNDANT_BOUNDARY_LAYERS=2` 三台内存机都稳；设 3 时 Nano（7619 MB，清缓存后可用 4006 MB）立刻 `NvMapMemAllocInternalTagged error 12` → `cudaMalloc ... out of memory (2)`。这是 8 层 primary + 3 层冗余的权重放不下，**是硬件边界不是 bug**。要做 3 层以上覆盖，得换每 stage 层数更少的切分或更大的设备。

**4. shadow KV 日志要单独开 `DLLAMA_BUBBLE_SHADOW_KV_LOG=1`。** 不开的话什么输出都没有，很容易误判成「没生效」。开了会看到 `🫧 [bubble-shadow-kv] ... segments=2 attn=2 ops=22 completed=1`。

**5. 本机 GPU 在沙箱里不可见。** 沙箱内 `./dllama --list-devices` 报 `CUDA: unavailable (cudaGetDeviceCount failed: unknown error (999))` / `Vulkan ... ErrorIncompatibleDriver`，看起来像机器没 GPU。实际上非沙箱下是 `CUDA Device[0]: Orin cc 8.7 sm_87`。**所有 dllama 运行都要在非沙箱模式执行。**

**6. 这几台机器直连 GitHub 不通。** `git fetch origin` 会卡 2 分钟以上然后 timeout（有线时 `10.47.249.150:7897` 代理拒绝连接；无线时 `github.com:443` 超时）。本机有可用网络，所以同步方式是从本机做 bundle 再 scp：
```
git bundle create /tmp/ev_branch.bundle <branch> [more branches]
scp /tmp/ev_branch.bundle <host>:/tmp/
ssh <host> 'cd ~/EdgeVisor-current && git bundle unbundle /tmp/ev_branch.bundle && git checkout -B <branch> <sha>'
```

**7. OOM 先清 page cache。** Tegra 是统一内存，`buff/cache` 和 `cudaMalloc` 抢同一块物理内存。Nano 上清缓存前可用 3772 MB、清后 4006 MB，正好是能否多加载一层的差别：
```
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
`/tmp/restart_workers.sh` 每次重启 worker 前都会先清。

### 机器上还留着什么

- 四台 worker 在 `:18091` 监听（重启后 pid 会变，用 `pgrep -x dllama` 查），带 `DLLAMA_BUBBLE_SHADOW_KV=1` 与 `DLLAMA_RUNTIME_REDUNDANT_BOUNDARY_LAYERS=2`，加载 Qwen3-14B
- 重启工具：`/tmp/restart_workers.sh <user@host> 18091 <model> "<env 串>"`，已内置清 page cache 与带时间戳的日志
- 驱动工具（本轮新增，**在仓库里**）：`scripts/uds_migrate.py`，连 root 的 plan UDS 发多次迁移
- 各机 `/tmp/w_HHMMSS.log`（worker）、本机 `/tmp/five_*.log` 与 `/tmp/rep{{2,3,4,5}}_root.log`（root）、`/tmp/build.log`、`/tmp/ev_branch.bundle`
- 三台 Jetson 各有 stash `pre-multifault local smoke fixes 2026-11-10_11:14`（切换分支前保存的未提交五设备 smoke 修复，**不要丢**）
- ROG 有 4 个仓库副本，本轮用的是 `~/EdgeVisor-current`

### 下一步

用户说「继续」时：

1. **merge 由用户指挥**，不要自己合进 `exp/multifault-weights`。本轮唯一的代码改动是新增 `scripts/uds_migrate.py`，其余是 `handsoff.md`。
2. `EDGEVISOR_WEIGHT_RESIDENCY=on_demand` 的权重上传路径（`sendWeightLoadRequest` → `applyPendingWeightLoads`）在五机上**还没触发过**。本轮所有迁移都走 shadow 分支，而 shadow 分支的 `weightResidencyPreallocated` 判断在这条路径上没被踩到。要验证得跑 `DLLAMA_SHADOW_KV_MODE=disabled_transfer` 和 `disabled_recompute` 两个基线臂。
3. 多跳迁移（0→1→2 或反向）还没测，本轮只做了同方向的 0→1 连续两次。
4. 无线下每个配置只跑了一次，1.40 vs 1.79 里有多少是 WiFi 抖动没分离。要报数得各跑 3–5 次取中位数。

### 已经定下来的结论

> 下面第 1–8 条与 8B 结果表**来自另一台机器上的 `exp/multifault-weights` 工作**（`/home/byh/B01`），不是本机跑出来的。方法论仍然适用，但数字不可在本机复现，因为本机没有那批 `/tmp` 数据。留着是为了不让下一位重做已经定过的方法论。

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

> 以下数据来自另一台机器，本机无法复现。保留作方法论参考。

episode 完成时间，从触发起算：

| 触发点 | bounded | transfer | pipelive | ev_hit | ev_miss | replay |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 325.9 s | 250.5 s | 248.5 s | **199.1 s** | 248.4 s | 477.9 s |
| 2048 | 568.0 s | 484.5 s | 484.4 s | **441.1 s** | 487.5 s | 未跑 |
| 4096 | 1163.3 s | 1063.2 s | 1056.9 s | **926.7 s** | 1056.7 s | 未跑 |

`stall_ms`：EV 0–1 ms，transfer 1509 / 2426 / 4247 ms。KV：101 / 202 / 405 MB。

未解释现象：EV 的前置吞吐稳定高 15–30%（6.92 / 5.37 / 4.77 vs 其他臂 5.26 / 4.84 / 4.14）。怀疑是气泡填充利用了 stage 间空隙，但也可能是其他臂单线程没跑满。**这个差异要么在论文里解释清楚，要么归一化掉，不能当作 EV 的收益直接报。**

### 五设备现场状态（本机 / 各 worker）

- 四台 worker 正在 `:18091` 监听（nx2 pid 15784、nano1 102348、nano2 36372、rog 10354），带 shadow KV 环境变量，加载的是 Qwen3-14B
- `/tmp/w.log`（各 worker）、`/tmp/five_*.log`（本机 root 日志）、`/tmp/build.log`（各机编译日志）、`/tmp/ev_branch.bundle`
- 三台 Jetson 各有一个 stash：`pre-multifault local smoke fixes 2026-11-10_11:14`，切换分支前保存的未提交五设备 smoke 修复，**不要丢**
- ROG 有 4 个仓库副本，本轮用的是 `~/EdgeVisor-current`

### 相关 commit

- 本分支基点 `81cb8b7`（`exp/multifault-weights` head），本轮未改代码
- 该 head 上的关键实现：`ed7e695`（权重驻留成本 + 多窗口故障 + ownership 解析）、`6639ac5`（decode 期分块 KV 补丁）、`a8a956b`（增量 KV 导出 + replay 单样本）、`7b2e195`（`DLLAMA_IGNORE_EOS`）
- 关键文件：`EdgeVisor/src/nn/nn-executor.cpp`（`faultScheduleConfig`、`countExecutedLayers`、forward 惩罚点）、`EdgeVisor/src/app.cpp`（`sendWeightLoadRequest`、`resolvePpMigrationLayersByOwnership`、`applyPendingWeightLoads`、`runPipeLivePatch`）、`EdgeVisor/src/llm.cpp`（`loadLlmNetWeightUnevenFiltered`）

没说继续不要开 3B/14B 长实验。

### 约束

- 不要合到 `main`，不要强推
- 本分支 `exp/multifault-5dev-lab` 是给用户 review 用，**merge 由用户指挥**，不要自己合进 `exp/multifault-weights`
- 不要提交 `/tmp` 下的结果目录、论文 PDF、`baseline/` 仓库的东西（那个仓库单独的 remote 是 `BianYanhui/EdgeVisor_baseline_reproduc`）
- 不要提交 `._*`
- 每次改代码后同步更新本文件的「可变交接」段
