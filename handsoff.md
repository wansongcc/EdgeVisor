# handsoff

## 文档说明（固定，不要改）

这份文件是给下一个 Agent 的交接，不是设计文档，也不是用 `git diff` 代替进度说明。

用户经常在两个 Agent 之间交替做同一件事。换之前，刚做完的 Agent 更新下面的「当前交接」，提交并推送。新 Agent 开工前先读这份文件。读完应能接着做，不需要用户再讲一遍背景。

固定规则：

- 只改「当前交接」。上面这一节保持不动。
- 交接写的是工作状态。正在做什么、做到哪一步、结果、没改代码的实验、下一步，都要写。没有代码改动也要更新。
- 写清事实：哪台机器、哪条命令、日志在哪、进程是否还在跑。没验证的猜测不要写成结论。
- 不要写密码、密钥、token。
- 提交时只加入这份文件，除非用户明确要求把别的改动一起提交。不要提交 `._*` 这类 macOS 附属文件。
- 提交并推送到当前开发分支。远程若不能快进，先把这份提交接到远程分支之上再推，不要强推。

## 当前交接

更新：2026-10-19（Asia/Shanghai）。下面替代 2026-10-18 的 smoke 交接；旧 PID、旧机器地址和旧测试结论不是当前事实。

### 1. 当前状态（Current state）

用户本轮要求：先把未来实验规划写入本文件，然后开始「实验 A：机会是否真实」，并依次补大测试套件、解决 14B/5 机 nano OOM。规划 `ec5d4cf`、A 试点 `644210b`、多轮 `98335ad`、大测试 `2cddaef`、14B 恢复修复 `94ec488`、14B 记录 `76bcafb` 均已推送。没有合并 `main`，没有改 ROG 安全配置。

- 当前开发分支 `refactor/experiment-foundation`；API 修复 `4032df5`、测试记录+交接 `20530db`、实验规划 `ec5d4cf`、A 试点 `644210b`、A 多轮 `98335ad`、A 大测试 `2cddaef`、14B 恢复修复 `94ec488`、14B 记录 `76bcafb`。`main` 仍为 `bb97cab`。
- **实验 A 单轮结果**：单机 8B Qwen3 Q40，真实工具（沙箱 coding 测试、120k 文档 BM25、1M 行只读 SQLite），9 次真实工具调用窗口 **9.6 ms–691 ms，中位 473 ms**；coding 中位 430 ms、search 636 ms、SQL 353 ms。论文 Fig.5 的 1.096/1.20 s 中位未复现，实测约为其 35–60%。同一 workload 内查询选择性可差近两个数量级（SQL COUNT 9.6 ms vs GROUP BY day 618 ms；search 唯一 token 0.57 ms vs 宽查询 819 ms）。
- **实验 A 多轮结果**：7 个 episode、19 次调用、每 episode 2–4 次；累计窗口 **0.13–1.06 s**（coding fix 0.13–0.28 s；SQL 链 0.57–0.64 s；跨工具 search 链 0.98–1.06 s），仍远低于论文 W1 的 5×1.096 s≈5.5 s。6/7 完成；coding_fix_rle 在 180 s 客户端超时。第一个复杂查询贡献大部分窗口，后续索引查询只有 87–101 ms。
- **实验 A 大测试上界**：heavy matrix 3.08 s、topk 3.93+4.01=7.94 s、logparse 1.78 s；参考测试本身 1.5–4.6 s。与微测试合并后，累计机会范围 **0.13–7.94 s**；W1 的 5.5 s 落在范围内但非普适。
- **14B/5 机复核**：18/18/2/1/1 + context 1024 + 180 s 预算解决 nano OOM；search 1 call 622 ms、SQL 链 3 calls 530 ms、heavy topk 3 calls **11.80 s**（2.1× W1），三者均正确完成。工具窗口与 8B 一致；生成约 1.2 token/s，不是性能配置。期间发现并修复 runtime 重建时裸 `NnTransferSocketException` 逃出 main 导致 terminate（`94ec488`）。
- 详细记录：`docs/test_records/18`–`21`；harness `scripts/experiments/opportunity_study.py`；图 `docs/test_records/assets/experiment_a_*_2026101{9,20}.png`。
- API 修复的改动：`dllama-api.cpp` 捕获生成期 timeout/offline/session-restart/worker-socket/executor 异常，返回 503（流式则 SSE error）并展开 runtime 重建；客户端写失败不再混入推理故障；HTTP 状态短语修正；固定拓扑 KV/control-ACK/尾采样默认预算 5000→30000 ms；session restart 后最多 10×500 ms 等待 worker relisten。
- 已实机验证（0.6B Q40）：
  - 双机强制 `DLLAMA_KV_ACK_TIMEOUT_MS=1`：返回 503 `peer_timeout`，进程存活，`/v1/models` 恢复，改 30 s 后 completion 返回 200。
  - 五机默认 API gate：5 s 时第三个 cold 请求 503（复现原故障），30 s 时 20/20 通过；10 次生成 6.28–9.63 s，cold 约 7.56 s。
  - 双机运行中 SIGKILL worker 并重启：root 在 pos 22 session restart，重放后完成 96 token（9.32 token/s），无 Critical error；未触发新增的 relisten 重试分支。
  - 五机默认 `five_mixed_auto` 复跑：96 token、10.20 token/s、无重启。修复前同命令在 pos 22 大迁移后 node2 掉线并级联失败；复跑通过不能证明原始断连根因消除。
  - nx1 `make -j8 check`：10 项 C++ 检查全部通过。
- **设备版本不一致**：只有 nx1 的 `dllama`/`dllama-api` 重新编译到 `4032df5`；nx2/nano1/nano2/rog 仍是 `e431cbd` worker 二进制（协议兼容，本轮只改 root 行为）。最终实验前必须把五台统一到同一提交并重新核对二进制哈希。

### 2. 已走过的失败与对照（Dead ends）

- 5 s 默认预算是确定性问题：五机 cold replay 的尾采样等待超过 5 s；30 s 后同一 gate 通过。不要把“调大预算”单独说成可用性修复；本轮同时加了 503 + runtime 重建。
- `NnPeerTimeoutException` 原样逃出 `server()` 会触发 `std::terminate`；只在 `main()` 加宽泛 `catch (std::exception)` 会掩盖致命启动错误（例如 trace 文件已存在的 `Cannot exclusively create EDGEVISOR_TRACE_PATH` 会无限重试）。恢复必须是请求级分类 + 显式 runtime 重建，不能退回全局吞异常。
- 默认五机 auto 的旧失败链：pos 22 规划 23 层 1→2 迁移，导出 529 行后 node2 被判定 offline；node1 的 failover 因 shadow-unfilled 抛错并关闭 root 连接；root 重启后 node1 尚未 relisten，`startup peer=1 offline, retreat` 最终 exit 1。复跑未重现，根因未定位。
- 双机掉线测试里 worker 重启够快，没有真正触发“relisten 等待”分支；该分支目前只有代码审查和条件测试，不能声称已覆盖原 auto 竞态。
- 流式请求在“已发 SSE 头之后”才失败的路径未做设备实测；非流式 503 和超时后恢复已实测。

### 3. 未完成事项（Unresolved items）

1. 默认五机 auto 的间歇性 peer 断连仍未定位：需同时抓 root/worker 退出码、`dmesg`/OOM、ZeroTier/链路状态和迁移前后 socket 事件，区分真实断链、POLLHUP 误判和 worker 崩溃。
2. 30 s 是等待期限，不是 watchdog；4096 token cold replay 仍可能超过 30 s。超时后 API 会 503 并重建，但尚未做长上下文回归。
3. 流式 post-header 错误、并发请求/压力、14B API/auto/长稳、worker 无重启 shadow 仍未验证。
4. MobiSys 补强路线仍是 S0/S1 基础，S2–S7 未完成，详见 `docs/research/HANDOFF_2026-10-14.md`。S2 的 migration transaction/epoch/frontier/closure 仍是下一阶段主线。
5. ROG 失守密码和公网 SSH 入口仍未处理；用户此前明确暂不动安全配置。
6. nano2 根盘约余 6.1 GiB、ROG 约余 5.5 GiB；长实验前复查容量，不擅自删除用户文件。
7. 实验 A 已完成单机 8B 的单轮 + 多轮 + 大测试，以及 14B/5 机的 OOM 修复与复核：机会范围 0.13–11.8 s。仍未做远程工具、14B 均衡性能配置、decode bubble（API 模式没有 worker per-stage timing）、heavy matrix 的 14B 重跑、大样本 p95。

### 4. 第一项下一步（First next action）

A 的主体已完成（见第 6 节），下一步按顺序：
1. **14B per-stage token 埋点 + bubble**：API 模式目前没有 worker `[token-prof]`，无法拆 stage exec/sync；先补埋点，再测 14B/5 机的 bubble 和 stage idle。
2. **均衡 14B 配置**：当前 18/18/2/1/1 生成约 1.2 token/s；尝试 context 512/768 或 16/16/4/2/2 等更均衡的分配，目标 >3 token/s。
3. **大样本**：heavy coding 每类 3–5 个任务、14B 至少 2–3 个任务，统计完成率与窗口分布；再进 B 的 5-rep pilot。
4. B 的窗口输入用 0.1–12 s 的逐窗口分布，不用固定 1.1 s。

A 的 harness、任务集和原始数据要保留；代码改动 fresh build + 实机回归后 commit/push。未经新请求不合 main，不直接开始论文性能主实验。

重跑入口：Mac 项目父目录 `work/device_smoke_20261018/` 下新增 `api_recovery_20261019.py`、`api_five_default_20261019.py`、`failover_retry_20261019.py`；原有 `final_checks.py`、`smoke_14b.py`、`lab.py`、`smoke.py`、`api_smoke.py` 仍在。精确逐场景命令见 `cases/*.command.txt`。这些是证据/控制脚本，不在 Git 中，运行中不要编辑源码或脚本。

### 5. 证据、清理与约束（Optional context）

- Mac repo：`/Users/yhbian/Documents/Codex/2026-10-08/github-plugin-github-openai-curated-remote-2/work/EdgeVisor`。测试记录 17–21；实验 A harness `scripts/experiments/opportunity_study.py`。
- 实验 A 原始数据：`work/opportunity_20261019/fixtures/`（120k 文档、1M 行 SQLite、coding 测试）、`runs/pilot_8b_*`、`runs/multi_8b`、`runs/heavy_8b`、`runs/opp14b5_*`（episode/summary JSON）；不在 Git。
- 图：`docs/test_records/assets/experiment_a_tool_windows_20261019.png`、`..._cumulative_windows_20261019.png`、`..._opportunity_range_20261020.png`、`..._14b_topk_20261020.png`。
- 14B 日志：`nx1:/home/jetson/edgevisor_runs/opp14b5_20261020.log` / `.trace.jsonl`；worker 日志在 `~/edgevisor_runs/smoke_20261018_e431cbd/opp14b5_20261020.log`。
- 设备日志：`nx1:/home/jetson/edgevisor_runs/smoke_20261018_e431cbd/api_five_default_20261019_r2b.api.log`（5 s 失败）、同目录 `r2c`（30 s 通过）、`nx1:/home/jetson/edgevisor_runs/api_recovery_20261019/`、`failover_retry_20261019.log`、`auto_rerun_20261019.trace.jsonl`。
- 本轮结束已清理测试 root/worker/API 进程和 19581/19584/19590 监听；未停无关服务，未改驱动、频率、网络、SSH 或安全设置。ROG 旧未提交目录和用户文件保持不动。
- 沿用用户授权的 git 管理：只提交本轮代码/文档，推当前开发分支；远端变化则接到远端之上，不强推，不夹带 `._*`，未经新请求不合 main。新会话读取交接后仍须复查实际状态，不能把历史空闲状态当当前事实。

### 6. 后续实验规划（2026-10-19）

目标：把论文从“机制正确性”补到“系统收益性”。主张写成条件式：**在同等内存/准备算力/带宽预算下，对真实 agent 任务和真实扰动，EdgeVisor 相对最强 proactive baseline 降低任务 p95 完成时间/提高截止期内成功率，且正常路径不劣化；收益来自 admission+JIT，而不是多准备 KV。**

**实验 A（主体完成，待大样本/均衡/bubble）：机会是否真实。**
- 单轮：3 类真实任务 × 3 任务，9 次工具调用；窗口 9.6 ms–691 ms，中位 473 ms（coding 430、search 636、SQL 353）；论文 1.096/1.20 s 中位未复现；查询选择性导致近两个数量级差异。
- 多轮：7 个 episode、19 次调用、2–4 次/episode；累计窗口 0.13–1.06 s；6/7 完成。
- 大测试：heavy matrix 3.08 s、topk 7.94 s、logparse 1.78 s；参考测试 1.5–4.6 s；合并微测试后范围 **0.13–7.94 s**。
- 14B/5 机：18/18/2/1/1 + context 1024 + 180 s 跑通；search 622 ms、SQL 链 530 ms、heavy topk **11.80 s**（2.1× W1），均完成；生成约 1.2 token/s。
- 未完成：远程工具延迟、14B 均衡性能配置、worker per-stage bubble、heavy matrix 的 14B 重跑、大样本 p95。
- 下一步：先补 bubble 埋点与均衡 14B，再扩样本；B 的窗口输入用 0.1–12 s 分布。
- 失败判据：14B/均衡配置下累计窗口仍 < 准备一个候选的成本 → 机会不足，需改 workload/claim。

**实验 B（A 之后，主实验）：收益 + 归因。**
- Arms：Boundary-Defer、KV-Transfer、Replay、**Pre-copy-all**、**Warm-spare**、EdgeVisor full、w/o admission、w/o JIT；所有臂同一内存/准备算力/带宽预算。
- 3 workloads × 3 扰动（compute clamp、link loss、worker removal）× 3 强度/持续时间 × ≥10 个配对 episode；先 5 reps pilot 估 effect size，再决定是否扩到 20。
- 主指标预注册：受扰任务 p95 完成时间，或截止期内成功率；副指标：p99 inter-token stall、F2FT、任务质量/成功率、能耗/task、峰值内存、prepared-but-unused。
- 分析：同任务/同扰动 trace 配对，随机化臂顺序，paired difference + 95% CI，保留全部失败/OOM/abort。
- 失败判据：Pre-copy-all 在多数 cell 不劣于 full → novelty 不成立；要么找 selection 的 crossover，要么把论文收缩为“pre-copy 足够，VDC 提供正确性/安全切换”。

**实验 C（B 之后）：边界与代价。**
- 另一模型家族/尺寸（至少 0.6B/3B/8B 之一）、2/3/5 节点、真实 Wi-Fi/蜂窝、30–60 min 热/功耗。
- 输出：收益 vs 额外内存/焦耳/task 的 Pareto 曲线；(故障持续时间, 剩余工作, 内存预算) 的 phase diagram，标出值得/不值得准备的区域。

**横贯要求：**
- F2FT 归因：拆开 84% residual；否则主结果仍是 0.006% 的组件优化。
- 统计：3 reps → 10+；0/5 故障 → 0/30；JIT 16 cells → 100+，calibration/test 分开。
- 先修 E2 +18 s、M1 90.3 vs 95.0、E8B patch 等证据链；所有臂冻结同一代码版本。
- 章节重排，不往当前 12 页稿硬塞。

当前状态：A 进行中；B/C 未开始；不改 `main`。
