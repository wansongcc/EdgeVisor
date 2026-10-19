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

用户本轮要求：先修复已有问题和 bug，过程中做好 git 管理。已完成 API 生成失败恢复、等待预算修复和启动重连加固；没有合并 `main`，没有改 ROG 安全配置。

- 当前开发分支 `refactor/experiment-foundation`，代码提交 `4032df5`（已推送）。本交接随后作为单独文档提交推进分支。`main` 仍为 `bb97cab`。
- 改动：`dllama-api.cpp` 捕获生成期 timeout/offline/session-restart/worker-socket/executor 异常，返回 503（流式则 SSE error）并展开 runtime 重建；客户端写失败不再混入推理故障；HTTP 状态短语修正；固定拓扑 KV/control-ACK/尾采样默认预算 5000→30000 ms；session restart 后最多 10×500 ms 等待 worker relisten。
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

### 4. 第一项下一步（First next action）

先核对 `git status`、`git ls-remote` 和五台实际 HEAD/二进制哈希；本轮只有 nx1 是 `4032df5`。若继续修 bug，优先复现默认五机 auto 并保留 worker 退出码与 `dmesg`，确认 529 行导出后 node2 是真实断链还是误判；若是启动竞态，验证 relisten 重试是否覆盖。若转向论文主线，按 `docs/research/HANDOFF_2026-10-14.md` 做 S2 最小迁移事务。每个引擎改动都要 fresh build + 实机模型回归后 commit/push，不直接开始性能实验。

重跑入口：Mac 项目父目录 `work/device_smoke_20261018/` 下新增 `api_recovery_20261019.py`、`api_five_default_20261019.py`、`failover_retry_20261019.py`；原有 `final_checks.py`、`smoke_14b.py`、`lab.py`、`smoke.py`、`api_smoke.py` 仍在。精确逐场景命令见 `cases/*.command.txt`。这些是证据/控制脚本，不在 Git 中，运行中不要编辑源码或脚本。

### 5. 证据、清理与约束（Optional context）

- Mac repo：`/Users/yhbian/Documents/Codex/2026-10-08/github-plugin-github-openai-curated-remote-2/work/EdgeVisor`。新测试记录：`docs/test_records/17_API生成失败恢复与等待预算_2026-10-19.md`。
- 设备日志：`nx1:/home/jetson/edgevisor_runs/smoke_20261018_e431cbd/api_five_default_20261019_r2b.api.log`（5 s 失败）、同目录 `r2c`（30 s 通过）、`nx1:/home/jetson/edgevisor_runs/api_recovery_20261019/`、`failover_retry_20261019.log`、`auto_rerun_20261019.trace.jsonl`。
- 本轮结束已清理测试 root/worker 进程和 19581/19584 监听；未停无关服务，未改驱动、频率、网络、SSH 或安全设置。ROG 旧未提交目录和用户文件保持不动。
- 沿用用户授权的 git 管理：只提交本轮代码/文档，推当前开发分支；远端变化则接到远端之上，不强推，不夹带 `._*`，未经新请求不合 main。新会话读取交接后仍须复查实际状态，不能把历史空闲状态当当前事实。
