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

更新：2026-10-18（Asia/Shanghai）。下面替代旧的 2026-10-06 演示交接；旧 PID、旧机器地址和旧 UI 自测状态不是当前事实，不要据此停止进程或启动演示。

### 1. 当前状态（Current state）

用户本轮要求：先不调整 ROG 安全配置，把 4 Jetson + 1 ROG 更新到 GitHub 最新版，实测 README/现有功能的 smoke。本轮测试和清理已完成；用户随后要求更新本文件。没有要求继续实现优化，也没有要求合并 main。

- 当前开发分支 `refactor/experiment-foundation`；测试引擎源码 `e431cbddf96c5784833093e7ce9aa9a0d43bac1c`。测试报告提交 `53fe98ea6004cb3501f55dbb80379dd08c5d482a` 已推送；五台源码均同步该提交，CLI/API 二进制与测试时相同，tracked tree 干净。本交接再以单独文档提交推进分支；不要为纯文档改动重编引擎。
- `main` 仍为 `bb97cab`，本轮未合并、未修改引擎/API/客户端源码。ROG 旧目录有未提交内容，保留未覆盖。
- SSH 别名：`nx1 nx2 nano1 nano2 rog`。五台新入口统一 `~/EdgeVisor-current/EdgeVisor/`，须先进入此目录再运行，Vulkan shader 有相对路径依赖。Jetson 默认 CUDA，ROG 默认 Vulkan；ROG 走现有 ZeroTier，不能拿旧有线地址推断可达性。
- 各设备记录：`~/edgevisor_runs/smoke_20261018_e431cbd/`。nx1、ROG 另有测试源码 e431cbd 的 `~/EdgeVisor-current-cpu/`；这是 CPU 对照工作树，未随纯文档提交更新。
- 完整报告：`docs/test_records/16_五设备新版功能Smoke_2026-10-18.md`。结论是“基础及受控配置可用，默认配置并非全通过”。

已完成的可核对结果：

- 五个 GPU 构建、nx1/ROG 两个纯 CPU 构建均有 CLI/API；每个配置的 10 项模型无关 C++ 检查通过。Python 25 项通过（工具输出已见，未单独留存该次 unittest stdout）。五台 0.6B Q40 默认单机均生成 20 token，不显式传线程/上下文参数。
- 两机显式/速度分层、四 CUDA 固定/auto、五机 CUDA+Vulkan 固定分层通过。四 CUDA auto 本次无真实迁移，不能算自动迁移提交验证。根端及尾端往返各两次真实提交、140 token 与参照一致；CUDA↔Vulkan 单层往返两次提交、92 token 一致、无重启。
- 根端 shadow 在完成 token 的受控边界注入掉线，113 token 一致、无重启；worker 前驱场景只有安全会话重启通过，未证明该位置热接管。普通中间/尾节点掉线可重启完成；position=0 预留加入通过；延迟采样超时恢复及重试上限通过；4096 prompt 接受、4097 明确拒绝。
- API 单机/四机根采样/四机尾采样/五机尾采样共 80 项请求检查通过，含预期 HTTP 400；五机用显式 30 秒预算。覆盖冷热历史一致、精确 KV frontier/复用、非法请求隔离、采样重置、SSE。3B 单机和四 Jetson 各一例真实自治 calculator，返回 126；四机用 30 秒预算，无强制动作/答案。
- 五机已有 14B Q40 完整模型/分词器哈希一致，固定 `14/14/2/5/5`、context 256、线程 1、IO/KV 30 秒，生成 20 token、零重启。不是 14B 默认 4096、auto/API/故障或性能验收。0.6B 五台、3B 四 Jetson 的完整文件哈希也已核对；未外网下载模型。

### 2. 已走过的失败与对照（Dead ends）

- 默认 `five_mixed_auto` 未通过：ROG 初始探测被作为预留设备，后实际加入 12 行 KV；position 22 附近尝试 23 层 CUDA→Vulkan 迁移，导出 529 行后连接失败、重启/启动回退，最终 exit 1。没有证据直接归因 OOM。
- `five_mixed_auto_bounded` 同时改为初始 `7/7/4/5/5`、单次最多 1 层、冗余边界 3 层、所有参与进程 IO/KV 30 秒，完成 108 token 和一次真实 Vulkan→CUDA 提交，无重启。这是多参数对照，不能归因到单一开关，不能说默认策略已修复。
- 默认 3B 四机尾采样 API 因 control ACK 超时退出；默认 0.6B 五机 API 前两次请求成功、第三次冷历史重放因 tail sampled token 超时退出。保持模型/拓扑，将根端 `DLLAMA_KV_ACK_TIMEOUT_MS` 和 `DLLAMA_IO_TIMEOUT_MS` 改为 30000 后相应完整 gate 通过；增加预算不等于修好异常恢复。
- ROG 从错误 cwd 跑失败、迁移提示词提前 EOS、无可分配非尾端 donor 的预留拓扑、未实际注入故障的钩子调试都已保留，不计成功。不要把退出 0、旧 `case_index.json` 或同名日志当验收依据；预期 exit 1 的容量/重试上限用例反而是通过。
- worker 掉线触发 buffered-send 歧义，按设计安全重启；不能把 token 一致说成无重启热接管。根端无重启用例使用单独链接的测试边界入口，部署生产二进制未改。

### 3. 未完成事项（Unresolved items）

1. 默认五机 auto：拆分核验可达性探测、实测/硬编码速度单位、大迁移准备/提交预算与失败回退；当前尚未确定单一根因。
2. API 可用性 bug：`NnPeerTimeoutException` 继承 runtime_error，控制 ACK/尾端采样默认 5000 ms，异常未被 API 恢复边界兜住，可能终止服务。需要安全错误响应、失效旧 lineage、重建/恢复 runtime；不要直接继续使用半完成流水线。
3. worker 无重启 shadow 尚未实机证明；14B 全功能、长时稳定性、并发多会话、论文性能/任务质量统计均未做。当前网络/文件缓存不受控，不可把 smoke 时间写成公平性能数据或调度速度表。
4. MobiSys 补强路线仍是 S0/S1 已有基础，S2–S7 未完成，详见 `docs/research/HANDOFF_2026-10-14.md`。同步 generation ACK 不是迁移 epoch fencing；S2 仍需真实 transaction/identity/frontier/closure、旧 ACK/重复/部分 payload/commit 故障拒绝。不要先给共享 GPU scratch 加后台线程。
5. 结束时 nano2 根盘约余 6.1 GiB、ROG 约余 5.5 GiB；后续长实验先复查容量，不擅自删除用户文件。yhs1 上次容器状态见旧 research 交接，本轮没有重新检查，不能断言现在空闲。

### 4. 第一项下一步（First next action）

当前用户只要求记录，不启动新实验或修复。下一轮若获准修复，第一步先核对 `git status`/GitHub 分支尖端，读取本报告及 `api_five_mixed`、`agent_four_tail` 失败日志，再检查 `EdgeVisor/src/dllama-api.cpp` 的请求/服务异常边界与 `EdgeVisor/src/nn/nn-network.hpp` 的 timeout 类型，确定超时后如何安全废弃 runtime。先解决 API 退出，再分别隔离默认 auto 的因素；每个引擎改动都 fresh build + 实机模型回归后 commit/push，不直接开始论文性能实验。

重跑入口（均会启动模型，先确认设备空闲，仅按下一次授权运行）：Mac 项目父目录的 `work/device_smoke_20261018/` 下执行 `python3 final_checks.py`（串行 API 30s/工具/故障/容量/后端）或 `python3 smoke_14b.py`（五机 14B 短测）。辅助模块为 `lab.py`、`smoke.py`、`api_smoke.py`、`edge_cases.py`；精确逐场景命令见 `cases/*.command.txt`。这些是本轮证据/控制脚本，不在 Git 中，不应在模型运行中编辑脚本或源码。

### 5. 证据、清理与约束（Optional context）

- Mac repo：`/Users/yhbian/Documents/Codex/2026-10-08/github-plugin-github-openai-curated-remote-2/work/EdgeVisor`。
- Mac 证据：`/Users/yhbian/Documents/Codex/2026-10-08/github-plugin-github-openai-curated-remote-2/work/device_smoke_20261018/`；`acceptance_summary.json` 是明确范围的 partial_pass 汇总，`artifact_manifest.json` 是 360 个留存文件的 SHA256/大小清单（生成后若新增文件需重建，不含 manifest 自身与 pycache）。包含命令、build、模型/源码/二进制身份、raw trace、API 请求响应、成功/失败/对照。不要删除失败试次。
- `*_final_audit.json` 核对五台本轮 owned 进程为空、测试/传输端口 19581–19584 已关闭；审计先于纯文档同步，故其中 HEAD 是 e431cbd。`*_final_sync.json` 才是随后同步 53fe98e、tracked tree 干净及已测二进制哈希不变的证据。无推理/worker/API 留跑，无临时模型传输服务；模型副本保留。ROG 最后无 compute 进程、15 MiB、40°C（瞬时利用率 9%），不是长期安全监控结论。
- 未停无关服务，未改驱动、频率、网络、SSH 或安全设置；旧 UI PID 不再是当前指令。用户要求暂不继续 ROG 安全配置，本交接不记录凭据/攻击原始详情。原有 ROG 未提交目录和用户文件保持不动。
- 沿用用户授权的 git 管理：只提交这次 `handsoff.md`，推当前开发分支；远端变化则接到远端之上，不强推，不夹带 `._*`，未经新请求不合 main。新会话读取交接后仍须复查实际状态，不能把历史空闲状态当当前事实。
