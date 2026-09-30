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
2026-09-30 17:48 CST。自动冗余分层已经写进 `refactor/productize` 并推送。五机 14B 的错 token 修完，重新跑通一句完整句子。nx1、nx2、nano1、nano2、笔记本上都没有 `dllama`。`18091` 没有在听。没有容器。

现在在做的事：阶段 2 收尾里的自动冗余。主切分先定，每条边界左右两边各自用剩余显存放副本。一次运行里不再加副本。调度步长跟已放好的深度走。显式 `--runtime-redundant-boundary-layers` 仍是旧的统一跨度。

已经定下来的结果：

- 副本按常驻层计价：`--auto` 的 KV 是 F32，不是同步字节。70% 空闲、留一层。上下文用 `--max-seq-len`，否则用模型文件的 seqLen。
- 0.6B 两机（nx1+nx2，不写 `--ratios`）打出 `redundancy: 0->1 5,21`，短生成是连贯的思考。步长跟着深度，没有波动也会在开头窗口挪层。
- 14B 五机原生上下文，顺序 nx1、nx2 `192.168.137.15`、笔记本 `192.168.137.31`、nano2 `192.168.137.16`、nano1 `192.168.137.18`，端口 `18091`。切分大约 `1@16*1@14*1@1*1@6*1@3`，冗余 `0->1 0,0  1->2 0,3  2->3 3,0  3->4 0,2`。两台 NX 的主层已经吃满常驻预算，所以 NX 之间是 `0,0`。
- 修过的三件事：满权重切分收到常驻层上限；对端 socket 没了就抛 `Socket offline`，还没出字就走启动退让，出过字仍走会话重开；末级采样只加本 stage 自己的词表切分。之前把五个 stage 的词表加在一起，steps 48 解出 `token=456182 vocab=151669`。
- 修好后再跑：`--steps 48`，提示 `What is the capital of France?`，退出码 0。正文是思考过程，说到法国首都是 Paris。`tokens/s: 2.98 (335.39 ms/tok)`。日志 `/tmp/edgevisor_14bfix4_root.log`（nx1）。有一行 `reject pp command route=4->3 reason=target stage lacks provisioned layer 37`，那是步长想搬没有副本的层，句子没有坏。更早的失败日志还在：`/tmp/edgevisor_14bfix_root.log`、`/tmp/edgevisor_14bfix2_root.log`、`/tmp/edgevisor_14bfix3_root.log`。
- Vulkan 报的是设备本地堆总量，不是空闲。笔记本上的副本可能偏乐观。这次没有单独改。

源码提交 `261a14ec9372228a50ded9904dfeb1876fc44efc`，作者 Yanhui。说明是 Keep automatic overlap inside real KV memory and stop a dead peer from inventing tokens. 9 个文件：`EdgeVisor/src/app.cpp`、`app.hpp`、`device_profile.hpp`、`dllama.cpp`、`llm.cpp`、`nn/nn-network.cpp`、`nn/nn-network.hpp`、`test/test_device_profile.cpp`、`tokenizer.cpp`。推送 `5553911..261a14e` 到 `origin/refactor/productize`（`git@github.com:wansongcc/EdgeVisor.git`）。

五台二进制都是 2026-09-30 17:43–17:45 编的这份源码。只有 nx1 `/home/jetson/cc/edgevisor_fresh` 的 git 在 `261a14e`。nx2、nano1、nano2、笔记本 `/home/cc/edgevisor_fresh` 的 git HEAD 仍是更早的提交（笔记本是 `45deaab`），源码是从 nx1 拷过去的，不是 git pull。笔记本只链 Vulkan。

模型：14B 在 Jetson 是 `/home/jetson/cc/models/qwen3_14b_q40/`，笔记本是 `/home/cc/models/qwen3_14b_q40/`。worker 用 `setsid ./dllama worker --port 18091 --model <绝对路径>`。

不要合到 main。不要开阶段 3。不要做名单外的中途加入。不要提交 `._*`。README 仍是旧的，这次没改。

下一步：用户说「继续」之前不要开新实验。没有还在跑的进程。
