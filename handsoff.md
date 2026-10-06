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

## 可变交接

2026-10-06 15:56 CST。用户说了「继续」。自测没有重跑。Windows 上的编排脚本改不到，nx2 的 worker 没有停。

现在在做的事：五机 14B 可视化演示自测。卡在脚本用日志判断「已在听」。

工作状态：失败后停着。没有正在跑的自测，不要再开一份。

机器上还留着什么：

- nx2 pid `1072240`，已跑约 1 天 17 小时。命令仍是 `./dllama worker --port 18091 --model /home/jetson/cc/models/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m`。状态 S。`0.0.0.0:18091` 正在听。日志 `/tmp/ui_demo_worker.log`。没有 `gpu_hog`。这次没有停它，因为脚本还没改，停了也跑不了自测。不要再起一个 worker。
- nx1 没有 `dllama`，没有 tmux 会话 `evdemo`。
- nano1、nano2、笔记本都没有 `dllama`。
- `10.47.229.41` 是 2026-10-04 21:33 连上 nx1 的那台（`last`）。15:56 ping 不通，22、445、3389、5985、135 都不开。这台 Mac `10.47.240.59` 和 nx1 上都没有 `ui_live_demo.ps1`。

已经定下来的结论：

- `Listening on` 仍在 `/home/jetson/cc/edgevisor_fresh/EdgeVisor/src/nn/nn-network.cpp` 的 `createServerSocket`，`printf` 之后没有 `fflush`。stdout 重定向到文件时是全缓冲，worker 停在 accept 时这行不会进日志。用日志判断已在听会误判。下一步改脚本查端口，不要改这行 `printf`。
- 编排脚本仍是 `C:\Users\wuwzh\Desktop\Yanhui\20260925-1008\B01Proj\ui_live_demo.ps1`。根节点命令仍是 `/home/jetson/cc/edgevisor_fresh/scripts/ui_live_demo_root.sh`。nx1 不能 SSH 到其他机器。
- 顺序不变：nx1 根 `192.168.137.13`，nx2 `.15`，笔记本 `.31`，nano2 `.16`，nano1 `.18`，端口 `18091`。演示先起 nx2、笔记本、nano2，nano1 留成唯一预留槽。
- 2026-10-04 自测是 `timeline: FAIL pool`，随后是 `timeline: FAIL listen nx2`。UI 那批改动仍未提交。上次跑通的 14B 日志仍是 `/tmp/edgevisor_14bfix4_root.log`。

下一步：等 `10.47.229.41` 回到网上，或用户把 `ui_live_demo.ps1` 放到当前机器能改的路径。然后先停 nx2 pid `1072240`，把「已在听」改成查 `18091` 端口，再跑 `powershell -ExecutionPolicy Bypass -File C:\Users\wuwzh\Desktop\Yanhui\20260925-1008\B01Proj\ui_live_demo.ps1 -SelfTest`。只有 `timeline: OK` 才能告诉用户脚本好了。在此之前不要开长推理。

约束：不要合到 main，不要开阶段 3，不要强推。不要提交未提交的 UI 改动。不要提交 `._*`。

相关 commit：HEAD 是 `0c72489`，和 `origin/refactor/productize` 一致。这次没有功能代码改动。未提交、且不要带进这次提交的文件：

- 已修改：`EdgeVisor/Makefile`、`EdgeVisor/src/app.cpp`、`EdgeVisor/src/app.hpp`、`EdgeVisor/src/dllama.cpp`、`EdgeVisor/src/dynamic/dynamic_tpot.cpp`
- 未跟踪：`EdgeVisor/src/terminal_ui.cpp`、`EdgeVisor/src/terminal_ui.hpp`、`EdgeVisor/src/terminal_ui_preview.cpp`、`EdgeVisor/ui-preview`、`scripts/ui_live_demo_root.sh`

这次只提交 `handsoff.md`。
