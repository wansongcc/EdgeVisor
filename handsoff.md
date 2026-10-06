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

2026-10-06 16:03 CST。编排脚本已经放进 nx1 的仓库。自测没有重跑。

现在在做的事：五机 14B 可视化演示。脚本还没跑通，不要告诉用户可以手动跑。

工作状态：失败后停着。没有正在跑的自测，不要再开一份。

机器上还留着什么：

- nx2 pid `1072240`，已跑约 1 天 18 小时。命令是 `./dllama worker --port 18091 --model /home/jetson/cc/models/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m`。`0.0.0.0:18091` 正在听。日志 `/tmp/ui_demo_worker.log`。没有 `gpu_hog`。用户说「继续」之前不要停它，也不要再起一个 worker。
- nx1 没有 `dllama`，没有 tmux 会话 `evdemo`。
- nano1、nano2、笔记本都没有 `dllama`。

已经定下来的结论：

- 编排脚本的仓库副本是 `/home/jetson/cc/edgevisor_fresh/scripts/ui_live_demo.ps1`。本机 Windows 同一份在 `C:\Users\wuwzh\Desktop\Yanhui\20260925-1008\B01Proj\ui_live_demo.ps1`。两边内容这次是对齐的。改脚本改仓库这份。
- 这个 ps1 必须在有 SSH 别名 `nx1`、`nx2`、`nano1`、`nano2`、`rog` 的 Windows 上跑。nx1 不能 SSH 到其他机器，不要在 nx1 上执行它。
- 根节点命令仍是 `/home/jetson/cc/edgevisor_fresh/scripts/ui_live_demo_root.sh`。它在磁盘上，还没提交。
- `Listening on` 在 `EdgeVisor/src/nn/nn-network.cpp` 的 `createServerSocket`，`printf` 之后没有 `fflush`。stdout 进文件时是全缓冲，worker 停在 accept 时这行不会进日志。nx2 这个 worker 就是这样：端口在听，日志里没有这几个字。下一步改仓库里的 ps1，改成查 `18091` 端口，不要改这行 `printf`。
- 顺序不变：nx1 根 `192.168.137.13`，nx2 `.15`，笔记本 `.31`，nano2 `.16`，nano1 `.18`，端口 `18091`。先起 nx2、笔记本、nano2，nano1 留成唯一预留槽。
- 2026-10-04 自测是 `timeline: FAIL pool`，随后是 `timeline: FAIL listen nx2`。UI 那批改动仍未提交。上次跑通的 14B 日志仍是 `/tmp/edgevisor_14bfix4_root.log`。

下一步：用户说「继续」时，先停 nx2 pid `1072240`，再改仓库里的 `scripts/ui_live_demo.ps1`，把「已在听」改成查端口。改完拷回 Windows 那份再跑 `powershell -ExecutionPolicy Bypass -File C:\Users\wuwzh\Desktop\Yanhui\20260925-1008\B01Proj\ui_live_demo.ps1 -SelfTest`。只有 `timeline: OK` 才能告诉用户脚本好了。没说继续，不要开这次长推理。

约束：不要合到 main，不要开阶段 3，不要强推。不要提交 UI 源码、`EdgeVisor/ui-preview`、`scripts/ui_live_demo_root.sh`，除非用户明确要求。不要提交 `._*`。

相关 commit：上一份交接是 `0c72489`。这次用户明确要求把编排脚本放进仓库，所以这次提交带上 `handsoff.md` 和 `scripts/ui_live_demo.ps1`。不要带上下面这些：

- 已修改：`EdgeVisor/Makefile`、`EdgeVisor/src/app.cpp`、`EdgeVisor/src/app.hpp`、`EdgeVisor/src/dllama.cpp`、`EdgeVisor/src/dynamic/dynamic_tpot.cpp`
- 未跟踪：`EdgeVisor/src/terminal_ui.cpp`、`EdgeVisor/src/terminal_ui.hpp`、`EdgeVisor/src/terminal_ui_preview.cpp`、`EdgeVisor/ui-preview`、`scripts/ui_live_demo_root.sh`
