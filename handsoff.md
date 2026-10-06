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

2026-10-06 15:05 CST。五机可视化演示脚本还没准备好，自测没有通过。不要告诉用户这个脚本可以手动跑。

现在在做的事：准备一个用户稍后在 Windows 上手动跑的五机 14B 演示。真实推理过程中，根节点终端要看到设备池、设备上线、设备下线、设备劣化、任务迁移、影子 KV。五台都要用。脚本还没跑通。

工作状态：失败后停着。没有正在跑的自测，不要再开一份长推理。

机器上还留着什么：

- nx2 pid `1072240`，已跑约 1 天 17 小时。命令是 `./dllama worker --port 18091 --model /home/jetson/cc/models/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m`。状态 S，父进程 1，停在 `inet_csk_accept`。`0.0.0.0:18091` 正在听。日志 `/tmp/ui_demo_worker.log` 只有 189 字节，是 CUDA 设备行，里面没有 `Listening on`。没有 `gpu_hog`。用户说「继续」之前不要停它，也不要再起一个 worker。
- nx1 没有 `dllama`，没有 tmux 会话 `evdemo`。`docker ps -q` 为空。
- nano1、nano2、笔记本都没有 `dllama`。

已经定下来的结论：

- 编排在 Windows：`C:\Users\wuwzh\Desktop\Yanhui\20260925-1008\B01Proj\ui_live_demo.ps1`。这个目录不是 git 仓库。根节点命令在 nx1：`/home/jetson/cc/edgevisor_fresh/scripts/ui_live_demo_root.sh`。nx1 不能 SSH 到其他机器，编排不能写成从 nx1 往外 SSH。
- 控制面是 ZeroTier `10.47.x`，激活和 KV 走 `192.168.137.0/24`。不要在数据面 SSH，不要把网卡 down。远程 `--ratios` 要加引号。PowerShell 会吃掉远程双引号；管道符必须留在 ssh 参数里面。不要用 `pgrep -af dllama`。停进程用 `pkill -x dllama` 或 `pkill -x gpu_hog`。
- 顺序和地址：nx1 根 `192.168.137.13`，nx2 `.15`，笔记本 `.31`，nano2 `.16`，nano1 `.18`，端口都是 `18091`。14B 在 Jetson 是 `/home/jetson/cc/models/qwen3_14b_q40/`，笔记本是 `/home/cc/models/qwen3_14b_q40/`。演示先起 nx2、笔记本、nano2，把 nano1 留成唯一预留槽。
- 2026-10-04 自测结果是 `timeline: FAIL pool`。面板画出来了，预留设备被画成离线，日志里没有 `reserved`。`EdgeVisorUiDevice` 和 `EdgeVisorUiStage` 的 bool 当时没有默认值。同一次后来在 `Tokenizer::encode` 的 `strLen == 0` 断言退出，没走到上线、迁移、下线。
- 2026-10-04 21:43 头文件已写成 `bool offline = false` 等默认值。21:56 nx1 的 `EdgeVisor/dllama` 重新编过，`make dllama -j4` 退出码 0。worker 没重编。nx2 的 `dllama` 仍是 2026-09-30 17:44。
- `/tmp/tok_try` 对 14B tokenizer 测过：裸句 `What is the capital of France?` 和 ChatML 包装后的字符串都能 encode。这只说明分词，不能说明五机演示已经通过。
- 随后一次自测是 `timeline: FAIL listen nx2`。脚本在 90 秒里等日志出现 `Listening on`。现在这个 worker 已经在听 `18091`，日志里仍然没有这几个字。用这行日志判断“已在听”会误判。
- 更早一份根日志是 `ratios=1@40*1@0*1@0*1@0*1@0`，当时 worker 还没起来。脚本后来要求 `ratios=` 里的 `1@0` 正好一个。那次自测没走到这里。
- 上次跑通的 14B 切分大约 `1@16*1@14*1@1*1@6*1@3`，冗余 `0->1 0,0 1->2 0,3 2->3 3,0 3->4 0,2`。日志 `/tmp/edgevisor_14bfix4_root.log`。中间层的影子盖不住整段，杀掉 nano2 后 failover 常会 `partial-cover`。画面仍应先标出离线。
- UI 只在根节点，靠 `--ui`。黄是预留，蓝是离线，红是劣化和迁移源，绿是迁入，品红是影子行。

下一步：用户说「继续」时再做。先停 nx2 的 pid `1072240`，把“已在听”改成查端口，然后重跑 `powershell -ExecutionPolicy Bypass -File C:\Users\wuwzh\Desktop\Yanhui\20260925-1008\B01Proj\ui_live_demo.ps1 -SelfTest`。只有打出 `timeline: OK` 才能告诉用户脚本好了，以及从 `B01Proj` 怎么手动跑。没说继续，不要开这次长推理。

约束：不要合到 main，不要开阶段 3，不要强推。不要提交下面这些未提交的 UI 改动，除非用户明确要求。不要提交 `._*`。

相关 commit：HEAD 是 `8e9c03d`，和 `origin/refactor/productize` 一致。这次交接之前没有新的功能提交。未提交、且不要带进这次提交的文件：

- 已修改：`EdgeVisor/Makefile`、`EdgeVisor/src/app.cpp`、`EdgeVisor/src/app.hpp`、`EdgeVisor/src/dllama.cpp`、`EdgeVisor/src/dynamic/dynamic_tpot.cpp`
- 未跟踪：`EdgeVisor/src/terminal_ui.cpp`、`EdgeVisor/src/terminal_ui.hpp`、`EdgeVisor/src/terminal_ui_preview.cpp`、`EdgeVisor/ui-preview`、`scripts/ui_live_demo_root.sh`

这次只提交 `handsoff.md`。
