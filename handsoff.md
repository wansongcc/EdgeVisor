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

更新时间：2026-09-28

### 正在做什么

EdgeVisor 是改过的 dllama，按流水线把一层层模型摊到多台机器上。现在用的是五台机器上已有的 Qwen3-14B（`qwen3_14b_q40`，40 层），不用 Llama 3.2 3B。

当前这件事是五机 smoke：nano1、nano2、nx1、nx2 做 CUDA worker，rog 做 CPU root，确认五台能连上并正常生成 token。这不是中途把一台设备加入的测试，五台从一开始都在线。

这次交接之前，用户要求的就是这个 smoke，以及把交接写进仓库。五机 smoke 没有生成出 token，进程已经停掉。用户还没有在此之后布置新的改代码任务。

### 设备

控制面是 ZeroTier，只用来 SSH。激活和 KV 走有线数据面 `192.168.137.0/24`。不要从数据面 SSH。故障注入会把数据网卡 down 掉。

从这台 Windows 开发机可以用 SSH 别名登录，BatchMode 公钥可用。Jetson 用户是 `jetson`，rog 用户是 `cc`。

| 别名 | 控制面 | 数据面 | 代码目录 | 这次的角色 |
| --- | --- | --- | --- | --- |
| nano1 | 10.47.107.34 | 192.168.137.18 | `/home/jetson/cc/EdgeVisor` | CUDA worker |
| nano2 | 10.47.215.50 | 192.168.137.16 | 同上 | CUDA worker |
| nx1 | 10.47.145.51 | 192.168.137.13 | 同上 | CUDA worker |
| nx2 | 10.47.235.49 | 192.168.137.15 | 同上 | CUDA worker |
| rog | 10.47.72.162 | 192.168.137.31 | `/home/cc/EdgeVisor` | CPU root（avx2） |

二进制是仓库里的 `EdgeVisor/dllama`。Jetson 编译用 `DLLAMA_CUDA=1 make dllama`。不要跑裸的 `make`，默认目标是 clean。rog 用 `make dllama`。

模型目录：Jetson 是 `/home/jetson/cc/models/qwen3_14b_q40/`，rog 是 `/home/cc/models/qwen3_14b_q40/`。权重文件 `dllama_model_qwen3_14b_q40.m` 约 11 GB，tokenizer 是同目录的 `dllama_tokenizer_qwen3_14b_q40.t`。词表大小 151669 对模型 151936 的警告是预期的，不是失败。

这次 smoke 的 worker 顺序是 nano1、nx2、nano2、nx1，端口都是 9999。自动切分打出来的是 `1@1*1@1*1@22*1@1*1@15`：rog 1 层、nano1 1 层、nx2 22 层、nano2 1 层、nx1 15 层。

### 这次做了什么

没有在这次交接里新写功能代码。做了运行实验，然后写这份文件。

设备中途加入此前已经跑通。Qwen3-14B 解码时把预留的 nx2（`192.168.137.15:9999`）拉进流水线，大约在 pos=43 加入，带 215 行 KV，后面的句子仍然连贯。跨过加入点的句子是 “epsilon for a long time”，后面的希腊字母续写是对的。`zeta` 被分成 `z` 和 `eta` 是 BPE，不是 KV 损坏。

当时低序号 worker 会去拨 `127.0.0.1`。修复是让加入包带上对端数据面地址 `peerHost`。这套改动已经在远程提交 `fbcc67d`（Stop replaying a finished token when a covered stage drops.）里，包括 `peerHost`、executor 里同一次错误只登记一次、failover 增加 `replayActivation`，以及 `test_pp_tp.cpp` 里 `NnStageDef` 改成 `{}` 初始化。写这份文件时，nx1 的 HEAD 还停在 `7adcee8`，工作区里那 6 个源文件的内容和 `fbcc67d` 相同，只是 HEAD 还没快进；`test_pp_tp.cpp` 当时仍是旧内容。这份交接提交接在 `fbcc67d` 之上。另外四台机器未必已经是这个提交，动手前先看 `git status`。

五机一起在线的 smoke 没有跑完，这次没有为它改代码。四台 worker 都连上了，40 层权重加载成功，提示词 `The capital of France is` 已经打印出来。随后 nx2 把 nano1 标成离线，日志是 `deactivated node=1 socket=1`，以及 `fast-path deadNode=1 ejectedStage=1 targetStage=0 layers=[1,2)`。当时 `ss` 显示 nano1 和 nx2 之间没有 TCP 连接，rog 到四台 worker、以及其它 worker 之间的连接还在。nx2 建 22 层执行器时有一次 `NvMapMemAlloc ... error 12`，随后仍打印了权重加载成功。rog 以大约 300% CPU 空转到约 9 分钟，一个 token 都没有。五台 `dllama` 都已停掉，并确认进程不在。

日志还在机器上：rog 的 `/tmp/smoke-root.log`，四台 Jetson 的 `/tmp/smoke-worker.log`。

root 命令是在 `/home/cc/EdgeVisor/EdgeVisor` 里跑的，没有加 `--info`，也没有设 memory limit：

```
./dllama inference --prompt "The capital of France is" --steps 16 --model /home/cc/models/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m --tokenizer /home/cc/models/qwen3_14b_q40/dllama_tokenizer_qwen3_14b_q40.t --buffer-float-type q80 --nthreads 4 --max-seq-len 128 --workers 192.168.137.18:9999 192.168.137.15:9999 192.168.137.16:9999 192.168.137.13:9999
```

这次用的二进制是 2026-09-27 编的：Jetson 大约 16:04–16:05，rog 大约 01:04。源码内容和 `fbcc67d` 的主体一致。

### 仓库

- 远程：`git@github.com:wansongcc/EdgeVisor.git`
- 分支：`feat/device-fault-tolerance`
- 这次只提交 `handsoff.md`
- 不要提交 `EdgeVisor/src/._*` 和 `EdgeVisor/src/nn/._*`。那是 macOS 拷贝带出来的附属文件

### 下一步

用户若说继续，就从这次失败的五机 smoke 接着查：第一步时为什么 nano1 和 nx2 之间的连接断了，以及怎样让五台连上并吐出连贯 token。先读上面的两处日志。不要改用 3B。原因没查清之前，不要把同一次实验再挂成长时间的 CPU 空转。

### 操作时注意

- 这台开发机的 shell 是 PowerShell。发给远端的管道和引号放进单引号，否则会被本地拆开。
- `ssh -f` 或远端 `setsid ... &` 之后，本地 SSH 经常不退出，即使远端已经打出结果。远端进程起来之后，关掉卡住的本地 SSH。不要留着没用的终端，也不要误关用户自己的交互终端。
- root 第一次连接 worker 时，探测会先做一次真正的 TCP accept。worker 会看到 `handshake ended: Socket closed`，然后重新监听。若报 Socket offline，等 worker 重新 Listening 后再试一次。
- 生成算不算成功，要看 token 语义，不能只看进程没崩。
