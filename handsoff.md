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

五机 14B smoke 已经跑通。rog 做 CPU root，四台 Jetson 做 CUDA worker，从一开始都在线。模型仍是 Qwen3-14B Q40，没有改用 3B。

上一份交接里的失败原因已经查清，并且用改过默认超时的二进制复现通过。`nn-network.cpp` 的改动还没提交。五台 `dllama` 在跑完后已经退出。

### 失败原因

nano1 和 nx2 的连接一开始是通的。四台 worker 都完成了 mesh，40 层权重都加载成功，而且都收到了预填控制包 `Batch=12, Pos=0`。断线发生在第一次前向期间，不是建连失败。

默认 TCP keepalive 是空闲 1 秒、间隔 1 秒、3 次，用户超时约 4 秒。自动切分把 22 层放在 nx2 上。nx2 在算这 22 层时，它和 nano1 之间已经没有新数据，内核把这条空闲连接判成对端死亡。nx2 日志是 `deactivated node=1 socket=1`，接着 `fast-path deadNode=1 ejectedStage=1 targetStage=0 layers=[1,2) replay=0`。后面是连锁：nx1 看到 nano2 关闭，root 看到 nx1 关闭，然后会话重开，rog 空转，一个 token 都没有。

每台 worker 日志开头的 `handshake ended: Socket closed` 是 root 探测时的那一次 accept，接着会重新监听。那不是这次断线。

nx2 建执行器时仍有一次 `NvMapMemAlloc ... error 12`。权重随后加载成功，这次复跑也没有再因为这个报错停住。

旧日志还在：rog `/tmp/smoke-root.log`，四台 Jetson `/tmp/smoke-worker.log`。

### 这次怎么通过的

`setTcpKeepAlive` 的默认改成空闲 30 秒、间隔 10 秒、6 次，用户超时 180 秒。环境变量名字没变，显式设置仍然覆盖默认值。五台都已用这份源码重新编译：Jetson 是 `make dllama DLLAMA_CUDA=1`，rog 是清掉旧的 `.o` 之后 `make dllama`（直接增量链接会撞上上次 Vulkan 留下的 `app.o`）。

确认跑没有设置 `DLLAMA_TCP_*`。worker 顺序和端口与失败的那次相同：`192.168.137.18:9999 192.168.137.15:9999 192.168.137.16:9999 192.168.137.13:9999`，`--steps 28`，其余标志与上一份交接里的 root 命令相同。root 日志没有 `deactivated`。nx2 的 `/tmp/ft_smoke5b/w.log` 里 `deactivated` 次数是 0。生成文本是：

`<think>Okay, the user asked, "The capital of France is..." I`

日志：rog `/tmp/ft_smoke5b/root.log`，四台 Jetson `/tmp/ft_smoke5b/w.log`。

更早一次用环境变量把超时放到 180 秒、`--steps 16` 的跑也出了 `<think> Okay,`，日志在 `/tmp/ft_smoke5/`。那次二进制还是旧默认值。

### 仓库

- 远程：`git@github.com:wansongcc/EdgeVisor.git`
- 设备上的分支：`feat/device-fault-tolerance`，HEAD 曾是 `f7c52c4`。五台的 `EdgeVisor/src/nn/nn-network.cpp` 现在是未提交的超时改动。
- 开发机上的 `/tmp/edgevisor_ft` 在 `refactor/productize`，同一处改动也还没提交。
- 不要提交 `EdgeVisor/src/._*` 和 `EdgeVisor/src/nn/._*`。

### 下一步

用户若要留下这个修复，再提交 `nn-network.cpp`。没有新的失败要接着查。不要为了这次 smoke 再把超时改回 4 秒。

### 操作时注意

- 这台开发机的 shell 是 PowerShell。发给远端的管道和引号放进单引号，否则会被本地拆开。
- `ssh -f` 或远端 `setsid ... &` 之后，本地 SSH 经常不退出，即使远端已经打出结果。远端进程起来之后，关掉卡住的本地 SSH。不要留着没用的终端，也不要误关用户自己的交互终端。
- root 第一次连接 worker 时，探测会先做一次真正的 TCP accept。worker 会看到 `handshake ended: Socket closed`，然后重新监听。若报 Socket offline，等 worker 重新 Listening 后再试一次。
- 生成算不算成功，要看 token 语义，不能只看进程没崩。
