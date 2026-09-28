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

EdgeVisor 产品化，分支 `refactor/productize`。阶段 0、阶段 1 用户已经认可通过。阶段 2 的三件代码已经写完，编过，也用 0.6B 跑通了两机，但这些代码还没提交。

用户明确放下的一件事先不要做：名单里没有的机器，不能在推理中途插入执行图。上线仍然只填开跑时 `--workers` 里预留的槽。用户说这件事留到以后做 mDNS 再考虑。

### 仓库和机器

- 远程：`git@github.com:wansongcc/EdgeVisor.git`
- 当前开发分支：`refactor/productize`。已推送的 HEAD 是 `45deaab`（默认日志收干净）。
- 开发副本：`/tmp/edgevisor_ft`，分支 `refactor/productize`，没有 `origin` 这个 remote。推送用 `git push git@github.com:wansongcc/EdgeVisor.git HEAD:refactor/productize`。不要 `--set-upstream`，不要强推。
- 这份副本里阶段 2 的改动是未提交的。`git status` 会看到 `app.cpp`、`app.hpp`、`device_profile.cpp`、`device_profile.hpp`、`dllama.cpp`、`nn-cuda.cu`、`nn-cuda.hpp`、`nn-network.hpp`、`nn-vulkan.cpp`、`nn-vulkan.hpp`、`test_device_profile.cpp`，以及未跟踪的 `EdgeVisor/src/nn/vulkan/profile_gemv.comp`。不要把这些和 `handsoff.md` 一起提交，除非用户明确要求提交代码。
- 提交作者用环境变量，不要改 git config：`GIT_AUTHOR_NAME=Yanhui`、`GIT_AUTHOR_EMAIL=fromthefox@icloud.com`，committer 同样。
- nx1 上的旧目录 `/home/jetson/cc/EdgeVisor` 是 `feat/device-fault-tolerance`，和这条产品化线不是同一份工作区。不要在那个脏目录里继续改产品化代码。

五台设备，控制面走 ZeroTier，数据面是 `192.168.137.0/24`。SSH 走 ZeroTier，不要走数据面。

| 机器 | SSH | 数据面 | 新目录 |
| --- | --- | --- | --- |
| nx1，CUDA，root | `jetson@10.47.145.51` | `192.168.137.13` | `/home/jetson/cc/edgevisor_fresh` |
| nx2，CUDA | `jetson@10.47.235.49` | `192.168.137.15` | `/home/jetson/cc/edgevisor_fresh` |
| nano1，CUDA | `jetson@10.47.107.34` | `192.168.137.18` | `/home/jetson/cc/edgevisor_fresh` |
| nano2，CUDA | `jetson@10.47.215.50` | `192.168.137.16` | `/home/jetson/cc/edgevisor_fresh` |
| 笔记本，Vulkan | `cc@10.47.72.162` | `192.168.137.31` | `/home/cc/edgevisor_fresh` |

阶段 2 的源码已经 rsync 进 nx1、nx2、笔记本的新目录，并且 `make dllama` 成功。nano1、nano2 的新目录还停在已推送的 `45deaab`，没有这批未提交改动。

密钥登录可用，`ssh -o BatchMode=yes`。不要重新创建或打印密码。

### 阶段 0 和阶段 1

阶段 0 已在 `b9b029b`。快路径要求 token 一致；session restart 只要求句子连贯。中间层有缓存就跳过，没有缓存或者最后一台挂了就整句重开。API 只绑 `127.0.0.1`。这些不要退回去。

阶段 1 用户在 2026-09-28 认可通过。已推送的相关提交从 `6b620fb` 到 `45deaab`：`make dllama` 自动识别 CUDA/Vulkan；`--backend auto`；Q40 默认 q80；实验室地址走 speed-pack；默认日志只有设备、buffer、切分、加载、生成文本和 tokens/s。

五台都从空目录克隆过 `refactor/productize` 到上面的新目录，`make dllama` 成功。Jetson 自动选 CUDA。笔记本自动选 RTX 3060，不是核显。huggingface.co 从 nx1 超时，0.6B 是用 hf-mirror 下的。

单机 nx1，Qwen3-0.6B，日志 `/tmp/edgevisor_fresh_quiet.log`：

```
device: backend=cuda threads=8 ...
buffer: q80
topology: single device
loading weights
<think>
Okay,
tokens/s: 18.15 (55.10 ms/tok)
```

两机 nx1 + nx2，不传 `--ratios`，日志 `/tmp/edgevisor_fresh_two.log`：`topology: speed-pack classes=nx,nx ratios=1@22*1@6`，连贯句子，`tokens/s: 8.90`。当时的 worker 已经停掉。

### 阶段 2 已写进工作区、还没提交的内容

1. 层数上限取实验室表和 `estimateLayerCap(空闲显存, 当前模型)` 的较小值。毫秒/层仍是 NX 5.8、Nano 7.4、笔记本 8.3。显存更大时表里的上限还在，所以 0.6B 两机仍然是 `1@22*1@6`。
2. 陌生地址不再用一次小 GEMM 换算。在已经选定的 CUDA 或 Vulkan 上计时一层解码的七个投影（Q、K、V、O、gate、up、down）。Vulkan 着色器是 `EdgeVisor/src/nn/vulkan/profile_gemv.comp`。表里的地址探测失败时退回原数字。
3. `--auto` 才打开动态 TPOT 和 PP migration。冷启动切分不变。默认日志会多一行 `runtime: pp-migration dynamic-tpot`。
4. 自动切分在启动时如果 `cudaMalloc` 失败，或者 worker 在建图时断开（`Socket closed` 这一类），就从装不下的那台挪走一层再试，最多 12 次。用户写了 `--ratios` 时不改。

`device-profile-test` 在本机和 nx1 上都打印了 `device profile checks passed`。

阶段 2 源码上的两机复跑，2026-09-28，端口 18081，worker 已经停掉（最后停的是 pid 909407）：

- nx2：`./dllama worker --port 18081 --model models/qwen3_0.6b_q40/dllama_model_qwen3_0.6b_q40.m`
- nx1：同一模型，`--steps 32 --workers 192.168.137.15:18081`，没有 `--ratios`，也没有 `--auto`
- 日志：nx1 `/tmp/edgevisor_p2_two.log`，nx2 `/tmp/edgevisor_p2_worker.log`
- 结果：`EXIT:0`，`topology: speed-pack classes=nx,nx ratios=1@22*1@6`，生成了连贯句子，`tokens/s: 10.09`

### 还没在机器上验证的

- `--auto` 的完整推理。二进制里这个开关已经接上，没有跑过一轮会挪层的生成。
- 陌生地址的层计时。这次两机用的是表内地址 `192.168.137.15`，不会走那条探测。要测的话用 nx2 的 ZeroTier 地址 `10.47.235.49`，它不在表里。数据面仍然优先，这个地址只为了触发探测。
- 8B 装不下之后逐层退让。更早一次 Qwen3-8B 两机、旧二进制，speed-pack 分成 `1@22*1@14`，nx2 报需要 7231 MB，然后 `cudaMalloc` 失败。日志在 nx1 `/tmp/edgevisor_fresh_two.log` 被后来的 0.6B 覆盖过；当时 worker 日志是 `/tmp/edgevisor_fresh_worker.log`。新的退让逻辑只有单测，没有在 8B 上跑过。
- nano1、nano2 还没有这批未提交源码。

### 模型在哪

- Qwen3-0.6B Q40：nx1 和 nx2 的 `edgevisor_fresh/EdgeVisor/models/qwen3_0.6b_q40/`。模型约 914MB，tokenizer 约 2.1MB。笔记本、nano1、nano2 没有这份。
- Qwen3-8B Q40：四台 Jetson 的 `/home/jetson/cc/models/qwen3_8b_q40/`。笔记本没有。
- Qwen3-14B Q40：Jetson `/home/jetson/cc/models/qwen3_14b_q40/`，笔记本 `/home/cc/models/qwen3_14b_q40/`。

### 下一步

先问用户要不要把阶段 2 这批源码单独提交并推到 `refactor/productize`。用户没说提交之前，不要提交。

提交之后，在 nx1 + nx2 上补三件还没跑的验证：`--auto` 一轮短生成、用 `10.47.235.49` 做一次陌生地址探测、8B 不写 `--ratios` 看装不下时会不会逐层退让并最终吐出文本。不要做名单外设备的中途加入。不要合并到 main。

### 操作时注意

- 本机 shell 是 zsh。需要 `set -- $spec` 这种分词的脚本用 bash。远端管道不要用 `ssh -n` 去喂 heredoc；把脚本写成文件再执行。
- `--ratios` 在远端要加引号，否则 `1@16*` 会被当成通配。
- 不要 `pgrep -af dllama`。停进程用 `pkill -9 -x dllama` 或杀掉明确的 pid。不要拆网卡。
- `ssh -f` 或远端 `setsid ... &` 之后，本地 SSH 经常不退出。远端起来之后关掉卡住的本地 SSH。
- 旧目录 `/home/jetson/cc/EdgeVisor` 里可能还有没提交的 TCP keepalive 改动。那是更早的 smoke，不是这批产品化改动。不要顺手提交进去。
- 生成算不算成功，要看 token 语义，不能只看进程没崩。
