# handsoff

本段到下面的分隔线为止是固定说明。后来的 Agent 不要改写、不要删、不要把当次进度写进这一段。每次交接只整段替换分隔线以下的内容。

## 这份文档是干什么的

同一个人会在两个 Agent 之间交替做这个仓库的开发。一边做完一段工作，就把分隔线下面更新掉；另一边先读这份文件，然后接着做，不用再口头补一大段上下文。

## 接手时

1. 先读分隔线下面的全文，再看它点名的文件和命令，不要凭记忆补实验设计。
2. 用户说「继续」时，按「下一步」做。没写进下一步的长实验不要自己开。
3. 改代码、跑实验、只查了状态，都算一段工作。离开前更新分隔线下面。

## 交接时怎么写

替换分隔线以下的全部内容，保留分隔线本身。写清这些，有代码 diff 就写，没有也要写：

- 现在在做哪件事，做到哪一步。
- 机器上还留着什么：进程、端口、GPU、容器、没跑完的队列。
- 已经定下来的结论和约束。不要只贴 git diff。
- 卡住的原因，以及下一次打开终端应该先跑的命令或先改的函数。
- 相关 commit、关键路径、明确不要提交或不要推送的文件。

不要把对话原文、完整 diff 或论文 PDF 贴进来。事实写短，路径写全。

---

## 可变交接

更新时间：2026-11-19 15:45 CST。分支 `main`，已与 `origin/main` 对齐，HEAD `4756d18`（Merge pull request #7）。工作状态：接管乱码已修并合进 `main`。机器上没有正在跑的实验。

### 现在在做哪件事

算法 2 的流水线接管。死掉一台之后，句子不再掉进 `!'-39?EKQW]ciou`。做到：T4 上杀掉中间那台、不杀、以及整段交给后一台再重新加入，这三种都保持英文。还没做完的是：机器重新回到计算路径之后，句子会离开原来那句，另起一段。

### 已经定下来的结论

- 根节点自己算被抢走的第 8 层时，这一层的结果和它本来就由根节点计算时一致。坏的是后一台机器左边的副本：那些层的 KV 是用本机已经算完、准备发出去的激活填的。
- 修法在三处。收到的激活另存一份，左边影子层读这份。上一台机器右边的影子链继续接到后面各台已经持有的连续副本；这次死掉的是第 8 到 11 层，链要接到第 14 层，只接到 11、12 或 13 层时乱码还在。接管时，后一台机器上与这条链重复的层不再分给它。
- 不杀进程时，同一提示、temperature 0，句子是 “Is there something on your mind that you'd like to talk about…”。杀掉第 8 到 11 层那一台之后，前半句与此相同。日志在 `/tmp/ev_textfix_nokill/root.log` 和 `/tmp/ev_textfix_rejoin/root.log`。
- 重新加入之后乱码没有出现，但原文停住，模型另起 “I'd be happy to help you with any questions…”。这不是已经修好的那串乱码。
- 整段交给后一台（depths `0,0;0,4;1,1`，不要再加 `--runtime-redundant-boundary-layers`）时，接管和重新加入都能跑完，句子是英文。日志 `/tmp/ev_textfix_ack2/root.log`。
- 用 `kill` 记下的 PID 停掉进程，不能写成物理断电的结果。
- 2026-11-09 那份交接里的 8B 矩阵已经跑完。`/tmp/ev_matrix_v2/matrix.log` 末行是 `2026-11-10T05:01:20+00:00 === 8B matrix done ===`。当时的 pid 3057489、3113960 已经不在。不要再开那一轮。

### 两套设备

#### 这台机器上的四张 T4

控制机就是这台，四张 Tesla T4，当前空闲（每张 0 MiB）。GPU 0 当根，1、2、3 当 worker。本机编译：

`make -C /home/byh/B01/EdgeVisor-wansongcc/EdgeVisor -j dllama`

二进制是 `/home/byh/B01/EdgeVisor-wansongcc/EdgeVisor/dllama`，CUDA 架构 75。模型用 `/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m`。分词器没有 llama3.2 那份，用 `/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t`。

现成脚本：`/home/byh/B01/EdgeVisor-wansongcc/experiments/algo2/t4_rejoin.sh`。它用 `--ratios '1@8*1@4*1@8*1@8'`、`--runtime-redundant-boundary-layers 3`、`--workers 127.0.0.1:29241` 起三台、`--nthreads 1`、`--buffer-float-type q80`、`--enable-pp-migration`。影子 KV 必须 `--nthreads 1`，否则直接拒绝启动。层段用 `--ratios` 钉死；不钉的话速度装箱可能排出 29 层然后中止。对比句子时用 `--temperature 0 --seed 1`。停进程用脚本记下来的 PID，`kill` 那个 PID。不要用 `pkill -f`，那会把远程 shell 一起杀掉。

#### 边缘五台：rog、nx1、nx2、nano1、nano2

从这台控制机用 SSH 别名 `rog`、`nx1`、`nx2`、`nano1`、`nano2`。推理走有线 `192.168.137.0/24`，不要走 WiFi。

| 别名 | 有线 | 角色 |
| --- | --- | --- |
| rog | 192.168.137.20 | 流水线两头的主控，Vulkan/CPU 都在这台起 inference |
| nx1 | 192.168.137.13 | 第一台 Jetson，也是 aarch64 的编译机 |
| nx2 | 192.168.137.15 | 中间那台，演示里拔的是它 |
| nano1 | 192.168.137.18 | Jetson |
| nano2 | 192.168.137.16 | 流水线另一头 |

跳板：Jetson 不从控制机直连。先 `ssh rog`，再由 rog 用 `~/.ssh/id_ed25519_lab` 登 `jetson@192.168.137.x`。脚本里的写法在 `/home/byh/B01/EdgeVisor-wansongcc/scripts/demo_dropout_rejoin.sh` 和 `scripts/demo_power_cycle.sh`。

WiFi `192.168.8.46` 和 `192.168.8.23` 在速度表里算 NX，`192.168.8.45` 和 `192.168.8.53` 算 Nano（`EdgeVisor/src/device_profile.hpp` 的 `knownSpeedProfile`）。源码没有把每个 WiFi 地址再绑到某一台别名上。这些地址探得通，但撑不住一次完整的模型加载。重新加入和正式推理都用上面的有线地址，`--workers` 也填有线地址。

模型：rog 用 `/home/cc/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m` 和 `/home/cc/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t`。四台 Jetson 用 `/home/jetson/cc/models/` 下的同名文件。没有 llama3.2 的分词器。

两条演示：

- 杀进程再拉起：`bash /home/byh/B01/EdgeVisor-wansongcc/scripts/demo_dropout_rejoin.sh`。二进制在各机 `/tmp/ev_demo1/EdgeVisor/dllama`，重启机器会丢。`--rebuild` 才重编。
- 人手断电：`bash /home/byh/B01/EdgeVisor-wansongcc/scripts/demo_power_cycle.sh`。必须在真实终端里跑，面板要 100 列以上。二进制放在掉电也不丢的目录：rog 是 `/home/cc/ev_demo2`，Jetson 是 `/home/jetson/ev_demo2`。脚本自己不拔电。出 token 之后人手拔 nx2。不要拔 rog，也不要拔 nano2，它们在流水线两头。nx2 上电并能 ping 通 `192.168.137.15` 之后，再跑脚本写到 `/tmp/ev_demo2_next.txt` 的那条重新加入命令。

边缘上的二进制经常比 `main` 旧。`source_rev` 和当前 HEAD 不一致时，演示脚本会重编：rog 上 `DLLAMA_VULKAN=0 DLLAMA_CUDA=0 make dllama`，nx1 上编 aarch64，再拷到另外三台。合进 `4756d18` 之后还没在五台上重编，下一次边缘实验先让脚本按 `source_rev` 重编，不要拿旧二进制报结果。

rog 若改跑 Vulkan，工作目录必须能打开 `./src/nn/vulkan/`。加载函数按这个相对路径找文件（`EdgeVisor/src/nn/nn-vulkan.cpp` 的 `readShader`）。文件名是 `matmul-forward-q80-q40-f32.spv` 这一类，没有 `matmul.spv`。现在这两条演示脚本编的是 CPU 版，工作目录是持久目录，里面没有这些 shader。

停边缘进程用 `pkill -x dllama`（演示脚本就是这样），不要用 `pkill -f`。

### 机器上还留着什么

- 没有 `dllama` 进程。四张 T4 都是 0 MiB。端口 9992–9994、294xx 都没有在听。
- 未跟踪、不要提交：`/home/byh/B01/EdgeVisor-wansongcc/experiments/algo2/logs/t4_rejoin/`。
- 8B 矩阵结果还在 `/tmp/ev_matrix_v2/`，作废对照在 `/tmp/ev_e1_deprecated/`，3B 单线程六臂在 `/tmp/ev_st3b/`。都不要删，也不要提交。
- `/home/byh/B01/baseline` 在分支 `feat/remp-pp`，另有未跟踪的 `_harness/dynapipe_ab.py` 和 `_harness/dynapipe_sampling.py`。那个仓库的 remote 不是这份 EdgeVisor。不要从这边顺手提交。

### 下一步

用户说「继续」时：先看 `/tmp/ev_textfix_rejoin/root.log` 里 `rejoined stage` 之后的句子。重新加入本身已经成功（KV 追到 pos 46，乱码没有出现），但句子离开了 “something on your mind…”。先查交还时 worker 用的是不是刚拷过去的那份 KV，再决定要不要改 `EdgeVisor/src/app.cpp` 的重新加入路径。没说继续就不要重跑 T4，也不要上五台边缘机。

### 约束

- PR 用 `gh --repo BianYanhui/EdgeVisor_fork`。不带 `--repo` 会打到 `wansongcc/EdgeVisor`。
- 不要提交 `*.log`、`experiments/algo2/logs/`、`/tmp` 下的结果、论文 PDF、`._*`。
- 引擎是 C++11，编译带 `-Werror`。
- 影子 KV 开着时线程数必须是 1。

### 相关 commit

- `4756d18` 把 `algorithm_2_ver.2` 合进 `main`。其中 `8ccdcc2` 是接管句子这次的修改：`EdgeVisor/src/llm.cpp`（收到的激活另存、右边界链往后接）、`EdgeVisor/src/app.cpp`（后一台重复的层不再算进后缀，并丢掉插在 logit 前面的加入确认）、`EdgeVisor/src/nn/nn-network.cpp` 与 `nn-network.hpp`（丢掉缓冲的确认字，速度帧不挡交付询问）。
- 同一条分支上已包含：`1456269` 按邻居耗时切重叠层，`2594a19` 追赶中的设备先不进解码，`5fa2491` 设备回来后关掉邻居已经接管的冗余层。
- 这次交接没有改这些源码。
