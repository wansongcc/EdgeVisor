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

更新时间：2026-09-29

### 正在做什么

EdgeVisor 产品化，分支 `refactor/productize`。阶段 0、阶段 1 用户已认可。阶段 2 源码还没提交。当前在测 `--auto`：注入 GPU 波动之后，推理变慢、自动触发迁移、迁移完成、迁移后比变慢那段更快。一步只挪 1 层的那次已经跑完。把步长调到 2 和 4 的重跑都在 KV 交接成功后停住，没有测到更大的恢复。状态：失败后停着。

### 仓库和机器

- 远程：`git@github.com:wansongcc/EdgeVisor.git`
- 当前开发分支：`refactor/productize`。已推送的 HEAD 是 `97cc6b0`。这次交接提交上去之后会再往前一个。
- 开发副本是 nx1 的 `/home/jetson/cc/edgevisor_fresh`。`/tmp/edgevisor_ft` 的 `.git` 已经不在，不要在那里提交。
- 推送用这个仓库自己的 `origin`。不要 `--set-upstream`，不要强推。远程若不能快进，先把这份提交接到远程分支之上再推。
- 提交作者用环境变量，不要改 git config：`GIT_AUTHOR_NAME=Yanhui`、`GIT_AUTHOR_EMAIL=fromthefox@icloud.com`，committer 同样。
- 阶段 2 源码不要和 `handsoff.md` 一起提交，除非用户明确要求。

2026-09-29 17:34 查过 nx1 `10.47.145.51` 和 nx2 `10.47.235.49`：没有 `dllama`，没有 `gpu_hog`，18081 没有在听。没有容器，没有没跑完的队列。

| 机器 | SSH | 数据面 | 新目录 |
| --- | --- | --- | --- |
| nx1，CUDA，root | `jetson@10.47.145.51` | `192.168.137.13` | `/home/jetson/cc/edgevisor_fresh` |
| nx2，CUDA | `jetson@10.47.235.49` | `192.168.137.15` | `/home/jetson/cc/edgevisor_fresh` |
| nano1，CUDA | `jetson@10.47.107.34` | `192.168.137.18` | `/home/jetson/cc/edgevisor_fresh` |
| nano2，CUDA | `jetson@10.47.215.50` | `192.168.137.16` | `/home/jetson/cc/edgevisor_fresh` |
| 笔记本，Vulkan | `cc@10.47.72.162` | `192.168.137.31` | `/home/cc/edgevisor_fresh` |

SSH 走 ZeroTier，不要走数据面 `192.168.137.0/24`。阶段 2 源码在 nx1、nx2、笔记本的新目录里。nano1、nano2 的新目录仍是 `45deaab`，没有这批未提交改动。`/home/jetson/cc/EdgeVisor` 是 `feat/device-fault-tolerance`，HEAD `931c1fb`。那里的 `handsoff.md` 停在今天上午，不是这份。不要在那个脏目录里改产品化代码。

### 已经定下来的

- 名单里没有的机器，不能在推理中途插入执行图。用户说留到以后做 mDNS。不要做。
- 冷启动切分不变。没写 `--ratios` 时，实验室地址仍走 speed-pack：NX 5.8 ms、上限 22 层，Nano 7.4 ms、上限 8 层，笔记本 8.3 ms、上限 11 层。
- 一次 PP 决策默认只挪 1 层。`maxPpLayerMove` 默认 1，`--runtime-redundant-boundary-layers` 默认也是 1。打开动态 TPOT 时，冗余边界会被抬到至少和步长一样。能挪的是邻居事先以冗余副本持有的边界重叠。`DLLAMA_TPOT_MAX_PP_LAYER_MOVE` 可设 1–64。这次没有改默认值，只在测试进程的环境变量里调过。
- 阶段 0 不要退：快路径 token 一致；重开只要求句子连贯；API 只绑 `127.0.0.1`。

### 波动测试

模型 Qwen3-0.6B Q40，路径 `edgevisor_fresh/EdgeVisor/models/qwen3_0.6b_q40/`。nx1 做 root，nx2 做 worker，不写 `--ratios`，切分 `1@22*1@6`。提示词 `The capital of France is`。波动是 nx2 上的 `/tmp/gpu_hog 4096`，源文件 `/tmp/gpu_hog.cu`。不要用需要 sudo 的频率脚本。

一步 1 层，`--steps 400`，`--auto`，worker `192.168.137.15:18081`。注入点是第 12 条 `[token-e2e]`。日志：nx1 `/tmp/edgevisor_p2_auto.log`、`/tmp/edgevisor_p2_tpot.log`。

- 变慢了。控制器过冲约 402%，基准大约 52 ms/token。注入后的窗口是 346 ms、333 ms，决定迁移时是 263 ms。
- 触发了。pos 35，`pp_move`，stage 1 到 stage 0，第 22 层，预计收益 16.6 ms，阈值 7.9 ms，`note=migration_issued`。
- 交接完成。KV 日志 `recover mode=enabled status=ok`。调度器 `note=verify_ok`。
- 恢复有限。核对窗口从 263 ms 到 241 ms。后面的窗口仍在 236–299 ms。整段 `tokens/s: 3.55 (281.51 ms/tok)`。再挪一层的预计收益变成负数，所以没有第二次迁移。

步长 4 跑了两次，步长 2 跑了一次。波动都没注入进去。调度器在开局窗口（230–290 ms/token）就发出了反向的多层移动：4 层是第 18–21 层，预计收益 27–32 ms；2 层是第 20–21 层，预计收益 11.4 ms。方向都是 stage 0 到 stage 1。KV `recover status=ok` 之后不再出 token，`[token-e2e]` 停在 9 条。日志：`/tmp/edgevisor_p2_auto4.log`、`/tmp/edgevisor_p2_auto4b.log`、`/tmp/edgevisor_p2_auto2.log`，以及对应的 `tpot4.log`、`tpot4b.log`、`tpot2.log`。

### 同日更早、日志还在的三件

worker 侧日志已经不在。root 日志还在 nx1。

- `/tmp/edgevisor_auto_root.log`：`--steps 24`，`tokens/s: 17.52 (57.09 ms/tok)`，切分仍是 `1@22*1@6`。里面的 `root migration route=0->1 layers=[21] layerCount=1 pinnedByEnv=no` 是启动时武装的路由，没有后续的 `apply pp command`。
- `/tmp/edgevisor_profile_root.log`：清空速度表后仍用数据面。`profile: local cuda 0.85 ms/layer cap=28`，`profile: 192.168.137.15 cuda 1.28 ms/layer cap=28`，切分 `1@27*1@1`，`tokens/s: 9.59`。测完已把临时代码删掉。`knownSpeedProfile` 里 `.13`/`.15` 仍是 nx，5.8 ms，上限 22。
- `/tmp/edgevisor_8b_root.log`：Qwen3-8B 在 `/home/jetson/cc/models/qwen3_8b_q40/`，不写 `--ratios`，切分一直是 `1@22*1@14`，没有第二次 `topology:`，没有逐层退让。`tokens/s: 3.97 (251.84 ms/tok)`。这次 8B 装下了。

### 未提交的代码

都在 `/home/jetson/cc/edgevisor_fresh`。阶段 2 原有未提交文件：`EdgeVisor/src/app.cpp`、`app.hpp`、`device_profile.cpp`、`device_profile.hpp`、`dllama.cpp`、`nn/nn-cuda.cu`、`nn/nn-cuda.hpp`、`nn/nn-network.hpp`、`nn/nn-vulkan.cpp`、`nn/nn-vulkan.hpp`、`test/test_device_profile.cpp`，以及未跟踪的 `EdgeVisor/src/nn/vulkan/profile_gemv.comp`。

为了让 1 层迁移发得出去，又改了两处。nx1 的 `dllama` 已重新编过。nx2 的 worker 二进制没有这两处：

- `EdgeVisor/src/dynamic/dynamic_tpot.cpp` 的 `makePpCommandRequest` 补上 `predictedBenefitMs`（用 `candidate.gainMs`）、`predictedCostMs=0`、`survivalProbability=1`、`safetyMarginMs=0`。否则计划套接字因缺少估计值拒绝。失败时 `note` 会带上原因。
- `EdgeVisor/src/app.cpp` 等 KV ack 时，如果读到 `LLM_WORKER_FRAME_MAGIC`，按帧头里的长度丢掉再继续等。跳过次数从 4 改成 32。否则末级采样帧会被当成坏 ack。

### 下一步

用户说「继续」时，先查多层挪动为什么在 `recover status=ok` 之后不再出 token。日志停在 `[kv-collector] layer=21 pos=21`。当时 layer-gate 已经把源阶段那几层的 primary 和 redundant 都设成 `enabled=0`。先看 `RootLlmInference` 里这次交接之后的下一次 forward，不要先再开一轮波动实验。生成能继续之后，再用同样的 0.6B、nx1+nx2、`/tmp/gpu_hog 4096`、第 12 条 token 之后注入，看步长大于 1 时速度是否比 263 ms 到 241 ms 恢复得更多。

用户没说继续，不要自己开实验，也不要提交阶段 2 源码。不要做名单外中途加入。不要合并到 main。8B 这次装下了，逐层退让还没在真机上走到。

### 操作时注意

- 本机 shell 是 zsh。需要分词的远端脚本用 bash。`--ratios` 加引号。
- 不要 `pgrep -af dllama`。停进程用 `pkill -9 -x dllama` 或明确的 pid。`gpu_hog` 同样。不要拆网卡。
- `ssh -f` 或远端 `setsid ... &` 之后，本地 SSH 经常不退出。远端起来之后关掉卡住的本地 SSH。
- 生成算不算成功，要看 token 语义，不能只看进程没崩。
