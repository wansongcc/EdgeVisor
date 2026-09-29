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

EdgeVisor 产品化，分支 `refactor/productize`。阶段 0、阶段 1 用户已经认可通过。阶段 2 的代码还没提交。用户要求补跑三件里的两件，并改了第三件的做法：陌生地址探测不要走 ZeroTier，清空速度表后仍用数据面 IP。名单外机器中途插入执行图，用户说不用管。

三件都跑完了。跑完后 nx1、nx2 上没有 `dllama` 进程。

### 仓库和机器

- 远程：`git@github.com:wansongcc/EdgeVisor.git`
- 当前开发分支：`refactor/productize`。已推送的 HEAD 仍是 `45deaab`。阶段 2 源码还没提交。
- 开发副本：`/tmp/edgevisor_ft`，分支 `refactor/productize`，没有 `origin`。推送用 `git push git@github.com:wansongcc/EdgeVisor.git HEAD:refactor/productize`。不要 `--set-upstream`，不要强推。
- 这份交接写在 nx1 的 `/home/jetson/cc/EdgeVisor/handsoff.md`，那份仓库是 `feat/device-fault-tolerance`，不要在那个脏目录里改产品化代码。
- 提交作者用环境变量，不要改 git config：`GIT_AUTHOR_NAME=Yanhui`、`GIT_AUTHOR_EMAIL=fromthefox@icloud.com`，committer 同样。
- 阶段 2 未提交文件仍是 `app.cpp`、`app.hpp`、`device_profile.cpp`、`device_profile.hpp`、`dllama.cpp`、`nn-cuda.cu`、`nn-cuda.hpp`、`nn-network.hpp`、`nn-vulkan.cpp`、`nn-vulkan.hpp`、`test_device_profile.cpp`，以及未跟踪的 `EdgeVisor/src/nn/vulkan/profile_gemv.comp`。不要和 `handsoff.md` 一起提交，除非用户明确要求提交代码。

五台设备，控制面走 ZeroTier，数据面是 `192.168.137.0/24`。SSH 走 ZeroTier，不要走数据面。

| 机器 | SSH | 数据面 | 新目录 |
| --- | --- | --- | --- |
| nx1，CUDA，root | `jetson@10.47.145.51` | `192.168.137.13` | `/home/jetson/cc/edgevisor_fresh` |
| nx2，CUDA | `jetson@10.47.235.49` | `192.168.137.15` | `/home/jetson/cc/edgevisor_fresh` |
| nano1，CUDA | `jetson@10.47.107.34` | `192.168.137.18` | `/home/jetson/cc/edgevisor_fresh` |
| nano2，CUDA | `jetson@10.47.215.50` | `192.168.137.16` | `/home/jetson/cc/edgevisor_fresh` |
| 笔记本，Vulkan | `cc@10.47.72.162` | `192.168.137.31` | `/home/cc/edgevisor_fresh` |

这次只用了 nx1 和 nx2。nano1、nano2 的新目录仍停在 `45deaab`，没有阶段 2 的未提交源码。

### 这次跑完的结果

模型都是 Qwen。0.6B 在 `edgevisor_fresh/EdgeVisor/models/qwen3_0.6b_q40/`。8B 在 `/home/jetson/cc/models/qwen3_8b_q40/`。提示词都是 `The capital of France is`。没有写 `--ratios`。跑完后 worker 已停。

1. `--auto` 短生成，2026-09-29 重跑。端口 18082。nx2 worker 用 0.6B，nx1 root 加 `--auto --steps 24`，worker 地址是数据面 `192.168.137.15:18082`。日志：nx1 `/tmp/edgevisor_auto_root.log`，nx2 `/tmp/edgevisor_auto_worker.log`。

   结果：进程正常结束。日志有 `runtime: pp-migration dynamic-tpot`，切分仍是 `topology: speed-pack classes=nx,nx ratios=1@22*1@6`。生成过程中发生了一次挪层：`root migration route=0->1 layers=[21] layerCount=1`。文本是 `<think>` 后面的 “Okay, so the user is asking about the capital”，`tokens/s: 17.52`。语义连贯。

2. 清空速度表后的探测，仍走数据面，没有用 ZeroTier 地址。做法是让 `knownSpeedProfile` 直接 `return false`，在 nx1 上 `make dllama`，然后用 `192.168.137.15:18084` 跑 0.6B、`--steps 16`，没有 `--auto`。测完把这行 `return false` 删掉，nx1 又 `make dllama` 一次。源码里的实验室地址表现在是原样。日志：nx1 `/tmp/edgevisor_profile_root.log`，nx2 `/tmp/edgevisor_profile_worker.log`。

   结果：进程正常结束。探测打出来的是 `profile: local cuda 0.85 ms/layer cap=28` 和 `profile: 192.168.137.15 cuda 1.28 ms/layer cap=28`。切分变成 `topology: speed-pack classes=cuda,cuda ratios=1@27*1@1`。文本是 `<think>` 后的 “Okay,”，`tokens/s: 9.59`。这证明空表时会按数据面地址现场计时，而不是查 5.8 ms / 22 层那张表。测到的毫秒数比实验室表小，因为计的是当前这个 0.6B 的一层投影，不是表里那组固定数字。

3. 8B 不写 `--ratios`。端口 18083。同一提示词，`--steps 16`。2026-09-28 夜里和 2026-09-29 早上各跑完一次，结果一样。日志：nx1 `/tmp/edgevisor_8b_root.log`，nx2 `/tmp/edgevisor_8b_worker.log`。早上这次是 `tokens/s: 3.97`。

   切分一直是 `topology: speed-pack classes=nx,nx ratios=1@22*1@14`。没有出现第二次 `topology:`，也没有逐层退让。权重加载成功，生成了 `<think>` 后的 “Okay,”，worker 收到 prefill 和几次 decode 后正常 stop。所以这次 8B 在两台上装下了，退让那条路径没有被走到。更早一次旧二进制上 nx2 `cudaMalloc` 失败、需要 7231 MB 的情况，这次没有复现。

### 没有做的

- 名单外的机器在推理中途插入执行图。用户说不用管。
- 阶段 2 源码仍然没有提交。
- 没有合并到 main。

### 下一步

先问用户要不要把阶段 2 这批源码单独提交并推到 `refactor/productize`。用户没说提交之前，不要提交。

8B 这次装下了，所以还没有在真机上看到逐层退让。如果用户还想看退让，需要一次确实会 `cudaMalloc` 失败或建图时断开的启动，而不是把这次已经成功的 8B 命令再跑一遍。

### 操作时注意

- 这台开发机的 shell 是 PowerShell。发给远端的双引号经常会被吃掉，带空格的 `--prompt` 要用远端单引号，否则会报 `Unknown option: capital`。
- `--ratios` 在远端要加引号，否则 `1@16*` 会被当成通配。
- 不要 `pgrep -af dllama`。停进程用明确的 pid，或 `pkill -9 -x dllama`。不要拆网卡。
- `ssh -f` 或远端 `setsid ... &` 之后，本地 SSH 经常不退出。远端起来之后关掉卡住的本地 SSH。
- 旧目录 `/home/jetson/cc/EdgeVisor` 是脏的，里面有和这次产品化无关的未提交文件，还有 `._*`。不要顺手提交进去。
- 生成算不算成功，要看 token 语义，不能只看进程没崩。
