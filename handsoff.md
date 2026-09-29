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

2026-09-30 凌晨。阶段 2 按用户要求标成完成。源码提交 `797658e` 已推到 `refactor/productize`。nx1 `10.47.145.51` 和 nx2 `10.47.235.49` 上没有 `dllama`，没有 `gpu_hog`，没有 `mem_hold`。`18091` 没有在听。

阶段 2 是这四件：层数上限取实验室表和空闲显存估计的较小值；陌生地址在数据面计时一层的七个投影；`--auto` 打开动态迁移，冷启动切分不变；启动时显存或断连装不下就逐层退让，最多 12 次。用户传了 `--ratios` 就不退让。

已经定下来的结果：

- 数据面陌生地址探测做过。nx2 `192.168.137.15` 是 `cuda 1.28 ms/layer cap=28`，切分 `1@27*1@1`。
- 真实设备变慢会自动迁移，并且有正向收益。nx2 `/tmp/gpu_hog 4096` 在生成开始前就在跑，不是卡在第 12 条 token 之后才注入。一步 1 层：stage 1 到 0，layer 22，263.35 ms 到 240.69 ms。步长 2：layers 22 和 23，251.90 ms 到 200.16 ms，预测 53.49 ms，验证通过，层留下。再提议 layer 24 被拒绝，静态图没有这一层的冗余副本。一步 1 层日志 `/tmp/edgevisor_p2_auto.log`、`/tmp/edgevisor_p2_tpot.log`。步长 2 日志 `/tmp/edgevisor_p2_auto_fix4.log`、`/tmp/edgevisor_p2_tpot_fix4.log`。
- 往后面的 stage 交层时，root 等 KV ack 不再丢掉采样帧，采样帧、profile 帧和 stage-bypass ack 进缓存。原来停在 `[kv-collector] layer=21 pos=21` 的 `route=0->1 layers=[20,21]` 能交完并继续出 token。人为 root 延迟那次变慢并回滚，那不是真实设备变慢。
- 层数上限用 14B、不写 `--ratios`，打出切分就停，没有装权重。空闲约 12.6 GB 时是 `1@22*1@18`，表里的 22 生效。nx1 `/tmp/mem_hold` 占住 5 GiB 后，`cudaMemGetInfo` 剩 8573284352 字节，切分变成 `1@20*1@20`。14B 一层约 262 MB，70% 剩余显存只够 20 层。磁盘上留下的是第二次：`/tmp/edgevisor_cap_root.log`、`/tmp/mem_hold.log`。worker 是 `192.168.137.15:18091`。
- 逐层退让不写 `--ratios`。`prlimit --as` 限制 root 地址空间，`cudaMemGetInfo` 仍看到满显存，所以第一次切分还是 0.6B 的 `1@22*1@6`。6 GiB、`--steps 16`：root 每次减 1 层。`1@14*1@14` 装上权重后前向仍 `out of memory`，再退到 `1@13*1@15`，生成结束，`tokens/s: 10.32 (96.90 ms/tok)`。日志 `/tmp/edgevisor_retreat_root.log`。同一晚 3 GiB 连退 12 次，停在 `1@10*1@18` 仍是 `cudaMalloc ... out of memory`，没有第 13 次；那次日志被这次盖掉了。

源码提交 `797658e89790b7ecdf6452daea6476171b51d6a1`，作者 Yanhui。说明是 Complete phase 2 layer caps, startup retreat, and automatic migration. 13 个文件，新文件是 `/home/jetson/cc/edgevisor_fresh/EdgeVisor/src/nn/vulkan/profile_gemv.comp`。推送 `c986cf7..797658e` 到 `origin/refactor/productize`（`git@github.com:wansongcc/EdgeVisor.git`）。nx1 的 `/home/jetson/cc/edgevisor_fresh/EdgeVisor/dllama` 是 2026-09-29 23:17 编的这份源码。nx2 的同名二进制仍是 2026-09-28 17:51，没有重编。

不要合到 main。不要做名单外的中途加入。nano1、nano2 这次没有同步。不要提交 `._*`。

下一步：用户说「继续」之前不要开新实验。阶段 2 已经完成，没有写明的下一阶段。
