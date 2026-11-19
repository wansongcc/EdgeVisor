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

更新时间：2026-11-19 18:41 CST。分支 `perf/t4-cuda-speed`，基于 `main` `657900f`。工作状态：已完成 4 轮 4×T4 CUDA 推理速度优化并验证通过，进程全停，等下一步指令。4 个提交还没有推到远程，`origin` 上连这个分支都还没有。

### 现在在做哪件事

4×T4 单机四段的 CUDA 解码提速。4 轮都做完、都提交了。实测 Arm A 从 16.1–16.3 提到 51.6–51.7 tok/s（约 3.2×），Arm B 从 15.8 提到 49.8 tok/s（约 3.16×）。现在停下来，等用户决定是继续第五轮，还是先推 PR。

### 已经定下来的结论和约束

- 单请求 PP 气泡和 Shadow KV 的语义都要保留。这 4 轮只动 GPU 侧执行和搬运，没有改调度协议、没有改 PP 语义。`--ratios '1@7*1@7*1@7*1@7'`、`--nthreads 1`，以及 Arm B 的 `--enable-pp-migration --runtime-redundant-boundary-layers 3`，都是验证基线，不要为了刷分改掉。
- 不盲目强上 `ppInflight > 1` 去挤干气泡。`EdgeVisor/src/dllama.cpp` 第 701–706 行：只有 remote sampling 可用时 `ppInflight` 才允许多于 1，否则强制回到 1，并打印 “remote last-stage sampling is unavailable; using ppInflight=1 to keep logits sampling correct”。调大会破坏 logits 采样正确性。气泡留着，靠 Shadow KV 藏。
- 不加 `--use_fast_math`。实测它翻转 RMSNorm/softmax/SiLU 的舍入，最后一段的分词就变了；而且并不比纯 `-O3` 快（c1 阶段 nofm 15.11/15.13/15.05 对 fastmath 14.94/14.37/15.02，日志 `/tmp/ev_perf_c1/cmp/c1nofm_A`、`c1fm_A`）。这条已经写进 `EdgeVisor/Makefile` 第 109–115 行的注释，别删。
- 不全局加 `-DNDEBUG`。全局关掉会让单元测试的 `assert()` 失效，`make check` 就白跑。Makefile 里也是按这个写的。
- 这轮只做 CUDA 路径。CPU 和 Vulkan 的行为没碰：`NnDevice` 上新加的 `beginForward/endForward/syncPipeToHost/invalidatePipeOnDevice/setLayerPerfSegments` 都是空默认实现；多设备、或一个节点里混了 CPU 回退段的图，仍走原来的 host 中转路径。
- 边缘五台（rog/nx1/nx2/nano1/nano2）这一轮完全没动，上面的二进制还是 `4756d18` 之后那版。等设备联网后再按 `source_rev` 重编验证。

### 四个提交（都在 `657900f` 之后，按时间顺序）

1. `2c2a040` perf(cuda): batch segment transfers, cache op bindings, skip single-node syncs
   `EdgeVisor/src/nn/nn-cuda.cu`、`EdgeVisor/src/nn/nn-cuda.hpp`、`EdgeVisor/src/nn/nn-network.cpp`、`EdgeVisor/Makefile`。段的输入输出改成走各自的 pinned 块、一次 `cudaMemcpyAsync`，去掉每条 pipe 上的 `cudaStreamSynchronize`，`forward()` 不再无条件做整设备同步；每个 op 的显存指针和布局每 plan epoch 解析一次存进 `opBindings`，epoch 变更或 `refreshPointers` 时才重建；纯 host 段（只含 `OP_PLAN_BARRIER`/`OP_PLAN_APPLY`）不搬字节也不发 sync，barrier 不再把 plan 包推给设备；单节点 stage 的 `syncWithRoot`/`syncNodeSlices` 直接返回，这正是 4×T4 的拓扑。Makefile 只加了 `NVCCFLAGS += -O3` 和上面那条注释。
2. `b4d0a12` perf(cuda): dp4a warp-tiled Q80xQ40 matmul and parallel reduction kernels
   只改 `EdgeVisor/src/nn/nn-cuda.cu`。主 kernel `matmulQ80Q40LargeK` 改成一次 warp 算一行加 warp shuffle 归约，加 `__dp4a` 快路（q40 的 nibble 保持 0..15，每 block 减一次 `8*sum(x)`，与旧的 `(w-8)*x` 整数等价），x 组整体进 shared memory，grid 按行组乘以 batch 排。`q80Quantize` 改成一 warp 一个 32 元素块，`invRms` 改成一 block 一列，`multiheadAttScore` 把 query 切片缓存进 shared memory。`NnBlockQ40`(18B)/`NnBlockQ80`(34B) 只保证 2 字节对齐，所以 `qs` 按 uint16 读再拼 uint32，绝不 cast 成 `int*`。
3. `6e21fa8` perf(cuda): four rows per warp in Q80xQ40 Large-K and gate logits stats
   `EdgeVisor/src/nn/nn-cuda.cu`、`EdgeVisor/src/dllama.cpp`。同一个 kernel 改成每 warp 四行（block 32 行，grid 行维缩 4×），x 侧一份 shared memory 被四行复用，每行保留独立累加器，结果与一行一版逐位相同；尾部 warp 用 `rowCount` 收口，非活跃行的权重指针夹到本 warp 最后一行，避免越界。`dllama.cpp` 新增 `logitsStatsNeeded()`，128256 个 logits 的扫描只在 `DLLAMA_DEBUG_TOPK_LOGITS`、`productLogLevel() >= 1`、或 root logits 分片导出时才做，正常推理路径省掉两次全扫。
4. `1a66663` perf(cuda): keep pipes resident on the GPU for the whole single-device forward
   `EdgeVisor/src/nn/nn-cuda.cu`、`EdgeVisor/src/nn/nn-cuda.hpp`、`EdgeVisor/src/nn/nn-executor.cpp`、`EdgeVisor/src/nn/nn-executor.hpp`。单设备（一台顶一个 stage）时，整段前向的 pipe 留在显存：段之间不再下到 host 再上传；只有 PP 发送、logits 汇聚、多节点 slice 交换这几处会 `syncPipeToHost`，其余 pipe 标成 device-dirty；每个 sync 之后只失效它可能写过的 pipe，别的 pipe 保持 device-authoritative。中间段因此省掉整条 PCIe 往返和每段一次的 `cudaStreamSynchronize`。

### 实测结果（4×T4，同机交替跑）

- 脚本 `/tmp/ev_c4_cmp.sh <rounds> <ArmA|ArmB>`，端口 29581–29583，`--ratios '1@7*1@7*1@7*1@7'`、`--nthreads 1`、`--prompt "What is 2+2? Answer briefly."`、`--steps 64`、`--temperature 0 --seed 1`、`--benchmark --buffer-float-type q80`。Arm B 额外 `DLLAMA_BUBBLE_SHADOW_KV=1`、`DLLAMA_BUBBLE_SHADOW_KV_DURING_FORWARD=1`、`--enable-pp-migration --runtime-redundant-boundary-layers 3`。每轮按 base → c3 → c4 轮转，各 3 轮。
- 数字在 `/tmp/ev_perf_c4/cmp_A.log`、`/tmp/ev_perf_c4/cmp_B.log`，中位数（ms/tok 是每 token 耗时）：
  - Arm A：base 16.32（16.32/16.11/16.62）→ c3 43.56 → c4 51.73（51.90/51.59/51.73），**3.17×**
  - Arm B：base 15.78（16.04/15.77/15.78）→ c3 42.99 → c4 49.81（49.49/49.81/50.76），**3.16×**
- 文本一致性：`/tmp/ev_perf_c4/cmp/{base,c3,c4}/{ArmA,ArmB}/r{1,2,3}/txt.txt` 共 18 份，md5 全部是 `fa945fe6b651db1da18c55e91978bd24`，与 base 逐字节相同，18 次对比零差异。
- `nn-cuda-test`：`/tmp/ev_perf_c4/nnct3.log`，133/133 全 `ok`（`grep -c ": ok"` 得 133），`max_abs_error=0`，含 PR12 部分 KV 导出/应用。
- `make check`：`t_*.log` 那 9 项全过（`algo2 checks passed`、`device profile checks passed`、`dynamic_tpot_algorithm PASS`、`failover gate checks passed` 等，`t_token-timing-test.log` 是空输出），日志在 `/tmp/ev_perf_c4/`。`pp-failover-network-test` 不能跟着默认跑：默认 TCP 超时下它挂在 `testBufferedSendNoAnswer` 的 `assert(elapsed >= 400 && elapsed < 2500)`（见 `/tmp/ev_perf_base/make_check.log`、`/tmp/ev_perf_c1/make_check_final.log`），要带 `DLLAMA_TCP_USER_TIMEOUT_MS=1000 ./pp-failover-network-test` 单独跑，末尾是 `PP recovery, buffered tail frames, bounded timeouts, protocol rejection and shadow KV passed`（`/tmp/ev_perf_c4/ppfail.log`）。
- 4×T4 掉线重入：`/tmp/ev_perf_c4/rejoin_out.txt` 和 `rejoin/summary.txt`。杀掉 w1 再拉起，`node=1 rejoined stage=1 layers=[8,12) kv through 50`，输出仍是英文，没有回到乱码。
- Algo2 边界层耗时回归：`/tmp/ev_perf_c4/boundary/`（c4）、`boundary_c3/`（c3）、`boundary_nodelay/`（不注入延迟的对照），都跑到 token 77，统计的是后 65–69 个 token 的 `stage_compute` 中位数。`stage_compute` 中位数，各 stage 依次是 0/1/2/3：c4 = 4.09/3.97/105.9/1.75，c3 = 5.08/4.97/106.4/6.63；不注入延迟的对照是 4.04/3.93/3.94/1.52（合计 13.44 ms）。结论：在真实 PP 排空下 c4 的边界段没有被拉长，与 c3 同量级，迁移触发的判定阈值不用改。

### 机器上还留着什么

- 没有 `dllama` 进程：`ps -eo pid,cmd | grep -E "dllama|t4_"` 为空。四张 T4 都是 0 MiB / 0%，`nvidia-smi --query-compute-apps` 也是空的。
- 端口全空：`ss -ltn` 里 9990–9999、294xx、295xx 都没有在听。
- `EdgeVisor/dllama` 还在，md5 `cf7488d4902c8f72b44d62434f22a691`，与 `/tmp/ev_perf_c4/dllama_c4` 同一个二进制，就是当前 HEAD 编出来的 CUDA 版（架构 75，2.6 MB）。它被 `EdgeVisor/.gitignore` 的 `/dllama` 忽略，不会进提交。
- 未跟踪、不要提交：`experiments/algo2/logs/t4_rejoin/`。`git status --short` 只有这一行。
- `/tmp` 下这轮的产物都留着，不要删：`/tmp/ev_perf_base`（88K，base 的 make check/ppfail 日志）、`/tmp/ev_perf_c1`（17M，含 fast-math 与 nofm 对照）、`/tmp/ev_perf_c2`（3.3M）、`/tmp/ev_perf_c3`（11M）、`/tmp/ev_perf_c4`（9.1M，含 `dllama_base`、`dllama_c3`、`dllama_c4` 三个二进制和这一轮全部日志）。脚本是 `/tmp/ev_c4_cmp.sh`、`/tmp/ev_c4_rejoin.sh`、`/tmp/ev_c4_boundary.sh`、`/tmp/ev_c4_boundary_c3.sh`、`/tmp/ev_c4_boundary_nd.sh`、`/tmp/ev_c4_ioprobe.sh`、`/tmp/ev_c4_ioprobe16.sh`、`/tmp/ev_c4_smoke.sh`。
- 原有的还在：`/tmp/ev_matrix_v2/`（1.2G，8B 矩阵，已跑完，别重开）、`/tmp/ev_e1_deprecated/`（74M）、`/tmp/ev_st3b/`（7.3M）。以及 `/home/byh/B01/baseline`，在分支 `feat/remp-pp`，带未跟踪的 `_harness/dynapipe_ab.py` 和 `_harness/dynapipe_sampling.py`。都不要删，也不要从这边顺手提交。
- 磁盘：`/` 98G 用了 72%，`/home` 7.3T 用了 69%，还有空间开下一轮。

### 下一步

用户说「继续」之前，不要开长实验，也不要碰边缘五台。说继续时可以选：

1. 第五轮优化：把 Shadow KV 的 scratch buffer 和主计算图隔离开（现在它和主计算图共用同一批 pipe/staging），让影子 KV 真正异步藏进 PP 气泡；顺带补尾段采样与延迟的 profile 收集，`boundary/` 下目前只量到 `stage_compute`，尾段采样耗时还没有单独拆出来。
2. 先合入：把 `perf/t4-cuda-speed` 推到 `origin`（`BianYanhui/EdgeVisor_fork`），开 PR。推之前要告诉用户，这条分支上有 4 个未推送提交。
3. 边缘设备联网后再同步五台（rog/nx1/nx2/nano1/nano2）重编，跑一遍编译加演示脚本，确认 CUDA/Vulkan 两条路径都没被这轮改动带坏。

### 约束

- PR 用 `gh --repo BianYanhui/EdgeVisor_fork`。不带 `--repo` 会打到 `wansongcc/EdgeVisor`。
- 不要提交 `*.log`、`experiments/algo2/logs/`、`/tmp` 下的结果、论文 PDF、`._*`。
- 引擎是 C++11，编译带 `-Werror`。
- 影子 KV 开着时线程数必须是 1，否则直接拒绝启动。
- 这台机器上的对比一律交替跑：同一个脚本里 base/c3/c4 轮转，不要单跑一遍就报倍速。T4 有 clock drift，单次对比会差 10% 以上。

### 相关 commit

- 基线 `657900f`，已在 `origin/main`，接管乱码修复已经进去。
- 本轮 4 个：`2c2a040`、`b4d0a12`、`6e21fa8`、`1a66663`，全在 `perf/t4-cuda-speed` 上，都没推送。
- 改动面：`EdgeVisor/src/nn/nn-cuda.cu`、`EdgeVisor/src/nn/nn-cuda.hpp`、`EdgeVisor/src/nn/nn-executor.cpp`、`EdgeVisor/src/nn/nn-executor.hpp`、`EdgeVisor/src/nn/nn-network.cpp`、`EdgeVisor/src/dllama.cpp`、`EdgeVisor/Makefile`，共 7 个文件，+1399/−130。
- 这次交接只改 `handsoff.md`，源码没有新改动。
