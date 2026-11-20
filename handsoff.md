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

更新时间：2026-11-20 16:35 CST。分支 `perf/t4-cuda-speed`，基于 `main` `657900f`。工作状态：第 1–4 轮 4×T4 CUDA 提速、第 5 轮（影子 KV 异步 + 尾段采样下沉 + profile 延迟收集）都做完并验证过；4×T4 与 5 机边缘集群都已实测归档，进程全停。现在停下来等用户确认后推分支。

### 现在在做哪件事

`perf/t4-cuda-speed` 上共 8 个 commit，一个都没推送，`origin` 上连这个分支都不存在（`git ls-remote --heads origin perf/t4-cuda-speed` 为空，`git rev-list --count origin/main..HEAD` = 8）。第 5 轮已收尾，结论是：4×T4 单机不掉速（B 臂略升），5 机边缘集群拿到 2.68–2.99× 的端到端提升，掉线接管与重入在 5 机和 4×Jetson 上都复现通过。

### 已经定下来的结论和约束

- 单请求 PP 气泡和 Shadow KV 的语义都要保留。这几轮只动 GPU 侧执行、搬运和采样时序，没有改调度协议、没有改 PP 语义。`--ratios '1@7*1@7*1@7*1@7'`、`--nthreads 1`（影子 KV 要求单线程），以及 Arm B 的 `--enable-pp-migration --runtime-redundant-boundary-layers 3`，都是验证基线，不要为了刷分改掉。
- 不盲目强上 `ppInflight > 1` 去挤干气泡。`EdgeVisor/src/dllama.cpp` 第 701–706 行：只有 remote sampling 可用时 `ppInflight` 才允许多于 1，否则强制回到 1，并打印 “remote last-stage sampling is unavailable; using ppInflight=1 to keep logits sampling correct”。调大会破坏 logits 采样正确性。气泡留着，靠 Shadow KV 藏。
- 不加 `--use_fast_math`。实测它翻转 RMSNorm/softmax/SiLU 的舍入，最后一段的分词就变了；而且并不比纯 `-O3` 快（c1 阶段 nofm 15.11/15.13/15.05 对 fastmath 14.94/14.37/15.02，日志 `/tmp/ev_perf_c1/cmp/c1nofm_A`、`c1fm_A`）。这条已经写进 `EdgeVisor/Makefile` 第 109–115 行的注释，别删。
- 不全局加 `-DNDEBUG`。全局关掉会让单元测试的 `assert()` 失效，`make check` 就白跑。Makefile 里也是按这个写的。
- `DLLAMA_DEFERRED_PROFILE` 的延迟收集**必须**按 `c92796e` 那样门控在尾段采样路径上（`args->lastStageSampling` + 有网络 + 尾段落在 worker 上的 ≥2 段 plan）。lss=0 时尾段用 `SYNC_NODE_SLICES_EXCEPT_ROOT` 直接怼原始 LG float 数组，没有 `LlmWorkerFrameHeader`，延迟一个 token 收 profile 会把上一轮的 PROFILE 帧当 header 读，流直接错位成 “Invalid worker frame magic or version”。别把门控拆掉。
- Algo2 的 `e2eByToken` 必须挂在 `ControllerRuntime` 上（`c92796e`），不能退回 `stepAlgo2()` 里的局部 map。延迟路径下 `algo2NoteTokenComplete(t)` 早于 t 的 PROFILE 帧提交，局部 map 会丢完成事件，决策报 `metric=missing`，实测掉 52% 吞吐。
- 这轮只做 CUDA 路径。CPU 和 Vulkan 的执行语义没动：`NnDevice` 上新加的 `beginForward/endForward/syncPipeToHost/invalidatePipeOnDevice/setLayerPerfSegments` 都是空默认实现。Vulkan 侧只验证了「还能跑、结果对」，没有为 Vulkan 做专门优化。
- 边缘集群的冗余覆盖是配置约束不是 bug：`--runtime-redundant-boundary-layers N` 只覆盖 `2N` 层。stage 有 4 层时 N=2 会留 `suffix=0`，掉线按 `partial-cover` 直接中止；要 4/4 全覆盖就钉 `DLLAMA_REDUNDANT_BOUNDARY_DEPTHS=2,2;2,2;2,2;2,2`，**不要**用 `DLLAMA_REDUNDANT_BOUNDARY_BUDGET`（没接进 plan 构造，root 与 worker 会 desync，报 `Cannot locate op name='block_matmul_q'`）。另外显式传 `--runtime-redundant-boundary-layers` 会把 explicit-boundary-layers 打开并清掉已钉的 depths，两者别同时用。
- 5 机配比 `--ratios '1@8*1@6*1@4*1@5*1@5'`（合计 28 层）是在 8 个候选里实测最快的（12.00 tok/s，次优 `1@8*1@6*1@6*1@5*1@3` 11.45）。依据是 rog 的 Vulkan stage 0 每层约 2.2 ms 最慢，所以它给 8 层；nano 每层约 2.6 ms，给 4 层。
- `rog` 上 RTX 3060 的 Vulkan 索引是 **1**，不是 0：`vulkaninfo --summary` 里 GPU0 是 AMD Radeon（RADV RENOIR，核显），GPU1 才是 RTX 3060。核显必须避开。四台 Jetson 的 Tegra GPU 都是索引 0。Vulkan 二进制运行时需要 `EdgeVisor/src/nn/vulkan/*.spv`（24 个 shader），必须在含该目录的 cwd 下执行。
- 二进制名必须正好叫 `dllama`，因为边缘机上的清理是 `pkill -9 -x dllama`；名字长了 `comm` 会截断，孤儿进程杀不掉。

### 八个提交（都在 `657900f` 之后，按时间顺序）

第 1–4 轮（`657900f..1a66663`，7 个文件，+1399/−130）：

1. `2c2a040` perf(cuda): batch segment transfers, cache op bindings, skip single-node syncs
   `EdgeVisor/src/nn/nn-cuda.cu`、`EdgeVisor/src/nn/nn-cuda.hpp`、`EdgeVisor/src/nn/nn-network.cpp`、`EdgeVisor/Makefile`。段的输入输出改成走各自的 pinned 块、一次 `cudaMemcpyAsync`，去掉每条 pipe 上的 `cudaStreamSynchronize`，`forward()` 不再无条件做整设备同步；每个 op 的显存指针和布局每 plan epoch 解析一次存进 `opBindings`，epoch 变更或 `refreshPointers` 时才重建；纯 host 段（只含 `OP_PLAN_BARRIER`/`OP_PLAN_APPLY`）不搬字节也不发 sync，barrier 不再把 plan 包推给设备；单节点 stage 的 `syncWithRoot`/`syncNodeSlices` 直接返回。Makefile 只加了 `NVCCFLAGS += -O3` 和上面那条注释。
2. `b4d0a12` perf(cuda): dp4a warp-tiled Q80xQ40 matmul and parallel reduction kernels
   只改 `EdgeVisor/src/nn/nn-cuda.cu`。主 kernel `matmulQ80Q40LargeK` 改成一次 warp 算一行加 warp shuffle 归约，加 `__dp4a` 快路（q40 的 nibble 保持 0..15，每 block 减一次 `8*sum(x)`，与旧的 `(w-8)*x` 整数等价），x 组整体进 shared memory，grid 按行组乘以 batch 排。`q80Quantize` 改成一 warp 一个 32 元素块，`invRms` 改成一 block 一列，`multiheadAttScore` 把 query 切片缓存进 shared memory。`NnBlockQ40`(18B)/`NnBlockQ80`(34B) 只保证 2 字节对齐，所以 `qs` 按 uint16 读再拼 uint32，绝不 cast 成 `int*`。
3. `6e21fa8` perf(cuda): four rows per warp in Q80xQ40 Large-K and gate logits stats
   `EdgeVisor/src/nn/nn-cuda.cu`、`EdgeVisor/src/dllama.cpp`。同一个 kernel 改成每 warp 四行（block 32 行，grid 行维缩 4×），x 侧一份 shared memory 被四行复用，每行保留独立累加器，结果与一行一版逐位相同；尾部 warp 用 `rowCount` 收口，非活跃行的权重指针夹到本 warp 最后一行，避免越界。`dllama.cpp` 新增 `logitsStatsNeeded()`，128256 个 logits 的扫描只在 `DLLAMA_DEBUG_TOPK_LOGITS`、`productLogLevel() >= 1`、或 root logits 分片导出时才做。
4. `1a66663` perf(cuda): keep pipes resident on the GPU for the whole single-device forward
   `EdgeVisor/src/nn/nn-cuda.cu`、`EdgeVisor/src/nn/nn-cuda.hpp`、`EdgeVisor/src/nn/nn-executor.cpp`、`EdgeVisor/src/nn/nn-executor.hpp`。单设备（一台顶一个 stage）时，整段前向的 pipe 留在显存：段之间不再下到 host 再上传；只有 PP 发送、logits 汇聚、多节点 slice 交换这几处会 `syncPipeToHost`，其余 pipe 标成 device-dirty；每个 sync 之后只失效它可能写过的 pipe。中间段因此省掉整条 PCIe 往返和每段一次的 `cudaStreamSynchronize`。这一条也是边缘 5 机拿到 2.7–3.0× 的主因。

第 5 轮（`1a66663..c92796e`，7 个文件，+540/−167）：

5. `5288900` perf: isolate shadow KV scratch buffers and run root_wait shadow async
   `EdgeVisor/src/llm.cpp`、`EdgeVisor/src/nn/nn-executor.cpp`、`EdgeVisor/src/nn/nn-executor.hpp`。影子 KV 原来和主计算共用 scratch，只能等所有 executor 线程返回后才跑。现在给冗余 pass 独立 buffer，再交给 helper 线程在 PP `root_wait` 那段死时间里跑；没有 root-wait 步的节点仍走 forward 后的 drain 兜底，`forward(deferBubbleShadow)` 让尾段在 logits D2H 已发布后自己收尾 drain。
6. `dd5eb16` perf: split last-stage sampling and defer profile collection
   `EdgeVisor/src/app.cpp`、`EdgeVisor/src/app.hpp`、`EdgeVisor/src/dllama.cpp`。尾段原来要跑完整个 forward 才采样；root 原来在 `forward()` 之后立刻排空每个 worker 的 PROFILE 帧，两处都在每 token 的关键路径上。现在尾段在 logits D2H 一落地就采样并先发 `SAMPLED_TOKEN`，再跑 `runDeferredBubbleShadowKv()` 和 profile 包；root 侧 profile 排空延迟一个 token（`DLLAMA_DEFERRED_PROFILE`，默认开），`forward()` 收 `deferProfileCollect` 并用 `didLastForwardCollectProfile()` 回报本 token 有没有真排空，保证 failover 路径正确；计账搬进 `recordProfileForToken()`，循环末尾有 flush，最后一个 token 不会丢。
7. `c92796e` fix: gate deferred profile on last-stage sampling and persist Algo2 e2e map
   `EdgeVisor/src/dllama.cpp`、`EdgeVisor/src/dynamic/dynamic_tpot.cpp`。修 `dd5eb16` 带出的两个回归：(a) lss=0 时延迟排空会让上一 token 的 PROFILE/STAGE_SPEED 帧顶着原始 logits 传输，root 把帧头当 logits 读，流错位；现在延迟只在 `args->lastStageSampling` + 有网络 + 尾段在 worker 上的 ≥2 段 plan 下生效，Arm A/B（lss=0）从 3/3 崩变成全清。(b) `stepAlgo2()` 的 token→interval map 是局部变量，延迟路径下完成事件先于 PROFILE 提交，观测找不到 interval 就报 `metric=missing`；现在挂到 `ControllerRuntime::e2eByToken`，消费即删、按 1024 条最旧优先裁剪，只有 controller 线程碰它。
8. `919e4df` Update handsoff for 4xT4 CUDA inference speedups — 只改 `handsoff.md`，第 4 轮收尾时的交接文档。

### 实测结果（4×T4，同机交替跑）

- 第 5 轮定稿脚本 `/tmp/ev_r5_fix1_arms.sh <rounds>`（Arm A/B/C）与 `/tmp/ev_r5_fix1_bmore.sh`（B 臂补测），日志 `/tmp/ev_r5_t4_verify/fix1/summary.txt`、`fix1_Bmore/summary.txt`。公共配置：`--ratios '1@7*1@7*1@7*1@7'`、`--nthreads 1`、`--steps 64`、`--temperature 0 --seed 1`、`--benchmark --buffer-float-type q80 --max-seq-len 512`、`--prompt "What is 2+2? Answer briefly."`。A = 纯 PP；B = Shadow KV + migration + `--runtime-redundant-boundary-layers 3`（lss=0）；C = B + `DLLAMA_LAST_STAGE_SAMPLING=1`。
- 数值（tok/s）：A 49.72 / 52.39，B 53.15 / 52.13，C 53.39 / 50.94；B 臂另补 4 次 51.26 / 53.76 / 47.34 / 53.87。对照第 4 轮的 c4（A 51.73、B 49.81），第 5 轮在 4×T4 上不掉速、B 臂略有上升。**4×T4 不是第 5 轮的主战场**，第 5 轮的收益主要在 5 机边缘集群上体现（见下一节）。
- 文本一致性：上面全部 run 的 `txt.md5` 都是 `fa945fe6b651db1da18c55e91978bd24`，与本轮 base 逐字节相同。第 5 轮改动没有动采样分布。
- `nn-cuda-test`：133/133 全 `ok`，rc=0（`/tmp/ev_r5_t4_verify/nn-cuda-test.log`，本次又在 HEAD 上复跑到同一结果 `/tmp/ev_r5_nnct_final.log`）。绝大多数 `max_abs_error=0`，少数归约项是 2.98e-08 / 5.96e-08 / 1.19e-07 的浮点噪声。
- `make check`：`CHECK_TESTS` 共 10 项，其中 9 项常规项全过（`algo2 checks passed`、`dynamic_tpot_algorithm PASS`、`device profile checks passed`、`failover gate checks passed` 等）；第 10 项 `pp-failover-network-test` 不能跟着默认跑，默认 TCP 超时下它会挂在 `testBufferedSendNoAnswer` 的 `assert(elapsed >= 400 && elapsed < 2500)`（`/tmp/ev_make_check.log` 就是这次失败），要带 `DLLAMA_TCP_USER_TIMEOUT_MS=1000 ./pp-failover-network-test` 单独跑，末尾是 `PP recovery, buffered tail frames, bounded timeouts, protocol rejection and shadow KV passed`。带该环境变量时整个 `make check`（含 `reserved-join-test`、`delayed-sample-worker-test`）一次通过，日志 `/tmp/ev_r5_check_final.log`。
- Algo2 恢复：`/tmp/ev_r5_algo2_fix.log`。`DEFERRED_PROFILE=1` 下 `5.85 tok/s (170.84 ms/tok)`、`metric_missing=0`、`migration_issued=1`、`lss_auto=1`；修复前是 `2.60 tok/s`、`metric_missing=105`（`/tmp/ev_r5_t4_verify/algo2_def/def1_a/tpot.log`）。另一轮 `def1_b` 5.84、`def0_a` 5.94，三条都是 rc=0。即延迟 profile 现在不再饿死控制器。
- 4×T4 掉线重入：`/tmp/ev_r5_t4_verify/rejoin_out.txt`。杀 w1 再拉起，`[failover] split deadNode=1 layers=[8,12) prefix=4 suffix=0`、`[pool] rejoining ... at pos=40`、`catch-up ... seal=1`、`node=1 rejoined stage=1 layers=[8,12) kv through 50`，跑到 `[e1-end] pos=500`，`5.34 tok/s (187.12 ms/tok)`，root 退出码 0。输出仍是英文，没有回到乱码。

### 实测结果（5 机边缘集群）

- 拓扑：`rog` = stage 0 root（Vulkan，RTX 3060 Laptop，6 GB 显存，8 层）、`nx1` = stage 1（Orin cc 8.7，15.6 GB，6 层）、`nx2` = stage 2（Orin cc 8.7，15.6 GB，4 层）、`nano1` = stage 3（Orin cc 8.7，7.6 GB，5 层）、`nano2` = stage 4（Orin cc 8.7，7.6 GB，5 层）。`1@8*1@6*1@4*1@5*1@5` = 28 层，llama3.2-3b-instruct Q40。四台 Jetson 是 CUDA `sm_87`。
- 主表：驱动 `/tmp/ev_r5_bench.sh`，产物 `/tmp/ev_r5_edge_verify/gpu_5dev/`（`matrix.log` 是驱动日志，每个 run 目录带 `rc.txt`、`valid.txt`、`wall_s.txt`）。3 轮交错，中位数（tok/s，括号内 ms/tok）：

  | 臂 | `657900f` | `c92796e` | 加速比 |
  | --- | --- | --- | --- |
  | E-A 纯 PP | 4.79（208.98） | **12.83**（77.93） | **2.68×** |
  | E-B Shadow KV | 4.45（224.72） | **12.99**（76.97） | **2.92×** |
  | E-C Shadow + 尾段采样 | 4.63（216.04） | **13.85**（72.20） | **2.99×** |

  逐轮：E-A base 4.61/4.84/4.79、r5 12.83/9.78/13.00；E-B base 4.45/4.70/4.43、r5 12.96/12.99/13.05；E-C base 4.64/4.63/4.60、r5 13.81/13.88/13.85。r2 的 E-A 9.78 是唯一低样本，它的 stage 0 exec 是 22.2 ms 而另外两轮 4.5 ms，属 rog 宿主机噪声，中位数没受影响。
- 收益从哪来（E-B，c92796e，ms）：stage 0 `rootWait` 84.1（原 211.8）、stage 1 `sync` 4.8（原 25.8）、stage 2 `sync` 19.1（原 64.1）、stage 3 `sync` 29.1（原 92.6）；exec 几乎不动（stage 0 4.4 对 25.5）。所以赢的是 root-wait/PP-sync 串行链塌缩，不是矩阵算得更快。E-C 再把 stage 4 exec 从 18.2 打到 1.0 ms（`LAST_STAGE_SAMPLING`），这就是比 E-B 多的那约 0.9 tok/s。E-B/C 多付一点 bubble/drain（stage 0 bubble 0.8–1.1 ms、drain 约 1 ms），是影子 KV 机器的预期代价。
- 文本一致性（**已复核**）：9 次 base 全部同一份文本、9 次 r5 全部另一份，各臂各轮内部逐字节相同。base = `fa945fe6b651db1da18c55e91978bd24`（256 B，含尾换行），与 4×T4 的参考文本是同一份；r5 = `282a87b320ce0aba06317853f364a481`。两版前 213 字节完全相同，只在尾部差最后一次 detokenize 窗口：base 收在 `... I'm glad I could correct my earlier response`，r5 收在 `... I made a mistake earlier. The correct answer`。这是尾段采样路径改变的预期后果，不是采样分叉（同 prompt、同 seed、同权重）。
- 注意 `SUMMARY.md` 里写的 `70a851d2b6ee4d25...` / `127322d3b9ddb029...` 这两个 md5 **复现不出来**，按它的算法重算得到的是上面 `434ad643...` / `4485ed8b...`（不含尾换行）或 `fa945fe6...` / `282a87b3...`（含尾换行）。`REPORT.md` 里的 `434ad64357849fc65d1ceb798de45c78` / `4485ed8b38f0503ee11b62b62a9fcd9b` 才对得上。以 `REPORT.md` 和本文为准。
- 5 机 GPU 掉线重连（c92796e）：`/tmp/ev_r5_edge_verify/gpu_5dev_rejoin/`，配置同上但 `--steps 200`、prompt 换成 “Explain pipeline parallelism and fault tolerance in edge clusters.”，端口 18097。`nx2` 上装本地看门狗轮询自己的 worker 日志（50 ms），所以 kill 精确落在目标位置而不是晚一个 SSH 往返：`SELFKILL pos=40`。root 侧 4/4 全覆盖：

  ```
  [network-membership] deactivated node=2 socket=1
  [failover] split deadNode=2 layers=[14,18) prefix=2 suffix=2 prevMs=16 nextMs=13 front=[14,16) back=[16,18)
  [failover] takeover deadNode=2 stage=2 prev=1 layers=[14,16) next=3 layers=[16,18) myNode=0
  ```

  两个邻居都确认接管（`nx1` myNode=1 拿 `[14,16)`，`nano1` myNode=3 拿 `[16,18)`）。8 s 停机期间推理继续解码，没有 stall、没有重启、root 没退出。重新拉起后 `[pool] rejoining ... at pos=120`，KV 按 16 位置一块追平到 `kv=[0,138) seal=1`，`node=2 rejoined stage=2 layers=[14,18) kv through 138`，最后 `[e1-end] pos=200`、`tokens/s: 9.23 (108.30 ms/tok)`、root exit=0，驱动记 `took_over=1`。
- 5 机 CPU 校验（c92796e）：`DLLAMA_CUDA=0 DLLAMA_VULKAN=0` 在 nx1 上编好发给其余 Jetson，rog + 4 Jetson 全 `--backend cpu`、64 步，`tokens/s: 1.87 (534.36 ms/tok)`，rc=0（`/tmp/ev_r5_edge_verify/gpu_5dev/cpu1.out`）。约比 GPU 混跑慢 7×，但它证明同一套 plan 和重入管线在没有任何加速器时也成立。

### 实测结果（4×Jetson CUDA，无 rog）

- `nx1` 作 root，四台 Jetson 全 CUDA，`--ratios '1@9*1@7*1@6*1@6'`，3 轮交错，产物 `/tmp/ev_r5_edge_verify/logs/ev_cmp_out/js_*`、汇总 `js_sweep_summary.txt`。中位数：

  | 臂 | `657900f` | `c92796e` | 加速比 |
  | --- | --- | --- | --- |
  | J-A 纯 PP | 4.40 | **10.93** | 2.48× |
  | J-B Shadow KV | 4.29 | **10.88** | 2.54× |
  | J-C Shadow + 尾段采样 | 4.48 | **11.56** | 2.58× |

  方向与 5 机一致，量级略低，因为少了 rog 那个 Vulkan stage 0。

### 机器上还留着什么

- 全部空闲，本轮已确认：`yhs1` 四张 T4 都是 0 MiB / 0%，`nvidia-smi --query-compute-apps` 为空。`rog`、`nx1`、`nx2`、`nano1`、`nano2` 上 `pgrep -x dllama` 全空。
- 五台边缘机上唯一在听的是 `:9993`，那是 `zerotier-one.service`（`ss -ltne` 显示 `cgroup:/system.slice/zerotier-one.service`），不是 dllama，别去杀。
- `EdgeVisor/dllama` 还在（2.6 MB，md5 `a8f527b968e62e5514302d70f560a1bc`），是 HEAD 编出来的 CUDA 版；被 `EdgeVisor/.gitignore:17` 的 `/dllama` 忽略，不会进提交。
- **未跟踪、绝对不要动**：`experiments/algo2/logs/t4_rejoin/`。`git status --short` 至今只有这一行。
- `/tmp` 下不要删的产物：`/tmp/ev_r5_edge_verify/`（5.5 MB，含 `SUMMARY.md`、`REPORT.md`、`gpu_5dev/`、`gpu_5dev_rejoin/`、`logs/5node/`、`logs/ev_cmp_out/`、`src/` 里两版源码 tar）、`/tmp/ev_r5_t4_verify/`（4.4 MB，第 5 轮 4×T4 全部 run）、`/tmp/ev_perf_base`、`/tmp/ev_perf_c1`（17M，含 fast-math 对照）、`/tmp/ev_perf_c2`、`/tmp/ev_perf_c3`、`/tmp/ev_perf_c4`（9.1 MB，第 4 轮全部日志）。脚本：`/tmp/ev_c4_cmp.sh`、`/tmp/ev_r5_fix1_arms.sh`、`/tmp/ev_r5_fix1_bmore.sh`、`/tmp/ev_r5_algo2_fix.sh`、`/tmp/ev_r5_bench.sh`、`/tmp/ev_r5_cmp.sh`、`/tmp/ev_r5_dropout.sh`、`/tmp/ev_r5_summary.py`、`/tmp/ev_final_table.py`、`/tmp/ev_md5.py`、`/tmp/ev_textcmp.py`。
- 原有的还在：`/tmp/ev_matrix_v2/`（1.2G，8B 矩阵，已跑完，别重开）、`/tmp/ev_e1_deprecated/`（74M）、`/tmp/ev_st3b/`（7.3M）。以及 `/home/byh/B01/baseline`（分支 `feat/remp-pp`，带未跟踪的 `_harness/dynapipe_ab.py`、`_harness/dynapipe_sampling.py`）。都不要删，也不要从这边顺手提交。
- **`rog` 的 `/tmp` 已被清空**：第 5 轮中途 `rog` 因显存打爆掉卡（29 个进程抢 6 GB，`nvidia-smi: Unknown Error`），无 sudo 密码只能整机重启，`/tmp` 在根分区所以旧目录（`/tmp/ev_demo1`、`/tmp/ev_edge_*`、`/tmp/ev_algo2_*`）在 rog 上没了。不是谁删的，别去别处找。`nx1/nx2/nano1/nano2` 上的对应目录都完好。
- 磁盘：`yhs1` 的 `/` 98G 用了 69%（30G 空），`/home` 7.3T 用了 73%（1.9T 空）。

### 下一步

用户说「继续」之前不要开长实验、不要推分支。说继续时按顺序：

1. **先问用户能不能推**，然后 `git push -u origin perf/t4-cuda-speed`。这条分支上有 8 个未推送 commit，`origin` 上还没有这个分支，推完再按需开 PR（`gh --repo BianYanhui/EdgeVisor_fork`）。
2. 如果要继续提速：边缘 5 机上 stage 0（rog，Vulkan）仍是最大单项，每 token 的 `rootWait` 84 ms 还在；下一步该看 Vulkan 侧的段搬运和 `rootWait` 能不能像 CUDA 那样异步化，而不是继续动 CUDA kernel。另外 `E-C` 的 stage 4 exec 已经打到 1.0 ms，尾段基本榨干。
3. `SUMMARY.md` 的 md5 字段需要修（见上文），修完再当对外结论引用。

### 约束

- PR 用 `gh --repo BianYanhui/EdgeVisor_fork`。不带 `--repo` 会打到 `wansongcc/EdgeVisor`。
- 不要提交 `*.log`、`experiments/algo2/logs/`、`/tmp` 下的结果、论文 PDF、`._*`。
- 引擎是 C++11，编译带 `-Werror`。
- 影子 KV 开着时线程数必须是 1，否则直接拒绝启动。
- 这台机器上的对比一律交替跑：同一个脚本里 base/r5 轮转，不要单跑一遍就报倍速。T4 有 clock drift，单次对比会差 10% 以上。边缘 5 机同样要交替，且每个 run 都要过有效性门（rc=0、跑到 `[e1-end] pos=64`、四个 worker 都加载权重、日志里没有 `deactivated`/`peer offline`）。
- 5 机上有别的会话在并行作业（`nx2` 上的 `ev_r5_w_covchk_*`、`ev_r5_selfkill_rejoin{1..4}.log` 等），它的清理按名字杀 `dllama` 不管端口，会把别人的 run 打断（rc=137 或 worker 半路消失）。看到诡异失败先怀疑这个，重跑前把失败目录清掉，别把废数据留在表里。

### 相关 commit

- 基线 `657900f`，已在 `origin/main`，接管乱码修复已经进去。
- 第 1–4 轮：`2c2a040`、`b4d0a12`、`6e21fa8`、`1a66663`。
- 第 5 轮：`5288900`、`dd5eb16`、`c92796e`（当前 HEAD）。
- 文档：`919e4df`（第 4 轮 handsoff）。
- 8 个 commit 全在 `perf/t4-cuda-speed` 上，都没推送。
- 改动面（不含 `handsoff.md`）：11 个文件，+1941/−299。即 `EdgeVisor/Makefile`、`EdgeVisor/src/app.cpp`、`EdgeVisor/src/app.hpp`、`EdgeVisor/src/dllama.cpp`、`EdgeVisor/src/dynamic/dynamic_tpot.cpp`、`EdgeVisor/src/llm.cpp`、`EdgeVisor/src/nn/nn-cuda.cu`、`EdgeVisor/src/nn/nn-cuda.hpp`、`EdgeVisor/src/nn/nn-executor.cpp`、`EdgeVisor/src/nn/nn-executor.hpp`、`EdgeVisor/src/nn/nn-network.cpp`。其中第 5 轮新增/改动的是 `app.cpp`、`app.hpp`、`dllama.cpp`、`dynamic/dynamic_tpot.cpp`、`llm.cpp`、`nn/nn-executor.cpp`、`nn/nn-executor.hpp` 这 7 个。
- 这次交接只改 `handsoff.md`，源码没有新改动。
