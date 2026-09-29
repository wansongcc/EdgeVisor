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

2026-09-29 夜里查完多层挪动卡住的原因，并复测过。nx1 `10.47.145.51` 和 nx2 `10.47.235.49` 上现在没有 `dllama`，没有 `gpu_hog`。

卡住的原因：往后面的 stage 交层时，root 和 worker 用同一条 socket。worker 在读到 KV 控制包之前，已经把这个 token 的采样帧写出去了。root 等 KV ack 时把 `LLM_WORKER_FRAME_SAMPLED_TOKEN` 当成干扰帧读掉并丢弃。`forward()` 返回后，`tryReceiveLastStageSampledToken` 还在等这帧，worker 已经在等下一条控制包，两边停住。日志最后一行 `[kv-collector] layer=21 pos=21` 是采集线程打的，主线程已经打过 `recover status=ok`。反向挪动（stage 1 到 0）不读这条 KV ack，所以一步 1 层那次 263 ms 到 241 ms 的成功不受这个丢帧影响。

改动只在 nx1 的 `/home/jetson/cc/edgevisor_fresh/EdgeVisor/src/app.cpp`，还没提交。等 KV ack 时，采样帧、profile 帧、stage-bypass ack 放进原来的缓存。收 profile 时如果先读到采样帧，也放进缓存。nx1 已 `make dllama`（cuda=1 vulkan=1）。nx2 的 worker 二进制仍是 2026-09-28 17:51，没有重编。

复测用的还是 Qwen3-0.6B Q40，nx1 root，nx2 worker `192.168.137.15:18081`，不写 `--ratios`，切分 `1@22*1@6`，提示词 `The capital of France is`，`DLLAMA_TPOT_MAX_PP_LAYER_MOVE=2`。

- 没有波动时，步长 2 的候选收益低于阈值，没有发出迁移。生成自己停在 EOS，`tokens/s: 20.15 (49.64 ms/tok)`。日志 `/tmp/edgevisor_p2_auto_fix.log`、`/tmp/edgevisor_p2_tpot_fix.log`。
- nx2 上 `/tmp/gpu_hog 4096` 在生成开始前就已经在跑，不是严格卡在第 12 条 `[token-e2e]` 之后。调度器发出 `route=1->0 layers=[22,23]`，`recover status=ok stallMs=0.247 stateBytes=0`。token 继续到 pos=281，句子还是通的。窗口从 193.6 ms 到大约 188–191 ms，比一步 1 层的 263 ms 到 241 ms 恢复得少。再提议 layer 24 时被拒绝：`target stage lacks provisioned layer 24`，冗余边界只有这两层。日志 `/tmp/edgevisor_p2_auto_fix4.log`、`/tmp/edgevisor_p2_tpot_fix4.log`。
- 原来卡住的方向用 root 侧延迟打出来：`DLLAMA_TEST_COMPUTE_DELAY_US=8000`，node 0，第 12 个 decode token 之后，只延迟 primary。`route=0->1 layers=[20,21]`，`ack batch complete`，`recover status=ok stallMs=53.282`，之后 token 一直到结束（pos=119，`tokens/s: 1.41`）。验证窗口 423 ms 变成 736–745 ms，调度器回滚；回滚之后仍然继续出 token。这次变慢是 worker 上的冗余副本比 root 上被延迟的 primary 更贵，不是再卡住。日志 `/tmp/edgevisor_p2_auto_fix5.log`、`/tmp/edgevisor_p2_tpot_fix5.log`。

phase-2 源码仍然不要提交，除非用户明确要求。这次 `app.cpp` 的修复也不要和那些未提交改动捆在一起提交。不要合到 main。不要做名单外的中途加入。
