# 在线迁移（PlanCommand + UDS 控制器）使用与验证指南

本文说明如何在 `dllama` 推理过程中，通过 **UDS（Unix Domain Socket）控制器**下发一次性迁移命令，并验证迁移已经生效。

> 适用范围：当前实现主要面向 **CPU 路线**的 online repartition 实验（plan barrier/apply 是 CPU-only hook）。

---

## 1. 背景与机制

在启用 plan barrier 后，执行图里会插入两类 CPU-only op：

- **`OP_PLAN_BARRIER`**：每层/每次 forward 到达 barrier 时读取本地缓存的 `PlanCommand`。
  - 若命中触发条件：在 plan pipe 里写入一条“迁移控制消息”（epoch + kind + from/to + move…），并把该 `PlanCommand` **one-shot 消费**（清空缓存）。
  - 只有“stage root”节点会真正 emit（避免多点重复触发）。

- **`OP_PLAN_APPLY`**：读取 plan pipe 的控制消息。
  - 当 `msgEpoch > curEpoch` 且消息有效时，更新 `NnUnevenPartitionPlan`（head/ffn split 长度与起点），并调用 `device->setPlanEpoch(msgEpoch)`。
  - 下一次执行到来前，`NnCpuDeviceSegment::forward()` 会检测 epoch 变化并自动 `refreshPointers()`，使后续 op 使用新切分。

因此，“迁移起效”的可观测结果至少应包括：
- barrier 侧：确实触发并 emit 了一个更大的 epoch；
- apply 侧：节点把 plan 更新并 bump epoch；
- 执行侧：pointer/slice 配置随 epoch 更新发生变化。

通信实现基于Unix Domain Sockets 用于同一台主机上的进程间通信（IPC）。它的 API 和网络套接字（TCP/IP）非常相似，但不通过网络卡，而是通过文件系统中的一个“路径”作为地址，因此速度极快且开销很小。

---

## 2. 开启与启动

### 2.1 必要开关
#### 作为环境变量开关在启动时开启

- `DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock`
  - 设置后 root 进程会启动 UDS 控制器线程。
  - 进程启动后会在 stderr 打印：`[plan-uds] listening on /tmp/dllama_plan.sock`。


UDS 控制器协议说明与客户端示例见：
- [docs/README_ENV_VARS.md](README_ENV_VARS.md)
- 客户端脚本：[examples/plan-uds-client.py](../examples/plan-uds-client.py)

kv冗余计算
 - 单个值：`--kv-redundancy 2`  所有节点使用 2 个冗余 head
 - 每个节点：`--kv-redundancy 2,3,2,3`  每个节点指定不同数量
 - **ps:kv冗余计算配置为全局配置，如果代码没有彻底退出，将一直保留该配置**

### 2.2 运行

用 3 个 worker 进程模拟 4 节点，验证 root→worker 分发与 apply 是否都正常。
#### 容器准备
```
$cloud25
cd /home/cc/docker_file
export WORKER_NAME=worker-node-0X
docker compose -f docker-compose-rpi.yml up -d
```
 -已经配置好了网络连接，使用`rpi-net`连接各个节点，在连接是不需要知道ip，使用容器名即可
#### 终端 A：启动 worker

```bash
cd /workspace/dllama/distributed-llama
./dllama worker --port 9999 --nthreads 4
```

#### 终端 B：启动 root 推理（inference）

下面示例用仓库自带模型（可换成你的模型）。关键点是：
- `--workers 127.0.0.1:9999` 让 root 连接本地 worker；
- `--ratios 1:1` 形成 stage 内 2 节点（示例）；
- 开启 `DLLAMA_ENABLE_PLAN_BARRIER` + `DLLAMA_PLAN_CTRL_SOCKET`。

```bash
cd /workspace/dllama/distributed-llama
./dllama inference \
  --interactive \
  --prompt "The capital of France is" \
  --steps 64 \
  --model /workspace/dllama/distributed-llama/models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m \
  --tokenizer /workspace/dllama/distributed-llama/models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t \
  --buffer-float-type q80 \
  --enable-plan-barrier \ #启用在线迁移
  --kv-redundancy 2 \
  --nthreads 2 \
  --max-seq-len 2048 \
  --benchmark \
  --workers rpi-trixie5:9999 rpi-trixie2:9999 rpi-trixie6:9999 \
  --ratios "1:1*1:1*1:1"
```

> 如果你使用多 stage（如 `--ratios 1:1*1:1`），请在命令里用正确的 `stageIndex` 指向想迁移的 stage。

---

## 3. 下发迁移命令（UDS）

UDS 是“一次连接一条请求”的 JSON 行协议：
- 客户端发一行 JSON（以 `\n` 结尾）
- 服务端回一行 JSON

### 3.1 先确认控制器可用

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock ping
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock status
```

`status` 会返回：
- 当前缓存的 `PlanCommand`（含 `mode/seq/stageIndex/from/to/...`）
- 以及 root 推理的 `position/batchSize/perfSamples`（用于确认推理在跑）

### 3.2 下一次 barrier 触发模式：`next_barrier`

“下一次 barrier 立即触发”（用于真实运行时最稳的方式）：

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock set_plan \
  --seq 1 --mode next_barrier \
  --stage 0 --from 0 --to 1 \
  --kind 3 --heads 1 --ffn 256
```

含义（默认字段）：
- `mode=next_barrier`：下一次 barrier 触发后立刻 one-shot 消费
- `stage=0`：只允许该 stage 的 stage root emit
- `from/to`：在该 stage 内从哪个 node 迁移到哪个 node
- `kind=3`：`1=headSplit` `2=ffnSplit` `3=both`
- `heads/ffn`：迁移量（注意：head 迁移在 GQA 情况下会按 KV-head lockstep 处理）
- **`seq` 信令编号：同一次代码运行中（没有退出代码，而不是同一次推理中）携带相同的seq的指令二次接收到后会被抛弃，如果在同一次运行中，每次的指令的seq需要递增**

### 3.3 调试触发模式：`exact`

只有当到达“指定 token position + layer”才触发：

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock set_plan \
  --seq 2 --mode exact \
  --stage 0 --from 0 --to 1 --kind 3 --heads 1 --ffn 256 \
  --trigger-pos 30 --trigger-layer 5
```

用于：
- 精准复现某个 layer/pos 处的迁移行为
- 对照迁移前后某个固定点的 slice/pointer/kv 范围变化
- ps：seq同上，注意递增

---

## 4. 如何验证“迁移起效了”

这里给出两种验证强度：

### 4.1 轻量验证（不改二进制）：确认命令被消费 + 推理仍在跑

1) 下发 `set_plan`（建议用 `next_barrier`）
2) 反复 `status`

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock status
```

你应该看到：
- `status.cmd.mode` 从 `next_barrier/exact` 变回 `none`（one-shot 被消费）
- `position` 持续增长（说明推理在进行，barrier 有机会执行）

这能确认：
- UDS → root cache 写入正常
- root → worker 分发（通过 control packet piggyback）能触发缓存更新
- barrier 至少在 stage root 上命中过一次并消费了命令

但它 **不能 100% 证明 apply 改变了 plan**（只证明触发路径工作）。想严格证明 apply 生效，建议用 4.2。

### 4.2 严格验证（推荐）：开启在线迁移日志并观察 emit/apply

`OP_PLAN_BARRIER` 和 `OP_PLAN_APPLY` 的关键日志位于编译期宏 `DLLAMA_DEBUG_ONLINE_CHANGE` 下。推荐用它来验证迁移确实 apply 并 bump epoch。

重新编译（只影响 dllama）：

```bash
cd /workspace/dllama/distributed-llama
make clean
DLLAMA_DEBUG_ONLINE_CHANGE=1 make -j dllama
```

然后按第 2 节启动 root/worker，并下发 `set_plan`。

观察日志：
- barrier emit（只在 stage root）：`🧭 [plan][emit] ... epoch=... from=... to=...`
- apply（stage 内所有相关节点）：`🧭 [plan][apply] ... epoch=... headLen[...] ...` 或 `ffnLen[...] ...`

当你看到 `apply` 日志并且 epoch 递增，基本可判定迁移已生效（并会触发后续 `refreshPointers()`）。

### 4.3 进一步验证（可选）：用 slice/KV tracing 观察“切分范围变化”

如果你还想观察迁移前后某些 op 的 slice 范围变化，可以配合注意力调试能力（需要 `DLLAMA_DEBUG_ATTN=1` 编译）使用：
- KV 范围：`kvcache_debug=1`（或 `DLLAMA_DEBUG_KV_RANGE=1`）
- per-head dump：`DLLAMA_DEBUG_KVCACHE_PER_HEAD=1`

相关变量清单见：[docs/README_ENV_VARS.md](README_ENV_VARS.md)

---

## 5. 常见问题排查

### 5.1 socket 文件没有出现 / `ping` 连接失败

- 确认 root 进程设置了：
  - `DLLAMA_ENABLE_PLAN_BARRIER=1`
  - `DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock`
- root stderr 应该打印：`[plan-uds] listening on ...`
- 若提示 bind 失败：删除旧 socket：`rm -f /tmp/dllama_plan.sock`

### 5.2 `status` 里 cmd 一直不变（没有被消费）

- 你下发的是 `exact` 模式，但 `(triggerPos, triggerLayer)` 从未命中：
  - 改用 `next_barrier`
  - 或降低 `--trigger-pos`，并保证推理步数足够大
- `stageIndex` 不匹配导致 stage root 不 emit：
  - 先用单 stage（`--ratios 1:1`，`stageIndex=0`）跑通

### 5.3 看到了消费，但不确定 apply 是否真的改变了 plan

- 使用 4.2 的 `DLLAMA_DEBUG_ONLINE_CHANGE=1` 重新编译并观察 `🧭 [plan][apply]` 日志
- 或配合 4.3 的 KV/slice tracing 做更“黑盒”的证明

### 5.4 from/to 不在同一 stage 或迁移量非法

`OP_PLAN_APPLY` 只会在 `from` 和 `to` 都属于目标 stage 时生效，并且：
- `headMove` 不能超过 `from` 的可迁移 head 数
- `ffnMove` 不能超过 `from` 的可迁移 ffn 长度

遇到不生效时建议先用很小的迁移量（例如 `--heads 1 --ffn 256`）跑通。

---

## 6. 推荐的测试顺序（最省时间）

1) 单机 2 进程（root+worker）跑通 + `ping/status` 正常
2) `next_barrier` 下发命令，确认 `status` 里 cmd 变回 `none`
3) 打开 `DLLAMA_DEBUG_ONLINE_CHANGE=1` 重编译，确认 `emit/apply` 日志
4) 需要更强证据时，再加 `DLLAMA_DEBUG_ATTN=1` 的 KV/slice tracing
