# Dynamic Scheduling UDS API (Plan Controller)

本文档描述 **Plan UDS Controller** 的 JSON-line 接口，用于：

1. **检测运行时耗时**（stage / node 的执行与同步耗时；以及 layer 级别的 attention/ffn 计算耗时）。
2. **下发动态 plan 修改指令**（在线迁移 head/ffn slice），并在下一次 barrier 生效。

> 适用场景：多机/多节点推理时，外部进程（调度器）根据实时性能数据触发迁移。

---

## 1. 启用条件（必须）

### 1.1 启用 UDS Server

Plan UDS Controller 仅在设置环境变量后启动：

- `DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock`

Root 进程启动后会监听该 Unix Domain Socket。

### 1.2 启用推理侧 plan barrier（决定“是否真的会迁移”）

**必须**在启动推理（root 和 workers）时启用 `--enable-plan-barrier`，否则推理图里不会插入 `OP_PLAN_BARRIER/OP_PLAN_APPLY`，即使 `set_plan` 返回 `ok:true` 也不会真正 emit/apply。

> workers 侧是否启用由 root bootstrap 下发；无需额外在 worker 进程传同样参数（但允许）。

---

## 2. 协议概览

- 传输：Unix Domain Socket（stream）
- 编码：UTF-8 文本
- 每次请求：**一行 JSON**（以 `\n` 结尾）
- 每次响应：**一行 JSON**（以 `\n` 结尾）
- 连接模型：客户端每次请求建立一次连接，请求→响应→断开

### 2.1 通用字段

请求格式：

```json
{"op":"<operation>", ...}
```

响应格式：

```json
{"ok":true, ...}
```

失败时：

```json
{"ok":false, "error":"..."}
```

---

## 3. 如何检测 stage 运行时间

目前提供两类观测：

1. **粗粒度**：每个 node 的执行/同步耗时（`perf`）
2. **细粒度（layer 级）**：每层每 node 的 attention/ffn 计算耗时（`layer_prof`）

### 3.1 `op=status`（运行状态 + 是否开启 barrier）

用途：

- 确认 UDS 正常工作
- 确认 `enablePlanBarrier` 是否为 `true`
- 获取推理 position / batchSize 等信息（如果 inference 已挂载）

请求：

```json
{"op":"status"}
```

响应（示例）：

```json
{
  "ok": true,
  "enablePlanBarrier": true,
  "cacheSeq": 138,
  "cmd": { "mode": "next_barrier", "seq": 69, ... },
  "position": 128,
  "batchSize": 1,
  "perfSamples": 4
}
```

说明：

- `enablePlanBarrier=false` 时，`set_plan` 只会更新 cache，不会触发迁移日志。

### 3.2 `op=perf`（node 执行/同步耗时）

用途：快速判断“哪个 node 慢”、以及慢主要来自执行还是同步。

请求：

```json
{"op":"perf"}
```

响应（示例）：

```json
{
  "ok": true,
  "perf": [
    {"position":128,"batchSize":1,"nodeIndex":0,"stageIndex":0,"execUs":24000,"syncUs":5000},
    {"position":128,"batchSize":1,"nodeIndex":1,"stageIndex":0,"execUs":31000,"syncUs":8000}
  ]
}
```

字段：

- `execUs`: 本轮执行耗时（微秒）
- `syncUs`: 同步/通信耗时（微秒）

### 3.3 `op=layer_prof`（layer 粒度 attention/ffn 统计）

用途：做“最细粒度”的调度判断，定位每层每 node 的 attention/ffn 计算耗时。

该接口读取 stage root 写出的 mmap snapshot 文件（默认在 `/tmp` 下），并返回 JSON。

请求（查询指定 layer）：

```json
{"op":"layer_prof","layerIndex":0,"stageIndex":0,"rootNodeIndex":0}
```

也可指定 snapshot 文件路径：

```json
{"op":"layer_prof","path":"/tmp/dllama_layer_prof_stage0_root0.bin","layerIndex":0}
```

响应（示例，结构简化）：

```json
{
  "ok": true,
  "layer_prof": {
    "epoch": 373,
    "layerIndex": 0,
    "nodes": [
      {"ok":true,"nodeIndex":0,"attnUs":12000,"ffnUs":8000,"commUs":0},
      {"ok":true,"nodeIndex":1,"attnUs":19000,"ffnUs":13000,"commUs":0}
    ]
  }
}
```

说明：

- `epoch`：snapshot 发布序号；外部调度器通常用它检测“新数据到来”。
- `attnUs/ffnUs`：该层 attention/ffn 计算耗时（微秒）。
- `commUs`：若实现/启用通信拆分统计，则为该层通信耗时（微秒）；为 0 表示未提供或不可用。

---

## 4. 如何下发 plan 修改指令

Plan 指令写入一个共享的 PlanCommand cache。

推理侧（stage root）在 `OP_PLAN_BARRIER` 里读取该 cache，并在满足条件时：

1. 打印 `🧭 [plan][emit] ...`（只在 stage root）
2. 通过 plan pipe 广播控制包
3. 各节点在 `OP_PLAN_APPLY` 里应用并打印 `🧭 [plan][apply] ...` 或 `reject: ...`

### 4.1 `op=set_plan`（单次迁移：legacy single-edge）

用途：把 `fromNodeIndex -> toNodeIndex` 的迁移请求写入 cache。

请求：

```json
{
  "op":"set_plan",
  "cmd": {
    "seq": 69,
    "mode": "next_barrier",
    "stageIndex": 0,
    "cmdKind": 1,
    "fromNodeIndex": 1,
    "toNodeIndex": 0,
    "nHeadsToMove": 1,
    "nFfnToMove": 256
  }
}
```

字段：

- `seq`: 指令序号（递增即可）。推理侧用它做 one-shot 防重复。
- `mode`:
  - `next_barrier`: 下一次遇到 barrier 就触发（推荐做在线调度）
  - `exact`: 只有在指定 layer/pos 触发（需要额外 `triggerLayer/triggerPos`）
- `stageIndex`: 目标 stage；或用 `0xFFFFFFFF` 表示“任意 stage”
- `cmdKind`:
  - `1`：迁移 head slice
  - `2`：迁移 ffn slice
  - `3`：同时迁移 head+ffn
- `nHeadsToMove` / `nFfnToMove`: 迁移数量（单位为 head 数 / ffn hidden slice 单元）

响应：

```json
{"ok":true,"cacheSeq":138,"cmd":{...}}
```

### 4.2 `op=set_plan`（多段迁移：v2 moves[]）

用途：一次下发多条迁移（例如 stage 内多跳平衡）。

请求：

```json
{
  "op":"set_plan",
  "cmd": {
    "seq": 70,
    "mode": "next_barrier",
    "stageIndex": 0,
    "moves": [
      {"fromNodeIndex":1,"toNodeIndex":0,"cmdKind":3,"headMove":1,"ffnMove":256},
      {"fromNodeIndex":2,"toNodeIndex":1,"cmdKind":3,"headMove":1,"ffnMove":256}
    ]
  }
}
```

说明：

- `moves[]` 模式要求推理侧支持 v2 move list apply。
- 迁移约束：通常要求 stage 内 **相邻节点** 迁移；否则 `OP_PLAN_APPLY` 会打印 `reject: ...`。

### 4.3 `op=clear`

清空 cache（回到无指令状态）：

```json
{"op":"clear"}
```

---

## 5. 使用 `examples/plan-uds-client.py`

### 5.1 启动推理（示例）

- 在 root 进程环境中设置：

```bash
export DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock
```

- 启动 inference 时加入：

```bash
./dllama inference ... --enable-plan-barrier ...
```

### 5.2 查询状态/耗时

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock status
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock perf
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock layer_prof --layer 0 --stage 0 --root 0
```

### 5.3 watch + 自动触发 set_plan（差值阈值）

推荐使用差值阈值（慢-快差距）避免整体负载漂移影响：

```bash
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock watch_layer_prof \
  --path /tmp/dllama_layer_prof_stage0_root0.bin \
  --layer 0 \
  --delta-threshold-us 8000 \
  --interval-ms 200 \
  --stage 0 \
  --kind 1 \
  --heads 1
```

参数：

- `--interval-ms`: 轮询间隔（毫秒）。越小响应越快，但开销越高。
- `--delta-threshold-us`: 触发条件为 `delta=(slow_score-fast_score) >= threshold`。
- `--threshold-us`: 旧的绝对阈值（如果不提供 delta，则用它）。

---

## 6. 常见问题（FAQ）

### Q1: `set_plan` 返回 ok，但推理侧没有任何 `🧭 [plan]` 日志

检查：

1. `status.enablePlanBarrier` 是否为 `true`
2. inference 启动命令是否包含 `--enable-plan-barrier`
3. 是否连接了 workers 并进入实际推理循环（需要 forward 跑起来才会触发 barrier/apply）

### Q2: 有 `🧭 [plan][emit]`，但出现 `🧭 [plan][apply] reject: ...`

这通常是约束不满足（例如 from/to 不相邻、节点不在同一 stage、KV 安全限制等）。
按 reject 文本修正 `fromNodeIndex/toNodeIndex` 或降低移动量。

