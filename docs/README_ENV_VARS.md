# 环境变量开关指南（distributed-llama）

本文汇总了仓库代码中**通过读取环境变量来切换行为**的所有开关，并给出默认值与用法示例。

> 约定：
> - **“存在即启用”**：代码只判断 `getenv(VAR) != nullptr`，即使你设置成空串或 `0` 也算“启用”。这类开关要禁用必须 `unset VAR`。
> - **“解析数值”**：代码用 `atoi/strtol/strtoul` 解析，通常 `0` 代表关闭，非 0 代表开启。
> - 下面标注了每个变量属于哪种规则。

---

## 1) 运行时开关（无需重新编译）

### 1.1 Buffer 初始化与 NaN/未写检测

- `DLLAMA_ZERO_INIT_BUFFERS`（存在即启用）
  - 默认：关闭
  - 作用：CPU 侧分配每个 buffer 后先 `memset(..., 0)`；便于排查未初始化值。
  - 示例：`DLLAMA_ZERO_INIT_BUFFERS=1 ./dllama chat ...`

- `DLLAMA_POISON_INIT_BUFFERS`（字符串模式）
  - 默认：关闭
  - 作用：CPU 侧分配 buffer 后用特定“毒值”填充，帮助定位 read-before-write / partial-write。
  - 支持值：
    - `cd` / `0xcd` / `byte`：填充字节 `0xCD`
    - `nan` / `NaN`：若 buffer 为 `f32`，填充 `qNaN`；否则回退为 `0xCD`
  - 示例：`DLLAMA_POISON_INIT_BUFFERS=nan ./dllama inference ...`

- `DLLAMA_POISON_CHECK_NAN`（存在即启用）
  - 默认：关闭
  - 作用：在每个 op 执行前抽样扫描关键输入（以及一些 op 的“隐式二次读取”buffer/pipe），若检测到 NaN 则打印最近写入者并输出上下文。
  - 建议组合：`DLLAMA_POISON_INIT_BUFFERS=nan` 一起用效果最好。

- `DLLAMA_POISON_CHECK_WRITE_NAN`（存在即启用）
  - 默认：关闭
  - 作用：在每个 op 执行后抽样扫描 primary output（仅扫描本次运行的 active batch 范围），若写出 NaN 则打印该 op 的输出指针配置。

- `DLLAMA_POISON_CHECK_LIMIT`（无符号整数）
  - 默认：`200`
  - 作用：限制 poison 相关日志总条数；`0` 表示不限制。

- `DLLAMA_POISON_CHECK_SAMPLES`（无符号整数）
  - 默认：`64`
  - 作用：每次抽样检查的采样点数量（越大越慢，但更容易命中 NaN）。

- `DLLAMA_POISON_CHECK_FILTER`（字符串）
  - 默认：不过滤
  - 作用：仅对 opName 包含该子串的 op 做 poison 检查。

- `DLLAMA_POISON_CHECK_PTRS`（无符号整数）
  - 默认：`1`
  - 作用：对 primary input 检查前 N 个 pointer（有些 op 的 input 是 pointer 数组）。

示例（定位 NaN 的首次读/写）：

```bash
export DLLAMA_POISON_INIT_BUFFERS=nan
export DLLAMA_POISON_CHECK_NAN=1
export DLLAMA_POISON_CHECK_WRITE_NAN=1
export DLLAMA_POISON_CHECK_FILTER=matmul
export DLLAMA_POISON_CHECK_LIMIT=50
./dllama inference ...
```

### 1.2 Embedding token 合法性检查

- `DLLAMA_CHECK_EMBEDDING_TOKENS`（存在即启用，且进程内缓存一次）
  - 默认：关闭
  - 作用：embedding op 会检查 token 是否有限值且在 `[0, vocabSize)`，否则抛异常终止。
  - 注意：代码只判断“是否存在”，即使设置为 `0` 也会启用；要关闭请 `unset DLLAMA_CHECK_EMBEDDING_TOKENS`。

### 1.3 网络/执行路径行为开关

- `--enable-stage-full-weights`（命令行参数）
  - 默认：关闭
  - 作用：在 uneven 分片构建中启用“stage full residency”模式：
    - stage 内权重按 full 方式加载
    - attention/ffn/logits activation buffer 按 full 分配
    - op 通过 `PNTR_BATCHED_SLICE`（必要时显式 start/stride）在 full buffer 上运行 slice
  - 备注：该开关由 root 通过 bootstrap 同步到 workers。

- `DLLAMA_FORCE_MATMUL_VIEWS`（存在即启用）
  - 默认：关闭
  - 作用：强制所有 `OP_MATMUL` 走“view-aware（offset=0）”代码路径，用于分布式 A/B 对照。
  - 影响：可能禁用部分 fast path（例如 llamafile sgemm）。

- `DLLAMA_KV_AGGREGATE`（存在即启用）
  - 默认：关闭
  - 作用：构建 net 时额外创建 KV 聚合管线（`KC`/`VC` pipes），用于 KV 聚合相关实验/验证。

### 1.3.1 分布式分层性能（layer-prof）导出

> 说明：该能力仅在进程启用了 layer profiling 时才会生效（当前通常跟 `--benchmark` 一起开启）。

- `DLLAMA_LAYER_PROF_PRINT`（解析数值）
  - 默认：`0`（关闭）
  - 作用：在 **stage root** 上把每层汇总到的各节点耗时打印到 stdout。
  - 示例：`DLLAMA_LAYER_PROF_PRINT=1 ./dllama inference ...`

- `DLLAMA_LAYER_PROF_PATH`（字符串：文件路径）
  - 默认：未设置时使用 `/tmp/dllama_layer_prof_stage<stage>_root<root>.bin`
  - 作用：在 **stage root** 上创建/更新一个可被其它进程读取的 snapshot 文件（mmap + msync）。
  - 数据布局（v1）：`Header(DLPS)` + `NnLayerPerfMsg[nLayers][nStageNodes]`。
  - 示例：`DLLAMA_LAYER_PROF_PATH=/tmp/layerprof.bin ./dllama inference ...`

### 1.4 在线迁移 / 计划屏障（测试钩子）

完整使用与验证流程见：[docs/HOW_TO_ONLINE_MIGRATION.md](HOW_TO_ONLINE_MIGRATION.md)

- `--enable-plan-barrier`（命令行参数）
  - 默认：关闭
  - 作用：启用“每层一个 plan barrier + plan apply”的 CPU-only 测试钩子，用于 online repartition 实验。
  - 备注：该开关会由 Root 通过 bootstrap 同步到 Workers。

- `DLLAMA_PLAN_CTRL_SOCKET`（字符串：UDS 路径）
  - 默认：未设置（不启动控制器）
  - 作用：在 root 进程内启动 UDS 控制器线程，供外部监控/规划进程下发 `PlanCommand`（推荐方式）。
  - 示例：
    - `export DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock`
    - `./dllama inference --enable-plan-barrier ...`

UDS 协议为“单行 JSON 请求 → 单行 JSON 响应”（每次连接只处理 1 条请求）。支持的 `op`：`ping` / `status` / `perf` / `set_plan` / `clear`。

示例（使用仓库自带客户端 [examples/plan-uds-client.py](../examples/plan-uds-client.py)）：

```bash
# 1) 启动 root 推理时开启：
export DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock
./dllama inference --enable-plan-barrier ...

# 2) 基本连通性
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock ping
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock status

# 3) 下发一次性“下一次 barrier 触发”的迁移命令（推荐真实运行方式）
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock set_plan \
  --seq 1 --mode next_barrier --stage 0 --from 0 --to 1 --kind 3 --heads 1 --ffn 256

# 4) 调试用：精确触发（到达指定 position + layer 才触发）
python3 examples/plan-uds-client.py /tmp/dllama_plan.sock set_plan \
  --seq 2 --mode exact --stage 0 --from 0 --to 1 --kind 3 --heads 1 --ffn 256 \
  --trigger-pos 64 --trigger-layer 10
```

- `DLLAMA_HARD_MIGRATE_POS`（整数）
- `DLLAMA_HARD_MIGRATE_LAYER`（整数）
- `DLLAMA_HARD_MIGRATE_KIND`（整数；`1=headSplit` `2=ffnSplit` `3=both`）
- `DLLAMA_HARD_MIGRATE_HEAD_MOVE`（整数）
- `DLLAMA_HARD_MIGRATE_FFN_MOVE`（整数）
  - 状态：**Deprecated（兼容兜底）**
  - 作用：不再直接驱动 barrier 的“静态触发器”。当检测到这些变量时，root 会在启动时**自动生成一条 `PlanCommand(mode=exact)`** 并打印 deprecate warning；外部 UDS 下发命令优先。

- `--enable-kv-redundancy-during-migration`（命令行参数）
  - 默认：开启（`1`）
  - 作用：当启用 plan barrier/migration 时，控制 migration 期间是否保留 KV redundancy（用于“无额外通信”的实验）。
  - 用法：
    - 开启：`--enable-kv-redundancy-during-migration` 或 `--enable-kv-redundancy-during-migration 1`
    - 关闭：`--enable-kv-redundancy-during-migration 0`
  - 备注：该开关由 root 通过 bootstrap 同步到 workers。

- `DLLAMA_MIGRATION_LAYER_COUNT`（整数，推荐）
  - 默认：`1`
  - 作用：控制 runtime migration/redundant 的 boundary layer 数量（等价于 CLI 的 `--runtime-redundant-boundary-layers`）。
  - 兼容：若未设置该变量，会回退读取 `DLLAMA_RUNTIME_REDUNDANT_BOUNDARY_LAYERS`。
  - 优先级：若同时传了 CLI `--runtime-redundant-boundary-layers`，以 CLI 为准。
  - 示例：
    - `export DLLAMA_MIGRATION_LAYER_COUNT=2`
    - `./dllama inference ...`

- `DLLAMA_MIGRATION_LAYER_LIST`（逗号分隔整数列表）
  - 默认：未设置
  - 作用：指定在线迁移要批量处理的 layer 列表（transfer/ack/switch 按该列表批量执行）。
  - 优先级：高于 `DLLAMA_MIGRATION_LAYER_COUNT`（设置列表后，count 只影响冗余构图跨度，不决定迁移目标列表）。
  - 示例：
    - `export DLLAMA_MIGRATION_LAYER_LIST=13,12,11,10`
    - `export DLLAMA_ASYNC_KV_COLLECT_POS=31`
    - `./dllama inference ...`

- `DLLAMA_MIGRATION_DIRECTION`（字符串）
  - 默认：`next`
  - 作用：控制 layer 迁移方向（目标 stage）。
    - `next` / `forward` / `right`：迁移到下一 stage（默认）
    - `prev` / `backward` / `left`：迁移到上一 stage
  - 与 `DLLAMA_MIGRATION_LAYER_COUNT` 组合时：
    - `next`：默认从右边界层向左扩展 `count` 层
    - `prev`：默认从左边界层向右扩展 `count` 层
  - 示例：
    - `export DLLAMA_MIGRATION_DIRECTION=prev`
    - `export DLLAMA_MIGRATION_LAYER_COUNT=3`

### 1.6 分布式：Root 同步环境变量到 Workers

- `DLLAMA_SYNC_ENV_VARS`（逗号分隔列表）
  - 默认：空
  - 作用：Root 会把一组“默认名单 + 你额外指定的名单”里的环境变量（存在与否以及具体值）同步到所有 worker 进程。
  - 用法：
    - 例如你想把 `kvcache_debug`、`kvcache_debug_limit` 也一起同步：
      - `export DLLAMA_SYNC_ENV_VARS="kvcache_debug,kvcache_debug_limit"`
  - 备注：默认同步名单在代码里硬编码（见 nn-network.cpp），包括 `DLLAMA_FORCE_MATMUL_VIEWS`、`DLLAMA_KV_AGGREGATE` 等。

### 1.7 E2E Matmul-View（offset=0）对照检查

- `DLLAMA_E2E_MATMUL_VIEW0_CHECK`（严格布尔：仅当值为 `1/true/TRUE` 才启用）
  - 默认：关闭
  - 作用：在 inference mode 下跑两次 prompt eval：
    - Pass A：清空所有 matmul 的 a/b/c view（legacy）
    - Pass B：保留 view（offset=0 plumbing）
    - 对每次 forward 的 logits 做对照，帮助验证 view-aware 代码路径的等价性
  - 限制：当前要求单机（`--n-workers 0`），否则会直接报错。

---

## 2) 调试/Tracing 开关（需要编译时启用 `DLLAMA_DEBUG_ATTN=1`）

很多注意力/KV/同步相关的 tracing 代码被 `#if DLLAMA_DEBUG_ATTN` 包住；要使用下面这些环境变量，请先重新编译：

```bash
DLLAMA_DEBUG_ATTN=1 make dllama
```

### 2.1 Slice / Pointer / Weight 范围调试

- `DLLAMA_DEBUG_SLICE_PARAMS`（存在即启用）
- `DLLAMA_DEBUG_SLICE_PARAMS_FILTER`（子串过滤）
  - 作用：打印 build-time 的 slice/view 参数（更偏配置验证）。

- `DLLAMA_DEBUG_SLICE_PARAMS_FWD`（存在即启用）
- `DLLAMA_DEBUG_SLICE_PARAMS_FWD_REPEAT`（把 `0`/`false` 视为关闭；其他值视为开启）
- `DLLAMA_DEBUG_SLICE_PARAMS_FILTER`（同上）
  - 作用：forward-time 打印 slice 参数与 pointer 样本，用于验证运行时 resolve 是否符合预期。

- `DLLAMA_DEBUG_RESOLVE_POINTER`（把 `0`/`false` 视为关闭；其他值视为开启）
- `DLLAMA_DEBUG_RESOLVE_POINTER_FILTER`（既支持数字 buffer/pipe index，也支持如 `HEAD`/`KV` 等 tag 子串）
- `DLLAMA_DEBUG_RESOLVE_POINTER_LIMIT`（整数，默认 `200`，上限 `100000`；`<=0` 视为 0）
  - 作用：输出 resolvePointer 相关调试信息。

- `DLLAMA_DEBUG_WEIGHT_RANGES`（存在即启用）
- `DLLAMA_DEBUG_WEIGHT_RANGES_FILTER`（子串过滤）
- `DLLAMA_DEBUG_WEIGHT_RANGES_LIMIT`（整数；用于限制 matmul weight-read 范围日志次数）
  - 作用：打印“可能读到的权重范围”与（matmul）更精确的按调用读范围。

- `DLLAMA_DEBUG_MATMUL_VIEWS`（存在即启用）
- `DLLAMA_DEBUG_MATMUL_VIEWS_FILTER`（子串过滤）
- `DLLAMA_DEBUG_MATMUL_VIEWS_LIMIT`（整数，默认 `50`）
  - 作用：当 matmul 走 llamafile fast path 时打印 view 配置，确认是否命中预期路径。

### 2.2 KV range tracing（粗粒度）

- `kvcache_debug`（存在即启用）
- `DLLAMA_DEBUG_KV_RANGE`（存在即启用；legacy alias）
- `kvcache_debug_limit`（整数，默认 `8`；`0` 不限制）
  - 作用：打印各类 KV 相关 op（K/V projection、RoPE、KV cache write、attention read）的有效 KV range。

### 2.3 KV cache per-head 细粒度 dump

- `DLLAMA_DEBUG_KVCACHE_PER_HEAD`（数值解析；非 0 启用）
- `DLLAMA_DEBUG_KVCACHE_PER_HEAD_FILTER`（子串过滤）
- `DLLAMA_DEBUG_KVCACHE_PER_HEAD_LIMIT`（整数，默认 `32`；`0` 不限制）
- `DLLAMA_DEBUG_KVCACHE_PER_HEAD_BATCH`（整数，默认 `0`）
- `DLLAMA_DEBUG_KVCACHE_PER_HEAD_POS`（整数，默认 `-1`：所有位置；`>=0`：仅该 pos）
- `DLLAMA_DEBUG_KVCACHE_PER_HEAD_DIMS`（整数，默认 `8`；`0` 打印 full headDim）
- `DLLAMA_DEBUG_KVCACHE_PER_HEAD_KVHEAD`（整数，默认 `-1`：所有 owned kvHeads；`>=0`：仅该 global kvHead）
- `DLLAMA_DEBUG_KVCACHE_PER_HEAD_KVHEADS`（字符串，例：`"4,5"` 或 `"4-7"`，优先级高于 KVHEAD）
- `DLLAMA_DEBUG_KVCACHE_PER_HEAD_GLOBAL`（数值解析；非 0 启用）
- `DLLAMA_DEBUG_KVCACHE_PER_HEAD_SHIFT`（数值解析；非 0 启用）
- `DLLAMA_DEBUG_KVCACHE_PER_HEAD_HEADDIM`（整数；覆盖 headDim）

### 2.4 Attention 计算调试

- `DLLAMA_DEBUG_ATT`（数值解析；非 0 启用）
- `DLLAMA_DEBUG_ATT_FILTER`（子串过滤）
- `DLLAMA_DEBUG_ATT_LIMIT`（整数；`0` 不限制）
- `DLLAMA_DEBUG_ATT_BATCH`（整数，默认 `0`）
- `DLLAMA_DEBUG_ATT_HEAD`（整数，默认 `0`；`-1` 打印所有 local q-head）

- `DLLAMA_DEBUG_ATT_QK`（数值解析；非 0 启用）
- `DLLAMA_DEBUG_ATT_QK_DIMS`（整数，默认 `16`；`0` full headDim）
- `DLLAMA_DEBUG_ATT_QK_TOPK`（整数，默认 `5`）
- `DLLAMA_DEBUG_ATT_QK_POS`（整数，默认 `-1`：所有 pos；`>=0`：仅该 pos）

- `DLLAMA_DEBUG_ATT_SCORES`（数值解析；非 0 启用）
- `DLLAMA_DEBUG_ATT_SCORES_MAXLEN`（整数，默认 `64`；`0` full length（有 cap））

### 2.5 Matmul / RoPE I/O 摘要日志

- `DLLAMA_DEBUG_MATMUL_IO`（存在即启用）
- `DLLAMA_DEBUG_MATMUL_IO_FILTER`（子串过滤）
- `DLLAMA_DEBUG_MATMUL_IO_LIMIT`（整数，默认 `20`；`0` 不限制）
- `DLLAMA_DEBUG_MATMUL_IO_OUT_OFFSET`（整数，默认 `0`）
- `DLLAMA_DEBUG_MATMUL_IO_OUT_LEN`（整数，默认 `64`；`0` 使用内部上限）

- `DLLAMA_DEBUG_ROPE_IO`（存在即启用）
- `DLLAMA_DEBUG_ROPE_IO_FILTER`（子串过滤）
- `DLLAMA_DEBUG_ROPE_IO_LIMIT`（整数，默认 `50`；`0` 不限制）
- `DLLAMA_DEBUG_ROPE_IO_OFFSET`（整数，默认 `0`）
- `DLLAMA_DEBUG_ROPE_IO_LEN`（整数，默认 `256`；`0` 使用内部上限）

### 2.6 同步与 KV 聚合验证

- `DLLAMA_KV_AGGREGATE_PROOF`（存在即启用）
- `DLLAMA_KV_AGGREGATE_PROOF_ALL_BATCHES`（存在即启用）
- `DLLAMA_SYNC_TRACE`（存在即启用）
- `DLLAMA_SYNC_PROFILE`（存在即启用；会启用 spin-wait barrier 做全线程 wall-time 计时）

---

## 3) 需要重新编译的“编译期开关”（make 变量/宏）

这些不是运行时 `getenv()` 读取，而是**编译期宏**，常见用法是通过 `make` 变量传入：

- `DLLAMA_DEBUG_ATTN`（默认 `0`）
  - 用法：`DLLAMA_DEBUG_ATTN=1 make dllama`
  - 作用：编译进大量 attention/KV/sync 的 tracing/调试分支（上面第 2 节的环境变量多数依赖它）。

- `DLLAMA_DEBUG_ONLINE_CHANGE`（默认 `0`）
  - 用法：`DLLAMA_DEBUG_ONLINE_CHANGE=1 make dllama`
  - 作用：输出 online plan apply / migration 的更详细日志。

- `DLLAMA_CONTROL_LOG`（默认 `0`）
  - 用法：`DLLAMA_CONTROL_LOG=1 make dllama`
  - 作用：打印 root 侧 control packet 发送日志。

- `DEBUG_OP_INPUT_OUTPUT`（默认 `0`）
  - 用法：`DEBUG_OP_INPUT_OUTPUT=1 make dllama`
  - 作用：编译进 op 输入输出调试打印（用于算子级排查）。

- `TOPK=1`（默认关闭）
  - 用法：`TOPK=1 make dllama`
  - 作用：编译进 TopK logits 调试能力；运行时再配合：
    - `DLLAMA_DEBUG_LOGITS_BATCH`（整数；默认 `0`，`-1` 表示所有 batch）

- `DLLAMA_VULKAN`（通过 `make` 环境变量存在与否判断）
  - 用法：`DLLAMA_VULKAN=1 make dllama` 或 `DLLAMA_VULKAN=1 make dllama-api`
  - 作用：启用 Vulkan 相关代码与 shader 编译。

---

## 4) 脚本工具相关环境变量

- `DLLAMA_SHARDING_UPDATE_FILE`（文件路径）
  - 使用处：Python 工具 [src/sharding_scheduler.py](../src/sharding_scheduler.py) 与示例脚本 [examples/sharding-update.sh](../examples/sharding-update.sh)
  - 作用：指定“sharding 请求文件”的路径（默认 `/tmp/dllama_sharding_req.txt`）。
  - 示例：

```bash
export DLLAMA_SHARDING_UPDATE_FILE=/tmp/dllama_sharding_req.txt
python3 src/sharding_scheduler.py --plan 'layer=21 pos=4'
```
