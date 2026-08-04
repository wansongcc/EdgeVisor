# Runtime-layout-aware Stage Bypass 修复设计

## 目标

修复 PP 迁移后 Stage Bypass 仍使用启动时静态 stage layer 范围的问题。
当 bypass 命令到达 Root 时，ejected stage 的 required layers 必须来自已提交的
runtime primary-owner snapshot。支持当前 primary layer 为非空连续区间的单 stage、
单 target、active-chain 相邻 bypass，并保持现有 generation、ACK 与 TopologyFence
一致性闭环。

典型流程为：

```text
启动布局 18/16/5/1
    └─ PP1: Stage2 -> Stage1, [34,35,36,37]
       runtime primary: S0=[0,18), S1=[18,38), S2=[38,39), S3=[39,40)
    └─ bypass: Stage2 -> Stage3
       required/applied: [38]
       verified runtime layout: S0=[0,18), S1=[18,38), S2=[], S3=[38,40)
       active chain: [0,1,3]
```

## 范围与约束

包含：

- PP1 完成后按 runtime owner 计算 bypass layer 范围；
- 非空、严格连续、越界检查；
- target 对 required layers 的 runtime redundant 校验；
- 当前 active-chain 相邻 stage 的单 stage bypass；
- PP pending/未验证时拒绝；
- root apply generation、三类 participant ACK、verifiedGeneration 与
  TopologyFence 的现有闭环；
- bypass 后的 PP2 候选生成与 `15/15/5/5` 回归。

不包含：

- 非连续 layer list bypass；
- 多 stage 同时 bypass；
- bypass 与未验证 PP migration 并发提交；
- 自动选择 bypass stage；
- 修改静态启动布局定义或降低 ACK 校验强度。

## 数据模型

现有启动时 `RuntimeStageLayerPlan` 继续作为静态 provision map，供启动图、
普通 PP migration 的 provision/rollback 检查以及 segment 分类使用。它不再作为
bypass 的 owner 或覆盖范围来源。

Root 新增一个可提交的 runtime layout snapshot。snapshot 初始复制静态 provision
角色，随后只在已完成的 PP ownership switch 或已验证的 bypass 后更新：

- 正常 PP migration 完成后，将 source 的当前 primary layer 置为 disabled，
  将 target 的对应 provisioned layer 置为 primary；
- bypass ACK 全部通过后，将 ejected required layers 置为 disabled，target 的
  redundant layers 置为 primary；
- 每次提交前验证全模型层数范围内每一层恰有一个 primary owner；
- static provision map 与 mutable runtime snapshot 分离，避免为了反映当前 owner
  而破坏普通 PP 回滚所需的静态冗余信息。

新增的纯 helper 负责以下职责，便于单元测试并保证拒绝路径原子性：

1. 校验 runtime snapshot 的 layer 数、角色矩阵边界、每层 primary owner 唯一；
2. 提取 `currentOwnedLayers(stage)`，要求结果非空、升序连续且完全在模型范围内；
3. 校验 ejected/target 存在于当前 active chain，且 target 是 ejected 的相邻 stage；
4. 校验 target 对每个 required layer 的角色为 `RUNTIME_LAYER_REDUNDANT`；
5. 在临时副本上应用 ownership transition，验证通过后再交换 snapshot。

建议的接口语义为：

```cpp
bool validateRuntimeStageLayerPlan(
    const RuntimeStageLayerPlan &layout, std::string *reason);

bool currentOwnedLayers(
    const RuntimeStageLayerPlan &layout,
    NnUint stageIndex,
    std::vector<NnUint> &layers,
    std::string *reason);

bool resolveStageBypassLayers(
    const NnUnevenPartitionPlan *plan,
    const RuntimeStageLayerPlan *runtimeLayout,
    NnUint ejectedStageIndex,
    NnUint targetStageIndex,
    std::vector<NnUint> &layers,
    std::string *reason);

bool applyRuntimeLayerOwnershipMove(
    RuntimeStageLayerPlan &layout,
    NnUint sourceStageIndex,
    NnUint targetStageIndex,
    const std::vector<NnUint> &layers,
    bool requireRedundantTarget,
    std::string *reason);
```

## Bypass admission 与提交流程

Root 处理 `set_stage_bypass` 时按以下顺序执行：

1. 若存在 pending KV/state transfer、等待 KV ACK、待发送 layer switch 或其他
   未完成 PP 状态，拒绝并记录
   `bypass deferred: pending PP generation is unverified`；
2. 若已有 root-applied bypass 且 `verifiedGeneration < appliedGeneration`，拒绝并
   保持当前布局、路由和 executor 不变；
3. 对当前 runtime snapshot 做完整性校验，并调用
   `resolveStageBypassLayers()`；空 owner、非连续 owner、越界、unknown stage、
   非 active-chain 相邻或冗余缺失均拒绝，使用明确原因，例如：
   `current ejected-stage layers are empty`、
   `current ejected-stage layers are non-contiguous`、
   `target lacks redundant runtime layer 38`；
4. 仅把返回的 required layers 保存为本次 bypass 的
   `runtimeRequiredLayers`，并作为唯一 layer-switch/KV-switch batch；不从
   `NnStageConfig::startLayer/endLayer` 重建范围；
5. root 分配 generation、发送 batch 并执行现有 root route apply。root apply 成功
   后写入 `rootApplyGeneration`/`appliedGeneration` 与 applied layer snapshot，
   但在 ACK 验证前不提交 runtime owner snapshot 的 bypass 变化；
6. worker 完成 layer switch、PP route 与 sync role 后发送 ACK；root 验证全部
   participant 后才提交 runtime snapshot，并令
   `verifiedGeneration = appliedGeneration`。

拒绝或任何 admission failure 都不得消费为成功 apply：不得修改 `plan` 的 route、
runtime snapshot、executor layer gate 或 pending switch batch。失败必须出现在 status
和日志中，而不是仅输出 `invalid stage` 或静默消费命令。

## ACK 校验

继续使用现有 dedicated stage-bypass ACK frame，不改变 layer-switch 固定 wire ABI。
每个 ACK 必须携带并由 root 校验：

- bypass generation；
- reporting node/stage、ejected stage、target stage；
- role flags；
- 当前 bypass 实际范围 `startLayer/endLayer/layerCount`；
- 完整 active stage chain。

本次只支持连续范围，因此 ACK 仍使用 `startLayer/endLayer/layerCount` 表达范围。
三个受影响 participant 为 previous stage、ejected stage、target stage。三者的
range/count 都必须与本次 `runtimeRequiredLayers` 的连续区间一致，target ACK 特别
必须精确为 `[38,39)`，不能回退到初始 `[34,39)`。generation、node/stage、role、
range/count、active chain 任一不一致都不推进 verifiedGeneration；重复、过期或
未知节点 ACK 也保持未验证状态并设置 failureReason。

## Status、日志与 scheduler

`status.stageBypass` 至少暴露：

```json
{
  "runtimeRequiredLayers": [38],
  "appliedLayers": [38],
  "layerCount": 1,
  "ejectedStage": 2,
  "targetStage": 3,
  "activeStageChain": [0, 1, 3],
  "rootApplyGeneration": 1,
  "verifiedGeneration": 1,
  "failureReason": ""
}
```

拒绝原因必须与具体校验一致，至少覆盖 pending PP、未验证 generation、空/非连续/
越界 owner、unknown/inactive stage、非相邻 target、target 缺少 redundant runtime
layer 与 ACK mismatch。成功后 `runtimeRequiredLayers == appliedLayers`，并保持
layer count 与连续范围一致。

动态 TPOT scheduler 继续把 root apply 视为 topology fence 起点，只有全部 ACK
验证且 `verifiedGeneration >= appliedGeneration` 才解除 fence。已提交的 runtime
logical PP layout 在 PP1 后为 Stage2 `[38,39)`；bypass verified 后合并 target，
得到 `[0,1,3]` 和 Stage3 `[38,40)`，不恢复或重复迁移 `[34..37]`。随后 Stage0
降频可以生成 PP2 `0 -> 1`。

## 测试设计

测试按 TDD 先写失败用例，再实现最小 helper/状态变更：

- `18/16/5/1` + 4-layer redundancy 初始 snapshot 满足全层唯一 owner；
- PP1 `2 -> 1 [34..37]` 后 runtime owner 为 Stage2 `[38]`；
- bypass `2 -> 3` 返回并只发送 `[38]`，active chain 为 `[0,1,3]`；
- target 缺少 redundant layer 38 时拒绝且 snapshot/route/executor 状态不变；
- 当前 owner 为空、非连续或越界时拒绝且原因明确；
- pending PP 或未验证 bypass generation 时拒绝；
- previous/ejected/target ACK 的 generation、role、range/count、chain 任一错误
  都不推进 verifiedGeneration；完整 ACK 才推进；
- scheduler 在 bypass verified 后保留 PP1 的 `[34..37]` 迁移结果，并允许 PP2；
- 初始 `15/15/5/5` 的既有 bypass 行为回归通过。

实现完成后运行 app/runtime helper、dynamic TPOT、ACK/status 相关定向测试及现有
工程回归测试，并以新鲜命令输出作为完成依据。
