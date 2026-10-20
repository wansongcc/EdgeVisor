# B 任务级结果：确定性中间故障下 fixed 0/3 vs shadow 3/3（2026-10-20）

## 1. 结论

**首次得到任务级的 fixed vs shadow 对比：同一中间 stage 故障下，固定臂 3/3 失败（503），shadow 臂 3/3 成功（答案 126），且 shadow 的完成时间接近无故障基线。**

- 任务：calculator `(17+25)*3`，答案 126。
- 拓扑：3B 3 机（nx1 12 层 / nx2 12 层 / nano1 4 层，context 1024）。
- 故障：worker node1 在 decode **pos=150** 确定性退出（`EDGEVISOR_TEST_WORKER_EXIT_POS=150`）。
- 无故障 fixed：126，75.2 s。
- 故障 fixed：**0/3 成功**，503，30.1/30.3/30.9 s，failover `partial-cover`。
- 故障 shadow：**3/3 成功**，126，72.3/72.2/72.0 s，failover `ejectedStage=1 targetStage=0 layers=[12,24) myNode=0 replay=1`，无 runtime restart。

## 2. 结果表

| arm | 故障 | 成功 | 最终答案 | wall | failover |
|---|---|---:|---|---:|---|
| fixed no-fault | 无 | 1/1 | 126 | 75.2 s | — |
| fixed fault | node1 pos=150 | **0/3** | — | 30.1–30.9 s | `partial-cover` → 503 |
| shadow fault | node1 pos=150 | **3/3** | 126 | 72.0–72.3 s | bypass，`replay=1`，无 restart |

- 固定臂的 503 发生在 ~30 s：root 收到离线，没有 shadow 覆盖，直接 `PP peer offline without ready shadow cache`。
- shadow 臂的第一个 generation 在故障后继续完成，工具调用 ~18 ms，第二个 generation 输出 `final_answer 126`。
- shadow 的 wall（72 s）接近无故障 fixed（75 s），说明接管 + replay 的成本没有显著超过正常一次执行；固定臂的“快”只是快速失败。

## 3. 解释

- 这是 B 需要的“系统收益”形态：**同一故障、同一任务、同一预算下，shadow 把失败变成成功**；不是把 shadow 的 wall 和 fixed 的失败 wall 比快慢。
- `replay=1` 表示当前 token 需要重放；重放成本被计入 72 s，但仍接近无故障 75 s。
- 固定臂 0/3 是设计边界（无覆盖 → 重启/503），不是 bug。
- 该结果只覆盖一个故障位置（pos=150）、一个极短任务（calculator，工具 18 ms），不能替代完整 B 矩阵。

## 4. 限制

- 3B 模型、3 机、context 1024；8B/14B 未参与该对比。
- 任务太短：工具窗口 18 ms，不包含 A 里 0.1–12 s 的长工具窗口；不能说明长窗口下的收益。
- 只测 pos=150；pos=160/200 的 shadow readiness 与接管率未扫。
- 没有无故障 shadow 基线，shadow 的正常路径开销未单独测（之前 12/12/4 的 `bubble` 39–51 ms/node 是间接证据）。
- 只有 fixed vs full，没有 Boundary-Defer/KV-Transfer/Replay/pre-copy 等 arms；3 reps 样本小。
- 8B 3 机 shadow 在 nano 4/2 层均 OOM，8B 任务级对比仍不可行。

## 5. 对 B 的含义

- **机制 + 任务级收益都已成立**（至少在这个 smoke 范围）：shadow 接管能避免中间 worker 故障导致的任务失败。
- 下一步要把“短 calculator”换成长工具窗口的真实任务（coding/search），并加无故障 shadow 基线，才能把收益和成本同时量化。
- 3B 的 finalize 问题在 calculator 上不存在；search/coding 仍会重复工具，需要更强的提示或换 8B。
- 8B shadow 需要解决 nano 内存（1 层 + context 512，或增加第三个 8B 设备），否则 8B 只能做 fixed 基线。

## 6. 证据

- no-fault fixed：`work/opportunity_20261019/runs/b_calc_fixed`；log `nx1:/home/jetson/edgevisor_runs/b_calc_fixed.log`。
- fault fixed：`b_calc_fixed2`、`b_calc_fixed_r2`、`b_calc_fixed_r3`；logs `b_calc_fixed2.log`、`b_calc_fixed_r{2,3}.log`。
- fault shadow：`b_calc_shadow`、`b_calc_shadow_r2`、`b_calc_shadow_r3`；logs `b_calc_shadow*.log`。
- 机制日志：`⚡ [failover] fast-path deadNode=1 ejectedStage=1 targetStage=0 layers=[12,24) myNode=0 replay=1`。
- worker 日志含 `🧪 [test] worker node=1 exit at pos=150`。
- Harness：`scripts/experiments/opportunity_study.py` 新增 `calculator` 工具与 `calc_001` 任务。
