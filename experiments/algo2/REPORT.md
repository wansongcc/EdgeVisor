# Algorithm 2 report

This is a mechanism report for the revised online controller. It does not claim a globally optimal allocation, and it does not claim a migration net benefit.

## What the code does

`DLLAMA_ALGO2_MODE` selects the controller inside the existing dynamic-TPOT thread.

| Mode | Behavior |
| --- | --- |
| unset or `legacy` | The previous controller, including the 1.3 trend cap and the full-KV fallback |
| `static` | Same initial split and the same algorithm-1 shadow. Decisions are logged and not issued |
| `revised_single_layer` | One ready layer, and only when the whole action clears the threshold |
| `revised_variable_step` | The largest ready k whose predicted per-token gain clears the threshold. Ties prefer the smaller k |

The revised path compares per-token gain with `max(3 ms, 0.03 * (source + target))`, plus calibrated noise when that value is set. The controller passes each stage's compute time and its separate communication wait. A measured boundary layer is layer compute. Interior layers are the average of the compute time that remains after those boundaries are removed, and that average is labeled as extrapolation. A stage with no measured layer and no declared fixed overhead still does not move. Communication wait is not part of the gain. Uncalibrated noise stays out of the threshold.

A move is issued only when the recorded shadow cursor covers the current token. In revised mode the receiver must answer that transaction. Silence is not acceptance: the root waits up to 200 ms, and a missing answer stays unconfirmed. One later token may replay that same transaction id. A second miss stays unconfirmed and does not start another transaction. The legacy path still treats silence as acceptance and can still copy KV from position 0.

A revised reservation miss with a known gap of at most `DLLAMA_ALGO2_ROLLBACK_CATCHUP` tokens (default 128) may retransmit that tail, or recompute it if the retransmit does not install. That is catch-up, not a ready reservation. A larger or unknown gap aborts.

The health reference is the recovery anchor. A normal window does not replace it with a slightly slower median. Context growth can raise the comparison value only after normal windows span `DLLAMA_ALGO2_LONG_SPAN` tokens (default 128) and lie on one line. A kept move does not replace the anchor and does not mark the run recovered. The next search uses the new allocation's own window. One worse tail sub-window is not a rollback. A worse median rolls back immediately. A tail that recovers still has to beat the median improvement bar; otherwise the window stays inconclusive until the verify cap. A tail that stays worse for two non-overlapping verify windows rolls back. The switch stall stays in that token's request interval and stays out of the health and verify windows. Predicted layer gain and measured end-to-end gain are both logged.

Default decode remains one request in `inferenceRunOnce`. The token clock is that request's completion interval on the root. A token with no completion interval is not a health sample. Continuous batching is not on this path.

## Audit

| Item | Finding |
| --- | --- |
| Serial decode | `inferenceRunOnce` forwards one token, samples, then continues. The T4 runs used that path |
| Legacy stage time | The old controller sums per-stage exec. That sum is not called end-to-end time |
| Legacy trend cap | `clampTrendPenalty` still caps the old path at 1.3. The revised estimator does not use it |
| Legacy verify | The old path still uses its 1.05 window. The revised path uses its own settle, verify, and cooldown |
| Shadow cursor | `shadowFilledThrough` is the minimum exclusive end of redundant and shadow-KV layers on that node. `0xFFFFFFFF` means the node has no such layer |
| Readiness | Cursor `>= token + 1` is ready. Unknown stays unknown. Weights being present is not readiness |
| Foreground switch | Not measured below 1 ms. The old revised path treated 200 ms of silence as acceptance, and the smoke below recorded 200.615 ms for that wait. The current revised path names that wait as protocol wait and accepts only an explicit applied acknowledgement |
| Containers | Docker is installed and the nvidia runtime is present. Pulling `nvidia/cuda:12.0.0-base-ubuntu22.04` failed: the configured proxy `192.168.1.11:10810` refused the connection. The T4 run was on the host. No host `tc`, no clock change, no power change |
| Edge set | nx1, nx2, nano1, nano2, and rog answered SSH. The Q40 model is on nx1. nvcc 12.6 is on nx1. No algorithm-2 inference was run there |

## Refinement against the smoke

The smoke below was produced by the previous revised binary. It is not a re-run of the current code.

| Point | Previous run | Current code |
| --- | --- | --- |
| Health anchor | Keeping the 179 ms move cleared the 58 ms reference, and the next window stored about 180 ms as healthy | The 58 ms anchor stays. The move can be kept, health stays unrecovered, and after cooldown the search continues against the new allocation |
| Commit | 200 ms of silence was recorded as a successful switch | Revised mode requires an applied acknowledgement for that transaction. Timeout is unconfirmed. One same-id replay is allowed |
| Short KV gap | Any reservation miss aborted | A known gap of at most 128 tokens may be retransmitted or recomputed. A longer gap still aborts, with no copy from position 0 |
| Cost | The k=6 prediction used a stage average that mixed fixed and communication time into the layer cost | Boundary-layer compute is the measured cost. Interior layers are an average and are labeled as extrapolation. Communication wait is not gain |
| Tail | One verify window decided keep or rollback | A worse median rolls back. A worse tail must repeat on a second non-overlapping window |

`algo2-test`, `dynamic-tpot-test`, and `dllama` were rebuilt after this change. The tests passed. No new T4 or edge inference was run, so the acknowledgement path and the tail catch-up have not been shown on a live pipeline.

## T4 smoke

Four Tesla T4 GPUs, CUDA arch 75, driver 13020, runtime 12080. Model `llama3.2-3b-instruct` Q40, tokenizer `llama_3_1`. Split `1@7*1@7*1@7*1@7`. Algorithm-1 equal shadow under cap 6: `6,3;3,3;3,6`. Prompt `Hi`, temperature 0, seed 1, `DLLAMA_IGNORE_EOS=1`. The decision window, settle, and verify were shortened for this smoke. Those overrides are in `t4_smoke.sh`. They are not new defaults.

Raw logs:

- `logs/t4_static_root.txt`, `logs/t4_static_tpot.txt`
- `logs/t4_revised_delay_root.txt`, `logs/t4_revised_delay_tpot.txt`

### Static, 48 steps, no injection

Exit 0. 16.18 tokens/s, 61.82 ms/token. The controller logged 36 observe records, issued nothing, and rolled nothing back. The first full window set the reference at 56.4 ms. Later healthy windows replaced it, up to 58.4 ms. The last tokens were `suspect` with cause `mixed`, still inside the confirmation count, so static mode correctly kept the allocation. Every logged stage cursor was `token + 1`.

### Revised variable step, 120 steps, injected sleep

These numbers are from the previous binary. This is an injected per-step sleep, not a GPU throttle and not a natural fault. Node 2 slept 15 ms on its compute steps after 24 decode tokens.

Exit 0. The whole run, including the healthy prefix and the stall, was 4.57 tokens/s, 219.02 ms/token. That average is not the post-move speed.

The healthy reference settled at 58.006 ms. The first slow window was `suspect` at token 40, median 555.414 ms. At token 48 the window was `stable_degraded`, cause `mixed`, median 558.134 ms, and the controller issued k=6 from stage 2 to stage 3, layers 15 through 20, predicted gain 380.919 ms, threshold 14.997 ms. The root log shows the reservation served the move: `stateBytes=0`, `recomputeUnits=0`, position 51. Commit completed by token 52. Verify median was 179.274 ms and p95 was 180.161 ms, so the transaction was kept as `clear_improvement` against the degraded window. It was not marked recovered. The reference was cleared and the following healthy-looking window rebuilt it at 180.08 ms.

Stage 3's cursor stopped at 52 after the move, while the other stages kept advancing. A later move that depends on stage 3 is not ready.

## What this does and does not show

The statistics are the logs above: one serial Q40 run per mode, with per-token phase, health, cursor, and the root migration line.

The previous binary fired on the injected run. Static mode with the same initial split did not move. That transaction checked a live cursor, used the reservation, passed settle and verify, and kept the move under the old rule that then cleared the health reference. The unit tests now also cover an unseparated fixed overhead, a kept move that leaves the health anchor in place, a gradual climb that stays under 10 percent per step, context-adjusted growth, one tail spike, a recovered tail whose median is not good enough, a missing end-to-end sample, an unconfirmed commit, separate predicted and measured gain, and a short versus long rollback gap.

That older run is one injected episode. Its 200.615 ms stall is the acceptance grace of the previous binary. The paired runs below use the current build.

## Paired T4 diagnostic

Build `8a23bff`. Host GPUs 0–3, not a container. Model Llama 3.2 3B Q40, split `1@7*1@7*1@7*1@7`, prompt `Hi`, 80 steps, seed 1. Node 2 sleeps 15 ms before each executed op after 24 decode tokens. Three pairs, order alternating, logs in `logs/t4_pair_b/`. Command: `experiments/algo2/t4_pair.sh`.

The 4 ms and 379 ms figures are not one action. The 4 ms figure is the unit test whose layers cost 10 ms and 8 ms. The 379 ms figure is 558 − 179 from the older smoke, copied into that same test so the two clocks are stored separately. The older smoke itself predicted 380.919 ms.

All six runs exited 0. Static observed the slowdown and issued nothing. Revised, at token 50 on every pair, moved k=6 from stage 2 to stage 3. The candidate table also contained smaller steps on that boundary and on the boundary with stage 1. k=6 had the largest predicted gain, about 415 ms, against a threshold of about 15 ms. The cost source is `boundary_measured_interior_extrapolated`, confidence medium. The receiver answered transaction 1. The stall was 1.619 ms, 1.586 ms, and 1.589 ms. The health anchor stayed near 59 ms.

End-to-end medians, ms/token:

| Pair | Before the move, tokens 33–49, static | Before, revised | After, tokens 64–74, static | After, revised | Paired end-to-end difference | Predicted gain | Revised's own median change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 555.8 | 555.8 | 557.3 | 178.8 | 378.5 | 415.0 | 555.6 → 178.6 |
| 2 | 556.0 | 556.2 | 556.4 | 178.9 | 377.5 | 414.6 | 556.2 → 178.4 |
| 3 | 555.7 | 556.2 | 556.6 | 178.1 | 378.4 | 415.2 | 556.5 → 178.6 |

The paired difference and the controller's own before/after median agree at about 378 ms/token. The frozen cost table predicts about 415 ms/token, so it overstates the end-to-end change by about 37 ms/token. The sleep sits inside the step timer, so the layer-cost snapshot counts it as compute. Most of that timed cost did move with the six layers. The logs do not separate the remaining 37 ms into communication, a fixed part of the sleep, or the one layer left on node 2.

No other step was executed. The table shows the program selected the highest passing score. It does not show that a smaller k would have ranked the same way on the machine. Stage 3's shadow cursor stopped advancing after the move, so a later move onto that stage was not ready. One repeat was kept by `clear_improvement` and then `reverse_suppressed`.

An earlier pair on the previous build, `logs/t4_pair/`, refused every boundary with `fixed_overhead_not_separated` because the decision snapshot did not yet receive the boundary-layer timers.

## k=1 against k=6

Build `82bc5e8`, same injection and the same 80-step window as the paired diagnostic. Each revised run could issue only one move. k=1 was forced on the stage-2 to stage-3 boundary. k=6 is the step the search selects there. Three repeats, logs in `logs/t4_k_compare/`. Command: `experiments/algo2/t4_k_compare.sh`.

All nine runs exited 0. The move was issued at token 50, 51, or 52. Each keep used 8 verify samples. Medians over tokens 70–78, ms/token:

| Arm | End-to-end | Full-chain compute sum | Stage 2 compute | Stage 3 compute | Predicted gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 557.0 | 538.8 | 471.7 | 28.0 | — |
| k=1 | 496.3 | 478.4 | 408.6 | 30.7 | 28.8 |
| k=6 | 178.9 | 176.6 | 93.4 | 44.1 | 415.0 |

Against static, k=1 lowers the token interval by 60.6 ms and the compute sum by 60.4 ms. k=6 lowers the interval by 378.1 ms and the compute sum by 362.2 ms. On the machine, 6 layers is better than 1 layer by about 317 ms/token. The predicted order is the same, and the gap is large.

For k=6, stage 2's compute falls by 378.3 ms and stage 3's rises by 16.1 ms. The token interval falls by 378.1 ms, the same amount as stage 2. The 16 ms added on the receiver does not stack onto the interval. The cost table's 415 ms is 37 ms above that stage-2 drop. That is the whole of the earlier 36.8 ms gap: the table overstates how much timed compute leaves the slow stage. It is not a separate communication term in the token interval.

For k=1, stage 2 falls by 63.0 ms and the interval falls by 60.6 ms. The boundary-layer estimate only predicted 28.8 ms, so one measured boundary layer does not capture the whole stage-time change of moving that layer.

## Not run

- The 512-token, four-strategy matrix
- A forced alternative step at the same position
- A natural GPU fault, clock cap, or power cap
- Edge-set Q40 inference
- Noise calibration. The threshold noise term stayed uncalibrated
- Rollback on a real reservation miss. The unit test covers a failed rollback. This T4 move hit the reservation

## Reproduce

```bash
cd /home/byh/B01/EdgeVisor-wansongcc/EdgeVisor
make algo2-test dynamic-tpot-test dllama
./algo2-test
./dynamic-tpot-test
bash ../experiments/algo2/t4_smoke.sh
OUT=../experiments/algo2/logs/t4_pair_b bash ../experiments/algo2/t4_pair.sh
bash ../experiments/algo2/t4_k_compare.sh
```

`algo2-test` and `dynamic-tpot-test` passed after the controller was linked. `dllama` was built with `DLLAMA_CUDA` and arch 75.
