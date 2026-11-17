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

The revised path compares per-token gain with `max(3 ms, 0.03 * (source + target))`, plus calibrated noise when that value is set. A per-layer vector is a high-confidence cost. A declared fixed overhead of 0 or more is a stage average with the fixed part and any included communication wait removed; that average is monotonic in k and is medium confidence. An unknown fixed overhead is not a layer cost, so the search does not issue. The live controller still does not pass a per-layer vector or a declared fixed overhead, so a revised run will not move until one of those is supplied. Uncalibrated noise stays out of the threshold.

A move is issued only when the recorded shadow cursor covers the current token. In revised mode the receiver must answer that transaction. Silence is not acceptance: the root waits up to 200 ms, and a missing answer stays unconfirmed. One later token may replay that same transaction id. A second miss stays unconfirmed and does not start another transaction. The legacy path still treats silence as acceptance and can still copy KV from position 0.

A revised reservation miss with a known gap of at most `DLLAMA_ALGO2_ROLLBACK_CATCHUP` tokens (default 128) may retransmit that tail, or recompute it if the retransmit does not install. That is catch-up, not a ready reservation. A larger or unknown gap aborts.

The health reference is the recovery anchor. A kept move does not replace it and does not mark the run recovered. The next search uses the new allocation's own window. One worse tail sub-window is not a rollback. A worse median rolls back immediately. A tail that stays worse for two non-overlapping verify windows rolls back. A spike that does not repeat is kept.

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
| Cost | The k=6 prediction used a stage average that mixed fixed and communication time into the layer cost | That prediction is not issued unless a per-layer vector or an explicit fixed overhead is supplied |
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

The previous binary fired on the injected run. Static mode with the same initial split did not move. That transaction checked a live cursor, used the reservation, passed settle and verify, and kept the move under the old rule that then cleared the health reference. The unit tests now also cover an unseparated fixed overhead, a kept move that leaves the health anchor in place, one tail spike, a repeated tail, a missing end-to-end sample, an unconfirmed commit, and a short versus long rollback gap.

A net benefit is not shown. There is no paired static run under the same injection. The kept move improved 558 ms to 179 ms inside one injected run, and the rebuilt reference stayed near 180 ms, far from the 58 ms healthy reference. The 200.615 ms stall is the acceptance grace, so the foreground switch itself was not shown to be under 1 ms.

## Not run

- The 512-token, four-strategy, three-repeat matrix
- A paired static run with the same injection
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
```

`algo2-test` and `dynamic-tpot-test` passed after the controller was linked. `dllama` was built with `DLLAMA_CUDA` and arch 75.
