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

The revised path compares per-token gain with `max(3 ms, 0.03 * (source + target))`. Uncalibrated prediction noise stays out of that threshold. A move is issued only when the recorded shadow cursor covers the current token. A reservation miss aborts. It does not copy KV from position 0. The legacy path still does.

Default decode remains one request in `inferenceRunOnce`. The token clock is that request's completion interval on the root. The first completed token has no interval, so its metric is the stage-compute sum. Later tokens use the completion interval. Continuous batching is not on this path.

## Audit

| Item | Finding |
| --- | --- |
| Serial decode | `inferenceRunOnce` forwards one token, samples, then continues. The T4 runs used that path |
| Legacy stage time | The old controller sums per-stage exec. That sum is not called end-to-end time |
| Legacy trend cap | `clampTrendPenalty` still caps the old path at 1.3. The revised estimator does not use it |
| Legacy verify | The old path still uses its 1.05 window. The revised path uses its own settle, verify, and cooldown |
| Shadow cursor | `shadowFilledThrough` is the minimum exclusive end of redundant and shadow-KV layers on that node. `0xFFFFFFFF` means the node has no such layer |
| Readiness | Cursor `>= token + 1` is ready. Unknown stays unknown. Weights being present is not readiness |
| Foreground switch | Not measured below 1 ms. The reservation-hit path waits a fixed 200 ms grace because silence means acceptance. The recorded stall was 200.615 ms with `stateBytes=0` |
| Containers | Docker is installed and the nvidia runtime is present. Pulling `nvidia/cuda:12.0.0-base-ubuntu22.04` failed: the configured proxy `192.168.1.11:10810` refused the connection. The T4 run was on the host. No host `tc`, no clock change, no power change |
| Edge set | nx1, nx2, nano1, nano2, and rog answered SSH. The Q40 model is on nx1. nvcc 12.6 is on nx1. No algorithm-2 inference was run there |

## T4 smoke

Four Tesla T4 GPUs, CUDA arch 75, driver 13020, runtime 12080. Model `llama3.2-3b-instruct` Q40, tokenizer `llama_3_1`. Split `1@7*1@7*1@7*1@7`. Algorithm-1 equal shadow under cap 6: `6,3;3,3;3,6`. Prompt `Hi`, temperature 0, seed 1, `DLLAMA_IGNORE_EOS=1`. The decision window, settle, and verify were shortened for this smoke. Those overrides are in `run_t4_smoke.sh`. They are not new defaults.

Raw logs:

- `logs/t4_static_root.log`, `logs/t4_static_tpot.log`
- `logs/t4_revised_delay_root.log`, `logs/t4_revised_delay_tpot.log`

### Static, 48 steps, no injection

Exit 0. 16.18 tokens/s, 61.82 ms/token. The controller logged 36 observe records, issued nothing, and rolled nothing back. The first full window set the reference at 56.4 ms. Later healthy windows replaced it, up to 58.4 ms. The last tokens were `suspect` with cause `mixed`, still inside the confirmation count, so static mode correctly kept the allocation. Every logged stage cursor was `token + 1`.

### Revised variable step, 120 steps, injected sleep

This is an injected per-step sleep, not a GPU throttle and not a natural fault. Node 2 slept 15 ms on its compute steps after 24 decode tokens.

Exit 0. The whole run, including the healthy prefix and the stall, was 4.57 tokens/s, 219.02 ms/token. That average is not the post-move speed.

The healthy reference settled at 58.006 ms. The first slow window was `suspect` at token 40, median 555.414 ms. At token 48 the window was `stable_degraded`, cause `mixed`, median 558.134 ms, and the controller issued k=6 from stage 2 to stage 3, layers 15 through 20, predicted gain 380.919 ms, threshold 14.997 ms. The root log shows the reservation served the move: `stateBytes=0`, `recomputeUnits=0`, position 51. Commit completed by token 52. Verify median was 179.274 ms and p95 was 180.161 ms, so the transaction was kept as `clear_improvement` against the degraded window. It was not marked recovered. The reference was cleared and the following healthy-looking window rebuilt it at 180.08 ms.

Stage 3's cursor stopped at 52 after the move, while the other stages kept advancing. A later move that depends on stage 3 is not ready.

## What this does and does not show

The statistics are the logs above: one serial Q40 run per mode, with per-token phase, health, cursor, and the root migration line.

The mechanism fired on the injected run. Static mode with the same initial split did not move. The revised transaction checked a live cursor, used the reservation, passed settle and verify, and kept the move. The unit tests cover the smaller cases: k=1 below the 3 ms bar, unknown and behind cursors, no shadow, zero memory spare, overlapping requests, fixed overhead, regression, inconclusive rollback, a failed rollback, and a local-search case that is not the exhaustive optimum.

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
bash ../experiments/algo2/run_t4_smoke.sh
```

`algo2-test` and `dynamic-tpot-test` passed after the controller was linked. `dllama` was built with `DLLAMA_CUDA` and arch 75.
