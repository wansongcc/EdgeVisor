# EdgeVisor Boundary Condition Experiment

- Run: `boundary_20260526_082514`
- Remote output: `/home/cc/yhbian/B01_Copy_API/edgevisor_boundary_results/boundary_20260526_082514`
- Setup: 8 Docker containers, active `3/3/2` VG layout, `ratios=1:1:1*1:1:1*1:1`.
- Condition: stable high bandwidth `1000 Mbps`, short-context state sizes, no injected network/CPU perturbation.
- Migration for dynamic variants: Stage 1 `Node4 -> Node3`, move 1 KV/GQA head via UDS `next_barrier`.
- Variants: `with_shadow_kv` uses `kv-redundancy=2`; `without_shadow_transfer` uses `kv-redundancy=0` plus real Docker TCP KV-state transfer; `static_no_edgevisor` disables plan barrier/full-weight residency/Shadow KV under the same 3/3/2 topology.
- Validity: `18` live rows, `64` transfer rows, failed live rows = `0`.

## Boundary Summary

| model | context_len | net_delta_ms_positive_means_edgevisor_slower | edgevisor_vs_static_delta_ms | with_shadow_effective_tokens_s | without_shadow_effective_tokens_s | static_effective_tokens_s | edgevisor_vs_static_throughput_delta_pct | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3B | 128 | -1711.56 | 2382.68 | 0.8506 | 0.8167 | 0.8954 | -5.006 | EdgeVisor loses vs static baseline |
| 3B | 512 | 23739.039 | 17351.212 | 0.4071 | 0.7857 | 0.615 | -33.801 | EdgeVisor loses vs no-shadow transfer |
| 3B | 2048 | -7115.703 | 11445.822 | 0.8161 | 0.6463 | 1.4144 | -42.3 | EdgeVisor loses vs static baseline |
| 8B | 128 | 2663.164 | 41728.363 | 0.293 | 0.3162 | 0.6378 | -54.065 | EdgeVisor loses vs no-shadow transfer |
| 8B | 512 | 52855.853 | 88608.12 | 0.1759 | 0.2854 | 0.5322 | -66.943 | EdgeVisor loses vs no-shadow transfer |
| 8B | 2048 | 17760.342 | 52010.211 | 0.2309 | 0.2604 | 0.4515 | -48.864 | EdgeVisor loses vs no-shadow transfer |

## Interpretation

- EdgeVisor/Shadow KV is not a universal win in this stable high-bandwidth regime: `with_shadow_kv` is slower than `static_no_edgevisor` in `6/6` live pairs.
- It is also slower than the no-shadow transfer path in `4/6` live pairs. The clearest 8B rows are 128/512/2048 context, where EdgeVisor adds `2.66s`, `52.86s`, and `17.76s` over no-shadow transfer respectively.
- For 3B, no-shadow transfer has more run-to-run noise: 128 and 2048 context still favor Shadow KV over transfer, but all 3B rows are slower than the static stable baseline. This is expected for the no-wave condition because the baseline pays no redundancy cost and does not need migration.
- The boundary condition is therefore: short or moderate KV state, high/stable bandwidth, and no sustained straggler. In that region the avoided transfer cost is too small, or unnecessary, so EdgeVisor redundancy can dominate end-to-end time.
- `threshold_context_tokens_est` is derived from measured live redundancy overhead and measured transfer time. Positive values estimate the context length where transfer cost would start to match redundancy overhead; below that threshold, no-shadow transfer can be competitive.

## Transfer Microbench Evidence

| model | context_len | bandwidth_mbps | state_MB | avg_transfer_ms | avg_transfer_mbps | repeats |
| --- | --- | --- | --- | --- | --- | --- |
| 3B | 128 | 1000 | 1.125 | 118.266 | 79.845 | 2 |
| 3B | 512 | 1000 | 4.5 | 144.43 | 261.62 | 2 |
| 3B | 2048 | 1000 | 18.0 | 274.943 | 549.663 | 2 |
| 3B | 16384 | 1000 | 144.0 | 1363.157 | 886.173 | 2 |
| 8B | 128 | 1000 | 1.375 | 117.383 | 98.358 | 2 |
| 8B | 512 | 1000 | 5.5 | 160.348 | 287.979 | 2 |
| 8B | 2048 | 1000 | 22.0 | 294.663 | 626.539 | 2 |
| 8B | 16384 | 1000 | 176.0 | 1646.027 | 896.947 | 2 |

## Files

- `edgevisor_boundary_condition_results.xlsx`: Excel-compatible workbook with README, boundary summary, live rows, transfer summary, transfer raw rows, and design JSON.
- `boundary_summary.csv`: paired with/without/static comparison and estimated boundary.
- `live_runs.csv`: raw live distributed inference rows with apply evidence.
- `transfer_microbench.csv`: repeated real TCP state transfer measurements.
