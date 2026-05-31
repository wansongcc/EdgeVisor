# EdgeVisor Boundary Condition Experiment

- Run: `boundary_20260526_082103`
- Remote output: `/home/cc/yhbian/B01_Copy_API/edgevisor_boundary_results/boundary_20260526_082103`
- Setup: 8 Docker containers, active `3/3/2` VG layout, `ratios=1:1:1*1:1:1*1:1`.
- Condition: stable high bandwidth `1000 Mbps`, short-context state sizes, no injected network/CPU perturbation.
- Migration: Stage 1 `Node4 -> Node3`, move 1 KV/GQA head via UDS `next_barrier`.
- Variants: `with_shadow_kv` uses `kv-redundancy=2`; `without_shadow_transfer` uses `kv-redundancy=0` and performs real Docker TCP KV-state transfer before migration.

## Boundary Summary

| model | context_len | net_delta_ms_positive_means_edgevisor_slower | edgevisor_vs_static_delta_ms | redundancy_overhead_ms_est | threshold_context_tokens_est | throughput_delta_pct | edgevisor_vs_static_throughput_delta_pct | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3B | 128 | -19064.457 | 13144.135 | -18879.502 |  |  |  | EdgeVisor loses vs static baseline |

## Interpretation

No live row showed a strict EdgeVisor loss in this run. Use the transfer summary and threshold estimates to locate the crossover, or rerun with a higher `B01_BOUNDARY_SHADOW_KV` value to stress redundant KV maintenance.

The `threshold_context_tokens_est` column estimates where the avoided transfer cost would equal the measured redundancy overhead. Contexts below that estimate are the region where no-shadow transfer can be competitive or faster under this stable high-bandwidth setup.

## Files

- `edgevisor_boundary_condition_results.xlsx`: Excel-compatible workbook with summary, live runs, transfer summary, raw transfer rows, and design JSON.
- `live_runs.csv`: live distributed inference timing rows.
- `transfer_microbench.csv`: repeated real TCP state transfer measurements.
- `boundary_summary.csv`: paired with/without comparison and estimated boundary.
