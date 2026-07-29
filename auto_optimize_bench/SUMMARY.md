# EdgeVisor Auto-Optimization - Findings & Methodology

## Setup
- Branch: `auto_optimization` (forked from `perf/ttft-tpot-opts`).
- Test server: `yhs1` (4x Tesla T4).
- Model: Llama 3.2 3B Instruct (Q40, fp32 activations).
- Per-token timings measured at the root inference loop via steady_clock:
  - TTFT proxy: `Evaluation (root wall-clock)` ms/tok
  - TPOT proxy: `Prediction (root wall-clock)` ms/tok
- Test scenarios (each run 3-5 times to absorb noise):
  1. `static_uneven`: GPU0 root + GPU1/GPU2 workers, ratios 2:3:3, 128 steps
  2. `dynamic_heads`: same topology + UDS-triggered TP head migration, 96 steps
  3. `pp_migration`: 3-stage pipeline 1@8*1@10*1@10 + UDS-triggered PP migration, 64 steps
  4. `agentic_proxy`: longer (256 steps) sustained decode, same uneven topology

## Baseline (3-run average, summary/baseline_multi.txt)
| Scenario              | TTFT (ms/tok) | TPOT (ms/tok) |
|-----------------------|--------------:|--------------:|
| static_uneven         |       147.88 |        28.15 |
| dynamic_heads         |       129.58 |        33.54 |
| pp_migration          |       251.12 |        33.66 |
| agentic_proxy         |       282.87 |        26.40 |

GPU profile breakdown (from `--benchmark` run): per-token total ~37 ms,
split into exec ~22 ms (Vulkan dispatch + compute) and sync ~15 ms
(per-step cross-node all-gather for head/FFN outputs).

## Optimizations attempted (all reverted; none kept)

| # | Idea                                                                | Outcome                                                                                  |
|---|---------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| 1 | SIMD-ize `matmul_Q80_Q40_F32_colSlice` (AVX2/AVX512 unrolled)     | CPU path only; benchmarked tests use Vulkan/GPU, no measurable gain. |
| 2 | Trim `executeStep` per-step branch cascade                          | Slight static_uneven TPOT gain (~3%) but agentic_proxy regression (~5-9%); reverted. |
| 3 | 1-thread fast path in `executorThreadHandler` (skip doneCount atomic)| Mixed: helps static_uneven (~2%) but agentic_proxy ~1% worse; within noise. Reverted. |
| 4 | Add `MSG_WAITALL` to all `recv` syscalls                            | Hurts agentic_proxy by 4.5%; re-conflicts with EAGAIN polling. Reverted.                |
| 5 | Skip `doneThreadCount.store(0)` in single-thread path                | Caused test stalls; reverted.                                                           |
| 6 | Pre-allocate / reuse `NnSocketIo` vector in `syncNodeSlices`         | Did not produce a measurable signal in three runs; reverted.                            |
| 7 | Bump Vulkan smallk matmul `TILE_SIZE_D` 8 -> 16                     | 33% worse TPOT on pp_migration, +13.8% worse agentic_proxy; reverted.                   |
| 8 | `SO_BUSY_POLL` + larger socket buffers (50us poll)                  | Within noise across all 4 scenarios; reverted.                                          |
| 9 | `SO_BUSY_POLL` (100us poll)                                         | Slightly worse than baseline; reverted.                                                 |

After 9 candidates the **stable** improvement is <= 3% on a single scenario
and within noise averaged across the suite, well below the 6% / 10% bar.
None of the changes were kept on the branch.

## Why the speed-up was small
- The per-step sync overhead is dominated by kernel TCP loopback syscalls
  on small (~3-5 KB) messages. Reducing CPU work does not move this.
- The matmul runs on the Vulkan shader; the CPU `matmul_*_colSlice`
  functions are never exercised on the GPU-path benchmark.
- The per-step executor overhead is small (~0.5 ms / token) compared to
  GPU dispatch + sync (~36 ms / token), so micro-optimizing the loop body
  cannot move the wall-clock more than ~1-2%.
- The biggest GPU-side cost is per-dispatch overhead (~30us x 728 dispatches
  = ~22 ms / token), which is dominated by the Vulkan driver and the
  underlying command buffer recording; bumping the matmul tile size
  (TILE_SIZE_D 8 -> 16) actually regressed, suggesting the T4 has
  register-pressure limits at the larger tile.

## Kept changes
- `src/dllama.cpp`: added `Evaluation (root wall-clock)` printout so the
  benchmark harness has a reliable TTFT proxy in non-continuous mode.
- `auto_optimize_bench/`: harness directory (kept untracked by design).

## Suggested next steps (out of scope of this round)
1. **Shader fusion**: combine `rms_norm + matmul` (and similar) pairs into a
   single dispatch; could remove ~28 dispatches per forward on the 3B
   model. This is the largest remaining GPU-side win.
2. **Pipeline-overlap executor**: re-order segments so layer-N FFN sync
   overlaps with layer-(N+1) ATTN compute.
3. **Unix-domain sockets for local workers**: replace AF_INET loopback
   with AF_UNIX to avoid TCP overhead per sync step.
4. **sendmmsg/recvmmsg batched I/O**: collect multiple writeMany
   targets into one syscall batch per forward.
