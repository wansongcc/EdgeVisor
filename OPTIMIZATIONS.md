# two-level-slack — Optimization Summary

## Real wins (performance / capability)

### 1. Shadow KV Drain off the critical path (L2 main story)

| Workload | Before (L1, drain on critical path) | After (L2, stash → tool-wait) | Gain |
|---|---|---|---|
| CPU 2pp async, `bubbleDrain/fwd` | 2.78 ms | **0.00 ms** | drain removed from critical path |
| 8B b8, stage1 ms/fwd | L1: 319.51 (drain 4.05) | **L2: 173.36** (drain 0.00) | **46% faster than L1, 42% faster than off** |
| 4×T4 b8, stage1 ms/fwd | L1: 183.80 (drain 2.15) | L2: 161.02 (drain 0.00) | **12% faster than L1** |
| 4×T4 b2, stage1 ms/fwd | L1: 86.34 (drain 0.95) | L2: 83.29 (drain 0.00) | ~equal (L1 was already fast) |

- **Mechanism**: shadow work that doesn't fit in the bubble window is no longer drained at forward-end; instead it is stashed as "debt" and replayed inside the tool-wait window.
- **Token md5 consistency**: all 12 matrix cases (3B off/L1/L2 × batch 1/2/4/8) + 8B b1/b8 × off/L1/L2 produce identical tokens → zero semantic regression.

### 2. 8B model first-time end-to-end on 4-GPU PP

- New capability: `1@8*1@8*1@8*1@8` topology, 3B q40 + 8B q40 both pass matrix off/L1/L2 on 4×T4 with shared ~3.7 GB headroom.
- 8B b8 L2 (173 ms/fwd) is the single most convincing number — the L2 design story reproduces at the larger model.

### 3. CUDA backend end-to-end + real agentic web_search

- Added `--edge-backend {auto,cuda,vulkan}` (was previously forced to vulkan via `BACKEND_AUTO`).
- Verified on 4×T4 CUDA: real Bing `web_search` (511 ms) → tool-wait → L2 catches up 48 shadow entries in 149 ms → next generation continues.
- Two pitfall fixes recorded: `CUDA_VISIBLE_DEVICES` mask, `cudaSetDevice failed: invalid device ordinal (101)` for out-of-mask worker.

### 4. LSS compatibility with single-node-stage topology

- Previously unusable on `1@N*1@N*...` (single-node stage) PP — three pre-existing bugs fixed:
  - `27ce884`: sampler vocabSize summed over all nodes → off-by-vocab argmax → decode assert crash.
  - `9bd0f8c` / `69b4b5d`: last-stage sampled-token packets drained for batchSize==1 prefill in both inference and chat modes.
- After fix: 12 matrix cases pass LSS, identical generation to non-LSS.

## Correctness guard-rails that unblock the above wins

### 5. L1 Bubble Shadow KV numerical fixes

| Bug | Before | After |
|---|---|---|
| Right-boundary shadow input from `OP_MERGE_ADD zqPipe→xBuffer` | double-add (xBuffer already merged by pp_stage_merge) | reads `pp_stage_out` snapshot; red_k/red_v bit-equal to reference |
| Right-boundary shadow shared `kTemp/vTemp` with takeover | latent corruption risk | private buffers (`shadowKTempSlicePtr` etc.) |
| Right-boundary shadow running before pp_send | race | gated by `segmentReadyAfterStep` |
| Left-boundary shadow | inherently off-by-one layer | default off; `DLLAMA_SHADOW_LEFT_ENABLE=1` to restore |

- E2E migration (layer 14 → stage0, pos 34): shadow / recompute / baseline all produce identical tokens.
- Main-path non-pollution: bubble-on vs clean baseline → identical tokens + main-path K/V bit-zero.

### 6. Critical racefix (`8d9730c`)

- Symptom: mid-flight catch-up join overwrites `execution->batchSize` with `savedBatchSize` (old value) after the handler has set it for the new forward → executor runs batchSize=1 on a batch=23 prefill → `pp_send` emits 1 row → worker waits for 22 → **deadlock**.
- Repro: gdb backtrace (`syncNodeSlices → readMany`), strace (root `sendto` 12288 B = 1 row only).
- Fix: re-assert `batchSize/position` + POS/SLT pipe contents after join.
- Without this, **L2 hangs at any large debt + interrupt scenario**.

### 7. Device-thread pinning (`0b5106c`)

- bubble / L2 catch-up thread entry calls `NnDevice::setCurrentThreadDevice()` (CPU no-op; CUDA = `cudaSetDevice(gpuIndex)`).
- Without it, kernel launches from a thread spawned on the wrong device land in the wrong CUDA context → silent miscompute.
- CPU build clean; nvcc 12.8 compile check passes.

### 8. Non-blocking control peek

- catch-up thread does non-blocking peek on root socket → returns to main loop on new control packet.
- UDS `tool_window_begin` is fire-and-forget, no ACK, never blocks inference.

## "No regression" coverage (also part of the win)

| Suite | Pass rate |
|---|---|
| semantic CPU | 6/8 (2 pre-existing: kvDim%3≠0, F32×Q40 no kernel — main fails identically) |
| semantic GPU | 4/5 (1 pre-existing: vulkan no F32 shader) |
| GPU pp/migration/patch | 4/4 |
| CB + bubble + L2 combination | RC=0, 0 error lines, bubbleDrain=0 |
| 12 LSS matrix cases | md5 all-match |
| 12 shadow-KV matrix cases (3B b1-b8 × off/L1/L2) | md5 all-match |
| 4 LSS 8B cases (b1/b8 × off/L1/L2) | md5 all-match |
| 4 8B matrix cases (b1/b8 × off/L1/L2) | md5 all-match |

## Honest disclosure: auto-optimize (committed `auto_optimize_bench/SUMMARY.md`)

**9 optimization candidates attempted — all reverted, none kept.** Each candidate sat within the ±3% noise floor; the strict 6% / 10% bar was not met.

- SIMD-ize `matmul_Q80_Q40_F32_colSlice`: not on the GPU benchmark path.
- Trim `executeStep` per-step branch cascade: static_uneven +3%, agentic_proxy −5-9% → reverted.
- 1-thread fast path in `executorThreadHandler`: mixed within noise.
- `MSG_WAITALL`: −4.5% on agentic_proxy.
- Skip `doneThreadCount.store(0)` in single-thread path: stalled tests.
- Pre-allocate `NnSocketIo` vector: no measurable signal.
- Vulkan smallk matmul `TILE_SIZE_D` 8→16: +33% TPOT on pp_migration, +13.8% on agentic_proxy.
- `SO_BUSY_POLL` 50µs / 100µs: within noise / slightly worse.

**Root cause of low headroom**: per-step sync (~15 ms/tok) is kernel TCP loopback on small messages; per-step GPU dispatch overhead (~22 ms/tok) is Vulkan driver + command-buffer recording. Micro-optimizing the executor loop body cannot move wall-clock more than ~1-2%.

**Kept change**: `Evaluation (root wall-clock)` printout in `src/dllama.cpp` so the bench harness has a reliable TTFT proxy in non-continuous mode.

**Suggested next steps (out of scope of this round)**:
1. **Shader fusion** (rms_norm + matmul): remove ~28 dispatches per forward on the 3B model — biggest remaining GPU win.
2. **AF_UNIX for local workers**: replace AF_INET loopback to avoid TCP overhead per sync step.
3. **`sendmmsg` / `recvmmsg`** batched I/O: collect multiple `writeMany` targets into one syscall batch.
4. **Pipeline-overlap executor**: re-order segments so layer-N FFN sync overlaps with layer-(N+1) ATTN compute.

## TL;DR

**Headline**: Shadow KV L2 moves bubble-overflow drain from the critical path into tool-wait catch-up — **8B b8 is 46% faster than L1 and 42% faster than off** with zero semantic regression.

**Sub-headline**: CUDA backend E2E agentic web_search + 8B on 4-GPU PP both unlocked.

**Prerequisites that had to land first**: L1 numerical correctness, LSS bug fixes, catch-up racefix, device-thread pinning — without these, L2 would be a numerically-wrong deadlock-prone feature.

**What did *not* move**: auto-optimize micro-opts (9 candidates reverted, honest) — the bottleneck is in TCP loopback sync and Vulkan driver dispatch, not in the executor loop body.