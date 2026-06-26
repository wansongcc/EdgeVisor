# CUDA support matrix

| Area | CPU | Vulkan | CUDA |
|---|---:|---:|---:|
| Device-independent pointer/slice layout | Yes | Yes | Used by CUDA diagnostics |
| `PNTR_RAW`, `PNTR_BATCH`, `PNTR_BATCHED_SLICE` | Yes | Yes | Common resolver available; PR4-PR12 ops use resolved layouts |
| Tagged uneven and stage-local slices | Yes | Yes | Yes for CUDA operator coverage, static TP/PP, and dynamic plan refresh |
| `NnTensorView` validation | Yes | Yes | Yes for PR4 ops, RoPE, Softmax, and F32 Matmul A/C views |
| CUDA build (`DLLAMA_CUDA=1`, optional `CUDA_ARCHS=75 87 ...`) | N/A | N/A | Yes; defaults to local GPU auto-detect and falls back to `75` if detection is unavailable |
| CUDA device enumeration and selection | N/A | N/A | Yes |
| CUDA stream lifecycle | N/A | N/A | Yes |
| CUDA hardware-aware launch configuration | N/A | N/A | Yes; detects compute capability/integrated GPU and selects block sizes per kernel family |
| CUDA pipe mirrors | N/A | N/A | Yes |
| CUDA buffer mirrors | N/A | N/A | Yes |
| CUDA weight buffers and chunked `loadWeight()` | N/A | N/A | Yes |
| CUDA segment boundary upload/download | N/A | N/A | Yes |
| `CAST F32->F32` | Yes | Yes | Yes |
| `CAST F32->Q80` | Yes | Yes | Yes |
| `SILU`, `GELU` | Yes | Yes | Yes |
| `MUL`, `SCALE` | Yes | Yes | Yes |
| `MERGE_ADD`, `MERGE_SUM` | Yes | Yes | Yes; `MERGE_ADD Q80->F32` also supported |
| `REPEAT_Z F32->Q80` | Yes | Yes | Yes |
| `EMBEDDING` | Yes | Yes | Yes for F32/Q40/Q80 weights to F32 output |
| `INV_RMS F32->F32` | Yes | Yes | Yes |
| `RMS_NORM F32->F32` | Yes | Yes | Yes |
| Llama/Falcon/Llama 3.1 `ROPE F32->F32` | Yes | Yes | Yes |
| `SHIFT F32->F32` with strided KV destination | Yes | Yes | Yes |
| `SOFTMAX F32->F32` | Yes | Yes | Yes |
| `MATMUL F32xF32->F32` | Yes | Yes | Yes via cuBLAS |
| `MATMUL Q80xQ40->F32` | Yes | Yes | Yes via custom CUDA kernels |
| `NnMatmulOpConfig.view` 0/1/2 | Yes | Yes | Yes for F32 and Q80xQ40 |
| Matmul batch and active expert Z | Yes | Yes | Yes for F32 and Q80xQ40 |
| `MULTIHEAD_ATT F32->F32` | Yes | Yes | Yes, naive attention |
| GQA and non-compact KV cache strides | Yes | Yes | Yes for CUDA attention |
| Llama/Qwen3 dense single-GPU op set | Yes | Yes | CUDA operator dispatch complete through PR8 |
| `MOE_GATE F32->F32` | Yes | Yes | Yes |
| Qwen3-MoE repeat/gate/expert/scale/merge chain | Yes | Yes | Unit-tested for Q80/Q40 expert matmul path |
| CUDA host-pipe segment boundary sync | N/A | N/A | Yes; CUDA segment outputs are downloaded before executor/network sync |
| CUDA static TP/PP with existing TCP host pipe | Yes | Yes | Yes; uses the same host pipe protocol, no NCCL/GPU-direct dependency |
| CPU/CUDA mixed execution via `--gpu-segments` | Yes | N/A | Yes; unit-tested for CUDA->host sync before CPU segment consumption |
| `OP_PLAN_BARRIER`, `OP_PLAN_APPLY` | Yes | Yes | Yes; CUDA emits/applies plan commands and logs epoch/ranges |
| CUDA `setPartitionPlan()` / `refreshPointers()` | N/A | N/A | Yes; refreshes slice sizes, MHA strides, Shift destination, Matmul views, Norm/Mul views |
| CUDA KV export/apply for PP layer migration | Yes | Yes | Yes; full row and partial range covered by unit tests |
| Runtime primary/redundant segment gate | Yes | Yes | Yes; CUDA segments use the same executor/runtime gate |
| Bubble Shadow KV | Yes | Yes | Yes; CUDA segments participate through executor shadow KV flow |
| Other CUDA operators | N/A | N/A | Not implemented |

Current milestone: PR 12 CUDA static distribution, dynamic plan refresh, and KV migration
coverage. CUDA build defaults to detecting local GPU compute capability via `nvidia-smi` when `CUDA_ARCHS` is not explicitly set; cross-compilation can still pass `CUDA_ARCHS=75 87 ...`. At runtime CUDA detects the selected device compute capability (`sm_xy`), warp size, integrated/discrete type, and max block size, then chooses launch parameters for elementwise, reduction, attention, Q80xQ40 matmul, softmax, and MoE gate kernels. CUDA can allocate pipe and buffer
mirrors with `cudaMalloc`, use pinned host staging plus `cudaMemcpyAsync`, load
weights by byte offset, synchronize segment inputs/outputs at the segment
boundary, execute the PR4 elementwise/cast/aggregate operators, refresh
position/index pipes at every segment boundary, and execute the PR5 operators
above. It also uses cuBLAS on the segment stream for F32 matmul view 0/1/2,
including A/C tensor views, row slices, column slices, resident-relative starts,
batch execution, and active expert Z. Q80xQ40 matmul now directly reads
`NnBlockQ80` and `NnBlockQ40`, uses block-scale conversion, Q40 nibble unpacking,
integer accumulation, and FP32 output through small-K and tiled large-K custom
kernels. CUDA `MULTIHEAD_ATT` uses the same naive attention semantics as the
CPU/Vulkan backends, supports GQA, `qStart`/`qStride`/`kvStart`/`kvStride`, and
non-compact KV cache buffers for multi-token prefill and single-token decode.
CUDA `MOE_GATE` matches CPU/Vulkan top-k semantics, writes active expert indexes,
normalizes top-k weights when requested, and is covered together with the
Qwen3-MoE `REPEAT_Z -> gate -> expert matmul -> SCALE -> MERGE_SUM` chain.

PR10–PR12 add CUDA participation in the existing host-pipe distributed runtime:
segment outputs are downloaded before host/network synchronization, host-updated
pipes are re-uploaded before the next CUDA segment, and CPU/CUDA mixed execution
uses the same `--gpu-segments` routing as Vulkan. CUDA now implements dynamic
`OP_PLAN_BARRIER`/`OP_PLAN_APPLY`, `setPartitionPlan()`, and `refreshPointers()`
for head/FFN repartitioning. CUDA also implements `exportLayerKvRow()` and
`applyTransferredKvRow()` for full-row and partial-range PP KV transfer. These
features are covered by program-generated CUDA unit tests in `nn-cuda-test`;
model-level L3/L4/L5 runs are driven by environment-configured scripts so model
paths, GPU IDs, workers, and ports are not hard-coded.

Validation entry points:

- `scripts/cuda_static_semantics.sh`: build + CUDA unit tests, optional single
  GPU model run, optional static TP/PP/TP+PP root runs using `EDGEVISOR_WORKERS`.
- `scripts/cuda_dynamic_semantics.sh`: build + CUDA PR11/PR12 unit coverage,
  optional dynamic model run with `--enable-plan-barrier`, PP migration, and
  Bubble Shadow KV flags.
- `scripts/cuda_full_regression.sh`: CPU L0, CUDA unit tests, optional
  Compute Sanitizer and model-level gates.

Required model-level environment:

- `EDGEVISOR_MODEL3`
- `EDGEVISOR_TOKENIZER`
- optional `EDGEVISOR_MOE_MODEL`
- optional `EDGEVISOR_WORKERS` for L4/L5 distributed root runs
- optional `EDGEVISOR_GPU_INDEX`, `EDGEVISOR_PORT_BASE`, and
  `EDGEVISOR_PLAN_SOCKET`
