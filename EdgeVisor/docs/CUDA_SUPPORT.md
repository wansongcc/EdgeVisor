# CUDA support matrix

| Area | CPU | Vulkan | CUDA |
|---|---:|---:|---:|
| Device-independent pointer/slice layout | Yes | Yes | Used by CUDA diagnostics |
| `PNTR_RAW`, `PNTR_BATCH`, `PNTR_BATCHED_SLICE` | Yes | Yes | Common resolver available; PR4-PR9 ops use resolved layouts |
| Tagged uneven and stage-local slices | Yes | Yes | Common resolver available; operator coverage is still limited |
| `NnTensorView` validation | Yes | Yes | Yes for PR4 ops, RoPE, Softmax, and F32 Matmul A/C views |
| CUDA build (`DLLAMA_CUDA=1 CUDA_ARCHS=75`) | N/A | N/A | Yes |
| CUDA device enumeration and selection | N/A | N/A | Yes |
| CUDA stream lifecycle | N/A | N/A | Yes |
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
| Other CUDA operators | N/A | N/A | Not implemented |

Current milestone: PR 9 CUDA Qwen3-MoE operator coverage. CUDA can allocate pipe and buffer
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
