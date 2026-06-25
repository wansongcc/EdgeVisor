# CUDA support matrix

| Area | CPU | Vulkan | CUDA |
|---|---:|---:|---:|
| Device-independent pointer/slice layout | Yes | Yes | Used by CUDA diagnostics |
| `PNTR_RAW`, `PNTR_BATCH`, `PNTR_BATCHED_SLICE` | Yes | Yes | Common resolver available; PR4-PR7 ops use resolved layouts |
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
| `MERGE_ADD`, `MERGE_SUM` | Yes | Yes | Yes |
| `REPEAT_Z F32->Q80` | Yes | Yes | Yes |
| `EMBEDDING F32->F32` | Yes | Yes | Yes |
| `INV_RMS F32->F32` | Yes | Yes | Yes |
| `RMS_NORM F32->F32` | Yes | Yes | Yes |
| Llama/Falcon/Llama 3.1 `ROPE F32->F32` | Yes | Yes | Yes |
| `SHIFT F32->F32` with strided KV destination | Yes | Yes | Yes |
| `SOFTMAX F32->F32` | Yes | Yes | Yes |
| `MATMUL F32xF32->F32` | Yes | Yes | Yes via cuBLAS |
| `MATMUL Q80xQ40->F32` | Yes | Yes | Yes via custom CUDA kernels |
| `NnMatmulOpConfig.view` 0/1/2 | Yes | Yes | Yes for F32 and Q80xQ40 |
| Matmul batch and active expert Z | Yes | Yes | Yes for F32 and Q80xQ40 |
| Other CUDA operators | N/A | N/A | Not implemented |

Current milestone: PR 7 CUDA Q80xQ40 matmul. CUDA can allocate pipe and buffer
mirrors with `cudaMalloc`, use pinned host staging plus `cudaMemcpyAsync`, load
weights by byte offset, synchronize segment inputs/outputs at the segment
boundary, execute the PR4 elementwise/cast/aggregate operators, refresh
position/index pipes at every segment boundary, and execute the PR5 operators
above. It also uses cuBLAS on the segment stream for F32 matmul view 0/1/2,
including A/C tensor views, row slices, column slices, resident-relative starts,
batch execution, and active expert Z. Q80xQ40 matmul now directly reads
`NnBlockQ80` and `NnBlockQ40`, uses block-scale conversion, Q40 nibble unpacking,
integer accumulation, and FP32 output through small-K and tiled large-K custom
kernels. Model-level CUDA inference is not claimed correct yet because attention
and full MoE remain later PRs.
