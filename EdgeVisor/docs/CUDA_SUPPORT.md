# CUDA support matrix

| Area | CPU | Vulkan | CUDA |
|---|---:|---:|---:|
| Device-independent pointer/slice layout | Yes | Yes | Used by CUDA diagnostics |
| `PNTR_RAW`, `PNTR_BATCH`, `PNTR_BATCHED_SLICE` | Yes | Yes | Common resolver available; PR4 ops use resolved layouts |
| Tagged uneven and stage-local slices | Yes | Yes | Common resolver available; operator coverage is still limited |
| `NnTensorView` validation | Yes | Yes | Yes for PR4 CAST/SILU/GELU/MUL/SCALE kernels |
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
| Other CUDA operators | N/A | N/A | Not implemented |

Current milestone: PR 4 CUDA basic operators. CUDA can allocate pipe and buffer
mirrors with `cudaMalloc`, use pinned host staging plus `cudaMemcpyAsync`, load
weights by byte offset, synchronize segment inputs/outputs at the segment
boundary, and execute the PR4 elementwise/cast/aggregate operators above.
Model-level CUDA inference is not claimed correct yet because attention, norm,
RoPE, softmax, matmul, and quantized matmul remain later PRs.
