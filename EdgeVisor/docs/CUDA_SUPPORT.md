# CUDA support matrix

| Area | CPU | Vulkan | CUDA |
|---|---:|---:|---:|
| Device-independent pointer/slice layout | Yes | Yes | Foundation only |
| `PNTR_RAW`, `PNTR_BATCH`, `PNTR_BATCHED_SLICE` | Yes | Yes | Common resolver available; no CUDA memory binding yet |
| Tagged uneven and stage-local slices | Yes | Yes | Common resolver available; no CUDA memory binding yet |
| `NnTensorView` validation | Yes | Yes | Foundation only |
| CUDA build (`DLLAMA_CUDA=1 CUDA_ARCHS=75`) | N/A | N/A | Yes |
| CUDA device enumeration and selection | N/A | N/A | Yes |
| CUDA stream lifecycle | N/A | N/A | Yes |
| CUDA memory and operators | N/A | N/A | Not implemented |

Current milestone: PR 2 build/CLI/device skeleton. CUDA can be selected with
`--backend cuda --gpu-index <index>` and `nn-cuda-test` prints device
properties. Real CUDA buffers, weight loading, data transfer, and operators begin
in PR 3 and later; trying to execute a model with CUDA now fails explicitly
instead of falling back to CPU.
