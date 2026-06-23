# CUDA support matrix

| Area | CPU | Vulkan | CUDA |
|---|---:|---:|---:|
| Device-independent pointer/slice layout | Yes | Yes | Foundation only |
| `PNTR_RAW`, `PNTR_BATCH`, `PNTR_BATCHED_SLICE` | Yes | Yes | Not implemented |
| Tagged uneven and stage-local slices | Yes | Yes | Not implemented |
| `NnTensorView` validation | Yes | Yes | Foundation only |
| CUDA device, memory, and operators | N/A | N/A | Not implemented |

Current milestone: PR 1 test and common-semantics baseline. CUDA build and device
support begin in PR 2.
