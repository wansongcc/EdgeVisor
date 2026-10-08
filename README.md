# EdgeVisor

Pipeline-parallel Q40 inference for a few Linux machines on a LAN.

## One machine

Prepare a compatible EdgeVisor/DLLama Q40 model (`.m`) and its matching tokenizer (`.t`) yourself. EdgeVisor does not provide model downloads or distribute model weights. The examples use Qwen3 0.6B Q40; place your files in the paths below, or replace those paths.

From a clean Linux checkout:

```bash
make dllama
cd EdgeVisor
./dllama inference \
  --model models/qwen3_0.6b_q40/dllama_model_qwen3_0.6b_q40.m \
  --tokenizer models/qwen3_0.6b_q40/dllama_tokenizer_qwen3_0.6b_q40.t \
  --prompt "The capital of France is" \
  --max-seq-len 4096 \
  --steps 256
```

You need `g++` and `make`, plus the development tools for your GPU backend. Python 3 is needed only for optional conversion utilities. The tested 0.6B Q40 model file is about 958 MB (914 MiB), plus a 2.2 MB tokenizer.

`make dllama` compiles CUDA when `nvcc` is on the path, and Vulkan when `glslc` or `glslangValidator` is on the path. Otherwise it builds the CPU binary. `make DLLAMA_CUDA=0 DLLAMA_VULKAN=0 dllama` forces CPU.

`./dllama --list-devices` prints the compiled backends and exits. The default backend is `auto`: CUDA, then Vulkan, then CPU. `--backend cpu` forces CPU. The thread count defaults to the number of cores. Q40 weights use q80 activation buffers by default; these are separate from the KV cache.

For a single GPU or a pipeline stage with one device, `--nthreads 1` can reduce executor overhead, especially when the GPU is shared with other jobs. A successful memory allocation does not guarantee predictable latency under competing GPU workloads.

Default output includes the device, buffer choice, generated text, and root wall-clock tokens/s. `--verbose` adds handshake and per-token detail. Execution timing requires `--benchmark` (or the profiling enabled by `--auto`).

### Context capacity

The default maximum context is **4096 tokens**, capped by the model's native window. It includes prompt and generated tokens. `--steps` limits total positions, not just new tokens; very small values can stop before the answer appears.

Set `--max-seq-len N` explicitly for a different capacity. `--max-seq-len 0` requests the native window and can exhaust memory. For the tested model, a native window of 40960 requires about 8.75 GiB just for the full 28-layer F32 KV cache; 4096 requires about 896 MiB. Weights, scratch buffers, duplicated layers, and other processes need additional memory.

Choose a capacity from the memory budget before launching. 2048, 4096, 8192, etc. are convenient settings, but any supported integer works. Do not probe capacity by allocating increasingly large caches: a device can be killed by the OS before the program can recover. Distributed allocations also depend on each stage's layer count and overlap. A successful 4096-token allocation does not imply every model fits every device.

## A second machine

Copy the same model onto the second machine, build there, and start a worker from its `EdgeVisor` directory:

```bash
./dllama worker --port 9999 \
  --model models/qwen3_0.6b_q40/dllama_model_qwen3_0.6b_q40.m
```

On the first machine, pass that worker. The worker order is the pipeline order. The root supplies the context configuration to workers.

```bash
./dllama inference \
  --model models/qwen3_0.6b_q40/dllama_model_qwen3_0.6b_q40.m \
  --tokenizer models/qwen3_0.6b_q40/dllama_tokenizer_qwen3_0.6b_q40.t \
  --prompt "The capital of France is" \
  --max-seq-len 4096 \
  --steps 256 \
  --workers 192.168.1.20:9999
```

Leave `--ratios` unset to use speed packing: each active stage receives at least one layer, then faster machines fill up to a memory cap. Addresses in the lab table (`192.168.137.13`, `.15`, `.16`, `.18`, `.31`) use built-in measurements. Other machines run a short backend layer benchmark and report a memory budget; CUDA uses free device memory, whereas Vulkan currently uses heap capacity. Check the resulting allocation on machines with other GPU workloads.

Explicit ratios skip automatic placement/profiling. Their layer counts must sum to the model's layer count. For this **28-layer** example, `--ratios '1@14*1@14'` assigns 14 layers to each of two stages.

## Joining and failure recovery

Normally the root samples tokens after gathering the last stage's logits. `--auto` enables dynamic placement/migration and last-stage sampling; results and profiling return to the root over framed messages.

- A failed middle stage can be bypassed when its predecessor has both the complete redundant weights and ready shadow KV history. Otherwise the root restarts from the prompt plus text already generated. Shadow KV is opt-in (`DLLAMA_BUBBLE_SHADOW_KV=1`), currently requires `--nthreads 1` on participating stages, and needs enough `--runtime-redundant-boundary-layers` to cover the entire failed stage. Weight overlap alone does not make the cache ready.
- A failed final stage uses session restart. Restart can change subsequent tokens; it is not seamless cache recovery.
- An unavailable worker listed at startup may keep a reserved slot. When it becomes reachable, it loads weights and the complete preceding KV history, acknowledges that history, and joins between tokens. Dynamic scheduling waits for this join to commit, then resumes using the active pipeline order and updated layer ownership. This is reserved-slot joining, not arbitrary hot-add of an unlisted device.

Older guides in `docs/archive/` describe experiments and may not match current behavior.
