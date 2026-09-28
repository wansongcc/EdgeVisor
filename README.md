# EdgeVisor

Pipeline-parallel inference for a few machines on a LAN. One command builds the binary. The first run uses a small model.

## One machine

From a clean Linux checkout:

```bash
make dllama
cd EdgeVisor
python3 launch.py qwen3_0.6b_q40 -y -skip-run
./dllama inference \
  --model models/qwen3_0.6b_q40/dllama_model_qwen3_0.6b_q40.m \
  --tokenizer models/qwen3_0.6b_q40/dllama_tokenizer_qwen3_0.6b_q40.t \
  --prompt "The capital of France is" \
  --steps 16
```

`make dllama` compiles CUDA when `nvcc` is on the path, and Vulkan when `glslc` or `glslangValidator` is on the path. Otherwise it builds the CPU binary. `make DLLAMA_CUDA=0 DLLAMA_VULKAN=0 dllama` forces CPU.

You need `g++`, `make`, and Python 3. The 0.6B download is about 400 MB.

`./dllama --list-devices` prints the compiled backends and then exits. The default backend is `auto`: CUDA, then Vulkan, then CPU. `--backend cpu` forces the CPU. The thread count defaults to the number of cores. A Q40 model uses q80 activation buffers unless you pass `--buffer-float-type`.

Default output is the device line, the buffer choice, generated text, and tokens/s. `--verbose` adds handshake and per-token detail.

## A second machine

Copy the same model files onto the second machine, build there, and start a worker:

```bash
./dllama worker --port 9999 \
  --model models/qwen3_0.6b_q40/dllama_model_qwen3_0.6b_q40.m
```

On the first machine, pass that worker. The worker order is the pipeline order.

```bash
./dllama inference \
  --model models/qwen3_0.6b_q40/dllama_model_qwen3_0.6b_q40.m \
  --tokenizer models/qwen3_0.6b_q40/dllama_tokenizer_qwen3_0.6b_q40.t \
  --prompt "The capital of France is" \
  --steps 16 \
  --workers 192.168.1.20:9999
```

Leave `--ratios` unset. Each machine is given one layer, then faster machines fill up to a memory cap. Addresses in the lab table (`192.168.137.13`, `.15`, `.16`, `.18`, `.31`) use the measured table. Any other address is profiled once: one matmul on that machine's backend, plus a layer cap from free memory. `--ratios 1@8*1@8` skips both.

## What the program is doing

The root holds the prompt and samples the next token. Each worker holds a contiguous range of transformer layers. One token walks root to the last worker, then the logits come back. Adding a machine splits that walk; it does not change the model.

Three behaviors matter once a machine drops:

- A middle machine whose layers are already cached on the previous stage is skipped, and decoding continues on the same tokens.
- If that cache is not ready, or the last machine dies, the root starts the sentence again from the text produced so far.
- An offline worker can keep a reserved slot and join back between tokens.

Older guides are in `docs/archive/`.
