# How to Run Distributed Llama on 🧠 GPU

Distributed Llama can run on GPU devices using Vulkan API. This article describes how to build and run the project on GPU.

Before you start here, please check how to build and run Distributed Llama on CPU:
* [🍓 How to Run on Raspberry Pi](./HOW_TO_RUN_RASPBERRYPI.md)
* [💻 How to Run on Linux, MacOS or Windows](./HOW_TO_RUN_LINUX_MACOS_WIN.md)

To run on GPU, please follow these steps:

1. Install Vulkan SDK for your platform.
  * Linux: please check [this article](https://vulkan.lunarg.com/doc/view/latest/linux/getting_started_ubuntu.html).
  * MacOS: download SDK [here](https://vulkan.lunarg.com/sdk/home#mac).
2. Build Distributed Llama with GPU support:

```bash
DLLAMA_VULKAN=1 make dllama
DLLAMA_VULKAN=1 make dllama-api
```

Notes for NVIDIA Jetson:
- Jetson (JetPack / Ubuntu aarch64) can usually run the Vulkan backend if Vulkan drivers are installed.
- If you build on one Jetson and want to run on a different ARM device, consider `DLLAMA_PORTABLE=1`:
  - `DLLAMA_VULKAN=1 make DLLAMA_PORTABLE=1 dllama`

3. Now `dllama` and `dllama-api` binaries supports arguments related to GPU usage.

```
--gpu-index <index>   Use GPU device with given index (use `0` for first device)
```

4. You can run the root node or worker node on GPU by specifying the `--gpu-index` argument. Vulkan backend requires single thread, so you should also set `--nthreads 1`.

```bash
./dllama inference ... --nthreads 1 --gpu-index 0 
./dllama chat      ... --nthreads 1 --gpu-index 0 
./dllama worker    ... --nthreads 1 --gpu-index 0 
./dllama-api       ... --nthreads 1 --gpu-index 0 
```

## Online Migration Notes

GPU/Jetson online migration uses the same root/worker commands plus the current
runtime switches documented in:

- [README_ENV_VARS.md](README_ENV_VARS.md) for the full environment variable and
  CLI switch reference.
- [HOW_TO_ONLINE_MIGRATION.md](HOW_TO_ONLINE_MIGRATION.md) for manual UDS and
  automatic TPOT migration workflows.

For automatic PP/TPOT scheduling, start root inference with
`--enable-dynamic-tpot --plan-ctrl-socket <path>` and add
`--runtime-redundant-boundary-layers 1` when PP layer migration needs redundant
boundary weights.

## GPU Compute Disturbance Injection

For PP auto-migration experiments that need strong GPU-side contention, use the
Jetson-side disturbance helper:

```bash
cd /home/jetson/cc/EdgeVisor/EdgeVisor

./scripts/gpu_compute_disturbance.sh start
```

The default workload is `1xQwen3-14B + 1xQwen3-8B` Vulkan inference on
`--gpu-index 0`, with a 240 second disturbance window and 60 second preheat.
It creates `/tmp/edgevisor_models -> /home/jetson/cc/models` when that source
directory exists, and writes logs/pid files under `/tmp/edgevisor_disturbance`.

Check status and logs:

```bash
./scripts/gpu_compute_disturbance.sh status

tail -n 40 /tmp/edgevisor_disturbance/gpu_disturb_14b.log
tail -n 40 /tmp/edgevisor_disturbance/gpu_disturb_8b.log
```

Stop only the disturbance processes:

```bash
./scripts/gpu_compute_disturbance.sh stop
```

Do not use `pkill -x dllama` after workers are running; it will also kill the
EdgeVisor worker on the same Jetson. The helper records its own process groups
and stops only those groups.

Useful overrides:

```bash
./scripts/gpu_compute_disturbance.sh start \
  --duration 240 \
  --preheat 60 \
  --bin-dir /home/jetson/cc/EdgeVisor/EdgeVisor \
  --models-src /home/jetson/cc/models \
  --gpu-index 0
```

