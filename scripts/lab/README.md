# Four-GPU Docker lab

This is a model-backed smoke gate for a Linux SSH host with four CUDA GPUs,
Docker access, an existing CUDA development image with `tc`, and user-prepared
EdgeVisor/DLLama Q40 model and tokenizer files. It does not download models or
images, change host traffic control, change GPU clocks, or stop other workloads.

The first validated deployment uses four Tesla T4s on `ssh yhs1`; the engine
snapshot is `bb97cab`. Results and limitations are recorded in
`docs/test_records/12_yhs1_四T4_Docker实验环境验收_2026-10-13.md`.

## Deployment contract

Choose a unique run ID and a fresh directory owned by your SSH user. Create one
Docker bridge network with `--internal` and label `edgevisor.lab=<run-id>`.
Create four containers with the same label and one distinct GPU UUID per
container, using `--gpus device=<UUID>`. Do not use `--privileged`, host network
or a Docker socket mount. The lab uses `--cap-drop ALL --cap-add NET_ADMIN`
and `--security-opt no-new-privileges`; only `tc` setup needs container root.
Inference and probe processes run as the ordinary remote UID/GID.

The tested resource limits are four CPUs, 6 GiB memory, no container swap and
256 PIDs per container. The root filesystem is read-only with a 256 MiB `/tmp`
tmpfs. Set an application CUDA cache path in `/tmp` when needed. GPU device
visibility is restricted, but GPU memory and scheduling are still shared with
other processes using that physical GPU.

Each container needs these bind mounts:

| Destination | Source | Mode |
| --- | --- | --- |
| `/source` | `<remote_base>/source` | read-only |
| `/models` | existing compatible model/tokenizer directory | read-only |
| `/labtools` | `<remote_base>/tools`, containing compiled `net_probe` | read-only |
| `/out` | `<remote_base>/results/nodeN`, owned by the remote user | read-write |

Use an explicit `/bin/bash` entrypoint and keep the containers idle with
`sleep infinity`. No host ports need to be published. Stop/start the four
owned containers between coding sessions if desired; keep the network and
mount configuration for reuse.

## Source/build preparation

From the repository root:

```bash
python3 scripts/lab/package_sources.py /tmp/edgevisor-build-input
```

Upload `source.tar.gz` and `source_manifest.json` to the dedicated remote
directory. Extract the archive into its `source` subdirectory. Copy
`net_probe.cpp` into its `tools` subdirectory. These source archives include
only the two Makefiles and C/C++/CUDA/shader source files below `EdgeVisor/src`.

Compile in a separate unprivileged container using the same image and ordinary
remote UID/GID; mount only the new source/tools directories read-write and
use `--network none --cap-drop ALL`:

```bash
make -j4 DLLAMA_CUDA=1 DLLAMA_VULKAN=0 CUDA_ARCHS=75 dllama check
g++ -std=c++11 -O2 -Wall -Wextra -Werror /tools/net_probe.cpp -o /tools/net_probe
```

`75` is the tested T4 compute capability. For another GPU, select its actual
architecture. Use a fresh source directory or clean rebuild after coding;
retain the build log and binary/model/tokenizer SHA256. Models remain in the
existing cache and are not uploaded with the source archive.

Copy `lab_config.example.json` outside the repository, replace its placeholders
with this deployment's host, label, paths, UID/GID and container names, and do
not add credentials. The model/tokenizer paths are **inside** the containers.
The fixture checks the local and deployed build-source manifests before it
starts tests, so it refuses to silently test a preceding coding snapshot.

## Run checks

Run with Python 3.9+ on the SSH client; Python is not needed in the containers.

```bash
python3 scripts/lab/docker_gpu_check.py all \
  --config /path/to/lab_config.json --output /path/to/new-run-results
```

Actions `network`, `inference`, `fault` and `cleanup` can be run separately.
Run one fixture at a time because the owned workers and qdiscs are shared
between cases. Use a fresh output/run directory for evidence you intend to
retain; these smoke cases overwrite same-named logs when rerun.

- Network: transfer 4 MiB between two containers and wait for receiver
  completion; verify unshaped, 5 Mbit/s TBF, 20 Mbit/s + 50 ms netem, and reset.
  Accept a practical tolerance around requested rates, not exact equality.
- Inference: short Paris answer; single-GPU and four-stage 28-layer PP; tail
  rate/delay; reset; `--auto`; and a stopped middle worker. All successful
  inference uses one thread, 512 context, temperature 0 and seed 42.
- Static rate/delay/reset cases compare `(position, token)` to the same
  four-stage baseline. Automatic migration reports comparison but does not
  promise bitwise identity after changing the mapping.
- Fault: only stop a PID recorded by this run after checking its executable
  inside the owned worker container. Expect successful completion and one
  session restart, rather than hot takeover with unprepared Shadow KV.

The current four-stage fixture uses `1@7*1@7*1@7*1@7`, so it is for a **28-layer
model**. Configure a separate model-specific fixture before trying a 40-layer
14B model. It is not a general model benchmark or statistical performance claim.

TC changes are confined to the owned container's `eth0`; each case removes
its qdisc. Workers and probes are bounded/cleaned up, and `cleanup` checks
the configured inference port (default 19393), probe port 19400 and test processes. Containers can remain idle for the next
coding run. Root-coordinator failover, distributed physical hosts, mobile
thermal behaviour and isolated GPU-clock control are outside this smoke gate.
