# EdgeVisor GPU-Based Usage Guide

This document describes how to build and run the GPU-enabled EdgeVisor version under
`/home/cc/yhbian/B01_Copy_API/EdgeVisor`.

## Current GPU Support Scope

The GPU path uses the Vulkan backend. It supports the EdgeVisor dynamic runtime on
compact-contiguous partitions:

- non-uniform TP within one stage, for example `2:3:3`;
- non-uniform PP / multi-stage execution, for example `1@8*1@10*1@10`;
- UDS-triggered dynamic heads migration inside a TP stage;
- UDS-triggered PP layer migration with GPU KV row export/apply;
- CPU-only build and inference are preserved.

Important limitation: the Vulkan backend is still a compact computation backend. It
does not support arbitrary sparse head ownership such as `{0,2,7}` on one GPU unless
those heads are first materialized into a contiguous view or a new sparse head-map
kernel is added. Current dynamic head migration changes each node's contiguous
`start + length` range and refreshes Vulkan pointer/operator metadata.

## Verified Environment

Server:

```bash
ssh cc@192.168.2.125
```

Project:

```bash
cd /home/cc/yhbian/B01_Copy_API/EdgeVisor
```

Vulkan dependency path:

```bash
/home/cc/yhbian/B01_Copy_API/tools/vulkan_deps
```

Models:

```bash
# 3B model
/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m

# 8B model
/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_model_llama3.1_instruct_q40.m

# tokenizer
/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t
```

GPU rule used in testing:

- Use GPU0, GPU1, GPU2 only.
- Do not use GPU3.

## Build GPU/Vulkan Version

```bash
cd /home/cc/yhbian/B01_Copy_API/EdgeVisor

PATH=/home/cc/yhbian/B01_Copy_API/tools/vulkan_deps/root/usr/bin:$PATH \
CPATH=/home/cc/yhbian/B01_Copy_API/tools/vulkan_deps/root/usr/include \
make DLLAMA_VULKAN=1 \
  LIBS='-L/home/cc/yhbian/B01_Copy_API/tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu -lvulkan -lpthread' \
  dllama -j4
```

To rebuild from scratch:

```bash
make clean
PATH=/home/cc/yhbian/B01_Copy_API/tools/vulkan_deps/root/usr/bin:$PATH \
CPATH=/home/cc/yhbian/B01_Copy_API/tools/vulkan_deps/root/usr/include \
make DLLAMA_VULKAN=1 \
  LIBS='-L/home/cc/yhbian/B01_Copy_API/tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu -lvulkan -lpthread' \
  dllama -j4
```

## Static Non-Uniform GPU TP

Start two workers on GPU1 and GPU2:

```bash
cd /home/cc/yhbian/B01_Copy_API/EdgeVisor

./dllama worker --port 19101 --nthreads 1 --gpu-index 1 > /tmp/ev_worker_gpu1.log 2>&1 &
./dllama worker --port 19102 --nthreads 1 --gpu-index 2 > /tmp/ev_worker_gpu2.log 2>&1 &
```

Run root on GPU0 with non-uniform TP:

```bash
MODEL3=/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m
TOK=/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t

./dllama inference \
  --prompt "Hi" \
  --steps 16 \
  --model "$MODEL3" \
  --tokenizer "$TOK" \
  --buffer-float-type q80 \
  --nthreads 1 \
  --max-seq-len 512 \
  --gpu-index 0 \
  --workers "127.0.0.1:19101" "127.0.0.1:19102" \
  --ratios "2:3:3"
```

The `2:3:3` ratio was used because Q40/GQA alignment can make some visually simple
ratios invalid. Avoid `1:1:1` unless you have verified head alignment for the model.

Convenience script:

```bash
/home/cc/yhbian/B01_Copy_API/run_gpu_patch_regression.sh static
```

Latest verified log:

```text
/home/cc/yhbian/B01_Copy_API/gpu_test_logs/gpu_patch_static_20260529_051336
```

## Dynamic Heads Migration Through UDS

Use the same 3-GPU non-uniform TP setup, but enable plan barriers and KV redundancy:

```bash
SOCK=/tmp/dllama_gpu_patch_plan.sock
MODEL3=/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m
TOK=/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t

DLLAMA_PLAN_CTRL_SOCKET="$SOCK" ./dllama inference \
  --prompt "Hi" \
  --steps 16 \
  --model "$MODEL3" \
  --tokenizer "$TOK" \
  --buffer-float-type q80 \
  --nthreads 1 \
  --max-seq-len 512 \
  --gpu-index 0 \
  --workers "127.0.0.1:19101" "127.0.0.1:19102" \
  --ratios "2:3:3" \
  --enable-plan-barrier \
  --enable-kv-redundancy-during-migration 1 \
  --kv-redundancy 2
```

Send a dynamic migration command:

```bash
python3 examples/plan-uds-client.py "$SOCK" set_plan \
  --seq 101 \
  --mode next_barrier \
  --stage 0 \
  --from 1 \
  --to 2 \
  --kind 1 \
  --heads 1 \
  --ffn 0
```

Expected evidence in logs:

```text
[plan][emit][gpu]
[plan][apply][gpu] node=0 ...
[plan][apply][gpu] node=1 ...
[plan][apply][gpu] node=2 ...
```

Convenience script:

```bash
DYN_STEPS=16 /home/cc/yhbian/B01_Copy_API/run_gpu_patch_regression.sh dynamic
```

Latest verified log:

```text
/home/cc/yhbian/B01_Copy_API/gpu_test_logs/gpu_patch_dynamic_20260529_051426
```

## Static Multi-Stage GPU PP

Example 3-stage split on GPU0/GPU1/GPU2:

```bash
./dllama inference \
  --prompt "Hi" \
  --steps 16 \
  --model "$MODEL3" \
  --tokenizer "$TOK" \
  --buffer-float-type q80 \
  --nthreads 1 \
  --max-seq-len 512 \
  --gpu-index 0 \
  --workers "127.0.0.1:19201" "127.0.0.1:19202" \
  --ratios "1@8*1@10*1@10"
```

Meaning:

- stage 0: node0, layers `[0, 8)`;
- stage 1: node1, layers `[8, 18)`;
- stage 2: node2, layers `[18, 28)`.

## Dynamic PP Layer Migration

PP layer migration uses GPU KV row export/apply to transfer the history rows for the
migrated boundary layer, then switches layer ownership after ack.

Convenience script:

```bash
/home/cc/yhbian/B01_Copy_API/run_gpu_pp_migration.sh
```

Expected evidence:

```text
[kv-export-gpu] node=0 ...
[kv-write-gpu] node=1 ...
[kv-migrate] ack batch complete layers=1 -> switch ownership
[layer-gate] node=0 role=primary layer=7 enabled=0
[worker-switch] node=1 activate redundant layer=7
```

Latest verified log:

```text
/home/cc/yhbian/B01_Copy_API/gpu_test_logs/gpu_pp_migration_20260529_051507
```

## 8B GPU Check

The 8B model also runs on GPU0/GPU1/GPU2 with the same non-uniform TP pattern:

```bash
MODEL8=/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_model_llama3.1_instruct_q40.m

./dllama inference \
  --prompt "Hi" \
  --steps 8 \
  --model "$MODEL8" \
  --tokenizer "$TOK" \
  --buffer-float-type q80 \
  --nthreads 1 \
  --max-seq-len 512 \
  --gpu-index 0 \
  --workers "127.0.0.1:19401" "127.0.0.1:19402" \
  --ratios "2:3:3"
```

Previously verified log:

```text
/home/cc/yhbian/B01_Copy_API/gpu_test_logs/gpu_8b_static_20260527_160422
```

## CPU Build Still Works

CPU-only build:

```bash
cd /home/cc/yhbian/B01_Copy_API/EdgeVisor
make clean
make dllama -j4
```

CPU regression log from the GPU adaptation work:

```text
/home/cc/yhbian/B01_Copy_API/gpu_test_logs/cpu_single_after_gpu_kv_patch.log
```

## Troubleshooting

- If a ratio fails with Q40/GQA alignment, use `2:3:3` for 3-way TP.
- If a single-GPU run OOMs on Tesla T4, use distributed GPU mode.
- Keep GPU3 unused if another process is present.
- Dynamic heads migration currently assumes compact-contiguous head ranges.
- Arbitrary sparse head assignment requires an additional sparse head map or
  weight/KV repacking path in the Vulkan kernels.
