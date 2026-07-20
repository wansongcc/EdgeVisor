# Jetson Chunked Weight Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove root token-embedding mmap peaks and last-stage CUDA pinned-staging peaks without changing tensor layout or model partitioning.

**Architecture:** Root embedding loading will mmap one bounded F32 row chunk at a time, quantize it to Q80, upload it, and release the mapping before the next chunk. Contiguous row-matrix slices in the local loader will upload in bounded row-aligned chunks, preserving each chunk's file and device offsets.

**Tech Stack:** C++11, POSIX mmap, CUDA runtime pinned host memory, existing C++ test/build system.

## Global Constraints

- Preserve model file offsets and final device-buffer layouts.
- Preserve CPU, Vulkan, and CUDA loader semantics.
- Do not change PP ratios, runtime migration policy, or redundant-layer behavior.
- Keep the existing root embedding quantization format and chunk row size semantics.

### Task 1: Add failing regression coverage for bounded loading

**Files:**
- Create: `EdgeVisor/src/test/test_chunked_weight_loading.cpp`
- Modify: `EdgeVisor/Makefile` only if required to build the focused test.

**Interfaces:**
- Test the pure chunk-range calculation helpers before integrating them into mmap and loader code.
- Expected helper behavior: split a contiguous row range into row-aligned chunks no larger than the configured byte cap, including a short final chunk.

- [ ] **Step 1: Write the failing test**

Test a 100-byte row stride with a 250-byte cap and assert chunks are `[0,200]`, `[200,200]`, `[400,100]`; assert a single row larger than the cap remains one row-sized chunk.

- [ ] **Step 2: Run the focused test and verify it fails**

Run the repository's focused C++ test command after adding the test target. Expected result: compile failure because the chunk helper does not exist.

### Task 2: Implement root embedding chunked mmap

**Files:**
- Modify: `EdgeVisor/src/llm.cpp`
- Modify: `EdgeVisor/src/llm.hpp` only if a shared helper declaration is needed.

**Interfaces:**
- Keep `loadRootTokenEmbeddingQ80()` behavior unchanged for callers that already provide a complete range.
- Add a loader-local chunked path that maps `[fileOffset + rowStart * dim * sizeof(float), chunkBytes]`, invokes the existing quantizer/uploader for that chunk, then releases the mapping.

- [ ] **Step 1: Implement a bounded embedding-map loop**

Use the existing `rowsPerChunk = 1024` quantization granularity. For each chunk, call `mapWeightRange()` with only that chunk's F32 bytes, call `loadRootTokenEmbeddingQ80()` with a chunk-adjusted header/input, and reset the mapping immediately after upload.

- [ ] **Step 2: Preserve final file offset validation**

Advance `fileOffset` by the complete embedding byte count exactly once and retain the existing layer alignment checks.

- [ ] **Step 3: Run the focused regression test**

Expected result: root range test passes and no complete `embeddingBytes` mapping is emitted by the tested loading helper.

### Task 3: Implement bounded contiguous row-slice uploads

**Files:**
- Modify: `EdgeVisor/src/nn/nn-network-local.cpp`
- Modify: `EdgeVisor/src/nn/nn-network-local.hpp` only if a helper declaration is required.

**Interfaces:**
- Preserve `loadRowMatmulSlicesUneven()` return value as the full tensor byte size.
- Preserve `slice.sliceSize.nBytes` as the total device-resident slice size.
- Split only the upload calls; do not split strided column slices.

- [ ] **Step 1: Add a row-aligned chunk loop**

Compute `bytesPerRow` using the existing block-size calculation. Select a fixed bounded byte cap, round it down to at least one complete row, and issue repeated `executor->loadWeight()` calls with:

```cpp
deviceOffset + chunkOffset
chunkBytes
weight + fileByteOffset + chunkOffset
```

- [ ] **Step 2: Cover the final partial chunk**

Ensure the final call handles any remaining rows and that the total uploaded bytes equal `slice.sliceSize.nBytes`.

- [ ] **Step 3: Run focused tests**

Expected result: row-slice chunk offsets are contiguous and no chunk exceeds the cap except when a single row itself exceeds the cap.

### Task 4: Build and regression verification

**Files:**
- Modify: none beyond Tasks 1–3.

- [ ] **Step 1: Run syntax and focused tests**

```bash
git diff --check
bash -n EdgeVisor/scripts/jetson_freq_inject.sh
```

- [ ] **Step 2: Run existing semantic tests**

```bash
python3 tests/semantic/test_jetson_freq_injector.py
```

- [ ] **Step 3: Run host C++ checks**

```bash
make -C EdgeVisor -n DLLAMA_CUDA=1 CUDA_ARCHS=87 dllama
```

On a Jetson with CUDA installed, run the real build:

```bash
make -C EdgeVisor clean
make -C EdgeVisor -j$(nproc) DLLAMA_CUDA=1
```

- [ ] **Step 4: Commit the implementation**

```bash
git add EdgeVisor/src/llm.cpp EdgeVisor/src/nn/nn-network-local.cpp EdgeVisor/src/test/test_chunked_weight_loading.cpp
git commit -m "fix: chunk root and final-stage weight uploads"
```
