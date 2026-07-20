# Jetson root 节点 token embedding mmap 失败

## 现象

CUDA 后端 root 节点启动时失败：

```text
Mmap failed: path=/home/jetson/cc/models/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m offset=136 size=3111649280 mapping_size=3111649416 errno=12 (Cannot allocate memory)
```

## 已确认的原因

错误区间从 offset `136` 开始，大小为 `3111649280` 字节：

```text
151936 vocab × 5120 hidden_dim × 4 bytes(F32)
= 3111649280 bytes
```

因此该 mmap 对应的是完整的 Qwen3 14B F32 token embedding，而不是某个 Transformer layer。

`loadLlmNetWeightUneven()` 当前在 root 节点执行：

```cpp
mapWeightRange(fileOffset, embeddingBytes);
loadRootTokenEmbeddingQ80(loader, h, b);
```

这会先一次性 mmap 约 2.9 GiB 的 F32 embedding，再按 1024 行分块量化成 Q80。虽然量化上传本身是分块的，但源文件 mmap 不是分块的。

root 节点在此之前已经创建 CUDA context、pipe、buffer、layer weight buffer 和 CUDA runtime 状态。`--memory-limit-gib` 使用 `RLIMIT_AS`，所以该 2.9 GiB 映射需要与已有进程虚拟地址空间共同满足限制；失败 errno=12 表示 mmap 地址空间/映射分配失败。

## 与 worker last-stage OOM 的关系

这是另一条角色相关的内存峰值路径：

- root：一次性 mmap 完整 F32 token embedding；
- last-stage worker：一次性申请完整 final logits 的 CUDA pinned staging；
- 中间 worker：通常不会触发上述两个完整 vocab/embedding 路径。

因此错误可以跟随网络角色变化，而不跟随物理 Jetson 设备变化。

## 修复方向

将 root embedding 的加载改为与量化粒度一致的分块 mmap：

1. 每次只 mmap 固定数量的 embedding rows；
2. 量化并上传该 chunk；
3. 立即 `munmap` 该 chunk；
4. 继续下一个 chunk。

这样 root 的源 F32 映射峰值从约 2.9 GiB 降为单个 chunk，而不改变最终 Q80 embedding 的内容。

同类地，last-stage 的 final logits 上传也应采用分块 `loadWeight()`，避免单次 `cudaMallocHost()` 申请完整 logits slice。

## 当前状态

问题已确认，尚未实现 embedding 分块 mmap 和 final logits 分块上传修复。
