from pathlib import Path

root = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor")

def patch(path, fn):
    p = root / path
    s = p.read_text()
    ns = fn(s)
    if ns != s:
        p.write_text(ns)

def repl(s, old, new, label):
    if new in s:
        return s
    if old not in s:
        raise RuntimeError(f"{label} pattern not found")
    return s.replace(old, new, 1)

def patch_executor_hpp(s):
    s = repl(s,
        """    virtual void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) = 0;
    virtual void setPartitionPlan(const NnUnevenPartitionPlan * /*plan*/) {}
    virtual void refreshPointers() {}
};""",
        """    virtual void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) = 0;
    virtual void setPartitionPlan(const NnUnevenPartitionPlan * /*plan*/) {}
    virtual void refreshPointers() {}
    virtual bool exportLayerKvRow(NnUint /*layerIndex*/, NnUint /*position*/, NnUint /*kvDim*/, std::vector<float> & /*kRow*/, std::vector<float> & /*vRow*/) { return false; }
    virtual bool applyTransferredKvRow(NnUint /*layerIndex*/, NnUint /*position*/, const std::vector<float> & /*kRow*/, const std::vector<float> & /*vRow*/) { return false; }
};""",
        "executor hpp virtuals")
    return s

def patch_vulkan_hpp(s):
    s = repl(s,
        """    NnVulkanBuffer *resolvePipeByIndex(NnUint pipeIndex);
    NnVulkanBuffer *resolveBufferByIndex(NnUint bufferIndex);
    void setPartitionPlan(const NnUnevenPartitionPlan *plan);""",
        """    NnVulkanBuffer *resolvePipeByIndex(NnUint pipeIndex);
    NnVulkanBuffer *resolveBufferByIndex(NnUint bufferIndex);
    NnUint getBufferCount() const;
    const NnSize3D *getBufferSize(NnUint bufferIndex) const;
    void setPartitionPlan(const NnUnevenPartitionPlan *plan);""",
        "vulkan data header helpers")
    s = repl(s,
        """    void setPartitionPlan(const NnUnevenPartitionPlan *plan) override;
    void refreshPointers() override;
};""",
        """    void setPartitionPlan(const NnUnevenPartitionPlan *plan) override;
    void refreshPointers() override;
    bool exportLayerKvRow(NnUint layerIndex, NnUint position, NnUint kvDim, std::vector<float> &kRow, std::vector<float> &vRow) override;
    bool applyTransferredKvRow(NnUint layerIndex, NnUint position, const std::vector<float> &kRow, const std::vector<float> &vRow) override;
};""",
        "vulkan segment kv virtuals")
    return s

def patch_vulkan_cpp(s):
    s = repl(s,
        """        context->device.invalidateMappedMemoryRanges({ {deviceMemory, offset, (vk::DeviceSize)size} });
        std::memcpy(data, hostPointer, size);
""",
        """        context->device.invalidateMappedMemoryRanges({ {deviceMemory, offset, (vk::DeviceSize)size} });
        std::memcpy(data, &hostPointer[offset], size);
""",
        "vulkan host-visible read offset")
    s = repl(s,
        """NnVulkanBuffer *NnVulkanDeviceData::resolveBufferByIndex(NnUint bufferIndex) {
    assert(bufferIndex < nodeConfig->nBuffers);
    return buffers[bufferIndex].get();
}
""",
        """NnVulkanBuffer *NnVulkanDeviceData::resolveBufferByIndex(NnUint bufferIndex) {
    assert(bufferIndex < nodeConfig->nBuffers);
    return buffers[bufferIndex].get();
}

NnUint NnVulkanDeviceData::getBufferCount() const {
    return nodeConfig != nullptr ? nodeConfig->nBuffers : 0u;
}

const NnSize3D *NnVulkanDeviceData::getBufferSize(NnUint bufferIndex) const {
    if (nodeConfig == nullptr || bufferIndex >= nodeConfig->nBuffers) return nullptr;
    return &nodeConfig->buffers[bufferIndex].size;
}
""",
        "vulkan data buffer helpers")
    insert_after = """void NnVulkanDeviceSegment::refreshPointers() {
"""
    # Insert methods immediately before refreshPointers so they can use class internals.
    marker = """void NnVulkanDeviceSegment::refreshPointers() {
"""
    methods = r'''
bool NnVulkanDeviceSegment::exportLayerKvRow(NnUint layerIndex, NnUint position, NnUint kvDim, std::vector<float> &kRow, std::vector<float> &vRow) {
    kRow.assign(kvDim, 0.0f);
    vRow.assign(kvDim, 0.0f);
    if (data == nullptr || segmentConfig == nullptr || kvDim == 0u) return false;

    bool readAny = false;
    for (NnUint i = 0; i < segmentConfig->nOps; ++i) {
        const NnOpConfig &op = segmentConfig->ops[i];
        if (op.code != OP_MULTIHEAD_ATT || op.index != layerIndex) continue;
        const auto *cfg = (const NnMultiHeadAttOpConfig *)op.config;
        if (cfg == nullptr) continue;

        auto readOne = [&](NnUint bufIdx, std::vector<float> &dstRow, const char *tag) {
            if (bufIdx >= data->getBufferCount()) return;
            const NnSize3D *bSize = data->getBufferSize(bufIdx);
            if (bSize == nullptr || bSize->floatType != F_32 || bSize->x == 0u || bSize->y == 0u) return;
            if (position >= bSize->y) return;

            const NnUint srcStart = cfg->kvStart;
            const NnUint needLen = cfg->kvDim0;
            if (srcStart >= bSize->x || srcStart >= dstRow.size() || needLen == 0u) return;

            const NnUint srcAvail = bSize->x - srcStart;
            const NnUint dstAvail = (NnUint)dstRow.size() - srcStart;
            const NnUint readLen = std::min(needLen, std::min(srcAvail, dstAvail));
            if (readLen == 0u) return;

            context->device.waitIdle();
            NnVulkanBuffer *buffer = data->resolveBufferByIndex(bufIdx);
            const NnSize offset = ((NnSize)position * (NnSize)bSize->x + (NnSize)srcStart) * sizeof(float);
            const NnSize nBytes = (NnSize)readLen * sizeof(float);
            buffer->read((NnByte *)(dstRow.data() + srcStart), offset, nBytes);
            readAny = true;

            std::printf("🧩 [kv-export-gpu] node=%u seg=%u layer=%u pos=%u %sBuf=%u range=[%u,%u)\n",
                (unsigned)(data ? data->getNodeIndex() : 0u),
                (unsigned)segmentIndex,
                (unsigned)layerIndex,
                (unsigned)position,
                tag,
                (unsigned)bufIdx,
                (unsigned)srcStart,
                (unsigned)(srcStart + readLen));
            std::fflush(stdout);
        };

        readOne(cfg->keyCacheBufferIndex, kRow, "k");
        readOne(cfg->valueCacheBufferIndex, vRow, "v");
    }
    return readAny;
}

bool NnVulkanDeviceSegment::applyTransferredKvRow(NnUint layerIndex, NnUint position, const std::vector<float> &kRow, const std::vector<float> &vRow) {
    if (data == nullptr || segmentConfig == nullptr) return false;
    if (kRow.empty() || vRow.empty() || kRow.size() != vRow.size()) return false;

    bool wroteAny = false;
    for (NnUint i = 0; i < segmentConfig->nOps; ++i) {
        const NnOpConfig &op = segmentConfig->ops[i];
        if (op.code != OP_MULTIHEAD_ATT || op.index != layerIndex) continue;
        const auto *cfg = (const NnMultiHeadAttOpConfig *)op.config;
        if (cfg == nullptr) continue;

        auto writeOne = [&](NnUint bufIdx, const std::vector<float> &srcRow, const char *tag) {
            if (bufIdx >= data->getBufferCount()) return;
            const NnSize3D *bSize = data->getBufferSize(bufIdx);
            if (bSize == nullptr || bSize->floatType != F_32 || bSize->x == 0u || bSize->y == 0u) return;
            if (position >= bSize->y) return;

            const NnUint srcStart = cfg->kvStart;
            const NnUint needLen = cfg->kvDim0;
            if (srcStart >= srcRow.size() || srcStart >= bSize->x || needLen == 0u) return;

            const NnUint srcAvail = (NnUint)srcRow.size() - srcStart;
            const NnUint dstAvail = bSize->x - srcStart;
            const NnUint writeLen = std::min(needLen, std::min(srcAvail, dstAvail));
            if (writeLen == 0u) return;

            context->device.waitIdle();
            NnVulkanBuffer *buffer = data->resolveBufferByIndex(bufIdx);
            const NnSize offset = ((NnSize)position * (NnSize)bSize->x + (NnSize)srcStart) * sizeof(float);
            const NnSize nBytes = (NnSize)writeLen * sizeof(float);
            buffer->write((const NnByte *)(srcRow.data() + srcStart), offset, nBytes);
            wroteAny = true;

            std::printf("🧩 [kv-write-gpu] node=%u seg=%u layer=%u pos=%u %sBuf=%u range=[%u,%u)\n",
                (unsigned)(data ? data->getNodeIndex() : 0u),
                (unsigned)segmentIndex,
                (unsigned)layerIndex,
                (unsigned)position,
                tag,
                (unsigned)bufIdx,
                (unsigned)srcStart,
                (unsigned)(srcStart + writeLen));
            std::fflush(stdout);
        };

        writeOne(cfg->keyCacheBufferIndex, kRow, "k");
        writeOne(cfg->valueCacheBufferIndex, vRow, "v");
    }
    return wroteAny;
}

'''
    if methods.strip() not in s:
        if marker not in s:
            raise RuntimeError("refreshPointers marker not found")
        s = s.replace(marker, methods + marker, 1)
    return s

def patch_executor_cpp(s):
    # Add a GPU/general fallback for non-CPU segments in export/apply loops.
    s = repl(s,
        """        auto *cpuSeg = dynamic_cast<NnCpuDeviceSegment *>(baseSeg);
        if (cpuSeg == nullptr || cpuSeg->device == nullptr) continue;

        NnSegmentConfig *seg = &nodeConfig->segments[s];
""",
        """        auto *cpuSeg = dynamic_cast<NnCpuDeviceSegment *>(baseSeg);
        if (cpuSeg == nullptr || cpuSeg->device == nullptr) {
            if (baseSeg->exportLayerKvRow(layerIndex, position, kvDim, kRow, vRow)) {
                readAny = true;
            }
            continue;
        }

        NnSegmentConfig *seg = &nodeConfig->segments[s];
""",
        "executor export fallback")
    s = repl(s,
        """        auto *cpuSeg = dynamic_cast<NnCpuDeviceSegment *>(baseSeg);
        if (cpuSeg == nullptr || cpuSeg->device == nullptr) continue;

        NnSegmentConfig *seg = &nodeConfig->segments[s];
""",
        """        auto *cpuSeg = dynamic_cast<NnCpuDeviceSegment *>(baseSeg);
        if (cpuSeg == nullptr || cpuSeg->device == nullptr) {
            if (baseSeg->applyTransferredKvRow(layerIndex, position, kRow, vRow)) {
                wroteAny = true;
            }
            continue;
        }

        NnSegmentConfig *seg = &nodeConfig->segments[s];
""",
        "executor apply fallback")
    return s

patch("src/nn/nn-executor.hpp", patch_executor_hpp)
patch("src/nn/nn-vulkan.hpp", patch_vulkan_hpp)
patch("src/nn/nn-vulkan.cpp", patch_vulkan_cpp)
patch("src/nn/nn-executor.cpp", patch_executor_cpp)
print("added GPU KV row export/apply support")
