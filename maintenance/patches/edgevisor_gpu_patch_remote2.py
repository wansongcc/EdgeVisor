from pathlib import Path

root = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor")


def replace_once(path: Path, old: str, new: str) -> None:
    s = path.read_text()
    if new in s:
        return
    if old not in s:
        raise RuntimeError(f"pattern not found in {path}: {old[:100]!r}")
    path.write_text(s.replace(old, new, 1))


def insert_before_once(path: Path, needle: str, text: str) -> None:
    s = path.read_text()
    if text.strip() in s:
        return
    if needle not in s:
        raise RuntimeError(f"needle not found in {path}: {needle[:100]!r}")
    path.write_text(s.replace(needle, text + needle, 1))


# CPU segment declaration and implementation.
p = root / "src/nn/nn-cpu.hpp"
replace_once(
    p,
    """    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;

    // CPU-only: re-resolve PNTR_* pointers/sizes after partition plan updates.
    void refreshPointers();""",
    """    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
    void setPartitionPlan(const NnUnevenPartitionPlan *plan) override;

    // Re-resolve PNTR_* pointers/sizes after partition plan updates.
    void refreshPointers() override;""",
)

p = root / "src/nn/nn-cpu.cpp"
insert_before_once(
    p,
    "void NnCpuDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {",
    """void NnCpuDeviceSegment::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    if (device != nullptr) {
        device->setPartitionPlan(plan);
    }
}

""",
)

# Executor generic hooks.
p = root / "src/nn/nn-executor.cpp"
replace_once(
    p,
    """void NnExecutor::refreshPointers() {
    if (context.isAlive.load()) {
        throw std::runtime_error("Cannot refresh pointers while executor is running");
    }
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnDeviceSegment *segment = segments[segmentIndex].get();
        if (segment == nullptr)
            continue;
        if (auto *cpuSeg = dynamic_cast<NnCpuDeviceSegment *>(segment)) {
            cpuSeg->refreshPointers();
        }
    }
}

void NnExecutor::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    if (context.isAlive.load()) {
        throw std::runtime_error("Cannot update partition plan while executor is running");
    }
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnDeviceSegment *segment = segments[segmentIndex].get();
        if (segment == nullptr)
            continue;
        if (auto *cpuSeg = dynamic_cast<NnCpuDeviceSegment *>(segment)) {
            if (cpuSeg->device != nullptr) {
                cpuSeg->device->setPartitionPlan(plan);
            }
        }
    }
}
""",
    """void NnExecutor::refreshPointers() {
    if (context.isAlive.load()) {
        throw std::runtime_error("Cannot refresh pointers while executor is running");
    }
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnDeviceSegment *segment = segments[segmentIndex].get();
        if (segment == nullptr)
            continue;
        segment->refreshPointers();
    }
}

void NnExecutor::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    if (context.isAlive.load()) {
        throw std::runtime_error("Cannot update partition plan while executor is running");
    }
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnDeviceSegment *segment = segments[segmentIndex].get();
        if (segment == nullptr)
            continue;
        segment->setPartitionPlan(plan);
    }
}
""",
)

# Vulkan declarations.
p = root / "src/nn/nn-vulkan.hpp"
replace_once(
    p,
    "#include <vulkan/vulkan.hpp>\n#include <vector>",
    "#include <vulkan/vulkan.hpp>\n#include <vector>\n#include <atomic>",
)
replace_once(
    p,
    """    NnVulkanBuffer *resolvePipeByIndex(NnUint pipeIndex);
    NnVulkanBuffer *resolveBufferByIndex(NnUint bufferIndex);
};""",
    """    NnVulkanBuffer *resolvePipeByIndex(NnUint pipeIndex);
    NnVulkanBuffer *resolveBufferByIndex(NnUint bufferIndex);
    void setPartitionPlan(const NnUnevenPartitionPlan *plan);
    const NnUnevenPartitionPlan *getPartitionPlan() const;
    NnUint getNodeIndex() const;
};""",
)
replace_once(
    p,
    """    NnNetExecution *netExecution;
public:
    NnVulkanDeviceData data;
    NnVulkanDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);""",
    """    NnNetExecution *netExecution;
    const NnUnevenPartitionPlan *partitionPlan;
    std::atomic_uint planEpoch{0u};
    std::atomic_uint lastPlanCmdSeqEmitted{0u};
public:
    NnVulkanDeviceData data;
    NnVulkanDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan = nullptr);""",
)
replace_once(
    p,
    """    ~NnVulkanDevice() override;
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
};""",
    """    ~NnVulkanDevice() override;
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
    void setPartitionPlan(const NnUnevenPartitionPlan *plan);
    const NnUnevenPartitionPlan *getPartitionPlan() const { return partitionPlan; }
    unsigned int getPlanEpoch() const { return planEpoch.load(std::memory_order_acquire); }
    void setPlanEpoch(unsigned int e) { planEpoch.store(e, std::memory_order_release); }
    unsigned int getLastPlanCmdSeqEmitted() const { return lastPlanCmdSeqEmitted.load(std::memory_order_acquire); }
    void setLastPlanCmdSeqEmitted(unsigned int s) { lastPlanCmdSeqEmitted.store(s, std::memory_order_release); }
    NnUint getNodeIndex() const { return nodeConfig ? nodeConfig->nodeIndex : 0u; }
};""",
)
replace_once(
    p,
    """    NnNetExecution *netExecution;
    std::unique_ptr<NnVulkanDeviceSegmentData> segmentData;""",
    """    NnNetExecution *netExecution;
    NnVulkanDevice *ownerDevice;
    std::unique_ptr<NnVulkanDeviceSegmentData> segmentData;""",
)
replace_once(
    p,
    """    NnUint lastBatchSize;
public:
    NnVulkanDeviceSegment(NnVulkanContext *context, NnVulkanStagingCopier *copier, NnVulkanBufferFactory *bufferFactory, NnVulkanDeviceData *data, NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution);
    ~NnVulkanDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
};""",
    """    NnUint lastBatchSize;
    bool commandBufferDirty;
    std::atomic_uint planEpochReady{0u};
public:
    NnVulkanDeviceSegment(NnVulkanDevice *ownerDevice, NnVulkanContext *context, NnVulkanStagingCopier *copier, NnVulkanBufferFactory *bufferFactory, NnVulkanDeviceData *data, NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution);
    ~NnVulkanDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
    void setPartitionPlan(const NnUnevenPartitionPlan *plan) override;
    void refreshPointers() override;
};""",
)

# App passes initial plan into Vulkan device.
p = root / "src/app.cpp"
replace_once(
    p,
    "new NnVulkanDevice(args->gpuIndex, netConfig, nodeConfig, netExecution),",
    "new NnVulkanDevice(args->gpuIndex, netConfig, nodeConfig, netExecution, plan),",
)

print("gpu interface patch 2 applied")
