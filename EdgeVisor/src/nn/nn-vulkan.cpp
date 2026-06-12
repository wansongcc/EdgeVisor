#include "nn-vulkan.hpp"
#include "ablation.hpp"
#include "plan-command.hpp"
#include "llm.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#define DEBUG_VULKAN_BUFFERS false
#define DEBUG_VULKAN_TRACE false

#if DEBUG_VULKAN_TRACE
    #define VULKAN_TRACE(...) printf("VULKAN_TRACE: "); printf(__VA_ARGS__); printf("\n");
#else
    #define VULKAN_TRACE(...)
#endif

static bool hasPortabilityExtension() {
#ifdef __APPLE__
    const std::vector<vk::ExtensionProperties> extensionProperties = vk::enumerateInstanceExtensionProperties();
    for (const auto& extension : extensionProperties) {
        if (std::strcmp(extension.extensionName, "VK_KHR_portability_enumeration") == 0)
            return true;
    }
#endif
    return false;
}

static void logVulkanPlanApplyAblationEvent(
    NnUint stageIndex,
    NnUint fromNode,
    NnUint toNode,
    NnUint layerIndex,
    NnUint pos,
    NnUint cmd,
    bool success,
    const char *reason) {
    EdgeVisorAblationEvent ev;
    ev.eventId = "plan_command_apply";
    ev.triggerPos = pos;
    ev.triggerLayer = layerIndex;
    ev.affectedStage = stageIndex;
    ev.fromNode = fromNode;
    ev.toNode = toNode;
    ev.selectedPolicy = std::string("gpu_apply_cmd_") + std::to_string((unsigned)cmd);
    ev.bindingUpdateCount = 1u;
    ev.applySuccess = success;
    auto appendFallbackReason = [&](const char *fallbackReason) {
        if (fallbackReason == nullptr || fallbackReason[0] == '\0') return;
        if (!ev.fallbackReason.empty()) ev.fallbackReason += ";";
        ev.fallbackReason += fallbackReason;
    };
    appendFallbackReason(reason);
    const EdgeVisorAblationConfig &cfg = getEdgeVisorAblationConfig();
    if (cfg.pointerSwizzlingMode == PointerSwizzlingMode::OPERATOR_REBUILD) {
        appendFallbackReason("operator_rebuild_substitutes_lightweight_pointer_swizzling");
        ev.fallbackCount = 1u;
    } else if (cfg.pointerSwizzlingMode == PointerSwizzlingMode::WEIGHT_REMATERIALIZE) {
        appendFallbackReason("weight_rematerialize_substitutes_lightweight_pointer_swizzling");
        ev.materializedBytes = ev.bindingUpdateCount;
        ev.fallbackCount = 1u;
    }
    edgevisorAblationLogEvent(ev);
}

static bool hasValidationLayer() {
    const std::vector<vk::LayerProperties> layerProperties = vk::enumerateInstanceLayerProperties();
    for (const auto& layer : layerProperties) {
        if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0)
            return true;
    }
    return false;
}

static void assertDeviceExtensionSupport(const vk::PhysicalDevice &physicalDevice, const std::vector<const char *> &requiredExtensions) {
    auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    for (const auto& ext : requiredExtensions) {
        bool found = false;
        for (const auto& extension : availableExtensions) {
            if (std::strcmp(extension.extensionName, ext) == 0) {
                found = true;
                break;
            }
        }
        if (!found)
            throw std::runtime_error(std::string("Device extension ") + ext + " is not supported");
    }
}

#define MEMORY_TYPE_INDEX_NOT_FOUND ~0

static uint32_t findMemoryTypeIndex(const vk::PhysicalDevice *physicalDevice, vk::MemoryPropertyFlags expectedFlags) {
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice->getMemoryProperties();
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; index++) {
        auto flags = memoryProperties.memoryTypes[index].propertyFlags;
        if ((flags & expectedFlags) == expectedFlags) {
            return index;
        }
    }
    return MEMORY_TYPE_INDEX_NOT_FOUND;
}

NnVulkanContext::NnVulkanContext(const NnUint gpuIndex) {
    vk::InstanceCreateFlags createInstanceFlags(0);
    std::vector<const char*> instanceLayers = {};
    std::vector<const char*> instanceExtensions = {};
    std::vector<const char*> deviceExtension = {};

    if (hasValidationLayer()) {
        instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
    }
    if (hasPortabilityExtension()) {
        createInstanceFlags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        instanceExtensions.push_back("VK_KHR_portability_enumeration");
        deviceExtension.push_back("VK_KHR_portability_subset");
    }
    deviceExtension.push_back("VK_KHR_8bit_storage");
    deviceExtension.push_back("VK_KHR_16bit_storage");
    deviceExtension.push_back("VK_KHR_maintenance4");

    vk::ApplicationInfo appInfo {"Distributed Llama", 1, nullptr, 0, VK_API_VERSION_1_2 };
    vk::InstanceCreateInfo instanceCreateInfo(createInstanceFlags, &appInfo, instanceLayers, instanceExtensions);
    instance = vk::createInstance(instanceCreateInfo);

    auto physicalDevices = instance.enumeratePhysicalDevices();
    const NnSize nDevices = physicalDevices.size();
    if (gpuIndex >= nDevices)
        throw std::runtime_error("Invalid GPU index, found " + std::to_string(nDevices) + " GPUs");
    physicalDevice = physicalDevices[gpuIndex];
    assertDeviceExtensionSupport(physicalDevice, deviceExtension);

    vk::PhysicalDeviceProperties deviceProps = physicalDevice.getProperties();
    printf("🌋 Device: %s\n", (char*)deviceProps.deviceName);
    printf("🌋 DeviceApiVersion: %d.%d.%d\n", VK_VERSION_MAJOR(deviceProps.apiVersion), VK_VERSION_MINOR(deviceProps.apiVersion), VK_VERSION_PATCH(deviceProps.apiVersion));
    printf("🌋 MaxComputeSharedMemory: %d kB\n", deviceProps.limits.maxComputeSharedMemorySize / 1024);
    printf("🌋 NonCoherentAtomSize: %lu bytes\n", (NnSize)deviceProps.limits.nonCoherentAtomSize);

    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
    for (unsigned int h = 0; h < memoryProperties.memoryHeapCount; h++) {
        if (memoryProperties.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
            printf("🌋 Heap[%u]: %lu MB\n", h, ((NnSize)memoryProperties.memoryHeaps[h].size) / (1024 * 1024));
    }

    vk::PhysicalDeviceFeatures deviceFeatures = physicalDevice.getFeatures();
    VkPhysicalDeviceFeatures2 deviceFeatures2;
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.pNext = nullptr;
    deviceFeatures2.features = (VkPhysicalDeviceFeatures)deviceFeatures;

    VkPhysicalDeviceVulkan11Features vk11Features;
    vk11Features.pNext = nullptr;
    vk11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    deviceFeatures2.pNext = &vk11Features;

    VkPhysicalDeviceVulkan12Features vk12Features;
    vk12Features.pNext = nullptr;
    vk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11Features.pNext = &vk12Features;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures2);

    std::vector<vk::QueueFamilyProperties> queueFamilyProps = physicalDevice.getQueueFamilyProperties();
    auto propIt = std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(), [](const vk::QueueFamilyProperties& Prop) {
        return Prop.queueFlags & vk::QueueFlagBits::eCompute;
    });
    queueFamilyIndex = std::distance(queueFamilyProps.begin(), propIt);

    const float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &queuePriority);

    vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), deviceQueueCreateInfo);
    deviceCreateInfo.enabledExtensionCount = deviceExtension.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtension.data();
    deviceCreateInfo.setPNext(&deviceFeatures2);
    device = physicalDevice.createDevice(deviceCreateInfo);

    vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer), queueFamilyIndex);
    commandPool = device.createCommandPool(commandPoolCreateInfo);
    queue = device.getQueue(queueFamilyIndex, 0);
    nonCoherentAtomSize = deviceProps.limits.nonCoherentAtomSize;
    VULKAN_TRACE("Context created");
}

NnVulkanContext::~NnVulkanContext() {
    device.destroyCommandPool(commandPool);
    device.destroy();
    instance.destroy();
    VULKAN_TRACE("Context destroyed");
}

std::pair<vk::Buffer, vk::DeviceMemory> NnVulkanContext::createRawBuffer(const uint32_t memoryTypeIndex, const vk::DeviceSize bufferSize, const vk::BufferUsageFlags usageFlags) {
    vk::BufferCreateInfo bufferCreateInfo {
        vk::BufferCreateFlags(),
        bufferSize,
        usageFlags,
        vk::SharingMode::eExclusive,
        1,
        &queueFamilyIndex
    };
    vk::Buffer buffer = device.createBuffer(bufferCreateInfo);

    vk::MemoryRequirements memoryRequirements = device.getBufferMemoryRequirements(buffer);
    vk::MemoryAllocateInfo bufferMemoryAllocateInfo(memoryRequirements.size, memoryTypeIndex);
    vk::DeviceMemory bufferMemory = device.allocateMemory(bufferMemoryAllocateInfo);

    device.bindBufferMemory(buffer, bufferMemory, 0);
    return std::make_pair(buffer, bufferMemory);
}

NnVulkanStagingCopier::NnVulkanStagingCopier(NnVulkanContext *context) :
    context(context)
{
    allocatedSize = 0u;

    memoryTypeIndex = findMemoryTypeIndex(&context->physicalDevice, vk::MemoryPropertyFlagBits::eHostVisible);
    if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
        throw std::runtime_error("Cannot find host visible memory type");
}

NnVulkanStagingCopier::~NnVulkanStagingCopier() {
    tryRelease();
}

void NnVulkanStagingCopier::tryRelease() {
    if (allocatedSize > 0u) {
        context->device.unmapMemory(hostMemory);
        context->device.freeMemory(hostMemory);
        context->device.destroyBuffer(hostBuffer);
        allocatedSize = 0u;
    }
}

void NnVulkanStagingCopier::allocate(const NnSize size) {
    if (allocatedSize != size) {
        tryRelease();

        auto b = context->createRawBuffer(memoryTypeIndex, size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
        hostBuffer = b.first;
        hostMemory = b.second;
        hostPointer = context->device.mapMemory(hostMemory, 0, size);
        allocatedSize = size;
    }
}

void NnVulkanStagingCopier::copy(NnByte *data, const NnSize size, const NnVulkanStagingCopierDirection direction) {
    assert(size == allocatedSize);

    switch (direction) {
    case COPY_TO_DEVICE:
        std::memcpy(hostPointer, data, size);
        break;
    case COPY_FROM_DEVICE:
        std::memcpy(data, hostPointer, size);
        break;
    }
}

void NnVulkanStagingCopier::addCopyCommand(vk::CommandBuffer& commandBuffer, vk::Buffer& target, const NnSize offset, const NnSize size, const NnVulkanStagingCopierDirection direction) {
    assert(size == allocatedSize);

    VkBufferCopy copyRegion;
    copyRegion.size = size;
    switch (direction) {
    case COPY_TO_DEVICE:
        copyRegion.srcOffset = 0u;
        copyRegion.dstOffset = offset;
        vkCmdCopyBuffer(commandBuffer, hostBuffer, target, 1, &copyRegion);
        break;
    case COPY_FROM_DEVICE:
        copyRegion.srcOffset = offset;
        copyRegion.dstOffset = 0u;
        vkCmdCopyBuffer(commandBuffer, target, hostBuffer, 1, &copyRegion);
        break;
    }
}

void NnVulkanStagingCopier::executeCopyCommand(vk::Buffer& target, const NnSize offset, const NnSize size, const NnVulkanStagingCopierDirection direction) {
    vk::CommandBufferAllocateInfo allocInfo(context->commandPool, vk::CommandBufferLevel::ePrimary, 1);
    const std::vector<vk::CommandBuffer> cmdBuffers = context->device.allocateCommandBuffers(allocInfo);
    vk::CommandBuffer commandBuffer = cmdBuffers.front();
    commandBuffer.begin({ vk::CommandBufferUsageFlags{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit } });
    addCopyCommand(commandBuffer, target, offset, size, direction);
    commandBuffer.end();

    vk::Fence fence = context->device.createFence(vk::FenceCreateInfo());
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
    context->queue.submit({ submitInfo }, fence);
    assert(context->device.waitForFences({ fence }, true, uint64_t(-1)) == vk::Result::eSuccess);

    context->device.destroyFence(fence);
    context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);
}

NnVulkanBuffer::NnVulkanBuffer(NnVulkanContext *context, NnVulkanStagingCopier *copier, const char *name, const NnSize bufferSize, const bool isSliceable, vk::BufferUsageFlags usageFlags, bool fastAccess) :
    context(context),
    copier(copier),
    name(name),
    bufferSize(bufferSize),
    isSliceable(isSliceable),
    usageFlags(usageFlags)
{
    this->hostPointer = nullptr;

    isHostVisible = false;

    VULKAN_TRACE("Creating buffer of size %zu (fastAccess=%d)", bufferSize, fastAccess);

    uint32_t memoryTypeIndex = MEMORY_TYPE_INDEX_NOT_FOUND;
    if (fastAccess) {
        memoryTypeIndex = findMemoryTypeIndex(
            &context->physicalDevice,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent |
            vk::MemoryPropertyFlagBits::eHostCached
        );
        if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
            memoryTypeIndex = findMemoryTypeIndex(
                &context->physicalDevice,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent
            );
        if (memoryTypeIndex != MEMORY_TYPE_INDEX_NOT_FOUND)
            isHostVisible = true;
    }
    if (!isHostVisible) {
        memoryTypeIndex = findMemoryTypeIndex(
            &context->physicalDevice,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
        if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
            throw std::runtime_error("Cannot find host visible memory type");
    }

    auto b = context->createRawBuffer(memoryTypeIndex, bufferSize, usageFlags | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    deviceBuffer = b.first;
    deviceMemory = b.second;
    if (isHostVisible)
        hostPointer = (NnByte *)context->device.mapMemory(deviceMemory, 0, bufferSize);
    VULKAN_TRACE("Created buffer of size %zu (fastAccess=%d)", bufferSize, fastAccess);
}

NnVulkanBuffer::~NnVulkanBuffer() {
    if (hostPointer != nullptr)
        context->device.unmapMemory(deviceMemory);
    context->device.freeMemory(deviceMemory);
    context->device.destroyBuffer(deviceBuffer);
    VULKAN_TRACE("Destroyed buffer of size %zu", bufferSize);
}

void NnVulkanBuffer::write(const NnByte *data) {
    write(data, 0u, bufferSize);
}

void NnVulkanBuffer::write(const NnByte *data, const NnSize offset, const NnSize size) {
    assert(offset + size <= bufferSize);

    if (isHostVisible && hostPointer != nullptr) {
        std::memcpy(&hostPointer[offset], data, size);
        context->device.flushMappedMemoryRanges({ { deviceMemory, offset, (vk::DeviceSize)size } });
        VULKAN_TRACE("Wrote %zu bytes to host visible buffer", size);
    } else {
        copier->allocate(size);
        copier->copy((NnByte *)data, size, COPY_TO_DEVICE);
        copier->executeCopyCommand(deviceBuffer, offset, size, COPY_TO_DEVICE);
        VULKAN_TRACE("Wrote %zu bytes to buffer", size);
    }
}

void NnVulkanBuffer::read(NnByte *data) {
    read(data, 0u, bufferSize);
}

void NnVulkanBuffer::read(NnByte *data, const NnSize offset, const NnSize size) {
    assert(offset + size <= bufferSize);

    if (isHostVisible && hostPointer != nullptr) {
        context->device.invalidateMappedMemoryRanges({ {deviceMemory, offset, (vk::DeviceSize)size} });
        std::memcpy(data, &hostPointer[offset], size);

        VULKAN_TRACE("Read %zu bytes from host visible buffer", size);
    } else {
        copier->allocate(size);
        copier->executeCopyCommand(deviceBuffer, offset, size, COPY_FROM_DEVICE);
        copier->copy(data, size, COPY_FROM_DEVICE);

        VULKAN_TRACE("Read %zu bytes from buffer", size);
    }
}

NnSize NnVulkanBuffer::calcSliceSize(const NnSize nominator, const NnSize denominator) {
    if (!isSliceable)
        return bufferSize;

    assert(bufferSize % denominator == 0);

    NnSize size = (bufferSize / denominator) * nominator;
    if (context->nonCoherentAtomSize != 0) {
        // TODO: this alignment is not needed for coherent memory
        size += context->nonCoherentAtomSize - (size % context->nonCoherentAtomSize);
        size = std::min(size, bufferSize);
    }
    return size;
}

NnVulkanBufferFactory::NnVulkanBufferFactory(NnVulkanContext *context, NnVulkanStagingCopier *copier) :
    context(context),
    copier(copier)
{}

std::unique_ptr<NnVulkanBuffer> NnVulkanBufferFactory::create(const char *name, const NnSize bufferSize, const bool isSliceable, vk::BufferUsageFlags usageFlags, bool fastAccess) {
    return std::unique_ptr<NnVulkanBuffer>(new NnVulkanBuffer(context, copier, name, bufferSize, isSliceable, usageFlags, fastAccess));
}

static void collectOpConfigs(std::vector<NnByte *> &out, NnNodeConfig *nodeConfig, NnOpCode opCode) {
    if (nodeConfig == nullptr) return;
    for (NnUint i = 0; i < nodeConfig->nSegments; i++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[i];
        for (NnUint j = 0; j < segmentConfig->nOps; j++) {
            if (segmentConfig->ops[j].code == opCode)
                out.push_back(segmentConfig->ops[j].config);
        }
    }
}


static bool isVulkanControlOp(NnOpCode code) {
    return code == OP_PLAN_BARRIER || code == OP_PLAN_APPLY;
}

static NnUint getSplitTotalVulkan(const NnDimSplit *split, NnUint nNodes) {
    if (split == nullptr || split->lengths == nullptr) return 0u;
    NnUint total = 0u;
    for (NnUint i = 0; i < nNodes; ++i) total += split->lengths[i];
    return total;
}

static const NnStageConfig *findStageForNodeVulkan(const NnUnevenPartitionPlan *plan, NnUint nodeIndex) {
    if (plan == nullptr || plan->stages == nullptr) return nullptr;
    for (NnUint s = 0; s < plan->nStages; ++s) {
        const NnStageConfig *st = &plan->stages[s];
        for (NnUint i = 0; i < st->nNodes; ++i) {
            if (st->nodeIndices[i] == nodeIndex) return st;
        }
    }
    return nullptr;
}

static NnUint findStageRankVulkan(const NnStageConfig *stage, NnUint nodeIndex) {
    if (stage == nullptr) return nodeIndex;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        if (stage->nodeIndices[i] == nodeIndex) return i;
    }
    return nodeIndex;
}

static NnUint getSplitTotalForStageVulkan(const NnDimSplit *split, const NnStageConfig *stage) {
    if (split == nullptr || split->lengths == nullptr || stage == nullptr) return 0u;
    NnUint total = 0u;
    for (NnUint i = 0; i < stage->nNodes; ++i) total += split->lengths[stage->nodeIndices[i]];
    return total;
}

static bool hasHeadDeltaForStageVulkan(const std::vector<int> &deltaHeadOrKv, const NnStageConfig *stage) {
    if (stage == nullptr) return false;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        const NnUint n = stage->nodeIndices[i];
        if (n < deltaHeadOrKv.size() && deltaHeadOrKv[n] != 0) return true;
    }
    return false;
}

static bool validateKvShadowCoverageVulkan(
    const NnUnevenPartitionPlan *plan,
    const NnStageConfig *stage,
    const std::vector<int> &deltaHeadOrKv,
    char *reason,
    size_t reasonSize
) {
    auto setReason = [&](const char *value) {
        if (reason != nullptr && reasonSize != 0u) {
            std::snprintf(reason, reasonSize, "%s", value != nullptr ? value : "kv_shadow_invalid");
        }
    };
    if (!getEnableStageFullWeights()) {
        setReason("kv_shadow_requires_stage_full_weights");
        return false;
    }
    if (!getEnableKvRedundancyDuringMigration()) {
        setReason("kv_shadow_redundancy_disabled");
        return false;
    }
    if (plan == nullptr || stage == nullptr ||
        plan->kvHeadSplit.starts == nullptr || plan->kvHeadSplit.lengths == nullptr ||
        plan->kvHeadComputeSplit.starts == nullptr || plan->kvHeadComputeSplit.lengths == nullptr) {
        setReason("kv_shadow_split_missing");
        return false;
    }

    NnUint newStart = 0u;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        const NnUint n = stage->nodeIndices[i];
        if (n >= deltaHeadOrKv.size()) {
            setReason("kv_shadow_delta_missing");
            return false;
        }
        const int newLenSigned = (int)plan->kvHeadSplit.lengths[n] + deltaHeadOrKv[n];
        if (newLenSigned < 0) {
            setReason("kv_shadow_underflow");
            return false;
        }

        const NnUint newLen = (NnUint)newLenSigned;
        const NnUint newEnd = newStart + newLen;
        const NnUint shadowStart = plan->kvHeadComputeSplit.starts[n];
        const NnUint shadowEnd = shadowStart + plan->kvHeadComputeSplit.lengths[n];
        if (newStart < shadowStart || newEnd > shadowEnd) {
            setReason("kv_shadow_coverage_miss");
            return false;
        }
        newStart = newEnd;
    }
    setReason("");
    return true;
}

static bool resolveUnevenSliceVulkan(
    NnNetConfig *netConfig,
    NnNodeConfig *nodeConfig,
    NnPointerConfig *pointerConfig,
    const NnSize3D &sourceSize,
    NnUint *outOffset,
    NnUint *outLength
) {
    if (outOffset == nullptr || outLength == nullptr) return false;
    if (pointerConfig->type != PNTR_BATCHED_SLICE) return false;

    const NnUnevenPartitionPlan *plan = (nodeConfig != nullptr) ? nodeConfig->partitionPlan : nullptr;
    const NnUint nodeIdx = (nodeConfig != nullptr) ? nodeConfig->nodeIndex : 0u;
    bool stackedByNode = (pointerConfig->sliceTag == NN_SLICE_STACKED_BY_NODE);
    bool splitFound = false;
    NnUint myOffset = 0u;
    NnUint myLength = 0u;

    const NnStageConfig *myStage = (plan != nullptr && plan->nStages > 0u)
        ? findStageForNodeVulkan(plan, nodeIdx)
        : nullptr;

    if (plan != nullptr && netConfig != nullptr && netConfig->nNodes == plan->nNodes) {
        const NnUint totalDim = sourceSize.x;

        if (!stackedByNode && pointerConfig->source == SRC_PIPE && myStage != nullptr) {
            const NnUint dimTotal = getSplitTotalForStageVulkan(&plan->dimSplit, myStage);
            if (dimTotal > 0u && totalDim == dimTotal * netConfig->nNodes) {
                stackedByNode = true;
            }
        }

        auto tryApplySplit = [&](const NnDimSplit &split, bool allowMultiplier) -> bool {
            if (stackedByNode || split.starts == nullptr || split.lengths == nullptr) return false;
            const NnUint splitTotal = (myStage != nullptr)
                ? getSplitTotalForStageVulkan(&split, myStage)
                : getSplitTotalVulkan(&split, plan->nNodes);
            if (splitTotal == 0u || totalDim % splitTotal != 0u) return false;
            const NnUint multiplier = totalDim / splitTotal;
            if (!allowMultiplier && multiplier != 1u) return false;
            myOffset = split.starts[nodeIdx] * multiplier;
            myLength = split.lengths[nodeIdx] * multiplier;
            if (sourceSize.floatType == F_Q80) {
                if ((myOffset % Q80_BLOCK_SIZE) != 0u || (myLength % Q80_BLOCK_SIZE) != 0u) {
                    return false;
                }
            }
            return true;
        };

        if (pointerConfig->sliceTag != NN_SLICE_AUTO) {
            switch (pointerConfig->sliceTag) {
                case NN_SLICE_VOCAB:
                    splitFound = tryApplySplit(plan->vocabSplit, false);
                    break;
                case NN_SLICE_FFN:
                    splitFound = tryApplySplit(plan->ffnSplit, false);
                    break;
                case NN_SLICE_DIM:
                    splitFound = tryApplySplit(plan->dimSplit, false);
                    break;
                case NN_SLICE_HEAD:
                    splitFound = tryApplySplit(plan->headSplit, true);
                    break;
                case NN_SLICE_KV_HEAD:
                    splitFound = tryApplySplit(plan->kvHeadSplit, true);
                    break;
                case NN_SLICE_STACKED_BY_NODE:
                case NN_SLICE_AUTO:
                default:
                    break;
            }
        } else {
            splitFound = tryApplySplit(plan->vocabSplit, false);
            if (!splitFound) splitFound = tryApplySplit(plan->ffnSplit, false);
            if (!splitFound) splitFound = tryApplySplit(plan->dimSplit, false);
            if (!splitFound && sourceSize.floatType != F_Q80) splitFound = tryApplySplit(plan->headSplit, true);
            if (!splitFound && sourceSize.floatType != F_Q80) splitFound = tryApplySplit(plan->kvHeadSplit, true);
        }
    }

    if (!splitFound) {
        if (stackedByNode && netConfig != nullptr && netConfig->nNodes != 0u) {
            myLength = sourceSize.x / netConfig->nNodes;
            myOffset = myLength * nodeIdx;
        } else {
            const NnUint nSplitNodes = (myStage != nullptr) ? myStage->nNodes : (netConfig ? netConfig->nNodes : 1u);
            const NnUint rank = (myStage != nullptr) ? findStageRankVulkan(myStage, nodeIdx) : nodeIdx;
            myLength = (nSplitNodes != 0u) ? sourceSize.x / nSplitNodes : sourceSize.x;
            myOffset = myLength * rank;
        }
    }

    if (myOffset > sourceSize.x) {
        myOffset = 0u;
        myLength = 0u;
    }
    *outOffset = myOffset;
    *outLength = myLength;
    return true;
}

static bool handleVulkanPlanBarrier(
    NnVulkanDevice *device,
    NnNetExecution *netExecution,
    NnSegmentConfig *segmentConfig,
    NnUint opIndex,
    NnUint batchSize
) {
    if (device == nullptr || netExecution == nullptr || segmentConfig == nullptr) return true;
    NnOpConfig *op = &segmentConfig->ops[opIndex];
    if (op->input.source != SRC_PIPE || op->output.source != SRC_PIPE) return true;

    const float *posPipe = (const float *)netExecution->pipes[op->input.pointerIndex];
    float *planPipe = (float *)netExecution->pipes[op->output.pointerIndex];
    const NnUint layerIndex = op->index;
    const NnUint pos = (posPipe != nullptr && posPipe[0] >= 0.0f) ? (NnUint)posPipe[0] : 0u;

    const NnUnevenPartitionPlan *planConst = device->getPartitionPlan();
    const NnUint myNode = device->getNodeIndex();
    const NnStageConfig *myStage = (planConst != nullptr && planConst->nStages > 0u)
        ? findStageForNodeVulkan(planConst, myNode)
        : nullptr;

    const PlanCommandSnapshot snap = planCommandCache().load();
    const PlanCommand &pc = snap.cmd;
    const bool hasPlanPayload =
        (pc.cmdKind == PLAN_CMD_KIND_HEAD ||
         pc.cmdKind == PLAN_CMD_KIND_FFN ||
         pc.cmdKind == PLAN_CMD_KIND_BOTH ||
         (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves != 0u));
    const bool hasCmd =
        (pc.magic == DLLAMA_PLAN_CMD_MAGIC) &&
        (pc.version == DLLAMA_PLAN_CMD_VERSION_V1 || pc.version == DLLAMA_PLAN_CMD_VERSION_V2) &&
        (pc.mode == PLAN_CMD_MODE_EXACT || pc.mode == PLAN_CMD_MODE_NEXT_BARRIER) &&
        hasPlanPayload;

    const bool stageOk = (pc.stageIndex == 0xFFFFFFFFu) || (myStage != nullptr && myStage->stageIndex == pc.stageIndex);
    const bool isStageRoot = stageOk && (myStage != nullptr) && (myStage->rootNodeIndex == myNode);
    const unsigned int curEpoch = device->getPlanEpoch();
    unsigned int emitEpoch = curEpoch;
    unsigned int cmd = 0u;
    unsigned int headMove = 0u;
    unsigned int ffnMove = 0u;
    unsigned int fromNode = 0u;
    unsigned int toNode = 0u;

    bool trigger = false;
    if (hasCmd && isStageRoot) {
        const unsigned int lastEmitted = device->getLastPlanCmdSeqEmitted();
        if (!(pc.seq != 0u && pc.seq <= lastEmitted)) {
            if (pc.mode == PLAN_CMD_MODE_EXACT) {
                trigger = (layerIndex == pc.triggerLayer) && (pos == pc.triggerPos);
            } else {
                trigger = true;
            }
        }
    }

    if (trigger) {
        emitEpoch = curEpoch + 1u;
        if (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves != 0u) {
            cmd = 4u;
            fromNode = pc.seq;
            printf("🧭 [plan][emit][gpu] node=%u stage=%u layer=%u pos=%u epoch=%u kind=cmdlist seq=%u moves=%u\n",
                (unsigned)myNode,
                (unsigned)(myStage != nullptr ? myStage->stageIndex : 0u),
                (unsigned)layerIndex,
                (unsigned)pos,
                (unsigned)emitEpoch,
                (unsigned)pc.seq,
                (unsigned)pc.nMoves);
        } else {
            cmd = pc.cmdKind;
            headMove = pc.nHeadsToMove;
            ffnMove = pc.nFfnToMove;
            fromNode = pc.fromNodeIndex;
            toNode = pc.toNodeIndex;
            printf("🧭 [plan][emit][gpu] node=%u stage=%u layer=%u pos=%u epoch=%u kind=%u headMove=%u ffnMove=%u from=%u to=%u\n",
                (unsigned)myNode,
                (unsigned)(myStage != nullptr ? myStage->stageIndex : 0u),
                (unsigned)layerIndex,
                (unsigned)pos,
                (unsigned)emitEpoch,
                (unsigned)cmd,
                (unsigned)headMove,
                (unsigned)ffnMove,
                (unsigned)fromNode,
                (unsigned)toNode);
        }
        device->setLastPlanCmdSeqEmitted(pc.seq);
        std::fflush(stdout);
    }

    for (NnUint b = 0; b < batchSize; ++b) {
        float *out = planPipe + b * 8u;
        out[0] = (float)emitEpoch;
        out[1] = (float)cmd;
        out[2] = (float)fromNode;
        out[3] = (float)toNode;
        out[4] = (float)headMove;
        out[5] = (float)layerIndex;
        out[6] = (float)pos;
        out[7] = (float)ffnMove;
    }
    return true;
}

static bool handleVulkanPlanApply(
    NnVulkanDevice *device,
    NnNetExecution *netExecution,
    NnSegmentConfig *segmentConfig,
    NnUint opIndex
) {
    if (device == nullptr || netExecution == nullptr || segmentConfig == nullptr) return true;
    NnOpConfig *op = &segmentConfig->ops[opIndex];
    if (op->input.source != SRC_PIPE) return true;

    const float *in0 = (const float *)netExecution->pipes[op->input.pointerIndex];
    if (in0 == nullptr) return true;
    const unsigned int curEpoch = device->getPlanEpoch();
    const unsigned int msgEpoch = (in0[0] >= 0.0f) ? (unsigned int)in0[0] : 0u;
    const unsigned int cmd = (in0[1] >= 0.0f) ? (unsigned int)in0[1] : 0u;
    const unsigned int fromNode = (in0[2] >= 0.0f) ? (unsigned int)in0[2] : 0u;
    const unsigned int toNode = (in0[3] >= 0.0f) ? (unsigned int)in0[3] : 0u;
    const unsigned int headMove = (in0[4] >= 0.0f) ? (unsigned int)in0[4] : 0u;
    const unsigned int layerIndex = (in0[5] >= 0.0f) ? (unsigned int)in0[5] : 0u;
    const unsigned int pos = (in0[6] >= 0.0f) ? (unsigned int)in0[6] : 0u;
    const unsigned int ffnMove = (in0[7] >= 0.0f) ? (unsigned int)in0[7] : 0u;

    if (!(cmd == 1u || cmd == 2u || cmd == 3u || cmd == 4u) || msgEpoch <= curEpoch) return true;

    auto *plan = const_cast<NnUnevenPartitionPlan *>(device->getPartitionPlan());
    if (plan == nullptr || plan->nStages == 0u) return true;
    const auto *cfg = (const NnPlanApplyOpCodeConfig *)op->config;
    const NnUint onlyStage = cfg != nullptr ? cfg->onlyStageIndex : 0u;
    if (onlyStage >= plan->nStages) return true;
    const NnUint myNode = device->getNodeIndex();
    const NnStageConfig *myStage = findStageForNodeVulkan(plan, myNode);
    if (myStage == nullptr || myStage->stageIndex != onlyStage) return true;
    const NnStageConfig *st = &plan->stages[onlyStage];

    auto inStage = [&](NnUint node) -> bool {
        for (NnUint i = 0; i < st->nNodes; ++i) if (st->nodeIndices[i] == node) return true;
        return false;
    };
    auto stageRank = [&](NnUint node) -> int {
        for (NnUint i = 0; i < st->nNodes; ++i) if (st->nodeIndices[i] == node) return (int)i;
        return -1;
    };
    auto adjacent = [&](NnUint a, NnUint b) -> bool {
        const int ra = stageRank(a), rb = stageRank(b);
        if (ra < 0 || rb < 0) return false;
        const int d = (ra >= rb) ? (ra - rb) : (rb - ra);
        return d == 1;
    };
    auto recomputeStarts = [&](NnDimSplit &split) {
        NnUint run = 0u;
        for (NnUint i = 0; i < st->nNodes; ++i) {
            const NnUint n = st->nodeIndices[i];
            split.starts[n] = run;
            run += split.lengths[n];
        }
    };

    NnUint stageQHeadsTotal = 0u, stageKvHeadsTotal = 0u;
    if (plan->kvHeadSplit.lengths != nullptr && plan->headSplit.lengths != nullptr) {
        for (NnUint i = 0; i < st->nNodes; ++i) {
            const NnUint n = st->nodeIndices[i];
            stageQHeadsTotal += plan->headSplit.lengths[n];
            stageKvHeadsTotal += plan->kvHeadSplit.lengths[n];
        }
    }
    NnUint gqaGroupSize = 1u;
    if (stageKvHeadsTotal != 0u && stageQHeadsTotal != 0u && (stageQHeadsTotal % stageKvHeadsTotal) == 0u) {
        gqaGroupSize = stageQHeadsTotal / stageKvHeadsTotal;
        if (gqaGroupSize == 0u) gqaGroupSize = 1u;
    }

    std::vector<int> deltaHeadOrKv(plan->nNodes, 0);
    std::vector<int> deltaFfn(plan->nNodes, 0);
    bool reject = false;

    auto addMove = [&](NnUint f, NnUint t, NnUint h, NnUint ffn) {
        if (!inStage(f) || !inStage(t) || f == t || !adjacent(f, t)) { reject = true; return; }
        if (h != 0u) {
            if (gqaGroupSize > 1u) {
                if (!getEnableKvRedundancyDuringMigration() || h > NN_KV_REDUNDANCY_PAD_HEADS) { reject = true; return; }
            }
            deltaHeadOrKv[f] -= (int)h;
            deltaHeadOrKv[t] += (int)h;
        }
        if (ffn != 0u) {
            deltaFfn[f] -= (int)ffn;
            deltaFfn[t] += (int)ffn;
        }
    };

    if (cmd == 4u) {
        const unsigned int wantSeq = fromNode;
        const PlanCommandSnapshot snap2 = planCommandCache().load();
        const PlanCommand &pc2 = snap2.cmd;
        if (!isValidPlanCommandHeader(pc2) || pc2.seq != wantSeq || !planCommandHasMoveList(pc2)) {
            printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u layer=%u pos=%u cmdlist missing/mismatch\n",
                (unsigned)myNode, (unsigned)onlyStage, (unsigned)msgEpoch, (unsigned)layerIndex, (unsigned)pos);
            std::fflush(stdout);
            logVulkanPlanApplyAblationEvent(onlyStage, fromNode, toNode, layerIndex, pos, cmd, false, "cmdlist_missing_or_mismatch");
            return true;
        }
        const uint32_t maxMovesAllowed = std::min<uint32_t>(DLLAMA_PLAN_CMD_MAX_MOVES, (uint32_t)(2u * st->nNodes));
        if (pc2.nMoves > maxMovesAllowed) {
            printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u reject: nMoves=%u > maxAllowed=%u\n",
                (unsigned)myNode, (unsigned)onlyStage, (unsigned)msgEpoch, (unsigned)pc2.nMoves, (unsigned)maxMovesAllowed);
            std::fflush(stdout);
            logVulkanPlanApplyAblationEvent(onlyStage, fromNode, toNode, layerIndex, pos, cmd, false, "too_many_moves");
            return true;
        }
        for (uint32_t i = 0; i < pc2.nMoves; ++i) {
            addMove(pc2.moves[i].fromNodeIndex, pc2.moves[i].toNodeIndex, pc2.moves[i].headMove, pc2.moves[i].ffnMove);
            if (reject) break;
        }
    } else {
        addMove(fromNode, toNode, (cmd == 1u || cmd == 3u) ? headMove : 0u, (cmd == 2u || cmd == 3u) ? ffnMove : 0u);
    }

    if (reject) {
        printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u layer=%u pos=%u reject: bad move\n",
            (unsigned)myNode, (unsigned)onlyStage, (unsigned)msgEpoch, (unsigned)layerIndex, (unsigned)pos);
        std::fflush(stdout);
        logVulkanPlanApplyAblationEvent(onlyStage, fromNode, toNode, layerIndex, pos, cmd, false, "bad_move");
        return true;
    }

    if (gqaGroupSize > 1u && hasHeadDeltaForStageVulkan(deltaHeadOrKv, st)) {
        char reason[96] = {0};
        if (!validateKvShadowCoverageVulkan(plan, st, deltaHeadOrKv, reason, sizeof(reason))) {
            printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u layer=%u pos=%u reject: %s\n",
                (unsigned)myNode,
                (unsigned)onlyStage,
                (unsigned)msgEpoch,
                (unsigned)layerIndex,
                (unsigned)pos,
                reason);
            std::fflush(stdout);
            logVulkanPlanApplyAblationEvent(onlyStage, fromNode, toNode, layerIndex, pos, cmd, false, reason);
            return true;
        }
    }

    bool changed = false;
    if (gqaGroupSize > 1u && plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
        for (NnUint i = 0; i < st->nNodes; ++i) {
            const NnUint n = st->nodeIndices[i];
            const int newLen = (int)plan->kvHeadSplit.lengths[n] + deltaHeadOrKv[n];
            if (newLen < 0) reject = true;
        }
        if (!reject) {
            for (NnUint i = 0; i < st->nNodes; ++i) {
                const NnUint n = st->nodeIndices[i];
                if (deltaHeadOrKv[n] != 0) {
                    plan->kvHeadSplit.lengths[n] = (NnUint)((int)plan->kvHeadSplit.lengths[n] + deltaHeadOrKv[n]);
                    changed = true;
                }
            }
            if (changed) {
                recomputeStarts(plan->kvHeadSplit);
                for (NnUint i = 0; i < st->nNodes; ++i) {
                    const NnUint n = st->nodeIndices[i];
                    plan->headSplit.starts[n] = plan->kvHeadSplit.starts[n] * gqaGroupSize;
                    plan->headSplit.lengths[n] = plan->kvHeadSplit.lengths[n] * gqaGroupSize;
                }
            }
        }
    } else if (plan->headSplit.starts && plan->headSplit.lengths) {
        for (NnUint i = 0; i < st->nNodes; ++i) {
            const NnUint n = st->nodeIndices[i];
            const int newLen = (int)plan->headSplit.lengths[n] + deltaHeadOrKv[n];
            if (newLen < 0) reject = true;
        }
        if (!reject) {
            for (NnUint i = 0; i < st->nNodes; ++i) {
                const NnUint n = st->nodeIndices[i];
                if (deltaHeadOrKv[n] != 0) {
                    plan->headSplit.lengths[n] = (NnUint)((int)plan->headSplit.lengths[n] + deltaHeadOrKv[n]);
                    changed = true;
                }
            }
            if (changed) recomputeStarts(plan->headSplit);
        }
    }

    bool ffnChanged = false;
    if (!reject && plan->ffnSplit.starts && plan->ffnSplit.lengths) {
        for (NnUint i = 0; i < st->nNodes; ++i) {
            const NnUint n = st->nodeIndices[i];
            const int newLen = (int)plan->ffnSplit.lengths[n] + deltaFfn[n];
            if (newLen < 0) reject = true;
        }
        if (!reject) {
            for (NnUint i = 0; i < st->nNodes; ++i) {
                const NnUint n = st->nodeIndices[i];
                if (deltaFfn[n] != 0) {
                    plan->ffnSplit.lengths[n] = (NnUint)((int)plan->ffnSplit.lengths[n] + deltaFfn[n]);
                    ffnChanged = true;
                }
            }
            if (ffnChanged) {
                recomputeStarts(plan->ffnSplit);
                changed = true;
            }
        }
    }

    if (reject) {
        printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u layer=%u pos=%u reject: split underflow\n",
            (unsigned)myNode, (unsigned)onlyStage, (unsigned)msgEpoch, (unsigned)layerIndex, (unsigned)pos);
        std::fflush(stdout);
        logVulkanPlanApplyAblationEvent(onlyStage, fromNode, toNode, layerIndex, pos, cmd, false, "split_underflow");
        return true;
    }

    if (changed) {
        printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u layer=%u pos=%u cmd=%u kvHeads=%u qHeads=%u ffnDim=%u\n",
            (unsigned)myNode,
            (unsigned)onlyStage,
            (unsigned)msgEpoch,
            (unsigned)layerIndex,
            (unsigned)pos,
            (unsigned)cmd,
            (unsigned)(plan->kvHeadSplit.lengths ? plan->kvHeadSplit.lengths[myNode] : 0u),
            (unsigned)(plan->headSplit.lengths ? plan->headSplit.lengths[myNode] : 0u),
            (unsigned)(plan->ffnSplit.lengths ? plan->ffnSplit.lengths[myNode] : 0u));
        std::fflush(stdout);
        logVulkanPlanApplyAblationEvent(onlyStage, fromNode, toNode, layerIndex, pos, cmd, true, "");
        device->setPlanEpoch(msgEpoch);
    }
    return true;
}

NnVulkanDeviceData::NnVulkanDeviceData(NnVulkanBufferFactory *bufferFactory, NnNetConfig *netConfig, NnNodeConfig *nodeConfig) :
    netConfig(netConfig),
    nodeConfig(nodeConfig),
    pipes(netConfig->nPipes),
    buffers(nodeConfig->nBuffers),
    internalBuffers()
{
    for (NnUint i = 0; i < netConfig->nPipes; i++) {
        const NnPipeConfig *config = &netConfig->pipes[i];
        const bool isSliceable = config->size.z == 1u;
        pipes[i] = bufferFactory->create(config->name, config->size.nBytes, isSliceable, vk::BufferUsageFlagBits::eStorageBuffer, true);
    }
    for (NnUint i = 0; i < nodeConfig->nBuffers; i++) {
        const NnBufferConfig *config = &nodeConfig->buffers[i];
        const bool isSliceable = config->size.z == 1u;
        buffers[i] = bufferFactory->create(config->name, config->size.nBytes, isSliceable, vk::BufferUsageFlagBits::eStorageBuffer, false);
    }

    std::vector<NnByte *> ropeOpConfigs;
    collectOpConfigs(ropeOpConfigs, nodeConfig, OP_ROPE);
    std::vector<NnUint> initializedRopeBuffers;
    for (NnByte *rawConfig : ropeOpConfigs) {
        NnRopeOpConfig *ropeLlamaOpConfig = (NnRopeOpConfig *)rawConfig;
        if (ropeLlamaOpConfig == nullptr) continue;
        const NnUint bufferIndex = ropeLlamaOpConfig->ropeCacheBufferIndex;
        if (std::find(initializedRopeBuffers.begin(), initializedRopeBuffers.end(), bufferIndex) != initializedRopeBuffers.end())
            continue;
        assert(bufferIndex < nodeConfig->nBuffers);
        NnVulkanBuffer *buffer = buffers[bufferIndex].get();
        std::vector<NnByte> ropeCache(ropeLlamaOpConfig->slice.cacheSize.nBytes);
        fullfillRopeCache(ropeLlamaOpConfig, (float *)ropeCache.data());
        buffer->write(ropeCache.data());
        initializedRopeBuffers.push_back(bufferIndex);
    }
}

NnVulkanDeviceData::~NnVulkanDeviceData() {
    pipes.clear();
    buffers.clear();
    internalBuffers.clear();
    VULKAN_TRACE("Destroyed device data");
}

NnSize3D NnVulkanDeviceData::resolveBufferSize(NnPointerConfig *config) {
    if (config->source == SRC_BUFFER)
        return nodeConfig->buffers[config->pointerIndex].size;
    if (config->source == SRC_PIPE)
        return netConfig->pipes[config->pointerIndex].size;
    throw std::invalid_argument("Unsupported pointer config");
}

NnVulkanBuffer *NnVulkanDeviceData::resolvePointerVulkanBuffer(NnPointerConfig *config) {
    if (config->source == SRC_BUFFER)
        return resolveBufferByIndex(config->pointerIndex);
    if (config->source == SRC_PIPE)
        return resolvePipeByIndex(config->pointerIndex);
    throw std::invalid_argument("Unsupported pointer config");
}

NnUint NnVulkanDeviceData::resolveBufferBatchOffset(NnPointerConfig *config, NnUint batchIndex, NnUint zIndex) {
    assert(batchIndex < netConfig->nBatches);
    if (config->type == PNTR_RAW)
        return 0;

    const NnSize3D bufferSize = resolveBufferSize(config);
    const NnSize blockSize = getBlockSize(bufferSize.floatType);
    assert(bufferSize.x % blockSize == 0);
    const NnUint sizeX = bufferSize.x / blockSize;
    const NnUint offsetZ = sizeX * netConfig->nBatches * zIndex;

    if (config->type == PNTR_BATCH)
        return offsetZ + sizeX * batchIndex;
    if (config->type == PNTR_BATCHED_SLICE) {
        NnUint logicalOffset = 0u;
        NnUint logicalLength = bufferSize.x;
        resolveUnevenSliceVulkan(netConfig, nodeConfig, config, bufferSize, &logicalOffset, &logicalLength);
        (void)logicalLength;
        assert((logicalOffset % blockSize) == 0u);
        return offsetZ + sizeX * batchIndex + (logicalOffset / blockSize);
    }
    throw std::runtime_error("Cannot determine buffer offset");
}

NnUint NnVulkanDeviceData::resolveBufferBatchWidth(NnPointerConfig *config) {
    const NnSize3D bufferSize = resolveBufferSize(config);
    const NnSize blockSize = getBlockSize(bufferSize.floatType);
    assert(bufferSize.x % blockSize == 0);
    const NnUint sizeX = bufferSize.x / blockSize;

    if (config->type == PNTR_RAW)
        return sizeX;
    if (config->type == PNTR_BATCH)
        return sizeX;
    if (config->type == PNTR_BATCHED_SLICE) {
        NnUint logicalOffset = 0u;
        NnUint logicalLength = bufferSize.x;
        resolveUnevenSliceVulkan(netConfig, nodeConfig, config, bufferSize, &logicalOffset, &logicalLength);
        (void)logicalOffset;
        assert((logicalLength % blockSize) == 0u);
        return logicalLength / blockSize;
    }
    throw std::runtime_error("Cannot determine buffer width");
}

NnVulkanBuffer *NnVulkanDeviceData::resolvePipeByIndex(NnUint pipeIndex) {
    assert(pipeIndex < netConfig->nPipes);
    return pipes[pipeIndex].get();
}

NnVulkanBuffer *NnVulkanDeviceData::resolveBufferByIndex(NnUint bufferIndex) {
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


void NnVulkanDeviceData::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    if (nodeConfig != nullptr) {
        nodeConfig->partitionPlan = plan;
    }
}

const NnUnevenPartitionPlan *NnVulkanDeviceData::getPartitionPlan() const {
    return nodeConfig != nullptr ? nodeConfig->partitionPlan : nullptr;
}

NnUint NnVulkanDeviceData::getNodeIndex() const {
    return nodeConfig != nullptr ? nodeConfig->nodeIndex : 0u;
}

NnVulkanDevice::NnVulkanDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan) :
    context(gpuIndex),
    copier(&context),
    bufferFactory(&context, &copier),
    netConfig(netConfig),
    nodeConfig(nodeConfig),
    netExecution(netExecution),
    partitionPlan(partitionPlan),
    data(&bufferFactory, netConfig, nodeConfig)
{
    setPartitionPlan(partitionPlan);
}

NnVulkanDevice::~NnVulkanDevice() {}


void NnVulkanDevice::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    partitionPlan = plan;
    if (nodeConfig != nullptr) {
        nodeConfig->partitionPlan = plan;
    }
    data.setPartitionPlan(plan);
}

NnUint NnVulkanDevice::maxNThreads() {
    return 1;
}

NnDeviceSegment *NnVulkanDevice::createSegment(NnUint segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    return new NnVulkanDeviceSegment(this, &context, &copier, &bufferFactory, &data, netConfig, segmentIndex, segmentConfig, netExecution);
};

static const char *getShaderFileName(const NnOpCode opCode, const NnOpQuantType quantType) {
    if (opCode == OP_MERGE_ADD) {
        if (quantType == F32_F32_F32) return "merge-add-forward-f32-f32.spv";
        if (quantType == Q80_Q80_F32) return "merge-add-forward-q80-f32.spv";
    }
    if (opCode == OP_MERGE_SUM) {
        if (quantType == F32_F32_F32) return "merge-sum-forward-f32-f32.spv";
    }
    if (opCode == OP_EMBEDDING) {
        if (quantType == F32_F32_F32) return "embedding-forward-f32-f32.spv";
    }
    if (opCode == OP_ROPE) {
        if (quantType == F32_F32_F32) return "rope-forward-f32-f32.spv";
    }
    if (opCode == OP_INV_RMS) {
        if (quantType == F32_F32_F32) return "inv-rms-forward-f32-f32.spv";
    }
    if (opCode == OP_MATMUL) {
        if (quantType == F32_F32_F32) return "matmul-forward-f32-f32-f32.spv";
        if (quantType == Q80_Q40_F32) return "matmul-forward-q80-q40-f32.spv";
    }
    if (opCode == OP_MULTIHEAD_ATT) {
        if (quantType == F32_F32_F32) return "multi-head-att-forward-f32-f32.spv";
    }
    if (opCode == OP_RMS_NORM) {
        if (quantType == F32_F32_F32) return "rms-norm-forward-f32-f32-f32.spv";
    }
    if (opCode == OP_SILU) {
        if (quantType == F32_F32_F32) return "silu-forward-f32-f32.spv";
    }
    if (opCode == OP_MUL) {
        if (quantType == F32_F32_F32) return "mul-forward-f32-f32.spv";
    }
    if (opCode == OP_SCALE) {
        if (quantType == F32_F32_F32) return "scale-forward-f32-f32.spv";
    }
    if (opCode == OP_CAST) {
        if (quantType == F32_F32_F32) return "cast-forward-f32-f32.spv";
        if (quantType == F32_F32_Q80) return "cast-forward-f32-q80.spv";
    }
    if (opCode == OP_REPEAT_Z) {
        if (quantType == F32_F32_Q80) return "repeatz-forward-f32-q80.spv";
    }
    if (opCode == OP_SHIFT) {
        if (quantType == F32_F32_F32) return "shift-forward-f32-f32.spv";
    }
    if (opCode == OP_SOFTMAX) {
        if (quantType == F32_F32_F32) return "softmax-forward-f32-f32.spv";
    }
    if (opCode == OP_MOE_GATE) {
        if (quantType == F32_F32_F32) return "moe-gate-forward-f32-f32.spv";
    }
    throw std::invalid_argument(std::string("Unsupported shader: ") + opCodeToString(opCode) + "/" + opQuantTypeToString(quantType));
}

static void buildShaderLayout(std::vector<NnOpBufferAccess> &a, NnVulkanDeviceData *data, NnVulkanDeviceSegmentData *segmentData, NnUint opIndex, NnOpConfig *opConfig) {
    // input
    a.push_back({ACCESS_READONLY, data->resolvePointerVulkanBuffer(&opConfig->input)});
    // output
    a.push_back({ACCESS_READ_WRITE, data->resolvePointerVulkanBuffer(&opConfig->output)});
    // batch info
    a.push_back({ACCESS_IMMUTABLE, segmentData->resolveOpBatchInfoVulkanBuffer(opIndex)});
    // weight
    if (opConfig->weightSize.nBytes > 0)
        a.push_back({ACCESS_IMMUTABLE, segmentData->resolveOpWeightVulkanBuffer(opIndex)});
    // config
    if (opConfig->configSize > 0) {
        a.push_back({ACCESS_IMMUTABLE, segmentData->resolveOpConfigVulkanBuffer(opIndex)});

        switch (opConfig->code) {
        case OP_RMS_NORM: {
            const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->invRmsBufferIndex)});
        } break;
        case OP_MUL: {
            const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->multiplierBufferIndex)});
        } break;
        case OP_MATMUL: {
            const NnMatmulOpConfig *config = (NnMatmulOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->activeExpertIndexesBufferIndex)});
        } break;
        case OP_SCALE: {
            const NnScaleOpCodeConfig *config = (NnScaleOpCodeConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->scaleBufferIndex)});
        } break;
        case OP_SHIFT: {
            const NnShiftOpCodeConfig *config = (NnShiftOpCodeConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->indexPipeIndex)});
        } break;
        case OP_ROPE: {
            const NnRopeOpConfig *config = (NnRopeOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->positionPipeIndex)});
            a.push_back({ACCESS_IMMUTABLE, data->resolveBufferByIndex(config->ropeCacheBufferIndex)});
        } break;
        case OP_MULTIHEAD_ATT: {
            const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->positionPipeIndex)});
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->queryBufferIndex)});
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->keyCacheBufferIndex)});
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->valueCacheBufferIndex)});
            a.push_back({ACCESS_READ_WRITE, data->resolveBufferByIndex(config->attBufferIndex)});
        } break;
        case OP_MOE_GATE: {
            const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)opConfig->config;
            a.push_back({ACCESS_READ_WRITE, data->resolveBufferByIndex(config->indexesBufferIndex)});
        } break;
        default:
            break;
        }
    }
}

static bool containsBuffer(std::vector<NnVulkanBuffer *> &buffers, NnVulkanBuffer *buffer) {
    for (NnVulkanBuffer *b : buffers) {
        if (b == buffer) return true;
    }
    return false;
}

static void resolveBuffersToSync(std::vector<std::vector<NnVulkanBuffer *>>& buffersToSync, std::vector<std::vector<NnOpBufferAccess>>& accesses) {
    std::vector<NnVulkanBuffer *> modifiedBuffers;
    for (NnUint opIndex = 0; opIndex < accesses.size(); opIndex++) {
        if (opIndex > 0) {
            bool flush = false;
            for (const NnOpBufferAccess &access : accesses[opIndex]) {
                if (access.type == ACCESS_READONLY && containsBuffer(modifiedBuffers, access.buffer)) {
                    flush = true;
                    break;
                }
            }
            if (flush) {
                for (NnVulkanBuffer *buffer : modifiedBuffers)
                    buffersToSync[opIndex].push_back(buffer);
                modifiedBuffers.clear();
            }
        }
        for (const NnOpBufferAccess &access : accesses[opIndex]) {
            if (access.type == ACCESS_READ_WRITE && !containsBuffer(modifiedBuffers, access.buffer))
                modifiedBuffers.push_back(access.buffer);
        }
    }
}

static NnUint resolveNumberOfBatchInfoZ(const NnSize3D inputSize, const NnSize3D outputSize) {
    return std::max(inputSize.z, outputSize.z);
}

static std::vector<NnVulkanBatchInfo> buildBatchInfo(NnOpConfig *opConfig, NnVulkanDeviceData *data, NnUint nBatches) {
    const NnUint nZ = resolveNumberOfBatchInfoZ(
        data->resolveBufferSize(&opConfig->input),
        data->resolveBufferSize(&opConfig->output)
    );

    std::vector<NnVulkanBatchInfo> batchInfo(nZ * nBatches);
    for (NnUint zIndex = 0; zIndex < nZ; zIndex++) {
        for (NnUint batchIndex = 0; batchIndex < nBatches; batchIndex++) {
            const NnUint b = zIndex * nBatches + batchIndex;
            batchInfo[b].inputOffset = data->resolveBufferBatchOffset(&opConfig->input, batchIndex, zIndex);
            batchInfo[b].inputSizeX = data->resolveBufferBatchWidth(&opConfig->input);
            batchInfo[b].outputOffset = data->resolveBufferBatchOffset(&opConfig->output, batchIndex, zIndex);
            batchInfo[b].outputSizeX = data->resolveBufferBatchWidth(&opConfig->output);
        }
    }
    return batchInfo;
}

static NnSize3D resolveShaderLogicalSize(NnVulkanDeviceData *data, NnPointerConfig *config) {
    NnSize3D size = data->resolveBufferSize(config);
    if (config->type == PNTR_BATCHED_SLICE) {
        size.x = data->resolveBufferBatchWidth(config) * getBlockSize(size.floatType);
    }
    return size;
}

static std::vector<NnUint> resolveShaderConstants(const NnOpConfig *opConfig, const NnUint nBatches, const NnSize3D inputSize, const NnSize3D outputSize) {
    std::vector<NnUint> consts;
    consts.push_back(nBatches);

    if (
        opConfig->code == OP_CAST ||
        opConfig->code == OP_MATMUL ||
        opConfig->code == OP_MERGE_SUM ||
        opConfig->code == OP_MUL ||
        opConfig->code == OP_REPEAT_Z ||
        opConfig->code == OP_SCALE ||
        opConfig->code == OP_SILU ||
        opConfig->code == OP_SOFTMAX
    ) {
        const NnUint nZ = resolveNumberOfBatchInfoZ(inputSize, outputSize);
        consts.push_back(nZ);
    }
    if (opConfig->code == OP_MATMUL) {
        if (opConfig->weightSize.floatType == F_Q40) {
            // The q80*q40 shader uses local_size_x as a specialization constant.
            // Online head/FFN repartition can change a matmul input width after
            // pipeline creation, so do not specialize this to the initial slice.
            // The shader guards against threads beyond BatchInfo.inputSizeX.
            constexpr NnUint maxThreads = 256u;
            assert(inputSize.x % Q40_BLOCK_SIZE == 0);
            consts.push_back(maxThreads);
            consts.push_back(opConfig->weightSize.y / Q40_BLOCK_SIZE); // physical weight row stride, in Q40 blocks
            consts.push_back(opConfig->weightSize.x); // physical output rows per expert
        } else if (opConfig->weightSize.floatType == F_32) {
            consts.push_back(opConfig->weightSize.y); // physical weight row stride, in elements
            consts.push_back(opConfig->weightSize.x); // physical output rows per expert
        }
    }
    if (opConfig->code == OP_MOE_GATE) {
        const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)opConfig->config;
        consts.push_back(inputSize.x);
        consts.push_back(config->k);
    }
    return consts;
}

static NnUint resolveShaderNumberOfWorkGroupsX(const NnOpConfig *opConfig, const NnSize3D inputSize, const NnSize3D outputSize) {
    if (
        opConfig->code == OP_CAST ||
        opConfig->code == OP_REPEAT_Z
    ) {
        if (outputSize.floatType == F_Q80) {
            return outputSize.x / Q80_BLOCK_SIZE;
        } else {
            constexpr NnUint chunkSize = 4u; // Shader constant
            return outputSize.x / chunkSize;
        }
    }
    if (opConfig->code == OP_MERGE_ADD) {
        if (inputSize.floatType == F_Q80) {
            return outputSize.x / Q80_BLOCK_SIZE; // Yes, outputSize is used here
        } else {
            return 32u;
        }
    }
    if (opConfig->code == OP_MATMUL) {
        if (opConfig->weightSize.floatType == F_Q40) {
            constexpr NnUint tileSizeD = 8u; // Shader constant
            return (outputSize.x + tileSizeD - 1u) / tileSizeD;
        } else {
            return 32u;
        }
    }
    if (opConfig->code == OP_MULTIHEAD_ATT)
        return ((NnMultiHeadAttOpConfig *)opConfig->config)->nHeads0;
    if (opConfig->code == OP_INV_RMS)
        return ((NnInvRmsOpConfig *)opConfig->config)->nColumns;
    if (
        opConfig->code == OP_EMBEDDING ||
        opConfig->code == OP_RMS_NORM ||
        opConfig->code == OP_MUL ||
        opConfig->code == OP_SCALE ||
        opConfig->code == OP_SILU ||
        opConfig->code == OP_SHIFT ||
        opConfig->code == OP_MERGE_SUM
    ) {
        constexpr NnUint chunkSize = 4u; // Shader constant
        assert(outputSize.x % chunkSize == 0);
        return outputSize.x / chunkSize;
    }
    return 1u;
}

static NnUint resolveShaderNumberOfWorkGroupsZ(const NnOpConfig *opConfig, const NnSize3D inputSize, const NnSize3D outputSize) {
    if (
        opConfig->code == OP_MERGE_SUM ||
        opConfig->code == OP_REPEAT_Z ||
        opConfig->code == OP_MOE_GATE
    ) {
        return 1u;
    }
    return resolveNumberOfBatchInfoZ(inputSize, outputSize);
}

static std::vector<uint32_t> readShader(const char *fileName) {
    std::vector<uint32_t> code;
    std::string path = std::string("./src/nn/vulkan/") + fileName;
    FILE *file = fopen(path.c_str(), "rb");
    if (!file)
        throw std::runtime_error("Failed to open shader file: " + path);
    constexpr size_t maxSize = 16384;
    uint32_t chunk[maxSize];
    size_t bytesRead = fread(chunk, 1, maxSize, file);
    assert(bytesRead < maxSize); // Check if the file is too large
    if (bytesRead > 0)
        code.insert(code.end(), chunk, chunk + bytesRead);
    if (ferror(file)) {
        fclose(file);
        throw std::runtime_error("Failed to read shader file: " + path);
    }
    fclose(file);
    return code;
}

static vk::DescriptorType toDescriptorType(NnVulkanBuffer *buffer) {
    if (buffer->usageFlags & vk::BufferUsageFlagBits::eUniformBuffer)
        return vk::DescriptorType::eUniformBuffer;
    if (buffer->usageFlags & vk::BufferUsageFlagBits::eStorageBuffer)
        return vk::DescriptorType::eStorageBuffer;
    throw std::invalid_argument("Unsupported buffer usage");
}

NnVulkanDeviceSegmentData::NnVulkanDeviceSegmentData(NnVulkanBufferFactory *bufferFactory, NnVulkanDeviceData *data, NnSegmentConfig *segmentConfig, NnUint nBatches) :
    data(data),
    batchInfoBufferIndex(segmentConfig->nOps, UINT32_MAX),
    weightBufferIndex(segmentConfig->nOps, UINT32_MAX),
    configBufferIndex(segmentConfig->nOps, UINT32_MAX)
{
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        if (isVulkanControlOp(opConfig->code)) {
            continue;
        }

        std::vector<NnVulkanBatchInfo> batchInfo = buildBatchInfo(opConfig, data, nBatches);
        VULKAN_TRACE("Resolved for %s %zu batch infos", opCodeToString(opConfig->code), batchInfo.size());

        NnSize batchInfoSize = sizeof(NnVulkanBatchInfo) * batchInfo.size();
        data->internalBuffers.push_back(
            bufferFactory->create("batchInfo", batchInfoSize, false, vk::BufferUsageFlagBits::eUniformBuffer, false)
        );
        NnVulkanBuffer *batchInfoBuffer = data->internalBuffers.back().get();
        batchInfoBuffer->write((NnByte *)batchInfo.data());
        batchInfoBufferIndex[opIndex] = data->internalBuffers.size() - 1;

        if (opConfig->weightSize.nBytes > 0) {
            data->internalBuffers.push_back(
                bufferFactory->create("weights", opConfig->weightSize.nBytes, false, vk::BufferUsageFlagBits::eStorageBuffer, false)
            );
            weightBufferIndex[opIndex] = data->internalBuffers.size() - 1;
        }
        if (opConfig->configSize > 0) {
            data->internalBuffers.push_back(
                bufferFactory->create("config", opConfig->configSize, false, vk::BufferUsageFlagBits::eUniformBuffer, false)
            );
            NnVulkanBuffer *configBuffer = data->internalBuffers.back().get();
            configBuffer->write(opConfig->config);
            configBufferIndex[opIndex] = data->internalBuffers.size() - 1;
        }
    }
}

NnVulkanBuffer *NnVulkanDeviceSegmentData::resolveOpBatchInfoVulkanBuffer(NnUint opIndex) {
    assert(batchInfoBufferIndex[opIndex] != UINT32_MAX);
    return data->internalBuffers[batchInfoBufferIndex[opIndex]].get();
}

NnVulkanBuffer *NnVulkanDeviceSegmentData::resolveOpConfigVulkanBuffer(NnUint opIndex) {
    assert(configBufferIndex[opIndex] != UINT32_MAX);
    return data->internalBuffers[configBufferIndex[opIndex]].get();
}

NnVulkanBuffer *NnVulkanDeviceSegmentData::resolveOpWeightVulkanBuffer(NnUint opIndex) {
    assert(weightBufferIndex[opIndex] != UINT32_MAX);
    return data->internalBuffers[weightBufferIndex[opIndex]].get();
}

NnVulkanDeviceSegment::NnVulkanDeviceSegment(NnVulkanDevice *ownerDevice, NnVulkanContext *context, NnVulkanStagingCopier *copier, NnVulkanBufferFactory *bufferFactory, NnVulkanDeviceData *data, NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution) :
    context(context),
    copier(copier),
    data(data),
    netConfig(netConfig),
    segmentIndex(segmentIndex),
    segmentConfig(segmentConfig),
    netExecution(netExecution),
    ownerDevice(ownerDevice),
    shaderModules(segmentConfig->nOps),
    descriptorSets(segmentConfig->nOps),
    descriptorSetLayouts(segmentConfig->nOps),
    pipelineLayouts(segmentConfig->nOps),
    pipelines(segmentConfig->nOps),
    buffersToSync(segmentConfig->nOps)
{
    this->segmentData.reset(new NnVulkanDeviceSegmentData(bufferFactory, data, segmentConfig, netExecution->nBatches));
    this->lastBatchSize = 0;
    this->commandBufferDirty = true;

    bool hasVulkanComputeOps = false;
    for (NnUint i = 0; i < segmentConfig->nOps; ++i) {
        if (!isVulkanControlOp(segmentConfig->ops[i].code)) {
            hasVulkanComputeOps = true;
            break;
        }
    }
    if (!hasVulkanComputeOps) {
        return;
    }

    std::vector<vk::PipelineShaderStageCreateInfo> shaderCreateInfos(segmentConfig->nOps);
    std::vector<std::vector<NnOpBufferAccess>> opBufferAccesses(segmentConfig->nOps);

    std::vector<vk::SpecializationInfo> specInfos(segmentConfig->nOps);
    std::vector<std::vector<vk::SpecializationMapEntry>> specEntries(segmentConfig->nOps);
    std::vector<std::vector<NnUint>> constants(segmentConfig->nOps);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        if (isVulkanControlOp(opConfig->code)) {
            continue;
        }
        NnSize3D inputSize = resolveShaderLogicalSize(data, &opConfig->input);
        NnSize3D outputSize = resolveShaderLogicalSize(data, &opConfig->output);
        NnOpQuantType opQuant = getOpQuantType(
            inputSize.floatType,
            opConfig->weightSize.floatType,
            outputSize.floatType
        );
        const char *shaderFileName = getShaderFileName(opConfig->code, opQuant);
        std::vector<uint32_t> code = readShader(shaderFileName);

        buildShaderLayout(opBufferAccesses[opIndex], data, segmentData.get(), opIndex, opConfig);

        VULKAN_TRACE("Loading shader: %s", shaderFileName);
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            code.size(),
            code.data()
        );

        constants[opIndex] = resolveShaderConstants(opConfig, netConfig->nBatches, inputSize, outputSize);
        const std::vector<NnUint> *opConstants = &constants[opIndex];

        specEntries[opIndex].resize(opConstants->size());
        for (NnUint i = 0; i < opConstants->size(); i++) {
            specEntries[opIndex][i] = vk::SpecializationMapEntry(
                i,
                i * sizeof(NnUint),
                sizeof(NnUint)
            );
            VULKAN_TRACE("Spec constant %d: %u", i, opConstants->at(i));
        }
        specInfos[opIndex] = vk::SpecializationInfo(
            opConstants->size(),
            specEntries[opIndex].data(),
            opConstants->size() * sizeof(NnUint),
            opConstants->data()
        );

        vk::ShaderModule shaderModule = context->device.createShaderModule(shaderModuleCreateInfo);
        vk::PipelineShaderStageCreateInfo shaderCreateInfo(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            shaderModule,
            "main",
            &specInfos[opIndex]
        );

        shaderModules[opIndex] = shaderModule;
        shaderCreateInfos[opIndex] = shaderCreateInfo;
        VULKAN_TRACE("Segment %d, opIndex: %d, buffers: %zu", segmentIndex, opIndex, opBufferAccesses.size());
    }

    NnUint nUniformBuffers = 0;
    NnUint nStorageBuffers = 0;
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (isVulkanControlOp(segmentConfig->ops[opIndex].code)) {
            vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
                vk::DescriptorSetLayoutCreateFlags(),
                0,
                nullptr
            );
            descriptorSetLayouts[opIndex] = context->device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);
            continue;
        }
        std::vector<NnOpBufferAccess> &accesses = opBufferAccesses[opIndex];
        std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings(accesses.size());
        for (NnUint i = 0; i < accesses.size(); i++) {
            vk::DescriptorType descriptorType = toDescriptorType(accesses[i].buffer);
            descriptorSetLayoutBindings[i] = vk::DescriptorSetLayoutBinding(
                i,
                descriptorType,
                1,
                vk::ShaderStageFlagBits::eCompute
            );
            if (descriptorType == vk::DescriptorType::eUniformBuffer)
                nUniformBuffers++;
            else if (descriptorType == vk::DescriptorType::eStorageBuffer)
                nStorageBuffers++;
        }

        vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            descriptorSetLayoutBindings.size(),
            descriptorSetLayoutBindings.data()
        );
        vk::DescriptorSetLayout descriptorSetLayout = context->device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

        descriptorSetLayouts[opIndex] = descriptorSetLayout;
    }

    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;
    if (nStorageBuffers > 0)
        descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, nStorageBuffers));
    if (nUniformBuffers > 0)
        descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, nUniformBuffers));
    if (descriptorPoolSizes.empty())
        descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1));

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 
        segmentConfig->nOps,
        descriptorPoolSizes.size(),
        descriptorPoolSizes.data()
    );
    descriptorPool = context->device.createDescriptorPool(descriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo descriptorSetAllocInfo(descriptorPool, descriptorSetLayouts.size(), descriptorSetLayouts.data());
    descriptorSets = context->device.allocateDescriptorSets(descriptorSetAllocInfo);

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets(nUniformBuffers + nStorageBuffers);
    std::vector<vk::DescriptorBufferInfo> bufferInfos(nUniformBuffers + nStorageBuffers);
    NnUint writeDescriptorSetIndex = 0;
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (isVulkanControlOp(segmentConfig->ops[opIndex].code)) {
            continue;
        }
        std::vector<NnOpBufferAccess> &accesses = opBufferAccesses[opIndex];
        for (NnUint i = 0; i < accesses.size(); i++) {
            NnVulkanBuffer *buffer = accesses[i].buffer;
            bufferInfos[writeDescriptorSetIndex] = vk::DescriptorBufferInfo(
                buffer->deviceBuffer,
                0,
                buffer->bufferSize
            );
            vk::DescriptorType descriptorType = toDescriptorType(buffer);
            writeDescriptorSets[writeDescriptorSetIndex] = vk::WriteDescriptorSet(
                descriptorSets[opIndex],
                i,
                0,
                1,
                descriptorType,
                nullptr,
                &bufferInfos.data()[writeDescriptorSetIndex],
                nullptr
            );
            writeDescriptorSetIndex++;
        }
    }

    context->device.updateDescriptorSets(writeDescriptorSets, nullptr);

    pipelineCache = context->device.createPipelineCache(vk::PipelineCacheCreateInfo());

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (isVulkanControlOp(segmentConfig->ops[opIndex].code)) {
            continue;
        }
        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &descriptorSetLayouts[opIndex]);
        pipelineLayouts[opIndex] = context->device.createPipelineLayout(pipelineLayoutCreateInfo);

        vk::ComputePipelineCreateInfo pipelineInfo(vk::PipelineCreateFlags(), shaderCreateInfos[opIndex], pipelineLayouts[opIndex], vk::Pipeline(), 0);
        pipelines[opIndex] = context->device.createComputePipelines(pipelineCache, pipelineInfo).value.front();
    }

    fence = context->device.createFence(vk::FenceCreateInfo());

    vk::CommandBufferAllocateInfo commandBufferAllocInfo(context->commandPool, vk::CommandBufferLevel::ePrimary, 1);
    const std::vector<vk::CommandBuffer> cmdBuffers = context->device.allocateCommandBuffers(commandBufferAllocInfo);
    commandBuffer = cmdBuffers.front();

    resolveBuffersToSync(buffersToSync, opBufferAccesses);
}

NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
    context->device.waitIdle();
    if (commandBuffer) context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);
    if (descriptorPool) context->device.destroyDescriptorPool(descriptorPool);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (pipelines[opIndex]) context->device.destroyPipeline(pipelines[opIndex]);
        if (pipelineLayouts[opIndex]) context->device.destroyPipelineLayout(pipelineLayouts[opIndex]);
    }
    if (fence) context->device.destroyFence(fence);
    if (pipelineCache) context->device.destroyPipelineCache(pipelineCache);
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (descriptorSetLayouts[opIndex]) context->device.destroyDescriptorSetLayout(descriptorSetLayouts[opIndex]);
        if (shaderModules[opIndex]) context->device.destroyShaderModule(shaderModules[opIndex]);
    }
    VULKAN_TRACE("Destroyed segment");
}


void NnVulkanDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    assert(opIndex < segmentConfig->nOps);
    assert(offset + nBytes <= segmentConfig->ops[opIndex].weightSize.nBytes);
    NnVulkanBuffer *buffer = segmentData->resolveOpWeightVulkanBuffer(opIndex);
    buffer->write(weight, offset, nBytes);
}


void NnVulkanDeviceSegment::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    if (ownerDevice != nullptr) {
        ownerDevice->setPartitionPlan(plan);
    } else if (data != nullptr) {
        data->setPartitionPlan(plan);
    }
    refreshPointers();
}


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

void NnVulkanDeviceSegment::refreshPointers() {
    if (data == nullptr || segmentConfig == nullptr || segmentData == nullptr) return;
    const NnUnevenPartitionPlan *plan = (ownerDevice != nullptr) ? ownerDevice->getPartitionPlan() : data->getPartitionPlan();
    data->setPartitionPlan(plan);
    const NnUint nodeIndex = data->getNodeIndex();

    auto getSplitStart = [&](NnSliceTag tag, NnUint *outStart) -> bool {
        if (plan == nullptr || outStart == nullptr) return false;
        switch (tag) {
            case NN_SLICE_VOCAB:
                if (plan->vocabSplit.starts) { *outStart = plan->vocabSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_FFN:
                if (plan->ffnSplit.starts) { *outStart = plan->ffnSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_DIM:
                if (plan->dimSplit.starts) { *outStart = plan->dimSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_HEAD:
                if (plan->headSplit.starts) { *outStart = plan->headSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_KV_HEAD:
                if (plan->kvHeadSplit.starts) { *outStart = plan->kvHeadSplit.starts[nodeIndex]; return true; }
                return false;
            default:
                return false;
        }
    };

    auto findRmsNormColSize = [&](NnUint fromOpIndex, const NnSize3D &inputSizeNow) -> NnUint {
        for (NnUint j = fromOpIndex + 1u; j < segmentConfig->nOps && j < fromOpIndex + 8u; ++j) {
            NnOpConfig *c = &segmentConfig->ops[j];
            if (c->code != OP_RMS_NORM) continue;
            if (c->weightSize.x == 0u) continue;
            if (inputSizeNow.x % c->weightSize.x != 0u) continue;
            return c->weightSize.x;
        }
        return 0u;
    };

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; ++opIndex) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        if (isVulkanControlOp(opConfig->code)) continue;

        std::vector<NnVulkanBatchInfo> batchInfo = buildBatchInfo(opConfig, data, netExecution->nBatches);
        NnVulkanBuffer *batchInfoBuffer = segmentData->resolveOpBatchInfoVulkanBuffer(opIndex);
        batchInfoBuffer->write((NnByte *)batchInfo.data());

        NnSize3D inputSize = resolveShaderLogicalSize(data, &opConfig->input);
        NnSize3D outputSize = resolveShaderLogicalSize(data, &opConfig->output);
        if (opConfig->code == OP_CAST &&
            opConfig->output.type == PNTR_BATCHED_SLICE &&
            inputSize.x != outputSize.x) {
            outputSize = size3D(outputSize.floatType, outputSize.z, outputSize.y, inputSize.x);
        }

        if (plan != nullptr && opConfig->config != nullptr) {
            if (opConfig->code == OP_MULTIHEAD_ATT) {
                auto *cfg = (NnMultiHeadAttOpConfig *)opConfig->config;
                const bool fullAttBuffer =
                    opConfig->input.type == PNTR_BATCHED_SLICE ||
                    opConfig->output.type == PNTR_BATCHED_SLICE;
                if (plan->headSplit.starts && plan->headSplit.lengths) {
                    cfg->nHeads0 = plan->headSplit.lengths[nodeIndex];
                    cfg->qSliceD0 = cfg->nHeads0 * cfg->headDim;
                    if (fullAttBuffer) {
                        cfg->qStart = plan->headSplit.starts[nodeIndex] * cfg->headDim;
                        cfg->qStride = cfg->nHeads * cfg->headDim;
                    } else {
                        cfg->qStart = 0u;
                        cfg->qStride = cfg->qSliceD0;
                    }
                }
                if (plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
                    const NnUint kvHeads0 = plan->kvHeadSplit.lengths[nodeIndex];
                    cfg->kvDim0 = kvHeads0 * cfg->headDim;
                    if (fullAttBuffer) {
                        cfg->kvStart = plan->kvHeadSplit.starts[nodeIndex] * cfg->headDim;
                        cfg->kvStride = cfg->nKvHeads * cfg->headDim;
                    } else {
                        cfg->kvStart = 0u;
                        cfg->kvStride = cfg->kvDim0;
                    }
                }
            } else if (opConfig->code == OP_SHIFT) {
                auto *cfg = (NnShiftOpCodeConfig *)opConfig->config;
                if (cfg->dstRowStride != 0u && cfg->dstColStartUnit != 0u &&
                    plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
                    cfg->dstColStart = plan->kvHeadSplit.starts[nodeIndex] * cfg->dstColStartUnit;
                }
            } else if (opConfig->code == OP_MATMUL) {
                auto *cfg = (NnMatmulOpConfig *)opConfig->config;
                if (cfg->view == 1u && cfg->outSliceTag != NN_SLICE_AUTO && cfg->outStartUnit != 0u) {
                    NnUint st = 0u;
                    if (getSplitStart(cfg->outSliceTag, &st)) cfg->outStart = st * cfg->outStartUnit;
                }
                if (cfg->view == 2u && cfg->inSliceTag != NN_SLICE_AUTO && cfg->inStartUnit != 0u) {
                    NnUint st = 0u;
                    if (getSplitStart(cfg->inSliceTag, &st)) cfg->inStart = st * cfg->inStartUnit;
                }
                if (cfg->aView.sizeX != 0u) cfg->aView.sizeX = inputSize.x;
                if (cfg->cView.sizeX != 0u) cfg->cView.sizeX = outputSize.x;
            } else if (opConfig->code == OP_MUL) {
                auto *cfg = (NnMulOpCodeConfig *)opConfig->config;
                if (plan->ffnSplit.starts && plan->ffnSplit.lengths && cfg->view.sizeX != 0u) {
                    cfg->view.offset = plan->ffnSplit.starts[nodeIndex];
                    cfg->view.sizeX = plan->ffnSplit.lengths[nodeIndex];
                    cfg->view.strideX = 1u;
                }
            } else if (opConfig->code == OP_INV_RMS) {
                auto *cfg = (NnInvRmsOpConfig *)opConfig->config;
                const NnUint colSize = findRmsNormColSize(opIndex, inputSize);
                if (colSize != 0u && inputSize.x % colSize == 0u) {
                    const NnUint newCols = inputSize.x / colSize;
                    if (outputSize.x >= newCols && newCols != 0u) cfg->nColumns = newCols;
                }
            } else if (opConfig->code == OP_RMS_NORM) {
                auto *cfg = (NnRmsNormOpConfig *)opConfig->config;
                const NnUint colSize = opConfig->weightSize.x;
                if (colSize != 0u && inputSize.x % colSize == 0u) {
                    const NnUint newCols = inputSize.x / colSize;
                    if (newCols != 0u) cfg->nColumns = newCols;
                }
            } else if (opConfig->code == OP_ROPE) {
                auto *cfg = (NnRopeOpConfig *)opConfig->config;
                if (cfg->isQ == 1u) cfg->slice.qDim0 = inputSize.x;
                else cfg->slice.kvDim0 = inputSize.x;
            }
        }

        if (opConfig->configSize > 0) {
            NnVulkanBuffer *configBuffer = segmentData->resolveOpConfigVulkanBuffer(opIndex);
            configBuffer->write(opConfig->config);
        }
    }

    commandBufferDirty = true;
    lastBatchSize = 0u;
    if (ownerDevice != nullptr) {
        planEpochReady.store(ownerDevice->getPlanEpoch(), std::memory_order_release);
    }
}

void NnVulkanDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize)  {
    assert(threadIndex == 0);

    if (opIndex != 0) {
        // TODO: this is a design problem, executor tries to forward all ops in a segment
        return;
    }

    if (ownerDevice != nullptr) {
        const unsigned int deviceEpoch = ownerDevice->getPlanEpoch();
        const unsigned int readyEpoch = planEpochReady.load(std::memory_order_acquire);
        if (readyEpoch != deviceEpoch) {
            refreshPointers();
            planEpochReady.store(deviceEpoch, std::memory_order_release);
        }
    }

    // TODO: this should be called only after weights loading is done
    copier->tryRelease();

    // TODO: refactor this block
    {
        // PreSync pipes (POS today) are consumed through op config bindings by
        // ROPE/MHA/SHIFT, not necessarily as the primary op input. In PP mode a
        // non-first stage can have sync-only leading segments, so restricting
        // this upload to segment 0/1 leaves the first real Vulkan segment with a
        // stale position buffer and corrupts KV-cache writes during prefill.
        for (NnUint i = 0; i < netConfig->nPreSyncs; i++) {
            NnPreSyncConfig *preSyncConfig = &netConfig->preSyncs[i];
            NnByte *pipeData = netExecution->pipes[preSyncConfig->pipeIndex];
            NnVulkanBuffer *buffer = data->pipes[preSyncConfig->pipeIndex].get();
            buffer->write(pipeData, 0u, buffer->calcSliceSize(batchSize, netConfig->nBatches));
        }

        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (opConfig->input.source == SRC_PIPE) {
                NnByte *pipeData = netExecution->pipes[opConfig->input.pointerIndex];
                NnVulkanBuffer *buffer = data->pipes[opConfig->input.pointerIndex].get();
                buffer->write(pipeData, 0u, buffer->calcSliceSize(batchSize, netConfig->nBatches));
            }
        }
    }


    bool hasOnlyControlOps = true;
    for (NnUint i = 0; i < segmentConfig->nOps; ++i) {
        NnOpConfig *opConfig = &segmentConfig->ops[i];
        if (opConfig->code == OP_PLAN_BARRIER) {
            handleVulkanPlanBarrier(ownerDevice, netExecution, segmentConfig, i, batchSize);
        } else if (opConfig->code == OP_PLAN_APPLY) {
            handleVulkanPlanApply(ownerDevice, netExecution, segmentConfig, i);
        } else {
            hasOnlyControlOps = false;
        }
    }
    if (hasOnlyControlOps) {
        return;
    }

    if (lastBatchSize != batchSize || commandBufferDirty) {
        lastBatchSize = batchSize;
        commandBufferDirty = false;
        commandBuffer.begin({ vk::CommandBufferUsageFlags{} });

        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            if (isVulkanControlOp(segmentConfig->ops[opIndex].code)) {
                continue;
            }
            std::vector<vk::BufferMemoryBarrier> memoryBarriers;
            for (NnVulkanBuffer *buffer : buffersToSync[opIndex]) {
                vk::BufferMemoryBarrier barrier(
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead,
                    context->queueFamilyIndex,
                    context->queueFamilyIndex,
                    buffer->deviceBuffer,
                    0,
                    buffer->calcSliceSize(batchSize, netConfig->nBatches)
                );
                memoryBarriers.push_back(barrier);
            }
            if (!memoryBarriers.empty()) {
                commandBuffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlags(),
                    nullptr,
                    memoryBarriers,
                    nullptr
                );
                VULKAN_TRACE("Created memory barrier for %zu buffers", memoryBarriers.size());
            }

            commandBuffer.bindPipeline(
                vk::PipelineBindPoint::eCompute,
                pipelines[opIndex]
            );
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,
                pipelineLayouts[opIndex],
                0,
                { descriptorSets[opIndex] },
                {}
            );

            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            const NnSize3D inputSize = resolveShaderLogicalSize(data, &opConfig->input);
            const NnSize3D outputSize = resolveShaderLogicalSize(data, &opConfig->output);

            const NnUint groupCountX = resolveShaderNumberOfWorkGroupsX(opConfig, inputSize, outputSize);
            const NnUint groupCountY = batchSize;
            const NnUint groupCountZ = resolveShaderNumberOfWorkGroupsZ(opConfig, inputSize, outputSize);
            commandBuffer.dispatch(groupCountX, groupCountY, groupCountZ);
            VULKAN_TRACE("Dispatched %s (%d, %d, %d)", opCodeToString(opConfig->code), groupCountX, groupCountY, groupCountZ);
        }
        commandBuffer.end();
    }

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
    context->queue.submit({ submitInfo }, fence);
    assert(context->device.waitForFences({ fence }, true, uint64_t(-1)) == vk::Result::eSuccess);

    context->device.resetFences({ fence });
    VULKAN_TRACE("Forwarded");

    // TODO: refactor this block
    {
        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (opConfig->output.source == SRC_PIPE) {
                NnByte *pipeData = netExecution->pipes[opConfig->output.pointerIndex];
                NnVulkanBuffer *buffer = data->pipes[opConfig->output.pointerIndex].get();
                buffer->read(pipeData, 0u, buffer->calcSliceSize(batchSize, netConfig->nBatches));
            }
        }
    }

#if DEBUG_VULKAN_BUFFERS
    NnUint nBuffers = data->buffers.size();
    for (NnUint i = 0; i < nBuffers; i++) {
        NnVulkanBuffer *buffer = data->buffers[i].get();
        printf("[%3d:%3d:%10s] ", segmentIndex, i, buffer->name);
        std::vector<NnByte> data(buffer->bufferSize);
        buffer->read(data.data());
        if (strncmp(buffer->name, "q_", 2) == 0) {
            NnUint nBytes = 32;
            if (buffer->bufferSize < nBytes)
                nBytes = buffer->bufferSize;
            for (NnUint j = 0; j < nBytes; j++)
                printf(" %x", data.data()[j]);
        } else {
            NnUint nNumbers = buffer->bufferSize / sizeof(float);
            if (nNumbers > 16)
                nNumbers = 16;
            float *nums = (float *)data.data();
            for (NnUint j = 0; j < nNumbers; j++)
                printf(" %.4f", nums[j]);
        }
        printf("\n");
    }
#endif
}
