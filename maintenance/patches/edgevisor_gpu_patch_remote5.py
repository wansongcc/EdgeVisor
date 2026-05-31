from pathlib import Path

p = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor/src/nn/nn-vulkan.cpp")
s = p.read_text()

old = """    this->segmentData.reset(new NnVulkanDeviceSegmentData(bufferFactory, data, segmentConfig, netExecution->nBatches));
    this->lastBatchSize = 0;
    this->commandBufferDirty = true;

    std::vector<vk::PipelineShaderStageCreateInfo> shaderCreateInfos(segmentConfig->nOps);
"""
new = """    this->segmentData.reset(new NnVulkanDeviceSegmentData(bufferFactory, data, segmentConfig, netExecution->nBatches));
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
"""
if new not in s:
    if old not in s:
        raise RuntimeError("constructor insertion point not found")
    s = s.replace(old, new, 1)

old = """NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
    context->device.waitIdle();
    context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);
    context->device.destroyDescriptorPool(descriptorPool);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (pipelines[opIndex]) context->device.destroyPipeline(pipelines[opIndex]);
        if (pipelineLayouts[opIndex]) context->device.destroyPipelineLayout(pipelineLayouts[opIndex]);
    }
    context->device.destroyFence(fence);
    context->device.destroyPipelineCache(pipelineCache);
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (descriptorSetLayouts[opIndex]) context->device.destroyDescriptorSetLayout(descriptorSetLayouts[opIndex]);
        if (shaderModules[opIndex]) context->device.destroyShaderModule(shaderModules[opIndex]);
    }
    VULKAN_TRACE("Destroyed segment");
}
"""
new = """NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
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
"""
if new not in s:
    if old not in s:
        raise RuntimeError("destructor replacement point not found")
    s = s.replace(old, new, 1)

p.write_text(s)
print("skipped Vulkan resource allocation for control-only segments and guarded teardown")
