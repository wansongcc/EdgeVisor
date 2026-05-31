from pathlib import Path

p = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor/src/nn/nn-vulkan.cpp")
s = p.read_text()

if "#include <cstdlib>\n" not in s:
    s = s.replace("#include <cstring>\n", "#include <cstring>\n#include <cstdlib>\n", 1)

old = """NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
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
new = """NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
    const bool traceDtor = std::getenv("DLLAMA_VK_DTOR_TRACE") != nullptr;
    auto trace = [&](const char *step, NnUint opIndex = UINT32_MAX) {
        if (!traceDtor) return;
        if (opIndex == UINT32_MAX)
            std::fprintf(stderr, "[vk-dtor] seg=%u step=%s ops=%u\\n", (unsigned)segmentIndex, step, (unsigned)segmentConfig->nOps);
        else
            std::fprintf(stderr, "[vk-dtor] seg=%u op=%u step=%s code=%s\\n", (unsigned)segmentIndex, (unsigned)opIndex, step, opCodeToString(segmentConfig->ops[opIndex].code));
        std::fflush(stderr);
    };
    trace("waitIdle");
    context->device.waitIdle();
    if (commandBuffer) { trace("freeCommandBuffers"); context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer); }
    if (descriptorPool) { trace("destroyDescriptorPool"); context->device.destroyDescriptorPool(descriptorPool); }

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (pipelines[opIndex]) { trace("destroyPipeline", opIndex); context->device.destroyPipeline(pipelines[opIndex]); }
        if (pipelineLayouts[opIndex]) { trace("destroyPipelineLayout", opIndex); context->device.destroyPipelineLayout(pipelineLayouts[opIndex]); }
    }
    if (fence) { trace("destroyFence"); context->device.destroyFence(fence); }
    if (pipelineCache) { trace("destroyPipelineCache"); context->device.destroyPipelineCache(pipelineCache); }
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (descriptorSetLayouts[opIndex]) { trace("destroyDescriptorSetLayout", opIndex); context->device.destroyDescriptorSetLayout(descriptorSetLayouts[opIndex]); }
        if (shaderModules[opIndex]) { trace("destroyShaderModule", opIndex); context->device.destroyShaderModule(shaderModules[opIndex]); }
    }
    trace("done");
    VULKAN_TRACE("Destroyed segment");
}
"""
if old not in s and new not in s:
    raise RuntimeError("destructor pattern not found")
if new not in s:
    s = s.replace(old, new, 1)

p.write_text(s)
print("added optional Vulkan destructor trace")
