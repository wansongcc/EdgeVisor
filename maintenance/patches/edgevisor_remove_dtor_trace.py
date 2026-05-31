from pathlib import Path

p = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor/src/nn/nn-vulkan.cpp")
s = p.read_text()
s = s.replace("#include <cstdlib>\n", "")

start = s.find("NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {")
if start < 0:
    raise RuntimeError("destructor start not found")
end = s.find("\n\nvoid NnVulkanDeviceSegment::loadWeight", start)
if end < 0:
    raise RuntimeError("destructor end not found")

clean = """NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
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

s = s[:start] + clean + s[end:]
p.write_text(s)
print("removed temporary Vulkan destructor trace")
