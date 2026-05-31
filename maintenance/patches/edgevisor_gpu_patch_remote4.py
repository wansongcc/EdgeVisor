from pathlib import Path

p = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor/src/nn/nn-vulkan.cpp")
s = p.read_text()

old = """NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
    context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);
"""
new = """NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
    context->device.waitIdle();
    context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);
"""
if new not in s:
    if old not in s:
        raise RuntimeError("destructor pattern not found")
    s = s.replace(old, new, 1)

p.write_text(s)
print("added Vulkan segment waitIdle before teardown")
