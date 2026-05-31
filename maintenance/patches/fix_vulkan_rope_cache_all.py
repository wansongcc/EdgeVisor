from pathlib import Path

p = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor/src/nn/nn-vulkan.cpp")
s = p.read_text()

old = """static NnByte *findFirstOpConfig(NnNodeConfig *nodeConfig, NnOpCode opCode) {
    for (NnUint i = 0; i < nodeConfig->nSegments; i++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[i];
        for (NnUint j = 0; j < segmentConfig->nOps; j++) {
            if (segmentConfig->ops[j].code == opCode)
                return segmentConfig->ops[j].config;
        }
    }
    return nullptr;
}
"""

new = """static void collectOpConfigs(std::vector<NnByte *> &out, NnNodeConfig *nodeConfig, NnOpCode opCode) {
    if (nodeConfig == nullptr) return;
    for (NnUint i = 0; i < nodeConfig->nSegments; i++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[i];
        for (NnUint j = 0; j < segmentConfig->nOps; j++) {
            if (segmentConfig->ops[j].code == opCode)
                out.push_back(segmentConfig->ops[j].config);
        }
    }
}
"""

if old not in s:
    raise SystemExit("findFirstOpConfig block not found")
s = s.replace(old, new, 1)

old2 = """    NnRopeOpConfig *ropeLlamaOpConfig = (NnRopeOpConfig *)findFirstOpConfig(nodeConfig, OP_ROPE);
    if (ropeLlamaOpConfig != nullptr) {
        assert(ropeLlamaOpConfig->ropeCacheBufferIndex < nodeConfig->nBuffers);
        NnVulkanBuffer *buffer = buffers[ropeLlamaOpConfig->ropeCacheBufferIndex].get();
        std::vector<NnByte> ropeCache(ropeLlamaOpConfig->slice.cacheSize.nBytes);
        fullfillRopeCache(ropeLlamaOpConfig, (float *)ropeCache.data());
        buffer->write(ropeCache.data());
    }
"""

new2 = """    std::vector<NnByte *> ropeOpConfigs;
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
"""

if old2 not in s:
    raise SystemExit("rope cache init block not found")
s = s.replace(old2, new2, 1)

p.write_text(s)
print("patched src/nn/nn-vulkan.cpp rope cache initialization")
