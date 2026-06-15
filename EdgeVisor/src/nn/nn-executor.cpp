#include <cassert>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <sstream>
#include <string>
#include <vector>
#include "nn-executor.hpp"
#include "nn-cpu.hpp"
#include "ablation.hpp"

// Segment kind codes for per-layer compute profiling.
static constexpr NnByte SEG_KIND_OTHER = 0;
static constexpr NnByte SEG_KIND_ATTN  = 1;
static constexpr NnByte SEG_KIND_FFN   = 2;
static constexpr NnByte SEG_ROLE_UNGUARDED = 0;
static constexpr NnByte SEG_ROLE_PRIMARY = 1;
static constexpr NnByte SEG_ROLE_REDUNDANT = 2;

static inline bool nameHas(const char *name, const char *needle) {
    if (name == nullptr || needle == nullptr) return false;
    return std::strstr(name, needle) != nullptr;
}

static NnByte classifySegmentKind(const NnSegmentConfig &seg) {
    bool hasAttn = false;
    bool hasFfn = false;
    for (NnUint i = 0; i < seg.nOps; ++i) {
        const char *n = seg.ops[i].name;
        // FFN / MoE markers
        if (nameHas(n, "block_matmul_w1") || nameHas(n, "block_matmul_w2") || nameHas(n, "block_matmul_w3") ||
            nameHas(n, "block_moe_") || nameHas(n, "moe_") || nameHas(n, "_moe_")) {
            hasFfn = true;
        }
        // Attention markers
        if (nameHas(n, "block_matmul_q") || nameHas(n, "block_matmul_k") || nameHas(n, "block_matmul_v") ||
            nameHas(n, "block_matmul_wo") || nameHas(n, "block_multihead_att") || nameHas(n, "block_rope_") ||
            nameHas(n, "_att")) {
            hasAttn = true;
        }
    }
    if (hasFfn) return SEG_KIND_FFN;
    if (hasAttn) return SEG_KIND_ATTN;
    return SEG_KIND_OTHER;
}

static const char *segmentKindToString(NnByte kind) {
    if (kind == SEG_KIND_ATTN) return "attn";
    if (kind == SEG_KIND_FFN) return "ffn";
    return "other";
}

static const char *segmentRoleToString(NnByte role) {
    if (role == SEG_ROLE_PRIMARY) return "primary";
    if (role == SEG_ROLE_REDUNDANT) return "redundant";
    return "unguarded";
}

static void inferActiveAndRedundantLayer(const NnSegmentConfig *seg, int *activeLayer, int *redundantLayer) {
    if (activeLayer != nullptr) *activeLayer = -1;
    if (redundantLayer != nullptr) *redundantLayer = -1;
    if (seg == nullptr || seg->ops == nullptr) return;

    for (NnUint i = 0; i < seg->nOps; ++i) {
        const NnOpConfig &op = seg->ops[i];
        const bool isRedundant = nameHas(op.name, "runtime_redundant_");
        if (isRedundant) {
            if (redundantLayer != nullptr && *redundantLayer < 0) {
                *redundantLayer = (int)op.index;
            }
        } else {
            if (activeLayer != nullptr && *activeLayer < 0) {
                *activeLayer = (int)op.index;
            }
        }
    }
}

static bool segmentHasOpName(const NnSegmentConfig *seg, const char *needle) {
    if (seg == nullptr || seg->ops == nullptr || needle == nullptr) return false;
    for (NnUint i = 0; i < seg->nOps; ++i) {
        if (nameHas(seg->ops[i].name, needle)) return true;
    }
    return false;
}

static bool parseEnvBoolOr(const char *name, bool fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "False") == 0)
        return false;
    return true;
}

static bool isSegRuntimePrintEnabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = parseEnvBoolOr("DLLAMA_SEG_RUNTIME_PRINT", false) ? 1 : 0;
    }
    return cached == 1;
}

static std::unordered_set<NnUint> parseLayerSetEnv(const char *name) {
    std::unordered_set<NnUint> out;
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return out;

    std::stringstream ss{std::string(v)};
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        try {
            int x = std::stoi(token);
            if (x >= 0) out.insert((NnUint)x);
        } catch (...) {
        }
    }
    return out;
}

static NnByte classifySegmentRuntimeRole(const NnSegmentConfig *seg, NnByte segKind) {
    if (seg == nullptr) return SEG_ROLE_UNGUARDED;
    int activeLayer = -1;
    int redundantLayer = -1;
    inferActiveAndRedundantLayer(seg, &activeLayer, &redundantLayer);

    if (redundantLayer >= 0) {
        return SEG_ROLE_REDUNDANT;
    }

    const bool looksPrimaryLayerSeg =
        (segKind == SEG_KIND_ATTN || segKind == SEG_KIND_FFN) ||
        segmentHasOpName(seg, "plan_barrier") ||
        segmentHasOpName(seg, "plan_apply");

    if (looksPrimaryLayerSeg && activeLayer >= 0) {
        return SEG_ROLE_PRIMARY;
    }
    return SEG_ROLE_UNGUARDED;
}

static NnUint inferMaxLayerIndex(const NnNodeConfig *nodeConfig) {
    NnUint maxIdx = 0u;
    if (nodeConfig == nullptr || nodeConfig->segments == nullptr) return 0u;
    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        const NnSegmentConfig &seg = nodeConfig->segments[s];
        for (NnUint i = 0; i < seg.nOps; ++i) {
            maxIdx = std::max(maxIdx, seg.ops[i].index);
        }
    }
    return maxIdx;
}

struct PointerBindingWorkStats {
    uint64_t bindingUpdates = 0u;
    uint64_t materializedBytes = 0u;
    uint64_t fallbackCount = 0u;
};

struct PointerRebuildDescriptor {
    uint32_t opCode;
    uint32_t opIndex;
    uint32_t inputKey;
    uint32_t outputKey;
    uint32_t weightHash;
    uint32_t dependencyMask;
    uint64_t descriptorHash;
};

static volatile uint64_t gPointerBindingSink = 0u;

static uint64_t parseEnvUint64Or(const char *name, uint64_t fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v) return fallback;
    return (uint64_t)parsed;
}

static uint64_t estimateOpWeightBytes(const NnOpConfig &op) {
    if (op.weightSize.nBytes > 0u) return (uint64_t)op.weightSize.nBytes;
    if (op.weightSize.length > 0u) return (uint64_t)op.weightSize.length * sizeof(float);
    return 0u;
}

static uint64_t estimateNodeWeightBytes(const NnNodeConfig *nodeConfig) {
    uint64_t total = 0u;
    if (nodeConfig == nullptr || nodeConfig->segments == nullptr) return total;
    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        const NnSegmentConfig &seg = nodeConfig->segments[s];
        for (NnUint i = 0; i < seg.nOps; ++i) {
            total += estimateOpWeightBytes(seg.ops[i]);
            total += (uint64_t)seg.ops[i].configSize;
        }
    }
    return total;
}

static PointerBindingWorkStats simulateOperatorRebuildBindingWork(const NnNodeConfig *nodeConfig) {
    PointerBindingWorkStats stats;
    if (nodeConfig == nullptr || nodeConfig->segments == nullptr) return stats;

    uint64_t totalOps = 0u;
    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        totalOps += (uint64_t)nodeConfig->segments[s].nOps;
    }
    if (totalOps == 0u) return stats;

    const uint64_t rounds = parseEnvUint64Or("EDGEVISOR_OPERATOR_REBUILD_ROUNDS", 96u);
    std::vector<PointerRebuildDescriptor> graph;
    graph.reserve((size_t)totalOps);
    uint64_t checksum = 1469598103934665603ull;

    for (uint64_t r = 0u; r < rounds; ++r) {
        graph.clear();
        for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
            const NnSegmentConfig &seg = nodeConfig->segments[s];
            for (NnUint i = 0; i < seg.nOps; ++i) {
                const NnOpConfig &op = seg.ops[i];
                PointerRebuildDescriptor d;
                d.opCode = (uint32_t)op.code;
                d.opIndex = (uint32_t)op.index;
                d.inputKey = ((uint32_t)op.input.source << 24) ^ ((uint32_t)op.input.type << 16) ^
                             ((uint32_t)op.input.sliceTag << 8) ^ (uint32_t)op.input.pointerIndex;
                d.outputKey = ((uint32_t)op.output.source << 24) ^ ((uint32_t)op.output.type << 16) ^
                              ((uint32_t)op.output.sliceTag << 8) ^ (uint32_t)op.output.pointerIndex;
                d.weightHash = (uint32_t)((estimateOpWeightBytes(op) >> 6) ^ op.configSize ^ (uint64_t)(s * 131u + i));
                d.dependencyMask = (uint32_t)((d.inputKey * 16777619u) ^ (d.outputKey + d.weightHash + (uint32_t)r));
                d.descriptorHash = ((uint64_t)d.inputKey << 32) ^ d.outputKey ^ ((uint64_t)d.weightHash << 7) ^ r;
                graph.push_back(d);
            }
        }

        for (size_t i = 0; i < graph.size(); ++i) {
            PointerRebuildDescriptor &d = graph[i];
            const size_t prev = (i == 0u) ? (graph.size() - 1u) : (i - 1u);
            const size_t next = (i + 1u == graph.size()) ? 0u : (i + 1u);
            d.dependencyMask ^= graph[prev].outputKey + graph[next].inputKey + (uint32_t)i;
            d.descriptorHash ^= ((uint64_t)d.dependencyMask << 17) | ((uint64_t)d.opCode << 3);
            checksum ^= d.descriptorHash + 0x9e3779b97f4a7c15ull + (checksum << 6) + (checksum >> 2);
        }
    }

    gPointerBindingSink ^= checksum;
    stats.bindingUpdates = totalOps;
    stats.fallbackCount = 1u;
    return stats;
}

static PointerBindingWorkStats simulateWeightRematerializeBindingWork(const NnNodeConfig *nodeConfig) {
    PointerBindingWorkStats stats;
    uint64_t targetBytes = parseEnvUint64Or("EDGEVISOR_WEIGHT_REMATERIALIZE_BYTES", 0u);
    if (targetBytes == 0u) {
        const uint64_t estimated = estimateNodeWeightBytes(nodeConfig);
        targetBytes = estimated > 0u ? estimated : (64ull * 1024ull * 1024ull);
    }
    const uint64_t capBytes = parseEnvUint64Or("EDGEVISOR_WEIGHT_REMATERIALIZE_MAX_BYTES", 256ull * 1024ull * 1024ull);
    if (capBytes > 0u && targetBytes > capBytes) targetBytes = capBytes;
    const uint64_t passes = std::max<uint64_t>(1u, parseEnvUint64Or("EDGEVISOR_WEIGHT_REMATERIALIZE_PASSES", 1u));
    const uint64_t chunkBytesRaw = parseEnvUint64Or("EDGEVISOR_WEIGHT_REMATERIALIZE_CHUNK_BYTES", 8ull * 1024ull * 1024ull);
    const size_t chunkBytes = (size_t)std::max<uint64_t>(1024u, std::min<uint64_t>(chunkBytesRaw, targetBytes));

    std::vector<NnByte> host(chunkBytes);
    std::vector<NnByte> device(chunkBytes);
    for (size_t i = 0u; i < host.size(); ++i) {
        host[i] = (NnByte)((i * 131u + 17u) & 0xffu);
    }

    uint64_t copied = 0u;
    uint64_t checksum = 1099511628211ull;
    for (uint64_t p = 0u; p < passes; ++p) {
        uint64_t passCopied = 0u;
        while (passCopied < targetBytes) {
            const size_t n = (size_t)std::min<uint64_t>((uint64_t)chunkBytes, targetBytes - passCopied);
            std::memcpy(device.data(), host.data(), n);
            if (n > 0u) {
                device[0] ^= (NnByte)((copied + p) & 0xffu);
                checksum ^= (uint64_t)device[0] + ((uint64_t)device[n - 1u] << 8) + copied;
                checksum *= 1099511628211ull;
            }
            copied += (uint64_t)n;
            passCopied += (uint64_t)n;
        }
    }

    gPointerBindingSink ^= checksum;
    stats.bindingUpdates = (nodeConfig != nullptr) ? (uint64_t)nodeConfig->nSegments : 0u;
    stats.materializedBytes = targetBytes * passes;
    stats.fallbackCount = 1u;
    return stats;
}

void NnFakeNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    // Nothing
}

NnNetExecution::NnNetExecution(NnUint nThreads, NnNetConfig *netConfig) {
    this->nThreads = nThreads;
    this->nBatches = netConfig->nBatches;
    this->nPipes = netConfig->nPipes;
    this->batchSize = 0; // This value must be overwritten before calling forward

    pipes = new NnByte *[netConfig->nPipes];
    for (NnUint pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &netConfig->pipes[pipeIndex];
        NnByte *pipe = new NnByte[pipeConfig->size.nBytes];
        std::memset(pipe, 0, pipeConfig->size.nBytes);
        pipes[pipeIndex] = pipe;
    }

    layerPerf = nullptr;
}

NnNetExecution::~NnNetExecution() {
    if (layerPerf != nullptr) {
        delete layerPerf;
        layerPerf = nullptr;
    }
    for (NnUint pipeIndex = 0; pipeIndex < nPipes; pipeIndex++)
        delete[] pipes[pipeIndex];
    delete[] pipes;
}

void NnNetExecution::setBatchSize(NnUint batchSize) {
    assert(batchSize <= nBatches);
    this->batchSize = batchSize;
}

NnExecutorDevice::NnExecutorDevice(NnDevice *device, int segmentFrom, int segmentTo) {
    this->device = std::unique_ptr<NnDevice>(device);
    this->segmentFrom = segmentFrom;
    this->segmentTo = segmentTo;
}

NnExecutorException::NnExecutorException(const std::string message)
    : std::runtime_error(message) 
{}

NnExecutor::NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, std::vector<NnExecutorDevice> *devices, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark)
    : netExecution(netExecution), nodeConfig(nodeConfig), segments(nodeConfig->nSegments), steps(), segmentKinds(), segmentRuntimeRoles(), segmentLayerIndex(), segmentHasExecOps(), segmentEnabled(nullptr), threads(nullptr)
{
    NnUint maxNThreads = 0;
    for (NnExecutorDevice &d : *devices) {
        if (d.device->maxNThreads() > maxNThreads)
            maxNThreads = d.device->maxNThreads();
    }
    if (netExecution->nThreads > maxNThreads)
        throw std::invalid_argument("This configuration supports max " + std::to_string(maxNThreads) + " threads");

    // Lazily allocate per-layer profiling state when benchmark/profile is enabled.
    if (benchmark && this->netExecution != nullptr && this->netExecution->layerPerf == nullptr) {
        const NnUint maxLayer = inferMaxLayerIndex(nodeConfig);
        const NnUint nLayers = maxLayer + 1u;
        auto *st = new NnNetExecution::LayerPerfState();
        st->nLayers = nLayers;
        st->attnUs.resize(nLayers, 0ull);
        st->ffnUs.resize(nLayers, 0ull);
        this->netExecution->layerPerf = st;
    }

    bool useSynchronizer = netConfig->nNodes > 1;

    // Build segment kind table for this node config.
    segmentKinds.assign(nodeConfig->nSegments, SEG_KIND_OTHER);
    segmentRuntimeRoles.assign(nodeConfig->nSegments, SEG_ROLE_UNGUARDED);
    segmentLayerIndex.assign(nodeConfig->nSegments, -1);
    segmentHasExecOps.assign(nodeConfig->nSegments, 0u);
    std::vector<int> segmentActiveLayers(nodeConfig->nSegments, -1);
    const std::unordered_set<NnUint> skipPrimaryLayers = parseLayerSetEnv("DLLAMA_RUNTIME_PRIMARY_SKIP_LAYERS");
    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        segmentKinds[s] = classifySegmentKind(nodeConfig->segments[s]);
        segmentRuntimeRoles[s] = classifySegmentRuntimeRole(&nodeConfig->segments[s], segmentKinds[s]);
        int activeLayer = -1;
        int redundantLayer = -1;
        inferActiveAndRedundantLayer(&nodeConfig->segments[s], &activeLayer, &redundantLayer);
        segmentActiveLayers[s] = activeLayer;
        segmentLayerIndex[s] = (activeLayer >= 0) ? activeLayer : redundantLayer;
        segmentHasExecOps[s] = (nodeConfig->segments[s].nOps > 0) ? 1u : 0u;
    }

    segmentEnabled.reset(new std::atomic_uint8_t[nodeConfig->nSegments]);
    const bool enablePrimaryByDefault = parseEnvBoolOr("DLLAMA_RUNTIME_ACTIVE_SEG_ENABLED", true);
    const bool enableRedundantByDefault = parseEnvBoolOr("DLLAMA_RUNTIME_REDUNDANT_SEG_ENABLED", false);
    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        const NnByte role = segmentRuntimeRoles[s];
        const bool enabled =
            (role == SEG_ROLE_PRIMARY) ? enablePrimaryByDefault :
            (role == SEG_ROLE_REDUNDANT) ? enableRedundantByDefault :
            true;
        bool finalEnabled = enabled;
        if (role == SEG_ROLE_PRIMARY && segmentActiveLayers[s] >= 0 &&
            skipPrimaryLayers.find((NnUint)segmentActiveLayers[s]) != skipPrimaryLayers.end()) {
            finalEnabled = false;
        }
        segmentEnabled[s].store(finalEnabled ? 1u : 0u, std::memory_order_relaxed);
    }

    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnDevice *device = nullptr;
        for (NnExecutorDevice &d : *devices) {
            if (
                (d.segmentFrom == -1 && d.segmentTo == -1) ||
                (segmentIndex >= d.segmentFrom && segmentIndex <= d.segmentTo)
            ) {
                device = d.device.get();
                break;
            }
        }
        if (device == nullptr)
            throw std::invalid_argument("Cannot locate device for segment " + std::to_string(segmentIndex));

        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        const NnByte segKind = segmentKinds[segmentIndex];
        const NnByte segRole = segmentRuntimeRoles[segmentIndex];
        const bool hasExecOps = segmentConfig->nOps > 0;
        const bool hasSyncOps = useSynchronizer && segmentConfig->nSyncs > 0;
        const bool segEnabled = segmentEnabled[segmentIndex].load(std::memory_order_relaxed) != 0u;
        int activeLayer = -1;
        int redundantLayer = -1;
        inferActiveAndRedundantLayer(segmentConfig, &activeLayer, &redundantLayer);

        std::printf(
            "🧱 [executor-init] node=%u segment=%u kind=%s role=%s enabled=%u activeLayer=%d redundantLayer=%d nOps=%u nSyncs=%u createExec=%u createSync=%u\n",
            (unsigned)(nodeConfig ? nodeConfig->nodeIndex : 0u),
            (unsigned)segmentIndex,
            segmentKindToString(segKind),
            segmentRoleToString(segRole),
            segEnabled ? 1u : 0u,
            activeLayer,
            redundantLayer,
            (unsigned)segmentConfig->nOps,
            (unsigned)segmentConfig->nSyncs,
            hasExecOps ? 1u : 0u,
            hasSyncOps ? 1u : 0u);

        if (segmentConfig->nOps > 0) {
            NnDeviceSegment *segment = device->createSegment(segmentIndex);
            segments[segmentIndex] = std::unique_ptr<NnDeviceSegment>(segment);

            for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
                steps.push_back(NnExecutorStep{ STEP_EXECUTE_OP, segment, opIndex, &segmentConfig->ops[opIndex], segmentIndex });
            }
        }
        if (useSynchronizer && segmentConfig->nSyncs > 0){
            steps.push_back(NnExecutorStep{ STEP_SYNC_NODES, nullptr, segmentIndex, nullptr, segmentIndex });
        }

    }

    steps.shrink_to_fit();

    context.nThreads = netExecution->nThreads;
    context.synchronizer = synchronizer;
    context.nSteps = (NnUint)steps.size();
    context.steps = steps.data();
    context.layerPerf = (this->netExecution != nullptr) ? this->netExecution->layerPerf : nullptr;
    context.segmentKinds = segmentKinds.empty() ? nullptr : segmentKinds.data();
    context.segmentRuntimeRoles = segmentRuntimeRoles.empty() ? nullptr : segmentRuntimeRoles.data();
    context.segmentEnabled = segmentEnabled.get();
    context.segmentLayerIndex = segmentLayerIndex.empty() ? nullptr : segmentLayerIndex.data();
    context.segmentHasExecOps = segmentHasExecOps.empty() ? nullptr : segmentHasExecOps.data();
    context.nSegments = (this->nodeConfig != nullptr) ? this->nodeConfig->nSegments : 0u;
    context.currentStepIndex.store(0u);
    context.doneThreadCount.store(0u);
    context.isAlive.store(false);
    context.batchSize = 0u;
    if (benchmark)
        context.timer = new Timer();
    else
        context.timer = nullptr;

    threads = new NnExecutorThread[netExecution->nThreads];
    for (NnUint threadIndex = 0; threadIndex < netExecution->nThreads; threadIndex++) {
        NnExecutorThread *thread = &threads[threadIndex];
        thread->threadIndex = threadIndex;
        thread->context = &context;
    }
}

NnExecutor::~NnExecutor() {
    if (context.timer != nullptr)
        delete context.timer;
    delete[] threads;
}

void NnExecutor::loadWeight(const char *name, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        for (NnUint i = 0; i < segmentConfig->nOps; i++) {
            NnOpConfig *opConfig = &segmentConfig->ops[i];
            if (opConfig->index == opIndex && std::strcmp(opConfig->name, name) == 0) {
                NnDeviceSegment *segment = segments[segmentIndex].get();
                assert(segment != nullptr);
                segment->loadWeight(i, offset, nBytes, weight);
                return;
            }
        }
    }
    throw std::invalid_argument(
        "Cannot locate op name='" + std::string(name) +
        "' index=" + std::to_string(opIndex) +
        " on node=" + std::to_string(nodeConfig ? nodeConfig->nodeIndex : 0u) +
        ". (Likely plan/config mismatch between root and worker binaries or --ratios)"
    );
}

inline void executeStep(NnExecutorStep *step, NnUint nThreads, NnExecutorThread *thread, NnExecutorContext *context) {
    const bool segRuntimePrint = isSegRuntimePrintEnabled();
    const bool isThread0 = (thread != nullptr && thread->threadIndex == 0u);
    const bool canInspectSeg =
        (context != nullptr && step != nullptr && step->segmentIndex < context->nSegments &&
         context->segmentEnabled != nullptr && context->segmentLayerIndex != nullptr &&
         context->segmentHasExecOps != nullptr && context->segmentRuntimeRoles != nullptr);
    const bool isFirstExecOpStep = (step->type == STEP_EXECUTE_OP && step->arg0 == 0u);
    const bool isSyncOnlyStep = (step->type == STEP_SYNC_NODES && canInspectSeg && context->segmentHasExecOps[step->segmentIndex] == 0u);

    if (context != nullptr && context->segmentEnabled != nullptr && step->segmentIndex < context->nSegments) {
        const bool enabled = context->segmentEnabled[step->segmentIndex].load(std::memory_order_relaxed) != 0u;

        if (segRuntimePrint && isThread0 && canInspectSeg && (isFirstExecOpStep || isSyncOnlyStep)) {
            const int layer = context->segmentLayerIndex[step->segmentIndex];
            const NnByte role = context->segmentRuntimeRoles[step->segmentIndex];
            std::printf("🧮 [seg-runtime] segment=%u layer=%d role=%s status=%s\n",
                (unsigned)step->segmentIndex,
                layer,
                segmentRoleToString(role),
                enabled ? "执行" : "休眠");
        }

        if (!enabled) {
            return;
        }
    }

    if (step->type == STEP_EXECUTE_OP) {
        step->segment->forward(step->arg0, nThreads, thread->threadIndex, context->batchSize);
    } else if (step->type == STEP_SYNC_NODES) {
        context->synchronizer->sync(step->arg0, nThreads, thread->threadIndex);
    } else {
        throw std::invalid_argument("Unsupported step type");
    }
}

static inline void *executorThreadHandler(void *arg) {
    NnExecutorThread *thread = (NnExecutorThread *)arg;
    NnExecutorContext *context = thread->context;
    NnUint nThreads = context->nThreads;
    NnUint doneCount = nThreads - 1;

    while (context->isAlive.load()) {
        const unsigned int currentStepIndex = context->currentStepIndex.load();
        if (currentStepIndex == context->nSteps)
            break;

        NnExecutorStep *step = &context->steps[currentStepIndex];
        try {
            executeStep(step, nThreads, thread, context);
        } catch (const std::runtime_error &e) {
            context->isAlive.store(false);
            printf("Execution error: %s\n", e.what());
            break;
        }

        NnUint currentCount = context->doneThreadCount.fetch_add(1);
        if (currentCount == doneCount) {
            if (context->timer != nullptr) {
                NnUint time = context->timer->elapsedMicroseconds();
                context->totalTime[step->type] += time;

                // Per-layer compute profiling (ATTN/FFN only).
                if (context->layerPerf != nullptr && context->segmentKinds != nullptr && step->type == STEP_EXECUTE_OP && step->opConfig != nullptr) {
                    const NnUint layerIndex = step->opConfig->index;
                    if (layerIndex < context->layerPerf->nLayers && step->segmentIndex < context->nSegments) {
                        const NnByte kind = context->segmentKinds[step->segmentIndex];
                        if (kind == SEG_KIND_ATTN) {
                            context->layerPerf->attnUs[layerIndex] += (unsigned long long)time;
                        } else if (kind == SEG_KIND_FFN) {
                            context->layerPerf->ffnUs[layerIndex] += (unsigned long long)time;
                        }
                    }
                }

                context->timer->reset();
            }

            // Optional hook after a full sync step (safe: all threads finished sync()).
            if (step->type == STEP_SYNC_NODES && context->synchronizer != nullptr) {
                context->synchronizer->onSyncStepComplete(step->arg0);
            }

            context->doneThreadCount.store(0);
            context->currentStepIndex.fetch_add(1);
        } else {
            while (
                context->currentStepIndex.load() == currentStepIndex &&
                context->isAlive.load()
            );
        }
    }
    return nullptr;
}

void NnExecutor::forward() {
    assert(netExecution->batchSize > 0);

    NnUint nThreads = netExecution->nThreads;
    context.isAlive.exchange(true);
    context.currentStepIndex.exchange(0);
    context.doneThreadCount.exchange(0);
    context.batchSize = netExecution->batchSize;

    if (context.timer != nullptr) {
        std::memset(context.totalTime, 0, sizeof(context.totalTime));
        context.timer->reset();
    }

    if (netExecution != nullptr && netExecution->layerPerf != nullptr) {
        netExecution->layerPerf->reset();
    }

    NnUint threadIndex;
    for (threadIndex = 1; threadIndex < nThreads; threadIndex++) {
        int result = pthread_create(&threads[threadIndex].handler, NULL, (PthreadFunc)executorThreadHandler, (void *)&threads[threadIndex]);
        assert(result == 0 && "Failed to create thread");
    }
    executorThreadHandler((void *)&threads[0]);
    for (threadIndex = 1; threadIndex < nThreads; threadIndex++)
        pthread_join(threads[threadIndex].handler, NULL);

    if (!context.isAlive.load())
        throw NnExecutorException("Execution failed in one of the threads");
}

void NnExecutor::refreshPointers() {
    if (context.isAlive.load()) {
        throw std::runtime_error("Cannot refresh pointers while executor is running");
    }
    const EdgeVisorAblationConfig &cfg = getEdgeVisorAblationConfig();
    EdgeVisorAblationEvent event;
    event.eventId = "binding_refresh";
    event.selectedPolicy = std::string("pointer_") + toString(cfg.pointerSwizzlingMode);
    event.bindingUpdateCount = nodeConfig != nullptr ? nodeConfig->nSegments : 0u;

    const auto t0 = std::chrono::high_resolution_clock::now();
    PointerBindingWorkStats stats;
    if (cfg.pointerSwizzlingMode == PointerSwizzlingMode::OPERATOR_REBUILD) {
        stats = simulateOperatorRebuildBindingWork(nodeConfig);
        event.fallbackReason = "operator_rebuild_substitutes_lightweight_pointer_swizzling";
        event.stallTimeMs = 0.0;
    } else if (cfg.pointerSwizzlingMode == PointerSwizzlingMode::WEIGHT_REMATERIALIZE) {
        stats = simulateWeightRematerializeBindingWork(nodeConfig);
        event.fallbackReason = "weight_rematerialize_substitutes_lightweight_pointer_swizzling";
        event.stallTimeMs = 0.0;
    }

    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnDeviceSegment *segment = segments[segmentIndex].get();
        if (segment == nullptr)
            continue;
        segment->refreshPointers();
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    event.tBindMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (stats.bindingUpdates > 0u) event.bindingUpdateCount = stats.bindingUpdates;
    event.materializedBytes = stats.materializedBytes;
    event.fallbackCount = stats.fallbackCount;
    edgevisorAblationLogEvent(event);
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

void NnExecutor::applyPartitionPlan(const NnUnevenPartitionPlan *plan) {
    setPartitionPlan(plan);
    refreshPointers();
}

void NnExecutor::setSegmentEnabled(NnUint segmentIndex, bool enabled) {
    if (segmentEnabled == nullptr || nodeConfig == nullptr || segmentIndex >= nodeConfig->nSegments) {
        throw std::out_of_range("segmentIndex out of range");
    }
    segmentEnabled[segmentIndex].store(enabled ? 1u : 0u, std::memory_order_relaxed);
}

void NnExecutor::setRuntimeLayerGate(bool enablePrimarySegments, bool enableRedundantSegments) {
    if (segmentEnabled == nullptr || nodeConfig == nullptr) return;
    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        const NnByte role = (s < segmentRuntimeRoles.size()) ? segmentRuntimeRoles[s] : SEG_ROLE_UNGUARDED;
        if (role == SEG_ROLE_PRIMARY) {
            segmentEnabled[s].store(enablePrimarySegments ? 1u : 0u, std::memory_order_relaxed);
        } else if (role == SEG_ROLE_REDUNDANT) {
            segmentEnabled[s].store(enableRedundantSegments ? 1u : 0u, std::memory_order_relaxed);
        }
    }
}

void NnExecutor::setPrimaryLayerEnabled(NnUint layerIndex, bool enabled) {
    if (segmentEnabled == nullptr || nodeConfig == nullptr) return;
    NnUint matched = 0u;
    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        if (s >= segmentRuntimeRoles.size() || segmentRuntimeRoles[s] != SEG_ROLE_PRIMARY) continue;
        const NnSegmentConfig *seg = &nodeConfig->segments[s];
        int activeLayer = -1;
        int redundantLayer = -1;
        inferActiveAndRedundantLayer(seg, &activeLayer, &redundantLayer);
        if (activeLayer >= 0 && (NnUint)activeLayer == layerIndex) {
            segmentEnabled[s].store(enabled ? 1u : 0u, std::memory_order_relaxed);
            matched += 1u;
        }
    }
    std::printf("🛂 [layer-gate] node=%u role=primary layer=%u enabled=%u matchedSegs=%u\n",
        (unsigned)(nodeConfig ? nodeConfig->nodeIndex : 0u),
        (unsigned)layerIndex,
        enabled ? 1u : 0u,
        (unsigned)matched);
    std::fflush(stdout);
}

void NnExecutor::setRedundantLayerEnabled(NnUint layerIndex, bool enabled) {
    if (segmentEnabled == nullptr || nodeConfig == nullptr) return;
    NnUint matched = 0u;
    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        if (s >= segmentRuntimeRoles.size() || segmentRuntimeRoles[s] != SEG_ROLE_REDUNDANT) continue;
        const NnSegmentConfig *seg = &nodeConfig->segments[s];
        int activeLayer = -1;
        int redundantLayer = -1;
        inferActiveAndRedundantLayer(seg, &activeLayer, &redundantLayer);
        if (redundantLayer >= 0 && (NnUint)redundantLayer == layerIndex) {
            segmentEnabled[s].store(enabled ? 1u : 0u, std::memory_order_relaxed);
            matched += 1u;
        }
    }
    std::printf("🛂 [layer-gate] node=%u role=redundant layer=%u enabled=%u matchedSegs=%u\n",
        (unsigned)(nodeConfig ? nodeConfig->nodeIndex : 0u),
        (unsigned)layerIndex,
        enabled ? 1u : 0u,
        (unsigned)matched);
    std::fflush(stdout);
}

bool NnExecutor::exportLayerKvRow(
    NnUint layerIndex,
    NnUint position,
    NnUint kvDim,
    std::vector<float> &kRow,
    std::vector<float> &vRow,
    NnUint rangeStart,
    NnUint rangeLen) {
    const bool partial = rangeLen != 0u;
    const NnUint outDim = partial ? rangeLen : kvDim;
    kRow.assign(outDim, 0.0f);
    vRow.assign(outDim, 0.0f);
    if (nodeConfig == nullptr || segments.empty() || kvDim == 0u || outDim == 0u) return false;
    if (partial && rangeStart + rangeLen > kvDim) return false;

    bool readAny = false;
    std::unordered_set<NnUint> readK;
    std::unordered_set<NnUint> readV;

    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        if (s >= segments.size()) continue;
        NnDeviceSegment *baseSeg = segments[s].get();
        if (baseSeg == nullptr) continue;
        auto *cpuSeg = dynamic_cast<NnCpuDeviceSegment *>(baseSeg);
        if (cpuSeg == nullptr || cpuSeg->device == nullptr) {
            if (baseSeg->exportLayerKvRow(layerIndex, position, kvDim, kRow, vRow, rangeStart, rangeLen)) {
                readAny = true;
            }
            continue;
        }

        NnSegmentConfig *seg = &nodeConfig->segments[s];
        for (NnUint i = 0; i < seg->nOps; ++i) {
            const NnOpConfig &op = seg->ops[i];
            if (op.code != OP_MULTIHEAD_ATT || op.index != layerIndex) continue;
            const auto *cfg = (const NnMultiHeadAttOpConfig *)op.config;
            if (cfg == nullptr) continue;

            auto readOne = [&](NnUint bufIdx, std::vector<float> &dstRow, std::unordered_set<NnUint> &seen) {
                if (seen.find(bufIdx) != seen.end()) return;
                if (bufIdx >= cpuSeg->device->getBufferCount()) return;

                const NnSize3D &bSize = nodeConfig->buffers[bufIdx].size;
                if (bSize.floatType != F_32 || bSize.x == 0u || bSize.y == 0u) return;
                if (position >= bSize.y) return;

                const NnUint srcStart = partial ? rangeStart : cfg->kvStart;
                const NnUint dstStart = partial ? 0u : cfg->kvStart;
                const NnUint needLen = partial ? rangeLen : cfg->kvDim0;
                if (srcStart >= bSize.x || dstStart >= dstRow.size() || needLen == 0u) return;

                const NnUint srcAvail = bSize.x - srcStart;
                const NnUint dstAvail = (NnUint)dstRow.size() - dstStart;
                const NnUint readLen = std::min(needLen, std::min(srcAvail, dstAvail));
                if (readLen == 0u) return;

                const float *buf = (const float *)cpuSeg->device->buffers[bufIdx];
                const float *src = buf + (NnSize)position * (NnSize)bSize.x + (NnSize)srcStart;
                std::memcpy(dstRow.data() + dstStart, src, (size_t)readLen * sizeof(float));
                seen.insert(bufIdx);
                readAny = true;

                std::printf("🧩 [kv-export] node=%u seg=%u layer=%u pos=%u buf=%u srcRange=[%u,%u) dstRange=[%u,%u) partial=%u\n",
                    (unsigned)(nodeConfig ? nodeConfig->nodeIndex : 0u),
                    (unsigned)s,
                    (unsigned)layerIndex,
                    (unsigned)position,
                    (unsigned)bufIdx,
                    (unsigned)srcStart,
                    (unsigned)(srcStart + readLen),
                    (unsigned)dstStart,
                    (unsigned)(dstStart + readLen),
                    partial ? 1u : 0u);
            };

            readOne(cfg->keyCacheBufferIndex, kRow, readK);
            readOne(cfg->valueCacheBufferIndex, vRow, readV);
        }
    }

    if (!readAny) {
        std::printf("⚠️  [kv-export] no source buffers read for layer=%u pos=%u\n",
            (unsigned)layerIndex,
            (unsigned)position);
        std::fflush(stdout);
    } else {
        std::fflush(stdout);
    }
    return readAny;
}

bool NnExecutor::applyTransferredKvRow(
    NnUint layerIndex,
    NnUint position,
    const std::vector<float> &kRow,
    const std::vector<float> &vRow,
    NnUint rangeStart,
    NnUint rangeLen) {
    if (nodeConfig == nullptr || segments.empty()) return false;
    if (kRow.empty() || vRow.empty() || kRow.size() != vRow.size()) return false;
    if (rangeLen != 0u && rangeLen != kRow.size()) return false;

    bool wroteAny = false;
    std::unordered_set<NnUint> writtenK;
    std::unordered_set<NnUint> writtenV;

    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        if (s >= segments.size()) continue;
        NnDeviceSegment *baseSeg = segments[s].get();
        if (baseSeg == nullptr) continue;
        auto *cpuSeg = dynamic_cast<NnCpuDeviceSegment *>(baseSeg);
        if (cpuSeg == nullptr || cpuSeg->device == nullptr) {
            if (baseSeg->applyTransferredKvRow(layerIndex, position, kRow, vRow, rangeStart, rangeLen)) {
                wroteAny = true;
            }
            continue;
        }

        NnSegmentConfig *seg = &nodeConfig->segments[s];
        for (NnUint i = 0; i < seg->nOps; ++i) {
            const NnOpConfig &op = seg->ops[i];
            if (op.code != OP_MULTIHEAD_ATT || op.index != layerIndex) continue;

            const auto *cfg = (const NnMultiHeadAttOpConfig *)op.config;
            if (cfg == nullptr) continue;

            auto writeOne = [&](NnUint bufIdx, const std::vector<float> &srcRow, std::unordered_set<NnUint> &written, const char *tag) {
                if (written.find(bufIdx) != written.end()) return;
                if (bufIdx >= cpuSeg->device->getBufferCount()) return;

                const NnSize3D &bSize = nodeConfig->buffers[bufIdx].size;
                if (bSize.floatType != F_32 || bSize.x == 0u || bSize.y == 0u) return;
                if (position >= bSize.y) return;

                const bool partial = rangeLen != 0u;
                const NnUint dstStart = partial ? rangeStart : cfg->kvStart;
                const NnUint srcStart = partial ? 0u : cfg->kvStart;
                const NnUint needLen = partial ? rangeLen : cfg->kvDim0;
                if (srcStart >= srcRow.size() || dstStart >= bSize.x || needLen == 0u) return;

                const NnUint srcAvail = (NnUint)srcRow.size() - srcStart;
                const NnUint dstAvail = bSize.x - dstStart;
                const NnUint writeLen = std::min(needLen, std::min(srcAvail, dstAvail));
                if (writeLen == 0u) return;

                float *buf = (float *)cpuSeg->device->buffers[bufIdx];
                float *dst = buf + (NnSize)position * (NnSize)bSize.x + (NnSize)dstStart;
                std::memcpy(dst, srcRow.data() + srcStart, (size_t)writeLen * sizeof(float));
                written.insert(bufIdx);
                wroteAny = true;

                std::printf("🧩 [kv-write] node=%u seg=%u layer=%u pos=%u %sBuf=%u bufX=%u srcRange=[%u,%u) dstRange=[%u,%u) partial=%u\n",
                    (unsigned)(nodeConfig ? nodeConfig->nodeIndex : 0u),
                    (unsigned)s,
                    (unsigned)layerIndex,
                    (unsigned)position,
                    tag,
                    (unsigned)bufIdx,
                    (unsigned)bSize.x,
                    (unsigned)srcStart,
                    (unsigned)(srcStart + writeLen),
                    (unsigned)dstStart,
                    (unsigned)(dstStart + writeLen),
                    partial ? 1u : 0u);
                std::fflush(stdout);
            };

            writeOne(cfg->keyCacheBufferIndex, kRow, writtenK, "k");
            writeOne(cfg->valueCacheBufferIndex, vRow, writtenV, "v");
        }
    }

    if (!wroteAny) {
        std::printf("⚠️  [kv-write] no target buffers written for layer=%u pos=%u (mapping check failed)\n",
            (unsigned)layerIndex,
            (unsigned)position);
        std::fflush(stdout);
    }
    return wroteAny;
}

NnUint NnExecutor::getTotalTime(NnExecutorStepType type) {
    assert((NnUint)type < N_STEP_TYPES);
    return context.totalTime[type];
}
