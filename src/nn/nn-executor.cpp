#include <cassert>
#include <cstring>
#include <algorithm>
#include "nn-executor.hpp"
#include "nn-cpu.hpp"

// Segment kind codes for per-layer compute profiling.
static constexpr NnByte SEG_KIND_OTHER = 0;
static constexpr NnByte SEG_KIND_ATTN  = 1;
static constexpr NnByte SEG_KIND_FFN   = 2;

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
    : netExecution(netExecution), nodeConfig(nodeConfig), segments(nodeConfig->nSegments), steps(), segmentKinds(), threads(nullptr)
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
    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        segmentKinds[s] = classifySegmentKind(nodeConfig->segments[s]);
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
    context.nSegments = (this->nodeConfig != nullptr) ? this->nodeConfig->nSegments : 0u;
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

void NnExecutor::applyPartitionPlan(const NnUnevenPartitionPlan *plan) {
    setPartitionPlan(plan);
    refreshPointers();
}

NnUint NnExecutor::getTotalTime(NnExecutorStepType type) {
    assert((NnUint)type < N_STEP_TYPES);
    return context.totalTime[type];
}
