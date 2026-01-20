#include "nn-cpu.hpp"
#include "nn-cpu-ops.hpp"

#ifndef NN_LOGITS_COMM_DATA_LOG
#define NN_LOGITS_COMM_DATA_LOG 0
#endif

#include "nn-core.hpp"
#include <cassert>
#include <iostream> 
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define DEBUG_CPU_OP_QUANTS false

#define BUFFER_ALIGNMENT 64


static NnByte *allocAlignedBuffer(NnSize size) {
    NnByte *buffer;
#ifdef _WIN32
    buffer = (NnByte *)_aligned_malloc(size, BUFFER_ALIGNMENT);
    if (buffer == NULL)
        throw std::runtime_error("_aligned_malloc failed");
#else
    if (posix_memalign((void **)&buffer, BUFFER_ALIGNMENT, size) != 0)
        throw std::runtime_error("posix_memalign failed");
    mlock(buffer, size);
#endif
    return buffer;
}

#if NN_LOGITS_COMM_DATA_LOG
static inline std::uint64_t hashBytes(const void *data, NnSize size) {
    const std::uint8_t *p = (const std::uint8_t *)data;
    std::uint64_t h = 1469598103934665603ull;
    for (NnSize i = 0; i < size; ++i) {
        h ^= (std::uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void logF32Stats(const char *tag, const float *data, NnUint n) {
    if (data == nullptr || n == 0u) return;
    float minv = data[0];
    float maxv = data[0];
    NnUint nnz = 0u;
    for (NnUint i = 0u; i < n; ++i) {
        const float v = data[i];
        if (v != 0.0f) nnz++;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }
    printf("%s n=%u min=%g max=%g nnz=%u\n", tag, n, minv, maxv, nnz);
}
#endif

static void releaseAlignedBuffer(NnByte *buffer) {
#ifdef _WIN32
    _aligned_free(buffer);
#else
    free(buffer);
#endif
}

static NnUint getSplitTotal(const NnDimSplit* split, NnUint nNodes) {
    if (!split || !split->lengths) return 0;
    NnUint sum = 0;
    for(NnUint i=0; i<nNodes; ++i) sum += split->lengths[i];
    return sum;
}

static const NnStageConfig* findStageForNode(const NnUnevenPartitionPlan* plan, NnUint nodeIndex) {
    if (!plan) return nullptr;
    for (NnUint s = 0; s < plan->nStages; ++s) {
        const NnStageConfig* st = &plan->stages[s];
        for (NnUint i = 0; i < st->nNodes; ++i) {
            if (st->nodeIndices[i] == nodeIndex) return st;
        }
    }
    return nullptr;
}

static NnUint findStageRank(const NnStageConfig* stage, NnUint nodeIndex) {
    if (!stage) return 0;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        if (stage->nodeIndices[i] == nodeIndex) return i;
    }
    return 0;
}

static NnUint getSplitTotalForStage(const NnDimSplit* split, const NnStageConfig* stage) {
    if (!split || !split->lengths || !stage) return 0;
    NnUint sum = 0;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        sum += split->lengths[stage->nodeIndices[i]];
    }
    return sum;
}


NnCpuDevice::NnCpuDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan, NnLayerShardingTable *layerSharding) {
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->netExecution = netExecution;
    this->partitionPlan = partitionPlan;
    this->layerSharding = layerSharding;

    printCpuInstructionSet();

    nBuffers = nodeConfig->nBuffers;
    buffers = new NnByte *[nBuffers];
    for (NnUint bufferIndex = 0; bufferIndex < nBuffers; bufferIndex++) {
        NnBufferConfig *config = &nodeConfig->buffers[bufferIndex];
        NnByte *buffer = allocAlignedBuffer(config->size.nBytes);
        buffers[bufferIndex] = buffer;
    }

    bufferFlags = new NnByte[nBuffers];
    std::memset(bufferFlags, 0, nBuffers * sizeof(NnByte));
    
    #ifndef NN_LOGITS_COMM_DATA_LOG
    #define NN_LOGITS_COMM_DATA_LOG 0
    #endif
}

NnCpuDevice::~NnCpuDevice() {
    for (NnUint bufferIndex = 0; bufferIndex < nBuffers; bufferIndex++) {
        releaseAlignedBuffer(buffers[bufferIndex]);
    }
    delete[] buffers;
    delete[] bufferFlags;
}

NnUint NnCpuDevice::maxNThreads() {
    return std::thread::hardware_concurrency();
}

NnDeviceSegment *NnCpuDevice::createSegment(NnUint segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    assert(segmentConfig->nOps > 0);

    std::vector<NnOpQuantType> opQuants(segmentConfig->nOps);
    std::vector<NnCpuOpForward> opForwardLocal(segmentConfig->nOps);
    std::vector<NnSize3D> inputSizes(segmentConfig->nOps);
    std::vector<NnSize3D> outputSizes(segmentConfig->nOps);

    std::vector<std::vector<NnByte *>> inputsPtr(segmentConfig->nOps);
    std::vector<std::vector<NnByte *>> outputsPtr(segmentConfig->nOps);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnSize3D inputSize;
        NnSize3D outputSize;
        inputsPtr[opIndex] = resolvePointer(&inputSize, &opConfig->input);
        outputsPtr[opIndex] = resolvePointer(&outputSize, &opConfig->output);

        // [Patch Start] Logits Pipe 尺寸修正补丁
        // 在非均匀切分模式下，resolvePointer 可能会根据 Pipe 的总大小计算出一个“均匀”的 Output Slice。
        // 但 inputSize (来自本地 Buffer) 是真实的“非均匀”大小。
        // 如果它们不匹配，且是 OP_CAST (常用于输出到 Pipe)，我们信任 Input 的大小。
        if (opConfig->code == OP_CAST && 
            opConfig->output.type == PNTR_BATCHED_SLICE && 
            inputSize.x != outputSize.x) {
            
            // 重新计算 3D 尺寸，保持 type, z, y 不变，仅更新 x (及其衍生的 nBytes)
            outputSize = size3D(outputSize.floatType, outputSize.z, outputSize.y, inputSize.x);
        }
        // [Patch End]

        NnOpQuantType opQuant = getOpQuantType(
            inputSize.floatType,
            opConfig->weightSize.floatType,
            outputSize.floatType);
#if DEBUG_CPU_OP_QUANTS
            printf("%20s %2d: %s\n", opConfig->name, opConfig->index, opQuantTypeToString(opQuant));
#endif
        NnCpuOpForward forward = getCpuOpForward(opConfig->code, opQuant);
        if (forward == nullptr) {
            throw std::invalid_argument(
                std::string("Unsupported CPU op code: ") + opCodeToString(opConfig->code) + 
                ", quant: " + opQuantTypeToString(opQuant) +
                ", op name: " + opConfig->name);
        }
        inputSizes[opIndex] = inputSize;
        outputSizes[opIndex] = outputSize;
        opQuants[opIndex] = opQuant;
        opForwardLocal[opIndex] = forward;
    }

    NnCpuOpForward *opForward = new NnCpuOpForward[segmentConfig->nOps];
    NnCpuOpContext *opContexts = new NnCpuOpContext[segmentConfig->nOps];

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnCpuOpContext *opContext = &opContexts[opIndex];
        NnCpuOpForwardInit opInit = getCpuOpForwardInit(opConfig->code, opQuants[opIndex]);
        opContext->name = opConfig->name;
        opContext->nodeIndex = nodeConfig->nodeIndex;
        opContext->opConfig = opConfig->config;
        opContext->opCode = (NnUint)opConfig->code;
        opContext->opIndex = opIndex;
        opContext->opConfigSize = opConfig->configSize;
        opContext->weightSize = opConfig->weightSize;

        // NOTE: In distributed mode, NnOpConfig is received over the wire and may not carry
        // newer fields (or those fields may be uninitialized). Never trust opConfig->weightAllocBytes.
        // Instead, derive the allocation size from stable metadata:
        // - Default: weightSize.nBytes
        // - Replicated matmul: replicateExpertStrideBytes * nExperts
        NnSize weightAllocBytes = opConfig->weightSize.nBytes;
        if (opConfig->code == OP_MATMUL && opConfig->configSize >= sizeof(NnMatmulOpConfig)) {
            const NnMatmulOpConfig *cfg = (const NnMatmulOpConfig *)opConfig->config;
            if (cfg != nullptr && cfg->replicateMode != 0u) {
                if (cfg->replicateExpertStrideBytes > 0) {
                    // For non-MoE matmul, cfg->nExperts is 0 but we still have one matrix.
                    const size_t nExpertsOr1 = (cfg->nExperts == 0u) ? 1u : (size_t)cfg->nExperts;
                    const size_t total = (size_t)cfg->replicateExpertStrideBytes * nExpertsOr1;
                    weightAllocBytes = (NnSize)total;
                }
            }
        }
        if (weightAllocBytes < opConfig->weightSize.nBytes)
            weightAllocBytes = opConfig->weightSize.nBytes;
        opContext->weightAllocBytes = weightAllocBytes;
        opContext->nBatches = netConfig->nBatches;
        opContext->pipes = netExecution->pipes;
        opContext->pipeConfigs = netConfig->pipes;
        opContext->nPipes = netConfig->nPipes;
        opContext->buffers = buffers;
        opContext->bufferConfigs = nodeConfig->buffers;
        opContext->bufferFlags = bufferFlags;
        opContext->layerSharding = this->layerSharding;

        opContext->input = new NnByte *[inputsPtr[opIndex].size()];
        opContext->inputSize = inputSizes[opIndex];
        opContext->hasInputContinuousMemory = hasPointerContinuousMemory(&opConfig->input);
        std::memcpy(opContext->input, inputsPtr[opIndex].data(), inputsPtr[opIndex].size() * sizeof(NnByte *));

        opContext->output = new NnByte *[outputsPtr[opIndex].size()];
        opContext->outputSize = outputSizes[opIndex];
        opContext->hasOutputContinuousMemory = hasPointerContinuousMemory(&opConfig->output);
        std::memcpy(opContext->output, outputsPtr[opIndex].data(), outputsPtr[opIndex].size() * sizeof(NnByte *));

#if not(DEBUG_USE_MMAP_FOR_WEIGHTS)
        if (opContext->weightAllocBytes > 0)
            opContext->weight = allocAlignedBuffer(opContext->weightAllocBytes);
        else
            opContext->weight = nullptr;
#endif

        // Debug: confirm stage-local replicated weight packing is active.
        // Printed at segment creation time.
        bool replicationEnabled = true;
        const char *rep = std::getenv("DLLAMA_STAGE_REPLICATE_WEIGHTS");
        if (rep != nullptr) {
            if (std::strcmp(rep, "0") == 0 || std::strcmp(rep, "false") == 0 || std::strcmp(rep, "FALSE") == 0 ||
                std::strcmp(rep, "no") == 0 || std::strcmp(rep, "NO") == 0) {
                replicationEnabled = false;
            }
        }
        if (opConfig->code == OP_MATMUL && replicationEnabled) {
            const NnMatmulOpConfig *cfg = (const NnMatmulOpConfig *)opContext->opConfig;
            if (cfg != nullptr && cfg->replicateMode != 0u) {
                std::printf(
                    "🧩 ReplicateWeights node=%u op=%s[%u] alloc=%zu slice=%zu myOff=%zu stride=%zu nExperts=%u\n",
                    nodeConfig->nodeIndex,
                    opConfig->name,
                    opConfig->index,
                    (size_t)opContext->weightAllocBytes,
                    (size_t)opContext->weightSize.nBytes,
                    (size_t)cfg->replicateMyOffsetBytes,
                    (size_t)cfg->replicateExpertStrideBytes,
                    cfg->nExperts);
            }
        }

        if (opInit != nullptr)
            opInit(opContext);
        opForward[opIndex] = opForwardLocal[opIndex];
    }
    return new NnCpuDeviceSegment(opForward, opContexts, segmentConfig->nOps);
}

NnCpuDeviceSegment::~NnCpuDeviceSegment() {
    for (NnUint opIndex = 0; opIndex < nOps; opIndex++) {
        NnCpuOpContext *context = &opContexts[opIndex];
        delete[] context->input;
        delete[] context->output;
#if not(DEBUG_USE_MMAP_FOR_WEIGHTS)
    if (context->weightAllocBytes > 0)
            releaseAlignedBuffer(context->weight);
#endif
    }
    delete[] opForward;
    delete[] opContexts;
}

std::vector<NnByte *> NnCpuDevice::resolvePointer(NnSize3D *pntrSize, NnPointerConfig *pointerConfig) {
    NnByte *source;
    NnSize3D *sourceSize;

    switch (pointerConfig->source) {
    case SRC_BUFFER:
        source = buffers[pointerConfig->pointerIndex];
        sourceSize = &nodeConfig->buffers[pointerConfig->pointerIndex].size;
        break;
    case SRC_PIPE:
        source = netExecution->pipes[pointerConfig->pointerIndex];
        sourceSize = &netConfig->pipes[pointerConfig->pointerIndex].size;
        break;
    default:
        throw std::invalid_argument("Unsupported pointer type");
    }

    switch (pointerConfig->type) {
    case PNTR_RAW: {
        *pntrSize = size1D(sourceSize->floatType, sourceSize->length);
        return std::vector<NnByte *>{source};
    }
    case PNTR_BATCH:
    case PNTR_BATCHED_SLICE: {
        ASSERT_EQ(sourceSize->y, netConfig->nBatches);
        std::vector<NnByte *> pntr(sourceSize->z * sourceSize->y);

        NnSize batchBytes = getBytes(sourceSize->floatType, sourceSize->x);
        for (NnUint z = 0u; z < sourceSize->z; z++) {
            for (NnUint y = 0u; y < sourceSize->y; y++)
                pntr[z * sourceSize->y + y] = &source[(z * sourceSize->y + y) * batchBytes];
        }
        *pntrSize = *sourceSize;

        if (pointerConfig->type == PNTR_BATCHED_SLICE) {
            // ====================================================
            // [重写] 智能非均匀切分逻辑
            // ====================================================
            NnUint myOffset = 0;
            NnUint myLength = 0;
            bool splitFound = false;
            bool stackedByNode = false;

            // 1. 尝试查阅 Partition Plan 来获取精确的非均匀 Offset/Length
            if (partitionPlan != nullptr && netConfig->nNodes == partitionPlan->nNodes) {
                NnUint totalDim = sourceSize->x; // 管道的总维度
                NnUint nodeIdx = nodeConfig->nodeIndex;
                const NnStageConfig* myStage = (partitionPlan->nStages > 0) ? findStageForNode(partitionPlan, nodeIdx) : nullptr;

                // ----------------------------------------------------
                // [PP Fix] ZQ 等“按 node 堆叠”的 pipe：总维度 == dim * nNodes
                // 这种 pipe 的每个 node slice 应该是固定长度 dim，且按【全局 nodeIndex】排布。
                // 不能用 dim/head split 进行匹配，否则会把 offset 解释成 interleaved 布局，导致跨 stage 污染。
                // ----------------------------------------------------
                if (partitionPlan->nStages > 0 && myStage != nullptr) {
                    NnUint dimTotal = getSplitTotalForStage(&partitionPlan->dimSplit, myStage);
                    if (dimTotal > 0 && totalDim == dimTotal * netConfig->nNodes) {
                        stackedByNode = true;
                    }
                }
                
                // Lambda: 检查给定的 split 是否匹配当前维度
                auto tryApplySplit = [&](const NnDimSplit& split, bool allowMultiplier, const char* name) -> bool {
                    (void)name;
                    if (stackedByNode) {
                        // 对于 stacked-by-node 的 pipe，不允许用任何 split 匹配。
                        return false;
                    }
                    // In PP mode, split arrays are stage-local but stored in a single [nNodes] table.
                    // Summing across all nodes would produce (dim * nStages) and fail matching.
                    // So for PP, compute totals only within my stage.
                    NnUint splitTotal = (myStage != nullptr) ? getSplitTotalForStage(&split, myStage)
                                                           : getSplitTotal(&split, partitionPlan->nNodes);
                    if (splitTotal > 0 && totalDim % splitTotal == 0) {
                        // 命中！计算倍率 (例如 HeadDim) 并应用
                        NnUint multiplier = totalDim / splitTotal;
                        
                        // [Fix] Prevent aggressive matching for ZQ pipe (dim * nNodes)
                        // If allowMultiplier is false, we require exact match (multiplier == 1)
                        if (!allowMultiplier && multiplier != 1) return false;

                        myOffset = split.starts[nodeIdx] * multiplier;
                        myLength = split.lengths[nodeIdx] * multiplier;

                        return true;
                    }
                    return false;
                };

                // 按优先级尝试匹配 (Vocab > FFN > Heads)
                // Vocab, FFN, Dim should match exactly (multiplier 1) to avoid matching concatenated pipes like ZQ
                if (!splitFound) splitFound = tryApplySplit(partitionPlan->vocabSplit, false, "vocab");
                if (!splitFound) splitFound = tryApplySplit(partitionPlan->ffnSplit, false, "ffn");
                if (!splitFound) splitFound = tryApplySplit(partitionPlan->dimSplit, false, "dim");
                // Heads have headDim multiplier; but for Q80 activation pipes (e.g., ZQ) avoid head split to prevent misaligned offsets
                if (sourceSize->floatType != F_Q80) {
                    if (!splitFound) splitFound = tryApplySplit(partitionPlan->headSplit, true, "head");
                    if (!splitFound) splitFound = tryApplySplit(partitionPlan->kvHeadSplit, true, "kvhead");
                }
            }

            // 2. 如果没有 Plan 或没找到匹配，回退到 Legacy 均匀切分
            if (!splitFound) {
                // In PP mode:
                // - If pipe is stacked-by-node (e.g., ZQ = dim * nNodes), use global node slots.
                // - Otherwise, fall back to uniform split within my stage.
                const NnStageConfig* myStage = (partitionPlan && partitionPlan->nStages > 0)
                    ? findStageForNode(partitionPlan, nodeConfig->nodeIndex)
                    : nullptr;

                if (stackedByNode) {
                    myLength = sourceSize->x / netConfig->nNodes;
                    myOffset = myLength * nodeConfig->nodeIndex;
                } else {
                    NnUint nSplitNodes = (myStage != nullptr) ? myStage->nNodes : netConfig->nNodes;
                    NnUint rank = (myStage != nullptr) ? findStageRank(myStage, nodeConfig->nodeIndex) : nodeConfig->nodeIndex;

                    myLength = sourceSize->x / nSplitNodes;
                    myOffset = myLength * rank;
                }
            }

            // 3. 应用偏移量 (带越界保护)
            // [Fix] For Q80, getBytes(F_Q80, offset) assumes offset is block-aligned.
            // If offset is NOT block aligned (e.g. 1024 elements, block 32 -> aligned), it works.
            // But if offset is not aligned, getBytes might throw or return wrong value?
            // getBytes implementation: assert(n % Q80_BLOCK_SIZE == 0);
            // So we MUST ensure myOffset is block aligned for Q80.
            
            if (sourceSize->floatType == F_Q80 && myOffset % Q80_BLOCK_SIZE != 0) {
                 // This is bad. Q80 requires block alignment.
                 // But usually dim splits are aligned to 32 or more.
                 // Let's hope it is aligned.
            }

            NnSize offsetBytes = getBytes(sourceSize->floatType, myOffset);
            NnSize totalBytes = getBytes(sourceSize->floatType, sourceSize->x);
            
            if (offsetBytes >= totalBytes) {
                offsetBytes = 0;
                myLength = 0;
            }

            for (NnUint z = 0u; z < sourceSize->z; z++) {
                for (NnUint y = 0u; y < sourceSize->y; y++)
                    pntr[z * sourceSize->y + y] += offsetBytes; // [Fix] Use += on NnByte* directly
            }
            
            // 更新 size 为实际计算出的 length
            *pntrSize = size3D(sourceSize->floatType, sourceSize->z, sourceSize->y, myLength);

        }
        return pntr;
    }
    default:
        throw std::invalid_argument("Unsupported pointer config");
    }
}

void NnCpuDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    assert(opIndex >= 0u);
    assert(opIndex < nOps);
    NnCpuOpContext *context = &opContexts[opIndex];
    if (offset + nBytes > context->weightAllocBytes) {
        std::cerr << "🚨 CRITICAL ERROR in loadWeight:" << std::endl;
        std::cerr << "   Op Name: " << (context->name ? context->name : "Unknown") << std::endl;
        std::cerr << "   Op Index: " << opIndex << std::endl;
        std::cerr << "   Offset: " << offset << std::endl;
        std::cerr << "   Write Bytes: " << nBytes << std::endl;
        std::cerr << "   Required (Offset + Bytes): " << (offset + nBytes) << std::endl;
        std::cerr << "   Allocated Size: " << context->weightAllocBytes << std::endl;
        std::cerr << "   Diff: " << (long long)(offset + nBytes) - (long long)context->weightAllocBytes << std::endl;
    }
    assert(offset + nBytes <= context->weightAllocBytes);
#if DEBUG_USE_MMAP_FOR_WEIGHTS
    assert(offset == 0u);
    context->weight = weight;
#else
    std::memcpy(&context->weight[offset], weight, nBytes);
#endif
}

void NnCpuDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) {
    NnCpuOpContext *context = &opContexts[opIndex];
    // printf("forward: %d %s (%d/%d)\n", opIndex, context->name, threadIndex + 1, nThreads); fflush(stdout);

    opForward[opIndex](nThreads, threadIndex, batchSize, context);

#if NN_LOGITS_COMM_DATA_LOG
    if (threadIndex == 0u && context->name != nullptr) {
        if (std::strncmp(context->name, "final_", 6) == 0) {
            printf("[LOGITS][OP] node=%u op=%s[%u] inPtr=%p outPtr=%p inX=%u outX=%u inFt=%d outFt=%d\n",
                context->nodeIndex,
                context->name,
                context->opIndex,
                (context->input && context->input[0]) ? (void *)context->input[0] : nullptr,
                (context->output && context->output[0]) ? (void *)context->output[0] : nullptr,
                context->inputSize.x,
                context->outputSize.x,
                (int)context->inputSize.floatType,
                (int)context->outputSize.floatType);
        }
        if (std::strcmp(context->name, "final_norm") == 0) {
            printf("[LOGITS][OP] node=%u op=%s[%u] outPtr=%p outX=%u outFt=%d\n",
                context->nodeIndex,
                context->name,
                context->opIndex,
                (context->output && context->output[0]) ? (void *)context->output[0] : nullptr,
                context->outputSize.x,
                (int)context->outputSize.floatType);
            if (context->output && context->output[0] && context->outputSize.floatType == F_32) {
                logF32Stats("[LOGITS][NORM]", (const float *)context->output[0], context->outputSize.x);
            }
        }

        if (std::strcmp(context->name, "final_cast_y") == 0) {
            printf("[LOGITS][OP] node=%u op=%s[%u] inPtr=%p outPtr=%p inX=%u outX=%u inFt=%d outFt=%d\n",
                context->nodeIndex,
                context->name,
                context->opIndex,
                (context->input && context->input[0]) ? (void *)context->input[0] : nullptr,
                (context->output && context->output[0]) ? (void *)context->output[0] : nullptr,
                context->inputSize.x,
                context->outputSize.x,
                (int)context->inputSize.floatType,
                (int)context->outputSize.floatType);
            if (context->input && context->input[0] && context->inputSize.floatType == F_32) {
                logF32Stats("[LOGITS][CAST_IN ]", (const float *)context->input[0], context->inputSize.x);
            }
            if (context->output && context->output[0]) {
                const NnSize bytes = getBytes(context->outputSize.floatType, context->outputSize.x);
                const NnSize hashLen = std::min(bytes, (NnSize)65536);
                const std::uint64_t h = hashBytes(context->output[0], hashLen);
                printf("[LOGITS][CAST_OUT] bytes=%zu hashLen=%zu hash=0x%016llx\n",
                    (size_t)bytes, (size_t)hashLen, (unsigned long long)h);
            }
        }

        if (std::strcmp(context->name, "final_matmul_logits") == 0 || std::strcmp(context->name, "final_cast_logits") == 0) {
            printf("[LOGITS][OP] node=%u op=%s[%u] inPtr=%p outPtr=%p inX=%u outX=%u inFt=%d outFt=%d\n",
                context->nodeIndex,
                context->name,
                context->opIndex,
                (context->input && context->input[0]) ? (void *)context->input[0] : nullptr,
                (context->output && context->output[0]) ? (void *)context->output[0] : nullptr,
                context->inputSize.x,
                context->outputSize.x,
                (int)context->inputSize.floatType,
                (int)context->outputSize.floatType);

            if (context->input && context->input[0] && context->inputSize.floatType == F_32) {
                logF32Stats("[LOGITS][IN ]", (const float *)context->input[0], context->inputSize.x);
            } else if (context->input && context->input[0]) {
                const NnSize bytes = getBytes(context->inputSize.floatType, context->inputSize.x);
                const NnSize hashLen = std::min(bytes, (NnSize)65536);
                const std::uint64_t h = hashBytes(context->input[0], hashLen);
                printf("[LOGITS][INQ] bytes=%zu hashLen=%zu hash=0x%016llx\n",
                    (size_t)bytes, (size_t)hashLen, (unsigned long long)h);
            }
            if (context->output && context->output[0] && context->outputSize.floatType == F_32) {
                logF32Stats("[LOGITS][OUT]", (const float *)context->output[0], context->outputSize.x);
            }
            if (std::strcmp(context->name, "final_matmul_logits") == 0 && context->weight != nullptr) {
                const NnSize wBytes = context->weightAllocBytes;
                const NnSize hashLen = std::min(wBytes, (NnSize)65536);
                const std::uint64_t h = hashBytes(context->weight, hashLen);
                printf("[LOGITS][W] bytes=%zu hashLen=%zu hash=0x%016llx\n",
                    (size_t)wBytes, (size_t)hashLen, (unsigned long long)h);
            }
        }
    }

    if (threadIndex == 0u && context->output != nullptr && context->output[0] != nullptr) {
        NnByte *out0 = context->output[0];
        if (context->pipes != nullptr && context->pipeConfigs != nullptr && context->nPipes > 0u) {
            for (NnUint pi = 0u; pi < context->nPipes; ++pi) {
                NnByte *base = context->pipes[pi];
                const NnSize pipeBytes = context->pipeConfigs[pi].size.nBytes;
                if (base != nullptr && out0 >= base && out0 < (base + pipeBytes)) {
                    const NnSize off = (NnSize)(out0 - base);
                    printf("[PIPE][WRITE] node=%u op=%s[%u] pipe=%u outPtr=%p off=%zu pipeBytes=%zu\n",
                        context->nodeIndex,
                        context->name ? context->name : "(null)",
                        context->opIndex,
                        pi,
                        (void *)out0,
                        (size_t)off,
                        (size_t)pipeBytes);
                    break;
                }
            }
        }
    }
#endif

}
