#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network-local.hpp"
#include "nn/nn-network.hpp"
#include "mmap.hpp"
#include "llm.hpp"
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <functional>
#include <utility>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

static bool envFlag(const char *name) {
    const char *v = std::getenv(name);
    if (v == nullptr) return false;
    if (std::strcmp(v, "1") == 0) return true;
    if (std::strcmp(v, "true") == 0) return true;
    if (std::strcmp(v, "TRUE") == 0) return true;
    if (std::strcmp(v, "yes") == 0) return true;
    if (std::strcmp(v, "YES") == 0) return true;
    return false;
}

static bool envFlagDefaultTrue(const char *name) {
    const char *v = std::getenv(name);
    if (v == nullptr) return true;
    if (std::strcmp(v, "0") == 0) return false;
    if (std::strcmp(v, "false") == 0) return false;
    if (std::strcmp(v, "FALSE") == 0) return false;
    if (std::strcmp(v, "no") == 0) return false;
    if (std::strcmp(v, "NO") == 0) return false;
    // Treat any other set value as enabled.
    return true;
}

static void maybeWillNeedPages(const void *addr, size_t length) {
    if (addr == nullptr || length == 0) return;
#if defined(__linux__) || defined(__APPLE__)
    // Best-effort prefetch to page cache. Safe no-op if kernel ignores.
    (void)madvise(const_cast<void *>(addr), length, MADV_WILLNEED);
#else
    (void)addr;
    (void)length;
#endif
}

static const char *hiddenActToString(LlmHiddenAct act) {
    if (act == HIDDEN_ACT_GELU) return "Gelu";
    if (act == HIDDEN_ACT_SILU) return "Silu";
    throw std::runtime_error("Unsupported hidden act");
}

static const char *ropeTypeToString(NnRopeType type) {
    if (type == ROPE_LLAMA) return "Llama";
    if (type == ROPE_LLAMA3_1) return "Llama3.1";
    if (type == ROPE_FALCON) return "Falcon";
    throw std::runtime_error("Unsupported rope type");
}

static const char *archTypeToString(LlmArchType type) {
    if (type == LLAMA) return "Llama";
    if (type == QWEN3) return "Qwen3";
    if (type == QWEN3_MOE) return "Qwen3 MoE";
    throw std::runtime_error("Unsupported architecture");
}

static float convertNormEpsilon(int value) {
    if (value == 5) return 1e-05f;
    if (value == 6) return 1e-06f;
    throw std::runtime_error("Unsupported norm epsilon");
}

static NnSize calculateLayerBytes(LlmHeader* h, NnSize3D moeGateSize, NnSize3D rmsNormSize, NnSize3D qkRmsNormSize) {
    NnSize bytes = 0;
    // Q, K, V, WO
    bytes += size2D(h->weightType, h->dim, h->qDim).nBytes;
    bytes += size2D(h->weightType, h->dim, h->kvDim).nBytes * 2; 
    bytes += size2D(h->weightType, h->qDim, h->dim).nBytes; 

    // FFN / MoE
    NnUint ffDim = (h->archType == QWEN3_MOE) ? h->moeHiddenDim : h->hiddenDim;
    if (h->nExperts > 0) {
        bytes += moeGateSize.nBytes;
        // Experts * (W1 + W2 + W3) - 假设是 Interleaved 存储
        bytes += h->nExperts * (size2D(h->weightType, h->dim, ffDim).nBytes * 2 + size2D(h->weightType, ffDim, h->dim).nBytes);
    } else {
        bytes += size2D(h->weightType, h->dim, ffDim).nBytes * 2;
        bytes += size2D(h->weightType, ffDim, h->dim).nBytes;
    }

    // Norms
    if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
        bytes += qkRmsNormSize.nBytes * 2;
    }
    bytes += rmsNormSize.nBytes * 2; 

    return bytes;
}

LlmHeader loadLlmHeader(const char *path, const NnUint maxSeqLen, NnFloatType syncType) {
    LlmHeader header;
    std::memset(&header, 0, sizeof(LlmHeader));
    header.weightType = F_UNK;
    header.hiddenAct = HIDDEN_ACT_SILU;
    header.ropeType = ROPE_LLAMA;
    header.ropeTheta = 10000.0f;
    header.ropeScalingFactor = 1.0f;
    header.normEpsilon = 1e-5f;
    header.moeHiddenDim = 0u;

    std::unique_ptr<FILE, int(*)(FILE *)> fdPtr(fopen(path, "rb"), fclose);
    FILE *fd = fdPtr.get();
    if (fd == NULL)
        throw std::runtime_error(std::string("Cannot open model file (") + path + std::string("): ") + std::strerror(errno));

    int magic;
    if (fread(&magic, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read magic value");

    if (magic == 0xABCD00 || magic == 0xABCD01)
        throw std::runtime_error("Old model format is not supported");
    if (magic != 0xA00ABCD)
        throw std::runtime_error("Unsupported magic number");

    if (fread(&header.headerSize, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read header size");

    std::vector<int> bufferPtr(header.headerSize);
    int *buffer = &bufferPtr[0];
    if (fread(buffer, header.headerSize, 1, fd) != 1)
        throw std::runtime_error("Cannot read header values");

    int nKv = (header.headerSize - 2 * sizeof(int)) / sizeof(int);

    for (int i = 0; i < nKv; i += 2) {
        int key = buffer[i];
        int value = buffer[i + 1];
        if (key == VERSION) header.version = value;
        else if (key == ARCH_TYPE) header.archType = (LlmArchType)value;
        else if (key == DIM) header.dim = value;
        else if (key == HIDDEN_DIM) header.hiddenDim = value;
        else if (key == N_LAYERS) header.nLayers = value;
        else if (key == N_HEADS) header.nHeads = value;
        else if (key == N_KV_HEADS) header.nKvHeads = value;
        else if (key == N_EXPERTS) header.nExperts = value;
        else if (key == N_ACTIVE_EXPERTS) header.nActiveExperts = value;
        else if (key == VOCAB_SIZE) header.vocabSize = value;
        else if (key == SEQ_LEN) header.seqLen = value;
        else if (key == HIDDEN_ACT) header.hiddenAct = (LlmHiddenAct)value;
        else if (key == ROPE_THETA) header.ropeTheta = (float)value;
        else if (key == WEIGHT_FLOAT_TYPE) header.weightType = (NnFloatType)value;
        else if (key == ROPE_SCALING_FACTOR) header.ropeScalingFactor = (float)value;
        else if (key == ROPE_SCALING_LOW_FREQ_FACTOR) header.ropeScalingLowFreqFactor = (float)value;
        else if (key == ROPE_SCALING_HIGH_FREQ_FACTORY) header.ropeScalingHighFreqFactory = (float)value;
        else if (key == ROPE_SCALING_ORIG_MAX_SEQ_LEN) header.ropeScalingOrigMaxSeqLen = value;
        else if (key == ROPE_TYPE) header.ropeType = (NnRopeType)value;
        else if (key == HEAD_DIM) header.headDim = value;
        else if (key == NORM_EPSILON) header.normEpsilon = convertNormEpsilon(value);
        else if (key == MOE_HIDDEN_DIM) header.moeHiddenDim = value;
        else throw std::runtime_error("Unsupported header key");
    }

    if (header.weightType == F_UNK)
        throw std::runtime_error("Model does not specify weight type");

    header.origSeqLen = header.seqLen;
    if (maxSeqLen > 0 && header.seqLen > maxSeqLen)
        header.seqLen = maxSeqLen;

    if (header.headDim == 0)
        header.headDim = header.dim / header.nHeads;
    header.qDim = header.headDim * header.nHeads;
    header.kvDim = header.headDim * header.nKvHeads;
    header.syncType = syncType;
    header.fileSize = (NnSize)seekToEnd(fd);

    if (header.archType == QWEN3 || header.archType == QWEN3_MOE)
        header.ropeType = ROPE_FALCON;
    return header;
}

void printLlmHeader(LlmHeader *header) {
    printf("💡 Arch: %s\n", archTypeToString(header->archType));
    printf("💡 HiddenAct: %s\n", hiddenActToString(header->hiddenAct));
    printf("💡 Dim: %u\n", header->dim);
    printf("💡 HeadDim: %u\n", header->headDim);
    printf("💡 QDim: %u\n", header->qDim);
    printf("💡 KvDim: %u\n", header->kvDim);
    printf("💡 HiddenDim: %u\n", header->hiddenDim);
    printf("💡 VocabSize: %u\n", header->vocabSize);
    printf("💡 nLayers: %u\n", header->nLayers);
    printf("💡 nHeads: %u\n", header->nHeads);
    printf("💡 nKvHeads: %u\n", header->nKvHeads);
    if (header->seqLen != header->origSeqLen) {
        printf("💡 OrigSeqLen: %u\n", header->origSeqLen);
    }
    if (header->nExperts > 0) {
        printf("💡 nExperts: %u\n", header->nExperts);
        printf("💡 nActiveExperts: %u\n", header->nActiveExperts);
        printf("💡 MoeHiddenDim: %u\n", header->moeHiddenDim);
    }
    printf("💡 SeqLen: %u\n", header->seqLen);
    printf("💡 NormEpsilon: %f\n", header->normEpsilon);
    printf("💡 RopeType: %s\n", ropeTypeToString(header->ropeType));
    printf("💡 RopeTheta: %.0f\n", header->ropeTheta);
    if (header->ropeType == ROPE_LLAMA3_1) {
        printf("💡 RopeScaling: f=%.1f, l=%.1f, h=%.1f, o=%d\n",
            header->ropeScalingFactor,
            header->ropeScalingLowFreqFactor,
            header->ropeScalingHighFreqFactory,
            header->ropeScalingOrigMaxSeqLen);
    }
}

//get stage config for a given node index
static const NnStageConfig* getStageForNode(const NnUnevenPartitionPlan *plan, NnUint nodeIndex) {
    if (!plan || plan->nStages == 0) return nullptr;
    for (NnUint s = 0; s < plan->nStages; ++s) {
        const NnStageConfig& stage = plan->stages[s];
        for (NnUint i = 0; i < stage.nNodes; ++i) {
            if (stage.nodeIndices[i] == nodeIndex) return &stage;
        }
    }
    return nullptr;
}

LlmNet buildLlmNet(LlmHeader *h, NnUint nNodes, NnUint nBatches) {
    NnUint nExpertsOr1 = std::max(h->nExperts, 1u);
    NnUint nActiveExpertsOr1 = std::max(h->nActiveExperts, 1u);
    NnUint ffDim = h->hiddenDim;

    if (h->archType == QWEN3_MOE)
        ffDim = h->moeHiddenDim;

    LlmNet n;
    n.tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    n.rmsNormSize = size1D(F_32, h->dim);
    n.qkRmsNormSize = size1D(F_32, h->headDim);
    n.moeGateSize = size2D(F_32, h->dim, h->nExperts);
    NnKvCacheSlice kvCacheSlice = sliceKvCache(h->kvDim, h->seqLen, nNodes); //KVslice
    NnMultiHeadAttSlice multiHeadAttSlice = sliceMultiHeadAtt(h->nHeads, h->seqLen, nNodes, nBatches);

    n.qSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->qDim);
    n.kSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
    n.vSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
    n.woSlice = sliceColMatmul(h->weightType, nNodes, h->qDim, h->dim);

    n.w1Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, ffDim);
    n.w2Slice = sliceColMatmul(h->weightType, nNodes, ffDim, h->dim);
    n.w3Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, ffDim);
    n.wclsSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->vocabSize);
 
    NnUint nQNormColumns = 1;
    NnUint nKNormColumns = 1;
    NnUint nInvBufferColumns = 1;
    if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
        ASSERT_EQ(n.qSlice.d0 % h->headDim, 0);
        ASSERT_EQ(n.kSlice.d0 % h->headDim, 0);
        nQNormColumns = n.qSlice.d0 / h->headDim;
        nKNormColumns = n.kSlice.d0 / h->headDim;
        nInvBufferColumns = std::max(nQNormColumns, nKNormColumns);
    }

    NnNetConfigBuilder netBuilder(nNodes, nBatches);

    n.positionPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));
    n.tokenPipeIndex = netBuilder.addPipe("TOK", size2D(F_32, nBatches, 1));
    n.xPipeIndex = netBuilder.addPipe("X", size2D(F_32, nBatches, h->dim));
    n.logitsPipeIndex = netBuilder.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    const NnUint zqPipeIndex = netBuilder.addPipe("ZQ", size2D(h->syncType, nBatches, h->dim * nNodes));

    // Dedicated control pipe for stage-root sharding updates.
    // Sized in u32 words but stored as F_32 (4 bytes) for transport.
    const NnSize shardingBytes = sizeof(NnShardingUpdateHeader) + (NnSize)(N_SPLIT_KINDS * 2u * nNodes * sizeof(NnUint));
    const NnUint shardingWords = (NnUint)((shardingBytes + sizeof(NnUint) - 1u) / sizeof(NnUint));
    n.shardingPipeIndex = netBuilder.addPipe("SHD", size2D(F_32, nBatches, shardingWords));

    n.zqPipeIndex = zqPipeIndex;
    n.kPipeIndex = 0u;
    n.vPipeIndex = 0u;

    netBuilder.addPreSync(n.positionPipeIndex);

    n.header = h;
    n.netConfig = netBuilder.build();
    n.nodeConfigs = new NnNodeConfig[nNodes];

    for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
        NnRopeSlice ropeSlice = sliceRope(h->ropeType, h->qDim, h->kvDim, h->nKvHeads, nNodes, h->seqLen, h->headDim, h->ropeTheta, nodeIndex);
        NnNodeConfigBuilder nodeBuilder(nodeIndex);

        const NnUint xBufferIndex = nodeBuilder.addBuffer("x", size2D(F_32, nBatches, h->dim));
        const NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, nBatches, h->dim));
        const NnUint yqBufferIndex = h->syncType == F_32
            ? yBufferIndex
            : nodeBuilder.addBuffer("q_y", size2D(h->syncType, nBatches, h->dim));

        const NnUint zBufferIndex = nodeBuilder.addBuffer("z", size2D(F_32, nBatches, h->qDim));
        const NnUint zqSliceBufferIndex = nodeBuilder.addBuffer("q_z_slice", size2D(h->syncType, nBatches, h->qDim / nNodes));

        const NnUint qBufferIndex = nodeBuilder.addBuffer("q", size2D(F_32, nBatches, n.qSlice.d0));
        const NnUint kTempBufferIndex = nodeBuilder.addBuffer("k_temp", size2D(F_32, nBatches, n.kSlice.d0));
        const NnUint vTempBufferIndex = nodeBuilder.addBuffer("v_temp", size2D(F_32, nBatches, n.vSlice.d0));

        const NnUint invRmsBufferIndex = nodeBuilder.addBuffer("inv_rms", size2D(F_32, nBatches, nInvBufferColumns));

        const NnUint ropeCacheBufferIndex = nodeBuilder.addBuffer("rope_cache", ropeSlice.cacheSize);
        const NnUint attBufferIndex = nodeBuilder.addBuffer("att", multiHeadAttSlice.attSize);
        const NnUint logitsSliceBufferIndex = nodeBuilder.addBuffer("lg", size2D(F_32, nBatches, h->vocabSize / nNodes));

        // not moe
        const NnUint dBufferIndex = nodeBuilder.addBuffer("d", size2D(F_32, nBatches, n.w1Slice.d0));
        const NnUint dqBufferIndex = h->syncType == F_32
            ? dBufferIndex
            : nodeBuilder.addBuffer("q_d", size2D(h->syncType, nBatches, n.w1Slice.d0));
        const NnUint lBufferIndex = nodeBuilder.addBuffer("l", size2D(F_32, nBatches, n.w3Slice.d0));

        // moe
        const NnUint moeGtBufferIndex = nodeBuilder.addBuffer("gt", size2D(F_32, nBatches, nExpertsOr1));
        const NnUint moeExpertIndexesBufferIndex = nodeBuilder.addBuffer("act_exp_ix", size2D(F_32, nBatches, nActiveExpertsOr1));
        const NnUint moeYBufferIndex = nodeBuilder.addBuffer("moe_y", size3D(F_32, nActiveExpertsOr1, nBatches, h->dim));
        const NnUint moeYqBufferIndex = h->syncType == F_32
            ? moeYBufferIndex
            : nodeBuilder.addBuffer("q_moe_y", size3D(h->syncType, nActiveExpertsOr1, nBatches, h->dim));
        const NnUint moeDBufferIndex = nodeBuilder.addBuffer("moe_d", size3D(F_32, nActiveExpertsOr1, nBatches, n.w1Slice.d0));
        const NnUint moeDQBufferIndex = h->syncType == F_32
            ? moeDBufferIndex
            : nodeBuilder.addBuffer("q_moe_d", size3D(h->syncType, nActiveExpertsOr1, nBatches, n.w1Slice.d0));
        const NnUint moeLBufferIndex = nodeBuilder.addBuffer("moe_l", size3D(F_32, nActiveExpertsOr1, nBatches, n.w3Slice.d0));
        const NnUint moeSBufferIndex = nodeBuilder.addBuffer("moe_s", size3D(F_32, nActiveExpertsOr1, nBatches, 1));

        NnSegmentConfigBuilder start;
        if (nodeIndex == 0) {
            start.addOp(
                OP_EMBEDDING, "embedding", 0,
                pointerBatchConfig(SRC_PIPE, n.tokenPipeIndex),
                pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                n.tokenEmbeddingSize,
                NnEmbeddingOpConfig{});
        }
        start.addSync(n.xPipeIndex, SYNC_WITH_ROOT);
        nodeBuilder.addSegment(start.build());

        for (NnUint layerIndex = 0; layerIndex < h->nLayers; layerIndex++) {
            const NnUint kBufferIndex = nodeBuilder.addBuffer("k", kvCacheSlice.keySize);
            const NnUint vBufferIndex = nodeBuilder.addBuffer("v", kvCacheSlice.valueSize);

            NnSegmentConfigBuilder att;
            NnSegmentConfigBuilder ff;

            // att
            if (layerIndex == 0) {
                att.addOp(
                    OP_CAST, "block_cast_x", layerIndex,
                    pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{});
            } else {
                att.addOp(
                    OP_MERGE_ADD, "block_merge_add", layerIndex,
                    pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(),
                    NnMergeAddOpCodeConfig{});
            }

            att.addOp(
                OP_INV_RMS, "block_norm_pre_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon, 1});
            att.addOp(
                OP_RMS_NORM, "block_norm_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex, 1});
            if (yBufferIndex != yqBufferIndex) {
                att.addOp(
                    OP_CAST, "block_cast_y", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{});
            }
            att.addOp(
                OP_MATMUL, "block_matmul_q", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                size2D(h->weightType, n.qSlice.n, n.qSlice.d0),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
            att.addOp(
                OP_MATMUL, "block_matmul_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size2D(h->weightType, n.kSlice.n, n.kSlice.d0),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
            att.addOp(
                OP_MATMUL, "block_matmul_v", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                size2D(h->weightType, n.vSlice.n, n.vSlice.d0),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});

            if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
                att.addOp(OP_INV_RMS, "block_norm_pre_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                    size0(),
                    NnInvRmsOpConfig{h->normEpsilon, nQNormColumns});
                att.addOp(
                    OP_RMS_NORM, "block_norm_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    size2D(F_32, 1, n.header->headDim),
                    NnRmsNormOpConfig{invRmsBufferIndex, nQNormColumns});

                att.addOp(OP_INV_RMS, "block_norm_pre_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                    size0(),
                    NnInvRmsOpConfig{h->normEpsilon, nKNormColumns});
                att.addOp(
                    OP_RMS_NORM, "block_norm_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    size2D(F_32, 1, n.header->headDim),
                    NnRmsNormOpConfig{invRmsBufferIndex, nKNormColumns});
            }

            att.addOp(
                OP_ROPE, "block_rope_q", layerIndex,
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                size0(),
                NnRopeOpConfig{n.header->ropeType, 1, n.positionPipeIndex, ropeCacheBufferIndex, 
                    h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
                    ropeSlice});
            att.addOp(
                OP_ROPE, "block_rope_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size0(),
                NnRopeOpConfig{n.header->ropeType, 0, n.positionPipeIndex, ropeCacheBufferIndex, 
                    h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
                    ropeSlice});
            att.addOp(
                OP_SHIFT, "block_shift_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerRawConfig(SRC_BUFFER, kBufferIndex),
                size0(),
                NnShiftOpCodeConfig{n.positionPipeIndex});
            att.addOp(
                OP_SHIFT, "block_shift_v", layerIndex,
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                pointerRawConfig(SRC_BUFFER, vBufferIndex),
                size0(),
                NnShiftOpCodeConfig{n.positionPipeIndex});
            att.addOp(
                OP_MULTIHEAD_ATT, "block_multihead_att", layerIndex,
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                size0(),
                NnMultiHeadAttOpConfig{
                    nodeIndex, layerIndex, (NnUint)(nodeIndex * multiHeadAttSlice.nHeads0),
                    multiHeadAttSlice.nHeads, multiHeadAttSlice.nHeads0,
                    h->nKvHeads, h->headDim, h->seqLen, n.qSlice.d0, kvCacheSlice.kvDim0,
                    n.positionPipeIndex, qBufferIndex, kBufferIndex, vBufferIndex, attBufferIndex});
            att.addOp(
                OP_CAST, "block_cast_y2", layerIndex,
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                pointerBatchConfig(SRC_BUFFER, zqSliceBufferIndex),
                size0(),
                NnCastOpCodeConfig{});
            att.addOp(
                OP_MATMUL, "block_matmul_wo", layerIndex,
                pointerBatchConfig(SRC_BUFFER, zqSliceBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                size2D(h->weightType, n.woSlice.n0, n.woSlice.d),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
            att.addOp(
                OP_CAST, "block_cast_d", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
            att.addSync(zqPipeIndex, SYNC_NODE_SLICES);

            // ff
            ff.addOp(
                OP_MERGE_ADD, "block_merge_add2", layerIndex,
                pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                size0(),
                NnMergeAddOpCodeConfig{});
            ff.addOp(
                OP_INV_RMS, "block_norm_pre_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon, 1});
            ff.addOp(
                OP_RMS_NORM, "block_norm_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex, 1});

            if (h->archType == QWEN3_MOE) {
                ff.addOp(
                    OP_REPEAT_Z, "block_moe_y_repeat", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
                    size0(),
                    NnRepeatZOpCodeConfig{});
                ff.addOp(
                    OP_MATMUL, "block_moe_gate", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    n.moeGateSize,
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_SOFTMAX, "block_moe_softmax", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    size0(),
                    NnSoftmaxOpCodeConfig{});
                ff.addOp(
                    OP_MOE_GATE, "block_moe_gate2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeSBufferIndex),
                    size0(),
                    NnMoeGateOpCodeConfig{h->nActiveExperts, 1u, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_MATMUL, "block_matmul_w1", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    size3D(h->weightType, h->nExperts, n.w1Slice.n, n.w1Slice.d0),
                    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_MATMUL, "block_matmul_w3", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeLBufferIndex),
                    size3D(h->weightType, h->nExperts, n.w3Slice.n, n.w3Slice.d0),
                    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_SILU, "block_act", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    size0(),
                    NnSiluOpCodeConfig{});
                ff.addOp(
                    OP_MUL, "block_mul", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    size0(),
                    NnMulOpCodeConfig{moeLBufferIndex});
                if (moeDBufferIndex != moeDQBufferIndex) {
                    ff.addOp(
                        OP_CAST, "block_cast_d2", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex),
                        size0(),
                        NnCastOpCodeConfig{});
                }
                ff.addOp(
                    OP_MATMUL, "block_matmul_w2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    size3D(h->weightType, h->nExperts, n.w2Slice.n0, n.w2Slice.d),
                    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_SCALE, "block_moe_scale", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    size0(),
                    NnScaleOpCodeConfig{moeSBufferIndex});
                ff.addOp(
                    OP_MERGE_SUM, "block_moe_merge_sum", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    size0(),
                    NnMergeSumOpCodeConfig{});
            } else {
                if (yBufferIndex != yqBufferIndex) {
                    ff.addOp(
                        OP_CAST, "block_cast_y3", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                        size0(),
                        NnCastOpCodeConfig{});
                }
                ff.addOp(
                    OP_MATMUL, "block_matmul_w1", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    size2D(h->weightType, n.w1Slice.n, n.w1Slice.d0),
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_MATMUL, "block_matmul_w3", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, lBufferIndex),
                    size2D(h->weightType, n.w3Slice.n, n.w3Slice.d0),
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_SILU, "block_act", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    size0(),
                    NnSiluOpCodeConfig{});
                ff.addOp(
                    OP_MUL, "block_mul", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    size0(),
                    NnMulOpCodeConfig{lBufferIndex});
                if (dBufferIndex != dqBufferIndex) {
                    ff.addOp(
                        OP_CAST, "block_cast_d2", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
                        size0(),
                        NnCastOpCodeConfig{});
                }
                ff.addOp(
                    OP_MATMUL, "block_matmul_w2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    size2D(h->weightType, n.w2Slice.n0, n.w2Slice.d),
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
            }
            ff.addOp(
                OP_CAST, "block_cast_d3", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
            ff.addSync(zqPipeIndex, SYNC_NODE_SLICES);

            nodeBuilder.addSegment(att.build());
            nodeBuilder.addSegment(ff.build());
        }

        NnSegmentConfigBuilder end;
        end.addOp(
            OP_MERGE_ADD, "final_merge_add", 0,
            pointerBatchConfig(SRC_PIPE, zqPipeIndex),
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            size0(),
            NnMergeAddOpCodeConfig{});
        end.addOp(
            OP_INV_RMS, "final_norm_pre", 0,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
            size0(),
            NnInvRmsOpConfig{h->normEpsilon, 1});
        end.addOp(
            OP_RMS_NORM, "final_norm", 0,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            pointerBatchConfig(SRC_BUFFER, yBufferIndex),
            n.rmsNormSize,
            NnRmsNormOpConfig{invRmsBufferIndex, 1});
        if (yBufferIndex != yqBufferIndex) {
            end.addOp(
                OP_CAST, "final_cast_y", 0,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                size0(),
                NnCastOpCodeConfig{});
        }
        {
            NnMatmulOpConfig cfg{};
            cfg.layerIndex = (h->nLayers > 0u) ? (h->nLayers - 1u) : 0u;
            cfg.splitKind = (NnUint)SPLIT_VOCAB;
            cfg.splitAxis = 1u; // OUT_ROWS
            cfg.splitUnit = 1u;
            cfg.staticStartUnits = nodeIndex * n.wclsSlice.d0;
            cfg.staticLenUnits = n.wclsSlice.d0;
            end.addOp(
                OP_MATMUL, "final_matmul_logits", 0,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
                size2D(h->weightType, n.wclsSlice.n, n.wclsSlice.d0),
                cfg);
        }
        end.addOp(
            OP_CAST, "final_cast_logits", 0,
            pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
            pointerBatchedSliceConfig(SRC_PIPE, n.logitsPipeIndex),
            size0(),
            NnCastOpCodeConfig{});
        end.addSync(n.logitsPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);

        nodeBuilder.addSegment(end.build());
        n.nodeConfigs[nodeIndex] = nodeBuilder.build();
    }
    return n;
}

static NnNodeConfig buildLlmNodeInternal(
    NnUint nodeIndex, 
    LlmHeader *h, 
    LlmNet *n, // 为了获取全局 Pipe Index
    const NnUnevenPartitionPlan *plan,
    NnUint nBatches,
    NnUint startLayer,
    NnUint endLayer,
    bool isFirstStage,
    bool isLastStage
) {
    // 1. 准备参数
    NnUint nExpertsOr1 = std::max(h->nExperts, 1u);
    NnUint nActiveExpertsOr1 = std::max(h->nActiveExperts, 1u);
    NnUint ffDim = (h->archType == QWEN3_MOE) ? h->moeHiddenDim : h->hiddenDim;

    // 2. 计算切分 (Slicing)
    NnKvCacheSliceUneven kvCacheSlice = sliceKvCacheUneven(h->seqLen, h->headDim, plan, nodeIndex);
    NnMultiHeadAttSliceUneven multiHeadAttSlice = sliceMultiHeadAttUneven(nBatches, h->nHeads, h->seqLen, plan, nodeIndex);
    
    NnRowMatmulSliceUneven qSlice = sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->headSplit, h->qDim, nodeIndex);
    NnRowMatmulSliceUneven kSlice = sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, nodeIndex);
    NnRowMatmulSliceUneven vSlice = sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, nodeIndex);
    NnColMatmulSliceUneven woSlice = sliceColMatmulAttUneven(h->weightType, h->qDim, h->dim, h->headDim, plan, nodeIndex);

    NnRowMatmulSliceUneven w1Slice = sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, nodeIndex);
    NnColMatmulSliceUneven w2Slice = sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, nodeIndex);
    NnRowMatmulSliceUneven w3Slice = sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, nodeIndex);
    NnRowMatmulSliceUneven wclsSlice = sliceRowMatmulLogitsUneven(h->weightType, h->dim, h->vocabSize, plan, nodeIndex);

    NnRopeSliceUneven unevenRope = sliceRopeUneven(h->ropeType, h->seqLen, h->kvDim, h->nKvHeads, h->headDim, h->ropeTheta, plan, nodeIndex);

    const NnStageConfig* myStageCfg = (plan != nullptr && plan->nStages > 0)
        ? getStageForNode(plan, nodeIndex)
        : nullptr;
    const bool replicateStageWeights = true;

    auto computeReplicaRow = [&](const std::function<NnRowMatmulSliceUneven(NnUint)> &sliceFn) -> std::pair<NnSize, NnSize> {
        if (!replicateStageWeights || myStageCfg == nullptr || myStageCfg->nNodes == 0 || myStageCfg->nodeIndices == nullptr)
            return {0u, 0u};
        // In replicated mode we load the full global tensor (not packed slices), so per-expert stride is the global tensor size
        // and the per-node offset is derived from the first node's slice start.
        NnRowMatmulSliceUneven global = sliceFn(myStageCfg->nodeIndices[0]);
        return {global.size.nBytes, 0u};
    };

    auto computeReplicaCol = [&](const std::function<NnColMatmulSliceUneven(NnUint)> &sliceFn) -> std::pair<NnSize, NnSize> {
        if (!replicateStageWeights || myStageCfg == nullptr || myStageCfg->nNodes == 0 || myStageCfg->nodeIndices == nullptr)
            return {0u, 0u};
        
        // [FIX] 既然是全量加载 (replicateStageWeights=true)，直接加载 Node 0 (Global) 的全量信息
        NnColMatmulSliceUneven global = sliceFn(myStageCfg->nodeIndices[0]);
        
        // 强制从文件偏移 0 开始，加载全量大小
        return {global.size.nBytes, 0u}; 
    };
    
    // 适配旧版 Rope Config
    NnRopeSlice ropeSlice;
    std::memset(&ropeSlice, 0, sizeof(NnRopeSlice));
    ropeSlice.qDim0 = unevenRope.qDimLen;
    ropeSlice.qDimStart = unevenRope.qDimStart;
    ropeSlice.qDimEnd = unevenRope.qDimStart + unevenRope.qDimLen;
    ropeSlice.qShift = unevenRope.qShift;
    ropeSlice.kvDim = unevenRope.kvDim;
    ropeSlice.kvDim0 = unevenRope.kvDimLen;
    ropeSlice.kvDimStart = unevenRope.kvDimStart;
    ropeSlice.sliceDim = unevenRope.sliceDim;
    ropeSlice.seqLen = unevenRope.seqLen;
    ropeSlice.headDim = unevenRope.headDim;
    ropeSlice.nKvHeads = unevenRope.nKvHeads;
    ropeSlice.ropeTheta = unevenRope.ropeTheta;
    ropeSlice.cacheSize = unevenRope.cacheSize;

    NnUint nQNormColumns = 1, nKNormColumns = 1, nInvBufferColumns = 1;

    // 3. 构建 Node Config
    NnNodeConfigBuilder nodeBuilder(nodeIndex);

    // Global-dim preallocation: each node allocates buffers large enough to hold
    // the full layer tensors (all heads / full FFN dim). Runtime execution may
    // still operate on per-node slices.
    const NnUint stageMaxFfnLen = ffDim;
    const NnUint stageMaxHeadLen = h->nHeads;
    const NnUint stageMaxKvHeadLen = h->nKvHeads;
    const NnUint stageMaxQSliceD0 = h->qDim;
    const NnUint stageMaxKvDim0 = h->kvDim;

    // QWEN3/QWEN3_MOE: per-head norm uses nColumns=headCount. With stage-max buffers,
    // nColumns must match the allocated head count to keep x % nColumns == 0.
    if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
        nQNormColumns = std::max(stageMaxHeadLen, 1u);
        nKNormColumns = std::max(stageMaxKvHeadLen, 1u);
        nInvBufferColumns = std::max(nQNormColumns, nKNormColumns);
    }

    // Buffers
    const NnUint xBufferIndex = nodeBuilder.addBuffer("x", size2D(F_32, nBatches, h->dim));
    const NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, nBatches, h->dim));
    const NnUint yqBufferIndex = (h->syncType == F_32) ? yBufferIndex : nodeBuilder.addBuffer("q_y", size2D(h->syncType, nBatches, h->dim));
    
    const NnUint mhaOutBufferIndex = nodeBuilder.addBuffer("mha_out", size2D(F_32, nBatches, stageMaxQSliceD0));
    const NnUint mhaOutQBufferIndex = (h->syncType == F_32)
        ? mhaOutBufferIndex
        : nodeBuilder.addBuffer("q_mha_out", size2D(h->syncType, nBatches, stageMaxQSliceD0));
    
    const NnUint qBufferIndex = nodeBuilder.addBuffer("q", size2D(F_32, nBatches, stageMaxQSliceD0));
    const NnUint kTempBufferIndex = nodeBuilder.addBuffer("k_temp", size2D(F_32, nBatches, stageMaxKvDim0));
    const NnUint vTempBufferIndex = nodeBuilder.addBuffer("v_temp", size2D(F_32, nBatches, stageMaxKvDim0));
    const NnUint invRmsBufferIndex = nodeBuilder.addBuffer("inv_rms", size2D(F_32, nBatches, nInvBufferColumns));
    const NnUint ropeCacheBufferIndex = nodeBuilder.addBuffer("rope_cache", ropeSlice.cacheSize);
    const NnUint attBufferIndex = nodeBuilder.addBuffer("att", size2D(F_32, nBatches, stageMaxHeadLen * h->seqLen));
    const NnUint logitsSliceBufferIndex = nodeBuilder.addBuffer("lg", size2D(F_32, nBatches, wclsSlice.inLen));

    const NnUint dBufferIndex = nodeBuilder.addBuffer("d", size2D(F_32, nBatches, stageMaxFfnLen));
    const NnUint dqBufferIndex = (h->syncType == F_32) ? dBufferIndex : nodeBuilder.addBuffer("q_d", size2D(h->syncType, nBatches, stageMaxFfnLen));
    const NnUint lBufferIndex = nodeBuilder.addBuffer("l", size2D(F_32, nBatches, stageMaxFfnLen));

    const NnUint moeGtBufferIndex = nodeBuilder.addBuffer("gt", size2D(F_32, nBatches, nExpertsOr1));
    const NnUint moeExpertIndexesBufferIndex = nodeBuilder.addBuffer("act_exp_ix", size2D(F_32, nBatches, nActiveExpertsOr1));
    const NnUint moeYBufferIndex = nodeBuilder.addBuffer("moe_y", size3D(F_32, nActiveExpertsOr1, nBatches, h->dim));
    const NnUint moeYqBufferIndex = (h->syncType == F_32) ? moeYBufferIndex : nodeBuilder.addBuffer("q_moe_y", size3D(h->syncType, nActiveExpertsOr1, nBatches, h->dim));
    const NnUint moeDBufferIndex = nodeBuilder.addBuffer("moe_d", size3D(F_32, nActiveExpertsOr1, nBatches, stageMaxFfnLen));
    const NnUint moeDQBufferIndex = (h->syncType == F_32) ? moeDBufferIndex : nodeBuilder.addBuffer("q_moe_d", size3D(h->syncType, nActiveExpertsOr1, nBatches, stageMaxFfnLen));
    const NnUint moeLBufferIndex = nodeBuilder.addBuffer("moe_l", size3D(F_32, nActiveExpertsOr1, nBatches, stageMaxFfnLen));
    const NnUint moeSBufferIndex = nodeBuilder.addBuffer("moe_s", size3D(F_32, nActiveExpertsOr1, nBatches, 1));

    // 4. Start Segment (Embedding)
    NnSegmentConfigBuilder start;
    if (isFirstStage) {
        // [修改] First Stage 所有节点都负责 Embedding
        // 1. 先同步 Token (广播: Root -> Stage 0 Workers)
        // 注意：这里假设 SYNC_WITH_ROOT 能正确处理 Node 0 到 Stage 0 其他节点的广播

        NnSegmentConfigBuilder tokenSyncSeg;
        tokenSyncSeg.addSync(n->tokenPipeIndex, SYNC_WITH_ROOT);
        nodeBuilder.addSegment(tokenSyncSeg.build());

        // 2. 所有节点本地计算 Embedding (避免传输大的 Embedding 向量)
        start.addOp(OP_EMBEDDING, "embedding", 0, 
            pointerBatchConfig(SRC_PIPE, n->tokenPipeIndex),
            pointerBatchConfig(SRC_PIPE, n->xPipeIndex), 
            n->tokenEmbeddingSize, NnEmbeddingOpConfig{});
    }
    nodeBuilder.addSegment(start.build());

    if (!isFirstStage) {
        // 创建一个专门的 Segment 来处理接收
        NnSegmentConfigBuilder ppRecvSeg;
        
        // A. 接收 (仅 Stage Root 执行，在 Synchronizer 里判断)
        // 数据写入 n->xPipeIndex (复用这个 Pipe 作为 Buffer)
        ppRecvSeg.addSync(n->xPipeIndex, SYNC_PP_RECV);
        
        // B. 广播 (Stage Root -> Stage Workers)
        // 因为 SYNC_PP_RECV 只有 Root 有数据，必须马上广播给 TP 组内的其他人
        // 注意：SYNC_WITH_ROOT 默认是全局广播。你需要修改它的逻辑支持 Stage 组广播，
        // 或者简单点：PP 只支持 "Stage Root 也是 Cluster Root" 的情况？
        // 正确做法：修改 syncWithRoot 支持 group。
        // 暂时假设：Stage 内部使用 SYNC_WITH_ROOT 广播 (需要 syncWithRoot 支持局部广播)
        ppRecvSeg.addSync(n->xPipeIndex, SYNC_WITH_ROOT); 

        nodeBuilder.addSegment(ppRecvSeg.build());
    }

    // 5. Layers Loop (PP: 只构建负责的层)
    for (NnUint layerIndex = startLayer; layerIndex < endLayer; layerIndex++) {
        // ... (这里的 K/V Buffer 是 Layer Local 的，需要 Slice 信息) ...
        const NnUint kBufferIndex = nodeBuilder.addBuffer("k", size2D(F_32, h->seqLen, stageMaxKvDim0));
        const NnUint vBufferIndex = nodeBuilder.addBuffer("v", size2D(F_32, h->seqLen, stageMaxKvDim0));

        NnSegmentConfigBuilder att;
        NnSegmentConfigBuilder ff;

        // --- Dynamic sharding control point (per-layer boundary) ---
        // This must run BEFORE any compute ops for the layer so that runtime start/len
        // is visible to the layer's matmuls. Also note: a segment executes all ops first
        // then all syncs, so SYNC_STAGE_SHARDING_UPDATE must be in a separate segment
        // before OP_UPDATE_SHARDING.
        // Skip global first/last layer and stage-boundary layers: only middle layers can change online.
        if (layerIndex > 0u && (layerIndex + 1u) < h->nLayers &&
            layerIndex > startLayer && (layerIndex + 1u) < endLayer) {
            NnSegmentConfigBuilder shdSend;
            shdSend.addOp(OP_WRITE_U32, "sharding_layer_index", layerIndex,
                pointerBatchConfig(SRC_PIPE, n->shardingPipeIndex),
                pointerBatchConfig(SRC_PIPE, n->shardingPipeIndex),
                size0(),
                NnWriteU32OpConfig{layerIndex});
            shdSend.addSync(n->shardingPipeIndex, SYNC_STAGE_SHARDING_UPDATE);
            nodeBuilder.addSegment(shdSend.build());

            NnSegmentConfigBuilder shdApply;
            shdApply.addOp(OP_UPDATE_SHARDING, "update_sharding", layerIndex,
                pointerBatchConfig(SRC_PIPE, n->shardingPipeIndex),
                pointerBatchConfig(SRC_PIPE, n->shardingPipeIndex),
                size0(),
                NnUpdateShardingOpConfig{});
            nodeBuilder.addSegment(shdApply.build());
        }

        if (layerIndex == 0) {
            // Case A: 全局第0层 (Embedding -> Buffer)
            att.addOp(OP_CAST, "block_cast_x", layerIndex, 
                pointerBatchConfig(SRC_PIPE, n->xPipeIndex), 
                pointerBatchConfig(SRC_BUFFER, xBufferIndex), 
                size0(), NnCastOpCodeConfig{});
        } 
        else if (layerIndex == startLayer && !isFirstStage) {
            // Case B: Stage 起始层 (PP Recv Pipe -> Buffer)
            // 注意：PP_RECV 的 Sync 已经在循环外的 ppRecvSeg 做完了，数据在 X Pipe 中
            att.addOp(OP_CAST, "block_cast_x_pp", layerIndex, 
                pointerBatchConfig(SRC_PIPE, n->xPipeIndex), 
                pointerBatchConfig(SRC_BUFFER, xBufferIndex), 
                size0(), NnCastOpCodeConfig{});
        } 
        else {
            // Case C: 内部层 (ZQ Pipe Partial -> Merge Add -> Buffer)
            att.addOp(OP_MERGE_ADD, "block_merge_add", layerIndex, 
                pointerBatchConfig(SRC_PIPE, n->zqPipeIndex), 
                pointerBatchConfig(SRC_BUFFER, xBufferIndex), 
                size0(), NnMergeAddOpCodeConfig{});
        }

        // --- Attention Ops ---
        att.addOp(OP_INV_RMS, "block_norm_pre_0", layerIndex, pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex), size0(), NnInvRmsOpConfig{h->normEpsilon, 1});
        att.addOp(OP_RMS_NORM, "block_norm_0", layerIndex, pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex), n->rmsNormSize, NnRmsNormOpConfig{invRmsBufferIndex, 1});
        if (yBufferIndex != yqBufferIndex) {
            att.addOp(OP_CAST, "block_cast_y", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchConfig(SRC_BUFFER, yqBufferIndex), size0(), NnCastOpCodeConfig{});
        }

        {
            NnMatmulOpConfig cfg{0, 0, moeExpertIndexesBufferIndex};
            cfg.layerIndex = layerIndex;
            cfg.splitKind = (NnUint)SPLIT_HEAD;
            cfg.splitAxis = 1u; // OUT_ROWS
            cfg.splitUnit = h->headDim;
            cfg.staticStartUnits = multiHeadAttSlice.headStart;
            cfg.staticLenUnits = multiHeadAttSlice.headLen;
            NnSize allocBytes = 0;
            auto r = computeReplicaRow([&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->headSplit, h->qDim, idx); });
            if (r.first > 0) {
                cfg.replicateMode = 1u;
                cfg.replicateExpertStrideBytes = r.first;
                cfg.replicateMyOffsetBytes = r.second;
                allocBytes = r.first;
            }
            const NnSize3D wSize = (cfg.replicateMode != 0u) ? qSlice.size : qSlice.sliceSize;
            att.addOp(OP_MATMUL, "block_matmul_q", layerIndex, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, qBufferIndex), wSize, cfg, allocBytes);
        }
        {
            NnMatmulOpConfig cfg{0, 0, moeExpertIndexesBufferIndex};
            cfg.layerIndex = layerIndex;
            cfg.splitKind = (NnUint)SPLIT_KV_HEAD;
            cfg.splitAxis = 1u; // OUT_ROWS
            cfg.splitUnit = h->headDim;
            cfg.staticStartUnits = plan->kvHeadSplit.starts ? plan->kvHeadSplit.starts[nodeIndex] : 0u;
            cfg.staticLenUnits = plan->kvHeadSplit.lengths ? plan->kvHeadSplit.lengths[nodeIndex] : 0u;
            NnSize allocBytes = 0;
            auto r = computeReplicaRow([&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, idx); });
            if (r.first > 0) {
                cfg.replicateMode = 1u;
                cfg.replicateExpertStrideBytes = r.first;
                cfg.replicateMyOffsetBytes = r.second;
                allocBytes = r.first;
            }
            const NnSize3D wSize = (cfg.replicateMode != 0u) ? kSlice.size : kSlice.sliceSize;
            att.addOp(OP_MATMUL, "block_matmul_k", layerIndex, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), wSize, cfg, allocBytes);
        }
        {
            NnMatmulOpConfig cfg{0, 0, moeExpertIndexesBufferIndex};
            cfg.layerIndex = layerIndex;
            cfg.splitKind = (NnUint)SPLIT_KV_HEAD;
            cfg.splitAxis = 1u; // OUT_ROWS
            cfg.splitUnit = h->headDim;
            cfg.staticStartUnits = plan->kvHeadSplit.starts ? plan->kvHeadSplit.starts[nodeIndex] : 0u;
            cfg.staticLenUnits = plan->kvHeadSplit.lengths ? plan->kvHeadSplit.lengths[nodeIndex] : 0u;
            NnSize allocBytes = 0;
            auto r = computeReplicaRow([&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, idx); });
            if (r.first > 0) {
                cfg.replicateMode = 1u;
                cfg.replicateExpertStrideBytes = r.first;
                cfg.replicateMyOffsetBytes = r.second;
                allocBytes = r.first;
            }
            const NnSize3D wSize = (cfg.replicateMode != 0u) ? vSlice.size : vSlice.sliceSize;
            att.addOp(OP_MATMUL, "block_matmul_v", layerIndex, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, vTempBufferIndex), wSize, cfg, allocBytes);
        }

        if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
            att.addOp(OP_INV_RMS, "block_norm_pre_q", layerIndex, pointerBatchConfig(SRC_BUFFER, qBufferIndex), pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex), size0(), NnInvRmsOpConfig{h->normEpsilon, nQNormColumns});
            att.addOp(OP_RMS_NORM, "block_norm_q", layerIndex, pointerBatchConfig(SRC_BUFFER, qBufferIndex), pointerBatchConfig(SRC_BUFFER, qBufferIndex), size2D(F_32, 1, h->headDim), NnRmsNormOpConfig{invRmsBufferIndex, nQNormColumns});
            att.addOp(OP_INV_RMS, "block_norm_pre_k", layerIndex, pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex), size0(), NnInvRmsOpConfig{h->normEpsilon, nKNormColumns});
            att.addOp(OP_RMS_NORM, "block_norm_k", layerIndex, pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), size2D(F_32, 1, h->headDim), NnRmsNormOpConfig{invRmsBufferIndex, nKNormColumns});
        }

        {
            NnRopeOpConfig cfg{h->ropeType, 1, n->positionPipeIndex, ropeCacheBufferIndex,
                h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen, ropeSlice};
            cfg.layerIndex = layerIndex;
            cfg.splitKind = (NnUint)SPLIT_HEAD;
            cfg.splitUnit = h->headDim;
            cfg.staticLenUnits = multiHeadAttSlice.headLen;
            cfg.staticStartUnits = multiHeadAttSlice.headStart;
            att.addOp(OP_ROPE, "block_rope_q", layerIndex,
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                size0(),
                cfg);
        }
        {
            NnRopeOpConfig cfg{h->ropeType, 0, n->positionPipeIndex, ropeCacheBufferIndex,
                h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen, ropeSlice};
            cfg.layerIndex = layerIndex;
            cfg.splitKind = (NnUint)SPLIT_KV_HEAD;
            cfg.splitUnit = h->headDim;
            cfg.staticLenUnits = plan->kvHeadSplit.lengths ? plan->kvHeadSplit.lengths[nodeIndex] : 0u;
            cfg.staticStartUnits = plan->kvHeadSplit.starts ? plan->kvHeadSplit.starts[nodeIndex] : 0u;
            att.addOp(OP_ROPE, "block_rope_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size0(),
                cfg);
        }
        {
            NnCastOpCodeConfig cfg{layerIndex, (NnUint)SPLIT_KV_HEAD, h->headDim,
                plan->kvHeadSplit.lengths ? plan->kvHeadSplit.lengths[nodeIndex] : 0u};
            att.addOp(OP_CAST, "block_cast_k_pipe", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerBatchedSliceConfig(SRC_PIPE, n->kPipeIndex),
                size0(),
                cfg);
        }
        {
            NnCastOpCodeConfig cfg{layerIndex, (NnUint)SPLIT_KV_HEAD, h->headDim,
                plan->kvHeadSplit.lengths ? plan->kvHeadSplit.lengths[nodeIndex] : 0u};
            att.addOp(OP_CAST, "block_cast_v_pipe", layerIndex,
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                pointerBatchedSliceConfig(SRC_PIPE, n->vPipeIndex),
                size0(),
                cfg);
        }

        nodeBuilder.addSegment(att.build());

        NnSegmentConfigBuilder att2;
        {
            NnShiftOpCodeConfig cfg{n->positionPipeIndex};
            att2.addOp(OP_SHIFT, "block_shift_k", layerIndex,
                pointerBatchConfig(SRC_PIPE, n->kPipeIndex),
                pointerRawConfig(SRC_BUFFER, kBufferIndex),
                size0(),
                cfg);
        }
        {
            NnShiftOpCodeConfig cfg{n->positionPipeIndex};
            att2.addOp(OP_SHIFT, "block_shift_v", layerIndex,
                pointerBatchConfig(SRC_PIPE, n->vPipeIndex),
                pointerRawConfig(SRC_BUFFER, vBufferIndex),
                size0(),
                cfg);
        }

        {
            NnMultiHeadAttOpConfig cfg{
                nodeIndex,
                layerIndex,
                multiHeadAttSlice.headStart,
                h->nHeads,
                stageMaxHeadLen,
                h->nKvHeads,
                h->headDim,
                h->seqLen,
                stageMaxQSliceD0,
                kSlice.sliceSize.x,
                n->positionPipeIndex,
                qBufferIndex,
                kBufferIndex,
                vBufferIndex,
                attBufferIndex};
            // Keep the graph-time planned head length for fallback when runtime sharding is disabled.
            cfg.staticHeadLenUnits = multiHeadAttSlice.headLen;
            cfg.kvHeadStart = plan->kvHeadSplit.starts ? plan->kvHeadSplit.starts[nodeIndex] : 0u;

            att2.addOp(OP_MULTIHEAD_ATT, "block_multihead_att", layerIndex,
                pointerBatchConfig(SRC_BUFFER, mhaOutBufferIndex),
                pointerBatchConfig(SRC_BUFFER, mhaOutBufferIndex),
                size0(),
                cfg);
        }

        if (mhaOutBufferIndex != mhaOutQBufferIndex) {
            att2.addOp(OP_CAST, "block_cast_y2", layerIndex, pointerBatchConfig(SRC_BUFFER, mhaOutBufferIndex), pointerBatchConfig(SRC_BUFFER, mhaOutQBufferIndex), size0(), NnCastOpCodeConfig{});
        }
        {
            NnMatmulOpConfig cfg{0, 0, moeExpertIndexesBufferIndex};
            cfg.layerIndex = layerIndex;
            cfg.splitKind = (NnUint)SPLIT_HEAD;
            cfg.splitAxis = 2u; // IN_COLS (select input-dim shard)
            cfg.splitUnit = h->headDim;
            cfg.staticStartUnits = multiHeadAttSlice.headStart;
            cfg.staticLenUnits = multiHeadAttSlice.headLen;
            cfg.replicateGlobalInDim = h->qDim;
            NnSize allocBytes = 0;
            auto r = computeReplicaCol([&](NnUint idx) { return sliceColMatmulAttUneven(h->weightType, h->qDim, h->dim, h->headDim, plan, idx); });
            if (r.first > 0) {
                cfg.replicateMode = 1u;
                cfg.replicateExpertStrideBytes = r.first;
                cfg.replicateMyOffsetBytes = r.second;
                allocBytes = r.first;
            }
            const NnSize3D wSize = (cfg.replicateMode != 0u) ? woSlice.size : woSlice.sliceSize;
            att2.addOp(OP_MATMUL, "block_matmul_wo", layerIndex, pointerBatchConfig(SRC_BUFFER, mhaOutQBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex), wSize, cfg, allocBytes);
        }
        att2.addOp(OP_CAST, "block_cast_d", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchedSliceConfig(SRC_PIPE, n->zqPipeIndex), size0(), NnCastOpCodeConfig{});
        att2.addSync(n->zqPipeIndex, SYNC_NODE_SLICES);

        // --- FFN Ops ---
        ff.addOp(OP_MERGE_ADD, "block_merge_add2", layerIndex, pointerBatchConfig(SRC_PIPE, n->zqPipeIndex), pointerBatchConfig(SRC_BUFFER, xBufferIndex), size0(), NnMergeAddOpCodeConfig{});
        ff.addOp(OP_INV_RMS, "block_norm_pre_1", layerIndex, pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex), size0(), NnInvRmsOpConfig{h->normEpsilon, 1});
        ff.addOp(OP_RMS_NORM, "block_norm_1", layerIndex, pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex), n->rmsNormSize, NnRmsNormOpConfig{invRmsBufferIndex, 1});

        if (h->archType == QWEN3_MOE) {
            ff.addOp(OP_REPEAT_Z, "block_moe_y_repeat", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex), size0(), NnRepeatZOpCodeConfig{});
            ff.addOp(OP_MATMUL, "block_moe_gate", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex), n->moeGateSize, NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
            ff.addOp(OP_SOFTMAX, "block_moe_softmax", layerIndex, pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex), pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex), size0(), NnSoftmaxOpCodeConfig{});
            ff.addOp(OP_MOE_GATE, "block_moe_gate2", layerIndex, pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex), pointerBatchConfig(SRC_BUFFER, moeSBufferIndex), size0(), NnMoeGateOpCodeConfig{h->nActiveExperts, 1u, moeExpertIndexesBufferIndex});
            
            const NnRowMatmulSliceUneven w1Global = (myStageCfg != nullptr && myStageCfg->nNodes > 0) ? sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, myStageCfg->nodeIndices[0]) : w1Slice;
            const NnRowMatmulSliceUneven w3Global = (myStageCfg != nullptr && myStageCfg->nNodes > 0) ? sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, myStageCfg->nodeIndices[0]) : w3Slice;
            const NnColMatmulSliceUneven w2Global = (myStageCfg != nullptr && myStageCfg->nNodes > 0) ? sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, myStageCfg->nodeIndices[0]) : w2Slice;

            const NnSize3D w1ExpertGlobalSize = size3D(h->weightType, h->nExperts, w1Global.size.y, w1Global.size.x);
            const NnSize3D w3ExpertGlobalSize = size3D(h->weightType, h->nExperts, w3Global.size.y, w3Global.size.x);
            const NnSize3D w2ExpertGlobalSize = size3D(h->weightType, h->nExperts, w2Global.size.y, w2Global.size.x);

            const NnSize3D w1ExpertLocalSize = size3D(h->weightType, h->nExperts, w1Slice.sliceSize.y, w1Slice.sliceSize.x);
            const NnSize3D w3ExpertLocalSize = size3D(h->weightType, h->nExperts, w3Slice.sliceSize.y, w3Slice.sliceSize.x);
            const NnSize3D w2ExpertLocalSize = size3D(h->weightType, h->nExperts, w2Slice.sliceSize.y, w2Slice.sliceSize.x);

            {
                NnMatmulOpConfig cfg{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex};
                cfg.layerIndex = layerIndex;
                cfg.splitKind = (NnUint)SPLIT_FFN;
                cfg.splitAxis = 1u; // OUT_ROWS
                cfg.splitUnit = 1u;
                cfg.staticStartUnits = plan->ffnSplit.starts ? plan->ffnSplit.starts[nodeIndex] : 0u;
                cfg.staticLenUnits = plan->ffnSplit.lengths ? plan->ffnSplit.lengths[nodeIndex] : 0u;
                NnSize allocBytes = 0;
                auto r = computeReplicaRow([&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); });
                if (r.first > 0) {
                    cfg.replicateMode = 1u;
                    cfg.replicateExpertStrideBytes = r.first;
                    cfg.replicateMyOffsetBytes = r.second;
                    allocBytes = r.first * (NnSize)h->nExperts;
                }
                const NnSize3D wSize = (cfg.replicateMode != 0u) ? w1ExpertGlobalSize : w1ExpertLocalSize;
                ff.addOp(OP_MATMUL, "block_matmul_w1", layerIndex, pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex), pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), wSize, cfg, allocBytes);
            }
            {
                NnMatmulOpConfig cfg{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex};
                cfg.layerIndex = layerIndex;
                cfg.splitKind = (NnUint)SPLIT_FFN;
                cfg.splitAxis = 1u; // OUT_ROWS
                cfg.splitUnit = 1u;
                cfg.staticStartUnits = plan->ffnSplit.starts ? plan->ffnSplit.starts[nodeIndex] : 0u;
                cfg.staticLenUnits = plan->ffnSplit.lengths ? plan->ffnSplit.lengths[nodeIndex] : 0u;
                NnSize allocBytes = 0;
                auto r = computeReplicaRow([&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); });
                if (r.first > 0) {
                    cfg.replicateMode = 1u;
                    cfg.replicateExpertStrideBytes = r.first;
                    cfg.replicateMyOffsetBytes = r.second;
                    allocBytes = r.first * (NnSize)h->nExperts;
                }
                const NnSize3D wSize = (cfg.replicateMode != 0u) ? w3ExpertGlobalSize : w3ExpertLocalSize;
                ff.addOp(OP_MATMUL, "block_matmul_w3", layerIndex, pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex), pointerBatchConfig(SRC_BUFFER, moeLBufferIndex), wSize, cfg, allocBytes);
            }
            const NnUint ffnStaticLenUnits = (plan->ffnSplit.lengths != nullptr) ? plan->ffnSplit.lengths[nodeIndex] : w1Slice.inLen;
            ff.addOp(OP_SILU, "block_act", layerIndex,
                pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                size0(),
                NnSiluOpCodeConfig{layerIndex, (NnUint)SPLIT_FFN, 1u, ffnStaticLenUnits});
            ff.addOp(OP_MUL, "block_mul", layerIndex,
                pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                size0(),
                NnMulOpCodeConfig{moeLBufferIndex, layerIndex, (NnUint)SPLIT_FFN, 1u, ffnStaticLenUnits});
            if (moeDBufferIndex != moeDQBufferIndex) {
                ff.addOp(OP_CAST, "block_cast_d2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{layerIndex, (NnUint)SPLIT_FFN, 1u, ffnStaticLenUnits});
            }
            {
                NnMatmulOpConfig cfg{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex};
                cfg.layerIndex = layerIndex;
                cfg.splitKind = (NnUint)SPLIT_FFN;
                cfg.splitAxis = 2u; // IN_COLS
                cfg.splitUnit = 1u;
                cfg.staticStartUnits = plan->ffnSplit.starts ? plan->ffnSplit.starts[nodeIndex] : 0u;
                cfg.staticLenUnits = plan->ffnSplit.lengths ? plan->ffnSplit.lengths[nodeIndex] : 0u;
                cfg.replicateGlobalInDim = ffDim;
                NnSize allocBytes = 0;
                auto r = computeReplicaCol([&](NnUint idx) { return sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, idx); });
                if (r.first > 0) {
                    cfg.replicateMode = 1u;
                    cfg.replicateExpertStrideBytes = r.first;
                    cfg.replicateMyOffsetBytes = r.second;
                    allocBytes = r.first * (NnSize)h->nExperts;
                }
                const NnSize3D wSize = (cfg.replicateMode != 0u) ? w2ExpertGlobalSize : w2ExpertLocalSize;
                ff.addOp(OP_MATMUL, "block_matmul_w2", layerIndex, pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex), pointerBatchConfig(SRC_BUFFER, moeYBufferIndex), wSize, cfg, allocBytes);
            }
            ff.addOp(OP_SCALE, "block_moe_scale", layerIndex, pointerBatchConfig(SRC_BUFFER, moeYBufferIndex), pointerBatchConfig(SRC_BUFFER, moeYBufferIndex), size0(), NnScaleOpCodeConfig{moeSBufferIndex});
            ff.addOp(OP_MERGE_SUM, "block_moe_merge_sum", layerIndex, pointerBatchConfig(SRC_BUFFER, moeYBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex), size0(), NnMergeSumOpCodeConfig{});
        } else {
            if (yBufferIndex != yqBufferIndex) {
                ff.addOp(OP_CAST, "block_cast_y3", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchConfig(SRC_BUFFER, yqBufferIndex), size0(), NnCastOpCodeConfig{});
            }
            {
                NnMatmulOpConfig cfg{0, 0, moeExpertIndexesBufferIndex};
                cfg.layerIndex = layerIndex;
                cfg.splitKind = (NnUint)SPLIT_FFN;
                cfg.splitAxis = 1u; // OUT_ROWS
                cfg.splitUnit = 1u;
                cfg.staticStartUnits = plan->ffnSplit.starts ? plan->ffnSplit.starts[nodeIndex] : 0u;
                cfg.staticLenUnits = plan->ffnSplit.lengths ? plan->ffnSplit.lengths[nodeIndex] : 0u;
                NnSize allocBytes = 0;
                auto r = computeReplicaRow([&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); });
                if (r.first > 0) {
                    cfg.replicateMode = 1u;
                    cfg.replicateExpertStrideBytes = r.first;
                    cfg.replicateMyOffsetBytes = r.second;
                    allocBytes = r.first;
                }
                const NnSize3D wSize = (cfg.replicateMode != 0u) ? w1Slice.size : w1Slice.sliceSize;
                ff.addOp(OP_MATMUL, "block_matmul_w1", layerIndex, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, dBufferIndex), wSize, cfg, allocBytes);
            }
            {
                NnMatmulOpConfig cfg{0, 0, moeExpertIndexesBufferIndex};
                cfg.layerIndex = layerIndex;
                cfg.splitKind = (NnUint)SPLIT_FFN;
                cfg.splitAxis = 1u; // OUT_ROWS
                cfg.splitUnit = 1u;
                cfg.staticStartUnits = plan->ffnSplit.starts ? plan->ffnSplit.starts[nodeIndex] : 0u;
                cfg.staticLenUnits = plan->ffnSplit.lengths ? plan->ffnSplit.lengths[nodeIndex] : 0u;
                NnSize allocBytes = 0;
                auto r = computeReplicaRow([&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); });
                if (r.first > 0) {
                    cfg.replicateMode = 1u;
                    cfg.replicateExpertStrideBytes = r.first;
                    cfg.replicateMyOffsetBytes = r.second;
                    allocBytes = r.first;
                }
                const NnSize3D wSize = (cfg.replicateMode != 0u) ? w3Slice.size : w3Slice.sliceSize;
                ff.addOp(OP_MATMUL, "block_matmul_w3", layerIndex, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, lBufferIndex), wSize, cfg, allocBytes);
            }
            const NnUint ffnStaticLenUnits = (plan->ffnSplit.lengths != nullptr) ? plan->ffnSplit.lengths[nodeIndex] : w1Slice.inLen;
            ff.addOp(OP_SILU, "block_act", layerIndex,
                pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                size0(),
                NnSiluOpCodeConfig{layerIndex, (NnUint)SPLIT_FFN, 1u, ffnStaticLenUnits});
            ff.addOp(OP_MUL, "block_mul", layerIndex,
                pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                size0(),
                NnMulOpCodeConfig{lBufferIndex, layerIndex, (NnUint)SPLIT_FFN, 1u, ffnStaticLenUnits});
            if (dBufferIndex != dqBufferIndex) {
                ff.addOp(OP_CAST, "block_cast_d2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{layerIndex, (NnUint)SPLIT_FFN, 1u, ffnStaticLenUnits});
            }
            {
                NnMatmulOpConfig cfg{0, 0, moeExpertIndexesBufferIndex};
                cfg.layerIndex = layerIndex;
                cfg.splitKind = (NnUint)SPLIT_FFN;
                cfg.splitAxis = 2u; // IN_COLS
                cfg.splitUnit = 1u;
                cfg.staticStartUnits = plan->ffnSplit.starts ? plan->ffnSplit.starts[nodeIndex] : 0u;
                cfg.staticLenUnits = plan->ffnSplit.lengths ? plan->ffnSplit.lengths[nodeIndex] : 0u;
                cfg.replicateGlobalInDim = ffDim;
                NnSize allocBytes = 0;
                auto r = computeReplicaCol([&](NnUint idx) { return sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, idx); });
                if (r.first > 0) {
                    cfg.replicateMode = 1u;
                    cfg.replicateExpertStrideBytes = r.first;
                    cfg.replicateMyOffsetBytes = r.second;
                    allocBytes = r.first;
                }
                const NnSize3D wSize = (cfg.replicateMode != 0u) ? w2Slice.size : w2Slice.sliceSize;
                ff.addOp(OP_MATMUL, "block_matmul_w2", layerIndex, pointerBatchConfig(SRC_BUFFER, dqBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex), wSize, cfg, allocBytes);
            }
        }
        
        ff.addOp(OP_CAST, "block_cast_d3", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchedSliceConfig(SRC_PIPE, n->zqPipeIndex), size0(), NnCastOpCodeConfig{});
        ff.addSync(n->zqPipeIndex, SYNC_NODE_SLICES);

        nodeBuilder.addSegment(att2.build());
        nodeBuilder.addSegment(ff.build());
    }

    if (!isLastStage) {
        NnSegmentConfigBuilder ppSendSeg;
        
        // 1. Merge Add: 将本 Stage 最后一层的 TP 分片合并为完整激活值
        // 结果存入 xBufferIndex
        ppSendSeg.addOp(OP_MERGE_ADD, "pp_stage_merge", endLayer-1,
            pointerBatchConfig(SRC_PIPE, n->zqPipeIndex),
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            size0(), NnMergeAddOpCodeConfig{});

        // 2. Cast: 将完整激活值写入 X Pipe (复用通信管道)
        ppSendSeg.addOp(OP_CAST, "pp_cast_out", endLayer-1,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            pointerBatchConfig(SRC_PIPE, n->xPipeIndex),
            size0(), NnCastOpCodeConfig{});
            
        // 3. Send: 触发 PP 发送
        ppSendSeg.addSync(n->xPipeIndex, SYNC_PP_SEND);
        
        nodeBuilder.addSegment(ppSendSeg.build());
    }

    // 6. End Segment (Final Norm & Logits)
    NnSegmentConfigBuilder end;
    if (isLastStage) {
        end.addOp(OP_MERGE_ADD, "final_merge_add", 0, pointerBatchConfig(SRC_PIPE, n->zqPipeIndex), pointerBatchConfig(SRC_BUFFER, xBufferIndex), size0(), NnMergeAddOpCodeConfig{});
        end.addOp(OP_INV_RMS, "final_norm_pre", 0, pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex), size0(), NnInvRmsOpConfig{h->normEpsilon, 1});
        end.addOp(OP_RMS_NORM, "final_norm", 0, pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex), n->rmsNormSize, NnRmsNormOpConfig{invRmsBufferIndex, 1});
        
        if (yBufferIndex != yqBufferIndex) {
            end.addOp(OP_CAST, "final_cast_y", 0, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchConfig(SRC_BUFFER, yqBufferIndex), size0(), NnCastOpCodeConfig{});
        }
        
        {
            NnMatmulOpConfig cfg{};
            NnSize allocBytes = 0;
            auto r = computeReplicaRow([&](NnUint idx) { return sliceRowMatmulLogitsUneven(h->weightType, h->dim, h->vocabSize, plan, idx); });
            if (r.first > 0) {
                cfg.replicateMode = 1u;
                cfg.replicateExpertStrideBytes = r.first;
                cfg.replicateMyOffsetBytes = r.second;
                allocBytes = r.first;
            }
            cfg.layerIndex = (h->nLayers > 0u) ? (h->nLayers - 1u) : 0u;
            cfg.splitKind = (NnUint)SPLIT_VOCAB;
            cfg.splitAxis = 1u; // OUT_ROWS
            cfg.splitUnit = 1u;
            cfg.staticStartUnits = wclsSlice.inStart;
            cfg.staticLenUnits = wclsSlice.inLen;
            const NnSize3D wSize = (cfg.replicateMode != 0u) ? wclsSlice.size : wclsSlice.sliceSize;
            end.addOp(OP_MATMUL, "final_matmul_logits", 0, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex), wSize, cfg, allocBytes);
        }
        
        NnCastOpCodeConfig castLogitsCfg{};
        castLogitsCfg.layerIndex = (h->nLayers > 0u) ? (h->nLayers - 1u) : 0u;
        castLogitsCfg.splitKind = (NnUint)SPLIT_VOCAB;
        castLogitsCfg.splitUnit = 1u;
        castLogitsCfg.staticLenUnits = wclsSlice.inLen;

        end.addOp(OP_CAST, "final_cast_logits", 0, 
            pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex), 
            pointerBatchedSliceConfig(SRC_PIPE, n->logitsPipeIndex), 
            size0(), castLogitsCfg);
        
        end.addSync(n->logitsPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);
    }
    nodeBuilder.addSegment(end.build());
    if (nodeIndex == 0 && !isLastStage) {
        NnSegmentConfigBuilder rootWaitSeg;
        
        // 这是一个纯同步 Segment，不包含计算 Op
        // 语义：Node 0 等待 Last Stage 的节点发送 Logits 给它
        rootWaitSeg.addSync(n->logitsPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);
        
        nodeBuilder.addSegment(rootWaitSeg.build());
    }

    NnNodeConfig config = nodeBuilder.build();
    config.partitionPlan = plan;
    return config;
}

LlmNet buildLlmNetUneven(LlmHeader *h, NnUint nNodes, NnUint nBatches, const NnUnevenPartitionPlan* plan) {
    LlmNet n;
    n.header = h;

    // 1. Global Dimensions
    n.tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    n.rmsNormSize = size1D(F_32, h->dim);
    n.qkRmsNormSize = size1D(F_32, h->headDim);
    n.moeGateSize = size2D(F_32, h->dim, h->nExperts);

    // 2. Global Pipes
    NnNetConfigBuilder netBuilder(nNodes, nBatches);
    n.positionPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));
    n.tokenPipeIndex = netBuilder.addPipe("TOK", size2D(F_32, nBatches, 1));
    n.xPipeIndex = netBuilder.addPipe("X", size2D(F_32, nBatches, h->dim));
    n.logitsPipeIndex = netBuilder.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    n.zqPipeIndex = netBuilder.addPipe("ZQ", size2D(h->syncType, nBatches, h->dim * nNodes)); // Safe size

    // KV all-gather pipes (full kvDim per batch)
    n.kPipeIndex = netBuilder.addPipe("KP", size2D(F_32, nBatches, h->kvDim));
    n.vPipeIndex = netBuilder.addPipe("VP", size2D(F_32, nBatches, h->kvDim));

    // Dedicated control pipe for stage-root sharding updates.
    const NnSize shardingBytes = sizeof(NnShardingUpdateHeader) + (NnSize)(N_SPLIT_KINDS * 2u * nNodes * sizeof(NnUint));
    const NnUint shardingWords = (NnUint)((shardingBytes + sizeof(NnUint) - 1u) / sizeof(NnUint));
    n.shardingPipeIndex = netBuilder.addPipe("SHD", size2D(F_32, nBatches, shardingWords));

    netBuilder.addPreSync(n.positionPipeIndex);
    n.netConfig = netBuilder.build();
    n.nodeConfigs = new NnNodeConfig[nNodes];

    // 3. Loop Nodes and Build Internal Graph
    for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
        const NnStageConfig* myStage = getStageForNode(plan, nodeIndex);

        NnUint startLayer = 0;
        NnUint endLayer = h->nLayers;
        bool isFirstStage = true;
        bool isLastStage = true;

        if (myStage) {
            startLayer = myStage->startLayer;
            endLayer = myStage->endLayer;
            isFirstStage = (myStage->stageIndex == 0);
            isLastStage = (myStage->stageIndex == plan->nStages - 1);
        }
        n.nodeConfigs[nodeIndex] = buildLlmNodeInternal(
            nodeIndex, h, &n, plan, nBatches,
            startLayer, endLayer, isFirstStage, isLastStage
        );
        n.nodeConfigs[nodeIndex].partitionPlan = plan;
    }

    return n;
}

void releaseLlmNet(LlmNet *net) {
    for (NnUint nodeIndex = 0u; nodeIndex < net->netConfig.nNodes; nodeIndex++)
        releaseNodeConfig(&net->nodeConfigs[nodeIndex]);
    releaseNetConfig(&net->netConfig);
    delete[] net->nodeConfigs;
}

void loadLlmNetWeight(const char *path, LlmNet *net, NnRootWeightLoader *loader) {
    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);
#if DEBUG_USE_MMAP_FOR_WEIGHTS
    assert(net->netConfig.nNodes == 1u);
#else
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    printf("💿 Loading weights...\n");
#endif
    Timer timer;
    NnByte *data = (NnByte *)file.data;
    NnByte *b = &data[net->header->headerSize];
    b += loader->loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);

    for (NnUint layerIndex = 0u; layerIndex < net->header->nLayers; layerIndex++) {
        b += loader->loadRowMatmulSlices("block_matmul_q", layerIndex, 0u, &net->qSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_k", layerIndex, 0u, &net->kSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_v", layerIndex, 0u, &net->vSlice, b);
        b += loader->loadColMatmulSlices("block_matmul_wo", layerIndex, 0u, &net->woSlice, b);

        if (net->header->nExperts > 0u) {
            b += loader->loadAll("block_moe_gate", layerIndex, net->moeGateSize.nBytes, b);
            for (NnUint expertIndex = 0u; expertIndex < net->header->nExperts; expertIndex++) {
                b += loader->loadRowMatmulSlices("block_matmul_w1", layerIndex, expertIndex, &net->w1Slice, b);
                b += loader->loadColMatmulSlices("block_matmul_w2", layerIndex, expertIndex, &net->w2Slice, b);
                b += loader->loadRowMatmulSlices("block_matmul_w3", layerIndex, expertIndex, &net->w3Slice, b);
            }
        } else {
            b += loader->loadRowMatmulSlices("block_matmul_w1", layerIndex, 0u, &net->w1Slice, b);
            b += loader->loadColMatmulSlices("block_matmul_w2", layerIndex, 0u, &net->w2Slice, b);
            b += loader->loadRowMatmulSlices("block_matmul_w3", layerIndex, 0u, &net->w3Slice, b);
        }

        if (net->header->archType == QWEN3 || net->header->archType == QWEN3_MOE) {
            b += loader->loadAll("block_norm_q", layerIndex, net->qkRmsNormSize.nBytes, b);
            b += loader->loadAll("block_norm_k", layerIndex, net->qkRmsNormSize.nBytes, b);
        }

        b += loader->loadAll("block_norm_0", layerIndex, net->rmsNormSize.nBytes, b);
        b += loader->loadAll("block_norm_1", layerIndex, net->rmsNormSize.nBytes, b);

        if (timer.elapsedMiliseconds() > 10000) {
            printf("💿 Loaded %u/%u\n", layerIndex + 1, net->header->nLayers);
            timer.reset();
        }
    }

    b += loader->loadAll("final_norm", 0u, net->rmsNormSize.nBytes, b);
    b += loader->loadRowMatmulSlices("final_matmul_logits", 0u, 0u, &net->wclsSlice, b);

    long long missingBytes = (long long)(b - data) - net->header->fileSize;
    if (missingBytes != 0u)
        throw std::runtime_error("Missing bytes in weight file: " + std::to_string(missingBytes));
    printf("💿 Weights loaded\n");

    loader->finish();
}

void loadLlmNetWeightUneven(const char *path, LlmNet *net, NnLocalWeightLoader *loader, 
                            const NnUnevenPartitionPlan* plan, NnUint nodeIndex) {
    
    // 1. 自动计算层范围
    NnUint startLayer = 0;
    NnUint endLayer = net->header->nLayers;
    bool isFirstStage = true;
    bool isLastStage = true;

    // 查表确定身份
    const NnStageConfig* myStage = getStageForNode(plan, nodeIndex);
    if (myStage) {
        startLayer = myStage->startLayer;
        endLayer = myStage->endLayer;
        isFirstStage = (myStage->stageIndex == 0);
        isLastStage = (myStage->stageIndex == plan->nStages - 1);
        printf("   [PP] Node %u: Responsible for Layers %u-%u %s%s\n", 
            nodeIndex, startLayer, endLayer, 
            isFirstStage ? "[First]" : "", isLastStage ? "[Last]" : "");
    } else {
        // 如果找不到 Plan (或者纯 TP 模式)，默认负责所有层
        printf("   [PP] Node %u: No stage info found (assuming Full/TP mode)\n", nodeIndex);
    }

    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    
    printf("💿 Loading weights for Node %u (Layers [%u, %u))...\n", nodeIndex, startLayer, endLayer);

    const bool prefetchStageFull = envFlag("DLLAMA_STAGE_PREFETCH_FULL");
    const bool replicateStageWeights = true;

    Timer timer;
    NnByte *data = (NnByte *)file.data;
    NnByte *b = &data[net->header->headerSize];
    LlmHeader *h = net->header;

    // --- 1. Embedding ---
    if (isFirstStage) {
        if (prefetchStageFull) {
            maybeWillNeedPages(b, net->tokenEmbeddingSize.nBytes);
        }
        b += loader->loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);
    } else {
        b += net->tokenEmbeddingSize.nBytes; 
    }

    // --- 2. 逐层加载 ---
    for (NnUint layerIndex = 0u; layerIndex < h->nLayers; layerIndex++) {
        
        bool isMyLayer = (layerIndex >= startLayer && layerIndex < endLayer);
        
        // 预计算该层的理论大小 (用于 Skip 或 Verify)
        NnSize layerBytes = calculateLayerBytes(h, net->moeGateSize, net->rmsNormSize, net->qkRmsNormSize);

        if (isMyLayer) {
            NnByte* layerStartPtr = b;

            // Prefetch the full layer region so every device in the stage
            // pages in all weights for that stage's layers.
            if (prefetchStageFull) {
                maybeWillNeedPages(layerStartPtr, layerBytes);
            }

            // Attention
            if (replicateStageWeights && myStage != nullptr) {
                b += loader->loadRowMatmulSlicesUnevenReplicated("block_matmul_q", layerIndex, 0,
                    [&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->headSplit, h->qDim, idx); }, myStage, b);
                b += loader->loadRowMatmulSlicesUnevenReplicated("block_matmul_k", layerIndex, 0,
                    [&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, idx); }, myStage, b);
                b += loader->loadRowMatmulSlicesUnevenReplicated("block_matmul_v", layerIndex, 0,
                    [&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, idx); }, myStage, b);
                b += loader->loadColMatmulSlicesUnevenReplicated("block_matmul_wo", layerIndex, 0,
                    [&](NnUint idx) { return sliceColMatmulAttUneven(h->weightType, h->qDim, h->dim, h->headDim, plan, idx); }, myStage, b);
            } else {
                b += loader->loadRowMatmulSlicesUneven("block_matmul_q", layerIndex, 0, 
                    [&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->headSplit, h->qDim, idx); }, b);
                b += loader->loadRowMatmulSlicesUneven("block_matmul_k", layerIndex, 0, 
                    [&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, idx); }, b);
                b += loader->loadRowMatmulSlicesUneven("block_matmul_v", layerIndex, 0, 
                    [&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, idx); }, b);
                b += loader->loadColMatmulSlicesUneven("block_matmul_wo", layerIndex, 0, 
                    [&](NnUint idx) { return sliceColMatmulAttUneven(h->weightType, h->qDim, h->dim, h->headDim, plan, idx); }, b);
            }

            // FFN / MoE
            NnUint ffDim = (h->archType == QWEN3_MOE) ? h->moeHiddenDim : h->hiddenDim;
            if (h->nExperts > 0) {
                b += loader->loadAll("block_moe_gate", layerIndex, net->moeGateSize.nBytes, b);
                for (NnUint expertIndex = 0u; expertIndex < h->nExperts; expertIndex++) {
                    if (replicateStageWeights && myStage != nullptr) {
                        b += loader->loadRowMatmulSlicesUnevenReplicated("block_matmul_w1", layerIndex, expertIndex,
                            [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, myStage, b);
                        b += loader->loadColMatmulSlicesUnevenReplicated("block_matmul_w2", layerIndex, expertIndex,
                            [&](NnUint idx) { return sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, idx); }, myStage, b);
                        b += loader->loadRowMatmulSlicesUnevenReplicated("block_matmul_w3", layerIndex, expertIndex,
                            [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, myStage, b);
                    } else {
                        b += loader->loadRowMatmulSlicesUneven("block_matmul_w1", layerIndex, expertIndex, 
                            [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, b);
                        b += loader->loadColMatmulSlicesUneven("block_matmul_w2", layerIndex, expertIndex, 
                            [&](NnUint idx) { return sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, idx); }, b);
                        b += loader->loadRowMatmulSlicesUneven("block_matmul_w3", layerIndex, expertIndex, 
                            [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, b);
                    }
                }
            } else {
                if (replicateStageWeights && myStage != nullptr) {
                    b += loader->loadRowMatmulSlicesUnevenReplicated("block_matmul_w1", layerIndex, 0,
                        [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, myStage, b);
                    b += loader->loadColMatmulSlicesUnevenReplicated("block_matmul_w2", layerIndex, 0,
                        [&](NnUint idx) { return sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, idx); }, myStage, b);
                    b += loader->loadRowMatmulSlicesUnevenReplicated("block_matmul_w3", layerIndex, 0,
                        [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, myStage, b);
                } else {
                    b += loader->loadRowMatmulSlicesUneven("block_matmul_w1", layerIndex, 0, 
                        [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, b);
                    b += loader->loadColMatmulSlicesUneven("block_matmul_w2", layerIndex, 0, 
                        [&](NnUint idx) { return sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, idx); }, b);
                    b += loader->loadRowMatmulSlicesUneven("block_matmul_w3", layerIndex, 0, 
                        [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, b);
                }
            }

            // Norms
            if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
                b += loader->loadAll("block_norm_q", layerIndex, net->qkRmsNormSize.nBytes, b);
                b += loader->loadAll("block_norm_k", layerIndex, net->qkRmsNormSize.nBytes, b);
            }
            b += loader->loadAll("block_norm_0", layerIndex, net->rmsNormSize.nBytes, b);
            b += loader->loadAll("block_norm_1", layerIndex, net->rmsNormSize.nBytes, b);

            // 校验
            NnSize actualBytes = (NnSize)(b - layerStartPtr);
            
            if (actualBytes != layerBytes) {
                // 如果这里报错，说明 calculateLayerBytes 算错了，或者文件里有 Padding
                printf("🚨 Layer %u Mismatch!\n", layerIndex);
                printf("   Expected (Skip Size): %zu bytes\n", layerBytes);
                printf("   Actual   (Load Size): %zu bytes\n", actualBytes);
                printf("   Diff: %ld bytes\n", (long)(actualBytes - layerBytes));
                throw std::runtime_error("Weight file alignment error");
            } else {
                // 校验通过，说明 Skip 逻辑是安全的
                // printf("✅ Layer %u alignment verified.\n", layerIndex);
            }

        } else {
            // [Skip]
            b += layerBytes;
        }

        if (timer.elapsedMiliseconds() > 5000) {
            printf("💿 Loaded %u/%u layers...\n", layerIndex + 1, h->nLayers);
            timer.reset();
        }
    }

    // --- 3. Final Layers ---
    NnSize finalBlockBytes = net->rmsNormSize.nBytes + size2D(h->weightType, h->dim, h->vocabSize).nBytes;
    
    if (isLastStage) {
        NnByte* finalStart = b;
        if (prefetchStageFull) {
            maybeWillNeedPages(finalStart, finalBlockBytes);
        }
        b += loader->loadAll("final_norm", 0u, net->rmsNormSize.nBytes, b);
        if (replicateStageWeights && myStage != nullptr) {
            b += loader->loadRowMatmulSlicesUnevenReplicated("final_matmul_logits", 0u, 0u,
                [&](NnUint idx) {
                    return sliceRowMatmulLogitsUneven(h->weightType, h->dim, h->vocabSize, plan, idx);
                }, myStage, b);
        } else {
            b += loader->loadRowMatmulSlicesUneven("final_matmul_logits", 0u, 0u, 
                [&](NnUint idx) { 
                    return sliceRowMatmulLogitsUneven(h->weightType, h->dim, h->vocabSize, plan, idx); 
                }, b);
        }
        
        if ((NnSize)(b - finalStart) != finalBlockBytes) {
             throw std::runtime_error("Final block size mismatch");
        }
    } else {
        b += finalBlockBytes;
    }

    // --- 4. 结束检查 ---
    long long diff = (long long)(b - data) - net->header->fileSize;
    if (diff != 0) {
        printf("⚠️ Warning: File pointer drift by %lld bytes (Padding or Error?)\n", diff);
    }
    
    loader->finish();
}