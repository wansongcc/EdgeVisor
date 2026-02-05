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

    // Opt-in: force all OP_MATMUL ops to take the view-aware code path.
    // This is useful for distributed A/B checks while keeping buffer layouts unchanged (offset=0).
    // NOTE: This will likely disable some fast paths (e.g. llamafile sgemm) when enabled.
    const bool forceMatmulViews = (std::getenv("DLLAMA_FORCE_MATMUL_VIEWS") != nullptr);
    auto makeMatmulCfg = [&](NnUint nExperts, NnUint nActiveExperts, NnUint activeExpertIndexesBufferIndex, NnUint inLen, NnUint outLen) -> NnMatmulOpConfig {
        NnMatmulOpConfig cfg{};
        cfg.nExperts = nExperts;
        cfg.nActiveExperts = nActiveExperts;
        cfg.activeExpertIndexesBufferIndex = activeExpertIndexesBufferIndex;
        if (forceMatmulViews) {
            cfg.aView = NnTensorView{0u, 0u, inLen, 0u, 0u};
            cfg.cView = NnTensorView{0u, 0u, outLen, 0u, 0u};
        }
        return cfg;
    };

    NnNetConfigBuilder netBuilder(nNodes, nBatches);

    n.positionPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));
    n.tokenPipeIndex = netBuilder.addPipe("TOK", size2D(F_32, nBatches, 1));
    n.xPipeIndex = netBuilder.addPipe("X", size2D(F_32, nBatches, h->dim));
    n.logitsPipeIndex = netBuilder.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    const NnUint zqPipeIndex = netBuilder.addPipe("ZQ", size2D(h->syncType, nBatches, h->dim * nNodes));
    n.zqPipeIndex = zqPipeIndex;
    // Keep the same control pipe layout as the uneven build (unused in uniform mode).
    n.planPipeIndex = netBuilder.addPipe("PLN", size2D(F_32, nBatches, 8));

    n.kvAggKPipeIndex = (NnUint)-1;
    n.kvAggVPipeIndex = (NnUint)-1;
    if (std::getenv("DLLAMA_KV_AGGREGATE") != nullptr) {
        // KC/VC holds the per-token KV vectors for the current batch window.
        // Each row corresponds to POS[batchIndex] (position + batchIndex).
        n.kvAggKPipeIndex = netBuilder.addPipe("KC", size2D(F_32, nBatches, h->kvDim));
        n.kvAggVPipeIndex = netBuilder.addPipe("VC", size2D(F_32, nBatches, h->kvDim));
    }

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
                makeMatmulCfg(0u, 0u, moeExpertIndexesBufferIndex, n.qSlice.n, n.qSlice.d0));
            att.addOp(
                OP_MATMUL, "block_matmul_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size2D(h->weightType, n.kSlice.n, n.kSlice.d0),
                makeMatmulCfg(0u, 0u, moeExpertIndexesBufferIndex, n.kSlice.n, n.kSlice.d0));
            att.addOp(
                OP_MATMUL, "block_matmul_v", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                size2D(h->weightType, n.vSlice.n, n.vSlice.d0),
                makeMatmulCfg(0u, 0u, moeExpertIndexesBufferIndex, n.vSlice.n, n.vSlice.d0));

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
                    multiHeadAttSlice.nHeads, multiHeadAttSlice.nHeads0,
                    h->nKvHeads, h->headDim, h->seqLen, n.qSlice.d0, kvCacheSlice.kvDim0,
                    0u, n.qSlice.d0,
                    0u, kvCacheSlice.kvDim0,
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
                makeMatmulCfg(0u, 0u, moeExpertIndexesBufferIndex, n.woSlice.n0, n.woSlice.d));
            att.addOp(
                OP_CAST, "block_cast_d", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchedSliceConfigTagged(SRC_PIPE, zqPipeIndex, NN_SLICE_STACKED_BY_NODE),
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
                    makeMatmulCfg(0u, 0u, moeExpertIndexesBufferIndex, h->dim, nExpertsOr1));
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
                    makeMatmulCfg(h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex, n.w1Slice.n, n.w1Slice.d0));
                ff.addOp(
                    OP_MATMUL, "block_matmul_w3", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeLBufferIndex),
                    size3D(h->weightType, h->nExperts, n.w3Slice.n, n.w3Slice.d0),
                    makeMatmulCfg(h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex, n.w3Slice.n, n.w3Slice.d0));
                if (h->hiddenAct == HIDDEN_ACT_GELU) {
                    ff.addOp(
                        OP_GELU, "block_act", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                        size0(),
                        NnGeluOpCodeConfig{NnTensorView{0u, 0u, 0u, 0u, 1u}});
                } else {
                    ff.addOp(
                        OP_SILU, "block_act", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                        size0(),
                        NnSiluOpCodeConfig{NnTensorView{0u, 0u, 0u, 0u, 1u}});
                }
                ff.addOp(
                    OP_MUL, "block_mul", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    size0(),
                    NnMulOpCodeConfig{moeLBufferIndex, NnTensorView{0u, 0u, 0u, 0u, 1u}});
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
                    makeMatmulCfg(h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex, n.w2Slice.n0, n.w2Slice.d));
                ff.addOp(
                    OP_SCALE, "block_moe_scale", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    size0(),
                    NnScaleOpCodeConfig{moeSBufferIndex, NnTensorView{0u, 0u, 0u, 0u, 1u}});
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
                    makeMatmulCfg(0u, 0u, moeExpertIndexesBufferIndex, n.w1Slice.n, n.w1Slice.d0));
                ff.addOp(
                    OP_MATMUL, "block_matmul_w3", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, lBufferIndex),
                    size2D(h->weightType, n.w3Slice.n, n.w3Slice.d0),
                    makeMatmulCfg(0u, 0u, moeExpertIndexesBufferIndex, n.w3Slice.n, n.w3Slice.d0));
                if (h->hiddenAct == HIDDEN_ACT_GELU) {
                    ff.addOp(
                        OP_GELU, "block_act", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                        size0(),
                        NnGeluOpCodeConfig{NnTensorView{0u, 0u, 0u, 0u, 1u}});
                } else {
                    ff.addOp(
                        OP_SILU, "block_act", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                        size0(),
                        NnSiluOpCodeConfig{NnTensorView{0u, 0u, 0u, 0u, 1u}});
                }
                ff.addOp(
                    OP_MUL, "block_mul", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    size0(),
                    NnMulOpCodeConfig{lBufferIndex, NnTensorView{0u, 0u, 0u, 0u, 1u}});
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
                    makeMatmulCfg(0u, 0u, moeExpertIndexesBufferIndex, n.w2Slice.n0, n.w2Slice.d));
            }
            ff.addOp(
                OP_CAST, "block_cast_d3", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchedSliceConfigTagged(SRC_PIPE, zqPipeIndex, NN_SLICE_STACKED_BY_NODE),
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
        end.addOp(
            OP_MATMUL, "final_matmul_logits", 0,
            pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
            pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
            size2D(h->weightType, n.wclsSlice.n, n.wclsSlice.d0),
            makeMatmulCfg(0u, 0u, 0u, n.wclsSlice.n, n.wclsSlice.d0));
        end.addOp(
            OP_CAST, "final_cast_logits", 0,
            pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
            pointerBatchedSliceConfigTagged(SRC_PIPE, n.logitsPipeIndex, NN_SLICE_VOCAB),
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

    const NnStageConfig* myStage = getStageForNode(plan, nodeIndex);
    const bool stageFullWeights = (std::getenv("DLLAMA_STAGE_FULL_WEIGHTS") != nullptr) && (myStage != nullptr);
    // When stageFullWeights is enabled, we treat this as the single “full residency” mode toggle:
    // - stage-local weights are loaded in full
    // - attention/ffn/logits activations are allocated as full buffers
    // - ops run on per-node slices via PNTR_BATCHED_SLICE (and explicit start/stride when needed)
    const bool fullAttBuffers = stageFullWeights && (plan != nullptr);
    const bool fullFfnBuffers = stageFullWeights && (plan != nullptr);
    const bool fullLogitsBuffers = stageFullWeights && (plan != nullptr);

    auto getenvIntOr = [](const char *name, int fallback) -> int {
        const char *v = std::getenv(name);
        if (v == nullptr || v[0] == '\0') return fallback;
        char *end = nullptr;
        long x = std::strtol(v, &end, 10);
        if (end == v) return fallback;
        return (int)x;
    };

    // Per-layer plan barrier/apply is a CPU-only hook for online repartition.
    // Keep it opt-in for normal runs.
    //
    // NOTE: Legacy env hard-migrate variables are treated as a deprecated *enable signal*
    // only. Actual trigger/move parameters are now provided via PlanCommand cache.
    const bool legacyHardMigrateRequested =
        (std::getenv("DLLAMA_HARD_MIGRATE_POS") != nullptr) ||
        (std::getenv("DLLAMA_HARD_MIGRATE_LAYER") != nullptr) ||
        (std::getenv("DLLAMA_HARD_MIGRATE_KIND") != nullptr) ||
        (std::getenv("DLLAMA_HARD_MIGRATE_HEAD_MOVE") != nullptr) ||
        (std::getenv("DLLAMA_HARD_MIGRATE_FFN_MOVE") != nullptr);

    const bool enablePlanBarrier =
        (std::getenv("DLLAMA_ENABLE_PLAN_BARRIER") != nullptr) || legacyHardMigrateRequested;

    // 2. 计算切分 (Slicing)
    NnKvCacheSliceUneven kvCacheSlice = sliceKvCacheUneven(h->seqLen, h->headDim, plan, nodeIndex);
    NnMultiHeadAttSliceUneven multiHeadAttSlice = sliceMultiHeadAttUneven(nBatches, h->nHeads, h->seqLen, plan, nodeIndex);
    
    NnRowMatmulSliceUneven qSlice = sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->headSplit, h->qDim, nodeIndex);
    NnRowMatmulSliceUneven kOwnedSlice = sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, nodeIndex);
    NnRowMatmulSliceUneven vOwnedSlice = sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, nodeIndex);

    // KV redundancy compute slice:
    // - Only meaningful when stageFullWeights/fullAttBuffers are enabled (full KV cache buffer exists).
    // - Uses kvHeadComputeSplit which may overlap across nodes; do NOT use it for pointer slicing.
    bool enableKvRedundancy = fullAttBuffers && plan != nullptr &&
        plan->kvHeadComputeSplit.starts != nullptr && plan->kvHeadComputeSplit.lengths != nullptr;

    // Online migration test hook: by default we keep the older, correctness-first behavior and
    // disable KV redundancy when plan barrier/migration is enabled.
    // If you want redundancy to provide a "no extra comm" migration experiment, opt-in via:
    //   DLLAMA_ENABLE_KV_REDUNDANCY_DURING_MIGRATION=1
    if (enablePlanBarrier && std::getenv("DLLAMA_ENABLE_KV_REDUNDANCY_DURING_MIGRATION") == nullptr) {
        enableKvRedundancy = false;
    }

    NnRowMatmulSliceUneven kSlice = enableKvRedundancy
        ? sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadComputeSplit, h->kvDim, nodeIndex)
        : kOwnedSlice;
    NnRowMatmulSliceUneven vSlice = enableKvRedundancy
        ? sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadComputeSplit, h->kvDim, nodeIndex)
        : vOwnedSlice;
    NnColMatmulSliceUneven woSlice = sliceColMatmulAttUneven(h->weightType, h->qDim, h->dim, h->headDim, plan, nodeIndex);

    NnRowMatmulSliceUneven w1Slice = sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, nodeIndex);
    NnColMatmulSliceUneven w2Slice = sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, nodeIndex);
    NnRowMatmulSliceUneven w3Slice = sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, nodeIndex);
    NnRowMatmulSliceUneven wclsSlice = sliceRowMatmulLogitsUneven(h->weightType, h->dim, h->vocabSize, plan, nodeIndex);

    NnRopeSliceUneven unevenRope = sliceRopeUneven(h->ropeType, h->seqLen, h->kvDim, h->nKvHeads, h->headDim, h->ropeTheta, plan, nodeIndex);
    
    // 适配旧版 Rope Config
    // Build separate RoPE slices/caches for Q and K so KV redundancy can extend kvDimStart/kvDim0
    // without affecting Q's cache layout.
    NnRopeSlice ropeSliceQ;
    std::memset(&ropeSliceQ, 0, sizeof(NnRopeSlice));
    ropeSliceQ.qDim0 = unevenRope.qDimLen;
    ropeSliceQ.qDimStart = unevenRope.qDimStart;
    ropeSliceQ.qDimEnd = unevenRope.qDimStart + unevenRope.qDimLen;
    ropeSliceQ.qShift = unevenRope.qShift;
    ropeSliceQ.kvDim = unevenRope.kvDim;
    ropeSliceQ.kvDim0 = unevenRope.kvDimLen;
    ropeSliceQ.kvDimStart = unevenRope.kvDimStart;
    ropeSliceQ.sliceDim = unevenRope.sliceDim;
    ropeSliceQ.seqLen = unevenRope.seqLen;
    ropeSliceQ.headDim = unevenRope.headDim;
    ropeSliceQ.nKvHeads = unevenRope.nKvHeads;
    ropeSliceQ.ropeTheta = unevenRope.ropeTheta;
    ropeSliceQ.cacheSize = unevenRope.cacheSize;

    // K RoPE cache should cover exactly the K buffer range.
    // IMPORTANT: when KV redundancy extends kvHeadComputeSplit beyond kvHeadSplit, using Q-derived
    // (qDimStart/qDimEnd) for K cache can under-fill the cache or cause OOB reads in ropeForward
    // (especially when nHeads == nKvHeads, i.e. gqaGroupSize==1).
    NnRopeSlice ropeSliceK = ropeSliceQ;
    ropeSliceK.kvDim0 = kSlice.inLen;
    ropeSliceK.kvDimStart = kSlice.inStart;

    // Make K cache self-contained: [kvDimStart, kvDimStart+kvDim0)
    ropeSliceK.qDimStart = ropeSliceK.kvDimStart;
    ropeSliceK.qDim0 = ropeSliceK.kvDim0;
    ropeSliceK.qDimEnd = ropeSliceK.qDimStart + ropeSliceK.qDim0;
    ropeSliceK.qShift = 0u;

    if (h->ropeType == ROPE_LLAMA || h->ropeType == ROPE_LLAMA3_1) {
        ropeSliceK.sliceDim = ropeSliceK.kvDim0;
        assert((ropeSliceK.sliceDim % 2u) == 0u);
        ropeSliceK.cacheSize = size2D(F_32, h->seqLen, ropeSliceK.sliceDim);
    } else if (h->ropeType == ROPE_FALCON) {
        ropeSliceK.sliceDim = h->headDim;
        ropeSliceK.cacheSize = size2D(F_32, h->seqLen, h->headDim);
    }

#if DLLAMA_DEBUG_ATTN
    printf("🔍 [Node %u DEBUG] RoPE Q Slice: qStart=%u qLen=%u kvStart=%u kvLen=%u headDim=%u\n",
        nodeIndex, ropeSliceQ.qDimStart, ropeSliceQ.qDim0, ropeSliceQ.kvDimStart, ropeSliceQ.kvDim0, ropeSliceQ.headDim);
    printf("🔍 [Node %u DEBUG] RoPE K Slice%s: qStart=%u qLen=%u kvStart=%u kvLen=%u headDim=%u\n",
        nodeIndex,
        enableKvRedundancy ? " (redundant)" : "",
        ropeSliceK.qDimStart, ropeSliceK.qDim0, ropeSliceK.kvDimStart, ropeSliceK.kvDim0, ropeSliceK.headDim);
#endif

    NnUint nQNormColumns = 1, nKNormColumns = 1, nInvBufferColumns = 1;
    if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
        nQNormColumns = qSlice.inLen / h->headDim;
        nKNormColumns = kSlice.inLen / h->headDim;
        // Online head migration may change per-node q/k slice lengths.
        // Allocate inv_rms buffer for the worst case (up to full head counts) so refresh-time
        // recomputation of nColumns stays within bounds.
        if (plan != nullptr && myStage != nullptr) {
            nInvBufferColumns = std::max(h->nHeads, h->nKvHeads);
        } else {
            nInvBufferColumns = std::max(nQNormColumns, nKNormColumns);
        }
    }

    // 3. 构建 Node Config
    NnNodeConfigBuilder nodeBuilder(nodeIndex);

    // Buffers
    const NnUint xBufferIndex = nodeBuilder.addBuffer("x", size2D(F_32, nBatches, h->dim));
    const NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, nBatches, h->dim));
    const NnUint yqBufferIndex = (h->syncType == F_32) ? yBufferIndex : nodeBuilder.addBuffer("q_y", size2D(h->syncType, nBatches, h->dim));
    
    const NnUint mhaOutBufferIndex = nodeBuilder.addBuffer(
        "mha_out",
        size2D(F_32, nBatches, fullAttBuffers ? h->qDim : qSlice.inLen));
    const NnUint mhaOutQBufferIndex = (h->syncType == F_32)
        ? mhaOutBufferIndex
        : nodeBuilder.addBuffer(
            "q_mha_out",
            size2D(h->syncType, nBatches, fullAttBuffers ? h->qDim : qSlice.inLen));

    const NnUint qBufferIndex = nodeBuilder.addBuffer(
        "q",
        size2D(F_32, nBatches, fullAttBuffers ? h->qDim : qSlice.inLen));
    const NnUint kTempBufferIndex = nodeBuilder.addBuffer(
        "k_temp",
        size2D(F_32, nBatches, (fullAttBuffers && enableKvRedundancy) ? kSlice.inLen : (fullAttBuffers ? h->kvDim : kSlice.inLen)));
    const NnUint vTempBufferIndex = nodeBuilder.addBuffer(
        "v_temp",
        size2D(F_32, nBatches, (fullAttBuffers && enableKvRedundancy) ? vSlice.inLen : (fullAttBuffers ? h->kvDim : vSlice.inLen)));
    const NnUint invRmsBufferIndex = nodeBuilder.addBuffer("inv_rms", size2D(F_32, nBatches, nInvBufferColumns));
    const NnUint ropeCacheQBufferIndex = nodeBuilder.addBuffer("rope_cache_q", ropeSliceQ.cacheSize);
    const NnUint ropeCacheKBufferIndex = nodeBuilder.addBuffer("rope_cache_k", ropeSliceK.cacheSize);
    // Attention scratch buffer: sized by local headLen * seqLen in the static plan.
    // With online repartition (plan barrier/apply), nHeads0 may change at runtime,
    // but buffers are not resized. Allocate a safe upper bound to avoid OOB writes
    // into att during/after migration.
    NnSize3D attSize = multiHeadAttSlice.attSize;
    if (enablePlanBarrier) {
        attSize = size2D(F_32, nBatches, h->nHeads * h->seqLen);
    }
    const NnUint attBufferIndex = nodeBuilder.addBuffer("att", attSize);
    const NnUint logitsSliceBufferIndex = nodeBuilder.addBuffer(
        "lg",
        size2D(F_32, nBatches, fullLogitsBuffers ? h->vocabSize : wclsSlice.inLen));

    const NnUint dBufferIndex = nodeBuilder.addBuffer(
        "d",
        size2D(F_32, nBatches, fullFfnBuffers ? ffDim : w1Slice.inLen));
    const NnUint dqBufferIndex = (h->syncType == F_32)
        ? dBufferIndex
        : nodeBuilder.addBuffer(
            "q_d",
            size2D(h->syncType, nBatches, fullFfnBuffers ? ffDim : w1Slice.inLen));
    const NnUint lBufferIndex = nodeBuilder.addBuffer(
        "l",
        size2D(F_32, nBatches, fullFfnBuffers ? ffDim : w3Slice.inLen));

    const NnUint moeGtBufferIndex = nodeBuilder.addBuffer("gt", size2D(F_32, nBatches, nExpertsOr1));
    const NnUint moeExpertIndexesBufferIndex = nodeBuilder.addBuffer("act_exp_ix", size2D(F_32, nBatches, nActiveExpertsOr1));
    const NnUint moeYBufferIndex = nodeBuilder.addBuffer("moe_y", size3D(F_32, nActiveExpertsOr1, nBatches, h->dim));
    const NnUint moeYqBufferIndex = (h->syncType == F_32) ? moeYBufferIndex : nodeBuilder.addBuffer("q_moe_y", size3D(h->syncType, nActiveExpertsOr1, nBatches, h->dim));
    const NnUint moeDBufferIndex = nodeBuilder.addBuffer("moe_d", size3D(F_32, nActiveExpertsOr1, nBatches, w1Slice.inLen));
    const NnUint moeDQBufferIndex = (h->syncType == F_32) ? moeDBufferIndex : nodeBuilder.addBuffer("q_moe_d", size3D(h->syncType, nActiveExpertsOr1, nBatches, w1Slice.inLen));
    const NnUint moeLBufferIndex = nodeBuilder.addBuffer("moe_l", size3D(F_32, nActiveExpertsOr1, nBatches, w3Slice.inLen));
    const NnUint moeSBufferIndex = nodeBuilder.addBuffer("moe_s", size3D(F_32, nActiveExpertsOr1, nBatches, 1));

    // Opt-in: force all OP_MATMUL ops to take the view-aware code path.
    // This is useful for distributed A/B checks while keeping buffer layouts unchanged (offset=0).
    const bool forceMatmulViews = (std::getenv("DLLAMA_FORCE_MATMUL_VIEWS") != nullptr);

    auto makeRowMatmulCfg = [&](NnUint inLen, NnUint outLen, NnUint outStart) -> NnMatmulOpConfig {
        NnMatmulOpConfig cfg{};
        cfg.activeExpertIndexesBufferIndex = moeExpertIndexesBufferIndex;
        if (stageFullWeights) {
            cfg.view = 1u;
            cfg.outStart = outStart;
        }
        // In the current uneven path, buffers are still allocated as packed local slices.
        // We keep filling C view to maintain existing plumbing behavior.
        if (forceMatmulViews) {
            cfg.aView = NnTensorView{0u, 0u, inLen, 0u, 0u};
            cfg.cView = NnTensorView{0u, 0u, outLen, 0u, 0u};
        } else {
            cfg.cView = NnTensorView{0u, 0u, outLen, 0u, 1u};
        }
        return cfg;
    };

    auto makeRowMatmulCfgTagged = [&](NnSliceTag outTag, NnUint outUnit, NnUint inLen, NnUint outLen, NnUint outStart) -> NnMatmulOpConfig {
        NnMatmulOpConfig cfg = makeRowMatmulCfg(inLen, outLen, outStart);
        cfg.outSliceTag = outTag;
        cfg.outStartUnit = outUnit;
        return cfg;
    };

    auto makeMoeRowMatmulCfg = [&](NnUint inLen, NnUint outLen, NnUint outStart) -> NnMatmulOpConfig {
        NnMatmulOpConfig cfg = makeRowMatmulCfg(inLen, outLen, outStart);
        cfg.nExperts = h->nExperts;
        cfg.nActiveExperts = h->nActiveExperts;
        return cfg;
    };

    auto makeColMatmulCfg = [&](NnUint inLen, NnUint outLen, NnUint inStart) -> NnMatmulOpConfig {
        NnMatmulOpConfig cfg{};
        cfg.activeExpertIndexesBufferIndex = moeExpertIndexesBufferIndex;
        if (stageFullWeights) {
            cfg.view = 2u;
            cfg.inStart = inStart;
            cfg.outStart = 0u;
        }
        if (forceMatmulViews) {
            cfg.aView = NnTensorView{0u, 0u, inLen, 0u, 0u};
            cfg.cView = NnTensorView{0u, 0u, outLen, 0u, 0u};
        }
        return cfg;
    };

    auto makeColMatmulCfgTagged = [&](NnSliceTag inTag, NnUint inUnit, NnUint inLen, NnUint outLen, NnUint inStart) -> NnMatmulOpConfig {
        NnMatmulOpConfig cfg = makeColMatmulCfg(inLen, outLen, inStart);
        cfg.inSliceTag = inTag;
        cfg.inStartUnit = inUnit;
        return cfg;
    };

    auto makeMoeColMatmulCfg = [&](NnUint inLen, NnUint outLen, NnUint inStart) -> NnMatmulOpConfig {
        NnMatmulOpConfig cfg = makeColMatmulCfg(inLen, outLen, inStart);
        cfg.nExperts = h->nExperts;
        cfg.nActiveExperts = h->nActiveExperts;
        return cfg;
    };

    // 4. Start Segment (Embedding)
    NnSegmentConfigBuilder start;
    if (isFirstStage) {
        // [修改] First Stage 所有节点都负责 Embedding
        // 1. 先同步 Token (广播: Root -> Stage 0 Workers)
        // 注意：这里假设 SYNC_WITH_ROOT 能正确处理 Node 0 到 Stage 0 其他节点的广播
        start.addSync(n->tokenPipeIndex, SYNC_WITH_ROOT);

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
        const NnUint kBufferIndex = nodeBuilder.addBuffer(
            "k",
            fullAttBuffers ? size2D(F_32, h->seqLen, h->kvDim) : kvCacheSlice.keySize);
        const NnUint vBufferIndex = nodeBuilder.addBuffer(
            "v",
            fullAttBuffers ? size2D(F_32, h->seqLen, h->kvDim) : kvCacheSlice.valueSize);

        NnSegmentConfigBuilder att;
        NnSegmentConfigBuilder ff;

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

        // IMPORTANT: when using PNTR_BATCHED_SLICE with sliceTag=AUTO, resolvePointer() may
        // accidentally match kvHeadSplit (multiplier = gqaGroup * headDim) instead of headSplit
        // for Q buffers (qDim is divisible by both totals). After online migration, headSplit totals
        // can be uneven/stage-local and AUTO matching becomes fragile, leading to Q-heads reading 0.
        // Force NN_SLICE_HEAD so matmul_q/rope_q and MHA agree on the same qStart slice.
        const NnPointerConfig qSlicePtr = fullAttBuffers
            ? pointerBatchedSliceConfigTagged(SRC_BUFFER, qBufferIndex, NN_SLICE_HEAD)
            : pointerBatchConfig(SRC_BUFFER, qBufferIndex);
        const NnPointerConfig kTempSlicePtr = (fullAttBuffers && !enableKvRedundancy)
            ? pointerBatchedSliceConfig(SRC_BUFFER, kTempBufferIndex)
            : pointerBatchConfig(SRC_BUFFER, kTempBufferIndex);
        const NnPointerConfig vTempSlicePtr = (fullAttBuffers && !enableKvRedundancy)
            ? pointerBatchedSliceConfig(SRC_BUFFER, vTempBufferIndex)
            : pointerBatchConfig(SRC_BUFFER, vTempBufferIndex);
        const NnPointerConfig mhaOutSlicePtr = fullAttBuffers
            ? pointerBatchedSliceConfig(SRC_BUFFER, mhaOutBufferIndex)
            : pointerBatchConfig(SRC_BUFFER, mhaOutBufferIndex);
        const NnPointerConfig mhaOutQSlicePtr = fullAttBuffers
            ? pointerBatchedSliceConfigTagged(SRC_BUFFER, mhaOutQBufferIndex, NN_SLICE_HEAD)
            : pointerBatchConfig(SRC_BUFFER, mhaOutQBufferIndex);

        att.addOp(OP_MATMUL, "block_matmul_q", layerIndex, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), qSlicePtr, stageFullWeights ? qSlice.size : qSlice.sliceSize, makeRowMatmulCfgTagged(NN_SLICE_HEAD, h->headDim, qSlice.n, qSlice.inLen, qSlice.inStart));
        NnMatmulOpConfig kCfg = enableKvRedundancy
            ? makeRowMatmulCfg(kSlice.n, kSlice.inLen, kSlice.inStart)
            : makeRowMatmulCfgTagged(NN_SLICE_KV_HEAD, h->headDim, kSlice.n, kSlice.inLen, kSlice.inStart);
        NnMatmulOpConfig vCfg = enableKvRedundancy
            ? makeRowMatmulCfg(vSlice.n, vSlice.inLen, vSlice.inStart)
            : makeRowMatmulCfgTagged(NN_SLICE_KV_HEAD, h->headDim, vSlice.n, vSlice.inLen, vSlice.inStart);
        if (enableKvRedundancy) {
            // Keep headDim unit for debug printing (do NOT set outSliceTag, to avoid refreshPointers remapping outStart).
            kCfg.outStartUnit = h->headDim;
            vCfg.outStartUnit = h->headDim;
        }
        att.addOp(OP_MATMUL, "block_matmul_k", layerIndex, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), kTempSlicePtr, stageFullWeights ? kSlice.size : kSlice.sliceSize, kCfg);
        att.addOp(OP_MATMUL, "block_matmul_v", layerIndex, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), vTempSlicePtr, stageFullWeights ? vSlice.size : vSlice.sliceSize, vCfg);

        if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
            att.addOp(OP_INV_RMS, "block_norm_pre_q", layerIndex, qSlicePtr, pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex), size0(), NnInvRmsOpConfig{h->normEpsilon, nQNormColumns});
            att.addOp(OP_RMS_NORM, "block_norm_q", layerIndex, qSlicePtr, qSlicePtr, size2D(F_32, 1, h->headDim), NnRmsNormOpConfig{invRmsBufferIndex, nQNormColumns});
            att.addOp(OP_INV_RMS, "block_norm_pre_k", layerIndex, kTempSlicePtr, pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex), size0(), NnInvRmsOpConfig{h->normEpsilon, nKNormColumns});
            att.addOp(OP_RMS_NORM, "block_norm_k", layerIndex, kTempSlicePtr, kTempSlicePtr, size2D(F_32, 1, h->headDim), NnRmsNormOpConfig{invRmsBufferIndex, nKNormColumns});
        }

        att.addOp(OP_ROPE, "block_rope_q", layerIndex, qSlicePtr, qSlicePtr, size0(), NnRopeOpConfig{h->ropeType, 1, n->positionPipeIndex, ropeCacheQBufferIndex, h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen, ropeSliceQ});
        att.addOp(OP_ROPE, "block_rope_k", layerIndex, kTempSlicePtr, kTempSlicePtr, size0(), NnRopeOpConfig{h->ropeType, 0, n->positionPipeIndex, ropeCacheKBufferIndex, h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen, ropeSliceK});

        const NnShiftOpCodeConfig shiftKCfg = fullAttBuffers
            ? NnShiftOpCodeConfig{n->positionPipeIndex, enableKvRedundancy ? kSlice.inStart : kvCacheSlice.kvStart, h->kvDim, enableKvRedundancy ? 0u : h->headDim}
            : NnShiftOpCodeConfig{n->positionPipeIndex};
        const NnShiftOpCodeConfig shiftVCfg = fullAttBuffers
            ? NnShiftOpCodeConfig{n->positionPipeIndex, enableKvRedundancy ? vSlice.inStart : kvCacheSlice.kvStart, h->kvDim, enableKvRedundancy ? 0u : h->headDim}
            : NnShiftOpCodeConfig{n->positionPipeIndex};
        att.addOp(OP_SHIFT, "block_shift_k", layerIndex, kTempSlicePtr, pointerRawConfig(SRC_BUFFER, kBufferIndex), size0(), shiftKCfg);
        att.addOp(OP_SHIFT, "block_shift_v", layerIndex, vTempSlicePtr, pointerRawConfig(SRC_BUFFER, vBufferIndex), size0(), shiftVCfg);

        // Basic invariants for head-aligned slices.
        assert(h->headDim != 0u);
        assert((qSlice.inStart % h->headDim) == 0u);
        assert((qSlice.inLen % h->headDim) == 0u);
        assert((kvCacheSlice.kvStart % h->headDim) == 0u);
        assert((kvCacheSlice.kvLen % h->headDim) == 0u);
        assert(kvCacheSlice.kvStart + kvCacheSlice.kvLen <= h->kvDim);
        if (enableKvRedundancy) {
            assert((kSlice.inStart % h->headDim) == 0u);
            assert((kSlice.inLen % h->headDim) == 0u);
            assert(kSlice.inStart + kSlice.inLen <= h->kvDim);
            assert((vSlice.inStart % h->headDim) == 0u);
            assert((vSlice.inLen % h->headDim) == 0u);
            assert(vSlice.inStart + vSlice.inLen <= h->kvDim);
        }

        const NnUint qStart = fullAttBuffers ? qSlice.inStart : 0u;
        const NnUint qStride = fullAttBuffers ? h->qDim : qSlice.inLen;

        // MHA reads owned KV heads (kvHeadSplit) from the KV cache.
        // KV redundancy only widens the set of heads that get written into this same KV cache,
        // so that after migration a node may already have history for newly-owned kvHeads.
        const NnUint kvStart = fullAttBuffers ? kvCacheSlice.kvStart : 0u;
        const NnUint kvDim0 = kvCacheSlice.kvLen;
        const NnUint kvStride = fullAttBuffers ? h->kvDim : kvCacheSlice.kvLen;
        att.addOp(OP_MULTIHEAD_ATT, "block_multihead_att", layerIndex, mhaOutSlicePtr, mhaOutSlicePtr, size0(),
            NnMultiHeadAttOpConfig{
                multiHeadAttSlice.nHeads,
                multiHeadAttSlice.nHeads0,
                h->nKvHeads,
                h->headDim,
                h->seqLen,
                qSlice.inLen,
                kvDim0,
                qStart,
                qStride,
                kvStart,
                kvStride,
                n->positionPipeIndex,
                qBufferIndex,
                kBufferIndex,
                vBufferIndex,
                attBufferIndex});

#if DLLAMA_DEBUG_ATTN
        printf(" [Node %u DEBUG] MHA: nHeads=%u nHeads0=%u qStart=%u qSliceD0=%u kvStart=%u kvDim0=%u kvStride=%u%s\n",
            nodeIndex,
            multiHeadAttSlice.nHeads,
            multiHeadAttSlice.nHeads0,
            qStart,
            qSlice.inLen,
            kvStart,
            kvDim0,
            kvStride,
            enableKvRedundancy ? " (kv-redundant)" : "");
#endif

        if (mhaOutBufferIndex != mhaOutQBufferIndex) {
             att.addOp(OP_CAST, "block_cast_y2", layerIndex, mhaOutSlicePtr, mhaOutQSlicePtr, size0(), NnCastOpCodeConfig{});
        }
        att.addOp(OP_MATMUL, "block_matmul_wo", layerIndex, mhaOutQSlicePtr, pointerBatchConfig(SRC_BUFFER, yBufferIndex), stageFullWeights ? woSlice.size : woSlice.sliceSize, makeColMatmulCfgTagged(NN_SLICE_HEAD, h->headDim, woSlice.n0, woSlice.d, woSlice.outStart));
        att.addOp(OP_CAST, "block_cast_d", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchedSliceConfigTagged(SRC_PIPE, n->zqPipeIndex, NN_SLICE_STACKED_BY_NODE), size0(), NnCastOpCodeConfig{});
        att.addSync(n->zqPipeIndex, SYNC_NODE_SLICES);

        // Side-effect only: aggregate full KV-cache (all nodes get a full copy).
        // This does NOT change any downstream compute (no ops read KC/VC).
        // NOTE: KV cache buffers are laid out as (seqLen, kvDimLocal) and are not batch-major.
        // Using PNTR_BATCH on them will trip ASSERT_EQ(sourceSize->y, nBatches).
        // Instead, aggregate the per-token KV vectors that were just written (kTemp/vTemp),
        // which are batch-major and correspond to the current POS window.
        if (n->kvAggKPipeIndex != (NnUint)-1 && n->kvAggVPipeIndex != (NnUint)-1) {
            att.addOp(
                OP_CAST, "kvagg_cast_k", layerIndex,
                kTempSlicePtr,
                pointerBatchedSliceConfigTagged(SRC_PIPE, n->kvAggKPipeIndex, NN_SLICE_KV_HEAD),
                size0(),
                NnCastOpCodeConfig{});
            att.addSync(n->kvAggKPipeIndex, SYNC_NODE_SLICES);

            att.addOp(
                OP_CAST, "kvagg_cast_v", layerIndex,
                vTempSlicePtr,
                pointerBatchedSliceConfigTagged(SRC_PIPE, n->kvAggVPipeIndex, NN_SLICE_KV_HEAD),
                size0(),
                NnCastOpCodeConfig{});
            att.addSync(n->kvAggVPipeIndex, SYNC_NODE_SLICES);
        }

        // --- FFN Ops ---
        ff.addOp(OP_MERGE_ADD, "block_merge_add2", layerIndex, pointerBatchConfig(SRC_PIPE, n->zqPipeIndex), pointerBatchConfig(SRC_BUFFER, xBufferIndex), size0(), NnMergeAddOpCodeConfig{});
        ff.addOp(OP_INV_RMS, "block_norm_pre_1", layerIndex, pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex), size0(), NnInvRmsOpConfig{h->normEpsilon, 1});
        ff.addOp(OP_RMS_NORM, "block_norm_1", layerIndex, pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex), n->rmsNormSize, NnRmsNormOpConfig{invRmsBufferIndex, 1});

        if (h->archType == QWEN3_MOE) {
            ff.addOp(OP_REPEAT_Z, "block_moe_y_repeat", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex), size0(), NnRepeatZOpCodeConfig{});
            ff.addOp(OP_MATMUL, "block_moe_gate", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex), n->moeGateSize, makeRowMatmulCfg(n->moeGateSize.x, nExpertsOr1, 0u));
            ff.addOp(OP_SOFTMAX, "block_moe_softmax", layerIndex, pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex), pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex), size0(), NnSoftmaxOpCodeConfig{});
            ff.addOp(OP_MOE_GATE, "block_moe_gate2", layerIndex, pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex), pointerBatchConfig(SRC_BUFFER, moeSBufferIndex), size0(), NnMoeGateOpCodeConfig{h->nActiveExperts, 1u, moeExpertIndexesBufferIndex});
            
            NnSize3D w1ExpertSliceSize = stageFullWeights
                ? size3D(h->weightType, h->nExperts, w1Slice.n, ffDim)
                : size3D(h->weightType, h->nExperts, w1Slice.n, w1Slice.inLen);
            NnSize3D w3ExpertSliceSize = stageFullWeights
                ? size3D(h->weightType, h->nExperts, w3Slice.n, ffDim)
                : size3D(h->weightType, h->nExperts, w3Slice.n, w3Slice.inLen);
            NnSize3D w2ExpertSliceSize = stageFullWeights
                ? size3D(h->weightType, h->nExperts, w2Slice.n, w2Slice.d)
                : size3D(h->weightType, h->nExperts, w2Slice.n0, w2Slice.d);

            {
                NnMatmulOpConfig w1Cfg = makeMoeRowMatmulCfg(w1Slice.n, w1Slice.inLen, w1Slice.inStart);
                w1Cfg.outSliceTag = NN_SLICE_FFN;
                w1Cfg.outStartUnit = 1u;
                ff.addOp(OP_MATMUL, "block_matmul_w1", layerIndex, pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex), pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), w1ExpertSliceSize, w1Cfg);
            }
            {
                NnMatmulOpConfig w3Cfg = makeMoeRowMatmulCfg(w3Slice.n, w3Slice.inLen, w3Slice.inStart);
                w3Cfg.outSliceTag = NN_SLICE_FFN;
                w3Cfg.outStartUnit = 1u;
                ff.addOp(OP_MATMUL, "block_matmul_w3", layerIndex, pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex), pointerBatchConfig(SRC_BUFFER, moeLBufferIndex), w3ExpertSliceSize, w3Cfg);
            }
            {
                const bool testSplit = (std::getenv("DLLAMA_TEST_SILU_VIEW_SPLIT") != nullptr);
                const NnUint actDim = w1Slice.inLen;
                if (testSplit && layerIndex == 0u && actDim >= 2u) {
                    const NnUint half = actDim / 2u;
                    ff.addOp(OP_SILU, "block_act_v0", layerIndex, pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), size0(), NnSiluOpCodeConfig{NnTensorView{0u, 0u, half, 0u, 1u}});
                    ff.addOp(OP_SILU, "block_act_v1", layerIndex, pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), size0(), NnSiluOpCodeConfig{NnTensorView{half, 0u, actDim - half, 0u, 1u}});
                } else {
                    ff.addOp(OP_SILU, "block_act", layerIndex, pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), size0(), NnSiluOpCodeConfig{NnTensorView{0u, 0u, 0u, 0u, 1u}});
                }
            }
            ff.addOp(OP_MUL, "block_mul", layerIndex, pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), size0(), NnMulOpCodeConfig{moeLBufferIndex});
            if (moeDBufferIndex != moeDQBufferIndex) {
                ff.addOp(OP_CAST, "block_cast_d2", layerIndex, pointerBatchConfig(SRC_BUFFER, moeDBufferIndex), pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex), size0(), NnCastOpCodeConfig{});
            }
            {
                NnMatmulOpConfig w2Cfg = makeMoeColMatmulCfg(w2Slice.n0, w2Slice.d, w2Slice.outStart);
                w2Cfg.inSliceTag = NN_SLICE_FFN;
                w2Cfg.inStartUnit = 1u;
                ff.addOp(OP_MATMUL, "block_matmul_w2", layerIndex, pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex), pointerBatchConfig(SRC_BUFFER, moeYBufferIndex), w2ExpertSliceSize, w2Cfg);
            }
            ff.addOp(OP_SCALE, "block_moe_scale", layerIndex, pointerBatchConfig(SRC_BUFFER, moeYBufferIndex), pointerBatchConfig(SRC_BUFFER, moeYBufferIndex), size0(), NnScaleOpCodeConfig{moeSBufferIndex});
            ff.addOp(OP_MERGE_SUM, "block_moe_merge_sum", layerIndex, pointerBatchConfig(SRC_BUFFER, moeYBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex), size0(), NnMergeSumOpCodeConfig{});
        } else {
            if (yBufferIndex != yqBufferIndex) {
                ff.addOp(OP_CAST, "block_cast_y3", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchConfig(SRC_BUFFER, yqBufferIndex), size0(), NnCastOpCodeConfig{});
            }
            const NnPointerConfig dSlicePtr = fullFfnBuffers
                ? pointerBatchedSliceConfigTagged(SRC_BUFFER, dBufferIndex, NN_SLICE_FFN)
                : pointerBatchConfig(SRC_BUFFER, dBufferIndex);
            const NnPointerConfig dqSlicePtr = fullFfnBuffers
                ? pointerBatchedSliceConfigTagged(SRC_BUFFER, dqBufferIndex, NN_SLICE_FFN)
                : pointerBatchConfig(SRC_BUFFER, dqBufferIndex);
            const NnPointerConfig lSlicePtr = fullFfnBuffers
                ? pointerBatchedSliceConfigTagged(SRC_BUFFER, lBufferIndex, NN_SLICE_FFN)
                : pointerBatchConfig(SRC_BUFFER, lBufferIndex);

            ff.addOp(OP_MATMUL, "block_matmul_w1", layerIndex, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), dSlicePtr, stageFullWeights ? w1Slice.size : w1Slice.sliceSize, makeRowMatmulCfgTagged(NN_SLICE_FFN, 1u, w1Slice.n, w1Slice.inLen, w1Slice.inStart));
            ff.addOp(OP_MATMUL, "block_matmul_w3", layerIndex, pointerBatchConfig(SRC_BUFFER, yqBufferIndex), lSlicePtr, stageFullWeights ? w3Slice.size : w3Slice.sliceSize, makeRowMatmulCfgTagged(NN_SLICE_FFN, 1u, w3Slice.n, w3Slice.inLen, w3Slice.inStart));
            {
                const bool testSplit = (std::getenv("DLLAMA_TEST_SILU_VIEW_SPLIT") != nullptr);
                const NnUint actDim = w1Slice.inLen;
                if (testSplit && layerIndex == 0u && actDim >= 2u) {
                    const NnUint half = actDim / 2u;
                    ff.addOp(OP_SILU, "block_act_v0", layerIndex, dSlicePtr, dSlicePtr, size0(), NnSiluOpCodeConfig{NnTensorView{0u, 0u, half, 0u, 1u}});
                    ff.addOp(OP_SILU, "block_act_v1", layerIndex, dSlicePtr, dSlicePtr, size0(), NnSiluOpCodeConfig{NnTensorView{half, 0u, actDim - half, 0u, 1u}});
                } else {
                    ff.addOp(OP_SILU, "block_act", layerIndex, dSlicePtr, dSlicePtr, size0(), NnSiluOpCodeConfig{NnTensorView{0u, 0u, 0u, 0u, 1u}});
                }
            }
            // OP_MUL reads multiplier from a buffer index (not via pointer config),
            // so when using full buffers we must use view(offset/len) to align d and l slices.
            // NOTE:
            // - We intentionally keep input/output pointer configs as pointerBatchConfig() so they point
            //   to the *base* of the full D buffer (offset=0).
            // - The slice selection is applied via mulCfg.view.offset/sizeX, which is also used to index
            //   into the multiplier buffer L (which is addressed by buffer index, not via pointer slicing).
            // If we instead used dSlicePtr here, we'd need a separate multiplier offset field to avoid
            // double-applying the offset.
            NnMulOpCodeConfig mulCfg = NnMulOpCodeConfig{lBufferIndex};
            if (fullFfnBuffers) {
                const NnUint ffnStart0 = (plan && plan->ffnSplit.starts) ? plan->ffnSplit.starts[nodeIndex] : 0u;
                const NnUint ffnLen0 = (plan && plan->ffnSplit.lengths) ? plan->ffnSplit.lengths[nodeIndex] : w1Slice.inLen;
                mulCfg.view = NnTensorView{ffnStart0, 0u, ffnLen0, 0u, 1u};
            }
            ff.addOp(OP_MUL, "block_mul", layerIndex,
                pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                size0(), mulCfg);
            if (dBufferIndex != dqBufferIndex) {
                ff.addOp(OP_CAST, "block_cast_d2", layerIndex, dSlicePtr, dqSlicePtr, size0(), NnCastOpCodeConfig{});
            }
            ff.addOp(OP_MATMUL, "block_matmul_w2", layerIndex, dqSlicePtr, pointerBatchConfig(SRC_BUFFER, yBufferIndex), stageFullWeights ? w2Slice.size : w2Slice.sliceSize, makeColMatmulCfgTagged(NN_SLICE_FFN, 1u, w2Slice.n0, w2Slice.d, w2Slice.outStart));
        }
        
        ff.addOp(OP_CAST, "block_cast_d3", layerIndex, pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchedSliceConfigTagged(SRC_PIPE, n->zqPipeIndex, NN_SLICE_STACKED_BY_NODE), size0(), NnCastOpCodeConfig{});
        ff.addSync(n->zqPipeIndex, SYNC_NODE_SLICES);

        nodeBuilder.addSegment(att.build());
        nodeBuilder.addSegment(ff.build());

        // ----------------------------------------------------------------------
        // Optional per-layer barrier/apply (CPU-only test hook for online repartition)
        // ----------------------------------------------------------------------
        if (enablePlanBarrier) {
            NnSegmentConfigBuilder planBarrier;
            planBarrier.addOp(
                OP_PLAN_BARRIER,
                "plan_barrier",
                layerIndex,
                pointerBatchConfig(SRC_PIPE, n->positionPipeIndex),
                pointerBatchConfig(SRC_PIPE, n->planPipeIndex),
                size0(),
                NnPlanBarrierOpCodeConfig{
                    n->positionPipeIndex,
                    n->planPipeIndex,
                    0xFFFFFFFFu, // triggerPos (deprecated: PlanCommand supplies trigger)
                    0xFFFFFFFFu, // triggerLayer (deprecated)
                    0u, // fromNodeIndex (deprecated)
                    0u, // toNodeIndex (deprecated)
                    0u, // nHeadsToMove (deprecated)
                    0u, // cmdKind (deprecated)
                    0u, // nFfnToMove (deprecated)
                    (myStage != nullptr ? myStage->stageIndex : 0u) // onlyStageIndex: my stage
                });
            // Broadcast the control packet within the current stage.
            planBarrier.addSync(n->planPipeIndex, SYNC_WITH_ROOT);
            nodeBuilder.addSegment(planBarrier.build());

            NnSegmentConfigBuilder planApply;
            planApply.addOp(
                OP_PLAN_APPLY,
                "plan_apply",
                layerIndex,
                pointerBatchConfig(SRC_PIPE, n->planPipeIndex),
                pointerBatchConfig(SRC_PIPE, n->planPipeIndex),
                size0(),
                NnPlanApplyOpCodeConfig{n->planPipeIndex, (myStage != nullptr ? myStage->stageIndex : 0u)});
            nodeBuilder.addSegment(planApply.build());
        }
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
        
        const NnPointerConfig logitsBufSlicePtr = fullLogitsBuffers
            ? pointerBatchedSliceConfigTagged(SRC_BUFFER, logitsSliceBufferIndex, NN_SLICE_VOCAB)
            : pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex);

        end.addOp(OP_MATMUL, "final_matmul_logits", 0,
            pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
            logitsBufSlicePtr,
            stageFullWeights ? wclsSlice.size : wclsSlice.sliceSize,
            makeRowMatmulCfgTagged(NN_SLICE_VOCAB, 1u, wclsSlice.n, wclsSlice.inLen, wclsSlice.inStart));
        
        end.addOp(OP_CAST, "final_cast_logits", 0, 
            logitsBufSlicePtr, 
            pointerBatchedSliceConfigTagged(SRC_PIPE, n->logitsPipeIndex, NN_SLICE_VOCAB), // <--- 改回这个！
            size0(), NnCastOpCodeConfig{});
        
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
    // Control pipe for hard migration barrier/apply (CPU-only test hook)
    n.planPipeIndex = netBuilder.addPipe("PLN", size2D(F_32, nBatches, 8));

    n.kvAggKPipeIndex = (NnUint)-1;
    n.kvAggVPipeIndex = (NnUint)-1;
    if (std::getenv("DLLAMA_KV_AGGREGATE") != nullptr) {
        n.kvAggKPipeIndex = netBuilder.addPipe("KC", size2D(F_32, nBatches, h->kvDim));
        n.kvAggVPipeIndex = netBuilder.addPipe("VC", size2D(F_32, nBatches, h->kvDim));
    }

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

        if (timer.elapsedMiliseconds() > 10000)
            printf("💿 Loaded %u/%u\n", layerIndex + 1, net->header->nLayers);
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

    // Opt-in: stage 内全量加载本 stage 的模型权重（每个节点都持有本 stage 的全量 weight tensor）。
    // 注意：这只改变“加载多少权重到本地”；要真正利用全量权重进行区间计算，还需要算子/建图侧用 view/outStart 等参数选择计算区间。
    const bool stageFullWeights = (std::getenv("DLLAMA_STAGE_FULL_WEIGHTS") != nullptr) && (myStage != nullptr);
    if (stageFullWeights) {
        printf("   [PP] Node %u: Stage full-weight loading ENABLED (DLLAMA_STAGE_FULL_WEIGHTS)\n", nodeIndex);
    }

    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    
    printf("💿 Loading weights for Node %u (Layers [%u, %u))...\n", nodeIndex, startLayer, endLayer);

    Timer timer;
    NnByte *data = (NnByte *)file.data;
    NnByte *b = &data[net->header->headerSize];
    LlmHeader *h = net->header;

    // --- 1. Embedding ---
    if (isFirstStage) {
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

            // Attention
            if (stageFullWeights) {
                b += loader->loadRowMatmulFullUneven("block_matmul_q", layerIndex, 0,
                    [&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->headSplit, h->qDim, idx); }, b);
                b += loader->loadRowMatmulFullUneven("block_matmul_k", layerIndex, 0,
                    [&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, idx); }, b);
                b += loader->loadRowMatmulFullUneven("block_matmul_v", layerIndex, 0,
                    [&](NnUint idx) { return sliceRowMatmulAttUneven(h->weightType, h->dim, h->headDim, &plan->kvHeadSplit, h->kvDim, idx); }, b);
                b += loader->loadColMatmulFullUneven("block_matmul_wo", layerIndex, 0,
                    [&](NnUint idx) { return sliceColMatmulAttUneven(h->weightType, h->qDim, h->dim, h->headDim, plan, idx); }, b);
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
                    if (stageFullWeights) {
                        b += loader->loadRowMatmulFullUneven("block_matmul_w1", layerIndex, expertIndex,
                            [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, b);
                        b += loader->loadColMatmulFullUneven("block_matmul_w2", layerIndex, expertIndex,
                            [&](NnUint idx) { return sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, idx); }, b);
                        b += loader->loadRowMatmulFullUneven("block_matmul_w3", layerIndex, expertIndex,
                            [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, b);
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
                if (stageFullWeights) {
                    b += loader->loadRowMatmulFullUneven("block_matmul_w1", layerIndex, 0,
                        [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, b);
                    b += loader->loadColMatmulFullUneven("block_matmul_w2", layerIndex, 0,
                        [&](NnUint idx) { return sliceColMatmulFfnUneven(h->weightType, ffDim, h->dim, plan, idx); }, b);
                    b += loader->loadRowMatmulFullUneven("block_matmul_w3", layerIndex, 0,
                        [&](NnUint idx) { return sliceRowMatmulFfnUneven(h->weightType, h->dim, ffDim, plan, idx); }, b);
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
        b += loader->loadAll("final_norm", 0u, net->rmsNormSize.nBytes, b);
        if (stageFullWeights) {
            b += loader->loadRowMatmulFullUneven("final_matmul_logits", 0u, 0u,
                [&](NnUint idx) {
                    return sliceRowMatmulLogitsUneven(h->weightType, h->dim, h->vocabSize, plan, idx);
                }, b);
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