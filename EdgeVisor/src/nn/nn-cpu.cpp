#include "nn-cpu.hpp"
#include "nn-cpu-ops.hpp"
#include "nn-core.hpp"
#include "ablation.hpp"
#include "plan-command.hpp"
#include "llm.hpp"
#include <cassert>
#include <iostream> 
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <atomic>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define DEBUG_CPU_OP_QUANTS false

#define BUFFER_ALIGNMENT 64

static void logCpuPlanApplyAblationEvent(
    NnUint stageIndex,
    NnUint fromNode,
    NnUint toNode,
    NnUint layerIndex,
    NnUint pos,
    NnUint cmd,
    bool success,
    const char *reason) {
    EdgeVisorAblationEvent ev;
    ev.eventId = "plan_command_apply";
    ev.triggerPos = pos;
    ev.triggerLayer = layerIndex;
    ev.affectedStage = stageIndex;
    ev.fromNode = fromNode;
    ev.toNode = toNode;
    ev.selectedPolicy = std::string("cpu_apply_cmd_") + std::to_string((unsigned)cmd);
    ev.bindingUpdateCount = 1u;
    ev.applySuccess = success;
    auto appendFallbackReason = [&](const char *fallbackReason) {
        if (fallbackReason == nullptr || fallbackReason[0] == '\0') return;
        if (!ev.fallbackReason.empty()) ev.fallbackReason += ";";
        ev.fallbackReason += fallbackReason;
    };
    appendFallbackReason(reason);
    const EdgeVisorAblationConfig &cfg = getEdgeVisorAblationConfig();
    if (cfg.pointerSwizzlingMode == PointerSwizzlingMode::OPERATOR_REBUILD) {
        appendFallbackReason("operator_rebuild_substitutes_lightweight_pointer_swizzling");
        ev.fallbackCount = 1u;
    } else if (cfg.pointerSwizzlingMode == PointerSwizzlingMode::WEIGHT_REMATERIALIZE) {
        appendFallbackReason("weight_rematerialize_substitutes_lightweight_pointer_swizzling");
        ev.materializedBytes = ev.bindingUpdateCount;
        ev.fallbackCount = 1u;
    }
    edgevisorAblationLogEvent(ev);
}

#if DLLAMA_DEBUG_ATTN

static bool debugWeightRangesEnabled(const char *opName) {
    if (std::getenv("DLLAMA_DEBUG_WEIGHT_RANGES") == nullptr)
        return false;
    const char *filter = std::getenv("DLLAMA_DEBUG_WEIGHT_RANGES_FILTER");
    if (filter == nullptr || filter[0] == '\0')
        return true;
    if (opName == nullptr)
        return false;
    return (std::strstr(opName, filter) != nullptr);
}

static bool debugSliceParamsEnabled(const char *opName) {
    if (std::getenv("DLLAMA_DEBUG_SLICE_PARAMS") == nullptr)
        return false;
    const char *filter = std::getenv("DLLAMA_DEBUG_SLICE_PARAMS_FILTER");
    if (filter == nullptr || filter[0] == '\0')
        return true;
    if (opName == nullptr)
        return false;
    return (std::strstr(opName, filter) != nullptr);
}

static bool debugSliceParamsForwardEnabled(const char *opName) {
    if (std::getenv("DLLAMA_DEBUG_SLICE_PARAMS_FWD") == nullptr)
        return false;
    const char *filter = std::getenv("DLLAMA_DEBUG_SLICE_PARAMS_FILTER");
    if (filter == nullptr || filter[0] == '\0')
        return true;
    if (opName == nullptr)
        return false;
    return (std::strstr(opName, filter) != nullptr);
}

static bool debugSliceParamsForwardRepeat() {
    const char *v = std::getenv("DLLAMA_DEBUG_SLICE_PARAMS_FWD_REPEAT");
    if (v == nullptr)
        return false;
    // Treat "0" / "false" as disabled.
    if (v[0] == '0')
        return false;
    if ((v[0] == 'f' || v[0] == 'F') && (v[1] == 'a' || v[1] == 'A'))
        return false;
    return true;
}

static inline int getenvIntOr(const char *name, int fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0')
        return fallback;
    try {
        return std::stoi(std::string(v));
    } catch (...) {
        return fallback;
    }
}

static inline bool debugResolvePointerEnabled() {
    const char *v = std::getenv("DLLAMA_DEBUG_RESOLVE_POINTER");
    if (v == nullptr) return false;
    if (v[0] == '0') return false;
    if ((v[0] == 'f' || v[0] == 'F') && (v[1] == 'a' || v[1] == 'A')) return false;
    return true;
}

static inline bool debugResolvePointerPassesFilter(NnPointerSource src, NnUint idx, NnPointerType type, NnSliceTag tag) {
    const char *filter = std::getenv("DLLAMA_DEBUG_RESOLVE_POINTER_FILTER");
    if (filter == nullptr || filter[0] == '\0') return true;

    // Accept either a numeric buffer/pipe index, or a tag substring like "HEAD" / "KV".
    char *end = nullptr;
    long want = std::strtol(filter, &end, 10);
    if (end != filter && *end == '\0') {
        return (want >= 0) ? ((NnUint)want == idx) : false;
    }

    const char *tagStr = sliceTagToString(tag);
    const char *srcStr = (src == SRC_BUFFER) ? "BUFFER" : (src == SRC_PIPE ? "PIPE" : "SRC?");
    const char *typeStr = (type == PNTR_BATCHED_SLICE) ? "BATCHED_SLICE" : (type == PNTR_BATCH ? "BATCH" : (type == PNTR_RAW ? "RAW" : "TYPE?"));

    if (tagStr && std::strstr(tagStr, filter) != nullptr) return true;
    if (std::strstr(srcStr, filter) != nullptr) return true;
    if (std::strstr(typeStr, filter) != nullptr) return true;
    return false;
}

static inline NnUint debugResolvePointerLimit() {
    const char *v = std::getenv("DLLAMA_DEBUG_RESOLVE_POINTER_LIMIT");
    if (v == nullptr) return 200u;
    long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 100000) return 100000u;
    return (NnUint)n;
}

static inline bool tpRangeTargetEnabled() {
    // Enable when either explicit debug env is present, or a PlanCommand(EXACT) target exists.
    if (std::getenv("DLLAMA_DEBUG_TP_RANGES") != nullptr)
        return true;
    const int p = getenvIntOr("DLLAMA_DEBUG_TP_RANGES_POS", -1);
    const int l = getenvIntOr("DLLAMA_DEBUG_TP_RANGES_LAYER", -1);
    if (p >= 0 && l >= 0)
        return true;
    const PlanCommandSnapshot snap = planCommandCache().load();
    const PlanCommand &pc = snap.cmd;
    if (pc.magic != DLLAMA_PLAN_CMD_MAGIC || pc.version != DLLAMA_PLAN_CMD_VERSION_V2)
        return false;
    return (pc.mode == PLAN_CMD_MODE_EXACT && pc.triggerPos != 0xFFFFFFFFu && pc.triggerLayer != 0xFFFFFFFFu);
}

static inline bool tpRangeTargetMatch(NnUint layerIndex, NnUint pos) {
    int targetPos = getenvIntOr("DLLAMA_DEBUG_TP_RANGES_POS", -1);
    int targetLayer = getenvIntOr("DLLAMA_DEBUG_TP_RANGES_LAYER", -1);
    int span = getenvIntOr("DLLAMA_DEBUG_TP_RANGES_LAYER_SPAN", -1);
    const bool explicitTarget = (targetPos >= 0 && targetLayer >= 0);
    if (targetPos < 0 || targetLayer < 0) {
        const PlanCommandSnapshot snap = planCommandCache().load();
        const PlanCommand &pc = snap.cmd;
        if (pc.magic == DLLAMA_PLAN_CMD_MAGIC && pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.mode == PLAN_CMD_MODE_EXACT) {
            targetPos = (pc.triggerPos == 0xFFFFFFFFu) ? -1 : (int)pc.triggerPos;
            targetLayer = (pc.triggerLayer == 0xFFFFFFFFu) ? -1 : (int)pc.triggerLayer;
            // Default behavior: when using PlanCommand(EXACT) target as the debug anchor,
            // print both (layer-1) and (layer) at the target pos so users can compare
            // compute ranges before/after the migration point.
            if (span < 0)
                span = 1;
        }
    }
    if (targetPos < 0 || targetLayer < 0)
        return false;
    if (span < 0)
        span = 0;
    if (pos != (NnUint)targetPos)
        return false;
    const int li = (int)layerIndex;
    const int tl = (int)targetLayer;
    const int d = (li >= tl) ? (li - tl) : (tl - li);
    // If the user explicitly specifies DLLAMA_DEBUG_TP_RANGES_LAYER, respect it strictly unless span was set.
    if (explicitTarget && getenvIntOr("DLLAMA_DEBUG_TP_RANGES_LAYER_SPAN", -1) < 0)
        return li == tl;
    return d <= span;
}

static inline NnUint loadPosFromPipe(const NnCpuOpContext *context, NnUint pipeIndex) {
    if (context == nullptr || context->pipes == nullptr)
        return 0xFFFFFFFFu;
    const float posF = *(const float *)(context->pipes[pipeIndex]);
    if (!(posF >= 0.0f))
        return 0xFFFFFFFFu;
    return (NnUint)posF;
}

static inline void maybeDumpHardcodedLayerKvCacheOnce(
    const NnCpuOpContext *context,
    const NnSegmentConfig *segmentConfig,
    NnUint opIndex,
    NnUint nodeIndex) {
    if (context == nullptr || segmentConfig == nullptr || context->name == nullptr) return;
    if (context->opCode != OP_MULTIHEAD_ATT) return;
    if (std::strstr(context->name, "block_multihead_att") == nullptr) return;

    const NnUint layerIndex = segmentConfig->ops[opIndex].index;
    static constexpr NnUint kTargetLayer = 15u;
    static constexpr NnUint kTargetPos = 32u;
    if (layerIndex != kTargetLayer) return;

    const auto *cfg = (const NnMultiHeadAttOpConfig *)context->opConfig;
    if (cfg == nullptr) return;
    const NnUint pos = loadPosFromPipe(context, cfg->positionPipeIndex);
    if (pos != kTargetPos) return;

    static std::atomic_uint8_t dumpedOnce{0u};
    const NnUint old = dumpedOnce.exchange(1u, std::memory_order_acq_rel);
    if (old != 0u) return;

    const NnUint kBufferIndex = cfg->keyCacheBufferIndex;
    const NnUint vBufferIndex = cfg->valueCacheBufferIndex;

    const NnSize3D kSize = context->bufferConfigs[kBufferIndex].size;
    const NnSize3D vSize = context->bufferConfigs[vBufferIndex].size;

    char kPath[256];
    char vPath[256];
    std::snprintf(kPath, sizeof(kPath), "/tmp/dllama_kvcache_layer%u_pos%u_node%u_k.bin",
        (unsigned)layerIndex, (unsigned)pos, (unsigned)nodeIndex);
    std::snprintf(vPath, sizeof(vPath), "/tmp/dllama_kvcache_layer%u_pos%u_node%u_v.bin",
        (unsigned)layerIndex, (unsigned)pos, (unsigned)nodeIndex);

    auto dumpOne = [&](const char *path, const NnSize3D &size, NnByte *data) {
        std::unique_ptr<FILE, int(*)(FILE *)> f(std::fopen(path, "wb"), std::fclose);
        if (!f) return false;
        const uint32_t magic = 0x4b564344u; // 'DVCK'
        const uint32_t version = 1u;
        std::fwrite(&magic, sizeof(magic), 1, f.get());
        std::fwrite(&version, sizeof(version), 1, f.get());
        std::fwrite(&nodeIndex, sizeof(nodeIndex), 1, f.get());
        std::fwrite(&layerIndex, sizeof(layerIndex), 1, f.get());
        std::fwrite(&pos, sizeof(pos), 1, f.get());
        std::fwrite(&size, sizeof(size), 1, f.get());
        std::fwrite(data, 1, size.nBytes, f.get());
        return true;
    };

    const bool kOk = dumpOne(kPath, kSize, context->buffers[kBufferIndex]);
    const bool vOk = dumpOne(vPath, vSize, context->buffers[vBufferIndex]);
    std::printf("🧠 [kvcache][dump] node=%u layer=%u pos=%u k=%s (%s) v=%s (%s)\n",
        (unsigned)nodeIndex,
        (unsigned)layerIndex,
        (unsigned)pos,
        kPath,
        kOk ? "ok" : "fail",
        vPath,
        vOk ? "ok" : "fail");
    std::fflush(stdout);
}

static inline void printTpAffectedRange(
    const NnCpuOpContext *context,
    const NnSegmentConfig *segmentConfig,
    NnUint opIndex,
    NnUint layerIndex,
    NnUint pos,
    NnUint nodeIndex,
    unsigned int epoch,
    NnUint stageIndex,
    NnUint stageRank,
    NnUint ffnStart,
    NnUint ffnLen) {
    if (context == nullptr)
        return;

    printf("📐 [tp-range] node=%u stage=%u rank=%u epoch=%u layer=%u pos=%u op=%u name=%s code=%s inX=%u outX=%u\n",
        (unsigned)nodeIndex,
        (unsigned)stageIndex,
        (unsigned)stageRank,
        (unsigned)epoch,
        (unsigned)layerIndex,
        (unsigned)pos,
        (unsigned)opIndex,
        (context->name ? context->name : "Unknown"),
        opCodeToString(context->opCode),
        (unsigned)context->inputSize.x,
        (unsigned)context->outputSize.x);

    // Best-effort: unify qStart/kvStart so node0/node1 complement is easy to see.
    // For non-attention ops these may be unknown.
    NnUint qStart = 0xFFFFFFFFu;
    NnUint kvStart = 0xFFFFFFFFu;
    if (context->opCode == OP_MULTIHEAD_ATT) {
        const auto *cfg = (const NnMultiHeadAttOpConfig *)context->opConfig;
        qStart = cfg->qStart;
        kvStart = cfg->kvStart;
    } else if (context->opCode == OP_ROPE) {
        const auto *cfg = (const NnRopeOpConfig *)context->opConfig;
        qStart = cfg->slice.qDimStart;
        kvStart = cfg->slice.kvDimStart;
    } else if (context->opCode == OP_SHIFT) {
        const auto *cfg = (const NnShiftOpCodeConfig *)context->opConfig;
        kvStart = cfg->dstColStart;
    }
    printf("    TP starts: qStart=%s kvStart=%s\n",
        (qStart == 0xFFFFFFFFu) ? "?" : std::to_string((unsigned)qStart).c_str(),
        (kvStart == 0xFFFFFFFFu) ? "?" : std::to_string((unsigned)kvStart).c_str());

    // Print FFN split only when it's relevant to the op.
    // Attention ops (e.g. block_matmul_wo) are driven by headSplit, not ffnSplit.
    bool ffnRelevant = false;
    if (context->opCode == OP_MUL || context->opCode == OP_SILU) {
        ffnRelevant = true;
    } else if (context->opCode == OP_MATMUL) {
        const auto *cfg = (const NnMatmulOpConfig *)context->opConfig;
        ffnRelevant = (cfg->inSliceTag == NN_SLICE_FFN) || (cfg->outSliceTag == NN_SLICE_FFN);
    }
    if (ffnRelevant && ffnStart != 0xFFFFFFFFu && ffnLen != 0xFFFFFFFFu) {
        printf("    FFN split: start=%u len=%u\n", (unsigned)ffnStart, (unsigned)ffnLen);
    }

    switch (context->opCode) {
    case OP_MULTIHEAD_ATT: {
        const auto *cfg = (const NnMultiHeadAttOpConfig *)context->opConfig;
        printf("    MHA q:  start=%u len=%u stride=%u (nHeads=%u nHeads0=%u headDim=%u)\n",
            (unsigned)cfg->qStart, (unsigned)cfg->qSliceD0, (unsigned)cfg->qStride,
            (unsigned)cfg->nHeads, (unsigned)cfg->nHeads0, (unsigned)cfg->headDim);
        printf("    MHA kv: start=%u len=%u stride=%u (nKvHeads=%u headDim=%u)\n",
            (unsigned)cfg->kvStart, (unsigned)cfg->kvDim0, (unsigned)cfg->kvStride,
            (unsigned)cfg->nKvHeads, (unsigned)cfg->headDim);
        break;
    }
    case OP_MATMUL: {
        const auto *cfg = (const NnMatmulOpConfig *)context->opConfig;
        const NnUint aLen = (cfg->aView.sizeX == 0u) ? context->inputSize.x : cfg->aView.sizeX;
        const NnUint cLen = (cfg->cView.sizeX == 0u) ? context->outputSize.x : cfg->cView.sizeX;
        printf("    MATMUL view=%u inStart=%u outStart=%u aView={off=%u sizeX=%u strideX=%u} cView={off=%u sizeX=%u strideX=%u}\n",
            (unsigned)cfg->view,
            (unsigned)cfg->inStart,
            (unsigned)cfg->outStart,
            (unsigned)cfg->aView.offset,
            (unsigned)cfg->aView.sizeX,
            (unsigned)cfg->aView.strideX,
            (unsigned)cfg->cView.offset,
            (unsigned)cfg->cView.sizeX,
            (unsigned)cfg->cView.strideX);
        printf("    MATMUL A: start=%u len=%u | C: start=%u len=%u\n",
            (unsigned)cfg->aView.offset,
            (unsigned)aLen,
            (unsigned)cfg->cView.offset,
            (unsigned)cLen);
        printf("    MATMUL tags: inTag=%s outTag=%s (units inStartUnit=%u outStartUnit=%u)\n",
            sliceTagToString(cfg->inSliceTag),
            sliceTagToString(cfg->outSliceTag),
            (unsigned)cfg->inStartUnit,
            (unsigned)cfg->outStartUnit);

        // Weight layout note (matches matmulForward_* implementation):
        // - weightSize.y is the global input dim (n)
        // - weightSize.x is the global output dim (d)
        // - row stride in bytes for one output row is getBytes(type, weightSize.y)
        printf("    WEIGHT dims(z,y,x)=(%u,%u,%u) type=%u (n=weightSize.y d=weightSize.x)\n",
            (unsigned)context->weightSize.z,
            (unsigned)context->weightSize.y,
            (unsigned)context->weightSize.x,
            (unsigned)context->weightSize.floatType);

        const NnSize rowStrideBytes = getBytes(context->weightSize.floatType, context->weightSize.y);
        // We don't know the active expert here; for non-MoE models this is always 0.
        const NnSize expertBase = 0u;
        NnSize begin = expertBase;
        NnSize end = expertBase + context->weightSize.nBytesXY;

        if (cfg->view == 1u || cfg->view == 2u) {
            begin = expertBase + (NnSize)cfg->outStart * rowStrideBytes;
            end = begin + (NnSize)cLen * rowStrideBytes;
        }
        // For view=2, actual reads are a column slice [inStart, inStart+aLen) within each row.
        if (cfg->view == 2u) {
            const NnSize colBegin = getBytes(context->weightSize.floatType, cfg->inStart);
            const NnSize colEnd = getBytes(context->weightSize.floatType, cfg->inStart + aLen);
            printf("    WEIGHT read(view2): rows=[%u,%u) cols=[%u,%u) rowStrideBytes=%zu approxRowSpan=[%zu,%zu)\n",
                (unsigned)cfg->outStart,
                (unsigned)(cfg->outStart + cLen),
                (unsigned)cfg->inStart,
                (unsigned)(cfg->inStart + aLen),
                (size_t)rowStrideBytes,
                (size_t)(begin + colBegin),
                (size_t)(begin + colEnd));
        }
        printf("    WEIGHT read: allocXY=[0,%zu) approxContig=[%zu,%zu) (expertBase=%zu)\n",
            (size_t)context->weightSize.nBytesXY,
            (size_t)begin,
            (size_t)end,
            (size_t)expertBase);
        break;
    }
    case OP_MUL: {
        const auto *cfg = (const NnMulOpCodeConfig *)context->opConfig;
        const NnUint viewLen = (cfg->view.sizeX != 0u) ? cfg->view.sizeX : context->inputSize.x;
        const NnUint strideX = (cfg->view.strideX != 0u) ? cfg->view.strideX : 1u;
        printf("    MUL multBuf=%u view={start=%u len=%u stride=%u} (raw:off=%u sizeX=%u strideX=%u)\n",
            (unsigned)cfg->multiplierBufferIndex,
            (unsigned)cfg->view.offset,
            (unsigned)viewLen,
            (unsigned)strideX,
            (unsigned)cfg->view.offset,
            (unsigned)cfg->view.sizeX,
            (unsigned)cfg->view.strideX);
        break;
    }
    case OP_SILU: {
        const auto *cfg = (const NnSiluOpCodeConfig *)context->opConfig;
        const NnUint viewLen = (cfg->view.sizeX != 0u) ? cfg->view.sizeX : context->inputSize.x;
        const NnUint strideX = (cfg->view.strideX != 0u) ? cfg->view.strideX : 1u;
        printf("    SILU view={start=%u len=%u stride=%u} (raw:off=%u sizeX=%u strideX=%u)\n",
            (unsigned)cfg->view.offset,
            (unsigned)viewLen,
            (unsigned)strideX,
            (unsigned)cfg->view.offset,
            (unsigned)cfg->view.sizeX,
            (unsigned)cfg->view.strideX);
        break;
    }
    case OP_CAST: {
        const auto *cfg = (const NnCastOpCodeConfig *)context->opConfig;
        const NnUint viewLen = (cfg->view.sizeX != 0u) ? cfg->view.sizeX : context->inputSize.x;
        const NnUint strideX = (cfg->view.strideX != 0u) ? cfg->view.strideX : 1u;
        printf("    CAST view={start=%u len=%u stride=%u} (raw:off=%u sizeX=%u strideX=%u)\n",
            (unsigned)cfg->view.offset,
            (unsigned)viewLen,
            (unsigned)strideX,
            (unsigned)cfg->view.offset,
            (unsigned)cfg->view.sizeX,
            (unsigned)cfg->view.strideX);
        break;
    }
    case OP_ROPE: {
        const auto *cfg = (const NnRopeOpConfig *)context->opConfig;
        const NnUint viewLen = (cfg->view.sizeX != 0u) ? cfg->view.sizeX : context->inputSize.x;
        printf("    ROPE isQ=%u view={off=%u len=%u strideX=%u} slice{qStart=%u qLen=%u kvStart=%u kvLen=%u headDim=%u}\n",
            (unsigned)cfg->isQ,
            (unsigned)cfg->view.offset,
            (unsigned)viewLen,
            (unsigned)cfg->view.strideX,
            (unsigned)cfg->slice.qDimStart,
            (unsigned)cfg->slice.qDim0,
            (unsigned)cfg->slice.kvDimStart,
            (unsigned)cfg->slice.kvDim0,
            (unsigned)cfg->slice.headDim);
        break;
    }
    case OP_SHIFT: {
        const auto *cfg = (const NnShiftOpCodeConfig *)context->opConfig;
        printf("    SHIFT dstColStart=%u dstRowStride=%u dstColStartUnit=%u indexPipe=%u\n",
            (unsigned)cfg->dstColStart,
            (unsigned)cfg->dstRowStride,
            (unsigned)cfg->dstColStartUnit,
            (unsigned)cfg->indexPipeIndex);
        break;
    }
    case OP_INV_RMS: {
        const auto *cfg = (const NnInvRmsOpConfig *)context->opConfig;
        const NnUint viewLen = (cfg->view.sizeX != 0u) ? cfg->view.sizeX : context->inputSize.x;
        printf("    INV_RMS nColumns=%u view={off=%u len=%u strideX=%u}\n",
            (unsigned)cfg->nColumns,
            (unsigned)cfg->view.offset,
            (unsigned)viewLen,
            (unsigned)cfg->view.strideX);
        break;
    }
    case OP_RMS_NORM: {
        const auto *cfg = (const NnRmsNormOpConfig *)context->opConfig;
        const NnUint viewLen = (cfg->view.sizeX != 0u) ? cfg->view.sizeX : context->inputSize.x;
        printf("    RMS_NORM nColumns=%u invRmsBuf=%u view={off=%u len=%u strideX=%u}\n",
            (unsigned)cfg->nColumns,
            (unsigned)cfg->invRmsBufferIndex,
            (unsigned)cfg->view.offset,
            (unsigned)viewLen,
            (unsigned)cfg->view.strideX);
        break;
    }
    default:
        // Only print detailed ranges for TP-sensitive ops.
        break;
    }
    (void)segmentConfig;
}

#endif // DLLAMA_DEBUG_ATTN

#if !DLLAMA_DEBUG_ATTN

static inline NnUint loadPosFromPipe(const NnCpuOpContext *context, NnUint pipeIndex) {
    if (context == nullptr || context->pipes == nullptr)
        return 0xFFFFFFFFu;
    const float posF = *(const float *)(context->pipes[pipeIndex]);
    if (!(posF >= 0.0f))
        return 0xFFFFFFFFu;
    return (NnUint)posF;
}

static inline void maybeDumpHardcodedLayerKvCacheOnce(
    const NnCpuOpContext *context,
    const NnSegmentConfig *segmentConfig,
    NnUint opIndex,
    NnUint nodeIndex) {
    if (context == nullptr || segmentConfig == nullptr || context->name == nullptr) return;
    if (context->opCode != OP_MULTIHEAD_ATT) return;
    if (std::strstr(context->name, "block_multihead_att") == nullptr) return;

    const NnUint layerIndex = segmentConfig->ops[opIndex].index;
    static constexpr NnUint kTargetLayer = 15u;
    static constexpr NnUint kTargetPos = 32u;
    if (layerIndex != kTargetLayer) return;

    const auto *cfg = (const NnMultiHeadAttOpConfig *)context->opConfig;
    if (cfg == nullptr) return;
    const NnUint pos = loadPosFromPipe(context, cfg->positionPipeIndex);
    if (pos != kTargetPos) return;

    static std::atomic_uint8_t dumpedOnce{0u};
    const NnUint old = dumpedOnce.exchange(1u, std::memory_order_acq_rel);
    if (old != 0u) return;

    const NnUint kBufferIndex = cfg->keyCacheBufferIndex;
    const NnUint vBufferIndex = cfg->valueCacheBufferIndex;

    const NnSize3D kSize = context->bufferConfigs[kBufferIndex].size;
    const NnSize3D vSize = context->bufferConfigs[vBufferIndex].size;

    char kPath[256];
    char vPath[256];
    std::snprintf(kPath, sizeof(kPath), "/tmp/dllama_kvcache_layer%u_pos%u_node%u_k.bin",
        (unsigned)layerIndex, (unsigned)pos, (unsigned)nodeIndex);
    std::snprintf(vPath, sizeof(vPath), "/tmp/dllama_kvcache_layer%u_pos%u_node%u_v.bin",
        (unsigned)layerIndex, (unsigned)pos, (unsigned)nodeIndex);

    auto dumpOne = [&](const char *path, const NnSize3D &size, NnByte *data) {
        std::unique_ptr<FILE, int(*)(FILE *)> f(std::fopen(path, "wb"), std::fclose);
        if (!f) return false;
        const uint32_t magic = 0x4b564344u; // 'DVCK'
        const uint32_t version = 1u;
        std::fwrite(&magic, sizeof(magic), 1, f.get());
        std::fwrite(&version, sizeof(version), 1, f.get());
        std::fwrite(&nodeIndex, sizeof(nodeIndex), 1, f.get());
        std::fwrite(&layerIndex, sizeof(layerIndex), 1, f.get());
        std::fwrite(&pos, sizeof(pos), 1, f.get());
        std::fwrite(&size, sizeof(size), 1, f.get());
        std::fwrite(data, 1, size.nBytes, f.get());
        return true;
    };

    const bool kOk = dumpOne(kPath, kSize, context->buffers[kBufferIndex]);
    const bool vOk = dumpOne(vPath, vSize, context->buffers[vBufferIndex]);
    std::printf("🧠 [kvcache][dump] node=%u layer=%u pos=%u k=%s (%s) v=%s (%s)\n",
        (unsigned)nodeIndex,
        (unsigned)layerIndex,
        (unsigned)pos,
        kPath,
        kOk ? "ok" : "fail",
        vPath,
        vOk ? "ok" : "fail");
    std::fflush(stdout);
}

#endif

static inline void maybeTraceRedundantInferencePath(
    const NnCpuOpContext *context,
    const NnSegmentConfig *segmentConfig,
    NnUint opIndex,
    NnUint nodeIndex,
    NnUint segmentIndex) {
    if (context == nullptr || segmentConfig == nullptr) return;
    if (context->opCode != OP_MULTIHEAD_ATT) return;

    bool segmentIsRuntimeRedundant = false;
    char opNameBuf[64];
    if (segmentConfig->ops != nullptr) {
        for (NnUint i = 0; i < segmentConfig->nOps; ++i) {
            const char *opName = segmentConfig->ops[i].name;
            if (opName != nullptr && std::strstr(opName, "runtime_redundant_") != nullptr) {
                segmentIsRuntimeRedundant = true;
                std::strncpy(opNameBuf, opName, sizeof(opNameBuf) - 1);
                break;
            }
        }
    }
    if (!segmentIsRuntimeRedundant) return;

    const auto *cfg = (const NnMultiHeadAttOpConfig *)context->opConfig;
    if (cfg == nullptr) return;

    const NnUint pos = loadPosFromPipe(context, cfg->positionPipeIndex);
    if (pos == 0xFFFFFFFFu) return;

    const NnUint layerIndex = segmentConfig->ops[opIndex].index;
    std::printf("✅ [redundant-infer] node=%u seg=%u layer=%u pos=%u op=%s opname=%s path=active\n",
        (unsigned)nodeIndex,
        (unsigned)segmentIndex,
        (unsigned)layerIndex,
        (unsigned)pos,
        (context->name != nullptr) ? context->name : "unknown",
        (opNameBuf[0] != '\0') ? opNameBuf : "unknown");
    std::fflush(stdout);
}

#if DLLAMA_DEBUG_ATTN

static inline void printPtrSampleDbg(const char *label, NnByte **ptrs, const NnSize3D &sz) {
    if (ptrs == nullptr) {
        printf("    %s=null\n", (label ? label : "ptr"));
        return;
    }
    // Print first pointer for z=0,y=0 plus stride between batches to detect slot alignment.
    NnByte *p00 = ptrs[0];
    NnByte *p01 = (sz.y >= 2u) ? ptrs[1] : nullptr;
    printf("    %s[0]=%p", (label ? label : "ptr"), (void *)p00);
    if (p01 != nullptr) {
        const ptrdiff_t delta = (ptrdiff_t)(p01 - p00);
        bool hasExpect = false;
        size_t expectRowBytes = 0;
        if (sz.x > 0u) {
            if (sz.floatType == F_Q80) {
                // getBytes(F_Q80, n) requires n to be block-aligned.
                if ((sz.x % Q80_BLOCK_SIZE) == 0u) {
                    expectRowBytes = (size_t)getBytes(sz.floatType, sz.x);
                    hasExpect = true;
                }
            } else {
                expectRowBytes = (size_t)getBytes(sz.floatType, sz.x);
                hasExpect = true;
            }
        }
        if (hasExpect) {
            printf(" %s[1]=%p deltaB=%td expectB=%zu", (label ? label : "ptr"), (void *)p01, delta, expectRowBytes);
        } else {
            printf(" %s[1]=%p deltaB=%td", (label ? label : "ptr"), (void *)p01, delta);
        }
    }
    printf("\n");
}

static inline void printTensorViewDbg(const char *label, const NnTensorView &v, const NnUint fallbackSizeX) {
    const NnUint sizeX = (v.sizeX != 0u) ? v.sizeX : fallbackSizeX;
    const NnUint strideX = (v.strideX != 0u) ? v.strideX : 1u;
    printf("    view[%s] start=%u len=%u stride=%u (raw:off=%u sizeX=%u strideX=%u)\n",
        (label ? label : "?"),
        (unsigned)v.offset,
        (unsigned)sizeX,
        (unsigned)strideX,
        (unsigned)v.offset,
        (unsigned)v.sizeX,
        (unsigned)v.strideX);
}

static void printOpSliceParamsDbg(const char *opName, const NnOpCode opCode, const void *opConfig, const NnPointerConfig *inPtr, const NnPointerConfig *outPtr, const NnSize3D &inSize, const NnSize3D &outSize) {
    // If caller provides pointer configs, gate by DLLAMA_DEBUG_SLICE_PARAMS (build-time).
    // If pointer configs are null, allow forward-time logging gate to control output.
    if (inPtr != nullptr || outPtr != nullptr) {
        if (!debugSliceParamsEnabled(opName))
            return;
    }
    printf("🧩 [slice] op=%s code=%s inTag=%s outTag=%s inX=%u outX=%u\n",
        (opName ? opName : "Unknown"),
        opCodeToString(opCode),
        (inPtr ? sliceTagToString(inPtr->sliceTag) : "?"),
        (outPtr ? sliceTagToString(outPtr->sliceTag) : "?"),
        (unsigned)inSize.x,
        (unsigned)outSize.x);

    if (opCode == OP_MULTIHEAD_ATT) {
        const auto *cfg = (const NnMultiHeadAttOpConfig *)opConfig;
        printf("    q:   start=%u len=%u stride=%u (nHeads=%u nHeads0=%u headDim=%u qSliceD0=%u)\n",
            (unsigned)cfg->qStart, (unsigned)cfg->qSliceD0, (unsigned)cfg->qStride,
            (unsigned)cfg->nHeads, (unsigned)cfg->nHeads0, (unsigned)cfg->headDim, (unsigned)cfg->qSliceD0);
        printf("    kv:  start=%u len=%u stride=%u (nKvHeads=%u headDim=%u kvDim0=%u)\n",
            (unsigned)cfg->kvStart, (unsigned)cfg->kvDim0, (unsigned)cfg->kvStride,
            (unsigned)cfg->nKvHeads, (unsigned)cfg->headDim, (unsigned)cfg->kvDim0);
        return;
    }

    if (opCode == OP_SHIFT) {
        const auto *cfg = (const NnShiftOpCodeConfig *)opConfig;
        if (cfg->dstRowStride != 0u) {
            printf("    dst: start=%u len=%u stride=%u (unit=%u)\n",
                (unsigned)cfg->dstColStart, (unsigned)outSize.x, (unsigned)cfg->dstRowStride, (unsigned)cfg->dstColStartUnit);
        } else {
            printf("    dst: packed len=%u\n", (unsigned)outSize.x);
        }
        return;
    }

    if (opCode == OP_MATMUL) {
        const auto *cfg = (const NnMatmulOpConfig *)opConfig;
        printf("    matmul: view=%u inStart=%u outStart=%u inTag=%s outTag=%s\n",
            (unsigned)cfg->view,
            (unsigned)cfg->inStart,
            (unsigned)cfg->outStart,
            sliceTagToString(cfg->inSliceTag),
            sliceTagToString(cfg->outSliceTag));
        if (cfg->aView.offset != 0u || cfg->aView.sizeX != 0u || cfg->aView.strideX != 0u)
            printTensorViewDbg("A", cfg->aView, inSize.x);
        if (cfg->cView.offset != 0u || cfg->cView.sizeX != 0u || cfg->cView.strideX != 0u)
            printTensorViewDbg("C", cfg->cView, outSize.x);
        return;
    }

    if (opCode == OP_ROPE) {
        const auto *cfg = (const NnRopeOpConfig *)opConfig;
        printTensorViewDbg(cfg->isQ ? "Q" : "K", cfg->view, inSize.x);
        return;
    }

    if (opCode == OP_SILU) {
        const auto *cfg = (const NnSiluOpCodeConfig *)opConfig;
        printTensorViewDbg("SILU", cfg->view, inSize.x);
        return;
    }
    if (opCode == OP_GELU) {
        const auto *cfg = (const NnGeluOpCodeConfig *)opConfig;
        printTensorViewDbg("GELU", cfg->view, inSize.x);
        return;
    }
    if (opCode == OP_MUL) {
        const auto *cfg = (const NnMulOpCodeConfig *)opConfig;
        printTensorViewDbg("MUL", cfg->view, inSize.x);
        return;
    }
    if (opCode == OP_SCALE) {
        const auto *cfg = (const NnScaleOpCodeConfig *)opConfig;
        printTensorViewDbg("SCALE", cfg->view, inSize.x);
        return;
    }
    if (opCode == OP_CAST) {
        const auto *cfg = (const NnCastOpCodeConfig *)opConfig;
        printTensorViewDbg("CAST", cfg->view, inSize.x);
        return;
    }
}

#endif // DLLAMA_DEBUG_ATTN

static inline bool migrationOpTraceEnabled() {
    const char *v = std::getenv("DLLAMA_MIGRATION_OP_TRACE");
    if (v == nullptr) return false;
    if (v[0] == '0') return false;
    if ((v[0] == 'f' || v[0] == 'F') && (v[1] == 'a' || v[1] == 'A')) return false;
    return true;
}

static inline bool finalNormInputTraceEnabled() {
    const char *v = std::getenv("DLLAMA_DEBUG_FINAL_NORM_INPUT");
    if (v == nullptr) return false;
    if (v[0] == '0') return false;
    if ((v[0] == 'f' || v[0] == 'F') && (v[1] == 'a' || v[1] == 'A')) return false;
    return true;
}

static inline NnUint finalNormInputTraceLimit() {
    const char *v = std::getenv("DLLAMA_DEBUG_FINAL_NORM_INPUT_LIMIT");
    if (v == nullptr) return 64u;
    long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline NnUint finalNormInputTraceSampleCount() {
    const char *v = std::getenv("DLLAMA_DEBUG_FINAL_NORM_INPUT_SAMPLE");
    if (v == nullptr) return 8u;
    long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 256) return 256u;
    return (NnUint)n;
}

static inline bool finalLogitsSliceTraceEnabled() {
    const char *v = std::getenv("DLLAMA_DEBUG_FINAL_LOGITS_SLICE");
    if (v == nullptr) return false;
    if (v[0] == '0') return false;
    if ((v[0] == 'f' || v[0] == 'F') && (v[1] == 'a' || v[1] == 'A')) return false;
    return true;
}

static inline NnUint finalLogitsSliceTraceLimit() {
    const char *v = std::getenv("DLLAMA_DEBUG_FINAL_LOGITS_SLICE_LIMIT");
    if (v == nullptr) return 128u;
    long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline bool layerOutTraceEnabled() {
    const char *v = std::getenv("DLLAMA_DEBUG_LAYER_OUT_TRACE");
    if (v == nullptr) return false;
    if (v[0] == '0') return false;
    if ((v[0] == 'f' || v[0] == 'F') && (v[1] == 'a' || v[1] == 'A')) return false;
    return true;
}

static inline int layerOutTracePosFilter() {
    const char *v = std::getenv("DLLAMA_DEBUG_LAYER_OUT_TRACE_POS");
    if (v == nullptr || v[0] == '\0') return -1;
    return std::atoi(v);
}

static inline int layerOutTraceLayerFilter() {
    const char *v = std::getenv("DLLAMA_DEBUG_LAYER_OUT_TRACE_LAYER");
    if (v == nullptr || v[0] == '\0') return -1;
    return std::atoi(v);
}

static inline NnUint layerOutTraceLimit() {
    const char *v = std::getenv("DLLAMA_DEBUG_LAYER_OUT_TRACE_LIMIT");
    if (v == nullptr) return 256u;
    long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline bool layerOutCompareEnabled() {
    const char *v = std::getenv("DLLAMA_DEBUG_LAYER_OUT_COMPARE");
    if (v == nullptr) return false;
    if (v[0] == '0') return false;
    if ((v[0] == 'f' || v[0] == 'F') && (v[1] == 'a' || v[1] == 'A')) return false;
    return true;
}

static inline bool residualChainTraceEnabled() {
    const char *v = std::getenv("DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE");
    if (v == nullptr) return false;
    if (v[0] == '0') return false;
    if ((v[0] == 'f' || v[0] == 'F') && (v[1] == 'a' || v[1] == 'A')) return false;
    return true;
}

static inline int residualChainTracePosFilter() {
    const char *v = std::getenv("DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_POS");
    if (v == nullptr || v[0] == '\0') return -1;
    return std::atoi(v);
}

static inline int residualChainTraceLayerFilter() {
    const char *v = std::getenv("DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_LAYER");
    if (v == nullptr || v[0] == '\0') return -1;
    return std::atoi(v);
}

static inline NnUint residualChainTraceLimit() {
    const char *v = std::getenv("DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_LIMIT");
    if (v == nullptr) return 256u;
    long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline NnUint residualChainTraceSliceLimit() {
    const char *v = std::getenv("DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_SLICES");
    if (v == nullptr) return 0u;
    long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1024) return 1024u;
    return (NnUint)n;
}

static inline NnUint residualChainTraceSamples() {
    const char *v = std::getenv("DLLAMA_DEBUG_RESIDUAL_CHAIN_TRACE_SAMPLES");
    if (v == nullptr) return 4u;
    long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 1u;
    if (n > 8) return 8u;
    return (NnUint)n;
}

static inline NnSize typedOffsetBytesNoAssert(NnFloatType type, NnUint xOffset) {
    if (type == F_32) return (NnSize)xOffset * sizeof(float);
    if (type == F_16) return (NnSize)xOffset * sizeof(NnFp16);
    if (type == F_Q40) {
        if ((xOffset % Q40_BLOCK_SIZE) != 0u) return (NnSize)-1;
        return (NnSize)(xOffset / Q40_BLOCK_SIZE) * sizeof(NnBlockQ40);
    }
    if (type == F_Q80) {
        if ((xOffset % Q80_BLOCK_SIZE) != 0u) return (NnSize)-1;
        return (NnSize)(xOffset / Q80_BLOCK_SIZE) * sizeof(NnBlockQ80);
    }
    return (NnSize)-1;
}

static inline bool readTypedValue(const NnByte *base, NnFloatType type, NnUint index, double *outValue) {
    if (base == nullptr || outValue == nullptr) return false;
    if (type == F_32) {
        const float *p = (const float *)base;
        *outValue = (double)p[index];
        return true;
    }
    if (type == F_16) {
        const NnFp16 *p = (const NnFp16 *)base;
        *outValue = (double)CONVERT_F16_TO_F32(p[index]);
        return true;
    }
    if (type == F_Q80) {
        const NnBlockQ80 *p = (const NnBlockQ80 *)base;
        const NnUint blockIndex = index / Q80_BLOCK_SIZE;
        const NnUint inBlockIndex = index % Q80_BLOCK_SIZE;
        const double d = (double)CONVERT_F16_TO_F32(p[blockIndex].d);
        *outValue = d * (double)p[blockIndex].qs[inBlockIndex];
        return true;
    }
    return false;
}

static inline void printResidualChainStats(
    const char *phase,
    const NnByte *base,
    NnFloatType type,
    NnUint x,
    const char *opName,
    const char *role,
    NnUint node,
    unsigned int epoch,
    NnUint layer,
    NnUint pos,
    NnUint sliceIndex,
    NnUint nSlices,
    NnUint sampleCount) {

    if (phase == nullptr) phase = "unknown";
    if (opName == nullptr) opName = "?";
    if (role == nullptr) role = "unknown";
    if (x == 0u || base == nullptr) {
        std::printf("[res-chain] phase=%s op=%s role=%s node=%u epoch=%u layer=%u pos=%u type=%s x=%u slice=%u/%u empty=1\n",
            phase,
            opName,
            role,
            (unsigned)node,
            (unsigned)epoch,
            (unsigned)layer,
            (unsigned)pos,
            floatTypeToString(type),
            (unsigned)x,
            (unsigned)sliceIndex,
            (unsigned)nSlices);
        return;
    }

    double first = 0.0;
    if (!readTypedValue(base, type, 0u, &first)) {
        std::printf("[res-chain] phase=%s op=%s role=%s node=%u epoch=%u layer=%u pos=%u type=%s x=%u slice=%u/%u unsupported=1\n",
            phase,
            opName,
            role,
            (unsigned)node,
            (unsigned)epoch,
            (unsigned)layer,
            (unsigned)pos,
            floatTypeToString(type),
            (unsigned)x,
            (unsigned)sliceIndex,
            (unsigned)nSlices);
        return;
    }

    double vMin = first;
    double vMax = first;
    double sumAbs = 0.0;
    NnUint nZero = 0u;
    NnUint nNan = 0u;
    NnUint nInf = 0u;

    for (NnUint i = 0u; i < x; ++i) {
        double v = 0.0;
        if (!readTypedValue(base, type, i, &v)) {
            nNan += 1u;
            continue;
        }
        if (std::isnan(v)) { nNan += 1u; continue; }
        if (!std::isfinite(v)) { nInf += 1u; continue; }
        if (v == 0.0) nZero += 1u;
        if (v < vMin) vMin = v;
        if (v > vMax) vMax = v;
        sumAbs += std::fabs(v);
    }

    double samples[8] = {0.0};
    if (sampleCount == 0u) sampleCount = 1u;
    if (sampleCount > 8u) sampleCount = 8u;
    for (NnUint i = 0u; i < sampleCount; ++i) {
        const NnUint si = (sampleCount == 1u) ? 0u : (NnUint)(((NnSize)i * (NnSize)(x - 1u)) / (NnSize)(sampleCount - 1u));
        double v = 0.0;
        if (!readTypedValue(base, type, si, &v)) {
            v = std::numeric_limits<double>::quiet_NaN();
        }
        samples[i] = v;
    }

    std::printf("[res-chain] phase=%s op=%s role=%s node=%u epoch=%u layer=%u pos=%u type=%s x=%u slice=%u/%u range=[%.7g,%.7g] meanAbs=%.7g zero=%u/%u nan=%u inf=%u s=[",
        phase,
        opName,
        role,
        (unsigned)node,
        (unsigned)epoch,
        (unsigned)layer,
        (unsigned)pos,
        floatTypeToString(type),
        (unsigned)x,
        (unsigned)sliceIndex,
        (unsigned)nSlices,
        vMin,
        vMax,
        (x > 0u ? (sumAbs / (double)x) : 0.0),
        (unsigned)nZero,
        (unsigned)x,
        (unsigned)nNan,
        (unsigned)nInf);
    for (NnUint i = 0u; i < sampleCount; ++i) {
        std::printf("%s%.6g", (i == 0u ? "" : ","), samples[i]);
    }
    std::printf("]\n");
}

static inline const char *layerOutCompareMode() {
    const char *v = std::getenv("DLLAMA_DEBUG_LAYER_OUT_COMPARE_MODE");
    return (v == nullptr || v[0] == '\0') ? "compare" : v;
}

static inline const char *layerOutCompareFile() {
    const char *v = std::getenv("DLLAMA_DEBUG_LAYER_OUT_COMPARE_FILE");
    return (v == nullptr || v[0] == '\0') ? "/tmp/dllama_layer_out_baseline.log" : v;
}

static inline bool layerOutCompareModeIs(const char *mode) {
    return std::strcmp(layerOutCompareMode(), mode) == 0;
}

static inline double layerOutCompareRelEps() {
    const char *v = std::getenv("DLLAMA_DEBUG_LAYER_OUT_COMPARE_REL_EPS");
    if (v == nullptr || v[0] == '\0') return 1e-3;
    const double x = std::strtod(v, nullptr);
    return (x > 0.0) ? x : 1e-3;
}

static inline double layerOutCompareAbsEps() {
    const char *v = std::getenv("DLLAMA_DEBUG_LAYER_OUT_COMPARE_ABS_EPS");
    if (v == nullptr || v[0] == '\0') return 1e-4;
    const double x = std::strtod(v, nullptr);
    return (x > 0.0) ? x : 1e-4;
}

static inline double relDiff(double cur, double base) {
    return std::fabs(cur - base) / (std::fabs(base) + 1e-12);
}

static void layerOutCompareRecord(
    const char *opName,
    NnUint node,
    NnUint layer,
    NnUint pos,
    NnUint outX,
    double vMin,
    double vMax,
    double meanAbs,
    NnUint nZero,
    NnUint nNan,
    NnUint nInf,
    double s0,
    double s1,
    double s2,
    double s3) {

    const char *path = layerOutCompareFile();
    FILE *f = std::fopen(path, "a");
    if (f == nullptr) return;
    std::fprintf(f,
        "%s %u %u %u %u %.10g %.10g %.10g %u %u %u %.10g %.10g %.10g %.10g\n",
        (opName != nullptr ? opName : "?"),
        (unsigned)node,
        (unsigned)layer,
        (unsigned)pos,
        (unsigned)outX,
        vMin,
        vMax,
        meanAbs,
        (unsigned)nZero,
        (unsigned)nNan,
        (unsigned)nInf,
        s0,
        s1,
        s2,
        s3);
    std::fclose(f);
}

static bool layerOutCompareLoadBaseline(
    const char *opName,
    NnUint node,
    NnUint layer,
    NnUint pos,
    NnUint *outX,
    double *vMin,
    double *vMax,
    double *meanAbs,
    NnUint *nZero,
    NnUint *nNan,
    NnUint *nInf,
    double *s0,
    double *s1,
    double *s2,
    double *s3) {

    const char *path = layerOutCompareFile();
    FILE *f = std::fopen(path, "r");
    if (f == nullptr) return false;

    char line[1024];
    bool found = false;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        if (line[0] == '\0' || line[0] == '#') continue;
        char op[128] = {0};
        unsigned n = 0u, l = 0u, p = 0u, x = 0u;
        unsigned z = 0u, nn = 0u, ni = 0u;
        double mn = 0.0, mx = 0.0, ma = 0.0;
        double q0 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0;
        const int k = std::sscanf(line,
            "%127s %u %u %u %u %lf %lf %lf %u %u %u %lf %lf %lf %lf",
            op, &n, &l, &p, &x, &mn, &mx, &ma, &z, &nn, &ni, &q0, &q1, &q2, &q3);
        if (k != 15) continue;
        if (std::strcmp(op, (opName != nullptr ? opName : "?")) != 0) continue;
        if (n != (unsigned)node || l != (unsigned)layer || p != (unsigned)pos) continue;

        if (outX) *outX = (NnUint)x;
        if (vMin) *vMin = mn;
        if (vMax) *vMax = mx;
        if (meanAbs) *meanAbs = ma;
        if (nZero) *nZero = (NnUint)z;
        if (nNan) *nNan = (NnUint)nn;
        if (nInf) *nInf = (NnUint)ni;
        if (s0) *s0 = q0;
        if (s1) *s1 = q1;
        if (s2) *s2 = q2;
        if (s3) *s3 = q3;
        found = true;
        break;
    }

    std::fclose(f);
    return found;
}

static void layerOutCompareRun(
    const char *opName,
    NnUint node,
    NnUint layer,
    NnUint pos,
    NnUint outX,
    double vMin,
    double vMax,
    double meanAbs,
    NnUint nZero,
    NnUint nNan,
    NnUint nInf,
    double s0,
    double s1,
    double s2,
    double s3) {

    if (!layerOutCompareEnabled()) return;

    if (layerOutCompareModeIs("record")) {
        layerOutCompareRecord(opName, node, layer, pos, outX, vMin, vMax, meanAbs, nZero, nNan, nInf, s0, s1, s2, s3);
        return;
    }
    if (!layerOutCompareModeIs("compare")) return;

    NnUint bx = 0u;
    double bMin = 0.0, bMax = 0.0, bMeanAbs = 0.0;
    NnUint bZero = 0u, bNan = 0u, bInf = 0u;
    double bs0 = 0.0, bs1 = 0.0, bs2 = 0.0, bs3 = 0.0;
    const bool hasBase = layerOutCompareLoadBaseline(opName, node, layer, pos,
        &bx, &bMin, &bMax, &bMeanAbs, &bZero, &bNan, &bInf, &bs0, &bs1, &bs2, &bs3);
    if (!hasBase) {
        std::printf("[layer-out-diff] op=%s node=%u layer=%u pos=%u baseline=missing file=%s\n",
            (opName != nullptr ? opName : "?"),
            (unsigned)node,
            (unsigned)layer,
            (unsigned)pos,
            layerOutCompareFile());
        return;
    }

    const double relMean = relDiff(meanAbs, bMeanAbs);
    const double relMin = relDiff(vMin, bMin);
    const double relMax = relDiff(vMax, bMax);
    const double abs0 = std::fabs(s0 - bs0);
    const double abs1 = std::fabs(s1 - bs1);
    const double abs2 = std::fabs(s2 - bs2);
    const double abs3 = std::fabs(s3 - bs3);

    const double relEps = layerOutCompareRelEps();
    const double absEps = layerOutCompareAbsEps();
    const bool mismatch =
        (bx != outX) ||
        (nZero != bZero) ||
        (nNan != bNan) ||
        (nInf != bInf) ||
        (relMean > relEps) ||
        (relMin > relEps) ||
        (relMax > relEps) ||
        (abs0 > absEps) ||
        (abs1 > absEps) ||
        (abs2 > absEps) ||
        (abs3 > absEps);

    std::printf("[layer-out-diff] op=%s node=%u layer=%u pos=%u mismatch=%u meanAbs=%.7g/%.7g relMean=%.3e range=[%.7g,%.7g]/[%.7g,%.7g] s=[%.6g,%.6g,%.6g,%.6g]/[%.6g,%.6g,%.6g,%.6g]\n",
        (opName != nullptr ? opName : "?"),
        (unsigned)node,
        (unsigned)layer,
        (unsigned)pos,
        (unsigned)(mismatch ? 1u : 0u),
        meanAbs,
        bMeanAbs,
        relMean,
        vMin,
        vMax,
        bMin,
        bMax,
        s0,
        s1,
        s2,
        s3,
        bs0,
        bs1,
        bs2,
        bs3);

    if (mismatch) {
        static std::atomic<NnUint> firstMismatchPrinted{0u};
        if (firstMismatchPrinted.exchange(1u, std::memory_order_acq_rel) == 0u) {
            std::printf("[layer-out-first-mismatch] op=%s node=%u layer=%u pos=%u file=%s relEps=%.3e absEps=%.3e\n",
                (opName != nullptr ? opName : "?"),
                (unsigned)node,
                (unsigned)layer,
                (unsigned)pos,
                layerOutCompareFile(),
                relEps,
                absEps);
        }
    }
}

static inline NnUint migrationOpTraceLimit() {
    const char *v = std::getenv("DLLAMA_MIGRATION_OP_TRACE_LIMIT");
    if (v == nullptr) return 0u;
    long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline NnUint migrationKvTraceDims() {
    const char *v = std::getenv("DLLAMA_MIGRATION_KV_TRACE_DIMS");
    if (v == nullptr) return 8u;
    long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1024) return 1024u;
    return (NnUint)n;
}

static inline bool nameHas(const char *name, const char *needle) {
    if (name == nullptr || needle == nullptr) return false;
    return std::strstr(name, needle) != nullptr;
}

static inline bool isFfnTraceOp(const NnCpuOpContext *context) {
    if (context == nullptr || context->name == nullptr) return false;
    if (context->opCode == OP_MATMUL) {
        return nameHas(context->name, "block_matmul_w1") ||
               nameHas(context->name, "block_matmul_w2") ||
               nameHas(context->name, "block_matmul_w3") ||
               nameHas(context->name, "block_moe_");
    }
    if (context->opCode == OP_MUL || context->opCode == OP_SILU || context->opCode == OP_GELU) {
        return nameHas(context->name, "block_mul") || nameHas(context->name, "block_act");
    }
    return false;
}

static inline void printTraceFloatPrefix(const char *tag, const float *ptr, NnUint n) {
    if (tag == nullptr) tag = "vec";
    if (ptr == nullptr || n == 0u) {
        std::printf("%s=[]", tag);
        return;
    }
    std::printf("%s=[", tag);
    for (NnUint i = 0u; i < n; ++i) {
        std::printf("%s%.6f", (i == 0u ? "" : ","), (double)ptr[i]);
    }
    std::printf("]");
}

static inline const char *migrationSegmentRoleForTrace(const NnSegmentConfig *segmentConfig, const NnCpuOpContext *context) {
    if (segmentConfig != nullptr && segmentConfig->ops != nullptr) {
        for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
            if (nameHas(segmentConfig->ops[i].name, "runtime_redundant_")) return "redundant";
        }
        return "primary";
    }
    if (context == nullptr || context->name == nullptr) return "unknown";
    if (nameHas(context->name, "runtime_redundant_")) return "redundant";
    return "primary";
}

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

static bool hasHeadDeltaForStage(const std::vector<int> &deltaHeadOrKv, const NnStageConfig *stage) {
    if (stage == nullptr) return false;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        const NnUint n = stage->nodeIndices[i];
        if (n < deltaHeadOrKv.size() && deltaHeadOrKv[n] != 0) return true;
    }
    return false;
}

static bool validateKvShadowCoverage(
    const NnUnevenPartitionPlan *plan,
    const NnStageConfig *stage,
    const std::vector<int> &deltaHeadOrKv,
    char *reason,
    size_t reasonSize
) {
    auto setReason = [&](const char *value) {
        if (reason != nullptr && reasonSize != 0u) {
            std::snprintf(reason, reasonSize, "%s", value != nullptr ? value : "kv_shadow_invalid");
        }
    };
    if (!getEnableStageFullWeights()) {
        setReason("kv_shadow_requires_stage_full_weights");
        return false;
    }
    if (!getEnableKvRedundancyDuringMigration()) {
        setReason("kv_shadow_redundancy_disabled");
        return false;
    }
    if (plan == nullptr || stage == nullptr ||
        plan->kvHeadSplit.starts == nullptr || plan->kvHeadSplit.lengths == nullptr ||
        plan->kvHeadComputeSplit.starts == nullptr || plan->kvHeadComputeSplit.lengths == nullptr) {
        setReason("kv_shadow_split_missing");
        return false;
    }

    NnUint newStart = 0u;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        const NnUint n = stage->nodeIndices[i];
        if (n >= deltaHeadOrKv.size()) {
            setReason("kv_shadow_delta_missing");
            return false;
        }
        const int newLenSigned = (int)plan->kvHeadSplit.lengths[n] + deltaHeadOrKv[n];
        if (newLenSigned < 0) {
            setReason("kv_shadow_underflow");
            return false;
        }

        const NnUint newLen = (NnUint)newLenSigned;
        const NnUint newEnd = newStart + newLen;
        const NnUint shadowStart = plan->kvHeadComputeSplit.starts[n];
        const NnUint shadowEnd = shadowStart + plan->kvHeadComputeSplit.lengths[n];
        if (newStart < shadowStart || newEnd > shadowEnd) {
            setReason("kv_shadow_coverage_miss");
            return false;
        }
        newStart = newEnd;
    }
    setReason("");
    return true;
}


NnCpuDevice::NnCpuDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan) {
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->netExecution = netExecution;
    this->partitionPlan = partitionPlan;

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
}

NnCpuDevice::~NnCpuDevice() {
    for (NnUint bufferIndex = 0; bufferIndex < nBuffers; bufferIndex++) {
        releaseAlignedBuffer(buffers[bufferIndex]);
    }
    delete[] buffers;
    delete[] bufferFlags;
}

void NnCpuDevice::setPartitionPlan(const NnUnevenPartitionPlan *newPlan) {
    this->partitionPlan = newPlan;
    if (this->nodeConfig != nullptr) {
        this->nodeConfig->partitionPlan = newPlan;
    }
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

        // Debug: print slice/view parameters for full-buffer + local-slice execution.
    #if DLLAMA_DEBUG_ATTN
        printOpSliceParamsDbg(opConfig->name, opConfig->code, opConfig->config, &opConfig->input, &opConfig->output, inputSize, outputSize);
    #endif

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
        opContext->opCode = opConfig->code;
        opContext->opIndex = opIndex;
        opContext->nodeIndex = nodeConfig ? nodeConfig->nodeIndex : 0u;
        opContext->opConfig = opConfig->config;
        opContext->weightSize = opConfig->weightSize;
        opContext->nBatches = netConfig->nBatches;
        opContext->pipes = netExecution->pipes;
        opContext->pipeConfigs = netConfig->pipes;
        opContext->buffers = buffers;
        opContext->bufferConfigs = nodeConfig->buffers;
        opContext->bufferFlags = bufferFlags;

        opContext->input = new NnByte *[inputsPtr[opIndex].size()];
        opContext->inputSize = inputSizes[opIndex];
        opContext->hasInputContinuousMemory = hasPointerContinuousMemory(&opConfig->input);
        std::memcpy(opContext->input, inputsPtr[opIndex].data(), inputsPtr[opIndex].size() * sizeof(NnByte *));

        opContext->output = new NnByte *[outputsPtr[opIndex].size()];
        opContext->outputSize = outputSizes[opIndex];
        opContext->hasOutputContinuousMemory = hasPointerContinuousMemory(&opConfig->output);
        std::memcpy(opContext->output, outputsPtr[opIndex].data(), outputsPtr[opIndex].size() * sizeof(NnByte *));

        opContext->weightLoadedMin = std::numeric_limits<NnSize>::max();
        opContext->weightLoadedMax = 0;
        opContext->weightLoadedBytes = 0;
        opContext->weightLoadCalls = 0u;
        opContext->weightReadPrinted = 0u;

#if not(DEBUG_USE_MMAP_FOR_WEIGHTS)
        if (opContext->weightSize.nBytes > 0)
            opContext->weight = allocAlignedBuffer(opContext->weightSize.nBytes);
        else
            opContext->weight = nullptr;
#endif

#if DLLAMA_DEBUG_ATTN
        if (opContext->weightSize.nBytes > 0 && debugWeightRangesEnabled(opContext->name)) {
            printf("🧱 [weights][alloc] op=%s idx=%u code=%s weightBytes=%zu required=[0,%zu) dims(z,y,x)=(%u,%u,%u) type=%u\n",
                opContext->name,
                opIndex,
                opCodeToString(opConfig->code),
                (size_t)opContext->weightSize.nBytes,
                (size_t)opContext->weightSize.nBytes,
                opContext->weightSize.z,
                opContext->weightSize.y,
                opContext->weightSize.x,
                (unsigned)opContext->weightSize.floatType);
            std::fflush(stdout);
        }
#endif

        if (opInit != nullptr)
            opInit(opContext);
        opForward[opIndex] = opForwardLocal[opIndex];
    }
    return new NnCpuDeviceSegment(this, segmentIndex, segmentConfig, opForward, opContexts, segmentConfig->nOps);
}

NnCpuDeviceSegment::~NnCpuDeviceSegment() {
    for (NnUint opIndex = 0; opIndex < nOps; opIndex++) {
        NnCpuOpContext *context = &opContexts[opIndex];
        delete[] context->input;
        delete[] context->output;
#if not(DEBUG_USE_MMAP_FOR_WEIGHTS)
        if (context->weightSize.nBytes > 0)
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
            bool stackedByNode = (pointerConfig->sliceTag == NN_SLICE_STACKED_BY_NODE);

            const char* matchedName = nullptr;
            NnUint matchedSplitTotal = 0u;
            NnUint matchedMultiplier = 0u;

#if DLLAMA_DEBUG_ATTN
            // Debug: trace how slice offsets/lengths are computed.
            static std::atomic<NnUint> resolvePrinted{0u};
            const bool rpDbg = debugResolvePointerEnabled() && debugResolvePointerPassesFilter(pointerConfig->source, pointerConfig->pointerIndex, pointerConfig->type, pointerConfig->sliceTag);
            const NnUint rpLimit = debugResolvePointerLimit();
            const NnUint rpIdx = rpDbg ? resolvePrinted.fetch_add(1u) : 0u;
            const bool rpAllow = rpDbg && (rpLimit == 0u || rpIdx < rpLimit);
            const NnStageConfig* dbgStage = nullptr;
            NnUint dbgRank = 0u;
            NnUint dbgStageNodes = 0u;
            if (rpAllow) {
                if (partitionPlan != nullptr && partitionPlan->nStages > 0) {
                    dbgStage = findStageForNode(partitionPlan, nodeConfig->nodeIndex);
                    if (dbgStage != nullptr) {
                        dbgStageNodes = dbgStage->nNodes;
                        dbgRank = findStageRank(dbgStage, nodeConfig->nodeIndex);
                    }
                }
                printf("🧭 [resolve][begin] src=%s idx=%u type=BATCHED_SLICE tag=%s totalX=%u floatType=%u stageNodes=%u rank=%u\n",
                    (pointerConfig->source == SRC_BUFFER ? "BUFFER" : "PIPE"),
                    (unsigned)pointerConfig->pointerIndex,
                    sliceTagToString(pointerConfig->sliceTag),
                    (unsigned)sourceSize->x,
                    (unsigned)sourceSize->floatType,
                    (unsigned)dbgStageNodes,
                    (unsigned)dbgRank);
            }
#endif

            // 1. 尝试查阅 Partition Plan 来获取精确的非均匀 Offset/Length
            if (partitionPlan != nullptr && netConfig->nNodes == partitionPlan->nNodes) {
                NnUint totalDim = sourceSize->x; // 管道的总维度
                NnUint nodeIdx = nodeConfig->nodeIndex;
                const NnStageConfig* myStage = (partitionPlan->nStages > 0) ? findStageForNode(partitionPlan, nodeIdx) : nullptr;

                // If tag explicitly says stacked-by-node, do not attempt split matching.
                // Otherwise, keep legacy stacked-by-node detection for backward compatibility.
                // IMPORTANT: this heuristic is only valid for PIPES (e.g. ZQ = dim * nNodes).
                // Applying it to BUFFERS would misclassify full Q/K/V buffers as stacked-by-node
                // and force a uniform fallback slice (e.g. off=1024), breaking head-based slicing.
                if (!stackedByNode && pointerConfig->source == SRC_PIPE) {
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

#if DLLAMA_DEBUG_ATTN
                        if (rpAllow) {
                            matchedName = name;
                            matchedSplitTotal = splitTotal;
                            matchedMultiplier = multiplier;
                        }
#endif

                        // Q80 requires block-aligned offsets and lengths.
                        // If we can't satisfy alignment, reject this split match.
                        if (sourceSize->floatType == F_Q80) {
                            if ((myOffset % Q80_BLOCK_SIZE) != 0 || (myLength % Q80_BLOCK_SIZE) != 0) {
                                return false;
                            }
                        }

                        return true;
                    }
                    return false;
                };

                // If sliceTag is explicitly set, use it deterministically.
                // Otherwise, fall back to legacy heuristic matching (Vocab > FFN > Dim > Heads).
                if (!splitFound && pointerConfig->sliceTag != NN_SLICE_AUTO) {
                    switch (pointerConfig->sliceTag) {
                        case NN_SLICE_VOCAB:
                            splitFound = tryApplySplit(partitionPlan->vocabSplit, false, "vocab");
                            break;
                        case NN_SLICE_FFN:
                            splitFound = tryApplySplit(partitionPlan->ffnSplit, false, "ffn");
                            break;
                        case NN_SLICE_DIM:
                            splitFound = tryApplySplit(partitionPlan->dimSplit, false, "dim");
                            break;
                        case NN_SLICE_HEAD:
                            // Heads have headDim multiplier; explicit tag means the caller knows this buffer is head-sliced.
                            splitFound = tryApplySplit(partitionPlan->headSplit, true, "head");
                            break;
                        case NN_SLICE_KV_HEAD:
                            splitFound = tryApplySplit(partitionPlan->kvHeadSplit, true, "kvhead");
                            break;
                        case NN_SLICE_STACKED_BY_NODE:
                            // handled by stackedByNode flag
                            break;
                        case NN_SLICE_AUTO:
                        default:
                            break;
                    }
                }

                // Legacy heuristic matching (kept for backward compatibility).
                if (!splitFound && pointerConfig->sliceTag == NN_SLICE_AUTO) {
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

#if DLLAMA_DEBUG_ATTN
                if (rpAllow) {
                    printf("🧭 [resolve][fallback] src=%s idx=%u tag=%s stackedByNode=%u nSplitNodes=%u rank=%u off=%u len=%u\n",
                        (pointerConfig->source == SRC_BUFFER ? "BUFFER" : "PIPE"),
                        (unsigned)pointerConfig->pointerIndex,
                        sliceTagToString(pointerConfig->sliceTag),
                        stackedByNode ? 1u : 0u,
                        (unsigned)((partitionPlan && partitionPlan->nStages > 0 && dbgStage != nullptr) ? dbgStage->nNodes : netConfig->nNodes),
                        (unsigned)((partitionPlan && partitionPlan->nStages > 0 && dbgStage != nullptr) ? dbgRank : nodeConfig->nodeIndex),
                        (unsigned)myOffset,
                        (unsigned)myLength);
                }
#endif
            }

            // 3. 应用偏移量 (带越界保护)
              // For Q80, getBytes(F_Q80, n) requires n to be block-aligned.
              // tryApplySplit() enforces alignment for explicit matches; legacy fallback must also be aligned.

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

#if DLLAMA_DEBUG_ATTN
            if (rpAllow) {
                const void *base0 = (const void *)&source[0];
                const void *ret0 = (const void *)pntr[0];
                printf("🧭 [resolve][done] src=%s idx=%u tag=%s splitFound=%u match=%s splitTotal=%u mult=%u off=%u len=%u base=%p ret=%p deltaB=%zu\n",
                    (pointerConfig->source == SRC_BUFFER ? "BUFFER" : "PIPE"),
                    (unsigned)pointerConfig->pointerIndex,
                    sliceTagToString(pointerConfig->sliceTag),
                    splitFound ? 1u : 0u,
                    (matchedName ? matchedName : "-"),
                    (unsigned)matchedSplitTotal,
                    (unsigned)matchedMultiplier,
                    (unsigned)myOffset,
                    (unsigned)myLength,
                    base0,
                    ret0,
                    (size_t)((const NnByte *)ret0 - (const NnByte *)base0));
            }
#endif

        }

        if (pointerConfig->type == PNTR_BATCH) {
            // PNTR_BATCH keeps full per-batch data without applying per-node slicing.
            return pntr;
        }
        return pntr;
    }
    default:
        throw std::invalid_argument("Unsupported pointer config");
    }
}

void NnCpuDeviceSegment::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    if (device != nullptr) {
        device->setPartitionPlan(plan);
    }
}

void NnCpuDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    assert(opIndex >= 0u);
    assert(opIndex < nOps);
    NnCpuOpContext *context = &opContexts[opIndex];

#if DLLAMA_DEBUG_ATTN
    if (context->weightSize.nBytes > 0 && debugWeightRangesEnabled(context->name)) {
        const NnSize end = offset + nBytes;
        context->weightLoadCalls += 1u;
        context->weightLoadedBytes += nBytes;
        if (offset < context->weightLoadedMin) context->weightLoadedMin = offset;
        if (end > context->weightLoadedMax) context->weightLoadedMax = end;

        printf("📥 [weights][load] op=%s idx=%u write=[%zu,%zu) bytes=%zu alloc=[0,%zu) calls=%u\n",
            (context->name ? context->name : "Unknown"),
            (unsigned)opIndex,
            (size_t)offset,
            (size_t)end,
            (size_t)nBytes,
            (size_t)context->weightSize.nBytes,
            (unsigned)context->weightLoadCalls);
        std::fflush(stdout);
    }
#endif

    if (offset + nBytes > context->weightSize.nBytes) {
        std::cerr << "🚨 CRITICAL ERROR in loadWeight:" << std::endl;
        std::cerr << "   Op Name: " << (context->name ? context->name : "Unknown") << std::endl;
        std::cerr << "   Op Index: " << opIndex << std::endl;
        std::cerr << "   Offset: " << offset << std::endl;
        std::cerr << "   Write Bytes: " << nBytes << std::endl;
        std::cerr << "   Required (Offset + Bytes): " << (offset + nBytes) << std::endl;
        std::cerr << "   Allocated Size: " << context->weightSize.nBytes << std::endl;
        std::cerr << "   Diff: " << (long long)(offset + nBytes) - (long long)context->weightSize.nBytes << std::endl;
    }
    assert(offset + nBytes <= context->weightSize.nBytes);
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

    // If the partition plan was updated (epoch bumped), refresh this segment's pointers/configs
    // before executing the next op.
    if (device != nullptr) {
        const unsigned int deviceEpoch = device->getPlanEpoch();
        const unsigned int readyEpoch = planEpochReady.load(std::memory_order_acquire);
        if (readyEpoch != deviceEpoch) {
            if (threadIndex == 0u) {
                refreshPointers();
                planEpochReady.store(deviceEpoch, std::memory_order_release);
            } else {
                while (planEpochReady.load(std::memory_order_acquire) != deviceEpoch) {
                }
            }
        }
    }

    if (threadIndex == 0u && device != nullptr) {
        maybeTraceRedundantInferencePath(context, segmentConfig, opIndex, device->getNodeIndex(), segmentIndex);
        maybeDumpHardcodedLayerKvCacheOnce(context, segmentConfig, opIndex, device->getNodeIndex());
    }

    // ------------------------------------------------------------------
    // CPU-only: plan barrier/apply ops (PlanCommand-driven online migration)
    // ------------------------------------------------------------------
    if (context->opCode == OP_PLAN_BARRIER) {
        if (threadIndex == 0u && device != nullptr && context->input != nullptr && context->output != nullptr) {
            const NnUnevenPartitionPlan *planConst = device->getPartitionPlan();

            const PlanCommandSnapshot snap = planCommandCache().load();
            const PlanCommand &pc = snap.cmd;
            const bool hasPlanPayload =
                (pc.cmdKind == PLAN_CMD_KIND_HEAD ||
                 pc.cmdKind == PLAN_CMD_KIND_FFN ||
                 pc.cmdKind == PLAN_CMD_KIND_BOTH ||
                 (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves != 0u));
            const bool hasCmd =
                (pc.magic == DLLAMA_PLAN_CMD_MAGIC) &&
                (pc.version == DLLAMA_PLAN_CMD_VERSION_V1 || pc.version == DLLAMA_PLAN_CMD_VERSION_V2) &&
                (pc.mode == PLAN_CMD_MODE_EXACT || pc.mode == PLAN_CMD_MODE_NEXT_BARRIER) &&
                hasPlanPayload;

            const NnUint layerIndex = (segmentConfig != nullptr) ? segmentConfig->ops[opIndex].index : 0u;
            const float posF = *(const float *)(context->input[0]);
            const NnUint pos = (posF >= 0.0f) ? (NnUint)posF : 0u;

            // Determine my stage/root.
            const NnUint myNode = device->getNodeIndex();
            const NnStageConfig *myStage = (planConst != nullptr && planConst->nStages > 0)
                ? findStageForNode(planConst, myNode)
                : nullptr;

            const bool stageOk = (pc.stageIndex == 0xFFFFFFFFu) || (myStage != nullptr && myStage->stageIndex == pc.stageIndex);
            const bool isStageRoot = stageOk && (myStage != nullptr) && (myStage->rootNodeIndex == myNode);

            const unsigned int curEpoch = device->getPlanEpoch();
            unsigned int emitEpoch = curEpoch;
            unsigned int cmd = 0u;
            unsigned int headMove = 0u;
            unsigned int ffnMove = 0u;
            unsigned int fromNode = 0u;
            unsigned int toNode = 0u;

            bool trigger = false;
            if (hasCmd && isStageRoot) {
                // One-shot guard: only emit once per command seq.
                const unsigned int lastEmitted = device->getLastPlanCmdSeqEmitted();
                if (pc.seq != 0u && pc.seq <= lastEmitted) {
                    trigger = false;
                } else {
                if (pc.mode == PLAN_CMD_MODE_EXACT) {
                    trigger = (layerIndex == pc.triggerLayer) && (pos == pc.triggerPos);
                } else if (pc.mode == PLAN_CMD_MODE_NEXT_BARRIER) {
                    trigger = true;
                }
                }
            }

            if (trigger) {
                emitEpoch = curEpoch + 1u;
                const bool useSingleMoveList = (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves == 1u);
                const bool useMoveList = (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves > 1u);
                if (useSingleMoveList) {
                    // Encode the single v2 move directly in the broadcast pipe. Workers do not
                    // share the root process planCommandCache, so cmdlist-by-seq is unsafe here.
                    const PlanMove &m = pc.moves[0];
                    cmd = m.cmdKind;
                    headMove = m.headMove;
                    ffnMove = m.ffnMove;
                    fromNode = m.fromNodeIndex;
                    toNode = m.toNodeIndex;
                    printf("🧭 [plan][emit] node=%u stage=%u layer=%u pos=%u epoch=%u kind=single-move headMove=%u ffnMove=%u from=%u to=%u seq=%u\n",
                        (unsigned)myNode,
                        (unsigned)(myStage != nullptr ? myStage->stageIndex : 0u),
                        (unsigned)layerIndex,
                        (unsigned)pos,
                        (unsigned)emitEpoch,
                        (unsigned)headMove,
                        (unsigned)ffnMove,
                        (unsigned)fromNode,
                        (unsigned)toNode,
                        (unsigned)pc.seq);
                } else if (useMoveList) {
                    // cmd=4 means: apply from PlanCommand move list, referenced by seq.
                    cmd = 4u;
                    headMove = 0u;
                    ffnMove = 0u;
                    fromNode = pc.seq; // overloaded: cmdSeq
                    toNode = 0u;
                    printf("🧭 [plan][emit] node=%u stage=%u layer=%u pos=%u epoch=%u kind=cmdlist seq=%u moves=%u\n",
                        (unsigned)myNode,
                        (unsigned)(myStage != nullptr ? myStage->stageIndex : 0u),
                        (unsigned)layerIndex,
                        (unsigned)pos,
                        (unsigned)emitEpoch,
                        (unsigned)pc.seq,
                        (unsigned)pc.nMoves);
                } else {
                    cmd = pc.cmdKind;
                    headMove = pc.nHeadsToMove;
                    ffnMove = pc.nFfnToMove;
                    fromNode = pc.fromNodeIndex;
                    toNode = pc.toNodeIndex;

                    const char *kind = (cmd == 1u) ? "head" : (cmd == 2u ? "ffn" : (cmd == 3u ? "both" : "unknown"));
                    if (cmd == 3u) {
                        printf("🧭 [plan][emit] node=%u stage=%u layer=%u pos=%u epoch=%u kind=%s headMove=%u ffnMove=%u from=%u to=%u\n",
                            (unsigned)myNode,
                            (unsigned)(myStage != nullptr ? myStage->stageIndex : 0u),
                            (unsigned)layerIndex,
                            (unsigned)pos,
                            (unsigned)emitEpoch,
                            kind,
                            (unsigned)headMove,
                            (unsigned)ffnMove,
                            (unsigned)fromNode,
                            (unsigned)toNode);
                    } else {
                        const unsigned int move = (cmd == 2u) ? ffnMove : headMove;
                        printf("🧭 [plan][emit] node=%u stage=%u layer=%u pos=%u epoch=%u kind=%s move=%u from=%u to=%u\n",
                        (unsigned)myNode,
                        (unsigned)(myStage != nullptr ? myStage->stageIndex : 0u),
                        (unsigned)layerIndex,
                        (unsigned)pos,
                        (unsigned)emitEpoch,
                        kind,
                        (unsigned)move,
                        (unsigned)fromNode,
                        (unsigned)toNode);
                    }
                }
                const char *kind = (cmd == 1u) ? "head" : (cmd == 2u ? "ffn" : (cmd == 3u ? "both" : "unknown"));
                std::fflush(stdout);
                std::fflush(stdout);

                device->setLastPlanCmdSeqEmitted(pc.seq);
            }

            // Write control packet into the plan pipe for all batches.
            for (NnUint b = 0; b < batchSize; ++b) {
                float *out = (float *)context->output[b];
                out[0] = (float)emitEpoch;
                out[1] = (float)cmd;
                out[2] = (float)fromNode;
                out[3] = (float)toNode;
                // Always encode both moves; cmdKind decides what to apply.
                out[4] = (float)headMove;
                out[5] = (float)layerIndex;
                out[6] = (float)pos;
                out[7] = (float)ffnMove;
            }
        }
        return;
    }

    if (context->opCode == OP_PLAN_APPLY) {
        if (threadIndex == 0u && device != nullptr && context->input != nullptr) {
            const auto *cfg = (const NnPlanApplyOpCodeConfig *)context->opConfig;
            (void)cfg;

            const unsigned int curEpoch = device->getPlanEpoch();
            const float *in0 = (const float *)context->input[0];
            const unsigned int msgEpoch = (in0[0] >= 0.0f) ? (unsigned int)in0[0] : 0u;
            const unsigned int cmd = (in0[1] >= 0.0f) ? (unsigned int)in0[1] : 0u;
            const unsigned int fromNode = (in0[2] >= 0.0f) ? (unsigned int)in0[2] : 0u;
            const unsigned int toNode = (in0[3] >= 0.0f) ? (unsigned int)in0[3] : 0u;
            const unsigned int headMove = (in0[4] >= 0.0f) ? (unsigned int)in0[4] : 0u;
            const unsigned int layerIndex = (in0[5] >= 0.0f) ? (unsigned int)in0[5] : 0u;
            const unsigned int pos = (in0[6] >= 0.0f) ? (unsigned int)in0[6] : 0u;
            const unsigned int ffnMove = (in0[7] >= 0.0f) ? (unsigned int)in0[7] : 0u;

            if ((cmd == 1u || cmd == 2u || cmd == 3u || cmd == 4u) && msgEpoch > curEpoch) {
                auto *plan = const_cast<NnUnevenPartitionPlan *>(device->getPartitionPlan());
                if (plan != nullptr && plan->nStages > 0) {
                    const NnUint myNode = device->getNodeIndex();
                    const NnStageConfig *myStage = findStageForNode(plan, myNode);
                    if (myStage != nullptr && myStage->stageIndex == ((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex) {
                        const NnStageConfig *st = &plan->stages[((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex];

                        auto inStage = [&](NnUint node) -> bool {
                            for (NnUint i = 0; i < st->nNodes; ++i) {
                                if (st->nodeIndices[i] == node) return true;
                            }
                            return false;
                        };

                        auto stageRank = [&](NnUint node) -> int {
                            for (NnUint i = 0; i < st->nNodes; ++i) {
                                if (st->nodeIndices[i] == node) return (int)i;
                            }
                            return -1;
                        };

                        auto areAdjacentInStage = [&](NnUint a, NnUint b) -> bool {
                            const int ra = stageRank(a);
                            const int rb = stageRank(b);
                            if (ra < 0 || rb < 0) return false;
                            const int d = (ra >= rb) ? (ra - rb) : (rb - ra);
                            return d == 1;
                        };

                        // Determine GQA group size (best-effort).
                        NnUint stageQHeadsTotal = 0u;
                        NnUint stageKvHeadsTotal = 0u;
                        if (st != nullptr && plan->kvHeadSplit.lengths != nullptr && plan->headSplit.lengths != nullptr) {
                            for (NnUint i = 0; i < st->nNodes; ++i) {
                                const NnUint n = st->nodeIndices[i];
                                stageQHeadsTotal += plan->headSplit.lengths[n];
                                stageKvHeadsTotal += plan->kvHeadSplit.lengths[n];
                            }
                        }
                        NnUint gqaGroupSize = 1u;
                        if (stageKvHeadsTotal != 0u && stageQHeadsTotal != 0u && (stageQHeadsTotal % stageKvHeadsTotal) == 0u) {
                            const NnUint g = stageQHeadsTotal / stageKvHeadsTotal;
                            if (g != 0u) gqaGroupSize = g;
                        }

                        bool changed = false;

                        // Helper: recompute contiguous starts within this stage.
                        auto recomputeStarts = [&](NnDimSplit &split) {
                            NnUint run = 0u;
                            for (NnUint i = 0; i < st->nNodes; ++i) {
                                const NnUint n = st->nodeIndices[i];
                                split.starts[n] = run;
                                run += split.lengths[n];
                            }
                        };

                        auto logLocalWorkSplit = [&](const char *reason) {
                            const NnUint stageIndex = ((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex;
                            const NnUint kvHeadLen = (plan->kvHeadSplit.lengths != nullptr) ? plan->kvHeadSplit.lengths[myNode] : 0u;
                            const NnUint qHeadLen = (plan->headSplit.lengths != nullptr) ? plan->headSplit.lengths[myNode] : 0u;
                            const NnUint ffnLen = (plan->ffnSplit.lengths != nullptr) ? plan->ffnSplit.lengths[myNode] : 0u;
                            printf("📊 [plan][work] node=%u stage=%u epoch=%u layer=%u pos=%u kvHeads=%u qHeads=%u ffnDim=%u reason=%s\n",
                                (unsigned)myNode,
                                (unsigned)stageIndex,
                                (unsigned)msgEpoch,
                                (unsigned)layerIndex,
                                (unsigned)pos,
                                (unsigned)kvHeadLen,
                                (unsigned)qHeadLen,
                                (unsigned)ffnLen,
                                reason);
                            std::fflush(stdout);
                        };

                        // v2 move list apply (cmd==4): aggregate deltas first, then apply once.
                        if (cmd == 4u) {
                            const unsigned int wantSeq = fromNode; // overloaded by barrier for cmdlist
                            const PlanCommandSnapshot snap2 = planCommandCache().load();
                            const PlanCommand &pc2 = snap2.cmd;
                            if (!isValidPlanCommandHeader(pc2) || pc2.seq != wantSeq || pc2.mode == PLAN_CMD_MODE_NONE) {
                                printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u cmdlist missing/mismatch (wantSeq=%u gotSeq=%u ver=%u mode=%u)\n",
                                    (unsigned)myNode,
                                    (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    (unsigned)msgEpoch,
                                    (unsigned)layerIndex,
                                    (unsigned)pos,
                                    (unsigned)wantSeq,
                                    (unsigned)pc2.seq,
                                    (unsigned)pc2.version,
                                    (unsigned)pc2.mode);
                                std::fflush(stdout);
                                return;
                            }
                            if (!planCommandHasMoveList(pc2)) {
                                printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u cmdlist has no moves (seq=%u)\n",
                                    (unsigned)myNode,
                                    (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    (unsigned)msgEpoch,
                                    (unsigned)layerIndex,
                                    (unsigned)pos,
                                    (unsigned)pc2.seq);
                                std::fflush(stdout);
                                return;
                            }

                            const uint32_t stageNodes = st->nNodes;
                            uint32_t maxMovesAllowed = (uint32_t)(2u * stageNodes);
                            if (maxMovesAllowed > DLLAMA_PLAN_CMD_MAX_MOVES) maxMovesAllowed = DLLAMA_PLAN_CMD_MAX_MOVES;
                            if (pc2.nMoves > maxMovesAllowed) {
                                printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: nMoves=%u > maxAllowed=%u (stageNodes=%u)\n",
                                    (unsigned)myNode,
                                    (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    (unsigned)msgEpoch,
                                    (unsigned)layerIndex,
                                    (unsigned)pos,
                                    (unsigned)pc2.nMoves,
                                    (unsigned)maxMovesAllowed,
                                    (unsigned)stageNodes);
                                std::fflush(stdout);
                                return;
                            }

                            std::vector<int> deltaHeadOrKv(plan->nNodes, 0);
                            std::vector<int> deltaFfn(plan->nNodes, 0);

                            bool reject = false;
                            const bool gqaLockstep = (gqaGroupSize > 1u) && plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths;
                            const bool kvRedundancyEnabled = getEnableKvRedundancyDuringMigration();
                            for (uint32_t i = 0; i < pc2.nMoves; ++i) {
                                const PlanMove &m = pc2.moves[i];
                                const NnUint f = (NnUint)m.fromNodeIndex;
                                const NnUint t = (NnUint)m.toNodeIndex;
                                if (!inStage(f) || !inStage(t) || f == t) { reject = true; break; }
                                if (!areAdjacentInStage(f, t)) { reject = true; break; }
                                if (m.headMove != 0u) {
                                    // KV-head migration safety (GQA lockstep): without KV cache transfer, we
                                    // require KV redundancy to be enabled AND moved heads within the precomputed pad.
                                    if (gqaLockstep) {
                                        if (!kvRedundancyEnabled) { reject = true; break; }
                                        if (m.headMove > NN_KV_REDUNDANCY_PAD_HEADS) { reject = true; break; }
                                    }
                                    deltaHeadOrKv[f] -= (int)m.headMove;
                                    deltaHeadOrKv[t] += (int)m.headMove;
                                }
                                if (m.ffnMove != 0u) {
                                    deltaFfn[f] -= (int)m.ffnMove;
                                    deltaFfn[t] += (int)m.ffnMove;
                                }
                            }
                            if (reject) {
                                printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: bad move (non-adjacent/out-of-stage/self or KV safety)\n",
                                    (unsigned)myNode,
                                    (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    (unsigned)msgEpoch,
                                    (unsigned)layerIndex,
                                    (unsigned)pos);
                                std::fflush(stdout);
                                return;
                            }

                            if (gqaLockstep && hasHeadDeltaForStage(deltaHeadOrKv, st)) {
                                char reason[96] = {0};
                                if (!validateKvShadowCoverage(plan, st, deltaHeadOrKv, reason, sizeof(reason))) {
                                    printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: %s\n",
                                        (unsigned)myNode,
                                        (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                        (unsigned)msgEpoch,
                                        (unsigned)layerIndex,
                                        (unsigned)pos,
                                        reason);
                                    std::fflush(stdout);
                                    logCpuPlanApplyAblationEvent(
                                        ((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                        pc2.nMoves > 0u ? pc2.moves[0].fromNodeIndex : 0u,
                                        pc2.nMoves > 0u ? pc2.moves[0].toNodeIndex : 0u,
                                        layerIndex,
                                        pos,
                                        cmd,
                                        false,
                                        reason);
                                    return;
                                }
                            }

                            // Apply head/KV deltas.
                            if (gqaLockstep) {
                                for (NnUint i = 0; i < st->nNodes; ++i) {
                                    const NnUint n = st->nodeIndices[i];
                                    const int d = deltaHeadOrKv[n];
                                    if (d == 0) continue;
                                    const int newLen = (int)plan->kvHeadSplit.lengths[n] + d;
                                    if (newLen < 0) { reject = true; break; }
                                }
                                if (reject) {
                                    printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: kvHead underflow\n",
                                        (unsigned)myNode,
                                        (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                        (unsigned)msgEpoch,
                                        (unsigned)layerIndex,
                                        (unsigned)pos);
                                    std::fflush(stdout);
                                    return;
                                }
                                for (NnUint i = 0; i < st->nNodes; ++i) {
                                    const NnUint n = st->nodeIndices[i];
                                    const int d = deltaHeadOrKv[n];
                                    if (d == 0) continue;
                                    plan->kvHeadSplit.lengths[n] = (NnUint)((int)plan->kvHeadSplit.lengths[n] + d);
                                    changed = true;
                                }
                                if (changed) {
                                    recomputeStarts(plan->kvHeadSplit);
                                    // Derive Q head split from KV head split.
                                    for (NnUint i = 0; i < st->nNodes; ++i) {
                                        const NnUint n = st->nodeIndices[i];
                                        plan->headSplit.starts[n] = plan->kvHeadSplit.starts[n] * gqaGroupSize;
                                        plan->headSplit.lengths[n] = plan->kvHeadSplit.lengths[n] * gqaGroupSize;
                                    }
                                }
                            } else {
                                // Non-GQA: treat headMove as Q heads.
                                if (!plan->headSplit.starts || !plan->headSplit.lengths) return;
                                for (NnUint i = 0; i < st->nNodes; ++i) {
                                    const NnUint n = st->nodeIndices[i];
                                    const int d = deltaHeadOrKv[n];
                                    if (d == 0) continue;
                                    const int newLen = (int)plan->headSplit.lengths[n] + d;
                                    if (newLen < 0) { reject = true; break; }
                                }
                                if (reject) {
                                    printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: head underflow\n",
                                        (unsigned)myNode,
                                        (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                        (unsigned)msgEpoch,
                                        (unsigned)layerIndex,
                                        (unsigned)pos);
                                    std::fflush(stdout);
                                    return;
                                }
                                for (NnUint i = 0; i < st->nNodes; ++i) {
                                    const NnUint n = st->nodeIndices[i];
                                    const int d = deltaHeadOrKv[n];
                                    if (d == 0) continue;
                                    plan->headSplit.lengths[n] = (NnUint)((int)plan->headSplit.lengths[n] + d);
                                    changed = true;
                                }
                                if (changed) recomputeStarts(plan->headSplit);
                            }

                            // Apply FFN deltas.
                            if (!plan->ffnSplit.starts || !plan->ffnSplit.lengths) return;
                            for (NnUint i = 0; i < st->nNodes; ++i) {
                                const NnUint n = st->nodeIndices[i];
                                const int d = deltaFfn[n];
                                if (d == 0) continue;
                                const int newLen = (int)plan->ffnSplit.lengths[n] + d;
                                if (newLen < 0) { reject = true; break; }
                            }
                            if (reject) {
                                printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: ffn underflow\n",
                                    (unsigned)myNode,
                                    (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    (unsigned)msgEpoch,
                                    (unsigned)layerIndex,
                                    (unsigned)pos);
                                std::fflush(stdout);
                                return;
                            }
                            bool ffnChanged = false;
                            for (NnUint i = 0; i < st->nNodes; ++i) {
                                const NnUint n = st->nodeIndices[i];
                                const int d = deltaFfn[n];
                                if (d == 0) continue;
                                plan->ffnSplit.lengths[n] = (NnUint)((int)plan->ffnSplit.lengths[n] + d);
                                ffnChanged = true;
                            }
                            if (ffnChanged) {
                                recomputeStarts(plan->ffnSplit);
                                changed = true;
                            }

                            if (changed) {
                                printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u cmdlist seq=%u moves=%u (gqaGroup=%u)\n",
                                    (unsigned)myNode,
                                    (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    (unsigned)msgEpoch,
                                    (unsigned)layerIndex,
                                    (unsigned)pos,
                                    (unsigned)pc2.seq,
                                    (unsigned)pc2.nMoves,
                                    (unsigned)gqaGroupSize);
                                std::fflush(stdout);
                                logLocalWorkSplit("cmdlist");
                                logCpuPlanApplyAblationEvent(
                                    ((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    pc2.nMoves > 0u ? pc2.moves[0].fromNodeIndex : 0u,
                                    pc2.nMoves > 0u ? pc2.moves[0].toNodeIndex : 0u,
                                    layerIndex,
                                    pos,
                                    cmd,
                                    true,
                                    "");
                                device->setPlanEpoch(msgEpoch);
                            }
                            return;
                        }

                        // Legacy single-edge apply (cmd 1/2/3)
                        if (inStage(fromNode) && inStage(toNode)) {
                            if (!areAdjacentInStage(fromNode, toNode)) {
                                printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: legacy move not adjacent (from=%u to=%u)\n",
                                    (unsigned)myNode,
                                    (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    (unsigned)msgEpoch,
                                    (unsigned)layerIndex,
                                    (unsigned)pos,
                                    (unsigned)fromNode,
                                    (unsigned)toNode);
                                std::fflush(stdout);
                                return;
                            }
                            // Keep previous behavior for single-edge mode.
                            bool legacyChanged = false;
                            bool legacyNoOp = false;

                            const char *legacyKind = (cmd == 1u) ? "head" : (cmd == 2u ? "ffn" : (cmd == 3u ? "both" : "unknown"));

                            auto applyHeadLegacy = [&]() {
                                if (headMove == 0u) return;
                                if (!plan->headSplit.starts || !plan->headSplit.lengths) {
                                    printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: missing head split arrays\n",
                                        (unsigned)myNode,
                                        (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                        (unsigned)msgEpoch,
                                        (unsigned)layerIndex,
                                        (unsigned)pos);
                                    std::fflush(stdout);
                                    legacyNoOp = true;
                                    return;
                                }
                                const bool gqaLockstep = (gqaGroupSize > 1u) && plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths;
                                if (gqaLockstep) {
                                    const bool kvRedundancyEnabled = getEnableKvRedundancyDuringMigration();
                                    if (!kvRedundancyEnabled) {
                                        printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: KV redundancy disabled (--enable-kv-redundancy-during-migration=0)\n",
                                            (unsigned)myNode,
                                            (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                            (unsigned)msgEpoch,
                                            (unsigned)layerIndex,
                                            (unsigned)pos);
                                        std::fflush(stdout);
                                        legacyNoOp = true;
                                        return;
                                    }
                                    const NnUint kvMove = headMove;
                                    if (kvMove > NN_KV_REDUNDANCY_PAD_HEADS) {
                                        printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: KV move=%u exceeds pad=%u\n",
                                            (unsigned)myNode,
                                            (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                            (unsigned)msgEpoch,
                                            (unsigned)layerIndex,
                                            (unsigned)pos,
                                            (unsigned)kvMove,
                                            (unsigned)NN_KV_REDUNDANCY_PAD_HEADS);
                                        std::fflush(stdout);
                                        legacyNoOp = true;
                                        return;
                                    }
                                    if (plan->kvHeadSplit.lengths[fromNode] < kvMove) {
                                        printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: kvHead insufficient (from=%u len=%u need=%u)\n",
                                            (unsigned)myNode,
                                            (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                            (unsigned)msgEpoch,
                                            (unsigned)layerIndex,
                                            (unsigned)pos,
                                            (unsigned)fromNode,
                                            (unsigned)plan->kvHeadSplit.lengths[fromNode],
                                            (unsigned)kvMove);
                                        std::fflush(stdout);
                                        legacyNoOp = true;
                                        return;
                                    }
                                    std::vector<int> deltaHeadOrKv(plan->nNodes, 0);
                                    deltaHeadOrKv[fromNode] -= (int)kvMove;
                                    deltaHeadOrKv[toNode] += (int)kvMove;
                                    char reason[96] = {0};
                                    if (!validateKvShadowCoverage(plan, st, deltaHeadOrKv, reason, sizeof(reason))) {
                                        printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: %s\n",
                                            (unsigned)myNode,
                                            (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                            (unsigned)msgEpoch,
                                            (unsigned)layerIndex,
                                            (unsigned)pos,
                                            reason);
                                        std::fflush(stdout);
                                        logCpuPlanApplyAblationEvent(
                                            ((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                            fromNode,
                                            toNode,
                                            layerIndex,
                                            pos,
                                            cmd,
                                            false,
                                            reason);
                                        legacyNoOp = true;
                                        return;
                                    }
                                    plan->kvHeadSplit.lengths[fromNode] -= kvMove;
                                    plan->kvHeadSplit.lengths[toNode] += kvMove;
                                    recomputeStarts(plan->kvHeadSplit);
                                    for (NnUint i = 0; i < st->nNodes; ++i) {
                                        const NnUint n = st->nodeIndices[i];
                                        plan->headSplit.starts[n] = plan->kvHeadSplit.starts[n] * gqaGroupSize;
                                        plan->headSplit.lengths[n] = plan->kvHeadSplit.lengths[n] * gqaGroupSize;
                                    }
                                    legacyChanged = true;
                                    return;
                                }
                                const NnUint qMove = headMove;
                                if (plan->headSplit.lengths[fromNode] < qMove) {
                                    printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: head insufficient (from=%u len=%u need=%u)\n",
                                        (unsigned)myNode,
                                        (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                        (unsigned)msgEpoch,
                                        (unsigned)layerIndex,
                                        (unsigned)pos,
                                        (unsigned)fromNode,
                                        (unsigned)plan->headSplit.lengths[fromNode],
                                        (unsigned)qMove);
                                    std::fflush(stdout);
                                    legacyNoOp = true;
                                    return;
                                }
                                plan->headSplit.lengths[fromNode] -= qMove;
                                plan->headSplit.lengths[toNode] += qMove;
                                recomputeStarts(plan->headSplit);
                                legacyChanged = true;
                            };

                            auto applyFfnLegacy = [&]() {
                                if (ffnMove == 0u) return;
                                if (!plan->ffnSplit.starts || !plan->ffnSplit.lengths) {
                                    printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: missing ffn split arrays\n",
                                        (unsigned)myNode,
                                        (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                        (unsigned)msgEpoch,
                                        (unsigned)layerIndex,
                                        (unsigned)pos);
                                    std::fflush(stdout);
                                    legacyNoOp = true;
                                    return;
                                }
                                if (plan->ffnSplit.lengths[fromNode] < ffnMove) {
                                    printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u reject: ffn insufficient (from=%u len=%u need=%u)\n",
                                        (unsigned)myNode,
                                        (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                        (unsigned)msgEpoch,
                                        (unsigned)layerIndex,
                                        (unsigned)pos,
                                        (unsigned)fromNode,
                                        (unsigned)plan->ffnSplit.lengths[fromNode],
                                        (unsigned)ffnMove);
                                    std::fflush(stdout);
                                    legacyNoOp = true;
                                    return;
                                }
                                plan->ffnSplit.lengths[fromNode] -= ffnMove;
                                plan->ffnSplit.lengths[toNode] += ffnMove;
                                recomputeStarts(plan->ffnSplit);
                                legacyChanged = true;
                            };

                            if (cmd == 1u) applyHeadLegacy();
                            else if (cmd == 2u) applyFfnLegacy();
                            else if (cmd == 3u) { applyHeadLegacy(); applyFfnLegacy(); }

                            if (legacyChanged) {
                                printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u kind=%s headMove=%u ffnMove=%u from=%u to=%u (gqaGroup=%u)\n",
                                    (unsigned)myNode,
                                    (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    (unsigned)msgEpoch,
                                    (unsigned)layerIndex,
                                    (unsigned)pos,
                                    legacyKind,
                                    (unsigned)headMove,
                                    (unsigned)ffnMove,
                                    (unsigned)fromNode,
                                    (unsigned)toNode,
                                    (unsigned)gqaGroupSize);
                                std::fflush(stdout);
                                logLocalWorkSplit("legacy");
                                logCpuPlanApplyAblationEvent(
                                    ((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    fromNode,
                                    toNode,
                                    layerIndex,
                                    pos,
                                    cmd,
                                    true,
                                    "");
                                device->setPlanEpoch(msgEpoch);
                            } else if (!legacyNoOp) {
                                // This happens if cmd asks to move 0, or cmd kind is unknown.
                                printf("🧭 [plan][apply] node=%u stage=%u epoch=%u layer=%u pos=%u no-op (kind=%s headMove=%u ffnMove=%u from=%u to=%u)\n",
                                    (unsigned)myNode,
                                    (unsigned)((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    (unsigned)msgEpoch,
                                    (unsigned)layerIndex,
                                    (unsigned)pos,
                                    legacyKind,
                                    (unsigned)headMove,
                                    (unsigned)ffnMove,
                                    (unsigned)fromNode,
                                    (unsigned)toNode);
                                std::fflush(stdout);
                                logCpuPlanApplyAblationEvent(
                                    ((const NnPlanApplyOpCodeConfig *)context->opConfig)->onlyStageIndex,
                                    fromNode,
                                    toNode,
                                    layerIndex,
                                    pos,
                                    cmd,
                                    false,
                                    "no_op");
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // Debug final_norm input readiness before RMS_NORM executes.
    if (threadIndex == 0u && device != nullptr &&
        context->opCode == OP_RMS_NORM &&
        context->name != nullptr && std::strstr(context->name, "final_norm") != nullptr &&
        finalNormInputTraceEnabled()) {

        static std::atomic<NnUint> finalNormPrinted{0u};
        const NnUint limit = finalNormInputTraceLimit();
        const NnUint idx = finalNormPrinted.fetch_add(1u, std::memory_order_relaxed);
        if (limit == 0u || idx < limit) {
            const auto *cfg = (const NnRmsNormOpConfig *)context->opConfig;
            const NnUint viewOffset = (cfg != nullptr) ? cfg->view.offset : 0u;
            const NnUint viewLenReq = (cfg != nullptr && cfg->view.sizeX != 0u) ? cfg->view.sizeX : context->inputSize.x;
            const NnUint viewStride = (cfg != nullptr && cfg->view.strideX != 0u) ? cfg->view.strideX : 1u;
            const NnUint invRmsBuf = (cfg != nullptr) ? cfg->invRmsBufferIndex : 0u;
            const NnUint nodeIndex = device->getNodeIndex();
            const unsigned int epoch = device->getPlanEpoch();
            const NnUint layerIndex = (segmentConfig != nullptr) ? segmentConfig->ops[opIndex].index : 0u;

            // Default to a single-stage/full-dim view when no uneven partition plan is attached
            // (e.g. uniform build path), so debug output stays meaningful.
            NnUint stageIndex = 0u;
            NnUint stageRank = nodeIndex;
            NnUint dimStart = 0u;
            NnUint dimLen = context->inputSize.x;
            NnUint stageDimTotal = context->inputSize.x;
            if (const NnUnevenPartitionPlan *plan = device->getPartitionPlan()) {
                if (plan->dimSplit.starts && plan->dimSplit.lengths) {
                    dimStart = plan->dimSplit.starts[nodeIndex];
                    dimLen = plan->dimSplit.lengths[nodeIndex];
                }
                if (plan->nStages > 0) {
                    if (const NnStageConfig *st = findStageForNode(plan, nodeIndex)) {
                        stageIndex = st->stageIndex;
                        stageDimTotal = getSplitTotalForStage(&plan->dimSplit, st);
                        stageRank = findStageRank(st, nodeIndex);
                    }
                }
            }

            const NnUint pos = loadPosFromPipe(context, 0u);

            NnUint validCount = 0u;
            NnUint lastIndex = viewOffset;
            if (viewOffset < context->inputSize.x && viewStride > 0u) {
                NnUint at = viewOffset;
                for (NnUint i = 0u; i < viewLenReq; ++i) {
                    if (at >= context->inputSize.x) break;
                    lastIndex = at;
                    validCount += 1u;
                    if (at > std::numeric_limits<NnUint>::max() - viewStride) break;
                    at += viewStride;
                }
            }

            std::printf("[final-norm][pre] node=%u stage=%u rank=%u epoch=%u layer=%u pos=%u seg=%u op=%u name=%s inType=%u inX=%u batchSize=%u view={off=%u len=%u stride=%u valid=%u} coverage=[%u,%u) dimSplit=[%u,%u) stageDimTotal=%u invRmsBuf=%u\n",
                (unsigned)nodeIndex,
                (unsigned)stageIndex,
                (unsigned)stageRank,
                (unsigned)epoch,
                (unsigned)layerIndex,
                (unsigned)pos,
                (unsigned)segmentIndex,
                (unsigned)opIndex,
                context->name,
                (unsigned)context->inputSize.floatType,
                (unsigned)context->inputSize.x,
                (unsigned)batchSize,
                (unsigned)viewOffset,
                (unsigned)viewLenReq,
                (unsigned)viewStride,
                (unsigned)validCount,
                (unsigned)viewOffset,
                (unsigned)((validCount > 0u) ? (lastIndex + 1u) : viewOffset),
                (unsigned)dimStart,
                (unsigned)(dimStart + dimLen),
                (unsigned)stageDimTotal,
                (unsigned)invRmsBuf);

            if (context->inputSize.floatType == F_32 && context->input != nullptr && batchSize > 0u) {
                const float *in0 = (const float *)context->input[0];
                if (in0 != nullptr && validCount > 0u) {
                    float vMin = in0[viewOffset];
                    float vMax = in0[viewOffset];
                    double sum = 0.0;
                    double absSum = 0.0;
                    NnUint nNan = 0u;
                    NnUint nInf = 0u;

                    for (NnUint i = 0u, at = viewOffset; i < validCount; ++i, at += viewStride) {
                        const float v = in0[at];
                        if (std::isnan(v)) {
                            nNan += 1u;
                            continue;
                        }
                        if (!std::isfinite(v)) {
                            nInf += 1u;
                            continue;
                        }
                        if (v < vMin) vMin = v;
                        if (v > vMax) vMax = v;
                        sum += (double)v;
                        absSum += (double)std::fabs((double)v);
                    }

                    std::printf("[final-norm][pre][stats] node=%u layer=%u pos=%u count=%u min=%.7g max=%.7g mean=%.7g meanAbs=%.7g nan=%u inf=%u\n",
                        (unsigned)nodeIndex,
                        (unsigned)layerIndex,
                        (unsigned)pos,
                        (unsigned)validCount,
                        (double)vMin,
                        (double)vMax,
                        (double)(validCount > 0u ? (sum / (double)validCount) : 0.0),
                        (double)(validCount > 0u ? (absSum / (double)validCount) : 0.0),
                        (unsigned)nNan,
                        (unsigned)nInf);

                    const NnUint sample = finalNormInputTraceSampleCount();
                    if (sample > 0u) {
                        const NnUint headN = std::min(sample, validCount);
                        std::printf("[final-norm][pre][head] node=%u vals=", (unsigned)nodeIndex);
                        for (NnUint i = 0u, at = viewOffset; i < headN; ++i, at += viewStride) {
                            std::printf("%s%.7g", (i == 0u ? "[" : ","), (double)in0[at]);
                        }
                        std::printf("]\n");

                        const NnUint tailN = std::min(sample, validCount);
                        const NnUint tailStart = validCount - tailN;
                        std::printf("[final-norm][pre][tail] node=%u vals=", (unsigned)nodeIndex);
                        for (NnUint i = tailStart; i < validCount; ++i) {
                            const NnUint at = viewOffset + i * viewStride;
                            std::printf("%s%.7g", (i == tailStart ? "[" : ","), (double)in0[at]);
                        }
                        std::printf("]\n");
                    }
                } else {
                    std::printf("[final-norm][pre][warn] node=%u input pointer is null or empty\n", (unsigned)nodeIndex);
                }
            } else {
                std::printf("[final-norm][pre][warn] node=%u unsupported input type=%u (only F_32 value dump is enabled)\n",
                    (unsigned)nodeIndex,
                    (unsigned)context->inputSize.floatType);
            }
            std::fflush(stdout);
        }
    }

    // // Always-on: print KV buffer range -> layer mapping every 5 tokens at MHA.
    // if (threadIndex == 0u && device != nullptr && context->opCode == OP_MULTIHEAD_ATT) {
    //     auto nameHas = [&](const char *needle) -> bool {
    //         if (context->name == nullptr || needle == nullptr) return false;
    //         return std::strstr(context->name, needle) != nullptr;
    //     };
    //     if (nameHas("block_multihead_att")) {
    //         const auto *cfg = (const NnMultiHeadAttOpConfig *)context->opConfig;
    //         const NnUint pos = (cfg != nullptr) ? loadPosFromPipe(context, cfg->positionPipeIndex) : 0xFFFFFFFFu;
    //         if (cfg != nullptr && pos != 0xFFFFFFFFu && (pos % 5u) == 0u) {
    //             const NnUint nodeIndex = device->getNodeIndex();
    //             const unsigned int epoch = device->getPlanEpoch();
    //             const NnUint layerIndex = (segmentConfig != nullptr) ? segmentConfig->ops[opIndex].index : 0u;
    //             const NnUint kvStart = cfg->kvStart;
    //             const NnUint kvEnd = cfg->kvStart + cfg->kvDim0;
    //             const NnUint kBuf = cfg->keyCacheBufferIndex;
    //             const NnUint vBuf = cfg->valueCacheBufferIndex;
    //             const NnUint kCols = context->bufferConfigs[kBuf].size.x;
    //             const NnUint vCols = context->bufferConfigs[vBuf].size.x;
    //             printf("🧠 [kv-range-map] node=%u epoch=%u pos=%u layer=%u kBuf=%u/%u vBuf=%u/%u kRange=[%u,%u) vRange=[%u,%u)\n",
    //                 (unsigned)nodeIndex,
    //                 (unsigned)epoch,
    //                 (unsigned)pos,
    //                 (unsigned)layerIndex,
    //                 (unsigned)kBuf,
    //                 (unsigned)kCols,
    //                 (unsigned)vBuf,
    //                 (unsigned)vCols,
    //                 (unsigned)kvStart,
    //                 (unsigned)kvEnd,
    //                 (unsigned)kvStart,
    //                 (unsigned)kvEnd);
    //             std::fflush(stdout);
    //         }
    //     }
    // }

    if (threadIndex == 0u && device != nullptr && migrationOpTraceEnabled()) {
        static std::atomic<NnUint> tracePrinted{0u};
        const NnUint limit = migrationOpTraceLimit();
        const NnUint idx = tracePrinted.fetch_add(1u, std::memory_order_relaxed);
        if (limit == 0u || idx < limit) {
            const NnUint nodeIndex = device->getNodeIndex();
            const NnUint layerIndex = (segmentConfig != nullptr) ? segmentConfig->ops[opIndex].index : 0u;

            NnUint pos = 0xFFFFFFFFu;
            if (context->opCode == OP_MULTIHEAD_ATT) {
                const auto *cfg = (const NnMultiHeadAttOpConfig *)context->opConfig;
                if (cfg != nullptr) pos = loadPosFromPipe(context, cfg->positionPipeIndex);
            } else if (context->opCode == OP_ROPE) {
                const auto *cfg = (const NnRopeOpConfig *)context->opConfig;
                if (cfg != nullptr) pos = loadPosFromPipe(context, cfg->positionPipeIndex);
            } else if (context->opCode == OP_SHIFT) {
                const auto *cfg = (const NnShiftOpCodeConfig *)context->opConfig;
                if (cfg != nullptr) pos = loadPosFromPipe(context, cfg->indexPipeIndex);
            } else {
                // FFN ops do not carry a position pipe in op config; read global POS pipe.
                pos = loadPosFromPipe(context, 0u);
            }

            if (pos != 0xFFFFFFFFu) {
                lastPos.store(pos, std::memory_order_release);
                device->setLastPos(pos);
            } else {
                pos = lastPos.load(std::memory_order_acquire);
                if (pos == 0xFFFFFFFFu) {
                    pos = device->getLastPos();
                }
            }

            const bool isAttn = (context->opCode == OP_MULTIHEAD_ATT) && nameHas(context->name, "block_multihead_att");
            const bool isFfn = isFfnTraceOp(context);
            if (isAttn || isFfn) {
                const int posPrint = (pos == 0xFFFFFFFFu) ? -1 : (int)pos;
                const char *segRole = migrationSegmentRoleForTrace(segmentConfig, context);
                std::printf("🧪 [op-trace] node=%u seg=%u role=%s layer=%u pos=%d phase=%s op=%s code=%s inX=%u outX=%u\n",
                    (unsigned)nodeIndex,
                    (unsigned)segmentIndex,
                    segRole,
                    (unsigned)layerIndex,
                    posPrint,
                    isAttn ? "attn" : "ffn",
                    (context->name ? context->name : "unknown"),
                    opCodeToString(context->opCode),
                    (unsigned)context->inputSize.x,
                    (unsigned)context->outputSize.x);

                if (isAttn && pos != 0xFFFFFFFFu) {
                    const auto *cfg = (const NnMultiHeadAttOpConfig *)context->opConfig;
                    if (cfg != nullptr) {
                        const NnUint kvStride = (cfg->kvStride == 0u) ? cfg->kvDim0 : cfg->kvStride;
                        const NnUint headDim = cfg->headDim;
                        const NnUint col0 = cfg->kvStart;
                        const NnUint dimsEnv = migrationKvTraceDims();
                        const NnUint dims = (dimsEnv == 0u || headDim == 0u) ? 0u : std::min(headDim, dimsEnv);

                        std::printf("🧪 [op-trace][attn-kv] node=%u layer=%u pos=%u kvStart=%u kvDim0=%u kvStride=%u headDim=%u\n",
                            (unsigned)nodeIndex,
                            (unsigned)layerIndex,
                            (unsigned)pos,
                            (unsigned)cfg->kvStart,
                            (unsigned)cfg->kvDim0,
                            (unsigned)kvStride,
                            (unsigned)cfg->headDim);

                        if (dims > 0u && kvStride > 0u && (col0 + dims) <= kvStride) {
                            const float *keyCache = (const float *)context->buffers[cfg->keyCacheBufferIndex];
                            const float *valueCache = (const float *)context->buffers[cfg->valueCacheBufferIndex];
                            const float *k0 = keyCache + col0;
                            const float *v0 = valueCache + col0;
                            const float *kp = keyCache + pos * kvStride + col0;
                            const float *vp = valueCache + pos * kvStride + col0;

                            std::printf("🧪 [op-trace][attn-kv] ");
                            printTraceFloatPrefix("K[t0]", k0, dims);
                            std::printf(" ");
                            printTraceFloatPrefix("V[t0]", v0, dims);
                            std::printf("\n");

                            std::printf("🧪 [op-trace][attn-kv] ");
                            printTraceFloatPrefix("K[tpos]", kp, dims);
                            std::printf(" ");
                            printTraceFloatPrefix("V[tpos]", vp, dims);
                            std::printf("\n");
                        } else {
                            std::printf("🧪 [op-trace][attn-kv] skip sample dims=%u kvStride=%u col0=%u\n",
                                (unsigned)dims,
                                (unsigned)kvStride,
                                (unsigned)col0);
                        }
                    }
                }
                std::fflush(stdout);
            }
        }
    }

    // ------------------------------------------------------------------
    // Debug: KV compute / attention KV range tracing
    // Enable with: kvcache_debug=1
    // Legacy alias: DLLAMA_DEBUG_KV_RANGE=1
    // This prints the effective KV ranges used by:
    // - K/V projection (matmul_k/matmul_v)
    // - K RoPE (rope_k)
    // - KV cache writes (shift_k/shift_v)
    // - Attention KV reads (multihead_att)
    // ------------------------------------------------------------------
#if DLLAMA_DEBUG_ATTN
    const bool kvCacheDebug = (std::getenv("kvcache_debug") != nullptr) || (std::getenv("DLLAMA_DEBUG_KV_RANGE") != nullptr);
    if (threadIndex == 0u && device != nullptr && kvCacheDebug) {
        const NnUint nodeIndex = device->getNodeIndex();
        const unsigned int epoch = device->getPlanEpoch();
        const NnUint layerIndex = (segmentConfig != nullptr) ? segmentConfig->ops[opIndex].index : 0u;

        auto nameHas = [&](const char *needle) -> bool {
            if (context->name == nullptr || needle == nullptr) return false;
            return std::strstr(context->name, needle) != nullptr;
        };

        if (context->opCode == OP_MATMUL && (nameHas("block_matmul_k") || nameHas("block_matmul_v"))) {
            const auto *cfg = (const NnMatmulOpConfig *)context->opConfig;
            const NnUint outStart = cfg ? cfg->outStart : 0u;
            const NnUint outUnit = cfg ? cfg->outStartUnit : 0u;
            const NnUint outLenElems = context->outputSize.x;
            if (outUnit != 0u) {
                printf("🧠 [kv][matmul] node=%u epoch=%u layer=%u op=%u name=%s outStart=%u outLen=%u (kvHeadStart=%u kvHeadLen=%u) view=%u\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (unsigned)outStart,
                    (unsigned)outLenElems,
                    (unsigned)(outStart / outUnit),
                    (unsigned)(outLenElems / outUnit),
                    (unsigned)(cfg ? cfg->view : 0u));
            } else {
                printf("🧠 [kv][matmul] node=%u epoch=%u layer=%u op=%u name=%s outStart=%u outLen=%u view=%u\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (unsigned)outStart,
                    (unsigned)outLenElems,
                    (unsigned)(cfg ? cfg->view : 0u));
            }
            std::fflush(stdout);
        } else if (context->opCode == OP_ROPE && nameHas("block_rope_k")) {
            const auto *cfg = (const NnRopeOpConfig *)context->opConfig;
            const NnUint headDim = (cfg != nullptr) ? cfg->slice.headDim : 0u;
            const NnUint kvStart = (cfg != nullptr) ? cfg->slice.kvDimStart : 0u;
            const NnUint kvLen = (cfg != nullptr) ? cfg->slice.kvDim0 : 0u;
            if (headDim != 0u) {
                printf("🧠 [kv][rope] node=%u epoch=%u layer=%u op=%u name=%s kvStart=%u kvLen=%u (kvHeadStart=%u kvHeadLen=%u) ropeType=%u\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (unsigned)kvStart,
                    (unsigned)kvLen,
                    (unsigned)(kvStart / headDim),
                    (unsigned)(kvLen / headDim),
                    (unsigned)(cfg ? cfg->type : 0u));
            } else {
                printf("🧠 [kv][rope] node=%u epoch=%u layer=%u op=%u name=%s kvStart=%u kvLen=%u ropeType=%u\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (unsigned)kvStart,
                    (unsigned)kvLen,
                    (unsigned)(cfg ? cfg->type : 0u));
            }
            std::fflush(stdout);
        } else if (context->opCode == OP_SHIFT && (nameHas("block_shift_k") || nameHas("block_shift_v"))) {
            const auto *cfg = (const NnShiftOpCodeConfig *)context->opConfig;
            const NnUint pos = (cfg != nullptr) ? loadPosFromPipe(context, cfg->indexPipeIndex) : 0xFFFFFFFFu;
            if (cfg != nullptr && cfg->dstRowStride != 0u) {
                printf("🧠 [kv][shift] node=%u epoch=%u layer=%u op=%u name=%s dstColStart=%u dstRowStride=%u inLen=%u\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (unsigned)cfg->dstColStart,
                    (unsigned)cfg->dstRowStride,
                    (unsigned)context->inputSize.x);
                printf("🧠 [kv][shift] node=%u epoch=%u layer=%u op=%u name=%s kvCacheBase=%p pos=%u writeCols=[%u,%u) dstColStartUnit=%u (kvHeadStart=%u kvHeadLen=%u)\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (void *)context->output[0],
                    (unsigned)pos,
                    (unsigned)cfg->dstColStart,
                    (unsigned)(cfg->dstColStart + context->inputSize.x),
                    (unsigned)cfg->dstColStartUnit,
                    (unsigned)((cfg->dstColStartUnit != 0u) ? (cfg->dstColStart / cfg->dstColStartUnit) : 0u),
                    (unsigned)((cfg->dstColStartUnit != 0u) ? (context->inputSize.x / cfg->dstColStartUnit) : 0u));

                if (pos != 0xFFFFFFFFu) {
                    // Print the concrete destination address for batch0.
                    const float *indexes = (cfg != nullptr) ? (const float *)context->pipes[cfg->indexPipeIndex] : nullptr;
                    const NnUint index0 = (indexes != nullptr) ? (NnUint)indexes[0] : 0u;
                    const NnSize rowStrideBytes = getBytes(F_32, cfg->dstRowStride);
                    const NnSize colStartBytes = getBytes(F_32, cfg->dstColStart);
                    printf("🧠 [kv][shift] node=%u epoch=%u layer=%u op=%u name=%s batch0 writePtr=%p (row=%u)\n",
                        (unsigned)nodeIndex,
                        (unsigned)epoch,
                        (unsigned)layerIndex,
                        (unsigned)opIndex,
                        context->name,
                        (void *)(context->output[0] + index0 * rowStrideBytes + colStartBytes),
                        (unsigned)index0);
                }
            } else {
                printf("🧠 [kv][shift] node=%u epoch=%u layer=%u op=%u name=%s (packed) inLen=%u\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (unsigned)context->inputSize.x);
                printf("🧠 [kv][shift] node=%u epoch=%u layer=%u op=%u name=%s kvCacheBase=%p pos=%u writeCols=[%u,%u)\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (void *)context->output[0],
                    (unsigned)pos,
                    0u,
                    (unsigned)context->inputSize.x);
            }
            std::fflush(stdout);
        } else if (context->opCode == OP_MULTIHEAD_ATT && nameHas("block_multihead_att")) {
            const auto *cfg = (const NnMultiHeadAttOpConfig *)context->opConfig;
            const NnUint headDim = (cfg != nullptr) ? cfg->headDim : 0u;
            if (cfg != nullptr && headDim != 0u) {
                const NnUint pos = loadPosFromPipe(context, cfg->positionPipeIndex);
                printf("🧠 [kv][att] node=%u epoch=%u layer=%u op=%u name=%s kvStart=%u kvDim0=%u kvStride=%u (kvHeadStart=%u kvHeadLen=%u)\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (unsigned)cfg->kvStart,
                    (unsigned)cfg->kvDim0,
                    (unsigned)cfg->kvStride,
                    (unsigned)(cfg->kvStart / headDim),
                    (unsigned)(cfg->kvDim0 / headDim));
                printf("🧠 [kv][att] node=%u epoch=%u layer=%u op=%u name=%s qStart=%u qSliceD0=%u qStride=%u (qHeadStart=%u qHeadLen=%u)\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (unsigned)cfg->qStart,
                    (unsigned)cfg->qSliceD0,
                    (unsigned)cfg->qStride,
                    (unsigned)(cfg->qStart / headDim),
                    (unsigned)(cfg->qSliceD0 / headDim));

                float *keyCache = (float *)context->buffers[cfg->keyCacheBufferIndex];
                float *valueCache = (float *)context->buffers[cfg->valueCacheBufferIndex];
                float *att = (float *)context->buffers[cfg->attBufferIndex];
                float *query = (float *)context->buffers[cfg->queryBufferIndex];
                const NnUint qStride = (cfg->qStride == 0u) ? cfg->qSliceD0 : cfg->qStride;
                const NnUint kvStride = (cfg->kvStride == 0u) ? cfg->kvDim0 : cfg->kvStride;

                printf("🧠 [kv][att] node=%u epoch=%u layer=%u op=%u name=%s pos=%u queryBase=%p keyCacheBase=%p valueCacheBase=%p attBase=%p\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (unsigned)pos,
                    (void *)query,
                    (void *)keyCache,
                    (void *)valueCache,
                    (void *)att);
                // Print the first element address of the KV range for token0 and token=pos (batch0).
                if (pos != 0xFFFFFFFFu) {
                    printf("🧠 [kv][att] node=%u epoch=%u layer=%u op=%u K[row0,colStart]=%p V[row0,colStart]=%p\n",
                        (unsigned)nodeIndex,
                        (unsigned)epoch,
                        (unsigned)layerIndex,
                        (unsigned)opIndex,
                        (void *)(keyCache + 0u * kvStride + cfg->kvStart),
                        (void *)(valueCache + 0u * kvStride + cfg->kvStart));
                    printf("🧠 [kv][att] node=%u epoch=%u layer=%u op=%u K[rowPos,colStart]=%p V[rowPos,colStart]=%p\n",
                        (unsigned)nodeIndex,
                        (unsigned)epoch,
                        (unsigned)layerIndex,
                        (unsigned)opIndex,
                        (void *)(keyCache + pos * kvStride + cfg->kvStart),
                        (void *)(valueCache + pos * kvStride + cfg->kvStart));
                    printf("🧠 [kv][att] node=%u epoch=%u layer=%u op=%u q[batch0,head0,dim0]=%p\n",
                        (unsigned)nodeIndex,
                        (unsigned)epoch,
                        (unsigned)layerIndex,
                        (unsigned)opIndex,
                        (void *)(query + 0u * qStride + cfg->qStart));
                }

                // GQA sanity hint: which KV heads are required by this Q-head range?
                // For GQA, kvHead = floor(qHead / groupSize).
                if (cfg->nKvHeads != 0u && cfg->nHeads % cfg->nKvHeads == 0u) {
                    const NnUint gqaGroup = cfg->nHeads / cfg->nKvHeads;
                    if (gqaGroup != 0u) {
                        const NnUint qHeadStart = cfg->qStart / headDim;
                        const NnUint qHeadLen = cfg->qSliceD0 / headDim;
                        const NnUint kvHeadOwnedStart = cfg->kvStart / headDim;
                        const NnUint kvHeadOwnedLen = cfg->kvDim0 / headDim;

                        const NnUint needKv0 = qHeadStart / gqaGroup;
                        const NnUint qHeadEnd = qHeadStart + qHeadLen;
                        const NnUint needKv1 = (qHeadEnd + gqaGroup - 1u) / gqaGroup; // exclusive

                        const NnUint ownedKv1 = kvHeadOwnedStart + kvHeadOwnedLen;
                        if (needKv0 < kvHeadOwnedStart || needKv1 > ownedKv1) {
                            printf("🧠 [kv][att][warn] node=%u epoch=%u layer=%u op=%u GQA group=%u qHeads=[%u,%u) requires kvHeads=[%u,%u) but owned kvHeads=[%u,%u)\n",
                                (unsigned)nodeIndex,
                                (unsigned)epoch,
                                (unsigned)layerIndex,
                                (unsigned)opIndex,
                                (unsigned)gqaGroup,
                                (unsigned)qHeadStart,
                                (unsigned)qHeadEnd,
                                (unsigned)needKv0,
                                (unsigned)needKv1,
                                (unsigned)kvHeadOwnedStart,
                                (unsigned)ownedKv1);
                        }
                    }
                }
            } else if (cfg != nullptr) {
                printf("🧠 [kv][att] node=%u epoch=%u layer=%u op=%u name=%s kvStart=%u kvDim0=%u kvStride=%u\n",
                    (unsigned)nodeIndex,
                    (unsigned)epoch,
                    (unsigned)layerIndex,
                    (unsigned)opIndex,
                    context->name,
                    (unsigned)cfg->kvStart,
                    (unsigned)cfg->kvDim0,
                    (unsigned)cfg->kvStride);
            }
            std::fflush(stdout);
        }
    }
#endif

    // // ------------------------------------------------------------------
    // // Targeted TP-range logging at a specific (pos, layer).
    // // Uses per-op config fields (start/len/view) to prove that online repartition affects compute.
    // // ------------------------------------------------------------------
    // if (threadIndex == 0u && device != nullptr && tpRangeTargetEnabled()) {
    //     const NnUint nodeIndex = device->getNodeIndex();
    //     const unsigned int epoch = device->getPlanEpoch();
    //     const NnUint layerIndex = (segmentConfig != nullptr) ? segmentConfig->ops[opIndex].index : 0u;

    //     // Stage metadata (if a partition plan exists): stage index and stage-local rank.
    //     NnUint stageIndex = 0xFFFFFFFFu;
    //     NnUint stageRank = 0xFFFFFFFFu;
    //     if (const NnUnevenPartitionPlan *plan = device->getPartitionPlan()) {
    //         if (const NnStageConfig *st = (plan->nStages > 0) ? findStageForNode(plan, nodeIndex) : nullptr) {
    //             stageIndex = st->stageIndex;
    //             for (NnUint i = 0; i < st->nNodes; ++i) {
    //                 if (st->nodeIndices[i] == nodeIndex) {
    //                     stageRank = i;
    //                     break;
    //                 }
    //             }
    //         }
    //     }

    //     // Try to determine pos for this op.
    //     NnUint pos = 0xFFFFFFFFu;
    //     if (context->opCode == OP_MULTIHEAD_ATT) {
    //         const auto *cfg = (const NnMultiHeadAttOpConfig *)context->opConfig;
    //         pos = loadPosFromPipe(context, cfg->positionPipeIndex);
    //     } else if (context->opCode == OP_ROPE) {
    //         const auto *cfg = (const NnRopeOpConfig *)context->opConfig;
    //         pos = loadPosFromPipe(context, cfg->positionPipeIndex);
    //     } else if (context->opCode == OP_SHIFT) {
    //         const auto *cfg = (const NnShiftOpCodeConfig *)context->opConfig;
    //         pos = loadPosFromPipe(context, cfg->indexPipeIndex);
    //     }

    //     if (pos != 0xFFFFFFFFu) {
    //         lastPos.store(pos, std::memory_order_release);
    //         if (device != nullptr) {
    //             device->setLastPos(pos);
    //         }
    //     } else {
    //         pos = lastPos.load(std::memory_order_acquire);
    //         if (pos == 0xFFFFFFFFu && device != nullptr) {
    //             pos = device->getLastPos();
    //         }
    //     }

    //     const bool isTpSensitive = (context->opCode == OP_MULTIHEAD_ATT) || (context->opCode == OP_MATMUL) ||
    //         (context->opCode == OP_SHIFT) || (context->opCode == OP_ROPE) || (context->opCode == OP_INV_RMS) || (context->opCode == OP_RMS_NORM) ||
    //         (context->opCode == OP_MUL) || (context->opCode == OP_SILU) || (context->opCode == OP_CAST);

    //     if (isTpSensitive && pos != 0xFFFFFFFFu && tpRangeTargetMatch(layerIndex, pos)) {
    //         if (opIndex < tpRangePrintedEpoch.size() && tpRangePrintedEpoch[opIndex] != epoch) {
    //             tpRangePrintedEpoch[opIndex] = epoch;
    //             NnUint ffnStart = 0xFFFFFFFFu;
    //             NnUint ffnLen = 0xFFFFFFFFu;
    //             if (const NnUnevenPartitionPlan *plan = device->getPartitionPlan()) {
    //                 if (plan->ffnSplit.starts && plan->ffnSplit.lengths) {
    //                     ffnStart = plan->ffnSplit.starts[nodeIndex];
    //                     ffnLen = plan->ffnSplit.lengths[nodeIndex];
    //                 }
    //             }
    //             printTpAffectedRange(context, segmentConfig, opIndex, layerIndex, pos, nodeIndex, epoch, stageIndex, stageRank, ffnStart, ffnLen);
    //             std::fflush(stdout);
    //         }
    //     }
    // }

#if DLLAMA_DEBUG_ATTN
    // Forward-time slice logging (prints once per op per segment).
    // This validates that each device/node sees the expected start/len/stride AND that
    // the resolved base pointers match the intended physical slots.
    if (threadIndex == 0u && debugSliceParamsForwardEnabled(context->name)) {
        const bool repeat = debugSliceParamsForwardRepeat();
        if (repeat || (opIndex < sliceFwdPrintedOnce.size() && sliceFwdPrintedOnce[opIndex] == 0u)) {
            if (!repeat && opIndex < sliceFwdPrintedOnce.size())
                sliceFwdPrintedOnce[opIndex] = 1u;
            const NnUint nodeIndex = (device != nullptr) ? device->getNodeIndex() : 0u;
            printf("🧩 [slice][fwd] node=%u seg=%u op=%u name=%s code=%s inX=%u outX=%u\n",
                (unsigned)nodeIndex,
                (unsigned)this->segmentIndex,
                (unsigned)opIndex,
                (context->name ? context->name : "Unknown"),
                opCodeToString(context->opCode),
                (unsigned)context->inputSize.x,
                (unsigned)context->outputSize.x);

            // Print opConfig-derived ranges (same helper as build-time).
            // Note: pointerConfig tags aren't directly available here; opConfig fields are the authority.
            printOpSliceParamsDbg(context->name, context->opCode, context->opConfig, nullptr, nullptr, context->inputSize, context->outputSize);
            // Print pointer samples to confirm physical slot alignment.
            printPtrSampleDbg("in", context->input, context->inputSize);
            printPtrSampleDbg("out", context->output, context->outputSize);
            std::fflush(stdout);
        }
    }

    if (threadIndex == 0u && context->weightSize.nBytes > 0 && context->weightReadPrinted == 0u && debugWeightRangesEnabled(context->name)) {
        context->weightReadPrinted = 1u;
        printf("📖 [weights][read-approx] op=%s idx=%u code=%s mayRead=[0,%zu) (matmul has detailed per-call ranges)\n",
            (context->name ? context->name : "Unknown"),
            (unsigned)opIndex,
            opCodeToString(context->opCode),
            (size_t)context->weightSize.nBytes);
        if (context->weightLoadCalls > 0u) {
            const NnSize min = (context->weightLoadedMin == std::numeric_limits<NnSize>::max()) ? 0u : context->weightLoadedMin;
            const NnSize max = context->weightLoadedMax;
            printf("📦 [weights][loaded] op=%s idx=%u loadedUnion=[%zu,%zu) loadedBytesSum=%zu calls=%u\n",
                (context->name ? context->name : "Unknown"),
                (unsigned)opIndex,
                (size_t)min,
                (size_t)max,
                (size_t)context->weightLoadedBytes,
                (unsigned)context->weightLoadCalls);
        }
        std::fflush(stdout);
    }
#endif

    static std::atomic<NnUint> residualChainTracePrinted{0u};
    const bool residualChainCandidate =
        (threadIndex == 0u && device != nullptr && residualChainTraceEnabled() && context->name != nullptr &&
         (std::strcmp(context->name, "block_cast_d") == 0 || std::strcmp(context->name, "block_merge_add2") == 0));

    bool residualChainTraceThisOp = false;
    bool residualChainIsMergeOp = false;
    NnUint residualChainNode = 0u;
    unsigned int residualChainEpoch = 0u;
    NnUint residualChainLayer = 0u;
    NnUint residualChainPos = 0xFFFFFFFFu;
    const char *residualRole = "unknown";

    if (residualChainCandidate) {
        residualChainNode = device->getNodeIndex();
        residualChainEpoch = device->getPlanEpoch();
        residualChainLayer = (segmentConfig != nullptr) ? segmentConfig->ops[opIndex].index : 0u;
        residualChainPos = loadPosFromPipe(context, 0u);
        residualRole = migrationSegmentRoleForTrace(segmentConfig, context);
        const int posFilter = residualChainTracePosFilter();
        const int layerFilter = residualChainTraceLayerFilter();
        const bool passPos = (posFilter < 0) || ((int)residualChainPos == posFilter);
        const bool passLayer = (layerFilter < 0) || ((int)residualChainLayer == layerFilter);
        if (passPos && passLayer) {
            const NnUint idx = residualChainTracePrinted.fetch_add(1u, std::memory_order_relaxed);
            const NnUint limit = residualChainTraceLimit();
            residualChainTraceThisOp = (limit == 0u || idx < limit);
        }

        residualChainIsMergeOp = (std::strcmp(context->name, "block_merge_add2") == 0);
        if (residualChainTraceThisOp && residualChainIsMergeOp && batchSize > 0u) {
            const NnUint sampleCount = residualChainTraceSamples();
            if (context->output != nullptr) {
                printResidualChainStats(
                    "merge-pre-out",
                    context->output[0],
                    context->outputSize.floatType,
                    context->outputSize.x,
                    context->name,
                    residualRole,
                    residualChainNode,
                    residualChainEpoch,
                    residualChainLayer,
                    residualChainPos,
                    0u,
                    1u,
                    sampleCount);
            }

            NnUint nSlices = 1u;
            if (context->outputSize.x > 0u && context->inputSize.x >= context->outputSize.x) {
                if ((context->inputSize.x % context->outputSize.x) == 0u) {
                    nSlices = context->inputSize.x / context->outputSize.x;
                }
            }

            NnUint slicesToPrint = nSlices;
            const NnUint sliceLimit = residualChainTraceSliceLimit();
            if (sliceLimit > 0u && sliceLimit < slicesToPrint) {
                slicesToPrint = sliceLimit;
            }

            const NnByte *in0 = (context->input != nullptr) ? context->input[0] : nullptr;
            if (in0 != nullptr) {
                for (NnUint sliceIndex = 0u; sliceIndex < slicesToPrint; ++sliceIndex) {
                    const NnUint sliceStart = sliceIndex * context->outputSize.x;
                    const NnSize offBytes = typedOffsetBytesNoAssert(context->inputSize.floatType, sliceStart);
                    if (offBytes == (NnSize)-1) {
                        std::printf("[res-chain] phase=merge-in-slice op=%s role=%s node=%u epoch=%u layer=%u pos=%u type=%s x=%u slice=%u/%u unsupported_offset=1\n",
                            context->name,
                            residualRole,
                            (unsigned)residualChainNode,
                            (unsigned)residualChainEpoch,
                            (unsigned)residualChainLayer,
                            (unsigned)residualChainPos,
                            floatTypeToString(context->inputSize.floatType),
                            (unsigned)context->outputSize.x,
                            (unsigned)(sliceIndex + 1u),
                            (unsigned)nSlices);
                        continue;
                    }

                    printResidualChainStats(
                        "merge-in-slice",
                        in0 + offBytes,
                        context->inputSize.floatType,
                        context->outputSize.x,
                        context->name,
                        residualRole,
                        residualChainNode,
                        residualChainEpoch,
                        residualChainLayer,
                        residualChainPos,
                        sliceIndex + 1u,
                        nSlices,
                        sampleCount);
                }
            }
            std::fflush(stdout);
        }
    }

    const bool traceFinalLogitsSlice =
        (threadIndex == 0u && device != nullptr && finalLogitsSliceTraceEnabled() &&
         context->name != nullptr &&
         (std::strstr(context->name, "final_matmul_logits") != nullptr ||
          std::strstr(context->name, "final_cast_logits") != nullptr));

    opForward[opIndex](nThreads, threadIndex, batchSize, context);

    if (residualChainTraceThisOp && batchSize > 0u) {
        const NnUint sampleCount = residualChainTraceSamples();
        if (std::strcmp(context->name, "block_cast_d") == 0) {
            if (context->output != nullptr) {
                printResidualChainStats(
                    "cast-out",
                    context->output[0],
                    context->outputSize.floatType,
                    context->outputSize.x,
                    context->name,
                    residualRole,
                    residualChainNode,
                    residualChainEpoch,
                    residualChainLayer,
                    residualChainPos,
                    1u,
                    1u,
                    sampleCount);
                std::fflush(stdout);
            }
        } else if (residualChainIsMergeOp) {
            if (context->output != nullptr) {
                printResidualChainStats(
                    "merge-post-out",
                    context->output[0],
                    context->outputSize.floatType,
                    context->outputSize.x,
                    context->name,
                    residualRole,
                    residualChainNode,
                    residualChainEpoch,
                    residualChainLayer,
                    residualChainPos,
                    1u,
                    1u,
                    sampleCount);
                std::fflush(stdout);
            }
        }
    }

    if (traceFinalLogitsSlice) {
        static std::atomic<NnUint> finalLogitsPrinted{0u};
        const NnUint limit = finalLogitsSliceTraceLimit();
        const NnUint idx = finalLogitsPrinted.fetch_add(1u, std::memory_order_relaxed);
        if (limit == 0u || idx < limit) {
            const NnUint nodeIndex = device->getNodeIndex();
            const NnUint layerIndex = (segmentConfig != nullptr) ? segmentConfig->ops[opIndex].index : 0u;
            const unsigned int epoch = device->getPlanEpoch();
            const NnUint pos = loadPosFromPipe(context, 0u);

            NnUint vocabStart = 0u;
            NnUint vocabLen = context->outputSize.x;
            if (const NnUnevenPartitionPlan *plan = device->getPartitionPlan()) {
                if (plan->vocabSplit.starts && plan->vocabSplit.lengths) {
                    vocabStart = plan->vocabSplit.starts[nodeIndex];
                    vocabLen = plan->vocabSplit.lengths[nodeIndex];
                }
            }

            if (context->outputSize.floatType == F_32 && context->output != nullptr && batchSize > 0u) {
                const float *out0 = (const float *)context->output[0];
                const NnUint n = context->outputSize.x;
                if (out0 != nullptr && n > 0u) {
                    float vMin = out0[0];
                    float vMax = out0[0];
                    NnUint vMaxIdx = 0u;
                    NnUint zeroCount = 0u;
                    NnUint nNan = 0u;
                    NnUint nInf = 0u;
                    double absSum = 0.0;

                    for (NnUint i = 0u; i < n; ++i) {
                        const float v = out0[i];
                        if (std::isnan(v)) { nNan += 1u; continue; }
                        if (!std::isfinite(v)) { nInf += 1u; continue; }
                        if (v == 0.0f) zeroCount += 1u;
                        if (v < vMin) vMin = v;
                        if (v > vMax) { vMax = v; vMaxIdx = i; }
                        absSum += std::fabs((double)v);
                    }

                    const NnUint globalMaxIdx = vocabStart + vMaxIdx;
                    std::printf("[final-logits-slice] op=%s node=%u epoch=%u layer=%u pos=%u outX=%u vocabSlice=[%u,%u) min=%.7g max=%.7g localMaxIdx=%u globalMaxIdx=%u meanAbs=%.7g zero=%u/%u nan=%u inf=%u\n",
                        context->name,
                        (unsigned)nodeIndex,
                        (unsigned)epoch,
                        (unsigned)layerIndex,
                        (unsigned)pos,
                        (unsigned)context->outputSize.x,
                        (unsigned)vocabStart,
                        (unsigned)(vocabStart + vocabLen),
                        (double)vMin,
                        (double)vMax,
                        (unsigned)vMaxIdx,
                        (unsigned)globalMaxIdx,
                        (double)(n > 0u ? (absSum / (double)n) : 0.0),
                        (unsigned)zeroCount,
                        (unsigned)n,
                        (unsigned)nNan,
                        (unsigned)nInf);
                    std::fflush(stdout);
                }
            }
        }
    }

    if (threadIndex == 0u && device != nullptr && layerOutTraceEnabled() &&
        context->opCode == OP_MERGE_ADD && context->name != nullptr &&
        (std::strstr(context->name, "block_merge_add2") != nullptr || std::strstr(context->name, "final_merge_add") != nullptr) &&
        context->outputSize.floatType == F_32 && context->output != nullptr && batchSize > 0u) {

        static std::atomic<NnUint> layerOutPrinted{0u};
        const NnUint limit = layerOutTraceLimit();
        const NnUint idx = layerOutPrinted.fetch_add(1u, std::memory_order_relaxed);
        if (limit == 0u || idx < limit) {
            const NnUint pos = loadPosFromPipe(context, 0u);
            const NnUint layerIndex = (segmentConfig != nullptr) ? segmentConfig->ops[opIndex].index : 0u;
            const int posFilter = layerOutTracePosFilter();
            const int layerFilter = layerOutTraceLayerFilter();
            const bool passPos = (posFilter < 0) || ((int)pos == posFilter);
            const bool passLayer = (layerFilter < 0) || ((int)layerIndex == layerFilter);
            if (passPos && passLayer) {
                const float *out0 = (const float *)context->output[0];
                const NnUint n = context->outputSize.x;
                if (out0 != nullptr && n > 0u) {
                    float vMin = out0[0];
                    float vMax = out0[0];
                    NnUint nNan = 0u;
                    NnUint nInf = 0u;
                    NnUint nZero = 0u;
                    double meanAbs = 0.0;
                    for (NnUint i = 0u; i < n; ++i) {
                        const float v = out0[i];
                        if (v == 0.0f) nZero += 1u;
                        if (std::isnan(v)) { nNan += 1u; continue; }
                        if (!std::isfinite(v)) { nInf += 1u; continue; }
                        if (v < vMin) vMin = v;
                        if (v > vMax) vMax = v;
                        meanAbs += std::fabs((double)v);
                    }
                    std::printf("[layer-out] op=%s node=%u stageEpoch=%u layer=%u pos=%u outX=%u range=[%.7g, %.7g] meanAbs=%.7g zero=%u/%u nan=%u inf=%u\n",
                        context->name,
                        (unsigned)device->getNodeIndex(),
                        (unsigned)device->getPlanEpoch(),
                        (unsigned)layerIndex,
                        (unsigned)pos,
                        (unsigned)n,
                        (double)vMin,
                        (double)vMax,
                        (double)(n > 0u ? (meanAbs / (double)n) : 0.0),
                        (unsigned)nZero,
                        (unsigned)n,
                        (unsigned)nNan,
                        (unsigned)nInf);
                    const NnUint i0 = 0u;
                    const NnUint i1 = n / 3u;
                    const NnUint i2 = (2u * n) / 3u;
                    const NnUint i3 = n - 1u;
                    layerOutCompareRun(
                        context->name,
                        device->getNodeIndex(),
                        layerIndex,
                        pos,
                        n,
                        (double)vMin,
                        (double)vMax,
                        (double)(n > 0u ? (meanAbs / (double)n) : 0.0),
                        nZero,
                        nNan,
                        nInf,
                        (double)out0[i0],
                        (double)out0[i1],
                        (double)out0[i2],
                        (double)out0[i3]);
                    std::fflush(stdout);
                }
            }
        }
    }
}

void NnCpuDeviceSegment::refreshPointers() {
    assert(device != nullptr);
    assert(segmentConfig != nullptr);
    assert(segmentConfig->nOps == nOps);

    const NnUnevenPartitionPlan *plan = device->getPartitionPlan();
    const NnUint nodeIndex = device->getNodeIndex();

    auto getSplitStart = [&](NnSliceTag tag, NnUint *outStart) -> bool {
        if (plan == nullptr || outStart == nullptr) return false;
        switch (tag) {
            case NN_SLICE_VOCAB:
                if (plan->vocabSplit.starts) { *outStart = plan->vocabSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_FFN:
                if (plan->ffnSplit.starts) { *outStart = plan->ffnSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_DIM:
                if (plan->dimSplit.starts) { *outStart = plan->dimSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_HEAD:
                if (plan->headSplit.starts) { *outStart = plan->headSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_KV_HEAD:
                if (plan->kvHeadSplit.starts) { *outStart = plan->kvHeadSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_STACKED_BY_NODE:
            case NN_SLICE_AUTO:
            default:
                return false;
        }
    };

    auto findRmsNormColSize = [&](NnUint fromOpIndex, const NnSize3D &inputSizeNow) -> NnUint {
        // Heuristic: the corresponding RMS_NORM usually comes right after INV_RMS.
        for (NnUint j = fromOpIndex + 1u; j < nOps && j < fromOpIndex + 8u; ++j) {
            NnCpuOpContext *c = &opContexts[j];
            if (c->opCode != OP_RMS_NORM) continue;
            if (c->weightSize.x == 0u) continue;
            if (inputSizeNow.x % c->weightSize.x != 0u) continue;
            return c->weightSize.x;
        }
        return 0u;
    };

    for (NnUint opIndex = 0; opIndex < nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnCpuOpContext *opContext = &opContexts[opIndex];

        NnSize3D inputSize;
        NnSize3D outputSize;
        std::vector<NnByte *> inputsPtr = device->resolvePointer(&inputSize, &opConfig->input);
        std::vector<NnByte *> outputsPtr = device->resolvePointer(&outputSize, &opConfig->output);

        // Keep the same logits pipe size fix as createSegment().
        if (opConfig->code == OP_CAST &&
            opConfig->output.type == PNTR_BATCHED_SLICE &&
            inputSize.x != outputSize.x) {
            outputSize = size3D(outputSize.floatType, outputSize.z, outputSize.y, inputSize.x);
        }

        // Update pointers in-place (pointer counts are stable: z*y for batch/slice, 1 for raw).
        std::memcpy(opContext->input, inputsPtr.data(), inputsPtr.size() * sizeof(NnByte *));
        std::memcpy(opContext->output, outputsPtr.data(), outputsPtr.size() * sizeof(NnByte *));
        opContext->inputSize = inputSize;
        opContext->outputSize = outputSize;

        // Refresh KV-related op configs for online repartition (CPU-only).
        // We only touch configs when plan data is available.
        if (plan != nullptr) {
            if (opContext->opCode == OP_MULTIHEAD_ATT) {
                auto *cfg = (NnMultiHeadAttOpConfig *)opContext->opConfig;
                const bool fullAttBuffer =
                    opConfig->input.type == PNTR_BATCHED_SLICE ||
                    opConfig->output.type == PNTR_BATCHED_SLICE;
                if (plan->headSplit.starts && plan->headSplit.lengths) {
                    cfg->nHeads0 = plan->headSplit.lengths[nodeIndex];
                    cfg->qSliceD0 = cfg->nHeads0 * cfg->headDim;
                    if (fullAttBuffer) {
                        cfg->qStart = plan->headSplit.starts[nodeIndex] * cfg->headDim;
                        cfg->qStride = cfg->nHeads * cfg->headDim;
                    } else {
                        cfg->qStart = 0u;
                        cfg->qStride = cfg->qSliceD0;
                    }
                }
                if (plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
                    const NnUint kvHeads0 = plan->kvHeadSplit.lengths[nodeIndex];
                    cfg->kvDim0 = kvHeads0 * cfg->headDim;
                    if (fullAttBuffer) {
                        cfg->kvStart = plan->kvHeadSplit.starts[nodeIndex] * cfg->headDim;
                        cfg->kvStride = cfg->nKvHeads * cfg->headDim;
                    } else {
                        cfg->kvStart = 0u;
                        cfg->kvStride = cfg->kvDim0;
                    }
                }
            } else if (opContext->opCode == OP_SHIFT) {
                auto *cfg = (NnShiftOpCodeConfig *)opContext->opConfig;
                // Only update when this SHIFT is using strided destination mode (KV cache full-buffer mode).
                if (cfg->dstRowStride != 0u && cfg->dstColStartUnit != 0u &&
                    plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
                    cfg->dstColStart = plan->kvHeadSplit.starts[nodeIndex] * cfg->dstColStartUnit;
                }
            } else if (opContext->opCode == OP_MATMUL) {
                auto *cfg = (NnMatmulOpConfig *)opContext->opConfig;
                if (cfg->view == 1u && cfg->outSliceTag != NN_SLICE_AUTO && cfg->outStartUnit != 0u) {
                    NnUint s = 0u;
                    if (getSplitStart(cfg->outSliceTag, &s)) {
                        cfg->outStart = s * cfg->outStartUnit;
                    }
                }
                if (cfg->view == 2u && cfg->inSliceTag != NN_SLICE_AUTO && cfg->inStartUnit != 0u) {
                    NnUint s = 0u;
                    if (getSplitStart(cfg->inSliceTag, &s)) {
                        cfg->inStart = s * cfg->inStartUnit;
                    }
                }

                // Keep A/C view logical sizes aligned with refreshed tensor sizes when views are used.
                if (cfg->aView.sizeX != 0u) {
                    cfg->aView.sizeX = inputSize.x;
                }
                if (cfg->cView.sizeX != 0u) {
                    cfg->cView.sizeX = outputSize.x;
                }
            } else if (opContext->opCode == OP_MUL) {
                auto *cfg = (NnMulOpCodeConfig *)opContext->opConfig;
                // When full FFN buffers are enabled, OP_MUL relies on view(offset/len)
                // to align the multiplier slice. Make it refreshable for online ffnSplit.
                if (plan->ffnSplit.starts && plan->ffnSplit.lengths && cfg->view.sizeX != 0u) {
                    cfg->view.offset = plan->ffnSplit.starts[nodeIndex];
                    cfg->view.sizeX = plan->ffnSplit.lengths[nodeIndex];
                    cfg->view.strideX = 1u;
                }
            } else if (opContext->opCode == OP_INV_RMS) {
                // QWEN3 Q/K pre-norm uses nColumns = nHeads0 (or nKvHeads0).
                // After head migration, recompute nColumns from current inputSize.x and the
                // per-column size (headDim), inferred from the subsequent RMS_NORM weight size.
                auto *cfg = (NnInvRmsOpConfig *)opContext->opConfig;
                const NnUint colSize = findRmsNormColSize(opIndex, inputSize);
                if (colSize != 0u && inputSize.x % colSize == 0u) {
                    const NnUint newCols = inputSize.x / colSize;
                    // Ensure output buffer can hold it.
                    if (outputSize.x >= newCols && newCols != 0u) {
                        cfg->nColumns = newCols;
                    }
                }
            } else if (opContext->opCode == OP_RMS_NORM) {
                // Recompute nColumns when weights are per-head (colSize = headDim).
                auto *cfg = (NnRmsNormOpConfig *)opContext->opConfig;
                const NnUint colSize = opContext->weightSize.x;
                if (colSize != 0u && inputSize.x % colSize == 0u) {
                    const NnUint newCols = inputSize.x / colSize;
                    // invRms buffer must have at least newCols columns; it is allocated with a safe upper bound.
                    if (newCols != 0u) {
                        cfg->nColumns = newCols;
                    }
                }
            } else if (opContext->opCode == OP_ROPE) {
                // RoPE config caches per-node slice dimensions (qDim0/kvDim0).
                // With online head migration and PNTR_BATCHED_SLICE, the local slice len changes.
                // Update dim0 so Falcon/Llama assertions stay valid.
                auto *cfg = (NnRopeOpConfig *)opContext->opConfig;
                if (cfg->isQ == 1u) {
                    cfg->slice.qDim0 = inputSize.x;
                } else {
                    cfg->slice.kvDim0 = inputSize.x;
                }
            }
        }

    #if DLLAMA_DEBUG_ATTN
        // Debug: after refresh, print updated params.
        printOpSliceParamsDbg(opContext->name, opContext->opCode, opContext->opConfig, &opConfig->input, &opConfig->output, opContext->inputSize, opContext->outputSize);
    #endif
    }

    // If we changed pointers/configs, allow forward-time logs to print again on next run.
    if (!sliceFwdPrintedOnce.empty()) {
        std::fill(sliceFwdPrintedOnce.begin(), sliceFwdPrintedOnce.end(), 0u);
    }

    // Mark this segment as up-to-date with current plan epoch.
    if (device != nullptr) {
        planEpochReady.store(device->getPlanEpoch(), std::memory_order_release);
    }
}
