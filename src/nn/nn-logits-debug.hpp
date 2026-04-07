#ifndef NN_LOGITS_DEBUG_HPP
#define NN_LOGITS_DEBUG_HPP

#include "nn-core.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static inline bool lgDebugParseBoolEnv(const char *name) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return false;
    if ((std::strcmp(v, "0") == 0) ||
        (std::strcmp(v, "false") == 0) ||
        (std::strcmp(v, "FALSE") == 0) ||
        (std::strcmp(v, "off") == 0) ||
        (std::strcmp(v, "OFF") == 0)) {
        return false;
    }
    return true;
}

static inline bool lgDebugEnabled() {
    return lgDebugParseBoolEnv("DLLAMA_DEBUG_LG");
}

static inline bool lgDebugParseIntFilter(const char *spec, int value) {
    if (spec == nullptr || spec[0] == '\0') return true;
    const char *p = spec;
    while (*p != '\0') {
        while (*p == ',' || std::isspace((unsigned char)*p)) ++p;
        if (*p == '\0') break;

        char *end = nullptr;
        long lo = std::strtol(p, &end, 10);
        if (end == p) break;
        long hi = lo;
        p = end;
        if (*p == '-') {
            ++p;
            hi = std::strtol(p, &end, 10);
            if (end == p) hi = lo;
            p = end;
        }
        if ((long)value >= lo && (long)value <= hi) return true;
        while (*p == ',' || std::isspace((unsigned char)*p)) ++p;
    }
    return false;
}

static inline bool lgDebugShouldLogPos(int pos) {
    if (!lgDebugEnabled()) return false;
    if (pos < 0) return lgDebugParseIntFilter(std::getenv("DLLAMA_DEBUG_LG_POS"), -1);
    return lgDebugParseIntFilter(std::getenv("DLLAMA_DEBUG_LG_POS"), pos);
}

static inline bool lgDebugShouldLogBatch(int batch) {
    if (!lgDebugEnabled()) return false;
    return lgDebugParseIntFilter(std::getenv("DLLAMA_DEBUG_LG_BATCH"), batch);
}

static inline bool lgDebugShouldLog(int pos, int batch) {
    if (!lgDebugEnabled()) return false;
    return lgDebugShouldLogPos(pos) && lgDebugShouldLogBatch(batch);
}

static inline int lgDebugFindPipeIndexByName(const NnPipeConfig *pipeConfigs, NnUint nPipes, const char *name) {
    if (pipeConfigs == nullptr || name == nullptr) return -1;
    for (NnUint i = 0; i < nPipes; ++i) {
        const char *pipeName = pipeConfigs[i].name;
        if (pipeName != nullptr && std::strcmp(pipeName, name) == 0) return (int)i;
    }
    return -1;
}

static inline int lgDebugLoadPosFromPipes(
    NnByte * const *pipes,
    const NnPipeConfig *pipeConfigs,
    NnUint nPipes,
    NnUint batchIndex) {
    if (pipes == nullptr || pipeConfigs == nullptr) return -1;
    const int posPipeIndex = lgDebugFindPipeIndexByName(pipeConfigs, nPipes, "POS");
    if (posPipeIndex < 0) return -1;
    const NnPipeConfig *posCfg = &pipeConfigs[(NnUint)posPipeIndex];
    if (posCfg->size.floatType != F_32) return -1;
    if (batchIndex >= posCfg->size.y) return -1;
    const NnSize posBatchBytes = getBytes(posCfg->size.floatType, posCfg->size.x);
    float posF = -1.0f;
    std::memcpy(&posF, &pipes[(NnUint)posPipeIndex][(NnSize)batchIndex * posBatchBytes], sizeof(float));
    if (!(posF >= 0.0f)) return -1;
    return (int)posF;
}

static inline NnUint lgDebugLgPipeVocabSize(const NnPipeConfig *pipeConfigs, NnUint nPipes) {
    const int lgPipeIndex = lgDebugFindPipeIndexByName(pipeConfigs, nPipes, "LG");
    if (lgPipeIndex < 0) return 0u;
    return pipeConfigs[(NnUint)lgPipeIndex].size.x;
}

static inline bool lgDebugGetVocabShard(
    const NnUnevenPartitionPlan *plan,
    NnUint nodeIndex,
    NnUint *outStart,
    NnUint *outLen) {
    if (outStart) *outStart = 0u;
    if (outLen) *outLen = 0u;
    if (plan == nullptr || plan->vocabSplit.starts == nullptr || plan->vocabSplit.lengths == nullptr) return false;
    if (outStart) *outStart = plan->vocabSplit.starts[nodeIndex];
    if (outLen) *outLen = plan->vocabSplit.lengths[nodeIndex];
    return true;
}

static inline std::uint64_t lgDebugHashSlice256(const float *base, unsigned len) {
    if (base == nullptr || len == 0u) return 0ull;
    std::uint64_t h = 1469598103934665603ull;
    const unsigned sampleN = std::min(len, 256u);
    for (unsigned i = 0; i < sampleN; ++i) {
        std::uint32_t u = 0u;
        std::memcpy(&u, &base[i], sizeof(u));
        h ^= (std::uint64_t)u;
        h *= 1099511628211ull;
    }
    return h;
}

static inline float lgDebugSliceValueAt(const float *base, unsigned len, unsigned idx) {
    if (base == nullptr || len == 0u) return 0.0f;
    if (idx >= len) idx = len - 1u;
    return base[idx];
}

static inline void dumpLogitsSliceStats(
    const char* tag,
    int node,
    int pos,
    int batch,
    const float* base,
    unsigned globalOff,
    unsigned len,
    unsigned vocabSize
) {
    if (!base || len == 0u) {
        std::printf("[LGDBG] tag=%s node=%d pos=%d batch=%d off=%u len=%u vocab=%u EMPTY\n",
                    (tag != nullptr ? tag : "(null)"),
                    node, pos, batch, globalOff, len, vocabSize);
        return;
    }

    float mn = 1e30f;
    float mx = -1e30f;
    unsigned zero = 0u;
    int localMaxIdx = -1;
    for (unsigned i = 0; i < len; ++i) {
        const float v = base[i];
        if (v < mn) mn = v;
        if (v > mx) {
            mx = v;
            localMaxIdx = (int)i;
        }
        if (v == 0.0f) zero++;
    }

    const std::uint64_t h = lgDebugHashSlice256(base, len);
    std::printf(
        "[LGDBG] tag=%s node=%d pos=%d batch=%d off=%u len=%u vocab=%u range=[%.6f,%.6f] "
        "localMax=%d globalMax=%d zero=%u/%u hash256=0x%016llx "
        "head={%.6f,%.6f,%.6f,%.6f} tail={%.6f,%.6f,%.6f,%.6f}\n",
        (tag != nullptr ? tag : "(null)"),
        node,
        pos,
        batch,
        globalOff,
        len,
        vocabSize,
        mn,
        mx,
        localMaxIdx,
        localMaxIdx >= 0 ? (int)(globalOff + (unsigned)localMaxIdx) : -1,
        zero,
        len,
        (unsigned long long)h,
        lgDebugSliceValueAt(base, len, 0u),
        lgDebugSliceValueAt(base, len, 1u),
        lgDebugSliceValueAt(base, len, 2u),
        lgDebugSliceValueAt(base, len, 3u),
        lgDebugSliceValueAt(base, len, len > 4u ? len - 4u : 0u),
        lgDebugSliceValueAt(base, len, len > 3u ? len - 3u : 0u),
        lgDebugSliceValueAt(base, len, len > 2u ? len - 2u : 0u),
        lgDebugSliceValueAt(base, len, len > 1u ? len - 1u : 0u));
}

#endif
