#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#elif defined(__AVX2__) || defined(__AVX512F__)
    #include <immintrin.h>
#endif
#include "nn-cpu-ops.hpp"
#include "nn-quants.hpp"
#include "llamafile/sgemm.hpp"

#ifndef DLLAMA_DEBUG_ATTN
#define DLLAMA_DEBUG_ATTN 0
#endif

#if DLLAMA_DEBUG_ATTN

static inline bool kvCacheDebugEnabled() {
    // Primary switch requested by users.
    if (std::getenv("kvcache_debug") != nullptr) return true;
    // Legacy alias used by existing tracing in nn-cpu.cpp.
    if (std::getenv("DLLAMA_DEBUG_KV_RANGE") != nullptr) return true;
    return false;
}

static inline bool kvCachePerHeadEnabled() {
    const char *v = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD");
    if (v == nullptr) return false;
    return std::atoi(v) != 0;
}

static inline bool kvCachePerHeadPassesFilter(const char *opName) {
    const char *filter = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD_FILTER");
    if (filter == nullptr || filter[0] == '\0') return true;
    if (opName == nullptr) return false;
    return std::strstr(opName, filter) != nullptr;
}

static inline NnUint kvCachePerHeadLimit() {
    // 0 means unlimited.
    const char *v = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD_LIMIT");
    if (v == nullptr) return 32u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline NnUint kvCachePerHeadBatch() {
    const char *v = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD_BATCH");
    if (v == nullptr) return 0u;
    const long n = std::strtol(v, nullptr, 10);
    if (n < 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline long kvCachePerHeadSelectKvHead() {
    // >=0: only print that global kvHead; -1: print all owned kvHeads.
    const char *v = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD_KVHEAD");
    if (v == nullptr) return -1;
    return std::strtol(v, nullptr, 10);
}

static inline const char *kvCachePerHeadSelectKvHeadsSpec() {
    // Optional: comma-separated list / ranges of global kvHeads to print, e.g. "4,5" or "4-7".
    // When set, this overrides DLLAMA_DEBUG_KVCACHE_PER_HEAD_KVHEAD.
    const char *v = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD_KVHEADS");
    if (v == nullptr || v[0] == '\0') return nullptr;
    return v;
}

static inline bool kvCachePerHeadGlobalEnabled() {
    // When enabled, printing in MHA uses global kvHead numbering directly (0..nKvHeads),
    // and can print heads outside the owned kvStart/kvDim0 slice (e.g. redundant compute heads),
    // as long as the KV cache buffers are allocated as full global tensors.
    const char *v = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD_GLOBAL");
    if (v == nullptr) return false;
    return std::atoi(v) != 0;
}

static inline bool kvHeadMatchesSelection(NnUint globalKvHead, const char *spec, long singleSel) {
    if (spec == nullptr || spec[0] == '\0') {
        return (singleSel < 0) ? true : ((NnUint)singleSel == globalKvHead);
    }

    // Parse tokens like: "4", "4-7", separated by comma or whitespace.
    const char *p = spec;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',') ++p;
        if (*p == '\0') break;

        char *end = nullptr;
        long a = std::strtol(p, &end, 10);
        if (end == p) {
            // Skip invalid token.
            while (*p && *p != ',' && *p != ' ' && *p != '\t') ++p;
            continue;
        }
        p = end;

        long b = a;
        if (*p == '-') {
            ++p;
            char *end2 = nullptr;
            b = std::strtol(p, &end2, 10);
            if (end2 != p) {
                p = end2;
            } else {
                // Trailing '-', treat as single.
                b = a;
            }
        }

        if (a < 0) a = 0;
        if (b < 0) b = 0;
        if (a > b) std::swap(a, b);
        if ((long)globalKvHead >= a && (long)globalKvHead <= b) return true;

        while (*p && *p != ',' && *p != ' ' && *p != '\t') ++p;
    }
    return false;
}

static inline long kvCachePerHeadSelectPos() {
    // >=0: only print at that position; -1: print all positions.
    const char *v = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD_POS");
    if (v == nullptr) return -1;
    return std::strtol(v, nullptr, 10);
}

static inline NnUint kvCachePerHeadDims() {
    // Number of dims to print per head. 0 means "print full headDim".
    const char *v = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD_DIMS");
    if (v == nullptr) return 8u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 4096) return 4096u;
    return (NnUint)n;
}

static inline bool kvCachePerHeadShiftEnabled() {
    const char *v = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD_SHIFT");
    if (v == nullptr) return false;
    return std::atoi(v) != 0;
}

static inline NnUint kvCachePerHeadHeadDimOverride() {
    const char *v = std::getenv("DLLAMA_DEBUG_KVCACHE_PER_HEAD_HEADDIM");
    if (v == nullptr) return 0u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 4096) return 4096u;
    return (NnUint)n;
}

static inline bool attDebugEnabled() {
    const char *v = std::getenv("DLLAMA_DEBUG_ATT");
    if (v == nullptr) return false;
    return std::atoi(v) != 0;
}

static inline bool attDebugPassesFilter(const char *opName) {
    const char *filter = std::getenv("DLLAMA_DEBUG_ATT_FILTER");
    if (filter == nullptr || filter[0] == '\0') return true;
    if (opName == nullptr) return false;
    return std::strstr(opName, filter) != nullptr;
}

static inline NnUint attDebugLimit() {
    // 0 means unlimited.
    const char *v = std::getenv("DLLAMA_DEBUG_ATT_LIMIT");
    if (v == nullptr) return 0u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline NnUint attDebugBatch() {
    const char *v = std::getenv("DLLAMA_DEBUG_ATT_BATCH");
    if (v == nullptr) return 0u;
    const long n = std::strtol(v, nullptr, 10);
    if (n < 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline long attDebugHead() {
    // >=0: print only that local q-head; -1: print all local q-heads.
    const char *v = std::getenv("DLLAMA_DEBUG_ATT_HEAD");
    if (v == nullptr) return 0;
    return std::strtol(v, nullptr, 10);
}

static inline bool attQkEnabled() {
    // Print attention inputs Q/K and dot(q,k) for the selected head.
    // Enable with: DLLAMA_DEBUG_ATT_QK=1
    const char *v = std::getenv("DLLAMA_DEBUG_ATT_QK");
    if (v == nullptr) return false;
    return std::atoi(v) != 0;
}

static inline NnUint attQkDims() {
    // Number of dims to print from Q/K vectors.
    // 0 means full headDim (capped).
    const char *v = std::getenv("DLLAMA_DEBUG_ATT_QK_DIMS");
    if (v == nullptr) return 16u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 4096) return 4096u;
    return (NnUint)n;
}

static inline NnUint attQkTopK() {
    // Print K vectors for the top-K attention weights (after softmax).
    const char *v = std::getenv("DLLAMA_DEBUG_ATT_QK_TOPK");
    if (v == nullptr) return 5u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 256) return 256u;
    return (NnUint)n;
}

static inline long attQkForcePos() {
    // If set to >=0, print only this position index (within [0,len)).
    const char *v = std::getenv("DLLAMA_DEBUG_ATT_QK_POS");
    if (v == nullptr) return -1;
    return std::strtol(v, nullptr, 10);
}

static inline bool attScoresEnabled() {
    // Print per-head attention scores/weights (the softmax row) as a list.
    // Enable with: DLLAMA_DEBUG_ATT_SCORES=1
    const char *v = std::getenv("DLLAMA_DEBUG_ATT_SCORES");
    if (v == nullptr) return false;
    return std::atoi(v) != 0;
}

static inline NnUint attScoresMaxLen() {
    // Max number of positions to print from the attention row.
    // 0 means print full length (capped).
    const char *v = std::getenv("DLLAMA_DEBUG_ATT_SCORES_MAXLEN");
    if (v == nullptr) return 64u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 8192) return 8192u;
    return (NnUint)n;
}

static void printAttRowValues(const float *attRow, NnUint len, NnUint maxLen) {
    if (attRow == nullptr || len == 0u) return;
    const NnUint cap = 8192u;
    const NnUint n = (maxLen == 0u) ? std::min(len, cap) : std::min(len, std::min(maxLen, cap));
    printf("🧪 [att][scores] len=%u printing=%u:", (unsigned)len, (unsigned)n);
    for (NnUint i = 0u; i < n; ++i) {
        printf(" %u:%.6f", (unsigned)i, (double)attRow[i]);
    }
    if (n < len) printf(" ...");
    printf("\n");
}

static void selectTopKIndices(const float *row, NnUint len, NnUint k, std::vector<NnUint> &outIdx) {
    outIdx.clear();
    if (row == nullptr || len == 0u || k == 0u) return;
    k = std::min(k, len);
    struct Item { float v; NnUint i; };
    std::vector<Item> items;
    items.reserve(len);
    for (NnUint i = 0u; i < len; ++i) items.push_back(Item{row[i], i});
    const auto cmp = [](const Item &a, const Item &b) { return a.v > b.v; };
    if (len > k) {
        std::nth_element(items.begin(), items.begin() + k, items.end(), cmp);
        items.resize(k);
    }
    std::sort(items.begin(), items.end(), cmp);
    outIdx.reserve(k);
    for (const auto &it : items) outIdx.push_back(it.i);
}

static void printVecDims(const char *tag, const float *v, NnUint n, NnUint maxPrint) {
    if (tag == nullptr) tag = "vec";
    if (v == nullptr || n == 0u) {
        printf("%s=[]", tag);
        return;
    }
    const NnUint cap = 256u;
    const NnUint dims = (maxPrint == 0u) ? std::min(n, cap) : std::min(n, std::min(maxPrint, cap));
    printf("%s=[", tag);
    for (NnUint i = 0u; i < dims; ++i) {
        if (i) printf(",");
        printf("%.6f", (double)v[i]);
    }
    if (dims < n) printf(",...");
    printf("]");
}

static double l2NormSq_F32(const float *v, NnUint n) {
    if (v == nullptr || n == 0u) return 0.0;
    double s = 0.0;
    for (NnUint i = 0u; i < n; ++i) {
        const double x = (double)v[i];
        s += x * x;
    }
    return s;
}

static inline bool matmulIoDebugEnabledForOp(const char *opName) {
    if (std::getenv("DLLAMA_DEBUG_MATMUL_IO") == nullptr) return false;
    const char *filter = std::getenv("DLLAMA_DEBUG_MATMUL_IO_FILTER");
    if (filter == nullptr || filter[0] == '\0') return true;
    if (opName == nullptr) return false;
    return (std::strstr(opName, filter) != nullptr);
}

static inline NnUint matmulIoDebugLimit() {
    const char *v = std::getenv("DLLAMA_DEBUG_MATMUL_IO_LIMIT");
    if (v == nullptr) return 20u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 100000) return 100000u;
    return (NnUint)n;
}

static inline NnUint matmulIoOutOffset() {
    const char *v = std::getenv("DLLAMA_DEBUG_MATMUL_IO_OUT_OFFSET");
    if (v == nullptr) return 0u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline NnUint matmulIoOutLen() {
    const char *v = std::getenv("DLLAMA_DEBUG_MATMUL_IO_OUT_LEN");
    if (v == nullptr) return 64u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static void printMatmulOutSummaryF32(
    const char *opName,
    NnUint opIndex,
    NnUint y,
    NnUint e,
    const float *outBase,
    NnUint outLen,
    NnUint cViewOffset,
    NnUint cLen,
    NnUint sliceOffset,
    NnUint sliceLen)
{
    if (outBase == nullptr || outLen == 0u) return;
    sliceOffset = std::min(sliceOffset, outLen);
    const NnUint maxSlice = outLen - sliceOffset;
    const NnUint n = (sliceLen == 0u) ? std::min(outLen, 256u) : std::min(sliceLen, maxSlice);
    const float *p = outBase + sliceOffset;

    float minV = p[0], maxV = p[0];
    double normSq = 0.0;
    NnUint zeros = 0u;
    for (NnUint i = 0u; i < n; ++i) {
        const float v = p[i];
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
        normSq += (double)v * (double)v;
        if (v == 0.0f) ++zeros;
    }

    printf("🔬 [matmul][io] op=%s opIndex=%u y=%u e=%u outBase=%p outLen=%u cViewOff=%u cLen=%u outSlice=[%u,%u) min=%.6f max=%.6f norm=%.6f zeros=%u/%u head=" ,
        (opName ? opName : "Unknown"),
        (unsigned)opIndex,
        (unsigned)y,
        (unsigned)e,
        (const void *)outBase,
        (unsigned)outLen,
        (unsigned)cViewOffset,
        (unsigned)cLen,
        (unsigned)sliceOffset,
        (unsigned)(sliceOffset + n),
        (double)minV,
        (double)maxV,
        (double)std::sqrt(normSq),
        (unsigned)zeros,
        (unsigned)n);
    const NnUint headN = std::min(n, 16u);
    for (NnUint i = 0u; i < headN; ++i) {
        printf("%s%.6f", (i == 0u ? "[" : ","), (double)p[i]);
    }
    printf("]\n");
}

static void printMatmulInSummaryQ80(
    const char *opName,
    NnUint opIndex,
    NnUint y,
    NnUint e,
    const NnBlockQ80 *x,
    NnUint aLen)
{
    if (x == nullptr || aLen == 0u) return;
    const NnUint blocks = aLen / Q80_BLOCK_SIZE;
    const NnUint scanBlocks = std::min(blocks, 16u);
    NnUint nonZero = 0u;
    for (NnUint b = 0u; b < scanBlocks; ++b) {
        for (NnUint i = 0u; i < Q80_BLOCK_SIZE; ++i) {
            if (x[b].qs[i] != 0) { ++nonZero; break; }
        }
    }
    printf("🔬 [matmul][io] op=%s opIndex=%u y=%u e=%u inType=Q80 x=%p aLen=%u blocks=%u scanBlocks=%u nonZeroBlocks=%u\n",
        (opName ? opName : "Unknown"),
        (unsigned)opIndex,
        (unsigned)y,
        (unsigned)e,
        (const void *)x,
        (unsigned)aLen,
        (unsigned)blocks,
        (unsigned)scanBlocks,
        (unsigned)nonZero);
}

static inline bool ropeIoDebugEnabledForOp(const char *opName) {
    if (std::getenv("DLLAMA_DEBUG_ROPE_IO") == nullptr) return false;
    const char *filter = std::getenv("DLLAMA_DEBUG_ROPE_IO_FILTER");
    if (filter == nullptr || filter[0] == '\0') return true;
    if (opName == nullptr) return false;
    return (std::strstr(opName, filter) != nullptr);
}

static inline NnUint ropeIoDebugLimit() {
    const char *v = std::getenv("DLLAMA_DEBUG_ROPE_IO_LIMIT");
    if (v == nullptr) return 50u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 100000) return 100000u;
    return (NnUint)n;
}

static inline NnUint ropeIoOffset() {
    const char *v = std::getenv("DLLAMA_DEBUG_ROPE_IO_OFFSET");
    if (v == nullptr) return 0u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static inline NnUint ropeIoLen() {
    const char *v = std::getenv("DLLAMA_DEBUG_ROPE_IO_LEN");
    if (v == nullptr) return 256u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 1000000) return 1000000u;
    return (NnUint)n;
}

static void printFloatSliceStats(const char *prefix, const char *opName, NnUint opIndex, const float *base, NnUint baseLen, NnUint sliceOff, NnUint sliceLen) {
    if (base == nullptr || baseLen == 0u) return;
    sliceOff = std::min(sliceOff, baseLen);
    const NnUint maxSlice = baseLen - sliceOff;
    const NnUint n = (sliceLen == 0u) ? std::min(maxSlice, 256u) : std::min(sliceLen, maxSlice);
    const float *p = base + sliceOff;

    float minV = p[0], maxV = p[0];
    double normSq = 0.0;
    NnUint zeros = 0u;
    for (NnUint i = 0u; i < n; ++i) {
        const float v = p[i];
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
        normSq += (double)v * (double)v;
        if (v == 0.0f) ++zeros;
    }
    printf("🔁 [%s][io] op=%s opIndex=%u base=%p baseLen=%u slice=[%u,%u) min=%.6f max=%.6f norm=%.6f zeros=%u/%u head=",
        (prefix ? prefix : "buf"),
        (opName ? opName : "Unknown"),
        (unsigned)opIndex,
        (const void *)base,
        (unsigned)baseLen,
        (unsigned)sliceOff,
        (unsigned)(sliceOff + n),
        (double)minV,
        (double)maxV,
        (double)std::sqrt(normSq),
        (unsigned)zeros,
        (unsigned)n);
    const NnUint headN = std::min(n, 16u);
    for (NnUint i = 0u; i < headN; ++i) {
        printf("%s%.6f", (i == 0u ? "[" : ","), (double)p[i]);
    }
    printf("]\n");
}

static inline NnUint kvCacheDebugLimit() {
    // Default: keep logs bounded.
    const char *v = std::getenv("kvcache_debug_limit");
    if (v == nullptr) return 8u;
    const long n = std::strtol(v, nullptr, 10);
    if (n <= 0) return 0u;
    if (n > 100000) return 100000u;
    return (NnUint)n;
}

static void printAttRowSummary(const float *attRow, NnUint len, NnUint globalQHead, NnUint globalKvHead) {
    if (attRow == nullptr || len == 0u) return;

    // Compute top-5 entries and a simple entropy metric.
    struct Top { float v; NnUint i; };
    Top top[5];
    for (int k = 0; k < 5; ++k) top[k] = Top{-1.0f, 0u};

    float maxV = -1.0f;
    NnUint maxI = 0u;
    double entropy = 0.0;

    for (NnUint i = 0; i < len; ++i) {
        const float p = attRow[i];
        if (p > maxV) { maxV = p; maxI = i; }
        if (p > 0.0f) entropy += -(double)p * std::log((double)p);

        for (int k = 0; k < 5; ++k) {
            if (p > top[k].v) {
                for (int s = 4; s > k; --s) top[s] = top[s - 1];
                top[k] = Top{p, i};
                break;
            }
        }
    }

    printf("🧪 [att] qHead=%u kvHead=%u len=%u max@%u=%.6f entropy=%.4f top5:",
        (unsigned)globalQHead,
        (unsigned)globalKvHead,
        (unsigned)len,
        (unsigned)maxI,
        (double)maxV,
        entropy);
    for (int k = 0; k < 5; ++k) {
        if (top[k].v < 0.0f) break;
        printf(" (%u,%.6f)", (unsigned)top[k].i, (double)top[k].v);
    }
    printf("\n");
}

#endif // DLLAMA_DEBUG_ATTN

static void noopForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    (void)nThreads;
    (void)threadIndex;
    (void)batchSize;
    (void)context;
}

static float dotProduct_F32(const float *a, const float *b, const unsigned int size);

#ifndef DEBUG_OP_INPUT_OUTPUT
    #define DEBUG_OP_INPUT_OUTPUT 0
#endif

#if DEBUG_OP_INPUT_OUTPUT
    #define DEBUG_VECTOR(context, suffix, v) \
        if (threadIndex == 0) { \
            printf("%20s.%6s: ", context->name, suffix); \
            for (int k = 0; k < 12; k++) printf("%f ", v[k]); \
            printf("\n"); \
        }

    #define DEBUG_SCALAR(context, suffix, scalar) \
        if (threadIndex == 0) \
            printf("%20s.%6s: %f\n", context->name, suffix, scalar);
#else
    #define DEBUG_VECTOR(context, suffix, vec)
    #define DEBUG_SCALAR(context, suffix, scalar)
#endif

#if defined(__ARM_NEON)
static inline float32x4_t expf_neon(float32x4_t x) {
    const float32x4_t ln2 = vdupq_n_f32(0.69314718056f);
    const float32x4_t inv_ln2 = vdupq_n_f32(1.44269504089f);
    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.1666666667f);
    const float32x4_t c4 = vdupq_n_f32(0.04166666667f);
    const float32x4_t c5 = vdupq_n_f32(0.008333333333f);

    x = vminq_f32(x, vdupq_n_f32(88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));

    float32x4_t kf = vaddq_f32(vmulq_f32(x, inv_ln2), vdupq_n_f32(0.5f));
    int32x4_t k = vcvtq_s32_f32(kf);
    kf = vcvtq_f32_s32(k);

    float32x4_t f = vmlsq_f32(x, kf, ln2);
    float32x4_t f2 = vmulq_f32(f, f);
    float32x4_t f3 = vmulq_f32(f2, f);
    float32x4_t f4 = vmulq_f32(f3, f);
    float32x4_t f5 = vmulq_f32(f4, f);
    float32x4_t p = c1;
    p = vaddq_f32(p, f);
    p = vaddq_f32(p, vmulq_f32(c2, f2));
    p = vaddq_f32(p, vmulq_f32(c3, f3));
    p = vaddq_f32(p, vmulq_f32(c4, f4));
    p = vaddq_f32(p, vmulq_f32(c5, f5));

    int32x4_t pow2k = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
    float32x4_t two_k = vreinterpretq_f32_s32(pow2k);
    return vmulq_f32(p, two_k);
}
#endif

#if defined(__AVX2__)
static inline float horizontalSum_avx2(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline float horizontalMax_avx2(__m256 v) {
    __m128 v_low = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(v_low, v_high);
    __m128 max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    __m128 max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(max32);
}

static inline __m256 expf_avx2(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 c4 = _mm256_set1_ps(0.009618129107628477f);
    __m256 y = _mm256_mul_ps(x, log2e);
    __m256i n = _mm256_cvtps_epi32(y);
    __m256 n_float = _mm256_cvtepi32_ps(n);
    __m256 f = _mm256_sub_ps(y, n_float);
    __m256 p = c4;
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);
    __m256i exponent = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    exponent = _mm256_slli_epi32(exponent, 23);
    __m256 two_n = _mm256_castsi256_ps(exponent);
    return _mm256_mul_ps(p, two_n);
}
#endif

static float invRms_F32(const float *x, const unsigned int size, const float epsilon) {
    float sum;
#if defined(__ARM_NEON)
    assert(size % 4 == 0);
    float32x4_t fsq;
    float32x4_t fs = vmovq_n_f32(0);
    for (unsigned int j = 0; j < size; j += 4) {
        fsq = vld1q_f32(&x[j]);
        fs = vmlaq_f32(fs, fsq, fsq);
    }
    sum = vaddvq_f32(fs);
#elif defined(__AVX2__)
    assert(size % 8 == 0);
    __m256 a;
    __m256 u = _mm256_set1_ps(0.0f);
    for (unsigned int j = 0; j < size; j += 8) {
        a = _mm256_loadu_ps(&x[j]);
        u = _mm256_fmadd_ps(a, a, u);
    }
    sum = horizontalSum_avx2(u);
#else
    sum = 0;
    for (unsigned int j = 0; j < size; j++) {
        sum += x[j] * x[j];
    }
#endif
    sum /= size;
    sum += epsilon;
    return 1.0f / sqrtf(sum);
}

static float invRms_F32_any(const float *x, const NnUint size, const float epsilon) {
    if (size == 0u)
        return 0.0f;
    float sum = 0.0f;
    for (NnUint i = 0; i < size; i++) {
        const float v = x[i];
        sum += v * v;
    }
    sum /= (float)size;
    sum += epsilon;
    return 1.0f / sqrtf(sum);
}

static void rmsNorm_F32(float *output, const float *x, const float invRms, const float *w, const NnUint size, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);
    unsigned int i = start;
#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 4);
    float32x4_t fw;
    float32x4_t fx;
    float32x4_t fss = vmovq_n_f32(invRms);
    for (; i < neonEnd; i += 4) {
        fw = vld1q_f32(&w[i]);
        fx = vld1q_f32(&x[i]);
        fx = vmulq_f32(fx, fw);
        fx = vmulq_f32(fx, fss);
        vst1q_f32(&output[i], fx);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);
    const __m256 invRmsVec = _mm256_set1_ps(invRms);
    for (; i < avxEnd; i += 8) {
        __m256 xVec = _mm256_loadu_ps(&x[i]);
        __m256 wVec = _mm256_loadu_ps(&w[i]);
        __m256 scaledX = _mm256_mul_ps(xVec, invRmsVec);
        __m256 result = _mm256_mul_ps(scaledX, wVec);
        _mm256_storeu_ps(output + i, result);
    }
#endif
    for (; i < end; i++)
        output[i] = w[i] * (invRms * x[i]);
}

static void rmsNorm_Q80_F32_F32(float *output, const NnBlockQ80 *x, const float invRms, const float *w, const NnUint size, const NnUint nThreads, const NnUint threadIndex) {
    assert(size % Q80_BLOCK_SIZE == 0);
    const NnUint nBlocks = size / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

    for (NnUint i = start; i < end; i++) {
        float d = CONVERT_F16_TO_F32(x[i].d);
        for (NnUint j = 0; j < Q80_BLOCK_SIZE; j++) {
            NnUint k = i * Q80_BLOCK_SIZE + j;
            output[k] = w[k] * (invRms * d * x[i].qs[j]);
        }
    }
}

static void matmul_F32_F32_F32(float *output, const float *x, const float *w, const NnUint n, const NnUint d, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, d, nThreads, threadIndex);
    unsigned int i, j;
#if defined(__ARM_NEON)
    assert(n % 4 == 0);
    float32x4_t q;
    float32x4_t p;
    float32x4_t z;
    for (i = start; i < end; i++) {
        z = vmovq_n_f32(0);
        for (j = 0; j < n; j += 4) {
            q = vld1q_f32(&x[j]);
            p = vld1q_f32(&w[i * n + j]);
            z = vfmaq_f32(z, q, p);
        }
        output[i] = vaddvq_f32(z);
    }
#elif defined(__AVX2__)
    assert(n % 8 == 0);
    __m256 a0, b0, u;
    for (i = start; i < end; i++) {
        u = _mm256_set1_ps(0.0f);
        for (j = 0; j < n; j += 8) {
            a0 = _mm256_loadu_ps(&x[j]);
            b0 = _mm256_loadu_ps(&w[i * n + j]);
            u = _mm256_fmadd_ps(a0, b0, u);
        }
        output[i] = horizontalSum_avx2(u);
    }
#else
    for (i = start; i < end; i++) {
        float val = 0.0f;
        for (j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        output[i] = val;
    }
#endif
}

static void matmul_F32_F32_F32_colSlice(
    float *output,
    const float *x,
    const float *wBase,
    const NnUint nTotal,
    const NnUint dLocal,
    const NnUint inStart,
    const NnUint nLocal,
    const NnUint outStart,
    const NnUint nThreads,
    const NnUint threadIndex) {
    // Computes local output rows [outStart, outStart + dLocal) against local input cols [inStart, inStart + nLocal).
    SPLIT_THREADS(start, end, dLocal, nThreads, threadIndex);
    for (NnUint i0 = start; i0 < end; i0++) {
        const NnUint i = outStart + i0;
        const float *wRow = &wBase[i * nTotal + inStart];
        output[i0] = dotProduct_F32(x, wRow, nLocal);
    }
}

static void matmul_Q80_Q40_F32(float *output, const NnBlockQ80 *x, const NnBlockQ40 *w, const NnUint n, const NnUint d, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, d, nThreads, threadIndex);
    assert(n % Q40_BLOCK_SIZE == 0);
    const unsigned int nBlocks = n / Q40_BLOCK_SIZE;

#if defined(__ARM_NEON)
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t s8b = vdupq_n_s8(0x8);

    for (unsigned int di = start; di < end; di++) {
        float32x4_t sumv0 = vmovq_n_f32(0.0f);
        float32x4_t sumv1 = vmovq_n_f32(0.0f);
        float32x4_t sumv2 = vmovq_n_f32(0.0f);
        float32x4_t sumv3 = vmovq_n_f32(0.0f);

        unsigned int j = 0;
        
#if defined(__ARM_FEATURE_DOTPROD)
        for (; j + 3 < nBlocks; j += 4) {
            __builtin_prefetch(&w[di * nBlocks + j + 4]);
            __builtin_prefetch(&x[j + 4]);

            const NnBlockQ40 *w0 = &w[di * nBlocks + j];
            const NnBlockQ40 *w1 = &w[di * nBlocks + j + 1];
            const NnBlockQ40 *w2 = &w[di * nBlocks + j + 2];
            const NnBlockQ40 *w3 = &w[di * nBlocks + j + 3];

            const NnBlockQ80 *x0 = &x[j];
            const NnBlockQ80 *x1 = &x[j + 1];
            const NnBlockQ80 *x2 = &x[j + 2];
            const NnBlockQ80 *x3 = &x[j + 3];

            int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w0->qs), m4b)), s8b);
            int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w0->qs), 4)), s8b);
            int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w1->qs), m4b)), s8b);
            int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w1->qs), 4)), s8b);
            int8x16_t w2l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w2->qs), m4b)), s8b);
            int8x16_t w2h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w2->qs), 4)), s8b);
            int8x16_t w3l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w3->qs), m4b)), s8b);
            int8x16_t w3h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w3->qs), 4)), s8b);

            const int8x16_t x0l = vld1q_s8(x0->qs);
            const int8x16_t x0h = vld1q_s8(x0->qs + 16);
            const int8x16_t x1l = vld1q_s8(x1->qs);
            const int8x16_t x1h = vld1q_s8(x1->qs + 16);
            const int8x16_t x2l = vld1q_s8(x2->qs);
            const int8x16_t x2h = vld1q_s8(x2->qs + 16);
            const int8x16_t x3l = vld1q_s8(x3->qs);
            const int8x16_t x3h = vld1q_s8(x3->qs + 16);

            const int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0l, x0l), w0h, x0h);
            const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w1l, x1l), w1h, x1h);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w2l, x2l), w2h, x2h);
            const int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w3l, x3l), w3h, x3h);

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), CONVERT_F16_TO_F32(w0->d) * CONVERT_F16_TO_F32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), CONVERT_F16_TO_F32(w1->d) * CONVERT_F16_TO_F32(x1->d));
            sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(p2), CONVERT_F16_TO_F32(w2->d) * CONVERT_F16_TO_F32(x2->d));
            sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(p3), CONVERT_F16_TO_F32(w3->d) * CONVERT_F16_TO_F32(x3->d));
        }
#else
        for (; j + 1 < nBlocks; j += 2) {
            const NnBlockQ40 *w0 = &w[di * nBlocks + j];
            const NnBlockQ40 *w1 = &w[di * nBlocks + j + 1];
            const NnBlockQ80 *x0 = &x[j];
            const NnBlockQ80 *x1 = &x[j + 1];

            const uint8x16_t w0qs = vld1q_u8(w0->qs);
            const uint8x16_t w1qs = vld1q_u8(w1->qs);
            
            int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w0qs, m4b)), s8b);
            int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w0qs, 4)), s8b);
            int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w1qs, m4b)), s8b);
            int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w1qs, 4)), s8b);

            const int8x16_t x0l = vld1q_s8(x0->qs);
            const int8x16_t x0h = vld1q_s8(x0->qs + 16);
            const int8x16_t x1l = vld1q_s8(x1->qs);
            const int8x16_t x1h = vld1q_s8(x1->qs + 16);

            const int16x8_t pl0l = vmull_s8(vget_low_s8(w0l), vget_low_s8(x0l));
            const int16x8_t pl0h = vmull_s8(vget_high_s8(w0l), vget_high_s8(x0l));
            const int16x8_t ph0l = vmull_s8(vget_low_s8(w0h), vget_low_s8(x0h));
            const int16x8_t ph0h = vmull_s8(vget_high_s8(w0h), vget_high_s8(x0h));
            
            const int16x8_t pl1l = vmull_s8(vget_low_s8(w1l), vget_low_s8(x1l));
            const int16x8_t pl1h = vmull_s8(vget_high_s8(w1l), vget_high_s8(x1l));
            const int16x8_t ph1l = vmull_s8(vget_low_s8(w1h), vget_low_s8(x1h));
            const int16x8_t ph1h = vmull_s8(vget_high_s8(w1h), vget_high_s8(x1h));

            const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
            const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
            const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
            const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), CONVERT_F16_TO_F32(w0->d) * CONVERT_F16_TO_F32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), CONVERT_F16_TO_F32(w1->d) * CONVERT_F16_TO_F32(x1->d));
        }
#endif

        for (; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[di * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];

            const uint8x16_t wqs = vld1q_u8(wb->qs);
            const int8x16_t wl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(wqs, m4b)), s8b);
            const int8x16_t wh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(wqs, 4)), s8b);

            const int8x16_t xl = vld1q_s8(xb->qs);
            const int8x16_t xh = vld1q_s8(xb->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
            const int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl, xl), wh, xh);
#else
            const int16x8_t pll = vmull_s8(vget_low_s8(wl), vget_low_s8(xl));
            const int16x8_t plh = vmull_s8(vget_high_s8(wl), vget_high_s8(xl));
            const int16x8_t phl = vmull_s8(vget_low_s8(wh), vget_low_s8(xh));
            const int16x8_t phh = vmull_s8(vget_high_s8(wh), vget_high_s8(xh));
            
            const int32x4_t pl = vaddq_s32(vpaddlq_s16(pll), vpaddlq_s16(plh));
            const int32x4_t ph = vaddq_s32(vpaddlq_s16(phl), vpaddlq_s16(phh));
            const int32x4_t p = vaddq_s32(pl, ph);
#endif
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), s);
        }

        output[di] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
    }
#elif defined(__AVX512F__)
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0f;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);

            __m128i w8 = _mm_loadu_si128((const __m128i*)wb->qs);
            __m128i v_w0 = _mm_and_si128(w8, _mm_set1_epi8(0x0F));
            __m128i v_w1 = _mm_srli_epi16(w8, 4);
            v_w1 = _mm_and_si128(v_w1, _mm_set1_epi8(0x0F));
            
            v_w0 = _mm_sub_epi8(v_w0, _mm_set1_epi8(8));
            v_w1 = _mm_sub_epi8(v_w1, _mm_set1_epi8(8));

            __m256i w8_combined = _mm256_set_m128i(v_w1, v_w0);
            __m512i w16 = _mm512_cvtepi8_epi16(w8_combined);

            __m256i x8 = _mm256_loadu_si256((const __m256i*)xb->qs);
            __m512i x16 = _mm512_cvtepi8_epi16(x8);

            __m512i products = _mm512_madd_epi16(w16, x16);
            sum += _mm512_reduce_add_epi32(products) * s;
        }
        output[i] = sum;
    }
#elif defined(__AVX2__)
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0f;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);

            __m128i w_packed = _mm_loadu_si128((const __m128i*)wb->qs);

            __m128i w0_low = _mm_and_si128(w_packed, _mm_set1_epi8(0x0F));
            __m128i w0 = _mm_sub_epi8(w0_low, _mm_set1_epi8(8));

            __m128i w1_high = _mm_srli_epi16(w_packed, 4);
            w1_high = _mm_and_si128(w1_high, _mm_set1_epi8(0x0F));
            __m128i w1 = _mm_sub_epi8(w1_high, _mm_set1_epi8(8));

            __m256i w0_16 = _mm256_cvtepi8_epi16(w0);
            __m256i w1_16 = _mm256_cvtepi8_epi16(w1);

            __m128i i1_8 = _mm_loadu_si128((const __m128i*)xb->qs);
            __m128i i2_8 = _mm_loadu_si128((const __m128i*)(xb->qs + Q40_BLOCK_SIZE / 2));

            __m256i i1_16 = _mm256_cvtepi8_epi16(i1_8);
            __m256i i2_16 = _mm256_cvtepi8_epi16(i2_8);

            __m256i prod0 = _mm256_mullo_epi16(w0_16, i1_16);
            __m256i prod1 = _mm256_mullo_epi16(w1_16, i2_16);
            __m256i sum_prod = _mm256_add_epi16(prod0, prod1);

            __m256i ones = _mm256_set1_epi16(1);
            __m256i sum32 = _mm256_madd_epi16(sum_prod, ones);

            __m128i sum_low = _mm256_castsi256_si128(sum32);
            __m128i sum_high = _mm256_extracti128_si256(sum32, 1);
            sum_low = _mm_add_epi32(sum_low, sum_high);
            sum_low = _mm_hadd_epi32(sum_low, sum_low);
            sum_low = _mm_hadd_epi32(sum_low, sum_low);
            int32_t block_sum = _mm_extract_epi32(sum_low, 0);

            sum += block_sum * s;
        }
        output[i] = sum;
    }
#else
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);
            for (NnUint k = 0; k < Q40_BLOCK_SIZE / 2; k++) {
                const int w0 = (wb->qs[k] & 0x0F) - 8;
                const int w1 = (wb->qs[k] >> 4) - 8;
                const int i1 = xb->qs[k];
                const int i2 = xb->qs[k + Q80_BLOCK_SIZE / 2];
                sum += (w0 * i1 + w1 * i2) * s;
            }
        }
        output[i] = sum;
    }
#endif
}

static void matmul_Q80_Q40_F32_colSlice(
    float *output,
    const NnBlockQ80 *x,
    const NnBlockQ40 *wBase,
    const NnUint nTotal,
    const NnUint dLocal,
    const NnUint inStart,
    const NnUint nLocal,
    const NnUint outStart,
    const NnUint nThreads,
    const NnUint threadIndex) {
    // Correctness-first strided col-slice matmul.
    // Weight rows are length nTotal elements (quantized in Q40 blocks).
    assert(nTotal % Q40_BLOCK_SIZE == 0);
    assert(nLocal % Q40_BLOCK_SIZE == 0);
    assert(inStart % Q40_BLOCK_SIZE == 0);

    const NnUint nTotalBlocks = nTotal / Q40_BLOCK_SIZE;
    const NnUint inStartBlock = inStart / Q40_BLOCK_SIZE;
    const NnUint nLocalBlocks = nLocal / Q40_BLOCK_SIZE;

    SPLIT_THREADS(start, end, dLocal, nThreads, threadIndex);
    for (NnUint di0 = start; di0 < end; di0++) {
        const NnUint di = outStart + di0;
        const NnBlockQ40 *wRow = &wBase[di * nTotalBlocks + inStartBlock];

        float sum = 0.0f;
        for (NnUint bi = 0; bi < nLocalBlocks; bi++) {
            const NnBlockQ80 *xb = &x[bi];
            const NnBlockQ40 *wb = &wRow[bi];

            const float dx = CONVERT_F16_TO_F32(xb->d);
            const float dw = CONVERT_F16_TO_F32(wb->d);

            int acc = 0;
            for (NnUint j = 0; j < Q40_BLOCK_SIZE / 2; j++) {
                const int w0 = (wb->qs[j] & 0x0F) - 8;
                const int w1 = (wb->qs[j] >> 4) - 8;
                acc += (int)xb->qs[j] * w0;
                acc += (int)xb->qs[j + Q40_BLOCK_SIZE / 2] * w1;
            }

            sum += (dx * dw) * (float)acc;
        }

        output[di0] = sum;
    }
}

#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A 0.044715f

static void gelu_F32(float *output, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        float x = output[i];
        output[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }
}

static void gelu_F32_strided(float *output, const NnUint n, const NnUint stride, const NnUint nThreads, const NnUint threadIndex) {
    assert(stride > 0u);
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (NnUint i = start; i < end; i++) {
        float *p = &output[i * stride];
        const float x = *p;
        *p = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }
}

static void silu_F32(float *output, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    unsigned int i = start;
#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 4);

    for (; i < neonEnd; i += 4) {
        float32x4_t x = vld1q_f32(&output[i]);
        float32x4_t neg_x = vnegq_f32(x);
        float32x4_t exp_negx = expf_neon(neg_x);
        float32x4_t denominator = vaddq_f32(exp_negx, vdupq_n_f32(1.0f));

        float32x4_t recip = vrecpeq_f32(denominator);
        recip = vmulq_f32(recip, vsubq_f32(vdupq_n_f32(2.0f), vmulq_f32(denominator, recip)));

        float32x4_t result = vmulq_f32(x, recip);
        vst1q_f32(output + i, result);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);

    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    for (; i < avxEnd; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(output + i);
        __m256 neg_x = _mm256_sub_ps(zero, x_vec);
        __m256 exp_negx = expf_avx2(neg_x);
        __m256 denominator = _mm256_add_ps(ones, exp_negx);
        __m256 result = _mm256_div_ps(x_vec, denominator);
        _mm256_storeu_ps(output + i, result);
    }
#endif
    for (; i < end; i++) {
        float x = output[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

static void silu_F32_strided(float *output, const NnUint n, const NnUint stride, const NnUint nThreads, const NnUint threadIndex) {
    assert(stride > 0u);
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (NnUint i = start; i < end; i++) {
        float *p = &output[i * stride];
        const float x = *p;
        *p = x / (1.0f + expf(-x));
    }
}

static void add_F32(float *output, const float *x, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        output[i] += x[i];
    }
}

static void add_Q80_F32(float *y, const NnBlockQ80 *x, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint nBlocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

#if defined(__ARM_NEON)
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);
        float *y_base = y + i * Q80_BLOCK_SIZE;
        const int8_t *qs = xi->qs;

        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j += 16) {
            // Load 16x 8-bit quantized values
            const int8x16_t q8 = vld1q_s8(qs + j);
            
            // Split into 8-bit high/low components
            const int8x8_t q8_low = vget_low_s8(q8);
            const int8x8_t q8_high = vget_high_s8(q8);
            
            // Sign extend to 16-bit
            const int16x8_t q16_low = vmovl_s8(q8_low);
            const int16x8_t q16_high = vmovl_s8(q8_high);
            
            // Sign extend to 32-bit and convert to float
            const float32x4_t qf_ll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_low)));
            const float32x4_t qf_lh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_low)));
            const float32x4_t qf_hl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_high)));
            const float32x4_t qf_hh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_high)));
            
            // Multiply by scale factor
            const float32x4_t sf_ll = vmulq_n_f32(qf_ll, xid);
            const float32x4_t sf_lh = vmulq_n_f32(qf_lh, xid);
            const float32x4_t sf_hl = vmulq_n_f32(qf_hl, xid);
            const float32x4_t sf_hh = vmulq_n_f32(qf_hh, xid);
            
            // Load existing y values
            float32x4_t y_ll = vld1q_f32(y_base + j);
            float32x4_t y_lh = vld1q_f32(y_base + j + 4);
            float32x4_t y_hl = vld1q_f32(y_base + j + 8);
            float32x4_t y_hh = vld1q_f32(y_base + j + 12);
            
            // Accumulate results
            y_ll = vaddq_f32(y_ll, sf_ll);
            y_lh = vaddq_f32(y_lh, sf_lh);
            y_hl = vaddq_f32(y_hl, sf_hl);
            y_hh = vaddq_f32(y_hh, sf_hh);
            
            // Store results back
            vst1q_f32(y_base + j, y_ll);
            vst1q_f32(y_base + j + 4, y_lh);
            vst1q_f32(y_base + j + 8, y_hl);
            vst1q_f32(y_base + j + 12, y_hh);
        }
    }
#elif defined(__AVX2__)
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);

        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j += 8) {
            __m128i i8_vec = _mm_loadl_epi64((const __m128i*)(xi->qs + j));

            __m256i i32_vec = _mm256_cvtepi8_epi32(i8_vec);

            __m256 f_vec = _mm256_cvtepi32_ps(i32_vec);

            __m256 scale = _mm256_set1_ps(xid);
            __m256 scaled = _mm256_mul_ps(f_vec, scale);

            float* y_ptr = y + i * Q80_BLOCK_SIZE + j;
            __m256 y_vec = _mm256_loadu_ps(y_ptr);
            y_vec = _mm256_add_ps(y_vec, scaled);
            _mm256_storeu_ps(y_ptr, y_vec);
        }
    }
#else
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);
        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j++) {
            y[i * Q80_BLOCK_SIZE + j] += xid * xi->qs[j];
        }
    }
#endif
}

void softmax_F32(float *x, const NnUint size) {
    if (size == 0)
        return;

#if defined(__ARM_NEON)
    NnUint j;
    float maxVal;
    if (size >= 4) {
        float32x4_t fs;
        float32x4_t fmaxv = vld1q_f32(&x[0]);
        j = size - (size % 4);
        for (NnUint i = 4; i < j; i += 4) {
            fs = vld1q_f32(&x[i]);
            fmaxv = vmaxq_f32(fmaxv, fs);
        }
        maxVal = vmaxvq_f32(fmaxv);
    } else {
        maxVal = x[0];
        j = 1;
    }
    for (; j < size; j++)
        maxVal = fmaxf(maxVal, x[j]);

    const float32x4_t maxVal_vec = vdupq_n_f32(maxVal);
    float32x4_t sumv = vdupq_n_f32(0.0f);
    NnUint i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t val = vld1q_f32(x + i);
        val = vsubq_f32(val, maxVal_vec);
        val = expf_neon(val);
        vst1q_f32(x + i, val);
        sumv = vaddq_f32(sumv, val);
    }

    float32x2_t sum_lo = vadd_f32(vget_low_f32(sumv), vget_high_f32(sumv));
    float sum = vget_lane_f32(sum_lo, 0) + vget_lane_f32(sum_lo, 1);

    for (; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    if (sum == 0.0f)
        sum = 0.000001f;

    const float inv_sum = 1.0f / sum;
    const float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);

    i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t val = vld1q_f32(x + i);
        val = vmulq_f32(val, inv_sum_vec);
        vst1q_f32(x + i, val);
    }
    for (; i < size; ++i)
        x[i] /= sum;
#elif defined(__AVX2__)
    float maxVal;
    const unsigned avxEnd = size - (size % 8);
    NnUint i = 0;

    if (avxEnd >= 8) {
        __m256 max_vec = _mm256_loadu_ps(x);
        i = 8;
        for (; i < avxEnd; i += 8) {
            __m256 vec = _mm256_loadu_ps(&x[i]);
            max_vec = _mm256_max_ps(max_vec, vec);
        }
        maxVal = horizontalMax_avx2(max_vec);
    } else {
        maxVal = x[0];
        i = 1;
    }
    for (; i < size; ++i) {
        if (x[i] > maxVal)
            maxVal = x[i];
    }

    __m256 max_val_vec = _mm256_set1_ps(maxVal);
    __m256 sum_vec = _mm256_setzero_ps();
    float sum = 0.0f;
    i = 0;
    for (; i < avxEnd; i += 8) {
        __m256 vec = _mm256_loadu_ps(&x[i]);
        vec = _mm256_sub_ps(vec, max_val_vec);
        vec = expf_avx2(vec);
        _mm256_storeu_ps(&x[i], vec);
        sum_vec = _mm256_add_ps(sum_vec, vec);
    }
    sum = horizontalSum_avx2(sum_vec);
    for (; i < size; ++i) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    if (sum == 0.0f)
        sum = 0.000001f;

    const float inv_sum = 1.0f / sum;
    const __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);

    i = 0;
    for (; i < avxEnd; i += 8) {
        __m256 vec = _mm256_loadu_ps(x + i);
        vec = _mm256_mul_ps(vec, inv_sum_vec);
        _mm256_storeu_ps(x + i, vec);
    }
    for (; i < size; i++)
        x[i] *= inv_sum;
#else
    float maxVal = x[0];
    for (NnUint i = 1; i < size; i++) {
        if (x[i] > maxVal)
            maxVal = x[i];
    }
    float sum = 0.0f;
    for (NnUint i = 0; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }
    if (sum == 0.0)
        sum = 0.000001;
    for (NnUint i = 0; i < size; i++)
        x[i] /= sum;
#endif
}

static void softmax_F32_strided(float *x, const NnUint size, const NnUint stride) {
    if (size == 0u)
        return;
    assert(stride > 0u);

    float maxVal = x[0];
    for (NnUint i = 1; i < size; i++) {
        const float v = x[i * stride];
        if (v > maxVal)
            maxVal = v;
    }

    float sum = 0.0f;
    for (NnUint i = 0; i < size; i++) {
        float v = expf(x[i * stride] - maxVal);
        x[i * stride] = v;
        sum += v;
    }
    if (sum == 0.0f)
        sum = 0.000001f;
    const float invSum = 1.0f / sum;
    for (NnUint i = 0; i < size; i++)
        x[i * stride] *= invSum;
}

static float dotProduct_F32(const float *a, const float *b, const unsigned int size) {
#if defined(__ARM_NEON)
    assert(size % 4 == 0);
    float32x4_t fa;
    float32x4_t fb;
    float32x4_t fs = vmovq_n_f32(0);
    for (unsigned int i = 0; i < size; i += 4) {
        fa = vld1q_f32(&a[i]);
        fb = vld1q_f32(&b[i]);
        fs = vmlaq_f32(fs, fa, fb);
    }
    return vaddvq_f32(fs);
#elif defined(__AVX2__)
    assert(size % 8 == 0);
    __m256 a0, b0;
    __m256 u = _mm256_set1_ps(0.0f);
    for (unsigned int i = 0; i < size; i += 8) {
        a0 = _mm256_loadu_ps(&a[i]);
        b0 = _mm256_loadu_ps(&b[i]);
        u = _mm256_fmadd_ps(a0, b0, u);
    }
    return horizontalSum_avx2(u);
#else
    float sum = 0.0f;
    for (unsigned int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

static void multiheadAtt_F32(
    float *y, const float *q, float *att, float *keyCache, float *valueCache,
    const NnUint pos, const NnUint nHeads, const NnUint nHeads0, const NnUint nKvHeads,
    const NnUint kvDim0, const NnUint kvStart, const NnUint kvStride,
    const NnUint qHeadStart,
    const NnUint headDim, const NnUint seqLen,
    const NnUint nThreads, const NnUint threadIndex) 
{
    SPLIT_THREADS(h0Start, h0End, nHeads0, nThreads, threadIndex);
    assert(nKvHeads != 0u);
    assert((nHeads % nKvHeads) == 0u);
    const NnUint kvMul = nHeads / nKvHeads;
    const float headDimRoot = sqrtf(headDim);
    assert(headDim != 0u);
    assert((kvStart % headDim) == 0u);
    assert((kvDim0 % headDim) == 0u);
    const NnUint kvHeadStart = kvStart / headDim;
    const NnUint kvHeads0 = kvDim0 / headDim;

    for (NnUint h0 = h0Start; h0 < h0End; h0++) {
        const float *hQ = &q[h0 * headDim];
        const NnUint globalQHead = qHeadStart + h0;
        const NnUint globalKvHead = globalQHead / kvMul;
        const long localKvHeadSigned = (long)globalKvHead - (long)kvHeadStart;
        assert(localKvHeadSigned >= 0);
        const NnUint localKvHead = (NnUint)localKvHeadSigned;
        assert(localKvHead < kvHeads0);

        const float *hKc = &keyCache[kvStart + localKvHead * headDim];
        const float *hVc = &valueCache[kvStart + localKvHead * headDim];
        float *hAtt = &att[h0 * seqLen];

        for (NnUint t = 0; t <= pos; t++) {
            const float *posK = &hKc[t * kvStride];
            const float score = dotProduct_F32(hQ, posK, headDim) / headDimRoot;
            hAtt[t] = score;
        }

        softmax_F32(hAtt, pos + 1);

        float *hY = &y[h0 * headDim];
        std::memset(hY, 0, headDim * sizeof(float));

        for (NnUint t = 0; t <= pos; t++) {
            const float *posV = &hVc[t * kvStride];
            const float posA = hAtt[t];
            for (int i = 0; i < headDim; i++) {
                hY[i] += posA * posV[i];
            }
        }
    }
}

static void mul_F32(float *y, const float *x, const float *m, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    unsigned int i = start;

#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 8);
    for (; i < neonEnd; i += 4) {
        float32x4_t out_vec = vld1q_f32(&x[i]);
        float32x4_t x_vec = vld1q_f32(&m[i]);
        float32x4_t res_vec = vmulq_f32(out_vec, x_vec);
        vst1q_f32(&y[i], res_vec);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);
    for (; i < avxEnd; i += 8) {
        __m256 out_vec = _mm256_loadu_ps(&x[i]);
        __m256 x_vec = _mm256_loadu_ps(&m[i]);
        __m256 res_vec = _mm256_mul_ps(out_vec, x_vec);
        _mm256_storeu_ps(&y[i], res_vec);
    }
#endif
    for (; i < end; i++)
        y[i] = x[i] * m[i];
}

static void mul_F32_strided(float *y, const float *x, const float *m, const NnUint n, const NnUint stride, const NnUint nThreads, const NnUint threadIndex) {
    assert(stride > 0u);
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (NnUint i = start; i < end; i++) {
        const NnUint k = i * stride;
        y[k] = x[k] * m[k];
    }
}

static void scale_F32(const float *i, float *o, const float s, NnSize size, NnUint nThreads, NnUint threadIndex) {
    for (NnUint x = threadIndex; x < size; x += nThreads)
        o[x] = i[x] * s;
}

static void scale_F32_strided(const float *i, float *o, const float s, const NnUint n, const NnUint stride, const NnUint nThreads, const NnUint threadIndex) {
    assert(stride > 0u);
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (NnUint t = start; t < end; t++) {
        const NnUint k = t * stride;
        o[k] = i[k] * s;
    }
}

static void mul_Q80_F32(float *y, const float *x, const NnBlockQ80 *m, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint nBlocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);
    for (NnUint i = start; i < end; i++) {
        const NnBlockQ80 *b = &m[i];
        float d = CONVERT_F16_TO_F32(b->d);
        for (NnUint j = 0; j < Q80_BLOCK_SIZE; j++) {
            NnUint k = i * Q80_BLOCK_SIZE + j;
            y[k] = x[k] * d * b->qs[j];
        }
    }
}

static void copy_UNK(NnByte *output, const NnByte *x, NnSize size, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);
    NnUint s = end - start;
    if (s != 0)
        std::memcpy(&output[start], &x[start], s);
}


static void ropeLlama_F32(float* x, const float *cache, bool isQ, const NnUint pos, const NnRopeSlice *slice, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint dim0Half = (isQ ? slice->qDim0 : slice->kvDim0) / 2;
    const NnUint shift = isQ ? slice->qShift : 0;
    SPLIT_THREADS(s, e, dim0Half, nThreads, threadIndex);
    const NnUint iStart = s * 2;
    const NnUint iEnd = e * 2;

    const float *posCache = &cache[pos * slice->sliceDim + shift];

    for (NnUint i = iStart; i < iEnd; i += 2) {
        const float fcr = posCache[i];
        const float fci = posCache[i + 1];
        const float v0 = x[i];
        const float v1 = x[i + 1];

        float x0 = v0 * fcr - v1 * fci;
        float x1 = v0 * fci + v1 * fcr;
        x[i] = x0;
        x[i + 1] = x1;
    }
}

static void ropeFalcon_F32(float* x, const float *cache, bool isQ, const NnUint pos, const NnRopeSlice *slice, const NnUint nThreads, const NnUint threadIndex) {
    unsigned int dim0 =  isQ ? slice->qDim0 : slice->kvDim0;
    assert(dim0 % slice->headDim == 0);
    unsigned int nHeads0 = dim0 / slice->headDim;
    SPLIT_THREADS(h0s, h0e, nHeads0, nThreads, threadIndex);

    const float *posCache = &cache[pos * slice->headDim];

    for (unsigned int h = h0s; h < h0e; h++) {
        const unsigned int o = h * slice->headDim;
        for (unsigned int j = 0; j < slice->headDim / 2; j++) {
            const float fcr0 = posCache[j];
            const float fci0 = posCache[j + slice->headDim / 2];

            float q0 = x[o + j];
            float q1 = x[o + j + slice->headDim / 2];
            x[o + j] = q0 * fcr0 - q1 * fci0;
            x[o + j + slice->headDim / 2] = q0 * fci0 + q1 * fcr0;
        }
    }
}

static void ropeFalcon_F32_view(
    float *xBase,
    const float *cache,
    const bool isQ,
    const NnUint pos,
    const NnRopeSlice *slice,
    const NnUint offset,
    const NnUint len,
    const NnUint nThreads,
    const NnUint threadIndex)
{
    (void)isQ;
    // Falcon RoPE cache is per-position and per-dimension within headDim.
    // View-slicing is only well-defined when slicing whole heads.
    assert(slice->headDim != 0u);
    assert((offset % slice->headDim) == 0u);
    assert((len % slice->headDim) == 0u);

    const NnUint nHeadsView = len / slice->headDim;
    float *x = xBase + offset;
    SPLIT_THREADS(hs, he, nHeadsView, nThreads, threadIndex);

    const float *posCache = &cache[pos * slice->headDim];

    for (NnUint h = hs; h < he; ++h) {
        const NnUint o = h * slice->headDim;
        for (NnUint j = 0u; j < slice->headDim / 2u; ++j) {
            const float fcr0 = posCache[j];
            const float fci0 = posCache[j + slice->headDim / 2u];

            const float v0 = x[o + j];
            const float v1 = x[o + j + slice->headDim / 2u];
            x[o + j] = v0 * fcr0 - v1 * fci0;
            x[o + j + slice->headDim / 2u] = v0 * fci0 + v1 * fcr0;
        }
    }
}

static void mergeSum_F32(float **output, float **input, const NnSize size, const NnSize batchSize, const NnSize nBatches, const NnSize nZ, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);

    for (NnUint y = 0u; y < batchSize; y++) {
        for (NnUint x = start; x < end; x++) {
            float s = 0.0f;
            for (NnUint z = 0u; z < nZ; z++)
                s += input[y + z * nBatches][x];
            output[y][x] = s;
        }
    }
}

static void topk_F32(const float *x, NnUint *y, NnSize size, NnUint k) {
    assert(k <= size);
    assert(k > 0u);

    std::vector<NnUint> items(size);
    for (NnSize i = 0u; i < size; i++)
        items[i] = i;

    std::sort(items.begin(), items.end(),
        [&x](int a, int b) {
            return x[a] > x[b];
        }
    );

    for (NnUint i = 0u; i < k; i++)
        y[i] = items[i];
}

//

static void mergeAddForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    NnUint nSlices = context->inputSize.x / context->outputSize.x;

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        float *input = (float *)context->input[batchIndex];
        for (NnUint sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
            float *i = &input[sliceIndex * context->outputSize.x];
            DEBUG_VECTOR(context, "input", i);
            add_F32(
                output,
                i,
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void mergeAddForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->inputSize.floatType == F_Q80);
    assert(context->outputSize.floatType == F_32);

    NnUint nSlices = context->inputSize.x / context->outputSize.x;
    NnUint xSize = context->outputSize.x / Q80_BLOCK_SIZE;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        for (NnUint sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
            add_Q80_F32(
                output,
                &input[sliceIndex * xSize],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void mergeSumForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.z, 1u);
    assert(context->inputSize.z >= 1u);

    mergeSum_F32(
        (float **)context->output,
        (float **)context->input,
        context->outputSize.x,
        batchSize,
        context->nBatches,
        context->inputSize.z,
        nThreads,
        threadIndex);
}

static void initEmbeddingForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, 1);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->weightSize.x, context->outputSize.x);
}

static void embeddingForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    NnSize dimSize = getBytes(F_32, context->outputSize.x);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnUint token = (NnUint)*((float *)context->input[batchIndex]);
        copy_UNK(
            context->output[batchIndex],
            &context->weight[token * dimSize],
            dimSize,
            nThreads,
            threadIndex);
    }
}

static void embeddingForward_F32_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    NnSize dimSize = getBytes(F_32, context->outputSize.x);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnUint token = (NnUint)*((float *)context->input[batchIndex]);
        quantizeF32toQ80(
            (float *)&context->weight[token * dimSize],
            (NnBlockQ80 *)context->output[batchIndex],
            context->outputSize.x,
            nThreads,
            threadIndex);
    }
}

static void initInvRmsForward(NnCpuOpContext *context) {
    const NnInvRmsOpConfig *config = (const NnInvRmsOpConfig *)context->opConfig;
    assert(context->outputSize.x >= config->nColumns);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    assert(context->inputSize.x % config->nColumns == 0u);
}

static void invRmsForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnInvRmsOpConfig *config = (NnInvRmsOpConfig *)context->opConfig;
    const NnUint colSize = context->inputSize.x / config->nColumns;

    // INV_RMS is a reduction. In this engine it is always computed over the full global
    // hidden size for each column (no view slicing).

    for (NnUint batchIndex = threadIndex; batchIndex < batchSize; batchIndex += nThreads) {
        const float *input = (const float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);
        for (NnUint colIndex = 0; colIndex < config->nColumns; colIndex++) {
            const NnUint colStart = colIndex * colSize;

            const float rms = invRms_F32(&input[colStart], colSize, config->epsilon);
            output[colIndex] = rms;
            DEBUG_SCALAR(context, "output", rms);
        }
    }
}

static void initRmsNormForward_ANY_F32_F32(NnCpuOpContext *context) {
    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    NnBufferConfig *rmsBufferConfig = &context->bufferConfigs[config->invRmsBufferIndex];
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.x % config->nColumns, 0);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->weightSize.floatType, F_32);
    ASSERT_EQ(context->weightSize.y, 1);
    ASSERT_EQ(context->weightSize.x, context->inputSize.x / config->nColumns);
    ASSERT_EQ(rmsBufferConfig->size.floatType, F_32);
    assert(rmsBufferConfig->size.x >= config->nColumns);
    ASSERT_EQ(rmsBufferConfig->size.y, context->nBatches);
}

static void rmsNormForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->inputSize.z, 1u);
    ASSERT_EQ(context->outputSize.z, 1u);

    const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    const float *weight = (float *)context->weight;
    const NnUint invRmsBatchSize = context->bufferConfigs[config->invRmsBufferIndex].size.x;
    const float *invRms = (float *)context->buffers[config->invRmsBufferIndex];

    const NnUint colSize = context->weightSize.x;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        const float *input = (const float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);

        // Always write the full global hidden size.
        for (NnUint xIndex = threadIndex; xIndex < context->outputSize.x; xIndex += nThreads) {
            const NnUint colIndex = xIndex / colSize;
            const NnUint local = xIndex - colIndex * colSize;
            assert(colIndex < config->nColumns);
            const float r = invRms[batchIndex * invRmsBatchSize + colIndex];
            output[xIndex] = weight[local] * (r * input[xIndex]);
        }

        DEBUG_VECTOR(context, "output", output);
    }
}

static void rmsNormForward_Q80_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);
    ASSERT_EQ(context->inputSize.z, 1u);
    ASSERT_EQ(context->outputSize.z, 1u);

    const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    ASSERT_EQ(config->nColumns, 1); // TODO: add support multiple columns

    const float *weight = (float *)context->weight;
    const NnUint invRmsBatchSize = context->bufferConfigs[config->invRmsBufferIndex].size.x;
    const float *invRms = (float *)context->buffers[config->invRmsBufferIndex];
    assert(invRmsBatchSize >= 1u);

    // Q80 path always writes the full global hidden size.
    const NnUint len = context->outputSize.x;
    assert((len % Q80_BLOCK_SIZE) == 0u);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        rmsNorm_Q80_F32_F32(
            output,
            input,
            invRms[batchIndex * invRmsBatchSize],
            weight,
            len,
            nThreads,
            threadIndex);
        DEBUG_VECTOR(context, "output", output);
    }
}

static void initMatmulForward(NnCpuOpContext *context) {
    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);

    const NnTensorView *aView = &config->aView;
    const NnTensorView *cView = &config->cView;

    // For now, matmul only supports per-batch 1D slicing (within each row) for A/C.
    assert(aView->sizeY == 0u);
    assert(cView->sizeY == 0u);
    // When sizeY==0, strideY is unused in the current CPU matmul implementation.
    assert(aView->strideX == 0u || aView->strideX == 1u);
    assert(cView->strideX == 0u || cView->strideX == 1u);

    const NnUint aLen = (aView->sizeX == 0u) ? context->inputSize.x : aView->sizeX;
    const NnUint cLen = (cView->sizeX == 0u) ? context->outputSize.x : cView->sizeX;
    assert(aView->offset + aLen <= context->inputSize.x);
    assert(cView->offset + cLen <= context->outputSize.x);

    if (config->view == 0u) {
        ASSERT_EQ(aLen, context->weightSize.y);
        ASSERT_EQ(cLen, context->weightSize.x);
    } else {
        // Legacy weight view mode: input/output can be slices into a full weight tensor.
        assert(aLen <= context->weightSize.y);
        assert(config->inStart + aLen <= context->weightSize.y);
        assert(cLen <= context->weightSize.x);
        assert(config->outStart + cLen <= context->weightSize.x);
    }
    ASSERT_EQ(context->inputSize.z, std::max(config->nActiveExperts, 1u));
    ASSERT_EQ(context->outputSize.z, std::max(config->nActiveExperts, 1u));
    ASSERT_EQ(context->weightSize.z, std::max(config->nExperts, 1u));


}

static inline bool debugWeightRangesEnabledForOp(const char *opName) {
    if (std::getenv("DLLAMA_DEBUG_WEIGHT_RANGES") == nullptr)
        return false;
    const char *filter = std::getenv("DLLAMA_DEBUG_WEIGHT_RANGES_FILTER");
    if (filter == nullptr || filter[0] == '\0')
        return true;
    if (opName == nullptr)
        return false;
    return (std::strstr(opName, filter) != nullptr);
}

static inline void debugMatmulWeightReadRangeOnce(
    const NnCpuOpContext *context,
    const NnMatmulOpConfig *config,
    NnUint activeExpertIndex,
    NnUint cLen) {
    if (!debugWeightRangesEnabledForOp(context->name))
        return;

    static std::atomic<NnUint> printed{0u};
    NnUint limit = 200u;
    if (const char *limitEnv = std::getenv("DLLAMA_DEBUG_WEIGHT_RANGES_LIMIT")) {
        try { limit = (NnUint)std::stoul(limitEnv); } catch (...) {}
    }
    const NnUint idx = printed.fetch_add(1u);
    if (idx >= limit)
        return;

    // Compute a conservative contiguous read range in bytes within context->weight.
    // Weight storage uses columns of length `weightSize.y` (byte stride = bytes(weightSize.y)).
    const NnSize expertBase = (NnSize)activeExpertIndex * context->weightSize.nBytesXY;
    NnSize begin = expertBase;
    NnSize end = expertBase + context->weightSize.nBytesXY;

    const NnSize colStrideBytes = getBytes(context->weightSize.floatType, context->weightSize.y);
    if (config->view == 1u || config->view == 2u) {
        begin = expertBase + (NnSize)config->outStart * colStrideBytes;
        end = begin + (NnSize)cLen * colStrideBytes;
    }

    printf("📖 [weights][matmul-read] op=%s view=%u expert=%u alloc=[0,%zu) read=[%zu,%zu) outStart=%u cLen=%u colStrideBytes=%zu\n",
        (context->name ? context->name : "Unknown"),
        (unsigned)config->view,
        (unsigned)activeExpertIndex,
        (size_t)context->weightSize.nBytes,
        (size_t)begin,
        (size_t)end,
        (unsigned)config->outStart,
        (unsigned)cLen,
        (size_t)colStrideBytes);
    std::fflush(stdout);
}

static bool matmulForward_llamafile(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    if (config->view != 0u)
        return false;
    if (config->aView.offset != 0u || config->aView.sizeX != 0u)
        return false;
    if (config->cView.offset != 0u || config->cView.sizeX != 0u)
        return false;
    if (batchSize == 1u || !context->hasInputContinuousMemory || !context->hasOutputContinuousMemory || context->inputSize.z != 1u)
        return false;

    // Debug hook: confirm when the llamafile fast path is taken.
    if (threadIndex == 0u && std::getenv("DLLAMA_DEBUG_MATMUL_VIEWS") != nullptr) {
        const char *filter = std::getenv("DLLAMA_DEBUG_MATMUL_VIEWS_FILTER");
        const bool passesFilter = (filter == nullptr) || (std::strstr(context->name, filter) != nullptr);
        if (passesFilter) {
            static std::atomic<NnUint> printed{0u};
            NnUint limit = 50u;
            if (const char *limitEnv = std::getenv("DLLAMA_DEBUG_MATMUL_VIEWS_LIMIT")) {
                try { limit = (NnUint)std::stoul(limitEnv); } catch (...) {}
            }
            const NnUint idx = printed.fetch_add(1u);
            if (idx < limit) {
                const bool viewAware = (config->aView.offset != 0u || config->aView.sizeX != 0u || config->cView.offset != 0u || config->cView.sizeX != 0u);
                const bool forceStyle = viewAware && (config->aView.strideX == 0u) && (config->cView.strideX == 0u);
                const bool offsetActive = (config->aView.offset != 0u || config->cView.offset != 0u);
#if DLLAMA_DEBUG_ATTN
                printf("🔎 [matmul][llamafile] op=%s viewMode=%u aView={off=%u sizeX=%u strideX=%u} cView={off=%u sizeX=%u strideX=%u} batch=%u nThreads=%u\n",
                    context->name,
                    config->view,
                    config->aView.offset, config->aView.sizeX, config->aView.strideX,
                    config->cView.offset, config->cView.sizeX, config->cView.strideX,
                    batchSize, nThreads);
                printf("🔎 [matmul][llamafile] op=%s viewAware=%u forceStyle=%u offsetActive=%u\n",
                    context->name,
                    viewAware ? 1u : 0u,
                    forceStyle ? 1u : 0u,
                    offsetActive ? 1u : 0u);
                std::fflush(stdout);
#endif
            }
        }
    }

    const NnUint n = context->weightSize.y / getBlockSize(context->inputSize.floatType);
    const NnUint d = context->weightSize.x;
    const bool ok = llamafile_sgemm(
        d, batchSize, n,
        context->weight, n,
        context->input[0], n,
        context->output[0], d,
        threadIndex, nThreads, 0,
        context->weightSize.floatType,
        context->inputSize.floatType,
        F_32
    );

    // Optional matmul IO debug summary (best-effort; prints from thread 0).
#if DLLAMA_DEBUG_ATTN
    if (ok && threadIndex == 0u) {
        static std::atomic<NnUint> ioPrinted{0u};
        const bool ioDbg = matmulIoDebugEnabledForOp(context->name);
        const NnUint ioLimit = matmulIoDebugLimit();
        const NnUint ioOutOff = matmulIoOutOffset();
        const NnUint ioOutLen = matmulIoOutLen();
        if (ioDbg) {
            const NnUint idx = ioPrinted.fetch_add(1u, std::memory_order_relaxed);
            if (ioLimit == 0u || idx < ioLimit) {
                // This fast path implies cView/aView offsets are 0 and cLen == outputSize.x.
                const float *outBase = (const float *)context->output[0];
                printMatmulOutSummaryF32(
                    context->name,
                    context->opIndex,
                    0u,
                    0u,
                    outBase,
                    context->outputSize.x,
                    0u,
                    context->outputSize.x,
                    ioOutOff,
                    ioOutLen);
            }
        }
    }
#endif

    return ok;
}

static void matmulForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
        return;

    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    const NnUint nActiveExpertsOr1 = std::max(config->nActiveExperts, 1u);
    const float *activeExpertIndexes = (const float *)context->buffers[config->activeExpertIndexesBufferIndex];

#if DLLAMA_DEBUG_ATTN
    static std::atomic<NnUint> ioPrinted{0u};
    const bool ioDbg = matmulIoDebugEnabledForOp(context->name);
    const NnUint ioLimit = matmulIoDebugLimit();
    const NnUint ioOutOff = matmulIoOutOffset();
    const NnUint ioOutLen = matmulIoOutLen();
#endif

    for (NnUint y = 0; y < batchSize; y++) {
        for (NnUint e = 0; e < nActiveExpertsOr1; e++) {
            const NnUint activeExpertIndex = config->nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[y * config->nActiveExperts + e];

            const NnTensorView *aView = &config->aView;
            const NnTensorView *cView = &config->cView;
            const NnUint aLen = (aView->sizeX == 0u) ? context->inputSize.x : aView->sizeX;
            const NnUint cLen = (cView->sizeX == 0u) ? context->outputSize.x : cView->sizeX;

            const bool viewAware = (aView->offset != 0u || aView->sizeX != 0u || cView->offset != 0u || cView->sizeX != 0u);
            const bool forceStyle = viewAware && (aView->strideX == 0u) && (cView->strideX == 0u);
            const bool offsetActive = (aView->offset != 0u || cView->offset != 0u);

            float *outputBase = (float *)context->output[e * context->outputSize.y + y];
            const float *xBase = (float *)context->input[e * context->inputSize.y + y];
            float *output = outputBase + cView->offset;
            const float *x = xBase + aView->offset;
            const float *wBase = (const float *)&context->weight[activeExpertIndex * context->weightSize.nBytesXY];

            if (threadIndex == 0u) {
                debugMatmulWeightReadRangeOnce(context, config, activeExpertIndex, cLen);
            }

            if (config->view == 0u) {
                matmul_F32_F32_F32(
                    output,
                    x,
                    wBase,
                    aLen,
                    cLen,
                    nThreads,
                    threadIndex);
            } else if (config->view == 1u) {
                const float *w = &wBase[config->outStart * context->weightSize.y];
                matmul_F32_F32_F32(
                    output,
                    x,
                    w,
                    context->weightSize.y,
                    cLen,
                    nThreads,
                    threadIndex);
            } else if (config->view == 2u) {
                matmul_F32_F32_F32_colSlice(
                    output,
                    x,
                    wBase,
                    context->weightSize.y,
                    cLen,
                    config->inStart,
                    aLen,
                    config->outStart,
                    nThreads,
                    threadIndex);
            } else {
                throw std::runtime_error("Unsupported matmul view mode");
            }
            DEBUG_VECTOR(context, "output", output);

#if DLLAMA_DEBUG_ATTN
            if (ioDbg && threadIndex == 0u) {
                const NnUint idx = ioPrinted.fetch_add(1u, std::memory_order_relaxed);
                if (ioLimit == 0u || idx < ioLimit) {
                    printMatmulOutSummaryF32(
                        context->name,
                        context->opIndex,
                        y,
                        e,
                        outputBase,
                        context->outputSize.x,
                        cView->offset,
                        cLen,
                        ioOutOff,
                        ioOutLen);
                }
            }
#endif
        }
    }
}

static void matmulForward_Q80_Q40_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
        return;

    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    const NnUint nActiveExpertsOr1 = std::max(config->nActiveExperts, 1u);
    const float *activeExpertIndexes = (const float *)context->buffers[config->activeExpertIndexesBufferIndex];

#if DLLAMA_DEBUG_ATTN
    static std::atomic<NnUint> ioPrinted{0u};
    const bool ioDbg = matmulIoDebugEnabledForOp(context->name);
    const NnUint ioLimit = matmulIoDebugLimit();
    const NnUint ioOutOff = matmulIoOutOffset();
    const NnUint ioOutLen = matmulIoOutLen();
#endif

    for (NnUint y = 0; y < batchSize; y++) {
        for (NnUint e = 0; e < nActiveExpertsOr1; e++) {
            const NnUint activeExpertIndex = config->nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[y * config->nActiveExperts + e];

            const NnTensorView *aView = &config->aView;
            const NnTensorView *cView = &config->cView;
            const NnUint aLen = (aView->sizeX == 0u) ? context->inputSize.x : aView->sizeX;
            const NnUint cLen = (cView->sizeX == 0u) ? context->outputSize.x : cView->sizeX;

            const bool viewAware = (aView->offset != 0u || aView->sizeX != 0u || cView->offset != 0u || cView->sizeX != 0u);
            const bool forceStyle = viewAware && (aView->strideX == 0u) && (cView->strideX == 0u);
            const bool offsetActive = (aView->offset != 0u || cView->offset != 0u);

            // Quantized inputs slice in full blocks.
            assert((aView->offset % Q80_BLOCK_SIZE) == 0u);
            assert((aLen % Q80_BLOCK_SIZE) == 0u);

            float *outputBase = (float *)context->output[e * context->outputSize.y + y];
            const NnBlockQ80 *xBase = (NnBlockQ80 *)context->input[e * context->inputSize.y + y];
            float *output = outputBase + cView->offset;
            const NnBlockQ80 *x = xBase + (aView->offset / Q80_BLOCK_SIZE);
            const NnBlockQ40 *wBase = (const NnBlockQ40 *)&context->weight[activeExpertIndex * context->weightSize.nBytesXY];

            if (threadIndex == 0u) {
                debugMatmulWeightReadRangeOnce(context, config, activeExpertIndex, cLen);
            }

            if (config->view == 0u) {
                matmul_Q80_Q40_F32(
                    output,
                    x,
                    wBase,
                    aLen,
                    cLen,
                    nThreads,
                    threadIndex);
            } else if (config->view == 1u) {
                const NnSize rowStrideBytes = getBytes(context->weightSize.floatType, context->weightSize.y);
                const NnBlockQ40 *w = (const NnBlockQ40 *)((const NnByte *)wBase + (NnSize)config->outStart * rowStrideBytes);
                matmul_Q80_Q40_F32(
                    output,
                    x,
                    w,
                    context->weightSize.y,
                    cLen,
                    nThreads,
                    threadIndex);
            } else if (config->view == 2u) {
                matmul_Q80_Q40_F32_colSlice(
                    output,
                    x,
                    wBase,
                    context->weightSize.y,
                    cLen,
                    config->inStart,
                    aLen,
                    config->outStart,
                    nThreads,
                    threadIndex);
            } else {
                throw std::runtime_error("Unsupported matmul view mode");
            }
            DEBUG_VECTOR(context, "output", output);

#if DLLAMA_DEBUG_ATTN
            if (ioDbg && threadIndex == 0u) {
                const NnUint idx = ioPrinted.fetch_add(1u, std::memory_order_relaxed);
                if (ioLimit == 0u || idx < ioLimit) {
                    printMatmulInSummaryQ80(context->name, context->opIndex, y, e, x, aLen);
                    printMatmulOutSummaryF32(
                        context->name,
                        context->opIndex,
                        y,
                        e,
                        outputBase,
                        context->outputSize.x,
                        cView->offset,
                        cLen,
                        ioOutOff,
                        ioOutLen);
                }
            }
#endif
        }
    }
}

static void siluForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    const NnSiluOpCodeConfig *config = (const NnSiluOpCodeConfig *)context->opConfig;
    const NnTensorView *view = &config->view;
    const NnUint strideX = (view->strideX == 0u) ? 1u : view->strideX;
    const NnUint len = (view->sizeX == 0u) ? context->outputSize.x : view->sizeX;
    const NnUint offset = view->offset;

    if (strideX == 1u) {
        assert(offset + len <= context->outputSize.x);
    } else {
        if (len > 0u)
            assert(offset + (len - 1u) * strideX < context->outputSize.x);
    }

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            float *output = (float *)context->output[z * context->outputSize.y + y];
            float *base = &output[offset];
            if (strideX == 1u) {
                silu_F32(base, len, nThreads, threadIndex);
            } else {
                silu_F32_strided(base, len, strideX, nThreads, threadIndex);
            }
        }
    }
}

static void geluForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    const NnGeluOpCodeConfig *config = (const NnGeluOpCodeConfig *)context->opConfig;
    const NnTensorView *view = &config->view;
    const NnUint strideX = (view->strideX == 0u) ? 1u : view->strideX;
    const NnUint len = (view->sizeX == 0u) ? context->outputSize.x : view->sizeX;
    const NnUint offset = view->offset;

    if (strideX == 1u) {
        assert(offset + len <= context->outputSize.x);
    } else {
        if (len > 0u)
            assert(offset + (len - 1u) * strideX < context->outputSize.x);
    }

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        float *base = &output[offset];
        if (strideX == 1u) {
            gelu_F32(base, len, nThreads, threadIndex);
        } else {
            gelu_F32_strided(base, len, strideX, nThreads, threadIndex);
        }
    }
}

static void initRopeForward_F32(NnCpuOpContext *context) {
    const NnRopeOpConfig *config = (NnRopeOpConfig *)context->opConfig;
    if (context->bufferFlags[config->ropeCacheBufferIndex] == 1)
        return;
    context->bufferFlags[config->ropeCacheBufferIndex] = 1;

    float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];
    fullfillRopeCache(config, cache);
}

static void ropeLlama_F32_view(
    float *xBase,
    const float *cache,
    const bool isQ,
    const NnUint pos,
    const NnRopeSlice *slice,
    const NnUint offset,
    const NnUint len,
    const NnUint nThreads,
    const NnUint threadIndex)
{
    assert((offset % 2u) == 0u);
    assert((len % 2u) == 0u);
    const NnUint shift = (isQ ? slice->qShift : 0u);
    const float *posCache = &cache[pos * slice->sliceDim + shift + offset];

    const NnUint pairCount = len / 2u;
    SPLIT_THREADS(s, e, pairCount, nThreads, threadIndex);
    const NnUint iStart = s * 2u;
    const NnUint iEnd = e * 2u;

    for (NnUint i = iStart; i < iEnd; i += 2u) {
        const float fcr = posCache[i];
        const float fci = posCache[i + 1u];
        const float v0 = xBase[i];
        const float v1 = xBase[i + 1u];

        xBase[i] = v0 * fcr - v1 * fci;
        xBase[i + 1u] = v0 * fci + v1 * fcr;
    }
}

static void ropeForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnRopeOpConfig *config = (NnRopeOpConfig *)context->opConfig;
    const NnRopeSlice *slice = &config->slice;
    const float *positions = (float *)context->pipes[config->positionPipeIndex];
    const float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];
    const bool isQ = config->isQ == 1;

    const NnTensorView *view = &config->view;
    const NnUint strideX = (view->strideX == 0u) ? 1u : view->strideX;
    const NnUint len = (view->sizeX == 0u) ? context->inputSize.x : view->sizeX;
    const NnUint offset = view->offset;

    assert(strideX == 1u);

#if DLLAMA_DEBUG_ATTN
    static std::atomic<NnUint> ropePrinted{0u};
    const bool ropeDbg = ropeIoDebugEnabledForOp(context->name);
    const NnUint ropeLimit = ropeIoDebugLimit();
    const NnUint ropeOff = ropeIoOffset();
    const NnUint ropeLen = ropeIoLen();
#endif

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *x = (float *)context->input[batchIndex];
        const NnUint pos = (NnUint)positions[batchIndex];

#if DLLAMA_DEBUG_ATTN
        if (ropeDbg && threadIndex == 0u) {
            const NnUint idx = ropePrinted.fetch_add(1u, std::memory_order_relaxed);
            if (ropeLimit == 0u || idx < ropeLimit) {
                printf("🔁 [rope][io] op=%s opIndex=%u batch=%u pos=%u isQ=%u viewOff=%u viewLen=%u\n",
                    (context->name ? context->name : "Unknown"),
                    (unsigned)context->opIndex,
                    (unsigned)batchIndex,
                    (unsigned)pos,
                    (unsigned)(isQ ? 1u : 0u),
                    (unsigned)offset,
                    (unsigned)len);
                printFloatSliceStats("rope-in", context->name, context->opIndex, x, context->inputSize.x, ropeOff, ropeLen);
            }
        }
#endif

        if (config->type == ROPE_LLAMA || config->type == ROPE_LLAMA3_1) {
            const NnUint dim0 = isQ ? slice->qDim0 : slice->kvDim0;
            assert(offset + len <= dim0);
            if (offset == 0u && len == dim0) {
                ropeLlama_F32(x, cache, isQ, pos, slice, nThreads, threadIndex);
            } else {
                ropeLlama_F32_view(x + offset, cache, isQ, pos, slice, offset, len, nThreads, threadIndex);
            }
        } else if (config->type == ROPE_FALCON) {
            const NnUint dim0 = isQ ? slice->qDim0 : slice->kvDim0;
            assert(offset + len <= dim0);
            if (offset == 0u && len == dim0) {
                ropeFalcon_F32(x, cache, isQ, pos, slice, nThreads, threadIndex);
            } else {
                ropeFalcon_F32_view(x, cache, isQ, pos, slice, offset, len, nThreads, threadIndex);
            }
        } else {
            throw std::runtime_error("Unsupported rope type");
        }

#if DLLAMA_DEBUG_ATTN
        if (ropeDbg && threadIndex == 0u) {
            const NnUint idx = ropePrinted.load(std::memory_order_relaxed);
            // Use the same budget as the pre-print.
            if (ropeLimit == 0u || (idx - 1u) < ropeLimit) {
                printFloatSliceStats("rope-out", context->name, context->opIndex, x, context->inputSize.x, ropeOff, ropeLen);
            }
        }
#endif
    }
}

static void initMultiHeadAttForward(NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->outputSize.x, config->qSliceD0);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    NnSize3D *querySize = &context->bufferConfigs[config->queryBufferIndex].size;
    const NnUint qStride = config->qStride == 0u ? config->qSliceD0 : config->qStride;
    ASSERT_EQ(querySize->x, qStride);
    NnSize3D *posSize = &context->pipeConfigs[config->positionPipeIndex].size;
    ASSERT_EQ(posSize->x, 1);
    ASSERT_EQ(posSize->y, context->nBatches);
}

static void multiHeadAttForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    float *query = (float *)context->buffers[config->queryBufferIndex];
    float *keyCache = (float *)context->buffers[config->keyCacheBufferIndex];
    float *valueCache = (float *)context->buffers[config->valueCacheBufferIndex];
    float *att = (float *)context->buffers[config->attBufferIndex];
    const float *positions = (float *)context->pipes[config->positionPipeIndex];

#if DLLAMA_DEBUG_ATTN
    static std::atomic<NnUint> dbgPrinted{0u};
    const bool dbg = kvCacheDebugEnabled();
    const NnUint dbgLimit = kvCacheDebugLimit();
    
    static std::atomic<NnUint> kvPerHeadPrinted{0u};
    const bool kvPerHeadDbg = kvCachePerHeadEnabled();
    const NnUint kvPerHeadLimit = kvCachePerHeadLimit();
    const NnUint kvPerHeadBatchSel = kvCachePerHeadBatch();
    const long kvPerHeadKvHeadSel = kvCachePerHeadSelectKvHead();
    const char *kvPerHeadKvHeadsSpec = kvCachePerHeadSelectKvHeadsSpec();
    const long kvPerHeadPosSel = kvCachePerHeadSelectPos();
    const NnUint kvPerHeadDimsSel = kvCachePerHeadDims();
    const bool kvPerHeadGlobal = kvCachePerHeadGlobalEnabled();
    
    static std::atomic<NnUint> attPrinted{0u};
    const bool attDbg = attDebugEnabled();
    const NnUint attLimit = attDebugLimit();
    const NnUint attBatch = attDebugBatch();
    const long attHeadSel = attDebugHead();
    const bool attScores = attScoresEnabled();
    const NnUint attScoresLen = attScoresMaxLen();
    const bool attQkDbg = attQkEnabled();
    const NnUint attQkDimsSel = attQkDims();
    const NnUint attQkTopKSel = attQkTopK();
    const long attQkPosSel = attQkForcePos();
#endif

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *y = (float *)context->output[batchIndex];
        const NnUint qStride = config->qStride == 0u ? config->qSliceD0 : config->qStride;
        float *q = &query[batchIndex * qStride + config->qStart];
        NnUint pos = (NnUint)positions[batchIndex];
        assert(pos < config->seqLen);

        const NnUint qHeadStart = (config->headDim != 0u) ? (config->qStart / config->headDim) : 0u;

        const NnUint kvStride = config->kvStride == 0u ? config->kvDim0 : config->kvStride;

        // Debug: print K/V cache values per owned KV-head on this node.
        // Enable with: DLLAMA_DEBUG_KVCACHE_PER_HEAD=1
        // Optional:
        // - DLLAMA_DEBUG_KVCACHE_PER_HEAD_FILTER=substring (match op name)
        // - DLLAMA_DEBUG_KVCACHE_PER_HEAD_LIMIT=N (default 32, 0 unlimited)
        // - DLLAMA_DEBUG_KVCACHE_PER_HEAD_BATCH=N (default 0)
        // - DLLAMA_DEBUG_KVCACHE_PER_HEAD_POS=P (default -1 = all)
        // - DLLAMA_DEBUG_KVCACHE_PER_HEAD_KVHEAD=H (default -1 = all owned)
        // - DLLAMA_DEBUG_KVCACHE_PER_HEAD_DIMS=D (default 8, 0 = full headDim)
    #if DLLAMA_DEBUG_ATTN
        if (kvPerHeadDbg && threadIndex == 0u && batchIndex == kvPerHeadBatchSel && kvCachePerHeadPassesFilter(context->name)) {
            if (kvPerHeadPosSel < 0 || (NnUint)kvPerHeadPosSel == pos) {
                const NnUint idx = kvPerHeadPrinted.fetch_add(1u, std::memory_order_relaxed);
                if (kvPerHeadLimit == 0u || idx < kvPerHeadLimit) {
                    const NnUint headDim = config->headDim;
                    const NnUint dimsToPrint = (kvPerHeadDimsSel == 0u) ? headDim : std::min(headDim, kvPerHeadDimsSel);
                    if (headDim == 0u || dimsToPrint == 0u) {
                        printf("🧠 [kvcache][kvhead][skip] node=%u op=%s opIndex=%u batch=%u pos=%u headDim=%u (invalid)\n",
                            (unsigned)context->nodeIndex,
                            (context->name ? context->name : "Unknown"),
                            (unsigned)context->opIndex,
                            (unsigned)batchIndex,
                            (unsigned)pos,
                            (unsigned)config->headDim);
                        std::fflush(stdout);
                    } else if (kvPerHeadGlobal) {
                        // Global kvHead print path: requires full KV cache buffers (global stride).
                        if (config->kvStride == 0u) {
                            printf("🧠 [kvcache][kvhead][skip] node=%u op=%s opIndex=%u batch=%u pos=%u global=1 requires full KV buffers (kvStride!=0)\n",
                                (unsigned)context->nodeIndex,
                                (context->name ? context->name : "Unknown"),
                                (unsigned)context->opIndex,
                                (unsigned)batchIndex,
                                (unsigned)pos);
                            std::fflush(stdout);
                        } else {
                            const NnUint globalKvDim = config->kvStride;
                            const NnUint maxHeadsByStride = (headDim != 0u) ? (globalKvDim / headDim) : 0u;
                            const NnUint nHeadsToScan = std::min(config->nKvHeads, maxHeadsByStride);
                            for (NnUint globalKvHead = 0u; globalKvHead < nHeadsToScan; ++globalKvHead) {
                                if (!kvHeadMatchesSelection(globalKvHead, kvPerHeadKvHeadsSpec, kvPerHeadKvHeadSel)) continue;
                                const NnUint col0 = globalKvHead * headDim;
                                const float *kPtr = &keyCache[pos * kvStride + col0];
                                const float *vPtr = &valueCache[pos * kvStride + col0];

                                printf("🧠 [kvcache][kvhead] node=%u op=%s opIndex=%u batch=%u pos=%u kvHead=%u colStart=%u dims=%u K=",
                                    (unsigned)context->nodeIndex,
                                    (context->name ? context->name : "Unknown"),
                                    (unsigned)context->opIndex,
                                    (unsigned)batchIndex,
                                    (unsigned)pos,
                                    (unsigned)globalKvHead,
                                    (unsigned)col0,
                                    (unsigned)dimsToPrint);
                                for (NnUint d = 0u; d < dimsToPrint; ++d) {
                                    printf("%s%.6f", (d == 0u ? "[" : ","), (double)kPtr[d]);
                                }
                                printf("] V=");
                                for (NnUint d = 0u; d < dimsToPrint; ++d) {
                                    printf("%s%.6f", (d == 0u ? "[" : ","), (double)vPtr[d]);
                                }
                                printf("]\n");
                            }
                            std::fflush(stdout);
                        }
                    } else {
                        // Owned kvHead slice print path (existing behavior).
                        const NnUint kvStart = config->kvStart;
                        const NnUint kvDim0 = config->kvDim0;
                        if ((kvStart % headDim) == 0u && (kvDim0 % headDim) == 0u) {
                            const NnUint kvHeadStart = kvStart / headDim;
                            const NnUint kvHeadLen = kvDim0 / headDim;
                            for (NnUint localKv = 0u; localKv < kvHeadLen; ++localKv) {
                                const NnUint globalKvHead = kvHeadStart + localKv;
                                if (!kvHeadMatchesSelection(globalKvHead, kvPerHeadKvHeadsSpec, kvPerHeadKvHeadSel)) continue;
                                const NnUint col0 = kvStart + localKv * headDim;
                                const float *kPtr = &keyCache[pos * kvStride + col0];
                                const float *vPtr = &valueCache[pos * kvStride + col0];

                                printf("🧠 [kvcache][kvhead] node=%u op=%s opIndex=%u batch=%u pos=%u kvHead=%u colStart=%u dims=%u K=",
                                    (unsigned)context->nodeIndex,
                                    (context->name ? context->name : "Unknown"),
                                    (unsigned)context->opIndex,
                                    (unsigned)batchIndex,
                                    (unsigned)pos,
                                    (unsigned)globalKvHead,
                                    (unsigned)col0,
                                    (unsigned)dimsToPrint);
                                for (NnUint d = 0u; d < dimsToPrint; ++d) {
                                    printf("%s%.6f", (d == 0u ? "[" : ","), (double)kPtr[d]);
                                }
                                printf("] V=");
                                for (NnUint d = 0u; d < dimsToPrint; ++d) {
                                    printf("%s%.6f", (d == 0u ? "[" : ","), (double)vPtr[d]);
                                }
                                printf("]\n");
                            }
                            std::fflush(stdout);
                        } else {
                            printf("🧠 [kvcache][kvhead][skip] node=%u op=%s opIndex=%u batch=%u pos=%u headDim=%u kvStart=%u kvDim0=%u (unaligned)\n",
                                (unsigned)context->nodeIndex,
                                (context->name ? context->name : "Unknown"),
                                (unsigned)context->opIndex,
                                (unsigned)batchIndex,
                                (unsigned)pos,
                                (unsigned)config->headDim,
                                (unsigned)config->kvStart,
                                (unsigned)config->kvDim0);
                            std::fflush(stdout);
                        }
                    }
                }
            }
        }

    #endif // DLLAMA_DEBUG_ATTN

        DEBUG_VECTOR(context, "input", y);
        DEBUG_VECTOR(context, "q", q);

        multiheadAtt_F32(y, q, 
            &att[batchIndex * config->nHeads0 * config->seqLen],
            keyCache, valueCache, pos,
            config->nHeads, config->nHeads0,
            config->nKvHeads,
            config->kvDim0, config->kvStart, kvStride,
            qHeadStart,
            config->headDim, config->seqLen, nThreads, threadIndex);

        // Attention debug: print per forward position (after softmax).
        // Enable with: DLLAMA_DEBUG_ATT=1
        // Optional:
        // - DLLAMA_DEBUG_ATT_FILTER=substring (only ops whose name matches)
        // - DLLAMA_DEBUG_ATT_BATCH=N (default 0)
        // - DLLAMA_DEBUG_ATT_HEAD=H (default 0), -1 prints all local heads
        // - DLLAMA_DEBUG_ATT_LIMIT=N (0 unlimited)
    #if DLLAMA_DEBUG_ATTN
        if (attDbg && threadIndex == 0u && batchIndex == attBatch && attDebugPassesFilter(context->name)) {
            const NnUint idx = attPrinted.fetch_add(1u, std::memory_order_relaxed);
            if (attLimit == 0u || idx < attLimit) {
                const NnUint len = pos + 1u;
                const float *attBatchPtr = &att[batchIndex * config->nHeads0 * config->seqLen];
                const NnUint globalQHeadStart = (config->headDim != 0u) ? (config->qStart / config->headDim) : 0u;
                const NnUint gqaGroup = (config->nKvHeads != 0u && (config->nHeads % config->nKvHeads) == 0u)
                    ? (config->nHeads / config->nKvHeads)
                    : 0u;

                    auto printHead = [&](NnUint localHead) {
                    if (localHead >= config->nHeads0) return;
                    const NnUint globalQHead = globalQHeadStart + localHead;
                    const NnUint globalKvHead = (gqaGroup != 0u) ? (globalQHead / gqaGroup) : 0u;
                    const float *row = &attBatchPtr[localHead * config->seqLen];
                    printf("🧪 [att][pos] op=%s opIndex=%u pos=%u batch=%u qHead=%u kvHead=%u qStart=%u kvStart=%u kvDim0=%u kvStride=%u\n",
                        (context->name ? context->name : "Unknown"),
                        (unsigned)context->opIndex,
                        (unsigned)pos,
                        (unsigned)batchIndex,
                        (unsigned)globalQHead,
                        (unsigned)globalKvHead,
                        (unsigned)config->qStart,
                        (unsigned)config->kvStart,
                        (unsigned)config->kvDim0,
                        (unsigned)(config->kvStride == 0u ? config->kvDim0 : config->kvStride));
                    printAttRowSummary(row, len, globalQHead, globalKvHead);
                    if (attScores) {
                        printAttRowValues(row, len, attScoresLen);
                    }
                    if (attQkDbg) {
                        const NnUint headDim = config->headDim;
                        if (headDim == 0u) {
                            printf("🧪 [att][qk][skip] headDim=0\n");
                            return;
                        }

                        const NnUint dimsToPrint = (attQkDimsSel == 0u) ? headDim : std::min(headDim, attQkDimsSel);
                        const float *qVec = q + localHead * headDim;

                        printf("🧪 [att][qk][ptr] query=%p qVec=%p keyCache=%p\n",
                            (const void *)query,
                            (const void *)qVec,
                            (const void *)keyCache);

                        printf("🧪 [att][qk] qHead=%u kvHead=%u headDim=%u ",
                            (unsigned)globalQHead,
                            (unsigned)globalKvHead,
                            (unsigned)headDim);
                        printVecDims("Q", qVec, headDim, dimsToPrint);
                        printf(" qNorm=%.6f\n", std::sqrt(l2NormSq_F32(qVec, headDim)));

                        auto kvColStart = [&](NnUint gKvHead, NnUint &col0Out) -> bool {
                            // Prefer full-stride addressing when kvStride is configured.
                            if (config->kvStride != 0u) {
                                const NnUint stride = config->kvStride;
                                const NnUint needed = (gKvHead + 1u) * headDim;
                                if (needed <= stride) {
                                    col0Out = gKvHead * headDim;
                                    return true;
                                }
                            }
                            // Fall back to slice addressing (owned heads only).
                            if ((config->kvStart % headDim) != 0u || (config->kvDim0 % headDim) != 0u) return false;
                            const NnUint sliceHeadStart = config->kvStart / headDim;
                            const NnUint sliceHeadLen = config->kvDim0 / headDim;
                            if (gKvHead < sliceHeadStart) return false;
                            const NnUint localKv = gKvHead - sliceHeadStart;
                            if (localKv >= sliceHeadLen) return false;
                            col0Out = config->kvStart + localKv * headDim;
                            return true;
                        };

                        std::vector<NnUint> idxs;
                        if (attQkPosSel >= 0) {
                            const NnUint p = (NnUint)attQkPosSel;
                            if (p < len) idxs.push_back(p);
                        } else {
                            selectTopKIndices(row, len, attQkTopKSel, idxs);
                        }

                        NnUint col0 = 0u;
                        if (!kvColStart(globalKvHead, col0)) {
                            printf("🧪 [att][qk][skip] cannot map kvHead=%u to K colStart (kvStart=%u kvDim0=%u kvStride=%u headDim=%u)\n",
                                (unsigned)globalKvHead,
                                (unsigned)config->kvStart,
                                (unsigned)config->kvDim0,
                                (unsigned)(config->kvStride == 0u ? config->kvDim0 : config->kvStride),
                                (unsigned)headDim);
                            return;
                        }

                        const double invSqrtHd = 1.0 / std::sqrt((double)headDim);
                        for (NnUint t = 0u; t < idxs.size(); ++t) {
                            const NnUint p = idxs[t];
                            const float w = row[p];
                            const float *kVec = &keyCache[p * kvStride + col0];
                            const float dot = dotProduct_F32(qVec, kVec, headDim);
                            const double kNorm = std::sqrt(l2NormSq_F32(kVec, headDim));

                            printf("🧪 [att][qk] pos=%u w=%.6f colStart=%u dot=%.6f dotScaled=%.6f kNorm=%.6f ",
                                (unsigned)p,
                                (double)w,
                                (unsigned)col0,
                                (double)dot,
                                (double)((double)dot * invSqrtHd),
                                (double)kNorm);
                            printVecDims("K", kVec, headDim, dimsToPrint);
                            printf("\n");
                        }
                    }
                };

                if (attHeadSel < 0) {
                    for (NnUint h = 0u; h < config->nHeads0; ++h) {
                        printHead(h);
                    }
                } else {
                    printHead((NnUint)attHeadSel);
                }
                std::fflush(stdout);
            }
        }

        // Debug: print a small attention summary for batch0/head0 at current pos.
        // This helps diagnose repetition / quality drops after repartition by checking whether
        // attention collapses to a single token or becomes inconsistent.
        if (dbg && dbgLimit != 0u && threadIndex == 0u && batchIndex == 0u) {
            const NnUint printed = dbgPrinted.fetch_add(1u, std::memory_order_relaxed);
            if (printed < dbgLimit) {
                const NnUint len = pos + 1u;
                const float *attBatch = &att[batchIndex * config->nHeads0 * config->seqLen];

                const NnUint localHead = 0u;
                const NnUint globalQHeadStart = (config->headDim != 0u) ? (config->qStart / config->headDim) : 0u;
                const NnUint globalQHead = globalQHeadStart + localHead;
                const NnUint globalKvHead = (config->nKvHeads != 0u && config->nHeads % config->nKvHeads == 0u)
                    ? (globalQHead / (config->nHeads / config->nKvHeads))
                    : 0u;

                const float *row = &attBatch[localHead * config->seqLen];
                printf("🧪 [att] op=%s opIndex=%u pos=%u qStart=%u kvStart=%u kvDim0=%u\n",
                    context->name,
                    (unsigned)context->opIndex,
                    (unsigned)pos,
                    (unsigned)config->qStart,
                    (unsigned)config->kvStart,
                    (unsigned)config->kvDim0);
                printAttRowSummary(row, len, globalQHead, globalKvHead);
                std::fflush(stdout);
            }
        }

	#endif // DLLAMA_DEBUG_ATTN

        DEBUG_VECTOR(context, "output", y);
    }
}

static void initMulForward(NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    ASSERT_EQ(context->inputSize.z, context->outputSize.z);

    const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)context->opConfig;
    const NnTensorView *view = &config->view;
    const NnUint strideX = (view->strideX == 0u) ? 1u : view->strideX;
    const NnUint len = (view->sizeX == 0u) ? context->outputSize.x : view->sizeX;
    const NnUint offset = view->offset;

    assert(view->sizeY == 0u);
    assert(view->strideY == 0u);

    // Output buffer bounds (sliced output is allowed).
    if (strideX == 1u) {
        assert(offset + len <= context->outputSize.x);
    } else {
        if (len > 0u)
            assert(offset + (len - 1u) * strideX < context->outputSize.x);
    }

    // Multiplier buffer uses its own per-row stride (often the full global FFN dim).
    // In full-buffer residency mode, outputSize.x may be a local slice length; using it
    // as the multiplier row stride would read the wrong region and silently corrupt results.
    const NnUint multRowStride = context->bufferConfigs[config->multiplierBufferIndex].size.x;
    assert(multRowStride > 0u);
    if (strideX == 1u) {
        assert(offset + len <= multRowStride);
    } else {
        if (len > 0u)
            assert(offset + (len - 1u) * strideX < multRowStride);
    }
}

static void mulForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)context->opConfig;
    const float *multiplier = (float *)context->buffers[config->multiplierBufferIndex];

    const NnUint multRowStride = context->bufferConfigs[config->multiplierBufferIndex].size.x;

    const NnTensorView *view = &config->view;
    const NnUint strideX = (view->strideX == 0u) ? 1u : view->strideX;
    const NnUint len = (view->sizeX == 0u) ? context->outputSize.x : view->sizeX;
    const NnUint offset = view->offset;

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            float *outBase = (float *)context->output[zOffset + y];
            const float *inBase = (float *)context->input[zOffset + y];
            const float *mBase = &multiplier[multRowStride * (zOffset + y)];

            float *out = &outBase[offset];
            const float *in = &inBase[offset];
            const float *m = &mBase[offset];

            if (strideX == 1u) {
                mul_F32(out, in, m, len, nThreads, threadIndex);
            } else {
                mul_F32_strided(out, in, m, len, strideX, nThreads, threadIndex);
            }
        }
    }
}

static void scaleForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnScaleOpCodeConfig *config = (NnScaleOpCodeConfig *)context->opConfig;
    const float *scale = (float *)context->buffers[config->scaleBufferIndex];

    const NnTensorView *view = &config->view;
    const NnUint strideX = (view->strideX == 0u) ? 1u : view->strideX;
    const NnUint len = (view->sizeX == 0u) ? context->inputSize.x : view->sizeX;
    const NnUint offset = view->offset;

    assert(view->sizeY == 0u);
    assert(view->strideY == 0u);

    if (strideX == 1u) {
        assert(offset + len <= context->inputSize.x);
    } else {
        if (len > 0u)
            assert(offset + (len - 1u) * strideX < context->inputSize.x);
    }

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            const NnUint index = z * context->inputSize.y + y;
            const float s = scale[index];
            const float *i = (float *)context->input[index];
            float *o = (float *)context->output[index];

            const float *iBase = &i[offset];
            float *oBase = &o[offset];
            if (strideX == 1u) {
                scale_F32(iBase, oBase, s, len, nThreads, threadIndex);
            } else {
                scale_F32_strided(iBase, oBase, s, len, strideX, nThreads, threadIndex);
            }
        }
    }
}

static void initCastForward(NnCpuOpContext *context) {
    // ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    // ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    // ASSERT_EQ(context->inputSize.z, context->outputSize.z);
}

static void castForward_ANY(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnCastOpCodeConfig *config = (NnCastOpCodeConfig *)context->opConfig;

    const NnTensorView *view = &config->view;
    const NnUint strideX = (view->strideX == 0u) ? 1u : view->strideX;
    const NnUint len = (view->sizeX == 0u) ? context->outputSize.x : view->sizeX;
    const NnUint offset = view->offset;

    assert(view->sizeY == 0u);
    assert(view->strideY == 0u);

    const NnSize blockSize = getBlockSize(context->outputSize.floatType);
    if (blockSize > 1u) {
        // Quantized types must slice in full blocks.
        assert(strideX == 1u);
        assert((offset % blockSize) == 0u);
        assert((len % blockSize) == 0u);
        assert(offset + len <= context->outputSize.x);
        assert(offset + len <= context->inputSize.x);
    } else {
        if (strideX == 1u) {
            assert(offset + len <= context->outputSize.x);
            assert(offset + len <= context->inputSize.x);
        } else {
            if (len > 0u) {
                assert(offset + (len - 1u) * strideX < context->outputSize.x);
                assert(offset + (len - 1u) * strideX < context->inputSize.x);
            }
        }
    }

    const NnSize offsetBytes = getBytes(context->outputSize.floatType, offset);
    const NnSize dimBytes = getBytes(context->outputSize.floatType, len);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            NnByte *oBase = context->output[zOffset + y];
            NnByte *iBase = context->input[zOffset + y];

            if (blockSize > 1u || strideX == 1u) {
                copy_UNK(
                    oBase + offsetBytes,
                    iBase + offsetBytes,
                    dimBytes,
                    nThreads,
                    threadIndex);
            } else {
                // Correctness-first strided element-wise copy.
                // Note: only meaningful for non-quantized types (blockSize==1).
                if (context->outputSize.floatType == F_32) {
                    float *o = (float *)oBase;
                    const float *i = (const float *)iBase;
                    for (NnUint t = threadIndex; t < len; t += nThreads) {
                        const NnUint xIndex = offset + t * strideX;
                        o[xIndex] = i[xIndex];
                    }
                } else if (context->outputSize.floatType == F_16) {
                    NnFp16 *o = (NnFp16 *)oBase;
                    const NnFp16 *i = (const NnFp16 *)iBase;
                    for (NnUint t = threadIndex; t < len; t += nThreads) {
                        const NnUint xIndex = offset + t * strideX;
                        o[xIndex] = i[xIndex];
                    }
                } else {
                    // Fallback: treat as bytes with element size 1 (should not happen).
                    assert(false && "Unsupported float type for strided castForward_ANY");
                }
            }
        }
    }
}

static void castForward_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_Q80);

    const NnCastOpCodeConfig *config = (NnCastOpCodeConfig *)context->opConfig;
    const NnTensorView *view = &config->view;
    const NnUint strideX = (view->strideX == 0u) ? 1u : view->strideX;
    const NnUint len = (view->sizeX == 0u) ? context->outputSize.x : view->sizeX;
    const NnUint offset = view->offset;

    assert(view->sizeY == 0u);
    assert(view->strideY == 0u);

    // For Q80, only support contiguous slicing aligned to block size.
    assert(strideX == 1u);
    assert((offset % Q80_BLOCK_SIZE) == 0u);
    assert((len % Q80_BLOCK_SIZE) == 0u);
    assert(offset + len <= context->outputSize.x);
    assert(offset + len <= context->inputSize.x);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            quantizeF32toQ80(
                ((float *)context->input[zOffset + y]) + offset,
                ((NnBlockQ80 *)context->output[zOffset + y]) + (offset / Q80_BLOCK_SIZE),
                len,
                nThreads,
                threadIndex);
        }
    }
}

static void castForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);
    ASSERT_EQ(context->outputSize.floatType, F_32);

    const NnCastOpCodeConfig *config = (NnCastOpCodeConfig *)context->opConfig;
    const NnTensorView *view = &config->view;
    const NnUint strideX = (view->strideX == 0u) ? 1u : view->strideX;
    const NnUint len = (view->sizeX == 0u) ? context->outputSize.x : view->sizeX;
    const NnUint offset = view->offset;

    assert(view->sizeY == 0u);
    assert(view->strideY == 0u);

    // For Q80, only support contiguous slicing aligned to block size.
    assert(strideX == 1u);
    assert((offset % Q80_BLOCK_SIZE) == 0u);
    assert((len % Q80_BLOCK_SIZE) == 0u);
    assert(offset + len <= context->outputSize.x);
    assert(offset + len <= context->inputSize.x);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            dequantizeQ80toF32(
                ((NnBlockQ80 *)context->input[zOffset + y]) + (offset / Q80_BLOCK_SIZE),
                ((float *)context->output[zOffset + y]) + offset,
                len,
                nThreads,
                threadIndex);
        }
    }
}

static void initRepeatZForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    ASSERT_EQ(context->inputSize.z, 1u);
    assert(context->inputSize.z <= context->outputSize.z);
    assert(context->inputSize.z > 0u);
}

static void repeatZForward_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_Q80);
    const NnSize dimSize = getBytes(F_Q80, context->outputSize.x);

    for (NnUint z = 0u; z < context->outputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            NnByte *output = context->output[z * context->outputSize.y + y];
            if (z == 0u) {
                quantizeF32toQ80(
                    (float *)context->input[y],
                    (NnBlockQ80 *)output,
                    context->outputSize.x,
                    nThreads,
                    threadIndex);
            } else {
                copy_UNK(
                    output,
                    context->output[y],
                    dimSize,
                    nThreads,
                    threadIndex);
            }
        }
    }
}

static void shiftForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    // OP_SHIFT is often used to write per-step K/V into a KV cache.
    // In full-buffer + per-node-slice mode, the input pointer may be PNTR_BATCHED_SLICE
    // (non-contiguous), which is fine because we already copy per batchIndex.
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.y, 1);

    const NnShiftOpCodeConfig *config = (NnShiftOpCodeConfig *)context->opConfig;
    const float *indexes = (float *)context->pipes[config->indexPipeIndex];
    const NnSize dimBytes = getBytes(F_32, context->inputSize.x);
    NnByte *output = context->output[0];

#if DLLAMA_DEBUG_ATTN
    // Optional debug: print KV cache values per KV-head written by this shift.
    // This is the right place to observe KV redundancy compute (kvHeadComputeSplit), because
    // MHA reads only owned kvHeadSplit but SHIFT may write a wider range.
    const bool kvPerHeadDbg = kvCachePerHeadEnabled() && kvCachePerHeadShiftEnabled() && kvCachePerHeadPassesFilter(context->name);
    const NnUint kvPerHeadLimit = kvCachePerHeadLimit();
    const long kvPerHeadKvHeadSel = kvCachePerHeadSelectKvHead();
    const long kvPerHeadPosSel = kvCachePerHeadSelectPos();
    const NnUint kvPerHeadDimsSel = kvCachePerHeadDims();
    const NnUint headDimOverride = kvCachePerHeadHeadDimOverride();
    static std::atomic<NnUint> kvShiftPrinted{0u};

    auto isShiftK = [&]() -> bool {
        return context->name != nullptr && std::strstr(context->name, "shift_k") != nullptr;
    };
    auto isShiftV = [&]() -> bool {
        return context->name != nullptr && std::strstr(context->name, "shift_v") != nullptr;
    };
#endif

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        const NnSize index = (NnSize)indexes[batchIndex];
        if (config->dstRowStride == 0u) {
            assert((index + 1) * context->inputSize.x <= context->outputSize.x);
            copy_UNK(
                &output[index * dimBytes],
                context->input[batchIndex],
                dimBytes,
                nThreads,
                threadIndex);
        } else {
            const NnSize rowStrideBytes = getBytes(F_32, config->dstRowStride);
            const NnSize colStartBytes = getBytes(F_32, config->dstColStart);
            const NnSize totalRows = context->outputSize.x / config->dstRowStride;
            assert(index < totalRows);
            assert(config->dstColStart + context->inputSize.x <= config->dstRowStride);
            copy_UNK(
                &output[index * rowStrideBytes + colStartBytes],
                context->input[batchIndex],
                dimBytes,
                nThreads,
                threadIndex);

#if DLLAMA_DEBUG_ATTN
            // Print after the write so we read from the actual cache.
            if (kvPerHeadDbg && threadIndex == 0u) {
                if (kvPerHeadPosSel < 0 || (NnUint)kvPerHeadPosSel == (NnUint)index) {
                    const NnUint printed = kvShiftPrinted.fetch_add(1u, std::memory_order_relaxed);
                    if (kvPerHeadLimit == 0u || printed < kvPerHeadLimit) {
                        NnUint headDim = config->dstColStartUnit;
                        if (headDim == 0u) headDim = headDimOverride;
                        if (headDim == 0u) {
                            printf("🧠 [kvcache][kvhead][shift][skip] node=%u op=%s opIndex=%u pos=%u need headDim; set DLLAMA_DEBUG_KVCACHE_PER_HEAD_HEADDIM=... (dstColStartUnit=0)\n",
                                (unsigned)context->nodeIndex,
                                (context->name ? context->name : "Unknown"),
                                (unsigned)context->opIndex,
                                (unsigned)index);
                            std::fflush(stdout);
                        } else if ((config->dstColStart % headDim) != 0u || (context->inputSize.x % headDim) != 0u) {
                            printf("🧠 [kvcache][kvhead][shift][skip] node=%u op=%s opIndex=%u pos=%u unaligned headDim=%u dstColStart=%u writeLen=%u\n",
                                (unsigned)context->nodeIndex,
                                (context->name ? context->name : "Unknown"),
                                (unsigned)context->opIndex,
                                (unsigned)index,
                                (unsigned)headDim,
                                (unsigned)config->dstColStart,
                                (unsigned)context->inputSize.x);
                            std::fflush(stdout);
                        } else {
                            const NnUint kvHeadStart = config->dstColStart / headDim;
                            const NnUint kvHeadLen = context->inputSize.x / headDim;
                            const NnUint dimsToPrint = (kvPerHeadDimsSel == 0u) ? headDim : std::min(headDim, kvPerHeadDimsSel);
                            const float *row = (const float *)(output + index * rowStrideBytes);
                            const char *kind = isShiftK() ? "K" : (isShiftV() ? "V" : "KV");

                            for (NnUint localKv = 0u; localKv < kvHeadLen; ++localKv) {
                                const NnUint globalKvHead = kvHeadStart + localKv;
                                if (kvPerHeadKvHeadSel >= 0 && (NnUint)kvPerHeadKvHeadSel != globalKvHead) continue;
                                const NnUint col0 = config->dstColStart + localKv * headDim;
                                const float *ptr = row + col0;

                                printf("🧠 [kvcache][kvhead][shift] node=%u op=%s opIndex=%u batch=%u pos=%u kvHead=%u colStart=%u dims=%u %s=",
                                    (unsigned)context->nodeIndex,
                                    (context->name ? context->name : "Unknown"),
                                    (unsigned)context->opIndex,
                                    (unsigned)batchIndex,
                                    (unsigned)index,
                                    (unsigned)globalKvHead,
                                    (unsigned)col0,
                                    (unsigned)dimsToPrint,
                                    kind);
                                for (NnUint d = 0u; d < dimsToPrint; ++d) {
                                    printf("%s%.6f", (d == 0u ? "[" : ","), (double)ptr[d]);
                                }
                                printf("]\n");
                            }
                            std::fflush(stdout);
                        }
                    }
                }
            }
#endif
        }
    }
}

static void softmaxForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(*context->input == *context->output);

    const NnSoftmaxOpCodeConfig *config = (const NnSoftmaxOpCodeConfig *)context->opConfig;
    const NnTensorView *view = &config->view;
    const NnUint strideX = (view->strideX == 0u) ? 1u : view->strideX;
    const NnUint len = (view->sizeX == 0u) ? context->outputSize.x : view->sizeX;
    const NnUint offset = view->offset;

    if (strideX == 1u) {
        assert(offset + len <= context->outputSize.x);
    } else {
        if (len > 0u)
            assert(offset + (len - 1u) * strideX < context->outputSize.x);
    }

    for (NnUint y = threadIndex; y < batchSize; y += nThreads)
    {
        float *out = (float *)context->output[y];
        float *base = &out[offset];
        if (strideX == 1u) {
            softmax_F32(base, len);
        } else {
            softmax_F32_strided(base, len, strideX);
        }
    }
}

static void initMoeGateForward(NnCpuOpContext *context) {
    const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.z, 1u);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    assert(context->inputSize.x >= config->k);
    ASSERT_EQ(context->outputSize.z, config->k);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.x, 1u);
}

static void moeGateForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)context->opConfig;
    float *indexes = (float *)context->buffers[config->indexesBufferIndex];

    std::vector<NnUint> pos(config->k);
    for (NnUint y = threadIndex; y < batchSize; y += nThreads) {
        float *input = (float *)context->input[y];

        topk_F32(input, pos.data(), context->inputSize.x, config->k);

        float sum;
        if (config->normTopk == 1u) {
            sum = 0.0f;
            for (NnUint i = 0u; i < config->k; i++)
                sum += input[pos[i]];
        } else {
            sum = 1.0f;
        }

        for (NnUint k = 0u; k < config->k; k++) {
            const NnUint p = pos[k];
            indexes[y * config->k + k] = (float)p;

            // (nActiveExperts, nBatches, 1)
            float *output = (float *)context->output[k * context->outputSize.y + y];
            *output = input[p] / sum;
        }

        DEBUG_VECTOR(context, "indexes", (&indexes[y * config->k]));
    }
}

// device

void printCpuInstructionSet() {
    printf("🧠 CPU:");
#if defined(__ARM_NEON)
    printf(" neon");
#if defined(__ARM_FEATURE_DOTPROD)
    printf(" dotprod");
#endif
#if defined(__ARM_FP16_FORMAT_IEEE)
    printf(" fp16");
#endif
#endif
#if defined(__AVX2__)
    printf(" avx2");
#endif
#if defined(__AVX512F__)
    printf(" avx512f");
#endif
    printf("\n");
}

NnCpuOpForwardInit getCpuOpForwardInit(NnOpCode code, NnOpQuantType quantType) {
    if (code == OP_EMBEDDING)
        return initEmbeddingForward;
    if (code == OP_INV_RMS)
        return initInvRmsForward;
    if (code == OP_RMS_NORM)
        return initRmsNormForward_ANY_F32_F32;
    if (code == OP_ROPE)
        return initRopeForward_F32;
    if (code == OP_MULTIHEAD_ATT)
        return initMultiHeadAttForward;
    if (code == OP_MATMUL)
        return initMatmulForward;
    if (code == OP_MUL)
        return initMulForward;
    if (code == OP_CAST)
        return initCastForward;
    if (code == OP_REPEAT_Z)
        return initRepeatZForward;
    if (code == OP_MOE_GATE)
        return initMoeGateForward;
    return nullptr;
}

NnCpuOpForward getCpuOpForward(NnOpCode code, NnOpQuantType quantType) {
    if (code == OP_PLAN_BARRIER || code == OP_PLAN_APPLY) {
        return noopForward;
    }
    if (code == OP_MERGE_ADD) {
        if (quantType == F32_F32_F32) return mergeAddForward_F32_F32;
        if (quantType == Q80_Q80_F32) return mergeAddForward_Q80_F32;
    }
    if (code == OP_MERGE_SUM) {
        if (quantType == F32_F32_F32) return mergeSumForward_F32_F32;
    }
    if (code == OP_EMBEDDING) {
        if (quantType == F32_F32_F32) return embeddingForward_F32_F32_F32;
        if (quantType == F32_F32_Q80) return embeddingForward_F32_F32_Q80;
    }
    if (code == OP_INV_RMS) {
        if (quantType == F32_F32_F32) return invRmsForward_F32_F32;
    }
    if (code == OP_RMS_NORM) {
        if (quantType == F32_F32_F32) return rmsNormForward_F32_F32_F32;
        if (quantType == Q80_F32_F32) return rmsNormForward_Q80_F32_F32;
    }
    if (code == OP_MATMUL) {
        if (quantType == F32_F32_F32) return matmulForward_F32_F32_F32;
        if (quantType == Q80_Q40_F32) return matmulForward_Q80_Q40_F32;
    }
    if (code == OP_ROPE) {
        if (quantType == F32_F32_F32) return ropeForward_F32_F32;
    }
    if (code == OP_MULTIHEAD_ATT) {
        if (quantType == F32_F32_F32) return multiHeadAttForward_F32_F32;
    }
    if (code == OP_GELU) {
        if (quantType == F32_F32_F32) return geluForward_F32_F32_F32;
    }
    if (code == OP_SILU) {
        if (quantType == F32_F32_F32) return siluForward_F32_F32;
    }
    if (code == OP_MUL) {
        if (quantType == F32_F32_F32) return mulForward_F32_F32;
    }
    if (code == OP_SCALE) {
        if (quantType == F32_F32_F32) return scaleForward_F32_F32;
    }
    if (code == OP_CAST) {
        if (quantType == F32_F32_F32) return castForward_ANY;
        if (quantType == F32_F32_Q80) return castForward_F32_Q80;
        if (quantType == Q80_Q80_Q80) return castForward_ANY;
        if (quantType == Q80_Q80_F32) return castForward_Q80_F32;
    }
    if (code == OP_REPEAT_Z) {
        if (quantType == F32_F32_Q80) return repeatZForward_F32_Q80;
    }
    if (code == OP_SHIFT) {
        if (quantType == F32_F32_F32) return shiftForward_F32_F32;
    }
    if (code == OP_SOFTMAX) {
        if (quantType == F32_F32_F32) return softmaxForward_F32_F32;
    }
    if (code == OP_MOE_GATE) {
        if (quantType == F32_F32_F32) return moeGateForward_F32_F32;
    }
    return nullptr;
}
