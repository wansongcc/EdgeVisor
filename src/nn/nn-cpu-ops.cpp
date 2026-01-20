#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cstdint>
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

#define DEBUG_OP_INPUT_OUTPUT false

static inline bool envFlagLocal(const char *name) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return false;
    if (std::strcmp(v, "0") == 0) return false;
    if (std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0) return false;
    if (std::strcmp(v, "no") == 0 || std::strcmp(v, "NO") == 0) return false;
    return true;
}

static std::atomic_uint g_dbgPrintedQInputMask{0u};

static inline bool shouldPrintQInputOnce(NnUint nodeIndex) {
    if (!envFlagLocal("DLLAMA_DEBUG_PRINT_Q_INPUT"))
        return false;
    if (nodeIndex > 31u)
        return false;
    const unsigned bit = (1u << nodeIndex);
    const unsigned prev = g_dbgPrintedQInputMask.fetch_or(bit, std::memory_order_acq_rel);
    return (prev & bit) == 0u;
}

static void checkTensorStats(const char* tag, NnUint nodeIndex, NnUint layerIndex, float* data, NnSize size) {
    if (size == 0) return;
    
    float minVal = 1e9, maxVal = -1e9;
    double sum = 0;
    double sumSq = 0;
    NnSize zeroCount = 0;
    
    for (NnSize i = 0; i < size; ++i) {
        float v = data[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
        sum += v;
        sumSq += v * v;
        if (std::abs(v) < 1e-9) zeroCount++;
    }
    
    double mean = sum / size;
    double variance = (sumSq / size) - (mean * mean);
    double stdDev = (variance > 0) ? std::sqrt(variance) : 0;
    float zeroRatio = (float)zeroCount / size;

    printf("📉 [STATS %s] Node=%u Layer=%u | Range=[%.4f, %.4f] Mean=%.4f Std=%.4f | Zeros=%zu (%.1f%%)\n", 
           tag, nodeIndex, layerIndex, minVal, maxVal, mean, stdDev, zeroCount, zeroRatio * 100.0f);
}

// Allow the stage root to enable matmul debug logging on workers via sharding-update header flags.
static std::atomic<int> g_forceShardingMatmulLog{0};

static inline bool shardingMatmulLogEnabled() {
    static int cached = -1;
    if (g_forceShardingMatmulLog.load(std::memory_order_acquire) != 0) return true;
    if (cached >= 0) return cached != 0;
    const char *v = std::getenv("DLLAMA_DEBUG_SHARDING_MATMUL");
    cached = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
    return cached != 0;
}

static inline int shardingMatmulLogLayerFilter() {
    static int cached = -2;
    if (cached != -2) return cached;
    const char *v = std::getenv("DLLAMA_DEBUG_SHARDING_LAYER");
    if (v == nullptr || v[0] == '\0') {
        cached = -1;
        return cached;
    }
    cached = std::atoi(v);
    return cached;
}

static inline int shardingMatmulLogNodeFilter() {
    static int cached = -2;
    if (cached != -2) return cached;
    const char *v = std::getenv("DLLAMA_DEBUG_SHARDING_NODE");
    if (v == nullptr || v[0] == '\0') {
        cached = -1;
        return cached;
    }
    cached = std::atoi(v);
    return cached;
}

static inline const char *splitKindName(NnUint kind) {
    switch ((NnSplitKind)kind) {
        case SPLIT_HEAD: return "HEAD";
        case SPLIT_KV_HEAD: return "KV_HEAD";
        case SPLIT_VOCAB: return "VOCAB";
        case SPLIT_FFN: return "FFN";
        case SPLIT_DIM: return "DIM";
        default: return "UNKNOWN";
    }
}

static inline const char *splitAxisName(NnUint axis) {
    // See nn-core.hpp: OUT_ROWS=1, IN_COLS=2 (historical encoding).
    if (axis == 1u) return "OUT_ROWS";
    if (axis == 2u) return "IN_COLS";
    return "NONE";
}

// Last applied sharding update range. Used only for debug logging.
static std::atomic<NnUint> g_lastAppliedShardingEpoch{0u};
static std::atomic<NnUint> g_lastAppliedShardingRangeStart{0u};
static std::atomic<NnUint> g_lastAppliedShardingRangeEnd{0u};
// Keep separate keys so BASELINE logging doesn't overwrite UPDATE logging (and vice versa).
// Otherwise, alternating layers can cause the same UPDATE line to be printed repeatedly.
static std::atomic<std::uint64_t> g_lastLoggedMatmulUpdateKey{0u};
static std::atomic<std::uint64_t> g_lastLoggedMatmulBaselineKey{0u};

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

#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A 0.044715f

static void gelu_F32(float *output, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        float x = output[i];
        output[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
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
    const NnUint pos, const NnUint nHeads, const NnUint nHeads0, const NnUint nKvHeads, const NnUint kvDim0, const NnUint headDim, const NnUint seqLen,
    const NnUint headStart,
    const NnUint kvHeadStart,
    const NnUint nThreads, const NnUint threadIndex) 
{
    SPLIT_THREADS(h0Start, h0End, nHeads0, nThreads, threadIndex);
    const NnUint kvMul = nHeads / nKvHeads;
    const float headDimRoot = sqrtf(headDim);

    for (NnUint h0 = h0Start; h0 < h0End; h0++) {
        const float *hQ = &q[h0 * headDim];
        // [Fix] Map Local Q Head (h0) -> Global Q Head -> Global KV Head
        const NnUint globalH = headStart + h0;
        const NnUint globalKvH = globalH / kvMul;

        // [Fix] Map Global KV Head -> Local KV Head (subtract kvOffset)
        // Ensure we fall back to 0 if out-of-range (should not happen in valid GQA)
        const NnUint localKvH = (globalKvH >= kvHeadStart) ? (globalKvH - kvHeadStart) : 0u;      

        const float *hKc = &keyCache[localKvH * headDim];
        const float *hVc = &valueCache[localKvH * headDim];
        float *hAtt = &att[h0 * seqLen];

        for (NnUint t = 0; t <= pos; t++) {
            const float *posK = &hKc[t * kvDim0];
            const float score = dotProduct_F32(hQ, posK, headDim) / headDimRoot;
            hAtt[t] = score;
        }

        softmax_F32(hAtt, pos + 1);

        float *hY = &y[h0 * headDim];
        std::memset(hY, 0, headDim * sizeof(float));

        for (NnUint t = 0; t <= pos; t++) {
            const float *posV = &hVc[t * kvDim0];
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

static void scale_F32(const float *i, float *o, const float s, NnSize size, NnUint nThreads, NnUint threadIndex) {
    for (NnUint x = threadIndex; x < size; x += nThreads)
        o[x] = i[x] * s;
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
        if (threadIndex == 0) {
            printf("[EMBEDDING CHECK] node=%u batch=%u token=%u\n", 
                   context->nodeIndex, batchIndex, token);
        }
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
    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    assert(context->outputSize.x >= config->nColumns);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
}

static void invRmsForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnInvRmsOpConfig *config = (NnInvRmsOpConfig *)context->opConfig;
    const NnUint colSize = context->inputSize.x / config->nColumns;

    for (NnUint batchIndex = threadIndex; batchIndex < batchSize; batchIndex += nThreads) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);
        for (NnUint colIndex = 0; colIndex < config->nColumns; colIndex++) {
            float rms = invRms_F32(
                &input[colIndex * colSize],
                colSize,
                config->epsilon);
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

    const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    const float *weight = (float *)context->weight;
    const NnUint invRmsBatchSize = context->bufferConfigs[config->invRmsBufferIndex].size.x;
    const float *invRms = (float *)context->buffers[config->invRmsBufferIndex];

    const NnUint colSize = context->weightSize.x;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);
        for (NnUint colIndex = 0; colIndex < config->nColumns; colIndex++) {
            rmsNorm_F32(
                &output[colIndex * colSize],
                &input[colIndex * colSize],
                invRms[batchIndex * invRmsBatchSize + colIndex],
                weight,
                colSize,
                nThreads,
                threadIndex);
        }
        DEBUG_VECTOR(context, "output", output);
    }
}

static void rmsNormForward_Q80_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);

    const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    ASSERT_EQ(config->nColumns, 1); // TODO: add support multiple columns

    const float *weight = (float *)context->weight;
    const float *invRms = (float *)context->buffers[config->invRmsBufferIndex];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        rmsNorm_Q80_F32_F32(
            output,
            input,
            invRms[batchIndex],
            weight,
            context->inputSize.x,
            nThreads,
            threadIndex);
        DEBUG_VECTOR(context, "output", output);
    }
}

static void initMatmulForward(NnCpuOpContext *context) {
    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.z, std::max(config->nActiveExperts, 1u));
    ASSERT_EQ(context->outputSize.z, std::max(config->nActiveExperts, 1u));
    ASSERT_EQ(context->weightSize.z, std::max(config->nExperts, 1u));

    // Legacy strict checks (no dynamic sharding / no stage-max preallocation)
    if (config->replicateMode == 0u && config->splitAxis == 0u) {
        ASSERT_EQ(context->inputSize.x, context->weightSize.y);
        ASSERT_EQ(context->outputSize.x, context->weightSize.x);
        return;
    }

    // OUT_ROWS: output is a prefix of a sharded vector (len can change online).
    if (config->splitAxis == 1u) {
        ASSERT_EQ(context->inputSize.x, context->weightSize.y);
        const NnUint unit = (config->splitUnit == 0u) ? 1u : config->splitUnit;
        if (config->staticLenUnits > 0u)
            assert(context->outputSize.x >= config->staticLenUnits * unit);
        return;
    }

    // IN_COLS: input is a prefix of a sharded vector (len can change online).
    if (config->splitAxis == 2u) {
        ASSERT_EQ(context->outputSize.x, context->weightSize.x);
        const NnUint unit = (config->splitUnit == 0u) ? 1u : config->splitUnit;
        if (config->staticLenUnits > 0u)
            assert(context->inputSize.x >= config->staticLenUnits * unit);
        return;
    }

    if (!context->hasInputContinuousMemory)
        printf("🚧 Op %s does not have contiguous memory for input\n", context->name);
    if (!context->hasOutputContinuousMemory)
        printf("🚧 Op %s does not have contiguous memory for output\n", context->name);

}

static bool matmulForward_llamafile(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (batchSize == 1u || !context->hasInputContinuousMemory || !context->hasOutputContinuousMemory || context->inputSize.z != 1u)
        return false;

    const NnUint n = context->weightSize.y / getBlockSize(context->inputSize.floatType);
    const NnUint d = context->weightSize.x;
    return llamafile_sgemm(
        d, batchSize, n,
        context->weight, n,
        context->input[0], n,
        context->output[0], d,
        threadIndex, nThreads, 0,
        context->weightSize.floatType,
        context->inputSize.floatType,
        F_32
    );
}

static inline const NnDimSplit *getLayerSplitPtr(const NnLayerSplits &ls, NnSplitKind kind) {
    switch (kind) {
        case SPLIT_HEAD: return &ls.headSplit;
        case SPLIT_KV_HEAD: return &ls.kvHeadSplit;
        case SPLIT_VOCAB: return &ls.vocabSplit;
        case SPLIT_FFN: return &ls.ffnSplit;
        case SPLIT_DIM: return &ls.dimSplit;
        default: return nullptr;
    }
}

static inline NnUint getSplitTotalUnits(const NnDimSplit *split, NnUint nNodes) {
    if (split == nullptr || split->lengths == nullptr) return 0u;
    NnUint sum = 0u;
    for (NnUint i = 0u; i < nNodes; ++i) sum += split->lengths[i];
    return sum;
}

static inline void buildOwnerMapFromSplit(
    const NnDimSplit *split,
    NnUint nNodes,
    NnUint **owners,
    NnUint *ownersLen
) {
    if (owners == nullptr || ownersLen == nullptr) return;
    const NnUint totalUnits = getSplitTotalUnits(split, nNodes);
    if (split == nullptr || split->starts == nullptr || split->lengths == nullptr || totalUnits == 0u) {
        if (*owners != nullptr) {
            delete[] *owners;
            *owners = nullptr;
        }
        *ownersLen = 0u;
        return;
    }

    if (*owners == nullptr || *ownersLen != totalUnits) {
        delete[] *owners;
        *owners = new NnUint[totalUnits];
        *ownersLen = totalUnits;
    }

    std::fill(*owners, *owners + *ownersLen, 0u);
    for (NnUint node = 0u; node < nNodes; ++node) {
        const NnUint start = split->starts[node];
        const NnUint len = split->lengths[node];
        for (NnUint u = 0u; u < len && (start + u) < totalUnits; ++u) {
            (*owners)[start + u] = node;
        }
    }
}

static inline bool getRuntimeStartLenUnits(
    const NnCpuOpContext *context,
    const NnMatmulOpConfig *config,
    NnUint *outStartUnits,
    NnUint *outLenUnits
) {
    if (outStartUnits == nullptr || outLenUnits == nullptr) return false;
    *outStartUnits = config->staticStartUnits;
    *outLenUnits = config->staticLenUnits;

    if (context->layerSharding == nullptr) return false;
    const NnUint epoch = context->layerSharding->epoch.load(std::memory_order_acquire);
    if (epoch == 0u) return false;
    if (config->splitKind >= (NnUint)N_SPLIT_KINDS) return false;
    if (config->layerIndex >= context->layerSharding->nLayers) return false;
    if (context->nodeIndex >= context->layerSharding->nNodes) return false;

    const NnLayerSplits &ls = context->layerSharding->layers[config->layerIndex];
    const NnDimSplit *sp = getLayerSplitPtr(ls, (NnSplitKind)config->splitKind);
    if (sp == nullptr || sp->starts == nullptr || sp->lengths == nullptr) return false;

    *outStartUnits = sp->starts[context->nodeIndex];
    *outLenUnits = sp->lengths[context->nodeIndex];
    return true;
}

static inline NnUint getRuntimeLenElems(
    const NnCpuOpContext *context,
    const NnUint layerIndex,
    const NnUint splitKind,
    const NnUint splitUnit,
    const NnUint staticLenUnits,
    const NnUint maxElems
) {
    const NnUint unit = (splitUnit == 0u) ? 1u : splitUnit;
    NnUint lenUnits = staticLenUnits;

    if (context->layerSharding != nullptr) {
        const NnUint epoch = context->layerSharding->epoch.load(std::memory_order_acquire);
        if (epoch != 0u && splitKind < (NnUint)N_SPLIT_KINDS && layerIndex < context->layerSharding->nLayers && context->nodeIndex < context->layerSharding->nNodes) {
            const NnLayerSplits &ls = context->layerSharding->layers[layerIndex];
            const NnDimSplit *sp = getLayerSplitPtr(ls, (NnSplitKind)splitKind);
            if (sp != nullptr && sp->lengths != nullptr) {
                lenUnits = sp->lengths[context->nodeIndex];
            }
        }
    }

    return std::min(lenUnits * unit, maxElems);
}

static void matmul_F32_F32_F32_stridedW(
    float *output,
    const float *x,
    const float *wGlobal,
    const NnUint nGlobal,
    const NnUint start,
    const NnUint nLocal,
    const NnUint d,
    const NnUint nThreads,
    const NnUint threadIndex
) {
    SPLIT_THREADS(outStart, outEnd, d, nThreads, threadIndex);
    for (NnUint i = outStart; i < outEnd; i++) {
        const float *wRow = &wGlobal[i * (NnSize)nGlobal + start];
        float val = 0.0f;
        for (NnUint j = 0; j < nLocal; j++) {
            val += wRow[j] * x[j];
        }
        output[i] = val;
    }
}

static void matmul_Q80_Q40_F32_stridedW(
    float *output,
    const NnBlockQ80 *x,
    const NnBlockQ40 *wGlobal,
    const NnUint nGlobal,
    const NnUint start,
    const NnUint nLocal,
    const NnUint d,
    const NnUint nThreads,
    const NnUint threadIndex
) {
    SPLIT_THREADS(outStart, outEnd, d, nThreads, threadIndex);
    assert(nGlobal % Q40_BLOCK_SIZE == 0);
    assert(nLocal % Q40_BLOCK_SIZE == 0);
    assert(start % Q40_BLOCK_SIZE == 0);
    const NnUint nGlobalBlocks = nGlobal / Q40_BLOCK_SIZE;
    const NnUint startBlock = start / Q40_BLOCK_SIZE;
    const NnUint nLocalBlocks = nLocal / Q40_BLOCK_SIZE;

    for (NnUint di = outStart; di < outEnd; di++) {
        float sum = 0.0f;
        const NnBlockQ40 *wRow = &wGlobal[di * (NnSize)nGlobalBlocks + startBlock];
        for (NnUint bj = 0; bj < nLocalBlocks; bj++) {
            // Reuse the existing q80/q40 dot routine by calling into matmul with adjusted pointers.
            // Here we do a simple scalar implementation using the dequant scales.
            // (This is not the fastest path but is correct and only used in replicated+strided mode.)
            const float dx = CONVERT_F16_TO_F32(x[bj].d);
            const float dw = CONVERT_F16_TO_F32(wRow[bj].d);
            for (NnUint k = 0; k < Q40_BLOCK_SIZE; k++) {
                // wRow[bj].qs packs 2x4-bit values per byte.
                const uint8_t packed = wRow[bj].qs[k / 2];
                const int8_t w4 = (k % 2 == 0)
                    ? (int8_t)(packed & 0x0F)
                    : (int8_t)((packed >> 4) & 0x0F);
                const int8_t wv = (int8_t)(w4 - 8);
                sum += (float)wv * (float)x[bj].qs[k] * dx * dw;
            }
        }
        output[di] = sum;
    }
}

static void matmulForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;

    // Runtime sharding selection (hoisted). Note that matmul may take the fast llamafile path;
    // keep this logic before the early return so we can still print debug info.
    NnUint startUnits = config->staticStartUnits;
    NnUint lenUnits = config->staticLenUnits;
    (void)getRuntimeStartLenUnits(context, config, &startUnits, &lenUnits);
    const NnUint unit = (config->splitUnit == 0u) ? 1u : config->splitUnit;
    const NnUint startElems = startUnits * unit;
    const NnUint lenElems = lenUnits * unit;
    const NnUint runtimeLenOut = std::min(lenElems, context->outputSize.x);
    const NnUint runtimeLenIn = std::min(lenElems, context->inputSize.x);

    const bool outRows = (config->splitAxis == 1u);
    const NnUint startOut = 0u;
    // OUT_ROWS 分片时，所有节点都应从 input[0] 读
    const NnUint startIn = outRows ? 0u : startElems;
    const NnUint maxRows = context->weightSize.x;
    const NnUint dCompute = std::min(runtimeLenOut, (startOut < maxRows) ? (maxRows - startOut) : 0u);

#if DLLAMA_CONTROL_LOG
    if (shardingMatmulLogEnabled() && threadIndex == 0u) {
        const int layerFilter = shardingMatmulLogLayerFilter();
        const int nodeFilter = shardingMatmulLogNodeFilter();
        if ((layerFilter < 0 || (NnUint)layerFilter == config->layerIndex) && (nodeFilter < 0 || (NnUint)nodeFilter == context->nodeIndex)) {
            const NnUint epoch = (context->layerSharding == nullptr) ? 0u : context->layerSharding->epoch.load(std::memory_order_acquire);
            const NnUint appliedEpoch = g_lastAppliedShardingEpoch.load(std::memory_order_acquire);
            const NnUint rangeStart = g_lastAppliedShardingRangeStart.load(std::memory_order_acquire);
            const NnUint rangeEnd = g_lastAppliedShardingRangeEnd.load(std::memory_order_acquire);
            const bool isUpdate = (epoch != 0u && epoch == appliedEpoch && config->layerIndex >= rangeStart && config->layerIndex < rangeEnd);
            const std::uint64_t runtimeKey = (epoch == 0u)
                ? 0u
                : ((std::uint64_t)epoch << 32) | (std::uint64_t)config->layerIndex;
            const std::uint64_t logKey = isUpdate ? runtimeKey : ((std::uint64_t)config->layerIndex);

            const std::uint64_t prev = isUpdate
                ? g_lastLoggedMatmulUpdateKey.exchange(logKey, std::memory_order_acq_rel)
                : g_lastLoggedMatmulBaselineKey.exchange(logKey, std::memory_order_acq_rel);
            if (prev != logKey) {
                printf("[SHARD][CPU][MATMUL][%s] node=%u op=%u name=%s layer=%u epoch=%u splitKind=%s(%u) splitAxis=%s(%u) unit=%u startUnits=%u lenUnits=%u startElems=%u lenElems=%u runtimeOut=%u runtimeIn=%u startOut=%u startIn=%u dCompute=%u repMode=%u\n",
                    isUpdate ? "UPDATE" : "BASELINE",
                    context->nodeIndex,
                    context->opIndex,
                    context->name ? context->name : "(null)",
                    config->layerIndex,
                    epoch,
                    splitKindName(config->splitKind),
                    config->splitKind,
                    splitAxisName(config->splitAxis),
                    config->splitAxis,
                    unit,
                    startUnits,
                    lenUnits,
                    startElems,
                    lenElems,
                    runtimeLenOut,
                    runtimeLenIn,
                    startOut,
                    startIn,
                    dCompute,
                    config->replicateMode);
            }
        }
    }
#endif

    if (config->replicateMode == 0u) {
        if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
            return;
    }
    const NnUint nActiveExpertsOr1 = std::max(config->nActiveExperts, 1u);
    const float *activeExpertIndexes = (const float *)context->buffers[config->activeExpertIndexesBufferIndex];

    for (NnUint y = 0; y < batchSize; y++) {
        for (NnUint e = 0; e < nActiveExpertsOr1; e++) {
            const NnUint activeExpertIndex = config->nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[y * config->nActiveExperts + e];

            const NnSize expertStride = config->replicateMode == 0u
                ? context->weightSize.nBytesXY
                : config->replicateExpertStrideBytes;
            const NnSize expertBase = (NnSize)activeExpertIndex * expertStride;

            const NnByte *expertWeightBase = &context->weight[expertBase];

            float *output = (float *)context->output[e * context->outputSize.y + y];

            if (config->replicateMode != 0u && config->splitAxis == 2u && config->replicateGlobalInDim > 0u) {
                // IN_COLS: weight stores full global tensor; select input-dim shard via strided access.
                const NnUint nLocal = runtimeLenIn;
                if (nLocal > 0u && startIn + nLocal <= config->replicateGlobalInDim) {
                    matmul_F32_F32_F32_stridedW(
                        output,
                        (float *)context->input[e * context->inputSize.y + y],
                        (float *)&context->weight[expertBase],
                        config->replicateGlobalInDim,
                        startIn,
                        nLocal,
                        context->weightSize.x,
                        nThreads,
                        threadIndex);
                }
            } else {
                // OUT_ROWS and legacy/default: compute only the valid prefix (dCompute).
                if (dCompute > 0u) {
                    const NnSize bytesPerRow = getBytes(context->weightSize.floatType, context->weightSize.x);
                    const NnSize rowOffsetBytes = (NnSize)startOut * bytesPerRow;
                    const float *w = (const float *)(expertWeightBase + rowOffsetBytes);

                    float *targetOutput = output;
                    matmul_F32_F32_F32(
                        targetOutput,
                        (float *)context->input[e * context->inputSize.y + y],
                        (float *)w,
                        context->weightSize.x,
                        dCompute,
                        nThreads,
                        threadIndex);
                }
                // if (threadIndex == 0u && dCompute < context->outputSize.x) {
                //     std::memset(&output[dCompute], 0, (context->outputSize.x - dCompute) * sizeof(float));
                // }

                
            }

            if (threadIndex == 0 && y == 0 && e == 0 && config->layerIndex == 0) {
                // 仅针对 Wo 打印
                if (context->name && std::strcmp(context->name, "block_matmul_wo") == 0) {
                    // 这里的 output 是 matmul 的结果 (Partial Sum)
                    // 对于 Node 1，如果 stridedW 工作正常，这里应该有数值
                    // 如果全是 0，说明 Node 1 没算对
                    checkTensorStats("Wo_PARTIAL", context->nodeIndex, config->layerIndex, output, context->outputSize.x);
                }
            }

            DEBUG_VECTOR(context, "output", output);
        }
    }
}

static void matmulForward_Q80_Q40_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    
    if (threadIndex == 0 && batchSize > 0) {
        // 1. Layer 1 Input Consistency Check (检查上一层 All-Reduce 是否成功)
        // 在 Layer 1 的 Q 算子处打印输入的前几个数和 Checksum
        if (config->layerIndex == 1 && context->name && std::strcmp(context->name, "block_matmul_q") == 0) {
            // 注意：input[0] 是 Q80 格式，我们需要解析它
            const NnBlockQ80* in_ptr = (const NnBlockQ80*)context->input[0];
            float tmp[Q80_BLOCK_SIZE];
            // 解压第一个 block 用于打印
            dequantizeQ80toF32(in_ptr, tmp, Q80_BLOCK_SIZE, 1, 0);
            
            // 计算简单的 Checksum (基于前 10 个 block 的 scale)
            float scaleSum = 0.0f;
            for(int i=0; i<10 && i < (int)(context->inputSize.x/Q80_BLOCK_SIZE); ++i) {
                scaleSum += CONVERT_F16_TO_F32(in_ptr[i].d);
            }

            printf("\n🕵️ [LAYER1 INPUT CHECK] Node=%u Layer=1 Op=Q\n", context->nodeIndex);
            printf("    First 8 Floats: %f %f %f %f %f %f %f %f\n", 
                   tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]);
            printf("    Scale Sum (first 10 blocks): %f\n", scaleSum);
            printf("--------------------------------------------------\n");
        }

        // 2. Wo Weight Fingerprint Check (检查权重加载是否错位)
        // 仅在 Layer 0 的 Wo 算子处打印
        if (config->layerIndex == 0 && context->name && std::strcmp(context->name, "block_matmul_wo") == 0) {
            const NnBlockQ40* w_ptr = (const NnBlockQ40*)context->weight;
            
            printf("\n🔥 [WEIGHT CHECK Wo] Node=%u Layer=0 Op=Wo\n", context->nodeIndex);
            printf("    Weight Ptr: %p\n", (const void*)w_ptr);
            printf("    First 4 Blocks (F16->F32):\n");
            for (int i = 0; i < 4; i++) {
                float d = CONVERT_F16_TO_F32(w_ptr[i].d);
                uint8_t q = w_ptr[i].qs[0];
                printf("      Block[%d]: d=%f, qs[0]=0x%02x\n", i, d, q);
            }
            printf("--------------------------------------------------\n");
        }
    }

    NnUint startUnits = config->staticStartUnits;
    NnUint lenUnits = config->staticLenUnits;
    (void)getRuntimeStartLenUnits(context, config, &startUnits, &lenUnits);
    const NnUint unit = (config->splitUnit == 0u) ? 1u : config->splitUnit;
    const NnUint startElems = startUnits * unit;
    const NnUint lenElems = lenUnits * unit;
    const NnUint runtimeLenOut = std::min(lenElems, context->outputSize.x);
    const NnUint runtimeLenIn = std::min(lenElems, context->inputSize.x);

    const bool outRows = (config->splitAxis == 1u);
    const NnUint startOut = (config->replicateMode != 0u && outRows) ? startElems : 0u;
    // OUT_ROWS 分片时，所有节点都应从 input[0] 读
    const NnUint startIn = outRows ? 0u : startElems;
    const NnUint maxRows = context->weightSize.x;
    const NnUint dCompute = std::min(runtimeLenOut, (startOut < maxRows) ? (maxRows - startOut) : 0u);

#if DLLAMA_CONTROL_LOG
    if (shardingMatmulLogEnabled() && threadIndex == 0u) {
        const int layerFilter = shardingMatmulLogLayerFilter();
        const int nodeFilter = shardingMatmulLogNodeFilter();
        if ((layerFilter < 0 || (NnUint)layerFilter == config->layerIndex) && (nodeFilter < 0 || (NnUint)nodeFilter == context->nodeIndex)) {
            const NnUint epoch = (context->layerSharding == nullptr) ? 0u : context->layerSharding->epoch.load(std::memory_order_acquire);
            const NnUint appliedEpoch = g_lastAppliedShardingEpoch.load(std::memory_order_acquire);
            const NnUint rangeStart = g_lastAppliedShardingRangeStart.load(std::memory_order_acquire);
            const NnUint rangeEnd = g_lastAppliedShardingRangeEnd.load(std::memory_order_acquire);
            const bool isUpdate = (epoch != 0u && epoch == appliedEpoch && config->layerIndex >= rangeStart && config->layerIndex < rangeEnd);
            const std::uint64_t runtimeKey = (epoch == 0u)
                ? 0u
                : ((std::uint64_t)epoch << 32) | (std::uint64_t)config->layerIndex;
            const std::uint64_t logKey = isUpdate ? runtimeKey : ((std::uint64_t)config->layerIndex);

            const std::uint64_t prev = isUpdate
                ? g_lastLoggedMatmulUpdateKey.exchange(logKey, std::memory_order_acq_rel)
                : g_lastLoggedMatmulBaselineKey.exchange(logKey, std::memory_order_acq_rel);
            if (prev != logKey) {
                printf("[SHARD][CPU][MATMUL][%s] node=%u op=%u name=%s layer=%u epoch=%u splitKind=%s(%u) splitAxis=%s(%u) unit=%u startUnits=%u lenUnits=%u startElems=%u lenElems=%u runtimeOut=%u runtimeIn=%u startOut=%u startIn=%u dCompute=%u repMode=%u\n",
                    isUpdate ? "UPDATE" : "BASELINE",
                    context->nodeIndex,
                    context->opIndex,
                    context->name ? context->name : "(null)",
                    config->layerIndex,
                    epoch,
                    splitKindName(config->splitKind),
                    config->splitKind,
                    splitAxisName(config->splitAxis),
                    config->splitAxis,
                    unit,
                    startUnits,
                    lenUnits,
                    startElems,
                    lenElems,
                    runtimeLenOut,
                    runtimeLenIn,
                    startOut,
                    startIn,
                    dCompute,
                    config->replicateMode);
            }
        }
    }
#endif

    if (config->replicateMode == 0u) {
        if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
            return;
    }
    const NnUint nActiveExpertsOr1 = std::max(config->nActiveExperts, 1u);
    const float *activeExpertIndexes = (const float *)context->buffers[config->activeExpertIndexesBufferIndex];

    for (NnUint y = 0; y < batchSize; y++) {
        for (NnUint e = 0; e < nActiveExpertsOr1; e++) {
            const NnUint activeExpertIndex = config->nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[y * config->nActiveExperts + e];

            const NnSize expertStride = config->replicateMode == 0u
                ? context->weightSize.nBytesXY
                : config->replicateExpertStrideBytes;
            const NnSize expertBase = (NnSize)activeExpertIndex * expertStride;

            const NnByte *expertWeightBase = &context->weight[expertBase];

            float *output = (float *)context->output[e * context->outputSize.y + y];

            if (threadIndex == 0u && y == 0u && e == 0u &&
                context->name != nullptr && std::strcmp(context->name, "block_matmul_q") == 0 &&
                config->layerIndex == 0u && (context->nodeIndex == 0u || context->nodeIndex == 1u) &&
                shouldPrintQInputOnce(context->nodeIndex)) {
                const NnBlockQ80 *in0 = (const NnBlockQ80 *)context->input[e * context->inputSize.y + y];
                float tmp[Q80_BLOCK_SIZE];
                std::memset(tmp, 0, sizeof(tmp));
                // Dequantize the first block to readable floats.
                dequantizeQ80toF32(in0, tmp, Q80_BLOCK_SIZE, 1u, 0u);

                std::printf(
                    "[QINPUT CHECK Q80] node=%u layer=%u op=%s axis=%u startIn=%u startOut=%u inType=%s inDim=%u inPtr=%p continuous=%u first8:",
                    context->nodeIndex,
                    config->layerIndex,
                    context->name,
                    config->splitAxis,
                    startIn,
                    startOut,
                    floatTypeToString(context->inputSize.floatType),
                    context->inputSize.x,
                    (const void *)in0,
                    context->hasInputContinuousMemory ? 1u : 0u);
                for (NnUint i = 0u; i < 8u; ++i) {
                    std::printf(" %g", tmp[i]);
                }
                std::printf("\n");
            }

            if (config->replicateMode != 0u && config->splitAxis == 2u && config->replicateGlobalInDim > 0u) {
                const NnUint nLocal = runtimeLenIn;
                if (nLocal > 0u && startIn + nLocal <= config->replicateGlobalInDim) {
                    if (nLocal % Q40_BLOCK_SIZE == 0u && startIn % Q40_BLOCK_SIZE == 0u && config->replicateGlobalInDim % Q40_BLOCK_SIZE == 0u) {
                        matmul_Q80_Q40_F32_stridedW(
                            output,
                            (NnBlockQ80 *)context->input[e * context->inputSize.y + y],
                            (NnBlockQ40 *)&context->weight[expertBase],
                            config->replicateGlobalInDim,
                            startIn,
                            nLocal,
                            context->weightSize.x,
                            nThreads,
                            threadIndex);
                    }
                }
            } else {
                if (dCompute > 0u) {
                    const NnSize bytesPerRow = getBytes(context->weightSize.floatType, context->weightSize.y);
                    const NnSize rowOffsetBytes = (NnSize)startOut * bytesPerRow;
                    const NnBlockQ40 *w = (const NnBlockQ40 *)(expertWeightBase + rowOffsetBytes);

                    float *targetOutput = output;

                    matmul_Q80_Q40_F32(
                        targetOutput,
                        (NnBlockQ80 *)context->input[e * context->inputSize.y + y],
                        (NnBlockQ40 *)w,
                        context->weightSize.y,
                        dCompute,
                        nThreads,
                        threadIndex);
                }
                // if (threadIndex == 0u && dCompute < context->outputSize.x) {
                //     std::memset(&output[dCompute], 0, (context->outputSize.x - dCompute) * sizeof(float));
                // }
            }
            DEBUG_VECTOR(context, "output", output);
        }
    }
}

static void siluForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnSiluOpCodeConfig *config = (NnSiluOpCodeConfig *)context->opConfig;
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    const NnUint nElems = (config->staticLenUnits == 0u && config->splitKind == 0u)
        ? context->outputSize.x
        : getRuntimeLenElems(context, config->layerIndex, config->splitKind, config->splitUnit, config->staticLenUnits, context->outputSize.x);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            float *output = (float *)context->output[z * context->outputSize.y + y];
            silu_F32(output, nElems, nThreads, threadIndex);
        }
    }
}

static void geluForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        gelu_F32(output, context->outputSize.x, nThreads, threadIndex);
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

static void ropeForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnRopeOpConfig *config = (NnRopeOpConfig *)context->opConfig;
    const NnRopeSlice *slice = &config->slice;
    const float *positions = (float *)context->pipes[config->positionPipeIndex];
    const float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];
    const bool isQ = config->isQ == 1;

    NnUint startUnits = config->staticStartUnits;

    const NnUint headDim = slice->headDim;
    const NnUint maxElems = context->inputSize.x;
    const NnUint fallbackElems = (config->staticLenUnits == 0u && config->splitKind == 0u)
        ? maxElems
        : getRuntimeLenElems(context, config->layerIndex, config->splitKind, config->splitUnit, config->staticLenUnits, maxElems);
    const NnUint runtimeElems = (headDim == 0u) ? 0u : ((fallbackElems / headDim) * headDim);
    const NnUint unit = (config->splitUnit == 0) ? 1 : config->splitUnit; // 通常为 headDim
    const NnUint startElems = startUnits * unit;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *x = (float *)context->input[batchIndex] + startElems;
        const NnUint pos = (NnUint)positions[batchIndex];

        if (threadIndex == 0 && batchIndex == 0 && config->layerIndex == 0) {
            printf("🔍 [ROPE DEBUG] Node=%u Token_Pos=%u\n", context->nodeIndex, pos);
        }

        if (config->type == ROPE_LLAMA || config->type == ROPE_LLAMA3_1) {
            // Apply rope within each head; cache repeats per-head, so we can ignore qShift.
            const NnUint pairCount = runtimeElems / 2u;
            SPLIT_THREADS(pStart, pEnd, pairCount, nThreads, threadIndex);
            const float *posCache = &cache[pos * slice->sliceDim];
            for (NnUint p = pStart; p < pEnd; p++) {
                const NnUint i = p * 2u;
                const NnUint j = i % headDim;
                const float fcr = posCache[j];
                const float fci = posCache[j + 1u];
                const float v0 = x[i];
                const float v1 = x[i + 1u];
                x[i] = v0 * fcr - v1 * fci;
                x[i + 1u] = v0 * fci + v1 * fcr;
            }
        } else if (config->type == ROPE_FALCON) {
            // Falcon kernel expects head-major layout [nHeads0, headDim].
            const NnUint dim0 = runtimeElems;
            assert(headDim > 0u);
            assert(dim0 % headDim == 0u);
            const NnUint nHeads0 = dim0 / headDim;
            SPLIT_THREADS(h0s, h0e, nHeads0, nThreads, threadIndex);
            const float *posCache = &cache[pos * headDim];

            for (NnUint h = h0s; h < h0e; h++) {
                const NnUint o = h * headDim;
                for (NnUint jj = 0u; jj < headDim / 2u; jj++) {
                    const float fcr0 = posCache[jj];
                    const float fci0 = posCache[jj + headDim / 2u];

                    float q0 = x[o + jj];
                    float q1 = x[o + jj + headDim / 2u];
                    x[o + jj] = q0 * fcr0 - q1 * fci0;
                    x[o + jj + headDim / 2u] = q0 * fci0 + q1 * fcr0;
                }
            }
        } else {
            throw std::runtime_error("Unsupported rope type");
        }
        if (threadIndex == 0 && batchIndex == 0 && config->layerIndex == 0) {
            // 只打印 Layer 0，减少刷屏
            char tag[32];
            snprintf(tag, 32, "ROPE_%s", isQ ? "Q" : "K");
            // 注意：此时 x 已经是加上了 startElems 偏移后的指针，指向当前节点负责的数据段
            checkTensorStats(tag, context->nodeIndex, config->layerIndex, x, runtimeElems);
        }
        // Clear tail to keep stage-max buffers safe.
        if (threadIndex == 0u && runtimeElems < maxElems) {
            std::memset(&x[runtimeElems], 0, (maxElems - runtimeElems) * sizeof(float));
        }
    }
}

static void initMultiHeadAttForward(NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->outputSize.x, config->qSliceD0);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    NnSize3D *querySize = &context->bufferConfigs[config->queryBufferIndex].size;
    ASSERT_EQ(querySize->x, config->qSliceD0);
    NnSize3D *posSize = &context->pipeConfigs[config->positionPipeIndex].size;
    ASSERT_EQ(posSize->x, 1);
    ASSERT_EQ(posSize->y, context->nBatches);
}

static void multiHeadAttForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    // Determine runtime local head count (in heads), with fallback to graph-time plan.
    NnUint runtimeHeads0 = (config->staticHeadLenUnits != 0u) ? config->staticHeadLenUnits : config->nHeads0;
    NnUint runtimeKvHeads0 = 0u;
    if (context->layerSharding != nullptr) {
        const NnUint epoch = context->layerSharding->epoch.load(std::memory_order_acquire);
        if (epoch != 0u && config->layerIndex < context->layerSharding->nLayers && config->nodeIndex < context->layerSharding->nNodes) {
            const NnLayerSplits &ls = context->layerSharding->layers[config->layerIndex];
            if (ls.headSplit.lengths != nullptr) {
                runtimeHeads0 = ls.headSplit.lengths[config->nodeIndex];
            }
            if (ls.kvHeadSplit.lengths != nullptr) {
                runtimeKvHeads0 = ls.kvHeadSplit.lengths[config->nodeIndex];
            }
        }
    }
    runtimeHeads0 = std::min(runtimeHeads0, config->nHeads0);

    // Defensive clamp: ensure we never read KV cache heads beyond what's available.
    // For GQA, kvMul = nHeads / nKvHeads.
    if (runtimeKvHeads0 != 0u && config->nKvHeads != 0u) {
        const NnUint kvMul = config->nHeads / config->nKvHeads;
        if (kvMul != 0u) {
            const NnUint maxHeadsByKv = runtimeKvHeads0 * kvMul;
            runtimeHeads0 = std::min(runtimeHeads0, maxHeadsByKv);
        }
    }
    const NnUint runtimeElems = runtimeHeads0 * config->headDim;



    float *query = (float *)context->buffers[config->queryBufferIndex];
    float *keyCache = (float *)context->buffers[config->keyCacheBufferIndex];
    float *valueCache = (float *)context->buffers[config->valueCacheBufferIndex];
    float *att = (float *)context->buffers[config->attBufferIndex];
    const float *positions = (float *)context->pipes[config->positionPipeIndex];

    // [FIX START] 计算输入指针偏移
    NnUint qPtrOffset = 0;
    NnUint kvPtrOffset = 0;

    // 启发式判断：如果 Buffer 维度 >= 全局维度，说明是全量 Buffer，需要偏移
    if (config->qSliceD0 >= config->nHeads * config->headDim) {
        qPtrOffset = config->headStart * config->headDim;
    }
    // KV 同理 (kvDim0 是 Buffer 的维度)
    if (config->kvDim0 >= config->nKvHeads * config->headDim) {
        kvPtrOffset = config->kvHeadStart * config->headDim;
    }
    // [FIX END]

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *y = (float *)context->output[batchIndex];
        float *q = &query[batchIndex * config->qSliceD0 + qPtrOffset];
        NnUint pos = (NnUint)positions[batchIndex];
        assert(pos < config->seqLen);

        DEBUG_VECTOR(context, "input", y);
        DEBUG_VECTOR(context, "q", q);

        float *batchKeyCache = keyCache; 
        float *batchValueCache = valueCache;

        batchKeyCache += kvPtrOffset;
        batchValueCache += kvPtrOffset;

        multiheadAtt_F32(y, q,
            &att[batchIndex * config->nHeads0 * config->seqLen],
            batchKeyCache, batchValueCache, pos,
            config->nHeads, runtimeHeads0,
            config->nKvHeads, config->kvDim0, config->headDim, config->seqLen, 
            config->headStart, // Global Q Start (static)
            config->kvHeadStart, // Global KV Start (static)
            nThreads, threadIndex);

        if (threadIndex == 0u && runtimeElems < config->qSliceD0) {
            std::memset(&y[runtimeElems], 0, (config->qSliceD0 - runtimeElems) * sizeof(float));
        }

        if (threadIndex == 0 && batchIndex == 0 && config->layerIndex == 0) {
            // y 指向的是 mha_out buffer
            checkTensorStats("MHA_OUT", context->nodeIndex, config->layerIndex, y, runtimeElems);
        }

        DEBUG_VECTOR(context, "output", y);
    }


}

static void initMulForward(NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    ASSERT_EQ(context->inputSize.z, context->outputSize.z);
}

static void mulForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)context->opConfig;
    const float *multiplier = (float *)context->buffers[config->multiplierBufferIndex];

    const NnUint nElems = (config->staticLenUnits == 0u && config->splitKind == 0u)
        ? context->outputSize.x
        : getRuntimeLenElems(context, config->layerIndex, config->splitKind, config->splitUnit, config->staticLenUnits, context->outputSize.x);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            mul_F32(
                (float *)context->output[zOffset + y],
                (float *)context->input[zOffset + y],
                &multiplier[context->outputSize.x * (zOffset + y)],
                nElems,
                nThreads,
                threadIndex);
        }
    }
}

static void scaleForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnScaleOpCodeConfig *config = (NnScaleOpCodeConfig *)context->opConfig;
    const float *scale = (float *)context->buffers[config->scaleBufferIndex];

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            const NnUint index = z * context->inputSize.y + y;
            const float s = scale[index];
            const float *i = (float *)context->input[index];
            float *o = (float *)context->output[index];
            scale_F32(i, o, s, context->inputSize.x, nThreads, threadIndex);
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

    const NnUint nElems = (config->staticLenUnits == 0u && config->splitKind == 0u)
        ? context->outputSize.x
        : getRuntimeLenElems(context, config->layerIndex, config->splitKind, config->splitUnit, config->staticLenUnits, context->outputSize.x);
    const NnUint rowBytes = (NnUint)getBytes(context->outputSize.floatType, nElems);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            copy_UNK(
                context->output[zOffset + y],
                context->input[zOffset + y],
                rowBytes,
                nThreads,
                threadIndex);
        }
    }
}

static void castForward_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnCastOpCodeConfig *config = (NnCastOpCodeConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_Q80);

    const NnUint nElems = (config->staticLenUnits == 0u && config->splitKind == 0u)
        ? context->outputSize.x
        : getRuntimeLenElems(context, config->layerIndex, config->splitKind, config->splitUnit, config->staticLenUnits, context->outputSize.x);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            quantizeF32toQ80(
                (float *)context->input[zOffset + y],
                (NnBlockQ80 *)context->output[zOffset + y],
                nElems,
                nThreads,
                threadIndex);
        }
    }
}

static void castForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnCastOpCodeConfig *config = (NnCastOpCodeConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.floatType, F_Q80);
    ASSERT_EQ(context->outputSize.floatType, F_32);

    const NnUint nElems = (config->staticLenUnits == 0u && config->splitKind == 0u)
        ? context->outputSize.x
        : getRuntimeLenElems(context, config->layerIndex, config->splitKind, config->splitUnit, config->staticLenUnits, context->outputSize.x);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            dequantizeQ80toF32(
                (NnBlockQ80 *)context->input[zOffset + y],
                (float *)context->output[zOffset + y],
                nElems,
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
    ASSERT_EQ(context->hasInputContinuousMemory, true);
    ASSERT_EQ(context->hasOutputContinuousMemory, true);
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.y, 1);

    const NnShiftOpCodeConfig *config = (NnShiftOpCodeConfig *)context->opConfig;
    const float *indexes = (float *)context->pipes[config->indexPipeIndex];
    const NnSize rowStrideBytes = getBytes(F_32, context->inputSize.x);
    const NnUint copyElems = (config->staticLenUnits == 0u && config->splitKind == 0u)
        ? context->inputSize.x
        : getRuntimeLenElems(context, config->layerIndex, config->splitKind, config->splitUnit, config->staticLenUnits, context->inputSize.x);
    const NnSize copyBytes = getBytes(F_32, copyElems);
    NnByte *output = context->output[0];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        const NnSize index = (NnSize)indexes[batchIndex];
        assert((index + 1) * context->inputSize.x <= context->outputSize.x);
        copy_UNK(
            &output[index * rowStrideBytes],
            context->input[batchIndex],
            copyBytes,
            nThreads,
            threadIndex);
    }
}

static void softmaxForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(*context->input == *context->output);

    for (NnUint y = threadIndex; y < batchSize; y += nThreads)
        softmax_F32(
            (float *)context->output[y],
            context->outputSize.x);
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

// =====================================================================================
// Dynamic Layer Sharding (control op)
// =====================================================================================

static void writeU32Forward_ANY(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    (void)nThreads;
    if (threadIndex != 0u)
        return;
    if (!context->hasOutputContinuousMemory)
        return;
    if (context->output == nullptr || batchSize == 0u)
        return;

    const NnWriteU32OpConfig *cfg = (const NnWriteU32OpConfig *)context->opConfig;
    for (NnUint b = 0u; b < batchSize; ++b) {
        NnUint *dst = (NnUint *)context->output[b];
        dst[0] = cfg ? cfg->value : 0u;
    }
}

static void updateShardingForward_ANY(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    (void)nThreads;
    (void)batchSize;
    if (threadIndex != 0u)
        return;

    // One-time diagnostics to explain why updates are not applied.
    static std::atomic<bool> printedNullSharding{false};
    static std::atomic<bool> printedNonContigInput{false};
    static std::atomic<bool> printedNullInput{false};
    static std::atomic<bool> printedMagicMismatch{false};
    static std::atomic<bool> printedVersionMismatch{false};

    if (context->layerSharding == nullptr)
    {
        if (shardingMatmulLogEnabled()) {
            bool expected = false;
            if (printedNullSharding.compare_exchange_strong(expected, true)) {
                printf("[SHARD][CPU][UPDATE][SKIP] layerSharding=null op=%u name=%s\n",
                    context->opIndex,
                    context->name ? context->name : "(null)");
            }
        }
        return;
    }
    if (!context->hasInputContinuousMemory)
    {
        if (shardingMatmulLogEnabled()) {
            bool expected = false;
            if (printedNonContigInput.compare_exchange_strong(expected, true)) {
                printf("[SHARD][CPU][UPDATE][SKIP] inputNotContiguous op=%u name=%s\n",
                    context->opIndex,
                    context->name ? context->name : "(null)");
            }
        }
        return;
    }

    if (context->input == nullptr || context->input[0] == nullptr) {
        if (shardingMatmulLogEnabled()) {
            bool expected = false;
            if (printedNullInput.compare_exchange_strong(expected, true)) {
                printf("[SHARD][CPU][UPDATE][SKIP] inputNull op=%u name=%s\n",
                    context->opIndex,
                    context->name ? context->name : "(null)");
            }
        }
        return;
    }

    const NnUint *u32 = (const NnUint *)context->input[0];
    const NnShardingUpdateHeader *hdr = (const NnShardingUpdateHeader *)u32;

    // Allow root to enable matmul sharding logs on all stage nodes.
    if ((hdr->flags & NN_SHARDING_UPDATE_FLAG_DEBUG_MATMUL) != 0u) {
        g_forceShardingMatmulLog.store(1, std::memory_order_release);
    }

    // A no-op header is encoded as magic=0. This is an expected state when
    // there is no pending online sharding update.
    if (hdr->magic == 0u) {
        return;
    }

    if (hdr->magic != NN_SHARDING_UPDATE_MAGIC)
    {
        if (shardingMatmulLogEnabled()) {
            bool expected = false;
            if (printedMagicMismatch.compare_exchange_strong(expected, true)) {
                printf("[SHARD][CPU][UPDATE][SKIP] magicMismatch op=%u name=%s u32[0]=0x%08x\n",
                    context->opIndex,
                    context->name ? context->name : "(null)",
                    (unsigned)u32[0]);
            }
        }
        return;
    }
    if (hdr->version != NN_SHARDING_UPDATE_VERSION)
    {
        if (shardingMatmulLogEnabled()) {
            bool expected = false;
            if (printedVersionMismatch.compare_exchange_strong(expected, true)) {
                printf("[SHARD][CPU][UPDATE][SKIP] versionMismatch op=%u name=%s version=%u expected=%u\n",
                    context->opIndex,
                    context->name ? context->name : "(null)",
                    hdr->version,
                    (unsigned)NN_SHARDING_UPDATE_VERSION);
            }
        }
        return;
    }
    // Version management: keep the newest scheme (monotonic epoch).
    const NnUint curEpoch = context->layerSharding->epoch.load(std::memory_order_acquire);
    if (hdr->epoch <= curEpoch) {
        return;
    }
    if (hdr->nNodes != context->layerSharding->nNodes)
        return;
    if (hdr->layerIndex >= context->layerSharding->nLayers)
        return;

    // Apply this update to a layer range. Default is the single layerIndex.
    NnUint applyStart = hdr->applyStartLayer;
    NnUint applyEnd = hdr->applyEndLayerExclusive;
    if (applyEnd == 0u || applyEnd > context->layerSharding->nLayers) {
        applyEnd = context->layerSharding->nLayers;
    }
    if (applyStart >= context->layerSharding->nLayers) {
        applyStart = hdr->layerIndex;
    }
    if (applyEnd <= applyStart) {
        applyStart = hdr->layerIndex;
        applyEnd = hdr->layerIndex + 1u;
    }
    if (applyEnd > context->layerSharding->nLayers) {
        applyEnd = context->layerSharding->nLayers;
    }

    const NnUint *payloadBase = (const NnUint *)(u32 + (sizeof(NnShardingUpdateHeader) / sizeof(NnUint)));

    const NnUint *startsByKind[N_SPLIT_KINDS];
    const NnUint *lensByKind[N_SPLIT_KINDS];
    for (NnUint k = 0u; k < (NnUint)N_SPLIT_KINDS; ++k) {
        const NnUint *p = payloadBase + (NnSize)k * 2u * hdr->nNodes;
        startsByKind[k] = p;
        lensByKind[k] = p + hdr->nNodes;
    }

    for (NnUint li = applyStart; li < applyEnd; ++li) {
        context->layerSharding->setLayerSplit(li, SPLIT_HEAD, startsByKind[SPLIT_HEAD], lensByKind[SPLIT_HEAD]);
        context->layerSharding->setLayerSplit(li, SPLIT_KV_HEAD, startsByKind[SPLIT_KV_HEAD], lensByKind[SPLIT_KV_HEAD]);
        context->layerSharding->setLayerSplit(li, SPLIT_VOCAB, startsByKind[SPLIT_VOCAB], lensByKind[SPLIT_VOCAB]);
        context->layerSharding->setLayerSplit(li, SPLIT_FFN, startsByKind[SPLIT_FFN], lensByKind[SPLIT_FFN]);
        context->layerSharding->setLayerSplit(li, SPLIT_DIM, startsByKind[SPLIT_DIM], lensByKind[SPLIT_DIM]);

        // Build per-unit owner maps for head / kv-head. These are used for KV cache migration.
        NnLayerSplits &ls = context->layerSharding->layers[li];
        buildOwnerMapFromSplit(&ls.headSplit, context->layerSharding->nNodes, &ls.headOwners, &ls.nHeadUnits);
        buildOwnerMapFromSplit(&ls.kvHeadSplit, context->layerSharding->nNodes, &ls.kvHeadOwners, &ls.nKvHeadUnits);
    }

    context->layerSharding->epoch.store(hdr->epoch, std::memory_order_release);

    // Record the last applied range so matmul can tag affected layers as UPDATE.
    g_lastAppliedShardingEpoch.store(hdr->epoch, std::memory_order_release);
    g_lastAppliedShardingRangeStart.store(applyStart, std::memory_order_release);
    g_lastAppliedShardingRangeEnd.store(applyEnd, std::memory_order_release);
    g_lastLoggedMatmulUpdateKey.store(0u, std::memory_order_release);
    g_lastLoggedMatmulBaselineKey.store(0u, std::memory_order_release);

    if (shardingMatmulLogEnabled()) {
        const int layerFilter = shardingMatmulLogLayerFilter();
        const int nodeFilter = shardingMatmulLogNodeFilter();
        if ((layerFilter < 0 || (NnUint)layerFilter == hdr->layerIndex) && (nodeFilter < 0 || (NnUint)nodeFilter == context->nodeIndex)) {
            const NnLayerSplits &ls = context->layerSharding->layers[hdr->layerIndex];
            const NnDimSplit *hs = getLayerSplitPtr(ls, SPLIT_HEAD);
            const NnDimSplit *khs = getLayerSplitPtr(ls, SPLIT_KV_HEAD);
            const NnDimSplit *vs = getLayerSplitPtr(ls, SPLIT_VOCAB);
            const NnDimSplit *fs = getLayerSplitPtr(ls, SPLIT_FFN);
            const NnDimSplit *ds = getLayerSplitPtr(ls, SPLIT_DIM);

            const NnUint n = context->nodeIndex;
            printf("[SHARD][CPU][UPDATE] node=%u layer=%u epoch=%u head=%u+%u kvHead=%u+%u vocab=%u+%u ffn=%u+%u dim=%u+%u\n",
                context->nodeIndex,
                hdr->layerIndex,
                hdr->epoch,
                (hs && hs->starts) ? hs->starts[n] : 0u,
                (hs && hs->lengths) ? hs->lengths[n] : 0u,
                (khs && khs->starts) ? khs->starts[n] : 0u,
                (khs && khs->lengths) ? khs->lengths[n] : 0u,
                (vs && vs->starts) ? vs->starts[n] : 0u,
                (vs && vs->lengths) ? vs->lengths[n] : 0u,
                (fs && fs->starts) ? fs->starts[n] : 0u,
                (fs && fs->lengths) ? fs->lengths[n] : 0u,
                (ds && ds->starts) ? ds->starts[n] : 0u,
                (ds && ds->lengths) ? ds->lengths[n] : 0u);

            // Minimal usage point: show owner mapping for this node's head/kv-head start.
            const NnUint headStart = (hs && hs->starts) ? hs->starts[n] : 0u;
            const NnUint kvHeadStart = (khs && khs->starts) ? khs->starts[n] : 0u;
            const NnUint headOwner = (ls.headOwners && headStart < ls.nHeadUnits) ? ls.headOwners[headStart] : 0u;
            const NnUint kvHeadOwner = (ls.kvHeadOwners && kvHeadStart < ls.nKvHeadUnits) ? ls.kvHeadOwners[kvHeadStart] : 0u;
            printf("[SHARD][CPU][MAP] node=%u layer=%u headStart=%u owner=%u kvHeadStart=%u owner=%u\n",
                context->nodeIndex,
                hdr->layerIndex,
                headStart,
                headOwner,
                kvHeadStart,
                kvHeadOwner);
        }
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
    if (code == OP_WRITE_U32) {
        return writeU32Forward_ANY;
    }
    if (code == OP_UPDATE_SHARDING) {
        return updateShardingForward_ANY;
    }
    return nullptr;
}
