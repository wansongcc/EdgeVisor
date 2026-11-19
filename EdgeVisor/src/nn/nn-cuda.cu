#include "nn-cuda.hpp"
#include "product_log.hpp"
#include "nn/io-profile.hpp"
#include "llm.hpp"
#include "plan-command.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

static std::string cudaErrorString(cudaError_t err, const char *expr) {
    std::ostringstream out;
    out << expr << " failed: " << cudaGetErrorString(err) << " (" << (int)err << ")";
    return out.str();
}

#define NN_CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) throw std::runtime_error(cudaErrorString(_err, #expr)); \
} while (0)

static std::string cublasErrorString(cublasStatus_t status, const char *expr) {
    std::ostringstream out;
    out << expr << " failed: ";
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: out << "CUBLAS_STATUS_SUCCESS"; break;
        case CUBLAS_STATUS_NOT_INITIALIZED: out << "CUBLAS_STATUS_NOT_INITIALIZED"; break;
        case CUBLAS_STATUS_ALLOC_FAILED: out << "CUBLAS_STATUS_ALLOC_FAILED"; break;
        case CUBLAS_STATUS_INVALID_VALUE: out << "CUBLAS_STATUS_INVALID_VALUE"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH: out << "CUBLAS_STATUS_ARCH_MISMATCH"; break;
        case CUBLAS_STATUS_MAPPING_ERROR: out << "CUBLAS_STATUS_MAPPING_ERROR"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED: out << "CUBLAS_STATUS_EXECUTION_FAILED"; break;
        case CUBLAS_STATUS_INTERNAL_ERROR: out << "CUBLAS_STATUS_INTERNAL_ERROR"; break;
        case CUBLAS_STATUS_NOT_SUPPORTED: out << "CUBLAS_STATUS_NOT_SUPPORTED"; break;
        case CUBLAS_STATUS_LICENSE_ERROR: out << "CUBLAS_STATUS_LICENSE_ERROR"; break;
        default: out << "unknown(" << (int)status << ")"; break;
    }
    return out.str();
}

#define NN_CUBLAS_CHECK(expr) do { \
    cublasStatus_t _status = (expr); \
    if (_status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(cublasErrorString(_status, #expr)); \
} while (0)

static std::string sizeToString(const NnSize3D &s) {
    std::ostringstream out;
    out << "{type=" << (int)s.floatType
        << ", z=" << s.z
        << ", y=" << s.y
        << ", x=" << s.x
        << ", bytes=" << s.nBytes
        << "}";
    return out.str();
}

static void validateRange(const char *what, const std::string &name, NnSize bufferSize, NnSize offset, NnSize nBytes) {
    if (offset > bufferSize || nBytes > bufferSize - offset) {
        std::ostringstream out;
        out << "CUDA " << what << " out of range"
            << " [buffer=" << name
            << ", bufferBytes=" << (unsigned long long)bufferSize
            << ", offset=" << (unsigned long long)offset
            << ", nBytes=" << (unsigned long long)nBytes
            << "]";
        throw std::runtime_error(out.str());
    }
}

struct NnCudaPointerRef {
    NnCudaBuffer *buffer;
    NnPointerLayout layout;
};

static NnCudaPointerRef resolveCudaPointer(
    NnCudaDevice *device,
    NnNetConfig *netConfig,
    NnNodeConfig *nodeConfig,
    const NnUnevenPartitionPlan *plan,
    const NnPointerConfig *config) {
    NnCudaPointerRef ref{};
    ref.layout = resolvePointerLayout(netConfig, nodeConfig, plan, config);
    ref.buffer = (config->source == SRC_PIPE)
        ? device->data.resolvePipe(config->pointerIndex)
        : device->data.resolveBuffer(config->pointerIndex);
    return ref;
}

static const NnTensorView *opTensorView(const NnOpConfig &op) {
    if (op.config == nullptr) return nullptr;
    switch (op.code) {
        case OP_CAST: return &((const NnCastOpCodeConfig *)op.config)->view;
        case OP_SILU: return &((const NnSiluOpCodeConfig *)op.config)->view;
        case OP_GELU: return &((const NnGeluOpCodeConfig *)op.config)->view;
        case OP_MUL: return &((const NnMulOpCodeConfig *)op.config)->view;
        case OP_SILU_MUL: return &((const NnMulOpCodeConfig *)op.config)->view;
        case OP_SCALE: return &((const NnScaleOpCodeConfig *)op.config)->view;
        case OP_INV_RMS: return &((const NnInvRmsOpConfig *)op.config)->view;
        case OP_RMS_NORM: return &((const NnRmsNormOpConfig *)op.config)->view;
        case OP_ROPE: return &((const NnRopeOpConfig *)op.config)->view;
        case OP_SOFTMAX: return &((const NnSoftmaxOpCodeConfig *)op.config)->view;
        default: return nullptr;
    }
}

static NnTensorViewLayout resolveOpView(const NnOpConfig &op, const NnSize3D &fallback, NnSize physicalElements) {
    return resolveTensorView(opTensorView(op), 0u, fallback.x, physicalElements);
}

static void validateQ80View(const char *opName, const NnTensorViewLayout &view) {
    if (view.strideX != 1u || (view.offset % Q80_BLOCK_SIZE) != 0u || (view.sizeX % Q80_BLOCK_SIZE) != 0u) {
        throw std::runtime_error(std::string("CUDA ") + opName + " requires contiguous Q80 block-aligned view");
    }
}

static inline unsigned int cudaBlocks(NnSize n, unsigned int blockSize = 256u) {
    return (unsigned int)((n + blockSize - 1u) / blockSize);
}

static unsigned int cudaFloorPowerOfTwo(unsigned int v) {
    if (v == 0u) return 0u;
    unsigned int p = 1u;
    while ((p << 1u) != 0u && (p << 1u) <= v) p <<= 1u;
    return p;
}

static unsigned int cudaClampLaunchBlockSize(unsigned int requested, unsigned int deviceMax, unsigned int hardMax) {
    unsigned int limit = std::min(deviceMax == 0u ? 1024u : deviceMax, hardMax == 0u ? 1024u : hardMax);
    if (limit < 32u) limit = 32u;
    unsigned int value = std::min(requested == 0u ? 32u : requested, limit);
    value = cudaFloorPowerOfTwo(value);
    if (value < 32u) value = 32u;
    return value;
}

static NnCudaLaunchConfig buildCudaLaunchConfig(const cudaDeviceProp &prop) {
    NnCudaLaunchConfig cfg{};
    cfg.computeCapabilityMajor = prop.major;
    cfg.computeCapabilityMinor = prop.minor;
    cfg.sm = prop.major * 10 + prop.minor;
    cfg.multiprocessorCount = (NnUint)std::max(prop.multiProcessorCount, 0);
    cfg.maxThreadsPerBlock = (NnUint)std::max(prop.maxThreadsPerBlock, 0);
    cfg.warpSize = (NnUint)std::max(prop.warpSize, 1);
    cfg.integrated = prop.integrated != 0;

    const unsigned int maxThreads = (unsigned int)std::max(prop.maxThreadsPerBlock, 32);
    const bool preVolta = cfg.sm > 0 && cfg.sm < 70;
    const bool ampereOrNewerDiscrete = cfg.sm >= 80 && !cfg.integrated;
    const bool integratedOrLowPower = cfg.integrated;

    const unsigned int elementwiseRequest = ampereOrNewerDiscrete ? 512u : (preVolta || integratedOrLowPower ? 128u : 256u);
    const unsigned int reductionRequest = preVolta || integratedOrLowPower ? 128u : 256u;
    const unsigned int attentionRequest = preVolta || integratedOrLowPower ? 128u : 256u;
    const unsigned int moeGateRequest = integratedOrLowPower ? 64u : 128u;

    cfg.elementwiseBlockSize = (NnUint)cudaClampLaunchBlockSize(elementwiseRequest, maxThreads, 1024u);
    cfg.reductionBlockSize = (NnUint)cudaClampLaunchBlockSize(reductionRequest, maxThreads, 256u);
    cfg.attentionBlockSize = (NnUint)cudaClampLaunchBlockSize(attentionRequest, maxThreads, 256u);
    cfg.q80q40SmallKBlockSize = (NnUint)cudaClampLaunchBlockSize(elementwiseRequest, maxThreads, 512u);
    cfg.q80q40LargeKBlockSize = cfg.reductionBlockSize;
    cfg.softmaxBlockSize = cfg.reductionBlockSize;
    cfg.moeGateBlockSize = (NnUint)cudaClampLaunchBlockSize(moeGateRequest, maxThreads, 256u);
    cfg.q80q40SmallKMaxBlocks = (NnUint)(ampereOrNewerDiscrete ? 8u : 4u);
    return cfg;
}

static std::string cudaLaunchConfigToString(const NnCudaLaunchConfig &cfg) {
    std::ostringstream out;
    out << "CUDA LaunchConfig: sm_" << cfg.sm
        << " integrated=" << (cfg.integrated ? 1 : 0)
        << " mp=" << cfg.multiprocessorCount
        << " warp=" << cfg.warpSize
        << " maxThreadsPerBlock=" << cfg.maxThreadsPerBlock
        << " elementwise=" << cfg.elementwiseBlockSize
        << " reduction=" << cfg.reductionBlockSize
        << " attention=" << cfg.attentionBlockSize
        << " q80q40SmallK=" << cfg.q80q40SmallKBlockSize
        << " q80q40LargeK=" << cfg.q80q40LargeKBlockSize
        << " softmax=" << cfg.softmaxBlockSize
        << " moeGate=" << cfg.moeGateBlockSize
        << " smallKMaxBlocks=" << cfg.q80q40SmallKMaxBlocks;
    return out.str();
}

__device__ static inline uint16_t cudaFloatToHalfBits(float x) {
    __half h = __float2half(x);
    return __half_as_ushort(h);
}

__device__ static inline float cudaHalfBitsToFloat(uint16_t h) {
    const int sign = (h & 0x8000u) ? -1 : 1;
    const int exp = (h >> 10) & 0x1fu;
    const int mant = h & 0x03ffu;
    if (exp == 0) {
        return sign * ldexpf((float)mant, -24);
    }
    if (exp == 31) {
        return mant == 0 ? sign * 3.4028234663852886e38f : 0.0f;
    }
    return sign * ldexpf((float)(1024 + mant), exp - 25);
}

/* __dp4a is a sm_61+ instruction. Host-side compilation passes and any device
 * build below sm_61 keep the scalar accumulation path. */
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610
#define NN_CUDA_HAS_DP4A 1
#else
#define NN_CUDA_HAS_DP4A 0
#endif

/* Largest aBlocks that still fits the packed x cache in the default 48KB
 * shared-memory budget: 1024 blocks needs 40KB. K of 32768 covers every
 * dimension the model uses; anything wider reads x from global memory while
 * staying on the warp dp4a path. */
#define NN_CUDA_MATMUL_SHARED_MAX_BLOCKS 1024u

/* NnBlockQ40 (18 bytes) and NnBlockQ80 (34 bytes) only guarantee 2-byte
 * alignment, so ->qs must never be reinterpreted as an int pointer. Two uint16
 * loads are always legal and assemble the same four packed int8 lanes. */
__device__ static inline int cudaLoadInt8x4(const int8_t *p) {
    const uint16_t *w = (const uint16_t *)p;
    return (int)((unsigned int)w[0] | ((unsigned int)w[1] << 16));
}

__device__ static inline int cudaSumInt8x4(int packed) {
#if NN_CUDA_HAS_DP4A
    return __dp4a(packed, 0x01010101, 0);
#else
    const unsigned int u = (unsigned int)packed;
    int sum = 0;
    #pragma unroll
    for (NnUint i = 0u; i < 4u; ++i) {
        sum += (int)(int8_t)((u >> (8u * i)) & 0xffu);
    }
    return sum;
#endif
}

/* Q80 x Q40 block dot product as a raw integer accumulator.
 * The q40 nibbles are kept in 0..15 so __dp4a can consume them directly (all
 * values stay inside the signed int8 range); the +8 bias this introduces is
 * removed once per block by subtracting 8 * sum(x). That is integer-identical
 * to the (w - 8) * x form while keeping the inner loop subtraction-free. */
__device__ static inline int cudaQ80Q40DotPacked(const int *xWords, int xSum, const uint8_t *wq) {
#if NN_CUDA_HAS_DP4A
    int acc = 0;
    #pragma unroll
    for (NnUint g = 0u; g < 4u; ++g) {
        const int w32 = cudaLoadInt8x4((const int8_t *)(wq + 4u * g));
        acc = __dp4a(xWords[g], w32 & 0x0f0f0f0f, acc);
        acc = __dp4a(xWords[4u + g], (w32 >> 4) & 0x0f0f0f0f, acc);
    }
    return acc - 8 * xSum;
#else
    const int8_t *xq = (const int8_t *)xWords;
    int acc = 0;
    for (NnUint k = 0u; k < Q40_BLOCK_SIZE / 2u; ++k) {
        const int w0 = (int)(wq[k] & 0x0fu) - 8;
        const int w1 = (int)(wq[k] >> 4) - 8;
        acc += w0 * (int)xq[k] + w1 * (int)xq[k + Q80_BLOCK_SIZE / 2u];
    }
    (void)xSum;
    return acc;
#endif
}

/* Same product straight out of global memory. The input block is packed once so
 * the dp4a path and the scalar fallback share a single read of x. */
__device__ static inline int cudaQ80Q40DotGlobal(const NnBlockQ80 *xb, const NnBlockQ40 *wb) {
    int xWords[8];
    int xSum = 0;
    #pragma unroll
    for (NnUint w = 0u; w < 8u; ++w) {
        xWords[w] = cudaLoadInt8x4(xb->qs + 4u * w);
        xSum += cudaSumInt8x4(xWords[w]);
    }
    return cudaQ80Q40DotPacked(xWords, xSum, wb->qs);
}

__device__ static inline float cudaWarpReduceMax(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static inline float cudaWarpReduceSum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

/* Block-wide max over a 32-slot staging array. Two barriers; the result is
 * re-read from the staging slot so a follow-up reduction on its own slot cannot
 * race this one. blockDim.x is always a power of two >= 32. */
__device__ static inline float cudaBlockReduceMax(float v, float *shared) {
    const NnUint nWarps = (blockDim.x + 31u) >> 5u;
    const NnUint lane = threadIdx.x & 31u;
    const NnUint wid = threadIdx.x >> 5u;
    v = cudaWarpReduceMax(v);
    if (lane == 0u) shared[wid] = v;
    __syncthreads();
    if (wid == 0u) {
        float w = lane < nWarps ? shared[lane] : -3.4028234663852886e38f;
        w = cudaWarpReduceMax(w);
        if (lane == 0u) shared[0] = w;
    }
    __syncthreads();
    return shared[0];
}

__device__ static inline float cudaBlockReduceSum(float v, float *shared) {
    const NnUint nWarps = (blockDim.x + 31u) >> 5u;
    const NnUint lane = threadIdx.x & 31u;
    const NnUint wid = threadIdx.x >> 5u;
    v = cudaWarpReduceSum(v);
    if (lane == 0u) shared[wid] = v;
    __syncthreads();
    if (wid == 0u) {
        float w = lane < nWarps ? shared[lane] : 0.0f;
        w = cudaWarpReduceSum(w);
        if (lane == 0u) shared[0] = w;
    }
    __syncthreads();
    return shared[0];
}

__device__ static inline NnByte *cudaRowBase(NnByte *base, NnPointerLayout layout, NnUint z, NnUint y) {
    return base + z * layout.zStrideBytes + y * layout.batchStrideBytes + layout.byteOffset;
}

static inline NnByte *cudaHostRowBase(NnByte *base, const NnPointerLayout &layout, NnUint z, NnUint y) {
    return base + z * layout.zStrideBytes + y * layout.batchStrideBytes + layout.byteOffset;
}

static bool isCudaControlOp(NnOpCode code) {
    return code == OP_PLAN_BARRIER || code == OP_PLAN_APPLY;
}

static const NnStageConfig *findStageForNodeCuda(const NnUnevenPartitionPlan *plan, NnUint nodeIndex) {
    if (plan == nullptr || plan->stages == nullptr) return nullptr;
    for (NnUint s = 0u; s < plan->nStages; ++s) {
        const NnStageConfig *st = &plan->stages[s];
        for (NnUint i = 0u; i < st->nNodes; ++i) {
            if (st->nodeIndices[i] == nodeIndex) return st;
        }
    }
    return nullptr;
}

static bool cudaStageContains(const NnStageConfig *stage, NnUint node) {
    if (stage == nullptr) return false;
    for (NnUint i = 0u; i < stage->nNodes; ++i) {
        if (stage->nodeIndices[i] == node) return true;
    }
    return false;
}

static int cudaStageRank(const NnStageConfig *stage, NnUint node) {
    if (stage == nullptr) return -1;
    for (NnUint i = 0u; i < stage->nNodes; ++i) {
        if (stage->nodeIndices[i] == node) return (int)i;
    }
    return -1;
}

static bool cudaStageAdjacent(const NnStageConfig *stage, NnUint a, NnUint b) {
    const int ra = cudaStageRank(stage, a);
    const int rb = cudaStageRank(stage, b);
    if (ra < 0 || rb < 0) return false;
    const int d = (ra >= rb) ? (ra - rb) : (rb - ra);
    return d == 1;
}

static void cudaRecomputeStarts(const NnStageConfig *stage, NnDimSplit &split) {
    if (stage == nullptr || split.starts == nullptr || split.lengths == nullptr) return;
    NnUint run = 0u;
    for (NnUint i = 0u; i < stage->nNodes; ++i) {
        const NnUint n = stage->nodeIndices[i];
        split.starts[n] = run;
        run += split.lengths[n];
    }
}

static NnUint cudaGqaGroupSize(const NnUnevenPartitionPlan *plan, const NnStageConfig *stage) {
    if (plan == nullptr || stage == nullptr || plan->headSplit.lengths == nullptr || plan->kvHeadSplit.lengths == nullptr) return 1u;
    NnUint q = 0u;
    NnUint kv = 0u;
    for (NnUint i = 0u; i < stage->nNodes; ++i) {
        const NnUint n = stage->nodeIndices[i];
        q += plan->headSplit.lengths[n];
        kv += plan->kvHeadSplit.lengths[n];
    }
    if (q != 0u && kv != 0u && (q % kv) == 0u) {
        const NnUint g = q / kv;
        return g == 0u ? 1u : g;
    }
    return 1u;
}

static bool cudaHasHeadDeltaForStage(const std::vector<int> &delta, const NnStageConfig *stage) {
    if (stage == nullptr) return false;
    for (NnUint i = 0u; i < stage->nNodes; ++i) {
        const NnUint n = stage->nodeIndices[i];
        if (n < delta.size() && delta[n] != 0) return true;
    }
    return false;
}

static bool cudaValidateKvShadowCoverage(
    const NnUnevenPartitionPlan *plan,
    const NnStageConfig *stage,
    const std::vector<int> &delta,
    char *reason,
    size_t reasonSize) {
    auto setReason = [&](const char *value) {
        if (reason != nullptr && reasonSize != 0u) std::snprintf(reason, reasonSize, "%s", value);
    };
    if (!getEnableStageFullWeights()) { setReason("kv_shadow_requires_stage_full_weights"); return false; }
    if (!getEnableKvRedundancyDuringMigration()) { setReason("kv_shadow_redundancy_disabled"); return false; }
    if (plan == nullptr || stage == nullptr ||
        plan->kvHeadSplit.lengths == nullptr ||
        plan->kvHeadComputeSplit.starts == nullptr ||
        plan->kvHeadComputeSplit.lengths == nullptr) {
        setReason("kv_shadow_split_missing");
        return false;
    }
    NnUint newStart = 0u;
    for (NnUint i = 0u; i < stage->nNodes; ++i) {
        const NnUint n = stage->nodeIndices[i];
        if (n >= delta.size()) { setReason("kv_shadow_delta_missing"); return false; }
        const int newLenSigned = (int)plan->kvHeadSplit.lengths[n] + delta[n];
        if (newLenSigned < 0) { setReason("kv_shadow_underflow"); return false; }
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

static bool cudaApplyPlanDeltas(
    NnCudaDevice *device,
    NnUnevenPartitionPlan *plan,
    const NnStageConfig *stage,
    const std::vector<int> &deltaHeadOrKv,
    const std::vector<int> &deltaFfn,
    NnUint msgEpoch,
    NnUint layerIndex,
    NnUint pos) {
    if (device == nullptr || plan == nullptr || stage == nullptr) return false;
    const NnUint gqaGroup = cudaGqaGroupSize(plan, stage);
    bool changed = false;

    if (cudaHasHeadDeltaForStage(deltaHeadOrKv, stage)) {
        if (gqaGroup > 1u && plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
            if (getEnableKvRedundancyDuringMigration() && !getAllowNoShadowHeadMigration()) {
                char reason[96] = {0};
                if (!cudaValidateKvShadowCoverage(plan, stage, deltaHeadOrKv, reason, sizeof(reason))) {
                    std::printf("🧭 [plan][apply][cuda] node=%u stage=%u epoch=%u layer=%u pos=%u reject: %s\n",
                        (unsigned)device->getNodeIndex(), (unsigned)stage->stageIndex, (unsigned)msgEpoch,
                        (unsigned)layerIndex, (unsigned)pos, reason);
                    std::fflush(stdout);
                    return false;
                }
            }
            for (NnUint i = 0u; i < stage->nNodes; ++i) {
                const NnUint n = stage->nodeIndices[i];
                const int newLen = (int)plan->kvHeadSplit.lengths[n] + deltaHeadOrKv[n];
                if (newLen < 0) return false;
            }
            for (NnUint i = 0u; i < stage->nNodes; ++i) {
                const NnUint n = stage->nodeIndices[i];
                if (deltaHeadOrKv[n] == 0) continue;
                plan->kvHeadSplit.lengths[n] = (NnUint)((int)plan->kvHeadSplit.lengths[n] + deltaHeadOrKv[n]);
                changed = true;
            }
            if (changed) {
                cudaRecomputeStarts(stage, plan->kvHeadSplit);
                for (NnUint i = 0u; i < stage->nNodes; ++i) {
                    const NnUint n = stage->nodeIndices[i];
                    plan->headSplit.starts[n] = plan->kvHeadSplit.starts[n] * gqaGroup;
                    plan->headSplit.lengths[n] = plan->kvHeadSplit.lengths[n] * gqaGroup;
                }
            }
        } else {
            if (!plan->headSplit.starts || !plan->headSplit.lengths) return false;
            for (NnUint i = 0u; i < stage->nNodes; ++i) {
                const NnUint n = stage->nodeIndices[i];
                const int newLen = (int)plan->headSplit.lengths[n] + deltaHeadOrKv[n];
                if (newLen < 0) return false;
            }
            for (NnUint i = 0u; i < stage->nNodes; ++i) {
                const NnUint n = stage->nodeIndices[i];
                if (deltaHeadOrKv[n] == 0) continue;
                plan->headSplit.lengths[n] = (NnUint)((int)plan->headSplit.lengths[n] + deltaHeadOrKv[n]);
                changed = true;
            }
            if (changed) cudaRecomputeStarts(stage, plan->headSplit);
        }
    }

    if (!deltaFfn.empty()) {
        if (!plan->ffnSplit.starts || !plan->ffnSplit.lengths) return false;
        for (NnUint i = 0u; i < stage->nNodes; ++i) {
            const NnUint n = stage->nodeIndices[i];
            const int d = n < deltaFfn.size() ? deltaFfn[n] : 0;
            const int newLen = (int)plan->ffnSplit.lengths[n] + d;
            if (newLen < 0) return false;
        }
        bool ffnChanged = false;
        for (NnUint i = 0u; i < stage->nNodes; ++i) {
            const NnUint n = stage->nodeIndices[i];
            const int d = n < deltaFfn.size() ? deltaFfn[n] : 0;
            if (d == 0) continue;
            plan->ffnSplit.lengths[n] = (NnUint)((int)plan->ffnSplit.lengths[n] + d);
            ffnChanged = true;
        }
        if (ffnChanged) {
            cudaRecomputeStarts(stage, plan->ffnSplit);
            changed = true;
        }
    }

    if (changed) {
        std::printf("🧭 [plan][apply][cuda] node=%u stage=%u epoch=%u layer=%u pos=%u applied qHeads=%u kvHeads=%u ffnDim=%u\n",
            (unsigned)device->getNodeIndex(),
            (unsigned)stage->stageIndex,
            (unsigned)msgEpoch,
            (unsigned)layerIndex,
            (unsigned)pos,
            (unsigned)(plan->headSplit.lengths ? plan->headSplit.lengths[device->getNodeIndex()] : 0u),
            (unsigned)(plan->kvHeadSplit.lengths ? plan->kvHeadSplit.lengths[device->getNodeIndex()] : 0u),
            (unsigned)(plan->ffnSplit.lengths ? plan->ffnSplit.lengths[device->getNodeIndex()] : 0u));
        std::fflush(stdout);
        device->setPlanEpoch(msgEpoch);
    }
    return changed;
}

__global__ static void castF32F32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    NnTensorViewLayout view,
    NnUint batchSize) {
    const NnSize total = (NnSize)output.logicalSize.z * batchSize * view.sizeY * view.sizeX;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % view.sizeX);
    NnSize t = tid / view.sizeX;
    const NnUint vy = (NnUint)(t % view.sizeY);
    t /= view.sizeY;
    const NnUint y = (NnUint)(t % batchSize);
    const NnUint z = (NnUint)(t / batchSize);
    const NnSize idx = view.offset + vy * view.strideY + x * view.strideX;
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, z, y);
    float *out = (float *)cudaRowBase(outputBase, output, z, y);
    out[idx] = in[idx];
}

__global__ static void siluKernel(
    NnByte *outputBase,
    NnPointerLayout output,
    NnTensorViewLayout view,
    NnUint batchSize) {
    const NnSize total = (NnSize)output.logicalSize.z * batchSize * view.sizeY * view.sizeX;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % view.sizeX);
    NnSize t = tid / view.sizeX;
    const NnUint vy = (NnUint)(t % view.sizeY);
    t /= view.sizeY;
    const NnUint y = (NnUint)(t % batchSize);
    const NnUint z = (NnUint)(t / batchSize);
    const NnSize idx = view.offset + vy * view.strideY + x * view.strideX;
    float *out = (float *)cudaRowBase(outputBase, output, z, y);
    const float v = out[idx];
    out[idx] = v / (1.0f + expf(-v));
}

__global__ static void geluKernel(
    NnByte *outputBase,
    NnPointerLayout output,
    NnTensorViewLayout view,
    NnUint batchSize) {
    const NnSize total = (NnSize)output.logicalSize.z * batchSize * view.sizeY * view.sizeX;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % view.sizeX);
    NnSize t = tid / view.sizeX;
    const NnUint vy = (NnUint)(t % view.sizeY);
    t /= view.sizeY;
    const NnUint y = (NnUint)(t % batchSize);
    const NnUint z = (NnUint)(t / batchSize);
    const NnSize idx = view.offset + vy * view.strideY + x * view.strideX;
    float *out = (float *)cudaRowBase(outputBase, output, z, y);
    const float v = out[idx];
    out[idx] = 0.5f * v * (1.0f + tanhf(0.79788456080286535587989211986876f * v * (1.0f + 0.044715f * v * v)));
}

__global__ static void mulKernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    const float *multiplier,
    NnUint multRowStride,
    NnTensorViewLayout view,
    NnUint batchSize) {
    const NnSize total = (NnSize)input.logicalSize.z * batchSize * view.sizeY * view.sizeX;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % view.sizeX);
    NnSize t = tid / view.sizeX;
    const NnUint vy = (NnUint)(t % view.sizeY);
    t /= view.sizeY;
    const NnUint y = (NnUint)(t % batchSize);
    const NnUint z = (NnUint)(t / batchSize);
    const NnUint row = z * input.logicalSize.y + y;
    const NnSize idx = view.offset + vy * view.strideY + x * view.strideX;
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, z, y);
    float *out = (float *)cudaRowBase(outputBase, output, z, y);
    const float *m = multiplier + row * multRowStride;
    out[idx] = in[idx] * m[idx];
}

__global__ static void siluMulKernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    const float *multiplier,
    NnUint multRowStride,
    NnTensorViewLayout view,
    NnUint batchSize) {
    const NnSize total = (NnSize)input.logicalSize.z * batchSize * view.sizeY * view.sizeX;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % view.sizeX);
    NnSize t = tid / view.sizeX;
    const NnUint vy = (NnUint)(t % view.sizeY);
    t /= view.sizeY;
    const NnUint y = (NnUint)(t % batchSize);
    const NnUint z = (NnUint)(t / batchSize);
    const NnUint row = z * input.logicalSize.y + y;
    const NnSize idx = view.offset + vy * view.strideY + x * view.strideX;
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, z, y);
    float *out = (float *)cudaRowBase(outputBase, output, z, y);
    const float *m = multiplier + row * multRowStride;
    const float v = in[idx];
    out[idx] = (v / (1.0f + expf(-v))) * m[idx];
}

__global__ static void scaleKernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    const float *scale,
    NnTensorViewLayout view,
    NnUint batchSize) {
    const NnSize total = (NnSize)input.logicalSize.z * batchSize * view.sizeY * view.sizeX;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % view.sizeX);
    NnSize t = tid / view.sizeX;
    const NnUint vy = (NnUint)(t % view.sizeY);
    t /= view.sizeY;
    const NnUint y = (NnUint)(t % batchSize);
    const NnUint z = (NnUint)(t / batchSize);
    const NnUint row = z * input.logicalSize.y + y;
    const NnSize idx = view.offset + vy * view.strideY + x * view.strideX;
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, z, y);
    float *out = (float *)cudaRowBase(outputBase, output, z, y);
    out[idx] = in[idx] * scale[row];
}

__global__ static void mergeAddF32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    NnUint nSlices,
    NnUint batchSize) {
    const NnSize total = (NnSize)batchSize * output.logicalSize.x;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % output.logicalSize.x);
    const NnUint y = (NnUint)(tid / output.logicalSize.x);
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, 0u, y);
    float *out = (float *)cudaRowBase(outputBase, output, 0u, y);
    float v = out[x];
    for (NnUint s = 0u; s < nSlices; ++s) v += in[s * output.logicalSize.x + x];
    out[x] = v;
}

__global__ static void mergeAddQ80F32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    NnUint nSlices,
    NnUint batchSize) {
    const NnSize total = (NnSize)batchSize * output.logicalSize.x;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % output.logicalSize.x);
    const NnUint y = (NnUint)(tid / output.logicalSize.x);
    const NnUint block = x / Q80_BLOCK_SIZE;
    const NnUint lane = x - block * Q80_BLOCK_SIZE;
    const NnUint blocksPerSlice = output.logicalSize.x / Q80_BLOCK_SIZE;
    const NnBlockQ80 *in = (const NnBlockQ80 *)cudaRowBase((NnByte *)inputBase, input, 0u, y);
    float *out = (float *)cudaRowBase(outputBase, output, 0u, y);
    float v = out[x];
    for (NnUint s = 0u; s < nSlices; ++s) {
        const NnBlockQ80 *b = in + s * blocksPerSlice + block;
        v += (float)b->qs[lane] * cudaHalfBitsToFloat(b->d);
    }
    out[x] = v;
}

__global__ static void mergeSumF32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    NnUint batchSize) {
    const NnSize total = (NnSize)batchSize * output.logicalSize.x;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % output.logicalSize.x);
    const NnUint y = (NnUint)(tid / output.logicalSize.x);
    float s = 0.0f;
    for (NnUint z = 0u; z < input.logicalSize.z; ++z) {
        const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, z, y);
        s += in[x];
    }
    float *out = (float *)cudaRowBase(outputBase, output, 0u, y);
    out[x] = s;
}

__global__ static void q80QuantizeKernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    NnUint offset,
    NnUint len,
    NnUint batchSize) {
    const NnUint blocksPerRow = len / Q80_BLOCK_SIZE;
    const NnSize totalBlocks = (NnSize)input.logicalSize.z * batchSize * blocksPerRow;
    /* One warp per 32-element block: each lane owns exactly one element. The
     * block-per-thread form left the GPU with len/32 threads for the whole
     * tensor (96 for dim=3072), far too few to fill a T4. */
    const NnUint lane = threadIdx.x & 31u;
    const NnSize blockIndex = ((NnSize)blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
    if (blockIndex >= totalBlocks) return;
    const NnUint b = (NnUint)(blockIndex % blocksPerRow);
    NnSize t = blockIndex / blocksPerRow;
    const NnUint y = (NnUint)(t % batchSize);
    const NnUint z = (NnUint)(t / batchSize);
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, z, y) + offset + b * Q80_BLOCK_SIZE;
    NnBlockQ80 *out = (NnBlockQ80 *)cudaRowBase(outputBase, output, z, y) + (offset / Q80_BLOCK_SIZE) + b;

    const float v = in[lane];
    float amax = cudaWarpReduceMax(fabsf(v));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    /* d is uniform across the warp, so one lane publishes it and the rest read
     * it back through the broadcast instead of racing the same half store. */
    if (lane == 0u) out->d = cudaFloatToHalfBits(d);
    int q = (int)nearbyintf(v * id);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    out->qs[lane] = (int8_t)q;
}

__global__ static void repeatZCopyQ80Kernel(
    NnByte *outputBase,
    NnPointerLayout output,
    NnUint batchSize) {
    const NnUint blocksPerRow = output.logicalSize.x / Q80_BLOCK_SIZE;
    const NnUint repeatZ = output.logicalSize.z > 0u ? output.logicalSize.z - 1u : 0u;
    const NnSize totalBlocks = (NnSize)repeatZ * batchSize * blocksPerRow;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalBlocks) return;
    const NnUint b = (NnUint)(tid % blocksPerRow);
    NnSize t = tid / blocksPerRow;
    const NnUint y = (NnUint)(t % batchSize);
    const NnUint z = (NnUint)(t / batchSize) + 1u;
    NnBlockQ80 *out = (NnBlockQ80 *)cudaRowBase(outputBase, output, z, y) + b;
    const NnBlockQ80 *src = (const NnBlockQ80 *)cudaRowBase(outputBase, output, 0u, y) + b;
    *out = *src;
}

__global__ static void embeddingF32F32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    const float *weight,
    NnUint batchSize) {
    const NnSize total = (NnSize)batchSize * output.logicalSize.x;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % output.logicalSize.x);
    const NnUint y = (NnUint)(tid / output.logicalSize.x);
    const float *tokenPtr = (const float *)cudaRowBase((NnByte *)inputBase, input, 0u, y);
    const NnUint token = (NnUint)tokenPtr[0];
    float *out = (float *)cudaRowBase(outputBase, output, 0u, y);
    out[x] = weight[(NnSize)token * output.logicalSize.x + x];
}

__global__ static void embeddingQ40F32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    const NnBlockQ40 *weight,
    NnUint rowBlocks,
    NnUint batchSize) {
    const NnSize total = (NnSize)batchSize * output.logicalSize.x;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % output.logicalSize.x);
    const NnUint y = (NnUint)(tid / output.logicalSize.x);
    const float *tokenPtr = (const float *)cudaRowBase((NnByte *)inputBase, input, 0u, y);
    const NnUint token = (NnUint)tokenPtr[0];
    const NnUint block = x / Q40_BLOCK_SIZE;
    const NnUint lane = x - block * Q40_BLOCK_SIZE;
    const NnBlockQ40 *b = weight + (NnSize)token * rowBlocks + block;
    const NnUint packed = lane < (Q40_BLOCK_SIZE / 2u) ? lane : lane - (Q40_BLOCK_SIZE / 2u);
    const int q = lane < (Q40_BLOCK_SIZE / 2u)
        ? (int)(b->qs[packed] & 0x0fu) - 8
        : (int)(b->qs[packed] >> 4) - 8;
    float *out = (float *)cudaRowBase(outputBase, output, 0u, y);
    out[x] = (float)q * cudaHalfBitsToFloat(b->d);
}

__global__ static void embeddingQ80F32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    const NnBlockQ80 *weight,
    NnUint rowBlocks,
    NnUint batchSize) {
    const NnSize total = (NnSize)batchSize * output.logicalSize.x;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % output.logicalSize.x);
    const NnUint y = (NnUint)(tid / output.logicalSize.x);
    const float *tokenPtr = (const float *)cudaRowBase((NnByte *)inputBase, input, 0u, y);
    const NnUint token = (NnUint)tokenPtr[0];
    const NnUint block = x / Q80_BLOCK_SIZE;
    const NnUint lane = x - block * Q80_BLOCK_SIZE;
    const NnBlockQ80 *b = weight + (NnSize)token * rowBlocks + block;
    float *out = (float *)cudaRowBase(outputBase, output, 0u, y);
    out[x] = (float)b->qs[lane] * cudaHalfBitsToFloat(b->d);
}

__global__ static void invRmsF32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    float epsilon,
    NnUint nColumns,
    NnUint batchSize) {
    /* One block per (y, col). The previous form gave each thread the whole
     * colSize (3072 element) accumulation; now the block splits it and folds it
     * with warp shuffles, leaving only the cross-warp step in shared memory. */
    __shared__ float shared[32];
    const NnUint col = blockIdx.x;
    const NnUint y = blockIdx.y;
    if (col >= nColumns || y >= batchSize) return;
    const NnUint colSize = input.logicalSize.x / nColumns;
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, 0u, y) + (NnSize)col * colSize;
    float ss = 0.0f;
    for (NnUint i = threadIdx.x; i < colSize; i += blockDim.x) {
        ss += in[i] * in[i];
    }
    ss = cudaBlockReduceSum(ss, shared);
    if (threadIdx.x == 0u) {
        float *out = (float *)cudaRowBase(outputBase, output, 0u, y);
        out[col] = rsqrtf(ss / (float)colSize + epsilon);
    }
}

__global__ static void rmsNormF32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    const float *weight,
    const float *invRms,
    NnUint invRmsRowStride,
    NnUint nColumns,
    NnUint batchSize) {
    const NnSize total = (NnSize)batchSize * output.logicalSize.x;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % output.logicalSize.x);
    const NnUint y = (NnUint)(tid / output.logicalSize.x);
    const NnUint colSize = output.logicalSize.x / nColumns;
    const NnUint col = x / colSize;
    const NnUint local = x - col * colSize;
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, 0u, y);
    float *out = (float *)cudaRowBase(outputBase, output, 0u, y);
    out[x] = weight[local] * (invRms[(NnSize)y * invRmsRowStride + col] * in[x]);
}

__global__ static void ropeF32Kernel(
    NnByte *inputBase,
    NnPointerLayout input,
    const float *positions,
    const float *cache,
    NnRopeOpConfig config,
    NnTensorViewLayout view,
    NnUint batchSize) {
    const bool isQ = config.isQ == 1u;
    const NnRopeSlice slice = config.slice;
    if (config.type == ROPE_LLAMA || config.type == ROPE_LLAMA3_1) {
        const NnUint pairCount = view.sizeX / 2u;
        const NnSize total = (NnSize)batchSize * pairCount;
        const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;
        const NnUint pair = (NnUint)(tid % pairCount);
        const NnUint y = (NnUint)(tid / pairCount);
        const NnUint pos = (NnUint)positions[y];
        const NnUint shift = isQ ? slice.qShift : 0u;
        const NnUint i = pair * 2u;
        const float *posCache = cache + (NnSize)pos * slice.sliceDim + shift + view.offset;
        float *x = (float *)cudaRowBase(inputBase, input, 0u, y);
        const NnUint x0Index = view.offset + i;
        const float fcr = posCache[i];
        const float fci = posCache[i + 1u];
        const float v0 = x[x0Index];
        const float v1 = x[x0Index + 1u];
        x[x0Index] = v0 * fcr - v1 * fci;
        x[x0Index + 1u] = v0 * fci + v1 * fcr;
        return;
    }
    if (config.type == ROPE_FALCON) {
        const NnUint headDim = slice.headDim;
        const NnUint nHeadsView = view.sizeX / headDim;
        const NnUint half = headDim / 2u;
        const NnSize total = (NnSize)batchSize * nHeadsView * half;
        const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;
        const NnUint j = (NnUint)(tid % half);
        NnSize t = tid / half;
        const NnUint h = (NnUint)(t % nHeadsView);
        const NnUint y = (NnUint)(t / nHeadsView);
        const NnUint pos = (NnUint)positions[y];
        const float *posCache = cache + (NnSize)pos * headDim;
        float *x = (float *)cudaRowBase(inputBase, input, 0u, y);
        const NnUint base = view.offset + h * headDim;
        const float fcr = posCache[j];
        const float fci = posCache[j + half];
        const float v0 = x[base + j];
        const float v1 = x[base + j + half];
        x[base + j] = v0 * fcr - v1 * fci;
        x[base + j + half] = v0 * fci + v1 * fcr;
    }
}

__global__ static void shiftF32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    const float *indexes,
    const float *slots,
    NnShiftOpCodeConfig config,
    NnUint batchSize) {
    const NnSize total = (NnSize)batchSize * input.logicalSize.x;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % input.logicalSize.x);
    const NnUint y = (NnUint)(tid / input.logicalSize.x);
    const NnUint row = (NnUint)indexes[y];
    const NnUint slotId = (config.kvSlotStride != 0u) ? (NnUint)slots[y] : 0u;
    const NnSize slotBase = (config.kvSlotStride != 0u) ? (NnSize)slotId * (NnSize)config.kvSlotStride : 0u;
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, 0u, y);
    float *outBase = (float *)(outputBase + output.byteOffset);
    const NnSize dst = config.dstRowStride == 0u
        ? slotBase + (NnSize)row * input.logicalSize.x + x
        : slotBase + (NnSize)row * config.dstRowStride + config.dstColStart + x;
    outBase[dst] = in[x];
}

__global__ static void softmaxF32Kernel(
    NnByte *outputBase,
    NnPointerLayout output,
    NnTensorViewLayout view,
    NnUint batchSize) {
    __shared__ float scratch[256];
    const NnUint row = blockIdx.x;
    const NnUint y = row % batchSize;
    const NnUint z = row / batchSize;
    float *out = (float *)cudaRowBase(outputBase, output, z, y);
    float maxv = -3.4028234663852886e38f;
    for (NnUint i = threadIdx.x; i < view.sizeX; i += blockDim.x) {
        const float v = out[view.offset + i * view.strideX];
        maxv = maxv > v ? maxv : v;
    }
    scratch[threadIdx.x] = maxv;
    __syncthreads();
    for (NnUint stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            const float v = scratch[threadIdx.x + stride];
            scratch[threadIdx.x] = scratch[threadIdx.x] > v ? scratch[threadIdx.x] : v;
        }
        __syncthreads();
    }
    maxv = scratch[0];
    float sum = 0.0f;
    for (NnUint i = threadIdx.x; i < view.sizeX; i += blockDim.x) {
        const NnUint idx = view.offset + i * view.strideX;
        const float v = expf(out[idx] - maxv);
        out[idx] = v;
        sum += v;
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (NnUint stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    sum = scratch[0];
    for (NnUint i = threadIdx.x; i < view.sizeX; i += blockDim.x) {
        const NnUint idx = view.offset + i * view.strideX;
        out[idx] = out[idx] / sum;
    }
}

__global__ static void softmaxF32SerialKernel(
    NnByte *outputBase,
    NnPointerLayout output,
    NnTensorViewLayout view,
    NnUint batchSize) {
    const NnUint row = blockIdx.x;
    const NnUint y = row % batchSize;
    const NnUint z = row / batchSize;
    float *out = (float *)cudaRowBase(outputBase, output, z, y);
    float maxv = -3.4028234663852886e38f;
    for (NnUint i = 0u; i < view.sizeX; ++i) {
        const float v = out[view.offset + i * view.strideX];
        maxv = maxv > v ? maxv : v;
    }
    float sum = 0.0f;
    for (NnUint i = 0u; i < view.sizeX; ++i) {
        const NnUint idx = view.offset + i * view.strideX;
        const float v = expf(out[idx] - maxv);
        out[idx] = v;
        sum += v;
    }
    for (NnUint i = 0u; i < view.sizeX; ++i) {
        const NnUint idx = view.offset + i * view.strideX;
        out[idx] = out[idx] / sum;
    }
}

__global__ static void moeGateF32Kernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    float *indexes,
    NnUint k,
    NnUint normTopk,
    NnUint indexRowStride,
    NnUint batchSize) {
    const NnUint y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= batchSize) return;

    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, 0u, y);
    float sum = 0.0f;
    for (NnUint rank = 0u; rank < k; ++rank) {
        float bestVal = -3.4028234663852886e38f;
        NnUint bestIdx = 0u;
        for (NnUint i = 0u; i < input.logicalSize.x; ++i) {
            bool alreadySelected = false;
            for (NnUint prev = 0u; prev < rank; ++prev) {
                if ((NnUint)indexes[(NnSize)y * indexRowStride + prev] == i) {
                    alreadySelected = true;
                    break;
                }
            }
            const float v = in[i];
            if (!alreadySelected && v > bestVal) {
                bestVal = v;
                bestIdx = i;
            }
        }

        indexes[(NnSize)y * indexRowStride + rank] = (float)bestIdx;
        float *out = (float *)cudaRowBase(outputBase, output, rank, y);
        out[0] = bestVal;
        sum += bestVal;
    }

    const float denom = normTopk == 1u ? sum : 1.0f;
    for (NnUint rank = 0u; rank < k; ++rank) {
        float *out = (float *)cudaRowBase(outputBase, output, rank, y);
        out[0] = out[0] / denom;
    }
}

__device__ static inline void cudaMhaOffsets(
    NnMultiHeadAttOpConfig config,
    NnUint h,
    NnUint y,
    NnUint *qOffset,
    NnUint *kvOffset,
    NnUint *attOffset) {
    const NnUint kvMul = config.nHeads / config.nKvHeads;
    const NnUint qHeadStart = config.qStart / config.headDim;
    const NnUint kvHeadStart = config.kvStart / config.headDim;
    const NnUint globalQHead = qHeadStart + h;
    const NnUint globalKvHead = globalQHead / kvMul;
    const NnUint localKvHead = globalKvHead - kvHeadStart;
    *qOffset = y * config.qStride + config.qStart + h * config.headDim;
    *kvOffset = config.kvStart + localKvHead * config.headDim;
    *attOffset = y * config.nHeads0 * config.seqLen + h * config.seqLen;
}

#define NN_CUDA_MHA_SHARED_QUERY_DIM 256u

__global__ static void multiheadAttScoreF32Kernel(
    const float *positions,
    const float *slots,
    const float *query,
    const float *keyCache,
    float *att,
    NnMultiHeadAttOpConfig config,
    NnUint batchSize) {
    /* The query slice is re-read once per key position. It is small (headDim,
     * typically 64 or 128) and identical for every p, so stage it in shared
     * memory when it fits and keep the dot product fed from there. */
    __shared__ float sQuery[NN_CUDA_MHA_SHARED_QUERY_DIM];
    const NnUint h = blockIdx.x;
    const NnUint y = blockIdx.y;
    if (h >= config.nHeads0 || y >= batchSize) return;
    const NnUint pos = (NnUint)positions[y];
    const NnUint slotId = (config.kvSlotStride != 0u) ? (NnUint)slots[y] : 0u;
    const NnSize slotBase = (config.kvSlotStride != 0u) ? (NnSize)slotId * (NnSize)config.kvSlotStride : 0u;
    NnUint qOffset = 0u;
    NnUint kvOffset = 0u;
    NnUint attOffset = 0u;
    cudaMhaOffsets(config, h, y, &qOffset, &kvOffset, &attOffset);
    const float invHeadDimRoot = rsqrtf((float)config.headDim);
    const bool sharedQuery = config.headDim <= NN_CUDA_MHA_SHARED_QUERY_DIM;
    if (sharedQuery) {
        for (NnUint i = threadIdx.x; i < config.headDim; i += blockDim.x) {
            sQuery[i] = query[qOffset + i];
        }
        __syncthreads();
    }
    for (NnUint p = threadIdx.x; p <= pos; p += blockDim.x) {
        float score = 0.0f;
        const NnSize kOffset = slotBase + (NnSize)p * config.kvStride + kvOffset;
        if (sharedQuery) {
            #pragma unroll 4
            for (NnUint i = 0u; i < config.headDim; ++i) {
                score += sQuery[i] * keyCache[kOffset + i];
            }
        } else {
            for (NnUint i = 0u; i < config.headDim; ++i) {
                score += query[qOffset + i] * keyCache[kOffset + i];
            }
        }
        att[attOffset + p] = score * invHeadDimRoot;
    }
}

__global__ static void multiheadAttSoftmaxSerialF32Kernel(
    const float *positions,
    float *att,
    NnMultiHeadAttOpConfig config,
    NnUint batchSize) {
    /* Parallel softmax over the same [0, pos] range the serial version walked:
     * both passes reduce with warp shuffles plus one shared stage, then every
     * thread rescales its own slots. attOffset and the numeric treatment of the
     * scores are unchanged, so results match the serial form up to the
     * reduction order. */
    __shared__ float shared[32];
    const NnUint h = blockIdx.x;
    const NnUint y = blockIdx.y;
    if (h >= config.nHeads0 || y >= batchSize) return;
    const NnUint pos = (NnUint)positions[y];
    const NnUint attOffset = y * config.nHeads0 * config.seqLen + h * config.seqLen;
    float maxScore = -3.4028234663852886e38f;
    for (NnUint p = threadIdx.x; p <= pos; p += blockDim.x) {
        const float v = att[attOffset + p];
        maxScore = maxScore > v ? maxScore : v;
    }
    maxScore = cudaBlockReduceMax(maxScore, shared);
    float sum = 0.0f;
    for (NnUint p = threadIdx.x; p <= pos; p += blockDim.x) {
        const float v = expf(att[attOffset + p] - maxScore);
        att[attOffset + p] = v;
        sum += v;
    }
    const float invSum = 1.0f / cudaBlockReduceSum(sum, shared);
    for (NnUint p = threadIdx.x; p <= pos; p += blockDim.x) {
        att[attOffset + p] *= invSum;
    }
}

__global__ static void multiheadAttValueF32Kernel(
    NnByte *outputBase,
    NnPointerLayout output,
    const float *positions,
    const float *slots,
    const float *valueCache,
    const float *att,
    NnMultiHeadAttOpConfig config,
    NnUint batchSize) {
    const NnUint h = blockIdx.x;
    const NnUint y = blockIdx.y;
    if (h >= config.nHeads0 || y >= batchSize) return;
    const NnUint pos = (NnUint)positions[y];
    const NnUint slotId = (config.kvSlotStride != 0u) ? (NnUint)slots[y] : 0u;
    const NnSize slotBase = (config.kvSlotStride != 0u) ? (NnSize)slotId * (NnSize)config.kvSlotStride : 0u;
    NnUint qOffset = 0u;
    NnUint kvOffset = 0u;
    NnUint attOffset = 0u;
    cudaMhaOffsets(config, h, y, &qOffset, &kvOffset, &attOffset);
    (void)qOffset;
    float *out = (float *)cudaRowBase(outputBase, output, 0u, y) + h * config.headDim;
    for (NnUint i = threadIdx.x; i < config.headDim; i += blockDim.x) {
        float acc = 0.0f;
        const NnUint vOffset = kvOffset + i;
        for (NnUint p = 0u; p <= pos; ++p) {
            acc += att[attOffset + p] * valueCache[slotBase + (NnSize)p * config.kvStride + vOffset];
        }
        out[i] = acc;
    }
}

__device__ static inline float q80q40BlockDot(const NnBlockQ80 *xb, const NnBlockQ40 *wb) {
    return (float)cudaQ80Q40DotGlobal(xb, wb) * cudaHalfBitsToFloat(xb->d) * cudaHalfBitsToFloat(wb->d);
}

__global__ static void matmulQ80Q40SmallKKernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    const NnBlockQ40 *weight,
    const float *activeExpertIndexes,
    NnUint activeExpertRowStride,
    NnUint nActiveExperts,
    NnUint nActiveExpertsOr1,
    NnUint weightExperts,
    NnUint weightBlocksPerRow,
    NnUint weightRows,
    NnUint view,
    NnUint aOffsetBlocks,
    NnUint cOffset,
    NnUint aBlocks,
    NnUint cLen,
    NnUint inStartBlocks,
    NnUint outStart,
    NnUint batchSize) {
    const NnSize total = (NnSize)batchSize * nActiveExpertsOr1 * cLen;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint outLocal = (NnUint)(tid % cLen);
    NnSize t = tid / cLen;
    const NnUint e = (NnUint)(t % nActiveExpertsOr1);
    const NnUint y = (NnUint)(t / nActiveExpertsOr1);
    const NnUint activeExpert = nActiveExperts == 0u ? 0u : (NnUint)activeExpertIndexes[(NnSize)y * activeExpertRowStride + e];
    if (activeExpert >= weightExperts) return;

    const NnBlockQ80 *x = (const NnBlockQ80 *)cudaRowBase((NnByte *)inputBase, input, e, y) + aOffsetBlocks;
    float *out = (float *)cudaRowBase(outputBase, output, e, y) + cOffset;
    const NnUint row = view == 0u ? outLocal : outStart + outLocal;
    const NnUint inBlock = view == 2u ? inStartBlocks : 0u;
    const NnBlockQ40 *w = weight + ((NnSize)activeExpert * weightRows + row) * weightBlocksPerRow + inBlock;
    float sum = 0.0f;
    for (NnUint b = 0u; b < aBlocks; ++b) {
        sum += q80q40BlockDot(x + b, w + b);
    }
    out[outLocal] = sum;
}

__global__ static void matmulQ80Q40LargeKKernel(
    const NnByte *inputBase,
    NnPointerLayout input,
    NnByte *outputBase,
    NnPointerLayout output,
    const NnBlockQ40 *weight,
    const float *activeExpertIndexes,
    NnUint activeExpertRowStride,
    NnUint nActiveExperts,
    NnUint nActiveExpertsOr1,
    NnUint weightExperts,
    NnUint weightBlocksPerRow,
    NnUint weightRows,
    NnUint view,
    NnUint aOffsetBlocks,
    NnUint cOffset,
    NnUint aBlocks,
    NnUint cLen,
    NnUint inStartBlocks,
    NnUint outStart,
    NnUint batchSize) {
    /* One warp per output row: every warp of the block works on rows from the
     * same (y, e) group, so a single shared copy of x feeds all of them and
     * global x traffic drops by the warp count (8 on a 256-thread block).
     * Grid is (row groups, batchSize * nActiveExpertsOr1). */
    extern __shared__ int smem[];
    const NnUint lane = threadIdx.x & 31u;
    const NnUint wid = threadIdx.x >> 5u;
    const NnUint nWarps = blockDim.x >> 5u;

    const NnUint y = (NnUint)(blockIdx.y % batchSize);
    const NnUint e = (NnUint)(blockIdx.y / batchSize);
    const NnUint activeExpert = nActiveExperts == 0u ? 0u : (NnUint)activeExpertIndexes[(NnSize)y * activeExpertRowStride + e];
    /* Uniform across the block, so this return cannot strand a barrier. */
    if (activeExpert >= weightExperts) return;

    const NnUint outLocal = blockIdx.x * nWarps + wid;
    const NnBlockQ80 *x = (const NnBlockQ80 *)cudaRowBase((NnByte *)inputBase, input, e, y) + aOffsetBlocks;

    /* cachedBlocks is a compile-time shared-memory budget; beyond it we read x
     * straight from global memory and stay on the warp dp4a path. */
    const bool cached = aBlocks <= NN_CUDA_MATMUL_SHARED_MAX_BLOCKS;
    int *xs = smem;
    float *xd = (float *)(xs + (cached ? aBlocks * 8u : 0u));
    int *xSum = (int *)(xd + (cached ? aBlocks : 0u));
    if (cached) {
        /* One thread per 32-element block: pack the block into eight int words
         * and fold its int8 sum once, so the dp4a inner loop needs no per-lane
         * pre-pass and no subtraction. */
        for (NnUint b = threadIdx.x; b < aBlocks; b += blockDim.x) {
            int s = 0;
            #pragma unroll
            for (NnUint g = 0u; g < 8u; ++g) {
                const int v = cudaLoadInt8x4(x[b].qs + 4u * g);
                xs[b * 8u + g] = v;
                s += cudaSumInt8x4(v);
            }
            xSum[b] = s;
            xd[b] = cudaHalfBitsToFloat(x[b].d);
        }
        __syncthreads();
    }
    /* Rows past cLen leave after the barrier; nothing below syncs. */
    if (outLocal >= cLen) return;

    const NnUint row = view == 0u ? outLocal : outStart + outLocal;
    const NnUint inBlock = view == 2u ? inStartBlocks : 0u;
    const NnBlockQ40 *w = weight + ((NnSize)activeExpert * weightRows + row) * weightBlocksPerRow + inBlock;

    float sum = 0.0f;
    if (cached) {
        for (NnUint b = lane; b < aBlocks; b += 32u) {
            const int ip = cudaQ80Q40DotPacked(xs + b * 8u, xSum[b], w[b].qs);
            sum += (float)ip * xd[b] * cudaHalfBitsToFloat(w[b].d);
        }
    } else {
        for (NnUint b = lane; b < aBlocks; b += 32u) {
            const int ip = cudaQ80Q40DotGlobal(x + b, w + b);
            sum += (float)ip * cudaHalfBitsToFloat(x[b].d) * cudaHalfBitsToFloat(w[b].d);
        }
    }
    sum = cudaWarpReduceSum(sum);
    if (lane == 0u) {
        float *out = (float *)cudaRowBase(outputBase, output, e, y) + cOffset;
        out[outLocal] = sum;
    }
}

/* Shared bytes for a cached-x Large-K block: packed int words, the per-block
 * f16 scales, and the per-block int8 sums. */
static NnSize cudaQ80Q40LargeKSharedBytes(NnUint aBlocks) {
    return (NnSize)aBlocks * (NnSize)(8u * sizeof(int) + sizeof(float) + sizeof(int));
}

int nnCudaDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaErrorString(err, "cudaGetDeviceCount"));
    }
    return count;
}

std::string nnCudaDeviceInfo(NnUint gpuIndex) {
    const int count = nnCudaDeviceCount();
    if ((int)gpuIndex < 0 || (int)gpuIndex >= count) {
        std::ostringstream out;
        out << "Invalid CUDA GPU index " << gpuIndex << "; available CUDA devices: " << count;
        throw std::runtime_error(out.str());
    }

    cudaDeviceProp prop;
    NN_CUDA_CHECK(cudaGetDeviceProperties(&prop, (int)gpuIndex));

    int driverVersion = 0;
    int runtimeVersion = 0;
    NN_CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    NN_CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));

    std::ostringstream out;
    out << "CUDA Device[" << gpuIndex << "]: " << prop.name
        << " cc " << prop.major << "." << prop.minor
        << " sm_" << (prop.major * 10 + prop.minor)
        << ", globalMem=" << (unsigned long long)(prop.totalGlobalMem / (1024ull * 1024ull)) << " MB"
        << ", multiProcessorCount=" << prop.multiProcessorCount
        << ", warpSize=" << prop.warpSize
        << ", maxThreadsPerBlock=" << prop.maxThreadsPerBlock
        << ", integrated=" << prop.integrated
        << ", driver=" << driverVersion
        << ", runtime=" << runtimeVersion;
    return out.str();
}

void nnCudaPrintDeviceInfo(NnUint gpuIndex) {
    std::printf("🔷 %s\n", nnCudaDeviceInfo(gpuIndex).c_str());
    std::fflush(stdout);
}

double nnCudaProfileGemmMs(NnUint gpuIndex, int n) {
    if (n < 16) n = 16;
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
    const size_t bytes = (size_t)n * (size_t)n * sizeof(float);
    float *a = nullptr;
    float *b = nullptr;
    float *c = nullptr;
    cublasHandle_t handle = nullptr;
    try {
        NN_CUDA_CHECK(cudaMalloc(&a, bytes));
        NN_CUDA_CHECK(cudaMalloc(&b, bytes));
        NN_CUDA_CHECK(cudaMalloc(&c, bytes));
        NN_CUDA_CHECK(cudaMemset(a, 0, bytes));
        NN_CUDA_CHECK(cudaMemset(b, 0, bytes));
        NN_CUBLAS_CHECK(cublasCreate(&handle));
        const float alpha = 1.0f;
        const float beta = 0.0f;
        NN_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, b, n, a, n, &beta, c, n));
        NN_CUDA_CHECK(cudaDeviceSynchronize());
        const auto start = std::chrono::steady_clock::now();
        const int iters = 3;
        for (int i = 0; i < iters; ++i) {
            NN_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, b, n, a, n, &beta, c, n));
        }
        NN_CUDA_CHECK(cudaDeviceSynchronize());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        cublasDestroy(handle);
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        return ms / (double)iters;
    } catch (...) {
        if (handle != nullptr) cublasDestroy(handle);
        if (a != nullptr) cudaFree(a);
        if (b != nullptr) cudaFree(b);
        if (c != nullptr) cudaFree(c);
        throw;
    }
}

static double cudaGemvMs(cublasHandle_t handle, float *weight, float *input, float *output, int rows, int cols) {
    if (rows < 1 || cols < 1) return 0.0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    NN_CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, weight, rows, input, 1, &beta, output, 1));
    NN_CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    const int iters = 3;
    for (int i = 0; i < iters; ++i) {
        NN_CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, weight, rows, input, 1, &beta, output, 1));
    }
    NN_CUDA_CHECK(cudaDeviceSynchronize());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return ms / (double)iters;
}

struct NnCudaLayerSoak::Impl {
    bool ok;
    int dim;
    int hidden;
    int kv;
    cublasHandle_t handle;
    float *weight;
    float *input;
    float *output;
    Impl() : ok(false), dim(0), hidden(0), kv(0), handle(nullptr), weight(nullptr), input(nullptr), output(nullptr) {}
};

NnCudaLayerSoak::NnCudaLayerSoak(unsigned gpuIndex, unsigned dim, unsigned hiddenDim, unsigned nHeads, unsigned nKvHeads)
    : impl(new Impl()) {
    if (dim == 0u) return;
    const unsigned hidden = hiddenDim == 0u ? dim : hiddenDim;
    const unsigned heads = nHeads == 0u ? 1u : nHeads;
    const unsigned kvHeads = nKvHeads == 0u ? heads : nKvHeads;
    const unsigned kvDim = (unsigned)(((unsigned long long)dim / heads) * kvHeads);
    const unsigned kv = kvDim == 0u ? dim : kvDim;
    const unsigned long long shapes[4] = {
        (unsigned long long)dim * dim,
        (unsigned long long)kv * dim,
        (unsigned long long)hidden * dim,
        (unsigned long long)dim * hidden
    };
    unsigned long long maxElems = shapes[0];
    for (int i = 1; i < 4; ++i) if (shapes[i] > maxElems) maxElems = shapes[i];
    const unsigned maxVector = std::max(dim, std::max(hidden, kv));
    try {
        NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
        NN_CUDA_CHECK(cudaMalloc(&impl->weight, (size_t)maxElems * sizeof(float)));
        NN_CUDA_CHECK(cudaMalloc(&impl->input, (size_t)maxVector * sizeof(float)));
        NN_CUDA_CHECK(cudaMalloc(&impl->output, (size_t)maxVector * sizeof(float)));
        NN_CUDA_CHECK(cudaMemset(impl->weight, 0, (size_t)maxElems * sizeof(float)));
        NN_CUBLAS_CHECK(cublasCreate(&impl->handle));
        impl->dim = (int)dim;
        impl->hidden = (int)hidden;
        impl->kv = (int)kv;
        impl->ok = true;
    } catch (...) {
        impl->ok = false;
    }
}

NnCudaLayerSoak::~NnCudaLayerSoak() {
    if (impl != nullptr) {
        if (impl->handle != nullptr) cublasDestroy(impl->handle);
        if (impl->weight != nullptr) cudaFree(impl->weight);
        if (impl->input != nullptr) cudaFree(impl->input);
        if (impl->output != nullptr) cudaFree(impl->output);
        delete impl;
    }
}

double NnCudaLayerSoak::runOnceMs() {
    if (impl == nullptr || !impl->ok) return -1.0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    try {
        const auto start = std::chrono::steady_clock::now();
        const int rows[7] = {impl->dim, impl->dim, impl->kv, impl->kv, impl->hidden, impl->hidden, impl->dim};
        const int cols[7] = {impl->dim, impl->dim, impl->dim, impl->dim, impl->dim, impl->dim, impl->hidden};
        for (int i = 0; i < 7; ++i) {
            NN_CUBLAS_CHECK(cublasSgemv(
                impl->handle, CUBLAS_OP_N, rows[i], cols[i], &alpha,
                impl->weight, rows[i], impl->input, 1, &beta, impl->output, 1));
        }
        NN_CUDA_CHECK(cudaDeviceSynchronize());
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    } catch (...) {
        return -1.0;
    }
}

double nnCudaProfileLayerMs(NnUint gpuIndex, unsigned dim, unsigned hiddenDim, unsigned nHeads, unsigned nKvHeads) {
    if (dim == 0u) return 0.0;
    const unsigned hidden = hiddenDim == 0u ? dim : hiddenDim;
    const unsigned heads = nHeads == 0u ? 1u : nHeads;
    const unsigned kvHeads = nKvHeads == 0u ? heads : nKvHeads;
    const unsigned kvDim = (unsigned)(((unsigned long long)dim / heads) * kvHeads);
    const unsigned kv = kvDim == 0u ? dim : kvDim;
    const unsigned long long shapes[4] = {
        (unsigned long long)dim * dim,
        (unsigned long long)kv * dim,
        (unsigned long long)hidden * dim,
        (unsigned long long)dim * hidden
    };
    unsigned long long maxElems = shapes[0];
    for (int i = 1; i < 4; ++i) if (shapes[i] > maxElems) maxElems = shapes[i];
    const unsigned maxVector = std::max(dim, std::max(hidden, kv));
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
    float *weight = nullptr;
    float *input = nullptr;
    float *output = nullptr;
    cublasHandle_t handle = nullptr;
    try {
        NN_CUDA_CHECK(cudaMalloc(&weight, (size_t)maxElems * sizeof(float)));
        NN_CUDA_CHECK(cudaMalloc(&input, (size_t)maxVector * sizeof(float)));
        NN_CUDA_CHECK(cudaMalloc(&output, (size_t)maxVector * sizeof(float)));
        NN_CUDA_CHECK(cudaMemset(weight, 0, (size_t)maxElems * sizeof(float)));
        NN_CUBLAS_CHECK(cublasCreate(&handle));
        const double q = cudaGemvMs(handle, weight, input, output, (int)dim, (int)dim);
        const double k = cudaGemvMs(handle, weight, input, output, (int)kv, (int)dim);
        const double up = cudaGemvMs(handle, weight, input, output, (int)hidden, (int)dim);
        const double down = cudaGemvMs(handle, weight, input, output, (int)dim, (int)hidden);
        cublasDestroy(handle);
        cudaFree(weight);
        cudaFree(input);
        cudaFree(output);
        return 2.0 * q + 2.0 * k + 2.0 * up + down;
    } catch (...) {
        if (handle != nullptr) cublasDestroy(handle);
        if (weight != nullptr) cudaFree(weight);
        if (input != nullptr) cudaFree(input);
        if (output != nullptr) cudaFree(output);
        throw;
    }
}

unsigned long long nnCudaFreeBytes(NnUint gpuIndex) {
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    NN_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    return (unsigned long long)freeBytes;
}

NnCudaPinnedStaging::NnCudaPinnedStaging()
    : hostPointer(nullptr), allocatedSize(0u) {}

NnCudaPinnedStaging::~NnCudaPinnedStaging() {
    release();
}

void *NnCudaPinnedStaging::ensure(NnSize size) {
    if (size == 0u) return nullptr;
    if (hostPointer != nullptr && allocatedSize >= size) return hostPointer;
    release();
    NN_CUDA_CHECK(cudaMallocHost(&hostPointer, size));
    allocatedSize = size;
    return hostPointer;
}

void NnCudaPinnedStaging::release() {
    if (hostPointer != nullptr) {
        cudaFreeHost(hostPointer);
        hostPointer = nullptr;
        allocatedSize = 0u;
    }
}

NnCudaBuffer::NnCudaBuffer(const char *name, NnSize bufferSize)
    : devicePointer(nullptr), name(name != nullptr ? name : "<unnamed>"), bufferSize(bufferSize) {
    if (bufferSize > 0u) {
        NN_CUDA_CHECK(cudaMalloc(&devicePointer, bufferSize));
        NN_CUDA_CHECK(cudaMemset(devicePointer, 0, bufferSize));
    }
}

NnCudaBuffer::~NnCudaBuffer() {
    if (devicePointer != nullptr) {
        cudaFree(devicePointer);
        devicePointer = nullptr;
    }
}

void NnCudaBuffer::write(const NnByte *data, NnSize offset, NnSize nBytes, void *stream, NnCudaPinnedStaging *staging) {
    if (nBytes == 0u) return;
    if (data == nullptr) throw std::runtime_error("CUDA write received null host pointer for buffer " + name);
    validateRange("write", name, bufferSize, offset, nBytes);
    const bool ioProfile = dllamaIoProbeEnabled();
    if (staging == nullptr) {
        const std::uint64_t copyStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
        NN_CUDA_CHECK(cudaMemcpyAsync((NnByte *)devicePointer + offset, data, nBytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
        if (ioProfile) dllamaIoProbeRecordCudaMemcpyAsync(DLLAMA_IO_PROBE_H2D, dllamaIoProbeNowUs() - copyStartUs, (std::uint64_t)nBytes);
        return;
    }
    void *host = staging->ensure(nBytes);
    const std::uint64_t memcpyStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
    std::memcpy(host, data, nBytes);
    if (ioProfile) dllamaIoProbeRecordHostMemcpy(DLLAMA_IO_PROBE_H2D, dllamaIoProbeNowUs() - memcpyStartUs, (std::uint64_t)nBytes);
    const std::uint64_t copyStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
    NN_CUDA_CHECK(cudaMemcpyAsync((NnByte *)devicePointer + offset, host, nBytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
    if (ioProfile) dllamaIoProbeRecordCudaMemcpyAsync(DLLAMA_IO_PROBE_H2D, dllamaIoProbeNowUs() - copyStartUs, (std::uint64_t)nBytes);
    const std::uint64_t syncStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
    NN_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
    if (ioProfile) dllamaIoProbeRecordCudaStreamSync(DLLAMA_IO_PROBE_H2D, dllamaIoProbeNowUs() - syncStartUs);
}

void NnCudaBuffer::read(NnByte *data, NnSize offset, NnSize nBytes, void *stream, NnCudaPinnedStaging *staging) {
    if (nBytes == 0u) return;
    if (data == nullptr) throw std::runtime_error("CUDA read received null host pointer for buffer " + name);
    validateRange("read", name, bufferSize, offset, nBytes);
    const bool ioProfile = dllamaIoProbeEnabled();
    if (staging == nullptr) {
        const std::uint64_t copyStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
        NN_CUDA_CHECK(cudaMemcpyAsync(data, (NnByte *)devicePointer + offset, nBytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
        if (ioProfile) dllamaIoProbeRecordCudaMemcpyAsync(DLLAMA_IO_PROBE_D2H, dllamaIoProbeNowUs() - copyStartUs, (std::uint64_t)nBytes);
        const std::uint64_t syncStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
        NN_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        if (ioProfile) dllamaIoProbeRecordCudaStreamSync(DLLAMA_IO_PROBE_D2H, dllamaIoProbeNowUs() - syncStartUs);
        return;
    }
    void *host = staging->ensure(nBytes);
    const std::uint64_t copyStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
    NN_CUDA_CHECK(cudaMemcpyAsync(host, (NnByte *)devicePointer + offset, nBytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
    if (ioProfile) dllamaIoProbeRecordCudaMemcpyAsync(DLLAMA_IO_PROBE_D2H, dllamaIoProbeNowUs() - copyStartUs, (std::uint64_t)nBytes);
    const std::uint64_t syncStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
    NN_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
    if (ioProfile) dllamaIoProbeRecordCudaStreamSync(DLLAMA_IO_PROBE_D2H, dllamaIoProbeNowUs() - syncStartUs);
    const std::uint64_t memcpyStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
    std::memcpy(data, host, nBytes);
    if (ioProfile) dllamaIoProbeRecordHostMemcpy(DLLAMA_IO_PROBE_D2H, dllamaIoProbeNowUs() - memcpyStartUs, (std::uint64_t)nBytes);
}

void NnCudaBuffer::clear(void *stream) {
    if (bufferSize == 0u) return;
    NN_CUDA_CHECK(cudaMemsetAsync(devicePointer, 0, bufferSize, (cudaStream_t)stream));
}

NnSize NnCudaBuffer::calcSliceSize(NnSize nominator, NnSize denominator) const {
    if (denominator == 0u) throw std::runtime_error("CUDA buffer slice denominator is zero for " + name);
    if (nominator > denominator) throw std::runtime_error("CUDA buffer slice nominator exceeds denominator for " + name);
    if (bufferSize % denominator != 0u) {
        std::ostringstream out;
        out << "CUDA buffer cannot be sliced evenly"
            << " [buffer=" << name
            << ", bytes=" << (unsigned long long)bufferSize
            << ", denominator=" << (unsigned long long)denominator
            << "]";
        throw std::runtime_error(out.str());
    }
    return (bufferSize / denominator) * nominator;
}

NnCudaDeviceData::NnCudaDeviceData()
    : netConfig(nullptr), nodeConfig(nullptr) {}

NnCudaDeviceData::NnCudaDeviceData(NnNetConfig *netConfig, NnNodeConfig *nodeConfig)
    : netConfig(netConfig), nodeConfig(nodeConfig) {
    if (netConfig != nullptr) {
        pipes.reserve(netConfig->nPipes);
        for (NnUint i = 0u; i < netConfig->nPipes; ++i) {
            const NnPipeConfig &pipe = netConfig->pipes[i];
            pipes.emplace_back(new NnCudaBuffer(pipe.name, pipe.size.nBytes));
        }
    }
    if (nodeConfig != nullptr) {
        buffers.reserve(nodeConfig->nBuffers);
        for (NnUint i = 0u; i < nodeConfig->nBuffers; ++i) {
            const NnBufferConfig &buffer = nodeConfig->buffers[i];
            buffers.emplace_back(new NnCudaBuffer(buffer.name, buffer.size.nBytes));
        }
    }
}

NnCudaDeviceData::NnCudaDeviceData(NnCudaDeviceData&& other) noexcept
    : netConfig(other.netConfig), nodeConfig(other.nodeConfig),
      pipes(std::move(other.pipes)), buffers(std::move(other.buffers)) {
    other.netConfig = nullptr;
    other.nodeConfig = nullptr;
}

NnCudaDeviceData& NnCudaDeviceData::operator=(NnCudaDeviceData&& other) noexcept {
    if (this != &other) {
        netConfig = other.netConfig;
        nodeConfig = other.nodeConfig;
        pipes = std::move(other.pipes);
        buffers = std::move(other.buffers);
        other.netConfig = nullptr;
        other.nodeConfig = nullptr;
    }
    return *this;
}

NnCudaBuffer *NnCudaDeviceData::resolvePipe(NnUint pipeIndex) {
    if (pipeIndex >= pipes.size()) throw std::runtime_error("CUDA pipe index out of range: " + std::to_string(pipeIndex));
    return pipes[pipeIndex].get();
}

NnCudaBuffer *NnCudaDeviceData::resolveBuffer(NnUint bufferIndex) {
    if (bufferIndex >= buffers.size()) throw std::runtime_error("CUDA buffer index out of range: " + std::to_string(bufferIndex));
    return buffers[bufferIndex].get();
}

NnCudaDevice::NnCudaDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan)
    : gpuIndex(gpuIndex), stream(nullptr), blasHandle(nullptr), netConfig(netConfig), nodeConfig(nodeConfig), netExecution(netExecution), partitionPlan(partitionPlan), launchConfig{}, staging(), data() {
    (void)this->netConfig;
    (void)this->netExecution;
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
    cudaDeviceProp prop{};
    NN_CUDA_CHECK(cudaGetDeviceProperties(&prop, (int)gpuIndex));
    launchConfig = buildCudaLaunchConfig(prop);
    if (productLogLevel() >= 1) {
        nnCudaPrintDeviceInfo(gpuIndex);
        std::printf("🔷 %s\n", launchConfigInfo().c_str());
        std::fflush(stdout);
    }
    cudaStream_t s = nullptr;
    NN_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    stream = (void *)s;
    cublasHandle_t h = nullptr;
    NN_CUBLAS_CHECK(cublasCreate(&h));
    NN_CUBLAS_CHECK(cublasSetStream(h, s));
    blasHandle = (void *)h;
    if (nodeConfig != nullptr) {
        nodeConfig->partitionPlan = partitionPlan;
    }
    data = NnCudaDeviceData(netConfig, nodeConfig);
}

NnCudaDevice::~NnCudaDevice() {
    if (stream != nullptr) {
        cudaSetDevice((int)gpuIndex);
        cudaStreamSynchronize((cudaStream_t)stream);
    }
    if (blasHandle != nullptr) {
        cudaSetDevice((int)gpuIndex);
        cublasDestroy((cublasHandle_t)blasHandle);
        blasHandle = nullptr;
    }
    if (stream != nullptr) {
        cudaSetDevice((int)gpuIndex);
        cudaStreamDestroy((cudaStream_t)stream);
        stream = nullptr;
    }
}

NnUint NnCudaDevice::maxNThreads() {
    // Same model as the Vulkan backend: compute stays on thread 0, extra
    // threads parallelize per-peer socket exchanges in sync steps.
    const NnUint nThreads = std::thread::hardware_concurrency();
    return nThreads == 0u ? 1u : nThreads;
}

std::string NnCudaDevice::launchConfigInfo() const {
    return cudaLaunchConfigToString(launchConfig);
}

NnDeviceSegment *NnCudaDevice::createSegment(NnUint segmentIndex) {
    if (nodeConfig == nullptr || segmentIndex >= nodeConfig->nSegments) {
        throw std::runtime_error("CUDA backend cannot create segment " + std::to_string(segmentIndex) + ": segment index out of range");
    }
    return new NnCudaDeviceSegment(this, segmentIndex, &nodeConfig->segments[segmentIndex], netExecution);
}

void NnCudaDevice::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    partitionPlan = plan;
    if (nodeConfig != nullptr) {
        nodeConfig->partitionPlan = plan;
    }
    planEpoch.fetch_add(1u, std::memory_order_acq_rel);
}

NnSize3D NnCudaDevice::resolvePointerLogicalSize(const NnPointerConfig *config) const {
    if (config == nullptr) return size0();
    return resolvePointerLayout(netConfig, nodeConfig, partitionPlan, config).logicalSize;
}

void NnCudaDevice::writePipe(NnUint pipeIndex, const NnByte *hostData, NnSize offset, NnSize nBytes) {
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
    data.resolvePipe(pipeIndex)->write(hostData, offset, nBytes, stream, &staging);
}

void NnCudaDevice::readPipe(NnUint pipeIndex, NnByte *hostData, NnSize offset, NnSize nBytes) {
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
    data.resolvePipe(pipeIndex)->read(hostData, offset, nBytes, stream, &staging);
}

void NnCudaDevice::writeBuffer(NnUint bufferIndex, const NnByte *hostData, NnSize offset, NnSize nBytes) {
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
    data.resolveBuffer(bufferIndex)->write(hostData, offset, nBytes, stream, &staging);
}

void NnCudaDevice::readBuffer(NnUint bufferIndex, NnByte *hostData, NnSize offset, NnSize nBytes) {
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
    data.resolveBuffer(bufferIndex)->read(hostData, offset, nBytes, stream, &staging);
}

void NnCudaDevice::synchronize() {
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
    NN_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
}

static void validateCudaQ80Q40MatmulAtCreate(NnCudaDevice *device, const NnOpConfig &op) {
    if (device == nullptr || op.code != OP_MATMUL || op.config == nullptr) return;
    if (op.weightSize.floatType != F_Q40) return;
    NnNetConfig *netConfig = device->getNetConfig();
    NnNodeConfig *nodeConfig = device->getNodeConfig();
    if (netConfig == nullptr || nodeConfig == nullptr) return;
    NnPointerLayout in = resolvePointerLayout(netConfig, nodeConfig, device->getPartitionPlan(), &op.input);
    NnPointerLayout out = resolvePointerLayout(netConfig, nodeConfig, device->getPartitionPlan(), &op.output);
    if (in.logicalSize.floatType != F_Q80 || out.logicalSize.floatType != F_32) return;
    const NnMatmulOpConfig *config = (const NnMatmulOpConfig *)op.config;
    if ((config->aView.strideX != 0u && config->aView.strideX != 1u) ||
        (config->cView.strideX != 0u && config->cView.strideX != 1u) ||
        config->aView.sizeY != 0u ||
        config->cView.sizeY != 0u) {
        throw std::runtime_error("CUDA Q80xQ40 MATMUL supports only contiguous per-row A/C tensor views");
    }
    const NnTensorViewLayout aView = resolveTensorView(&config->aView, 0u, in.logicalSize.x, in.logicalSize.x);
    const NnTensorViewLayout cView = resolveTensorView(&config->cView, 0u, out.logicalSize.x, out.logicalSize.x);
    if ((op.weightSize.y % Q40_BLOCK_SIZE) != 0u ||
        (aView.offset % Q80_BLOCK_SIZE) != 0u ||
        (aView.sizeX % Q80_BLOCK_SIZE) != 0u ||
        (config->inStart % Q40_BLOCK_SIZE) != 0u) {
        throw std::runtime_error("CUDA Q80xQ40 MATMUL requires K, input offset, inStart and A slice to be 32-element block aligned");
    }
    if (config->view == 0u) {
        if (aView.sizeX != op.weightSize.y || cView.sizeX != op.weightSize.x) {
            throw std::runtime_error("CUDA Q80xQ40 MATMUL view=0 shape mismatch");
        }
    } else if (config->view == 1u) {
        if (aView.sizeX != op.weightSize.y || config->outStart + cView.sizeX > op.weightSize.x) {
            throw std::runtime_error("CUDA Q80xQ40 MATMUL view=1 shape mismatch");
        }
    } else if (config->view == 2u) {
        if (config->inStart + aView.sizeX > op.weightSize.y ||
            config->outStart + cView.sizeX > op.weightSize.x) {
            throw std::runtime_error("CUDA Q80xQ40 MATMUL view=2 shape mismatch");
        }
    } else {
        throw std::runtime_error("CUDA Q80xQ40 MATMUL unsupported view mode");
    }
}

NnCudaDeviceSegment::NnCudaDeviceSegment(NnCudaDevice *device, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution)
    : device(device), segmentIndex(segmentIndex), segmentConfig(segmentConfig), netExecution(netExecution),
      bindingsEpoch(0u), bindingsValid(false), isHostOnlyPlanSegment(false), hasControlOps(false) {
    (void)this->netExecution;
    if (device != nullptr) {
        planEpochReady.store(device->getPlanEpoch(), std::memory_order_release);
    }
    if (segmentConfig != nullptr) {
        weightBuffers.reserve(segmentConfig->nOps);
        configBuffers.reserve(segmentConfig->nOps);
        for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
            const NnOpConfig &op = segmentConfig->ops[i];
            validateCudaQ80Q40MatmulAtCreate(device, op);
            std::string weightName = std::string(op.name != nullptr ? op.name : "op") + ".weight";
            std::string configName = std::string(op.name != nullptr ? op.name : "op") + ".config";
            weightBuffers.emplace_back(new NnCudaBuffer(weightName.c_str(), op.weightSize.nBytes));
            configBuffers.emplace_back(new NnCudaBuffer(configName.c_str(), op.configSize));
            if (op.config != nullptr && op.configSize > 0u && device != nullptr) {
                configBuffers.back()->write(op.config, 0u, op.configSize, device->getStream(), &device->staging);
            }
            if (op.code == OP_ROPE && op.config != nullptr && device != nullptr) {
                const NnRopeOpConfig *ropeConfig = (const NnRopeOpConfig *)op.config;
                if (ropeConfig->ropeCacheBufferIndex >= device->data.buffers.size()) {
                    throw std::runtime_error("CUDA ROPE cache buffer index out of range");
                }
                std::vector<float> ropeCache(ropeConfig->slice.cacheSize.length, 0.0f);
                fullfillRopeCache(ropeConfig, ropeCache.data());
                device->data.resolveBuffer(ropeConfig->ropeCacheBufferIndex)->write(
                    (const NnByte *)ropeCache.data(),
                    0u,
                    ropeConfig->slice.cacheSize.nBytes,
                    device->getStream(),
                    &device->staging);
            }
        }
        if (device != nullptr) {
            device->synchronize();
        }
    }
    buildPipeIndexSets();
}

// Collect the pipes this segment really has to move across the PCIe link.
// OP_PLAN_BARRIER / OP_PLAN_APPLY are pure host ops (they read and write
// netExecution->pipes directly and never touch device memory), so their inputs
// and outputs are excluded.
void NnCudaDeviceSegment::buildPipeIndexSets() {
    uploadPipeIndices.clear();
    downloadPipeIndices.clear();
    isHostOnlyPlanSegment = false;
    hasControlOps = false;
    if (device == nullptr || segmentConfig == nullptr || netExecution == nullptr) return;
    NnNetConfig *netConfig = device->getNetConfig();
    if (netConfig == nullptr) return;

    // Pass 1: classify the segment. A segment made only of barrier/apply ops
    // runs entirely on the host, so it must not move any bytes at all.
    bool sawNonControlOp = false;
    for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
        if (isCudaControlOp(segmentConfig->ops[i].code)) hasControlOps = true;
        else sawNonControlOp = true;
    }
    isHostOnlyPlanSegment = hasControlOps && !sawNonControlOp;
    if (isHostOnlyPlanSegment) return;

    std::vector<unsigned char> uploadSeen(netConfig->nPipes, 0u);
    std::vector<unsigned char> downloadSeen(netConfig->nPipes, 0u);

    const auto addUpload = [&](NnUint pipeIndex) {
        if (pipeIndex >= netConfig->nPipes) {
            throw std::runtime_error("CUDA segment pipe index out of range during upload planning");
        }
        if (uploadSeen[pipeIndex] != 0u) return;
        uploadSeen[pipeIndex] = 1u;
        uploadPipeIndices.push_back(pipeIndex);
    };
    const auto addDownload = [&](NnUint pipeIndex) {
        if (pipeIndex >= netConfig->nPipes) {
            throw std::runtime_error("CUDA segment pipe index out of range during download planning");
        }
        if (downloadSeen[pipeIndex] != 0u) return;
        downloadSeen[pipeIndex] = 1u;
        downloadPipeIndices.push_back(pipeIndex);
    };

    for (NnUint i = 0u; i < netConfig->nPreSyncs; ++i) {
        addUpload(netConfig->preSyncs[i].pipeIndex);
    }

    for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
        const NnOpConfig &op = segmentConfig->ops[i];
        if (isCudaControlOp(op.code)) continue;
        if (op.input.source == SRC_PIPE) addUpload(op.input.pointerIndex);
        if (op.output.source == SRC_PIPE) addDownload(op.output.pointerIndex);
        if (op.code == OP_ROPE && op.config != nullptr) {
            const NnRopeOpConfig *config = (const NnRopeOpConfig *)op.config;
            addUpload(config->positionPipeIndex);
        } else if (op.code == OP_MULTIHEAD_ATT && op.config != nullptr) {
            const NnMultiHeadAttOpConfig *config = (const NnMultiHeadAttOpConfig *)op.config;
            addUpload(config->positionPipeIndex);
            if (config->kvSlotStride != 0u) addUpload(config->slotPipeIndex);
        } else if (op.code == OP_SHIFT && op.config != nullptr) {
            const NnShiftOpCodeConfig *config = (const NnShiftOpCodeConfig *)op.config;
            addUpload(config->indexPipeIndex);
            if (config->kvSlotStride != 0u) addUpload(config->slotPipeIndex);
        }
    }
}

// Resolve every op's device buffers and plan-dependent layouts once per plan
// epoch. refreshPointers() has already rewritten the op configs (slices,
// offsets, column counts) for the current plan, so this must run after it.
void NnCudaDeviceSegment::rebuildBindings() {
    opBindings.clear();
    bindingsValid = false;
    bindingsEpoch = device != nullptr ? device->getPlanEpoch() : 0u;
    if (device == nullptr || segmentConfig == nullptr) return;
    NnNetConfig *netConfig = device->getNetConfig();
    NnNodeConfig *nodeConfig = device->getNodeConfig();
    if (netConfig == nullptr || nodeConfig == nullptr) return;

    const NnUnevenPartitionPlan *plan = device->getPartitionPlan();
    opBindings.resize(segmentConfig->nOps);
    for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
        const NnOpConfig &op = segmentConfig->ops[i];
        const NnCudaPointerRef in = resolveCudaPointer(device, netConfig, nodeConfig, plan, &op.input);
        const NnCudaPointerRef out = resolveCudaPointer(device, netConfig, nodeConfig, plan, &op.output);
        opBindings[i].inBuf = in.buffer;
        opBindings[i].outBuf = out.buffer;
        opBindings[i].inLayout = in.layout;
        opBindings[i].outLayout = out.layout;
    }
    bindingsValid = true;
}

NnCudaDeviceSegment::~NnCudaDeviceSegment() = default;

std::string NnCudaDeviceSegment::unsupportedOpMessage(NnUint opIndex) const {
    std::ostringstream out;
    out << "CUDA backend skeleton does not implement real operators yet"
        << " [device=" << (device ? device->getGpuIndex() : 0u)
        << ", node=" << (device ? device->getNodeIndex() : 0u)
        << ", segment=" << segmentIndex
        << ", opIndex=" << opIndex;
    if (segmentConfig != nullptr && opIndex < segmentConfig->nOps) {
        const NnOpConfig &op = segmentConfig->ops[opIndex];
        NnSize3D inputSize = size0();
        NnSize3D outputSize = size0();
        if (device != nullptr) {
            inputSize = device->resolvePointerLogicalSize(&op.input);
            outputSize = device->resolvePointerLogicalSize(&op.output);
        }
        out << ", opName=" << (op.name ? op.name : "<unnamed>")
            << ", opCode=" << opCodeToString(op.code)
            << "(" << (int)op.code << ")"
            << ", input=" << sizeToString(inputSize)
            << ", output=" << sizeToString(outputSize)
            << ", weight=" << sizeToString(op.weightSize);
    }
    out << "]";
    return out.str();
}

void NnCudaDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    if (nBytes == 0u) return;
    if (device == nullptr) throw std::runtime_error("CUDA loadWeight has no device");
    if (opIndex >= weightBuffers.size()) throw std::runtime_error("CUDA loadWeight op index out of range: " + std::to_string(opIndex));
    weightBuffers[opIndex]->write(weight, offset, nBytes, device->getStream(), &device->staging);
}

void NnCudaDeviceSegment::readWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *out) {
    if (nBytes == 0u) return;
    if (device == nullptr) throw std::runtime_error("CUDA readWeight has no device");
    if (opIndex >= weightBuffers.size()) throw std::runtime_error("CUDA readWeight op index out of range: " + std::to_string(opIndex));
    weightBuffers[opIndex]->read(out, offset, nBytes, device->getStream(), &device->staging);
}

void NnCudaDeviceSegment::uploadSegmentInputs(NnUint batchSize) {
    if (device == nullptr || segmentConfig == nullptr || netExecution == nullptr) return;
    NnNetConfig *netConfig = device->getNetConfig();
    if (netConfig == nullptr) return;
    if (batchSize > netConfig->nBatches) {
        throw std::runtime_error("CUDA segment batchSize exceeds configured nBatches");
    }
    if (isHostOnlyPlanSegment || uploadPipeIndices.empty()) return;

    // Size the transfer set first so the whole segment goes through one pinned
    // block and one round of async copies, with no per-pipe stream sync.
    NnSize totalBytes = 0u;
    std::vector<NnSize> offsets(uploadPipeIndices.size(), 0u);
    std::vector<NnSize> sizes(uploadPipeIndices.size(), 0u);
    for (std::size_t i = 0u; i < uploadPipeIndices.size(); ++i) {
        NnCudaBuffer *pipe = device->data.resolvePipe(uploadPipeIndices[i]);
        const NnSize nBytes = pipe->calcSliceSize(batchSize, netConfig->nBatches);
        offsets[i] = totalBytes;
        sizes[i] = nBytes;
        totalBytes += nBytes;
    }
    if (totalBytes == 0u) return;

    NnByte *host = (NnByte *)device->uploadStaging.ensure(totalBytes);
    if (host == nullptr) throw std::runtime_error("CUDA segment upload staging allocation failed");

    const bool ioProfile = dllamaIoProbeEnabled();
    cudaStream_t stream = (cudaStream_t)device->getStream();

    for (std::size_t i = 0u; i < uploadPipeIndices.size(); ++i) {
        const NnUint pipeIndex = uploadPipeIndices[i];
        if (sizes[i] == 0u) continue;
        NnByte *src = netExecution->pipes[pipeIndex];
        if (src == nullptr) {
            throw std::runtime_error("CUDA segment upload received null host pipe " + std::to_string(pipeIndex));
        }
        const std::uint64_t memcpyStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
        std::memcpy(host + offsets[i], src, sizes[i]);
        if (ioProfile) dllamaIoProbeRecordHostMemcpy(DLLAMA_IO_PROBE_H2D, dllamaIoProbeNowUs() - memcpyStartUs, (std::uint64_t)sizes[i]);

        NnCudaBuffer *pipe = device->data.resolvePipe(pipeIndex);
        const std::uint64_t copyStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
        NN_CUDA_CHECK(cudaMemcpyAsync(pipe->data(), host + offsets[i], sizes[i], cudaMemcpyHostToDevice, stream));
        if (ioProfile) dllamaIoProbeRecordCudaMemcpyAsync(DLLAMA_IO_PROBE_H2D, dllamaIoProbeNowUs() - copyStartUs, (std::uint64_t)sizes[i]);
    }
}

bool NnCudaDeviceSegment::downloadSegmentOutputs(NnUint batchSize) {
    if (device == nullptr || segmentConfig == nullptr || netExecution == nullptr) return false;
    NnNetConfig *netConfig = device->getNetConfig();
    if (netConfig == nullptr) return false;
    if (batchSize > netConfig->nBatches) {
        throw std::runtime_error("CUDA segment batchSize exceeds configured nBatches");
    }
    if (isHostOnlyPlanSegment || downloadPipeIndices.empty()) return false;

    NnSize totalBytes = 0u;
    std::vector<NnSize> offsets(downloadPipeIndices.size(), 0u);
    std::vector<NnSize> sizes(downloadPipeIndices.size(), 0u);
    for (std::size_t i = 0u; i < downloadPipeIndices.size(); ++i) {
        NnCudaBuffer *pipe = device->data.resolvePipe(downloadPipeIndices[i]);
        const NnSize nBytes = pipe->calcSliceSize(batchSize, netConfig->nBatches);
        offsets[i] = totalBytes;
        sizes[i] = nBytes;
        totalBytes += nBytes;
    }
    if (totalBytes == 0u) return false;

    NnByte *host = (NnByte *)device->downloadStaging.ensure(totalBytes);
    if (host == nullptr) throw std::runtime_error("CUDA segment download staging allocation failed");

    const bool ioProfile = dllamaIoProbeEnabled();
    cudaStream_t stream = (cudaStream_t)device->getStream();

    for (std::size_t i = 0u; i < downloadPipeIndices.size(); ++i) {
        if (sizes[i] == 0u) continue;
        NnCudaBuffer *pipe = device->data.resolvePipe(downloadPipeIndices[i]);
        const std::uint64_t copyStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
        NN_CUDA_CHECK(cudaMemcpyAsync(host + offsets[i], pipe->data(), sizes[i], cudaMemcpyDeviceToHost, stream));
        if (ioProfile) dllamaIoProbeRecordCudaMemcpyAsync(DLLAMA_IO_PROBE_D2H, dllamaIoProbeNowUs() - copyStartUs, (std::uint64_t)sizes[i]);
    }

    // Exactly one sync for the whole download set.
    const std::uint64_t syncStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
    NN_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (ioProfile) dllamaIoProbeRecordCudaStreamSync(DLLAMA_IO_PROBE_D2H, dllamaIoProbeNowUs() - syncStartUs);

    for (std::size_t i = 0u; i < downloadPipeIndices.size(); ++i) {
        if (sizes[i] == 0u) continue;
        NnByte *dst = netExecution->pipes[downloadPipeIndices[i]];
        if (dst == nullptr) {
            throw std::runtime_error("CUDA segment download received null host pipe " + std::to_string(downloadPipeIndices[i]));
        }
        const std::uint64_t memcpyStartUs = ioProfile ? dllamaIoProbeNowUs() : 0u;
        std::memcpy(dst, host + offsets[i], sizes[i]);
        if (ioProfile) dllamaIoProbeRecordHostMemcpy(DLLAMA_IO_PROBE_D2H, dllamaIoProbeNowUs() - memcpyStartUs, (std::uint64_t)sizes[i]);
    }
    return true;
}

void NnCudaDeviceSegment::executeOp(NnUint opIndex, NnUint batchSize) {
    if (device == nullptr || segmentConfig == nullptr) return;
    NnNetConfig *netConfig = device->getNetConfig();
    NnNodeConfig *nodeConfig = device->getNodeConfig();
    if (netConfig == nullptr || nodeConfig == nullptr) return;
    if (opIndex >= segmentConfig->nOps) {
        throw std::runtime_error("CUDA executeOp op index out of range: " + std::to_string(opIndex));
    }

    const NnOpConfig &op = segmentConfig->ops[opIndex];
    if (!bindingsValid || opIndex >= opBindings.size()) {
        throw std::runtime_error("CUDA executeOp has no cached binding for op index: " + std::to_string(opIndex));
    }
    NnCudaPointerRef in;
    NnCudaPointerRef out;
    in.buffer = opBindings[opIndex].inBuf;
    in.layout = opBindings[opIndex].inLayout;
    out.buffer = opBindings[opIndex].outBuf;
    out.layout = opBindings[opIndex].outLayout;
    cudaStream_t stream = (cudaStream_t)device->getStream();
    const NnCudaLaunchConfig &launch = device->getLaunchConfig();
    const unsigned int elementwiseBlock = (unsigned int)launch.elementwiseBlockSize;
    const unsigned int attentionBlock = (unsigned int)launch.attentionBlockSize;
    const unsigned int q80q40SmallKBlock = (unsigned int)launch.q80q40SmallKBlockSize;
    const unsigned int q80q40LargeKBlock = (unsigned int)launch.q80q40LargeKBlockSize;
    const unsigned int softmaxBlock = (unsigned int)launch.softmaxBlockSize;
    const unsigned int moeGateBlock = (unsigned int)launch.moeGateBlockSize;

    switch (op.code) {
        case OP_EMBEDDING: {
            if (in.layout.logicalSize.floatType != F_32 || out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            if (op.weightSize.x != out.layout.logicalSize.x) {
                throw std::runtime_error("CUDA EMBEDDING weight/output width mismatch");
            }
            const NnSize total = (NnSize)batchSize * out.layout.logicalSize.x;
            if (total == 0u) return;
            if (op.weightSize.floatType == F_32) {
                embeddingF32F32Kernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    (const float *)weightBuffers[opIndex]->data(),
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (op.weightSize.floatType == F_Q40) {
                if ((op.weightSize.x % Q40_BLOCK_SIZE) != 0u) {
                    throw std::runtime_error("CUDA EMBEDDING Q40 requires block-aligned embedding width");
                }
                embeddingQ40F32Kernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    (const NnBlockQ40 *)weightBuffers[opIndex]->data(),
                    op.weightSize.x / Q40_BLOCK_SIZE,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (op.weightSize.floatType == F_Q80) {
                if ((op.weightSize.x % Q80_BLOCK_SIZE) != 0u) {
                    throw std::runtime_error("CUDA EMBEDDING Q80 requires block-aligned embedding width");
                }
                embeddingQ80F32Kernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    (const NnBlockQ80 *)weightBuffers[opIndex]->data(),
                    op.weightSize.x / Q80_BLOCK_SIZE,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
                return;
            }
            throw std::runtime_error(unsupportedOpMessage(opIndex));
        }
        case OP_INV_RMS: {
            const NnInvRmsOpConfig *config = (const NnInvRmsOpConfig *)op.config;
            if (config == nullptr) throw std::runtime_error("CUDA INV_RMS missing config");
            if (in.layout.logicalSize.floatType != F_32 || out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            if (config->nColumns == 0u || (in.layout.logicalSize.x % config->nColumns) != 0u) {
                throw std::runtime_error("CUDA INV_RMS invalid nColumns");
            }
            const NnSize total = (NnSize)batchSize * config->nColumns;
            if (total != 0u) {
                /* One block per (y, col): the block splits the colSize
                 * accumulation instead of one thread walking all of it. */
                dim3 invRmsGrid(config->nColumns, batchSize, 1u);
                invRmsF32Kernel<<<invRmsGrid, (unsigned int)launch.reductionBlockSize, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    config->epsilon,
                    config->nColumns,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_RMS_NORM: {
            const NnRmsNormOpConfig *config = (const NnRmsNormOpConfig *)op.config;
            if (config == nullptr) throw std::runtime_error("CUDA RMS_NORM missing config");
            if (in.layout.logicalSize.floatType != F_32 ||
                op.weightSize.floatType != F_32 ||
                out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            if (config->nColumns == 0u || (out.layout.logicalSize.x % config->nColumns) != 0u) {
                throw std::runtime_error("CUDA RMS_NORM invalid nColumns");
            }
            NnCudaBuffer *invRmsBuf = device->data.resolveBuffer(config->invRmsBufferIndex);
            const NnUint invRmsRowStride = nodeConfig->buffers[config->invRmsBufferIndex].size.x;
            const NnSize total = (NnSize)batchSize * out.layout.logicalSize.x;
            if (total != 0u) {
                rmsNormF32Kernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    (const float *)weightBuffers[opIndex]->data(),
                    (const float *)invRmsBuf->data(),
                    invRmsRowStride,
                    config->nColumns,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_MATMUL: {
            const NnMatmulOpConfig *config = (const NnMatmulOpConfig *)op.config;
            if (config == nullptr) throw std::runtime_error("CUDA MATMUL missing config");
            if (in.layout.logicalSize.floatType == F_Q80 &&
                op.weightSize.floatType == F_Q40 &&
                out.layout.logicalSize.floatType == F_32) {
                if ((config->aView.strideX != 0u && config->aView.strideX != 1u) ||
                    (config->cView.strideX != 0u && config->cView.strideX != 1u) ||
                    config->aView.sizeY != 0u ||
                    config->cView.sizeY != 0u) {
                    throw std::runtime_error("CUDA Q80xQ40 MATMUL supports only contiguous per-row A/C tensor views");
                }
                const NnTensorViewLayout aView = resolveTensorView(&config->aView, 0u, in.layout.logicalSize.x, in.layout.logicalSize.x);
                const NnTensorViewLayout cView = resolveTensorView(&config->cView, 0u, out.layout.logicalSize.x, out.layout.logicalSize.x);
                const NnUint aLen = (NnUint)aView.sizeX;
                const NnUint cLen = (NnUint)cView.sizeX;
                if (aLen == 0u || cLen == 0u) return;
                if ((op.weightSize.y % Q40_BLOCK_SIZE) != 0u ||
                    (aView.offset % Q80_BLOCK_SIZE) != 0u ||
                    (aLen % Q80_BLOCK_SIZE) != 0u ||
                    (config->inStart % Q40_BLOCK_SIZE) != 0u) {
                    throw std::runtime_error("CUDA Q80xQ40 MATMUL requires K, input offset, inStart and A slice to be 32-element block aligned");
                }
                if (config->view == 0u) {
                    if (aLen != op.weightSize.y || cLen != op.weightSize.x) {
                        throw std::runtime_error("CUDA Q80xQ40 MATMUL view=0 shape mismatch");
                    }
                } else if (config->view == 1u) {
                    if (aLen != op.weightSize.y || config->outStart + cLen > op.weightSize.x) {
                        throw std::runtime_error("CUDA Q80xQ40 MATMUL view=1 shape mismatch");
                    }
                } else if (config->view == 2u) {
                    if (config->inStart + aLen > op.weightSize.y ||
                        config->outStart + cLen > op.weightSize.x) {
                        throw std::runtime_error("CUDA Q80xQ40 MATMUL view=2 shape mismatch");
                    }
                } else {
                    throw std::runtime_error("CUDA Q80xQ40 MATMUL unsupported view mode");
                }
                const NnUint nActiveExpertsOr1 = config->nActiveExperts == 0u ? 1u : config->nActiveExperts;
                NnUint activeExpertRowStride = 0u;
                const float *activeExpertIndexes = nullptr;
                if (config->nActiveExperts != 0u) {
                    NnCudaBuffer *idxBuf = device->data.resolveBuffer(config->activeExpertIndexesBufferIndex);
                    activeExpertIndexes = (const float *)idxBuf->data();
                    activeExpertRowStride = nodeConfig->buffers[config->activeExpertIndexesBufferIndex].size.x;
                }
                const NnUint weightBlocksPerRow = op.weightSize.y / Q40_BLOCK_SIZE;
                const NnUint aBlocks = aLen / Q80_BLOCK_SIZE;
                const NnUint aOffsetBlocks = (NnUint)aView.offset / Q80_BLOCK_SIZE;
                const NnUint inStartBlocks = config->inStart / Q40_BLOCK_SIZE;
                const NnSize totalOutputs = (NnSize)batchSize * nActiveExpertsOr1 * cLen;
                if (aBlocks <= launch.q80q40SmallKMaxBlocks) {
                    matmulQ80Q40SmallKKernel<<<cudaBlocks(totalOutputs, q80q40SmallKBlock), q80q40SmallKBlock, 0, stream>>>(
                        (const NnByte *)in.buffer->data(), in.layout,
                        (NnByte *)out.buffer->data(), out.layout,
                        (const NnBlockQ40 *)weightBuffers[opIndex]->data(),
                        activeExpertIndexes,
                        activeExpertRowStride,
                        config->nActiveExperts,
                        nActiveExpertsOr1,
                        op.weightSize.z,
                        weightBlocksPerRow,
                        op.weightSize.x,
                        config->view,
                        aOffsetBlocks,
                        (NnUint)cView.offset,
                        aBlocks,
                        cLen,
                        inStartBlocks,
                        config->outStart,
                        batchSize);
                    NN_CUDA_CHECK(cudaGetLastError());
                } else {
                    /* Warp-per-row blocks: 8 rows per block, one block per
                     * (row group, y, e) with the whole x group cached in
                     * shared memory when it fits. */
                    const unsigned int largeKBlock = q80q40LargeKBlock;
                    const unsigned int warpsPerBlock = largeKBlock >> 5u;
                    const NnSize rowGroups = ((NnSize)cLen + warpsPerBlock - 1u) / warpsPerBlock;
                    const NnSize sharedBytes = aBlocks <= NN_CUDA_MATMUL_SHARED_MAX_BLOCKS
                        ? cudaQ80Q40LargeKSharedBytes(aBlocks)
                        : 0u;
                    dim3 largeKGrid((unsigned int)rowGroups, (unsigned int)((NnSize)batchSize * nActiveExpertsOr1), 1u);
                    matmulQ80Q40LargeKKernel<<<largeKGrid, largeKBlock, (size_t)sharedBytes, stream>>>(
                        (const NnByte *)in.buffer->data(), in.layout,
                        (NnByte *)out.buffer->data(), out.layout,
                        (const NnBlockQ40 *)weightBuffers[opIndex]->data(),
                        activeExpertIndexes,
                        activeExpertRowStride,
                        config->nActiveExperts,
                        nActiveExpertsOr1,
                        op.weightSize.z,
                        weightBlocksPerRow,
                        op.weightSize.x,
                        config->view,
                        aOffsetBlocks,
                        (NnUint)cView.offset,
                        aBlocks,
                        cLen,
                        inStartBlocks,
                        config->outStart,
                        batchSize);
                    NN_CUDA_CHECK(cudaGetLastError());
                }
                return;
            }
            if (in.layout.logicalSize.floatType != F_32 ||
                op.weightSize.floatType != F_32 ||
                out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            if ((config->aView.strideX != 0u && config->aView.strideX != 1u) ||
                (config->cView.strideX != 0u && config->cView.strideX != 1u) ||
                config->aView.sizeY != 0u ||
                config->cView.sizeY != 0u) {
                throw std::runtime_error("CUDA MATMUL supports only contiguous per-row A/C tensor views");
            }
            const NnTensorViewLayout aView = resolveTensorView(&config->aView, 0u, in.layout.logicalSize.x, in.layout.logicalSize.x);
            const NnTensorViewLayout cView = resolveTensorView(&config->cView, 0u, out.layout.logicalSize.x, out.layout.logicalSize.x);
            const NnUint aLen = (NnUint)aView.sizeX;
            const NnUint cLen = (NnUint)cView.sizeX;
            if (aLen == 0u || cLen == 0u) return;
            if (config->view == 0u) {
                if (aLen != op.weightSize.y || cLen != op.weightSize.x) {
                    throw std::runtime_error("CUDA MATMUL view=0 shape mismatch");
                }
            } else if (config->view == 1u) {
                if (aLen != op.weightSize.y || config->outStart + cLen > op.weightSize.x) {
                    throw std::runtime_error("CUDA MATMUL view=1 shape mismatch");
                }
            } else if (config->view == 2u) {
                if (config->inStart + aLen > op.weightSize.y ||
                    config->outStart + cLen > op.weightSize.x) {
                    throw std::runtime_error("CUDA MATMUL view=2 shape mismatch");
                }
            } else {
                throw std::runtime_error("CUDA MATMUL unsupported view mode");
            }

            cublasHandle_t handle = (cublasHandle_t)device->getBlasHandle();
            if (handle == nullptr) throw std::runtime_error("CUDA MATMUL missing cuBLAS handle");
            const float alpha = 1.0f;
            const float beta = 0.0f;
            const NnUint nActiveExpertsOr1 = config->nActiveExperts == 0u ? 1u : config->nActiveExperts;
            NnUint activeExpertRowStride = 0u;
            std::vector<float> activeExpertIndexesHost;
            if (config->nActiveExperts != 0u) {
                NnCudaBuffer *idxBuf = device->data.resolveBuffer(config->activeExpertIndexesBufferIndex);
                activeExpertRowStride = nodeConfig->buffers[config->activeExpertIndexesBufferIndex].size.x;
                activeExpertIndexesHost.resize((NnSize)batchSize * activeExpertRowStride);
                idxBuf->read(
                    (NnByte *)activeExpertIndexesHost.data(),
                    0u,
                    getBytes(F_32, (NnSize)batchSize * activeExpertRowStride),
                    device->getStream(),
                    &device->staging);
            }
            const NnSize weightExpertFloats = (NnSize)op.weightSize.y * op.weightSize.x;
            for (NnUint y = 0u; y < batchSize; ++y) {
                for (NnUint e = 0u; e < nActiveExpertsOr1; ++e) {
                    NnUint activeExpertIndex = 0u;
                    if (config->nActiveExperts != 0u) {
                        activeExpertIndex = (NnUint)activeExpertIndexesHost[(NnSize)y * activeExpertRowStride + e];
                    }
                    if (activeExpertIndex >= std::max(config->nExperts, 1u) || activeExpertIndex >= op.weightSize.z) {
                        throw std::runtime_error("CUDA MATMUL active expert index out of range");
                    }
                    const float *x = (const float *)cudaHostRowBase((NnByte *)in.buffer->data(), in.layout, e, y) + aView.offset;
                    float *c = (float *)cudaHostRowBase((NnByte *)out.buffer->data(), out.layout, e, y) + cView.offset;
                    const float *wBase = (const float *)weightBuffers[opIndex]->data() + (NnSize)activeExpertIndex * weightExpertFloats;
                    const float *w = nullptr;
                    int lda = 0;
                    int m = (int)aLen;
                    int n = (int)cLen;
                    if (config->view == 0u) {
                        w = wBase;
                        lda = (int)aLen;
                    } else if (config->view == 1u) {
                        w = wBase + (NnSize)config->outStart * op.weightSize.y;
                        lda = (int)op.weightSize.y;
                    } else {
                        w = wBase + (NnSize)config->outStart * op.weightSize.y + config->inStart;
                        lda = (int)op.weightSize.y;
                    }
                    NN_CUBLAS_CHECK(cublasSgemv(
                        handle,
                        CUBLAS_OP_T,
                        m,
                        n,
                        &alpha,
                        w,
                        lda,
                        x,
                        1,
                        &beta,
                        c,
                        1));
                }
            }
            return;
        }
        case OP_ROPE: {
            const NnRopeOpConfig *config = (const NnRopeOpConfig *)op.config;
            if (config == nullptr) throw std::runtime_error("CUDA ROPE missing config");
            if (in.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            const NnTensorViewLayout view = resolveOpView(op, in.layout.logicalSize, in.layout.logicalSize.x);
            if ((view.sizeX % 2u) != 0u || view.strideX != 1u) {
                throw std::runtime_error("CUDA ROPE requires contiguous even-width view");
            }
            if (config->type == ROPE_FALCON &&
                (config->slice.headDim == 0u ||
                 (view.offset % config->slice.headDim) != 0u ||
                 (view.sizeX % config->slice.headDim) != 0u)) {
                throw std::runtime_error("CUDA FALCON ROPE requires whole-head view");
            }
            NnCudaBuffer *positionPipe = device->data.resolvePipe(config->positionPipeIndex);
            NnCudaBuffer *ropeCache = device->data.resolveBuffer(config->ropeCacheBufferIndex);
            const NnSize total = config->type == ROPE_FALCON
                ? (NnSize)batchSize * (view.sizeX / config->slice.headDim) * (config->slice.headDim / 2u)
                : (NnSize)batchSize * (view.sizeX / 2u);
            if (total != 0u) {
                ropeF32Kernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (NnByte *)in.buffer->data(), in.layout,
                    (const float *)positionPipe->data(),
                    (const float *)ropeCache->data(),
                    *config,
                    view,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_MULTIHEAD_ATT: {
            const NnMultiHeadAttOpConfig *configPtr = (const NnMultiHeadAttOpConfig *)op.config;
            if (configPtr == nullptr) throw std::runtime_error("CUDA MULTIHEAD_ATT missing config");
            if (out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            NnMultiHeadAttOpConfig config = *configPtr;
            config.qStride = config.qStride == 0u ? config.qSliceD0 : config.qStride;
            config.kvStride = config.kvStride == 0u ? config.kvDim0 : config.kvStride;
            if (config.nHeads == 0u || config.nHeads0 == 0u || config.nKvHeads == 0u ||
                config.headDim == 0u || config.seqLen == 0u) {
                throw std::runtime_error("CUDA MULTIHEAD_ATT invalid zero dimension");
            }
            if ((config.nHeads % config.nKvHeads) != 0u) {
                throw std::runtime_error("CUDA MULTIHEAD_ATT requires nHeads divisible by nKvHeads for GQA");
            }
            if ((config.qStart % config.headDim) != 0u ||
                (config.kvStart % config.headDim) != 0u ||
                (config.qSliceD0 % config.headDim) != 0u ||
                (config.kvDim0 % config.headDim) != 0u) {
                throw std::runtime_error("CUDA MULTIHEAD_ATT requires q/kv ranges aligned to headDim");
            }
            if (config.nHeads0 * config.headDim != config.qSliceD0 ||
                out.layout.logicalSize.x < config.qSliceD0 ||
                config.qStart + config.qSliceD0 > config.qStride ||
                config.kvStart + config.kvDim0 > config.kvStride) {
                throw std::runtime_error("CUDA MULTIHEAD_ATT shape/stride mismatch");
            }
            const NnUint qHeadStart = config.qStart / config.headDim;
            const NnUint kvHeadStart = config.kvStart / config.headDim;
            const NnUint kvHeads0 = config.kvDim0 / config.headDim;
            if (qHeadStart + config.nHeads0 > config.nHeads ||
                kvHeadStart + kvHeads0 > config.nKvHeads) {
                throw std::runtime_error("CUDA MULTIHEAD_ATT local head range exceeds global head count");
            }
            const NnUint kvMul = config.nHeads / config.nKvHeads;
            const NnUint firstGlobalKvHead = qHeadStart / kvMul;
            const NnUint lastGlobalKvHead = (qHeadStart + config.nHeads0 - 1u) / kvMul;
            if (firstGlobalKvHead < kvHeadStart || lastGlobalKvHead >= kvHeadStart + kvHeads0) {
                throw std::runtime_error("CUDA MULTIHEAD_ATT q-head range is not covered by local KV range");
            }
            NnCudaBuffer *positionPipe = device->data.resolvePipe(config.positionPipeIndex);
            NnCudaBuffer *slotPipe = device->data.resolvePipe(config.slotPipeIndex);
            NnCudaBuffer *queryBuffer = device->data.resolveBuffer(config.queryBufferIndex);
            NnCudaBuffer *keyCacheBuffer = device->data.resolveBuffer(config.keyCacheBufferIndex);
            NnCudaBuffer *valueCacheBuffer = device->data.resolveBuffer(config.valueCacheBufferIndex);
            NnCudaBuffer *attBuffer = device->data.resolveBuffer(config.attBufferIndex);
            const NnSize3D &querySize = nodeConfig->buffers[config.queryBufferIndex].size;
            const NnSize3D &keySize = nodeConfig->buffers[config.keyCacheBufferIndex].size;
            const NnSize3D &valueSize = nodeConfig->buffers[config.valueCacheBufferIndex].size;
            const NnSize3D &attSize = nodeConfig->buffers[config.attBufferIndex].size;
            if (querySize.floatType != F_32 || keySize.floatType != F_32 ||
                valueSize.floatType != F_32 || attSize.floatType != F_32 ||
                querySize.x < config.qStride ||
                keySize.length < (NnSize)config.seqLen * config.kvStride ||
                valueSize.length < (NnSize)config.seqLen * config.kvStride ||
                attSize.length < (NnSize)netConfig->nBatches * config.nHeads0 * config.seqLen) {
                throw std::runtime_error("CUDA MULTIHEAD_ATT backing buffer shape mismatch");
            }
            dim3 grid(config.nHeads0, batchSize, 1u);
            if (batchSize != 0u) {
                multiheadAttScoreF32Kernel<<<grid, attentionBlock, 0, stream>>>(
                    (const float *)positionPipe->data(),
                    (const float *)slotPipe->data(),
                    (const float *)queryBuffer->data(),
                    (const float *)keyCacheBuffer->data(),
                    (float *)attBuffer->data(),
                    config,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
                multiheadAttSoftmaxSerialF32Kernel<<<grid, softmaxBlock, 0, stream>>>(
                    (const float *)positionPipe->data(),
                    (float *)attBuffer->data(),
                    config,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
                multiheadAttValueF32Kernel<<<grid, attentionBlock, 0, stream>>>(
                    (NnByte *)out.buffer->data(), out.layout,
                    (const float *)positionPipe->data(),
                    (const float *)slotPipe->data(),
                    (const float *)valueCacheBuffer->data(),
                    (const float *)attBuffer->data(),
                    config,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_CAST: {
            if (in.layout.logicalSize.floatType == F_32 && out.layout.logicalSize.floatType == F_32) {
                const NnTensorViewLayout view = resolveOpView(op, out.layout.logicalSize, out.layout.logicalSize.x);
                const NnSize total = (NnSize)out.layout.logicalSize.z * batchSize * view.sizeY * view.sizeX;
                if (total != 0u) {
                    castF32F32Kernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                        (const NnByte *)in.buffer->data(), in.layout,
                        (NnByte *)out.buffer->data(), out.layout,
                        view, batchSize);
                    NN_CUDA_CHECK(cudaGetLastError());
                }
                return;
            }
            if (in.layout.logicalSize.floatType == F_32 && out.layout.logicalSize.floatType == F_Q80) {
                const NnTensorViewLayout view = resolveOpView(op, out.layout.logicalSize, out.layout.logicalSize.x);
                validateQ80View("CAST F32->Q80", view);
                const NnSize totalBlocks = (NnSize)in.layout.logicalSize.z * batchSize * (view.sizeX / Q80_BLOCK_SIZE);
                if (totalBlocks != 0u) {
                    /* One warp per 32-element block, so one thread per element. */
                    q80QuantizeKernel<<<cudaBlocks(totalBlocks * 32u, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                        (const NnByte *)in.buffer->data(), in.layout,
                        (NnByte *)out.buffer->data(), out.layout,
                        (NnUint)view.offset, (NnUint)view.sizeX, batchSize);
                    NN_CUDA_CHECK(cudaGetLastError());
                }
                return;
            }
            throw std::runtime_error(unsupportedOpMessage(opIndex));
        }
        case OP_SILU: {
            const NnTensorViewLayout view = resolveOpView(op, out.layout.logicalSize, out.layout.logicalSize.x);
            const NnSize total = (NnSize)out.layout.logicalSize.z * batchSize * view.sizeY * view.sizeX;
            if (total != 0u) {
                siluKernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (NnByte *)out.buffer->data(), out.layout, view, batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_GELU: {
            const NnTensorViewLayout view = resolveOpView(op, out.layout.logicalSize, out.layout.logicalSize.x);
            const NnSize total = (NnSize)out.layout.logicalSize.z * batchSize * view.sizeY * view.sizeX;
            if (total != 0u) {
                geluKernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (NnByte *)out.buffer->data(), out.layout, view, batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_MUL: {
            const NnMulOpCodeConfig *config = (const NnMulOpCodeConfig *)op.config;
            if (config == nullptr) throw std::runtime_error("CUDA MUL missing config");
            const NnTensorViewLayout view = resolveOpView(op, out.layout.logicalSize, out.layout.logicalSize.x);
            NnCudaBuffer *multBuf = device->data.resolveBuffer(config->multiplierBufferIndex);
            const NnUint multRowStride = nodeConfig->buffers[config->multiplierBufferIndex].size.x;
            const NnSize total = (NnSize)in.layout.logicalSize.z * batchSize * view.sizeY * view.sizeX;
            if (total != 0u) {
                mulKernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    (const float *)multBuf->data(), multRowStride,
                    view, batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_SILU_MUL: {
            const NnMulOpCodeConfig *config = (const NnMulOpCodeConfig *)op.config;
            if (config == nullptr) throw std::runtime_error("CUDA SILU_MUL missing config");
            const NnTensorViewLayout view = resolveOpView(op, out.layout.logicalSize, out.layout.logicalSize.x);
            NnCudaBuffer *multBuf = device->data.resolveBuffer(config->multiplierBufferIndex);
            const NnUint multRowStride = nodeConfig->buffers[config->multiplierBufferIndex].size.x;
            const NnSize total = (NnSize)in.layout.logicalSize.z * batchSize * view.sizeY * view.sizeX;
            if (total != 0u) {
                siluMulKernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    (const float *)multBuf->data(), multRowStride,
                    view, batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_SCALE: {
            const NnScaleOpCodeConfig *config = (const NnScaleOpCodeConfig *)op.config;
            if (config == nullptr) throw std::runtime_error("CUDA SCALE missing config");
            const NnTensorViewLayout view = resolveOpView(op, in.layout.logicalSize, in.layout.logicalSize.x);
            NnCudaBuffer *scaleBuf = device->data.resolveBuffer(config->scaleBufferIndex);
            const NnSize total = (NnSize)in.layout.logicalSize.z * batchSize * view.sizeY * view.sizeX;
            if (total != 0u) {
                scaleKernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    (const float *)scaleBuf->data(),
                    view, batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_MERGE_ADD: {
            if (out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            if (out.layout.logicalSize.x == 0u || (in.layout.logicalSize.x % out.layout.logicalSize.x) != 0u) {
                throw std::runtime_error("CUDA MERGE_ADD invalid input/output widths");
            }
            const NnUint nSlices = in.layout.logicalSize.x / out.layout.logicalSize.x;
            const NnSize total = (NnSize)batchSize * out.layout.logicalSize.x;
            if (total == 0u) return;
            if (in.layout.logicalSize.floatType == F_32) {
                mergeAddF32Kernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    nSlices, batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (in.layout.logicalSize.floatType == F_Q80) {
                if ((out.layout.logicalSize.x % Q80_BLOCK_SIZE) != 0u) {
                    throw std::runtime_error("CUDA MERGE_ADD Q80 input requires block-aligned output width");
                }
                mergeAddQ80F32Kernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    nSlices, batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
                return;
            }
            throw std::runtime_error(unsupportedOpMessage(opIndex));
        }
        case OP_MERGE_SUM: {
            if (in.layout.logicalSize.floatType != F_32 || out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            const NnSize total = (NnSize)batchSize * out.layout.logicalSize.x;
            if (total != 0u) {
                mergeSumF32Kernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_REPEAT_Z: {
            if (in.layout.logicalSize.floatType != F_32 || out.layout.logicalSize.floatType != F_Q80) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            if ((out.layout.logicalSize.x % Q80_BLOCK_SIZE) != 0u) {
                throw std::runtime_error("CUDA REPEAT_Z requires Q80 block-aligned width");
            }
            const NnSize quantBlocks = (NnSize)batchSize * (out.layout.logicalSize.x / Q80_BLOCK_SIZE);
            if (quantBlocks != 0u) {
                /* One warp per block here too, so scale the grid by 32. */
                q80QuantizeKernel<<<cudaBlocks(quantBlocks * 32u, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    0u, out.layout.logicalSize.x,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            const NnSize copyBlocks = (NnSize)(out.layout.logicalSize.z > 0u ? out.layout.logicalSize.z - 1u : 0u)
                * batchSize * (out.layout.logicalSize.x / Q80_BLOCK_SIZE);
            if (copyBlocks != 0u) {
                repeatZCopyQ80Kernel<<<cudaBlocks(copyBlocks, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (NnByte *)out.buffer->data(), out.layout,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_SHIFT: {
            const NnShiftOpCodeConfig *config = (const NnShiftOpCodeConfig *)op.config;
            if (config == nullptr) throw std::runtime_error("CUDA SHIFT missing config");
            if (in.layout.logicalSize.floatType != F_32 || out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            NnCudaBuffer *indexPipe = device->data.resolvePipe(config->indexPipeIndex);
            NnCudaBuffer *slotPipe = device->data.resolvePipe(config->slotPipeIndex);
            const NnSize total = (NnSize)batchSize * in.layout.logicalSize.x;
            if (total != 0u) {
                shiftF32Kernel<<<cudaBlocks(total, elementwiseBlock), elementwiseBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    (const float *)indexPipe->data(),
                    (const float *)slotPipe->data(),
                    *config,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_SOFTMAX: {
            if (out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            const NnTensorViewLayout view = resolveOpView(op, out.layout.logicalSize, out.layout.logicalSize.x);
            const NnUint rows = out.layout.logicalSize.z * batchSize;
            if (rows != 0u && view.sizeX != 0u) {
                if (view.sizeX <= 32u) {
                    softmaxF32SerialKernel<<<rows, 1, 0, stream>>>(
                        (NnByte *)out.buffer->data(), out.layout,
                        view,
                        batchSize);
                } else {
                    softmaxF32Kernel<<<rows, softmaxBlock, 0, stream>>>(
                        (NnByte *)out.buffer->data(), out.layout,
                        view,
                        batchSize);
                }
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_MOE_GATE: {
            const NnMoeGateOpCodeConfig *config = (const NnMoeGateOpCodeConfig *)op.config;
            if (config == nullptr) throw std::runtime_error("CUDA MOE_GATE missing config");
            if (in.layout.logicalSize.floatType != F_32 || out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            if (config->k == 0u ||
                in.layout.logicalSize.z != 1u ||
                in.layout.logicalSize.x < config->k ||
                out.layout.logicalSize.z != config->k ||
                out.layout.logicalSize.x != 1u ||
                out.layout.logicalSize.y < batchSize) {
                throw std::runtime_error("CUDA MOE_GATE shape mismatch");
            }
            if (config->indexesBufferIndex >= nodeConfig->nBuffers) {
                throw std::runtime_error("CUDA MOE_GATE indexes buffer index out of range");
            }
            const NnSize3D &idxSize = nodeConfig->buffers[config->indexesBufferIndex].size;
            if (idxSize.floatType != F_32 || idxSize.x < config->k || idxSize.y < batchSize) {
                throw std::runtime_error("CUDA MOE_GATE indexes buffer shape mismatch");
            }
            NnCudaBuffer *indexesBuf = device->data.resolveBuffer(config->indexesBufferIndex);
            if (batchSize != 0u) {
                moeGateF32Kernel<<<cudaBlocks(batchSize, moeGateBlock), moeGateBlock, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    (float *)indexesBuf->data(),
                    config->k,
                    config->normTopk,
                    idxSize.x,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_PLAN_BARRIER: {
            if (netExecution == nullptr || op.input.source != SRC_PIPE || op.output.source != SRC_PIPE) return;
            const float *posPipe = (const float *)netExecution->pipes[op.input.pointerIndex];
            float *planPipe = (float *)netExecution->pipes[op.output.pointerIndex];
            if (posPipe == nullptr || planPipe == nullptr) return;
            const NnUint layerIndex = op.index;
            const NnUint pos = posPipe[0] >= 0.0f ? (NnUint)posPipe[0] : 0u;
            const NnUnevenPartitionPlan *plan = device->getPartitionPlan();
            const NnUint myNode = device->getNodeIndex();
            const NnStageConfig *stage = findStageForNodeCuda(plan, myNode);
            const PlanCommandSnapshot snap = planCommandCache().load();
            const PlanCommand &pc = snap.cmd;
            const bool hasPayload =
                pc.cmdKind == PLAN_CMD_KIND_HEAD ||
                pc.cmdKind == PLAN_CMD_KIND_FFN ||
                pc.cmdKind == PLAN_CMD_KIND_BOTH ||
                (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves != 0u);
            const bool hasCmd =
                isValidPlanCommandHeader(pc) &&
                (pc.mode == PLAN_CMD_MODE_EXACT || pc.mode == PLAN_CMD_MODE_NEXT_BARRIER) &&
                hasPayload;
            const bool stageOk = (pc.stageIndex == 0xFFFFFFFFu) || (stage != nullptr && stage->stageIndex == pc.stageIndex);
            const bool isStageRoot = stageOk && stage != nullptr && stage->rootNodeIndex == myNode;

            unsigned int emitEpoch = device->getPlanEpoch();
            unsigned int cmd = 0u;
            unsigned int fromNode = 0u;
            unsigned int toNode = 0u;
            unsigned int headMove = 0u;
            unsigned int ffnMove = 0u;
            bool trigger = false;
            if (hasCmd && isStageRoot) {
                const unsigned int lastEmitted = device->getLastPlanCmdSeqEmitted();
                if (!(pc.seq != 0u && pc.seq <= lastEmitted)) {
                    if (pc.mode == PLAN_CMD_MODE_EXACT) {
                        const bool exactMatch = layerIndex == pc.triggerLayer && pos == pc.triggerPos;
                        const bool missedExact =
                            pc.triggerPos != 0xFFFFFFFFu &&
                            pc.triggerLayer != 0xFFFFFFFFu &&
                            (pos > pc.triggerPos || (pos == pc.triggerPos && layerIndex > pc.triggerLayer));
                        trigger = exactMatch || missedExact;
                    } else {
                        trigger = true;
                    }
                }
            }
            if (trigger) {
                emitEpoch += 1u;
                if (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves == 1u) {
                    const PlanMove &m = pc.moves[0];
                    cmd = m.cmdKind;
                    fromNode = m.fromNodeIndex;
                    toNode = m.toNodeIndex;
                    headMove = m.headMove;
                    ffnMove = m.ffnMove;
                    if (cmd == 0u) {
                        if (headMove != 0u && ffnMove != 0u) cmd = PLAN_CMD_KIND_BOTH;
                        else if (headMove != 0u) cmd = PLAN_CMD_KIND_HEAD;
                        else if (ffnMove != 0u) cmd = PLAN_CMD_KIND_FFN;
                    }
                } else if (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves > 1u) {
                    cmd = 4u;
                    fromNode = pc.seq;
                } else {
                    cmd = pc.cmdKind;
                    fromNode = pc.fromNodeIndex;
                    toNode = pc.toNodeIndex;
                    headMove = pc.nHeadsToMove;
                    ffnMove = pc.nFfnToMove;
                }
                device->setLastPlanCmdSeqEmitted(pc.seq);
                std::printf("🧭 [plan][emit][cuda] node=%u stage=%u layer=%u pos=%u epoch=%u cmd=%u headMove=%u ffnMove=%u from=%u to=%u seq=%u\n",
                    (unsigned)myNode,
                    (unsigned)(stage != nullptr ? stage->stageIndex : 0u),
                    (unsigned)layerIndex,
                    (unsigned)pos,
                    (unsigned)emitEpoch,
                    (unsigned)cmd,
                    (unsigned)headMove,
                    (unsigned)ffnMove,
                    (unsigned)fromNode,
                    (unsigned)toNode,
                    (unsigned)pc.seq);
                std::fflush(stdout);
            }
            for (NnUint b = 0u; b < batchSize; ++b) {
                float *row = planPipe + b * 8u;
                row[0] = (float)emitEpoch;
                row[1] = (float)cmd;
                row[2] = (float)fromNode;
                row[3] = (float)toNode;
                row[4] = (float)headMove;
                row[5] = (float)layerIndex;
                row[6] = (float)pos;
                row[7] = (float)ffnMove;
            }
            // No GPU op in the real graph reads the plan pipe, so its device-side
            // copy is dead weight: the write below also costs a stream sync. Keep
            // it only if some non-control op in this segment actually binds this
            // pipe as an input or output.
            const bool deviceVisible =
                std::find(uploadPipeIndices.begin(), uploadPipeIndices.end(), op.output.pointerIndex) != uploadPipeIndices.end() ||
                std::find(downloadPipeIndices.begin(), downloadPipeIndices.end(), op.output.pointerIndex) != downloadPipeIndices.end();
            if (deviceVisible && op.output.pointerIndex < netConfig->nPipes) {
                NnCudaBuffer *pipe = device->data.resolvePipe(op.output.pointerIndex);
                const NnSize nBytes = pipe->calcSliceSize(batchSize, netConfig->nBatches);
                pipe->write((const NnByte *)planPipe, 0u, nBytes, stream, &device->staging);
            }
            return;
        }
        case OP_PLAN_APPLY: {
            if (netExecution == nullptr || op.input.source != SRC_PIPE) return;
            const float *in0 = (const float *)netExecution->pipes[op.input.pointerIndex];
            if (in0 == nullptr) return;
            const unsigned int curEpoch = device->getPlanEpoch();
            const unsigned int msgEpoch = in0[0] >= 0.0f ? (unsigned int)in0[0] : 0u;
            const unsigned int cmd = in0[1] >= 0.0f ? (unsigned int)in0[1] : 0u;
            const unsigned int fromNode = in0[2] >= 0.0f ? (unsigned int)in0[2] : 0u;
            const unsigned int toNode = in0[3] >= 0.0f ? (unsigned int)in0[3] : 0u;
            const unsigned int headMove = in0[4] >= 0.0f ? (unsigned int)in0[4] : 0u;
            const unsigned int layerIndex = in0[5] >= 0.0f ? (unsigned int)in0[5] : 0u;
            const unsigned int pos = in0[6] >= 0.0f ? (unsigned int)in0[6] : 0u;
            const unsigned int ffnMove = in0[7] >= 0.0f ? (unsigned int)in0[7] : 0u;
            if (!(cmd == 1u || cmd == 2u || cmd == 3u || cmd == 4u) || msgEpoch <= curEpoch) return;
            NnUnevenPartitionPlan *plan = const_cast<NnUnevenPartitionPlan *>(device->getPartitionPlan());
            if (plan == nullptr || plan->nStages == 0u) return;
            const NnPlanApplyOpCodeConfig *config = (const NnPlanApplyOpCodeConfig *)op.config;
            const NnUint onlyStage = config != nullptr ? config->onlyStageIndex : 0u;
            if (onlyStage >= plan->nStages) return;
            const NnStageConfig *stage = findStageForNodeCuda(plan, device->getNodeIndex());
            if (stage == nullptr || stage->stageIndex != onlyStage) return;

            std::vector<int> deltaHeadOrKv(plan->nNodes, 0);
            std::vector<int> deltaFfn(plan->nNodes, 0);
            bool reject = false;
            auto addMove = [&](NnUint f, NnUint t, NnUint h, NnUint ffn) {
                if (!cudaStageContains(stage, f) || !cudaStageContains(stage, t) || f == t || !cudaStageAdjacent(stage, f, t)) {
                    reject = true;
                    return;
                }
                if (h != 0u) { deltaHeadOrKv[f] -= (int)h; deltaHeadOrKv[t] += (int)h; }
                if (ffn != 0u) { deltaFfn[f] -= (int)ffn; deltaFfn[t] += (int)ffn; }
            };
            if (cmd == 4u) {
                const PlanCommandSnapshot snap = planCommandCache().load();
                const PlanCommand &pc = snap.cmd;
                if (!isValidPlanCommandHeader(pc) || pc.seq != fromNode || !planCommandHasMoveList(pc)) {
                    std::printf("🧭 [plan][apply][cuda] node=%u stage=%u epoch=%u reject: cmdlist missing/mismatch\n",
                        (unsigned)device->getNodeIndex(), (unsigned)onlyStage, (unsigned)msgEpoch);
                    std::fflush(stdout);
                    return;
                }
                const uint32_t maxMovesAllowed = std::min<uint32_t>(DLLAMA_PLAN_CMD_MAX_MOVES, (uint32_t)(2u * stage->nNodes));
                if (pc.nMoves > maxMovesAllowed) return;
                for (uint32_t i = 0u; i < pc.nMoves; ++i) {
                    const PlanMove &m = pc.moves[i];
                    addMove(m.fromNodeIndex, m.toNodeIndex, m.headMove, m.ffnMove);
                    if (reject) break;
                }
            } else {
                addMove(fromNode, toNode, (cmd == 1u || cmd == 3u) ? headMove : 0u, (cmd == 2u || cmd == 3u) ? ffnMove : 0u);
            }
            if (reject) {
                std::printf("🧭 [plan][apply][cuda] node=%u stage=%u epoch=%u layer=%u pos=%u reject: bad move\n",
                    (unsigned)device->getNodeIndex(), (unsigned)onlyStage, (unsigned)msgEpoch,
                    (unsigned)layerIndex, (unsigned)pos);
                std::fflush(stdout);
                return;
            }
            cudaApplyPlanDeltas(device, plan, stage, deltaHeadOrKv, deltaFfn, msgEpoch, layerIndex, pos);
            return;
        }
        default:
            throw std::runtime_error(unsupportedOpMessage(opIndex));
    }
}

void NnCudaDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) {
    (void)nThreads;
    if (threadIndex != 0u) return;
    if (opIndex != 0u) return;
    if (segmentConfig == nullptr) {
        throw std::runtime_error("CUDA backend received null segment config for segment " + std::to_string(segmentIndex));
    }
    unsigned int deviceEpoch = 0u;
    if (device != nullptr) {
        deviceEpoch = device->getPlanEpoch();
        if (planEpochReady.load(std::memory_order_acquire) != deviceEpoch) {
            refreshPointers();
            deviceEpoch = device->getPlanEpoch();
            planEpochReady.store(deviceEpoch, std::memory_order_release);
        }
    }
    // Bindings are resolved from the op configs refreshPointers() just rewrote,
    // so they are rebuilt whenever the plan epoch moved. `bindingsValid` covers
    // the first forward, where readyEpoch == deviceEpoch == 0 and nothing moved.
    if (!bindingsValid || bindingsEpoch != deviceEpoch) {
        rebuildBindings();
    }

    uploadSegmentInputs(batchSize);
    for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
        executeOp(i, batchSize);
        // OP_PLAN_APPLY mutates the plan (and bumps the device epoch) from the
        // host in the middle of a segment, which would leave every later op with
        // stale cached slices. Re-bind before continuing. Control ops never touch
        // opBindings themselves, so a barrier/apply-only segment can skip this
        // and let the next real segment pick the new epoch up. Segments with no
        // control op skip the check entirely and keep the hot loop free of reads.
        if (hasControlOps && !isHostOnlyPlanSegment) {
            const unsigned int epochNow = device != nullptr ? device->getPlanEpoch() : 0u;
            if (epochNow != bindingsEpoch) {
                if (device != nullptr && planEpochReady.load(std::memory_order_acquire) != epochNow) {
                    // Rewrites the op configs and rebuilds the bindings.
                    refreshPointers();
                } else {
                    rebuildBindings();
                }
            }
        }
    }
    const bool synced = downloadSegmentOutputs(batchSize);
    if (!synced && !isHostOnlyPlanSegment && device != nullptr) {
        device->synchronize();
    }
}

void NnCudaDeviceSegment::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    if (device != nullptr) {
        if (device->getPartitionPlan() != plan) {
            device->setPartitionPlan(plan);
        }
        planEpochReady.store(device->getPlanEpoch(), std::memory_order_release);
        // The cached bindings belong to the previous plan; drop them so the next
        // forward re-resolves them against the current partition.
        bindingsValid = false;
    }
}

void NnCudaDeviceSegment::refreshPointers() {
    if (device == nullptr || segmentConfig == nullptr) return;
    NnNetConfig *netConfig = device->getNetConfig();
    NnNodeConfig *nodeConfig = device->getNodeConfig();
    const NnUnevenPartitionPlan *plan = device->getPartitionPlan();
    if (netConfig == nullptr || nodeConfig == nullptr) return;
    if (nodeConfig->partitionPlan != plan) nodeConfig->partitionPlan = plan;
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
            default:
                return false;
        }
    };

    auto findRmsNormColSize = [&](NnUint fromOpIndex, const NnSize3D &inputSizeNow) -> NnUint {
        for (NnUint j = fromOpIndex + 1u; j < segmentConfig->nOps && j < fromOpIndex + 8u; ++j) {
            NnOpConfig *c = &segmentConfig->ops[j];
            if (c->code != OP_RMS_NORM) continue;
            if (c->weightSize.x == 0u) continue;
            if (inputSizeNow.x % c->weightSize.x != 0u) continue;
            return c->weightSize.x;
        }
        return 0u;
    };

    for (NnUint opIndex = 0u; opIndex < segmentConfig->nOps; ++opIndex) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        if (isCudaControlOp(opConfig->code)) continue;

        NnPointerLayout inLayout = resolvePointerLayout(netConfig, nodeConfig, plan, &opConfig->input);
        NnPointerLayout outLayout = resolvePointerLayout(netConfig, nodeConfig, plan, &opConfig->output);
        NnSize3D inputSize = inLayout.logicalSize;
        NnSize3D outputSize = outLayout.logicalSize;
        if (opConfig->code == OP_CAST &&
            opConfig->output.type == PNTR_BATCHED_SLICE &&
            inputSize.x != outputSize.x) {
            outputSize = size3D(outputSize.floatType, outputSize.z, outputSize.y, inputSize.x);
        }

        if (plan != nullptr && opConfig->config != nullptr) {
            if (opConfig->code == OP_MULTIHEAD_ATT) {
                auto *cfg = (NnMultiHeadAttOpConfig *)opConfig->config;
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
            } else if (opConfig->code == OP_SHIFT) {
                auto *cfg = (NnShiftOpCodeConfig *)opConfig->config;
                if (cfg->dstRowStride != 0u && cfg->dstColStartUnit != 0u &&
                    plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
                    cfg->dstColStart = plan->kvHeadSplit.starts[nodeIndex] * cfg->dstColStartUnit;
                }
            } else if (opConfig->code == OP_MATMUL) {
                auto *cfg = (NnMatmulOpConfig *)opConfig->config;
                if (cfg->view == 1u && cfg->outSliceTag != NN_SLICE_AUTO && cfg->outStartUnit != 0u) {
                    NnUint st = 0u;
                    if (getSplitStart(cfg->outSliceTag, &st)) {
                        cfg->outStart = st * cfg->outStartUnit;
                        if (cfg->outResidentStart != 0u && cfg->outStart >= cfg->outResidentStart) {
                            cfg->outStart -= cfg->outResidentStart;
                        }
                    }
                }
                if (cfg->view == 2u && cfg->inSliceTag != NN_SLICE_AUTO && cfg->inStartUnit != 0u) {
                    NnUint st = 0u;
                    if (getSplitStart(cfg->inSliceTag, &st)) {
                        cfg->inStart = st * cfg->inStartUnit;
                        if (cfg->inResidentStart != 0u && cfg->inStart >= cfg->inResidentStart) {
                            cfg->inStart -= cfg->inResidentStart;
                        }
                    }
                }
                if (cfg->aView.sizeX != 0u) cfg->aView.sizeX = inputSize.x;
                if (cfg->cView.sizeX != 0u) cfg->cView.sizeX = outputSize.x;
            } else if (opConfig->code == OP_MUL) {
                auto *cfg = (NnMulOpCodeConfig *)opConfig->config;
                if (plan->ffnSplit.starts && plan->ffnSplit.lengths && cfg->view.sizeX != 0u) {
                    cfg->view.offset = plan->ffnSplit.starts[nodeIndex];
                    cfg->view.sizeX = plan->ffnSplit.lengths[nodeIndex];
                    cfg->view.strideX = 1u;
                }
            } else if (opConfig->code == OP_INV_RMS) {
                auto *cfg = (NnInvRmsOpConfig *)opConfig->config;
                const NnUint colSize = findRmsNormColSize(opIndex, inputSize);
                if (colSize != 0u && inputSize.x % colSize == 0u) {
                    const NnUint newCols = inputSize.x / colSize;
                    if (outputSize.x >= newCols && newCols != 0u) cfg->nColumns = newCols;
                }
            } else if (opConfig->code == OP_RMS_NORM) {
                auto *cfg = (NnRmsNormOpConfig *)opConfig->config;
                const NnUint colSize = opConfig->weightSize.x;
                if (colSize != 0u && inputSize.x % colSize == 0u) {
                    const NnUint newCols = inputSize.x / colSize;
                    if (newCols != 0u) cfg->nColumns = newCols;
                }
            } else if (opConfig->code == OP_ROPE) {
                auto *cfg = (NnRopeOpConfig *)opConfig->config;
                if (cfg->isQ == 1u) cfg->slice.qDim0 = inputSize.x;
                else cfg->slice.kvDim0 = inputSize.x;
            }
        }

        if (opConfig->config != nullptr && opConfig->configSize > 0u && opIndex < configBuffers.size()) {
            configBuffers[opIndex]->write(opConfig->config, 0u, opConfig->configSize, device->getStream(), &device->staging);
        }
    }

    device->synchronize();
    planEpochReady.store(device->getPlanEpoch(), std::memory_order_release);
    // This is the one place where op configs (slices, offsets, column counts)
    // are rewritten, so it is also the point at which the cached per-op
    // bindings become stale.
    rebuildBindings();
}

bool NnCudaDeviceSegment::exportLayerKvRow(
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
    if (device == nullptr || segmentConfig == nullptr || kvDim == 0u || outDim == 0u) return false;
    if (partial && rangeStart + rangeLen > kvDim) return false;
    NnNodeConfig *nodeConfig = device->getNodeConfig();
    if (nodeConfig == nullptr) return false;

    bool readAny = false;
    for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
        const NnOpConfig &op = segmentConfig->ops[i];
        if (op.code != OP_MULTIHEAD_ATT || op.index != layerIndex) continue;
        const auto *cfg = (const NnMultiHeadAttOpConfig *)op.config;
        if (cfg == nullptr) continue;

        auto readOne = [&](NnUint bufIdx, std::vector<float> &dstRow, const char *tag) {
            if (bufIdx >= nodeConfig->nBuffers) return;
            const NnSize3D &bSize = nodeConfig->buffers[bufIdx].size;
            if (bSize.floatType != F_32 || bSize.x == 0u || bSize.y == 0u) return;
            if (position >= bSize.y) return;
            const NnUint srcStart = partial ? rangeStart : cfg->kvStart;
            const NnUint dstStart = partial ? 0u : cfg->kvStart;
            const NnUint needLen = partial ? rangeLen : cfg->kvDim0;
            if (srcStart >= bSize.x || dstStart >= dstRow.size() || needLen == 0u) return;
            const NnUint readLen = std::min(needLen, std::min(bSize.x - srcStart, (NnUint)dstRow.size() - dstStart));
            if (readLen == 0u) return;
            const NnSize offset = ((NnSize)position * (NnSize)bSize.x + (NnSize)srcStart) * sizeof(float);
            device->readBuffer(bufIdx, (NnByte *)(dstRow.data() + dstStart), offset, (NnSize)readLen * sizeof(float));
            readAny = true;
            std::printf("🧩 [kv-export-cuda] node=%u seg=%u layer=%u pos=%u %sBuf=%u srcRange=[%u,%u) dstRange=[%u,%u) partial=%u\n",
                (unsigned)device->getNodeIndex(),
                (unsigned)segmentIndex,
                (unsigned)layerIndex,
                (unsigned)position,
                tag,
                (unsigned)bufIdx,
                (unsigned)srcStart,
                (unsigned)(srcStart + readLen),
                (unsigned)dstStart,
                (unsigned)(dstStart + readLen),
                partial ? 1u : 0u);
            std::fflush(stdout);
        };

        readOne(cfg->keyCacheBufferIndex, kRow, "k");
        readOne(cfg->valueCacheBufferIndex, vRow, "v");
    }
    return readAny;
}

bool NnCudaDeviceSegment::applyTransferredKvRow(
    NnUint layerIndex,
    NnUint position,
    const std::vector<float> &kRow,
    const std::vector<float> &vRow,
    NnUint rangeStart,
    NnUint rangeLen) {
    if (device == nullptr || segmentConfig == nullptr) return false;
    if (kRow.empty() || vRow.empty() || kRow.size() != vRow.size()) return false;
    if (rangeLen != 0u && rangeLen != kRow.size()) return false;
    NnNodeConfig *nodeConfig = device->getNodeConfig();
    if (nodeConfig == nullptr) return false;

    bool wroteAny = false;
    for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
        const NnOpConfig &op = segmentConfig->ops[i];
        if (op.code != OP_MULTIHEAD_ATT || op.index != layerIndex) continue;
        const auto *cfg = (const NnMultiHeadAttOpConfig *)op.config;
        if (cfg == nullptr) continue;

        auto writeOne = [&](NnUint bufIdx, const std::vector<float> &srcRow, const char *tag) {
            if (bufIdx >= nodeConfig->nBuffers) return;
            const NnSize3D &bSize = nodeConfig->buffers[bufIdx].size;
            if (bSize.floatType != F_32 || bSize.x == 0u || bSize.y == 0u) return;
            if (position >= bSize.y) return;
            const bool isPartial = rangeLen != 0u;
            const NnUint dstStart = isPartial ? rangeStart : cfg->kvStart;
            const NnUint srcStart = isPartial ? 0u : cfg->kvStart;
            const NnUint needLen = isPartial ? rangeLen : cfg->kvDim0;
            if (srcStart >= srcRow.size() || dstStart >= bSize.x || needLen == 0u) return;
            const NnUint writeLen = std::min(needLen, std::min((NnUint)srcRow.size() - srcStart, bSize.x - dstStart));
            if (writeLen == 0u) return;
            const NnSize offset = ((NnSize)position * (NnSize)bSize.x + (NnSize)dstStart) * sizeof(float);
            device->writeBuffer(bufIdx, (const NnByte *)(srcRow.data() + srcStart), offset, (NnSize)writeLen * sizeof(float));
            wroteAny = true;
            if (std::getenv("DLLAMA_MIGRATION_VERBOSE_LOG") != nullptr && std::strcmp(std::getenv("DLLAMA_MIGRATION_VERBOSE_LOG"), "0") != 0) {
                std::printf("🧩 [kv-write-cuda] node=%u seg=%u layer=%u pos=%u %sBuf=%u srcRange=[%u,%u) dstRange=[%u,%u) partial=%u\n",
                    (unsigned)device->getNodeIndex(),
                    (unsigned)segmentIndex,
                    (unsigned)layerIndex,
                    (unsigned)position,
                    tag,
                    (unsigned)bufIdx,
                    (unsigned)srcStart,
                    (unsigned)(srcStart + writeLen),
                    (unsigned)dstStart,
                    (unsigned)(dstStart + writeLen),
                    isPartial ? 1u : 0u);
                std::fflush(stdout);
            }
        };

        writeOne(cfg->keyCacheBufferIndex, kRow, "k");
        writeOne(cfg->valueCacheBufferIndex, vRow, "v");
    }
    if (wroteAny) device->synchronize();
    return wroteAny;
}
