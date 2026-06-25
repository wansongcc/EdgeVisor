#include "nn-cuda.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>
#include <cmath>
#include <cstdio>
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

__device__ static inline NnByte *cudaRowBase(NnByte *base, NnPointerLayout layout, NnUint z, NnUint y) {
    return base + z * layout.zStrideBytes + y * layout.batchStrideBytes + layout.byteOffset;
}

static inline NnByte *cudaHostRowBase(NnByte *base, const NnPointerLayout &layout, NnUint z, NnUint y) {
    return base + z * layout.zStrideBytes + y * layout.batchStrideBytes + layout.byteOffset;
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
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalBlocks) return;
    const NnUint b = (NnUint)(tid % blocksPerRow);
    NnSize t = tid / blocksPerRow;
    const NnUint y = (NnUint)(t % batchSize);
    const NnUint z = (NnUint)(t / batchSize);
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, z, y) + offset + b * Q80_BLOCK_SIZE;
    NnBlockQ80 *out = (NnBlockQ80 *)cudaRowBase(outputBase, output, z, y) + (offset / Q80_BLOCK_SIZE) + b;
    float amax = 0.0f;
    for (NnUint j = 0u; j < Q80_BLOCK_SIZE; ++j) {
        const float av = fabsf(in[j]);
        amax = amax > av ? amax : av;
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    out->d = cudaFloatToHalfBits(d);
    for (NnUint j = 0u; j < Q80_BLOCK_SIZE; ++j) {
        int q = (int)nearbyintf(in[j] * id);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out->qs[j] = (int8_t)q;
    }
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
    const NnSize total = (NnSize)batchSize * nColumns;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint col = (NnUint)(tid % nColumns);
    const NnUint y = (NnUint)(tid / nColumns);
    const NnUint colSize = input.logicalSize.x / nColumns;
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, 0u, y) + (NnSize)col * colSize;
    float ss = 0.0f;
    for (NnUint i = 0u; i < colSize; ++i) ss += in[i] * in[i];
    float *out = (float *)cudaRowBase(outputBase, output, 0u, y);
    out[col] = rsqrtf(ss / (float)colSize + epsilon);
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
    NnShiftOpCodeConfig config,
    NnUint batchSize) {
    const NnSize total = (NnSize)batchSize * input.logicalSize.x;
    const NnSize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const NnUint x = (NnUint)(tid % input.logicalSize.x);
    const NnUint y = (NnUint)(tid / input.logicalSize.x);
    const NnUint row = (NnUint)indexes[y];
    const float *in = (const float *)cudaRowBase((NnByte *)inputBase, input, 0u, y);
    float *outBase = (float *)(outputBase + output.byteOffset);
    const NnSize dst = config.dstRowStride == 0u
        ? (NnSize)row * input.logicalSize.x + x
        : (NnSize)row * config.dstRowStride + config.dstColStart + x;
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

__global__ static void multiheadAttScoreF32Kernel(
    const float *positions,
    const float *query,
    const float *keyCache,
    float *att,
    NnMultiHeadAttOpConfig config,
    NnUint batchSize) {
    const NnUint h = blockIdx.x;
    const NnUint y = blockIdx.y;
    if (h >= config.nHeads0 || y >= batchSize) return;
    const NnUint pos = (NnUint)positions[y];
    NnUint qOffset = 0u;
    NnUint kvOffset = 0u;
    NnUint attOffset = 0u;
    cudaMhaOffsets(config, h, y, &qOffset, &kvOffset, &attOffset);
    const float invHeadDimRoot = rsqrtf((float)config.headDim);
    for (NnUint p = threadIdx.x; p <= pos; p += blockDim.x) {
        float score = 0.0f;
        const NnUint kOffset = p * config.kvStride + kvOffset;
        for (NnUint i = 0u; i < config.headDim; ++i) {
            score += query[qOffset + i] * keyCache[kOffset + i];
        }
        att[attOffset + p] = score * invHeadDimRoot;
    }
}

__global__ static void multiheadAttSoftmaxSerialF32Kernel(
    const float *positions,
    float *att,
    NnMultiHeadAttOpConfig config,
    NnUint batchSize) {
    const NnUint h = blockIdx.x;
    const NnUint y = blockIdx.y;
    if (h >= config.nHeads0 || y >= batchSize) return;
    const NnUint pos = (NnUint)positions[y];
    const NnUint attOffset = y * config.nHeads0 * config.seqLen + h * config.seqLen;
    float maxScore = -3.4028234663852886e38f;
    for (NnUint p = 0u; p <= pos; ++p) {
        const float v = att[attOffset + p];
        maxScore = maxScore > v ? maxScore : v;
    }
    float sum = 0.0f;
    for (NnUint p = 0u; p <= pos; ++p) {
        const float v = expf(att[attOffset + p] - maxScore);
        att[attOffset + p] = v;
        sum += v;
    }
    const float invSum = 1.0f / sum;
    for (NnUint p = 0u; p <= pos; ++p) {
        att[attOffset + p] *= invSum;
    }
}

__global__ static void multiheadAttValueF32Kernel(
    NnByte *outputBase,
    NnPointerLayout output,
    const float *positions,
    const float *valueCache,
    const float *att,
    NnMultiHeadAttOpConfig config,
    NnUint batchSize) {
    const NnUint h = blockIdx.x;
    const NnUint y = blockIdx.y;
    if (h >= config.nHeads0 || y >= batchSize) return;
    const NnUint pos = (NnUint)positions[y];
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
            acc += att[attOffset + p] * valueCache[p * config.kvStride + vOffset];
        }
        out[i] = acc;
    }
}

__device__ static inline float q80q40BlockDot(const NnBlockQ80 *xb, const NnBlockQ40 *wb) {
    int acc = 0;
    for (NnUint k = 0u; k < Q40_BLOCK_SIZE / 2u; ++k) {
        const int w0 = (int)(wb->qs[k] & 0x0fu) - 8;
        const int w1 = (int)(wb->qs[k] >> 4) - 8;
        const int x0 = (int)xb->qs[k];
        const int x1 = (int)xb->qs[k + Q80_BLOCK_SIZE / 2u];
        acc += w0 * x0 + w1 * x1;
    }
    return (float)acc * cudaHalfBitsToFloat(xb->d) * cudaHalfBitsToFloat(wb->d);
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
    __shared__ float partial[256];
    const NnSize oid = blockIdx.x;
    const NnUint outLocal = (NnUint)(oid % cLen);
    NnSize t = oid / cLen;
    const NnUint e = (NnUint)(t % nActiveExpertsOr1);
    const NnUint y = (NnUint)(t / nActiveExpertsOr1);
    if (y >= batchSize) return;
    const NnUint activeExpert = nActiveExperts == 0u ? 0u : (NnUint)activeExpertIndexes[(NnSize)y * activeExpertRowStride + e];
    float sum = 0.0f;
    if (activeExpert < weightExperts) {
        const NnBlockQ80 *x = (const NnBlockQ80 *)cudaRowBase((NnByte *)inputBase, input, e, y) + aOffsetBlocks;
        const NnUint row = view == 0u ? outLocal : outStart + outLocal;
        const NnUint inBlock = view == 2u ? inStartBlocks : 0u;
        const NnBlockQ40 *w = weight + ((NnSize)activeExpert * weightRows + row) * weightBlocksPerRow + inBlock;
        for (NnUint b = threadIdx.x; b < aBlocks; b += blockDim.x) {
            sum += q80q40BlockDot(x + b, w + b);
        }
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (NnUint stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u && activeExpert < weightExperts) {
        float *out = (float *)cudaRowBase(outputBase, output, e, y) + cOffset;
        out[outLocal] = partial[0];
    }
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
        << ", globalMem=" << (unsigned long long)(prop.totalGlobalMem / (1024ull * 1024ull)) << " MB"
        << ", multiProcessorCount=" << prop.multiProcessorCount
        << ", maxThreadsPerBlock=" << prop.maxThreadsPerBlock
        << ", driver=" << driverVersion
        << ", runtime=" << runtimeVersion;
    return out.str();
}

void nnCudaPrintDeviceInfo(NnUint gpuIndex) {
    std::printf("🔷 %s\n", nnCudaDeviceInfo(gpuIndex).c_str());
    std::fflush(stdout);
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
    if (staging == nullptr) {
        NN_CUDA_CHECK(cudaMemcpyAsync((NnByte *)devicePointer + offset, data, nBytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
        return;
    }
    void *host = staging->ensure(nBytes);
    std::memcpy(host, data, nBytes);
    NN_CUDA_CHECK(cudaMemcpyAsync((NnByte *)devicePointer + offset, host, nBytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
    NN_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
}

void NnCudaBuffer::read(NnByte *data, NnSize offset, NnSize nBytes, void *stream, NnCudaPinnedStaging *staging) {
    if (nBytes == 0u) return;
    if (data == nullptr) throw std::runtime_error("CUDA read received null host pointer for buffer " + name);
    validateRange("read", name, bufferSize, offset, nBytes);
    if (staging == nullptr) {
        NN_CUDA_CHECK(cudaMemcpyAsync(data, (NnByte *)devicePointer + offset, nBytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
        NN_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        return;
    }
    void *host = staging->ensure(nBytes);
    NN_CUDA_CHECK(cudaMemcpyAsync(host, (NnByte *)devicePointer + offset, nBytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
    NN_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
    std::memcpy(data, host, nBytes);
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
    : gpuIndex(gpuIndex), stream(nullptr), blasHandle(nullptr), netConfig(netConfig), nodeConfig(nodeConfig), netExecution(netExecution), partitionPlan(partitionPlan), staging(), data() {
    (void)this->netConfig;
    (void)this->netExecution;
    nnCudaPrintDeviceInfo(gpuIndex);
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
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
    return 1u;
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
    : device(device), segmentIndex(segmentIndex), segmentConfig(segmentConfig), netExecution(netExecution) {
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

    std::vector<unsigned char> uploaded(netConfig->nPipes, 0u);
    const auto uploadPipe = [&](NnUint pipeIndex) {
        if (pipeIndex >= netConfig->nPipes) {
            throw std::runtime_error("CUDA segment pipe index out of range during upload");
        }
        if (uploaded[pipeIndex] != 0u) return;
        NnCudaBuffer *pipe = device->data.resolvePipe(pipeIndex);
        const NnSize nBytes = pipe->calcSliceSize(batchSize, netConfig->nBatches);
        pipe->write(netExecution->pipes[pipeIndex], 0u, nBytes, device->getStream(), &device->staging);
        uploaded[pipeIndex] = 1u;
    };

    for (NnUint i = 0u; i < netConfig->nPreSyncs; ++i) {
        uploadPipe(netConfig->preSyncs[i].pipeIndex);
    }

    for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
        const NnOpConfig &op = segmentConfig->ops[i];
        if (op.input.source == SRC_PIPE) uploadPipe(op.input.pointerIndex);
        if (op.code == OP_ROPE && op.config != nullptr) {
            const NnRopeOpConfig *config = (const NnRopeOpConfig *)op.config;
            uploadPipe(config->positionPipeIndex);
        } else if (op.code == OP_MULTIHEAD_ATT && op.config != nullptr) {
            const NnMultiHeadAttOpConfig *config = (const NnMultiHeadAttOpConfig *)op.config;
            uploadPipe(config->positionPipeIndex);
        } else if (op.code == OP_SHIFT && op.config != nullptr) {
            const NnShiftOpCodeConfig *config = (const NnShiftOpCodeConfig *)op.config;
            uploadPipe(config->indexPipeIndex);
        }
    }
}

void NnCudaDeviceSegment::downloadSegmentOutputs(NnUint batchSize) {
    if (device == nullptr || segmentConfig == nullptr || netExecution == nullptr) return;
    NnNetConfig *netConfig = device->getNetConfig();
    if (netConfig == nullptr) return;
    if (batchSize > netConfig->nBatches) {
        throw std::runtime_error("CUDA segment batchSize exceeds configured nBatches");
    }

    for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
        const NnOpConfig &op = segmentConfig->ops[i];
        if (op.output.source != SRC_PIPE) continue;
        NnCudaBuffer *pipe = device->data.resolvePipe(op.output.pointerIndex);
        const NnSize nBytes = pipe->calcSliceSize(batchSize, netConfig->nBatches);
        pipe->read(netExecution->pipes[op.output.pointerIndex], 0u, nBytes, device->getStream(), &device->staging);
    }
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
    NnCudaPointerRef in = resolveCudaPointer(device, netConfig, nodeConfig, device->getPartitionPlan(), &op.input);
    NnCudaPointerRef out = resolveCudaPointer(device, netConfig, nodeConfig, device->getPartitionPlan(), &op.output);
    cudaStream_t stream = (cudaStream_t)device->getStream();

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
                embeddingF32F32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                embeddingQ40F32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                embeddingQ80F32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                invRmsF32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                rmsNormF32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                if (aBlocks <= 4u) {
                    matmulQ80Q40SmallKKernel<<<cudaBlocks(totalOutputs), 256, 0, stream>>>(
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
                    matmulQ80Q40LargeKKernel<<<(unsigned int)totalOutputs, 256, 0, stream>>>(
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
                ropeF32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                multiheadAttScoreF32Kernel<<<grid, 256, 0, stream>>>(
                    (const float *)positionPipe->data(),
                    (const float *)queryBuffer->data(),
                    (const float *)keyCacheBuffer->data(),
                    (float *)attBuffer->data(),
                    config,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
                multiheadAttSoftmaxSerialF32Kernel<<<grid, 1, 0, stream>>>(
                    (const float *)positionPipe->data(),
                    (float *)attBuffer->data(),
                    config,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
                multiheadAttValueF32Kernel<<<grid, 256, 0, stream>>>(
                    (NnByte *)out.buffer->data(), out.layout,
                    (const float *)positionPipe->data(),
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
                    castF32F32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                    q80QuantizeKernel<<<cudaBlocks(totalBlocks), 256, 0, stream>>>(
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
                siluKernel<<<cudaBlocks(total), 256, 0, stream>>>(
                    (NnByte *)out.buffer->data(), out.layout, view, batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
        case OP_GELU: {
            const NnTensorViewLayout view = resolveOpView(op, out.layout.logicalSize, out.layout.logicalSize.x);
            const NnSize total = (NnSize)out.layout.logicalSize.z * batchSize * view.sizeY * view.sizeX;
            if (total != 0u) {
                geluKernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                mulKernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                scaleKernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                mergeAddF32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                mergeAddQ80F32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                mergeSumF32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
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
                q80QuantizeKernel<<<cudaBlocks(quantBlocks), 256, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    0u, out.layout.logicalSize.x,
                    batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            const NnSize copyBlocks = (NnSize)(out.layout.logicalSize.z > 0u ? out.layout.logicalSize.z - 1u : 0u)
                * batchSize * (out.layout.logicalSize.x / Q80_BLOCK_SIZE);
            if (copyBlocks != 0u) {
                repeatZCopyQ80Kernel<<<cudaBlocks(copyBlocks), 256, 0, stream>>>(
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
            const NnSize total = (NnSize)batchSize * in.layout.logicalSize.x;
            if (total != 0u) {
                shiftF32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    (const float *)indexPipe->data(),
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
                    softmaxF32Kernel<<<rows, 256, 0, stream>>>(
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
                moeGateF32Kernel<<<cudaBlocks(batchSize), 256, 0, stream>>>(
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
    if (device != nullptr) {
        const unsigned int deviceEpoch = device->getPlanEpoch();
        const unsigned int readyEpoch = planEpochReady.load(std::memory_order_acquire);
        if (readyEpoch != deviceEpoch) {
            refreshPointers();
            planEpochReady.store(deviceEpoch, std::memory_order_release);
        }
    }

    uploadSegmentInputs(batchSize);
    for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
        executeOp(i, batchSize);
    }
    downloadSegmentOutputs(batchSize);
    if (device != nullptr) {
        device->synchronize();
    }
}

void NnCudaDeviceSegment::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    if (device != nullptr) {
        device->setPartitionPlan(plan);
        planEpochReady.store(device->getPlanEpoch(), std::memory_order_release);
    }
}

void NnCudaDeviceSegment::refreshPointers() {
    if (device != nullptr) {
        planEpochReady.store(device->getPlanEpoch(), std::memory_order_release);
    }
}
