#include "nn-cuda.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <utility>

static std::string cudaErrorString(cudaError_t err, const char *expr) {
    std::ostringstream out;
    out << expr << " failed: " << cudaGetErrorString(err) << " (" << (int)err << ")";
    return out.str();
}

#define NN_CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) throw std::runtime_error(cudaErrorString(_err, #expr)); \
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

__device__ static inline NnByte *cudaRowBase(NnByte *base, NnPointerLayout layout, NnUint z, NnUint y) {
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
    : gpuIndex(gpuIndex), stream(nullptr), netConfig(netConfig), nodeConfig(nodeConfig), netExecution(netExecution), partitionPlan(partitionPlan), staging(), data() {
    (void)this->netConfig;
    (void)this->netExecution;
    nnCudaPrintDeviceInfo(gpuIndex);
    NN_CUDA_CHECK(cudaSetDevice((int)gpuIndex));
    cudaStream_t s = nullptr;
    NN_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    stream = (void *)s;
    if (nodeConfig != nullptr) {
        nodeConfig->partitionPlan = partitionPlan;
    }
    data = NnCudaDeviceData(netConfig, nodeConfig);
}

NnCudaDevice::~NnCudaDevice() {
    if (stream != nullptr) {
        cudaSetDevice((int)gpuIndex);
        cudaStreamSynchronize((cudaStream_t)stream);
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
            std::string weightName = std::string(op.name != nullptr ? op.name : "op") + ".weight";
            std::string configName = std::string(op.name != nullptr ? op.name : "op") + ".config";
            weightBuffers.emplace_back(new NnCudaBuffer(weightName.c_str(), op.weightSize.nBytes));
            configBuffers.emplace_back(new NnCudaBuffer(configName.c_str(), op.configSize));
            if (op.config != nullptr && op.configSize > 0u && device != nullptr) {
                configBuffers.back()->write(op.config, 0u, op.configSize, device->getStream(), &device->staging);
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

    for (NnUint i = 0u; i < netConfig->nPreSyncs; ++i) {
        const NnPreSyncConfig &pre = netConfig->preSyncs[i];
        NnCudaBuffer *pipe = device->data.resolvePipe(pre.pipeIndex);
        const NnSize nBytes = pipe->calcSliceSize(batchSize, netConfig->nBatches);
        pipe->write(netExecution->pipes[pre.pipeIndex], 0u, nBytes, device->getStream(), &device->staging);
    }

    for (NnUint i = 0u; i < segmentConfig->nOps; ++i) {
        const NnOpConfig &op = segmentConfig->ops[i];
        if (op.input.source != SRC_PIPE) continue;
        NnCudaBuffer *pipe = device->data.resolvePipe(op.input.pointerIndex);
        const NnSize nBytes = pipe->calcSliceSize(batchSize, netConfig->nBatches);
        pipe->write(netExecution->pipes[op.input.pointerIndex], 0u, nBytes, device->getStream(), &device->staging);
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
            if (in.layout.logicalSize.floatType != F_32 || out.layout.logicalSize.floatType != F_32) {
                throw std::runtime_error(unsupportedOpMessage(opIndex));
            }
            if (out.layout.logicalSize.x == 0u || (in.layout.logicalSize.x % out.layout.logicalSize.x) != 0u) {
                throw std::runtime_error("CUDA MERGE_ADD invalid input/output widths");
            }
            const NnUint nSlices = in.layout.logicalSize.x / out.layout.logicalSize.x;
            const NnSize total = (NnSize)batchSize * out.layout.logicalSize.x;
            if (total != 0u) {
                mergeAddF32Kernel<<<cudaBlocks(total), 256, 0, stream>>>(
                    (const NnByte *)in.buffer->data(), in.layout,
                    (NnByte *)out.buffer->data(), out.layout,
                    nSlices, batchSize);
                NN_CUDA_CHECK(cudaGetLastError());
            }
            return;
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
