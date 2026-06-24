#include "nn-cuda.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <sstream>
#include <stdexcept>

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

NnCudaDevice::NnCudaDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan)
    : gpuIndex(gpuIndex), stream(nullptr), netConfig(netConfig), nodeConfig(nodeConfig), netExecution(netExecution), partitionPlan(partitionPlan) {
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
}

NnCudaDevice::~NnCudaDevice() {
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

NnCudaDeviceSegment::NnCudaDeviceSegment(NnCudaDevice *device, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution)
    : device(device), segmentIndex(segmentIndex), segmentConfig(segmentConfig), netExecution(netExecution) {
    (void)this->netExecution;
    if (device != nullptr) {
        planEpochReady.store(device->getPlanEpoch(), std::memory_order_release);
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
    (void)offset;
    (void)weight;
    if (nBytes == 0u) return;
    throw std::runtime_error(unsupportedOpMessage(opIndex) + " while loading " + std::to_string((unsigned long long)nBytes) + " weight bytes");
}

void NnCudaDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) {
    (void)nThreads;
    (void)threadIndex;
    (void)batchSize;
    if (segmentConfig == nullptr || opIndex >= segmentConfig->nOps) {
        throw std::runtime_error("CUDA backend received invalid opIndex " + std::to_string(opIndex) + " for segment " + std::to_string(segmentIndex));
    }
    throw std::runtime_error(unsupportedOpMessage(opIndex));
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
