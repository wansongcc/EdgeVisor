#ifndef NN_CUDA_HPP
#define NN_CUDA_HPP

#include <atomic>
#include <string>
#include "nn-executor.hpp"

int nnCudaDeviceCount();
std::string nnCudaDeviceInfo(NnUint gpuIndex);
void nnCudaPrintDeviceInfo(NnUint gpuIndex);

class NnCudaDevice : public NnDevice {
private:
    NnUint gpuIndex;
    void *stream;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    NnNetExecution *netExecution;
    const NnUnevenPartitionPlan *partitionPlan;
    std::atomic_uint planEpoch{0u};
public:
    NnCudaDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan = nullptr);
    ~NnCudaDevice() override;
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
    void setPartitionPlan(const NnUnevenPartitionPlan *plan);
    const NnUnevenPartitionPlan *getPartitionPlan() const { return partitionPlan; }
    unsigned int getPlanEpoch() const { return planEpoch.load(std::memory_order_acquire); }
    void setPlanEpoch(unsigned int e) { planEpoch.store(e, std::memory_order_release); }
    NnUint getNodeIndex() const { return nodeConfig ? nodeConfig->nodeIndex : 0u; }
    NnUint getGpuIndex() const { return gpuIndex; }
    void *getStream() const { return stream; }
    NnSize3D resolvePointerLogicalSize(const NnPointerConfig *config) const;
};

class NnCudaDeviceSegment : public NnDeviceSegment {
private:
    NnCudaDevice *device;
    NnUint segmentIndex;
    NnSegmentConfig *segmentConfig;
    NnNetExecution *netExecution;
    std::atomic_uint planEpochReady{0u};

    std::string unsupportedOpMessage(NnUint opIndex) const;
public:
    NnCudaDeviceSegment(NnCudaDevice *device, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution);
    ~NnCudaDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
    void setPartitionPlan(const NnUnevenPartitionPlan *plan) override;
    void refreshPointers() override;
};

#endif
