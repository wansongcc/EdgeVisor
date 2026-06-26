#ifndef NN_CUDA_HPP
#define NN_CUDA_HPP

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include "nn-executor.hpp"

int nnCudaDeviceCount();
std::string nnCudaDeviceInfo(NnUint gpuIndex);
void nnCudaPrintDeviceInfo(NnUint gpuIndex);

struct NnCudaLaunchConfig {
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    int sm;
    NnUint multiprocessorCount;
    NnUint maxThreadsPerBlock;
    NnUint warpSize;
    bool integrated;
    NnUint elementwiseBlockSize;
    NnUint reductionBlockSize;
    NnUint attentionBlockSize;
    NnUint q80q40SmallKBlockSize;
    NnUint q80q40LargeKBlockSize;
    NnUint softmaxBlockSize;
    NnUint moeGateBlockSize;
    NnUint q80q40SmallKMaxBlocks;
};

class NnCudaPinnedStaging {
private:
    void *hostPointer;
    NnSize allocatedSize;
public:
    NnCudaPinnedStaging();
    ~NnCudaPinnedStaging();
    void *ensure(NnSize size);
    void release();
};

class NnCudaBuffer {
private:
    void *devicePointer;
public:
    std::string name;
    NnSize bufferSize;

    NnCudaBuffer(const char *name, NnSize bufferSize);
    ~NnCudaBuffer();
    NnCudaBuffer(const NnCudaBuffer&) = delete;
    NnCudaBuffer& operator=(const NnCudaBuffer&) = delete;

    void write(const NnByte *data, NnSize offset, NnSize nBytes, void *stream, NnCudaPinnedStaging *staging);
    void read(NnByte *data, NnSize offset, NnSize nBytes, void *stream, NnCudaPinnedStaging *staging);
    void clear(void *stream);
    NnSize calcSliceSize(NnSize nominator, NnSize denominator) const;
    void *data() const { return devicePointer; }
};

class NnCudaDeviceData {
private:
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
public:
    std::vector<std::unique_ptr<NnCudaBuffer>> pipes;
    std::vector<std::unique_ptr<NnCudaBuffer>> buffers;

    NnCudaDeviceData();
    NnCudaDeviceData(NnNetConfig *netConfig, NnNodeConfig *nodeConfig);
    NnCudaDeviceData(NnCudaDeviceData&& other) noexcept;
    NnCudaDeviceData& operator=(NnCudaDeviceData&& other) noexcept;
    NnCudaDeviceData(const NnCudaDeviceData&) = delete;
    NnCudaDeviceData& operator=(const NnCudaDeviceData&) = delete;
    NnCudaBuffer *resolvePipe(NnUint pipeIndex);
    NnCudaBuffer *resolveBuffer(NnUint bufferIndex);
};

class NnCudaDevice : public NnDevice {
private:
    NnUint gpuIndex;
    void *stream;
    void *blasHandle;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    NnNetExecution *netExecution;
    const NnUnevenPartitionPlan *partitionPlan;
    std::atomic_uint planEpoch{0u};
    std::atomic_uint lastPlanCmdSeqEmitted{0u};
    NnCudaLaunchConfig launchConfig;
public:
    NnCudaPinnedStaging staging;
    NnCudaDeviceData data;
    NnCudaDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan = nullptr);
    ~NnCudaDevice() override;
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
    void setPartitionPlan(const NnUnevenPartitionPlan *plan);
    const NnUnevenPartitionPlan *getPartitionPlan() const { return partitionPlan; }
    unsigned int getPlanEpoch() const { return planEpoch.load(std::memory_order_acquire); }
    void setPlanEpoch(unsigned int e) { planEpoch.store(e, std::memory_order_release); }
    unsigned int getLastPlanCmdSeqEmitted() const { return lastPlanCmdSeqEmitted.load(std::memory_order_acquire); }
    void setLastPlanCmdSeqEmitted(unsigned int s) { lastPlanCmdSeqEmitted.store(s, std::memory_order_release); }
    NnUint getNodeIndex() const { return nodeConfig ? nodeConfig->nodeIndex : 0u; }
    NnUint getGpuIndex() const { return gpuIndex; }
    void *getStream() const { return stream; }
    void *getBlasHandle() const { return blasHandle; }
    const NnCudaLaunchConfig &getLaunchConfig() const { return launchConfig; }
    std::string launchConfigInfo() const;
    NnSize3D resolvePointerLogicalSize(const NnPointerConfig *config) const;
    NnNetConfig *getNetConfig() const { return netConfig; }
    NnNodeConfig *getNodeConfig() const { return nodeConfig; }
    NnNetExecution *getNetExecution() const { return netExecution; }

    void writePipe(NnUint pipeIndex, const NnByte *data, NnSize offset, NnSize nBytes);
    void readPipe(NnUint pipeIndex, NnByte *data, NnSize offset, NnSize nBytes);
    void writeBuffer(NnUint bufferIndex, const NnByte *data, NnSize offset, NnSize nBytes);
    void readBuffer(NnUint bufferIndex, NnByte *data, NnSize offset, NnSize nBytes);
    void synchronize();
};

class NnCudaDeviceSegment : public NnDeviceSegment {
private:
    NnCudaDevice *device;
    NnUint segmentIndex;
    NnSegmentConfig *segmentConfig;
    NnNetExecution *netExecution;
    std::atomic_uint planEpochReady{0u};
    std::vector<std::unique_ptr<NnCudaBuffer>> weightBuffers;
    std::vector<std::unique_ptr<NnCudaBuffer>> configBuffers;

    std::string unsupportedOpMessage(NnUint opIndex) const;
    void uploadSegmentInputs(NnUint batchSize);
    void downloadSegmentOutputs(NnUint batchSize);
    void executeOp(NnUint opIndex, NnUint batchSize);
public:
    NnCudaDeviceSegment(NnCudaDevice *device, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution);
    ~NnCudaDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
    void setPartitionPlan(const NnUnevenPartitionPlan *plan) override;
    void refreshPointers() override;
    bool exportLayerKvRow(
        NnUint layerIndex,
        NnUint position,
        NnUint kvDim,
        std::vector<float> &kRow,
        std::vector<float> &vRow,
        NnUint rangeStart = 0u,
        NnUint rangeLen = 0u) override;
    bool applyTransferredKvRow(
        NnUint layerIndex,
        NnUint position,
        const std::vector<float> &kRow,
        const std::vector<float> &vRow,
        NnUint rangeStart = 0u,
        NnUint rangeLen = 0u) override;
    void readWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *out);
};

#endif
