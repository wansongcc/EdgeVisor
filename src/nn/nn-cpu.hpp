#ifndef NN_CPU_H
#define NN_CPU_H

#include <vector>
#include <atomic>
#include "nn-executor.hpp"
#include "nn-cpu-ops.hpp"
#include "nn-core.hpp"

#define DEBUG_USE_MMAP_FOR_WEIGHTS false

class NnCpuDevice : public NnDevice {
public:
    NnByte **buffers;
private:
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    NnNetExecution *netExecution;
    const NnUnevenPartitionPlan *partitionPlan;
    std::atomic_uint planEpoch{0u};
    // Cached position for tp-range logging across segments.
    // Attention/RoPE ops read POS pipe, but FFN ops usually don't.
    std::atomic_uint lastPos{0xFFFFFFFFu};
    NnUint nBuffers;
    NnByte *bufferFlags;
public:
    NnCpuDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan = nullptr);
    ~NnCpuDevice() override;
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
    std::vector<NnByte *> resolvePointer(NnSize3D *pntrSize, NnPointerConfig *pointerConfig);

    // Online TP repartition (CPU-only): update the plan used by resolvePointer.
    void setPartitionPlan(const NnUnevenPartitionPlan *newPlan);
    const NnUnevenPartitionPlan *getPartitionPlan() const { return partitionPlan; }
    unsigned int getPlanEpoch() const { return planEpoch.load(std::memory_order_acquire); }
    void setPlanEpoch(unsigned int e) { planEpoch.store(e, std::memory_order_release); }
    NnUint getNodeIndex() const { return nodeConfig ? nodeConfig->nodeIndex : 0u; }

    NnUint getLastPos() const { return lastPos.load(std::memory_order_acquire); }
    void setLastPos(NnUint p) { lastPos.store(p, std::memory_order_release); }
};

class NnCpuDeviceSegment : public NnDeviceSegment {
public:
    NnUint segmentIndex;
    NnUint nOps;
    NnCpuOpForward *opForward;
    NnCpuOpContext *opContexts;
    NnCpuDevice *device;
    NnSegmentConfig *segmentConfig;
    std::vector<unsigned char> sliceFwdPrintedOnce;
    // Print tp-range once per op per epoch (so online repartition before/after can be compared).
    std::vector<unsigned int> tpRangePrintedEpoch;
    // Cached position (from ops that read POS pipe) to allow other ops in the same layer
    // to be logged with a stable pos value.
    std::atomic_uint lastPos{0xFFFFFFFFu};
    std::atomic_uint planEpochReady{0u};
    NnCpuDeviceSegment(NnCpuDevice *device, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnCpuOpForward *opForward, NnCpuOpContext *opContexts, NnUint nOps)
        : segmentIndex(segmentIndex), nOps(nOps), opForward(opForward), opContexts(opContexts), device(device), segmentConfig(segmentConfig), sliceFwdPrintedOnce(nOps, 0u), tpRangePrintedEpoch(nOps, 0xFFFFFFFFu) {
            if (device != nullptr) {
                planEpochReady.store(device->getPlanEpoch(), std::memory_order_release);
            }
        }
    ~NnCpuDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;

    // CPU-only: re-resolve PNTR_* pointers/sizes after partition plan updates.
    void refreshPointers();
};

#endif