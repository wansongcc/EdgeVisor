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
    // One-shot trigger guard for PlanCommand MODE_NEXT_BARRIER and repeated forwards.
    // We intentionally do NOT consume the global planCommandCache in barrier, so apply can
    // still read the full move list (v2). This seq guards against repeated emits.
    std::atomic_uint lastPlanCmdSeqEmitted{0u};
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

    NnUint getBufferCount() const { return nBuffers; }
    NnUint getPipeCount() const { return netConfig ? netConfig->nPipes : 0u; }

    // Online TP repartition (CPU-only): update the plan used by resolvePointer.
    void setPartitionPlan(const NnUnevenPartitionPlan *newPlan);
    const NnUnevenPartitionPlan *getPartitionPlan() const { return partitionPlan; }
    unsigned int getPlanEpoch() const { return planEpoch.load(std::memory_order_acquire); }
    void setPlanEpoch(unsigned int e) { planEpoch.store(e, std::memory_order_release); }
    NnUint getNodeIndex() const { return nodeConfig ? nodeConfig->nodeIndex : 0u; }

    unsigned int getLastPlanCmdSeqEmitted() const { return lastPlanCmdSeqEmitted.load(std::memory_order_acquire); }
    void setLastPlanCmdSeqEmitted(unsigned int s) { lastPlanCmdSeqEmitted.store(s, std::memory_order_release); }

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
    // Debug: track last writer op per buffer/pipe (best-effort, segment-local).
    std::vector<int> lastWriteOpByBuffer;
    std::vector<const char *> lastWriteNameByBuffer;
    std::vector<int> lastWriteOpByPipe;
    std::vector<const char *> lastWriteNameByPipe;
    NnCpuDeviceSegment(NnCpuDevice *device, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnCpuOpForward *opForward, NnCpuOpContext *opContexts, NnUint nOps)
        : segmentIndex(segmentIndex), nOps(nOps), opForward(opForward), opContexts(opContexts), device(device), segmentConfig(segmentConfig), sliceFwdPrintedOnce(nOps, 0u), tpRangePrintedEpoch(nOps, 0xFFFFFFFFu) {
            if (device != nullptr) {
                planEpochReady.store(device->getPlanEpoch(), std::memory_order_release);
                const NnUint nBuffers = device->getBufferCount();
                const NnUint nPipes = device->getPipeCount();
                lastWriteOpByBuffer.assign(nBuffers, -1);
                lastWriteNameByBuffer.assign(nBuffers, nullptr);
                lastWriteOpByPipe.assign(nPipes, -1);
                lastWriteNameByPipe.assign(nPipes, nullptr);
            }
        }
    ~NnCpuDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
    void setPartitionPlan(const NnUnevenPartitionPlan *plan) override;

    // Re-resolve PNTR_* pointers/sizes after partition plan updates.
    void refreshPointers() override;
};

#endif