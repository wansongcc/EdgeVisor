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
double nnCudaProfileGemmMs(NnUint gpuIndex, int n);
double nnCudaProfileLayerMs(NnUint gpuIndex, unsigned dim, unsigned hiddenDim, unsigned nHeads, unsigned nKvHeads);
unsigned long long nnCudaFreeBytes(NnUint gpuIndex);

// One allocated layer, repeated. runOnceMs times a single pass of the same
// seven projections the one-shot profile averages, and returns a negative
// value when the context could not be created.
class NnCudaLayerSoak {
public:
    NnCudaLayerSoak(unsigned gpuIndex, unsigned dim, unsigned hiddenDim, unsigned nHeads, unsigned nKvHeads);
    ~NnCudaLayerSoak();
    double runOnceMs();
private:
    struct Impl;
    Impl *impl;
    NnCudaLayerSoak(const NnCudaLayerSoak &);
    NnCudaLayerSoak &operator=(const NnCudaLayerSoak &);
};

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

// Per-layer GPU timing target: which segment feeds which (layer, kind) slot of
// NnNetExecution::LayerPerfState. The struct itself is declared in
// nn-executor.hpp, next to the NnDevice hooks that carry it.

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

    // --- Single-device pipe residency -------------------------------------
    // While `residencyActive` a pipe's authoritative bytes sit on the GPU
    // (`pipeDeviceDirty`) or the host copy is newer (`pipeHostDirty`). Between
    // forwards both flags are false only for the pipes endForward() just
    // published, because there the two copies agree.
    bool residencyActive;
    bool deviceWorkPending;
    void *uploadDoneEvent;
    bool uploadEventPending;
    std::vector<bool> pipeHostDirty;
    std::vector<bool> pipeDeviceDirty;
    // Pipes whose host copy the rest of the process reads after a forward: ZQ
    // (last layer's attention output, merged by pp_prepare), LG (logits the
    // sampler reads), KC/VC (aggregated KV rows the migration collector copies).
    std::vector<NnUint> trailingPipes;
    // Per-segment cudaEvents for per-layer GPU timing. Indexed by segmentIndex
    // and only created for segments the executor registered as profile targets.
    std::vector<void *> segmentStartEvents;
    std::vector<void *> segmentEndEvents;
    // Per-forward "this segment actually ran and recorded both events".
    std::vector<unsigned char> segmentRecorded;
    std::vector<unsigned char> segmentPerfTarget;
    std::vector<NnLayerPerfSegmentBinding> perfBindings;
    bool perfTimingActive;
    void resolveTrailingPipes();
    void releaseSegmentEvents();
public:
    NnCudaPinnedStaging staging;
    // Dedicated pinned buffers for the per-op segment input upload and output
    // download. Kept separate from `staging` (which is reused by plan-refresh
    // and weight loading in the middle of a forward) so that batching a whole
    // segment's transfers into one pinned block can never alias them.
    NnCudaPinnedStaging uploadStaging;
    NnCudaPinnedStaging downloadStaging;
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

    // --- Single-device pipe residency API ---------------------------------
    void beginForward(NnNetExecution *execution) override;
    void endForward(NnNetExecution *execution) override;
    void syncPipeToHost(NnUint pipeIndex, NnUint batchSize) override;
    void invalidatePipeOnDevice(NnUint pipeIndex) override;
    void setLayerPerfSegments(const std::vector<NnLayerPerfSegmentBinding> &bindings) override;

    bool isResidencyActive() const { return residencyActive; }
    bool isPipeHostDirty(NnUint pipeIndex) const {
        return pipeIndex < pipeHostDirty.size() && pipeHostDirty[pipeIndex];
    }
    bool isPipeDeviceDirty(NnUint pipeIndex) const {
        return pipeIndex < pipeDeviceDirty.size() && pipeDeviceDirty[pipeIndex];
    }
    // The host copy is authoritative (device copy absent or stale).
    void markPipeHostOwned(NnUint pipeIndex);
    // A kernel has just overwritten this pipe on the device. The host copy is
    // now stale, but it is the destination of a later publish on purpose.
    void markPipeDeviceOwned(NnUint pipeIndex);
    // Both copies now hold the same bytes, so neither needs another transfer
    // until something writes one of them.
    void markPipeSynchronized(NnUint pipeIndex);
    void markDeviceWork() { deviceWorkPending = true; }
    // Wait for the previous segment's H2D batch to stop reading uploadStaging.
    void waitUploadStagingIdle();
    void noteUploadBatchIssued();
    // One batched D2H of `pipes` plus a single stream synchronize. Returns true
    // when bytes were copied. Both dirty flags end up clear for every pipe it
    // published, because after the memcpy the two copies agree.
    bool publishPipesToHost(const std::vector<NnUint> &pipes, NnUint batchSize);
    // Publish every pipe whose authoritative bytes are still on the device. Used
    // when residency is off and the host has to be the channel again.
    bool publishAllDeviceDirtyPipes(NnUint batchSize);
    // Per-layer GPU timing hooks, all no-ops unless endForward() needs them.
    bool isLayerPerfSegment(NnUint segmentIndex) const {
        return segmentIndex < segmentPerfTarget.size() && segmentPerfTarget[segmentIndex] != 0u;
    }
    void recordSegmentStart(NnUint segmentIndex);
    void recordSegmentEnd(NnUint segmentIndex);
};

// Per-op pointer binding for the CUDA segment: the resolved device buffers and
// the plan-dependent layouts, cached so the hot op loop never re-resolves them.
struct NnCudaOpBinding {
    NnCudaBuffer *inBuf;
    NnCudaBuffer *outBuf;
    NnPointerLayout inLayout;
    NnPointerLayout outLayout;
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

    // Cached per-op bindings and the deduplicated host<->device pipe sets the
    // segment needs. All three are built once in the constructor.
    std::vector<NnCudaOpBinding> opBindings;
    std::vector<NnUint> uploadPipeIndices;
    std::vector<NnUint> downloadPipeIndices;
    // Pipes whose contents this segment writes on the device. Used to flip the
    // residency dirty flags without a per-op pipe lookup.
    std::vector<NnUint> writtenPipeIndices;
    // Pipes this segment's own syncs need to read on the host, precomputed once
    // per segment from segmentConfig->syncs (see buildPipeIndexSets).
    std::vector<NnUint> syncReadPipeIndices;
    bool hasHostVisibleSync;
    unsigned int bindingsEpoch;
    bool bindingsValid;
    bool isHostOnlyPlanSegment;
    bool hasControlOps;

    std::string unsupportedOpMessage(NnUint opIndex) const;
    void buildPipeIndexSets();
    void rebuildBindings();
    // `onlyHostDirty` is the residency path: it uploads only the pipes whose
    // host copy is newer than the device copy, so a pipe whose authoritative
    // bytes already sit on the GPU costs nothing.
    void uploadSegmentInputs(NnUint batchSize, bool onlyHostDirty = false);
    bool downloadSegmentOutputs(NnUint batchSize);
    // Residency path: publish just the pipes this segment's own syncs read on
    // the host and leave every other output device-authoritative.
    void publishResidentOutputs(NnUint batchSize);
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
