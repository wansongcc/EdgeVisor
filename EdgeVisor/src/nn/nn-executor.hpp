#ifndef NN_EXECUTOR_H
#define NN_EXECUTOR_H

#include "nn-core.hpp"
#include <atomic>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <thread>
#include "pthread.h"

class NnExecutor;
class NnNetExecution;

// Segment kind codes for per-layer compute profiling. The executor fills
// LayerPerfState from these, and the CUDA backend uses the same codes to know
// which slot a per-segment GPU timing sample belongs to.
enum NnSegmentKindCode {
    NN_SEG_KIND_OTHER = 0,
    NN_SEG_KIND_ATTN  = 1,
    NN_SEG_KIND_FFN   = 2
};

// Per-layer GPU timing target: which segment feeds which (layer, kind) slot of
// NnNetExecution::LayerPerfState. Derived from the same ATTN/FFN segment
// classification the executor uses for its CPU-side accounting.
struct NnLayerPerfSegmentBinding {
    NnUint segmentIndex;
    NnUint layerIndex;
    NnByte kind;
};

class NnDeviceSegment {public:
    virtual ~NnDeviceSegment() {};
    virtual void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) = 0;
    virtual void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) = 0;
    virtual void setPartitionPlan(const NnUnevenPartitionPlan * /*plan*/) {}
    virtual void refreshPointers() {}
    virtual bool exportLayerKvRow(
        NnUint /*layerIndex*/,
        NnUint /*position*/,
        NnUint /*kvDim*/,
        std::vector<float> & /*kRow*/,
        std::vector<float> & /*vRow*/,
        NnUint /*rangeStart*/ = 0u,
        NnUint /*rangeLen*/ = 0u) { return false; }
    virtual bool applyTransferredKvRow(
        NnUint /*layerIndex*/,
        NnUint /*position*/,
        const std::vector<float> & /*kRow*/,
        const std::vector<float> & /*vRow*/,
        NnUint /*rangeStart*/ = 0u,
        NnUint /*rangeLen*/ = 0u) { return false; }
};

class NnDevice {
public:
    virtual NnUint maxNThreads() = 0;
    virtual ~NnDevice() {}
    virtual NnDeviceSegment *createSegment(NnUint segmentIndex) = 0;

    // Single-device forward residency hooks. A device that keeps its pipes
    // resident on the GPU across the whole forward() implements these; the
    // defaults below leave every other backend byte-for-byte unchanged.
    //
    // beginForward() is called once before the executor threads start,
    // endForward() once after the last executor thread has returned (and after
    // the bubble-shadow drain, so every shadow op on this device is finished).
    // syncPipeToHost()/invalidatePipeOnDevice() let the executor tell the device
    // that a sync step is about to read a pipe on the host, or that a sync step
    // overwrote a pipe on the host and the device copy is therefore stale.
    virtual void beginForward(NnNetExecution * /*execution*/) {}
    virtual void endForward(NnNetExecution * /*execution*/) {}
    virtual void syncPipeToHost(NnUint /*pipeIndex*/, NnUint /*batchSize*/) {}
    virtual void invalidatePipeOnDevice(NnUint /*pipeIndex*/) {}
    // Declares which segments carry per-layer GPU timing. Called once by the
    // executor after it has classified every segment; the device stores the
    // list and uses it to record per-segment cudaEvents.
    virtual void setLayerPerfSegments(const std::vector<NnLayerPerfSegmentBinding> & /*bindings*/) {}
};

class NnNodeSynchronizer {
public:
    virtual ~NnNodeSynchronizer() {};
    virtual void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) = 0;
    // Called exactly once per STEP_SYNC_NODES executor step, after all executor threads
    // have returned from sync(). This is safe for additional network I/O on shared sockets.
    virtual void onSyncStepComplete(NnUint /*segmentIndex*/) {}
};

class NnFakeNodeSynchronizer : public NnNodeSynchronizer {
public:
    ~NnFakeNodeSynchronizer() override {};
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
};

class NnNetExecution {
public:
    NnUint nThreads;
    NnUint nPipes;
    NnByte **pipes;
    NnUint batchSize;
    NnUint position;
    NnUint nBatches;

    // Optional per-layer profiling state shared between executor and synchronizer.
    // Allocated lazily when benchmark/profile is enabled.
    struct LayerPerfState;
    LayerPerfState *layerPerf = nullptr;

    NnNetExecution(NnUint nThreads, NnNetConfig *netConfig);
    ~NnNetExecution();
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
};

// Per-layer CPU compute timing (microseconds), filled by executor and consumed by synchronizer.
struct NnNetExecution::LayerPerfState {
    NnUint nLayers = 0u;
    // Indexed by layerIndex.
    std::vector<unsigned long long> attnUs;
    std::vector<unsigned long long> ffnUs;
    void reset() {
        std::fill(attnUs.begin(), attnUs.end(), 0ull);
        std::fill(ffnUs.begin(), ffnUs.end(), 0ull);
    }
};

enum NnExecutorStepType {
    STEP_EXECUTE_OP,
    STEP_SYNC_NODES,
};

#define N_STEP_TYPES STEP_SYNC_NODES + 1

struct NnExecutorSyncProfile {
    unsigned long long ppSendUs = 0ull;
    unsigned long long ppRecvUs = 0ull;
    unsigned long long rootWaitUs = 0ull;
    unsigned long long logitsUs = 0ull;
    unsigned long long otherUs = 0ull;

    void reset() {
        ppSendUs = 0ull;
        ppRecvUs = 0ull;
        rootWaitUs = 0ull;
        logitsUs = 0ull;
        otherUs = 0ull;
    }
};

class NnExecutorDevice {
public:
    std::unique_ptr<NnDevice> device;
    int segmentFrom;
    int segmentTo;
    NnExecutorDevice(NnDevice *device, int segmentFrom, int segmentTo);
};

typedef struct {
    NnExecutorStepType type;
    NnDeviceSegment *segment;
    NnUint arg0;
    NnOpConfig *opConfig;
    NnUint segmentIndex;
} NnExecutorStep;

typedef struct {
    NnUint nThreads;
    NnUint nSteps;
    NnExecutorStep *steps;
    NnNodeSynchronizer *synchronizer;
    std::atomic_uint currentStepIndex;
    std::atomic_uint doneThreadCount;
    std::atomic_bool isAlive;
    NnUint batchSize;
    NnUint position;
    Timer *timer;
    NnUint totalTime[N_STEP_TYPES];

    // Optional per-layer compute profiling.
    NnNetExecution::LayerPerfState *layerPerf;
    const NnByte *segmentKinds;
    const NnByte *segmentRuntimeRoles;
    const std::atomic_uint8_t *segmentEnabled;
    const int *segmentLayerIndex;
    const NnByte *segmentHasExecOps;
    NnUint nSegments;
    NnUint nodeIndex;
    NnExecutorSyncProfile *syncProfile;
    const NnByte *segmentSyncProfileKinds;
    // Set when one device owns every segment on this node (see NnExecutor's
    // residencyDevice); the executor threads use it to invalidate device pipes
    // after each sync step. `residencyStage` is this node's stage from the
    // partition plan (null when it has none).
    NnDevice *residencyDevice;
    const NnStageConfig *residencyStage;
    const NnNodeConfig *nodeConfig;
    // Owner, so the root-wait path can start the shadow KV pass while the
    // executor threads are blocked in sync(). Null in tests that drive
    // executeStep without an executor.
    NnExecutor *executor;
} NnExecutorContext;

typedef struct {
    NnUint threadIndex;
    NnExecutorContext *context;
    PthreadHandler handler;
} NnExecutorThread;

typedef struct {
    NnUint segmentsVisited;
    NnUint opStepsExecuted;
    NnUint skippedSyncSteps;
    NnUint attnSegments;
    NnUint ffnSegments;
    NnUint otherSegments;
    NnUint uniqueLayers;
    NnUint budgetHit;
    NnUint completed;
    NnUint drainUs;
    unsigned long long elapsedUs;
} NnBubbleShadowStats;

class NnExecutorException : public std::runtime_error {
public:
    NnExecutorException(const std::string message);
};

class NnExecutor {
private:
    NnNetExecution *netExecution;
    NnNodeConfig *nodeConfig;
    std::vector<std::unique_ptr<NnDeviceSegment>> segments;
    std::vector<NnExecutorStep> steps;
    // Segment classification (ATTN/FFN/OTHER) for per-layer compute profiling.
    std::vector<NnByte> segmentKinds;
    // Segment runtime role for gate control: 0=unguarded, 1=primary(active), 2=redundant.
    std::vector<NnByte> segmentRuntimeRoles;
    std::vector<int> segmentLayerIndex;
    std::vector<NnByte> segmentHasExecOps;
    std::unique_ptr<std::atomic_uint8_t[]> segmentEnabled;
    NnExecutorThread *threads;
    NnExecutorContext context;
    mutable std::mutex bubbleShadowMutex;
    NnBubbleShadowStats lastBubbleShadowStats;
    NnExecutorSyncProfile lastSyncProfile;
    bool bubbleShadowComplete;
    NnUint bubbleShadowCursor;
    NnUint bubbleShadowDrainUs;
    std::vector<NnUint> bubbleShadowStepIndices;
    // Exclusive end position whose shadow KV has been written, per layer.
    std::vector<NnUint> shadowFilledThrough_;
    std::vector<NnByte> segmentSyncProfileKinds;
    // Single-device pipe residency. Non-null only when this node has exactly one
    // device entry and it covers every segment, so one device owns the whole
    // forward and may keep pipes resident between segments. Multi-device nodes
    // (CUDA + CPU fallback ranges, tests) leave this null and keep the old
    // per-segment upload/download behaviour.
    NnDevice *residencyDevice;
    NnBubbleShadowStats runBubbleShadowRedundantInternal(NnUint budgetUs, bool allowWhileRunning);
    NnBubbleShadowStats runBubbleShadowRedundantChunk(NnUint budgetUs, bool allowWhileRunning);
    bool isRedundantLayerActive(NnUint layerIndex) const;
    // Layers this node currently executes, derived from the runtime gate. The
    // fault penalty scales by this so a stage that gave layers away does not
    // keep paying for work it no longer does.
    NnUint countExecutedLayers() const;
    void noteShadowFilled(NnUint layerIndex);
    void replayRedundantLayer(NnUint layerIndex);
    void resetBubbleShadowStateForForward();
    void drainBubbleShadowKv();
    // True when the PP root-wait bubble on this node can host the shadow KV
    // pass: shadow KV on, async bubble enabled, single executor thread, and the
    // step really is a pure root-wait sync with work still queued.
    bool shouldRunBubbleShadowDuringSync(const NnExecutorStep *step, NnUint nThreads, NnUint threadIndex) const;
    bool hasPendingBubbleShadow() const;
    friend void executeStep(NnExecutorStep *step, NnUint nThreads, NnExecutorThread *thread, NnExecutorContext *context);
public:
    NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, std::vector<NnExecutorDevice> *device, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark);
    ~NnExecutor();
    void loadWeight(const char *name, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight);
    // deferBubbleShadow=true leaves the shadow KV drain to the caller, which
    // runs it later with runDeferredBubbleShadowKv(). That is how the root hides
    // the pass in its PP wait instead of charging it after the forward.
    void forward(bool deferBubbleShadow);
    void forward();
    // Drains whatever shadow KV work forward(true) deferred. No-op when shadow
    // KV is off, when the pass already completed inside forward, or when there
    // is nothing queued.
    void runDeferredBubbleShadowKv();
    NnBubbleShadowStats runBubbleShadowRedundant(NnUint budgetUs);
    NnBubbleShadowStats getLastBubbleShadowStats() const;
    NnExecutorSyncProfile getLastSyncProfile() const;
    // CPU-only today: update partition plan used for PNTR_BATCHED_SLICE resolution.
    void setPartitionPlan(const NnUnevenPartitionPlan *plan);
    // CPU-only today: re-resolve segment pointers after updating partition plan.
    void refreshPointers();
    // Convenience: set plan + refresh pointers/configs as one atomic reconfigure step.
    void applyPartitionPlan(const NnUnevenPartitionPlan *plan);
    // Runtime segment gate APIs.
    void setSegmentEnabled(NnUint segmentIndex, bool enabled);
    void setRuntimeLayerGate(bool enablePrimarySegments, bool enableRedundantSegments);
    void setPrimaryLayerEnabled(NnUint layerIndex, bool enabled);
    void setRedundantLayerEnabled(NnUint layerIndex, bool enabled);
    // Run enabled right-boundary layers, then copy the stage-output cache back to the PP send pipe.
    void spliceRedundantLayersIntoSend(NnUint beginLayer, NnUint endLayer);
    NnUint executionPosition() const;
    bool shadowCovers(NnUint beginLayer, NnUint endLayer, NnUint position) const;
    // False when this node has no redundant or shadow-KV layer. Otherwise
    // filledThrough is the minimum exclusive end position across those layers.
    // A layer that has not been written contributes 0, which is not ready.
    bool redundantShadowCursor(NnUint *filledThrough) const;
    void setShiftedPpStartLayerEnabled(NnUint layerIndex, bool enabled);
    bool isSegmentEnabled(NnUint segmentIndex) const;
    void setPpSyncEnabled(bool enabled);
    bool exportLayerKvRow(
        NnUint layerIndex,
        NnUint position,
        NnUint kvDim,
        std::vector<float> &kRow,
        std::vector<float> &vRow,
        NnUint rangeStart = 0u,
        NnUint rangeLen = 0u);
    bool applyTransferredKvRow(
        NnUint layerIndex,
        NnUint position,
        const std::vector<float> &kRow,
        const std::vector<float> &vRow,
        NnUint rangeStart = 0u,
        NnUint rangeLen = 0u);
    NnUint getTotalTime(NnExecutorStepType type);
};

#endif
