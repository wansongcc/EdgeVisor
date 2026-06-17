#ifndef NN_EXECUTOR_H
#define NN_EXECUTOR_H

#include "nn-core.hpp"
#include <atomic>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include "pthread.h"

class NnDeviceSegment {
public:
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
    NnUint nBatches;

    // Optional per-layer profiling state shared between executor and synchronizer.
    // Allocated lazily when benchmark/profile is enabled.
    struct LayerPerfState;
    LayerPerfState *layerPerf = nullptr;

    NnNetExecution(NnUint nThreads, NnNetConfig *netConfig);
    ~NnNetExecution();
    void setBatchSize(NnUint batchSize);
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
    NnUint budgetHit;
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
public:
    NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, std::vector<NnExecutorDevice> *device, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark);
    ~NnExecutor();
    void loadWeight(const char *name, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight);
    void forward();
    NnBubbleShadowStats runBubbleShadowRedundant(NnUint budgetUs);
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
