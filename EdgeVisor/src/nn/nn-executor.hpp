#ifndef NN_EXECUTOR_H
#define NN_EXECUTOR_H

#include "nn-core.hpp"
#include <atomic>
#include <algorithm>
#include <deque>
#include <functional>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <thread>
#include "pthread.h"

class NnExecutor;

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
    // Backend-agnostic raw node-buffer access (shadow L2 stash snapshot/restore).
    virtual bool readNodeBuffer(NnUint /*bufferIndex*/, NnByte * /*dst*/, NnSize /*nBytes*/) { return false; }
    virtual bool writeNodeBuffer(NnUint /*bufferIndex*/, const NnByte * /*src*/, NnSize /*nBytes*/) { return false; }
};

class NnDevice {
public:
    virtual NnUint maxNThreads() = 0;
    virtual ~NnDevice() {}
    virtual NnDeviceSegment *createSegment(NnUint segmentIndex) = 0;
    // Ensure this thread targets the right compute device (no-op on CPU).
    // Must be called at the entry of any thread that launches device work
    // (e.g. CUDA kernels) outside the executor's own worker threads.
    virtual void setCurrentThreadDevice() {}
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
    NnExecutor *owner;
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
    // Shadow L2 (tool-wait catch-up) counters.
    NnUint stashEntries;        // current stash (debt) depth
    NnUint stashForcedDrains;   // entries force-drained on the critical path (stash cap)
    NnUint catchupEntries;      // entries completed in tool-wait catch-up windows (cumulative)
    NnUint catchupOps;          // op steps executed during catch-up (cumulative)
    unsigned long long catchupUs; // wall time spent in catch-up (cumulative)
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
    // Raw device pointers (owned by the caller's NnExecutorDevice list) used to
    // route spawned execution threads (bubble shadow, shadow L2 catch-up) to
    // the correct compute device at thread entry.
    std::vector<NnDevice *> executorDevices;
    void applyCurrentThreadDevice();
    NnExecutorContext context;
    std::thread bubbleShadowThread;
    mutable std::mutex bubbleShadowMutex;
    NnBubbleShadowStats lastBubbleShadowStats;
    NnExecutorSyncProfile lastSyncProfile;
    bool bubbleShadowAsyncRunning;
    bool bubbleShadowAsyncStarted;
    bool bubbleShadowStopRequested;
    bool bubbleShadowComplete;
    NnUint bubbleShadowCursor;
    NnUint bubbleShadowDrainUs;
    std::vector<NnUint> bubbleShadowStepIndices;
    std::vector<NnByte> segmentSyncProfileKinds;
    // Per-segment ready-point gating for right-boundary shadow KV segments:
    // the segment may only execute after the main path passed this step index
    // (the pp_send segment's pp_stage_cache CAST writing pp_stage_out).
    // 0 means "no gating".
    std::vector<NnUint> segmentReadyAfterStep;

    // ---- Shadow L2: tool-wait catch-up (debt stash) ----
    // When enabled (setShadowL2Config with a valid pp_stage_out buffer), an
    // unfinished shadow pass at the end of forward is stashed as a "debt"
    // entry (input snapshot + POS/SLT + progress cursor) instead of being
    // drained on the critical path. Debt is repaid later via
    // runShadowCatchup() during tool-wait windows; when the stash byte cap is
    // exceeded, the oldest entries are force-drained at the forward end
    // (bounded-memory fallback).
    struct ShadowL2StashEntry {
        NnUint batchSize;
        std::vector<float> act;  // batchSize * dim, snapshot of pp_stage_out
        std::vector<float> pos;  // batchSize
        std::vector<float> slot; // batchSize (empty when no slot pipe)
        NnUint cursor;           // progress within bubbleShadowStepIndices
    };
    bool shadowL2Enabled;
    NnUint shadowL2PpStageOutBufferIndex;
    NnUint shadowL2PosPipeIndex;
    NnUint shadowL2SlotPipeIndex;
    NnSize shadowL2StashCapBytes;
    std::deque<ShadowL2StashEntry> shadowL2Stash;
    NnSize shadowL2StashBytes;
    NnUint shadowL2ForcedDrains;
    NnUint shadowL2CatchupEntries;
    NnUint shadowL2CatchupOps;
    unsigned long long shadowL2CatchupUs;
    bool readShadowL2NodeBuffer(NnUint bufferIndex, NnByte *dst, NnSize nBytes);
    bool writeShadowL2NodeBuffer(NnUint bufferIndex, const NnByte *src, NnSize nBytes);
    NnSize shadowL2EntryBytes(const ShadowL2StashEntry &entry) const;
    void stashShadowL2Debt();
    void enforceShadowL2StashCap();
    bool restoreShadowL2Entry(const ShadowL2StashEntry &entry);
    NnBubbleShadowStats runBubbleShadowRedundantInternal(NnUint budgetUs, bool allowWhileRunning);
    NnBubbleShadowStats runBubbleShadowRedundantChunk(
        NnUint budgetUs,
        bool stopOnRequest,
        bool allowWhileRunning,
        const std::function<bool()> &externalStop = nullptr);
    bool isRedundantLayerActive(NnUint layerIndex) const;
    void resetBubbleShadowStateForForward();
public:
    NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, std::vector<NnExecutorDevice> *device, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark);
    ~NnExecutor();
    void loadWeight(const char *name, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight);
    void forward();
    NnBubbleShadowStats runBubbleShadowRedundant(NnUint budgetUs);
    bool isBubbleShadowAsyncModeEnabled() const;
    void maybeStartBubbleShadowAsyncBeforeSync();
    void joinBubbleShadowAsync();
    void pauseBubbleShadowAsyncAfterSync();
    void drainBubbleShadowAsync();
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
    // Debug-only (DLLAMA_DUMP_KV_DIR): dump the destination row written by every
    // OP_SHIFT (block_shift_k/block_shift_v) in both main att segments and
    // shadow-kv segments, for numerical verification of redundant KV.
    void dumpKvShiftDebug(const char *dir, NnUint batchSize);
    // Shadow L2 (tool-wait catch-up) public API.
    void setShadowL2Config(NnUint ppStageOutBufferIndex, NnUint posPipeIndex, NnUint slotPipeIndex, NnSize stashCapBytes);
    bool isShadowL2Enabled() const { return shadowL2Enabled; }
    NnUint getShadowL2DebtEntries();
    NnSize getShadowL2DebtBytes();
    // Repay stashed shadow debt (FIFO). shouldStop (optional) is polled at
    // segment/entry boundaries; when it returns true, the current entry keeps
    // its progress and is retried in a later window. Returns completed entries.
    // Must not run concurrently with forward().
    NnUint runShadowCatchup(const std::function<bool()> &shouldStop);
};

#endif
