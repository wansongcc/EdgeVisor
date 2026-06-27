#ifndef APP_HPP
#define APP_HPP

#include <vector>
#include <deque>
#include <mutex>
#include <chrono>
#include <utility>
#include "nn/nn-core.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-executor.hpp"
#include "nn/nn-network-local.hpp"
#include "tokenizer.hpp"
#include "llm.hpp"
#include "plan-command.hpp"
#include "ablation.hpp"

class AppCliArgs {
public:
    enum Backend {
        BACKEND_AUTO,
        BACKEND_CPU,
        BACKEND_VULKAN,
        BACKEND_CUDA,
    };

    char *mode;
    NnUint nThreads;
    NnUint nBatches;
    bool info;
    bool help;
    Backend backend;
    char *backendStr;

    // inference
    char *modelPath;
    char *tokenizerPath;
    char *prompt;
    bool interactive;
    NnFloatType syncType;
    NnUint nWorkers;
    NnUint workerAllocCount;
    char **workerHosts;
    NnUint *workerPorts;
    float temperature;
    float topp;
    NnUint steps;
    bool benchmark;
    bool lastStageSampling;
    unsigned long long seed;
    ChatTemplateType chatTemplateType;
    NnUint maxSeqLen;
    bool netTurbo;
    int gpuIndex;
    int gpuSegmentFrom;
    int gpuSegmentTo;

    char *ratiosStr;
    bool warmupEnabled; // Auto-select ratios before inference when --ratios is omitted
    NnUint warmupSteps; // Probe generation steps per candidate
    NnUint warmupBudget; // Maximum number of candidates to probe
    char *warmupCandidatesStr; // Optional candidate override, separated by whitespace/semicolon
    char *kvRedundancyStr; // KV redundancy per node, format: "2" (all) or "2,3,2,3" (per-node)
    bool enablePlanBarrier; // Enable plan barrier for online migration
    bool enableStageFullWeights; // Enable stage full residency (full weights and buffers)
    bool enableKvRedundancyDuringMigration; // Keep KV redundancy enabled during online migration
    bool allowNoShadowHeadMigration; // Allow head migration when runtime will recover KV state
    bool enableKvAggregate; // Build KV aggregate pipes (KC/VC)
    bool enablePpMigration; // Enable PP layer migration control path
    NnUint runtimeRedundantBoundaryLayers; // Runtime redundant boundary span in layers
    bool runtimeActiveSegEnabled; // Default gate for primary segments
    bool runtimeRedundantSegEnabled; // Default gate for redundant segments
    char *runtimePrimarySkipLayersStr; // Comma-separated primary layers to disable, e.g. "14,15"
    char *edgevisorAblationConfigPath;
    char *shadowKvModeStr;
    char *pointerSwizzlingModeStr;
    char *jitModeStr;
    char *vgModeStr;
    char *disableShardingControllerStr;
    char *disablePipelineBalancerStr;
    char *fallbackPolicyStr;
    char *ablationLogPath;
    char *experimentId;

    // worker
    NnUint port;

    static AppCliArgs parse(int argc, char **argv, bool hasMode);
    static const char *backendToString(Backend backend);
    ~AppCliArgs();

};

typedef struct {
    NnUint position;
    NnUint batchSize; // 0 = stop signal
    NnUint flags;     // bit0: enable per-token profiling
    NnUint planCmdSeq; // command seq; valid only when LLM_CTRL_HAS_PLAN_CMD is set
} LlmControlPacket;

enum LlmControlFlags : NnUint {
    LLM_CTRL_PROFILE = 1u << 0,
    LLM_CTRL_HAS_PLAN_CMD = 1u << 1,
    LLM_CTRL_HAS_KV_TRANSFER = 1u << 2,
    LLM_CTRL_HAS_LAYER_SWITCH = 1u << 3,
    LLM_CTRL_HAS_KV_EXPORT_REQUEST = 1u << 4,
    LLM_CTRL_CONTROL_ONLY = 1u << 5,
};

typedef struct {
    NnUint magic;          // 'KVTR'
    NnUint version;        // 1
    NnUint layerIndex;
    NnUint position;
    NnUint kvDim;          // Payload width in floats. Full-row transfers use model kvDim.
    NnUint fromNodeIndex;
    NnUint targetNodeIndex;
    NnUint rangeStart;     // Destination/source column offset for partial KV transfers.
    NnUint rangeLen;       // 0 means full-row legacy semantics; otherwise payload width.
} LlmKvTransferHeader;

typedef struct {
    NnUint magic;          // 'KVAK'
    NnUint version;        // 1
    NnUint layerIndex;
    NnUint position;
    NnUint fromNodeIndex;
    NnUint toNodeIndex;
} LlmKvAckPacket;

typedef struct {
    NnUint magic;          // 'KVTB'
    NnUint version;        // 1
    NnUint count;
    NnUint reserved;
} LlmKvTransferBatchHeader;

typedef struct {
    NnUint magic;          // 'KABT'
    NnUint version;        // 1
    NnUint count;
    NnUint reserved;
} LlmKvAckBatchHeader;

typedef struct {
    NnUint magic;          // 'LVSW'
    NnUint version;        // 1
    NnUint boundaryLayer;
    NnUint fromNodeIndex;
    NnUint toNodeIndex;
    NnUint reserved0;
    NnUint reserved1;
    NnUint reserved2;
} LlmLayerSwitchPacket;

typedef struct {
    NnUint magic;          // 'LSWB'
    NnUint version;        // 1
    NnUint count;
    NnUint reserved;
} LlmLayerSwitchBatchHeader;

typedef struct {
    NnUint magic;          // 'KXTR'
    NnUint version;        // 1
    NnUint requestId;
    NnUint fromNodeIndex;
    NnUint targetStageRootNodeIndex;
    NnUint endPosition;
    NnUint layerCount;
    NnUint kvDim;
    NnUint rangeStart;
    NnUint rangeLen;
} LlmKvExportRequestHeader;

typedef struct {
    NnUint magic;          // 'KXRS'
    NnUint version;        // 1
    NnUint requestId;
    NnUint fromNodeIndex;
    NnUint rowCount;
    NnUint kvDim;
    NnUint exportedRows;
    NnUint reserved;
} LlmKvExportResponseHeader;

static constexpr NnUint LLM_KV_TRANSFER_MAGIC = 0x5254564bu; // 'KVTR'
static constexpr NnUint LLM_KV_ACK_MAGIC = 0x4b41564bu;      // 'KVAK'
static constexpr NnUint LLM_LAYER_SWITCH_MAGIC = 0x5753564cu; // 'LVSW'
static constexpr NnUint LLM_KV_TRANSFER_BATCH_MAGIC = 0x4254564bu; // 'KVTB'
static constexpr NnUint LLM_KV_ACK_BATCH_MAGIC = 0x5442414bu; // 'KABT'
static constexpr NnUint LLM_LAYER_SWITCH_BATCH_MAGIC = 0x4257534cu; // 'LSWB'
static constexpr NnUint LLM_KV_EXPORT_REQUEST_MAGIC = 0x5254584bu; // 'KXTR'
static constexpr NnUint LLM_KV_EXPORT_RESPONSE_MAGIC = 0x5352584bu; // 'KXRS'
static constexpr NnUint LLM_KV_TRANSFER_VERSION = 1u;
static constexpr NnUint LLM_KV_ACK_VERSION = 1u;
static constexpr NnUint LLM_LAYER_SWITCH_VERSION = 1u;
static constexpr NnUint LLM_KV_TRANSFER_BATCH_VERSION = 1u;
static constexpr NnUint LLM_KV_ACK_BATCH_VERSION = 1u;
static constexpr NnUint LLM_LAYER_SWITCH_BATCH_VERSION = 1u;
static constexpr NnUint LLM_KV_EXPORT_REQUEST_VERSION = 1u;
static constexpr NnUint LLM_KV_EXPORT_RESPONSE_VERSION = 1u;

typedef struct {
    NnUint position;
    NnUint batchSize;
    NnUint nodeIndex;
    NnUint stageIndex;
    NnUint execUs;
    NnUint syncUs;
    NnUint syncPpSendUs;
    NnUint syncPpRecvUs;
    NnUint syncRootWaitUs;
    NnUint syncLogitsUs;
    NnUint syncOtherUs;
    NnUint bubbleUs;
    NnUint bubbleSegments;
    NnUint bubbleOps;
    NnUint bubbleSkippedSyncs;
    NnUint bubbleDrainUs;
    NnUint bubbleCompleted;
} LlmPerfPacket;

// Bootstrap settings sent from root to worker after socket connect and before
// sending net/node configs. This removes the need for workers to pass --model/--ratios.
enum LlmBootstrapFlags : NnUint {
    LLM_BOOTSTRAP_HAS_MODEL_PATH          = 1u << 0,
    LLM_BOOTSTRAP_HAS_RATIOS              = 1u << 1,
    LLM_BOOTSTRAP_ENABLE_PLAN_BARRIER    = 1u << 2,
    LLM_BOOTSTRAP_ENABLE_STAGE_FULL_WEIGHTS = 1u << 3,
    LLM_BOOTSTRAP_ENABLE_KV_REDUNDANCY_DURING_MIGRATION = 1u << 4,
    LLM_BOOTSTRAP_HAS_PRIMARY_SKIP_LAYERS = 1u << 5,
    LLM_BOOTSTRAP_ENABLE_KV_AGGREGATE = 1u << 6,
    LLM_BOOTSTRAP_HAS_KV_REDUNDANCY = 1u << 7,
    LLM_BOOTSTRAP_ENABLE_BUBBLE_SHADOW_KV = 1u << 8,
    LLM_BOOTSTRAP_DISABLE_BUBBLE_SHADOW_KV_ASYNC = 1u << 9,
    LLM_BOOTSTRAP_LAST_STAGE_SAMPLING = 1u << 10,
};

typedef struct {
    NnUint magic;      // 'DLBM'
    NnUint version;    // 2
    NnUint flags;      // LlmBootstrapFlags
    NnUint benchmarkEnabled; // 0/1, enables executor timer on workers
    NnUint enablePlanBarrier; // 0/1, enables plan barrier for online migration
    NnUint enableStageFullWeights; // 0/1, enables stage full residency (full weights and buffers)
    NnUint enableKvRedundancyDuringMigration; // 0/1, keeps KV redundancy enabled during migration
    NnUint allowNoShadowHeadMigration; // 0/1, permits no-shadow head migration with recovery
    NnUint enableKvAggregate; // 0/1, builds KV aggregate pipes (KC/VC)
    NnUint runtimeRedundantBoundaryLayers; // boundary span in layers for runtime redundant plan
    NnUint runtimeActiveSegEnabled; // 0/1 default primary segment gate
    NnUint runtimeRedundantSegEnabled; // 0/1 default redundant segment gate
    NnUint maxSeqLen;  // forwarded from root args
    NnUint syncType;   // NnFloatType (as NnUint)
    NnUint modelPathLen; // bytes including '\0' if present
    NnUint ratiosLen;    // bytes including '\0' if present
    NnUint primarySkipLayersLen; // bytes including '\0' if present
    NnUint kvRedundancyLen; // bytes including '\0' if present
    NnUint bubbleShadowKvEnabled; // 0/1, compute runtime redundant segments after primary forward
    float samplerTemperature;
    float samplerTopP;
    unsigned long long samplerSeed;
} LlmBootstrapPacket;

static constexpr NnUint LLM_BOOTSTRAP_MAGIC = 0x4d424c44u; // 'DLBM' little-endian
static constexpr NnUint LLM_BOOTSTRAP_VERSION = 12u;

static constexpr NnUint LLM_SAMPLED_TOKEN_MAGIC = 0x4b545344u; // 'DSTK' little-endian
static constexpr NnUint LLM_SAMPLED_TOKEN_VERSION = 1u;

typedef struct {
    NnUint magic;
    NnUint version;
    NnUint nodeIndex;
    NnUint position;
    NnUint token;
    float logit;
} LlmSampledTokenPacket;

typedef struct {
    NnUint layerIndex;
    NnUint position;
    std::vector<float> kRow;
    std::vector<float> vRow;
} RootKvAggRowPacket;

typedef struct {
    LlmKvTransferHeader header;
    std::vector<float> kRow;
    std::vector<float> vRow;
} PendingKvTransferItem;

class RootLlmInference {
public:
    enum KvTransferSubmitStatus : NnUint {
        KV_TRANSFER_SUBMIT_OK = 0u,
        KV_TRANSFER_SUBMIT_NO_NETWORK = 1u,
        KV_TRANSFER_SUBMIT_NO_TARGET_STAGE = 2u,
        KV_TRANSFER_SUBMIT_LAYER_NOT_IN_LIST = 3u,
        KV_TRANSFER_SUBMIT_INVALID_ROW = 4u,
        KV_TRANSFER_SUBMIT_WAITING_ACK = 5u,
        KV_TRANSFER_SUBMIT_DUP_LAYER_PENDING = 6u,
    };

    float *logitsPipe;
    const std::vector<LlmPerfPacket>& getLastPerf() const { return lastPerf; }
    NnUint getPosition() const { return controlPacket.position; }
    NnUint getBatchSize() const { return controlPacket.batchSize; }
    const NnUnevenPartitionPlan* getPartitionPlan() const { return plan; }
    NnUint getNodeIndex() const { return 0u; }
    NnUint getKvDim() const { return header != nullptr ? header->kvDim : 0u; }
    NnUint getSeqLen() const { return header != nullptr ? header->seqLen : 0u; }
    int getAsyncKvCollectLayer() const { return asyncKvCollectLayer; }
    int getAsyncKvCollectPos() const { return asyncKvCollectPos; }
    NnUint getMigrationFromNodeIndex() const { return migrationFromNodeIndex; }
    NnUint getMigrationTargetNodeIndex() const { return nextStageRootNode; }
    const std::vector<NnUint>& getMigrationLayers() const { return migrationLayers; }
    int getMigrationLayerCount() const { return migrationLayerCount; }
    bool isMigrationLayerListPinnedByEnv() const { return migrationLayerListPinnedByEnv; }
    bool isPpMigrationEnabled() const { return ppMigrationEnabled; }
    bool hasMigrationAck() const { return migrationAckSeen; }
    int getMigrationAckPos() const { return migrationAckPos; }
    int getMigrationAckLayer() const { return migrationAckLayer; }
    bool tryReceiveLastStageSampledToken(NnUint &token, float *logit = nullptr);
    bool hasAsyncKvCollector() const;
    bool tryPopAsyncKvRow(RootKvAggRowPacket &packet);
    KvTransferSubmitStatus submitBoundaryKvTransferDetailed(
        NnUint layerIndex,
        NnUint position,
        const std::vector<float> &kRow,
        const std::vector<float> &vRow,
        NnUint rangeStart = 0u,
        NnUint rangeLen = 0u);
    bool submitBoundaryKvTransfer(NnUint layerIndex, NnUint position, const std::vector<float> &kRow, const std::vector<float> &vRow);
private:
    float *tokenPipe = nullptr;
    float *positionPipe = nullptr;
    float *kvAggKPipe = nullptr;
    float *kvAggVPipe = nullptr;
    LlmHeader *header;
    NnNetExecution *execution;
    NnExecutor *executor;
    NnNetwork *network;
    LlmControlPacket controlPacket;
    bool profileEnabled = false;
    const NnUnevenPartitionPlan* plan = nullptr;
    const RuntimeStageLayerPlan* runtimePlan = nullptr;
    std::vector<LlmPerfPacket> lastPerf;
    NnUint lastPlanCmdSeqSent = 0u;
    NnBubbleShadowStats lastBubbleShadowStats{};
    int asyncKvCollectLayer = -1;
    int asyncKvCollectPos = -1;
    std::deque<RootKvAggRowPacket> asyncKvRows;
    mutable std::mutex asyncKvRowsMutex;
    std::mutex kvTransferMutex;
    std::vector<PendingKvTransferItem> pendingKvTransfers;
    bool waitingKvAck = false;
    NnUint waitingKvAckExpectedCount = 0u;
    std::vector<NnUint> waitingKvAckLayers;
    NnUint waitingKvAckReceivedCount = 0u;
    std::vector<NnUint> waitingKvAckReceivedLayers;
    std::vector<NnUint> waitingKvAckNodes;
    std::vector<NnUint> waitingKvAckNodeExpected;
    std::vector<NnUint> waitingKvAckNodeReceived;
    std::vector<NnUint> pendingLayerSwitchLayers;
    std::vector<NnUint> migrationLayers;
    std::vector<int> tokenHistory;
    int migrationStageStartLayer = -1;
    int migrationStageEndLayer = -1;
    int migrationLayerCount = 1;
    bool ppMigrationEnabled = false;
    bool migrationBatchSubmitted = false;
    bool migrationExportRequested = false;
    bool migrationLayerListPinnedByEnv = false;
    int boundaryLayerForMigration = -1;
    bool migrationAckSeen = false;
    int migrationAckPos = -1;
    int migrationAckLayer = -1;
    unsigned long long lastPpPlanCacheSeqApplied = 0ull;
    unsigned long long lastHeadRecoveryPlanCacheSeqApplied = 0ull;
    unsigned long long headRecoveryRangeCacheSeq = 0ull;
    bool headRecoveryRangeCached = false;
    bool headRecoveryPlanAfterApply = false;
    NnUint headRecoveryStageIndex = 0xFFFFFFFFu;
    NnUint headRecoveryFromNode = 0u;
    NnUint headRecoveryToNode = 0u;
    NnUint headRecoveryRangeHeadStart = 0u;
    NnUint headRecoveryRangeHeadLen = 0u;
    NnUint migrationFromNodeIndex = 0u;
    NnUint nextStageRootNode = (NnUint)-1;
    int kvAckSocketIndex = -1;
    NnUint nextKvExportRequestId = 1u;
    uint64_t lastMigrationStateTransferBytes = 0u;
    uint64_t lastMigrationExportedRows = 0u;
    bool collectSourceStageKvTransfers(NnUint endPos, NnUint *exportedRows, NnUint *queuedRows, uint64_t *sourceTransferBytes);
    bool collectHeadKvTransfers(const PlanCommand &cmd, NnUint endPos, NnUint *exportedRows, NnUint *queuedRows, uint64_t *sourceTransferBytes);
    bool flushPendingKvTransfersControlOnly(uint64_t *targetTransferBytes);
    bool sendPendingLayerSwitchControlOnly();
    void maybeEnableShiftedPpStartForSourceStage(
        const std::vector<NnUint> &switchLayers,
        NnUint sourceNodeIndex,
        NnUint targetNodeIndex,
        bool selfIsSource);
    void collectProfilePackets();
    bool replayHistoryForMigrationRecompute(NnUint endPos, double *recomputeMs, uint64_t *recomputeTokens);
    bool replayHistoryForHeadMigrationRecompute(const PlanCommand &cmd, NnUint endPos, double *recomputeMs, uint64_t *recomputeTokens);
    bool recoverHeadMigrationNoShadow(const PlanCommand &cmd, NnUint triggerPos);
    void emitPpMigrationRecoverEvent(
        bool applySuccess,
        uint64_t stateTransferBytes,
        uint64_t recomputeTokensOrLayers,
        double statePrepareMs,
        double recoverMs,
        double stallMs,
        const std::string &fallbackReason);
public:
    RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network, const NnUnevenPartitionPlan* plan, bool profileEnabled, bool ppMigrationEnabled);
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
    void setToken(NnUint batchIndex, NnUint token);
    void setRuntimeLayerGate(bool enablePrimarySegments, bool enableRedundantSegments);
    void setPrimaryLayerEnabled(NnUint layerIndex, bool enabled);
    void setShiftedPpStartLayerEnabled(NnUint layerIndex, bool enabled);
    void forward();
    void finish();
};

class WorkerLlmInference {
public:
    bool isFinished;
    NnUint position() const { return controlPacket.position; }
    NnUint batchSize() const { return controlPacket.batchSize; }
    NnUint flags() const { return controlPacket.flags; }
    bool consumeLayerSwitch(LlmLayerSwitchPacket &packet);
    void flushPendingKvAck();
    bool consumePendingKvTransfer(LlmKvTransferHeader &hdr, std::vector<float> &kRow, std::vector<float> &vRow);
    bool consumeKvExportRequest(LlmKvExportRequestHeader &hdr, std::vector<NnUint> &layers);
    void flushKvExportResponse(
        const LlmKvExportRequestHeader &req,
        const std::vector<NnUint> &layers,
        NnUint kvDim,
        NnExecutor *executor);
    void maybeSendLastStageSampledToken(const NnUnevenPartitionPlan *plan);
private:
    float *positionPipe;
    float *logitsPipe;
    NnNetExecution *execution;
    NnNetwork *network;
    Sampler *lastStageSampler;
    NnUint logitsPipeIndex = (NnUint)-1;
    NnUint localNodeIndex = 0u;
    LlmControlPacket controlPacket;
    NnUint lastPlanCmdSeqRecv = 0u;
    std::deque<PendingKvTransferItem> pendingKvTransfers;
    std::vector<LlmKvAckPacket> pendingKvAcks;
    std::deque<LlmLayerSwitchPacket> pendingLayerSwitches;
    std::deque<std::pair<LlmKvExportRequestHeader, std::vector<NnUint>>> pendingKvExportRequests;
public:
    WorkerLlmInference(NnNetExecution *execution, NnNetwork *network, NnUint localNodeIndex, NnUint logitsPipeIndex, Sampler *lastStageSampler);
    bool tryReadControlPacket();
};

typedef struct {
    AppCliArgs *args;
    LlmHeader *header;
    RootLlmInference *inference;
    Tokenizer *tokenizer;
    Sampler *sampler;
    NnNetwork *network;
    NnExecutor *executor;
    NnNodeConfig *nodeConfig;
} AppInferenceContext;

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context));
void runWorkerApp(AppCliArgs *args);

#endif
