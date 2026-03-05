#ifndef APP_HPP
#define APP_HPP

#include <vector>
#include <deque>
#include <mutex>
#include <chrono>
#include "nn/nn-core.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-executor.hpp"
#include "nn/nn-network-local.hpp"
#include "tokenizer.hpp"
#include "llm.hpp"
#include "plan-command.hpp"

class AppCliArgs {
public:
    char *mode;
    NnUint nThreads;
    NnUint nBatches;
    bool info;
    bool help;

    // inference
    char *modelPath;
    char *tokenizerPath;
    char *prompt;
    bool interactive;
    NnFloatType syncType;
    NnUint nWorkers;
    char **workerHosts;
    NnUint *workerPorts;
    float temperature;
    float topp;
    NnUint steps;
    bool benchmark;
    unsigned long long seed;
    ChatTemplateType chatTemplateType;
    NnUint maxSeqLen;
    bool netTurbo;
    int gpuIndex;
    int gpuSegmentFrom;
    int gpuSegmentTo;

    char *ratiosStr;
    char *kvRedundancyStr; // KV redundancy per node, format: "2" (all) or "2,3,2,3" (per-node)
    bool enablePlanBarrier; // Enable plan barrier for online migration
    bool enableStageFullWeights; // Enable stage full residency (full weights and buffers)
    bool enableKvRedundancyDuringMigration; // Keep KV redundancy enabled during online migration
    bool enableKvAggregate; // Build KV aggregate pipes (KC/VC)
    bool enablePpMigration; // Enable PP layer migration control path
    NnUint runtimeRedundantBoundaryLayers; // Runtime redundant boundary span in layers
    bool runtimeActiveSegEnabled; // Default gate for primary segments
    bool runtimeRedundantSegEnabled; // Default gate for redundant segments
    char *runtimePrimarySkipLayersStr; // Comma-separated primary layers to disable, e.g. "14,15"

    // worker
    NnUint port;

    static AppCliArgs parse(int argc, char **argv, bool hasMode);
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
};

typedef struct {
    NnUint magic;          // 'KVTR'
    NnUint version;        // 1
    NnUint layerIndex;
    NnUint position;
    NnUint kvDim;
    NnUint fromNodeIndex;
    NnUint targetNodeIndex;
    NnUint reserved;
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

static constexpr NnUint LLM_KV_TRANSFER_MAGIC = 0x5254564bu; // 'KVTR'
static constexpr NnUint LLM_KV_ACK_MAGIC = 0x4b41564bu;      // 'KVAK'
static constexpr NnUint LLM_LAYER_SWITCH_MAGIC = 0x5753564cu; // 'LVSW'
static constexpr NnUint LLM_KV_TRANSFER_BATCH_MAGIC = 0x4254564bu; // 'KVTB'
static constexpr NnUint LLM_KV_ACK_BATCH_MAGIC = 0x5442414bu; // 'KABT'
static constexpr NnUint LLM_LAYER_SWITCH_BATCH_MAGIC = 0x4257534cu; // 'LSWB'
static constexpr NnUint LLM_KV_TRANSFER_VERSION = 1u;
static constexpr NnUint LLM_KV_ACK_VERSION = 1u;
static constexpr NnUint LLM_LAYER_SWITCH_VERSION = 1u;
static constexpr NnUint LLM_KV_TRANSFER_BATCH_VERSION = 1u;
static constexpr NnUint LLM_KV_ACK_BATCH_VERSION = 1u;
static constexpr NnUint LLM_LAYER_SWITCH_BATCH_VERSION = 1u;

typedef struct {
    NnUint position;
    NnUint batchSize;
    NnUint nodeIndex;
    NnUint stageIndex;
    NnUint execUs;
    NnUint syncUs;
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
};

typedef struct {
    NnUint magic;      // 'DLBM'
    NnUint version;    // 2
    NnUint flags;      // LlmBootstrapFlags
    NnUint benchmarkEnabled; // 0/1, enables executor timer on workers
    NnUint enablePlanBarrier; // 0/1, enables plan barrier for online migration
    NnUint enableStageFullWeights; // 0/1, enables stage full residency (full weights and buffers)
    NnUint enableKvRedundancyDuringMigration; // 0/1, keeps KV redundancy enabled during migration
    NnUint enableKvAggregate; // 0/1, builds KV aggregate pipes (KC/VC)
    NnUint runtimeRedundantBoundaryLayers; // boundary span in layers for runtime redundant plan
    NnUint runtimeActiveSegEnabled; // 0/1 default primary segment gate
    NnUint runtimeRedundantSegEnabled; // 0/1 default redundant segment gate
    NnUint maxSeqLen;  // forwarded from root args
    NnUint syncType;   // NnFloatType (as NnUint)
    NnUint modelPathLen; // bytes including '\0' if present
    NnUint ratiosLen;    // bytes including '\0' if present
    NnUint primarySkipLayersLen; // bytes including '\0' if present
} LlmBootstrapPacket;

static constexpr NnUint LLM_BOOTSTRAP_MAGIC = 0x4d424c44u; // 'DLBM' little-endian
static constexpr NnUint LLM_BOOTSTRAP_VERSION = 8u;

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
    bool hasAsyncKvCollector() const;
    bool tryPopAsyncKvRow(RootKvAggRowPacket &packet);
    KvTransferSubmitStatus submitBoundaryKvTransferDetailed(NnUint layerIndex, NnUint position, const std::vector<float> &kRow, const std::vector<float> &vRow);
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
    std::vector<LlmPerfPacket> lastPerf;
    NnUint lastPlanCmdSeqSent = 0u;
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
    int migrationStageStartLayer = -1;
    int migrationStageEndLayer = -1;
    int migrationLayerCount = 1;
    bool ppMigrationEnabled = false;
    bool migrationBatchSubmitted = false;
    bool migrationLayerListPinnedByEnv = false;
    int boundaryLayerForMigration = -1;
    bool migrationAckSeen = false;
    int migrationAckPos = -1;
    int migrationAckLayer = -1;
    unsigned long long lastPpPlanCacheSeqApplied = 0ull;
    NnUint migrationFromNodeIndex = 0u;
    NnUint nextStageRootNode = (NnUint)-1;
    int kvAckSocketIndex = -1;
public:
    RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network, const NnUnevenPartitionPlan* plan, bool profileEnabled, bool ppMigrationEnabled);
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
    void setToken(NnUint batchIndex, NnUint token);
    void setRuntimeLayerGate(bool enablePrimarySegments, bool enableRedundantSegments);
    void setPrimaryLayerEnabled(NnUint layerIndex, bool enabled);
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
private:
    float *positionPipe;
    NnNetExecution *execution;
    NnNetwork *network;
    NnUint localNodeIndex = 0u;
    LlmControlPacket controlPacket;
    NnUint lastPlanCmdSeqRecv = 0u;
    std::deque<PendingKvTransferItem> pendingKvTransfers;
    std::vector<LlmKvAckPacket> pendingKvAcks;
    std::deque<LlmLayerSwitchPacket> pendingLayerSwitches;
public:
    WorkerLlmInference(NnNetExecution *execution, NnNetwork *network, NnUint localNodeIndex);
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
