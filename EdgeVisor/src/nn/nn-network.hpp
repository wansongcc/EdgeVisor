#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "nn-executor.hpp"
#include "nn-core.hpp"

#include <atomic>
#include <vector>
#include <string>
#include <cstdint>

#define ROOT_SOCKET_INDEX 0

void initSockets();
void cleanupSockets();
int acceptSocket(int serverSocket);
void setReuseAddr(int socket);
void writeSocket(int socket, const void* data, NnSize size);
void readSocket(int socket, void* data, NnSize size);
int createServerSocket(int port);
void destroySocket(int serverSocket);

class NnConnectionSocketException : public std::runtime_error {
public:
    NnConnectionSocketException(const std::string message);
};

class NnTransferSocketException : public std::runtime_error {
public:
    int code;
    NnTransferSocketException(int code, const std::string message);
};

class NnSocket {
public:
    int fd;
    NnSocket();
    NnSocket(int fd);
    ~NnSocket();
    void assign(int fd);
    int release();
};

struct NnSocketIo {
    NnUint socketIndex;
    const void *data;
    NnSize size;
};

struct NnCommProfileStats {
    std::uint64_t sendCalls = 0;
    std::uint64_t recvCalls = 0;
    std::uint64_t sendBytes = 0;
    std::uint64_t recvBytes = 0;
    std::uint64_t smallSendCalls = 0;
    std::uint64_t smallRecvCalls = 0;
    std::uint64_t smallSendBytes = 0;
    std::uint64_t smallRecvBytes = 0;
    std::uint64_t sendEagain = 0;
    std::uint64_t recvEagain = 0;
    std::uint64_t ackWrites = 0;
    std::uint64_t ackReads = 0;
    std::uint64_t writeManyCalls = 0;
    std::uint64_t readManyCalls = 0;
    std::uint64_t syncSteps = 0;
};

class NnNetwork {
private:
    int *sockets;
    NnUint *peerNodeBySocket;
    NnSize *sentBytes;
    NnSize *recvBytes;
    bool commProfileEnabled = false;
    std::uint64_t commProfileSeq = 0;
    std::atomic<std::uint64_t> commSendCalls{0};
    std::atomic<std::uint64_t> commRecvCalls{0};
    std::atomic<std::uint64_t> commSendBytes{0};
    std::atomic<std::uint64_t> commRecvBytes{0};
    std::atomic<std::uint64_t> commSmallSendCalls{0};
    std::atomic<std::uint64_t> commSmallRecvCalls{0};
    std::atomic<std::uint64_t> commSmallSendBytes{0};
    std::atomic<std::uint64_t> commSmallRecvBytes{0};
    std::atomic<std::uint64_t> commSendEagain{0};
    std::atomic<std::uint64_t> commRecvEagain{0};
    std::atomic<std::uint64_t> commAckWrites{0};
    std::atomic<std::uint64_t> commAckReads{0};
    std::atomic<std::uint64_t> commWriteManyCalls{0};
    std::atomic<std::uint64_t> commReadManyCalls{0};
    std::atomic<std::uint64_t> commSyncSteps{0};

    void recordCommSend(NnSize size);
    void recordCommRecv(NnSize size);
    void recordCommSendEagain();
    void recordCommRecvEagain();
    NnCommProfileStats getCommProfileStats() const;
    void resetCommProfileStats();
    void printCommProfile(const char *label, const NnCommProfileStats &stats, bool includeSeq);

public:
    static std::unique_ptr<NnNetwork> serve(int port);
    static std::unique_ptr<NnNetwork> connect(NnUint nSockets, char **hosts, NnUint *ports);

    NnUint nSockets;

    NnNetwork(std::vector<NnSocket> *sockets, std::vector<NnUint> *peerNodeBySocket);
    ~NnNetwork();

    void setTurbo(bool enabled);
    void write(const NnUint socketIndex, const void *data, const NnSize size);
    void read(const NnUint socketIndex, void *data, const NnSize size);
    void writeAck(const NnUint socketIndex);
    void readAck(const NnUint socketIndex);
    void readAckWithTimeout(const NnUint socketIndex, unsigned long timeoutMs);
    // ACK packet followed by a fixed-size payload.
    void writeAckWithPayload(const NnUint socketIndex, const void *payload, const NnSize payloadSize);
    void readAckWithPayload(const NnUint socketIndex, void *payload, const NnSize payloadSize);
    bool tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts);
    void writeMany(NnUint n, NnSocketIo *ios);
    void writeAll(const void *data, NnSize size);
    void readMany(NnUint n, NnSocketIo *ios);
    void getStats(NnSize *sentBytes, NnSize *recvBytes);
    void sendToNode(NnUint targetNodeIndex, NnUint myNodeIndex, const void* data, NnSize size);
    bool isCommProfileEnabled() const;
    void recordSyncStepComplete();
    void recvFromNode(NnUint sourceNodeIndex, NnUint myNodeIndex, void* data, NnSize size);
    int getSocketIndexForNode(NnUint targetNodeIndex, NnUint myNodeIndex) const;
    void resetStats();
    
};

// Fixed-size per-layer compute timing message sent from node -> stage root.
// Times are in microseconds and cover CPU compute only (no comm).
typedef struct {
    NnUint magic;     // 'DLPR'
    NnUint version;   // 1
    NnUint stageIndex;
    NnUint nodeIndex;
    NnUint layerIndex;
    NnUint attnUs;
    NnUint ffnUs;
} NnLayerPerfMsg;

static constexpr NnUint NN_LAYER_PERF_MAGIC = 0x52504c44u; // 'DLPR' little-endian
static constexpr NnUint NN_LAYER_PERF_VERSION = 1u;

// Shared snapshot format for cross-process query (mmap file written by stage root).
typedef struct {
    NnUint magic;         // 'DLPS'
    NnUint version;       // 1
    NnUint stageIndex;
    NnUint rootNodeIndex;
    NnUint nLayers;
    NnUint nStageNodes;
    NnUint reserved0;
    NnUint reserved1;
    unsigned long long epoch; // increments on each layer-end publish
} NnLayerPerfSnapshotHeader;

static constexpr NnUint NN_LAYER_PERF_SNAPSHOT_MAGIC = 0x53504c44u; // 'DLPS'
static constexpr NnUint NN_LAYER_PERF_SNAPSHOT_VERSION = 1u;

class NnNetworkNodeSynchronizer : public NnNodeSynchronizer {
private:
    NnNetwork *network;
    NnNetExecution *execution;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    const NnUnevenPartitionPlan *plan;
    const NnStageConfig* myStage = nullptr;

    struct SyncProfileSlot {
        std::atomic<NnUint> epoch{0u};
        std::atomic<NnUint> arrived{0u};
        std::atomic<NnUint> done{0u};
        std::atomic<long long> startUs{0};
        std::atomic<long long> endUs{0};

        SyncProfileSlot() = default;
        SyncProfileSlot(const SyncProfileSlot &other)
            : epoch(other.epoch.load(std::memory_order_relaxed)),
              arrived(other.arrived.load(std::memory_order_relaxed)),
              done(other.done.load(std::memory_order_relaxed)),
              startUs(other.startUs.load(std::memory_order_relaxed)),
              endUs(other.endUs.load(std::memory_order_relaxed)) {}

        SyncProfileSlot &operator=(const SyncProfileSlot &other) {
            if (this == &other) return *this;
            epoch.store(other.epoch.load(std::memory_order_relaxed), std::memory_order_relaxed);
            arrived.store(other.arrived.load(std::memory_order_relaxed), std::memory_order_relaxed);
            done.store(other.done.load(std::memory_order_relaxed), std::memory_order_relaxed);
            startUs.store(other.startUs.load(std::memory_order_relaxed), std::memory_order_relaxed);
            endUs.store(other.endUs.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }

        SyncProfileSlot(SyncProfileSlot &&other) noexcept
            : SyncProfileSlot(static_cast<const SyncProfileSlot &>(other)) {}

        SyncProfileSlot &operator=(SyncProfileSlot &&other) noexcept {
            return (*this = static_cast<const SyncProfileSlot &>(other));
        }
    };

    std::vector<NnUint> syncProfileBaseBySegment;
    std::vector<SyncProfileSlot> syncProfileSlots;

    // Thread barrier slots for layer profiling (separate from syncProfileSlots to avoid interference).
    std::vector<NnUint> layerPerfBaseBySegment;
    std::vector<SyncProfileSlot> layerPerfSlots;

    // Layer profiling (stage-local)
    bool layerProfileEnabled = false;
    struct SegmentMeta {
        NnUint layerIndex;
        NnByte kind; // 0 other, 1 attn, 2 ffn
    };
    std::vector<SegmentMeta> segmentMeta;
    std::vector<int> stageLocalIndexByGlobalNode;
    std::vector<std::vector<NnLayerPerfMsg>> lastLayerPerfByLayer; // [layer][stageLocalNode]

    // Cross-process query interface: stage root publishes snapshots to an mmap file.
    bool layerPerfPrintEnabled = false;
    std::string layerPerfPath;
    int layerPerfFd = -1;
    void *layerPerfMap = nullptr;
    NnSize layerPerfMapSize = 0;
    unsigned long long layerPerfEpoch = 0ull;

    void maybeInitLayerPerfSnapshot();
    void publishLayerPerfSnapshot(NnUint layerIndex);
public:
    NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, const NnUnevenPartitionPlan *plan = nullptr, bool layerProfileEnabled = false);
        ~NnNetworkNodeSynchronizer() override;
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
    void onSyncStepComplete(NnUint segmentIndex) override;
};

class NnRootConfigWriter {
private:
    NnNetwork *network;
public:
    NnRootConfigWriter(NnNetwork *network);
    void writeNet(NnUint socketIndex, NnNetConfig *config);
    void writeNode(NnUint socketIndex, NnNodeConfig *config);
    void writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs);
};

class NnWorkerConfigReader {
private:
    NnNetwork *network;
public:
    NnWorkerConfigReader(NnNetwork *network);
    NnNetConfig readNet();
    NnNodeConfig readNode();
};

class NnRootWeightLoader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnUint nNodes;
    NnByte *temp;
    NnSize tempSize;
public:
    NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnUint nNodes);
    ~NnRootWeightLoader();
    void writeWeight(NnUint nodeIndex, const char *opName, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight);
    NnSize loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadRowMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight);
    NnSize loadColMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight);
    void finish();
private:
    void allocate(NnSize size);};

class NnWorkerWeightReader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnByte *temp;
    NnUint tempSize;
public:
    NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network);
    ~NnWorkerWeightReader();
    void read();
private:
    void allocate(NnUint size);
};

#endif
