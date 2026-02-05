#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "nn-executor.hpp"
#include "nn-core.hpp"

#include <atomic>
#include <vector>

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

class NnNetwork {
private:
    int *sockets;
    NnSize *sentBytes;
    NnSize *recvBytes;

public:
    static std::unique_ptr<NnNetwork> serve(int port);
    static std::unique_ptr<NnNetwork> connect(NnUint nSockets, char **hosts, NnUint *ports);

    NnUint nSockets;

    NnNetwork(std::vector<NnSocket> *sockets);
    ~NnNetwork();

    void setTurbo(bool enabled);
    void write(const NnUint socketIndex, const void *data, const NnSize size);
    void read(const NnUint socketIndex, void *data, const NnSize size);
    void writeAck(const NnUint socketIndex);
    void readAck(const NnUint socketIndex);
    bool tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts);
    void writeMany(NnUint n, NnSocketIo *ios);
    void writeAll(const void *data, NnSize size);
    void readMany(NnUint n, NnSocketIo *ios);
    void getStats(NnSize *sentBytes, NnSize *recvBytes);
    void sendToNode(NnUint targetNodeIndex, NnUint myNodeIndex, const void* data, NnSize size);
    void recvFromNode(NnUint sourceNodeIndex, NnUint myNodeIndex, void* data, NnSize size);
    int getSocketIndexForNode(NnUint targetNodeIndex, NnUint myNodeIndex) const;
    void resetStats();
    
};

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
public:
    NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, const NnUnevenPartitionPlan *plan = nullptr);
    ~NnNetworkNodeSynchronizer() override {};
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
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
