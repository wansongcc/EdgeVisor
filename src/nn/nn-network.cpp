#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h> // For inet_addr and other functions
#include <windows.h>  // For SSIZE_T
typedef SSIZE_T ssize_t;
#else
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>  // for getaddrinfo
#endif
#include "nn-network.hpp"
#include "nn-core.hpp"
#include <cassert>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <set>
#include <chrono>
#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>
#include <sched.h>
#endif
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

#define ACK 23571114
#define MAX_CHUNK_SIZE 65536

static inline long long nowMsSteady() {
    return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static inline unsigned long getIoTimeoutMs() {
    // 0 means disabled (legacy behavior: spin until done).
    static std::atomic<unsigned long> cached{0ul};
    static std::atomic<bool> inited{false};
    if (!inited.load(std::memory_order_acquire)) {
        unsigned long v = 0ul;
        if (const char *p = std::getenv("DLLAMA_IO_TIMEOUT_MS")) {
            try {
                v = std::stoul(std::string(p));
            } catch (...) {
                v = 0ul;
            }
        }
        cached.store(v, std::memory_order_release);
        inited.store(true, std::memory_order_release);
    }
    return cached.load(std::memory_order_acquire);
}

static inline bool isEagainError() {
    #ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
    #else
    return SOCKET_LAST_ERRCODE == EAGAIN;
    #endif
}

static inline unsigned int getIoEagainYieldEvery() {
    // 0 disables cooperative yielding (legacy behavior).
    static std::atomic<unsigned int> cached{256u};
    static std::atomic<bool> inited{false};
    if (!inited.load(std::memory_order_acquire)) {
        unsigned int v = 256u;
        if (const char *p = std::getenv("DLLAMA_IO_EAGAIN_YIELD_EVERY")) {
            try {
                v = (unsigned int)std::stoul(std::string(p));
            } catch (...) {
                v = 256u;
            }
        }
        cached.store(v, std::memory_order_release);
        inited.store(true, std::memory_order_release);
    }
    return cached.load(std::memory_order_acquire);
}

static inline unsigned int getIoEagainSleepUs() {
    // 0 disables sleeping.
    static std::atomic<unsigned int> cached{0u};
    static std::atomic<bool> inited{false};
    if (!inited.load(std::memory_order_acquire)) {
        unsigned int v = 0u;
        if (const char *p = std::getenv("DLLAMA_IO_EAGAIN_SLEEP_US")) {
            try {
                v = (unsigned int)std::stoul(std::string(p));
            } catch (...) {
                v = 0u;
            }
        }
        cached.store(v, std::memory_order_release);
        inited.store(true, std::memory_order_release);
    }
    return cached.load(std::memory_order_acquire);
}

static inline void backoffOnEagain(unsigned int &spinCount) {
    const unsigned int yieldEvery = getIoEagainYieldEvery();
    if (yieldEvery == 0u) return;
    spinCount++;
    if ((spinCount % yieldEvery) != 0u) return;

#ifdef _WIN32
    Sleep(0);
#else
    sched_yield();
    const unsigned int sleepUs = getIoEagainSleepUs();
    if (sleepUs > 0u) {
        usleep(sleepUs);
    }
#endif
}

static NnUint getSplitTotal(const NnDimSplit& split, NnUint nNodes) {
    if (!split.lengths) return 0;
    NnUint sum = 0;
    for(NnUint i=0; i<nNodes; ++i) sum += split.lengths[i];
    return sum;
}

static NnUint getGroupRootIndex(const NnStageConfig* stage) {
    if (stage != nullptr) {
        // 如果是在某个 Stage 内部同步，Root 是该 Stage 的第一个节点
        return stage->rootNodeIndex; 
    }
    // 如果是全局同步，Root 是全局 Node 0
    return 0;
}

static inline void setNonBlocking(int socket, bool enabled) {
#ifdef _WIN32
    u_long mode = enabled ? 1 : 0;
    if (ioctlsocket(socket, FIONBIO, &mode) != 0) {
        throw std::runtime_error("Error setting socket to non-blocking");
    }
#else
    int flags = fcntl(socket, F_GETFL, 0);
    if (enabled) {
        flags |= O_NONBLOCK;
    } else {
        flags = flags & (~O_NONBLOCK);
    }
    if (fcntl(socket, F_SETFL, flags) < 0)
        throw std::runtime_error("Error setting socket to non-blocking");
#endif
}

static inline void setNoDelay(int socket) {
    int flag = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int)) < 0)
        throw std::runtime_error("Error setting socket to no-delay");
}

static inline void setQuickAck(int socket) {
#ifndef _WIN32
#ifdef TCP_QUICKACK
    int value = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_QUICKACK, (char*)&value, sizeof(int)) < 0)
        throw std::runtime_error("Error setting quick ack");
#endif
#endif
}

void setReuseAddr(int socket) {
    int opt = 1;
    #ifdef _WIN32
    int iresult = setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
    if (iresult == SOCKET_ERROR) {
        closesocket(socket);
        throw std::runtime_error("setsockopt failed: " + std::to_string(WSAGetLastError()));
    }
    #else
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        close(socket);
        throw std::runtime_error("setsockopt failed: " + std::string(strerror(errno)));
    }
    #endif
}

static NnUint getUnevenSliceSize(const NnUnevenPartitionPlan *plan, NnUint totalSize, NnUint nodeIndex) {
    if (!plan) return totalSize / plan->nNodes; // Fallback

    // 尝试匹配 Vocab
    NnUint vocabTotal = getSplitTotal(plan->vocabSplit, plan->nNodes);
    if (vocabTotal > 0 && totalSize % vocabTotal == 0) {
        return plan->vocabSplit.lengths[nodeIndex] * (totalSize / vocabTotal);
    }
    // 尝试匹配 FFN
    NnUint ffnTotal = getSplitTotal(plan->ffnSplit, plan->nNodes);
    if (ffnTotal > 0 && totalSize % ffnTotal == 0) {
        return plan->ffnSplit.lengths[nodeIndex] * (totalSize / ffnTotal);
    }
    // 尝试匹配 Heads
    NnUint headTotal = getSplitTotal(plan->headSplit, plan->nNodes);
    if (headTotal > 0 && totalSize % headTotal == 0) {
        return plan->headSplit.lengths[nodeIndex] * (totalSize / headTotal);
    }
    
    // 默认均匀
    return totalSize / plan->nNodes;
}

static NnUint getUnevenSliceOffset(const NnUnevenPartitionPlan *plan, NnUint totalSize, NnUint nodeIndex) {
    if (!plan) return (totalSize / plan->nNodes) * nodeIndex;

    // (同上逻辑，使用 starts 而不是 lengths)
    NnUint vocabTotal = getSplitTotal(plan->vocabSplit, plan->nNodes);
    if (vocabTotal > 0 && totalSize % vocabTotal == 0) {
        return plan->vocabSplit.starts[nodeIndex] * (totalSize / vocabTotal);
    }
    // ... FFN, Heads ...
    
    return (totalSize / plan->nNodes) * nodeIndex;
}

static NnUint getSplitTotalForStageNodes(const NnDimSplit& split, const NnStageConfig* stage, NnUint nNodes) {
    if (!split.lengths) return 0;
    if (!stage) return getSplitTotal(split, nNodes);
    NnUint sum = 0;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        sum += split.lengths[stage->nodeIndices[i]];
    }
    return sum;
}

static void fillUnevenSlices(const NnUnevenPartitionPlan *plan, NnUint nNodes, NnSize totalBytes, 
                             std::vector<NnSize>& offsets, std::vector<NnSize>& sizes,
                             NnFloatType floatType,
                             const NnStageConfig* stageForSplit,
                             NnSliceTag forcedTag,
                             NnUint totalElements = 0) {
    bool matchFound = false;
    bool stackedByNode = false;

    // Default to zeros for nodes not in stageForSplit
    for (NnUint i = 0; i < nNodes; ++i) {
        offsets[i] = 0;
        sizes[i] = 0;
    }

    if (plan && plan->nNodes == nNodes) {
        // ----------------------------------------------------
        // [PP Fix] ZQ 等“按 node 堆叠”的 pipe：totalElements == dim * nNodes
        // 对这种 pipe，slice 应该按【全局 nodeIndex】映射到固定 slot：
        //   offset = nodeIndex * (totalBytes / nNodes)
        //   size   = totalBytes / nNodes
        // 不能用 dim/head split 去匹配，否则会把布局解释错。
        // ----------------------------------------------------
        if (stageForSplit != nullptr && plan->nStages > 0 && totalElements > 0) {
            NnUint dimTotal = getSplitTotalForStageNodes(plan->dimSplit, stageForSplit, nNodes);
            if (dimTotal > 0 && totalElements == dimTotal * nNodes) {
                stackedByNode = true;
            }
        }

        if (stackedByNode) {
            if (totalBytes % nNodes != 0) {
                // Should not happen, but keep safe fallback.
            } else {
                NnSize slotBytes = totalBytes / nNodes;
                for (NnUint k = 0; k < stageForSplit->nNodes; ++k) {
                    NnUint node = stageForSplit->nodeIndices[k];
                    offsets[node] = (NnSize)node * slotBytes;
                    sizes[node] = slotBytes;
                }
                matchFound = true;
            }
        }

        // Helper lambda to check if a specific split configuration matches the current buffer size
        auto tryMatch = [&](const NnDimSplit& split, bool allowMultiplier, const char* name) -> bool {
            (void)name;
            if (stackedByNode) return false;
            // In PP mode, the split arrays are stage-local but stored across all nodes.
            // So we must compute totals only for the relevant stage.
            NnUint totalUnits = getSplitTotalForStageNodes(split, stageForSplit, nNodes);
            
            // [Fix] Priority: Match by Element Count first (if provided)
            if (totalElements > 0) {
                if (totalUnits > 0 && totalElements % totalUnits == 0) {
                    // Found match by elements!
                    NnUint multiplier = totalElements / totalUnits; // e.g. HeadDim

                    // [Fix] Prevent aggressive matching for ZQ pipe (dim * nNodes)
                    if (!allowMultiplier && multiplier != 1u) return false;

                    if (stageForSplit) {
                        for (NnUint k = 0; k < stageForSplit->nNodes; ++k) {
                            NnUint node = stageForSplit->nodeIndices[k];
                            NnUint offElems = split.starts[node] * multiplier;
                            NnUint lenElems = split.lengths[node] * multiplier;
                            offsets[node] = getBytes(floatType, offElems);
                            sizes[node] = getBytes(floatType, lenElems);
                        }
                    } else {
                        for (NnUint node = 0; node < nNodes; ++node) {
                            NnUint offElems = split.starts[node] * multiplier;
                            NnUint lenElems = split.lengths[node] * multiplier;
                            offsets[node] = getBytes(floatType, offElems);
                            sizes[node] = getBytes(floatType, lenElems);
                        }
                    }
                    return true;
                }
            }

            // Fallback: Match by Byte Count (Original Logic)
            // Check if totalBytes is exactly divisible by the total units (e.g. total heads)
            if (totalUnits > 0 && totalBytes % totalUnits == 0) {
                // If we are here, totalElements was 0 or didn't match.
                // If allowMultiplier is false, we should probably also enforce strict matching?
                // But byte matching is ambiguous. Let's assume if totalElements is not provided,
                // we rely on byte matching which implies multiplier=1 usually (unless bytes per unit is large).
                // For safety, let's skip this check if allowMultiplier is false and we suspect a multiplier.
                // But we don't know the multiplier here.
                
                NnSize bytesPerUnit = totalBytes / totalUnits;

                if (stageForSplit) {
                    for (NnUint k = 0; k < stageForSplit->nNodes; ++k) {
                        NnUint node = stageForSplit->nodeIndices[k];
                        offsets[node] = (NnSize)split.starts[node] * bytesPerUnit;
                        sizes[node] = (NnSize)split.lengths[node] * bytesPerUnit;
                    }
                } else {
                    for (NnUint node = 0; node < nNodes; ++node) {
                        offsets[node] = (NnSize)split.starts[node] * bytesPerUnit;
                        sizes[node] = (NnSize)split.lengths[node] * bytesPerUnit;
                    }
                }
                return true;
            }
            return false;
        };

        // Forced matching (used to disambiguate headSplit vs kvHeadSplit for kvDim-sized pipes)
        if (!matchFound && forcedTag != NN_SLICE_AUTO) {
            if (forcedTag == NN_SLICE_HEAD) {
                matchFound = tryMatch(plan->headSplit, true, "head(forced)");
            } else if (forcedTag == NN_SLICE_KV_HEAD) {
                matchFound = tryMatch(plan->kvHeadSplit, true, "kvhead(forced)");
            } else if (forcedTag == NN_SLICE_FFN) {
                matchFound = tryMatch(plan->ffnSplit, false, "ffn(forced)");
            } else if (forcedTag == NN_SLICE_DIM) {
                matchFound = tryMatch(plan->dimSplit, false, "dim(forced)");
            } else if (forcedTag == NN_SLICE_VOCAB) {
                matchFound = tryMatch(plan->vocabSplit, false, "vocab(forced)");
            }
        }

        // Priority order for matching:
        // 1. Vocab (Logits)
        if (!matchFound) matchFound = tryMatch(plan->vocabSplit, false, "vocab");
        // 2. FFN
        if (!matchFound) matchFound = tryMatch(plan->ffnSplit, false, "ffn");
        // 3. Dim
        if (!matchFound) matchFound = tryMatch(plan->dimSplit, false, "dim");
        // 4. Heads
        if (!matchFound) matchFound = tryMatch(plan->headSplit, true, "head");
        // 5. KV Heads
        if (!matchFound) matchFound = tryMatch(plan->kvHeadSplit, true, "kvhead");
    }

    // Fallback: Uniform partitioning
    if (!matchFound) {
        if (stageForSplit) {
            // In PP mode, prefer global node slots when possible (e.g., ZQ-like pipes).
            if (totalBytes % nNodes == 0) {
                NnSize slotBytes = totalBytes / nNodes;
                for (NnUint k = 0; k < stageForSplit->nNodes; ++k) {
                    NnUint node = stageForSplit->nodeIndices[k];
                    offsets[node] = (NnSize)node * slotBytes;
                    sizes[node] = slotBytes;
                }
            } else {
                NnUint m = stageForSplit->nNodes;
                if (m == 0) return;
                NnSize avgBytes = totalBytes / m;
                NnSize currentOffset = 0;
                for (NnUint k = 0; k < m; ++k) {
                    NnUint node = stageForSplit->nodeIndices[k];
                    offsets[node] = currentOffset;
                    if (k + 1 == m) {
                        sizes[node] = totalBytes - currentOffset;
                    } else {
                        sizes[node] = avgBytes;
                    }
                    currentOffset += sizes[node];
                }
            }
        } else {
            NnSize avgBytes = totalBytes / nNodes;
            for (NnUint i = 0; i < nNodes; ++i) {
                sizes[i] = avgBytes;
                offsets[i] = i * avgBytes;
            }
            // Fix rounding error for the last node
            offsets[nNodes - 1] = (nNodes - 1) * avgBytes;
            sizes[nNodes - 1] = totalBytes - offsets[nNodes - 1];
        }
    }
}

// ---------------------------------------------------------
// 通信数据打印开关（载荷字节/哈希/同步诊断）
// 默认关闭：避免刷屏并减少性能影响。
// 如需开启：编译时增加 -DNN_NETWORK_COMM_DATA_LOG=1
// ---------------------------------------------------------
#ifndef NN_NETWORK_COMM_DATA_LOG
#define NN_NETWORK_COMM_DATA_LOG 0
#endif

static void printBytes(const char* prefix, const void* data, NnSize size) {
#if NN_NETWORK_COMM_DATA_LOG
    const unsigned char* bytes = (const unsigned char*)data;
    printf("%s size=%zu bytes=", prefix, size);
    for (size_t i = 0; i < std::min((size_t)size, (size_t)16); ++i) {
        printf("%02x ", bytes[i]);
    }
    printf("\n");
#else
    (void)prefix;
    (void)data;
    (void)size;
#endif
}

[[maybe_unused]] static inline std::uint64_t fnv1a64(const void* data, NnSize size) {
    const std::uint8_t* p = (const std::uint8_t*)data;
    std::uint64_t h = 1469598103934665603ull;
    for (NnSize i = 0; i < size; ++i) {
        h ^= (std::uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

void writeSocket(int socket, const void *data, NnSize size) {
    printBytes("DEBUG: writeSocket", data, size);
    unsigned int eagainSpins = 0u;
    while (size > 0) {
        ssize_t s = send(socket, (const char*)data, size, 0);
        if (s < 0) {
            if (isEagainError()) {
                backoffOnEagain(eagainSpins);
                continue;
            }
            throw NnTransferSocketException(0, "Error writing to socket");
        } else if (s == 0) {
            throw NnTransferSocketException(0, "Socket closed");
        }
        eagainSpins = 0u;
        size -= s;
        data = (const char*)data + s;
    }
}

static inline bool tryReadSocket(int socket, void *data, NnSize size, unsigned long maxAttempts) {
    // maxAttempts = 0 means infinite attempts
    NnSize s = size;
    unsigned int eagainSpins = 0u;
    while (s > 0) {
        ssize_t r = recv(socket, (char*)data, s, 0);
        if (r < 0) {
            if (isEagainError()) {
                backoffOnEagain(eagainSpins);
                if (s == size && maxAttempts > 0) {
                    maxAttempts--;
                    if (maxAttempts == 0) {
                        return false;
                    }
                }
                continue;
            }
            throw NnTransferSocketException(0, "Error reading from socket");
        } else if (r == 0) {
            throw NnTransferSocketException(0, "Socket closed");
        }
        printBytes("DEBUG: readSocket", data, r);
        eagainSpins = 0u;
        data = (char*)data + r;
        s -= r;
    }
    return true;
}

void readSocket(int socket, void *data, NnSize size) {
    if (!tryReadSocket(socket, data, size, 0)) {
        throw std::runtime_error("Error reading from socket");
    }
}

static void readAckPacket(int socket) {
    NnUint packet;
    readSocket(socket, &packet, sizeof(packet));
    if (packet != ACK)
        throw std::runtime_error("Invalid ack packet");
}

static void writeAckPacket(int socket) {
    NnUint packet = ACK;
    writeSocket(socket, &packet, sizeof(packet));
}

static constexpr NnUint WORKER_PEER_HELLO_MAGIC = 0x314b5057u; // 'WPK1'

static void exchangeWorkerPeerIndex(int socket, NnUint myWorkerIndex, NnUint *peerWorkerIndex) {
    NnUint out[2] = {WORKER_PEER_HELLO_MAGIC, myWorkerIndex};
    writeSocket(socket, out, sizeof(out));

    NnUint in[2] = {0u, 0u};
    readSocket(socket, in, sizeof(in));
    if (in[0] != WORKER_PEER_HELLO_MAGIC) {
        throw std::runtime_error("Invalid worker-peer hello magic");
    }
    *peerWorkerIndex = in[1];
}

static inline NnUint peerWorkerToSocketIndex(NnUint myWorkerIndex, NnUint peerWorkerIndex) {
    // socket[0] is root. Worker-peer sockets use compressed worker index space
    // where local worker index is removed:
    //   peer < my  -> socket = peer + 1
    //   peer > my  -> socket = peer
    return (peerWorkerIndex < myWorkerIndex) ? (peerWorkerIndex + 1u) : peerWorkerIndex;
}

static inline int connectSocket(char *host, int port) {
    struct addrinfo hints;
    struct addrinfo *addr = NULL;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char portStr[11];
    snprintf(portStr, sizeof(portStr), "%d", port);

    int addrinfoError = getaddrinfo(host, portStr, &hints, &addr);
    if (addrinfoError != 0 || addr == NULL) {
        printf("Cannot resolve target %s (%s)\n", host, gai_strerror(addrinfoError));
        throw NnConnectionSocketException("Cannot resolve address");
    }

    int sock = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (sock < 0)
        throw std::runtime_error("Cannot create socket");

    int connectResult = ::connect(sock, addr->ai_addr, addr->ai_addrlen);
    if (connectResult != 0) {
        printf("Cannot connect to %s:%d (%s)\n", host, port, SOCKET_LAST_ERROR);
        throw NnConnectionSocketException("Cannot connect");
    }

    setNoDelay(sock);
    setQuickAck(sock);
    return sock;
}

int createServerSocket(int port) {
    const char *host = "0.0.0.0";
    struct sockaddr_in serverAddr;

    int serverSocket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (serverSocket < 0)
        throw std::runtime_error("Cannot create socket");
    setReuseAddr(serverSocket);

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = inet_addr(host);

    int bindResult;
    #ifdef _WIN32
    bindResult = bind(serverSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr));
    if (bindResult == SOCKET_ERROR) {
        int error = WSAGetLastError();
        closesocket(serverSocket);
        throw std::runtime_error("Cannot bind port: " + std::to_string(error));
    }
    #else
    bindResult = bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (bindResult < 0) {
        close(serverSocket);
        throw std::runtime_error("Cannot bind port: " + std::string(strerror(errno)));
    }
    #endif

    int listenResult = listen(serverSocket, SOMAXCONN);
    if (listenResult != 0) {
        #ifdef _WIN32
        closesocket(serverSocket);
        throw std::runtime_error("Cannot listen on port: " + std::to_string(WSAGetLastError()));
        #else
        close(serverSocket);
        throw std::runtime_error("Cannot listen on port: " + std::string(strerror(errno)));
        #endif
    }

    printf("Listening on %s:%d...\n", host, port);

    setNoDelay(serverSocket);
    setQuickAck(serverSocket);
    return serverSocket;
}

void destroySocket(int serverSocket) {
    shutdown(serverSocket, 2);
    #ifdef _WIN32
    closesocket(serverSocket);
    #else
    close(serverSocket);
    #endif
}

int acceptSocket(int serverSocket) {
    struct sockaddr_in clientAddr;
    socklen_t clientAddrSize = sizeof(clientAddr);
    int clientSocket = ::accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrSize);
    if (clientSocket < 0)
        throw std::runtime_error("Error accepting connection");
    setNoDelay(clientSocket);
    setQuickAck(clientSocket);
    return clientSocket;
}

void initSockets() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(WSAGetLastError()));
    }
#endif
}

void cleanupSockets() {
#ifdef _WIN32
    WSACleanup();
#endif
}

NnConnectionSocketException::NnConnectionSocketException(const std::string message)
    : std::runtime_error(message)
{}

NnTransferSocketException::NnTransferSocketException(int code, const std::string message)
    : code(code), std::runtime_error(message)
{}

NnSocket::NnSocket() {
    this->fd = -1;
}

NnSocket::NnSocket(int fd) : NnSocket() {
    assign(fd);
}

NnSocket::~NnSocket() {
    if (this->fd >= 0)
        destroySocket(this->fd);
}

void NnSocket::assign(int fd) {
    assert(this->fd == -1);
    assert(fd >= 0);
    this->fd = fd;
}

int NnSocket::release() {
    assert(this->fd >= 0);
    int fd = this->fd;
    this->fd = -1;
    return fd;
}

std::unique_ptr<NnNetwork> NnNetwork::serve(int port) {
    NnSocket socketSocket(createServerSocket(port));

    NnUint nSockets;
    NnUint nodeIndex;
    int rootSocketFd = acceptSocket(socketSocket.fd);
    NnSocket rootSocket(rootSocketFd);
    printf("⭕ The root node has connected\n");

    readSocket(rootSocketFd, &nSockets, sizeof(nSockets));
    NnUint nNodes = nSockets - 1; // nSockets - 1 root node
    printf("⭕ peerWorkers: %d\n", nNodes);
    readSocket(rootSocketFd, &nodeIndex, sizeof(nodeIndex));
    printf("⭕ NodeIndex: %d\n", nodeIndex);

    std::vector<NnSocket> sockets(nSockets);
    std::vector<NnUint> peerNodeBySocket(nSockets, 0u);
    std::vector<bool> slotAssigned(nSockets, false);
    sockets[0].assign(rootSocket.release());
    peerNodeBySocket[0] = 0u;
    slotAssigned[0] = true;

    printf("⭕ Socket[0]: accepted root node\n");
    std::vector<std::unique_ptr<char[]>> hosts(nNodes);
    std::vector<int> ports(nNodes);

    NnUint hostLen;
    for (NnUint i = 0; i < nNodes; i++) {
        readSocket(rootSocketFd, &hostLen, sizeof(hostLen));

        std::unique_ptr<char[]> host(new char[hostLen]);
        readSocket(rootSocketFd, host.get(), hostLen);
        hosts[i] = std::move(host);

        readSocket(rootSocketFd, &ports[i], sizeof(ports[i]));
    }

    writeAckPacket(rootSocketFd);

    // We need to wait here until the root node will send a "root is ready" packet
    readAckPacket(rootSocketFd);

    for (NnUint i = 0; i < nNodes; i++) {
        char *host = hosts[i].get();
        int port = ports[i];
        const NnUint peerWorkerIndex = (i < nodeIndex) ? i : (i + 1u);

        if (i >= nodeIndex) {
            printf("⭕ Connect worker-pair: localWorker=%u -> expectedPeerWorker=%u (%s:%d)\n",
                   nodeIndex,
                   peerWorkerIndex,
                   host,
                   port);

            int fd = connectSocket(host, port);
            NnUint actualPeerWorkerIndex = 0u;
            exchangeWorkerPeerIndex(fd, nodeIndex, &actualPeerWorkerIndex);

            if (actualPeerWorkerIndex != peerWorkerIndex) {
                throw std::runtime_error("Worker-peer handshake mismatch on connect path");
            }

            const NnUint socketIndex = peerWorkerToSocketIndex(nodeIndex, actualPeerWorkerIndex);
            if (socketIndex >= nSockets) {
                throw std::runtime_error("Worker-peer handshake produced out-of-range socket index");
            }
            if (slotAssigned[socketIndex]) {
                throw std::runtime_error("Duplicate worker-peer socket detected (connect path)");
            }

            sockets[socketIndex].assign(fd);
            peerNodeBySocket[socketIndex] = actualPeerWorkerIndex + 1u;
            slotAssigned[socketIndex] = true;

            printf("⭕ Socket[%u]: connected to worker(globalNode=%u, workerIndex=%u)\n",
                   socketIndex,
                   actualPeerWorkerIndex + 1u,
                   actualPeerWorkerIndex);
        } else {
            printf("⭕ Waiting worker-pair accept: localWorker=%u expecting one of [0..%u]\n",
                   nodeIndex,
                   nodeIndex == 0 ? 0 : (nodeIndex - 1u));

            int fd = acceptSocket(socketSocket.fd);
            NnUint actualPeerWorkerIndex = 0u;
            exchangeWorkerPeerIndex(fd, nodeIndex, &actualPeerWorkerIndex);

            if (actualPeerWorkerIndex >= nodeIndex) {
                throw std::runtime_error("Worker-peer handshake mismatch on accept path");
            }

            const NnUint socketIndex = peerWorkerToSocketIndex(nodeIndex, actualPeerWorkerIndex);
            if (socketIndex >= nSockets) {
                throw std::runtime_error("Worker-peer handshake produced out-of-range socket index");
            }
            if (slotAssigned[socketIndex]) {
                throw std::runtime_error("Duplicate worker-peer socket detected (accept path)");
            }

            sockets[socketIndex].assign(fd);
            peerNodeBySocket[socketIndex] = actualPeerWorkerIndex + 1u;
            slotAssigned[socketIndex] = true;

            printf("⭕ Socket[%u]: accepted worker(globalNode=%u, workerIndex=%u)\n",
                   socketIndex,
                   actualPeerWorkerIndex + 1u,
                   actualPeerWorkerIndex);
        }
    }

    for (NnUint workerIndex = 0; workerIndex < nNodes; ++workerIndex) {
        if (workerIndex == nodeIndex) continue;
        const NnUint socketIndex = peerWorkerToSocketIndex(nodeIndex, workerIndex);
        if (socketIndex >= nSockets || !slotAssigned[socketIndex]) {
            throw std::runtime_error("Missing worker-peer socket at startup");
        }
    }

    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(&sockets, &peerNodeBySocket));
}

std::unique_ptr<NnNetwork> NnNetwork::connect(NnUint nSockets, char **hosts, NnUint *ports) {
    assert(nSockets > 0);

    std::vector<NnSocket> sockets(nSockets);
    std::vector<NnUint> peerNodeBySocket(nSockets, 0u);
    struct sockaddr_in addr;
    for (NnUint i = 0; i < nSockets; i++) {
        printf("⭕ Socket[%d]: connecting to %s:%d worker\n", i, hosts[i], ports[i]);
        int fd = connectSocket(hosts[i], ports[i]);
        sockets[i].assign(fd);
        writeSocket(fd, &nSockets, sizeof(nSockets));
        writeSocket(fd, &i, sizeof(i)); // send node index
        for (NnUint j = 0; j < nSockets; j++) {
            if (j == i)
                continue;
            NnUint hostLen = strlen(hosts[j]) + 1;
            writeSocket(fd, &hostLen, sizeof(hostLen));
            writeSocket(fd, hosts[j], hostLen);
            writeSocket(fd, &ports[j], sizeof(ports[j]));
        }
        readAckPacket(fd);
        printf("⭕ Socket[%d]: connected\n", i);
        peerNodeBySocket[i] = i + 1u;
    }
    for (NnUint i = 0; i < nSockets; i++) {
        writeAckPacket(sockets[i].fd);
    }
    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(&sockets, &peerNodeBySocket));
}

NnNetwork::NnNetwork(std::vector<NnSocket> *sockets, std::vector<NnUint> *peerNodeBySocket) {
    this->nSockets = sockets->size();
    this->sockets = new int[nSockets];
    for (NnUint i = 0; i < nSockets; i++)
        this->sockets[i] = sockets->at(i).release();
    this->peerNodeBySocket = new NnUint[nSockets];
    for (NnUint i = 0; i < nSockets; i++) {
        this->peerNodeBySocket[i] = (peerNodeBySocket != nullptr && i < peerNodeBySocket->size())
            ? peerNodeBySocket->at(i)
            : 0u;
    }
    this->sentBytes = new NnSize[nSockets];
    this->recvBytes = new NnSize[nSockets];
}

NnNetwork::~NnNetwork() {
    delete[] peerNodeBySocket;
    delete[] sentBytes;
    delete[] recvBytes;
    for (NnUint i = 0; i < nSockets; i++)
        destroySocket(sockets[i]);
    delete[] sockets;
    printf("⭕ Network is closed\n");
}

void NnNetwork::setTurbo(bool enabled) {
    for (NnUint i = 0; i < nSockets; i++) {
        ::setNonBlocking(sockets[i], enabled);
    }
}

void NnNetwork::write(const NnUint socketIndex, const void *data, const NnSize size) {
    assert(socketIndex < nSockets);

    NnByte *current = (NnByte *)data;
    int s = sockets[socketIndex];
    for (NnSize chunk = 0; chunk < size; chunk += MAX_CHUNK_SIZE) {
        NnSize chunkSize = chunk + MAX_CHUNK_SIZE < size ? MAX_CHUNK_SIZE : size - chunk;
        writeSocket(s, current, chunkSize);
        current += chunkSize;
    }
    sentBytes[socketIndex] += size;
}

void NnNetwork::read(const NnUint socketIndex, void *data, const NnSize size) {
    assert(socketIndex < nSockets);

    NnByte *current = (NnByte *)data;
    int s = sockets[socketIndex];
    for (NnSize chunk = 0; chunk < size; chunk += MAX_CHUNK_SIZE) {
        NnSize chunkSize = chunk + MAX_CHUNK_SIZE < size ? MAX_CHUNK_SIZE : size - chunk;
        readSocket(s, current, chunkSize);
        current += chunkSize;
    }
    recvBytes[socketIndex] += size;
}

void NnNetwork::writeAck(const NnUint socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    writeAckPacket(sockets[socketIndex]);
}

void NnNetwork::readAck(const NnUint socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    readAckPacket(sockets[socketIndex]);
}

void NnNetwork::readAckWithTimeout(const NnUint socketIndex, unsigned long timeoutMs) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    if (timeoutMs == 0ul) {
        readAck(socketIndex);
        return;
    }

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(sockets[socketIndex], &rfds);

    struct timeval tv;
    tv.tv_sec = (long)(timeoutMs / 1000ul);
    tv.tv_usec = (long)((timeoutMs % 1000ul) * 1000ul);

    const int rc = select(sockets[socketIndex] + 1, &rfds, nullptr, nullptr, &tv);
    if (rc == 0) {
        throw NnTransferSocketException(ETIMEDOUT, "Socket ack timeout (DLLAMA_IO_TIMEOUT_MS)");
    }
    if (rc < 0) {
        throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
    }
    readAck(socketIndex);
}

void NnNetwork::writeAckWithPayload(const NnUint socketIndex, const void *payload, const NnSize payloadSize) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    writeAckPacket(sockets[socketIndex]);
    sentBytes[socketIndex] += sizeof(int);
    if (payloadSize > 0) {
        write(socketIndex, payload, payloadSize);
    }
}

void NnNetwork::readAckWithPayload(const NnUint socketIndex, void *payload, const NnSize payloadSize) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    readAckPacket(sockets[socketIndex]);
    recvBytes[socketIndex] += sizeof(int);
    if (payloadSize > 0) {
        read(socketIndex, payload, payloadSize);
    }
}

static constexpr NnByte SEG_KIND_OTHER = 0;
static constexpr NnByte SEG_KIND_ATTN  = 1;
static constexpr NnByte SEG_KIND_FFN   = 2;

static inline bool nameHas(const char *name, const char *needle) {
    if (name == nullptr || needle == nullptr) return false;
    return std::strstr(name, needle) != nullptr;
}

static NnByte classifySegmentKind(const NnSegmentConfig &seg) {
    bool hasAttn = false;
    bool hasFfn = false;
    for (NnUint i = 0; i < seg.nOps; ++i) {
        const char *n = seg.ops[i].name;
        if (nameHas(n, "block_matmul_w1") || nameHas(n, "block_matmul_w2") || nameHas(n, "block_matmul_w3") ||
            nameHas(n, "block_moe_") || nameHas(n, "moe_") || nameHas(n, "_moe_")) {
            hasFfn = true;
        }
        if (nameHas(n, "block_matmul_q") || nameHas(n, "block_matmul_k") || nameHas(n, "block_matmul_v") ||
            nameHas(n, "block_matmul_wo") || nameHas(n, "block_multihead_att") || nameHas(n, "block_rope_")) {
            hasAttn = true;
        }
    }
    if (hasFfn) return SEG_KIND_FFN;
    if (hasAttn) return SEG_KIND_ATTN;
    return SEG_KIND_OTHER;
}

bool NnNetwork::tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    if (tryReadSocket(sockets[socketIndex], data, size, maxAttempts)) {
        recvBytes[socketIndex] += size;
        return true;
    }
    return false;
}

void NnNetwork::writeMany(NnUint n, NnSocketIo *ios) {
    bool isWriting;
    NnSize nBytes = 0;
    const unsigned long timeoutMs = getIoTimeoutMs();
    const long long startMs = (timeoutMs > 0ul) ? nowMsSteady() : 0ll;
    unsigned int eagainSpins = 0u;
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        sentBytes[io->socketIndex] += io->size;
    }
    do {
        if (timeoutMs > 0ul) {
            const long long elapsed = nowMsSteady() - startMs;
            if (elapsed > (long long)timeoutMs) {
                throw NnTransferSocketException(ETIMEDOUT, "Socket write timeout (DLLAMA_IO_TIMEOUT_MS)");
            }
        }
        isWriting = false;
        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isWriting = true;
                int socket = sockets[io->socketIndex];
                ssize_t chunkSize = io->size > MAX_CHUNK_SIZE ? MAX_CHUNK_SIZE : io->size;
                ssize_t s = send(socket, (const char*)io->data, chunkSize, 0);
                if (s < 0) {
                    if (isEagainError()) {
                        backoffOnEagain(eagainSpins);
                        continue;
                    }
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (s == 0) {
                    throw NnTransferSocketException(0, "Socket closed");
                }
                eagainSpins = 0u;
                io->size -= s;
                io->data = (char*)io->data + s;
            }
        }
    } while (isWriting);
}

void NnNetwork::writeAll(const void *data, NnSize size) {
    std::vector<NnSocketIo> ios(nSockets);
    for (NnUint i = 0; i < nSockets; i++) {
        NnSocketIo *io = &ios[i];
        io->socketIndex = i;
        io->data = data;
        io->size = size;
    }
    writeMany(nSockets, &ios[0]);
}

void NnNetwork::readMany(NnUint n, NnSocketIo *ios) {
    bool isReading;
    NnSize nBytes = 0;
    const unsigned long timeoutMs = getIoTimeoutMs();
    const long long startMs = (timeoutMs > 0ul) ? nowMsSteady() : 0ll;
    unsigned int eagainSpins = 0u;
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        recvBytes[io->socketIndex] += io->size;
    }
    do {
        if (timeoutMs > 0ul) {
            const long long elapsed = nowMsSteady() - startMs;
            if (elapsed > (long long)timeoutMs) {
                throw NnTransferSocketException(ETIMEDOUT, "Socket read timeout (DLLAMA_IO_TIMEOUT_MS)");
            }
        }
        isReading = false;
        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isReading = true;
                int socket = sockets[io->socketIndex];
                ssize_t r = recv(socket, (char*)io->data, io->size, 0);
                if (r < 0) {
                    if (isEagainError()) {
                        backoffOnEagain(eagainSpins);
                        continue;
                    }
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (r == 0) {
                    throw NnTransferSocketException(0, "Socket closed");
                }
                eagainSpins = 0u;
                io->size -= r;
                io->data = (char*)io->data + r;
            }
        }
    } while (isReading);
}

void NnNetwork::getStats(NnSize *sentBytes, NnSize *recvBytes) {
    *sentBytes = 0;
    *recvBytes = 0;
    for (NnUint i = 0; i < nSockets; i++) {
        *sentBytes += this->sentBytes[i];
        *recvBytes += this->recvBytes[i];
    }
    resetStats();
}

void NnNetwork::resetStats() {
    for (NnUint i = 0; i < nSockets; i++) {
        sentBytes[i] = 0;
        recvBytes[i] = 0;
    }
}

int NnNetwork::getSocketIndexForNode(NnUint targetNodeIndex, NnUint myNodeIndex) const {
    (void)myNodeIndex;
    for (NnUint i = 0; i < nSockets; ++i) {
        if (peerNodeBySocket[i] == targetNodeIndex) {
            return (int)i;
        }
    }
    return -1;
}

void NnNetwork::sendToNode(NnUint targetNodeIndex, NnUint myNodeIndex, const void* data, NnSize size) {
    // 1. 获取对应的 Socket Index
    int socketIndex = getSocketIndexForNode(targetNodeIndex, myNodeIndex);
    
    // 2. 这里的 socketIndex 就是 sockets 数组的下标
    // write 函数内部会查找 this->sockets[socketIndex]
    if (socketIndex >= 0) {
        write(socketIndex, data, size);
    } else {
        // Error or Self
        printf("❌ Error: sendToNode target=%u my=%u invalid socket index\n", targetNodeIndex, myNodeIndex);
    }
}

void NnNetwork::recvFromNode(NnUint sourceNodeIndex, NnUint myNodeIndex, void* data, NnSize size) {
    int socketIndex = getSocketIndexForNode(sourceNodeIndex, myNodeIndex);
    if (socketIndex >= 0) {
        read(socketIndex, data, size);
    } else {
        printf("❌ Error: recvFromNode source=%u my=%u invalid socket index\n", sourceNodeIndex, myNodeIndex);
    }
}

static void syncWithRoot(
    NnNetwork *network, 
    NnUint myNodeIndex, 
    NnByte *buffer, 
    NnSize nBytes, 
    NnUint nThreads, 
    NnUint threadIndex,
    const NnStageConfig *stage // [新增] 传入 Stage 信息
) {
    // 1. 确定谁是 Root
    NnUint groupRootIndex = getGroupRootIndex(stage); // 复用之前的辅助函数
    bool amIRoot = (myNodeIndex == groupRootIndex);
    
    if (amIRoot) {
        // --- Root 发送 (Broadcast) ---
        
        // 确定目标节点列表
        std::vector<int> targetSockets;
        std::vector<NnUint> targetNodes;
        if (stage) {
            // Stage 内广播：只发给组内其他节点
            for(NnUint i=0; i<stage->nNodes; ++i) {
                NnUint target = stage->nodeIndices[i];
                if(target != myNodeIndex) {
                    int sock = network->getSocketIndexForNode(target, myNodeIndex);
                    if(sock >= 0) {
                        targetSockets.push_back(sock);
                        targetNodes.push_back(target);
                    } else {
                        throw std::runtime_error("syncWithRoot: target node not reachable by socket mapping");
                    }
                }
            }
            if (targetSockets.size() != (size_t)(stage->nNodes - 1u)) {
                throw std::runtime_error("syncWithRoot: stage broadcast target set is incomplete");
            }
        } else {
            // 全局广播：发给所有 Socket (简单处理)
            // 注意：这假设 network->nSockets 包含了所有 Worker
            for(NnUint i=0; i<network->nSockets; ++i) {
                targetSockets.push_back(i);
                targetNodes.push_back(0u);
            }
        }

        NnUint nTargets = targetSockets.size();
        if (nTargets == 0) return;

        // 分配给线程
        NnUint nSocketsPerThread = nTargets / nThreads + (nTargets % nThreads > threadIndex ? 1 : 0);
        if (nSocketsPerThread == 0) return;

        NnUint startIdx = 0;
        for (NnUint t = 0; t < threadIndex; ++t) {
            startIdx += nTargets / nThreads + (nTargets % nThreads > t ? 1 : 0);
        }

        std::vector<NnSocketIo> ios(nSocketsPerThread);
        const unsigned long ackTimeoutMs = getIoTimeoutMs();
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            ios[i].socketIndex = targetSockets[startIdx + i]; // 使用真实的 Socket Index
            ios[i].data = buffer;
            ios[i].size = nBytes;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);

        // [新增] Root 等待 Workers 确认 (ACK)
        // 确保 Workers 已经接收完数据，实现同步屏障
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            try {
                network->readAckWithTimeout(ios[i].socketIndex, ackTimeoutMs);
            } catch (const std::exception &e) {
                const NnUint targetNode = targetNodes[startIdx + i];
                std::fprintf(stderr,
                             "❌ syncWithRoot ack timeout/fail root=%u targetNode=%u socket=%u stageRoot=%u stageNodes=%u err=%s\n",
                             (unsigned)myNodeIndex,
                             (unsigned)targetNode,
                             (unsigned)ios[i].socketIndex,
                             (unsigned)groupRootIndex,
                             stage ? (unsigned)stage->nNodes : 0u,
                             e.what());
                throw;
            }
        }

    } else {
        // --- Worker 接收 ---

        if (threadIndex != 0) return; // 接收通常只需要一个线程

        int rootSocketIndex = network->getSocketIndexForNode(groupRootIndex, myNodeIndex);
        if (rootSocketIndex < 0) {
            throw std::runtime_error("syncWithRoot: cannot resolve root socket index");
        }

        NnSocketIo ios;
        ios.data = buffer;
        ios.size = nBytes;
        ios.socketIndex = rootSocketIndex; // [修正] 使用查找到的 Socket，而不是硬编码 0
        network->readMany(1, &ios);

        // [新增] Worker 发送确认 (ACK) 给 Root
        network->writeAck(rootSocketIndex);
    }
}

static void syncNodeSlices(
    bool onlyFromWorkerToRoot, 
    NnNetwork *network, 
    NnUint myNodeIndex, 
    NnUint nTotalNodes, 
    NnByte *buffer, 
    NnSize nBytes, 
    NnFloatType floatType,
    NnUint nThreads, 
    NnUint threadIndex, 
    const NnUnevenPartitionPlan *plan,
    const NnStageConfig *stage, // 指定同步组
    NnSliceTag forcedTag, // disambiguate split kind when needed
    NnUint totalElements = 0 // [New] Total elements for Q80 matching
) {
    // ---------------------------------------------------------
    // 0. [核心修改] 确定当前组的 Root 身份
    // ---------------------------------------------------------
    NnUint groupRootIndex = getGroupRootIndex(stage);
    bool amIRoot = (myNodeIndex == groupRootIndex);

    // 1. 确定参与同步的节点列表 (Peers)
    const NnUint* groupNodes = stage ? stage->nodeIndices : nullptr;
    NnUint nGroupNodes = stage ? stage->nNodes : nTotalNodes;

    // 2. 筛选出需要通信的 Socket
    std::vector<int> targetSockets;
    std::vector<NnUint> targetNodeIndices;

    for (NnUint i = 0; i < nGroupNodes; ++i) {
        // 获取目标的全局 ID
        NnUint targetNode = groupNodes ? groupNodes[i] : i;
        
        // 跳过自己
        if (targetNode == myNodeIndex) continue;

        // [修改] 动态的 Root/Worker 判定逻辑
        if (onlyFromWorkerToRoot) {
            // 检查是否是 Logits 收集 (Global gather with PP plan)
            // 如果是，只有 Last Stage 的节点需要发送给 Root
            bool isLogitsGather = (plan != nullptr && plan->nStages > 0 && stage == nullptr);

            // Case A: 我是 Worker (不是本组 Root)
            if (!amIRoot) {
                // Worker 只理会本组的 Root
                if (targetNode != groupRootIndex) continue; 

                if (isLogitsGather) {
                    const NnStageConfig& lastStage = plan->stages[plan->nStages - 1];
                    bool amInLastStage = false;
                    for(unsigned k=0; k<lastStage.nNodes; ++k) {
                        if (lastStage.nodeIndices[k] == myNodeIndex) {
                            amInLastStage = true;
                            break;
                        }
                    }
                    if (!amInLastStage) continue;
                }
            }
            // Case B: 我是 Root (本组 Root)
            else { 
                // Root 理会所有人 (接收)
                // 但如果是 Logits 收集，Root 只接收 Last Stage 的数据
                if (isLogitsGather) {
                    const NnStageConfig& lastStage = plan->stages[plan->nStages - 1];
                    bool targetInLastStage = false;
                    for(unsigned k=0; k<lastStage.nNodes; ++k) {
                        if (lastStage.nodeIndices[k] == targetNode) {
                            targetInLastStage = true;
                            break;
                        }
                    }
                    if (!targetInLastStage) continue;
                }
            }
        }

        int socketIndex = network->getSocketIndexForNode(targetNode, myNodeIndex);
        if (socketIndex >= 0) {
            targetSockets.push_back(socketIndex);
            targetNodeIndices.push_back(targetNode);
        }
    }

    // 3. 任务分配给线程 (保持不变)
    NnUint nActiveSockets = targetSockets.size();
    NnUint nSocketsPerThread = nActiveSockets / nThreads + (nActiveSockets % nThreads > threadIndex ? 1 : 0);
    if (nSocketsPerThread == 0) return;

    NnUint startIdx = 0;
    for (NnUint t = 0; t < threadIndex; ++t) {
        startIdx += nActiveSockets / nThreads + (nActiveSockets % nThreads > t ? 1 : 0);
    }

    // 4. 准备切分信息 (Plan Aware) (保持不变)
    std::vector<NnSize> sliceOffsets(nTotalNodes);
    std::vector<NnSize> sliceSizes(nTotalNodes);
    
    // For PP, split tables are stage-local. For logits gather, slices come from the LAST stage.
    const bool isLogitsGather = (onlyFromWorkerToRoot && plan != nullptr && plan->nStages > 0 && stage == nullptr);
    const NnStageConfig* stageForSplit = stage;
    if (isLogitsGather) {
        stageForSplit = &plan->stages[plan->nStages - 1];
    }

    // [Fix] Pass floatType + total elements for accurate matching (incl. Q80)
    fillUnevenSlices(plan, nTotalNodes, nBytes, sliceOffsets, sliceSizes, floatType, stageForSplit, forcedTag, totalElements);



    // --- logits gather (LastStage -> Root) 调试：打印 Root 端的目标节点与切片信息 ---
#if NN_NETWORK_COMM_DATA_LOG
    if (onlyFromWorkerToRoot && amIRoot && plan != nullptr && plan->nStages > 0 && stage == nullptr && threadIndex == 0) {
        const NnStageConfig& lastStage = plan->stages[plan->nStages - 1];
        printf("LOGITS GATHER root=%u lastStageNodes=", myNodeIndex);
        for (unsigned k = 0; k < lastStage.nNodes; ++k) {
            printf("%u%s", lastStage.nodeIndices[k], (k + 1 < lastStage.nNodes) ? "," : "");
        }
        printf(" targetsBuilt=");
        for (size_t k = 0; k < targetNodeIndices.size(); ++k) {
            printf("%u%s", targetNodeIndices[k], (k + 1 < targetNodeIndices.size()) ? "," : "");
        }
        printf(" nTargets=%zu\n", targetNodeIndices.size());

        for (unsigned k = 0; k < lastStage.nNodes; ++k) {
            NnUint node = lastStage.nodeIndices[k];
            if (node == myNodeIndex) continue;
            int sock = network->getSocketIndexForNode(node, myNodeIndex);
            printf(
                "LOGITS GATHER root recv-candidate node=%u sock=%d off=%zu size=%zu\n",
                node,
                sock,
                (size_t)sliceOffsets[node],
                (size_t)sliceSizes[node]
            );
        }
    }
#endif


    std::vector<NnSocketIo> ios(nSocketsPerThread);

    // My own slice (may be unused depending on mode)
    const NnSize mySliceOffset = sliceOffsets[myNodeIndex];
    NnByte *mySliceData = &buffer[mySliceOffset];
    const NnSize mySliceSize = sliceSizes[myNodeIndex];

    // --- 发送阶段 (Send) ---
    bool iShouldSend = true;
    
    // [修改] 如果我是 Root 且模式是 Worker->Root，我不发送
    if (onlyFromWorkerToRoot && amIRoot) iShouldSend = false; 

    // logits gather（LastStage -> Root）：发送端应发送自己在 pipe 中写入的那段（全局 offset 语义）。
    // 早期为了绕开 split 匹配失败导致的“写入在 0，但按全局 offset 读”而临时使用过 LOCAL0。
    // 现在 split 已 stage-aware 修复后，应回到全局 offset，否则 offset!=0 的 shard 会发送未写入区域（全 0）。
    // NOTE: isLogitsGather already computed above for stage-aware slicing.

    if (iShouldSend) {
        // 通信预览（载荷字节/哈希）：默认关闭
#if NN_NETWORK_COMM_DATA_LOG
        if (threadIndex == 0 && mySliceSize > 0) {
            // logits gather 时尽量打印全量 peer，避免因 preview 截断误判“没收到某个节点”
            NnUint preview = isLogitsGather ? nSocketsPerThread : std::min(nSocketsPerThread, (NnUint)2);
            for (NnUint i = 0; i < preview; i++) {
                NnUint idx = startIdx + i;
                char prefix[128];
                std::snprintf(prefix, sizeof(prefix), "SYNC send node %u -> %u bytes=%zu off=%zu mode=%s", myNodeIndex, targetNodeIndices[idx], (size_t)mySliceSize, (size_t)mySliceOffset, "GLOBALOFF");
                printBytes(prefix, mySliceData, mySliceSize);

                // hash 校验（限制最多 64KB，避免太慢）
                const NnSize hashLen = std::min(mySliceSize, (NnSize)65536);
                std::uint64_t h = fnv1a64(mySliceData, hashLen);
                printf("SYNC send hash node %u -> %u len=%zu hash=0x%016llx\n", myNodeIndex, targetNodeIndices[idx], (size_t)hashLen, (unsigned long long)h);
            }
        }
#endif

        // NOTE: actual sends happen below (ordered exchange) unless this is worker->root mode.
    }

    // --- 接收阶段 (Receive) ---
    bool iShouldRecv = true;
    
    // [修改] 如果我是 Worker 且模式是 Worker->Root，我不接收
    if (onlyFromWorkerToRoot && !amIRoot) iShouldRecv = false; 

    // ---------------------------------------------------------
    // Communication
    // ---------------------------------------------------------
    // For the unidirectional gather (worker -> root), keep the legacy behavior:
    // - senders: writeMany()
    // - receivers: readMany()
    // This mode doesn't have a symmetric exchange and thus doesn't benefit from
    // per-peer ordering.
    if (onlyFromWorkerToRoot) {
        if (iShouldSend) {
            for (NnUint i = 0; i < nSocketsPerThread; i++) {
                const NnUint idx = startIdx + i;
                ios[i].socketIndex = targetSockets[idx];
                ios[i].data = mySliceData;
                ios[i].size = mySliceSize;
            }
            network->writeMany(nSocketsPerThread, &ios[0]);
        }

        if (iShouldRecv) {
            for (NnUint i = 0; i < nSocketsPerThread; i++) {
                const NnUint idx = startIdx + i;
                const NnUint targetNode = targetNodeIndices[idx];

                ios[i].socketIndex = targetSockets[idx];
                ios[i].data = &buffer[sliceOffsets[targetNode]];
                ios[i].size = sliceSizes[targetNode];
            }
            network->readMany(nSocketsPerThread, &ios[0]);
        }
    } else {
        // Full all-gather style exchange: for each peer connection, enforce a total order
        // based on node indices so that one side is send-first and the other is recv-first.
        // This prevents cross-node deadlocks/hangs caused by symmetric send/recv ordering.
        std::vector<NnSocketIo> sendFirst;
        std::vector<NnSocketIo> recvFirst;
        std::vector<NnSocketIo> sendSecond;
        std::vector<NnSocketIo> recvSecond;
        sendFirst.reserve(nSocketsPerThread);
        recvFirst.reserve(nSocketsPerThread);
        sendSecond.reserve(nSocketsPerThread);
        recvSecond.reserve(nSocketsPerThread);

        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            const NnUint idx = startIdx + i;
            const NnUint targetNode = targetNodeIndices[idx];
            const int sock = targetSockets[idx];

            const bool amSendFirst = (myNodeIndex < targetNode);

            if (iShouldSend) {
                NnSocketIo s{};
                s.socketIndex = (NnUint)sock;
                s.data = mySliceData;
                s.size = mySliceSize;
                if (amSendFirst) {
                    sendFirst.push_back(s);
                } else {
                    sendSecond.push_back(s);
                }
            }

            if (iShouldRecv) {
                NnSocketIo r{};
                r.socketIndex = (NnUint)sock;
                r.data = &buffer[sliceOffsets[targetNode]];
                r.size = sliceSizes[targetNode];
                if (amSendFirst) {
                    recvSecond.push_back(r);
                } else {
                    recvFirst.push_back(r);
                }
            }
        }

        if (iShouldSend && !sendFirst.empty()) {
            network->writeMany((NnUint)sendFirst.size(), sendFirst.data());
        }
        if (iShouldRecv && !recvFirst.empty()) {
            network->readMany((NnUint)recvFirst.size(), recvFirst.data());
        }
        if (iShouldRecv && !recvSecond.empty()) {
            network->readMany((NnUint)recvSecond.size(), recvSecond.data());
        }
        if (iShouldSend && !sendSecond.empty()) {
            network->writeMany((NnUint)sendSecond.size(), sendSecond.data());
        }
    }

    if (iShouldRecv) {

        // 通信预览（载荷字节/哈希）：默认关闭
#if NN_NETWORK_COMM_DATA_LOG
        // 注意：在 logits gather 模式下，目标 socket 会被分配到多个线程；只在 threadIndex==0 打印会“看起来缺节点”。
        if (isLogitsGather || threadIndex == 0) {
            NnUint preview = isLogitsGather ? nSocketsPerThread : std::min(nSocketsPerThread, (NnUint)2);
            for (NnUint i = 0; i < preview; i++) {
                NnUint idx = startIdx + i;
                NnUint targetNode = targetNodeIndices[idx];
                NnSize sliceSize = sliceSizes[targetNode];
                if (sliceSize == 0) {
                    if (isLogitsGather) {
                        printf("SYNC recv(t=%u) node %u <- %u skip size=0\n", threadIndex, myNodeIndex, targetNode);
                    }
                    continue;
                }
                char prefix[160];
                std::snprintf(prefix, sizeof(prefix), "SYNC recv(t=%u) node %u <- %u bytes=%zu", threadIndex, myNodeIndex, targetNode, (size_t)sliceSize);
                printBytes(prefix, &buffer[sliceOffsets[targetNode]], sliceSize);

                // hash 校验（限制最多 64KB，避免太慢）
                const NnSize hashLen = std::min(sliceSize, (NnSize)65536);
                std::uint64_t h = fnv1a64(&buffer[sliceOffsets[targetNode]], hashLen);
                printf(
                    "SYNC recv hash(t=%u) node %u <- %u len=%zu hash=0x%016llx\n",
                    threadIndex,
                    myNodeIndex,
                    targetNode,
                    (size_t)hashLen,
                    (unsigned long long)h
                );
            }
        }
#endif
    }
}

static void syncPpSend(NnNetwork *network, NnUint myNodeIndex, NnByte *buffer, NnSize nBytes, 
                       const NnUnevenPartitionPlan *plan) {
    // 1. 找到我所在的 Stage
    const NnStageConfig* myStage = nullptr;
    const NnStageConfig* nextStage = nullptr;
    
    for (NnUint s = 0; s < plan->nStages; ++s) {
        // 检查我是否是该 Stage 的成员
        for (NnUint i = 0; i < plan->stages[s].nNodes; ++i) {
            if (plan->stages[s].nodeIndices[i] == myNodeIndex) {
                myStage = &plan->stages[s];
                // 如果还有下一个 Stage
                if (s + 1 < plan->nStages) {
                    nextStage = &plan->stages[s+1];
                }
                break;
            }
        }
        if (myStage) break;
    }

    // 2. 只有当前 Stage 的 Root 负责发送
    if (myStage && myStage->rootNodeIndex == myNodeIndex) {
        if (nextStage) {
            // 发送给下一阶段的 Root
            // printf("🚀 [PP] Node %u sending %zu bytes to Node %u (Stage %u)\n", 
            //        myNodeIndex, nBytes, nextStage->rootNodeIndex, nextStage->stageIndex);
            
            // 注意：这里需要 network 实现点对点 write
            // 如果网络拓扑不支持直连，可能需要通过 Node 0 中转
            network->sendToNode(nextStage->rootNodeIndex, myNodeIndex, buffer, nBytes);
        }
    }
}

static void syncPpRecv(NnNetwork *network, NnUint myNodeIndex, NnByte *buffer, NnSize nBytes, 
                       const NnUnevenPartitionPlan *plan) {
    const NnStageConfig* myStage = nullptr;
    const NnStageConfig* prevStage = nullptr;

    for (NnUint s = 0; s < plan->nStages; ++s) {
        for (NnUint i = 0; i < plan->stages[s].nNodes; ++i) {
            if (plan->stages[s].nodeIndices[i] == myNodeIndex) {
                myStage = &plan->stages[s];
                if (s > 0) {
                    prevStage = &plan->stages[s-1];
                }
                break;
            }
        }
        if (myStage) break;
    }

    // 只有当前 Stage 的 Root 负责接收
    if (myStage && myStage->rootNodeIndex == myNodeIndex) {
        if (prevStage) {
            // 从上一阶段的 Root 接收
            // printf("📥 [PP] Node %u receiving %zu bytes from Node %u (Stage %u)\n", 
            //        myNodeIndex, nBytes, prevStage->rootNodeIndex, prevStage->stageIndex);
                   
            network->recvFromNode(prevStage->rootNodeIndex, myNodeIndex, buffer, nBytes);
        } else {
            // 如果是 Stage 0 的第一层，数据应该来自 Embedding/Input，理论上不走 PP_RECV
            // 除非我们在架构设计上把 Embedding 视为 "Stage -1"
        }
    }
}

NnNetworkNodeSynchronizer::NnNetworkNodeSynchronizer(
    NnNetwork *network,
    NnNetExecution *execution,
    NnNetConfig *netConfig,
    NnNodeConfig *nodeConfig,
    const NnUnevenPartitionPlan *plan,
    bool layerProfileEnabled) {
    this->network = network;
    this->execution = execution;
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->plan = plan;
    this->layerProfileEnabled = layerProfileEnabled;

    // Env toggles for layer profiling.
    // - DLLAMA_LAYER_PROF_PRINT=1 enables stdout printing on stage root.
    // - DLLAMA_LAYER_PROF_PATH sets explicit snapshot file path (stage root only).
    const char *p = std::getenv("DLLAMA_LAYER_PROF_PRINT");
    layerPerfPrintEnabled = (p != nullptr && p[0] != '\0' && std::atoi(p) != 0);
    const char *pathEnv = std::getenv("DLLAMA_LAYER_PROF_PATH");
    if (pathEnv != nullptr && pathEnv[0] != '\0') {
        layerPerfPath = std::string(pathEnv);
    }
    // [新增] 构造时缓存 myStage，避免运行时重复查找
    this->myStage = nullptr;
    if (plan) {
        for (NnUint s = 0; s < plan->nStages; ++s) {
            for (NnUint i = 0; i < plan->stages[s].nNodes; ++i) {
                if (plan->stages[s].nodeIndices[i] == nodeConfig->nodeIndex) {
                    this->myStage = &plan->stages[s];
                    goto stage_found; // 跳出双层循环
                }
            }
        }
    }
stage_found:;

    // Build per-segment metadata for layer profiling.
    segmentMeta.clear();
    if (nodeConfig != nullptr && nodeConfig->segments != nullptr) {
        segmentMeta.resize(nodeConfig->nSegments);
        for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
            const NnSegmentConfig &seg = nodeConfig->segments[s];
            SegmentMeta m{};
            m.kind = classifySegmentKind(seg);
            if (seg.nOps > 0) {
                m.layerIndex = seg.ops[0].index;
            } else {
                m.layerIndex = 0xFFFFFFFFu;
            }
            segmentMeta[s] = m;
        }
    }

    // Stage node -> local index map (only meaningful for stage-local profiling).
    stageLocalIndexByGlobalNode.clear();
    if (netConfig != nullptr) {
        stageLocalIndexByGlobalNode.assign(netConfig->nNodes, -1);
        if (myStage != nullptr) {
            for (NnUint i = 0; i < myStage->nNodes; ++i) {
                const NnUint g = myStage->nodeIndices[i];
                if (g < netConfig->nNodes) stageLocalIndexByGlobalNode[g] = (int)i;
            }
        }
    }

    // Allocate storage for last per-layer per-node perf on stage root.
    lastLayerPerfByLayer.clear();
    if (this->layerProfileEnabled && execution != nullptr && execution->layerPerf != nullptr && myStage != nullptr) {
        const NnUint nLayers = execution->layerPerf->nLayers;
        lastLayerPerfByLayer.resize(nLayers);
        for (NnUint l = 0; l < nLayers; ++l) {
            lastLayerPerfByLayer[l].resize(myStage->nNodes);
        }
    }
    // Build a stable (segmentIndex, syncIndex) -> slot mapping for optional sync profiling.
    // This enables measuring full sync wall-time across all executor threads.
    syncProfileBaseBySegment.clear();
    syncProfileSlots.clear();
    if (nodeConfig != nullptr && nodeConfig->segments != nullptr) {
        syncProfileBaseBySegment.resize(nodeConfig->nSegments, 0u);
        NnUint totalSlots = 0u;
        for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
            syncProfileBaseBySegment[s] = totalSlots;
            totalSlots += nodeConfig->segments[s].nSyncs;
        }
        syncProfileSlots.resize(totalSlots);

        layerPerfBaseBySegment = syncProfileBaseBySegment;
        layerPerfSlots.resize(totalSlots);
    }
}

void NnNetworkNodeSynchronizer::maybeInitLayerPerfSnapshot() {
#ifdef _WIN32
    return;
#else
    if (!layerProfileEnabled) return;
    if (layerPerfMap != nullptr) return;
    if (execution == nullptr || execution->layerPerf == nullptr) return;
    if (myStage == nullptr || nodeConfig == nullptr) return;
    if (nodeConfig->nodeIndex != myStage->rootNodeIndex) return;

    // Default path includes stage+root to avoid collisions in multi-process setups.
    if (layerPerfPath.empty()) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "/tmp/dllama_layer_prof_stage%u_root%u.bin",
                      (unsigned)myStage->stageIndex, (unsigned)myStage->rootNodeIndex);
        layerPerfPath = std::string(buf);
    }

    const NnUint nLayers = execution->layerPerf->nLayers;
    const NnUint nStageNodes = myStage->nNodes;
    const size_t headerSize = sizeof(NnLayerPerfSnapshotHeader);
    const size_t bodySize = (size_t)nLayers * (size_t)nStageNodes * sizeof(NnLayerPerfMsg);
    const size_t totalSize = headerSize + bodySize;

    int fd = ::open(layerPerfPath.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
        std::perror("open(layer-prof snapshot)");
        return;
    }
    if (::ftruncate(fd, (off_t)totalSize) != 0) {
        std::perror("ftruncate(layer-prof snapshot)");
        ::close(fd);
        return;
    }
    void *mem = ::mmap(nullptr, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) {
        std::perror("mmap(layer-prof snapshot)");
        ::close(fd);
        return;
    }

    layerPerfFd = fd;
    layerPerfMap = mem;
    layerPerfMapSize = (NnSize)totalSize;
    layerPerfEpoch = 0ull;

    // Initialize header and zero payload.
    NnLayerPerfSnapshotHeader *hdr = (NnLayerPerfSnapshotHeader *)layerPerfMap;
    std::memset(hdr, 0, sizeof(*hdr));
    hdr->magic = NN_LAYER_PERF_SNAPSHOT_MAGIC;
    hdr->version = NN_LAYER_PERF_SNAPSHOT_VERSION;
    hdr->stageIndex = myStage->stageIndex;
    hdr->rootNodeIndex = myStage->rootNodeIndex;
    hdr->nLayers = nLayers;
    hdr->nStageNodes = nStageNodes;
    hdr->epoch = 0ull;

    std::memset((char *)layerPerfMap + headerSize, 0, bodySize);

    // Best-effort flush.
    (void)::msync(layerPerfMap, totalSize, MS_ASYNC);
#endif
}

void NnNetworkNodeSynchronizer::publishLayerPerfSnapshot(NnUint layerIndex) {
#ifdef _WIN32
    (void)layerIndex;
    return;
#else
    if (layerPerfMap == nullptr) return;
    if (execution == nullptr || execution->layerPerf == nullptr) return;
    if (myStage == nullptr) return;
    const NnUint nLayers = execution->layerPerf->nLayers;
    const NnUint nStageNodes = myStage->nNodes;
    if (layerIndex >= nLayers) return;
    if (lastLayerPerfByLayer.empty() || layerIndex >= lastLayerPerfByLayer.size()) return;

    const size_t headerSize = sizeof(NnLayerPerfSnapshotHeader);
    NnLayerPerfSnapshotHeader *hdr = (NnLayerPerfSnapshotHeader *)layerPerfMap;
    if (hdr->magic != NN_LAYER_PERF_SNAPSHOT_MAGIC || hdr->version != NN_LAYER_PERF_SNAPSHOT_VERSION) return;
    if (hdr->nLayers != nLayers || hdr->nStageNodes != nStageNodes) return;

    // Copy only this layer's row.
    NnLayerPerfMsg *base = (NnLayerPerfMsg *)((char *)layerPerfMap + headerSize);
    NnLayerPerfMsg *row = base + (size_t)layerIndex * (size_t)nStageNodes;
    for (NnUint i = 0; i < nStageNodes; ++i) {
        row[i] = lastLayerPerfByLayer[layerIndex][i];
    }

    hdr->epoch = ++layerPerfEpoch;
    (void)::msync(layerPerfMap, layerPerfMapSize, MS_ASYNC);
#endif
}

NnNetworkNodeSynchronizer::~NnNetworkNodeSynchronizer() {
#ifndef _WIN32
    if (layerPerfMap != nullptr) {
        ::munmap(layerPerfMap, (size_t)layerPerfMapSize);
        layerPerfMap = nullptr;
        layerPerfMapSize = 0;
    }
    if (layerPerfFd >= 0) {
        ::close(layerPerfFd);
        layerPerfFd = -1;
    }
#endif
}

void NnNetworkNodeSynchronizer::onSyncStepComplete(NnUint segmentIndex) {
    (void)segmentIndex;
    // NOTE: Do not send network traffic here.
    // Post-sync hooks are not aligned across nodes and can desynchronize socket streams.
}

void NnNetworkNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];

    const bool kvAggProof = (std::getenv("DLLAMA_KV_AGGREGATE_PROOF") != nullptr);
    const bool kvAggProofAllBatches = (std::getenv("DLLAMA_KV_AGGREGATE_PROOF_ALL_BATCHES") != nullptr);
    const bool syncTrace = (std::getenv("DLLAMA_SYNC_TRACE") != nullptr);
    const bool syncTraceAllThreads = (std::getenv("DLLAMA_SYNC_TRACE_ALL_THREADS") != nullptr);
    const bool syncProfile = (std::getenv("DLLAMA_SYNC_PROFILE") != nullptr);

    auto nowUs = []() -> long long {
        return (long long)std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    };

    auto findPipeIndexByName = [&](const char *name) -> int {
        for (NnUint i = 0; i < netConfig->nPipes; i++) {
            if (netConfig->pipes[i].name && std::strcmp(netConfig->pipes[i].name, name) == 0) return (int)i;
        }
        return -1;
    };

    const int posPipeIndex = kvAggProof ? findPipeIndexByName("POS") : -1;

    for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
        NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
        NnByte *pipe = execution->pipes[syncConfig->pipeIndex];
        NnPipeConfig *pipeConfig = &netConfig->pipes[syncConfig->pipeIndex];
        NnSize batchBytes = getBytes(pipeConfig->size.floatType, pipeConfig->size.x);
        NnUint totalElements = pipeConfig->size.x; // [New] Get total elements

        // Optional full-wall sync timing (across all executor threads).
        SyncProfileSlot *slot = nullptr;
        if (syncProfile && segmentIndex < syncProfileBaseBySegment.size()) {
            const NnUint base = syncProfileBaseBySegment[segmentIndex];
            const NnUint slotIndex = base + syncIndex;
            if (slotIndex < syncProfileSlots.size()) slot = &syncProfileSlots[slotIndex];
        }

        NnUint epoch = 0u;
        if (slot != nullptr) {
            if (threadIndex == 0) {
                epoch = slot->epoch.fetch_add(1u) + 1u;
                slot->arrived.store(0u, std::memory_order_release);
                slot->done.store(0u, std::memory_order_release);
                slot->startUs.store(nowUs(), std::memory_order_release);
            }

            // Barrier: wait for all threads to reach this syncIndex.
            // This makes duration represent the full executor sync step.
            // NOTE: spin-wait is OK for debugging; keep it behind env var.
            while ((epoch = slot->epoch.load(std::memory_order_acquire)) == 0u) {
            }
            slot->arrived.fetch_add(1u, std::memory_order_acq_rel);
            while (slot->arrived.load(std::memory_order_acquire) < nThreads) {
            }
        }

        auto printHeadTail16 = [&](const char *pipeName, NnUint pipeIndex, NnUint batchIndex, const NnByte *data, NnSize nBytes) {
            if (pipeName == nullptr) pipeName = "(null)";
            if (nBytes < 32) {
                printf("🧪 [kvagg][%s] node=%u seg=%u sync=%u batch=%u bytes=%zu (too small)\n",
                       pipeName, nodeConfig->nodeIndex, segmentIndex, syncIndex, batchIndex, (size_t)nBytes);
                return;
            }

            int posI = -1;
            if (posPipeIndex >= 0) {
                const NnPipeConfig *posCfg = &netConfig->pipes[(NnUint)posPipeIndex];
                const NnSize posBatchBytes = getBytes(posCfg->size.floatType, posCfg->size.x);
                const NnByte *posPipe = execution->pipes[(NnUint)posPipeIndex];
                float posF = 0.0f;
                std::memcpy(&posF, &posPipe[batchIndex * posBatchBytes], sizeof(float));
                posI = (int)posF;
            }

            printf("🧪 [kvagg][%s] node=%u seg=%u sync=%u pipe=%u batch=%u bytes=%zu pos=%d head=",
                   pipeName, nodeConfig->nodeIndex, segmentIndex, syncIndex, pipeIndex, batchIndex, (size_t)nBytes, posI);
            for (int i = 0; i < 16; ++i) printf("%02x", (unsigned)(unsigned char)data[i]);
            printf(" tail=");
            for (int i = 0; i < 16; ++i) printf("%02x", (unsigned)(unsigned char)data[(size_t)nBytes - 16 + (size_t)i]);
            printf("\n");
        };

        // Force KV-head slicing for KV aggregation pipes.
        // This avoids ambiguous AUTO matching and ensures slices are derived from kvHeadSplit
        // (including the multiplier = seqLen * headDim when totalElements == seqLen * kvDim).
        NnSliceTag forcedTag = NN_SLICE_AUTO;
        if (pipeConfig->name != nullptr) {
            if (std::strcmp(pipeConfig->name, "KC") == 0 || std::strcmp(pipeConfig->name, "VC") == 0) {
                forcedTag = NN_SLICE_KV_HEAD;
            }
        }

        // (Debug) Uncomment if you need per-sync pipe size info
        // if (threadIndex == 0) {
        //     printf("DEBUG sync entry: seg=%u sync=%u pipe=%u ft=%d x=%u batchBytes=%zu\n",
        //            segmentIndex, syncIndex, syncConfig->pipeIndex,
        //            pipeConfig->size.floatType, pipeConfig->size.x, (size_t)batchBytes);
        // }

        const char* syncTypeStr = "UNKNOWN";

        for (NnUint batchIndex = 0; batchIndex < execution->batchSize; batchIndex++) {
            NnByte *pipeBatch = &pipe[batchIndex * batchBytes];

            if (syncTrace && (syncTraceAllThreads || threadIndex == 0) && batchIndex == 0) {
                const char *pipeName = pipeConfig->name ? pipeConfig->name : "(null)";
                printf(
                    "⏱️ [sync] enter node=%u t=%u/%u seg=%u sync=%u type=%u pipe=%u name=%s bytes=%zu stageRoot=%u stageNodes=%u\n",
                    nodeConfig->nodeIndex,
                    (unsigned)threadIndex,
                    (unsigned)nThreads,
                    segmentIndex,
                    syncIndex,
                    (unsigned)syncConfig->syncType,
                    (unsigned)syncConfig->pipeIndex,
                    pipeName,
                    (size_t)batchBytes,
                    myStage ? (unsigned)myStage->rootNodeIndex : 0u,
                    myStage ? (unsigned)myStage->nNodes : (unsigned)netConfig->nNodes
                );
                std::fflush(stdout);
            }

            if (syncConfig->syncType == SYNC_WITH_ROOT) {
                syncTypeStr = "SYNC_WITH_ROOT";
                syncWithRoot(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, nThreads, threadIndex, this->myStage);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES) {
                syncTypeStr = "SYNC_NODE_SLICES";
                syncNodeSlices(false, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, pipeConfig->size.floatType, nThreads, threadIndex, plan, this->myStage, forcedTag, totalElements);
            }else if (syncConfig->syncType == SYNC_NODE_SLICES_EXCEPT_ROOT) {
                syncTypeStr = "SYNC_LOGITS";
                syncNodeSlices(true, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, pipeConfig->size.floatType, nThreads, threadIndex, plan, nullptr, forcedTag, totalElements);
            } 
            else if (syncConfig->syncType == SYNC_PP_SEND) {
                syncTypeStr = "PP_SEND";
                // PP 只要单线程执行一次
                if (threadIndex == 0) {
                    syncPpSend(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, plan);
                }
            }
            else if (syncConfig->syncType == SYNC_PP_RECV) {
                syncTypeStr = "PP_RECV";
                // PP 只要单线程执行一次
                if (threadIndex == 0) {
                    syncPpRecv(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, plan);
                }
            }else {
                throw std::invalid_argument("Unknown sync type");
            }

            if (syncTrace && (syncTraceAllThreads || threadIndex == 0) && batchIndex == 0) {
                const char *pipeName = pipeConfig->name ? pipeConfig->name : "(null)";
                printf(
                    "⏱️ [sync] exit  node=%u t=%u/%u seg=%u sync=%u type=%u pipe=%u name=%s\n",
                    nodeConfig->nodeIndex,
                    (unsigned)threadIndex,
                    (unsigned)nThreads,
                    segmentIndex,
                    syncIndex,
                    (unsigned)syncConfig->syncType,
                    (unsigned)syncConfig->pipeIndex,
                    pipeName
                );
                std::fflush(stdout);
            }

            if (threadIndex == 0) {
                // Proof: after KV aggregation all-gather, dump head/tail bytes.
                if (kvAggProof && syncConfig->syncType == SYNC_NODE_SLICES && pipeConfig->name != nullptr &&
                    (std::strcmp(pipeConfig->name, "KC") == 0 || std::strcmp(pipeConfig->name, "VC") == 0)) {
                    if (kvAggProofAllBatches || batchIndex == 0u) {
                        printHeadTail16(pipeConfig->name, syncConfig->pipeIndex, batchIndex, pipeBatch, batchBytes);
                    }
                }

                // (optional) extra debug hooks go here
            }
        }

        // Layer profiling: exchange per-layer compute times to stage root.
        // IMPORTANT: must run inside the sync call with a local thread barrier,
        // otherwise it can desynchronize socket streams across nodes.
        if (layerProfileEnabled && syncConfig->syncType == SYNC_NODE_SLICES &&
            execution != nullptr && execution->layerPerf != nullptr && myStage != nullptr &&
            segmentIndex < segmentMeta.size() && segmentMeta[segmentIndex].kind == SEG_KIND_FFN) {

            SyncProfileSlot *lp = nullptr;
            if (segmentIndex < layerPerfBaseBySegment.size()) {
                const NnUint base = layerPerfBaseBySegment[segmentIndex];
                const NnUint slotIndex = base + syncIndex;
                if (slotIndex < layerPerfSlots.size()) lp = &layerPerfSlots[slotIndex];
            }

            if (lp != nullptr) {
                // Reusable barrier (race-free): last arriving thread advances epoch.
                // This avoids a missed-epoch deadlock when one thread observes the
                // incremented epoch and then waits for the next increment.
                const NnUint observedEpoch = lp->epoch.load(std::memory_order_acquire);
                const NnUint arrived = lp->arrived.fetch_add(1u, std::memory_order_acq_rel) + 1u;
                if (arrived == nThreads) {
                    lp->arrived.store(0u, std::memory_order_release);
                    lp->done.store(0u, std::memory_order_release);
                    lp->epoch.store(observedEpoch + 1u, std::memory_order_release);
                } else {
                    while (lp->epoch.load(std::memory_order_acquire) == observedEpoch) {
                    }
                }

                if (threadIndex == 0) {
                    const NnUint myNode = nodeConfig->nodeIndex;
                    const NnUint stageRoot = myStage->rootNodeIndex;
                    const bool amRoot = (myNode == stageRoot);

                    const NnUint layerIndex = segmentMeta[segmentIndex].layerIndex;
                    if (layerIndex != 0xFFFFFFFFu && layerIndex < execution->layerPerf->nLayers) {
                        // Lazy init root cache.
                        if (lastLayerPerfByLayer.empty()) {
                            const NnUint nLayers = execution->layerPerf->nLayers;
                            lastLayerPerfByLayer.resize(nLayers);
                            for (NnUint l = 0; l < nLayers; ++l) {
                                lastLayerPerfByLayer[l].resize(myStage->nNodes);
                            }
                        }

                        NnLayerPerfMsg msg{};
                        msg.magic = NN_LAYER_PERF_MAGIC;
                        msg.version = NN_LAYER_PERF_VERSION;
                        msg.stageIndex = myStage->stageIndex;
                        msg.nodeIndex = myNode;
                        msg.layerIndex = layerIndex;
                        msg.attnUs = (NnUint)std::min<unsigned long long>(execution->layerPerf->attnUs[layerIndex], 0xFFFFFFFFull);
                        msg.ffnUs  = (NnUint)std::min<unsigned long long>(execution->layerPerf->ffnUs[layerIndex],  0xFFFFFFFFull);

                        if (!amRoot) {
                            network->sendToNode(stageRoot, myNode, &msg, sizeof(msg));
                        } else {
                            maybeInitLayerPerfSnapshot();
                            // Root: receive from all other stage nodes.
                            for (NnUint i = 0; i < myStage->nNodes; ++i) {
                                const NnUint peer = myStage->nodeIndices[i];
                                if (peer == myNode) continue;
                                NnLayerPerfMsg in{};
                                network->recvFromNode(peer, myNode, &in, sizeof(in));
                                if (in.magic != NN_LAYER_PERF_MAGIC || in.version != NN_LAYER_PERF_VERSION) continue;
                                if (in.layerIndex != layerIndex) continue;
                                const int loc = (in.nodeIndex < (NnUint)stageLocalIndexByGlobalNode.size()) ? stageLocalIndexByGlobalNode[in.nodeIndex] : -1;
                                if (loc >= 0 && in.layerIndex < lastLayerPerfByLayer.size()) {
                                    lastLayerPerfByLayer[in.layerIndex][(size_t)loc] = in;
                                }
                            }

                            // Store self.
                            const int myLocal = (myNode < (NnUint)stageLocalIndexByGlobalNode.size()) ? stageLocalIndexByGlobalNode[myNode] : -1;
                            if (myLocal >= 0 && layerIndex < lastLayerPerfByLayer.size()) {
                                lastLayerPerfByLayer[layerIndex][(size_t)myLocal] = msg;
                            }

                            publishLayerPerfSnapshot(layerIndex);

                            // Optional printing on stage root.
                            if (layerPerfPrintEnabled) {
                                for (NnUint i = 0; i < myStage->nNodes; ++i) {
                                    const NnLayerPerfMsg &p = lastLayerPerfByLayer[layerIndex][i];
                                    if (p.magic != NN_LAYER_PERF_MAGIC) continue;
                                    printf("[layer-prof] stage=%u layer=%u node=%u attn=%u us ffn=%u us\n",
                                           (unsigned)p.stageIndex, (unsigned)p.layerIndex, (unsigned)p.nodeIndex,
                                           (unsigned)p.attnUs, (unsigned)p.ffnUs);
                                }
                                fflush(stdout);
                            }
                        }
                    }
                }

                lp->done.fetch_add(1u, std::memory_order_acq_rel);
                while (lp->done.load(std::memory_order_acquire) < nThreads) {
                }
            }
        }

        if (slot != nullptr) {
            // Barrier end: wait all threads to finish this syncIndex.
            const NnUint done = slot->done.fetch_add(1u, std::memory_order_acq_rel) + 1u;
            if (done == nThreads) {
                slot->endUs.store(nowUs(), std::memory_order_release);
            }
            while (slot->done.load(std::memory_order_acquire) < nThreads) {
            }

            if (threadIndex == 0) {
                const long long startUs = slot->startUs.load(std::memory_order_acquire);
                const long long endUs = slot->endUs.load(std::memory_order_acquire);
                const double ms = (endUs > startUs) ? (double)(endUs - startUs) / 1000.0 : 0.0;
                const char *pipeName = pipeConfig->name ? pipeConfig->name : "(null)";
                const size_t totalBytes = (size_t)batchBytes * (size_t)execution->batchSize;
                printf(
                    "⏱️ [sync-prof] node=%u seg=%u sync=%u type=%s pipe=%u name=%s totalBytes=%zu wall=%.3f ms\n",
                    (unsigned)nodeConfig->nodeIndex,
                    (unsigned)segmentIndex,
                    (unsigned)syncIndex,
                    syncTypeStr,
                    (unsigned)syncConfig->pipeIndex,
                    pipeName,
                    totalBytes,
                    ms);
                std::fflush(stdout);
            }
        }
    }
}

static void writeString(NnNetwork *network, NnUint socketIndex, char *str) {
    NnUint bytes = std::strlen(str) + 1;
    network->write(socketIndex, &bytes, sizeof(NnUint));
    network->write(socketIndex, str, bytes);
}

static char *readString(NnNetwork *network, NnUint socketIndex) {
    NnUint bytes;
    network->read(socketIndex, &bytes, sizeof(NnUint));
    char *str = new char[bytes];
    network->read(socketIndex, str, bytes);
    return str;
}

static inline void setEnvPortable(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    setenv(name, value ? value : "", 1);
#endif
}

static inline void unsetEnvPortable(const char *name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

static std::vector<std::string> buildEnvSyncList() {
    // Default whitelist (debug/migration switches). Extend at runtime via:
    //   DLLAMA_SYNC_ENV_VARS="FOO,BAR,BAZ"
    static const char *kDefaults[] = {
        "DLLAMA_FORCE_MATMUL_VIEWS",
        "DLLAMA_KV_AGGREGATE",
        "DLLAMA_KV_AGGREGATE_PROOF",
        "DLLAMA_KV_AGGREGATE_PROOF_ALL_BATCHES",
        "DLLAMA_SYNC_TRACE",
        "DLLAMA_SYNC_TRACE_ALL_THREADS",
        "DLLAMA_IO_TIMEOUT_MS",
        "DLLAMA_CONTROL_LOG",
        "DLLAMA_DEBUG_KVCACHE_PER_HEAD_SHIFT"
    };

    std::set<std::string> names;
    for (size_t i = 0; i < sizeof(kDefaults) / sizeof(kDefaults[0]); ++i) {
        names.insert(std::string(kDefaults[i]));
    }

    const char *extra = std::getenv("DLLAMA_SYNC_ENV_VARS");
    if (extra != nullptr && extra[0] != '\0') {
        std::string s(extra);
        size_t start = 0;
        while (start < s.size()) {
            size_t end = s.find(',', start);
            if (end == std::string::npos) end = s.size();
            std::string token = s.substr(start, end - start);
            // trim spaces
            size_t l = 0;
            while (l < token.size() && (token[l] == ' ' || token[l] == '\t' || token[l] == '\n' || token[l] == '\r')) l++;
            size_t r = token.size();
            while (r > l && (token[r - 1] == ' ' || token[r - 1] == '\t' || token[r - 1] == '\n' || token[r - 1] == '\r')) r--;
            if (r > l) names.insert(token.substr(l, r - l));
            start = end + 1;
        }
    }

    std::vector<std::string> out;
    out.reserve(names.size());
    for (const auto &n : names) out.push_back(n);
    return out;
}

static void writeSyncedEnvVars(NnNetwork *network, NnUint socketIndex) {
    const std::vector<std::string> names = buildEnvSyncList();
    const NnUint n = (NnUint)names.size();
    network->write(socketIndex, &n, sizeof(n));
    for (const auto &name : names) {
        const char *cname = name.c_str();
        writeString(network, socketIndex, const_cast<char *>(cname));
        const char *val = std::getenv(cname);
        const NnUint present = (val != nullptr) ? 1u : 0u;
        network->write(socketIndex, &present, sizeof(present));
        if (present) {
            writeString(network, socketIndex, const_cast<char *>(val));
        }
    }
}

static void readAndApplySyncedEnvVars(NnNetwork *network, NnUint socketIndex) {
    NnUint n = 0;
    network->read(socketIndex, &n, sizeof(n));
    for (NnUint i = 0; i < n; ++i) {
        char *name = readString(network, socketIndex);
        NnUint present = 0;
        network->read(socketIndex, &present, sizeof(present));
        if (present) {
            char *val = readString(network, socketIndex);
            setEnvPortable(name, val);
            delete[] val;
        } else {
            unsetEnvPortable(name);
        }
        delete[] name;
    }
}

NnRootConfigWriter::NnRootConfigWriter(NnNetwork *network) {
    this->network = network;
}

void NnRootConfigWriter::writeNet(NnUint socketIndex, NnNetConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nBatches, sizeof(config->nBatches));
    network->write(socketIndex, &config->nNodes, sizeof(config->nNodes));
    network->write(socketIndex, &config->nPipes, sizeof(config->nPipes));
    for (NnUint pipeIndex = 0; pipeIndex < config->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config->pipes[pipeIndex];
        network->write(socketIndex, &pipeConfig->size, sizeof(pipeConfig->size));
        writeString(network, socketIndex, pipeConfig->name);
    }
    network->write(socketIndex, &config->nPreSyncs, sizeof(config->nPreSyncs));
    for (NnUint preSyncIndex = 0; preSyncIndex < config->nPreSyncs; preSyncIndex++) {
        NnPreSyncConfig *preSyncConfig = &config->preSyncs[preSyncIndex];
        network->write(socketIndex, &preSyncConfig->pipeIndex, sizeof(preSyncConfig->pipeIndex));
    }

    // Sync a whitelist of environment-variable switches from root to workers.
    // This runs during communication initialization, before any execution begins.
    writeSyncedEnvVars(network, socketIndex);

    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeNode(NnUint socketIndex, NnNodeConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nodeIndex, sizeof(config->nodeIndex));
    network->write(socketIndex, &config->nBuffers, sizeof(config->nBuffers));
    network->write(socketIndex, &config->nSegments, sizeof(config->nSegments));

    for (NnUint bufferIndex = 0; bufferIndex < config->nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config->buffers[bufferIndex];
        network->write(socketIndex, &bufferConfig->size, sizeof(bufferConfig->size));
        writeString(network, socketIndex, bufferConfig->name);
    }

    for (NnUint segmentIndex = 0; segmentIndex < config->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config->segments[segmentIndex];
        network->write(socketIndex, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->write(socketIndex, &segmentConfig->nOps, sizeof(segmentConfig->nOps));

        for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
            NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
            network->write(socketIndex, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
            network->write(socketIndex, &syncConfig->syncType, sizeof(syncConfig->syncType));
        }
        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            network->write(socketIndex, &opConfig->code, sizeof(opConfig->code));
            network->write(socketIndex, &opConfig->index, sizeof(opConfig->index));
            network->write(socketIndex, &opConfig->weightSize, sizeof(opConfig->weightSize));
            network->write(socketIndex, &opConfig->configSize, sizeof(opConfig->configSize));
            writeString(network, socketIndex, opConfig->name);
            network->write(socketIndex, &opConfig->input, sizeof(opConfig->input));
            network->write(socketIndex, &opConfig->output, sizeof(opConfig->output));
            if (opConfig->configSize > 0)
                network->write(socketIndex, opConfig->config, opConfig->configSize);
        }
    }
    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs) {
    for (NnUint nodeIndex = 1; nodeIndex < netConfig->nNodes; nodeIndex++) {
        NnUint socketIndex = nodeIndex - 1;
        writeNet(socketIndex, netConfig);
        writeNode(socketIndex, &nodeConfigs[nodeIndex]);
    }
}

NnWorkerConfigReader::NnWorkerConfigReader(NnNetwork *network) {
    this->network = network;
}

NnNetConfig NnWorkerConfigReader::readNet() {
    network->readAck(ROOT_SOCKET_INDEX);
    NnNetConfig config;
    network->read(ROOT_SOCKET_INDEX, &config.nBatches, sizeof(config.nBatches));
    network->read(ROOT_SOCKET_INDEX, &config.nNodes, sizeof(config.nNodes));
    network->read(ROOT_SOCKET_INDEX, &config.nPipes, sizeof(config.nPipes));
    config.pipes = new NnPipeConfig[config.nPipes];
    for (NnUint pipeIndex = 0; pipeIndex < config.nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config.pipes[pipeIndex];
        network->read(ROOT_SOCKET_INDEX, &pipeConfig->size, sizeof(pipeConfig->size));
        pipeConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }
    network->read(ROOT_SOCKET_INDEX, &config.nPreSyncs, sizeof(config.nPreSyncs));
    config.preSyncs = new NnPreSyncConfig[config.nPreSyncs];
    for (NnUint preSyncIndex = 0; preSyncIndex < config.nPreSyncs; preSyncIndex++) {
        NnPreSyncConfig *preSyncConfig = &config.preSyncs[preSyncIndex];
        network->read(ROOT_SOCKET_INDEX, &preSyncConfig->pipeIndex, sizeof(preSyncConfig->pipeIndex));
    }

    // Apply env switches synced from root.
    readAndApplySyncedEnvVars(network, ROOT_SOCKET_INDEX);

    network->writeAck(ROOT_SOCKET_INDEX);
    return config;
}

NnNodeConfig NnWorkerConfigReader::readNode() {
    network->readAck(ROOT_SOCKET_INDEX);

    NnNodeConfig config;
    network->read(ROOT_SOCKET_INDEX, &config.nodeIndex, sizeof(config.nodeIndex));
    network->read(ROOT_SOCKET_INDEX, &config.nBuffers, sizeof(config.nBuffers));
    network->read(ROOT_SOCKET_INDEX, &config.nSegments, sizeof(config.nSegments));

    config.buffers = new NnBufferConfig[config.nBuffers];
    config.segments = new NnSegmentConfig[config.nSegments];

    for (NnUint bufferIndex = 0; bufferIndex < config.nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config.buffers[bufferIndex];
        network->read(ROOT_SOCKET_INDEX, &bufferConfig->size, sizeof(bufferConfig->size));
        bufferConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }

    for (NnUint segmentIndex = 0; segmentIndex < config.nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config.segments[segmentIndex];
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nOps, sizeof(segmentConfig->nOps));

        if (segmentConfig->nSyncs > 0) {
            segmentConfig->syncs = new NnSyncConfig[segmentConfig->nSyncs];

            for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
                NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
                network->read(ROOT_SOCKET_INDEX, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->syncType, sizeof(syncConfig->syncType));
            }
        }

        if (segmentConfig->nOps > 0) {
            segmentConfig->ops = new NnOpConfig[segmentConfig->nOps];

            for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
                NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
                network->read(ROOT_SOCKET_INDEX, &opConfig->code, sizeof(opConfig->code));
                network->read(ROOT_SOCKET_INDEX, &opConfig->index, sizeof(opConfig->index));
                network->read(ROOT_SOCKET_INDEX, &opConfig->weightSize, sizeof(opConfig->weightSize));
                network->read(ROOT_SOCKET_INDEX, &opConfig->configSize, sizeof(opConfig->configSize));
                opConfig->name = readString(network, ROOT_SOCKET_INDEX);
                network->read(ROOT_SOCKET_INDEX, &opConfig->input, sizeof(opConfig->input));
                network->read(ROOT_SOCKET_INDEX, &opConfig->output, sizeof(opConfig->output));
                if (opConfig->configSize > 0) {
                    opConfig->config = new NnByte[opConfig->configSize];
                    network->read(ROOT_SOCKET_INDEX, opConfig->config, opConfig->configSize);
                }
            }
        }
    }
    network->writeAck(ROOT_SOCKET_INDEX);
    return config;
}

NnRootWeightLoader::NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnUint nNodes) {
    this->executor = executor;
    this->network = network;
    this->nNodes = nNodes;
    this->tempSize = 0;
}

NnRootWeightLoader::~NnRootWeightLoader() {
    if (tempSize > 0)
        delete[] temp;
}

void NnRootWeightLoader::finish() {
    NnUint zeroSize = 0;
    for (NnUint socketIndex = 0; socketIndex < nNodes - 1; socketIndex++) {
        network->write(socketIndex, &zeroSize, sizeof(zeroSize));
        network->readAck(socketIndex);
    }
    if (tempSize > 0) {
        delete[] temp;
        tempSize = 0;
    }
}

void NnRootWeightLoader::allocate(NnSize size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnRootWeightLoader::writeWeight(NnUint nodeIndex, const char *opName, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    NnUint nameSize = std::strlen(opName) + 1;
    NnUint socketIndex = nodeIndex - 1;
    network->write(socketIndex, &nameSize, sizeof(nameSize));
    network->write(socketIndex, opName, nameSize);
    network->write(socketIndex, &opIndex, sizeof(opIndex));
    network->write(socketIndex, &offset, sizeof(offset));
    network->write(socketIndex, &nBytes, sizeof(nBytes));
    network->write(socketIndex, weight, nBytes);
}

NnSize NnRootWeightLoader::loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    return nBytes;
}

NnSize NnRootWeightLoader::loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    executor->loadWeight(opName, opIndex, 0u, nBytes, weight);

    if (nNodes > 1u) {
        for (NnUint nodeIndex = 1u; nodeIndex < nNodes; nodeIndex++)
            writeWeight(nodeIndex, opName, opIndex, 0u, nBytes, weight);
    }
    return nBytes;
}

NnSize NnRootWeightLoader::loadRowMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight) {
    const NnUint offset = expertIndex * slice->sliceSize.nBytes;
    if (nNodes == 1u) {
        executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
    } else {
        allocate(slice->sliceSize.nBytes);
        for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
            splitRowMatmulWeight(slice, nodeIndex, weight, temp);
            if (nodeIndex == 0u)
                executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, temp);
            else
                writeWeight(nodeIndex, opName, opIndex, offset, slice->sliceSize.nBytes, temp);
        }
    }
    return slice->size.nBytes;
}

NnSize NnRootWeightLoader::loadColMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight) {
    const NnUint offset = expertIndex * slice->sliceSize.nBytes;
    if (nNodes == 1) {
        executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
    } else {
        allocate(slice->sliceSize.nBytes);
        for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
            splitColMatmulWeight(slice, nodeIndex, weight, temp);
            if (nodeIndex == 0)
                executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, temp);
            else
                writeWeight(nodeIndex, opName, opIndex, offset, slice->sliceSize.nBytes, temp);
        }
    }
    return slice->size.nBytes;
}

NnWorkerWeightReader::NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network) {
    this->executor = executor;
    this->network = network;
    this->tempSize = 0;
}

NnWorkerWeightReader::~NnWorkerWeightReader() {
    if (tempSize > 0)
        delete[] temp;
}

void NnWorkerWeightReader::allocate(NnUint size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnWorkerWeightReader::read() {
    NnUint nameSize;
    NnUint opIndex;
    NnSize offset;
    NnSize nBytes;
    while (true) {
        network->read(0, &nameSize, sizeof(nameSize));
        if (nameSize == 0) {
            network->writeAck(ROOT_SOCKET_INDEX);
            if (tempSize > 0) {
                delete temp;
                tempSize = 0;
            }
            break;
        }
        std::unique_ptr<char[]> opNamePtr(new char[nameSize]);
        char *opName = opNamePtr.get();
        network->read(ROOT_SOCKET_INDEX, opName, nameSize);
        network->read(ROOT_SOCKET_INDEX, &opIndex, sizeof(opIndex));
        network->read(ROOT_SOCKET_INDEX, &offset, sizeof(offset));
        network->read(ROOT_SOCKET_INDEX, &nBytes, sizeof(nBytes));
        allocate(nBytes);
        network->read(0, temp, nBytes);
        executor->loadWeight(opName, opIndex, offset, nBytes, temp);
        printf("💿 Loaded %22s %3d, %12zu kB\n", opName, opIndex, nBytes / 1024);
    }
    printf("💿 Weights loaded\n");
}
