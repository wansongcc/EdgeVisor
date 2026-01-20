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

#include <sys/stat.h>
#include "nn-core.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <fcntl.h>
#ifndef _WIN32
#include <poll.h>
#endif

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

#define ACK 23571114
#define MAX_CHUNK_SIZE 65536

// ---------------------------------------------------------
// Optional readMany timeout for debugging hangs.
// Default 0 (disabled; preserves legacy blocking behavior).
// Enable with: -DNN_NETWORK_READ_MANY_TIMEOUT_MS=5000
// ---------------------------------------------------------
#ifndef NN_NETWORK_READ_MANY_TIMEOUT_MS
#define NN_NETWORK_READ_MANY_TIMEOUT_MS 0
#endif

// Optional diagnostics for readMany timeout.
#ifndef NN_NETWORK_READ_MANY_DIAG
#define NN_NETWORK_READ_MANY_DIAG 1
#endif

static inline bool isEagainError() {
    #ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
    #else
    return SOCKET_LAST_ERRCODE == EAGAIN;
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

static inline bool shardingUpdateDebugEnabled() {
    const char *v = std::getenv("DLLAMA_DEBUG_SHARDING_UPDATE");
    return v != nullptr && v[0] == '1';
}

static inline bool shardingMatmulDebugEnabledOnRoot() {
    const char *v = std::getenv("DLLAMA_DEBUG_SHARDING_MATMUL");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
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

static NnUint getSplitTotalForStageNodesRuntime(const NnDimSplit *split, const NnStageConfig* stage, NnUint nNodes) {
    if (split == nullptr || split->lengths == nullptr) return 0u;
    if (!stage) {
        NnUint sum = 0u;
        for (NnUint i = 0u; i < nNodes; ++i) sum += split->lengths[i];
        return sum;
    }
    NnUint sum = 0u;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        sum += split->lengths[stage->nodeIndices[i]];
    }
    return sum;
}

static bool fillRuntimeSlicesFromSharding(
    const NnLayerShardingTable *layerSharding,
    NnUint layerIndex,
    NnUint nNodes,
    NnSize totalBytes,
    std::vector<NnSize>& offsets,
    std::vector<NnSize>& sizes,
    NnFloatType floatType,
    const NnStageConfig* stageForSplit,
    NnUint totalElements
) {
    if (layerSharding == nullptr) return false;
    const NnUint epoch = layerSharding->epoch.load(std::memory_order_acquire);
    if (epoch == 0u) return false;
    if (layerIndex >= layerSharding->nLayers) return false;

    bool matchFound = false;
    bool stackedByNode = false;

    for (NnUint i = 0; i < nNodes; ++i) {
        offsets[i] = 0;
        sizes[i] = 0;
    }

    const NnLayerSplits &ls = layerSharding->layers[layerIndex];

    if (stageForSplit != nullptr && totalElements > 0) {
        const NnUint dimTotal = getSplitTotalForStageNodesRuntime(&ls.dimSplit, stageForSplit, nNodes);
        if (dimTotal > 0 && totalElements == dimTotal * nNodes) {
            stackedByNode = true;
        }
    }

    if (stackedByNode) {
        if (totalBytes % nNodes == 0) {
            const NnSize slotBytes = totalBytes / nNodes;
            for (NnUint k = 0; k < stageForSplit->nNodes; ++k) {
                const NnUint node = stageForSplit->nodeIndices[k];
                offsets[node] = (NnSize)node * slotBytes;
                sizes[node] = slotBytes;
            }
            matchFound = true;
        }
    }

    auto tryMatch = [&](const NnDimSplit &split, bool allowMultiplier) -> bool {
        if (stackedByNode) return false;
        const NnUint totalUnits = getSplitTotalForStageNodesRuntime(&split, stageForSplit, nNodes);
        if (totalUnits == 0u) return false;

        if (totalElements > 0) {
            if (totalElements % totalUnits == 0u) {
                const NnUint multiplier = totalElements / totalUnits;
                if (!allowMultiplier && multiplier != 1u) return false;
                if (stageForSplit) {
                    for (NnUint k = 0; k < stageForSplit->nNodes; ++k) {
                        const NnUint node = stageForSplit->nodeIndices[k];
                        const NnUint offElems = split.starts[node] * multiplier;
                        const NnUint lenElems = split.lengths[node] * multiplier;
                        offsets[node] = getBytes(floatType, offElems);
                        sizes[node] = getBytes(floatType, lenElems);
                    }
                } else {
                    for (NnUint node = 0; node < nNodes; ++node) {
                        const NnUint offElems = split.starts[node] * multiplier;
                        const NnUint lenElems = split.lengths[node] * multiplier;
                        offsets[node] = getBytes(floatType, offElems);
                        sizes[node] = getBytes(floatType, lenElems);
                    }
                }
                return true;
            }
        }

        if (totalBytes % totalUnits == 0u) {
            const NnSize bytesPerUnit = totalBytes / totalUnits;
            if (stageForSplit) {
                for (NnUint k = 0; k < stageForSplit->nNodes; ++k) {
                    const NnUint node = stageForSplit->nodeIndices[k];
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

    if (!matchFound) matchFound = tryMatch(ls.vocabSplit, false);
    if (!matchFound) matchFound = tryMatch(ls.ffnSplit, false);
    if (!matchFound) matchFound = tryMatch(ls.dimSplit, false);
    if (!matchFound) matchFound = tryMatch(ls.headSplit, true);
    if (!matchFound) matchFound = tryMatch(ls.kvHeadSplit, true);

    return matchFound;
}

static void fillUnevenSlices(const NnUnevenPartitionPlan *plan, NnUint nNodes, NnSize totalBytes, 
                             std::vector<NnSize>& offsets, std::vector<NnSize>& sizes,
                             NnFloatType floatType,
                             const NnStageConfig* stageForSplit,
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

        // Priority order for matching:
        // 1. Vocab (Logits) - Largest, usually most critical for the "degeneration" bug
        if (!matchFound) matchFound = tryMatch(plan->vocabSplit, false, "vocab");
        // 2. FFN - Intermediate layers
        if (!matchFound) matchFound = tryMatch(plan->ffnSplit, false, "ffn");
        // 3. Dim - General dimension splits
        if (!matchFound) matchFound = tryMatch(plan->dimSplit, false, "dim");
        // 4. Heads - Attention Q
        if (!matchFound) matchFound = tryMatch(plan->headSplit, true, "head");
        // 5. KV Heads - Attention K/V
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
    while (size > 0) {
        ssize_t s = send(socket, (const char*)data, size, 0);
        if (s < 0) {
            if (isEagainError()) {
                continue;
            }
            throw NnTransferSocketException(0, "Error writing to socket");
        } else if (s == 0) {
            throw NnTransferSocketException(0, "Socket closed");
        }
        size -= s;
        data = (const char*)data + s;
    }
}

static inline bool tryReadSocket(int socket, void *data, NnSize size, unsigned long maxAttempts) {
    // maxAttempts = 0 means infinite attempts
    NnSize s = size;
    while (s > 0) {
        ssize_t r = recv(socket, (char*)data, s, 0);
        if (r < 0) {
            if (isEagainError()) {
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
    printf("⭕ nNodes: %d\n", nNodes);
    readSocket(rootSocketFd, &nodeIndex, sizeof(nodeIndex));
    printf("⭕ NodeIndex: %d\n", nodeIndex);

    std::vector<NnSocket> sockets(nSockets);
    sockets[0].assign(rootSocket.release());

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
        NnUint socketIndex = i + 1;
        if (i >= nodeIndex) {
            printf("⭕ Socket[%d]: connecting to %s:%d worker\n", socketIndex, host, port);
            sockets[socketIndex].assign(connectSocket(host, port));
            printf("⭕ Socket[%d]: connected\n", socketIndex);
        } else {
            printf("⭕ Socket[%d]: wait for %s:%d worker\n", socketIndex, host, port);
            sockets[socketIndex].assign(acceptSocket(socketSocket.fd));
            printf("⭕ Socket[%d]: accepted\n", socketIndex);
        }
    }

    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(&sockets));
}

std::unique_ptr<NnNetwork> NnNetwork::connect(NnUint nSockets, char **hosts, NnUint *ports) {
    assert(nSockets > 0);

    std::vector<NnSocket> sockets(nSockets);
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
    }
    for (NnUint i = 0; i < nSockets; i++) {
        writeAckPacket(sockets[i].fd);
    }
    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(&sockets));
}

NnNetwork::NnNetwork(std::vector<NnSocket> *sockets) {
    this->nSockets = sockets->size();
    this->sockets = new int[nSockets];
    for (NnUint i = 0; i < nSockets; i++)
        this->sockets[i] = sockets->at(i).release();
    this->sentBytes = new NnSize[nSockets];
    this->recvBytes = new NnSize[nSockets];
}

NnNetwork::~NnNetwork() {
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
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        sentBytes[io->socketIndex] += io->size;
    }
    do {
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
                        continue;
                    }
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (s == 0) {
                    throw NnTransferSocketException(0, "Socket closed");
                }
                io->size -= s;
                io->data = (char*)io->data + s;
            }
        }
    } while (isWriting);
}

void NnNetwork::writeAll(void *data, NnSize size) {
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
#if NN_NETWORK_READ_MANY_TIMEOUT_MS > 0 && !defined(_WIN32)
    // Use poll() to avoid indefinite blocking when a peer never sends.
    // This is especially useful for diagnosing deadlocks / missing replies.
    const auto start = std::chrono::steady_clock::now();
    const int timeoutMs = (int)NN_NETWORK_READ_MANY_TIMEOUT_MS;
#endif

    bool isReading;
    NnSize nBytes = 0;
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        recvBytes[io->socketIndex] += io->size;
    }
    do {
        isReading = false;

#if NN_NETWORK_READ_MANY_TIMEOUT_MS > 0 && !defined(_WIN32)
        // Build pollfd list for sockets that still need data.
        std::vector<pollfd> fds;
        fds.reserve(n);
        std::vector<NnUint> idxMap;
        idxMap.reserve(n);
        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size == 0) continue;
            pollfd pfd;
            pfd.fd = sockets[io->socketIndex];
            pfd.events = POLLIN;
            pfd.revents = 0;
            fds.push_back(pfd);
            idxMap.push_back(i);
        }

        if (!fds.empty()) {
            int pr = ::poll(fds.data(), fds.size(), timeoutMs);
            if (pr == 0) {
                // Timeout: print pending sockets and remaining bytes.
#if NN_NETWORK_READ_MANY_DIAG
                const auto now = std::chrono::steady_clock::now();
                const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                fprintf(stderr, "🚨 readMany timeout after %lldms (want=%u) pending=", (long long)elapsedMs, n);
                for (NnUint i = 0; i < n; i++) {
                    NnSocketIo *io = &ios[i];
                    if (io->size == 0) continue;
                    fprintf(stderr, "[%u sockIndex=%u fd=%d remaining=%zu] ", i, io->socketIndex, sockets[io->socketIndex], (size_t)io->size);
                }
                fprintf(stderr, "\n");
#endif
                throw NnTransferSocketException(ETIMEDOUT, "readMany timeout (peer did not send)");
            }
            if (pr < 0) {
                throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
            }
        }
#endif

        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isReading = true;
                int socket = sockets[io->socketIndex];
                ssize_t r = recv(socket, (char*)io->data, io->size, 0);
                if (r < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (r == 0) {
                    throw NnTransferSocketException(0, "Socket closed");
                }
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
    // 假设网络是全连接 Mesh，sockets 数组按 Node ID 排序 (跳过自己)
    if (targetNodeIndex < myNodeIndex) {
        return (int)targetNodeIndex;
    }
    if (targetNodeIndex > myNodeIndex) {
        return (int)targetNodeIndex - 1;
    }
    return -1; // Should not happen (target == self)
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
        if (stage) {
            // Stage 内广播：只发给组内其他节点
            for(NnUint i=0; i<stage->nNodes; ++i) {
                NnUint target = stage->nodeIndices[i];
                if(target != myNodeIndex) {
                    int sock = network->getSocketIndexForNode(target, myNodeIndex);
                    if(sock >= 0) targetSockets.push_back(sock);
                }
            }
        } else {
            // 全局广播：发给所有 Socket (简单处理)
            // 注意：这假设 network->nSockets 包含了所有 Worker
            for(NnUint i=0; i<network->nSockets; ++i) targetSockets.push_back(i);
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
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            ios[i].socketIndex = targetSockets[startIdx + i]; // 使用真实的 Socket Index
            ios[i].data = buffer;
            ios[i].size = nBytes;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);

        // [新增] Root 等待 Workers 确认 (ACK)
        // 确保 Workers 已经接收完数据，实现同步屏障
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            network->readAck(ios[i].socketIndex);
        }

    } else {
        // --- Worker 接收 ---

        if (threadIndex != 0) return; // 接收通常只需要一个线程

        int rootSocketIndex = network->getSocketIndexForNode(groupRootIndex, myNodeIndex);
        if (rootSocketIndex < 0) {
            // 异常：找不到 Root 的连接
            return; 
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

// =====================================================================================
// Stage-root sharding update decision + packed message
// =====================================================================================

static bool tryParseEnvUint(const char *name, NnUint *out) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0')
        return false;
    char *end = nullptr;
    unsigned long v = std::strtoul(env, &end, 10);
    if (end == env)
        return false;
    if (v > 0xfffffffful)
        return false;
    *out = (NnUint)v;
    return true;
}

static bool tryReadFileToBuffer(const char *path, char *buf, size_t bufSize) {
    if (path == nullptr || path[0] == '\0' || buf == nullptr || bufSize == 0)
        return false;
    FILE *f = std::fopen(path, "rb");
    if (!f)
        return false;
    const size_t n = std::fread(buf, 1, bufSize - 1, f);
    std::fclose(f);
    buf[n] = '\0';
    return n > 0;
}

static const char *getShardingUpdateRequestFilePath() {
    // Default request file path for runtime sharding updates.
    // Can be overridden by setting DLLAMA_SHARDING_UPDATE_FILE.
    const char *p = std::getenv("DLLAMA_SHARDING_UPDATE_FILE");
    if (p != nullptr && p[0] != '\0')
        return p;
    return "/tmp/dllama_sharding_update.req";
}

static bool parseUintList(const char *s, std::vector<NnUint> &out) {
    out.clear();
    if (s == nullptr) return false;
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
        if (*p == '\0') break;

        // Stop list when the next token isn't a number (e.g. next key like "head_lens=").
        if (*p < '0' || *p > '9')
            break;

        char *end = nullptr;
        unsigned long v = std::strtoul(p, &end, 10);
        if (end == p)
            break;
        out.push_back((NnUint)v);
        p = end;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == ',') {
            ++p;
            continue;
        }
        // Otherwise, next loop will skip whitespace and either parse another number
        // (if there's a comma-separated continuation) or stop on a non-number token.
    }
    return !out.empty();
}

static uint64_t fnv1a64(const char *s) {
    // FNV-1a 64-bit
    uint64_t h = 1469598103934665603ull;
    if (s == nullptr)
        return h;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        h ^= (uint64_t)(*p);
        h *= 1099511628211ull;
    }
    return h;
}

static bool extractLastNonEmptyLine(const char *buf, char *out, size_t outCap) {
    if (out == nullptr || outCap == 0)
        return false;
    out[0] = '\0';
    if (buf == nullptr)
        return false;

    // Find end
    const char *end = buf;
    while (*end)
        ++end;

    // Trim trailing whitespace/newlines
    const char *p = end;
    while (p > buf) {
        const char c = p[-1];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') {
            --p;
            continue;
        }
        break;
    }
    if (p <= buf)
        return false;

    // Find start of the last line
    const char *lineEnd = p;
    const char *lineStart = lineEnd;
    while (lineStart > buf && lineStart[-1] != '\n' && lineStart[-1] != '\r')
        --lineStart;

    // Trim leading whitespace
    while (lineStart < lineEnd && (*lineStart == ' ' || *lineStart == '\t' || *lineStart == '\r' || *lineStart == '\n'))
        ++lineStart;
    if (lineStart >= lineEnd)
        return false;

    const size_t n = (size_t)(lineEnd - lineStart);
    const size_t copyN = (n + 1 < outCap) ? n : (outCap - 1);
    std::memcpy(out, lineStart, copyN);
    out[copyN] = '\0';
    return true;
}

static bool pollShardingUpdateFile(
    const char *path,
    uint64_t *inoutLastLineHash,
    NnUint *outLayer,
    bool *outHasPos,
    NnUint *outPos,
    bool *outHasHeadLens,
    std::vector<NnUint> *outStageNodes,
    std::vector<NnUint> *outHeadLens
) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    char buf[1024];
    if (!tryReadFileToBuffer(path, buf, sizeof(buf))) {
        return false;
    }

    // Only the last non-empty line is treated as the command.
    char line[1024];
    if (!extractLastNonEmptyLine(buf, line, sizeof(line))) {
        return false;
    }

    // Gate by last line content (not mtime) to avoid re-triggering the same command.
    const uint64_t h = fnv1a64(line);
    if (inoutLastLineHash != nullptr && *inoutLastLineHash == h) {
        return false;
    }

    // Supported formats (single line):
    //   "layer=21 pos=4"  or  "21 4"  or  "layer=21"  or  "21"
    // Optional override (stage-local):
    //   "layer=21 pos=4 stage_nodes=2,3 head_lens=6,10"
    // Whitespace / newlines are ignored.
    NnUint layer = 0u;
    NnUint pos = 0u;
    bool hasLayer = false;
    bool hasPos = false;
    bool hasHeadLens = false;
    std::vector<NnUint> stageNodes;
    std::vector<NnUint> headLens;

    // Key-value parse (robust): locate each key via substring search.
    // This avoids relying on ASCII whitespace/token boundaries (e.g., non-breaking spaces).
    if (const char *p = std::strstr(line, "layer=")) {
        p += 6;
        char *end = nullptr;
        unsigned long v = std::strtoul(p, &end, 10);
        if (end != p) {
            layer = (NnUint)v;
            hasLayer = true;
        }
    }
    if (const char *p = std::strstr(line, "pos=")) {
        p += 4;
        char *end = nullptr;
        unsigned long v = std::strtoul(p, &end, 10);
        if (end != p) {
            pos = (NnUint)v;
            hasPos = true;
        }
    }
    if (const char *p = std::strstr(line, "stage_nodes=")) {
        p += 12;
        if (!parseUintList(p, stageNodes))
            stageNodes.clear();
    }
    if (const char *p = std::strstr(line, "head_lens=")) {
        p += 9;
        if (parseUintList(p, headLens)) {
            hasHeadLens = true;
        } else {
            headLens.clear();
        }
    }

    // Fallback positional parse
    if (!hasLayer) {
        unsigned long a = 0, b = 0;
        int n = std::sscanf(line, "%lu %lu", &a, &b);
        if (n >= 1) {
            layer = (NnUint)a;
            hasLayer = true;
        }
        if (n >= 2) {
            pos = (NnUint)b;
            hasPos = true;
        }
    }

    if (!hasLayer)
        return false;

    if (hasHeadLens) {
        if (stageNodes.empty() || headLens.empty() || stageNodes.size() != headLens.size()) {
            // Invalid override: ignore it, but still allow layer/pos update.
            hasHeadLens = false;
            stageNodes.clear();
            headLens.clear();
        }
    }

    if (inoutLastLineHash != nullptr)
        *inoutLastLineHash = h;
    if (outLayer) *outLayer = layer;
    if (outHasPos) *outHasPos = hasPos;
    if (outPos) *outPos = pos;
    if (outHasHeadLens) *outHasHeadLens = hasHeadLens;
    if (outStageNodes) *outStageNodes = stageNodes;
    if (outHeadLens) *outHeadLens = headLens;
    return true;
}

static NnUint getBasePositionFromExecution(const NnNetExecution *execution) {
    // Convention in this codebase: pipe 0 holds positions as float (see WorkerLlmInference).
    if (execution == nullptr || execution->pipes == nullptr || execution->nPipes == 0u)
        return 0u;
    const float *positions = (const float *)execution->pipes[0];
    if (positions == nullptr)
        return 0u;
    // Positions are small (<= seqLen), safe to round-tripping through float.
    return (NnUint)positions[0];
}

static bool shouldEmitEnvShardingUpdate(NnUint currentLayerInGraph, NnUint basePosition) {
    NnUint targetLayer = 0u;
    if (!tryParseEnvUint("DLLAMA_SHARDING_UPDATE_LAYER", &targetLayer))
        return false;
    if (currentLayerInGraph != targetLayer)
        return false;

    NnUint targetPos = 0u;
    if (tryParseEnvUint("DLLAMA_SHARDING_UPDATE_POS", &targetPos)) {
        return basePosition == targetPos;
    }
    return true;
}

static void computeStartsFromLengths(NnUint nNodes, const NnUint *lengths, NnUint *starts) {
    if (nNodes == 0u) return;
    starts[0] = 0u;
    for (NnUint i = 1u; i < nNodes; ++i) {
        starts[i] = starts[i - 1u] + lengths[i - 1u];
    }
}

static void writeNoShardingUpdate(NnByte *buffer, NnSize nBytes) {
    if (nBytes < sizeof(NnShardingUpdateHeader))
        return;
    NnShardingUpdateHeader hdr;
    hdr.magic = 0u;
    hdr.version = NN_SHARDING_UPDATE_VERSION;
    hdr.epoch = 0u;
    hdr.layerIndex = 0u;
    hdr.nNodes = 0u;
    hdr.flags = shardingMatmulDebugEnabledOnRoot() ? NN_SHARDING_UPDATE_FLAG_DEBUG_MATMUL : 0u;
    hdr.applyStartLayer = 0u;
    hdr.applyEndLayerExclusive = 0u;
    std::memcpy(buffer, &hdr, sizeof(hdr));
}



static void maybePrepareStageShardingUpdate(
    const NnUnevenPartitionPlan *plan,
    const NnStageConfig *stage,
    NnUint targetLayerIndex,
    bool hasOverrideHeadLens,
    const std::vector<NnUint> &overrideStageNodes,
    const std::vector<NnUint> &overrideHeadLens,
    NnByte *buffer,
    NnSize nBytes,
    std::atomic_uint *epochCounter,
    NnUint *outEpoch
) {
    // Default: no-op update (workers will ignore)
    writeNoShardingUpdate(buffer, nBytes);

    if (plan == nullptr || stage == nullptr)
        return;
    if (targetLayerIndex < stage->startLayer || targetLayerIndex >= stage->endLayer)
        return;
    if (outEpoch) *outEpoch = 0u;

    const NnUint nNodes = plan->nNodes;
    const NnSize required = sizeof(NnShardingUpdateHeader) + (NnSize)(N_SPLIT_KINDS * 2u * nNodes * sizeof(NnUint));
    if (nBytes < required)
        return;

    // Build new splits by defaulting to plan splits.
    std::vector<NnUint> headLen(nNodes), headStart(nNodes);
    std::vector<NnUint> kvHeadLen(nNodes), kvHeadStart(nNodes);
    std::vector<NnUint> vocabLen(nNodes), vocabStart(nNodes);
    std::vector<NnUint> ffnLen(nNodes), ffnStart(nNodes);
    std::vector<NnUint> dimLen(nNodes), dimStart(nNodes);

    for (NnUint i = 0u; i < nNodes; ++i) {
        headLen[i] = plan->headSplit.lengths ? plan->headSplit.lengths[i] : 0u;
        kvHeadLen[i] = plan->kvHeadSplit.lengths ? plan->kvHeadSplit.lengths[i] : 0u;
        vocabLen[i] = plan->vocabSplit.lengths ? plan->vocabSplit.lengths[i] : 0u;
        ffnLen[i] = plan->ffnSplit.lengths ? plan->ffnSplit.lengths[i] : 0u;
        dimLen[i] = plan->dimSplit.lengths ? plan->dimSplit.lengths[i] : 0u;
    }



    // Option A (new): explicit override from request file.
    // We only allow overriding exactly this stage's nodes.
    if (hasOverrideHeadLens && overrideStageNodes.size() == stage->nNodes && overrideHeadLens.size() == stage->nNodes) {
        bool matchesStage = true;
        for (NnUint i = 0u; i < stage->nNodes; ++i) {
            bool found = false;
            for (NnUint j = 0u; j < overrideStageNodes.size(); ++j) {
                if (overrideStageNodes[j] == stage->nodeIndices[i]) { found = true; break; }
            }
            if (!found) { matchesStage = false; break; }
        }



        if (matchesStage) {
            // Preserve total heads within this stage.
            NnUint stageSumPlan = 0u;
            for (NnUint i = 0u; i < stage->nNodes; ++i)
                stageSumPlan += headLen[stage->nodeIndices[i]];

            unsigned long long desiredSum = 0ull;
            for (NnUint i = 0u; i < stage->nNodes; ++i)
                desiredSum += (unsigned long long)overrideHeadLens[i];

            if (stageSumPlan > 0u && desiredSum > 0ull) {
                // Normalize overrideHeadLens to sum==stageSumPlan using largest remainder.
                std::vector<unsigned long long> rema(stage->nNodes);
                std::vector<NnUint> norm(stage->nNodes);
                NnUint sumNorm = 0u;
                for (NnUint i = 0u; i < stage->nNodes; ++i) {
                    const unsigned long long num = (unsigned long long)overrideHeadLens[i] * (unsigned long long)stageSumPlan;
                    const unsigned long long q = num / desiredSum;
                    const unsigned long long r = num % desiredSum;
                    norm[i] = (NnUint)q;
                    rema[i] = r;
                    sumNorm += norm[i];
                }
                while (sumNorm < stageSumPlan) {
                    NnUint best = 0u;
                    for (NnUint i = 1u; i < stage->nNodes; ++i) {
                        if (rema[i] > rema[best]) best = i;
                    }
                    norm[best] += 1u;
                    rema[best] = 0u;
                    sumNorm += 1u;
                }
                while (sumNorm > stageSumPlan) {
                    NnUint best = 0u;
                    for (NnUint i = 1u; i < stage->nNodes; ++i) {
                        if (norm[i] > norm[best]) best = i;
                    }
                    if (norm[best] == 0u) break;
                    norm[best] -= 1u;
                    sumNorm -= 1u;
                }

                // Apply mapping: overrideStageNodes may be permuted.
                for (NnUint j = 0u; j < stage->nNodes; ++j) {
                    const NnUint node = overrideStageNodes[j];
                    // Find corresponding normalized entry by same index in override arrays.
                    // (Scheduler should align stage_nodes and head_lens.)
                    headLen[node] = norm[j];
                }
            }
        }
    } else {
        // Option B (legacy): rotate head lengths among nodes in this stage.
        // This changes distribution while keeping totals constant.
        if (stage->nNodes >= 2u && plan->headSplit.lengths != nullptr) {
            std::vector<NnUint> oldStage(stage->nNodes);
            for (NnUint i = 0u; i < stage->nNodes; ++i) {
                oldStage[i] = headLen[stage->nodeIndices[i]];
            }
            for (NnUint i = 0u; i < stage->nNodes; ++i) {
                NnUint src = (i + 1u) % stage->nNodes;
                headLen[stage->nodeIndices[i]] = oldStage[src];
            }
        }
    }

    if (shardingUpdateDebugEnabled()) {
        printf("[SHARD][NET][UPDATE][PREP] layer=%u stage=%u nNodes=%u stageNodes=%u ",
            targetLayerIndex,
            stage->stageIndex,
            stage->nNodes,
            stage->nNodes);
        for (NnUint i = 0u; i < stage->nNodes; ++i) {
            const NnUint node = stage->nodeIndices[i];
            printf("%u:%u%s", node, headLen[node], (i + 1u == stage->nNodes) ? "" : ",");
        }
        printf("\n");
    }

    computeStartsFromLengths(nNodes, headLen.data(), headStart.data());
    computeStartsFromLengths(nNodes, kvHeadLen.data(), kvHeadStart.data());
    computeStartsFromLengths(nNodes, vocabLen.data(), vocabStart.data());
    computeStartsFromLengths(nNodes, ffnLen.data(), ffnStart.data());
    computeStartsFromLengths(nNodes, dimLen.data(), dimStart.data());

    // Write header + payload.
    NnShardingUpdateHeader hdr;
    hdr.magic = NN_SHARDING_UPDATE_MAGIC;
    hdr.version = NN_SHARDING_UPDATE_VERSION;
    // Monotonic epoch: only bumps when we actually emit an update.
    hdr.epoch = (epochCounter != nullptr)
        ? (epochCounter->fetch_add(1u, std::memory_order_acq_rel) + 1u)
        : 1u;
    hdr.layerIndex = targetLayerIndex;
    hdr.nNodes = nNodes;
    hdr.flags = shardingMatmulDebugEnabledOnRoot() ? NN_SHARDING_UPDATE_FLAG_DEBUG_MATMUL : 0u;
    // Apply this update to the entire stage.
    // Note: this cannot retroactively change layers that have already executed in the
    // current forward pass; it takes effect for those layers on the next step.
    hdr.applyStartLayer = stage->startLayer;
    hdr.applyEndLayerExclusive = stage->endLayer;
    std::memcpy(buffer, &hdr, sizeof(hdr));

    if (outEpoch) *outEpoch = hdr.epoch;

    NnUint *payload = (NnUint *)(buffer + sizeof(hdr));
    auto writeKind = [&](const std::vector<NnUint> &starts, const std::vector<NnUint> &lens) {
        std::memcpy(payload, starts.data(), nNodes * sizeof(NnUint));
        payload += nNodes;
        std::memcpy(payload, lens.data(), nNodes * sizeof(NnUint));
        payload += nNodes;
    };

    writeKind(headStart, headLen);
    writeKind(kvHeadStart, kvHeadLen);
    writeKind(vocabStart, vocabLen);
    writeKind(ffnStart, ffnLen);
    writeKind(dimStart, dimLen);

    (void)outEpoch;
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
    const NnLayerShardingTable *layerSharding,
    NnUint layerIndex,
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

    const NnUint shardingEpoch = (layerSharding != nullptr)
        ? layerSharding->epoch.load(std::memory_order_acquire)
        : 0u;
    bool runtimeMatched = fillRuntimeSlicesFromSharding(layerSharding, layerIndex, nTotalNodes, nBytes,
        sliceOffsets, sliceSizes, floatType, stageForSplit, totalElements);
    if (!runtimeMatched) {
        // [Fix] Pass floatType + total elements for accurate matching (incl. Q80)
        fillUnevenSlices(plan, nTotalNodes, nBytes, sliceOffsets, sliceSizes, floatType, stageForSplit, totalElements);
    }

    if (onlyFromWorkerToRoot && shardingUpdateDebugEnabled() && threadIndex == 0u && shardingEpoch > 0u) {
        printf("[SHARD][SYNC][LOGITS] layer=%u epoch=%u runtime=%s\n",
            layerIndex, shardingEpoch, runtimeMatched ? "matched" : "fallback");
    }



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

    // --- 发送阶段 (Send) ---
    bool iShouldSend = true;
    
    // [修改] 如果我是 Root 且模式是 Worker->Root，我不发送
    if (onlyFromWorkerToRoot && amIRoot) iShouldSend = false; 

    // logits gather（LastStage -> Root）：发送端应发送自己在 pipe 中写入的那段（全局 offset 语义）。
    // 早期为了绕开 split 匹配失败导致的“写入在 0，但按全局 offset 读”而临时使用过 LOCAL0。
    // 现在 split 已 stage-aware 修复后，应回到全局 offset，否则 offset!=0 的 shard 会发送未写入区域（全 0）。
    // NOTE: isLogitsGather already computed above for stage-aware slicing.

    if (iShouldSend) {
        //  - 此处展示 TP 组内的 Gather 模式
        // 注意：mySliceData 的偏移量依赖于 fillUnevenSlices 的逻辑
        // 如果使用了之前讨论的“局部偏移重置”，这里 sliceOffsets[myNodeIndex] 也是正确的
        NnSize mySliceOffset = sliceOffsets[myNodeIndex];
        NnByte *mySliceData = &buffer[mySliceOffset];
        NnSize mySliceSize = sliceSizes[myNodeIndex];

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

        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            NnUint idx = startIdx + i;
            ios[i].socketIndex = targetSockets[idx];
            ios[i].data = mySliceData;
            ios[i].size = mySliceSize;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);
    }

    // --- 接收阶段 (Receive) ---
    bool iShouldRecv = true;
    
    // [修改] 如果我是 Worker 且模式是 Worker->Root，我不接收
    if (onlyFromWorkerToRoot && !amIRoot) iShouldRecv = false; 

    if (iShouldRecv) {
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            NnUint idx = startIdx + i;
            NnUint targetNode = targetNodeIndices[idx];

            ios[i].socketIndex = targetSockets[idx];
            ios[i].data = &buffer[sliceOffsets[targetNode]];
            ios[i].size = sliceSizes[targetNode]; 
        }
        network->readMany(nSocketsPerThread, &ios[0]);

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

NnNetworkNodeSynchronizer::NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, const NnUnevenPartitionPlan *plan, const NnLayerShardingTable *layerSharding) {
    this->network = network;
    this->execution = execution;
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->plan = plan;
    this->layerSharding = layerSharding;

    // Root-only bookkeeping: env-trigger emits at most one update per layer.
    // Total layer count is derived from the partition plan (max endLayer).
    if (plan != nullptr) {
        NnUint totalLayers = 0u;
        for (NnUint s = 0; s < plan->nStages; ++s) {
            if (plan->stages[s].endLayer > totalLayers)
                totalLayers = plan->stages[s].endLayer;
        }
        if (totalLayers > 0u) {
            envLayerUpdated.assign(totalLayers, (NnByte)0);
        }
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
}

void NnNetworkNodeSynchronizer::localBarrier(NnUint nThreads, NnUint threadIndex) {
    (void)threadIndex;
    const NnUint phase = localBarrierPhase.load(std::memory_order_acquire);
    const NnUint arrived = localBarrierCount.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    if (arrived == nThreads) {
        localBarrierCount.store(0u, std::memory_order_release);
        localBarrierPhase.fetch_add(1u, std::memory_order_release);
        return;
    }
    while (localBarrierPhase.load(std::memory_order_acquire) == phase) {
        // spin
    }
}

void NnNetworkNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    NnUint segmentLayerIndex = 0u;
    if (segmentConfig->nOps > 0u) {
        segmentLayerIndex = segmentConfig->ops[0].index;
    }

    for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
        NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
        NnByte *pipe = execution->pipes[syncConfig->pipeIndex];
        NnPipeConfig *pipeConfig = &netConfig->pipes[syncConfig->pipeIndex];
        NnSize batchBytes = getBytes(pipeConfig->size.floatType, pipeConfig->size.x);
        NnUint totalElements = pipeConfig->size.x; // [New] Get total elements

        // (Debug) Uncomment if you need per-sync pipe size info
        // if (threadIndex == 0) {
        //     printf("DEBUG sync entry: seg=%u sync=%u pipe=%u ft=%d x=%u batchBytes=%zu\n",
        //            segmentIndex, syncIndex, syncConfig->pipeIndex,
        //            pipeConfig->size.floatType, pipeConfig->size.x, (size_t)batchBytes);
        // }

        auto start = std::chrono::high_resolution_clock::now();
        const char* syncTypeStr = "UNKNOWN";

        for (NnUint batchIndex = 0; batchIndex < execution->batchSize; batchIndex++) {
            NnByte *pipeBatch = &pipe[batchIndex * batchBytes];

            if (syncConfig->syncType == SYNC_WITH_ROOT) {
                syncTypeStr = "SYNC_WITH_ROOT";
                syncWithRoot(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, nThreads, threadIndex, this->myStage);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES) {
                syncTypeStr = "SYNC_NODE_SLICES";
                syncNodeSlices(false, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, pipeConfig->size.floatType, nThreads, threadIndex, plan, this->myStage, this->layerSharding, segmentLayerIndex, totalElements);
            } else if (syncConfig->syncType == SYNC_KV_ALLGATHER) {
                syncTypeStr = "SYNC_KV_ALLGATHER";
                // Stage-local all-gather of KV cache slices.
                syncNodeSlices(false, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, pipeConfig->size.floatType, nThreads, threadIndex, plan, this->myStage, this->layerSharding, segmentLayerIndex, totalElements);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES_EXCEPT_ROOT) {
                syncTypeStr = "SYNC_LOGITS";
                syncNodeSlices(true, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, pipeConfig->size.floatType, nThreads, threadIndex, plan, nullptr, this->layerSharding, segmentLayerIndex, totalElements);
            } else if (syncConfig->syncType == SYNC_STAGE_SHARDING_UPDATE) {
                syncTypeStr = "SYNC_STAGE_SHARDING_UPDATE";

                const NnStageConfig *stage = this->myStage;
                const NnUint groupRootIndex = getGroupRootIndex(stage);
                const bool amIRoot = (nodeConfig->nodeIndex == groupRootIndex);

                // Convention: the graph should write the intended layerIndex into the first NnUint of this pipe.
                // Root will overwrite the buffer with a packed sharding update (or a no-op header).
                NnUint layerIndex = 0u;
                if (batchBytes >= sizeof(NnUint)) {
                    std::memcpy(&layerIndex, pipeBatch, sizeof(layerIndex));
                }

                const NnUint basePos = getBasePositionFromExecution(execution);

                if (amIRoot && threadIndex == 0) {
                    // 0) Optional: one-shot forced sharding override for validation without a scheduler.
                    // Set `DLLAMA_FORCE_SHARDING_ONCE=1` to enable.
                    // Optional:
                    //   - DLLAMA_FORCE_SHARDING_LAYER=<uint>   (target layer index; default=current atLayer)
                    //   - DLLAMA_FORCE_SHARDING_POS=<uint>     (gate by basePos; default=emit ASAP)
                    bool installedForcedNow = false;
                    if (!this->forcedShardingOnceArmed && !this->forcedShardingOnceDone) {
                        const char *forceOnce = std::getenv("DLLAMA_FORCE_SHARDING_ONCE");
                        if (forceOnce != nullptr && forceOnce[0] == '1') {
                            this->forcedShardingOnceArmed = true;

                            NnUint forceLayer = layerIndex;
                            bool forceHasPos = false;
                            NnUint forcePos = 0u;

                            const char *layerEnv = std::getenv("DLLAMA_FORCE_SHARDING_LAYER");
                            if (layerEnv == nullptr || layerEnv[0] == '\0')
                                layerEnv = std::getenv("DLLAMA_FORCE_SHARDING_TARGET_LAYER");
                            if (layerEnv != nullptr && layerEnv[0] != '\0') {
                                char *end = nullptr;
                                unsigned long v = std::strtoul(layerEnv, &end, 10);
                                if (end != layerEnv)
                                    forceLayer = (NnUint)v;
                            }

                            const char *posEnv = std::getenv("DLLAMA_FORCE_SHARDING_POS");
                            if (posEnv == nullptr || posEnv[0] == '\0')
                                posEnv = std::getenv("DLLAMA_FORCE_SHARDING_TARGET_POS");
                            if (posEnv != nullptr && posEnv[0] != '\0') {
                                char *end = nullptr;
                                unsigned long v = std::strtoul(posEnv, &end, 10);
                                if (end != posEnv) {
                                    forceHasPos = true;
                                    forcePos = (NnUint)v;
                                }
                            }

                            // Prepare an override for this stage's HEAD split only.
                            // Use current plan head lengths as a baseline, then apply a small deterministic tweak
                            // that preserves the total heads within this stage.
                            std::vector<NnUint> stageNodes;
                            std::vector<NnUint> headLens;
                            stageNodes.reserve(stage->nNodes);
                            headLens.reserve(stage->nNodes);

                            for (NnUint i = 0u; i < stage->nNodes; ++i) {
                                const NnUint nodeIndex = stage->nodeIndices[i];
                                stageNodes.push_back(nodeIndex);
                                const NnUint baseLen = (plan && plan->headSplit.lengths) ? plan->headSplit.lengths[nodeIndex] : 0u;
                                headLens.push_back(baseLen);
                            }

                            // For validation: nudge the HEAD split by 1 head while preserving the total.
                            // Default behavior: move 1 head from the first stage node to the last stage node.
                            // Example: 8:8 -> 7:9 for a 2-node stage.
                            if (stage->nNodes >= 2u) {
                                const size_t last = (size_t)stage->nNodes - 1u;
                                if (headLens[0] > 0u) {
                                    headLens[0] -= 1u;
                                    headLens[last] += 1u;
                                }
                            }

                            if (shardingUpdateDebugEnabled()) {
                                printf("[SHARD][NET][FORCE] armed=1 layer=%u hasPos=%u pos=%u stage=%u nodes=",
                                    forceLayer,
                                    forceHasPos ? 1u : 0u,
                                    forcePos,
                                    stage->stageIndex);
                                for (NnUint i = 0u; i < stageNodes.size(); ++i) {
                                    printf("%u:%u%s", stageNodes[i], headLens[i], (i + 1u == stageNodes.size()) ? "" : ",");
                                }
                                printf("\n");
                            }

                            // Install as a pending file-style override (no pos gate => emit ASAP).
                            this->pendingShardingUpdate = true;
                            this->pendingShardingUpdateHasPos = forceHasPos;
                            this->pendingShardingUpdateLayer = forceLayer;
                            this->pendingShardingUpdatePos = forcePos;
                            this->pendingShardingUpdateHasHeadLens = true;
                            this->pendingShardingUpdateStageNodes = stageNodes;
                            this->pendingShardingUpdateHeadLens = headLens;
                            this->pendingShardingUpdateLocked = true;
                            installedForcedNow = true;
                        }
                    }

                    // 1) Poll runtime request file (optional). This lets root trigger updates at any time.
                    // const char *reqPath = getShardingUpdateRequestFilePath();
                    // NnUint fileLayer = 0u;
                    // NnUint filePos = 0u;
                    // bool fileHasPos = false;
                    // bool fileHasHeadLens = false;
                    // std::vector<NnUint> fileStageNodes;
                    // std::vector<NnUint> fileHeadLens;
                    // const bool lockedPending = (this->pendingShardingUpdate && this->pendingShardingUpdateLocked);
                    // if (!installedForcedNow && !lockedPending) {
                    //     if (pollShardingUpdateFile(reqPath, &this->shardingUpdateLastLineHash, &fileLayer, &fileHasPos, &filePos,
                    //             &fileHasHeadLens, &fileStageNodes, &fileHeadLens)) {
                    //         this->pendingShardingUpdate = true;
                    //         this->pendingShardingUpdateHasPos = fileHasPos;
                    //         this->pendingShardingUpdateLayer = fileLayer;
                    //         this->pendingShardingUpdatePos = filePos;

                    //         this->pendingShardingUpdateHasHeadLens = fileHasHeadLens;
                    //         this->pendingShardingUpdateStageNodes = fileStageNodes;
                    //         this->pendingShardingUpdateHeadLens = fileHeadLens;
                    //         this->pendingShardingUpdateLocked = false;
                    //     }
                    // }

                    // 2) Decide whether to emit an update now.
                    bool emit = false;
                    NnUint targetLayerIndex = layerIndex;

                    // File-trigger has priority.
                    if (this->pendingShardingUpdate) {
                        if (!this->pendingShardingUpdateHasPos || this->pendingShardingUpdatePos == basePos) {
                            emit = true;
                            targetLayerIndex = this->pendingShardingUpdateLayer;
                        }
                    } else {
                        // Env-trigger fallback (one-shot per layer).
                        if (shouldEmitEnvShardingUpdate(layerIndex, basePos)) {
                            const bool already = (layerIndex < envLayerUpdated.size())
                                ? (envLayerUpdated[layerIndex] != (NnByte)0)
                                : false;
                            if (!already) {
                                emit = true;
                                targetLayerIndex = layerIndex;
                                if (layerIndex < envLayerUpdated.size()) {
                                    envLayerUpdated[layerIndex] = (NnByte)1;
                                }
                            }
                        }
                    }

                    NnUint emittedEpoch = 0u;
                    if (emit) {
                        const bool hasOverride = this->pendingShardingUpdate && this->pendingShardingUpdateHasHeadLens;
                        maybePrepareStageShardingUpdate(
                            plan,
                            stage,
                            targetLayerIndex,
                            hasOverride,
                            this->pendingShardingUpdateStageNodes,
                            this->pendingShardingUpdateHeadLens,
                            pipeBatch,
                            batchBytes,
                            &this->shardingEpoch,
                            &emittedEpoch);
                        // Consume file-trigger request once emitted.
                        if (this->pendingShardingUpdate) {
                            if (this->pendingShardingUpdateLocked) {
                                this->forcedShardingOnceDone = true;
                            }
                            this->pendingShardingUpdate = false;
                            this->pendingShardingUpdateHasHeadLens = false;
                            this->pendingShardingUpdateLocked = false;
                            this->pendingShardingUpdateStageNodes.clear();
                            this->pendingShardingUpdateHeadLens.clear();
                        }
                    } else {
                        writeNoShardingUpdate(pipeBatch, batchBytes);
                    }
                }

                localBarrier(nThreads, threadIndex);
                syncWithRoot(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, nThreads, threadIndex, stage);

#if DLLAMA_CONTROL_LOG
                // Log whether an online sharding update actually changed the split (epoch advanced).
                // To avoid log spam, print once per sync (batch 0) on thread 0.
                if (threadIndex == 0u && batchIndex == 0u && batchBytes >= sizeof(NnShardingUpdateHeader)) {
                    NnShardingUpdateHeader hdr;
                    std::memcpy(&hdr, pipeBatch, sizeof(hdr));

                    const bool isValidUpdate = (hdr.magic == NN_SHARDING_UPDATE_MAGIC && hdr.version == NN_SHARDING_UPDATE_VERSION);
                    const NnUint prevEpoch = this->lastSeenShardingEpoch;
                    const bool changed = isValidUpdate && (hdr.epoch > prevEpoch);
                    if (changed) {
                        this->lastSeenShardingEpoch = hdr.epoch;
                    }

                    const NnUint stageIndex = (stage != nullptr) ? stage->stageIndex : 0u;
                    const NnUint msgLayer = isValidUpdate ? hdr.layerIndex : layerIndex;
                    printf("🔧 [ShardingUpdate] node=%u stage=%u layer=%u pos=%u changed=%s epoch=%u prev=%u\n",
                        nodeConfig->nodeIndex,
                        stageIndex,
                        msgLayer,
                        basePos,
                        changed ? "YES" : "NO",
                        isValidUpdate ? hdr.epoch : 0u,
                        prevEpoch);
                }
#endif
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
            if (threadIndex == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
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
