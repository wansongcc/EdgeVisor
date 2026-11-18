#include "sustained_profile.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

#if defined(__linux__)
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <linux/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

const uint32_t kLinkMagic = 0x4C494E4Bu;
const uint32_t kStopMagic = 0x53544F50u;
const unsigned kMaxRounds = 10000u;
const unsigned kMaxSamples = 10000u;

struct SustainedWireJob {
    uint32_t observeMs;
    uint32_t warmupRounds;
    uint32_t primaryLayers;
    uint32_t dim;
    uint32_t hiddenDim;
    uint32_t nHeads;
    uint32_t nKvHeads;
    uint32_t activationBytes;
    uint32_t timeoutMs;
    uint32_t peerPort;
    uint32_t peerStage;
    uint32_t pad;
    double stepMs;
    char peerHost[256];
};

struct SustainedWireRound {
    double timestampMs;
    double elapsedMs;
    uint32_t layerCount;
    uint32_t pad;
};

struct SustainedWireTick {
    double timestampMs;
    int32_t temperatureC;
    int32_t thermalLimited;
    int32_t freqMhz;
    int32_t freqMaxMhz;
    int32_t onExternalPower;
    int32_t batteryPercent;
};

struct SustainedWireXfer {
    double timestampMs;
    double elapsedMs;
    uint32_t bytes;
    uint32_t timedOut;
    int32_t retransDelta;
    uint32_t pad;
};

struct SustainedWireHeader {
    uint32_t nRounds;
    uint32_t nTicks;
    uint32_t nTransfers;
    uint32_t segmentStatus; // 0 unknown, 1 counts readable (rate still needs a non-zero denominator)
    uint32_t retransSegs;
    uint32_t dataSegsOut;
    uint32_t rebuilds;
    uint32_t pad;
};

static_assert(sizeof(SustainedWireHeader) == 32u, "sustained header must have a stable layout");

static_assert(sizeof(SustainedWireJob) == 312u, "sustained job must have a stable layout");
static_assert(sizeof(SustainedWireRound) == 24u, "sustained round must have a stable layout");
static_assert(sizeof(SustainedWireTick) == 32u, "sustained tick must have a stable layout");
static_assert(sizeof(SustainedWireXfer) == 32u, "sustained transfer must have a stable layout");

double elapsedMsSince(const std::chrono::steady_clock::time_point &start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void sleepMs(double ms) {
    if (!(ms > 0.0)) return;
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms));
}

std::string hostLabel(const std::string &host) {
    return host.empty() ? std::string("local") : host;
}

std::string endpointAddress(const std::string &host, int port) {
    if (host.empty() || port <= 0) return std::string();
    return host + ":" + std::to_string(port);
}

void printDoubleOrNa(const char *name, double value, bool known) {
    if (!known) std::printf(" %s=na", name);
    else std::printf(" %s=%.4f", name, value);
}

#if defined(__linux__)

bool writeFull(int fd, const void *data, size_t n) {
    const char *bytes = static_cast<const char *>(data);
    size_t done = 0u;
    while (done < n) {
        const ssize_t wrote = ::send(fd, bytes + done, n - done, MSG_NOSIGNAL);
        if (wrote < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (wrote == 0) return false;
        done += static_cast<size_t>(wrote);
    }
    return true;
}

bool readFullUntil(int fd, void *data, size_t n, const std::chrono::steady_clock::time_point &deadline) {
    char *bytes = static_cast<char *>(data);
    size_t done = 0u;
    while (done < n) {
        const int remain = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remain <= 0) return false;
        pollfd wait;
        wait.fd = fd;
        wait.events = POLLIN;
        wait.revents = 0;
        const int polled = ::poll(&wait, 1, remain);
        if (polled < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (polled == 0) return false;
        const ssize_t got = ::recv(fd, bytes + done, n - done, 0);
        if (got < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (got == 0) return false;
        done += static_cast<size_t>(got);
    }
    return true;
}

bool readFull(int fd, void *data, size_t n) {
    return readFullUntil(fd, data, n, std::chrono::steady_clock::now() + std::chrono::hours(1));
}

void setTcpLowLatency(int fd) {
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef TCP_QUICKACK
    ::setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
#endif
}

struct TcpSnapshot {
    bool known;
    uint32_t totalRetrans;
    uint32_t dataSegsOut;
};

TcpSnapshot readTcpSnapshot(int fd) {
    TcpSnapshot snap;
    snap.known = false;
    snap.totalRetrans = 0u;
    snap.dataSegsOut = 0u;
    struct tcp_info info;
    std::memset(&info, 0, sizeof(info));
    socklen_t len = sizeof(info);
    if (::getsockopt(fd, IPPROTO_TCP, TCP_INFO, &info, &len) != 0) return snap;
    const socklen_t need = (socklen_t)(offsetof(struct tcp_info, tcpi_data_segs_out) + sizeof(info.tcpi_data_segs_out));
    if (len < need) return snap;
    snap.known = true;
    snap.totalRetrans = info.tcpi_total_retrans;
    snap.dataSegsOut = info.tcpi_data_segs_out;
    return snap;
}

void setRecvTimeout(int fd, unsigned timeoutMs) {
    timeval limit;
    limit.tv_sec = (time_t)(timeoutMs / 1000u);
    limit.tv_usec = (suseconds_t)((timeoutMs % 1000u) * 1000u);
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &limit, sizeof(limit));
}

int connectWithin(const char *host, int port, int timeoutMs) {
    if (host == nullptr || host[0] == '\0' || port <= 0) return -1;
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0) ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (::inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        ::close(fd);
        return -1;
    }
    const int rc = ::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
    if (rc != 0 && errno != EINPROGRESS) {
        ::close(fd);
        return -1;
    }
    pollfd wait;
    wait.fd = fd;
    wait.events = POLLOUT;
    wait.revents = 0;
    const int polled = ::poll(&wait, 1, timeoutMs);
    if (polled <= 0) {
        ::close(fd);
        return -1;
    }
    int soError = 0;
    socklen_t len = sizeof(soError);
    if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soError, &len) != 0 || soError != 0) {
        ::close(fd);
        return -1;
    }
    if (flags >= 0) ::fcntl(fd, F_SETFL, flags);
    setTcpLowLatency(fd);
    return fd;
}

bool sendActivation(
    int fd,
    const std::vector<char> &payload,
    unsigned timeoutMs,
    double *elapsedMs,
    int *retransDelta,
    TcpSnapshot *beforeOut,
    TcpSnapshot *afterOut) {
    const uint32_t nbytes = (uint32_t)payload.size();
    const TcpSnapshot before = readTcpSnapshot(fd);
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    bool writeOk = writeFull(fd, &nbytes, sizeof(nbytes));
    if (writeOk && !payload.empty()) writeOk = writeFull(fd, payload.data(), payload.size());
    uint32_t ack = 0u;
    bool got = false;
    if (writeOk) {
        got = readFullUntil(
            fd, &ack, sizeof(ack), start + std::chrono::milliseconds(timeoutMs == 0u ? 1u : timeoutMs));
    }
    *elapsedMs = elapsedMsSince(start);
    const TcpSnapshot after = readTcpSnapshot(fd);
    if (beforeOut != nullptr) *beforeOut = before;
    if (afterOut != nullptr) *afterOut = after;
    if (before.known && after.known && after.totalRetrans >= before.totalRetrans
        && after.totalRetrans - before.totalRetrans <= 2147483647u)
        *retransDelta = (int)(after.totalRetrans - before.totalRetrans);
    else
        *retransDelta = -1;
    return writeOk && got && ack == 1u;
}

void serveOneLink(int fd, const std::atomic<bool> &stop, const std::chrono::steady_clock::time_point *stopAt) {
    std::vector<char> payload;
    while (true) {
        if (stop.load()) {
            const std::chrono::steady_clock::time_point deadline = stopAt == nullptr
                ? std::chrono::steady_clock::now()
                : *stopAt + std::chrono::seconds(1);
            if (std::chrono::steady_clock::now() >= deadline) return;
        }
        pollfd wait;
        wait.fd = fd;
        wait.events = POLLIN;
        wait.revents = 0;
        const int polled = ::poll(&wait, 1, 200);
        if (polled < 0) {
            if (errno == EINTR) continue;
            return;
        }
        if (polled == 0) continue;
        uint32_t nbytes = 0u;
        if (!readFullUntil(fd, &nbytes, sizeof(nbytes), std::chrono::steady_clock::now() + std::chrono::seconds(2)))
            return;
        if (nbytes == 0u) {
            const uint32_t ack = 1u;
            writeFull(fd, &ack, sizeof(ack));
            return;
        }
        if (nbytes > (16u << 20)) return;
        payload.resize(nbytes);
        if (!readFullUntil(fd, payload.data(), payload.size(), std::chrono::steady_clock::now() + std::chrono::seconds(2)))
            return;
        const uint32_t ack = 1u;
        if (!writeFull(fd, &ack, sizeof(ack))) return;
    }
}

struct LocalObservation {
    std::vector<ComputeRound> rounds;
    std::vector<HardwareTick> ticks;
    std::vector<TransferRound> transfers;
    LinkCounterWindow linkCounters;
};

LocalObservation observeLocal(
    const ModelShape &shape,
    unsigned primaryLayers,
    unsigned warmupRounds,
    double observeSeconds,
    double stepMs,
    unsigned activationBytes,
    unsigned timeoutMs,
    const std::string &peerHost,
    int peerPort) {
    LocalObservation observation;
    RepresentativeLayerBench bench(shape);
    const unsigned layers = primaryLayers == 0u ? 0u : primaryLayers;
    const bool send = !peerHost.empty() && peerPort > 0 && activationBytes > 0u && activationBytes <= (16u << 20);
    std::vector<char> payload(send ? activationBytes : 0u, 0);
    int linkFd = -1;
    int accountedFd = -1;
    unsigned failedConnects = 0u;

    auto pace = [&](double elapsed) {
        const double budget = stepMs > 0.0 ? stepMs : elapsed * 2.0;
        sleepMs(budget - elapsed);
    };
    auto reconnect = [&]() {
        if (linkFd >= 0) {
            ::close(linkFd);
            linkFd = -1;
        }
        linkFd = connectWithin(peerHost.c_str(), peerPort, 3000);
        if (linkFd < 0) return false;
        const uint32_t magic = kLinkMagic;
        if (!writeFull(linkFd, &magic, sizeof(magic))) {
            ::close(linkFd);
            linkFd = -1;
            return false;
        }
        setRecvTimeout(linkFd, timeoutMs == 0u ? kSustainedDefaultTimeoutMs : timeoutMs);
        return true;
    };
    auto oneTransfer = [&](bool record, double timestampMs) {
        if (!send || failedConnects >= 3u) return;
        if (linkFd < 0 && !reconnect()) {
            failedConnects += 1u;
            if (!record) return;
            TransferRound round;
            round.timestampMs = timestampMs;
            round.elapsedMs = (double)timeoutMs;
            round.bytes = activationBytes;
            round.timedOut = 1;
            round.retransDelta = -1;
            observation.transfers.push_back(round);
            return;
        }
        double elapsed = 0.0;
        int retrans = -1;
        TcpSnapshot beforeSnap;
        TcpSnapshot afterSnap;
        beforeSnap.known = false;
        afterSnap.known = false;
        const bool ok = sendActivation(linkFd, payload, timeoutMs, &elapsed, &retrans, &beforeSnap, &afterSnap);
        if (record) {
            const bool newConnection = accountedFd != linkFd;
            noteCountedSend(&observation.linkCounters, newConnection,
                beforeSnap.known, beforeSnap.totalRetrans, beforeSnap.dataSegsOut,
                afterSnap.known, afterSnap.totalRetrans, afterSnap.dataSegsOut);
            accountedFd = linkFd;
        }
        if (!ok) {
            if (observation.linkCounters.open) observation.linkCounters.endConnection();
            accountedFd = -1;
            if (linkFd >= 0) ::close(linkFd);
            linkFd = -1;
            failedConnects += 1u;
        } else {
            failedConnects = 0u;
        }
        if (!record) return;
        TransferRound round;
        round.timestampMs = timestampMs;
        round.elapsedMs = elapsed;
        round.bytes = activationBytes;
        round.timedOut = ok ? 0 : 1;
        round.retransDelta = retrans;
        observation.transfers.push_back(round);
    };

    if (layers > 0u) {
        for (unsigned warm = 0u; warm < warmupRounds; ++warm) {
            const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            for (unsigned layer = 0u; layer < layers; ++layer) bench.runOnceMs();
            pace(elapsedMsSince(start));
        }
    }
        if (send) {
        for (unsigned warm = 0u; warm < warmupRounds; ++warm) {
            const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            oneTransfer(false, 0.0);
            pace(elapsedMsSince(start));
        }
        failedConnects = 0u;
    }

    const std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point windowEnd =
        windowStart + std::chrono::milliseconds((int)(observeSeconds * 1000.0));
    std::atomic<bool> sampling(true);
    std::thread sampler([&]() {
        while (sampling.load()) {
            HardwareTick tick = readHardwareTick();
            tick.timestampMs = elapsedMsSince(windowStart);
            observation.ticks.push_back(tick);
            if (observation.ticks.size() >= kMaxSamples) break;
            const std::chrono::steady_clock::time_point next =
                std::chrono::steady_clock::now() + std::chrono::seconds(1);
            if (next >= windowEnd) break;
            std::this_thread::sleep_until(next);
        }
    });
    std::thread sender([&]() {
        if (!send) return;
        while (std::chrono::steady_clock::now() < windowEnd && observation.transfers.size() < kMaxRounds) {
            const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            oneTransfer(true, elapsedMsSince(windowStart));
            if (failedConnects >= 3u) break;
            pace(elapsedMsSince(start));
        }
        if (observation.linkCounters.open) observation.linkCounters.endConnection();
        if (linkFd >= 0) {
            const uint32_t end = 0u;
            writeFull(linkFd, &end, sizeof(end));
            ::close(linkFd);
            linkFd = -1;
        }
        accountedFd = -1;
    });

    if (layers > 0u) {
        while (std::chrono::steady_clock::now() < windowEnd && observation.rounds.size() < kMaxRounds) {
            const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            bool failed = false;
            for (unsigned layer = 0u; layer < layers; ++layer) {
                if (bench.runOnceMs() < 0.0) failed = true;
            }
            const double elapsed = elapsedMsSince(start);
            if (!failed) {
                ComputeRound round;
                round.timestampMs = elapsedMsSince(windowStart);
                round.elapsedMs = elapsed;
                round.layerCount = layers;
                observation.rounds.push_back(round);
            }
            if (std::chrono::steady_clock::now() >= windowEnd) break;
            pace(elapsed);
        }
    }

    sampling.store(false);
    if (sampler.joinable()) sampler.join();
    if (sender.joinable()) sender.join();
    if (linkFd >= 0) ::close(linkFd);
    return observation;
}

struct RemoteCapture {
    bool ok = false;
    std::vector<ComputeRound> rounds;
    std::vector<HardwareTick> ticks;
    std::vector<TransferRound> transfers;
    LinkCounterWindow linkCounters;
};

bool readRemoteResult(int fd, RemoteCapture *capture, const std::chrono::steady_clock::time_point &deadline) {
    SustainedWireHeader header;
    std::memset(&header, 0, sizeof(header));
    if (!readFullUntil(fd, &header, sizeof(header), deadline)) return false;
    if (header.nRounds > kMaxRounds || header.nTicks > kMaxSamples || header.nTransfers > kMaxRounds)
        return false;
    for (uint32_t i = 0u; i < header.nRounds; ++i) {
        SustainedWireRound wire;
        std::memset(&wire, 0, sizeof(wire));
        if (!readFullUntil(fd, &wire, sizeof(wire), deadline)) return false;
        ComputeRound round;
        round.timestampMs = wire.timestampMs;
        round.elapsedMs = wire.elapsedMs;
        round.layerCount = wire.layerCount;
        capture->rounds.push_back(round);
    }
    for (uint32_t i = 0u; i < header.nTicks; ++i) {
        SustainedWireTick wire;
        std::memset(&wire, 0, sizeof(wire));
        if (!readFullUntil(fd, &wire, sizeof(wire), deadline)) return false;
        HardwareTick tick;
        tick.timestampMs = wire.timestampMs;
        tick.temperatureC = wire.temperatureC;
        tick.thermalLimited = wire.thermalLimited;
        tick.freqMhz = wire.freqMhz;
        tick.freqMaxMhz = wire.freqMaxMhz;
        tick.onExternalPower = wire.onExternalPower;
        tick.batteryPercent = wire.batteryPercent;
        capture->ticks.push_back(tick);
    }
    for (uint32_t i = 0u; i < header.nTransfers; ++i) {
        SustainedWireXfer wire;
        std::memset(&wire, 0, sizeof(wire));
        if (!readFullUntil(fd, &wire, sizeof(wire), deadline)) return false;
        TransferRound round;
        round.timestampMs = wire.timestampMs;
        round.elapsedMs = wire.elapsedMs;
        round.bytes = wire.bytes;
        round.timedOut = (int)wire.timedOut;
        round.retransDelta = wire.retransDelta;
        capture->transfers.push_back(round);
    }
    if (header.segmentStatus == 1u) {
        capture->linkCounters.broken = false;
        capture->linkCounters.committed = 1u;
        capture->linkCounters.retransSegs = header.retransSegs;
        capture->linkCounters.dataSegsOut = header.dataSegsOut;
        capture->linkCounters.rebuilds = header.rebuilds;
    } else {
        capture->linkCounters.broken = header.rebuilds > 0u || header.nTransfers > 0u;
        capture->linkCounters.committed = 0u;
        capture->linkCounters.rebuilds = header.rebuilds;
    }
    return true;
}

void writeRemoteResult(int fd, const LocalObservation &observation) {
    SustainedWireHeader header;
    std::memset(&header, 0, sizeof(header));
    header.nRounds = (uint32_t)std::min(observation.rounds.size(), (size_t)kMaxRounds);
    header.nTicks = (uint32_t)std::min(observation.ticks.size(), (size_t)kMaxSamples);
    header.nTransfers = (uint32_t)std::min(observation.transfers.size(), (size_t)kMaxRounds);
    const bool countsKnown = !observation.linkCounters.broken && observation.linkCounters.committed > 0u
        && observation.linkCounters.retransSegs <= 0xffffffffull
        && observation.linkCounters.dataSegsOut <= 0xffffffffull;
    header.segmentStatus = countsKnown ? 1u : 0u;
    header.retransSegs = countsKnown ? (uint32_t)observation.linkCounters.retransSegs : 0u;
    header.dataSegsOut = countsKnown ? (uint32_t)observation.linkCounters.dataSegsOut : 0u;
    header.rebuilds = observation.linkCounters.rebuilds;
    if (!writeFull(fd, &header, sizeof(header))) return;
    for (uint32_t i = 0u; i < header.nRounds; ++i) {
        SustainedWireRound wire;
        std::memset(&wire, 0, sizeof(wire));
        wire.timestampMs = observation.rounds[i].timestampMs;
        wire.elapsedMs = observation.rounds[i].elapsedMs;
        wire.layerCount = observation.rounds[i].layerCount;
        if (!writeFull(fd, &wire, sizeof(wire))) return;
    }
    for (uint32_t i = 0u; i < header.nTicks; ++i) {
        SustainedWireTick wire;
        std::memset(&wire, 0, sizeof(wire));
        wire.timestampMs = observation.ticks[i].timestampMs;
        wire.temperatureC = observation.ticks[i].temperatureC;
        wire.thermalLimited = observation.ticks[i].thermalLimited;
        wire.freqMhz = observation.ticks[i].freqMhz;
        wire.freqMaxMhz = observation.ticks[i].freqMaxMhz;
        wire.onExternalPower = observation.ticks[i].onExternalPower;
        wire.batteryPercent = observation.ticks[i].batteryPercent;
        if (!writeFull(fd, &wire, sizeof(wire))) return;
    }
    for (uint32_t i = 0u; i < header.nTransfers; ++i) {
        SustainedWireXfer wire;
        std::memset(&wire, 0, sizeof(wire));
        wire.timestampMs = observation.transfers[i].timestampMs;
        wire.elapsedMs = observation.transfers[i].elapsedMs;
        wire.bytes = observation.transfers[i].bytes;
        wire.timedOut = (uint32_t)observation.transfers[i].timedOut;
        wire.retransDelta = observation.transfers[i].retransDelta;
        if (!writeFull(fd, &wire, sizeof(wire))) return;
    }
}

void acceptLinks(int listenFd, std::atomic<bool> *stop, std::chrono::steady_clock::time_point *stopAt) {
    while (!stop->load()) {
        pollfd wait;
        wait.fd = listenFd;
        wait.events = POLLIN;
        wait.revents = 0;
        const int polled = ::poll(&wait, 1, 200);
        if (polled < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (polled == 0 || stop->load()) continue;
        const int fd = ::accept(listenFd, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR) continue;
            continue;
        }
        setTcpLowLatency(fd);
        uint32_t magic = 0u;
        if (!readFullUntil(fd, &magic, sizeof(magic), std::chrono::steady_clock::now() + std::chrono::seconds(3))) {
            ::close(fd);
            continue;
        }
        if (magic == kStopMagic || stop->load()) {
            ::close(fd);
            break;
        }
        if (magic == kLinkMagic) serveOneLink(fd, *stop, stopAt);
        ::close(fd);
        if (stop->load()) break;
    }
}

void drainListen(int listenFd) {
    const int flags = ::fcntl(listenFd, F_GETFL, 0);
    if (flags >= 0) ::fcntl(listenFd, F_SETFL, flags | O_NONBLOCK);
    while (true) {
        const int fd = ::accept(listenFd, nullptr, nullptr);
        if (fd < 0) break;
        ::close(fd);
    }
    if (flags >= 0) ::fcntl(listenFd, F_SETFL, flags);
}

SustainedWireJob makeJob(
    const SustainedRunConfig &config,
    const SustainedEndpoint &endpoint,
    const std::string &peerHost,
    int peerPort,
    unsigned peerStage) {
    SustainedWireJob job;
    std::memset(&job, 0, sizeof(job));
    const double seconds = config.observeSeconds > 0.0 ? config.observeSeconds : 25.0;
    unsigned observeMs = (unsigned)(seconds * 1000.0);
    if (observeMs < 1000u) observeMs = 1000u;
    if (observeMs > 120000u) observeMs = 120000u;
    job.observeMs = observeMs;
    job.warmupRounds = std::min(config.warmupRounds, 32u);
    job.primaryLayers = std::min(endpoint.primaryLayers, 4096u);
    job.dim = config.shape.dim;
    job.hiddenDim = config.shape.hiddenDim;
    job.nHeads = config.shape.nHeads;
    job.nKvHeads = config.shape.nKvHeads;
    job.activationBytes = config.activationBytes <= (16u << 20) ? config.activationBytes : 0u;
    job.timeoutMs = config.timeoutMs == 0u ? kSustainedDefaultTimeoutMs : std::min(config.timeoutMs, 10000u);
    job.peerPort = (uint32_t)std::max(peerPort, 0);
    job.peerStage = peerStage;
    job.stepMs = config.stepMs > 0.0 ? config.stepMs : 0.0;
    if (!peerHost.empty() && peerPort > 0) {
        std::snprintf(job.peerHost, sizeof(job.peerHost), "%s", peerHost.c_str());
    }
    return job;
}

void fillLinkMissing(LinkSummary *link) {
    if (link == nullptr) return;
    if (link->transfers == 0u) link->missing.push_back("transfers");
    if (!link->meanKnown) link->missing.push_back("meanMs");
    if (!link->jitterKnown) link->missing.push_back("jitterMs");
    if (!link->timeoutKnown) link->missing.push_back("timeoutFraction");
    if (link->retransSegs < 0) link->missing.push_back("retransSegs");
    if (link->dataSegsOut < 0) link->missing.push_back("dataSegsOut");
    if (!link->segmentRateKnown) link->missing.push_back("segmentRetransRate");
}

bool writeTextFile(const std::string &path, const std::string &text) {
    std::ofstream file(path.c_str(), std::ios::out | std::ios::trunc);
    if (!file.is_open()) return false;
    file << text;
    return file.good();
}

#endif

}  // namespace

SustainedRunOutput runSustainedShadowProfile(const SustainedRunConfig &config) {
    SustainedRunOutput output;
    const double observeSeconds = config.observeSeconds > 0.0 ? config.observeSeconds : 25.0;
    output.report.observeSeconds = observeSeconds > 120.0 ? 120.0 : observeSeconds;
    output.report.warmupRounds = std::min(config.warmupRounds, 32u);
    output.report.pace = config.stepMs > 0.0 ? "pipeline" : "half-duty";
    output.report.reversed = config.reversed;
    output.report.note =
        "Short stability window recorded before redundancy allocation. "
        "A device score of 0 means read indicators stayed at or below their normal point; missing fields did not vote. "
        "credibility is metadata and is not a fused weight. "
        "Jitter uses successful transfers only. Timeout fraction uses timed-out attempts over all counted attempts. "
        "segmentRetransRate is the sender tcpi_total_retrans delta divided by the tcpi_data_segs_out delta on the same connections, warmup excluded. "
        "affectedRoundRate is the share of readable attempts whose retrans delta was positive, and it is not the segment rate. "
        "A slower hop finishes fewer paced attempts before the wall-clock window ends. "
        "The window can show early degradation and does not show whether a longer run would thermally throttle.";
    output.devicePriority.assign(config.endpoints.size(), kProtectionWeightUnknown);
    if (config.endpoints.size() >= 2u)
        output.linkPriority.assign(config.endpoints.size() - 1u, 0u);
    output.outputDir = config.outputDir.empty() ? std::string("shadow_profile") : config.outputDir;

#if !defined(__linux__)
    std::fprintf(stderr, "shadow-profile: sustained profiling is implemented for Linux only, using mid weights\n");
    for (std::size_t i = 0u; i < config.endpoints.size(); ++i) {
        DeviceSummary device = summarizeDevice((unsigned)i, config.endpoints[i].host, std::vector<ComputeRound>(), std::vector<HardwareTick>());
        device.missing.push_back("probe");
        output.report.devices.push_back(device);
    }
    return output;
#else
    std::printf("shadow-profile: observe=%.1fs warmup=%u pace=%s step=%.2fms activation=%u timeout=%u\n",
        output.report.observeSeconds, output.report.warmupRounds, output.report.pace.c_str(),
        config.stepMs > 0.0 ? config.stepMs : 0.0,
        config.activationBytes, config.timeoutMs == 0u ? kSustainedDefaultTimeoutMs : config.timeoutMs);
    std::fflush(stdout);

    const std::size_t n = config.endpoints.size();
    std::vector<RemoteCapture> remote(n);
    std::mutex gateMu;
    std::condition_variable gateCv;
    unsigned readyCount = 0u;
    bool start = false;
    std::vector<std::thread> threads;
    for (std::size_t i = 0u; i < n; ++i) {
        if (config.endpoints[i].host.empty()) continue;
        threads.push_back(std::thread([&, i]() {
            const SustainedEndpoint &endpoint = config.endpoints[i];
            std::string peerHost;
            int peerPort = 0;
            unsigned peerStage = 0u;
            if (i + 1u < n) {
                peerHost = config.endpoints[i + 1u].host;
                peerPort = config.endpoints[i + 1u].port;
                peerStage = config.endpoints[i + 1u].stage;
            }
            const int fd = connectWithin(endpoint.host.c_str(), endpoint.port, 3000);
            if (fd < 0) {
                std::lock_guard<std::mutex> lock(gateMu);
                readyCount += 1u;
                gateCv.notify_all();
                return;
            }
            const uint32_t magic = kSustainedProfileMagic;
            const SustainedWireJob job = makeJob(config, endpoint, peerHost, peerPort, peerStage);
            bool armed = writeFull(fd, &magic, sizeof(magic)) && writeFull(fd, &job, sizeof(job));
            uint32_t ready = 0u;
            if (armed) {
                armed = readFullUntil(fd, &ready, sizeof(ready),
                    std::chrono::steady_clock::now() + std::chrono::seconds(8)) && ready == 1u;
            }
            {
                std::unique_lock<std::mutex> lock(gateMu);
                readyCount += 1u;
                gateCv.notify_all();
                gateCv.wait_for(lock, std::chrono::seconds(15), [&]() { return start; });
            }
            if (!armed) {
                ::close(fd);
                return;
            }
            const uint32_t go = 1u;
            if (!writeFull(fd, &go, sizeof(go))) {
                ::close(fd);
                return;
            }
            const std::chrono::steady_clock::time_point deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds((int)(output.report.observeSeconds * 1000.0) + 20000);
            remote[i].ok = readRemoteResult(fd, &remote[i], deadline);
            ::close(fd);
        }));
    }

    {
        std::unique_lock<std::mutex> lock(gateMu);
        gateCv.wait_for(lock, std::chrono::seconds(20), [&]() { return readyCount >= threads.size(); });
        start = true;
        gateCv.notify_all();
    }

    LocalObservation local;
    int localIndex = -1;
    for (std::size_t i = 0u; i < n; ++i) {
        if (!config.endpoints[i].host.empty()) continue;
        localIndex = (int)i;
        std::string peerHost;
        int peerPort = 0;
        if (i + 1u < n) {
            peerHost = config.endpoints[i + 1u].host;
            peerPort = config.endpoints[i + 1u].port;
        }
        local = observeLocal(
            config.shape, config.endpoints[i].primaryLayers, output.report.warmupRounds,
            output.report.observeSeconds, config.stepMs, config.activationBytes,
            config.timeoutMs == 0u ? kSustainedDefaultTimeoutMs : config.timeoutMs,
            peerHost, peerPort);
        break;
    }
    for (std::size_t i = 0u; i < threads.size(); ++i) {
        if (threads[i].joinable()) threads[i].join();
    }

    std::vector<ComputeRawRow> computeRows;
    std::vector<HardwareRawRow> hardwareRows;
    std::vector<LinkRawRow> linkRows;
    for (std::size_t i = 0u; i < n; ++i) {
        const bool isLocal = (int)i == localIndex;
        const std::vector<ComputeRound> &rounds = isLocal ? local.rounds : remote[i].rounds;
        const std::vector<HardwareTick> &ticks = isLocal ? local.ticks : remote[i].ticks;
        const std::vector<TransferRound> &transfers = isLocal ? local.transfers : remote[i].transfers;
        const bool answered = isLocal || remote[i].ok;
        DeviceSummary device = summarizeDevice(config.endpoints[i].stage, config.endpoints[i].host, rounds, ticks);
        if (!answered) device.missing.push_back("probe");
        if (config.endpoints[i].extraNodes > 0u) device.missing.push_back("otherNodesOnStage");
        output.devicePriority[i] = device.appliedWeight;
        output.report.devices.push_back(device);
        for (std::size_t r = 0u; r < rounds.size(); ++r) {
            ComputeRawRow row;
            row.timestampMs = rounds[r].timestampMs;
            row.stage = config.endpoints[i].stage;
            row.round = (unsigned)r;
            row.layerCount = rounds[r].layerCount;
            row.elapsedMs = rounds[r].elapsedMs;
            computeRows.push_back(row);
        }
        for (std::size_t t = 0u; t < ticks.size(); ++t) {
            HardwareRawRow row;
            row.stage = config.endpoints[i].stage;
            row.tick = ticks[t];
            hardwareRows.push_back(row);
        }
        if (i + 1u >= n) continue;
        LinkSummary link = summarizeTransfers(transfers);
        applyLinkCounterWindow(&link, isLocal ? local.linkCounters : remote[i].linkCounters);
        link.fromStage = config.endpoints[i].stage;
        link.toStage = config.endpoints[i + 1u].stage;
        link.fromHost = hostLabel(config.endpoints[i].host);
        link.toHost = hostLabel(config.endpoints[i + 1u].host);
        link.address = endpointAddress(config.endpoints[i + 1u].host, config.endpoints[i + 1u].port);
        fillLinkMissing(&link);
        output.linkPriority[i] = link.appliedWeight;
        output.report.links.push_back(link);
        for (std::size_t r = 0u; r < transfers.size(); ++r) {
            LinkRawRow row;
            row.fromStage = link.fromStage;
            row.toStage = link.toStage;
            row.address = link.address;
            row.transfer = transfers[r];
            linkRows.push_back(row);
        }
        std::printf("shadow-link: %u->%u address=%s bytes=%u transfers=%u",
            link.fromStage, link.toStage, link.address.c_str(), link.bytes, link.transfers);
        printDoubleOrNa("mean", link.meanMs, link.meanKnown);
        printDoubleOrNa("jitter", link.jitterMs, link.jitterKnown);
        printDoubleOrNa("timeout", link.timeoutFraction, link.timeoutKnown);
        if (link.retransSegs < 0) std::printf(" retrans_segs=na");
        else std::printf(" retrans_segs=%lld", link.retransSegs);
        if (link.dataSegsOut < 0) std::printf(" data_segs_out=na");
        else std::printf(" data_segs_out=%lld", link.dataSegsOut);
        printDoubleOrNa("segment_retrans_rate", link.segmentRetransRate, link.segmentRateKnown);
        if (!link.affectedRoundRateKnown) std::printf(" affected_rounds=na");
        else std::printf(" affected_rounds=%u/%u", link.retransRounds, link.retransRoundsKnown);
        std::printf(" timeouts=%u/%u finished=%u rebuilds=%u",
            link.timeouts, link.transfers, link.finishedTransfers, link.rebuilds);
        printDoubleOrNa("t0", link.firstTransferMs, link.firstTransferMs >= 0.0);
        printDoubleOrNa("t1", link.lastTransferMs, link.lastTransferMs >= 0.0);
        if (link.risk < 0.0) std::printf(" risk=na");
        else std::printf(" risk=%.2f", link.risk);
        std::printf(" weight=%u priority=%s\n", link.appliedWeight, protectionPriorityName(link.priority));
    }
    for (std::size_t i = 0u; i < output.report.devices.size(); ++i) {
        const DeviceSummary &device = output.report.devices[i];
        std::printf("shadow-device: stage=%u host=%s rounds=%u layers=%u",
            device.stage, device.host.c_str(), device.compute.rounds, device.compute.layersPerRound);
        printDoubleOrNa("mean", device.compute.meanLayerMs, device.compute.meanKnown);
        printDoubleOrNa("jitter", device.compute.jitterLayerMs, device.compute.jitterKnown);
        printDoubleOrNa("late_early", device.compute.lateEarlyRatio, device.compute.ratioKnown);
        if (device.thermal.maxTempC < 0) std::printf(" temp_max=na");
        else std::printf(" temp_max=%d", device.thermal.maxTempC);
        printDoubleOrNa("rise", device.thermal.riseCPerSec, device.thermal.riseKnown);
        printDoubleOrNa("throttle", device.thermal.throttleFraction, device.thermal.throttleKnown);
        std::printf(" power=%s direction=%s battery=%d",
            powerStateName(device.power.onExternalPower),
            powerDirectionName(device.power.direction), device.power.batteryPercent);
        if (device.computeRisk < 0.0) std::printf(" compute_risk=na");
        else std::printf(" compute_risk=%.2f", device.computeRisk);
        if (device.powerRisk < 0.0) std::printf(" power_risk=na");
        else std::printf(" power_risk=%.2f", device.powerRisk);
        if (!device.scoreKnown) std::printf(" score=na");
        else std::printf(" score=%.2f", device.protectionScore);
        std::printf(" q=%.2f weight=%u priority=%s reasons=",
            device.credibility, device.appliedWeight, protectionPriorityName(device.appliedPriority));
        if (device.reasons.empty()) std::printf("-");
        for (std::size_t r = 0u; r < device.reasons.size(); ++r) {
            if (r > 0u) std::printf("|");
            std::printf("%s", device.reasons[r].c_str());
        }
        std::printf(" missing=");
        if (device.missing.empty()) std::printf("-");
        for (std::size_t r = 0u; r < device.missing.size(); ++r) {
            if (r > 0u) std::printf("|");
            std::printf("%s", device.missing[r].c_str());
        }
        std::printf("\n");
    }

    if (::mkdir(output.outputDir.c_str(), 0755) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "shadow-profile: cannot create %s\n", output.outputDir.c_str());
    } else {
        const bool computeOk = writeTextFile(output.outputDir + "/sustained_compute.csv", formatComputeCsv(computeRows));
        const bool hardwareOk = writeTextFile(output.outputDir + "/sustained_hardware.csv", formatHardwareCsv(hardwareRows));
        const bool linkOk = writeTextFile(output.outputDir + "/sustained_links.csv", formatLinkCsv(linkRows));
        const bool jsonOk = writeTextFile(output.outputDir + "/sustained_summary.json", formatSustainedSummaryJson(output.report));
        output.wroteFiles = computeOk && hardwareOk && linkOk && jsonOk;
        if (output.wroteFiles) {
            std::printf("shadow-profile: wrote %s\n", output.outputDir.c_str());
        } else {
            std::fprintf(stderr, "shadow-profile: failed to write %s\n", output.outputDir.c_str());
        }
    }
    std::fflush(stdout);
    return output;
#endif
}

void serveSustainedProfileProbe(int controlFd, int listenFd) {
#if !defined(__linux__)
    (void)controlFd;
    (void)listenFd;
#else
    try {
        SustainedWireJob job;
        std::memset(&job, 0, sizeof(job));
        if (!readFullUntil(controlFd, &job, sizeof(job), std::chrono::steady_clock::now() + std::chrono::seconds(8)))
            return;
        if (job.observeMs < 1000u) job.observeMs = 1000u;
        if (job.observeMs > 120000u) job.observeMs = 120000u;
        if (job.warmupRounds > 32u) job.warmupRounds = 32u;
        if (job.primaryLayers > 4096u) job.primaryLayers = 4096u;
        if (job.timeoutMs == 0u) job.timeoutMs = kSustainedDefaultTimeoutMs;
        if (job.timeoutMs > 10000u) job.timeoutMs = 10000u;
        if (job.activationBytes > (16u << 20)) job.activationBytes = 0u;
        job.peerHost[sizeof(job.peerHost) - 1u] = '\0';

        std::atomic<bool> stop(false);
        std::chrono::steady_clock::time_point stopAt = std::chrono::steady_clock::now();
        struct AcceptJoin {
            std::thread thread;
            std::atomic<bool> *stopFlag;
            std::chrono::steady_clock::time_point *stopAt;
            int listenFd;
            AcceptJoin() : stopFlag(nullptr), stopAt(nullptr), listenFd(-1) {}
            ~AcceptJoin() {
                if (stopFlag != nullptr && stopAt != nullptr) {
                    *stopAt = std::chrono::steady_clock::now();
                    stopFlag->store(true);
                }
                if (thread.joinable()) thread.join();
                if (listenFd >= 0) drainListen(listenFd);
            }
        } accepter;
        accepter.stopFlag = &stop;
        accepter.stopAt = &stopAt;
        accepter.listenFd = listenFd;
        if (listenFd >= 0) accepter.thread = std::thread(acceptLinks, listenFd, &stop, &stopAt);
        const uint32_t ready = 1u;
        if (!writeFull(controlFd, &ready, sizeof(ready))) return;
        uint32_t go = 0u;
        if (!readFullUntil(controlFd, &go, sizeof(go), std::chrono::steady_clock::now() + std::chrono::seconds(20))
            || go != 1u) {
            return;
        }
        ModelShape shape;
        shape.dim = job.dim;
        shape.hiddenDim = job.hiddenDim;
        shape.nHeads = job.nHeads;
        shape.nKvHeads = job.nKvHeads;
        shape.maxSeqLen = 0u;
        shape.nLayers = 0u;
        shape.weightType = 0;
        const std::string peerHost = job.peerPort == 0u ? std::string() : std::string(job.peerHost);
        LocalObservation observation = observeLocal(
            shape, job.primaryLayers, job.warmupRounds, (double)job.observeMs / 1000.0,
            job.stepMs, job.activationBytes, job.timeoutMs, peerHost, (int)job.peerPort);
        stopAt = std::chrono::steady_clock::now();
        stop.store(true);
        if (accepter.thread.joinable()) accepter.thread.join();
        writeRemoteResult(controlFd, observation);
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "shadow-profile: worker probe failed: %s\n", ex.what());
    } catch (...) {
        std::fprintf(stderr, "shadow-profile: worker probe failed\n");
    }
#endif
}
