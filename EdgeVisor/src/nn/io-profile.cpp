#include "nn/io-profile.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

struct IoProbeCounters {
    std::atomic<std::uint64_t> netWriteWallUs{0};
    std::atomic<std::uint64_t> netReadWallUs{0};
    std::atomic<std::uint64_t> netWriteOps{0};
    std::atomic<std::uint64_t> netReadOps{0};
    std::atomic<std::uint64_t> netSendSyscallUs{0};
    std::atomic<std::uint64_t> netRecvSyscallUs{0};
    std::atomic<std::uint64_t> netSendSyscalls{0};
    std::atomic<std::uint64_t> netRecvSyscalls{0};
    std::atomic<std::uint64_t> netSendBytes{0};
    std::atomic<std::uint64_t> netRecvBytes{0};
    std::atomic<std::uint64_t> netSendEagain{0};
    std::atomic<std::uint64_t> netRecvEagain{0};

    std::atomic<std::uint64_t> hostMemcpyH2DUs{0};
    std::atomic<std::uint64_t> hostMemcpyD2HUs{0};
    std::atomic<std::uint64_t> hostMemcpyH2DBytes{0};
    std::atomic<std::uint64_t> hostMemcpyD2HBytes{0};
    std::atomic<std::uint64_t> hostMemcpyH2DCalls{0};
    std::atomic<std::uint64_t> hostMemcpyD2HCalls{0};

    std::atomic<std::uint64_t> vulkanFlushUs{0};
    std::atomic<std::uint64_t> vulkanFlushBytes{0};
    std::atomic<std::uint64_t> vulkanFlushCalls{0};
    std::atomic<std::uint64_t> vulkanInvalidateUs{0};
    std::atomic<std::uint64_t> vulkanInvalidateBytes{0};
    std::atomic<std::uint64_t> vulkanInvalidateCalls{0};
    std::atomic<std::uint64_t> vulkanCopyH2DUs{0};
    std::atomic<std::uint64_t> vulkanCopyD2HUs{0};
    std::atomic<std::uint64_t> vulkanCopyH2DBytes{0};
    std::atomic<std::uint64_t> vulkanCopyD2HBytes{0};
    std::atomic<std::uint64_t> vulkanCopyH2DCalls{0};
    std::atomic<std::uint64_t> vulkanCopyD2HCalls{0};

    std::atomic<std::uint64_t> cudaMemcpyH2DUs{0};
    std::atomic<std::uint64_t> cudaMemcpyD2HUs{0};
    std::atomic<std::uint64_t> cudaMemcpyH2DBytes{0};
    std::atomic<std::uint64_t> cudaMemcpyD2HBytes{0};
    std::atomic<std::uint64_t> cudaMemcpyH2DCalls{0};
    std::atomic<std::uint64_t> cudaMemcpyD2HCalls{0};
    std::atomic<std::uint64_t> cudaSyncH2DUs{0};
    std::atomic<std::uint64_t> cudaSyncD2HUs{0};
    std::atomic<std::uint64_t> cudaSyncH2DCalls{0};
    std::atomic<std::uint64_t> cudaSyncD2HCalls{0};
};

struct IoProbeSnapshot {
    std::uint64_t netWriteWallUs = 0;
    std::uint64_t netReadWallUs = 0;
    std::uint64_t netWriteOps = 0;
    std::uint64_t netReadOps = 0;
    std::uint64_t netSendSyscallUs = 0;
    std::uint64_t netRecvSyscallUs = 0;
    std::uint64_t netSendSyscalls = 0;
    std::uint64_t netRecvSyscalls = 0;
    std::uint64_t netSendBytes = 0;
    std::uint64_t netRecvBytes = 0;
    std::uint64_t netSendEagain = 0;
    std::uint64_t netRecvEagain = 0;

    std::uint64_t hostMemcpyH2DUs = 0;
    std::uint64_t hostMemcpyD2HUs = 0;
    std::uint64_t hostMemcpyH2DBytes = 0;
    std::uint64_t hostMemcpyD2HBytes = 0;
    std::uint64_t hostMemcpyH2DCalls = 0;
    std::uint64_t hostMemcpyD2HCalls = 0;

    std::uint64_t vulkanFlushUs = 0;
    std::uint64_t vulkanFlushBytes = 0;
    std::uint64_t vulkanFlushCalls = 0;
    std::uint64_t vulkanInvalidateUs = 0;
    std::uint64_t vulkanInvalidateBytes = 0;
    std::uint64_t vulkanInvalidateCalls = 0;
    std::uint64_t vulkanCopyH2DUs = 0;
    std::uint64_t vulkanCopyD2HUs = 0;
    std::uint64_t vulkanCopyH2DBytes = 0;
    std::uint64_t vulkanCopyD2HBytes = 0;
    std::uint64_t vulkanCopyH2DCalls = 0;
    std::uint64_t vulkanCopyD2HCalls = 0;

    std::uint64_t cudaMemcpyH2DUs = 0;
    std::uint64_t cudaMemcpyD2HUs = 0;
    std::uint64_t cudaMemcpyH2DBytes = 0;
    std::uint64_t cudaMemcpyD2HBytes = 0;
    std::uint64_t cudaMemcpyH2DCalls = 0;
    std::uint64_t cudaMemcpyD2HCalls = 0;
    std::uint64_t cudaSyncH2DUs = 0;
    std::uint64_t cudaSyncD2HUs = 0;
    std::uint64_t cudaSyncH2DCalls = 0;
    std::uint64_t cudaSyncD2HCalls = 0;
};

struct IoProbeState {
    std::atomic<bool> configured{false};
    std::atomic<bool> enabled{false};
    std::atomic<bool> atexitRegistered{false};
    std::atomic<unsigned int> nodeIndex{0};
    std::atomic<unsigned int> stageIndex{0};
    std::atomic<unsigned long long> flushSeq{0};
    std::mutex pathMutex;
    std::string logPath;
    IoProbeCounters counters;
};

IoProbeState &state() {
    static IoProbeState s;
    return s;
}

std::uint64_t loadAndReset(std::atomic<std::uint64_t> &v) {
    return v.exchange(0u, std::memory_order_relaxed);
}

IoProbeSnapshot snapshotAndReset(IoProbeCounters &c) {
    IoProbeSnapshot s;
    s.netWriteWallUs = loadAndReset(c.netWriteWallUs);
    s.netReadWallUs = loadAndReset(c.netReadWallUs);
    s.netWriteOps = loadAndReset(c.netWriteOps);
    s.netReadOps = loadAndReset(c.netReadOps);
    s.netSendSyscallUs = loadAndReset(c.netSendSyscallUs);
    s.netRecvSyscallUs = loadAndReset(c.netRecvSyscallUs);
    s.netSendSyscalls = loadAndReset(c.netSendSyscalls);
    s.netRecvSyscalls = loadAndReset(c.netRecvSyscalls);
    s.netSendBytes = loadAndReset(c.netSendBytes);
    s.netRecvBytes = loadAndReset(c.netRecvBytes);
    s.netSendEagain = loadAndReset(c.netSendEagain);
    s.netRecvEagain = loadAndReset(c.netRecvEagain);

    s.hostMemcpyH2DUs = loadAndReset(c.hostMemcpyH2DUs);
    s.hostMemcpyD2HUs = loadAndReset(c.hostMemcpyD2HUs);
    s.hostMemcpyH2DBytes = loadAndReset(c.hostMemcpyH2DBytes);
    s.hostMemcpyD2HBytes = loadAndReset(c.hostMemcpyD2HBytes);
    s.hostMemcpyH2DCalls = loadAndReset(c.hostMemcpyH2DCalls);
    s.hostMemcpyD2HCalls = loadAndReset(c.hostMemcpyD2HCalls);

    s.vulkanFlushUs = loadAndReset(c.vulkanFlushUs);
    s.vulkanFlushBytes = loadAndReset(c.vulkanFlushBytes);
    s.vulkanFlushCalls = loadAndReset(c.vulkanFlushCalls);
    s.vulkanInvalidateUs = loadAndReset(c.vulkanInvalidateUs);
    s.vulkanInvalidateBytes = loadAndReset(c.vulkanInvalidateBytes);
    s.vulkanInvalidateCalls = loadAndReset(c.vulkanInvalidateCalls);
    s.vulkanCopyH2DUs = loadAndReset(c.vulkanCopyH2DUs);
    s.vulkanCopyD2HUs = loadAndReset(c.vulkanCopyD2HUs);
    s.vulkanCopyH2DBytes = loadAndReset(c.vulkanCopyH2DBytes);
    s.vulkanCopyD2HBytes = loadAndReset(c.vulkanCopyD2HBytes);
    s.vulkanCopyH2DCalls = loadAndReset(c.vulkanCopyH2DCalls);
    s.vulkanCopyD2HCalls = loadAndReset(c.vulkanCopyD2HCalls);

    s.cudaMemcpyH2DUs = loadAndReset(c.cudaMemcpyH2DUs);
    s.cudaMemcpyD2HUs = loadAndReset(c.cudaMemcpyD2HUs);
    s.cudaMemcpyH2DBytes = loadAndReset(c.cudaMemcpyH2DBytes);
    s.cudaMemcpyD2HBytes = loadAndReset(c.cudaMemcpyD2HBytes);
    s.cudaMemcpyH2DCalls = loadAndReset(c.cudaMemcpyH2DCalls);
    s.cudaMemcpyD2HCalls = loadAndReset(c.cudaMemcpyD2HCalls);
    s.cudaSyncH2DUs = loadAndReset(c.cudaSyncH2DUs);
    s.cudaSyncD2HUs = loadAndReset(c.cudaSyncD2HUs);
    s.cudaSyncH2DCalls = loadAndReset(c.cudaSyncH2DCalls);
    s.cudaSyncD2HCalls = loadAndReset(c.cudaSyncD2HCalls);
    return s;
}

bool snapshotHasData(const IoProbeSnapshot &s) {
    return
        s.netWriteWallUs || s.netReadWallUs || s.netWriteOps || s.netReadOps ||
        s.netSendSyscallUs || s.netRecvSyscallUs || s.netSendSyscalls || s.netRecvSyscalls ||
        s.netSendBytes || s.netRecvBytes || s.netSendEagain || s.netRecvEagain ||
        s.hostMemcpyH2DUs || s.hostMemcpyD2HUs || s.hostMemcpyH2DBytes || s.hostMemcpyD2HBytes ||
        s.hostMemcpyH2DCalls || s.hostMemcpyD2HCalls ||
        s.vulkanFlushUs || s.vulkanFlushBytes || s.vulkanFlushCalls ||
        s.vulkanInvalidateUs || s.vulkanInvalidateBytes || s.vulkanInvalidateCalls ||
        s.vulkanCopyH2DUs || s.vulkanCopyD2HUs || s.vulkanCopyH2DBytes || s.vulkanCopyD2HBytes ||
        s.vulkanCopyH2DCalls || s.vulkanCopyD2HCalls ||
        s.cudaMemcpyH2DUs || s.cudaMemcpyD2HUs || s.cudaMemcpyH2DBytes || s.cudaMemcpyD2HBytes ||
        s.cudaMemcpyH2DCalls || s.cudaMemcpyD2HCalls ||
        s.cudaSyncH2DUs || s.cudaSyncD2HUs || s.cudaSyncH2DCalls || s.cudaSyncD2HCalls;
}

unsigned long currentPid() {
#ifdef _WIN32
    return (unsigned long)GetCurrentProcessId();
#else
    return (unsigned long)getpid();
#endif
}

void flushAtExit() {
    dllamaIoProbeFlush("atexit");
}

void ensureConfiguredFromEnv(IoProbeState &s) {
    if (s.configured.load(std::memory_order_acquire)) return;
    const char *path = std::getenv("DLLAMA_IO_PROFILE_LOG");
    dllamaIoProbeConfigure(path);
}

void registerAtExit(IoProbeState &s) {
    bool expected = false;
    if (s.atexitRegistered.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        std::atexit(flushAtExit);
    }
}

void add(std::atomic<std::uint64_t> &dst, std::uint64_t v) {
    dst.fetch_add(v, std::memory_order_relaxed);
}

} // namespace

void dllamaIoProbeConfigure(const char *logPath) {
    IoProbeState &s = state();
    const bool hasPath = logPath != nullptr && logPath[0] != '\0';
    {
        std::lock_guard<std::mutex> lock(s.pathMutex);
        s.logPath = hasPath ? std::string(logPath) : std::string();
    }
    s.enabled.store(hasPath, std::memory_order_release);
    s.configured.store(true, std::memory_order_release);
    if (hasPath) registerAtExit(s);
}

void dllamaIoProbeConfigureFromEnv() {
    const char *path = std::getenv("DLLAMA_IO_PROFILE_LOG");
    dllamaIoProbeConfigure(path);
}

void dllamaIoProbeSetNode(unsigned int nodeIndex, unsigned int stageIndex) {
    IoProbeState &s = state();
    s.nodeIndex.store(nodeIndex, std::memory_order_relaxed);
    s.stageIndex.store(stageIndex, std::memory_order_relaxed);
}

bool dllamaIoProbeEnabled() {
    IoProbeState &s = state();
    ensureConfiguredFromEnv(s);
    return s.enabled.load(std::memory_order_acquire);
}

std::uint64_t dllamaIoProbeNowUs() {
    return (std::uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void dllamaIoProbeFlush(const char *reason) {
    IoProbeState &s = state();
    ensureConfiguredFromEnv(s);
    if (!s.enabled.load(std::memory_order_acquire)) return;

    IoProbeSnapshot snap = snapshotAndReset(s.counters);
    if (!snapshotHasData(snap)) return;

    std::string path;
    {
        std::lock_guard<std::mutex> lock(s.pathMutex);
        path = s.logPath;
    }
    if (path.empty()) return;

    FILE *f = std::fopen(path.c_str(), "a");
    if (f == nullptr) return;

    const unsigned long long seq = s.flushSeq.fetch_add(1u, std::memory_order_relaxed) + 1u;
    const std::uint64_t tsUs = (std::uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const unsigned int node = s.nodeIndex.load(std::memory_order_relaxed);
    const unsigned int stage = s.stageIndex.load(std::memory_order_relaxed);
    const char *label = reason == nullptr ? "flush" : reason;

    std::fprintf(f,
        "io_profile seq=%llu reason=%s ts_us=%llu pid=%lu node=%u stage=%u\n",
        seq, label, (unsigned long long)tsUs, currentPid(), node, stage);
    std::fprintf(f,
        "net write_ops=%llu read_ops=%llu write_wall_us=%llu read_wall_us=%llu send_syscalls=%llu recv_syscalls=%llu send_syscall_us=%llu recv_syscall_us=%llu send_bytes=%llu recv_bytes=%llu send_eagain=%llu recv_eagain=%llu\n",
        (unsigned long long)snap.netWriteOps,
        (unsigned long long)snap.netReadOps,
        (unsigned long long)snap.netWriteWallUs,
        (unsigned long long)snap.netReadWallUs,
        (unsigned long long)snap.netSendSyscalls,
        (unsigned long long)snap.netRecvSyscalls,
        (unsigned long long)snap.netSendSyscallUs,
        (unsigned long long)snap.netRecvSyscallUs,
        (unsigned long long)snap.netSendBytes,
        (unsigned long long)snap.netRecvBytes,
        (unsigned long long)snap.netSendEagain,
        (unsigned long long)snap.netRecvEagain);
    std::fprintf(f,
        "copy host_memcpy_h2d_calls=%llu host_memcpy_d2h_calls=%llu host_memcpy_h2d_us=%llu host_memcpy_d2h_us=%llu host_memcpy_h2d_bytes=%llu host_memcpy_d2h_bytes=%llu vulkan_flush_calls=%llu vulkan_flush_us=%llu vulkan_flush_bytes=%llu vulkan_invalidate_calls=%llu vulkan_invalidate_us=%llu vulkan_invalidate_bytes=%llu vulkan_copy_h2d_calls=%llu vulkan_copy_d2h_calls=%llu vulkan_copy_h2d_us=%llu vulkan_copy_d2h_us=%llu vulkan_copy_h2d_bytes=%llu vulkan_copy_d2h_bytes=%llu cuda_memcpy_h2d_calls=%llu cuda_memcpy_d2h_calls=%llu cuda_memcpy_h2d_us=%llu cuda_memcpy_d2h_us=%llu cuda_memcpy_h2d_bytes=%llu cuda_memcpy_d2h_bytes=%llu cuda_sync_h2d_calls=%llu cuda_sync_d2h_calls=%llu cuda_sync_h2d_us=%llu cuda_sync_d2h_us=%llu\n",
        (unsigned long long)snap.hostMemcpyH2DCalls,
        (unsigned long long)snap.hostMemcpyD2HCalls,
        (unsigned long long)snap.hostMemcpyH2DUs,
        (unsigned long long)snap.hostMemcpyD2HUs,
        (unsigned long long)snap.hostMemcpyH2DBytes,
        (unsigned long long)snap.hostMemcpyD2HBytes,
        (unsigned long long)snap.vulkanFlushCalls,
        (unsigned long long)snap.vulkanFlushUs,
        (unsigned long long)snap.vulkanFlushBytes,
        (unsigned long long)snap.vulkanInvalidateCalls,
        (unsigned long long)snap.vulkanInvalidateUs,
        (unsigned long long)snap.vulkanInvalidateBytes,
        (unsigned long long)snap.vulkanCopyH2DCalls,
        (unsigned long long)snap.vulkanCopyD2HCalls,
        (unsigned long long)snap.vulkanCopyH2DUs,
        (unsigned long long)snap.vulkanCopyD2HUs,
        (unsigned long long)snap.vulkanCopyH2DBytes,
        (unsigned long long)snap.vulkanCopyD2HBytes,
        (unsigned long long)snap.cudaMemcpyH2DCalls,
        (unsigned long long)snap.cudaMemcpyD2HCalls,
        (unsigned long long)snap.cudaMemcpyH2DUs,
        (unsigned long long)snap.cudaMemcpyD2HUs,
        (unsigned long long)snap.cudaMemcpyH2DBytes,
        (unsigned long long)snap.cudaMemcpyD2HBytes,
        (unsigned long long)snap.cudaSyncH2DCalls,
        (unsigned long long)snap.cudaSyncD2HCalls,
        (unsigned long long)snap.cudaSyncH2DUs,
        (unsigned long long)snap.cudaSyncD2HUs);
    std::fclose(f);
}

void dllamaIoProbeRecordNetWriteWall(std::uint64_t elapsedUs) {
    IoProbeCounters &c = state().counters;
    add(c.netWriteOps, 1u);
    add(c.netWriteWallUs, elapsedUs);
}

void dllamaIoProbeRecordNetReadWall(std::uint64_t elapsedUs) {
    IoProbeCounters &c = state().counters;
    add(c.netReadOps, 1u);
    add(c.netReadWallUs, elapsedUs);
}

void dllamaIoProbeRecordNetSendSyscall(std::uint64_t elapsedUs, std::uint64_t bytes) {
    IoProbeCounters &c = state().counters;
    add(c.netSendSyscalls, 1u);
    add(c.netSendSyscallUs, elapsedUs);
    add(c.netSendBytes, bytes);
}

void dllamaIoProbeRecordNetRecvSyscall(std::uint64_t elapsedUs, std::uint64_t bytes) {
    IoProbeCounters &c = state().counters;
    add(c.netRecvSyscalls, 1u);
    add(c.netRecvSyscallUs, elapsedUs);
    add(c.netRecvBytes, bytes);
}

void dllamaIoProbeRecordNetSendEagain() {
    add(state().counters.netSendEagain, 1u);
}

void dllamaIoProbeRecordNetRecvEagain() {
    add(state().counters.netRecvEagain, 1u);
}

void dllamaIoProbeRecordHostMemcpy(DllamaIoProbeDirection direction, std::uint64_t elapsedUs, std::uint64_t bytes) {
    IoProbeCounters &c = state().counters;
    if (direction == DLLAMA_IO_PROBE_H2D) {
        add(c.hostMemcpyH2DCalls, 1u);
        add(c.hostMemcpyH2DUs, elapsedUs);
        add(c.hostMemcpyH2DBytes, bytes);
    } else {
        add(c.hostMemcpyD2HCalls, 1u);
        add(c.hostMemcpyD2HUs, elapsedUs);
        add(c.hostMemcpyD2HBytes, bytes);
    }
}

void dllamaIoProbeRecordVulkanFlush(std::uint64_t elapsedUs, std::uint64_t bytes) {
    IoProbeCounters &c = state().counters;
    add(c.vulkanFlushCalls, 1u);
    add(c.vulkanFlushUs, elapsedUs);
    add(c.vulkanFlushBytes, bytes);
}

void dllamaIoProbeRecordVulkanInvalidate(std::uint64_t elapsedUs, std::uint64_t bytes) {
    IoProbeCounters &c = state().counters;
    add(c.vulkanInvalidateCalls, 1u);
    add(c.vulkanInvalidateUs, elapsedUs);
    add(c.vulkanInvalidateBytes, bytes);
}

void dllamaIoProbeRecordVulkanCopyCommand(DllamaIoProbeDirection direction, std::uint64_t elapsedUs, std::uint64_t bytes) {
    IoProbeCounters &c = state().counters;
    if (direction == DLLAMA_IO_PROBE_H2D) {
        add(c.vulkanCopyH2DCalls, 1u);
        add(c.vulkanCopyH2DUs, elapsedUs);
        add(c.vulkanCopyH2DBytes, bytes);
    } else {
        add(c.vulkanCopyD2HCalls, 1u);
        add(c.vulkanCopyD2HUs, elapsedUs);
        add(c.vulkanCopyD2HBytes, bytes);
    }
}

void dllamaIoProbeRecordCudaMemcpyAsync(DllamaIoProbeDirection direction, std::uint64_t elapsedUs, std::uint64_t bytes) {
    IoProbeCounters &c = state().counters;
    if (direction == DLLAMA_IO_PROBE_H2D) {
        add(c.cudaMemcpyH2DCalls, 1u);
        add(c.cudaMemcpyH2DUs, elapsedUs);
        add(c.cudaMemcpyH2DBytes, bytes);
    } else {
        add(c.cudaMemcpyD2HCalls, 1u);
        add(c.cudaMemcpyD2HUs, elapsedUs);
        add(c.cudaMemcpyD2HBytes, bytes);
    }
}

void dllamaIoProbeRecordCudaStreamSync(DllamaIoProbeDirection direction, std::uint64_t elapsedUs) {
    IoProbeCounters &c = state().counters;
    if (direction == DLLAMA_IO_PROBE_H2D) {
        add(c.cudaSyncH2DCalls, 1u);
        add(c.cudaSyncH2DUs, elapsedUs);
    } else {
        add(c.cudaSyncD2HCalls, 1u);
        add(c.cudaSyncD2HUs, elapsedUs);
    }
}
