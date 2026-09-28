#include "device_profile.hpp"
#include "nn/llamafile/sgemm.hpp"
#include "nn/nn-quants.hpp"

#include <chrono>
#include <cmath>
#include <map>
#include <thread>
#include <vector>

#include <unistd.h>
#if defined(__linux__)
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <fstream>
#endif

#if defined(DLLAMA_CUDA)
#include "nn/nn-cuda.hpp"
#endif
#if defined(DLLAMA_VULKAN)
#include "nn/nn-vulkan.hpp"
#endif

static int g_backend = PROFILE_CPU;
static int g_gpuIndex = 0;
static unsigned g_threads = 1;
static std::map<std::string, SpeedDeviceProfile> g_profileCache;

void publishProfileRuntime(int backend, int gpuIndex, unsigned nThreads) {
    g_backend = backend;
    g_gpuIndex = gpuIndex < 0 ? 0 : gpuIndex;
    g_threads = nThreads == 0u ? 1u : nThreads;
}

void cacheSpeedProfile(const std::string &key, const SpeedDeviceProfile &profile) {
    g_profileCache[key] = profile;
}

bool cachedSpeedProfile(const std::string &key, SpeedDeviceProfile *out) {
    if (out == nullptr) return false;
    const std::map<std::string, SpeedDeviceProfile>::const_iterator found = g_profileCache.find(key);
    if (found == g_profileCache.end()) return false;
    *out = found->second;
    return true;
}

bool localKnownSpeedProfile(SpeedDeviceProfile *out) {
#if defined(__linux__)
    struct ifaddrs *list = nullptr;
    if (getifaddrs(&list) != 0) return false;
    for (struct ifaddrs *ifa = list; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET) continue;
        char text[INET_ADDRSTRLEN];
        const struct sockaddr_in *addr = (const struct sockaddr_in *)ifa->ifa_addr;
        if (inet_ntop(AF_INET, &addr->sin_addr, text, sizeof(text)) == nullptr) continue;
        if (knownSpeedProfile(text, out)) {
            freeifaddrs(list);
            return true;
        }
    }
    freeifaddrs(list);
#else
    (void)out;
#endif
    return false;
}

static unsigned long long hostAvailableBytes() {
#if defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    unsigned long long value = 0;
    std::string unit;
    while (meminfo >> key >> value >> unit) {
        if (key == "MemAvailable:") return value * 1024ull;
    }
#endif
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && pageSize > 0) return (unsigned long long)pages * (unsigned long long)pageSize;
#endif
    return 4ull << 30;
}

static double cpuGemmMs(int n, unsigned nThreads) {
    std::vector<float> a((size_t)n * (size_t)n, 0.01f);
    std::vector<float> b((size_t)n * (size_t)n, 0.02f);
    std::vector<float> c((size_t)n * (size_t)n, 0.0f);
    const unsigned nth = std::max(1u, nThreads);
    auto once = [&]() {
        std::vector<std::thread> threads;
        threads.reserve(nth);
        for (unsigned ith = 0; ith < nth; ++ith) {
            threads.emplace_back([&, ith]() {
                llamafile_sgemm(n, n, n, a.data(), n, b.data(), n, c.data(), n,
                    (int)ith, (int)nth, 0, F_32, F_32, F_32);
            });
        }
        for (unsigned ith = 0; ith < nth; ++ith) threads[ith].join();
    };
    once();
    const auto start = std::chrono::steady_clock::now();
    const int iters = 3;
    for (int i = 0; i < iters; ++i) once();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return ms / (double)iters;
}

static double layerMsFromGemm(double gemmMs, int n, const ModelShape &shape) {
    const double flops = 2.0 * (double)n * (double)n * (double)n;
    if (!(gemmMs > 0.0) || !(flops > 0.0)) return 8.0;
    const double flopsPerSec = flops / (gemmMs / 1000.0);
    const unsigned heads = shape.nHeads == 0u ? 1u : shape.nHeads;
    const double kvScale = (double)(shape.nKvHeads == 0u ? heads : shape.nKvHeads) / (double)heads;
    const double layerFlops =
        2.0 * (double)shape.dim * (double)shape.hiddenDim * 3.0 +
        2.0 * (double)shape.dim * (double)shape.dim * (2.0 + 2.0 * kvScale);
    double ms = layerFlops / flopsPerSec * 1000.0;
    if (!(ms > 0.05)) ms = 0.05;
    return ms;
}

SpeedDeviceProfile measureLocalProfile(const ModelShape &shape) {
    double gemmMs = 0.0;
    unsigned long long freeBytes = hostAvailableBytes();
    const char *name = "cpu";
    int n = 256;
    bool measuredOnDevice = false;
    try {
#if defined(DLLAMA_CUDA)
        if (g_backend == PROFILE_CUDA) {
            n = 512;
            gemmMs = nnCudaProfileGemmMs((unsigned)g_gpuIndex, n);
            freeBytes = nnCudaFreeBytes((unsigned)g_gpuIndex);
            name = "cuda";
            measuredOnDevice = true;
        }
#endif
#if defined(DLLAMA_VULKAN)
        if (g_backend == PROFILE_VULKAN) {
            n = 256;
            gemmMs = nnVulkanProfileGemmMs((unsigned)g_gpuIndex, n);
            freeBytes = nnVulkanDeviceLocalBytes((unsigned)g_gpuIndex);
            name = "vulkan";
            measuredOnDevice = true;
        }
#endif
        if (!measuredOnDevice) {
            gemmMs = cpuGemmMs(n, g_threads);
            freeBytes = hostAvailableBytes();
            name = "cpu";
        }
    } catch (...) {
        return unknownSpeedProfile();
    }
    SpeedDeviceProfile profile = makeSpeedProfile(
        layerMsFromGemm(gemmMs, n, shape),
        estimateLayerCap(freeBytes, shape),
        name);
    if (profile.cap < 1u) profile.cap = 1u;
    return profile;
}
