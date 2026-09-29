#include "device_profile.hpp"

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

static double cpuGemvMs(unsigned rows, unsigned cols, unsigned nThreads) {
    if (rows == 0u || cols == 0u) return 0.0;
    std::vector<float> weight((size_t)rows * (size_t)cols, 0.01f);
    std::vector<float> input(cols, 0.02f);
    std::vector<float> output(rows, 0.0f);
    const unsigned nth = std::max(1u, nThreads);
    auto once = [&]() {
        std::vector<std::thread> threads;
        threads.reserve(nth);
        for (unsigned ith = 0; ith < nth; ++ith) {
            threads.emplace_back([&, ith]() {
                for (unsigned row = ith; row < rows; row += nth) {
                    const float *rowData = weight.data() + (size_t)row * cols;
                    float acc = 0.0f;
                    for (unsigned col = 0; col < cols; ++col) acc += rowData[col] * input[col];
                    output[row] = acc;
                }
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

static double cpuLayerMs(const ModelShape &shape) {
    const unsigned dim = shape.dim;
    const unsigned hidden = shape.hiddenDim == 0u ? dim : shape.hiddenDim;
    const unsigned heads = shape.nHeads == 0u ? 1u : shape.nHeads;
    const unsigned kvHeads = shape.nKvHeads == 0u ? heads : shape.nKvHeads;
    const unsigned kvDim = (unsigned)(((unsigned long long)dim / heads) * kvHeads);
    const unsigned kv = kvDim == 0u ? dim : kvDim;
    return 2.0 * cpuGemvMs(dim, dim, g_threads)
        + 2.0 * cpuGemvMs(kv, dim, g_threads)
        + 2.0 * cpuGemvMs(hidden, dim, g_threads)
        + cpuGemvMs(dim, hidden, g_threads);
}

unsigned long long profileDeviceFreeBytes() {
    try {
#if defined(DLLAMA_CUDA)
        if (g_backend == PROFILE_CUDA) return nnCudaFreeBytes((unsigned)g_gpuIndex);
#endif
#if defined(DLLAMA_VULKAN)
        if (g_backend == PROFILE_VULKAN) return nnVulkanDeviceLocalBytes((unsigned)g_gpuIndex);
#endif
    } catch (...) {
        return hostAvailableBytes();
    }
    return hostAvailableBytes();
}

SpeedDeviceProfile measureLocalProfile(const ModelShape &shape) {
    if (shape.dim == 0u) return unknownSpeedProfile();
    unsigned long long freeBytes = hostAvailableBytes();
    const char *name = "cpu";
    double layerMs = 0.0;
    try {
#if defined(DLLAMA_CUDA)
        if (g_backend == PROFILE_CUDA) {
            layerMs = nnCudaProfileLayerMs((unsigned)g_gpuIndex, shape.dim, shape.hiddenDim, shape.nHeads, shape.nKvHeads);
            freeBytes = nnCudaFreeBytes((unsigned)g_gpuIndex);
            name = "cuda";
        } else
#endif
#if defined(DLLAMA_VULKAN)
        if (g_backend == PROFILE_VULKAN) {
            layerMs = nnVulkanProfileLayerMs((unsigned)g_gpuIndex, shape.dim, shape.hiddenDim, shape.nHeads, shape.nKvHeads);
            freeBytes = nnVulkanDeviceLocalBytes((unsigned)g_gpuIndex);
            name = "vulkan";
        } else
#endif
        {
            layerMs = cpuLayerMs(shape);
            freeBytes = hostAvailableBytes();
            name = "cpu";
        }
    } catch (...) {
        return unknownSpeedProfile();
    }
    if (!(layerMs > 0.0)) layerMs = 0.05;
    SpeedDeviceProfile profile = makeSpeedProfile(layerMs, estimateLayerCap(freeBytes, shape), name);
    if (profile.cap < 1u) profile.cap = 1u;
    return profile;
}
