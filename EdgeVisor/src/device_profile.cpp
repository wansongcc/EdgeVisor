#include "device_profile.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>
#if defined(__linux__)
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <dirent.h>
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

// nvidia-smi is shelled out to rather than linked through NVML: this runs once
// per stage at startup and a missing driver must not fail the process.
static bool queryNvidiaCondition(int *tempC, int *thermalLimited) {
    *tempC = -1;
    *thermalLimited = -1;
#if defined(__linux__)
    char command[512];
    std::snprintf(command, sizeof(command),
        "timeout 2 nvidia-smi --query-gpu=temperature.gpu,"
        "clocks_event_reasons.hw_thermal_slowdown,"
        "clocks_event_reasons.sw_thermal_slowdown,"
        "clocks_event_reasons.hw_power_brake_slowdown "
        "--format=csv,noheader -i %d 2>/dev/null", g_gpuIndex);
    FILE *pipe = popen(command, "r");
    if (pipe == nullptr) return false;
    char line[512];
    line[0] = '\0';
    const bool gotLine = std::fgets(line, sizeof(line), pipe) != nullptr;
    // Closing the pipe reaps the child, so read before this.
    const int status = pclose(pipe);
    if (!gotLine || status != 0) return false;
    int parsedTempC = -1;
    int parsedThermalLimited = -1;
    if (!parseNvidiaTemperatureCsv(line, &parsedTempC, &parsedThermalLimited)) return false;
    *tempC = parsedTempC;
    *thermalLimited = parsedThermalLimited;
    return true;
#else
    return false;
#endif
}

#if defined(__linux__)
// One attribute of one supply, trimmed. An unreadable file leaves value empty.
static std::string readSupplyAttr(const char *path) {
    std::ifstream file(path);
    if (!file.is_open()) return std::string();
    std::string value;
    std::getline(file, value);
    const std::size_t first = value.find_first_not_of(" \t\r\n\v\f");
    if (first == std::string::npos) return std::string();
    const std::size_t last = value.find_last_not_of(" \t\r\n\v\f");
    return value.substr(first, last - first + 1u);
}

// Scans /sys/class/power_supply for a battery that is discharging, or anything
// that proves the machine is on mains. A desktop without a battery only reports
// online=1 for Mains/USB; a laptop on battery reports Discharging.
static int readOnExternalPower() {
    DIR *dir = opendir("/sys/class/power_supply");
    if (dir == nullptr) return externalPowerFromSupplies(false, false, false);
    bool sawSupply = false;
    bool batteryDischarging = false;
    bool externalOnline = false;
    while (struct dirent *entry = readdir(dir)) {
        if (entry->d_name[0] == '.') continue;
        const std::string base = std::string("/sys/class/power_supply/") + entry->d_name;
        const std::string type = readSupplyAttr((base + "/type").c_str());
        if (type.empty()) continue;
        sawSupply = true;
        if (type == "Battery") {
            const std::string status = readSupplyAttr((base + "/status").c_str());
            if (status == "Discharging") batteryDischarging = true;
            else if (status == "Charging" || status == "Full" || status == "Not charging") externalOnline = true;
        } else if (type == "Mains" || type == "USB") {
            if (readSupplyAttr((base + "/online").c_str()) == "1") externalOnline = true;
        }
    }
    closedir(dir);
    return externalPowerFromSupplies(sawSupply, batteryDischarging, externalOnline);
}
#endif

#if defined(__linux__)
// nvidia-smi on Jetson often prints [N/A] for temperature. The kernel still
// publishes gpu-thermal in millidegrees. The hottest zone wins when several
// GPUs share the node and this process was not given an index.
static int readGpuThermalZoneC() {
    DIR *dir = opendir("/sys/class/thermal");
    if (dir == nullptr) return -1;
    int best = -1;
    while (struct dirent *entry = readdir(dir)) {
        if (std::strncmp(entry->d_name, "thermal_zone", 12) != 0) continue;
        const std::string base = std::string("/sys/class/thermal/") + entry->d_name;
        if (readSupplyAttr((base + "/type").c_str()) != "gpu-thermal") continue;
        const std::string milli = readSupplyAttr((base + "/temp").c_str());
        char *stop = nullptr;
        const long parsed = std::strtol(milli.c_str(), &stop, 10);
        if (stop == milli.c_str() || *stop != '\0' || parsed < 0) continue;
        const int celsius = (int)(parsed / 1000L);
        if (celsius > best) best = celsius;
    }
    closedir(dir);
    return best;
}
#endif

DeviceCondition measureLocalCondition() {
    DeviceCondition condition;
    // Neither probe is allowed to take the process down: a machine without a
    // GPU or without sysfs simply reports those fields as unknown.
    queryNvidiaCondition(&condition.temperatureC, &condition.thermalLimited);
#if defined(__linux__)
    if (condition.temperatureC < 0) condition.temperatureC = readGpuThermalZoneC();
    condition.onExternalPower = readOnExternalPower();
#endif
    return condition;
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

#if defined(__linux__)
static int parseLeadingInt(const std::string &text) {
    if (text.find("N/A") != std::string::npos) return -1;
    char *stop = nullptr;
    const long parsed = std::strtol(text.c_str(), &stop, 10);
    if (stop == text.c_str()) return -1;
    return (int)parsed;
}

static std::string firstCsvField(const std::string &line) {
    const std::size_t comma = line.find(',');
    std::string field = comma == std::string::npos ? line : line.substr(0, comma);
    const std::size_t first = field.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return std::string();
    const std::size_t last = field.find_last_not_of(" \t\r\n");
    return field.substr(first, last - first + 1u);
}

static std::string secondCsvField(const std::string &line) {
    const std::size_t comma = line.find(',');
    if (comma == std::string::npos) return std::string();
    return firstCsvField(line.substr(comma + 1u));
}

static void readNvidiaClocks(int *freqMhz, int *freqMaxMhz) {
    *freqMhz = -1;
    *freqMaxMhz = -1;
    char command[256];
    std::snprintf(command, sizeof(command),
        "timeout 2 nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader -i %d 2>/dev/null",
        g_gpuIndex);
    FILE *pipe = popen(command, "r");
    if (pipe == nullptr) return;
    char line[256];
    line[0] = '\0';
    const bool got = std::fgets(line, sizeof(line), pipe) != nullptr;
    const int status = pclose(pipe);
    if (!got || status != 0) return;
    *freqMhz = parseLeadingInt(firstCsvField(line));
    *freqMaxMhz = parseLeadingInt(secondCsvField(line));
}

static void readDevfreqGpuClocks(int *freqMhz, int *freqMaxMhz) {
    if (*freqMhz >= 0 && *freqMaxMhz >= 0) return;
    DIR *dir = opendir("/sys/class/devfreq");
    if (dir == nullptr) return;
    while (struct dirent *entry = readdir(dir)) {
        if (entry->d_name[0] == '.') continue;
        const std::string name(entry->d_name);
        if (name.find("gpu") == std::string::npos && name.find("GPU") == std::string::npos) continue;
        const std::string base = std::string("/sys/class/devfreq/") + name;
        if (*freqMhz < 0) {
            const std::string cur = readSupplyAttr((base + "/cur_freq").c_str());
            const int mhz = parseLeadingInt(cur);
            if (mhz > 1000) *freqMhz = mhz / 1000000;
        }
        if (*freqMaxMhz < 0) {
            const std::string peak = readSupplyAttr((base + "/max_freq").c_str());
            const int mhz = parseLeadingInt(peak);
            if (mhz > 1000) *freqMaxMhz = mhz / 1000000;
        }
    }
    closedir(dir);
}

static int readBatteryPercent() {
    DIR *dir = opendir("/sys/class/power_supply");
    if (dir == nullptr) return -1;
    int percent = -1;
    while (struct dirent *entry = readdir(dir)) {
        if (entry->d_name[0] == '.') continue;
        const std::string base = std::string("/sys/class/power_supply/") + entry->d_name;
        if (readSupplyAttr((base + "/type").c_str()) != "Battery") continue;
        const int parsed = parseLeadingInt(readSupplyAttr((base + "/capacity").c_str()));
        if (parsed >= 0 && parsed <= 100) percent = parsed;
    }
    closedir(dir);
    return percent;
}
#endif

HardwareTick readHardwareTick() {
    HardwareTick tick;
    const DeviceCondition condition = measureLocalCondition();
    tick.temperatureC = condition.temperatureC;
    tick.thermalLimited = condition.thermalLimited;
    tick.onExternalPower = condition.onExternalPower;
#if defined(__linux__)
    tick.batteryPercent = readBatteryPercent();
    readNvidiaClocks(&tick.freqMhz, &tick.freqMaxMhz);
    readDevfreqGpuClocks(&tick.freqMhz, &tick.freqMaxMhz);
#endif
    return tick;
}

struct RepresentativeLayerBench::Impl {
    bool ready;
    unsigned dim;
    unsigned hidden;
    unsigned kv;
    unsigned threads;
#if defined(DLLAMA_CUDA)
    NnCudaLayerSoak *cuda;
#endif
#if defined(DLLAMA_VULKAN)
    bool vulkan;
    unsigned gpuIndex;
    unsigned nHeads;
    unsigned nKvHeads;
#endif
    std::vector<float> weight;
    std::vector<float> input;
    std::vector<float> output;
    Impl()
        : ready(false), dim(0u), hidden(0u), kv(0u), threads(1u)
#if defined(DLLAMA_CUDA)
        , cuda(nullptr)
#endif
#if defined(DLLAMA_VULKAN)
        , vulkan(false), gpuIndex(0u), nHeads(0u), nKvHeads(0u)
#endif
    {}
};

static void cpuGemvOnce(
    const float *weight,
    const float *input,
    float *output,
    unsigned rows,
    unsigned cols,
    unsigned nThreads) {
    const unsigned nth = std::max(1u, nThreads);
    std::vector<std::thread> threads;
    threads.reserve(nth);
    for (unsigned ith = 0u; ith < nth; ++ith) {
        threads.push_back(std::thread([&, ith]() {
            for (unsigned row = ith; row < rows; row += nth) {
                const float *rowData = weight + (size_t)row * cols;
                float acc = 0.0f;
                for (unsigned col = 0u; col < cols; ++col) acc += rowData[col] * input[col];
                output[row] = acc;
            }
        }));
    }
    for (unsigned ith = 0u; ith < nth; ++ith) threads[ith].join();
}

RepresentativeLayerBench::RepresentativeLayerBench(const ModelShape &shape) : impl(new Impl()) {
    if (shape.dim == 0u) return;
    const unsigned heads = shape.nHeads == 0u ? 1u : shape.nHeads;
    const unsigned kvHeads = shape.nKvHeads == 0u ? heads : shape.nKvHeads;
    const unsigned kvDim = (unsigned)(((unsigned long long)shape.dim / heads) * kvHeads);
    impl->dim = shape.dim;
    impl->hidden = shape.hiddenDim == 0u ? shape.dim : shape.hiddenDim;
    impl->kv = kvDim == 0u ? shape.dim : kvDim;
    impl->threads = g_threads == 0u ? 1u : g_threads;
#if defined(DLLAMA_CUDA)
    if (g_backend == PROFILE_CUDA) {
        try {
            impl->cuda = new NnCudaLayerSoak(
                (unsigned)g_gpuIndex, shape.dim, impl->hidden, shape.nHeads, shape.nKvHeads);
            impl->ready = true;
        } catch (...) {
            impl->ready = false;
        }
        return;
    }
#endif
#if defined(DLLAMA_VULKAN)
    if (g_backend == PROFILE_VULKAN) {
        impl->vulkan = true;
        impl->gpuIndex = (unsigned)g_gpuIndex;
        impl->nHeads = shape.nHeads;
        impl->nKvHeads = shape.nKvHeads;
        impl->ready = true;
        return;
    }
#endif
    const unsigned long long dimCells = (unsigned long long)impl->dim * (unsigned long long)impl->dim;
    const unsigned long long kvCells = (unsigned long long)impl->kv * (unsigned long long)impl->dim;
    const unsigned long long upCells = (unsigned long long)impl->hidden * (unsigned long long)impl->dim;
    const unsigned long long downCells = (unsigned long long)impl->dim * (unsigned long long)impl->hidden;
    const unsigned long long cells = std::max(dimCells, std::max(kvCells, std::max(upCells, downCells)));
    const unsigned maxVector = std::max(impl->dim, std::max(impl->hidden, impl->kv));
    try {
        impl->weight.assign((size_t)cells, 0.01f);
        impl->input.assign(maxVector, 0.02f);
        impl->output.assign(maxVector, 0.0f);
        impl->ready = true;
    } catch (...) {
        impl->ready = false;
    }
}

RepresentativeLayerBench::~RepresentativeLayerBench() {
    if (impl != nullptr) {
#if defined(DLLAMA_CUDA)
        delete impl->cuda;
        impl->cuda = nullptr;
#endif
        delete impl;
    }
}

double RepresentativeLayerBench::runOnceMs() {
    if (impl == nullptr || !impl->ready) return -1.0;
#if defined(DLLAMA_CUDA)
    if (impl->cuda != nullptr) return impl->cuda->runOnceMs();
#endif
#if defined(DLLAMA_VULKAN)
    if (impl->vulkan) {
        try {
            return nnVulkanProfileLayerMs(impl->gpuIndex, impl->dim, impl->hidden, impl->nHeads, impl->nKvHeads);
        } catch (...) {
            return -1.0;
        }
    }
#endif
    if (impl->weight.empty()) return -1.0;
    const auto once = [&](unsigned rows, unsigned cols) {
        cpuGemvOnce(impl->weight.data(), impl->input.data(), impl->output.data(), rows, cols, impl->threads);
    };
    const auto start = std::chrono::steady_clock::now();
    once(impl->dim, impl->dim);
    once(impl->dim, impl->dim);
    once(impl->kv, impl->dim);
    once(impl->kv, impl->dim);
    once(impl->hidden, impl->dim);
    once(impl->hidden, impl->dim);
    once(impl->dim, impl->hidden);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
