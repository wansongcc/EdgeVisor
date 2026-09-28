#ifndef DEVICE_PROFILE_HPP
#define DEVICE_PROFILE_HPP

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// Backend used for the one-shot matmul that feeds speed-pack.
enum ProfileBackend {
    PROFILE_CPU = 0,
    PROFILE_VULKAN = 1,
    PROFILE_CUDA = 2,
};

struct ModelShape {
    unsigned dim;
    unsigned hiddenDim;
    unsigned nHeads;
    unsigned nKvHeads;
    unsigned maxSeqLen;
    unsigned nLayers;
    int weightType;
};

struct SpeedDeviceProfile {
    double msPerLayer;
    unsigned cap;
    char name[16];
};

inline SpeedDeviceProfile makeSpeedProfile(double msPerLayer, unsigned cap, const char *name) {
    SpeedDeviceProfile profile;
    profile.msPerLayer = msPerLayer;
    profile.cap = cap;
    std::memset(profile.name, 0, sizeof(profile.name));
    if (name != nullptr) {
        std::strncpy(profile.name, name, sizeof(profile.name) - 1);
    }
    return profile;
}

inline SpeedDeviceProfile unknownSpeedProfile() {
    // Same cap the lab uses when a Nano-sized device is not in the table.
    // Profiling replaces this for machines we can measure.
    return makeSpeedProfile(7.4, 8u, "unknown");
}

// Lab data-plane addresses. An explicit --ratios still wins over this table.
inline bool knownSpeedProfile(const char *host, SpeedDeviceProfile *out) {
    if (host == nullptr || out == nullptr) return false;
    if (std::strcmp(host, "192.168.137.13") == 0 || std::strcmp(host, "192.168.137.15") == 0) {
        *out = makeSpeedProfile(5.8, 22u, "nx");
        return true;
    }
    if (std::strcmp(host, "192.168.137.16") == 0 || std::strcmp(host, "192.168.137.18") == 0) {
        *out = makeSpeedProfile(7.4, 8u, "nano");
        return true;
    }
    if (std::strcmp(host, "192.168.137.31") == 0) {
        *out = makeSpeedProfile(8.3, 11u, "laptop");
        return true;
    }
    return false;
}

// How many layers fit in freeBytes, leaving one layer of headroom for the
// redundant boundary graph. Q40 weights (weightType == 2) sync in q80.
inline unsigned estimateLayerCap(unsigned long long freeBytes, const ModelShape &shape) {
    if (shape.dim == 0u || shape.hiddenDim == 0u) return 8u;
    double bytesPerWeight = 4.0;
    if (shape.weightType == 2) bytesPerWeight = 18.0 / 32.0;
    else if (shape.weightType == 1) bytesPerWeight = 2.0;
    else if (shape.weightType == 3) bytesPerWeight = 34.0 / 32.0;
    const unsigned heads = shape.nHeads == 0u ? 1u : shape.nHeads;
    const unsigned kvHeads = shape.nKvHeads == 0u ? heads : shape.nKvHeads;
    const unsigned kvDim = (unsigned)(((unsigned long long)shape.dim / heads) * kvHeads);
    const unsigned long long params =
        2ull * shape.dim * shape.dim +
        2ull * shape.dim * kvDim +
        3ull * shape.dim * shape.hiddenDim;
    const double weightBytes = (double)params * bytesPerWeight;
    const unsigned seq = shape.maxSeqLen == 0u ? 2048u : shape.maxSeqLen;
    const double syncBytes = (shape.weightType == 2) ? (34.0 / 32.0) : 2.0;
    const double perLayer = weightBytes + (2.0 * (double)seq * (double)kvDim * syncBytes);
    if (perLayer < 1.0) return shape.nLayers == 0u ? 8u : shape.nLayers;
    const double budget = (double)freeBytes * 0.70;
    if (budget <= perLayer) return 1u;
    unsigned cap = (unsigned)((budget - perLayer) / perLayer);
    if (cap < 1u) cap = 1u;
    if (shape.nLayers > 0u && cap > shape.nLayers) cap = shape.nLayers;
    return cap;
}

// Preserve pipeline order. Faster devices fill first, up to cap.
// Leftover layers stay on the fastest device, matching the lab speed-pack.
inline std::string assignSpeedPack(
    const std::vector<SpeedDeviceProfile> &profiles,
    unsigned nLayers,
    const std::vector<char> *liveNodes) {
    const unsigned nNodes = (unsigned)profiles.size();
    std::vector<unsigned> layers(nNodes, 0u);
    auto isLive = [&](unsigned node) {
        return liveNodes == nullptr || node >= liveNodes->size() || (*liveNodes)[node] != 0;
    };
    if (nNodes == 0u || nLayers == 0u) return std::string();
    const unsigned base = nLayers >= nNodes ? 1u : 0u;
    unsigned left = nLayers;
    for (unsigned i = 0; i < nNodes && left > 0u; ++i) {
        if (!isLive(i)) continue;
        const unsigned give = std::min(base, left);
        layers[i] = give;
        left -= give;
    }
    std::vector<unsigned> order;
    order.reserve(nNodes);
    for (unsigned i = 0; i < nNodes; ++i) {
        if (isLive(i)) order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
        if (profiles[a].msPerLayer != profiles[b].msPerLayer)
            return profiles[a].msPerLayer < profiles[b].msPerLayer;
        return a < b;
    });
    for (unsigned index : order) {
        if (left == 0u) break;
        const unsigned room = profiles[index].cap > layers[index] ? profiles[index].cap - layers[index] : 0u;
        const unsigned take = std::min(room, left);
        layers[index] += take;
        left -= take;
    }
    if (left > 0u && !order.empty()) layers[order[0]] += left;
    std::ostringstream out;
    for (unsigned i = 0; i < nNodes; ++i) {
        if (i > 0u) out << '*';
        out << "1@" << layers[i];
    }
    return out.str();
}

bool localKnownSpeedProfile(SpeedDeviceProfile *out);
void publishProfileRuntime(int backend, int gpuIndex, unsigned nThreads);
void cacheSpeedProfile(const std::string &key, const SpeedDeviceProfile &profile);
bool cachedSpeedProfile(const std::string &key, SpeedDeviceProfile *out);
SpeedDeviceProfile measureLocalProfile(const ModelShape &shape);

#endif
