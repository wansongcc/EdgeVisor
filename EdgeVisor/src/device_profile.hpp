#ifndef DEVICE_PROFILE_HPP
#define DEVICE_PROFILE_HPP

#include <algorithm>
#include <cstdlib>
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
inline unsigned kvDimOf(const ModelShape &shape) {
    if (shape.dim == 0u) return 0u;
    const unsigned heads = shape.nHeads == 0u ? 1u : shape.nHeads;
    const unsigned kvHeads = shape.nKvHeads == 0u ? heads : shape.nKvHeads;
    return (unsigned)(((unsigned long long)shape.dim / heads) * kvHeads);
}

inline double estimateLayerWeightBytes(const ModelShape &shape) {
    double bytesPerWeight = 4.0;
    if (shape.weightType == 2) bytesPerWeight = 18.0 / 32.0;
    else if (shape.weightType == 1) bytesPerWeight = 2.0;
    else if (shape.weightType == 3) bytesPerWeight = 34.0 / 32.0;
    const unsigned kvDim = kvDimOf(shape);
    const unsigned long long params =
        2ull * shape.dim * shape.dim +
        2ull * shape.dim * kvDim +
        3ull * shape.dim * shape.hiddenDim;
    return (double)params * bytesPerWeight;
}

inline unsigned capFromLayerBytes(unsigned long long freeBytes, double perLayer, unsigned nLayers) {
    if (perLayer < 1.0) return nLayers == 0u ? 8u : nLayers;
    const double budget = (double)freeBytes * 0.70;
    if (budget <= perLayer) return 1u;
    unsigned cap = (unsigned)((budget - perLayer) / perLayer);
    if (cap < 1u) cap = 1u;
    if (nLayers > 0u && cap > nLayers) cap = nLayers;
    return cap;
}

inline unsigned estimateLayerCap(unsigned long long freeBytes, const ModelShape &shape) {
    if (shape.dim == 0u || shape.hiddenDim == 0u) return 8u;
    const unsigned seq = shape.maxSeqLen == 0u ? 2048u : shape.maxSeqLen;
    const double syncBytes = (shape.weightType == 2) ? (34.0 / 32.0) : 2.0;
    const double perLayer = estimateLayerWeightBytes(shape)
        + (2.0 * (double)seq * (double)kvDimOf(shape) * syncBytes);
    return capFromLayerBytes(freeBytes, perLayer, shape.nLayers);
}

// --auto keeps a full F32 K and V for every resident layer. The speed-pack cap
// prices that KV at the sync width, so it admits copies that do not fit.
inline unsigned estimateResidentLayerCap(unsigned long long freeBytes, const ModelShape &shape) {
    if (shape.dim == 0u || shape.hiddenDim == 0u) return 8u;
    const unsigned seq = shape.maxSeqLen == 0u ? 2048u : shape.maxSeqLen;
    const double perLayer = estimateLayerWeightBytes(shape)
        + (2.0 * (double)seq * (double)kvDimOf(shape) * 4.0);
    return capFromLayerBytes(freeBytes, perLayer, shape.nLayers);
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

// Lab table cap, tightened by what this model fits in free memory.
inline unsigned clampLayerCap(unsigned tableCap, unsigned long long freeBytes, const ModelShape &shape) {
    const unsigned measured = estimateLayerCap(freeBytes, shape);
    if (tableCap < 1u) return measured;
    return std::min(tableCap, measured);
}

inline bool parsePackedCounts(const std::string &ratios, std::vector<unsigned> *counts) {
    if (counts == nullptr || ratios.empty()) return false;
    counts->clear();
    std::string token;
    for (std::size_t i = 0; i <= ratios.size(); ++i) {
        const bool end = i == ratios.size() || ratios[i] == '*';
        if (!end) {
            token.push_back(ratios[i]);
            continue;
        }
        if (token.size() < 3u || token[0] != '1' || token[1] != '@') return false;
        char *stop = nullptr;
        const unsigned long value = std::strtoul(token.c_str() + 2, &stop, 10);
        if (stop == token.c_str() + 2 || *stop != '\0') return false;
        counts->push_back((unsigned)value);
        token.clear();
    }
    return !counts->empty();
}

inline std::string formatPackedCounts(const std::vector<unsigned> &counts) {
    std::ostringstream out;
    for (unsigned i = 0; i < counts.size(); ++i) {
        if (i > 0u) out << '*';
        out << "1@" << counts[i];
    }
    return out.str();
}

// Startup alloc failed on failedNode. That node's layer count can only shrink.
// The freed layer moves to whichever other node still has room under its ceiling.
inline bool relaxPackedRatios(
    const std::string &ratios,
    unsigned failedNode,
    std::vector<unsigned> *ceilings,
    std::string *out) {
    if (ceilings == nullptr || out == nullptr) return false;
    std::vector<unsigned> counts;
    if (!parsePackedCounts(ratios, &counts)) return false;
    if (failedNode >= counts.size() || counts[failedNode] <= 1u) return false;
    std::vector<unsigned> caps = *ceilings;
    if (caps.size() != counts.size()) caps.assign(counts.size(), 0xffffffffu);
    counts[failedNode] -= 1u;
    caps[failedNode] = counts[failedNode];
    unsigned best = 0xffffffffu;
    unsigned bestCount = 0xffffffffu;
    for (unsigned i = 0; i < counts.size(); ++i) {
        if (i == failedNode) continue;
        const unsigned room = caps[i] == 0xffffffffu
            ? 0xffffffffu
            : (caps[i] > counts[i] ? caps[i] - counts[i] : 0u);
        if (room == 0u) continue;
        if (best == 0xffffffffu || counts[i] < bestCount) {
            best = i;
            bestCount = counts[i];
        }
    }
    if (best == 0xffffffffu) return false;
    counts[best] += 1u;
    *ceilings = caps;
    *out = formatPackedCounts(counts);
    return true;
}

inline bool startupAllocFailure(const char *message) {
    if (message == nullptr) return false;
    return std::strstr(message, "out of memory") != nullptr
        || std::strstr(message, "cudaMalloc") != nullptr
        || std::strstr(message, "vkAllocate") != nullptr
        || std::strstr(message, "Socket closed") != nullptr
        || std::strstr(message, "Socket offline") != nullptr
        || std::strstr(message, "Error reading from socket") != nullptr
        || std::strstr(message, "Error writing to socket") != nullptr;
}

inline unsigned failedPackedNode(const char *message, const std::vector<unsigned> &counts) {
    const bool local = message != nullptr && (
        std::strstr(message, "out of memory") != nullptr
        || std::strstr(message, "cudaMalloc") != nullptr
        || std::strstr(message, "vkAllocate") != nullptr);
    if (local || counts.size() < 2u) return 0u;
    unsigned node = 1u;
    for (unsigned i = 2u; i < counts.size(); ++i) {
        if (counts[i] > counts[node]) node = i;
    }
    return node;
}

bool localKnownSpeedProfile(SpeedDeviceProfile *out);
unsigned long long profileDeviceFreeBytes();
void publishProfileRuntime(int backend, int gpuIndex, unsigned nThreads);
void cacheSpeedProfile(const std::string &key, const SpeedDeviceProfile &profile);
bool cachedSpeedProfile(const std::string &key, SpeedDeviceProfile *out);
SpeedDeviceProfile measureLocalProfile(const ModelShape &shape);

// One pipeline boundary. leftHolds is how many of the right stage's layers
// the left stage stores. rightHolds is the opposite direction.
struct BoundaryOverlap {
    unsigned leftHolds = 0u;
    unsigned rightHolds = 0u;
};

// After the primary split, each stage may keep extra layers in the memory
// cap that estimateLayerCap already computed (70% of free memory, one layer
// of headroom). A stage with two neighbors spends that spare on both
// boundaries. A stage is never emptied, so a neighbor with N primary layers
// can donate at most N-1.
inline std::vector<BoundaryOverlap> assignBoundaryOverlap(
    const std::vector<unsigned> &primary,
    const std::vector<unsigned> &memoryCap) {
    const unsigned n = (unsigned)std::min(primary.size(), memoryCap.size());
    std::vector<unsigned> spare(n, 0u);
    for (unsigned i = 0; i < n; ++i) {
        spare[i] = memoryCap[i] > primary[i] ? memoryCap[i] - primary[i] : 0u;
    }
    std::vector<BoundaryOverlap> overlaps(n < 2u ? 0u : n - 1u);
    bool grew = true;
    while (grew) {
        grew = false;
        for (unsigned boundary = 0u; boundary + 1u < n; ++boundary) {
            const unsigned left = boundary;
            const unsigned right = boundary + 1u;
            const unsigned maxLeftHolds = primary[right] > 0u ? primary[right] - 1u : 0u;
            const unsigned maxRightHolds = primary[left] > 0u ? primary[left] - 1u : 0u;
            if (spare[left] > 0u && overlaps[boundary].leftHolds < maxLeftHolds) {
                overlaps[boundary].leftHolds += 1u;
                spare[left] -= 1u;
                grew = true;
            }
            if (spare[right] > 0u && overlaps[boundary].rightHolds < maxRightHolds) {
                overlaps[boundary].rightHolds += 1u;
                spare[right] -= 1u;
                grew = true;
            }
        }
    }
    return overlaps;
}

inline std::string formatBoundaryOverlap(const std::vector<BoundaryOverlap> &overlaps) {
    std::ostringstream out;
    for (size_t i = 0u; i < overlaps.size(); ++i) {
        if (i > 0u) out << ';';
        out << overlaps[i].leftHolds << ',' << overlaps[i].rightHolds;
    }
    return out.str();
}

#endif
