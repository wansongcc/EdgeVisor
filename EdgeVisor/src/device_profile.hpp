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

// Test hook: pretend a device has this much free memory, whatever cudaMemGetInfo
// reports. Edge deployments are memory-constrained in ways a lab T4 is not, and
// the only honest way to study the shadow budget under that constraint is to
// vary it directly instead of hoping a bigger model happens to fit worse.
// 0 means "use the measured value".
inline unsigned long long &freeBytesOverride() {
    static unsigned long long value = 0ull;
    return value;
}

// Share of free memory the planner is allowed to spend. Defaults to 0.70, the
// same figure the caps have always used.
inline double &memoryBudgetFraction() {
    static double value = 0.70;
    return value;
}

inline unsigned long long effectiveFreeBytes(unsigned long long measured) {
    const unsigned long long overrideValue = freeBytesOverride();
    return overrideValue > 0ull ? overrideValue : measured;
}

inline unsigned capFromLayerBytes(unsigned long long freeBytes, double perLayer, unsigned nLayers) {
    if (perLayer < 1.0) return nLayers == 0u ? 8u : nLayers;
    const double fraction = memoryBudgetFraction();
    const double budget = (double)effectiveFreeBytes(freeBytes) * fraction;
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

// Distribute layers so the pipeline is balanced, not so the fastest device
// gets the most.
//
// A pipeline's token rate is set by its slowest stage. Stage time is
// layers * msPerLayer, so throughput is maximised by making those products
// equal, subject to each device's memory cap. Filling the fastest device first
// is right for replica serving, where a faster replica should absorb more
// requests, but in a pipeline it only moves the bottleneck: with four identical
// T4s it produced 25/1/1/1, which runs at the speed of a single card.
//
// Assignment is proportional: device i gets layers proportional to 1/ms_i, so
// layers_i * ms_i is the same everywhere. Devices whose share exceeds their cap
// are pinned at the cap and dropped from the pool, and the remaining layers are
// re-split among the rest. If every device hits its cap there is no balanced
// solution; the surplus goes to the slowest device, where it costs least.
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

    std::vector<unsigned> pool;
    pool.reserve(nNodes);
    for (unsigned i = 0; i < nNodes; ++i) {
        if (isLive(i)) pool.push_back(i);
    }
    if (pool.empty()) return std::string();

    // Every stage needs at least one layer or the chain breaks.
    unsigned left = nLayers;
    for (unsigned i : pool) {
        if (left == 0u) break;
        layers[i] = 1u;
        left -= 1u;
    }

    // Repeatedly split the remainder proportionally; pin and drop any device
    // whose proportional share no longer fits under its cap.
    while (left > 0u && !pool.empty()) {
        double sumInv = 0.0;
        for (unsigned i : pool) {
            const double ms = profiles[i].msPerLayer > 0.0 ? profiles[i].msPerLayer : 1.0;
            sumInv += 1.0 / ms;
        }
        if (sumInv <= 0.0) break;

        unsigned assigned = 0u;
        std::vector<unsigned> next;
        next.reserve(pool.size());
        for (unsigned i : pool) {
            const double ms = profiles[i].msPerLayer > 0.0 ? profiles[i].msPerLayer : 1.0;
            const double share = ((1.0 / ms) / sumInv) * (double)left;
            unsigned give = (unsigned)(share + 0.5);
            if (give == 0u) give = 1u;
            const unsigned room = profiles[i].cap > layers[i] ? profiles[i].cap - layers[i] : 0u;
            const unsigned take = std::min(give, room);
            layers[i] += take;
            assigned += take;
            // Under the cap and still had room: eligible for another round.
            if (layers[i] < profiles[i].cap) next.push_back(i);
        }
        if (assigned == 0u) break;
        left = left >= assigned ? left - assigned : 0u;
        if (left == 0u) break;
        if (next.size() == pool.size()) {
            // Nothing got pinned, so the split is already balanced; hand the
            // rounding remainder to the device with the smallest stage time.
            unsigned best = next[0];
            for (unsigned i : next) {
                const double ti = (double)(layers[i] + 1u) * profiles[i].msPerLayer;
                const double tb = (double)(layers[best] + 1u) * profiles[best].msPerLayer;
                if (ti < tb) best = i;
            }
            const unsigned room = profiles[best].cap > layers[best]
                ? profiles[best].cap - layers[best] : 0u;
            if (room == 0u) break;
            layers[best] += 1u;
            left -= 1u;
        }
        pool.swap(next);
    }

    // Unplaceable surplus. This should be rare: it only happens when every
    // device is at its cap and layers remain, which means the model does not
    // fit. Put the overflow on the slowest device and let the existing
    // allocation-failure path shrink it, rather than silently exceeding a cap
    // that was derived from free memory.
    if (left > 0u) {
        unsigned slowest = 0u;
        for (unsigned i = 0; i < nNodes; ++i) {
            if (isLive(i) && profiles[i].msPerLayer > profiles[slowest].msPerLayer) slowest = i;
        }
        layers[slowest] += left;
    }

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
        if (i == failedNode || counts[i] == 0u) continue;
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
    if (message != nullptr) {
        const char *peer = std::strstr(message, "peer=");
        if (peer != nullptr) {
            char *stop = nullptr;
            const unsigned long named = std::strtoul(peer + 5, &stop, 10);
            if (stop != peer + 5 && named < counts.size()) return (unsigned)named;
        }
    }
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

// How many extra layers fit in each device's pipeline bubble.
//
// A single-batch token walks the stages one after another, so device i is
// free for the whole time the other stages are computing. The startup probe
// already measured that compute as msPerLayer, and the primary split says how
// many layers each stage owns. idle_i / ms_i is therefore the shadow depth
// that finishes before the token moves on. A shadow pass skips the peer sync,
// so charging the full layer time is a little tighter than the real bubble.
// An unknown rate (ms <= 0) leaves the device uncapped rather than wiping its
// shadow on a missed probe.
inline std::vector<unsigned> bubbleShadowLayerCap(
    const std::vector<unsigned> &primary,
    const std::vector<double> &msPerLayer) {
    const unsigned n = (unsigned)std::min(primary.size(), msPerLayer.size());
    std::vector<unsigned> cap(primary.size(), 0xffffffffu);
    for (unsigned i = 0; i < n; ++i) {
        if (!(msPerLayer[i] > 0.0)) continue;
        double otherMs = 0.0;
        bool blocked = false;
        for (unsigned j = 0; j < n; ++j) {
            if (j == i || primary[j] == 0u) continue;
            if (!(msPerLayer[j] > 0.0)) {
                blocked = true;
                break;
            }
            otherMs += (double)primary[j] * msPerLayer[j];
        }
        if (blocked) continue;
        const double layers = otherMs / msPerLayer[i];
        if (!(layers > 0.0)) {
            cap[i] = 0u;
        } else if (layers >= (double)0xffffffffu) {
            cap[i] = 0xffffffffu;
        } else {
            cap[i] = (unsigned)(layers + 1e-9);
        }
    }
    return cap;
}

// Split one device's shadow budget between the two directions it can serve.
//
// budget is what this device can hold on top of its own primary layers.
// leftCap and rightCap are the most the left and right neighbor can take
// without being emptied, so nothing here can hand a neighbor more than
// primary-1. A zero cap or a zero weight means that side does not take part:
// the whole budget then goes to the other side, still under its cap. Two
// inactive sides leave both directions at zero.
//
// Both sides take their proportional floor first, computed with the largest
// remainder method so integer rounding is decided once instead of drifting
// over calls. The leftover goes one layer at a time to a side that is still
// under its cap and has the larger remainder; an exact tie goes left. A side
// that just took a layer drops its remainder to zero, so the next tie does not
// keep feeding the same side.
//
// The weights are a starting point for experiments, not a tuned result: they
// exist so a run can be biased without touching the caps.
inline void splitDeviceShadowBudget(
    unsigned budget,
    unsigned leftCap,
    unsigned rightCap,
    unsigned leftWeight,
    unsigned rightWeight,
    unsigned *xLeft,
    unsigned *xRight) {
    unsigned left = 0u;
    unsigned right = 0u;
    const bool leftActive = leftCap > 0u && leftWeight > 0u;
    const bool rightActive = rightCap > 0u && rightWeight > 0u;
    if (leftActive && rightActive) {
        const unsigned long long weightSum =
            (unsigned long long)leftWeight + (unsigned long long)rightWeight;
        const unsigned long long wanted = std::min<unsigned long long>(
            (unsigned long long)budget, (unsigned long long)leftCap + (unsigned long long)rightCap);
        const unsigned long long leftScaled = wanted * (unsigned long long)leftWeight;
        const unsigned long long rightScaled = wanted * (unsigned long long)rightWeight;
        left = (unsigned)std::min<unsigned long long>(leftScaled / weightSum, (unsigned long long)leftCap);
        right = (unsigned)std::min<unsigned long long>(rightScaled / weightSum, (unsigned long long)rightCap);
        unsigned long long leftRemainder = leftScaled % weightSum;
        unsigned long long rightRemainder = rightScaled % weightSum;
        unsigned long long leftover =
            wanted - (unsigned long long)left - (unsigned long long)right;
        while (leftover > 0ull) {
            const bool leftRoom = left < leftCap;
            const bool rightRoom = right < rightCap;
            if (!leftRoom && !rightRoom) break;
            bool takeLeft = leftRoom;
            if (leftRoom && rightRoom) takeLeft = leftRemainder >= rightRemainder;
            if (takeLeft) {
                left += 1u;
                leftRemainder = 0ull;
            } else {
                right += 1u;
                rightRemainder = 0ull;
            }
            leftover -= 1ull;
        }
    } else if (leftActive) {
        left = std::min(budget, leftCap);
    } else if (rightActive) {
        right = std::min(budget, rightCap);
    }
    if (xLeft != nullptr) *xLeft = left;
    if (xRight != nullptr) *xRight = right;
}

// What a device says about itself at startup, plus the round trip this side
// measured to it. A field keeps -1 (or zero probes) when nothing could be read,
// and unknown never counts as low risk.
struct DeviceCondition {
    int temperatureC = -1;     // -1 unknown
    int thermalLimited = -1;   // -1 unknown, 0 no, 1 yes
    int onExternalPower = -1;  // -1 unknown, 0 on battery, 1 on external power
    unsigned probesSent = 0u;
    unsigned probesLost = 0u;
    double rttMeanMs = -1.0;   // <0 unknown
    double rttJitterMs = -1.0; // <0 unknown. Jitter is the mean absolute deviation from the mean.
};

// The tier edges below are a fixed starting point chosen before the experiment
// ran; they are not fitted optimal parameters. Zero means "no evidence", and
// combineDeviceRiskWeight keeps that out of the low tier.
inline unsigned thermalRisk(const DeviceCondition &condition) {
    if (condition.thermalLimited == 1 || condition.temperatureC >= 83) return 4u;
    const bool temperatureKnown = condition.temperatureC >= 0;
    if (temperatureKnown && condition.thermalLimited == 0) {
        return condition.temperatureC >= 70 ? 3u : 2u;
    }
    return 0u;
}

// Loss is part of the tier even though a TCP probe cannot observe it: a caller
// that does know the loss can set probesLost and get the top tier.
inline unsigned networkRisk(const DeviceCondition &condition) {
    if (condition.probesSent == 0u || condition.rttMeanMs < 0.0) return 0u;
    const double jitter = condition.rttJitterMs < 0.0 ? 0.0 : condition.rttJitterMs;
    if (condition.probesLost == 0u && condition.rttMeanMs < 20.0 && jitter < 5.0) return 2u;
    if (condition.probesLost == 0u && condition.rttMeanMs < 80.0 && jitter < 20.0) return 3u;
    return 4u;
}

inline unsigned powerRisk(const DeviceCondition &condition) {
    if (condition.onExternalPower < 0) return 0u;
    return condition.onExternalPower == 0 ? 4u : 2u;
}

// The worst known tier wins, because a device only has to be bad in one way to
// slow the pipeline. With no evidence at all the device lands in the middle
// tier, since unknown is not the same as healthy.
inline unsigned combineDeviceRiskWeight(const DeviceCondition &condition) {
    const unsigned thermal = thermalRisk(condition);
    const unsigned network = networkRisk(condition);
    const unsigned power = powerRisk(condition);
    unsigned weight = thermal > network ? thermal : network;
    if (power > weight) weight = power;
    return weight > 0u ? weight : 3u;
}

// One `nvidia-smi --format=csv,noheader` line: the temperature followed by the
// hw thermal, sw thermal, and hw power brake slowdown reasons. Fewer than four
// fields is a malformed line and returns false. A field that does not parse
// leaves its own value unknown instead of failing the whole line.
inline bool parseNvidiaTemperatureCsv(const char *line, int *tempC, int *thermalLimited) {
    if (tempC != nullptr) *tempC = -1;
    if (thermalLimited != nullptr) *thermalLimited = -1;
    if (line == nullptr || tempC == nullptr || thermalLimited == nullptr) return false;
    std::vector<std::string> fields;
    std::string field;
    for (const char *cursor = line; ; ++cursor) {
        const bool atEnd = *cursor == '\0';
        if (!atEnd && *cursor != ',') {
            field.push_back(*cursor);
            continue;
        }
        const std::size_t first = field.find_first_not_of(" \t\r\n\v\f");
        const std::size_t last = field.find_last_not_of(" \t\r\n\v\f");
        fields.push_back(first == std::string::npos ? std::string() : field.substr(first, last - first + 1u));
        field.clear();
        if (atEnd) break;
    }
    if (fields.size() < 4u) return false;
    char *stop = nullptr;
    const long parsed = std::strtol(fields[0].c_str(), &stop, 10);
    if (stop != fields[0].c_str() && *stop == '\0') *tempC = (int)parsed;
    unsigned active = 0u;
    unsigned notActive = 0u;
    for (std::size_t i = 1u; i < 4u; ++i) {
        if (fields[i] == "Active") ++active;
        else if (fields[i] == "Not Active") ++notActive;
    }
    if (active > 0u) *thermalLimited = 1;
    else if (notActive == 3u) *thermalLimited = 0;
    return true;
}

// sysfs reports supplies one line at a time, so the caller does the scanning and
// this decides the single state. Discharging wins over an online main: a laptop
// on a weak charger is still spending battery.
inline int externalPowerFromSupplies(bool sawSupply, bool batteryDischarging, bool externalOnline) {
    if (!sawSupply) return -1;
    if (batteryDischarging) return 0;
    if (externalOnline) return 1;
    return -1;
}

// Fills the thermal and power fields from this machine. The network fields stay
// unknown: only the caller knows the peer it wants to measure.
DeviceCondition measureLocalCondition();

// Parses a shadow priority file: one stage per line, "<stage> <priority>
// [reason]". Blank lines and lines whose first non-space character is '#'
// are skipped, so config/shadow_priorities.example reads as written. The
// priority token is case-insensitive: low=2, mid and medium=3, high=4. A later
// line for the same stage wins, and a stage that never appears keeps mid, so
// an empty file means the equal split. On a bad stage index or an unknown
// token the reason lands in error and parsing stops; weights keep whatever was
// already written.
inline bool parseShadowPriorityText(
    const std::string &text,
    unsigned nStages,
    std::vector<unsigned> *weights,
    std::string *error) {
    if (weights == nullptr) return false;
    weights->assign(nStages, 3u);
    std::size_t pos = 0u;
    unsigned lineNumber = 0u;
    while (pos < text.size()) {
        const std::size_t newline = text.find('\n', pos);
        const std::size_t lineEnd = newline == std::string::npos ? text.size() : newline;
        std::string line = text.substr(pos, lineEnd - pos);
        pos = newline == std::string::npos ? text.size() : newline + 1u;
        ++lineNumber;
        const std::size_t first = line.find_first_not_of(" \t\r\v\f");
        if (first == std::string::npos) continue;
        if (line[first] == '#') continue;
        std::istringstream fields(line.substr(first));
        std::string indexToken;
        if (!(fields >> indexToken)) continue;
        char *stop = nullptr;
        const unsigned long index = std::strtoul(indexToken.c_str(), &stop, 10);
        if (stop == indexToken.c_str() || *stop != '\0') {
            if (error != nullptr) {
                *error = "line " + std::to_string(lineNumber) + ": stage '" + indexToken + "' is not a number";
            }
            return false;
        }
        if (index >= (unsigned long)nStages) {
            if (error != nullptr) {
                *error = "line " + std::to_string(lineNumber) + ": stage " + indexToken
                    + " is outside " + std::to_string(nStages) + " stages";
            }
            return false;
        }
        std::string priorityToken;
        if (!(fields >> priorityToken)) {
            if (error != nullptr) {
                *error = "line " + std::to_string(lineNumber) + ": stage " + indexToken + " has no priority";
            }
            return false;
        }
        std::string token = priorityToken;
        for (std::size_t i = 0u; i < token.size(); ++i) {
            if (token[i] >= 'A' && token[i] <= 'Z') token[i] = (char)(token[i] - 'A' + 'a');
        }
        unsigned weight = 0u;
        if (token == "low") weight = 2u;
        else if (token == "mid" || token == "medium") weight = 3u;
        else if (token == "high") weight = 4u;
        else {
            if (error != nullptr) {
                *error = "line " + std::to_string(lineNumber) + ": unknown priority '" + priorityToken + "'";
            }
            return false;
        }
        // A later line for the same stage wins.
        (*weights)[index] = weight;
    }
    return true;
}

// Low and high trade places. Reversed mode reads the same file as priority
// mode but asks the opposite question: whether the experiment holds up when
// the quiet neighbors get the shadow instead.
inline void reverseShadowPriorityWeights(std::vector<unsigned> *weights) {
    if (weights == nullptr) return;
    for (std::size_t i = 0u; i < weights->size(); ++i) {
        if ((*weights)[i] == 2u) (*weights)[i] = 4u;
        else if ((*weights)[i] == 4u) (*weights)[i] = 2u;
    }
}

// After the primary split, each stage may keep extra layers up to the tighter
// of two caps: leftover memory (70% of free memory, one layer of headroom)
// and the bubble depth from bubbleShadowLayerCap. A null bubble cap means
// "memory only", which is what the older callers asked for. A stage with two
// neighbors divides its spare between both boundaries with
// splitDeviceShadowBudget. A stage is never emptied, so a neighbor with N
// primary layers can take at most N-1 from this stage.
//
// priorityWeight scores the neighbor being protected, not the stage holding
// the copy: a high-priority neighbor pulls more shadow onto both of its peers.
// A null vector, or a vector shorter than the pipeline, means mid weights
// everywhere, which reproduces the equal split the older callers relied on.
inline std::vector<BoundaryOverlap> assignBoundaryOverlap(
    const std::vector<unsigned> &primary,
    const std::vector<unsigned> &memoryCap,
    const std::vector<unsigned> *bubbleCap = nullptr,
    const std::vector<unsigned> *priorityWeight = nullptr) {
    const unsigned n = (unsigned)std::min(primary.size(), memoryCap.size());
    std::vector<unsigned> spare(n, 0u);
    for (unsigned i = 0; i < n; ++i) {
        const unsigned memSpare = memoryCap[i] > primary[i] ? memoryCap[i] - primary[i] : 0u;
        unsigned limit = memSpare;
        if (bubbleCap != nullptr && i < bubbleCap->size())
            limit = std::min(limit, (*bubbleCap)[i]);
        spare[i] = limit;
    }
    std::vector<BoundaryOverlap> overlaps(n < 2u ? 0u : n - 1u);
    for (unsigned i = 0; i < n; ++i) {
        const unsigned leftCap = i > 0u && primary[i - 1u] > 0u ? primary[i - 1u] - 1u : 0u;
        const unsigned rightCap = i + 1u < n && primary[i + 1u] > 0u ? primary[i + 1u] - 1u : 0u;
        const unsigned leftWeight = i > 0u && priorityWeight != nullptr
            && i - 1u < priorityWeight->size() ? (*priorityWeight)[i - 1u] : 3u;
        const unsigned rightWeight = i + 1u < n && priorityWeight != nullptr
            && i + 1u < priorityWeight->size() ? (*priorityWeight)[i + 1u] : 3u;
        unsigned xLeft = 0u;
        unsigned xRight = 0u;
        splitDeviceShadowBudget(spare[i], leftCap, rightCap, leftWeight, rightWeight, &xLeft, &xRight);
        if (i > 0u) overlaps[i - 1u].rightHolds = xLeft;
        if (i + 1u < n) overlaps[i].leftHolds = xRight;
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
