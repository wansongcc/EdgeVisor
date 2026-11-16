#ifndef SUSTAINED_PROFILE_HPP
#define SUSTAINED_PROFILE_HPP

#include "device_profile.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

// Worker handshake. The worker answers this probe and then listens again.
static const unsigned kSustainedProfileMagic = 0x50524F48u;

// Starting points fixed before a run, not fitted to one. Each pair is the
// normal point and the strong-risk point of one indicator. A device with no
// evidence stays mid. A link that was not measured stays weight 0 and cannot
// pull a device weight down. Mean speed and mean delay are recorded only.
static const double kSustainedLateEarlyMid = 1.10;
static const double kSustainedLateEarlyHigh = 1.25;
static const double kSustainedJitterRatioMid = 0.20;
static const double kSustainedJitterRatioHigh = 0.50;
static const double kSustainedRiseMidCPerSec = 0.3;
static const double kSustainedRiseHighCPerSec = 1.0;
static const double kSustainedLinkJitterMidMs = 5.0;
static const double kSustainedLinkJitterHighMs = 20.0;
// Risk uses the same edges as a normal point and a strong-risk point. Mean
// speed and mean delay stay out of the score: they already describe capability.
static const double kSustainedHeadroomNormalC = 30.0;
static const double kSustainedHeadroomStrongC = 5.0;
static const double kSustainedBatteryNormalPercent = 80.0;
static const double kSustainedBatteryStrongPercent = 20.0;
static const double kSustainedBatteryDischargeRisk = 0.5;
static const double kSustainedLinkTimeoutStrong = 0.05;
static const double kSustainedLinkRetransStrong = 0.05;
// Presence weight for a readable indicator. It does not rescale the split
// while every present indicator shares it. Missing indicators use 0.
static const double kRiskCredibilityQ0 = 0.7;
static const unsigned kProtectionWeightLow = 2000u;
static const unsigned kProtectionWeightSpan = 2000u;
static const unsigned kProtectionWeightUnknown = 3000u;
static const unsigned kSustainedDefaultObserveMs = 25000u;
static const unsigned kSustainedDefaultWarmupRounds = 3u;
static const unsigned kSustainedDefaultTimeoutMs = 500u;

struct ComputeRound {
    double timestampMs = 0.0;
    double elapsedMs = 0.0;
    unsigned layerCount = 0u;
};

struct TransferRound {
    double timestampMs = 0.0;
    double elapsedMs = 0.0;
    unsigned bytes = 0u;
    int timedOut = 0;       // 1 when the transfer did not finish in time
    int retransDelta = -1;  // -1 unknown
};

struct ComputeSummary {
    unsigned rounds = 0u;
    unsigned layersPerRound = 0u;
    double meanLayerMs = 0.0;
    bool meanKnown = false;
    double jitterLayerMs = 0.0;
    bool jitterKnown = false;
    double lateEarlyRatio = 0.0;
    bool ratioKnown = false;
};

struct ThermalSummary {
    int maxTempC = -1;
    int tempChangeC = 0;
    double riseCPerSec = 0.0;
    bool riseKnown = false;
    double throttleFraction = 0.0;
    bool throttleKnown = false;
    int headroomC = -1;
    int freqMinMhz = -1;
    int freqMaxMhz = -1;
};

static const int kPowerUnknown = -1;
static const int kPowerSteadyExternal = 0;
static const int kPowerSteadyBattery = 1;
static const int kPowerExternalToBattery = 2;
static const int kPowerBatteryToExternal = 3;
static const int kPowerBothDirections = 4;

struct PowerSummary {
    int onExternalPower = -1; // last known sample
    int batteryPercent = -1;  // last known sample
    int changed = -1;         // -1 unknown, 0 no known-state change, 1 a known state changed
    int direction = kPowerUnknown;
};

struct DeviceSummary {
    unsigned stage = 0u;
    std::string host;
    ComputeSummary compute;
    ThermalSummary thermal;
    PowerSummary power;
    unsigned priority = 0u;        // coarse: 0 unknown, 2 low, 3 mid, 4 high
    unsigned appliedPriority = 3u; // after an optional low/high swap
    double computeRisk = -1.0;     // -1 when that group was not measured
    double powerRisk = -1.0;
    double protectionScore = -1.0; // strongest present group; 0 means no read indicator crossed its normal point
    double credibility = 0.0;
    unsigned appliedWeight = kProtectionWeightUnknown;
    bool scoreKnown = false;
    std::vector<std::string> reasons;
    std::vector<std::string> missing;
};

struct LinkSummary {
    unsigned fromStage = 0u;
    unsigned toStage = 0u;
    std::string fromHost;
    std::string toHost;
    std::string address;
    unsigned bytes = 0u;
    unsigned transfers = 0u;
    double meanMs = 0.0;
    bool meanKnown = false;
    double jitterMs = 0.0;
    bool jitterKnown = false;
    double timeoutFraction = 0.0;
    bool timeoutKnown = false;
    unsigned timeouts = 0u;
    unsigned finishedTransfers = 0u; // jitter uses only these successful attempts
    int retrans = -1;                // window tcpi_total_retrans delta, -1 unknown
    long long retransSegs = -1;      // same delta; -1 unknown, never a guessed packet count
    long long dataSegsOut = -1;      // window tcpi_data_segs_out delta, -1 unknown
    bool segmentRateKnown = false;
    double segmentRetransRate = 0.0; // retransSegs / dataSegsOut; undefined when the denominator is 0
    unsigned retransRounds = 0u;     // counted attempts whose per-attempt retrans delta was > 0
    unsigned retransRoundsKnown = 0u;
    bool affectedRoundRateKnown = false;
    double affectedRoundRate = 0.0;  // diagnostic only: retransRounds / retransRoundsKnown
    unsigned rebuilds = 0u;
    double firstTransferMs = -1.0;
    double lastTransferMs = -1.0;
    unsigned priority = 0u;
    double risk = -1.0;
    unsigned appliedWeight = 0u;
    std::vector<std::string> missing;
};

struct SustainedProfileReport {
    double observeSeconds = 0.0;
    unsigned warmupRounds = 0u;
    std::string pace;
    std::string note;
    bool reversed = false;
    std::vector<DeviceSummary> devices;
    std::vector<LinkSummary> links;
};

struct ComputeRawRow {
    double timestampMs = 0.0;
    unsigned stage = 0u;
    unsigned round = 0u;
    unsigned layerCount = 0u;
    double elapsedMs = 0.0;
};

struct HardwareRawRow {
    unsigned stage = 0u;
    HardwareTick tick;
};

struct LinkRawRow {
    unsigned fromStage = 0u;
    unsigned toStage = 0u;
    std::string address;
    TransferRound transfer;
};

inline const char *protectionPriorityName(unsigned weight) {
    if (weight == 2u) return "low";
    if (weight == 3u) return "mid";
    if (weight == 4u) return "high";
    return "unknown";
}

inline const char *powerStateName(int onExternalPower) {
    if (onExternalPower == 0) return "battery";
    if (onExternalPower == 1) return "external";
    return "unknown";
}

inline const char *powerDirectionName(int direction) {
    if (direction == kPowerSteadyExternal) return "steady_external";
    if (direction == kPowerSteadyBattery) return "steady_battery";
    if (direction == kPowerExternalToBattery) return "external_to_battery";
    if (direction == kPowerBatteryToExternal) return "battery_to_external";
    if (direction == kPowerBothDirections) return "both_directions";
    return "unknown";
}

// Larger is worse. x <= a is no sign, x >= b is a strong sign.
inline double risingRisk(double x, double a, double b) {
    if (!(b > a)) return 0.0;
    if (x <= a) return 0.0;
    if (x >= b) return 1.0;
    return (x - a) / (b - a);
}

// Smaller is worse. x >= a is no sign, x <= b is a strong sign.
inline double fallingRisk(double x, double a, double b) {
    if (!(a > b)) return 0.0;
    if (x >= a) return 0.0;
    if (x <= b) return 1.0;
    return (a - x) / (a - b);
}

inline unsigned tierFromRisk(double risk, bool known) {
    if (!known) return 0u;
    if (risk >= 1.0) return 4u;
    if (risk > 0.0) return 3u;
    return 2u;
}

// Measured r=0 stays at the low weight. Weight 0 is reserved for a side that
// was not measured and must not take the whole budget.
inline unsigned protectionWeightFromRisk(double risk, bool known) {
    if (!known) return kProtectionWeightUnknown;
    if (risk < 0.0) risk = 0.0;
    if (risk > 1.0) risk = 1.0;
    return kProtectionWeightLow + (unsigned)(kProtectionWeightSpan * risk + 0.5);
}

inline void takeStrongest(
    double *best,
    bool *known,
    std::vector<std::string> *reasons,
    const char *name,
    double risk,
    bool present) {
    if (!present) return;
    if (risk < 0.0) risk = 0.0;
    if (risk > 1.0) risk = 1.0;
    if (!*known || risk > *best) {
        *best = risk;
        *known = true;
        if (reasons != nullptr) {
            reasons->clear();
            if (risk > 0.0) reasons->push_back(name);
        }
        return;
    }
    if (reasons != nullptr && risk > 0.0 && risk == *best) reasons->push_back(name);
}

inline double computeRiskIntensity(const ComputeSummary &summary, bool *known, std::vector<std::string> *reasons) {
    double best = 0.0;
    bool present = false;
    if (summary.ratioKnown)
        takeStrongest(&best, &present, reasons, "late_early",
            risingRisk(summary.lateEarlyRatio, kSustainedLateEarlyMid, kSustainedLateEarlyHigh), true);
    if (summary.meanKnown && summary.jitterKnown && summary.meanLayerMs > 0.0)
        takeStrongest(&best, &present, reasons, "compute_jitter",
            risingRisk(summary.jitterLayerMs / summary.meanLayerMs, kSustainedJitterRatioMid, kSustainedJitterRatioHigh), true);
    if (known != nullptr) *known = present;
    return present ? best : -1.0;
}

inline double thermalRiskIntensity(const ThermalSummary &summary, bool *known, std::vector<std::string> *reasons) {
    double best = 0.0;
    bool present = false;
    const bool riseUsable = summary.riseKnown && (summary.tempChangeC >= 2 || summary.tempChangeC <= -2);
    if (riseUsable)
        takeStrongest(&best, &present, reasons, "temp_rise",
            risingRisk(summary.riseCPerSec, kSustainedRiseMidCPerSec, kSustainedRiseHighCPerSec), true);
    if (summary.headroomC >= 0)
        takeStrongest(&best, &present, reasons, "thermal_headroom",
            fallingRisk((double)summary.headroomC, kSustainedHeadroomNormalC, kSustainedHeadroomStrongC), true);
    if (summary.throttleKnown)
        takeStrongest(&best, &present, reasons, "throttle", summary.throttleFraction > 0.0 ? 1.0 : 0.0, true);
    if (known != nullptr) *known = present;
    return present ? best : -1.0;
}

inline double powerRiskIntensity(const PowerSummary &summary, bool *known, std::vector<std::string> *reasons) {
    double best = 0.0;
    bool present = false;
    // An unknown reading is not a transition. Plugging in is not a strong risk.
    // Losing external power during the window is.
    const bool lostExternal = summary.direction == kPowerExternalToBattery
        || summary.direction == kPowerBothDirections;
    if (lostExternal)
        takeStrongest(&best, &present, reasons, "external_to_battery", 1.0, true);
    if (summary.onExternalPower == 0) {
        takeStrongest(&best, &present, reasons, "battery_discharge", kSustainedBatteryDischargeRisk, true);
        if (summary.batteryPercent >= 0)
            takeStrongest(&best, &present, reasons, "battery_capacity",
                fallingRisk((double)summary.batteryPercent, kSustainedBatteryNormalPercent, kSustainedBatteryStrongPercent), true);
    } else if (summary.onExternalPower == 1 && !lostExternal) {
        takeStrongest(&best, &present, reasons, "external_power", 0.0, true);
    }
    if (known != nullptr) *known = present;
    return present ? best : -1.0;
}

inline double linkRiskIntensity(const LinkSummary &summary, bool *known, std::vector<std::string> *reasons) {
    double best = 0.0;
    bool present = false;
    if (summary.transfers == 0u) {
        if (known != nullptr) *known = false;
        return -1.0;
    }
    if (summary.jitterKnown)
        takeStrongest(&best, &present, reasons, "link_jitter",
            risingRisk(summary.jitterMs, kSustainedLinkJitterMidMs, kSustainedLinkJitterHighMs), true);
    if (summary.timeoutKnown)
        takeStrongest(&best, &present, reasons, "link_timeout",
            risingRisk(summary.timeoutFraction, 0.0, kSustainedLinkTimeoutStrong), true);
    if (summary.segmentRateKnown)
        takeStrongest(&best, &present, reasons, "link_retrans_segs",
            risingRisk(summary.segmentRetransRate, 0.0, kSustainedLinkRetransStrong), true);
    if (known != nullptr) *known = present;
    return present ? best : -1.0;
}

// Mean layer time is recorded and is not itself a priority. Speed-pack already
// used that number. What raises protection is the run getting worse, or a
// large spread, across the window.
inline unsigned computeStabilityPriority(const ComputeSummary &summary) {
    bool known = false;
    const double risk = computeRiskIntensity(summary, &known, nullptr);
    return tierFromRisk(risk, known);
}

inline unsigned thermalWindowPriority(const ThermalSummary &summary) {
    bool known = false;
    const double risk = thermalRiskIntensity(summary, &known, nullptr);
    return tierFromRisk(risk, known);
}

inline unsigned powerWindowPriority(const PowerSummary &summary) {
    bool known = false;
    const double risk = powerRiskIntensity(summary, &known, nullptr);
    return tierFromRisk(risk, known);
}

inline unsigned linkWindowPriority(const LinkSummary &summary) {
    bool known = false;
    const double risk = linkRiskIntensity(summary, &known, nullptr);
    return tierFromRisk(risk, known);
}

inline ComputeSummary summarizeComputeRounds(const std::vector<ComputeRound> &rounds) {
    ComputeSummary summary;
    std::vector<double> perLayer;
    unsigned layers = 0u;
    bool uniform = true;
    for (std::size_t i = 0u; i < rounds.size(); ++i) {
        if (rounds[i].layerCount == 0u || !(rounds[i].elapsedMs >= 0.0)) continue;
        perLayer.push_back(rounds[i].elapsedMs / (double)rounds[i].layerCount);
        if (layers == 0u) layers = rounds[i].layerCount;
        else if (rounds[i].layerCount != layers) uniform = false;
    }
    summary.rounds = (unsigned)perLayer.size();
    if (uniform && layers > 0u) summary.layersPerRound = layers;
    if (perLayer.empty()) return summary;
    double sum = 0.0;
    for (std::size_t i = 0u; i < perLayer.size(); ++i) sum += perLayer[i];
    const double mean = sum / (double)perLayer.size();
    double deviation = 0.0;
    for (std::size_t i = 0u; i < perLayer.size(); ++i)
        deviation += std::fabs(perLayer[i] - mean);
    summary.meanLayerMs = mean;
    summary.meanKnown = true;
    summary.jitterLayerMs = deviation / (double)perLayer.size();
    summary.jitterKnown = true;
    const std::size_t half = perLayer.size() / 2u;
    if (half >= 2u && perLayer.size() - half >= 2u) {
        double early = 0.0;
        double late = 0.0;
        for (std::size_t i = 0u; i < half; ++i) early += perLayer[i];
        for (std::size_t i = half; i < perLayer.size(); ++i) late += perLayer[i];
        early /= (double)half;
        late /= (double)(perLayer.size() - half);
        if (early > 0.0) {
            summary.lateEarlyRatio = late / early;
            summary.ratioKnown = true;
        }
    }
    return summary;
}

inline ThermalSummary summarizeThermalTicks(const std::vector<HardwareTick> &ticks) {
    ThermalSummary summary;
    int firstTemp = -1;
    int lastTemp = -1;
    double firstMs = 0.0;
    double lastMs = 0.0;
    unsigned knownLimit = 0u;
    unsigned activeLimit = 0u;
    for (std::size_t i = 0u; i < ticks.size(); ++i) {
        const HardwareTick &tick = ticks[i];
        if (tick.temperatureC >= 0) {
            if (summary.maxTempC < 0 || tick.temperatureC > summary.maxTempC)
                summary.maxTempC = tick.temperatureC;
            if (firstTemp < 0) {
                firstTemp = tick.temperatureC;
                firstMs = tick.timestampMs;
            }
            lastTemp = tick.temperatureC;
            lastMs = tick.timestampMs;
        }
        if (tick.thermalLimited >= 0) {
            knownLimit += 1u;
            if (tick.thermalLimited == 1) activeLimit += 1u;
        }
        if (tick.freqMhz >= 0 && (summary.freqMinMhz < 0 || tick.freqMhz < summary.freqMinMhz))
            summary.freqMinMhz = tick.freqMhz;
        if (tick.freqMaxMhz >= 0 && tick.freqMaxMhz > summary.freqMaxMhz)
            summary.freqMaxMhz = tick.freqMaxMhz;
        if (tick.thermalHeadroomC >= 0
            && (summary.headroomC < 0 || tick.thermalHeadroomC < summary.headroomC))
            summary.headroomC = tick.thermalHeadroomC;
    }
    if (firstTemp >= 0 && lastMs > firstMs + 500.0) {
        summary.tempChangeC = lastTemp - firstTemp;
        summary.riseCPerSec = (double)summary.tempChangeC / ((lastMs - firstMs) / 1000.0);
        summary.riseKnown = true;
    }
    if (knownLimit > 0u) {
        summary.throttleFraction = (double)activeLimit / (double)knownLimit;
        summary.throttleKnown = true;
    }
    return summary;
}

inline PowerSummary summarizePowerTicks(const std::vector<HardwareTick> &ticks) {
    PowerSummary summary;
    int previous = -1;
    int toBattery = 0;
    int toExternal = 0;
    int known = 0;
    for (std::size_t i = 0u; i < ticks.size(); ++i) {
        const HardwareTick &tick = ticks[i];
        if (tick.batteryPercent >= 0) summary.batteryPercent = tick.batteryPercent;
        if (tick.onExternalPower < 0) continue;
        known = 1;
        if (previous == 1 && tick.onExternalPower == 0) toBattery = 1;
        if (previous == 0 && tick.onExternalPower == 1) toExternal = 1;
        previous = tick.onExternalPower;
        summary.onExternalPower = tick.onExternalPower;
    }
    if (!known) {
        summary.direction = kPowerUnknown;
        summary.changed = -1;
        return summary;
    }
    if (toBattery && toExternal) summary.direction = kPowerBothDirections;
    else if (toBattery) summary.direction = kPowerExternalToBattery;
    else if (toExternal) summary.direction = kPowerBatteryToExternal;
    else if (summary.onExternalPower == 0) summary.direction = kPowerSteadyBattery;
    else summary.direction = kPowerSteadyExternal;
    summary.changed = (toBattery || toExternal) ? 1 : 0;
    return summary;
}

// Sender-side TCP counters for one counted window. Warmup is excluded by
// starting the snapshot at the first counted send. A rebuilt connection is
// committed on its own and then added; an unreadable or backwards counter
// makes the window total unknown instead of a partial or zero rate.
struct LinkCounterWindow {
    bool broken = false;
    unsigned committed = 0u;
    uint64_t retransSegs = 0u;
    uint64_t dataSegsOut = 0u;
    unsigned rebuilds = 0u;
    bool open = false;
    bool startKnown = false;
    bool latestKnown = false;
    uint32_t startRetrans = 0u;
    uint32_t startSegs = 0u;
    uint32_t latestRetrans = 0u;
    uint32_t latestSegs = 0u;

    void beginConnection(bool known, uint32_t retrans, uint32_t segs) {
        open = true;
        startKnown = known;
        latestKnown = known;
        startRetrans = retrans;
        startSegs = segs;
        latestRetrans = retrans;
        latestSegs = segs;
    }

    void observe(bool known, uint32_t retrans, uint32_t segs) {
        latestKnown = known;
        if (known) {
            latestRetrans = retrans;
            latestSegs = segs;
        }
    }

    void endConnection() {
        if (!open) return;
        open = false;
        if (!startKnown || !latestKnown || latestRetrans < startRetrans || latestSegs < startSegs) {
            broken = true;
            return;
        }
        if (broken) return;
        retransSegs += (uint64_t)latestRetrans - (uint64_t)startRetrans;
        dataSegsOut += (uint64_t)latestSegs - (uint64_t)startSegs;
        committed += 1u;
    }
};

inline void noteCountedSend(
    LinkCounterWindow *window,
    bool newConnection,
    bool beforeKnown,
    uint32_t beforeRetrans,
    uint32_t beforeSegs,
    bool afterKnown,
    uint32_t afterRetrans,
    uint32_t afterSegs) {
    if (window == nullptr) return;
    if (newConnection) {
        if (window->open || window->committed > 0u) window->rebuilds += 1u;
        if (window->open) window->endConnection();
        window->beginConnection(beforeKnown, beforeRetrans, beforeSegs);
    } else if (!window->open) {
        window->beginConnection(beforeKnown, beforeRetrans, beforeSegs);
    }
    window->observe(afterKnown, afterRetrans, afterSegs);
}

inline LinkSummary summarizeTransfers(const std::vector<TransferRound> &rounds) {
    LinkSummary summary;
    summary.transfers = (unsigned)rounds.size();
    unsigned timeouts = 0u;
    std::vector<double> finished;
    for (std::size_t i = 0u; i < rounds.size(); ++i) {
        if (rounds[i].bytes > summary.bytes) summary.bytes = rounds[i].bytes;
        if (rounds[i].timedOut) timeouts += 1u;
        else if (rounds[i].elapsedMs >= 0.0) finished.push_back(rounds[i].elapsedMs);
        if (rounds[i].retransDelta >= 0) {
            summary.retransRoundsKnown += 1u;
            if (rounds[i].retransDelta > 0) summary.retransRounds += 1u;
        }
    }
    summary.timeouts = timeouts;
    if (!rounds.empty()) {
        summary.timeoutFraction = (double)timeouts / (double)rounds.size();
        summary.timeoutKnown = true;
        summary.firstTransferMs = rounds.front().timestampMs;
        summary.lastTransferMs = rounds.back().timestampMs;
    }
    if (summary.retransRoundsKnown > 0u) {
        summary.affectedRoundRate = (double)summary.retransRounds / (double)summary.retransRoundsKnown;
        summary.affectedRoundRateKnown = true;
    }
    if (!finished.empty()) {
        double sum = 0.0;
        for (std::size_t i = 0u; i < finished.size(); ++i) sum += finished[i];
        const double mean = sum / (double)finished.size();
        double deviation = 0.0;
        for (std::size_t i = 0u; i < finished.size(); ++i)
            deviation += std::fabs(finished[i] - mean);
        summary.meanMs = mean;
        summary.meanKnown = true;
        summary.jitterMs = deviation / (double)finished.size();
        summary.jitterKnown = true;
    }
    summary.finishedTransfers = (unsigned)finished.size();
    bool known = false;
    summary.risk = linkRiskIntensity(summary, &known, nullptr);
    summary.priority = tierFromRisk(summary.risk, known);
    summary.appliedWeight = known ? protectionWeightFromRisk(summary.risk, true) : 0u;
    return summary;
}

inline void applyLinkCounterWindow(LinkSummary *link, const LinkCounterWindow &window) {
    if (link == nullptr) return;
    link->rebuilds = window.rebuilds;
    link->segmentRateKnown = false;
    link->retrans = -1;
    link->retransSegs = -1;
    link->dataSegsOut = -1;
    if (!window.broken && window.committed > 0u) {
        link->retransSegs = (long long)window.retransSegs;
        link->dataSegsOut = (long long)window.dataSegsOut;
        if (window.retransSegs <= 2147483647ull) link->retrans = (int)window.retransSegs;
        if (window.dataSegsOut > 0u) {
            link->segmentRetransRate = (double)window.retransSegs / (double)window.dataSegsOut;
            link->segmentRateKnown = true;
        }
    }
    bool known = false;
    link->risk = linkRiskIntensity(*link, &known, nullptr);
    link->priority = tierFromRisk(link->risk, known);
    link->appliedWeight = known ? protectionWeightFromRisk(link->risk, true) : 0u;
}

inline void takeGroup(
    double *best,
    bool *known,
    std::vector<std::string> *reasons,
    double risk,
    bool present,
    const std::vector<std::string> &groupReasons) {
    if (!present) return;
    if (!*known || risk > *best) {
        *best = risk;
        *known = true;
        if (reasons != nullptr) *reasons = groupReasons;
        return;
    }
    if (reasons != nullptr && risk == *best) {
        for (std::size_t i = 0u; i < groupReasons.size(); ++i)
            reasons->push_back(groupReasons[i]);
    }
}

inline void fillDeviceMissing(DeviceSummary *device) {
    if (device == nullptr) return;
    device->missing.clear();
    if (!device->compute.meanKnown) device->missing.push_back("compute.meanLayerMs");
    if (!device->compute.jitterKnown) device->missing.push_back("compute.jitterLayerMs");
    if (!device->compute.ratioKnown) device->missing.push_back("compute.lateEarlyRatio");
    if (device->thermal.maxTempC < 0) device->missing.push_back("thermal.maxTempC");
    if (!device->thermal.riseKnown) device->missing.push_back("thermal.riseCPerSec");
    if (!device->thermal.throttleKnown) device->missing.push_back("thermal.throttleFraction");
    if (device->thermal.headroomC < 0) device->missing.push_back("thermal.headroomC");
    if (device->thermal.freqMinMhz < 0) device->missing.push_back("thermal.freqMhz");
    if (device->power.onExternalPower < 0) device->missing.push_back("power.state");
    if (device->power.batteryPercent < 0) device->missing.push_back("power.batteryPercent");
}

inline DeviceSummary summarizeDevice(
    unsigned stage,
    const std::string &host,
    const std::vector<ComputeRound> &rounds,
    const std::vector<HardwareTick> &ticks) {
    DeviceSummary device;
    device.stage = stage;
    device.host = host.empty() ? std::string("local") : host;
    device.compute = summarizeComputeRounds(rounds);
    device.thermal = summarizeThermalTicks(ticks);
    device.power = summarizePowerTicks(ticks);
    bool computeKnown = false;
    bool thermalKnown = false;
    bool powerKnown = false;
    std::vector<std::string> computeReasons;
    std::vector<std::string> thermalReasons;
    std::vector<std::string> powerReasons;
    const double timingRisk = computeRiskIntensity(device.compute, &computeKnown, &computeReasons);
    const double thermalRisk = thermalRiskIntensity(device.thermal, &thermalKnown, &thermalReasons);
    const double supplyRisk = powerRiskIntensity(device.power, &powerKnown, &powerReasons);
    double computeGroup = 0.0;
    bool computeGroupKnown = false;
    std::vector<std::string> computeGroupReasons;
    takeGroup(&computeGroup, &computeGroupKnown, &computeGroupReasons, timingRisk, computeKnown, computeReasons);
    takeGroup(&computeGroup, &computeGroupKnown, &computeGroupReasons, thermalRisk, thermalKnown, thermalReasons);
    device.computeRisk = computeGroupKnown ? computeGroup : -1.0;
    device.powerRisk = powerKnown ? supplyRisk : -1.0;
    double score = 0.0;
    takeGroup(&score, &device.scoreKnown, &device.reasons, computeGroup, computeGroupKnown, computeGroupReasons);
    takeGroup(&score, &device.scoreKnown, &device.reasons, supplyRisk, powerKnown, powerReasons);
    device.protectionScore = device.scoreKnown ? score : -1.0;
    device.credibility = device.scoreKnown ? kRiskCredibilityQ0 : 0.0;
    device.appliedWeight = protectionWeightFromRisk(device.protectionScore, device.scoreKnown);
    device.priority = device.scoreKnown ? tierFromRisk(device.protectionScore, true) : 3u;
    device.appliedPriority = device.priority;
    fillDeviceMissing(&device);
    return device;
}

inline std::string formatComputeCsv(const std::vector<ComputeRawRow> &rows) {
    std::ostringstream out;
    out << "timestamp_ms,stage,round,layer_count,elapsed_ms\n";
    for (std::size_t i = 0u; i < rows.size(); ++i) {
        char line[160];
        std::snprintf(line, sizeof(line), "%.3f,%u,%u,%u,%.4f\n",
            rows[i].timestampMs, rows[i].stage, rows[i].round,
            rows[i].layerCount, rows[i].elapsedMs);
        out << line;
    }
    return out.str();
}

inline std::string formatHardwareCsv(const std::vector<HardwareRawRow> &rows) {
    std::ostringstream out;
    out << "timestamp_ms,stage,temp_c,freq_mhz,freq_max_mhz,thermal_limited,power,battery_pct,headroom_c\n";
    for (std::size_t i = 0u; i < rows.size(); ++i) {
        const HardwareTick &tick = rows[i].tick;
        char line[192];
        std::snprintf(line, sizeof(line), "%.3f,%u,%d,%d,%d,%d,%d,%d,%d\n",
            tick.timestampMs, rows[i].stage, tick.temperatureC, tick.freqMhz,
            tick.freqMaxMhz, tick.thermalLimited, tick.onExternalPower, tick.batteryPercent,
            tick.thermalHeadroomC);
        out << line;
    }
    return out.str();
}

inline std::string formatLinkCsv(const std::vector<LinkRawRow> &rows) {
    std::ostringstream out;
    out << "timestamp_ms,from_stage,to_stage,address,bytes,elapsed_ms,timeout,retrans_delta\n";
    for (std::size_t i = 0u; i < rows.size(); ++i) {
        char line[256];
        std::snprintf(line, sizeof(line), "%.3f,%u,%u,%s,%u,%.4f,%d,%d\n",
            rows[i].transfer.timestampMs, rows[i].fromStage, rows[i].toStage,
            rows[i].address.c_str(), rows[i].transfer.bytes, rows[i].transfer.elapsedMs,
            rows[i].transfer.timedOut, rows[i].transfer.retransDelta);
        out << line;
    }
    return out.str();
}

inline void jsonEscape(std::ostringstream &out, const std::string &value) {
    out << '"';
    for (std::size_t i = 0u; i < value.size(); ++i) {
        const unsigned char c = (unsigned char)value[i];
        if (c == '"' || c == '\\') out << '\\' << (char)c;
        else if (c < 0x20u) out << ' ';
        else out << (char)c;
    }
    out << '"';
}

inline void jsonDouble(std::ostringstream &out, double value, bool known) {
    if (!known) {
        out << "null";
        return;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.4f", value);
    out << buf;
}

inline void jsonIntOrNull(std::ostringstream &out, int value) {
    if (value < 0) out << "null";
    else out << value;
}

inline void jsonMissing(std::ostringstream &out, const std::vector<std::string> &missing) {
    out << '[';
    for (std::size_t i = 0u; i < missing.size(); ++i) {
        if (i > 0u) out << ',';
        jsonEscape(out, missing[i]);
    }
    out << ']';
}

inline std::string formatSustainedSummaryJson(const SustainedProfileReport &report) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"observeSeconds\": ";
    jsonDouble(out, report.observeSeconds, true);
    out << ",\n  \"warmupRounds\": " << report.warmupRounds;
    out << ",\n  \"pace\": ";
    jsonEscape(out, report.pace);
    out << ",\n  \"reversed\": " << (report.reversed ? "true" : "false");
    out << ",\n  \"note\": ";
    jsonEscape(out, report.note);
    out << ",\n  \"devices\": [\n";
    for (std::size_t i = 0u; i < report.devices.size(); ++i) {
        const DeviceSummary &device = report.devices[i];
        if (i > 0u) out << ",\n";
        out << "    {\n      \"stage\": " << device.stage << ",\n      \"host\": ";
        jsonEscape(out, device.host);
        out << ",\n      \"compute\": {\"rounds\": " << device.compute.rounds
            << ", \"layersPerRound\": " << device.compute.layersPerRound
            << ", \"meanLayerMs\": ";
        jsonDouble(out, device.compute.meanLayerMs, device.compute.meanKnown);
        out << ", \"jitterLayerMs\": ";
        jsonDouble(out, device.compute.jitterLayerMs, device.compute.jitterKnown);
        out << ", \"lateEarlyRatio\": ";
        jsonDouble(out, device.compute.lateEarlyRatio, device.compute.ratioKnown);
        out << "},\n      \"thermal\": {\"maxTempC\": ";
        jsonIntOrNull(out, device.thermal.maxTempC);
        out << ", \"tempChangeC\": ";
        if (!device.thermal.riseKnown) out << "null";
        else out << device.thermal.tempChangeC;
        out << ", \"riseCPerSec\": ";
        jsonDouble(out, device.thermal.riseCPerSec, device.thermal.riseKnown);
        out << ", \"throttleFraction\": ";
        jsonDouble(out, device.thermal.throttleFraction, device.thermal.throttleKnown);
        out << ", \"headroomC\": ";
        jsonIntOrNull(out, device.thermal.headroomC);
        out << ", \"freqMinMhz\": ";
        jsonIntOrNull(out, device.thermal.freqMinMhz);
        out << ", \"freqMaxMhz\": ";
        jsonIntOrNull(out, device.thermal.freqMaxMhz);
        out << "},\n      \"power\": {\"state\": ";
        jsonEscape(out, powerStateName(device.power.onExternalPower));
        out << ", \"batteryPercent\": ";
        jsonIntOrNull(out, device.power.batteryPercent);
        out << ", \"changed\": ";
        if (device.power.changed < 0) out << "null";
        else out << (device.power.changed ? "true" : "false");
        out << ", \"direction\": ";
        jsonEscape(out, powerDirectionName(device.power.direction));
        out << "},\n      \"computeRisk\": ";
        jsonDouble(out, device.computeRisk, device.computeRisk >= 0.0);
        out << ", \"powerRisk\": ";
        jsonDouble(out, device.powerRisk, device.powerRisk >= 0.0);
        out << ", \"protectionScore\": ";
        jsonDouble(out, device.protectionScore, device.scoreKnown);
        out << ", \"credibility\": ";
        jsonDouble(out, device.credibility, true);
        out << ", \"weight\": " << device.appliedWeight;
        out << ",\n      \"priority\": ";
        jsonEscape(out, protectionPriorityName(device.priority == 0u ? 0u : device.priority));
        out << ",\n      \"appliedPriority\": ";
        jsonEscape(out, protectionPriorityName(device.appliedPriority));
        out << ",\n      \"reasons\": ";
        jsonMissing(out, device.reasons);
        out << ",\n      \"missing\": ";
        jsonMissing(out, device.missing);
        out << "\n    }";
    }
    out << "\n  ],\n  \"links\": [\n";
    for (std::size_t i = 0u; i < report.links.size(); ++i) {
        const LinkSummary &link = report.links[i];
        if (i > 0u) out << ",\n";
        out << "    {\n      \"fromStage\": " << link.fromStage
            << ", \"toStage\": " << link.toStage << ",\n      \"fromHost\": ";
        jsonEscape(out, link.fromHost);
        out << ", \"toHost\": ";
        jsonEscape(out, link.toHost);
        out << ",\n      \"address\": ";
        jsonEscape(out, link.address);
        out << ", \"bytes\": " << link.bytes
            << ", \"transfers\": " << link.transfers
            << ",\n      \"meanMs\": ";
        jsonDouble(out, link.meanMs, link.meanKnown);
        out << ", \"jitterMs\": ";
        jsonDouble(out, link.jitterMs, link.jitterKnown);
        out << ", \"timeoutFraction\": ";
        jsonDouble(out, link.timeoutFraction, link.timeoutKnown);
        out << ", \"timeouts\": " << link.timeouts;
        out << ", \"finishedTransfers\": " << link.finishedTransfers;
        out << ", \"retransSegs\": ";
        if (link.retransSegs < 0) out << "null";
        else out << link.retransSegs;
        out << ", \"dataSegsOut\": ";
        if (link.dataSegsOut < 0) out << "null";
        else out << link.dataSegsOut;
        out << ", \"segmentRetransRate\": ";
        jsonDouble(out, link.segmentRetransRate, link.segmentRateKnown);
        out << ", \"retransRounds\": " << link.retransRounds;
        out << ", \"retransRoundsKnown\": " << link.retransRoundsKnown;
        out << ", \"affectedRoundRate\": ";
        jsonDouble(out, link.affectedRoundRate, link.affectedRoundRateKnown);
        out << ", \"rebuilds\": " << link.rebuilds;
        out << ", \"firstTransferMs\": ";
        jsonDouble(out, link.firstTransferMs, link.firstTransferMs >= 0.0);
        out << ", \"lastTransferMs\": ";
        jsonDouble(out, link.lastTransferMs, link.lastTransferMs >= 0.0);
        out << ",\n      \"risk\": ";
        jsonDouble(out, link.risk, link.risk >= 0.0);
        out << ", \"weight\": " << link.appliedWeight;
        out << ",\n      \"priority\": ";
        jsonEscape(out, protectionPriorityName(link.priority));
        out << ",\n      \"missing\": ";
        jsonMissing(out, link.missing);
        out << "\n    }";
    }
    out << "\n  ]\n}\n";
    return out.str();
}

struct SustainedEndpoint {
    unsigned stage = 0u;
    std::string host; // empty is this process
    int port = 0;
    unsigned primaryLayers = 0u;
    unsigned extraNodes = 0u;
};

struct SustainedRunConfig {
    double observeSeconds = 25.0;
    unsigned warmupRounds = kSustainedDefaultWarmupRounds;
    unsigned timeoutMs = kSustainedDefaultTimeoutMs;
    unsigned activationBytes = 0u;
    double stepMs = 0.0;
    bool reversed = false;
    std::string outputDir;
    ModelShape shape;
    std::vector<SustainedEndpoint> endpoints;
};

struct SustainedRunOutput {
    SustainedProfileReport report;
    std::vector<unsigned> devicePriority;
    std::vector<unsigned> linkPriority;
    bool wroteFiles = false;
    std::string outputDir;
};

// Runs the short window on this process and on each remote endpoint, writes
// the raw CSV and the summary JSON, and returns the priorities algorithm 1
// applies. A remote that does not answer stays unknown and becomes mid.
SustainedRunOutput runSustainedShadowProfile(const SustainedRunConfig &config);

// Worker side. listenFd is the inference accept socket; the upstream stage
// connects there for the activation-sized transfers. The worker then listens
// again. Exceptions stay inside this function.
void serveSustainedProfileProbe(int controlFd, int listenFd);

#endif
