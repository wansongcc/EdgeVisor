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

// Starting points fixed before a run, not fitted to one. A device with no
// evidence stays mid. A link that was not measured stays unknown (0) and
// cannot pull a device priority down.
static const double kSustainedLateEarlyMid = 1.10;
static const double kSustainedLateEarlyHigh = 1.25;
static const double kSustainedJitterRatioMid = 0.20;
static const double kSustainedJitterRatioHigh = 0.50;
static const int kSustainedTempMidC = 70;
static const int kSustainedTempHighC = 83;
static const double kSustainedRiseMidCPerSec = 0.3;
static const double kSustainedRiseHighCPerSec = 1.0;
static const double kSustainedLinkMeanMidMs = 20.0;
static const double kSustainedLinkJitterMidMs = 5.0;
static const double kSustainedLinkMeanHighMs = 80.0;
static const double kSustainedLinkJitterHighMs = 20.0;
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
    int freqMinMhz = -1;
    int freqMaxMhz = -1;
};

struct PowerSummary {
    int onExternalPower = -1; // last known sample
    int batteryPercent = -1;  // last known sample
    int changed = -1;         // -1 unknown, 0 steady, 1 the source changed
};

struct DeviceSummary {
    unsigned stage = 0u;
    std::string host;
    ComputeSummary compute;
    ThermalSummary thermal;
    PowerSummary power;
    unsigned priority = 0u;        // measured: 0 unknown, 2 low, 3 mid, 4 high
    unsigned appliedPriority = 3u; // after an optional low/high swap
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
    int retrans = -1;
    unsigned priority = 0u;
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

// Mean layer time is recorded and is not itself a priority. Speed-pack already
// used that number. What raises protection is the run getting worse, or a
// large spread, across the window.
inline unsigned computeStabilityPriority(const ComputeSummary &summary) {
    if (!summary.ratioKnown) return 0u;
    const double relativeJitter = (summary.meanKnown && summary.jitterKnown && summary.meanLayerMs > 0.0)
        ? summary.jitterLayerMs / summary.meanLayerMs : -1.0;
    if (summary.lateEarlyRatio >= kSustainedLateEarlyHigh || relativeJitter >= kSustainedJitterRatioHigh)
        return 4u;
    if (summary.lateEarlyRatio >= kSustainedLateEarlyMid || relativeJitter >= kSustainedJitterRatioMid)
        return 3u;
    return 2u;
}

inline unsigned thermalWindowPriority(const ThermalSummary &summary) {
    const bool throttling = summary.throttleKnown && summary.throttleFraction > 0.0;
    const bool hot = summary.maxTempC >= kSustainedTempHighC;
    // A one-degree step is the sensor resolution. The slope becomes evidence
    // only after the temperature has actually moved by two degrees.
    const bool riseUsable = summary.riseKnown && (summary.tempChangeC >= 2 || summary.tempChangeC <= -2);
    const bool risingFast = riseUsable && summary.riseCPerSec >= kSustainedRiseHighCPerSec;
    if (throttling || hot || risingFast) return 4u;
    if (summary.maxTempC >= kSustainedTempMidC) return 3u;
    if (riseUsable && summary.riseCPerSec >= kSustainedRiseMidCPerSec) return 3u;
    if (summary.maxTempC >= 0 || summary.throttleKnown) return 2u;
    return 0u;
}

inline unsigned powerWindowPriority(const PowerSummary &summary) {
    if (summary.onExternalPower < 0 && summary.changed < 0) return 0u;
    if (summary.onExternalPower == 0 || summary.changed == 1) return 4u;
    if (summary.onExternalPower == 1) return 2u;
    return 0u;
}

// Unknown dimensions drop out. With nothing known the device is mid, because
// a missing sample is not evidence that the machine is healthy.
inline unsigned combineProtectionPriority(unsigned compute, unsigned thermal, unsigned power) {
    unsigned weight = compute;
    if (thermal > weight) weight = thermal;
    if (power > weight) weight = power;
    return weight > 0u ? weight : 3u;
}

inline unsigned linkWindowPriority(const LinkSummary &summary) {
    if (summary.transfers == 0u) return 0u;
    const bool timedOut = summary.timeoutKnown && summary.timeoutFraction > 0.0;
    const bool retransmitted = summary.retrans > 0;
    const bool slow = summary.meanKnown && summary.meanMs >= kSustainedLinkMeanHighMs;
    const bool jumpy = summary.jitterKnown && summary.jitterMs >= kSustainedLinkJitterHighMs;
    if (timedOut || retransmitted || slow || jumpy) return 4u;
    if ((summary.meanKnown && summary.meanMs >= kSustainedLinkMeanMidMs)
        || (summary.jitterKnown && summary.jitterMs >= kSustainedLinkJitterMidMs))
        return 3u;
    if (summary.meanKnown) return 2u;
    return 0u;
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
    int sawBattery = 0;
    int sawExternal = 0;
    for (std::size_t i = 0u; i < ticks.size(); ++i) {
        const HardwareTick &tick = ticks[i];
        if (tick.batteryPercent >= 0) summary.batteryPercent = tick.batteryPercent;
        if (tick.onExternalPower == 0) sawBattery = 1;
        else if (tick.onExternalPower == 1) sawExternal = 1;
        if (tick.onExternalPower >= 0) summary.onExternalPower = tick.onExternalPower;
    }
    if (sawBattery || sawExternal) summary.changed = (sawBattery && sawExternal) ? 1 : 0;
    return summary;
}

inline LinkSummary summarizeTransfers(const std::vector<TransferRound> &rounds) {
    LinkSummary summary;
    summary.transfers = (unsigned)rounds.size();
    unsigned timeouts = 0u;
    int retransKnown = 0;
    int retransSum = 0;
    std::vector<double> finished;
    for (std::size_t i = 0u; i < rounds.size(); ++i) {
        if (rounds[i].bytes > summary.bytes) summary.bytes = rounds[i].bytes;
        if (rounds[i].timedOut) timeouts += 1u;
        else if (rounds[i].elapsedMs >= 0.0) finished.push_back(rounds[i].elapsedMs);
        if (rounds[i].retransDelta >= 0) {
            retransKnown = 1;
            retransSum += rounds[i].retransDelta;
        }
    }
    if (!rounds.empty()) {
        summary.timeoutFraction = (double)timeouts / (double)rounds.size();
        summary.timeoutKnown = true;
    }
    if (retransKnown) summary.retrans = retransSum;
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
    summary.priority = linkWindowPriority(summary);
    return summary;
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
    device.priority = combineProtectionPriority(
        computeStabilityPriority(device.compute),
        thermalWindowPriority(device.thermal),
        powerWindowPriority(device.power));
    device.appliedPriority = device.priority == 0u ? 3u : device.priority;
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
    out << "timestamp_ms,stage,temp_c,freq_mhz,freq_max_mhz,thermal_limited,power,battery_pct\n";
    for (std::size_t i = 0u; i < rows.size(); ++i) {
        const HardwareTick &tick = rows[i].tick;
        char line[192];
        std::snprintf(line, sizeof(line), "%.3f,%u,%d,%d,%d,%d,%d,%d\n",
            tick.timestampMs, rows[i].stage, tick.temperatureC, tick.freqMhz,
            tick.freqMaxMhz, tick.thermalLimited, tick.onExternalPower, tick.batteryPercent);
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
        out << "},\n      \"priority\": ";
        jsonEscape(out, protectionPriorityName(device.priority == 0u ? 0u : device.priority));
        out << ",\n      \"appliedPriority\": ";
        jsonEscape(out, protectionPriorityName(device.appliedPriority));
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
        out << ", \"retrans\": ";
        jsonIntOrNull(out, link.retrans);
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
