#include "dynamic/dynamic_tpot.hpp"

#include "app.hpp"
#include "dynamic/tpot_algorithm.hpp"
#include "dynamic/tpot_log.hpp"
#include "json.hpp"
#include "plan-command.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

using json = nlohmann::json;
namespace tpot = dllama::dynamic_tpot;

namespace {

enum class ControllerState {
    OBSERVE = 0,
    VERIFY,
    SETTLED,
};

struct NodeWindowStats {
    double totalMs = 0.0;
    double computeMs = 0.0;
    double attnMs = 0.0;
    double ffnMs = 0.0;
    uint32_t samples = 0u;
};

struct StageWindowStats {
    double stageTotalMs = 0.0;
    double stageComputeMs = 0.0;
    double boundaryCommMs = 0.0;
    double leftBoundaryLayerMs = 0.0;
    double rightBoundaryLayerMs = 0.0;
    uint32_t samples = 0u;
    std::map<uint32_t, NodeWindowStats> nodes;
};

struct WindowSummary {
    uint32_t posBegin = 0u;
    uint32_t posEnd = 0u;
    uint32_t samples = 0u;
    double tpotMs = 0.0;
    std::map<uint32_t, StageWindowStats> stages;
};

struct StageEwma {
    bool initialized = false;
    double previousStageMs = 0.0;
    double recentStageMs = 0.0;
};

struct SchedulerMetrics {
    uint32_t startPos = 0u;
    std::chrono::steady_clock::time_point startTime;
    bool sawImprovement = false;
    uint32_t firstImprovementTokens = 0u;
    double firstImprovementMs = 0.0;
    bool settled = false;
    uint32_t settleTokens = 0u;
    double settleMs = 0.0;
    uint32_t migrationCount = 0u;
    uint32_t rollbackCount = 0u;
    double steadyTpotMs = 0.0;
    double maxObservedTpotMs = 0.0;
    double baselineTpotMs = 0.0;
};

struct PendingAction {
    bool active = false;
    tpot::Candidate candidate;
    double beforeTpotMs = 0.0;
    uint32_t beforeSamples = 0u;
    uint32_t beforePosBegin = 0u;
    uint32_t beforePosEnd = 0u;
    uint32_t startPos = 0u;
    unsigned long long appliedGenerationBaseline = 0ull;
    bool ppApplied = false;
    bool rollback = false;
};

struct ControllerRuntime {
    tpot::SchedulerConfig cfg;
    ControllerState state = ControllerState::OBSERVE;
    uint32_t seq = 1u;
    uint64_t decisionSeq = 0u;
    int pollMs = 200;
    int timeoutMs = 2000;
    std::string logPath;
    uint32_t lastObservedPos = 0xFFFFFFFFu;
    bool haveObservedPos = false;
    unsigned long long lastPerfSeq = 0ull;
    uint32_t cooldownUntilPos = 0u;
    uint32_t stableWindows = 0u;
    WindowSummary window;
    std::map<uint32_t, StageEwma> ewmaByStage;
    std::map<uint32_t, uint32_t> softCapacityByStage;
    std::map<uint32_t, double> riskPenaltyByStage;
    std::vector<tpot::StageSnapshot> committedPpLayout;
    bool hasCommittedPpLayout = false;
    unsigned long long lastObservedAppliedGeneration = 0ull;
    unsigned long long lastObservedStageBypassGeneration = 0ull;
    unsigned long long topologyFenceGeneration = 0ull;
    uint32_t topologyFenceStartPos = 0u;
    uint32_t topologyFenceTimeoutTokens = 0u;
    bool controlPlaneFailed = false;
    std::vector<uint32_t> activeStageChain;
    PendingAction pending;
    tpot::PpLayoutGuard ppLayoutGuard;
    SchedulerMetrics metrics;
};

static bool parseEnvBool(const char *name, bool fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
        std::strcmp(v, "False") == 0 || std::strcmp(v, "off") == 0 ||
        std::strcmp(v, "OFF") == 0) {
        return false;
    }
    return true;
}

static int parseEnvInt(const char *name, int fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    char *end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return fallback;
    return (int)x;
}

static uint32_t parseEnvBoundedUint(const char *name, uint32_t fallback, long minimum, long maximum) {
    const char *v = std::getenv(name);
    if (v == nullptr) return fallback;
    errno = 0;
    char *end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0' || x < minimum || x > maximum) {
        throw std::runtime_error(std::string(name) + " must be a decimal integer in [" +
            std::to_string(minimum) + "," + std::to_string(maximum) + "]");
    }
    return (uint32_t)x;
}

static double parseEnvDouble(const char *name, double fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    char *end = nullptr;
    double x = std::strtod(v, &end);
    if (end == v) return fallback;
    return x;
}

static double parseEnvUnitInterval(const char *name, double fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    errno = 0;
    char *end = nullptr;
    const double x = std::strtod(v, &end);
    if (errno != 0 || end == v || *end != '\0' || !std::isfinite(x) || x < 0.0 || x > 1.0) {
        throw std::runtime_error(std::string(name) + " must be a finite number in [0,1]");
    }
    return x;
}

static tpot::SchedulerConfig loadSchedulerConfigFromEnvironmentImpl() {
    tpot::SchedulerConfig cfg;
    cfg.windowTokens = std::max(1, parseEnvInt("DLLAMA_TPOT_WINDOW_TOKENS", cfg.windowTokens));
    cfg.minSamples = std::max(1, parseEnvInt("DLLAMA_TPOT_MIN_SAMPLES", cfg.minSamples));
    cfg.cooldownTokens = std::max(0, parseEnvInt("DLLAMA_TPOT_COOLDOWN_TOKENS", cfg.cooldownTokens));
    cfg.rollbackWindow = std::max(1, parseEnvInt("DLLAMA_TPOT_ROLLBACK_WINDOW", cfg.rollbackWindow));
    cfg.ewmaAlpha = parseEnvDouble("DLLAMA_TPOT_EWMA_ALPHA", cfg.ewmaAlpha);
    if (cfg.ewmaAlpha <= 0.0 || cfg.ewmaAlpha > 1.0) cfg.ewmaAlpha = 0.2;
    cfg.minPpGainMs = parseEnvDouble("DLLAMA_TPOT_MIN_PP_GAIN_MS", cfg.minPpGainMs);
    cfg.minTpGainMs = parseEnvDouble("DLLAMA_TPOT_MIN_TP_GAIN_MS", cfg.minTpGainMs);
    cfg.loadPenaltyBeta = parseEnvDouble("DLLAMA_TPOT_LOAD_PENALTY_BETA", cfg.loadPenaltyBeta);
    cfg.ppGainRatio = parseEnvUnitInterval("DLLAMA_TPOT_PP_GAIN_RATIO", cfg.ppGainRatio);
    cfg.ppRiskMarginMs = parseEnvDouble("DLLAMA_TPOT_PP_RISK_MARGIN_MS", cfg.ppRiskMarginMs);
    cfg.tpRiskMarginMs = parseEnvDouble("DLLAMA_TPOT_TP_RISK_MARGIN_MS", cfg.tpRiskMarginMs);
    cfg.ppMigrationCostMs = parseEnvDouble("DLLAMA_TPOT_PP_MIGRATION_COST_MS", cfg.ppMigrationCostMs);
    cfg.tpMigrationCostMs = parseEnvDouble("DLLAMA_TPOT_TP_MIGRATION_COST_MS", cfg.tpMigrationCostMs);
    cfg.expectedRemainingTokens = std::max(1, parseEnvInt("DLLAMA_TPOT_EXPECTED_REMAINING_TOKENS", cfg.expectedRemainingTokens));
    cfg.maxPpLayerMove = parseEnvBoundedUint("DLLAMA_TPOT_MAX_PP_LAYER_MOVE", cfg.maxPpLayerMove, 1, 64);
    cfg.maxHeadMove = (uint32_t)std::max(1, parseEnvInt("DLLAMA_TPOT_MAX_HEAD_MOVE", (int)cfg.maxHeadMove));
    cfg.maxFfnMove = (uint32_t)std::max(1, parseEnvInt("DLLAMA_TPOT_MAX_FFN_MOVE", (int)cfg.maxFfnMove));
    return cfg;
}

static const char *stateName(ControllerState state) {
    switch (state) {
        case ControllerState::OBSERVE: return "OBSERVE";
        case ControllerState::VERIFY: return "VERIFY";
        case ControllerState::SETTLED: return "SETTLED";
        default: return "UNKNOWN";
    }
}

static void appendLog(const std::string &path, const std::string &line) {
    if (path.empty()) return;
    FILE *f = std::fopen(path.c_str(), "a");
    if (f == nullptr) return;
    std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
}

#ifndef _WIN32
static bool readLineFd(int fd, std::string &out) {
    out.clear();
    char ch;
    while (true) {
        ssize_t n = ::read(fd, &ch, 1);
        if (n == 0) return !out.empty();
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (ch == '\n') return true;
        out.push_back(ch);
        if (out.size() > 1024u * 256u) return false;
    }
}

static bool writeAllFd(int fd, const void *data, size_t len) {
    const char *p = (const char *)data;
    size_t left = len;
    while (left > 0u) {
        ssize_t w = ::write(fd, p, left);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (w == 0) return false;
        p += (size_t)w;
        left -= (size_t)w;
    }
    return true;
}

static json udsRequest(const std::string &socketPath, const json &req, int timeoutMs) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket(AF_UNIX) failed");

    struct timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (socketPath.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        throw std::runtime_error("socket path too long");
    }
    std::strncpy(addr.sun_path, socketPath.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(fd, (sockaddr *)&addr, sizeof(addr)) != 0) {
        ::close(fd);
        throw std::runtime_error("connect(AF_UNIX) failed");
    }

    const std::string line = req.dump() + "\n";
    if (!writeAllFd(fd, line.data(), line.size())) {
        ::close(fd);
        throw std::runtime_error("write request failed");
    }

    std::string respLine;
    if (!readLineFd(fd, respLine)) {
        ::close(fd);
        throw std::runtime_error("read response failed");
    }
    ::close(fd);
    return json::parse(respLine);
}
#endif

static double packetTimeMs(const json &p) {
    const double execUs = (double)p.value("execUs", 0u);
    const double syncUs = (double)p.value("syncUs", 0u);
    const double bubbleUs = (double)p.value("bubbleUs", 0u);
    return (execUs + syncUs + bubbleUs) / 1000.0;
}

static double packetComputeMs(const json &p) {
    const double execUs = (double)p.value("execUs", 0u);
    return execUs / 1000.0;
}

static double packetBoundaryCommMs(const json &p) {
    const double sendUs = (double)p.value("syncPpSendUs", 0u);
    const double recvUs = (double)p.value("syncPpRecvUs", 0u);
    return (sendUs + recvUs) / 1000.0;
}

static bool updateWindowFromPerfArray(ControllerRuntime &rt, const json &arr, uint32_t statusPos) {
    if (!arr.is_array() || arr.empty()) return false;

    uint32_t samplePos = statusPos;
    bool havePacket = false;
    bool isEvalSample = false;
    std::map<uint32_t, double> stageMaxMs;
    std::map<uint32_t, double> stageComputeMaxMs;
    std::map<uint32_t, double> stageBoundaryMs;
    std::map<uint32_t, double> stageLeftBoundaryMaxMs;
    std::map<uint32_t, double> stageRightBoundaryMaxMs;
    std::map<uint32_t, std::map<uint32_t, double> > nodeMs;
    std::map<uint32_t, std::map<uint32_t, double> > nodeComputeMs;

    for (size_t i = 0u; i < arr.size(); ++i) {
        const json &p = arr.at(i);
        if (!p.is_object()) continue;
        const uint32_t batchSize = p.value("batchSize", 0u);
        if (batchSize == 0u) continue;
        if (batchSize > 1u) isEvalSample = true;
        const uint32_t nodeIndex = p.value("nodeIndex", 0u);
        const uint32_t stageIndex = p.value("stageIndex", 0u);
        samplePos = p.value("position", samplePos);
        const double ms = packetTimeMs(p);
        const double computeMs = packetComputeMs(p);
        if (ms <= 0.0 && computeMs <= 0.0) continue;
        if (computeMs <= 0.0) continue;
        havePacket = true;
        if (stageMaxMs.find(stageIndex) == stageMaxMs.end() || ms > stageMaxMs[stageIndex]) {
            stageMaxMs[stageIndex] = ms;
        }
        if (stageComputeMaxMs.find(stageIndex) == stageComputeMaxMs.end() || computeMs > stageComputeMaxMs[stageIndex]) {
            stageComputeMaxMs[stageIndex] = computeMs;
        }
        const double commMs = packetBoundaryCommMs(p);
        if (stageBoundaryMs.find(stageIndex) == stageBoundaryMs.end() || commMs > stageBoundaryMs[stageIndex]) {
            stageBoundaryMs[stageIndex] = commMs;
        }
        const double leftBoundaryMs = (double)p.value("leftBoundaryLayerUs", 0u) / 1000.0;
        const double rightBoundaryMs = (double)p.value("rightBoundaryLayerUs", 0u) / 1000.0;
        if (stageLeftBoundaryMaxMs.find(stageIndex) == stageLeftBoundaryMaxMs.end() || leftBoundaryMs > stageLeftBoundaryMaxMs[stageIndex]) {
            stageLeftBoundaryMaxMs[stageIndex] = leftBoundaryMs;
        }
        if (stageRightBoundaryMaxMs.find(stageIndex) == stageRightBoundaryMaxMs.end() || rightBoundaryMs > stageRightBoundaryMaxMs[stageIndex]) {
            stageRightBoundaryMaxMs[stageIndex] = rightBoundaryMs;
        }
        nodeMs[stageIndex][nodeIndex] = ms;
        nodeComputeMs[stageIndex][nodeIndex] = computeMs;
    }

    // Eval phase uses batchSize>1 and has fundamentally different TPOT
    // characteristics; skip these samples so the scheduler window contains
    // only comparable pred-phase (batchSize=1) tokens.
    if (isEvalSample) {
        if (!rt.haveObservedPos || samplePos > rt.lastObservedPos) {
            rt.lastObservedPos = samplePos;
        }
        return false;
    }

    if (!havePacket || stageComputeMaxMs.empty()) return false;
    if (rt.haveObservedPos && samplePos <= rt.lastObservedPos) return false;
    if (!rt.haveObservedPos) {
        rt.metrics.startPos = samplePos;
        rt.metrics.startTime = std::chrono::steady_clock::now();
    }
    rt.haveObservedPos = true;
    rt.lastObservedPos = samplePos;

    if (rt.window.samples == 0u) rt.window.posBegin = samplePos;
    rt.window.posEnd = samplePos;
    rt.window.samples += 1u;

    double tokenTpot = 0.0;
    for (std::map<uint32_t, double>::const_iterator it = stageComputeMaxMs.begin(); it != stageComputeMaxMs.end(); ++it) {
        const uint32_t stageIndex = it->first;
        const double stageComputeMs = it->second;
        const double stageMs = stageMaxMs.count(stageIndex) != 0u ? stageMaxMs[stageIndex] : stageComputeMs;
        tokenTpot += stageComputeMs;
        StageWindowStats &st = rt.window.stages[stageIndex];
        st.stageTotalMs += stageMs;
        st.stageComputeMs += stageComputeMs;
        st.boundaryCommMs += stageBoundaryMs[stageIndex];
        st.leftBoundaryLayerMs += stageLeftBoundaryMaxMs[stageIndex];
        st.rightBoundaryLayerMs += stageRightBoundaryMaxMs[stageIndex];
        st.samples += 1u;
        const std::map<uint32_t, double> &nodes = nodeMs[stageIndex];
        for (std::map<uint32_t, double>::const_iterator nit = nodes.begin(); nit != nodes.end(); ++nit) {
            NodeWindowStats &nw = st.nodes[nit->first];
            nw.totalMs += nit->second;
            nw.computeMs += nodeComputeMs[stageIndex][nit->first];
            nw.samples += 1u;
        }
    }
    rt.window.tpotMs += tokenTpot;
    if (tokenTpot > rt.metrics.maxObservedTpotMs) rt.metrics.maxObservedTpotMs = tokenTpot;
    if (rt.metrics.baselineTpotMs <= 0.0) rt.metrics.baselineTpotMs = tokenTpot;
    return true;
}

static bool updateWindowFromPerf(ControllerRuntime &rt, const json &perfResp, uint32_t statusPos) {
    if (!perfResp.value("ok", false)) return false;

    bool updated = false;
    if (perfResp.contains("samples") && perfResp.at("samples").is_array()) {
        const json &samples = perfResp.at("samples");
        for (size_t i = 0u; i < samples.size(); ++i) {
            const json &sample = samples.at(i);
            if (!sample.is_object() || !sample.contains("perf") || !sample.at("perf").is_array()) continue;
            updated = updateWindowFromPerfArray(rt, sample.at("perf"), statusPos) || updated;
        }
        return updated;
    }

    if (!perfResp.contains("perf") || !perfResp.at("perf").is_array()) return false;
    return updateWindowFromPerfArray(rt, perfResp.at("perf"), statusPos);
}

static void finalizeWindow(WindowSummary &w) {
    if (w.samples == 0u) return;
    w.tpotMs /= (double)w.samples;
    for (std::map<uint32_t, StageWindowStats>::iterator it = w.stages.begin(); it != w.stages.end(); ++it) {
        StageWindowStats &st = it->second;
        if (st.samples == 0u) continue;
        st.stageTotalMs /= (double)st.samples;
        st.stageComputeMs /= (double)st.samples;
        st.boundaryCommMs /= (double)st.samples;
        st.leftBoundaryLayerMs /= (double)st.samples;
        st.rightBoundaryLayerMs /= (double)st.samples;
        for (std::map<uint32_t, NodeWindowStats>::iterator nit = st.nodes.begin(); nit != st.nodes.end(); ++nit) {
            NodeWindowStats &nw = nit->second;
            if (nw.samples == 0u) continue;
            nw.totalMs /= (double)nw.samples;
            nw.computeMs /= (double)nw.samples;
            nw.attnMs = nw.attnMs > 0.0 ? nw.attnMs / (double)nw.samples : 0.0;
            nw.ffnMs = nw.ffnMs > 0.0 ? nw.ffnMs / (double)nw.samples : 0.0;
        }
    }
}

static const NnStageConfig *findStageByIndex(const NnUnevenPartitionPlan *plan, uint32_t stageIndex) {
    if (plan == nullptr || plan->stages == nullptr) return nullptr;
    for (NnUint i = 0u; i < plan->nStages; ++i) {
        if (plan->stages[i].stageIndex == stageIndex) return &plan->stages[i];
    }
    return nullptr;
}

static bool stageRank(const NnStageConfig *stage, uint32_t nodeIndex, uint32_t *rank) {
    if (stage == nullptr || stage->nodeIndices == nullptr) return false;
    for (NnUint i = 0u; i < stage->nNodes; ++i) {
        if (stage->nodeIndices[i] == nodeIndex) {
            if (rank != nullptr) *rank = i;
            return true;
        }
    }
    return false;
}

static bool canMoveKvHeadToNeighbor(const NnUnevenPartitionPlan *plan, const NnStageConfig *stage, uint32_t fromNode, uint32_t toNode) {
    if (plan == nullptr || stage == nullptr) return false;
    if (plan->kvHeadSplit.starts == nullptr || plan->kvHeadSplit.lengths == nullptr) return false;
    if (fromNode >= plan->nNodes || toNode >= plan->nNodes) return false;
    if (plan->kvHeadSplit.lengths[fromNode] <= 1u) return false;
    uint32_t rf = 0u;
    uint32_t rt = 0u;
    if (!stageRank(stage, fromNode, &rf) || !stageRank(stage, toNode, &rt)) return false;
    const uint32_t dist = rf > rt ? rf - rt : rt - rf;
    if (dist != 1u) return false;
    if (plan->kvHeadComputeSplit.starts == nullptr || plan->kvHeadComputeSplit.lengths == nullptr) return false;
    const NnUint targetStart = plan->kvHeadSplit.starts[toNode];
    const NnUint targetLen = plan->kvHeadSplit.lengths[toNode];
    const NnUint shadowStart = plan->kvHeadComputeSplit.starts[toNode];
    const NnUint shadowEnd = shadowStart + plan->kvHeadComputeSplit.lengths[toNode];
    if (rt > rf) {
        const NnUint targetNeed = targetStart > 0u ? targetStart - 1u : 0u;
        return targetNeed >= shadowStart && targetNeed < shadowEnd;
    }
    const NnUint targetNeed = targetStart + targetLen;
    return targetNeed >= shadowStart && targetNeed < shadowEnd;
}

static std::vector<tpot::StageSnapshot> buildStageSnapshots(
    ControllerRuntime &rt,
    const NnUnevenPartitionPlan *plan,
    const WindowSummary &window,
    const json &status) {
    std::vector<tpot::StageSnapshot> out;
    if (plan == nullptr || plan->stages == nullptr) return out;

    const bool ppEnabled = status.contains("ppMigration") && status.at("ppMigration").is_object()
        ? status.at("ppMigration").value("enabled", false)
        : false;

    for (NnUint si = 0u; si < plan->nStages; ++si) {
        const NnStageConfig &stageCfg = plan->stages[si];
        if (stageCfg.stageIndex != 0u &&
            getPpPrevStageIndex(plan, stageCfg.stageIndex) == (NnUint)-1 &&
            getPpNextStageIndex(plan, stageCfg.stageIndex) == (NnUint)-1) continue;
        tpot::StageSnapshot stage;
        stage.stageIndex = stageCfg.stageIndex;
        stage.rootNodeIndex = stageCfg.rootNodeIndex;
        stage.startLayer = stageCfg.startLayer;
        stage.endLayer = stageCfg.endLayer;
        stage.nLayers = stageCfg.nLayers != 0u ? stageCfg.nLayers : (stageCfg.endLayer - stageCfg.startLayer);
        stage.softCapacity = rt.softCapacityByStage.count(stage.stageIndex) != 0u
            ? rt.softCapacityByStage[stage.stageIndex]
            : stage.nLayers;
        stage.hasFullWeights = ppEnabled;
        stage.riskPenalty = rt.riskPenaltyByStage.count(stage.stageIndex) != 0u
            ? rt.riskPenaltyByStage[stage.stageIndex]
            : 0.0;

        std::map<uint32_t, StageWindowStats>::const_iterator wit = window.stages.find(stage.stageIndex);
        if (wit != window.stages.end()) {
            const StageWindowStats &ws = wit->second;
            stage.stageTimeMs = ws.stageComputeMs > 0.0 ? ws.stageComputeMs : ws.stageTotalMs;
            stage.boundaryCommMs = ws.boundaryCommMs;
            stage.leftBoundaryLayerMs = ws.leftBoundaryLayerMs;
            stage.rightBoundaryLayerMs = ws.rightBoundaryLayerMs;
        }

        StageEwma &ewma = rt.ewmaByStage[stage.stageIndex];
        if (!ewma.initialized) {
            ewma.initialized = true;
            ewma.previousStageMs = stage.stageTimeMs;
            ewma.recentStageMs = stage.stageTimeMs;
        } else {
            ewma.previousStageMs = ewma.recentStageMs;
            ewma.recentStageMs = rt.cfg.ewmaAlpha * stage.stageTimeMs + (1.0 - rt.cfg.ewmaAlpha) * ewma.recentStageMs;
        }
        const double layerDiv = stage.nLayers > 0u ? (double)stage.nLayers : 1.0;
        stage.avgLayerMs = stage.stageTimeMs > 0.0 ? stage.stageTimeMs / layerDiv : 0.0;
        stage.recentAvgMs = ewma.recentStageMs / layerDiv;
        stage.previousAvgMs = ewma.previousStageMs / layerDiv;

        for (NnUint ni = 0u; ni < stageCfg.nNodes; ++ni) {
            const uint32_t nodeIndex = stageCfg.nodeIndices[ni];
            tpot::NodeSnapshot node;
            node.nodeIndex = nodeIndex;
            if (plan->kvHeadSplit.lengths != nullptr && nodeIndex < plan->nNodes) node.kvHeads = plan->kvHeadSplit.lengths[nodeIndex];
            else if (plan->headSplit.lengths != nullptr && nodeIndex < plan->nNodes) node.kvHeads = plan->headSplit.lengths[nodeIndex];
            if (plan->ffnSplit.lengths != nullptr && nodeIndex < plan->nNodes) node.ffnUnits = plan->ffnSplit.lengths[nodeIndex];
            if (wit != window.stages.end()) {
                std::map<uint32_t, NodeWindowStats>::const_iterator nit = wit->second.nodes.find(nodeIndex);
                if (nit != wit->second.nodes.end()) {
                    node.timeMs = nit->second.computeMs > 0.0 ? nit->second.computeMs : nit->second.totalMs;
                    node.attnMs = nit->second.attnMs;
                    node.ffnMs = nit->second.ffnMs;
                }
            }
            if (ni > 0u) node.canMoveHeadLeft = canMoveKvHeadToNeighbor(plan, &stageCfg, nodeIndex, stageCfg.nodeIndices[ni - 1u]);
            if (ni + 1u < stageCfg.nNodes) node.canMoveHeadRight = canMoveKvHeadToNeighbor(plan, &stageCfg, nodeIndex, stageCfg.nodeIndices[ni + 1u]);
            stage.nodes.push_back(node);
        }
        out.push_back(stage);
    }
    return out;
}

static bool commitAppliedStageBypass(ControllerRuntime &rt, const json &status) {
    if (!status.contains("stageBypass") || !status.at("stageBypass").is_object()) return false;
    const json &bypass = status.at("stageBypass");
    const unsigned long long generation = bypass.value("appliedGeneration", 0ull);
    if (generation <= rt.lastObservedStageBypassGeneration || !rt.hasCommittedPpLayout) return false;
    // A root-side bypass may arrive while an earlier PP command is still in
    // flight. Retain this generation in status and commit it only after the
    // pending command reaches its explicit terminal path below.
    if (rt.pending.active) return false;
    const uint32_t ejected = bypass.value("ejectedStage", 0xFFFFFFFFu);
    const uint32_t target = bypass.value("targetStage", 0xFFFFFFFFu);
    if (!bypass.contains("appliedLayers") || !bypass.at("appliedLayers").is_array()) return false;
    std::vector<uint32_t> appliedLayers;
    for (size_t i = 0u; i < bypass.at("appliedLayers").size(); ++i) {
        if (!bypass.at("appliedLayers").at(i).is_number_unsigned()) return false;
        appliedLayers.push_back(bypass.at("appliedLayers").at(i).get<uint32_t>());
    }
    uint32_t targetOldLayers = 0u;
    for (size_t i = 0u; i < rt.committedPpLayout.size(); ++i) {
        if (rt.committedPpLayout[i].stageIndex == target) targetOldLayers = rt.committedPpLayout[i].nLayers;
    }
    std::vector<tpot::StageSnapshot> mergedLayout = rt.committedPpLayout;
    std::vector<uint32_t> mergedChain;
    if (!tpot::commitStageBypassLayout(mergedLayout, ejected, target, appliedLayers, &mergedChain)) return false;
    rt.committedPpLayout.swap(mergedLayout);
    for (size_t i = 0u; i < rt.committedPpLayout.size(); ++i) {
        if (rt.committedPpLayout[i].stageIndex == target) {
            tpot::rebasePpSoftCapacity(rt.committedPpLayout[i], targetOldLayers);
            break;
        }
    }
    rt.softCapacityByStage[target] = 0u;
    for (size_t i = 0u; i < rt.committedPpLayout.size(); ++i) {
        if (rt.committedPpLayout[i].stageIndex == target) rt.softCapacityByStage[target] = rt.committedPpLayout[i].softCapacity;
    }
    rt.ewmaByStage.erase(ejected);
    rt.softCapacityByStage.erase(ejected);
    rt.riskPenaltyByStage.erase(ejected);
    rt.activeStageChain.swap(mergedChain);
    rt.ppLayoutGuard = tpot::PpLayoutGuard();
    rt.lastObservedStageBypassGeneration = generation;
    rt.topologyFenceGeneration = generation;
    rt.topologyFenceStartPos = rt.haveObservedPos ? rt.lastObservedPos : 0u;
    return true;
}

static void overlayCommittedPpLayout(
    ControllerRuntime &rt,
    std::vector<tpot::StageSnapshot> &stages) {
    if (!rt.hasCommittedPpLayout) return;
    for (size_t i = 0u; i < stages.size(); ++i) {
        for (size_t j = 0u; j < rt.committedPpLayout.size(); ++j) {
            if (stages[i].stageIndex != rt.committedPpLayout[j].stageIndex) continue;
            const uint32_t measuredLayers = stages[i].nLayers > 0u ? stages[i].nLayers : 1u;
            stages[i].startLayer = rt.committedPpLayout[j].startLayer;
            stages[i].endLayer = rt.committedPpLayout[j].endLayer;
            stages[i].nLayers = rt.committedPpLayout[j].nLayers;
            stages[i].softCapacity = rt.softCapacityByStage.count(stages[i].stageIndex) != 0u
                ? rt.softCapacityByStage[stages[i].stageIndex]
                : stages[i].nLayers;
            const double layerDiv = stages[i].nLayers > 0u ? (double)stages[i].nLayers : 1.0;
            stages[i].avgLayerMs = stages[i].stageTimeMs / layerDiv;
            stages[i].recentAvgMs = stages[i].recentAvgMs * (double)measuredLayers / layerDiv;
            stages[i].previousAvgMs = stages[i].previousAvgMs * (double)measuredLayers / layerDiv;
            break;
        }
    }
}

static bool appliedPpMatches(const json &status, const PendingAction &pending, unsigned long long *generation) {
    if (pending.candidate.kind != tpot::CandidateKind::PP_MOVE || !status.contains("ppMigration")) return false;
    const json &pp = status.at("ppMigration");
    if (!pp.is_object()) return false;
    const unsigned long long applied = pp.value("appliedGeneration", 0ull);
    if (generation != nullptr) *generation = applied;
    if (applied <= pending.appliedGenerationBaseline) return false;
    if (pp.value("appliedFromNodeIndex", 0xFFFFFFFFu) != pending.candidate.fromNodeIndex ||
        pp.value("appliedToNodeIndex", 0xFFFFFFFFu) != pending.candidate.toNodeIndex) return false;
    if (!pp.contains("appliedLayers") || !pp.at("appliedLayers").is_array() ||
        pp.at("appliedLayers").size() != pending.candidate.layerCount) return false;
    for (uint32_t i = 0u; i < pending.candidate.layerCount; ++i) {
        if (pp.at("appliedLayers").at(i).get<uint32_t>() != pending.candidate.layerIndex + i) return false;
    }
    return true;
}

static bool commitAppliedPpMove(ControllerRuntime &rt, const tpot::Candidate &candidate) {
    uint32_t fromOld = 0u;
    uint32_t toOld = 0u;
    for (size_t i = 0u; i < rt.committedPpLayout.size(); ++i) {
        if (rt.committedPpLayout[i].stageIndex == candidate.fromStageIndex) fromOld = rt.committedPpLayout[i].nLayers;
        if (rt.committedPpLayout[i].stageIndex == candidate.toStageIndex) toOld = rt.committedPpLayout[i].nLayers;
    }
    if (!tpot::applyPpMove(rt.committedPpLayout, candidate)) return false;
    for (size_t i = 0u; i < rt.committedPpLayout.size(); ++i) {
        tpot::StageSnapshot &stage = rt.committedPpLayout[i];
        if (stage.stageIndex == candidate.fromStageIndex) tpot::rebasePpSoftCapacity(stage, fromOld);
        if (stage.stageIndex == candidate.toStageIndex) tpot::rebasePpSoftCapacity(stage, toOld);
        if (stage.stageIndex == candidate.fromStageIndex || stage.stageIndex == candidate.toStageIndex) {
            rt.softCapacityByStage[stage.stageIndex] = stage.softCapacity;
        }
    }
    rt.ppLayoutGuard.markCommitted(candidate);
    return true;
}

static bool tpotJitterStable(const std::vector<double> &recent) {
    if (recent.size() < 3u) return false;
    double minV = recent[0];
    double maxV = recent[0];
    double sum = 0.0;
    for (size_t i = 0u; i < recent.size(); ++i) {
        minV = std::min(minV, recent[i]);
        maxV = std::max(maxV, recent[i]);
        sum += recent[i];
    }
    const double avg = sum / (double)recent.size();
    if (avg <= 0.0) return false;
    return ((maxV - minV) / avg) <= 0.03;
}

static json makeTpCommand(uint32_t seq, const tpot::Candidate &c) {
    uint32_t kind = PLAN_CMD_KIND_HEAD;
    if (c.headMove != 0u && c.ffnMove != 0u) kind = PLAN_CMD_KIND_BOTH;
    else if (c.ffnMove != 0u) kind = PLAN_CMD_KIND_FFN;
    json cmd;
    cmd["seq"] = seq;
    cmd["mode"] = "next_barrier";
    cmd["stageIndex"] = c.stageIndex;
    json moves = json::array();
    moves.push_back(json{
        {"fromNodeIndex", c.fromNodeIndex},
        {"toNodeIndex", c.toNodeIndex},
        {"cmdKind", kind},
        {"headMove", c.headMove},
        {"ffnMove", c.ffnMove},
    });
    cmd["moves"] = moves;
    return json{{"op", "set_plan"}, {"cmd", cmd}};
}

static bool issueCandidate(const std::string &socketPath, int timeoutMs, uint32_t seq, const tpot::Candidate &c, json *resp) {
#ifdef _WIN32
    (void)socketPath;
    (void)timeoutMs;
    (void)seq;
    (void)c;
    (void)resp;
    return false;
#else
    json req;
    if (c.kind == tpot::CandidateKind::PP_MOVE) req = tpot::makePpCommandRequest(seq, c);
    else if (c.kind == tpot::CandidateKind::TP_HEAD || c.kind == tpot::CandidateKind::TP_FFN) req = makeTpCommand(seq, c);
    else return false;
    const json out = udsRequest(socketPath, req, timeoutMs);
    if (resp != nullptr) *resp = out;
    return out.value("ok", false);
#endif
}

static std::string metricsString(const SchedulerMetrics &m, uint32_t currentPos) {
    std::ostringstream oss;
    oss << "time_to_first_improvement_tokens=" << (m.sawImprovement ? m.firstImprovementTokens : 0u)
        << " time_to_first_improvement_ms=" << (m.sawImprovement ? m.firstImprovementMs : 0.0)
        << " time_to_settle_tokens=" << (m.settled ? m.settleTokens : 0u)
        << " time_to_settle_ms=" << (m.settled ? m.settleMs : 0.0)
        << " migration_count=" << m.migrationCount
        << " rollback_count=" << m.rollbackCount
        << " steady_tpot=" << m.steadyTpotMs;
    const double base = m.baselineTpotMs > 0.0 ? m.baselineTpotMs : 1.0;
    const double overshoot = ((m.maxObservedTpotMs - base) / base) * 100.0;
    oss << " overshoot_pct=" << overshoot
        << " current_pos=" << currentPos;
    return oss.str();
}

static double tpotDeltaPct(double beforeMs, double afterMs) {
    if (beforeMs <= 0.0) return 0.0;
    return ((afterMs - beforeMs) / beforeMs) * 100.0;
}

static void logDecision(
    ControllerRuntime &rt,
    const WindowSummary &window,
    const tpot::Candidate &best,
    const tpot::Candidate &bestPp,
    bool issued,
    const char *extra,
    const PendingAction *comparison = nullptr,
    uint32_t verifyElapsedTokens = 0u) {
    const PendingAction *cmp = comparison;
    const bool hasCompare = (cmp != nullptr) || rt.pending.active;
    double beforeTpotMs = window.tpotMs;
    uint32_t beforeSamples = window.samples;
    uint32_t beforePosBegin = window.posBegin;
    uint32_t beforePosEnd = window.posEnd;
    uint32_t compareStartPos = window.posEnd;
    if (cmp != nullptr) {
        beforeTpotMs = cmp->beforeTpotMs;
        beforeSamples = cmp->beforeSamples;
        beforePosBegin = cmp->beforePosBegin;
        beforePosEnd = cmp->beforePosEnd;
        compareStartPos = cmp->startPos;
    } else if (rt.pending.active) {
        beforeTpotMs = rt.pending.beforeTpotMs;
        beforeSamples = rt.pending.beforeSamples;
        beforePosBegin = rt.pending.beforePosBegin;
        beforePosEnd = rt.pending.beforePosEnd;
        compareStartPos = rt.pending.startPos;
    }

    const double afterTpotMs = window.tpotMs;
    const double deltaMs = hasCompare ? (afterTpotMs - beforeTpotMs) : 0.0;
    const double improveMs = hasCompare ? (beforeTpotMs - afterTpotMs) : 0.0;
    const double deltaPct = hasCompare ? tpotDeltaPct(beforeTpotMs, afterTpotMs) : 0.0;
    const double improvePct = -deltaPct;
    const bool improved = hasCompare && deltaMs < 0.0;
    const bool degraded = hasCompare && beforeTpotMs > 0.0 && afterTpotMs > beforeTpotMs * 1.05;

    std::ostringstream oss;
    oss << "tpot_sched seq=" << rt.decisionSeq
        << " state=" << stateName(rt.state)
        << " " << tpot::formatSelectedCandidateLogFields(best)
        << " issued=" << (issued ? 1 : 0)
        << " tpot_before=" << beforeTpotMs
        << " tpot_after=" << afterTpotMs
        << " tpot_delta_ms=" << deltaMs
        << " tpot_delta_pct=" << deltaPct
        << " tpot_improve_ms=" << improveMs
        << " tpot_improve_pct=" << improvePct
        << " tpot_compare=" << (hasCompare ? 1 : 0)
        << " tpot_improved=" << (improved ? 1 : 0)
        << " tpot_degraded=" << (degraded ? 1 : 0)
        << " pre_samples=" << beforeSamples
        << " pre_pos_begin=" << beforePosBegin
        << " pre_pos_end=" << beforePosEnd
        << " post_samples=" << window.samples
        << " post_pos_begin=" << window.posBegin
        << " post_pos_end=" << window.posEnd
        << " verify_start_pos=" << compareStartPos
        << " verify_elapsed_tokens=" << verifyElapsedTokens
        << " settled=" << (rt.metrics.settled ? 1 : 0)
        << " migrations=" << rt.metrics.migrationCount
        << " rollbacks=" << rt.metrics.rollbackCount
        << " samples=" << window.samples
        << " pos_begin=" << window.posBegin
        << " pos_end=" << window.posEnd;
    oss << " " << tpot::formatPpCandidateLogFields(bestPp);
    if (extra != nullptr && extra[0] != '\0') oss << " note=" << extra;
    oss << " " << metricsString(rt.metrics, window.posEnd);
    appendLog(rt.logPath, oss.str());
}

} // namespace

namespace dllama {
namespace dynamic_tpot {

SchedulerConfig loadSchedulerConfigFromEnvironment() {
    return loadSchedulerConfigFromEnvironmentImpl();
}

nlohmann::json makePpCommandRequest(uint32_t seq, const Candidate &candidate) {
    json cmd;
    cmd["seq"] = seq;
    cmd["mode"] = "next_barrier";
    cmd["fromNodeIndex"] = candidate.fromNodeIndex;
    cmd["toNodeIndex"] = candidate.toNodeIndex;
    cmd["firstLayer"] = candidate.layerIndex;
    cmd["layerCount"] = ppCommandLayerCount(candidate);
    return json{{"op", "set_pp_migration"}, {"cmd", cmd}};
}

} // namespace dynamic_tpot
} // namespace dllama

std::unique_ptr<DynamicTpotController> DynamicTpotController::start(const std::string &socketPath, RootLlmInference *inference) {
#ifdef _WIN32
    (void)socketPath;
    (void)inference;
    return nullptr;
#else
    if (!parseEnvBool("DLLAMA_DYNAMIC_TPOT_ENABLE", false)) return nullptr;

    const char *logEnv = std::getenv("DLLAMA_TPOT_LOG");
    const std::string logPath = (logEnv != nullptr && logEnv[0] != '\0')
        ? std::string(logEnv)
        : std::string("/tmp/dllama_tpot_scheduler.log");

    tpot::SchedulerConfig config;
    try {
        config = tpot::loadSchedulerConfigFromEnvironment();
    } catch (const std::exception &e) {
        appendLog(logPath, std::string("tpot_sched seq=0 state=DISABLED best=none gain_ms=0 note=invalid_configuration:") + e.what());
        return nullptr;
    } catch (...) {
        appendLog(logPath, "tpot_sched seq=0 state=DISABLED best=none gain_ms=0 note=invalid_configuration:unknown_exception");
        return nullptr;
    }

    if (socketPath.empty()) {
        appendLog(logPath, "tpot_sched seq=0 state=DISABLED best=none gain_ms=0 note=missing_DLLAMA_PLAN_CTRL_SOCKET");
        return nullptr;
    }
    if (inference == nullptr) {
        appendLog(logPath, "tpot_sched seq=0 state=DISABLED best=none gain_ms=0 note=missing_inference");
        return nullptr;
    }
    if (!inference->isPpMigrationEnabled()) {
        appendLog(logPath, "tpot_sched seq=0 state=START best=none gain_ms=0 note=pp_migration_disabled_pp_candidates_filtered");
    }

    std::unique_ptr<DynamicTpotController> ctrl(new DynamicTpotController(socketPath, inference, config));
    DynamicTpotController *c = ctrl.get();
    ctrl->worker_ = std::thread([c]() { c->run(); });
    appendLog(logPath, "tpot_sched seq=0 state=START best=none gain_ms=0 note=controller_started");
    return ctrl;
#endif
}

DynamicTpotController::DynamicTpotController(
    const std::string &socketPath,
    RootLlmInference *inference,
    const tpot::SchedulerConfig &config)
    : socketPath_(socketPath), inference_(inference), config_(config) {}

DynamicTpotController::~DynamicTpotController() {
    stop_.store(true);
    if (worker_.joinable()) worker_.join();
}

void DynamicTpotController::run() {
#ifdef _WIN32
    return;
#else
    ControllerRuntime rt;
    rt.cfg = config_;
    rt.pollMs = std::max(10, parseEnvInt("DLLAMA_TPOT_POLL_MS", 200));
    rt.timeoutMs = std::max(100, parseEnvInt("DLLAMA_TPOT_UDS_TIMEOUT_MS", 2000));
    rt.topologyFenceTimeoutTokens = (uint32_t)std::max(1,
        parseEnvInt("DLLAMA_TPOT_TOPOLOGY_TIMEOUT_TOKENS", std::max(32, rt.cfg.rollbackWindow * 2)));
    const char *logEnv = std::getenv("DLLAMA_TPOT_LOG");
    rt.logPath = (logEnv != nullptr && logEnv[0] != '\0') ? std::string(logEnv) : std::string("/tmp/dllama_tpot_scheduler.log");

    std::vector<double> recentTpotWindows;

    while (!stop_.load()) {
        try {
            const NnUnevenPartitionPlan *plan = inference_ != nullptr ? inference_->getPartitionPlan() : nullptr;
            if (plan == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                continue;
            }

            json statusReq;
            statusReq["op"] = "status";
            const json status = udsRequest(socketPath_, statusReq, rt.timeoutMs);
            if (!status.value("ok", false)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                continue;
            }
            if (!status.value("enablePlanBarrier", false)) {
                appendLog(rt.logPath, "tpot_sched seq=0 state=DISABLED best=none gain_ms=0 note=enable_plan_barrier_false");
                std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                continue;
            }

            json perfReq;
            perfReq["op"] = "perf";
            perfReq["afterSeq"] = rt.lastPerfSeq;
            perfReq["maxSamples"] = std::max(1, rt.cfg.windowTokens * 4);
            const json perf = udsRequest(socketPath_, perfReq, rt.timeoutMs);
            const uint32_t statusPos = status.value("position", 0u);
            (void)updateWindowFromPerf(rt, perf, statusPos);
            rt.lastPerfSeq = perf.value("latestSeq", rt.lastPerfSeq);

            const bool enoughTokens = rt.window.samples >= (uint32_t)rt.cfg.windowTokens;
            const bool enoughSamples = rt.window.samples >= (uint32_t)rt.cfg.minSamples;
            if (!enoughTokens || !enoughSamples) {
                std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                continue;
            }

            WindowSummary window = rt.window;
            rt.window = WindowSummary();
            finalizeWindow(window);
            rt.decisionSeq += 1u;
            recentTpotWindows.push_back(window.tpotMs);
            if (recentTpotWindows.size() > 3u) recentTpotWindows.erase(recentTpotWindows.begin());

            const bool bypassCommitted = commitAppliedStageBypass(rt, status);
            std::vector<tpot::StageSnapshot> stages = buildStageSnapshots(rt, plan, window, status);
            if (!rt.hasCommittedPpLayout) {
                rt.committedPpLayout = stages;
                rt.hasCommittedPpLayout = true;
                for (size_t i = 0u; i < stages.size(); ++i) rt.activeStageChain.push_back(stages[i].stageIndex);
            }
            overlayCommittedPpLayout(rt, stages);
            if (bypassCommitted) {
                std::ostringstream topology;
                topology << "tpot_sched topology_generation=" << rt.lastObservedStageBypassGeneration
                    << " active_stage_chain=";
                for (size_t i = 0u; i < rt.activeStageChain.size(); ++i) {
                    if (i != 0u) topology << ",";
                    topology << rt.activeStageChain[i];
                }
                appendLog(rt.logPath, topology.str());
            }
            if (rt.topologyFenceGeneration != 0ull) {
                const json &bypass = status.contains("stageBypass") ? status.at("stageBypass") : json();
                const unsigned long long verified = bypass.is_object() ? bypass.value("verifiedGeneration", 0ull) : 0ull;
                const std::string failure = bypass.is_object() ? bypass.value("failureReason", std::string()) : std::string();
                const uint32_t elapsed = window.posEnd >= rt.topologyFenceStartPos
                    ? window.posEnd - rt.topologyFenceStartPos : 0u;
                if (!failure.empty() || elapsed >= rt.topologyFenceTimeoutTokens) {
                    rt.controlPlaneFailed = true;
                    appendLog(rt.logPath, "tpot_sched control_plane_failed reason=" +
                        (!failure.empty() ? failure : std::string("stage_bypass_verification_timeout")));
                    return;
                }
                if (verified < rt.topologyFenceGeneration) {
                    appendLog(rt.logPath, "tpot_sched topology_fence_wait generation=" +
                        std::to_string(rt.topologyFenceGeneration));
                    std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                    continue;
                }
                rt.topologyFenceGeneration = 0ull;
            }
            tpot::Candidate bestTp = tpot::bestTpCandidate(stages, rt.cfg);
            tpot::Candidate bestPp = tpot::bestPpCandidate(stages, window.tpotMs, rt.cfg);
            bestPp = rt.ppLayoutGuard.filter(bestPp);
            tpot::Candidate best = tpot::betterCandidate(bestTp, bestPp);

            bool issued = false;
            const char *note = "";

            if (rt.pending.active) {
                const PendingAction pendingForLog = rt.pending;
                bool verifyIssued = false;
                if (rt.pending.candidate.kind == tpot::CandidateKind::PP_MOVE && !rt.pending.ppApplied) {
                    unsigned long long appliedGeneration = 0ull;
                    if (!appliedPpMatches(status, rt.pending, &appliedGeneration)) {
                        const json &bypass = status.contains("stageBypass") ? status.at("stageBypass") : json();
                        const unsigned long long bypassGeneration = bypass.is_object()
                            ? bypass.value("rootApplyGeneration", bypass.value("appliedGeneration", 0ull)) : 0ull;
                        const uint32_t pendingElapsed = window.posEnd >= rt.pending.startPos
                            ? window.posEnd - rt.pending.startPos : 0u;
                        if (pendingElapsed >= rt.topologyFenceTimeoutTokens) {
                            appendLog(rt.logPath, "tpot_sched control_plane_failed reason=" +
                                std::string(bypassGeneration > rt.lastObservedStageBypassGeneration
                                    ? "pending_pp_cancelled_by_bypass_timeout" : "pending_pp_apply_timeout"));
                            return;
                        }
                        note = "pp_apply_wait";
                        logDecision(rt, window, pendingForLog.candidate, bestPp, false, note, &pendingForLog, 0u);
                        std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                        continue;
                    }
                    if (!commitAppliedPpMove(rt, rt.pending.candidate)) {
                        note = "pp_apply_commit_failed";
                        logDecision(rt, window, pendingForLog.candidate, bestPp, false, note, &pendingForLog, 0u);
                        std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                        continue;
                    }
                    rt.lastObservedAppliedGeneration = appliedGeneration;
                    rt.pending.ppApplied = true;
                    if (rt.pending.rollback) {
                        const uint32_t penalizedStage = rt.pending.candidate.fromStageIndex;
                        rt.riskPenaltyByStage[penalizedStage] += 0.1;
                        for (size_t i = 0u; i < rt.committedPpLayout.size(); ++i) {
                            if (rt.committedPpLayout[i].stageIndex != penalizedStage) continue;
                            tpot::applyRollbackPenalty(rt.committedPpLayout[i]);
                            rt.softCapacityByStage[penalizedStage] = rt.committedPpLayout[i].softCapacity;
                            break;
                        }
                        rt.pending.active = false;
                        rt.state = ControllerState::OBSERVE;
                        note = "rollback_applied";
                        logDecision(rt, window, pendingForLog.candidate, bestPp, false, note, &pendingForLog, 0u);
                        std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                        continue;
                    }
                }
                const uint32_t elapsed = window.posEnd >= rt.pending.startPos ? window.posEnd - rt.pending.startPos : 0u;
                if (elapsed >= (uint32_t)rt.cfg.rollbackWindow) {
                    const bool degraded = window.tpotMs > rt.pending.beforeTpotMs * 1.05;
                    if (degraded) {
                        tpot::Candidate reverse = tpot::reversePpCandidate(rt.pending.candidate);
                        json resp;
                        if (issueCandidate(socketPath_, rt.timeoutMs, rt.seq++, reverse, &resp)) {
                            rt.metrics.rollbackCount += 1u;
                            rt.cooldownUntilPos = window.posEnd + (uint32_t)std::max(64, rt.cfg.cooldownTokens);
                            if (rt.pending.candidate.kind == tpot::CandidateKind::PP_MOVE) {
                                rt.ppLayoutGuard.markIssued(reverse);
                                rt.pending.candidate = reverse;
                                rt.pending.appliedGenerationBaseline = rt.lastObservedAppliedGeneration;
                                rt.pending.ppApplied = false;
                                rt.pending.rollback = true;
                                rt.pending.startPos = window.posEnd;
                                verifyIssued = true;
                                note = "rollback_issued";
                                logDecision(rt, window, pendingForLog.candidate, bestPp, verifyIssued, note, &pendingForLog, elapsed);
                                std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                                continue;
                            }
                            verifyIssued = true;
                            note = "rollback_issued";
                        } else {
                            note = "rollback_failed";
                        }
                    } else {
                        if (!rt.metrics.sawImprovement && window.tpotMs < rt.pending.beforeTpotMs) {
                            rt.metrics.sawImprovement = true;
                            rt.metrics.firstImprovementTokens = window.posEnd >= rt.metrics.startPos ? window.posEnd - rt.metrics.startPos : 0u;
                            rt.metrics.firstImprovementMs = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - rt.metrics.startTime).count();
                        }
                        note = "verify_ok";
                    }
                    rt.pending.active = false;
                    rt.state = ControllerState::OBSERVE;
                } else {
                    note = "verify_wait";
                }
                logDecision(rt, window, pendingForLog.candidate, bestPp, verifyIssued, note, &pendingForLog, elapsed);
                std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                continue;
            }

            if (window.posEnd < rt.cooldownUntilPos) {
                note = "cooldown";
                logDecision(rt, window, best, bestPp, false, note);
                std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                continue;
            }

            if (best.valid) {
                json resp;
                issued = issueCandidate(socketPath_, rt.timeoutMs, rt.seq++, best, &resp);
                if (issued) {
                    rt.metrics.migrationCount += 1u;
                    if (best.kind == tpot::CandidateKind::PP_MOVE) {
                        rt.ppLayoutGuard.markIssued(best);
                        const json pp = status.value("ppMigration", json::object());
                        rt.pending.appliedGenerationBaseline = pp.value("appliedGeneration", 0ull);
                        rt.pending.ppApplied = false;
                        rt.pending.rollback = false;
                    }
                    rt.pending.active = true;
                    rt.pending.candidate = best;
                    rt.pending.beforeTpotMs = window.tpotMs;
                    rt.pending.beforeSamples = window.samples;
                    rt.pending.beforePosBegin = window.posBegin;
                    rt.pending.beforePosEnd = window.posEnd;
                    rt.pending.startPos = window.posEnd;
                    rt.cooldownUntilPos = window.posEnd + (uint32_t)rt.cfg.cooldownTokens;
                    rt.state = ControllerState::VERIFY;
                    rt.stableWindows = 0u;
                    note = "migration_issued";
                } else {
                    note = "migration_issue_failed";
                }
            } else {
                const bool stable = tpotJitterStable(recentTpotWindows);
                if (stable) rt.stableWindows += 1u;
                else rt.stableWindows = 0u;
                if (rt.stableWindows >= 3u) {
                    rt.state = ControllerState::SETTLED;
                    rt.metrics.settled = true;
                    rt.metrics.steadyTpotMs = window.tpotMs;
                    rt.metrics.settleTokens = window.posEnd >= rt.metrics.startPos ? window.posEnd - rt.metrics.startPos : 0u;
                    rt.metrics.settleMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - rt.metrics.startTime).count();
                } else {
                    rt.state = ControllerState::OBSERVE;
                }
                note = stable ? "no_gain_stable" : "no_gain";
            }

            logDecision(rt, window, best, bestPp, issued, note);
        } catch (const std::exception &e) {
            std::ostringstream oss;
            oss << "tpot_sched seq=" << rt.decisionSeq << " state=" << stateName(rt.state)
                << " best=none gain_ms=0 note=exception:" << e.what();
            appendLog(rt.logPath, oss.str());
        } catch (...) {
            std::ostringstream oss;
            oss << "tpot_sched seq=" << rt.decisionSeq << " state=" << stateName(rt.state)
                << " best=none gain_ms=0 note=unknown_exception";
            appendLog(rt.logPath, oss.str());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
    }
#endif
}
