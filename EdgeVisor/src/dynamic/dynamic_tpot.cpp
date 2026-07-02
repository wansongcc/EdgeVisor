#include "dynamic/dynamic_tpot.hpp"

#include "app.hpp"
#include "dynamic/tpot_algorithm.hpp"
#include "json.hpp"
#include "plan-command.hpp"

#include <algorithm>
#include <chrono>
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
    uint32_t startPos = 0u;
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
    uint32_t cooldownUntilPos = 0u;
    uint32_t stableWindows = 0u;
    WindowSummary window;
    std::map<uint32_t, StageEwma> ewmaByStage;
    std::map<uint32_t, uint32_t> softCapacityByStage;
    std::map<uint32_t, double> riskPenaltyByStage;
    PendingAction pending;
    std::set<std::string> issuedPpKeys;
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

static double parseEnvDouble(const char *name, double fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    char *end = nullptr;
    double x = std::strtod(v, &end);
    if (end == v) return fallback;
    return x;
}

static tpot::SchedulerConfig loadSchedulerConfig() {
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
    cfg.ppRiskMarginMs = parseEnvDouble("DLLAMA_TPOT_PP_RISK_MARGIN_MS", cfg.ppRiskMarginMs);
    cfg.tpRiskMarginMs = parseEnvDouble("DLLAMA_TPOT_TP_RISK_MARGIN_MS", cfg.tpRiskMarginMs);
    cfg.ppMigrationCostMs = parseEnvDouble("DLLAMA_TPOT_PP_MIGRATION_COST_MS", cfg.ppMigrationCostMs);
    cfg.tpMigrationCostMs = parseEnvDouble("DLLAMA_TPOT_TP_MIGRATION_COST_MS", cfg.tpMigrationCostMs);
    cfg.expectedRemainingTokens = std::max(1, parseEnvInt("DLLAMA_TPOT_EXPECTED_REMAINING_TOKENS", cfg.expectedRemainingTokens));
    cfg.maxPpLayerMove = (uint32_t)std::max(1, parseEnvInt("DLLAMA_TPOT_MAX_PP_LAYER_MOVE", (int)cfg.maxPpLayerMove));
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

static bool updateWindowFromPerf(ControllerRuntime &rt, const json &perfResp, uint32_t statusPos) {
    if (!perfResp.value("ok", false)) return false;
    if (!perfResp.contains("perf") || !perfResp.at("perf").is_array()) return false;
    const json &arr = perfResp.at("perf");
    if (arr.empty()) return false;

    uint32_t samplePos = statusPos;
    bool havePacket = false;
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
        const uint32_t nodeIndex = p.value("nodeIndex", 0u);
        const uint32_t stageIndex = p.value("stageIndex", 0u);
        samplePos = p.value("position", samplePos);
        const double ms = packetTimeMs(p);
        const double computeMs = packetComputeMs(p);
        if (ms <= 0.0 && computeMs <= 0.0) continue;
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

    if (!havePacket || stageMaxMs.empty()) return false;
    if (rt.haveObservedPos && samplePos == rt.lastObservedPos) return false;
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
    for (std::map<uint32_t, double>::const_iterator it = stageMaxMs.begin(); it != stageMaxMs.end(); ++it) {
        const uint32_t stageIndex = it->first;
        const double stageMs = it->second;
        const double stageComputeMs = stageComputeMaxMs[stageIndex];
        tokenTpot += stageMs;
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

static std::string commandName(const tpot::Candidate &c) {
    return tpot::candidateKindName(c.kind);
}

static std::string ppCandidateKey(const tpot::Candidate &c) {
    std::ostringstream oss;
    oss << c.fromStageIndex << ":" << c.toStageIndex << ":" << c.layerIndex;
    return oss.str();
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

static json makePpCommand(uint32_t seq, const tpot::Candidate &c) {
    json cmd;
    cmd["seq"] = seq;
    cmd["mode"] = "next_barrier";
    cmd["fromNodeIndex"] = c.fromNodeIndex;
    cmd["toNodeIndex"] = c.toNodeIndex;
    cmd["layerCount"] = 1u;
    return json{{"op", "set_pp_migration"}, {"cmd", cmd}};
}

static tpot::Candidate reverseCandidate(const tpot::Candidate &c) {
    tpot::Candidate r = c;
    std::swap(r.fromNodeIndex, r.toNodeIndex);
    std::swap(r.fromStageIndex, r.toStageIndex);
    return r;
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
    if (c.kind == tpot::CandidateKind::PP_MOVE) req = makePpCommand(seq, c);
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

static void logDecision(
    ControllerRuntime &rt,
    const WindowSummary &window,
    const tpot::Candidate &best,
    bool issued,
    const char *extra) {
    std::ostringstream oss;
    oss << "tpot_sched seq=" << rt.decisionSeq
        << " state=" << stateName(rt.state)
        << " best=" << commandName(best)
        << " gain_ms=" << best.gainMs
        << " threshold_ms=" << best.thresholdMs
        << " issued=" << (issued ? 1 : 0)
        << " from_stage=" << best.fromStageIndex
        << " to_stage=" << best.toStageIndex
        << " from_node=" << best.fromNodeIndex
        << " to_node=" << best.toNodeIndex
        << " layer=" << best.layerIndex
        << " head_move=" << best.headMove
        << " ffn_move=" << best.ffnMove
        << " tpot_before=" << (rt.pending.active ? rt.pending.beforeTpotMs : window.tpotMs)
        << " tpot_after=" << window.tpotMs
        << " settled=" << (rt.metrics.settled ? 1 : 0)
        << " migrations=" << rt.metrics.migrationCount
        << " rollbacks=" << rt.metrics.rollbackCount
        << " samples=" << window.samples
        << " pos_begin=" << window.posBegin
        << " pos_end=" << window.posEnd;
    if (extra != nullptr && extra[0] != '\0') oss << " note=" << extra;
    oss << " " << metricsString(rt.metrics, window.posEnd);
    appendLog(rt.logPath, oss.str());
}

} // namespace

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

    std::unique_ptr<DynamicTpotController> ctrl(new DynamicTpotController(socketPath, inference));
    DynamicTpotController *c = ctrl.get();
    ctrl->worker_ = std::thread([c]() { c->run(); });
    appendLog(logPath, "tpot_sched seq=0 state=START best=none gain_ms=0 note=controller_started");
    return ctrl;
#endif
}

DynamicTpotController::DynamicTpotController(const std::string &socketPath, RootLlmInference *inference)
    : socketPath_(socketPath), inference_(inference) {}

DynamicTpotController::~DynamicTpotController() {
    stop_.store(true);
    if (worker_.joinable()) worker_.join();
}

void DynamicTpotController::run() {
#ifdef _WIN32
    return;
#else
    ControllerRuntime rt;
    rt.cfg = loadSchedulerConfig();
    rt.pollMs = std::max(10, parseEnvInt("DLLAMA_TPOT_POLL_MS", 200));
    rt.timeoutMs = std::max(100, parseEnvInt("DLLAMA_TPOT_UDS_TIMEOUT_MS", 2000));
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
            const json perf = udsRequest(socketPath_, perfReq, rt.timeoutMs);
            const uint32_t statusPos = status.value("position", 0u);
            (void)updateWindowFromPerf(rt, perf, statusPos);

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

            std::vector<tpot::StageSnapshot> stages = buildStageSnapshots(rt, plan, window, status);
            tpot::Candidate bestTp = tpot::bestTpCandidate(stages, rt.cfg);
            tpot::Candidate bestPp = tpot::bestPpCandidate(stages, window.tpotMs, rt.cfg);
            if (bestPp.valid && rt.issuedPpKeys.count(ppCandidateKey(bestPp)) != 0u) {
                bestPp.valid = false;
                bestPp.reason = "pp boundary already issued";
            }
            tpot::Candidate best = tpot::betterCandidate(bestTp, bestPp);

            bool issued = false;
            const char *note = "";

            if (rt.pending.active) {
                const tpot::Candidate pendingForLog = rt.pending.candidate;
                bool verifyIssued = false;
                const uint32_t elapsed = window.posEnd >= rt.pending.startPos ? window.posEnd - rt.pending.startPos : 0u;
                if (elapsed >= (uint32_t)rt.cfg.rollbackWindow) {
                    const bool degraded = window.tpotMs > rt.pending.beforeTpotMs * 1.05;
                    if (degraded) {
                        tpot::Candidate reverse = reverseCandidate(rt.pending.candidate);
                        json resp;
                        if (issueCandidate(socketPath_, rt.timeoutMs, rt.seq++, reverse, &resp)) {
                            rt.metrics.rollbackCount += 1u;
                            rt.cooldownUntilPos = window.posEnd + (uint32_t)std::max(64, rt.cfg.cooldownTokens);
                            if (rt.pending.candidate.kind == tpot::CandidateKind::PP_MOVE) {
                                rt.riskPenaltyByStage[rt.pending.candidate.toStageIndex] += 0.1;
                                const NnStageConfig *target = findStageByIndex(plan, rt.pending.candidate.toStageIndex);
                                if (target != nullptr) {
                                    const uint32_t currentLayers = target->nLayers != 0u ? target->nLayers : (target->endLayer - target->startLayer);
                                    rt.softCapacityByStage[rt.pending.candidate.toStageIndex] =
                                        currentLayers > 1u ? currentLayers - 1u : 1u;
                                }
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
                logDecision(rt, window, pendingForLog, verifyIssued, note);
                std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                continue;
            }

            if (window.posEnd < rt.cooldownUntilPos) {
                note = "cooldown";
                logDecision(rt, window, best, false, note);
                std::this_thread::sleep_for(std::chrono::milliseconds(rt.pollMs));
                continue;
            }

            if (best.valid) {
                json resp;
                issued = issueCandidate(socketPath_, rt.timeoutMs, rt.seq++, best, &resp);
                if (issued) {
                    rt.metrics.migrationCount += 1u;
                    if (best.kind == tpot::CandidateKind::PP_MOVE) {
                        rt.issuedPpKeys.insert(ppCandidateKey(best));
                    }
                    rt.pending.active = true;
                    rt.pending.candidate = best;
                    rt.pending.beforeTpotMs = window.tpotMs;
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

            logDecision(rt, window, best, issued, note);
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
