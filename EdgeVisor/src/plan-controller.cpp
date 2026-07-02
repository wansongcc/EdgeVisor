#include "plan-controller.hpp"

#include "app.hpp"
#include "ablation.hpp"
#include "json.hpp"
#include "plan-command.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#endif

#ifndef _WIN32
#include <errno.h>
#endif

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

using json = nlohmann::json;

// Layer-prof snapshot reader (shared mmap file written by stage root).
// Keep this standalone to avoid pulling in nn-network headers.
static constexpr uint32_t kLayerPerfSnapshotMagic = 0x53504c44u; // 'DLPS'
static constexpr uint32_t kLayerPerfSnapshotVersion = 1u;
static constexpr uint32_t kLayerPerfMsgMagic = 0x52504c44u; // 'DLPR'
static constexpr uint32_t kLayerPerfMsgVersion = 1u;

struct LayerPerfSnapshotHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t stageIndex;
    uint32_t rootNodeIndex;
    uint32_t nLayers;
    uint32_t nStageNodes;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t epoch;
};

struct LayerPerfMsg {
    uint32_t magic;
    uint32_t version;
    uint32_t stageIndex;
    uint32_t nodeIndex;
    uint32_t layerIndex;
    uint32_t attnUs;
    uint32_t ffnUs;
};

static bool readFullFd(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    size_t left = len;
    while (left > 0) {
        ssize_t n = ::read(fd, p, left);
        if (n < 0) {
#ifndef _WIN32
            if (errno == EINTR) continue;
#endif
            return false;
        }
        if (n == 0) return false;
        p += (size_t)n;
        left -= (size_t)n;
    }
    return true;
}

static std::string defaultLayerPerfPath(uint32_t stageIndex, uint32_t rootNodeIndex) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "/tmp/dllama_layer_prof_stage%u_root%u.bin", (unsigned)stageIndex, (unsigned)rootNodeIndex);
    return std::string(buf);
}

static json readLayerPerfSnapshotJson(const json &req) {
    // Request fields:
    // - path (optional): explicit snapshot path
    // - stageIndex/rootNodeIndex (optional): for default path when env/path not set
    // - layerIndex (optional): return only a single layer row
    // - all (optional bool): return full [layer][node] table

    std::string path;
    if (req.contains("path")) {
        path = req.value("path", "");
    }
    if (path.empty()) {
        const char *env = std::getenv("DLLAMA_LAYER_PROF_PATH");
        if (env != nullptr && env[0] != '\0') path = std::string(env);
    }
    if (path.empty()) {
        const uint32_t stageIndex = req.value("stageIndex", 0u);
        const uint32_t rootNodeIndex = req.value("rootNodeIndex", 0u);
        path = defaultLayerPerfPath(stageIndex, rootNodeIndex);
    }

#ifdef _WIN32
    (void)path;
    throw std::runtime_error("layer_prof snapshot is not supported on Windows");
#else
    const bool all = req.value("all", false);
    const bool hasLayer = req.contains("layerIndex");
    const uint32_t layerIndex = req.value("layerIndex", 0u);

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(std::string("open snapshot failed: ") + path);
    }

    LayerPerfSnapshotHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    if (!readFullFd(fd, &hdr, sizeof(hdr))) {
        ::close(fd);
        throw std::runtime_error("read snapshot header failed");
    }
    if (hdr.magic != kLayerPerfSnapshotMagic || hdr.version != kLayerPerfSnapshotVersion) {
        ::close(fd);
        throw std::runtime_error("bad snapshot magic/version");
    }
    if (hdr.nLayers == 0u || hdr.nStageNodes == 0u) {
        ::close(fd);
        throw std::runtime_error("bad snapshot dimensions");
    }

    json out = json{{"path", path}, {"epoch", hdr.epoch}};
    out["header"] = json{
        {"magic", hdr.magic},
        {"version", hdr.version},
        {"stageIndex", hdr.stageIndex},
        {"rootNodeIndex", hdr.rootNodeIndex},
        {"nLayers", hdr.nLayers},
        {"nStageNodes", hdr.nStageNodes},
        {"epoch", hdr.epoch},
    };

    const size_t headerSize = sizeof(LayerPerfSnapshotHeader);
    const size_t rowSize = (size_t)hdr.nStageNodes * sizeof(LayerPerfMsg);
    const size_t tableSize = (size_t)hdr.nLayers * rowSize;

    auto readRow = [&](uint32_t li) -> json {
        if (li >= hdr.nLayers) throw std::runtime_error("layerIndex out of range");
        const off_t off = (off_t)(headerSize + (size_t)li * rowSize);
        if (::lseek(fd, off, SEEK_SET) < 0) throw std::runtime_error("lseek failed");
        std::vector<LayerPerfMsg> row(hdr.nStageNodes);
        if (!readFullFd(fd, row.data(), row.size() * sizeof(LayerPerfMsg))) {
            throw std::runtime_error("read row failed");
        }
        json arr = json::array();
        for (uint32_t i = 0; i < hdr.nStageNodes; ++i) {
            const LayerPerfMsg &m = row[i];
            if (m.magic != kLayerPerfMsgMagic || m.version != kLayerPerfMsgVersion) {
                // Keep placeholder to preserve indices.
                arr.push_back(json{{"ok", false}});
                continue;
            }
            arr.push_back(json{
                {"ok", true},
                {"stageIndex", m.stageIndex},
                {"nodeIndex", m.nodeIndex},
                {"layerIndex", m.layerIndex},
                {"attnUs", m.attnUs},
                {"ffnUs", m.ffnUs},
            });
        }
        return arr;
    };

    if (all) {
        // Read the full table.
        out["layers"] = json::array();
        for (uint32_t li = 0; li < hdr.nLayers; ++li) {
            out["layers"].push_back(readRow(li));
        }
    } else {
        if (!hasLayer) {
            // Default behavior: require an explicit layerIndex to keep response small.
            ::close(fd);
            throw std::runtime_error("missing layerIndex (or set all=true)");
        }
        out["layerIndex"] = layerIndex;
        out["nodes"] = readRow(layerIndex);
    }

    // Best-effort: validate file is at least header+table.
    (void)tableSize;
    ::close(fd);
    return out;
#endif
}

static bool readLine(int fd, std::string &out) {
    out.clear();
    char ch;
    while (true) {
        ssize_t n = ::read(fd, &ch, 1);
        if (n == 0) return !out.empty();
        if (n < 0) return false;
        if (ch == '\n') return true;
        out.push_back(ch);
        if (out.size() > 1024 * 256) return false;
    }
}

static void writeLine(int fd, const std::string &s) {
    auto writeFull = [&](const void *data, size_t len) -> bool {
        const char *p = (const char *)data;
        size_t left = len;
        while (left > 0) {
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
    };

    (void)writeFull(s.data(), s.size());
    const char nl = '\n';
    (void)writeFull(&nl, 1);
}

static uint32_t parseU32(const json &j, const char *key, uint32_t fallback) {
    if (!j.contains(key)) return fallback;
    try {
        return j.at(key).get<uint32_t>();
    } catch (...) {
        return fallback;
    }
}

static bool parseBool(const json &j, const char *key, bool fallback) {
    if (!j.contains(key)) return fallback;
    try {
        if (j.at(key).is_boolean()) {
            return j.at(key).get<bool>();
        }
        if (j.at(key).is_number_integer() || j.at(key).is_number_unsigned()) {
            return j.at(key).get<int>() != 0;
        }
    } catch (...) {
        return fallback;
    }
    return fallback;
}

static double parseDouble(const json &j, const char *key, double fallback) {
    if (!j.contains(key)) return fallback;
    try {
        if (j.at(key).is_number()) {
            return j.at(key).get<double>();
        }
    } catch (...) {
        return fallback;
    }
    return fallback;
}

static std::string modeToStr(uint32_t mode) {
    if (mode == PLAN_CMD_MODE_EXACT) return "exact";
    if (mode == PLAN_CMD_MODE_NEXT_BARRIER) return "next_barrier";
    return "none";
}

static json cmdToJson(const PlanCommand &c) {
    json out = json{
        {"seq", c.seq},
        {"mode", modeToStr(c.mode)},
        {"stageIndex", c.stageIndex},
        {"triggerPos", c.triggerPos},
        {"triggerLayer", c.triggerLayer},
        {"fromNodeIndex", c.fromNodeIndex},
        {"toNodeIndex", c.toNodeIndex},
        {"cmdKind", c.cmdKind},
        {"nHeadsToMove", c.nHeadsToMove},
        {"nFfnToMove", c.nFfnToMove},
        {"reserved0", c.reserved0}
    };

    if (c.version == DLLAMA_PLAN_CMD_VERSION_V2 && c.nMoves != 0u) {
        json arr = json::array();
        const uint32_t n = c.nMoves;
        for (uint32_t i = 0; i < n && i < DLLAMA_PLAN_CMD_MAX_MOVES; ++i) {
            const PlanMove &m = c.moves[i];
            arr.push_back(json{
                {"fromNodeIndex", m.fromNodeIndex},
                {"toNodeIndex", m.toNodeIndex},
                {"cmdKind", m.cmdKind},
                {"headMove", m.headMove},
                {"ffnMove", m.ffnMove}
            });
        }
        out["moves"] = arr;
    }
    return out;
}

static bool parseMode(const std::string &s, uint32_t &out) {
    if (s == "exact") { out = PLAN_CMD_MODE_EXACT; return true; }
    if (s == "next" || s == "next_barrier") { out = PLAN_CMD_MODE_NEXT_BARRIER; return true; }
    if (s == "none") { out = PLAN_CMD_MODE_NONE; return true; }
    return false;
}

static std::string nodePairString(const PlanCommand &cmd) {
    std::ostringstream oss;
    oss << "stage=" << cmd.stageIndex << ",from=" << cmd.fromNodeIndex << ",to=" << cmd.toNodeIndex;
    return oss.str();
}

static void populateAblationPlanEvent(
    EdgeVisorAblationEvent &ev,
    const PlanCommand &cmd,
    const json &jcmd,
    const char *eventId,
    uint64_t bindingCount) {
    const EdgeVisorAblationConfig &cfg = getEdgeVisorAblationConfig();
    ev.eventId = eventId;
    ev.triggerPos = cmd.triggerPos;
    ev.triggerLayer = cmd.triggerLayer;
    ev.affectedStage = cmd.stageIndex;
    ev.fromNode = cmd.fromNodeIndex;
    ev.toNode = cmd.toNodeIndex;
    ev.selectedPolicy = std::string("jit_") + toString(cfg.jitMode);
    ev.bindingUpdateCount = bindingCount;
    ev.tDecisionMs = parseDouble(jcmd, "tDecisionMs", 0.0);
    ev.tStatePrepareMs = parseDouble(jcmd, "tStatePrepareMs", 0.0);
    ev.tCommandMs = parseDouble(jcmd, "tCommandMs", 0.0);
    ev.tApplyMs = parseDouble(jcmd, "tApplyMs", 0.0);
    ev.tRecoverMs = parseDouble(jcmd, "tRecoverMs", 0.0);
    ev.stallTimeMs = parseDouble(jcmd, "stallTimeMs", 0.0);
    ev.stateTransferBytes = parseU32(jcmd, "stateTransferBytes", 0u);
    ev.recomputeTokensOrLayers = parseU32(jcmd, "recomputeTokensOrLayers", 0u);
    ev.candidateCount = parseU32(jcmd, "candidateCount", 1u);
    ev.rejectedMoves = parseU32(jcmd, "rejectedMoves", 0u);
    ev.fallbackCount = parseU32(jcmd, "fallbackCount", 0u);
    ev.vgMappingBefore = jcmd.value("vgMappingBefore", std::string("vg_") + toString(cfg.vgMode));
    ev.vgMappingAfter = jcmd.value("vgMappingAfter", std::string("vg_") + toString(cfg.vgMode));
    ev.physicalDeviceGroup = jcmd.value("physicalDeviceGroup", nodePairString(cmd));
    ev.logicalGroup = jcmd.value("logicalGroup", std::string("stage_") + std::to_string(cmd.stageIndex));
    ev.fallbackReason = jcmd.value("fallbackReason", jcmd.value("fallback_reason", std::string()));
    ev.applySuccess = true;

    if (cfg.shadowKvMode == ShadowKvMode::DISABLED_TRANSFER) {
        ev.fallbackReason = "shadow_kv_disabled_transfer_fallback";
    } else if (cfg.shadowKvMode == ShadowKvMode::DISABLED_RECOMPUTE) {
        ev.fallbackReason = "shadow_kv_disabled_recompute_fallback";
    }
    if (cfg.jitMode == JitMode::STATIC) {
        ev.fallbackReason = ev.fallbackReason.empty() ? "static_policy_replaces_runtime_jit" : ev.fallbackReason + ";static_policy_replaces_runtime_jit";
    } else if (cfg.jitMode == JitMode::GREEDY) {
        ev.fallbackReason = ev.fallbackReason.empty() ? "greedy_policy_ignores_enactment_cost" : ev.fallbackReason + ";greedy_policy_ignores_enactment_cost";
    } else if (cfg.jitMode == JitMode::ORACLE) {
        ev.fallbackReason = ev.fallbackReason.empty() ? "oracle_policy_for_experiment_upper_bound" : ev.fallbackReason + ";oracle_policy_for_experiment_upper_bound";
    }
    if (cfg.vgMode != VgMode::ENABLED) {
        const std::string reason = std::string("vg_mode_") + toString(cfg.vgMode);
        ev.fallbackReason = ev.fallbackReason.empty() ? reason : ev.fallbackReason + ";" + reason;
    }
}

static bool planCommandHasHeadMove(const PlanCommand &cmd) {
    if (cmd.version == DLLAMA_PLAN_CMD_VERSION_V2 && cmd.nMoves > 0u) {
        for (uint32_t i = 0; i < cmd.nMoves && i < DLLAMA_PLAN_CMD_MAX_MOVES; ++i) {
            if (cmd.moves[i].headMove > 0u) return true;
        }
        return false;
    }
    return (cmd.cmdKind == PLAN_CMD_KIND_HEAD || cmd.cmdKind == PLAN_CMD_KIND_BOTH) && cmd.nHeadsToMove > 0u;
}

static void logRejectedJitPlan(
    const PlanCommand &cmd,
    const json &jcmd,
    const char *reason,
    uint64_t bindingCount) {
    EdgeVisorAblationEvent ev;
    populateAblationPlanEvent(ev, cmd, jcmd, "jit_plan_rejected", bindingCount);
    ev.applySuccess = false;
    ev.fallbackReason = reason;
    ev.rejectedMoves = bindingCount;
    ev.fallbackCount = 1u;
    edgevisorAblationLogEvent(ev);
}

std::unique_ptr<PlanUdsController> PlanUdsController::start(const std::string &socketPath, RootLlmInference *inference) {
#ifdef _WIN32
    (void)socketPath;
    (void)inference;
    return nullptr;
#else
    std::unique_ptr<PlanUdsController> ctrl(new PlanUdsController(socketPath, inference));

    // Launch background thread (C++11: no init-capture).
    PlanUdsController *c = ctrl.get();
    ctrl->worker_ = std::thread([c]() { c->run(); });

    return ctrl;
#endif
}

PlanUdsController::PlanUdsController(const std::string &socketPath, RootLlmInference *inference)
    : socketPath_(socketPath), inference_(inference) {}

PlanUdsController::~PlanUdsController() {
#ifndef _WIN32
    stop_.store(true);
    if (!socketPath_.empty()) {
        int wakeFd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (wakeFd >= 0) {
            sockaddr_un addr{};
            addr.sun_family = AF_UNIX;
            std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socketPath_.c_str());
            (void)::connect(wakeFd, (sockaddr *)&addr, sizeof(addr));
            ::close(wakeFd);
        }
    }
    closeServer();
    if (worker_.joinable()) {
        worker_.join();
    }
#endif
}

void PlanUdsController::closeServer() {
#ifndef _WIN32
    if (serverFd_ >= 0) {
        ::close(serverFd_);
        serverFd_ = -1;
    }
    if (!socketPath_.empty()) {
        ::unlink(socketPath_.c_str());
    }
#endif
}

void PlanUdsController::run() {
#ifdef _WIN32
    return;
#else
    if (socketPath_.empty()) return;

    ::unlink(socketPath_.c_str());

    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        std::perror("socket(AF_UNIX)");
        return;
    }

    sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (socketPath_.size() >= sizeof(addr.sun_path)) {
        std::fprintf(stderr, "[plan-uds] socket path too long: %s\n", socketPath_.c_str());
        ::close(fd);
        return;
    }
    std::strncpy(addr.sun_path, socketPath_.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(fd, (sockaddr *)&addr, sizeof(addr)) != 0) {
        std::perror("bind(AF_UNIX)");
        ::close(fd);
        return;
    }

    if (::listen(fd, 16) != 0) {
        std::perror("listen(AF_UNIX)");
        ::close(fd);
        return;
    }

    serverFd_ = fd;
    std::fprintf(stderr, "[plan-uds] listening on %s\n", socketPath_.c_str());

    while (!stop_.load()) {
        int cfd = ::accept(fd, nullptr, nullptr);
        if (cfd < 0) {
            if (stop_.load()) break;
            continue;
        }
        if (stop_.load()) {
            ::close(cfd);
            break;
        }

        std::string line;
        if (!readLine(cfd, line)) {
            ::close(cfd);
            continue;
        }

        json resp;
        try {
            json req = json::parse(line);
            const std::string op = req.value("op", "");

            if (op == "set_plan") {
                if (!req.contains("cmd")) throw std::runtime_error("missing cmd");
                const json &jcmd = req.at("cmd");

                PlanCommand cmd = makeEmptyPlanCommand();
                cmd.magic = DLLAMA_PLAN_CMD_MAGIC;
                cmd.version = DLLAMA_PLAN_CMD_VERSION_V2;
                cmd.seq = parseU32(jcmd, "seq", 0u);

                const std::string modeStr = jcmd.value("mode", "");
                if (!parseMode(modeStr, cmd.mode)) {
                    throw std::runtime_error("bad mode (use exact/next_barrier)");
                }

                cmd.stageIndex = parseU32(jcmd, "stageIndex", 0u);

                const bool hasMovesField = jcmd.contains("moves");

                // Legacy single-edge fields are used only when moves[] is absent.
                // An explicit empty moves[] means a no-op command; do not fall back
                // to the historical default 0->1 head+ffn migration.
                if (!hasMovesField) {
                    cmd.fromNodeIndex = parseU32(jcmd, "fromNodeIndex", 0u);
                    cmd.toNodeIndex = parseU32(jcmd, "toNodeIndex", 1u);
                    cmd.cmdKind = parseU32(jcmd, "cmdKind", PLAN_CMD_KIND_BOTH);
                    cmd.nHeadsToMove = parseU32(jcmd, "nHeadsToMove", 1u);
                    cmd.nFfnToMove = parseU32(jcmd, "nFfnToMove", 256u);
                } else {
                    cmd.fromNodeIndex = parseU32(jcmd, "fromNodeIndex", 0u);
                    cmd.toNodeIndex = parseU32(jcmd, "toNodeIndex", 0u);
                    cmd.cmdKind = 0u;
                    cmd.nHeadsToMove = 0u;
                    cmd.nFfnToMove = 0u;
                }

                // Optional v2 move list: moves=[{fromNodeIndex,toNodeIndex,cmdKind,headMove,ffnMove}, ...]
                cmd.nMoves = 0u;
                if (hasMovesField) {
                    const json &jmoves = jcmd.at("moves");
                    if (!jmoves.is_array()) {
                        throw std::runtime_error("moves must be an array");
                    }
                    const size_t count = jmoves.size();
                    if (count > (size_t)DLLAMA_PLAN_CMD_MAX_MOVES) {
                        throw std::runtime_error("too many moves (exceeds DLLAMA_PLAN_CMD_MAX_MOVES)");
                    }
                    cmd.nMoves = (uint32_t)count;
                    for (size_t i = 0; i < count; ++i) {
                        const json &m = jmoves.at(i);
                        PlanMove pm;
                        std::memset(&pm, 0, sizeof(pm));
                        pm.fromNodeIndex = parseU32(m, "fromNodeIndex", 0u);
                        pm.toNodeIndex = parseU32(m, "toNodeIndex", 0u);
                        pm.cmdKind = parseU32(m, "cmdKind", PLAN_CMD_KIND_BOTH);
                        pm.headMove = parseU32(m, "headMove", 0u);
                        pm.ffnMove = parseU32(m, "ffnMove", 0u);
                        cmd.moves[i] = pm;
                    }
                }

                if (cmd.mode == PLAN_CMD_MODE_EXACT) {
                    if (!jcmd.contains("triggerPos") || !jcmd.contains("triggerLayer")) {
                        throw std::runtime_error("exact mode requires triggerPos + triggerLayer");
                    }
                    cmd.triggerPos = parseU32(jcmd, "triggerPos", 0xFFFFFFFFu);
                    cmd.triggerLayer = parseU32(jcmd, "triggerLayer", 0xFFFFFFFFu);
                } else {
                    cmd.triggerPos = 0xFFFFFFFFu;
                    cmd.triggerLayer = 0xFFFFFFFFu;
                }

                const EdgeVisorAblationConfig &cfg = getEdgeVisorAblationConfig();
                const bool hasHeadMove = planCommandHasHeadMove(cmd);
                if (cfg.disableShardingController && hasHeadMove) {
                    const uint64_t rejectedCount =
                        (cmd.version == DLLAMA_PLAN_CMD_VERSION_V2 && cmd.nMoves != 0u) ? cmd.nMoves : 1u;
                    logRejectedJitPlan(cmd, jcmd, "sharding_controller_disabled", rejectedCount);
                    resp = json{{"ok", false}, {"rejected", true}, {"reason", "sharding_controller_disabled"}, {"cmd", cmdToJson(cmd)}};
                    writeLine(cfd, resp.dump());
                    ::close(cfd);
                    continue;
                }

                const uint64_t cacheSeq = planCommandCache().store(cmd);
                EdgeVisorAblationEvent ev;
                populateAblationPlanEvent(
                    ev,
                    cmd,
                    jcmd,
                    "plan_command_emit",
                    (cmd.version == DLLAMA_PLAN_CMD_VERSION_V2 && cmd.nMoves != 0u) ? cmd.nMoves : 1u);
                edgevisorAblationLogEvent(ev);
                resp = json{{"ok", true}, {"cacheSeq", cacheSeq}, {"cmd", cmdToJson(cmd)}};
            } else if (op == "set_pp_migration") {
                if (!req.contains("cmd")) throw std::runtime_error("missing cmd");
                const json &jcmd = req.at("cmd");

                PlanCommand cmd = makeEmptyPlanCommand();
                cmd.magic = DLLAMA_PLAN_CMD_MAGIC;
                cmd.version = DLLAMA_PLAN_CMD_VERSION_V2;
                cmd.seq = parseU32(jcmd, "seq", 0u);

                const std::string modeStr = jcmd.value("mode", "");
                if (!parseMode(modeStr, cmd.mode)) {
                    throw std::runtime_error("bad mode (use exact/next_barrier)");
                }

                cmd.stageIndex = parseU32(jcmd, "stageIndex", 0xFFFFFFFFu);
                cmd.fromNodeIndex = parseU32(jcmd, "fromNodeIndex", 0u);
                cmd.toNodeIndex = parseU32(jcmd, "toNodeIndex", 1u);

                // cmdKind=0 + no moves => reserved for PP layer migration control path.
                cmd.cmdKind = 0u;
                cmd.nHeadsToMove = 0u;
                cmd.nFfnToMove = 0u;
                cmd.nMoves = 0u;
                cmd.reserved0 = parseU32(jcmd, "layerCount", 1u);
                if (cmd.reserved0 == 0u) {
                    throw std::runtime_error("layerCount must be >= 1");
                }

                if (cmd.mode == PLAN_CMD_MODE_EXACT) {
                    if (!jcmd.contains("triggerPos")) {
                        throw std::runtime_error("exact mode requires triggerPos");
                    }
                    cmd.triggerPos = parseU32(jcmd, "triggerPos", 0xFFFFFFFFu);
                    cmd.triggerLayer = parseU32(jcmd, "triggerLayer", 0xFFFFFFFFu);
                } else {
                    cmd.triggerPos = 0xFFFFFFFFu;
                    cmd.triggerLayer = 0xFFFFFFFFu;
                }

                const EdgeVisorAblationConfig &cfg = getEdgeVisorAblationConfig();
                if (cfg.disablePipelineBalancer) {
                    logRejectedJitPlan(cmd, jcmd, "pipeline_balancer_disabled", 1u);
                    resp = json{{"ok", false}, {"rejected", true}, {"reason", "pipeline_balancer_disabled"}, {"cmd", cmdToJson(cmd)}, {"ppMigration", true}};
                    writeLine(cfd, resp.dump());
                    ::close(cfd);
                    continue;
                }

                const uint64_t cacheSeq = planCommandCache().store(cmd);
                EdgeVisorAblationEvent ev;
                populateAblationPlanEvent(ev, cmd, jcmd, "pp_migration_emit", 1u);
                edgevisorAblationLogEvent(ev);
                resp = json{{"ok", true}, {"cacheSeq", cacheSeq}, {"cmd", cmdToJson(cmd)}, {"ppMigration", true}, {"layerCount", cmd.reserved0}};
            } else if (op == "set_runtime_gate") {
                if (inference_ == nullptr) throw std::runtime_error("inference not available");
                const bool primaryEnabled = parseBool(req, "primaryEnabled", true);
                const bool redundantEnabled = parseBool(req, "redundantEnabled", false);
                inference_->setRuntimeLayerGate(primaryEnabled, redundantEnabled);
                resp = json{{"ok", true}, {"primaryEnabled", primaryEnabled}, {"redundantEnabled", redundantEnabled}};
            } else if (op == "set_primary_layer") {
                if (inference_ == nullptr) throw std::runtime_error("inference not available");
                const uint32_t layerIndex = parseU32(req, "layerIndex", 0u);
                const bool enabled = parseBool(req, "enabled", true);
                inference_->setPrimaryLayerEnabled(layerIndex, enabled);
                resp = json{{"ok", true}, {"layerIndex", layerIndex}, {"enabled", enabled}};
            } else if (op == "clear") {
                const uint64_t cacheSeq = planCommandCache().clear();
                resp = json{{"ok", true}, {"cacheSeq", cacheSeq}};
            } else if (op == "status") {
                const PlanCommandSnapshot snap = planCommandCache().load();
                resp = json{{"ok", true}, {"cacheSeq", snap.cacheSeq}, {"cmd", cmdToJson(snap.cmd)}};
                resp["enablePlanBarrier"] = getEnablePlanBarrier();
                if (inference_ != nullptr) {
                    resp["position"] = inference_->getPosition();
                    resp["batchSize"] = inference_->getBatchSize();
                    resp["perfSamples"] = (uint32_t)inference_->getLastPerf().size();
                    json pp = json::object();
                    pp["fromNodeIndex"] = inference_->getMigrationFromNodeIndex();
                    pp["toNodeIndex"] = inference_->getMigrationTargetNodeIndex();
                    pp["collectPos"] = inference_->getAsyncKvCollectPos();
                    pp["armedTriggerPos"] = inference_->getAsyncKvCollectPos();
                    pp["collectLayer"] = inference_->getAsyncKvCollectLayer();
                    pp["enabled"] = inference_->isPpMigrationEnabled();
                    pp["layerCount"] = inference_->getMigrationLayerCount();
                    pp["layerListPinnedByEnv"] = inference_->isMigrationLayerListPinnedByEnv();
                    pp["ackSeen"] = inference_->hasMigrationAck();
                    pp["ackPos"] = inference_->getMigrationAckPos();
                    pp["ackLayer"] = inference_->getMigrationAckLayer();
                    json layers = json::array();
                    for (NnUint layer : inference_->getMigrationLayers()) {
                        layers.push_back(layer);
                    }
                    pp["layers"] = layers;
                    resp["ppMigration"] = pp;
                }
            } else if (op == "perf") {
                resp = json{{"ok", true}};
                json arr = json::array();
                if (inference_ != nullptr) {
                    for (const auto &p : inference_->getLastPerf()) {
                        arr.push_back(json{
                            {"position", p.position},
                            {"batchSize", p.batchSize},
                            {"nodeIndex", p.nodeIndex},
                            {"stageIndex", p.stageIndex},
                            {"execUs", p.execUs},
                            {"syncUs", p.syncUs},
                            {"leftBoundaryLayerUs", p.leftBoundaryLayerUs},
                            {"rightBoundaryLayerUs", p.rightBoundaryLayerUs}
                        });
                    }
                }
                resp["perf"] = arr;
            } else if (op == "layer_prof") {
                // Reads the mmap snapshot file written by stage root and returns JSON.
                resp = json{{"ok", true}};
                resp["layer_prof"] = readLayerPerfSnapshotJson(req);
            } else if (op == "ping") {
                resp = json{{"ok", true}};
            } else {
                resp = json{{"ok", false}, {"error", "unknown op"}};
            }
        } catch (const std::exception &e) {
            resp = json{{"ok", false}, {"error", e.what()}};
        } catch (...) {
            resp = json{{"ok", false}, {"error", "unknown error"}};
        }

        writeLine(cfd, resp.dump());
        ::close(cfd);
    }

    closeServer();
#endif
}
