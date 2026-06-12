#include "ablation.hpp"

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>

using json = nlohmann::json;

static EdgeVisorAblationConfig g_ablationConfig;
static std::mutex g_ablationLogMutex;

static std::string envValue(const char *name) {
    const char *v = std::getenv(name);
    return (v != nullptr) ? std::string(v) : std::string();
}

static bool hasValue(const char *value) {
    return value != nullptr && value[0] != '\0';
}

const char *toString(ShadowKvMode mode) {
    switch (mode) {
        case ShadowKvMode::ENABLED: return "enabled";
        case ShadowKvMode::DISABLED_TRANSFER: return "disabled_transfer";
        case ShadowKvMode::DISABLED_RECOMPUTE: return "disabled_recompute";
    }
    return "enabled";
}

const char *toString(PointerSwizzlingMode mode) {
    switch (mode) {
        case PointerSwizzlingMode::ENABLED: return "enabled";
        case PointerSwizzlingMode::OPERATOR_REBUILD: return "operator_rebuild";
        case PointerSwizzlingMode::WEIGHT_REMATERIALIZE: return "weight_rematerialize";
    }
    return "enabled";
}

const char *toString(JitMode mode) {
    switch (mode) {
        case JitMode::ENABLED: return "enabled";
        case JitMode::STATIC: return "static";
        case JitMode::GREEDY: return "greedy";
        case JitMode::ORACLE: return "oracle";
    }
    return "enabled";
}

const char *toString(VgMode mode) {
    switch (mode) {
        case VgMode::ENABLED: return "enabled";
        case VgMode::FLAT: return "flat";
        case VgMode::RANDOM: return "random";
        case VgMode::PURE_PP: return "pure_pp";
        case VgMode::NO_ELASTIC_VG: return "no_elastic_vg";
    }
    return "enabled";
}

bool parseShadowKvMode(const std::string &value, ShadowKvMode &mode) {
    if (value == "enabled") { mode = ShadowKvMode::ENABLED; return true; }
    if (value == "disabled_transfer") { mode = ShadowKvMode::DISABLED_TRANSFER; return true; }
    if (value == "disabled_recompute") { mode = ShadowKvMode::DISABLED_RECOMPUTE; return true; }
    return false;
}

bool parsePointerSwizzlingMode(const std::string &value, PointerSwizzlingMode &mode) {
    if (value == "enabled") { mode = PointerSwizzlingMode::ENABLED; return true; }
    if (value == "operator_rebuild") { mode = PointerSwizzlingMode::OPERATOR_REBUILD; return true; }
    if (value == "weight_rematerialize") { mode = PointerSwizzlingMode::WEIGHT_REMATERIALIZE; return true; }
    return false;
}

bool parseJitMode(const std::string &value, JitMode &mode) {
    if (value == "enabled") { mode = JitMode::ENABLED; return true; }
    if (value == "static") { mode = JitMode::STATIC; return true; }
    if (value == "greedy") { mode = JitMode::GREEDY; return true; }
    if (value == "oracle") { mode = JitMode::ORACLE; return true; }
    return false;
}

bool parseVgMode(const std::string &value, VgMode &mode) {
    if (value == "enabled") { mode = VgMode::ENABLED; return true; }
    if (value == "flat") { mode = VgMode::FLAT; return true; }
    if (value == "random") { mode = VgMode::RANDOM; return true; }
    if (value == "pure_pp") { mode = VgMode::PURE_PP; return true; }
    if (value == "no_elastic_vg") { mode = VgMode::NO_ELASTIC_VG; return true; }
    return false;
}

static void applyJsonConfig(EdgeVisorAblationConfig &cfg, const json &j) {
    if (j.contains("shadow_kv_mode")) {
        ShadowKvMode mode;
        const std::string value = j.value("shadow_kv_mode", "enabled");
        if (!parseShadowKvMode(value, mode)) throw std::runtime_error("Invalid shadow_kv_mode: " + value);
        cfg.shadowKvMode = mode;
    }
    if (j.contains("pointer_swizzling_mode")) {
        PointerSwizzlingMode mode;
        const std::string value = j.value("pointer_swizzling_mode", "enabled");
        if (!parsePointerSwizzlingMode(value, mode)) throw std::runtime_error("Invalid pointer_swizzling_mode: " + value);
        cfg.pointerSwizzlingMode = mode;
    }
    if (j.contains("jit_mode")) {
        JitMode mode;
        const std::string value = j.value("jit_mode", "enabled");
        if (!parseJitMode(value, mode)) throw std::runtime_error("Invalid jit_mode: " + value);
        cfg.jitMode = mode;
    }
    if (j.contains("vg_mode")) {
        VgMode mode;
        const std::string value = j.value("vg_mode", "enabled");
        if (!parseVgMode(value, mode)) throw std::runtime_error("Invalid vg_mode: " + value);
        cfg.vgMode = mode;
    }
    cfg.fallbackPolicy = j.value("fallback_policy", cfg.fallbackPolicy);
    cfg.ablationLogPath = j.value("ablation_log_path", cfg.ablationLogPath);
    cfg.experimentId = j.value("experiment_id", cfg.experimentId);
}

static void applyEnvConfig(EdgeVisorAblationConfig &cfg) {
    const std::string shadow = envValue("EDGEVISOR_SHADOW_KV_MODE");
    if (!shadow.empty() && !parseShadowKvMode(shadow, cfg.shadowKvMode)) {
        throw std::runtime_error("Invalid EDGEVISOR_SHADOW_KV_MODE: " + shadow);
    }
    const std::string pointer = envValue("EDGEVISOR_POINTER_SWIZZLING_MODE");
    if (!pointer.empty() && !parsePointerSwizzlingMode(pointer, cfg.pointerSwizzlingMode)) {
        throw std::runtime_error("Invalid EDGEVISOR_POINTER_SWIZZLING_MODE: " + pointer);
    }
    const std::string jit = envValue("EDGEVISOR_JIT_MODE");
    if (!jit.empty() && !parseJitMode(jit, cfg.jitMode)) {
        throw std::runtime_error("Invalid EDGEVISOR_JIT_MODE: " + jit);
    }
    const std::string vg = envValue("EDGEVISOR_VG_MODE");
    if (!vg.empty() && !parseVgMode(vg, cfg.vgMode)) {
        throw std::runtime_error("Invalid EDGEVISOR_VG_MODE: " + vg);
    }
    const std::string fallback = envValue("EDGEVISOR_FALLBACK_POLICY");
    if (!fallback.empty()) cfg.fallbackPolicy = fallback;
    const std::string logPath = envValue("EDGEVISOR_ABLATION_LOG_PATH");
    if (!logPath.empty()) cfg.ablationLogPath = logPath;
    const std::string experiment = envValue("EDGEVISOR_EXPERIMENT_ID");
    if (!experiment.empty()) cfg.experimentId = experiment;
}

EdgeVisorAblationConfig edgevisorAblationConfigFromSources(
    const char *jsonPath,
    const char *shadowKvModeCli,
    const char *pointerSwizzlingModeCli,
    const char *jitModeCli,
    const char *vgModeCli,
    const char *fallbackPolicyCli,
    const char *ablationLogPathCli,
    const char *experimentIdCli) {
    EdgeVisorAblationConfig cfg;

    if (hasValue(jsonPath)) {
        std::ifstream in(jsonPath);
        if (!in.is_open()) throw std::runtime_error(std::string("Cannot open ablation config: ") + jsonPath);
        json j;
        in >> j;
        applyJsonConfig(cfg, j);
    }

    applyEnvConfig(cfg);

    if (hasValue(shadowKvModeCli)) {
        const std::string value(shadowKvModeCli);
        if (!parseShadowKvMode(value, cfg.shadowKvMode)) throw std::runtime_error("Invalid --shadow-kv-mode: " + value);
    }
    if (hasValue(pointerSwizzlingModeCli)) {
        const std::string value(pointerSwizzlingModeCli);
        if (!parsePointerSwizzlingMode(value, cfg.pointerSwizzlingMode)) throw std::runtime_error("Invalid --pointer-swizzling-mode: " + value);
    }
    if (hasValue(jitModeCli)) {
        const std::string value(jitModeCli);
        if (!parseJitMode(value, cfg.jitMode)) throw std::runtime_error("Invalid --jit-mode: " + value);
    }
    if (hasValue(vgModeCli)) {
        const std::string value(vgModeCli);
        if (!parseVgMode(value, cfg.vgMode)) throw std::runtime_error("Invalid --vg-mode: " + value);
    }
    if (hasValue(fallbackPolicyCli)) cfg.fallbackPolicy = fallbackPolicyCli;
    if (hasValue(ablationLogPathCli)) cfg.ablationLogPath = ablationLogPathCli;
    if (hasValue(experimentIdCli)) cfg.experimentId = experimentIdCli;

    return cfg;
}

void setEdgeVisorAblationConfig(const EdgeVisorAblationConfig &config) {
    g_ablationConfig = config;
}

const EdgeVisorAblationConfig &getEdgeVisorAblationConfig() {
    return g_ablationConfig;
}

json edgevisorAblationConfigToJson(const EdgeVisorAblationConfig &config) {
    return json{
        {"shadow_kv_mode", toString(config.shadowKvMode)},
        {"pointer_swizzling_mode", toString(config.pointerSwizzlingMode)},
        {"jit_mode", toString(config.jitMode)},
        {"vg_mode", toString(config.vgMode)},
        {"fallback_policy", config.fallbackPolicy},
        {"ablation_log_path", config.ablationLogPath},
        {"experiment_id", config.experimentId},
        {"ablation_variant", edgevisorAblationVariantName(config)}
    };
}

std::string edgevisorAblationVariantName(const EdgeVisorAblationConfig &config) {
    return std::string("shadow_") + toString(config.shadowKvMode) +
           "__pointer_" + toString(config.pointerSwizzlingMode) +
           "__jit_" + toString(config.jitMode) +
           "__vg_" + toString(config.vgMode);
}

bool edgevisorAblationLogEnabled() {
    return !g_ablationConfig.ablationLogPath.empty();
}

void edgevisorAblationLogEvent(const EdgeVisorAblationEvent &event) {
    const EdgeVisorAblationConfig &cfg = getEdgeVisorAblationConfig();
    if (cfg.ablationLogPath.empty()) return;

    json j = edgevisorAblationConfigToJson(cfg);
    j["event_id"] = event.eventId;
    j["trigger_pos"] = event.triggerPos;
    j["trigger_layer"] = event.triggerLayer;
    j["affected_stage"] = event.affectedStage;
    j["from_node"] = event.fromNode;
    j["to_node"] = event.toNode;
    j["selected_policy"] = event.selectedPolicy;
    j["t_decision_ms"] = event.tDecisionMs;
    j["t_state_prepare_ms"] = event.tStatePrepareMs;
    j["t_bind_ms"] = event.tBindMs;
    j["t_command_ms"] = event.tCommandMs;
    j["t_apply_ms"] = event.tApplyMs;
    j["t_recover_ms"] = event.tRecoverMs;
    j["stall_time_ms"] = event.stallTimeMs;
    j["state_transfer_bytes"] = event.stateTransferBytes;
    j["recompute_tokens_or_layers"] = event.recomputeTokensOrLayers;
    j["binding_update_count"] = event.bindingUpdateCount;
    j["vg_mapping_before"] = event.vgMappingBefore;
    j["vg_mapping_after"] = event.vgMappingAfter;
    j["physical_device_group"] = event.physicalDeviceGroup;
    j["logical_group"] = event.logicalGroup;
    j["fallback_reason"] = event.fallbackReason;
    j["materialized_bytes"] = event.materializedBytes;
    j["candidate_count"] = event.candidateCount;
    j["rejected_moves"] = event.rejectedMoves;
    j["fallback_count"] = event.fallbackCount;
    j["apply_success"] = event.applySuccess;

    std::lock_guard<std::mutex> lock(g_ablationLogMutex);
    std::ofstream out(cfg.ablationLogPath.c_str(), std::ios::app);
    if (out.is_open()) {
        out << j.dump() << "\n";
    }
}

void edgevisorAblationLogSimpleEvent(const std::string &eventId, const std::string &selectedPolicy) {
    EdgeVisorAblationEvent event;
    event.eventId = eventId;
    event.selectedPolicy = selectedPolicy;
    edgevisorAblationLogEvent(event);
}
