#ifndef EDGEVISOR_ABLATION_HPP
#define EDGEVISOR_ABLATION_HPP

#include "json.hpp"
#include <cstdint>
#include <string>

enum class ShadowKvMode {
    ENABLED,
    DISABLED_TRANSFER,
    DISABLED_RECOMPUTE,
};

enum class PointerSwizzlingMode {
    ENABLED,
    OPERATOR_REBUILD,
    WEIGHT_REMATERIALIZE,
};

enum class JitMode {
    ENABLED,
    STATIC,
    GREEDY,
    ORACLE,
};

enum class VgMode {
    ENABLED,
    FLAT,
    RANDOM,
    PURE_PP,
    NO_ELASTIC_VG,
};

struct EdgeVisorAblationConfig {
    ShadowKvMode shadowKvMode = ShadowKvMode::ENABLED;
    PointerSwizzlingMode pointerSwizzlingMode = PointerSwizzlingMode::ENABLED;
    JitMode jitMode = JitMode::ENABLED;
    VgMode vgMode = VgMode::ENABLED;
    std::string fallbackPolicy = "disabled_unless_necessary";
    std::string ablationLogPath;
    std::string experimentId = "default";
};

struct EdgeVisorAblationEvent {
    std::string eventId = "event";
    uint32_t triggerPos = 0xFFFFFFFFu;
    uint32_t triggerLayer = 0xFFFFFFFFu;
    uint32_t affectedStage = 0xFFFFFFFFu;
    uint32_t fromNode = 0xFFFFFFFFu;
    uint32_t toNode = 0xFFFFFFFFu;
    std::string selectedPolicy;
    double tDecisionMs = 0.0;
    double tStatePrepareMs = 0.0;
    double tBindMs = 0.0;
    double tCommandMs = 0.0;
    double tApplyMs = 0.0;
    double tRecoverMs = 0.0;
    double stallTimeMs = 0.0;
    uint64_t stateTransferBytes = 0u;
    uint64_t recomputeTokensOrLayers = 0u;
    uint64_t bindingUpdateCount = 0u;
    std::string vgMappingBefore;
    std::string vgMappingAfter;
    std::string physicalDeviceGroup;
    std::string logicalGroup;
    std::string fallbackReason;
    uint64_t materializedBytes = 0u;
    uint64_t candidateCount = 0u;
    uint64_t rejectedMoves = 0u;
    uint64_t fallbackCount = 0u;
    bool applySuccess = true;
};

const char *toString(ShadowKvMode mode);
const char *toString(PointerSwizzlingMode mode);
const char *toString(JitMode mode);
const char *toString(VgMode mode);

bool parseShadowKvMode(const std::string &value, ShadowKvMode &mode);
bool parsePointerSwizzlingMode(const std::string &value, PointerSwizzlingMode &mode);
bool parseJitMode(const std::string &value, JitMode &mode);
bool parseVgMode(const std::string &value, VgMode &mode);

EdgeVisorAblationConfig edgevisorAblationConfigFromSources(
    const char *jsonPath,
    const char *shadowKvModeCli,
    const char *pointerSwizzlingModeCli,
    const char *jitModeCli,
    const char *vgModeCli,
    const char *fallbackPolicyCli,
    const char *ablationLogPathCli,
    const char *experimentIdCli);

void setEdgeVisorAblationConfig(const EdgeVisorAblationConfig &config);
const EdgeVisorAblationConfig &getEdgeVisorAblationConfig();
nlohmann::json edgevisorAblationConfigToJson(const EdgeVisorAblationConfig &config);
std::string edgevisorAblationVariantName(const EdgeVisorAblationConfig &config);
bool edgevisorAblationLogEnabled();
void edgevisorAblationLogEvent(const EdgeVisorAblationEvent &event);
void edgevisorAblationLogSimpleEvent(const std::string &eventId, const std::string &selectedPolicy);

#endif
