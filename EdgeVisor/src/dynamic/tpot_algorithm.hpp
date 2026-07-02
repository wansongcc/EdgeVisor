#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dllama {
namespace dynamic_tpot {

enum class CandidateKind {
    NONE = 0,
    TP_HEAD,
    TP_FFN,
    PP_MOVE,
};

struct SchedulerConfig {
    int windowTokens = 16;
    int minSamples = 8;
    int cooldownTokens = 32;
    int rollbackWindow = 16;
    double ewmaAlpha = 0.2;
    double minPpGainMs = 5.0;
    double minTpGainMs = 2.0;
    double loadPenaltyBeta = 0.08;
    double ppRiskMarginMs = 0.0;
    double tpRiskMarginMs = 0.0;
    double ppMigrationCostMs = 0.0;
    double tpMigrationCostMs = 0.0;
    int expectedRemainingTokens = 128;
    uint32_t maxPpLayerMove = 1u;
    uint32_t maxHeadMove = 1u;
    uint32_t maxFfnMove = 256u;
};

struct NodeSnapshot {
    uint32_t nodeIndex = 0u;
    double timeMs = 0.0;
    double attnMs = 0.0;
    double ffnMs = 0.0;
    uint32_t kvHeads = 0u;
    uint32_t ffnUnits = 0u;
    bool canMoveHeadLeft = true;
    bool canMoveHeadRight = true;
};

struct StageSnapshot {
    uint32_t stageIndex = 0u;
    uint32_t rootNodeIndex = 0u;
    uint32_t startLayer = 0u;
    uint32_t endLayer = 0u;
    uint32_t nLayers = 0u;
    uint32_t softCapacity = 0u;
    bool hasFullWeights = true;
    double avgLayerMs = 0.0;
    double recentAvgMs = 0.0;
    double previousAvgMs = 0.0;
    double stageTimeMs = 0.0;
    double boundaryCommMs = 0.0;
    double leftBoundaryLayerMs = 0.0;
    double rightBoundaryLayerMs = 0.0;
    double riskPenalty = 0.0;
    std::vector<NodeSnapshot> nodes;
};

struct Candidate {
    CandidateKind kind = CandidateKind::NONE;
    bool valid = false;
    std::string reason;
    double gainMs = 0.0;
    double thresholdMs = 0.0;

    uint32_t stageIndex = 0u;
    uint32_t fromStageIndex = 0u;
    uint32_t toStageIndex = 0u;
    uint32_t fromNodeIndex = 0u;
    uint32_t toNodeIndex = 0u;
    uint32_t layerIndex = 0u;
    uint32_t headMove = 0u;
    uint32_t ffnMove = 0u;
};

double clampTrendPenalty(double recentAvgMs, double previousAvgMs);
double stageCostMs(const StageSnapshot &stage, uint32_t nLayers, const SchedulerConfig &cfg);
double ppDeltaInMs(const StageSnapshot &stage, const SchedulerConfig &cfg);
double ppDeltaOutMs(const StageSnapshot &stage, const SchedulerConfig &cfg);
double ppGainThresholdMs(double currentTpotMs, const SchedulerConfig &cfg);
double tpGainThresholdMs(const StageSnapshot &stage, const SchedulerConfig &cfg);

Candidate bestPpCandidate(const std::vector<StageSnapshot> &stages, double currentTpotMs, const SchedulerConfig &cfg);
Candidate bestTpCandidate(const std::vector<StageSnapshot> &stages, const SchedulerConfig &cfg);
Candidate betterCandidate(const Candidate &a, const Candidate &b);

void applyPpMove(std::vector<StageSnapshot> &stages, const Candidate &candidate);
void applyRollbackPenalty(StageSnapshot &target);
const char *candidateKindName(CandidateKind kind);

} // namespace dynamic_tpot
} // namespace dllama
