#include "dynamic/tpot_algorithm.hpp"

#include <algorithm>
#include <cmath>

namespace dllama {
namespace dynamic_tpot {

static double maxDouble(double a, double b) {
    return a > b ? a : b;
}

static uint32_t stageLayerCount(const StageSnapshot &stage) {
    if (stage.nLayers != 0u) return stage.nLayers;
    if (stage.endLayer > stage.startLayer) return stage.endLayer - stage.startLayer;
    return 0u;
}

double clampTrendPenalty(double recentAvgMs, double previousAvgMs) {
    if (recentAvgMs <= 0.0 || previousAvgMs <= 0.0) return 1.0;
    double r = recentAvgMs / previousAvgMs;
    if (r < 1.0) r = 1.0;
    if (r > 1.3) r = 1.3;
    return r;
}

double stageCostMs(const StageSnapshot &stage, uint32_t nLayers, const SchedulerConfig &cfg) {
    if (nLayers == 0u || stage.avgLayerMs <= 0.0) return 0.0;
    const uint32_t soft = stage.softCapacity != 0u ? stage.softCapacity : nLayers;
    const double over = nLayers > soft ? (double)(nLayers - soft) : 0.0;
    const double loadPenalty = 1.0 + cfg.loadPenaltyBeta * over;
    const double trendPenalty = clampTrendPenalty(stage.recentAvgMs, stage.previousAvgMs);
    return stage.avgLayerMs * (double)nLayers * loadPenalty * trendPenalty;
}

double ppDeltaInMs(const StageSnapshot &stage, const SchedulerConfig &cfg) {
    const uint32_t n = stageLayerCount(stage);
    return stageCostMs(stage, n + 1u, cfg) - stageCostMs(stage, n, cfg);
}

double ppDeltaOutMs(const StageSnapshot &stage, const SchedulerConfig &cfg) {
    const uint32_t n = stageLayerCount(stage);
    if (n == 0u) return 0.0;
    return stageCostMs(stage, n, cfg) - stageCostMs(stage, n - 1u, cfg);
}

double ppGainThresholdMs(double currentTpotMs, const SchedulerConfig &cfg) {
    return maxDouble(cfg.minPpGainMs, 0.03 * currentTpotMs);
}

double tpGainThresholdMs(const StageSnapshot &stage, const SchedulerConfig &cfg) {
    return maxDouble(cfg.minTpGainMs, 0.03 * stage.stageTimeMs);
}

static Candidate makeRejected(CandidateKind kind, const char *reason) {
    Candidate c;
    c.kind = kind;
    c.valid = false;
    c.reason = reason;
    return c;
}

static void considerBest(Candidate &best, const Candidate &candidate) {
    if (!candidate.valid) return;
    if (!best.valid || candidate.gainMs > best.gainMs) best = candidate;
}

static double migrationCostPerToken(double costMs, int expectedRemainingTokens) {
    if (costMs <= 0.0) return 0.0;
    const int denom = expectedRemainingTokens > 0 ? expectedRemainingTokens : 1;
    return costMs / (double)denom;
}

static double ppBoundaryDeltaOutMs(const StageSnapshot &source, const StageSnapshot &target, const SchedulerConfig &cfg) {
    double measured = 0.0;
    if (target.stageIndex > source.stageIndex) {
        measured = source.rightBoundaryLayerMs;
    } else {
        measured = source.leftBoundaryLayerMs;
    }
    return measured > 0.0 ? measured : ppDeltaOutMs(source, cfg);
}

Candidate bestPpCandidate(const std::vector<StageSnapshot> &stages, double currentTpotMs, const SchedulerConfig &cfg) {
    if (stages.size() < 2u) return makeRejected(CandidateKind::PP_MOVE, "need at least two stages");

    Candidate best;
    best.kind = CandidateKind::PP_MOVE;
    best.thresholdMs = ppGainThresholdMs(currentTpotMs, cfg);
    best.reason = "no profitable pp move";

    const double migCost = migrationCostPerToken(cfg.ppMigrationCostMs, cfg.expectedRemainingTokens);
    for (size_t i = 0u; i + 1u < stages.size(); ++i) {
        const StageSnapshot &left = stages[i];
        const StageSnapshot &right = stages[i + 1u];

        const StageSnapshot *sources[2] = {&left, &right};
        const StageSnapshot *targets[2] = {&right, &left};
        for (int d = 0; d < 2; ++d) {
            const StageSnapshot &source = *sources[d];
            const StageSnapshot &target = *targets[d];
            const uint32_t sourceLayers = stageLayerCount(source);
            if (sourceLayers <= cfg.maxPpLayerMove) continue;
            if (!target.hasFullWeights) continue;

            const double gain =
                ppBoundaryDeltaOutMs(source, target, cfg) -
                ppDeltaInMs(target, cfg) -
                target.boundaryCommMs -
                migCost -
                cfg.ppRiskMarginMs -
                target.riskPenalty * maxDouble(1.0, target.avgLayerMs);

            Candidate c;
            c.kind = CandidateKind::PP_MOVE;
            c.valid = gain > best.thresholdMs;
            c.reason = c.valid ? "" : "gain below threshold";
            c.gainMs = gain;
            c.thresholdMs = best.thresholdMs;
            c.fromStageIndex = source.stageIndex;
            c.toStageIndex = target.stageIndex;
            c.fromNodeIndex = source.rootNodeIndex;
            c.toNodeIndex = target.rootNodeIndex;
            if (target.stageIndex > source.stageIndex) {
                c.layerIndex = source.endLayer > source.startLayer ? source.endLayer - 1u : source.startLayer;
            } else {
                c.layerIndex = source.startLayer;
            }
            considerBest(best, c);
        }
    }
    return best;
}

static double nodeUnitMs(double preferredMs, double fallbackMs, uint32_t units) {
    if (units == 0u) return 0.0;
    const double base = preferredMs > 0.0 ? preferredMs : fallbackMs;
    if (base <= 0.0) return 0.0;
    return base / (double)units;
}

static double stageMaxAfterMove(
    const StageSnapshot &stage,
    size_t slowIdx,
    size_t fastIdx,
    double slowAfter,
    double fastAfter) {
    double out = 0.0;
    for (size_t i = 0u; i < stage.nodes.size(); ++i) {
        double t = stage.nodes[i].timeMs;
        if (i == slowIdx) t = slowAfter;
        if (i == fastIdx) t = fastAfter;
        if (t > out) out = t;
    }
    return out;
}

static void considerTpMove(
    const StageSnapshot &stage,
    const SchedulerConfig &cfg,
    size_t slowIdx,
    size_t fastIdx,
    bool headMove,
    Candidate &best) {
    if (slowIdx >= stage.nodes.size() || fastIdx >= stage.nodes.size()) return;
    const NodeSnapshot &slow = stage.nodes[slowIdx];
    const NodeSnapshot &fast = stage.nodes[fastIdx];
    const bool movingRight = fastIdx > slowIdx;
    if (headMove) {
        if (slow.kvHeads <= cfg.maxHeadMove) return;
        if (movingRight && !slow.canMoveHeadRight) return;
        if (!movingRight && !slow.canMoveHeadLeft) return;
    } else {
        if (slow.ffnUnits <= cfg.maxFfnMove) return;
    }

    const uint32_t units = headMove ? cfg.maxHeadMove : cfg.maxFfnMove;
    const double slowUnit = headMove
        ? nodeUnitMs(slow.attnMs, slow.timeMs, slow.kvHeads)
        : nodeUnitMs(slow.ffnMs, slow.timeMs, slow.ffnUnits);
    const double fastUnit = headMove
        ? nodeUnitMs(fast.attnMs, fast.timeMs, fast.kvHeads == 0u ? 1u : fast.kvHeads)
        : nodeUnitMs(fast.ffnMs, fast.timeMs, fast.ffnUnits == 0u ? units : fast.ffnUnits);
    if (slowUnit <= 0.0 || fastUnit <= 0.0) return;

    double oldMax = 0.0;
    for (size_t i = 0u; i < stage.nodes.size(); ++i) {
        if (stage.nodes[i].timeMs > oldMax) oldMax = stage.nodes[i].timeMs;
    }
    const double slowAfter = maxDouble(0.0, slow.timeMs - slowUnit * (double)units);
    const double fastAfter = fast.timeMs + fastUnit * (double)units;
    const double newMax = stageMaxAfterMove(stage, slowIdx, fastIdx, slowAfter, fastAfter);
    const double gain = oldMax - newMax - cfg.tpMigrationCostMs - cfg.tpRiskMarginMs;

    Candidate c;
    c.kind = headMove ? CandidateKind::TP_HEAD : CandidateKind::TP_FFN;
    c.valid = gain > best.thresholdMs;
    c.reason = c.valid ? "" : "gain below threshold";
    c.gainMs = gain;
    c.thresholdMs = best.thresholdMs;
    c.stageIndex = stage.stageIndex;
    c.fromNodeIndex = slow.nodeIndex;
    c.toNodeIndex = fast.nodeIndex;
    c.headMove = headMove ? units : 0u;
    c.ffnMove = headMove ? 0u : units;
    considerBest(best, c);
}

Candidate bestTpCandidate(const std::vector<StageSnapshot> &stages, const SchedulerConfig &cfg) {
    Candidate best;
    best.kind = CandidateKind::TP_HEAD;
    best.reason = "no profitable tp move";

    for (size_t si = 0u; si < stages.size(); ++si) {
        const StageSnapshot &stage = stages[si];
        if (stage.nodes.size() < 2u) continue;

        size_t slowIdx = 0u;
        double slowTime = stage.nodes[0].timeMs;
        for (size_t i = 1u; i < stage.nodes.size(); ++i) {
            if (stage.nodes[i].timeMs > slowTime) {
                slowTime = stage.nodes[i].timeMs;
                slowIdx = i;
            }
        }

        Candidate stageBest;
        stageBest.kind = CandidateKind::TP_HEAD;
        stageBest.thresholdMs = tpGainThresholdMs(stage, cfg);
        stageBest.reason = "no profitable tp move";

        if (slowIdx > 0u) {
            considerTpMove(stage, cfg, slowIdx, slowIdx - 1u, true, stageBest);
            considerTpMove(stage, cfg, slowIdx, slowIdx - 1u, false, stageBest);
        }
        if (slowIdx + 1u < stage.nodes.size()) {
            considerTpMove(stage, cfg, slowIdx, slowIdx + 1u, true, stageBest);
            considerTpMove(stage, cfg, slowIdx, slowIdx + 1u, false, stageBest);
        }
        considerBest(best, stageBest);
    }

    return best;
}

Candidate betterCandidate(const Candidate &a, const Candidate &b) {
    if (a.valid && !b.valid) return a;
    if (b.valid && !a.valid) return b;
    if (!a.valid) return a;
    return b.gainMs > a.gainMs ? b : a;
}

void applyPpMove(std::vector<StageSnapshot> &stages, const Candidate &candidate) {
    if (candidate.kind != CandidateKind::PP_MOVE || !candidate.valid) return;
    StageSnapshot *from = nullptr;
    StageSnapshot *to = nullptr;
    for (size_t i = 0u; i < stages.size(); ++i) {
        if (stages[i].stageIndex == candidate.fromStageIndex) from = &stages[i];
        if (stages[i].stageIndex == candidate.toStageIndex) to = &stages[i];
    }
    if (from == nullptr || to == nullptr) return;
    if (from->nLayers == 0u) from->nLayers = stageLayerCount(*from);
    if (to->nLayers == 0u) to->nLayers = stageLayerCount(*to);
    if (from->nLayers <= 1u) return;
    from->nLayers -= 1u;
    to->nLayers += 1u;
    if (to->stageIndex > from->stageIndex) {
        if (from->endLayer > from->startLayer) from->endLayer -= 1u;
        if (to->startLayer > 0u) to->startLayer -= 1u;
    } else {
        from->startLayer += 1u;
        to->endLayer += 1u;
    }
}

void applyRollbackPenalty(StageSnapshot &target) {
    target.riskPenalty += 0.1;
    const uint32_t n = stageLayerCount(target);
    const uint32_t cap = n > 1u ? n - 1u : 1u;
    target.softCapacity = std::max<uint32_t>(target.softCapacity == 0u ? cap : std::min(target.softCapacity, cap), 1u);
}

const char *candidateKindName(CandidateKind kind) {
    switch (kind) {
        case CandidateKind::TP_HEAD: return "tp_head";
        case CandidateKind::TP_FFN: return "tp_ffn";
        case CandidateKind::PP_MOVE: return "pp_move";
        case CandidateKind::NONE:
        default:
            return "none";
    }
}

} // namespace dynamic_tpot
} // namespace dllama
