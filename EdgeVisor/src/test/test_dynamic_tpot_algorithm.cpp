#include "dynamic/tpot_algorithm.hpp"
#include "dynamic/tpot_log.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace tpot = dllama::dynamic_tpot;

static void require(bool cond, const char *msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static bool near(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

static tpot::StageSnapshot stage(uint32_t idx, uint32_t root, uint32_t start, uint32_t n, double avgLayerMs) {
    tpot::StageSnapshot s;
    s.stageIndex = idx;
    s.rootNodeIndex = root;
    s.startLayer = start;
    s.endLayer = start + n;
    s.nLayers = n;
    s.softCapacity = n;
    s.avgLayerMs = avgLayerMs;
    s.recentAvgMs = avgLayerMs;
    s.previousAvgMs = avgLayerMs;
    s.stageTimeMs = avgLayerMs * (double)n;
    s.hasFullWeights = true;
    return s;
}

static tpot::NodeSnapshot node(uint32_t idx, double ms, uint32_t heads, uint32_t ffn) {
    tpot::NodeSnapshot n;
    n.nodeIndex = idx;
    n.timeMs = ms;
    n.attnMs = ms;
    n.ffnMs = ms;
    n.kvHeads = heads;
    n.ffnUnits = ffn;
    return n;
}

static bool sameNode(const tpot::NodeSnapshot &a, const tpot::NodeSnapshot &b) {
    return a.nodeIndex == b.nodeIndex && near(a.timeMs, b.timeMs) && near(a.attnMs, b.attnMs) &&
        near(a.ffnMs, b.ffnMs) && a.kvHeads == b.kvHeads && a.ffnUnits == b.ffnUnits &&
        a.canMoveHeadLeft == b.canMoveHeadLeft && a.canMoveHeadRight == b.canMoveHeadRight;
}

static bool sameStage(const tpot::StageSnapshot &a, const tpot::StageSnapshot &b) {
    if (a.stageIndex != b.stageIndex || a.rootNodeIndex != b.rootNodeIndex ||
            a.startLayer != b.startLayer || a.endLayer != b.endLayer || a.nLayers != b.nLayers ||
            a.softCapacity != b.softCapacity || a.hasFullWeights != b.hasFullWeights ||
            !near(a.avgLayerMs, b.avgLayerMs) || !near(a.recentAvgMs, b.recentAvgMs) ||
            !near(a.previousAvgMs, b.previousAvgMs) || !near(a.stageTimeMs, b.stageTimeMs) ||
            !near(a.boundaryCommMs, b.boundaryCommMs) ||
            !near(a.leftBoundaryLayerMs, b.leftBoundaryLayerMs) ||
            !near(a.rightBoundaryLayerMs, b.rightBoundaryLayerMs) ||
            !near(a.riskPenalty, b.riskPenalty) || a.nodes.size() != b.nodes.size()) {
        return false;
    }
    for (size_t i = 0u; i < a.nodes.size(); ++i) {
        if (!sameNode(a.nodes[i], b.nodes[i])) return false;
    }
    return true;
}

static bool sameStages(const std::vector<tpot::StageSnapshot> &a, const std::vector<tpot::StageSnapshot> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0u; i < a.size(); ++i) {
        if (!sameStage(a[i], b[i])) return false;
    }
    return true;
}

int main() {
    tpot::SchedulerConfig cfg;
    require(near(cfg.loadPenaltyBeta, 0.0), "load penalty defaults to zero");
    require(near(cfg.ppGainRatio, 0.03), "PP gain ratio defaults to three percent");
    require(cfg.maxPpLayerMove == 1u, "PP batch size defaults to one layer");
    cfg.minPpGainMs = 5.0;
    cfg.minTpGainMs = 2.0;
    cfg.ppRiskMarginMs = 0.0;
    cfg.tpRiskMarginMs = 0.0;

    tpot::StageSnapshot scoredSource = stage(0, 0, 0, 6, 20.0);
    tpot::StageSnapshot scoredTarget = stage(1, 1, 6, 2, 5.0);
    scoredSource.rightBoundaryLayerMs = 20.0;
    scoredTarget.riskPenalty = 0.5;
    cfg.minPpGainMs = 1.0;
    cfg.ppGainRatio = 0.10;
    cfg.ppRiskMarginMs = 2.0;
    cfg.ppMigrationCostMs = 384.0;
    cfg.expectedRemainingTokens = 128;
    const tpot::Candidate scoredOne = tpot::ppCandidateForMove(scoredSource, scoredTarget, 1u, cfg);
    const tpot::Candidate scoredFour = tpot::ppCandidateForMove(scoredSource, scoredTarget, 4u, cfg);
    tpot::SchedulerConfig unpenalizedCfg = cfg;
    unpenalizedCfg.ppRiskMarginMs = 0.0;
    unpenalizedCfg.ppMigrationCostMs = 0.0;
    scoredTarget.riskPenalty = 0.0;
    const tpot::Candidate unpenalizedOne = tpot::ppCandidateForMove(scoredSource, scoredTarget, 1u, unpenalizedCfg);
    const tpot::Candidate unpenalizedFour = tpot::ppCandidateForMove(scoredSource, scoredTarget, 4u, unpenalizedCfg);
    require(near(unpenalizedOne.gainMs - scoredOne.gainMs, 7.5),
        "one-layer PP risk and migration deductions use one unit");
    require(near(unpenalizedFour.gainMs - scoredFour.gainMs, 30.0),
        "four-layer PP risk and migration deductions scale by four");
    require(near(scoredOne.thresholdMs, scoredFour.thresholdMs),
        "PP acceptance threshold does not scale with layer count");

    tpot::Candidate selectedForLog = scoredFour;
    selectedForLog.layerCount = 3u;
    tpot::Candidate freshBestForLog = scoredOne;
    freshBestForLog.layerCount = 1u;
    const std::string decisionLog = tpot::formatSelectedCandidateLogFields(selectedForLog) + " " +
        tpot::formatPpCandidateLogFields(freshBestForLog);
    require(decisionLog.find("layer_count=3") != std::string::npos,
        "selected or pending layer count remains stable in decision logs");
    require(decisionLog.find("pp_best_layer_count=1") != std::string::npos,
        "fresh best PP layer count is logged independently");

    tpot::Candidate layoutCandidate = scoredOne;
    layoutCandidate.valid = true;
    const tpot::Candidate blockedAfterMove = tpot::filterPpCandidateForStaticLayout(layoutCandidate, true);
    require(!blockedAfterMove.valid && blockedAfterMove.reason == "pp layout changed; restart or rollback required",
        "accepted PP layout changes suppress stale later PP candidates");
    const tpot::Candidate allowedAfterRollback = tpot::filterPpCandidateForStaticLayout(layoutCandidate, false);
    require(allowedAfterRollback.valid, "successful rollback restores PP candidate eligibility");

    tpot::PpLayoutGuard ppLayoutGuard;
    require(ppLayoutGuard.filter(layoutCandidate).valid,
        "PP boundary is initially eligible");
    ppLayoutGuard.markIssued(layoutCandidate);
    require(!ppLayoutGuard.filter(layoutCandidate).valid,
        "issued PP boundary is suppressed while migration is pending");
    ppLayoutGuard.markRolledBack(layoutCandidate);
    require(ppLayoutGuard.filter(layoutCandidate).valid,
        "rolled-back PP boundary becomes eligible again");

    std::vector<tpot::StageSnapshot> committedLayout;
    committedLayout.push_back(stage(0, 0, 0, 4, 10.0));
    committedLayout.push_back(stage(1, 1, 4, 3, 8.0));
    tpot::Candidate committedMove;
    committedMove.kind = tpot::CandidateKind::PP_MOVE;
    committedMove.valid = true;
    committedMove.fromStageIndex = 0u;
    committedMove.toStageIndex = 1u;
    committedMove.fromNodeIndex = 0u;
    committedMove.toNodeIndex = 1u;
    committedMove.layerIndex = 3u;
    committedMove.layerCount = 1u;
    tpot::PpLayoutGuard committedGuard;
    committedGuard.markIssued(committedMove);
    require(tpot::applyPpMove(committedLayout, committedMove),
        "committed PP move updates the logical layout");
    require(committedLayout[0].endLayer == 3u && committedLayout[0].nLayers == 3u &&
            committedLayout[1].startLayer == 3u && committedLayout[1].nLayers == 4u,
        "committed PP move shifts the next scheduler boundary");
    committedGuard.markCommitted(committedMove);
    require(committedGuard.filter(committedMove).valid,
        "committed PP move releases the guard for a later rebalance");

    std::vector<tpot::StageSnapshot> bypassLayout;
    bypassLayout.push_back(stage(0, 0, 0, 4, 5.0));
    bypassLayout.push_back(stage(1, 1, 4, 4, 5.0));
    bypassLayout.push_back(stage(2, 2, 8, 4, 5.0));
    bypassLayout.push_back(stage(3, 3, 12, 4, 20.0));
    std::vector<uint32_t> bypassActiveChain;
    const std::vector<uint32_t> bypassLayers = {8u, 9u, 10u, 11u};
    require(tpot::commitStageBypassLayout(bypassLayout, 2u, 3u, bypassLayers, &bypassActiveChain),
        "complete adjacent bypass commits its logical layout");
    require(bypassActiveChain == std::vector<uint32_t>({0u, 1u, 3u}),
        "bypass rebuilds active chain without ejected stage");
    require(bypassLayout[2].startLayer == 8u && bypassLayout[2].endLayer == 16u && bypassLayout[2].nLayers == 8u,
        "bypass merge derives range and count from endpoints");

    std::vector<tpot::StageSnapshot> rejectedBypassLayout;
    rejectedBypassLayout.push_back(stage(0, 0, 0, 4, 5.0));
    rejectedBypassLayout.push_back(stage(1, 1, 4, 4, 5.0));
    rejectedBypassLayout.push_back(stage(2, 2, 8, 4, 5.0));
    rejectedBypassLayout.push_back(stage(3, 3, 12, 4, 20.0));
    const std::vector<tpot::StageSnapshot> beforeRejectedBypass = rejectedBypassLayout;
    require(!tpot::commitStageBypassLayout(rejectedBypassLayout, 1u, 3u, std::vector<uint32_t>({4u, 5u, 6u, 7u}), nullptr) &&
            sameStages(rejectedBypassLayout, beforeRejectedBypass),
        "non-adjacent bypass telemetry cannot mutate layout");
    require(!tpot::commitStageBypassLayout(rejectedBypassLayout, 2u, 3u, std::vector<uint32_t>({8u, 10u}), nullptr) &&
            sameStages(rejectedBypassLayout, beforeRejectedBypass),
        "incomplete bypass layer telemetry cannot mutate layout");

    const std::vector<tpot::StageSnapshot> beforeInvalidMove = committedLayout;
    tpot::Candidate invalidMove = committedMove;
    invalidMove.layerIndex = 0u;
    require(!tpot::applyPpMove(committedLayout, invalidMove) && sameStages(committedLayout, beforeInvalidMove),
        "invalid PP move cannot partially mutate the committed layout");

    tpot::StageSnapshot fullCapacity = stage(2, 2, 7, 4, 6.0);
    fullCapacity.nLayers = 3u;
    tpot::rebasePpSoftCapacity(fullCapacity, 4u);
    require(fullCapacity.softCapacity == 3u,
        "full soft capacity follows a committed layer-count change");
    tpot::StageSnapshot conservativeCapacity = stage(3, 3, 10, 4, 6.0);
    conservativeCapacity.softCapacity = 2u;
    conservativeCapacity.nLayers = 3u;
    tpot::rebasePpSoftCapacity(conservativeCapacity, 4u);
    require(conservativeCapacity.softCapacity == 2u,
        "conservative soft capacity survives a committed layout change");

    std::vector<tpot::StageSnapshot> bypassChain;
    bypassChain.push_back(stage(0, 0, 0, 8, 5.0));
    bypassChain.push_back(stage(1, 1, 8, 8, 5.0));
    bypassChain.push_back(stage(3, 3, 16, 4, 20.0));
    bypassChain[2].leftBoundaryLayerMs = 20.0;
    cfg.minPpGainMs = 0.0;
    cfg.ppGainRatio = 0.0;
    const tpot::Candidate afterBypass = tpot::bestPpCandidate(bypassChain, 100.0, cfg);
    require(afterBypass.valid && afterBypass.fromStageIndex == 3u && afterBypass.toStageIndex == 1u &&
            afterBypass.layerIndex == 16u,
        "active chain 0,1,3 selects legal reverse PP2 without ejected stage 2");

    cfg = tpot::SchedulerConfig();
    cfg.minPpGainMs = 5.0;
    cfg.minTpGainMs = 2.0;
    cfg.ppRiskMarginMs = 0.0;
    cfg.tpRiskMarginMs = 0.0;

    tpot::StageSnapshot s = stage(0, 0, 0, 4, 10.0);
    require(near(tpot::stageCostMs(s, 4, cfg), 40.0), "F_s(n) base cost");
    require(near(tpot::ppDeltaInMs(s, cfg), 10.0), "default delta_in has no load penalty");
    cfg.loadPenaltyBeta = 0.08;
    require(near(tpot::ppDeltaInMs(s, cfg), 14.0), "explicit beta preserves legacy cost");
    cfg.loadPenaltyBeta = 0.0;
    require(near(tpot::ppDeltaOutMs(s, cfg), 10.0), "delta_out base layer cost");

    cfg.ppGainRatio = 0.03;
    require(near(tpot::ppGainThresholdMs(200.0, cfg), 6.0), "ratio controls PP threshold");
    cfg.ppGainRatio = 0.0;
    require(near(tpot::ppGainThresholdMs(200.0, cfg), 5.0), "zero ratio leaves absolute threshold");
    cfg.ppGainRatio = 0.10;
    require(near(tpot::ppGainThresholdMs(200.0, cfg), 20.0), "custom ratio changes threshold");
    cfg.ppGainRatio = 0.03;

    tpot::StageSnapshot over = s;
    over.softCapacity = 3;
    cfg.loadPenaltyBeta = 0.08;
    require(tpot::ppDeltaInMs(over, cfg) > tpot::ppDeltaInMs(s, cfg), "delta_in grows past soft capacity");
    cfg.loadPenaltyBeta = 0.0;

    tpot::StageSnapshot trend = s;
    trend.recentAvgMs = 13.0;
    trend.previousAvgMs = 10.0;
    require(near(tpot::clampTrendPenalty(trend.recentAvgMs, trend.previousAvgMs), 1.3), "trend penalty clamps at 1.3");

    std::vector<tpot::StageSnapshot> pp;
    pp.push_back(stage(0, 0, 0, 4, 25.0));
    pp.push_back(stage(1, 1, 4, 2, 5.0));

    cfg.minPpGainMs = 1.0;
    cfg.ppGainRatio = 0.0;
    const tpot::Candidate zeroRatioPp = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(zeroRatioPp.valid && near(zeroRatioPp.thresholdMs, 1.0),
        "zero PP ratio uses the absolute scheduler threshold");
    cfg.ppGainRatio = 0.03;
    const tpot::Candidate defaultRatioPp = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(defaultRatioPp.valid && near(defaultRatioPp.thresholdMs, 3.3),
        "default PP ratio sets the scheduler threshold");
    cfg.ppGainRatio = 0.20;
    const tpot::Candidate customRatioPp = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(!customRatioPp.valid && near(customRatioPp.thresholdMs, 22.0),
        "custom PP ratio rejects a scheduler candidate below its threshold");
    cfg.minPpGainMs = 5.0;
    cfg.ppGainRatio = 0.03;

    tpot::Candidate ppMove = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(ppMove.valid, "profitable PP candidate selected");
    require(ppMove.fromStageIndex == 0u && ppMove.toStageIndex == 1u, "PP candidate direction source slow to target fast");
    require(ppMove.layerIndex == 3u, "PP candidate moves source right boundary layer");

    cfg.minPpGainMs = 100.0;
    tpot::Candidate noPp = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(!noPp.valid, "PP candidate rejected below threshold");
    require(noPp.reason == "gain below threshold", "rejected candidate reports reason");
    require(noPp.fromStageIndex == 0u && noPp.toStageIndex == 1u, "rejected candidate retains direction");
    require(noPp.layerIndex == 3u, "rejected candidate retains boundary layer");
    require(noPp.gainMs > 0.0 && near(noPp.thresholdMs, 100.0), "rejected candidate retains scores");
    const std::string rejectedPpLog = tpot::formatPpCandidateLogFields(noPp);
    require(rejectedPpLog.find("pp_best_valid=0") != std::string::npos, "rejected PP log includes validity");
    require(rejectedPpLog.find("pp_best_gain_ms=") != std::string::npos, "rejected PP log includes gain");
    require(rejectedPpLog.find("pp_best_threshold_ms=100") != std::string::npos, "rejected PP log includes threshold");
    require(rejectedPpLog.find("pp_best_reason=gain_below_threshold") != std::string::npos, "rejected PP log sanitizes reason");
    require(rejectedPpLog.find("pp_best_from_stage=0") != std::string::npos, "rejected PP log includes source stage");
    require(rejectedPpLog.find("pp_best_to_stage=1") != std::string::npos, "rejected PP log includes target stage");
    require(rejectedPpLog.find("pp_best_layer=3") != std::string::npos, "rejected PP log includes layer");
    const std::string validPpLog = tpot::formatPpCandidateLogFields(ppMove);
    require(validPpLog.find("pp_best_valid=1") != std::string::npos, "valid PP log includes validity");
    require(validPpLog.find("pp_best_reason=none") != std::string::npos, "valid PP log includes no reason");
    cfg.minPpGainMs = 5.0;

    std::vector<tpot::StageSnapshot> mixed;
    mixed.push_back(stage(0, 0, 0, 4, 30.0));
    mixed.push_back(stage(1, 1, 4, 4, 5.0));
    mixed.push_back(stage(2, 2, 8, 4, 3.0));
    mixed[0].rightBoundaryLayerMs = 30.0;
    tpot::Candidate mixedBest = tpot::bestPpCandidate(mixed, 152.0, cfg);
    require(mixedBest.valid, "valid PP candidate wins over rejected alternatives");
    require(mixedBest.fromStageIndex == 0u && mixedBest.toStageIndex == 1u, "valid PP route remains selected");

    pp[1].riskPenalty = 20.0;
    tpot::Candidate riskPp = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(!riskPp.valid, "target risk penalty suppresses candidate");
    pp[1].riskPenalty = 0.0;

    std::vector<tpot::StageSnapshot> ppWithUnrelatedSlowStage;
    ppWithUnrelatedSlowStage.push_back(stage(0, 0, 0, 4, 25.0));
    ppWithUnrelatedSlowStage.push_back(stage(1, 1, 4, 2, 5.0));
    ppWithUnrelatedSlowStage.push_back(stage(2, 2, 6, 1, 1000.0));
    tpot::Candidate localPp = tpot::bestPpCandidate(ppWithUnrelatedSlowStage, 1110.0, cfg);
    require(localPp.valid, "PP threshold uses local affected stages, not unrelated global TPOT");
    require(localPp.fromStageIndex == 0u && localPp.toStageIndex == 1u, "local PP candidate is still the profitable boundary");

    pp[1].boundaryCommMs = 1000.0;
    tpot::Candidate commPp = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(commPp.valid, "PP automation decision ignores communication/bubble timing and uses compute time");
    pp[1].boundaryCommMs = 0.0;

    pp[0].nLayers = 1u;
    pp[0].endLayer = 1u;
    tpot::Candidate emptyPp = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(emptyPp.fromStageIndex != 0u || !emptyPp.valid, "PP candidate does not empty source stage");

    std::vector<tpot::StageSnapshot> singleStage;
    singleStage.push_back(stage(0, 0, 0, 4, 10.0));
    const tpot::Candidate singleStagePp = tpot::bestPpCandidate(singleStage, 40.0, cfg);
    require(!singleStagePp.valid && singleStagePp.reason == "no eligible pp candidate",
        "single-stage PP rejection uses the canonical reason");

    cfg.maxPpLayerMove = 4u;
    cfg.loadPenaltyBeta = 0.5;
    cfg.minPpGainMs = 0.0;
    cfg.ppGainRatio = 0.0;
    std::vector<tpot::StageSnapshot> multi;
    multi.push_back(stage(0, 0, 0, 6, 20.0));
    multi.push_back(stage(1, 1, 6, 2, 5.0));
    multi[0].rightBoundaryLayerMs = 20.0;
    const std::vector<tpot::StageSnapshot> originalMulti = multi;
    const tpot::Candidate multiMove = tpot::bestPpCandidate(multi, 130.0, cfg);
    require(multiMove.valid && multiMove.layerCount == 2u,
        "scheduler selects the interior optimum instead of the largest allowed batch");
    require(multiMove.layerIndex == multi[0].endLayer - multiMove.layerCount,
        "forward candidate identifies the first layer in the contiguous range");
    const std::string multiLog = tpot::formatPpCandidateLogFields(multiMove);
    require(multiLog.find("pp_best_layer_count=" + std::to_string(multiMove.layerCount)) != std::string::npos,
        "PP candidate log includes selected layer count");

    tpot::Candidate predicted = multiMove;
    tpot::applyPpMove(multi, predicted);
    require(multi[0].nLayers == 6u - predicted.layerCount &&
            multi[1].nLayers == 2u + predicted.layerCount,
        "predicted stage counts move the selected layer batch");

    std::vector<tpot::StageSnapshot> invalidPrediction;
    invalidPrediction.push_back(stage(0, 0, 0, 2, 20.0));
    invalidPrediction.push_back(stage(1, 1, 2, 2, 10.0));
    invalidPrediction.push_back(stage(2, 2, 4, 2, 5.0));
    tpot::Candidate nonAdjacentPrediction;
    nonAdjacentPrediction.kind = tpot::CandidateKind::PP_MOVE;
    nonAdjacentPrediction.valid = true;
    nonAdjacentPrediction.fromStageIndex = 0u;
    nonAdjacentPrediction.toStageIndex = 2u;
    nonAdjacentPrediction.layerIndex = 1u;
    nonAdjacentPrediction.layerCount = 1u;
    const std::vector<tpot::StageSnapshot> originalInvalidPrediction = invalidPrediction;
    tpot::applyPpMove(invalidPrediction, nonAdjacentPrediction);
    require(sameStages(invalidPrediction, originalInvalidPrediction),
        "prediction rejects non-adjacent PP stages");

    tpot::Candidate interiorForwardPrediction = nonAdjacentPrediction;
    interiorForwardPrediction.toStageIndex = 1u;
    interiorForwardPrediction.layerIndex = 0u;
    invalidPrediction = originalInvalidPrediction;
    tpot::applyPpMove(invalidPrediction, interiorForwardPrediction);
    require(sameStages(invalidPrediction, originalInvalidPrediction),
        "prediction rejects forward PP ranges away from the source boundary");

    tpot::Candidate interiorReversePrediction = nonAdjacentPrediction;
    interiorReversePrediction.fromStageIndex = 1u;
    interiorReversePrediction.toStageIndex = 0u;
    interiorReversePrediction.layerIndex = 3u;
    invalidPrediction = originalInvalidPrediction;
    tpot::applyPpMove(invalidPrediction, interiorReversePrediction);
    require(sameStages(invalidPrediction, originalInvalidPrediction),
        "prediction rejects reverse PP ranges away from the source boundary");

    const tpot::Candidate reverse = tpot::reversePpCandidate(predicted);
    require(reverse.layerCount == predicted.layerCount,
        "rollback preserves the selected PP layer count");
    require(tpot::ppCommandLayerCount(predicted) == predicted.layerCount,
        "PP command count uses the selected candidate count");

    cfg.ppMigrationCostMs = 384.0; // 3 ms per layer at 128 remaining tokens
    const tpot::Candidate costLimited = tpot::bestPpCandidate(originalMulti, 130.0, cfg);
    require(costLimited.layerCount == 1u,
        "per-layer migration cost can make a smaller batch optimal");

    std::vector<tpot::StageSnapshot> reverseStages;
    reverseStages.push_back(stage(0, 0, 0, 2, 5.0));
    reverseStages.push_back(stage(1, 1, 2, 6, 20.0));
    reverseStages[1].leftBoundaryLayerMs = 20.0;
    cfg.ppMigrationCostMs = 0.0;
    const tpot::Candidate reverseMove = tpot::bestPpCandidate(reverseStages, 130.0, cfg);
    require(reverseMove.fromStageIndex == 1u && reverseMove.toStageIndex == 0u &&
            reverseMove.layerIndex == reverseStages[1].startLayer,
        "reverse candidate begins at the source left boundary");

    std::vector<tpot::StageSnapshot> twoLayerSource;
    twoLayerSource.push_back(stage(0, 0, 0, 2, 20.0));
    twoLayerSource.push_back(stage(1, 1, 2, 2, 5.0));
    const tpot::Candidate leavesOne = tpot::bestPpCandidate(twoLayerSource, 50.0, cfg);
    require(leavesOne.layerCount <= 1u,
        "configured K never empties a two-layer source stage");

    cfg.maxPpLayerMove = 1u;
    cfg.loadPenaltyBeta = 0.0;
    cfg.ppMigrationCostMs = 0.0;
    cfg.minPpGainMs = 5.0;
    cfg.ppGainRatio = 0.03;

    std::vector<tpot::StageSnapshot> tp;
    tpot::StageSnapshot ts = stage(0, 0, 0, 2, 10.0);
    ts.nodes.push_back(node(0, 30.0, 3u, 1024u));
    ts.nodes.push_back(node(1, 10.0, 1u, 1024u));
    ts.stageTimeMs = 30.0;
    tp.push_back(ts);
    cfg.minTpGainMs = 2.0;
    cfg.maxHeadMove = 1u;
    cfg.maxFfnMove = 256u;
    tpot::Candidate tpMove = tpot::bestTpCandidate(tp, cfg);
    require(tpMove.valid, "TP water-filling selects profitable move");
    require(tpMove.fromNodeIndex == 0u && tpMove.toNodeIndex == 1u, "TP candidate moves from slow node to neighbor");
    require(tpMove.gainMs > 0.0, "TP candidate has positive gain");

    tp[0].stageTimeMs = 300.0;
    cfg.ppGainRatio = 0.03;
    const tpot::Candidate defaultRatioTp = tpot::bestTpCandidate(tp, cfg);
    cfg.ppGainRatio = 0.0;
    const tpot::Candidate zeroRatioTp = tpot::bestTpCandidate(tp, cfg);
    cfg.ppGainRatio = 0.75;
    const tpot::Candidate customRatioTp = tpot::bestTpCandidate(tp, cfg);
    require(defaultRatioTp.valid && zeroRatioTp.valid && customRatioTp.valid,
        "PP gain ratio preserves TP candidate validity");
    require(near(defaultRatioTp.thresholdMs, 9.0) && near(zeroRatioTp.thresholdMs, 9.0) &&
            near(customRatioTp.thresholdMs, 9.0),
        "PP gain ratio preserves TP scheduler thresholds");
    require(near(defaultRatioTp.gainMs, zeroRatioTp.gainMs) &&
            near(defaultRatioTp.gainMs, customRatioTp.gainMs),
        "PP gain ratio preserves TP scheduler gains");
    cfg.ppGainRatio = 0.03;

    tp[0].nodes[0].timeMs = 12.0;
    tp[0].nodes[1].timeMs = 11.0;
    tp[0].stageTimeMs = 12.0;
    cfg.minTpGainMs = 2.0;
    tpot::Candidate noTp = tpot::bestTpCandidate(tp, cfg);
    require(!noTp.valid, "TP candidate rejected when old_max-new_max is too small");

    std::printf("PASS dynamic_tpot_algorithm\n");
    return 0;
}
