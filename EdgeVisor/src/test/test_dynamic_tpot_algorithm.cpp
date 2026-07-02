#include "dynamic/tpot_algorithm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

int main() {
    tpot::SchedulerConfig cfg;
    cfg.minPpGainMs = 5.0;
    cfg.minTpGainMs = 2.0;
    cfg.loadPenaltyBeta = 0.08;
    cfg.ppRiskMarginMs = 0.0;
    cfg.tpRiskMarginMs = 0.0;

    tpot::StageSnapshot s = stage(0, 0, 0, 4, 10.0);
    require(near(tpot::stageCostMs(s, 4, cfg), 40.0), "F_s(n) base cost");
    require(near(tpot::ppDeltaInMs(s, cfg), 14.0), "delta_in includes soft-capacity load penalty");
    require(near(tpot::ppDeltaOutMs(s, cfg), 10.0), "delta_out base layer cost");

    tpot::StageSnapshot over = s;
    over.softCapacity = 3;
    require(tpot::ppDeltaInMs(over, cfg) > tpot::ppDeltaInMs(s, cfg), "delta_in grows past soft capacity");

    tpot::StageSnapshot trend = s;
    trend.recentAvgMs = 13.0;
    trend.previousAvgMs = 10.0;
    require(near(tpot::clampTrendPenalty(trend.recentAvgMs, trend.previousAvgMs), 1.3), "trend penalty clamps at 1.3");

    std::vector<tpot::StageSnapshot> pp;
    pp.push_back(stage(0, 0, 0, 4, 25.0));
    pp.push_back(stage(1, 1, 4, 2, 5.0));
    tpot::Candidate ppMove = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(ppMove.valid, "profitable PP candidate selected");
    require(ppMove.fromStageIndex == 0u && ppMove.toStageIndex == 1u, "PP candidate direction source slow to target fast");
    require(ppMove.layerIndex == 3u, "PP candidate moves source right boundary layer");

    cfg.minPpGainMs = 100.0;
    tpot::Candidate noPp = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(!noPp.valid, "PP candidate rejected below threshold");
    cfg.minPpGainMs = 5.0;

    pp[1].riskPenalty = 20.0;
    tpot::Candidate riskPp = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(!riskPp.valid, "target risk penalty suppresses candidate");
    pp[1].riskPenalty = 0.0;

    pp[0].nLayers = 1u;
    pp[0].endLayer = 1u;
    tpot::Candidate emptyPp = tpot::bestPpCandidate(pp, 110.0, cfg);
    require(emptyPp.fromStageIndex != 0u || !emptyPp.valid, "PP candidate does not empty source stage");

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

    tp[0].nodes[0].timeMs = 12.0;
    tp[0].nodes[1].timeMs = 11.0;
    tp[0].stageTimeMs = 12.0;
    cfg.minTpGainMs = 2.0;
    tpot::Candidate noTp = tpot::bestTpCandidate(tp, cfg);
    require(!noTp.valid, "TP candidate rejected when old_max-new_max is too small");

    std::printf("PASS dynamic_tpot_algorithm\n");
    return 0;
}
