#include "dynamic/algo2.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static tpot::Algo2Config testConfig() {
    tpot::Algo2Config cfg;
    cfg.metricsWindowTokens = 4;
    cfg.decisionWindowTokens = 4;
    cfg.decisionStrideTokens = 4;
    cfg.minDecisionIntervalMs = 0;
    cfg.degradationConfirmations = 2;
    cfg.recoveryConfirmations = 2;
    cfg.gainAbsMs = 3.0;
    cfg.gainRelativePair = 0.03;
    cfg.predictionNoiseMs = -1.0;
    cfg.settleTokens = 1;
    cfg.verifyTokens = 2;
    cfg.verifyMaxTokens = 4;
    cfg.cooldownTokens = 1;
    cfg.cooldownMinMs = 0;
    cfg.minLayersPerStage = 1u;
    cfg.maxMoveLayers = 0u;
    cfg.proactiveRebalance = false;
    cfg.verifyRatio = 0.05;
    cfg.watchdogMs = 1000;
    cfg.watchdogFactor = 5.0;
    cfg.reverseSuppressDecisions = 2;
    return cfg;
}

static tpot::Algo2StageInput stage(uint32_t id, uint32_t root, uint32_t start, uint32_t n, double perLayer) {
    tpot::Algo2StageInput s;
    s.snapshot.stageIndex = id;
    s.snapshot.rootNodeIndex = root;
    s.snapshot.startLayer = start;
    s.snapshot.endLayer = start + n;
    s.snapshot.nLayers = n;
    s.snapshot.avgLayerMs = perLayer;
    s.snapshot.stageTimeMs = perLayer * (double)n;
    s.snapshot.recentAvgMs = perLayer * 3.0;
    s.snapshot.previousAvgMs = perLayer;
    s.shadowCursor = 100;
    return s;
}

static tpot::Algo2TokenObs tok(uint32_t id, double ms) {
    tpot::Algo2TokenObs t;
    t.tokenId = id;
    t.contextLength = 32u + id;
    t.e2eKnown = true;
    t.e2eMs = ms;
    t.stageComputeMs[0] = ms * 0.7;
    t.stageComputeMs[1] = ms * 0.3;
    t.stageCommMs[0] = 1.0;
    t.stageCommMs[1] = 1.0;
    return t;
}

static void arm(tpot::Algo2Session *session, tpot::Algo2Mode mode, int targetCursor, int memorySpare, uint32_t rightReady) {
    session->setMode(mode);
    session->setSerialRequest(true);
    std::vector<tpot::Algo2StageInput> stages;
    stages.push_back(stage(0u, 0u, 0u, 4u, 10.0));
    stages.push_back(stage(1u, 1u, 4u, 4u, 8.0));
    stages[1].shadowCursor = targetCursor;
    session->setStages(stages);
    tpot::Algo2Boundary b;
    b.leftReady = 0u;
    b.rightReady = rightReady;
    b.memorySpareLayers = memorySpare;
    session->setBoundaries(std::vector<tpot::Algo2Boundary>(1u, b));
}

int main() {
    require(near(tpot::clampTrendPenalty(30.0, 10.0), 1.3), "legacy trend clamp remains 1.3");
    require(std::string(tpot::algo2ModeName(tpot::parseAlgo2Mode("revised_variable_step"))) == "revised_variable_step",
        "mode name");
    const char *modeEnv = std::getenv("DLLAMA_ALGO2_MODE");
    const bool revisedEnv = modeEnv != nullptr &&
        (std::strcmp(modeEnv, "revised_single_layer") == 0 || std::strcmp(modeEnv, "revised_variable_step") == 0);
    require(tpot::algo2ForbidsFullKvFallback() == revisedEnv, "full KV fallback stays available unless revised mode is selected");

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) d = session.observe(tok((uint32_t)i, 10.0), (double)i);
        require(d.health == tpot::Algo2Health::NORMAL && !d.issue, "first window only establishes the reference");
        for (int i = 4; i < 8; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(d.health == tpot::Algo2Health::SUSPECT && !d.issue, "one slow window is suspect, not a move");
        for (int i = 8; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(d.issue && d.candidate.valid, "confirmed degradation can move");
        require(d.candidate.layerCount == 2u, "k=1 stays under the 3 ms bar and k=2 clears it");
        require(near(d.candidate.gainMs, 4.0), "gain is k times the layer gap, with no 1.3 multiplier");
        require(d.candidate.fromStageIndex == 0u && d.candidate.toStageIndex == 1u, "move follows the cheaper neighbor");
        require(d.estimateMethod.find("stage_average_linear") != std::string::npos, "average estimate is named");
        require(d.estimateMethod.find("fixed_overhead_not_separated") != std::string::npos,
            "unseparated fixed overhead stays visible");
        require(d.cause == "compute_degradation", "stage compute locates the slowdown");
        require(!d.recovered, "issuing a move is not a recovery");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_SINGLE, 100, -1, 4u);
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) d = session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(!d.issue && d.note == "gain_below_threshold", "single-layer mode does not multiply a sub-threshold layer");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 0, -1, 4u);
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(!d.issue && d.note == "kv_behind", "a stale shadow cursor blocks the move");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, -1, -1, 4u);
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(!d.issue && d.note == "kv_readiness_unknown", "missing KV cursor is not treated as ready");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, -2, -1, 4u);
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(!d.issue && d.note == "shadow_not_provisioned", "a stage with no shadow cannot take layers");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, 0, 4u);
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(!d.issue && d.note == "memory_spare", "zero spare rejects even when shadow depth is positive");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::STATIC, 100, -1, 4u);
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(!d.issue && d.note == "static" &&
                (d.health == tpot::Algo2Health::DEGRADED || d.health == tpot::Algo2Health::STABLE_DEGRADED),
            "static mode records degradation and does not move");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 4u);
        session.setSerialRequest(false);
        tpot::Algo2Decision d;
        for (int i = 0; i < 12; ++i) d = session.observe(tok((uint32_t)i, i < 4 ? 10.0 : 20.0), (double)i);
        require(!d.issue && d.note == "overlapping_requests_unsupported", "overlapping requests do not use the serial search");
    }

    {
        tpot::Algo2Session session(testConfig());
        std::vector<tpot::Algo2StageInput> stages;
        stages.push_back(stage(0u, 0u, 0u, 4u, 10.0));
        stages.push_back(stage(1u, 1u, 4u, 4u, 5.0));
        stages[0].fixedOverheadMs = 32.0;
        stages[0].snapshot.stageTimeMs = 40.0;
        stages[1].fixedOverheadMs = 0.0;
        stages[1].snapshot.stageTimeMs = 20.0;
        stages[1].shadowCursor = 100;
        session.setMode(tpot::Algo2Mode::REVISED_VARIABLE);
        session.setStages(stages);
        tpot::Algo2Boundary b;
        b.rightReady = 4u;
        session.setBoundaries(std::vector<tpot::Algo2Boundary>(1u, b));
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(!d.issue, "fixed overhead is not moved as if it were layer compute");
        require(d.note == "gain_below_threshold", "the rejection names the threshold, not a fake layer gain");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 4u);
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(d.issue, "setup reaches a commit");
        session.noteCommitResult(true, "");
        d = session.observe(tok(12u, 20.0), 12.0);
        require(d.note == "settle_done", "one settle token opens verification");
        d = session.observe(tok(13u, 40.0), 13.0);
        require(d.note == "verify_wait", "verification waits for its window");
        d = session.observe(tok(14u, 40.0), 14.0);
        require(d.rollback && d.note == "clear_regression", "a worse verify window rolls back");
        session.noteCommitResult(false, "rollback_unsupported");
        d = session.observe(tok(15u, 40.0), 15.0);
        require(!d.issue && d.note == "rollback_failed", "a failed rollback stays failed instead of counting as success");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 4u);
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        tpot::Algo2Decision d;
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        session.noteCommitResult(true, "");
        session.observe(tok(12u, 20.0), 12.0);
        d = session.observe(tok(13u, 20.0), 13.0);
        require(d.note == "verify_wait", "verification waits for its window");
        d = session.observe(tok(14u, 20.0), 14.0);
        require(d.note == "inconclusive", "a verify window inside the band does not keep or roll back yet");
        d = session.observe(tok(15u, 20.0), 15.0);
        require(d.note == "inconclusive", "the band can be extended");
        d = session.observe(tok(16u, 20.0), 16.0);
        require(d.rollback && d.note == "inconclusive_rollback", "the verify cap rolls back an inconclusive move");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 4u);
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) session.observe(tok((uint32_t)i, 20.0), (double)i);
        session.noteCommitResult(true, "");
        session.observe(tok(12u, 20.0), 12.0);
        tpot::Algo2Decision d = session.observe(tok(13u, 10.0), 13.0);
        d = session.observe(tok(14u, 10.0), 14.0);
        require(d.note == "clear_improvement" && !d.recovered && !d.rollback,
            "a real improvement is kept and is not labeled recovered");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 4u);
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        tpot::Algo2Decision idle = session.pollIdle(5000.0);
        require(idle.health == tpot::Algo2Health::NO_PROGRESS && !idle.issue, "a stall is no_progress and does not move");
    }

    {
        tpot::Algo2TokenObs missing;
        missing.tokenId = 0u;
        tpot::Algo2Session session(testConfig());
        session.setMode(tpot::Algo2Mode::REVISED_VARIABLE);
        tpot::Algo2Decision d = session.observe(missing, 0.0);
        require(d.note == "metric_missing" && d.health == tpot::Algo2Health::INSUFFICIENT,
            "a token with no interval stays insufficient");
        missing.prefill = true;
        missing.e2eKnown = true;
        missing.e2eMs = 1000.0;
        d = session.observe(missing, 1.0);
        require(d.note == "prefill_excluded", "prefill does not enter the health window");
    }

    {
        const std::vector<std::vector<double>> easy = {
            {1, 10}, {1, 10}, {10, 1}, {10, 1},
        };
        const tpot::OfflineSearchResult matched =
            tpot::compareLocalSearchToExhaustive(easy, std::vector<uint32_t>{2u, 2u}, 1u);
        require(matched.exhaustive && matched.localIsOptimal && near(matched.optimalMs, 4.0),
            "a two-device boundary search can match the exhaustive assignment");

        const std::vector<std::vector<double>> trap = {
            {3, 2, 5}, {2, 8, 8}, {8, 7, 4}, {2, 8, 1}, {7, 7, 1}, {8, 5, 4},
        };
        const tpot::OfflineSearchResult shortOf =
            tpot::compareLocalSearchToExhaustive(trap, std::vector<uint32_t>{1u, 1u, 4u}, 1u);
        require(shortOf.exhaustive && !shortOf.localIsOptimal, "local search can stop short of the exhaustive assignment");
        require(near(shortOf.localMs, 21.0) && near(shortOf.optimalMs, 18.0),
            "the checker reports both the local stop and the better legal assignment");
    }

    std::printf("algo2 checks passed\n");
    return 0;
}
