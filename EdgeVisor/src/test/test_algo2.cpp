#include "dynamic/algo2.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace tpot = edgevisor::dynamic_tpot;

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
    s.fixedOverheadMs = 0.0;
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
    const char *modeEnv = std::getenv("EDGEVISOR_ALGO2_MODE");
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
        require(d.estimateMethod.find("stage_average_excluding_fixed") != std::string::npos, "average estimate names the separated fixed overhead");
        require(d.costConfidence == "medium", "a declared zero fixed overhead is a medium-confidence average");
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

    {
        tpot::Algo2Session session(testConfig());
        std::vector<tpot::Algo2StageInput> stages;
        stages.push_back(stage(0u, 0u, 0u, 4u, 10.0));
        stages.push_back(stage(1u, 1u, 4u, 4u, 8.0));
        stages[0].fixedOverheadMs = -1.0;
        stages[1].fixedOverheadMs = -1.0;
        session.setMode(tpot::Algo2Mode::REVISED_VARIABLE);
        session.setSerialRequest(true);
        session.setStages(stages);
        tpot::Algo2Boundary b;
        b.rightReady = 2u;
        session.setBoundaries(std::vector<tpot::Algo2Boundary>(1u, b));
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(!d.issue && d.note == "fixed_overhead_not_separated" && d.costConfidence != "high",
            "an average with unknown fixed overhead does not move");
    }

    {
        tpot::Algo2Session session(testConfig());
        std::vector<tpot::Algo2StageInput> stages;
        stages.push_back(stage(0u, 0u, 0u, 4u, 10.0));
        stages.push_back(stage(1u, 1u, 4u, 4u, 8.0));
        stages[0].snapshot.stageTimeMs = 40.0;
        stages[0].fixedOverheadMs = 0.0;
        stages[0].commWaitMs = 32.0;
        stages[0].commIncludedInStageTime = true;
        session.setMode(tpot::Algo2Mode::REVISED_VARIABLE);
        session.setSerialRequest(true);
        session.setStages(stages);
        tpot::Algo2Boundary b;
        b.rightReady = 2u;
        session.setBoundaries(std::vector<tpot::Algo2Boundary>(1u, b));
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 20.0), (double)i);
        require(!d.issue && d.note == "gain_below_threshold",
            "communication included in the stage time is not counted as movable gain");
    }

    {
        tpot::Algo2Config cfg = testConfig();
        cfg.decisionWindowTokens = 4;
        tpot::Algo2Session session(cfg);
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) d = session.observe(tok((uint32_t)i, 58.0), (double)i);
        require(d.health == tpot::Algo2Health::NORMAL && d.healthReferenceValid, "58 ms establishes the health anchor");
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 558.0), (double)i);
        require(d.issue, "the 558 ms fault issues the first move");
        const double healthAnchor = d.healthReferenceMs;
        session.noteCommitResult(true, "");
        session.observe(tok(12u, 558.0), 12.0);
        session.observe(tok(13u, 179.0), 13.0);
        d = session.observe(tok(14u, 179.0), 14.0);
        require(d.note == "clear_improvement" && !d.recovered && d.health != tpot::Algo2Health::NORMAL,
            "179 ms can keep the move without becoming the healthy state");
        require(near(d.healthReferenceMs, healthAnchor), "the health anchor stays at the pre-fault reference");
        require(near(d.predictedGainMs, 4.0) && near(d.measuredGainMs, 379.0),
            "predicted layer gain and measured end-to-end gain are separate");
        require(near(d.predictionErrorMs, 4.0 - 379.0), "prediction error is predicted minus measured");
        require(d.intervalRole == "verify" && d.intervalInController && near(d.requestIntervalMs, 179.0),
            "a verify sample stays in the controller window");
        d = session.observe(tok(15u, 179.0), 15.0);
        require(d.note == "cooldown_done", "cooldown ends without declaring recovery");
        for (int i = 16; i < 19; ++i) d = session.observe(tok((uint32_t)i, 179.0), (double)i);
        d = session.observe(tok(19u, 179.0), 19.0);
        require(d.issue && !d.recovered && d.health != tpot::Algo2Health::NORMAL,
            "after cooldown a still-unrecovered request can take another positive move");
        session.noteCommitResult(false, "stop_second");
        session.setBoundaries(std::vector<tpot::Algo2Boundary>(1u, tpot::Algo2Boundary()));
        for (int i = 20; i < 24; ++i) d = session.observe(tok((uint32_t)i, 179.0), (double)i);
        require(!d.issue && d.note == "degraded_no_improving_candidate" && d.health != tpot::Algo2Health::NORMAL,
            "no feasible candidate leaves the request unrecovered and does not spin");
        d = session.observe(tok(24u, 179.0), 24.0);
        require(d.note == "stride" && !d.issue, "the next token waits for the decision stride");
    }

    {
        tpot::Algo2Config cfg = testConfig();
        cfg.verifyTokens = 4;
        cfg.verifyMaxTokens = 12;
        cfg.tailConfirmations = 2;
        tpot::Algo2Session session(cfg);
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) session.observe(tok((uint32_t)i, 20.0), (double)i);
        session.noteCommitResult(true, "");
        session.observe(tok(12u, 20.0), 12.0);
        const double spiked[4] = {12.0, 12.0, 12.0, 40.0};
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) d = session.observe(tok(13u + (uint32_t)i, spiked[i]), 13.0 + i);
        require(d.note == "tail_unconfirmed" && !d.rollback, "one tail spike does not roll back");
        for (int i = 0; i < 4; ++i) d = session.observe(tok(17u + (uint32_t)i, 12.0), 17.0 + i);
        require((d.note == "spike_not_sustained" || d.note == "clear_improvement") && !d.rollback,
            "a tail spike that does not repeat is kept when the median improved");
    }

    {
        tpot::Algo2Config cfg = testConfig();
        cfg.verifyTokens = 4;
        cfg.verifyMaxTokens = 12;
        cfg.tailConfirmations = 2;
        tpot::Algo2Session session(cfg);
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) session.observe(tok((uint32_t)i, 20.0), (double)i);
        session.noteCommitResult(true, "");
        session.observe(tok(12u, 20.0), 12.0);
        tpot::Algo2Decision d;
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < 3; ++i) d = session.observe(tok(13u + (uint32_t)(pass * 4 + i), 12.0), 13.0);
            d = session.observe(tok(16u + (uint32_t)(pass * 4), 40.0), 16.0);
        }
        require(d.rollback && d.note == "tail_confirmed_rollback",
            "a tail that stays worse across two windows rolls back even if the median improved");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        tpot::Algo2TokenObs sumOnly;
        sumOnly.tokenId = 0u;
        sumOnly.stageSumKnown = true;
        sumOnly.stageComputeSumMs = 10.0;
        tpot::Algo2Decision d = session.observe(sumOnly, 0.0);
        require(d.note == "e2e_missing", "a stage-compute sum is not an end-to-end health sample");
        for (int i = 0; i < 4; ++i) d = session.observe(tok((uint32_t)i, 58.0), (double)i);
        require(d.health == tpot::Algo2Health::NORMAL && near(d.healthReferenceMs, 58.0),
            "the health anchor is built only from end-to-end intervals");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) session.observe(tok((uint32_t)i, 20.0), (double)i);
        session.noteCommitInDoubt("commit_ack_timeout");
        tpot::Algo2Decision d = session.observe(tok(12u, 20.0), 12.0);
        require(d.phase == tpot::Algo2Phase::RECONCILE && d.note == "reconcile_wait" && !d.issue,
            "a missing commit acknowledgement does not start another move");
        session.noteCommitResult(false, "commit_confirm_timeout");
        d = session.observe(tok(13u, 20.0), 13.0);
        require(d.phase == tpot::Algo2Phase::RECONCILE && !d.issue, "an unresolved commit stays in reconcile");
        session.noteCommitResult(true, "");
        d = session.observe(tok(14u, 20.0), 14.0);
        require(d.note == "settle" || d.note == "settle_done", "a later explicit acknowledgement can still settle the same transaction");
    }

    {
        tpot::Algo2Config cfg = testConfig();
        cfg.rollbackCatchupTokens = 4;
        cfg.verifyTokens = 2;
        tpot::Algo2Session session(cfg);
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) session.observe(tok((uint32_t)i, 20.0), (double)i);
        session.noteCommitResult(true, "");
        session.observe(tok(12u, 20.0), 12.0);
        std::vector<tpot::Algo2StageInput> behind;
        behind.push_back(stage(0u, 0u, 0u, 4u, 10.0));
        behind.push_back(stage(1u, 1u, 4u, 4u, 8.0));
        behind[0].shadowCursor = 0;
        session.setStages(behind);
        tpot::Algo2Decision d = session.observe(tok(13u, 40.0), 13.0);
        d = session.observe(tok(14u, 40.0), 14.0);
        require(!d.rollback && d.note == "rollback_kv_gap_too_large", "a long KV gap is not rolled back from stale state");
        d = session.observe(tok(15u, 40.0), 15.0);
        require(d.note == "rollback_failed", "a blocked rollback stays failed");
    }

    {
        tpot::Algo2Config cfg = testConfig();
        cfg.rollbackCatchupTokens = 8;
        tpot::Algo2Session session(cfg);
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) session.observe(tok((uint32_t)i, 20.0), (double)i);
        session.noteCommitResult(true, "");
        session.observe(tok(12u, 20.0), 12.0);
        std::vector<tpot::Algo2StageInput> closeGap;
        closeGap.push_back(stage(0u, 0u, 0u, 4u, 10.0));
        closeGap.push_back(stage(1u, 1u, 4u, 4u, 8.0));
        closeGap[0].shadowCursor = 12;
        session.setStages(closeGap);
        session.observe(tok(13u, 40.0), 13.0);
        tpot::Algo2Decision d = session.observe(tok(14u, 40.0), 14.0);
        require(d.rollback && d.note == "clear_regression" && d.catchupTokens == 3u && d.catchupMode == "retransmit",
            "a completed token 14 with cursor 12 is three missing positions");
    }

    {
        bool known = false;
        require(tpot::exclusiveCursorGap(81, 81u, &known) == 0u && known, "cursor 81 is ready for the token after 80");
        require(tpot::exclusiveCursorGap(70, 81u, &known) == 11u && known, "cursor 70 after token 80 misses 11 positions");
        known = true;
        require(tpot::exclusiveCursorGap(-1, 81u, &known) == 0u && !known, "an unknown cursor is not a gap of zero");
    }

    {
        tpot::Algo2Config cfg = testConfig();
        cfg.longReferenceSpanTokens = 128;
        tpot::Algo2Session session(cfg);
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        const double medians[4] = {40.0, 43.0, 46.0, 49.0};
        const uint32_t contexts[4] = {100u, 132u, 164u, 196u};
        tpot::Algo2Decision d;
        for (int w = 0; w < 4; ++w) {
            for (int i = 0; i < 4; ++i) {
                tpot::Algo2TokenObs t = tok((uint32_t)(w * 4 + i), medians[w]);
                t.contextLength = contexts[w];
                d = session.observe(t, (double)(w * 4 + i));
            }
        }
        require(d.health == tpot::Algo2Health::DEGRADED || d.health == tpot::Algo2Health::STABLE_DEGRADED,
            "40 to 49 ms within each 10 percent step is still degradation");
        require(near(d.healthReferenceMs, 40.0), "the long anchor does not walk up with each normal window");
        require(!d.contextSlopeTrusted, "a climb shorter than the context span is not treated as context growth");
    }

    {
        tpot::Algo2Config cfg = testConfig();
        cfg.longReferenceSpanTokens = 128;
        tpot::Algo2Session session(cfg);
        arm(&session, tpot::Algo2Mode::STATIC, 100, -1, 2u);
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) {
            tpot::Algo2TokenObs t = tok((uint32_t)i, 40.0);
            t.contextLength = 100u;
            d = session.observe(t, (double)i);
        }
        for (int i = 0; i < 4; ++i) {
            tpot::Algo2TokenObs t = tok(4u + (uint32_t)i, 42.0);
            t.contextLength = 228u;
            d = session.observe(t, 4.0 + i);
        }
        require(d.health == tpot::Algo2Health::NORMAL && d.contextSlopeTrusted, "a long context line can explain a small increase");
        for (int i = 0; i < 4; ++i) {
            tpot::Algo2TokenObs t = tok(8u + (uint32_t)i, 44.0);
            t.contextLength = 356u;
            d = session.observe(t, 8.0 + i);
        }
        require(d.health == tpot::Algo2Health::NORMAL && near(d.healthReferenceMs, 44.0),
            "the context-adjusted anchor follows the fitted line");
        for (int i = 0; i < 4; ++i) {
            tpot::Algo2TokenObs t = tok(12u + (uint32_t)i, 60.0);
            t.contextLength = 484u;
            d = session.observe(t, 12.0 + i);
        }
        require(d.health == tpot::Algo2Health::SUSPECT, "a jump off the context line is suspect");
    }

    {
        tpot::Algo2Config cfg = testConfig();
        cfg.verifyTokens = 4;
        cfg.verifyMaxTokens = 12;
        tpot::Algo2Session session(cfg);
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 20.0), (double)i);
        for (int i = 4; i < 12; ++i) session.observe(tok((uint32_t)i, 40.0), (double)i);
        session.noteCommitResult(true, "");
        session.observe(tok(12u, 40.0), 12.0);
        tpot::Algo2Decision d;
        for (int i = 0; i < 3; ++i) d = session.observe(tok(13u + (uint32_t)i, 20.0), 13.0);
        d = session.observe(tok(16u, 50.0), 16.0);
        require(d.note == "tail_unconfirmed" && !d.rollback, "the first worse tail is only a strike");
        for (int i = 0; i < 4; ++i) d = session.observe(tok(17u + (uint32_t)i, 40.0), 17.0);
        require(d.note == "inconclusive" && !d.rollback,
            "a recovered tail still has to clear the median improvement bar");
    }

    {
        auto feed = [](tpot::Algo2Session *session, double rightMs, double commMs) -> tpot::Algo2Decision {
            std::vector<tpot::Algo2StageInput> stages;
            stages.push_back(stage(0u, 0u, 0u, 4u, 10.0));
            stages.push_back(stage(1u, 1u, 4u, 4u, 5.0));
            stages[0].fixedOverheadMs = -1.0;
            stages[0].snapshot.stageTimeMs = 56.0;
            stages[0].snapshot.leftBoundaryLayerMs = 8.0;
            stages[0].snapshot.rightBoundaryLayerMs = rightMs;
            stages[0].commWaitMs = commMs;
            stages[0].commIncludedInStageTime = false;
            stages[1].fixedOverheadMs = -1.0;
            stages[1].snapshot.stageTimeMs = 20.0;
            stages[1].snapshot.leftBoundaryLayerMs = 5.0;
            stages[1].snapshot.rightBoundaryLayerMs = 5.0;
            stages[1].shadowCursor = 100;
            session->setStages(stages);
            tpot::Algo2Boundary b;
            b.rightReady = 2u;
            session->setBoundaries(std::vector<tpot::Algo2Boundary>(1u, b));
            tpot::Algo2Decision out;
            for (int i = 0; i < 4; ++i) session->observe(tok((uint32_t)i, 10.0), (double)i);
            for (int i = 4; i < 12; ++i) out = session->observe(tok((uint32_t)i, 30.0), (double)i);
            return out;
        };
        tpot::Algo2Session waited(testConfig());
        waited.setMode(tpot::Algo2Mode::REVISED_VARIABLE);
        waited.setSerialRequest(true);
        tpot::Algo2Decision withWait = feed(&waited, 20.0, 30.0);
        tpot::Algo2Session waitedMore(testConfig());
        waitedMore.setMode(tpot::Algo2Mode::REVISED_VARIABLE);
        waitedMore.setSerialRequest(true);
        tpot::Algo2Decision moreWait = feed(&waitedMore, 20.0, 80.0);
        require(withWait.issue && moreWait.issue && near(withWait.candidate.gainMs, moreWait.candidate.gainMs),
            "a larger communication wait does not increase movable gain");
        require(withWait.costConfidence == "medium" &&
            withWait.estimateMethod.find("interior_extrapolated") != std::string::npos,
            "an interior average is recorded as extrapolation");
        tpot::Algo2Session slower(testConfig());
        slower.setMode(tpot::Algo2Mode::REVISED_VARIABLE);
        slower.setSerialRequest(true);
        tpot::Algo2Decision slowLayer = feed(&slower, 30.0, 30.0);
        require(slowLayer.issue && slowLayer.candidate.gainMs > withWait.candidate.gainMs,
            "a slower measured boundary layer increases the gain");
    }

    {
        tpot::Algo2Session session(testConfig());
        arm(&session, tpot::Algo2Mode::REVISED_VARIABLE, 100, -1, 2u);
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) session.observe(tok((uint32_t)i, 20.0), (double)i);
        tpot::Algo2Decision d = session.observe(tok(12u, 50.0), 12.0);
        require(d.note == "commit_wait" && d.intervalRole == "request_only" && !d.intervalInController &&
            near(d.requestIntervalMs, 50.0),
            "the switch token stays in the request latency and out of the controller window");
    }

    {
        // Frozen device-layer costs. The boundary layer is 1 ms worse on the
        // source than on the target, and the layer inside it is 5 ms better.
        // k=1 stays under the 3 ms bar. k=2 clears it. The score is the full
        // assignment difference on that same table.
        tpot::Algo2Session session(testConfig());
        session.setMode(tpot::Algo2Mode::REVISED_VARIABLE);
        session.setSerialRequest(true);
        tpot::Algo2StageInput source = stage(0u, 0u, 0u, 4u, 10.0);
        tpot::Algo2StageInput target = stage(1u, 1u, 4u, 4u, 8.0);
        source.fixedOverheadMs = -1.0;
        target.fixedOverheadMs = -1.0;
        source.layerMs = {10.0, 10.0, 13.0, 7.0};
        source.layerMeasured = {1u, 1u, 1u, 1u};
        source.snapshot.stageTimeMs = 40.0;
        target.layerMs = {8.0, 8.0, 8.0, 8.0};
        target.layerMeasured = {1u, 1u, 1u, 1u};
        target.snapshot.stageTimeMs = 32.0;
        target.shadowCursor = 100;
        std::vector<tpot::Algo2StageInput> stages;
        stages.push_back(source);
        stages.push_back(target);
        session.setStages(stages);
        tpot::Algo2Boundary b;
        b.rightReady = 2u;
        session.setBoundaries(std::vector<tpot::Algo2Boundary>(1u, b));
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 30.0), (double)i);
        const double before = 10.0 + 10.0 + 13.0 + 7.0 + 32.0;
        const double after = 10.0 + 10.0 + 8.0 * 6.0;
        require(d.issue && d.candidate.layerCount == 2u && near(d.candidate.gainMs, before - after),
            "the selected step equals the frozen assignment difference");
        require(near(before - after, 4.0), "two layers clear a bar that one negative layer misses");
        require(d.candidateTrace.find("k=1,gain=-1") != std::string::npos, "the negative first layer stays in the candidate table");
        require(d.candidateTrace.find("k=2,gain=4") != std::string::npos, "the positive two-layer step stays in the candidate table");
        require(d.costConfidence == "high", "measured source and target layers are high confidence");
    }

    {
        tpot::Algo2Config cfg = testConfig();
        cfg.forceFromStage = 0;
        cfg.forceToStage = 1;
        cfg.forceK = 1;
        cfg.maxIssues = 1;
        tpot::Algo2Session session(cfg);
        session.setMode(tpot::Algo2Mode::REVISED_VARIABLE);
        session.setSerialRequest(true);
        tpot::Algo2StageInput source = stage(0u, 0u, 0u, 4u, 10.0);
        tpot::Algo2StageInput target = stage(1u, 1u, 4u, 4u, 8.0);
        source.fixedOverheadMs = -1.0;
        target.fixedOverheadMs = -1.0;
        source.layerMs = {10.0, 10.0, 20.0, 12.0};
        source.layerMeasured = {1u, 1u, 1u, 1u};
        source.snapshot.stageTimeMs = 52.0;
        target.layerMs = {8.0, 8.0, 8.0, 8.0};
        target.layerMeasured = {1u, 1u, 1u, 1u};
        target.snapshot.stageTimeMs = 32.0;
        target.shadowCursor = 100;
        std::vector<tpot::Algo2StageInput> stages;
        stages.push_back(source);
        stages.push_back(target);
        session.setStages(stages);
        tpot::Algo2Boundary b;
        b.rightReady = 2u;
        session.setBoundaries(std::vector<tpot::Algo2Boundary>(1u, b));
        tpot::Algo2Decision d;
        for (int i = 0; i < 4; ++i) session.observe(tok((uint32_t)i, 10.0), (double)i);
        for (int i = 4; i < 12; ++i) d = session.observe(tok((uint32_t)i, 30.0), (double)i);
        require(d.issue && d.candidate.layerCount == 1u && near(d.candidate.gainMs, 4.0),
            "a forced one-layer step is issued even though two layers score higher");
        session.noteCommitResult(false, "stop");
        for (int i = 12; i < 16; ++i) d = session.observe(tok((uint32_t)i, 30.0), (double)i);
        require(!d.issue && d.note == "issue_cap", "a one-move cap does not issue the next step");
    }

    std::printf("algo2 checks passed\n");
    return 0;
}
