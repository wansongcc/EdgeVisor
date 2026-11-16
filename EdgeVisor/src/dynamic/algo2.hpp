#pragma once

#include "dynamic/tpot_algorithm.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dllama {
namespace dynamic_tpot {

// Online discrete local search inside the shadow that is already ready.
// It is not a global optimum and it is not gradient descent.
enum class Algo2Mode {
    STATIC = 0,
    LEGACY = 1,
    REVISED_SINGLE = 2,
    REVISED_VARIABLE = 3,
};

enum class Algo2Health {
    INSUFFICIENT = 0,
    NORMAL,
    SUSPECT,
    DEGRADED,
    STABLE_DEGRADED,
    NO_PROGRESS,
};

enum class Algo2Phase {
    OBSERVE = 0,
    PLAN,
    COMMIT,
    SETTLE,
    VERIFY,
    KEEP,
    ROLLBACK,
    COOLDOWN,
};

struct Algo2Config {
    int metricsWindowTokens = 8;
    int decisionWindowTokens = 32;
    int decisionStrideTokens = 32;
    int minDecisionIntervalMs = 1000;
    double degradationTriggerRatio = 0.10;
    int degradationConfirmations = 2;
    double recoveryRatio = 0.05;
    int recoveryConfirmations = 2;
    double gainAbsMs = 3.0;
    double gainRelativePair = 0.03;
    // Negative means the prediction-noise term was not calibrated. It is not
    // treated as zero and it is not added into the threshold.
    double predictionNoiseMs = -1.0;
    int settleTokens = 12;
    int verifyTokens = 32;
    int verifyMaxTokens = 96;
    int cooldownTokens = 32;
    int cooldownMinMs = 1000;
    uint32_t minLayersPerStage = 1u;
    // 0 means the feasible ready range, not a hidden cap of 1.
    uint32_t maxMoveLayers = 0u;
    bool proactiveRebalance = false;
    // Fixed starting point for "the window is tight". Not a fitted noise floor.
    double stableBand = 0.10;
    // Fixed conservative verify band. Not a claim that 5% was calibrated here.
    double verifyRatio = 0.05;
    int watchdogMs = 5000;
    double watchdogFactor = 5.0;
    int reverseSuppressDecisions = 2;
};

struct Algo2Boundary {
    uint32_t leftReady = 0u;  // layers that can move from the right stage to the left
    uint32_t rightReady = 0u; // layers that can move from the left stage to the right
    // -1: spare was not measured again. Provisioned shadow is the cap.
    // >=0: tighten the move to this many extra layers.
    int memorySpareLayers = -1;
};

struct Algo2StageInput {
    StageSnapshot snapshot;
    // Negative: fixed stage overhead was not separated from the layer average.
    double fixedOverheadMs = -1.0;
    // Empty, or one entry per layer in [startLayer, endLayer).
    std::vector<double> layerMs;
    // Exclusive shadow cursor for this stage. -1 unknown. -2 this stage has
    // no redundant layers. Any other value is the exclusive end position whose
    // shadow KV has been written on every redundant layer.
    int shadowCursor = -1;
};

struct Algo2TokenObs {
    uint32_t tokenId = 0u;
    uint32_t contextLength = 0u;
    uint64_t partitionVersion = 0u;
    double e2eMs = -1.0;
    bool e2eKnown = false;
    double stageComputeSumMs = -1.0;
    bool stageSumKnown = false;
    std::map<uint32_t, double> stageComputeMs;
    std::map<uint32_t, double> stageCommMs;
    // Per stage: -1 unknown, -2 no redundant layers, otherwise exclusive shadow end.
    std::map<uint32_t, int> shadowCursor;
    bool prefill = false;
};

struct Algo2Decision {
    Algo2Phase phase = Algo2Phase::OBSERVE;
    Algo2Health health = Algo2Health::INSUFFICIENT;
    bool issue = false;
    bool rollback = false;
    Candidate candidate;
    std::string note;
    std::string cause; // compute_degradation, communication_or_wait_degradation, mixed, unknown
    std::string estimateMethod;
    bool noiseCalibrated = false;
    bool recovered = false;
    uint32_t locateStage = 0xFFFFFFFFu;
    double referenceMs = -1.0;
    double windowMedianMs = -1.0;
    double windowMeanMs = -1.0;
    double windowP95Ms = -1.0;
};

struct OfflineSearchResult {
    double optimalMs = 0.0;
    double localMs = 0.0;
    bool localIsOptimal = false;
    bool exhaustive = false;
    std::string note;
};

Algo2Mode parseAlgo2Mode(const char *text);
const char *algo2ModeName(Algo2Mode mode);
const char *algo2HealthName(Algo2Health health);
const char *algo2PhaseName(Algo2Phase phase);
Algo2Config algo2ConfigFromEnvironment();
bool algo2ForbidsFullKvFallback();

// Root decode loop. The first completion only starts the clock. Later calls
// record complete(t) - complete(t-1) on this host. The interval includes
// sampling and the return of the token to the root. It does not include the
// following printf.
void algo2ResetTokenClock();
void algo2NoteTokenComplete(uint32_t tokenId, uint32_t contextLength, bool includesSampling);

struct Algo2CompletedToken {
    uint32_t tokenId = 0u;
    uint32_t contextLength = 0u;
    double intervalMs = 0.0;
    bool includesSampling = false;
};
std::vector<Algo2CompletedToken> algo2DrainTokenCompletes();

class Algo2Session {
public:
    explicit Algo2Session(Algo2Config config);

    void setMode(Algo2Mode mode);
    void setSerialRequest(bool serial);
    void setStages(const std::vector<Algo2StageInput> &stages);
    void setBoundaries(const std::vector<Algo2Boundary> &boundaries);

    Algo2Decision observe(const Algo2TokenObs &token, double nowMs);
    Algo2Decision pollIdle(double nowMs);
    void noteCommitResult(bool applied, const std::string &reason);

    const Algo2Config &config() const { return cfg_; }
    std::string effectiveConfigText() const;

private:
    Algo2Decision make(const std::string &note) const;
    void captureStageReference(size_t windowTokens);
    void updateHealth(const std::vector<double> &window, double median, double p95);
    Candidate search(uint32_t tokenId, std::string *method, std::string *reject) const;

    Algo2Config cfg_;
    Algo2Mode mode_ = Algo2Mode::LEGACY;
    bool serial_ = true;
    std::vector<Algo2StageInput> stages_;
    std::vector<Algo2Boundary> boundaries_;

    struct Sample {
        uint32_t tokenId = 0u;
        uint32_t contextLength = 0u;
        double metric = 0.0;
        bool e2e = false;
        std::map<uint32_t, double> compute;
        std::map<uint32_t, double> comm;
    };
    std::vector<Sample> samples_;
    Algo2Phase phase_ = Algo2Phase::OBSERVE;
    Algo2Health health_ = Algo2Health::INSUFFICIENT;
    std::string cause_ = "unknown";
    uint32_t locateStage_ = 0xFFFFFFFFu;
    bool referenceValid_ = false;
    double referenceMs_ = -1.0;
    uint32_t referenceContext_ = 0u;
    std::map<uint32_t, double> referenceCompute_;
    std::map<uint32_t, double> referenceComm_;
    int breachCount_ = 0;
    int recoveryCount_ = 0;
    bool recovered_ = false;
    int observeTokens_ = 0;
    int lastDecisionTokens_ = -1000000;
    double lastDecisionMs_ = -1.0e100;
    double lastTokenMs_ = -1.0;
    bool haveTokenTime_ = false;
    int settleLeft_ = 0;
    std::vector<double> verify_;
    double beforeMedian_ = -1.0;
    double beforeP95_ = -1.0;
    int cooldownLeft_ = 0;
    double cooldownUntilMs_ = 0.0;
    Candidate pending_;
    bool havePending_ = false;
    bool sawKeptMove_ = false;
    uint32_t keptFrom_ = 0u;
    uint32_t keptTo_ = 0u;
    int reverseLeft_ = 0;
    bool rollbackBlocked_ = false;
    std::string lastNote_;
};

// costByLayerDevice[layer][device]. initialCounts sums to the layer count.
// Exhaustive only for a small instance. The local search is one boundary at
// a time and can stop short of the best legal assignment.
OfflineSearchResult compareLocalSearchToExhaustive(
    const std::vector<std::vector<double>> &costByLayerDevice,
    const std::vector<uint32_t> &initialCounts,
    uint32_t minLayers);

} // namespace dynamic_tpot
} // namespace dllama
