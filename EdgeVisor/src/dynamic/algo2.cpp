#include "dynamic/algo2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <sstream>

namespace dllama {
namespace dynamic_tpot {

namespace {

std::mutex g_tokenMu;
bool g_haveTokenClock = false;
double g_lastTokenMs = 0.0;
std::deque<Algo2CompletedToken> g_tokens;

double steadyMs() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

double medianOf(std::vector<double> values) {
    if (values.empty()) return -1.0;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if ((n % 2u) == 1u) return values[n / 2u];
    return 0.5 * (values[n / 2u - 1u] + values[n / 2u]);
}

double p95Of(std::vector<double> values) {
    if (values.empty()) return -1.0;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    size_t rank = (95u * n + 99u) / 100u;
    if (rank < 1u) rank = 1u;
    if (rank > n) rank = n;
    return values[rank - 1u];
}

double meanOf(const std::vector<double> &values) {
    if (values.empty()) return -1.0;
    double sum = 0.0;
    for (size_t i = 0u; i < values.size(); ++i) sum += values[i];
    return sum / (double)values.size();
}

const char *envOr(const char *name, const char *fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    return v;
}

int envInt(const char *name, int fallback, int minimum) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    char *end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || x < (long)minimum) return fallback;
    return (int)x;
}

double envDouble(const char *name, double fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    char *end = nullptr;
    const double x = std::strtod(v, &end);
    if (end == v || *end != '\0' || !std::isfinite(x)) return fallback;
    return x;
}

bool envBool(const char *name, bool fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    if (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "TRUE") == 0) return true;
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0) return false;
    return fallback;
}

struct LayerEstimate {
    double perLayerMs = 0.0;
    bool known = false;
    bool haveLayers = false;
    bool fixedSeparated = false;
    std::string method;
};

LayerEstimate estimateStage(const Algo2StageInput &stage) {
    LayerEstimate out;
    const uint32_t n = stage.snapshot.nLayers;
    if (n > 0u && stage.layerMs.size() == (size_t)n) {
        double sum = 0.0;
        bool ok = true;
        for (size_t i = 0u; i < stage.layerMs.size(); ++i) {
            if (!(stage.layerMs[i] >= 0.0)) ok = false;
            sum += stage.layerMs[i];
        }
        if (ok) {
            out.perLayerMs = sum / (double)n;
            out.known = true;
            out.haveLayers = true;
            out.fixedSeparated = true;
            out.method = "per_layer";
            return out;
        }
    }
    if (n > 0u && stage.fixedOverheadMs >= 0.0 && stage.snapshot.stageTimeMs >= stage.fixedOverheadMs) {
        double movable = stage.snapshot.stageTimeMs - stage.fixedOverheadMs;
        if (stage.commIncludedInStageTime && stage.commWaitMs > 0.0) movable -= stage.commWaitMs;
        if (movable < 0.0) movable = 0.0;
        out.perLayerMs = movable / (double)n;
        out.known = true;
        out.fixedSeparated = true;
        out.method = "stage_average_excluding_fixed_monotonic_k";
        return out;
    }
    // A stage average that still contains an unknown fixed cost is not a
    // movable layer cost. Naming it is not enough to let it drive a move.
    out.method = "fixed_overhead_not_separated";
    return out;
}

double movedSourceCost(const Algo2StageInput &source, uint32_t k, bool takeSuffix) {
    const LayerEstimate est = estimateStage(source);
    if (!est.known || k == 0u) return -1.0;
    if (est.haveLayers && source.layerMs.size() == (size_t)source.snapshot.nLayers && k <= source.snapshot.nLayers) {
        double sum = 0.0;
        const uint32_t begin = takeSuffix ? source.snapshot.nLayers - k : 0u;
        for (uint32_t i = 0u; i < k; ++i) sum += source.layerMs[(size_t)begin + (size_t)i];
        return sum;
    }
    return est.perLayerMs * (double)k;
}

bool betterMove(const Candidate &cand, const Candidate &best, uint32_t candMargin, uint32_t bestMargin) {
    if (!best.valid) return true;
    if (cand.gainMs > best.gainMs + 1e-9) return true;
    if (best.gainMs > cand.gainMs + 1e-9) return false;
    if (cand.layerCount < best.layerCount) return true;
    if (cand.layerCount > best.layerCount) return false;
    return candMargin > bestMargin;
}

double assignmentCost(
    const std::vector<std::vector<double>> &cost,
    const std::vector<uint32_t> &counts) {
    double total = 0.0;
    uint32_t layer = 0u;
    for (size_t d = 0u; d < counts.size(); ++d) {
        for (uint32_t i = 0u; i < counts[d]; ++i, ++layer) {
            if (layer >= cost.size() || d >= cost[layer].size()) return 1.0e300;
            total += cost[layer][d];
        }
    }
    return total;
}

void enumerateCounts(
    const std::vector<std::vector<double>> &cost,
    uint32_t minLayers,
    uint32_t layersLeft,
    uint32_t devicesLeft,
    std::vector<uint32_t> *prefix,
    double *best,
    std::vector<uint32_t> *bestCounts) {
    if (devicesLeft == 1u) {
        if (layersLeft < minLayers) return;
        prefix->push_back(layersLeft);
        const double total = assignmentCost(cost, *prefix);
        if (total < *best) {
            *best = total;
            *bestCounts = *prefix;
        }
        prefix->pop_back();
        return;
    }
    const uint32_t reserve = minLayers * (devicesLeft - 1u);
    if (layersLeft < minLayers + reserve) return;
    for (uint32_t take = minLayers; take + reserve <= layersLeft; ++take) {
        prefix->push_back(take);
        enumerateCounts(cost, minLayers, layersLeft - take, devicesLeft - 1u, prefix, best, bestCounts);
        prefix->pop_back();
    }
}

} // namespace

Algo2Mode parseAlgo2Mode(const char *text) {
    if (text == nullptr || text[0] == '\0' || std::strcmp(text, "legacy") == 0) return Algo2Mode::LEGACY;
    if (std::strcmp(text, "static") == 0) return Algo2Mode::STATIC;
    if (std::strcmp(text, "revised_single_layer") == 0) return Algo2Mode::REVISED_SINGLE;
    if (std::strcmp(text, "revised_variable_step") == 0) return Algo2Mode::REVISED_VARIABLE;
    return Algo2Mode::LEGACY;
}

const char *algo2ModeName(Algo2Mode mode) {
    switch (mode) {
        case Algo2Mode::STATIC: return "static";
        case Algo2Mode::LEGACY: return "legacy";
        case Algo2Mode::REVISED_SINGLE: return "revised_single_layer";
        case Algo2Mode::REVISED_VARIABLE: return "revised_variable_step";
        default: return "legacy";
    }
}

const char *algo2HealthName(Algo2Health health) {
    switch (health) {
        case Algo2Health::INSUFFICIENT: return "insufficient_data";
        case Algo2Health::NORMAL: return "normal";
        case Algo2Health::SUSPECT: return "suspect";
        case Algo2Health::DEGRADED: return "degraded";
        case Algo2Health::STABLE_DEGRADED: return "stable_degraded";
        case Algo2Health::NO_PROGRESS: return "no_progress";
        default: return "insufficient_data";
    }
}

const char *algo2PhaseName(Algo2Phase phase) {
    switch (phase) {
        case Algo2Phase::OBSERVE: return "OBSERVE";
        case Algo2Phase::PLAN: return "PLAN";
        case Algo2Phase::COMMIT: return "COMMIT";
        case Algo2Phase::SETTLE: return "SETTLE";
        case Algo2Phase::VERIFY: return "VERIFY";
        case Algo2Phase::KEEP: return "KEEP";
        case Algo2Phase::ROLLBACK: return "ROLLBACK";
        case Algo2Phase::COOLDOWN: return "COOLDOWN";
        case Algo2Phase::RECONCILE: return "RECONCILE";
        default: return "OBSERVE";
    }
}

Algo2Config algo2ConfigFromEnvironment() {
    Algo2Config cfg;
    cfg.metricsWindowTokens = envInt("DLLAMA_ALGO2_METRICS_WINDOW", cfg.metricsWindowTokens, 1);
    cfg.decisionWindowTokens = envInt("DLLAMA_ALGO2_DECISION_WINDOW", cfg.decisionWindowTokens, 1);
    cfg.decisionStrideTokens = envInt("DLLAMA_ALGO2_DECISION_STRIDE", cfg.decisionStrideTokens, 1);
    cfg.minDecisionIntervalMs = envInt("DLLAMA_ALGO2_MIN_DECISION_INTERVAL_MS", cfg.minDecisionIntervalMs, 0);
    cfg.degradationTriggerRatio = envDouble("DLLAMA_ALGO2_DEGRADATION_TRIGGER", cfg.degradationTriggerRatio);
    cfg.degradationConfirmations = envInt("DLLAMA_ALGO2_DEGRADATION_CONFIRMATIONS", cfg.degradationConfirmations, 1);
    cfg.recoveryRatio = envDouble("DLLAMA_ALGO2_RECOVERY_RATIO", cfg.recoveryRatio);
    cfg.recoveryConfirmations = envInt("DLLAMA_ALGO2_RECOVERY_CONFIRMATIONS", cfg.recoveryConfirmations, 1);
    cfg.gainAbsMs = envDouble("DLLAMA_ALGO2_GAIN_ABS_MS", cfg.gainAbsMs);
    cfg.gainRelativePair = envDouble("DLLAMA_ALGO2_GAIN_RELATIVE", cfg.gainRelativePair);
    if (std::getenv("DLLAMA_ALGO2_PREDICTION_NOISE_MS") != nullptr)
        cfg.predictionNoiseMs = envDouble("DLLAMA_ALGO2_PREDICTION_NOISE_MS", cfg.predictionNoiseMs);
    cfg.settleTokens = envInt("DLLAMA_ALGO2_SETTLE_TOKENS", cfg.settleTokens, 0);
    cfg.verifyTokens = envInt("DLLAMA_ALGO2_VERIFY_TOKENS", cfg.verifyTokens, 1);
    cfg.verifyMaxTokens = envInt("DLLAMA_ALGO2_VERIFY_MAX_TOKENS", cfg.verifyMaxTokens, cfg.verifyTokens);
    cfg.cooldownTokens = envInt("DLLAMA_ALGO2_COOLDOWN_TOKENS", cfg.cooldownTokens, 0);
    cfg.cooldownMinMs = envInt("DLLAMA_ALGO2_COOLDOWN_MIN_MS", cfg.cooldownMinMs, 0);
    cfg.minLayersPerStage = (uint32_t)envInt("DLLAMA_ALGO2_MIN_LAYERS", (int)cfg.minLayersPerStage, 1);
    cfg.maxMoveLayers = (uint32_t)envInt("DLLAMA_ALGO2_MAX_MOVE_LAYERS", (int)cfg.maxMoveLayers, 0);
    cfg.proactiveRebalance = envBool("DLLAMA_ALGO2_PROACTIVE", cfg.proactiveRebalance);
    cfg.stableBand = envDouble("DLLAMA_ALGO2_STABLE_BAND", cfg.stableBand);
    cfg.verifyRatio = envDouble("DLLAMA_ALGO2_VERIFY_RATIO", cfg.verifyRatio);
    cfg.watchdogMs = envInt("DLLAMA_ALGO2_WATCHDOG_MS", cfg.watchdogMs, 1);
    cfg.watchdogFactor = envDouble("DLLAMA_ALGO2_WATCHDOG_FACTOR", cfg.watchdogFactor);
    cfg.reverseSuppressDecisions = envInt("DLLAMA_ALGO2_REVERSE_SUPPRESS", cfg.reverseSuppressDecisions, 0);
    cfg.tailConfirmations = envInt("DLLAMA_ALGO2_TAIL_CONFIRMATIONS", cfg.tailConfirmations, 2);
    cfg.rollbackCatchupTokens = envInt("DLLAMA_ALGO2_ROLLBACK_CATCHUP", cfg.rollbackCatchupTokens, 0);
    if (cfg.verifyMaxTokens < cfg.verifyTokens) cfg.verifyMaxTokens = cfg.verifyTokens;
    return cfg;
}

bool algo2ForbidsFullKvFallback() {
    const Algo2Mode mode = parseAlgo2Mode(std::getenv("DLLAMA_ALGO2_MODE"));
    return mode == Algo2Mode::REVISED_SINGLE || mode == Algo2Mode::REVISED_VARIABLE;
}

void algo2ResetTokenClock() {
    std::lock_guard<std::mutex> lock(g_tokenMu);
    g_haveTokenClock = false;
    g_tokens.clear();
}

void algo2NoteTokenComplete(uint32_t tokenId, uint32_t contextLength, bool includesSampling) {
    const double now = steadyMs();
    std::lock_guard<std::mutex> lock(g_tokenMu);
    if (!g_haveTokenClock) {
        g_haveTokenClock = true;
        g_lastTokenMs = now;
        return;
    }
    Algo2CompletedToken item;
    item.tokenId = tokenId;
    item.contextLength = contextLength;
    item.intervalMs = now - g_lastTokenMs;
    item.includesSampling = includesSampling;
    g_lastTokenMs = now;
    g_tokens.push_back(item);
    while (g_tokens.size() > 4096u) g_tokens.pop_front();
}

std::vector<Algo2CompletedToken> algo2DrainTokenCompletes() {
    std::lock_guard<std::mutex> lock(g_tokenMu);
    std::vector<Algo2CompletedToken> out(g_tokens.begin(), g_tokens.end());
    g_tokens.clear();
    return out;
}

Algo2Session::Algo2Session(Algo2Config config) : cfg_(config) {
    mode_ = parseAlgo2Mode(envOr("DLLAMA_ALGO2_MODE", "legacy"));
}

void Algo2Session::setMode(Algo2Mode mode) { mode_ = mode; }
void Algo2Session::setSerialRequest(bool serial) { serial_ = serial; }
void Algo2Session::setStages(const std::vector<Algo2StageInput> &stages) { stages_ = stages; }
void Algo2Session::setBoundaries(const std::vector<Algo2Boundary> &boundaries) { boundaries_ = boundaries; }

std::string Algo2Session::effectiveConfigText() const {
    std::ostringstream out;
    out << "mode=" << algo2ModeName(mode_)
        << " metrics_window=" << cfg_.metricsWindowTokens
        << " decision_window=" << cfg_.decisionWindowTokens
        << " decision_stride=" << cfg_.decisionStrideTokens
        << " min_decision_interval_ms=" << cfg_.minDecisionIntervalMs
        << " trigger=" << cfg_.degradationTriggerRatio
        << " confirmations=" << cfg_.degradationConfirmations
        << " recovery=" << cfg_.recoveryRatio
        << " recovery_confirmations=" << cfg_.recoveryConfirmations
        << " gain_abs_ms=" << cfg_.gainAbsMs
        << " gain_relative_pair=" << cfg_.gainRelativePair
        << " prediction_noise_ms=" << cfg_.predictionNoiseMs
        << " prediction_noise=" << (cfg_.predictionNoiseMs >= 0.0 ? "calibrated" : "uncalibrated")
        << " settle=" << cfg_.settleTokens
        << " verify=" << cfg_.verifyTokens
        << " verify_max=" << cfg_.verifyMaxTokens
        << " verify_ratio=" << cfg_.verifyRatio
        << " verify_band=fixed_conservative_uncalibrated"
        << " tail_confirmations=" << cfg_.tailConfirmations
        << " rollback_catchup_tokens=" << cfg_.rollbackCatchupTokens
        << " cooldown_tokens=" << cfg_.cooldownTokens
        << " cooldown_min_ms=" << cfg_.cooldownMinMs
        << " min_layers=" << cfg_.minLayersPerStage
        << " max_move_layers=" << (cfg_.maxMoveLayers == 0u ? std::string("feasible_ready_range") : std::to_string(cfg_.maxMoveLayers))
        << " proactive=" << (cfg_.proactiveRebalance ? 1 : 0)
        << " stable_band=" << cfg_.stableBand
        << " stable_band_status=uncalibrated"
        << " watchdog_ms=" << cfg_.watchdogMs
        << " watchdog_factor=" << cfg_.watchdogFactor
        << " watchdog_status=uncalibrated"
        << " serial=" << (serial_ ? 1 : 0);
    return out.str();
}

Algo2Decision Algo2Session::make(const std::string &note) const {
    Algo2Decision d;
    d.phase = phase_;
    d.health = health_;
    d.note = note;
    d.cause = cause_;
    d.recovered = recovered_;
    d.locateStage = locateStage_;
    d.referenceMs = referenceValid_ ? referenceMs_ : -1.0;
    d.healthReferenceValid = referenceValid_;
    d.healthReferenceMs = referenceValid_ ? referenceMs_ : -1.0;
    d.healthReferenceContext = referenceContext_;
    d.healthReferenceReason = healthReferenceReason_;
    d.partitionReferenceValid = partitionReferenceValid_;
    d.partitionReferenceMs = partitionReferenceValid_ ? partitionReferenceMs_ : -1.0;
    d.preActionMedianMs = beforeMedian_;
    d.preActionP95Ms = beforeP95_;
    d.tailStrikes = tailStrikes_;
    d.verifySamples = verifySamplesSeen_;
    d.noiseCalibrated = cfg_.predictionNoiseMs >= 0.0;
    d.candidate.kind = CandidateKind::PP_MOVE;
    d.candidate.reason = note;
    return d;
}

void Algo2Session::captureStageReference(size_t windowTokens) {
    referenceCompute_.clear();
    referenceComm_.clear();
    if (samples_.empty() || windowTokens == 0u) return;
    const size_t begin = samples_.size() > windowTokens ? samples_.size() - windowTokens : 0u;
    std::map<uint32_t, std::vector<double>> comp;
    std::map<uint32_t, std::vector<double>> comm;
    for (size_t i = begin; i < samples_.size(); ++i) {
        for (std::map<uint32_t, double>::const_iterator it = samples_[i].compute.begin(); it != samples_[i].compute.end(); ++it)
            comp[it->first].push_back(it->second);
        for (std::map<uint32_t, double>::const_iterator it = samples_[i].comm.begin(); it != samples_[i].comm.end(); ++it)
            comm[it->first].push_back(it->second);
    }
    for (std::map<uint32_t, std::vector<double>>::iterator it = comp.begin(); it != comp.end(); ++it)
        referenceCompute_[it->first] = medianOf(it->second);
    for (std::map<uint32_t, std::vector<double>>::iterator it = comm.begin(); it != comm.end(); ++it)
        referenceComm_[it->first] = medianOf(it->second);
}

void Algo2Session::capturePartitionReference(const std::vector<Sample> &window) {
    partitionCompute_.clear();
    partitionComm_.clear();
    if (window.empty()) return;
    std::map<uint32_t, std::vector<double>> comp;
    std::map<uint32_t, std::vector<double>> comm;
    double metricSum = 0.0;
    for (size_t i = 0u; i < window.size(); ++i) {
        metricSum += window[i].metric;
        for (std::map<uint32_t, double>::const_iterator it = window[i].compute.begin(); it != window[i].compute.end(); ++it)
            comp[it->first].push_back(it->second);
        for (std::map<uint32_t, double>::const_iterator it = window[i].comm.begin(); it != window[i].comm.end(); ++it)
            comm[it->first].push_back(it->second);
    }
    std::vector<double> metrics;
    for (size_t i = 0u; i < window.size(); ++i) metrics.push_back(window[i].metric);
    partitionReferenceMs_ = medianOf(metrics);
    partitionReferenceValid_ = true;
    (void)metricSum;
    for (std::map<uint32_t, std::vector<double>>::iterator it = comp.begin(); it != comp.end(); ++it)
        partitionCompute_[it->first] = medianOf(it->second);
    for (std::map<uint32_t, std::vector<double>>::iterator it = comm.begin(); it != comm.end(); ++it)
        partitionComm_[it->first] = medianOf(it->second);
}

void Algo2Session::updateHealth(const std::vector<double> &window, double median, double p95) {
    if (!referenceValid_) {
        referenceValid_ = true;
        referenceMs_ = median;
        referenceContext_ = samples_.empty() ? 0u : samples_.back().contextLength;
        healthReferenceReason_ = "established";
        captureStageReference(window.size());
        health_ = Algo2Health::NORMAL;
        breachCount_ = 0;
        recoveryCount_ = 0;
        recovered_ = false;
        cause_ = "unknown";
        return;
    }

    const bool slow = referenceMs_ > 0.0 && median > referenceMs_ * (1.0 + cfg_.degradationTriggerRatio);
    const bool back = referenceMs_ > 0.0 && median <= referenceMs_ * (1.0 + cfg_.recoveryRatio);
    const bool tight = median > 0.0 && ((p95 - median) / median) <= cfg_.stableBand;

    if (health_ == Algo2Health::DEGRADED || health_ == Algo2Health::STABLE_DEGRADED) {
        if (back) {
            recoveryCount_ += 1;
            if (recoveryCount_ >= cfg_.recoveryConfirmations) {
                health_ = Algo2Health::NORMAL;
                recovered_ = true;
                referenceMs_ = median;
                referenceContext_ = samples_.empty() ? referenceContext_ : samples_.back().contextLength;
                healthReferenceReason_ = "recovered_to_health_anchor";
                captureStageReference(window.size());
                breachCount_ = 0;
                recoveryCount_ = 0;
            }
        } else {
            recoveryCount_ = 0;
            health_ = tight ? Algo2Health::STABLE_DEGRADED : Algo2Health::DEGRADED;
        }
    } else if (slow) {
        breachCount_ += 1;
        recoveryCount_ = 0;
        recovered_ = false;
        if (breachCount_ >= cfg_.degradationConfirmations) {
            health_ = tight ? Algo2Health::STABLE_DEGRADED : Algo2Health::DEGRADED;
        } else {
            health_ = Algo2Health::SUSPECT;
        }
    } else {
        breachCount_ = 0;
        health_ = Algo2Health::NORMAL;
        referenceMs_ = median;
        referenceContext_ = samples_.empty() ? referenceContext_ : samples_.back().contextLength;
        healthReferenceReason_ = "normal_update";
        captureStageReference(window.size());
        recovered_ = false;
    }

    locateStage_ = 0xFFFFFFFFu;
    bool computeUp = false;
    bool commUp = false;
    double worst = 1.0;
    if (!samples_.empty() && window.size() > 0u) {
        std::vector<const Sample *> epochSamples;
        for (size_t i = 0u; i < samples_.size(); ++i) {
            if (samples_[i].epoch == partitionEpoch_) epochSamples.push_back(&samples_[i]);
        }
        if (epochSamples.size() > window.size())
            epochSamples.erase(epochSamples.begin(), epochSamples.end() - (std::ptrdiff_t)window.size());
        std::map<uint32_t, std::vector<double>> comp;
        std::map<uint32_t, std::vector<double>> comm;
        for (size_t i = 0u; i < epochSamples.size(); ++i) {
            for (std::map<uint32_t, double>::const_iterator it = epochSamples[i]->compute.begin(); it != epochSamples[i]->compute.end(); ++it)
                comp[it->first].push_back(it->second);
            for (std::map<uint32_t, double>::const_iterator it = epochSamples[i]->comm.begin(); it != epochSamples[i]->comm.end(); ++it)
                comm[it->first].push_back(it->second);
        }
        for (std::map<uint32_t, std::vector<double>>::iterator it = comp.begin(); it != comp.end(); ++it) {
            const std::map<uint32_t, double> &computeRef = partitionReferenceValid_ ? partitionCompute_ : referenceCompute_;
            std::map<uint32_t, double>::const_iterator ref = computeRef.find(it->first);
            if (ref == computeRef.end() || !(ref->second > 0.0)) continue;
            const double now = medianOf(it->second);
            const double ratio = now / ref->second;
            if (ratio > 1.0 + cfg_.degradationTriggerRatio) {
                computeUp = true;
                if (ratio > worst) {
                    worst = ratio;
                    locateStage_ = it->first;
                }
            }
        }
        for (std::map<uint32_t, std::vector<double>>::iterator it = comm.begin(); it != comm.end(); ++it) {
            const std::map<uint32_t, double> &commRef = partitionReferenceValid_ ? partitionComm_ : referenceComm_;
            std::map<uint32_t, double>::const_iterator ref = commRef.find(it->first);
            if (ref == commRef.end() || !(ref->second > 0.0)) continue;
            const double now = medianOf(it->second);
            if (now > ref->second * (1.0 + cfg_.degradationTriggerRatio)) commUp = true;
        }
    }
    if (computeUp && commUp) cause_ = "mixed";
    else if (computeUp) cause_ = "compute_degradation";
    else if (commUp) cause_ = "communication_or_wait_degradation";
    else cause_ = "unknown";
}

bool Algo2Session::rollbackCatchup(uint32_t tokenId, uint32_t *gap, std::string *block) const {
    if (gap != nullptr) *gap = 0u;
    const Algo2StageInput *receiver = nullptr;
    for (size_t i = 0u; i < stages_.size(); ++i) {
        if (stages_[i].snapshot.stageIndex == pending_.fromStageIndex) {
            receiver = &stages_[i];
            break;
        }
    }
    if (receiver == nullptr) {
        if (block != nullptr) *block = "rollback_kv_unknown";
        return false;
    }
    if (receiver->shadowCursor < 0) {
        if (block != nullptr) *block = receiver->shadowCursor == -2 ? "rollback_shadow_missing" : "rollback_kv_unknown";
        return false;
    }
    const uint32_t need = tokenId + 1u;
    if ((uint32_t)receiver->shadowCursor >= need) return true;
    const uint32_t missing = need - (uint32_t)receiver->shadowCursor;
    if (cfg_.rollbackCatchupTokens >= 0 && (int)missing <= cfg_.rollbackCatchupTokens) {
        if (gap != nullptr) *gap = missing;
        return true;
    }
    if (block != nullptr) *block = "rollback_kv_gap_too_large";
    return false;
}

Candidate Algo2Session::search(uint32_t tokenId, std::string *method, std::string *reject,
        double *sourceCost, double *targetCost, std::string *confidence) const {
    Candidate best;
    best.kind = CandidateKind::PP_MOVE;
    best.reason = "no eligible pp candidate";
    uint32_t bestMargin = 0u;
    bool anyReject = false;
    std::string rejectReason = "no eligible pp candidate";
    if (method != nullptr) method->clear();

    if (stages_.size() < 2u || boundaries_.size() + 1u < stages_.size()) {
        if (reject != nullptr) *reject = "readiness_missing";
        best.reason = "readiness_missing";
        return best;
    }

    for (size_t i = 0u; i + 1u < stages_.size(); ++i) {
        const Algo2StageInput *pair[2] = {&stages_[i], &stages_[i + 1u]};
        const bool toRight[2] = {true, false};
        for (int d = 0; d < 2; ++d) {
            const Algo2StageInput &source = *pair[d == 0 ? 0 : 1];
            const Algo2StageInput &target = *pair[d == 0 ? 1 : 0];
            const bool takeSuffix = toRight[d];
            const Algo2Boundary &boundary = boundaries_[i];
            uint32_t ready = takeSuffix ? boundary.rightReady : boundary.leftReady;
            std::string why;
            if (target.shadowCursor < 0) {
                ready = 0u;
                why = target.shadowCursor == -2 ? "shadow_not_provisioned" : "kv_readiness_unknown";
            } else if ((uint32_t)target.shadowCursor < tokenId + 1u) {
                ready = 0u;
                why = "kv_behind";
            }
            if (boundary.memorySpareLayers >= 0)
                ready = std::min(ready, (uint32_t)boundary.memorySpareLayers);
            if (boundary.memorySpareLayers == 0) why = "memory_spare";
            const uint32_t sourceLayers = source.snapshot.nLayers;
            uint32_t room = sourceLayers > cfg_.minLayersPerStage ? sourceLayers - cfg_.minLayersPerStage : 0u;
            if (mode_ == Algo2Mode::REVISED_SINGLE) room = std::min(room, 1u);
            if (cfg_.maxMoveLayers > 0u) room = std::min(room, cfg_.maxMoveLayers);
            const uint32_t kmax = std::min(ready, room);
            if (kmax == 0u) {
                anyReject = true;
                if (!why.empty()) rejectReason = why;
                continue;
            }
            const LayerEstimate sourceEst = estimateStage(source);
            const LayerEstimate targetEst = estimateStage(target);
            if (!sourceEst.known || !targetEst.known) {
                anyReject = true;
                const bool unseparated = sourceEst.method == "fixed_overhead_not_separated" ||
                    targetEst.method == "fixed_overhead_not_separated";
                rejectReason = unseparated ? "fixed_overhead_not_separated" : "layer_cost_unknown";
                continue;
            }
            const double pairMs = std::max(0.0, source.snapshot.stageTimeMs) + std::max(0.0, target.snapshot.stageTimeMs);
            double threshold = std::max(cfg_.gainAbsMs, cfg_.gainRelativePair * pairMs);
            if (cfg_.predictionNoiseMs >= 0.0) threshold = std::max(threshold, cfg_.predictionNoiseMs);
            for (uint32_t k = 1u; k <= kmax; ++k) {
                const double movedCost = movedSourceCost(source, k, takeSuffix);
                if (!(movedCost >= 0.0)) continue;
                const double gain = movedCost - targetEst.perLayerMs * (double)k;
                Candidate cand;
                cand.kind = CandidateKind::PP_MOVE;
                cand.fromStageIndex = source.snapshot.stageIndex;
                cand.toStageIndex = target.snapshot.stageIndex;
                cand.fromNodeIndex = source.snapshot.rootNodeIndex;
                cand.toNodeIndex = target.snapshot.rootNodeIndex;
                cand.layerCount = k;
                cand.layerIndex = takeSuffix ? source.snapshot.endLayer - k : source.snapshot.startLayer;
                cand.gainMs = gain;
                cand.thresholdMs = threshold;
                cand.valid = gain > threshold;
                const std::string used = sourceEst.method;
                cand.reason = cand.valid ? used : "gain_below_threshold";
                if (sawKeptMove_ && reverseLeft_ > 0 && cand.fromStageIndex == keptTo_ && cand.toStageIndex == keptFrom_) {
                    cand.valid = false;
                    cand.reason = "reverse_suppressed";
                }
                if (cand.valid) {
                    const uint32_t margin = ready - k;
                    if (betterMove(cand, best, margin, bestMargin)) {
                        best = cand;
                        bestMargin = margin;
                        if (method != nullptr) *method = used;
                        if (sourceCost != nullptr) *sourceCost = movedCost;
                        if (targetCost != nullptr) *targetCost = targetEst.perLayerMs * (double)k;
                        if (confidence != nullptr) {
                            *confidence = sourceEst.haveLayers ? "high" : "medium";
                        }
                    }
                } else {
                    anyReject = true;
                    rejectReason = cand.reason;
                }
            }
        }
    }
    if (!best.valid && reject != nullptr) *reject = anyReject ? rejectReason : best.reason;
    if (!best.valid) best.reason = anyReject ? rejectReason : best.reason;
    return best;
}

Algo2Decision Algo2Session::pollIdle(double nowMs) {
    if (!haveTokenTime_) return make("idle");
    const double limit = std::max((double)cfg_.watchdogMs, cfg_.watchdogFactor * std::max(0.0, referenceMs_));
    if (nowMs - lastTokenMs_ > limit) {
        health_ = Algo2Health::NO_PROGRESS;
        return make("no_progress");
    }
    return make("idle");
}

Algo2Decision Algo2Session::observe(const Algo2TokenObs &token, double nowMs) {
    if (token.prefill) return make("prefill_excluded");
    const bool useE2e = token.e2eKnown && token.e2eMs >= 0.0;
    const bool useSum = token.stageSumKnown && token.stageComputeSumMs >= 0.0;
    if (!useE2e) return make(useSum ? "e2e_missing" : "metric_missing");

    Sample sample;
    sample.tokenId = token.tokenId;
    sample.contextLength = token.contextLength;
    sample.metric = token.e2eMs;
    sample.e2e = true;
    sample.epoch = partitionEpoch_;
    sample.compute = token.stageComputeMs;
    sample.comm = token.stageCommMs;

    if (haveTokenTime_) {
        const double limit = std::max((double)cfg_.watchdogMs, cfg_.watchdogFactor * std::max(0.0, referenceMs_));
        if (nowMs - lastTokenMs_ > limit) health_ = Algo2Health::NO_PROGRESS;
    }
    lastTokenMs_ = nowMs;
    haveTokenTime_ = true;

    if (phase_ == Algo2Phase::COMMIT) return make("commit_wait");
    if (phase_ == Algo2Phase::RECONCILE) return make("reconcile_wait");
    if (phase_ == Algo2Phase::ROLLBACK) return make(rollbackBlocked_ ? "rollback_failed" : "rollback_wait");
    if (phase_ == Algo2Phase::SETTLE) {
        if (settleLeft_ > 0) settleLeft_ -= 1;
        if (settleLeft_ <= 0) {
            phase_ = Algo2Phase::VERIFY;
            verify_.clear();
            verifySamplesSeen_ = 0;
            tailStrikes_ = 0;
            return make("settle_done");
        }
        return make("settle");
    }
    if (phase_ == Algo2Phase::VERIFY) {
        verify_.push_back(sample.metric);
        verifySamplesSeen_ += 1;
        if ((int)verify_.size() < cfg_.verifyTokens && verifySamplesSeen_ < cfg_.verifyMaxTokens)
            return make("verify_wait");
        const double med = medianOf(verify_);
        const double p95 = p95Of(verify_);
        const double mean = meanOf(verify_);
        double tol50 = std::max(0.0, cfg_.verifyRatio * beforeMedian_);
        double tol95 = std::max(0.0, cfg_.verifyRatio * beforeP95_);
        if (cfg_.predictionNoiseMs >= 0.0) {
            tol50 = std::max(tol50, cfg_.predictionNoiseMs);
            tol95 = std::max(tol95, cfg_.predictionNoiseMs);
        }
        const bool medianBetter = med < beforeMedian_ - tol50;
        const bool medianWorse = med > beforeMedian_ + tol50;
        const bool tailWorse = beforeP95_ >= 0.0 && p95 > beforeP95_ + tol95;
        Algo2Decision d = make("verify");
        d.windowMedianMs = med;
        d.windowMeanMs = mean;
        d.windowP95Ms = p95;
        d.preActionMedianMs = beforeMedian_;
        d.preActionP95Ms = beforeP95_;
        auto finishKeep = [&](const char *note) -> Algo2Decision {
            phase_ = Algo2Phase::COOLDOWN;
            cooldownLeft_ = cfg_.cooldownTokens;
            cooldownUntilMs_ = nowMs + (double)cfg_.cooldownMinMs;
            if (havePending_) {
                sawKeptMove_ = true;
                keptFrom_ = pending_.fromStageIndex;
                keptTo_ = pending_.toStageIndex;
                reverseLeft_ = cfg_.reverseSuppressDecisions;
            }
            havePending_ = false;
            pendingIsRollback_ = false;
            recovered_ = false;
            tailStrikes_ = 0;
            d.phase = phase_;
            d.note = note;
            d.recovered = false;
            d.healthReferenceValid = referenceValid_;
            d.healthReferenceMs = referenceValid_ ? referenceMs_ : -1.0;
            return d;
        };
        auto finishRollback = [&](const char *note) -> Algo2Decision {
            uint32_t gap = 0u;
            std::string block;
            if (!rollbackCatchup(token.tokenId, &gap, &block)) {
                phase_ = Algo2Phase::ROLLBACK;
                rollbackBlocked_ = true;
                havePending_ = false;
                pendingIsRollback_ = false;
                d.phase = phase_;
                d.rollback = false;
                d.note = block;
                return d;
            }
            phase_ = Algo2Phase::ROLLBACK;
            pendingIsRollback_ = true;
            d.phase = phase_;
            d.rollback = true;
            d.note = note;
            d.candidate = pending_;
            d.catchupTokens = gap;
            d.catchupMode = gap == 0u ? "reservation" : "retransmit";
            return d;
        };
        if (medianWorse) return finishRollback("clear_regression");
        if (tailWorse) {
            tailStrikes_ += 1;
            verify_.clear();
            d.tailStrikes = tailStrikes_;
            if (tailStrikes_ >= cfg_.tailConfirmations) return finishRollback("tail_confirmed_rollback");
            if (verifySamplesSeen_ >= cfg_.verifyMaxTokens) {
                if (medianBetter) return finishKeep("clear_improvement");
                return finishRollback("inconclusive_rollback");
            }
            d.note = "tail_unconfirmed";
            return d;
        }
        if (medianBetter) return finishKeep(tailStrikes_ > 0 ? "spike_not_sustained" : "clear_improvement");
        if (verifySamplesSeen_ < cfg_.verifyMaxTokens) {
            d.note = "inconclusive";
            return d;
        }
        return finishRollback("inconclusive_rollback");
    }
    if (phase_ == Algo2Phase::COOLDOWN) {
        if (cooldownLeft_ > 0) cooldownLeft_ -= 1;
        if (cooldownLeft_ <= 0 && nowMs >= cooldownUntilMs_) {
            phase_ = Algo2Phase::OBSERVE;
            return make("cooldown_done");
        }
        return make("cooldown");
    }

    samples_.push_back(sample);
    if (samples_.size() > 512u) samples_.erase(samples_.begin(), samples_.begin() + (samples_.size() - 512u));
    observeTokens_ += 1;
    if (mode_ == Algo2Mode::LEGACY) return make("legacy_controller");
    if (observeTokens_ - lastDecisionTokens_ < cfg_.decisionStrideTokens) return make("stride");
    if (lastDecisionMs_ > -1.0e90 && nowMs - lastDecisionMs_ < (double)cfg_.minDecisionIntervalMs) return make("min_interval");
    if ((int)samples_.size() < cfg_.decisionWindowTokens) {
        health_ = Algo2Health::INSUFFICIENT;
        return make("insufficient_data");
    }

    std::vector<Sample> epochSamples;
    for (size_t i = 0u; i < samples_.size(); ++i) {
        if (samples_[i].epoch == partitionEpoch_) epochSamples.push_back(samples_[i]);
    }
    if ((int)epochSamples.size() < cfg_.decisionWindowTokens) {
        lastDecisionTokens_ = observeTokens_;
        lastDecisionMs_ = nowMs;
        return make("partition_window_incomplete");
    }

    lastDecisionTokens_ = observeTokens_;
    lastDecisionMs_ = nowMs;
    const size_t epochBegin = epochSamples.size() - (size_t)cfg_.decisionWindowTokens;
    std::vector<Sample> epochWindow(epochSamples.begin() + (std::ptrdiff_t)epochBegin, epochSamples.end());
    std::vector<double> window;
    for (size_t i = 0u; i < epochWindow.size(); ++i) window.push_back(epochWindow[i].metric);
    const double med = medianOf(window);
    const double p95 = p95Of(window);
    const double mean = meanOf(window);
    if (!partitionReferenceValid_) capturePartitionReference(epochWindow);
    if (health_ != Algo2Health::NO_PROGRESS) updateHealth(window, med, p95);

    Algo2Decision d = make("observe");
    d.windowMedianMs = med;
    d.windowMeanMs = mean;
    d.windowP95Ms = p95;
    if (!serial_) {
        d.note = "overlapping_requests_unsupported";
        return d;
    }
    if (mode_ == Algo2Mode::STATIC) {
        d.note = "static";
        return d;
    }
    const bool degraded = health_ == Algo2Health::DEGRADED || health_ == Algo2Health::STABLE_DEGRADED;
    if (!degraded && !cfg_.proactiveRebalance) {
        d.note = algo2HealthName(health_);
        return d;
    }
    if (health_ == Algo2Health::NO_PROGRESS) {
        d.note = "no_progress";
        return d;
    }
    phase_ = Algo2Phase::PLAN;
    std::string method;
    std::string reject;
    double sourceCost = -1.0;
    double targetCost = -1.0;
    std::string confidence;
    Candidate chosen = search(token.tokenId, &method, &reject, &sourceCost, &targetCost, &confidence);
    d.sourceCostMs = sourceCost;
    d.targetCostMs = targetCost;
    d.costConfidence = confidence;
    if (reverseLeft_ > 0) reverseLeft_ -= 1;
    if (!chosen.valid) {
        phase_ = Algo2Phase::OBSERVE;
        d.phase = phase_;
        const bool nothingReady = reject.empty() || reject == "no eligible pp candidate";
        d.note = (degraded && nothingReady) ? "degraded_no_improving_candidate" : (reject.empty() ? "no_eligible_candidate" : reject);
        d.candidate = chosen;
        d.estimateMethod = method;
        return d;
    }
    phase_ = Algo2Phase::COMMIT;
    pending_ = chosen;
    havePending_ = true;
    beforeMedian_ = med;
    beforeP95_ = p95;
    d.phase = phase_;
    d.issue = true;
    d.candidate = chosen;
    d.estimateMethod = method;
    d.note = "migration_issued";
    return d;
}

void Algo2Session::noteCommitInDoubt(const std::string &reason) {
    if (phase_ != Algo2Phase::COMMIT && phase_ != Algo2Phase::ROLLBACK && phase_ != Algo2Phase::RECONCILE) return;
    phase_ = Algo2Phase::RECONCILE;
    commitInDoubt_ = true;
    lastNote_ = reason.empty() ? "commit_ack_timeout" : reason;
}

void Algo2Session::noteCommitResult(bool applied, const std::string &reason) {
    if (phase_ != Algo2Phase::COMMIT && phase_ != Algo2Phase::ROLLBACK && phase_ != Algo2Phase::RECONCILE) return;
    if (!applied) {
        if (phase_ == Algo2Phase::ROLLBACK) {
            rollbackBlocked_ = true;
            havePending_ = false;
            pendingIsRollback_ = false;
            lastNote_ = reason.empty() ? "rollback_failed" : reason;
            return;
        }
        if (phase_ == Algo2Phase::RECONCILE && reason != "commit_not_applied") {
            commitInDoubt_ = true;
            lastNote_ = reason.empty() ? "commit_confirm_timeout" : reason;
            return;
        }
        phase_ = Algo2Phase::OBSERVE;
        havePending_ = false;
        pendingIsRollback_ = false;
        commitInDoubt_ = false;
        lastNote_ = reason.empty() ? "commit_rejected" : reason;
        settleLeft_ = 0;
        return;
    }
    partitionEpoch_ += 1u;
    partitionReferenceValid_ = false;
    partitionReferenceMs_ = -1.0;
    partitionCompute_.clear();
    partitionComm_.clear();
    commitInDoubt_ = false;
    recovered_ = false;
    if (phase_ == Algo2Phase::ROLLBACK || pendingIsRollback_) {
        phase_ = Algo2Phase::COOLDOWN;
        cooldownLeft_ = cfg_.cooldownTokens;
        cooldownUntilMs_ = lastTokenMs_ + (double)cfg_.cooldownMinMs;
        havePending_ = false;
        pendingIsRollback_ = false;
        lastNote_ = "rollback_applied";
        return;
    }
    phase_ = Algo2Phase::SETTLE;
    settleLeft_ = cfg_.settleTokens;
    pendingIsRollback_ = false;
    lastNote_ = "commit_applied";
}

OfflineSearchResult compareLocalSearchToExhaustive(
    const std::vector<std::vector<double>> &costByLayerDevice,
    const std::vector<uint32_t> &initialCounts,
    uint32_t minLayers) {
    OfflineSearchResult out;
    if (costByLayerDevice.empty() || initialCounts.empty() || minLayers == 0u) {
        out.note = "empty";
        return out;
    }
    const uint32_t devices = (uint32_t)initialCounts.size();
    uint32_t layers = 0u;
    for (size_t i = 0u; i < initialCounts.size(); ++i) layers += initialCounts[i];
    if (layers != (uint32_t)costByLayerDevice.size() || devices > 3u || layers > 8u) {
        out.note = "exhaustive_skipped";
        out.localMs = assignmentCost(costByLayerDevice, initialCounts);
        return out;
    }
    for (size_t layer = 0u; layer < costByLayerDevice.size(); ++layer) {
        if (costByLayerDevice[layer].size() != (size_t)devices) {
            out.note = "ragged_cost";
            return out;
        }
    }
    double optimal = 1.0e300;
    std::vector<uint32_t> prefix;
    std::vector<uint32_t> bestCounts;
    enumerateCounts(costByLayerDevice, minLayers, layers, devices, &prefix, &optimal, &bestCounts);
    out.exhaustive = true;
    out.optimalMs = optimal;

    std::vector<uint32_t> counts = initialCounts;
    for (int guard = 0; guard < 32; ++guard) {
        const double current = assignmentCost(costByLayerDevice, counts);
        double bestGain = 0.0;
        std::vector<uint32_t> next = counts;
        for (uint32_t b = 0u; b + 1u < devices; ++b) {
            for (int dir = 0; dir < 2; ++dir) {
                const uint32_t src = dir == 0 ? b : b + 1u;
                const uint32_t dst = dir == 0 ? b + 1u : b;
                if (counts[src] <= minLayers) continue;
                const uint32_t kmax = counts[src] - minLayers;
                for (uint32_t k = 1u; k <= kmax; ++k) {
                    std::vector<uint32_t> trial = counts;
                    trial[src] -= k;
                    trial[dst] += k;
                    const double gain = current - assignmentCost(costByLayerDevice, trial);
                    if (gain > bestGain) {
                        bestGain = gain;
                        next = trial;
                    }
                }
            }
        }
        if (!(bestGain > 0.0)) break;
        counts.swap(next);
    }
    out.localMs = assignmentCost(costByLayerDevice, counts);
    out.localIsOptimal = std::fabs(out.localMs - out.optimalMs) <= 1e-6;
    out.note = out.localIsOptimal ? "local_matches_exhaustive" : "local_short_of_exhaustive";
    return out;
}

} // namespace dynamic_tpot
} // namespace dllama
