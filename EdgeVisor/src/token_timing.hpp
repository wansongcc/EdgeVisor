#ifndef TOKEN_TIMING_HPP
#define TOKEN_TIMING_HPP

#include <string>
#include <cstdint>
#include "json.hpp"

struct DllamaTokenNodeTiming {
    unsigned int nodeIndex = 0u;
    unsigned int stageIndex = 0u;
    bool hasStage = false;
    unsigned long long execUs = 0ull;
    unsigned long long syncUs = 0ull;
    unsigned long long bubbleUs = 0ull;
};

bool dllamaTokenTimingPrintEnabled();
std::string formatTokenE2eTimingLine(unsigned int pos, int token, double wallMs);
std::string formatTokenNodeTimingLine(unsigned int pos, const DllamaTokenNodeTiming &timing);

// Opt-in root-process events. Worker clocks must never be subtracted from this
// domain without an explicit measured mapping. Existing textual logs are legacy.
bool experimentTraceEnabled();
void experimentTracePoint(const char *phase, const nlohmann::json &attributes = nlohmann::json::object());
bool experimentTraceHealthy();
void experimentTraceNewContext();
void requireExperimentTraceHealthy();
class ExperimentTraceSpan {
public:
    explicit ExperimentTraceSpan(const char *phase,
        const nlohmann::json &attributes = nlohmann::json::object());
    ~ExperimentTraceSpan();
    void end();
    ExperimentTraceSpan(const ExperimentTraceSpan &) = delete;
    ExperimentTraceSpan &operator=(const ExperimentTraceSpan &) = delete;
private:
    uint64_t id_ = 0u;
    uint64_t parent_ = 0u;
    uint64_t start_ = 0u;
    std::string phase_;
    nlohmann::json attributes_;
};

#endif
