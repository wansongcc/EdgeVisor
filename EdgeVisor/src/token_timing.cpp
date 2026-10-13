#include "token_timing.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <atomic>
#include <exception>
#include <thread>

bool dllamaTokenTimingPrintEnabled() {
    const char *v = std::getenv("DLLAMA_TOKEN_TIMING_PRINT");
    if (v == nullptr || v[0] == '\0') return false;
    if (std::strcmp(v, "0") == 0 ||
        std::strcmp(v, "false") == 0 ||
        std::strcmp(v, "False") == 0 ||
        std::strcmp(v, "off") == 0 ||
        std::strcmp(v, "OFF") == 0) {
        return false;
    }
    return true;
}

std::string formatTokenE2eTimingLine(unsigned int pos, int token, double wallMs) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "[token-e2e] pos=" << pos
        << " token=" << token
        << " wall=" << wallMs << "ms";
    return oss.str();
}

std::string formatTokenNodeTimingLine(unsigned int pos, const DllamaTokenNodeTiming &timing) {
    const double execMs = (double)timing.execUs / 1000.0;
    const double syncMs = (double)timing.syncUs / 1000.0;
    const double bubbleMs = (double)timing.bubbleUs / 1000.0;
    const double totalMs = execMs + syncMs + bubbleMs;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "[token-prof] pos=" << pos
        << " node=" << timing.nodeIndex;
    if (timing.hasStage) oss << " stage=" << timing.stageIndex;
    else oss << " stage=unknown";
    oss << " total=" << totalMs << "ms"
        << " exec=" << execMs << "ms"
        << " sync=" << syncMs << "ms"
        << " bubble=" << bubbleMs << "ms";
    return oss.str();
}

namespace {
uint64_t traceNowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
thread_local std::vector<uint64_t> traceParents;
std::atomic<bool> traceHealthy{true};
std::atomic<uint64_t> traceContext{0u};
std::string traceActor() {
    std::ostringstream name;
    name << std::this_thread::get_id();
    return name.str();
}
struct RootExperimentTrace {
    FILE *file = nullptr;
    std::mutex mutex;
    uint64_t sequence = 0u;
    uint64_t nextSpan = 1u;
    std::string runId;
    std::string clockId;
    RootExperimentTrace() {
        const char *path = std::getenv("EDGEVISOR_TRACE_PATH");
        const char *run = std::getenv("EDGEVISOR_RUN_ID");
        if (run == nullptr || run[0] == '\0') throw std::runtime_error("EDGEVISOR_RUN_ID is required for experiment traces");
        runId = run;
        clockId = runId + ":root:" + std::to_string(traceNowNs());
        // Refuse to truncate or append to another run's evidence.
        file = std::fopen(path, "wx");
        if (file == nullptr) throw std::runtime_error("Cannot exclusively create EDGEVISOR_TRACE_PATH");
    }
    ~RootExperimentTrace() { if (file != nullptr) std::fclose(file); }
    void write(const char *type, const char *phase, uint64_t id, uint64_t parent,
               uint64_t timestamp, const nlohmann::json &attributes) {
        std::lock_guard<std::mutex> lock(mutex);
        nlohmann::json event = {{"schema_version", 1}, {"run_id", runId},
            {"clock_id", clockId}, {"clock", "steady"}, {"role", "root"},
            {"runtime_context_id", traceContext.load()}, {"actor_id", traceActor()},
            {"seq", sequence++}, {"ts_ns", timestamp}, {"type", type},
            {"phase", phase}, {"span_id", id}, {"parent_id", parent},
            {"attributes", attributes}};
        const std::string line = event.dump() + "\n";
        if (std::fwrite(line.data(), 1u, line.size(), file) != line.size() || std::fflush(file) != 0) {
            traceHealthy.store(false);
            throw std::runtime_error("Experiment trace write failed");
        }
    }
};
RootExperimentTrace &rootTrace() { static RootExperimentTrace trace; return trace; }
}

bool experimentTraceEnabled() {
    const char *path = std::getenv("EDGEVISOR_TRACE_PATH");
    return path != nullptr && path[0] != '\0';
}
bool experimentTraceHealthy() { return traceHealthy.load(); }
void experimentTraceNewContext() {
    if (!experimentTraceEnabled()) return;
    ++traceContext;
    experimentTracePoint("runtime_context_start");
}
void requireExperimentTraceHealthy() {
    if (!experimentTraceHealthy()) throw std::runtime_error("Experiment evidence is incomplete");
}
void experimentTracePoint(const char *phase, const nlohmann::json &attributes) {
    if (!experimentTraceEnabled()) return;
    rootTrace().write("point", phase, 0u, traceParents.empty() ? 0u : traceParents.back(), traceNowNs(), attributes);
}
ExperimentTraceSpan::ExperimentTraceSpan(const char *phase, const nlohmann::json &attributes) {
    if (!experimentTraceEnabled()) return;
    RootExperimentTrace &trace = rootTrace();
    {
        std::lock_guard<std::mutex> lock(trace.mutex);
        id_ = trace.nextSpan++;
    }
    parent_ = traceParents.empty() ? 0u : traceParents.back();
    phase_ = phase;
    attributes_ = attributes;
    start_ = traceNowNs();
    trace.write("begin", phase_.c_str(), id_, parent_, start_, attributes_);
    traceParents.push_back(id_);
}
void ExperimentTraceSpan::end() {
    if (id_ == 0u) return;
    const uint64_t timestamp = traceNowNs();
    attributes_["duration_ns"] = timestamp - start_;
    attributes_["status"] = std::uncaught_exception() ? "error" : "ok";
    const uint64_t id = id_;
    id_ = 0u;
    if (traceParents.empty() || traceParents.back() != id) {
        traceHealthy.store(false);
        throw std::runtime_error("Experiment trace spans closed out of order");
    }
    traceParents.pop_back();
    rootTrace().write("end", phase_.c_str(), id, parent_, timestamp, attributes_);
}
ExperimentTraceSpan::~ExperimentTraceSpan() {
    try { end(); }
    catch (const std::exception &e) {
        traceHealthy.store(false);
        std::fprintf(stderr, "[experiment-trace-error] %s\n", e.what());
    }
}
