#pragma once

#include "dynamic/tpot_algorithm.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

class RootLlmInference;

namespace dllama {
namespace dynamic_tpot {

SchedulerConfig loadSchedulerConfigFromEnvironment();

} // namespace dynamic_tpot
} // namespace dllama

// Root-side single-request TPOT scheduler.
// Enabled with DLLAMA_DYNAMIC_TPOT_ENABLE=1 and reuses the existing Plan UDS
// set_plan / set_pp_migration command paths.
class DynamicTpotController {
public:
    static std::unique_ptr<DynamicTpotController> start(const std::string &socketPath, RootLlmInference *inference);

    ~DynamicTpotController();

    DynamicTpotController(const DynamicTpotController &) = delete;
    DynamicTpotController &operator=(const DynamicTpotController &) = delete;

private:
    DynamicTpotController(
        const std::string &socketPath,
        RootLlmInference *inference,
        const dllama::dynamic_tpot::SchedulerConfig &config);

    void run();

    std::string socketPath_;
    RootLlmInference *inference_ = nullptr;
    dllama::dynamic_tpot::SchedulerConfig config_;

    std::atomic<bool> stop_{false};
    std::thread worker_;
};
