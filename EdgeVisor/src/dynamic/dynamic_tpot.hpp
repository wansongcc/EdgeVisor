#pragma once

#include "dynamic/tpot_algorithm.hpp"
#include "json.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

class RootLlmInference;

namespace edgevisor {
namespace dynamic_tpot {

SchedulerConfig loadSchedulerConfigFromEnvironment();
nlohmann::json makePpCommandRequest(uint32_t seq, const Candidate &candidate);

} // namespace dynamic_tpot
} // namespace edgevisor

// Root-side single-request TPOT scheduler.
// Enabled with EDGEVISOR_DYNAMIC_TPOT_ENABLE=1 and reuses the existing Plan UDS
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
        const edgevisor::dynamic_tpot::SchedulerConfig &config);

    void run();

    std::string socketPath_;
    RootLlmInference *inference_ = nullptr;
    edgevisor::dynamic_tpot::SchedulerConfig config_;

    std::atomic<bool> stop_{false};
    std::thread worker_;
};
