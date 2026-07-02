#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

class RootLlmInference;

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
    DynamicTpotController(const std::string &socketPath, RootLlmInference *inference);

    void run();

    std::string socketPath_;
    RootLlmInference *inference_ = nullptr;

    std::atomic<bool> stop_{false};
    std::thread worker_;
};
