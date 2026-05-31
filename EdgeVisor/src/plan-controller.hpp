#ifndef PLAN_CONTROLLER_HPP
#define PLAN_CONTROLLER_HPP

#include <atomic>
#include <memory>
#include <string>
#include <thread>

class RootLlmInference;

// Lightweight UDS controller for online migration PlanCommand.
// Protocol: one-line JSON request per connection (or per line), one-line JSON response.
class PlanUdsController {
public:
    static std::unique_ptr<PlanUdsController> start(const std::string &socketPath, RootLlmInference *inference);
    ~PlanUdsController();

    PlanUdsController(const PlanUdsController&) = delete;
    PlanUdsController& operator=(const PlanUdsController&) = delete;

private:
    PlanUdsController(const std::string &socketPath, RootLlmInference *inference);
    void run();

    std::string socketPath_;
    RootLlmInference *inference_;

    int serverFd_ = -1;
    std::atomic<bool> stop_{false};
    std::thread worker_;

    void closeServer();
};

#endif
