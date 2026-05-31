#pragma once

#include <atomic>
#include <memory>
#include <thread>

class RootLlmInference;

class RootKvCollector {
public:
    // Collector thread starts with root inference and waits until migration is armed
    // by runtime/UDS command flow (or env fallback trigger position).
    static std::unique_ptr<RootKvCollector> start(RootLlmInference *inference);

    ~RootKvCollector();

    RootKvCollector(const RootKvCollector &) = delete;
    RootKvCollector &operator=(const RootKvCollector &) = delete;

private:
    explicit RootKvCollector(RootLlmInference *inference);

    void run();

    RootLlmInference *inference_ = nullptr;
    std::atomic<bool> stop_{false};
    std::thread worker_;
};
