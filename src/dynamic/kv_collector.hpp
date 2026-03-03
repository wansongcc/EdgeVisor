#pragma once

#include <memory>

class RootLlmInference;

class RootKvCollector {
public:
    // Enabled by setting env: DLLAMA_ASYNC_KV_COLLECT_LAYER=<layer>
    static std::unique_ptr<RootKvCollector> start(RootLlmInference *inference);

    ~RootKvCollector();

    RootKvCollector(const RootKvCollector &) = delete;
    RootKvCollector &operator=(const RootKvCollector &) = delete;

private:
    explicit RootKvCollector(RootLlmInference *inference);

    void run();

    RootLlmInference *inference_ = nullptr;
    bool stop_ = false;
};
