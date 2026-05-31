from pathlib import Path

root = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor")

def replace(path, old, new):
    p = root / path
    s = p.read_text()
    if new in s:
        return
    if old not in s:
        raise RuntimeError(f"pattern not found in {path}: {old[:100]!r}")
    p.write_text(s.replace(old, new, 1))

# Headers: own the worker thread and make stop flags atomic.
replace(
    "src/dynamic/kv_collector.hpp",
    """#include <memory>

class RootLlmInference;
""",
    """#include <atomic>
#include <memory>
#include <thread>

class RootLlmInference;
""",
)
replace(
    "src/dynamic/kv_collector.hpp",
    """    RootLlmInference *inference_ = nullptr;
    bool stop_ = false;
};
""",
    """    RootLlmInference *inference_ = nullptr;
    std::atomic<bool> stop_{false};
    std::thread worker_;
};
""",
)

replace(
    "src/plan-controller.hpp",
    """#include <memory>
#include <string>
""",
    """#include <atomic>
#include <memory>
#include <string>
#include <thread>
""",
)
replace(
    "src/plan-controller.hpp",
    """    int serverFd_ = -1;
    bool stop_ = false;

    void closeServer();
};
""",
    """    int serverFd_ = -1;
    std::atomic<bool> stop_{false};
    std::thread worker_;

    void closeServer();
};
""",
)

replace(
    "src/dynamic/dynamic_layer.hpp",
    """#include <memory>
#include <string>
""",
    """#include <atomic>
#include <memory>
#include <string>
#include <thread>
""",
)
replace(
    "src/dynamic/dynamic_layer.hpp",
    """	bool stop_ = false;
};
""",
    """	std::atomic<bool> stop_{false};
	std::thread worker_;
};
""",
)

# Root KV collector: keep and join the thread.
replace(
    "src/dynamic/kv_collector.cpp",
    """    RootKvCollector *c = ctrl.get();
    std::thread t([c]() { c->run(); });
    t.detach();
""",
    """    RootKvCollector *c = ctrl.get();
    ctrl->worker_ = std::thread([c]() { c->run(); });
""",
)
replace(
    "src/dynamic/kv_collector.cpp",
    """RootKvCollector::~RootKvCollector() {
    stop_ = true;
}
""",
    """RootKvCollector::~RootKvCollector() {
    stop_.store(true);
    if (worker_.joinable()) {
        worker_.join();
    }
}
""",
)
replace(
    "src/dynamic/kv_collector.cpp",
    """    while (!stop_) {
""",
    """    while (!stop_.load()) {
""",
)

# Plan UDS controller: close fd to unblock accept and join.
replace(
    "src/plan-controller.cpp",
    """    PlanUdsController *c = ctrl.get();
    std::thread t([c]() { c->run(); });
    t.detach();
""",
    """    PlanUdsController *c = ctrl.get();
    ctrl->worker_ = std::thread([c]() { c->run(); });
""",
)
replace(
    "src/plan-controller.cpp",
    """PlanUdsController::~PlanUdsController() {
#ifndef _WIN32
    stop_ = true;
    closeServer();
#endif
}
""",
    """PlanUdsController::~PlanUdsController() {
#ifndef _WIN32
    stop_.store(true);
    closeServer();
    if (worker_.joinable()) {
        worker_.join();
    }
#endif
}
""",
)
replace(
    "src/plan-controller.cpp",
    """    while (!stop_) {
""",
    """    while (!stop_.load()) {
""",
)
replace(
    "src/plan-controller.cpp",
    """            if (stop_) break;
""",
    """            if (stop_.load()) break;
""",
)

# Dynamic layer controller: keep and join the thread.
replace(
    "src/dynamic/dynamic_layer.cpp",
    """	DynamicLayerController *c = ctrl.get();
	std::thread t([c]() { c->run(); });
	t.detach();
""",
    """	DynamicLayerController *c = ctrl.get();
	ctrl->worker_ = std::thread([c]() { c->run(); });
""",
)
replace(
    "src/dynamic/dynamic_layer.cpp",
    """DynamicLayerController::~DynamicLayerController() {
	stop_ = true;
}
""",
    """DynamicLayerController::~DynamicLayerController() {
	stop_.store(true);
	if (worker_.joinable()) {
		worker_.join();
	}
}
""",
)
replace(
    "src/dynamic/dynamic_layer.cpp",
    """	while (!stop_) {
""",
    """	while (!stop_.load()) {
""",
)

print("background controller threads now join on destruction")
