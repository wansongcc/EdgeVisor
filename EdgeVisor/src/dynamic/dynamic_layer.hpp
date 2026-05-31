
#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

class RootLlmInference;

// Internal dynamic layer scheduler:
// - Runs a background thread inside the root process.
// - Periodically queries the Plan UDS controller for layer_prof.
// - Detects a time drop between adjacent layers.
// - Runs an embedded kv_reused-style optimizer and issues set_plan via UDS.
class DynamicLayerController {
public:
	// Enabled by setting env: DLLAMA_DYNAMIC_LAYER_ENABLE=1
	static std::unique_ptr<DynamicLayerController> start(const std::string &socketPath, RootLlmInference *inference);

	~DynamicLayerController();

	DynamicLayerController(const DynamicLayerController &) = delete;
	DynamicLayerController &operator=(const DynamicLayerController &) = delete;

private:
	DynamicLayerController(const std::string &socketPath, RootLlmInference *inference);

	void run();

	std::string socketPath_;
	RootLlmInference *inference_ = nullptr;

	std::atomic<bool> stop_{false};
	std::thread worker_;
};

