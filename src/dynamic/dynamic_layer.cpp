
#include "dynamic_layer.hpp"

#include "app.hpp"
#include "json.hpp"

#include "nn/nn-core.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

using json = nlohmann::json;

// =====================================================================================
// Embedded kv_reused-style optimizer (moves output)
// NOTE: This is intentionally local to this translation unit to avoid cross-file calls.
// Units used here: KV heads (consistent with PlanCommand comment for GQA lockstep).
// =====================================================================================

struct DynHeadRange {
	int start;
	int end;
	int count() const { return end - start + 1; }
};

struct DynDeviceStatus {
	int device_id;               // global nodeIndex
	double execution_time_ms;    // per-layer time (ms)
	DynHeadRange current_compute;    // kvHeadSplit range (inclusive)
	DynHeadRange kv_cache_holding;   // kvHeadComputeSplit range (inclusive)
};

struct DynRebalanceMove {
	int from_device_id;
	int to_device_id;
	int cmdKind; // 1=headSplit 2=ffnSplit 3=both
	int headMove;
	int ffnMove;
};

static int dynFindBottleneckDeviceIndex(const std::vector<DynDeviceStatus> &devices) {
	int src_idx = -1;
	double max_time = -1.0;
	for (size_t i = 0; i < devices.size(); ++i) {
		if (devices[i].execution_time_ms > max_time) {
			max_time = devices[i].execution_time_ms;
			src_idx = (int)i;
		}
	}
	return src_idx;
}

static std::vector<DynRebalanceMove> dynRebalanceHeadMoves(const std::vector<DynDeviceStatus> &devices) {
	std::vector<DynRebalanceMove> moves;
	if (devices.empty()) return moves;

	const int src_idx = dynFindBottleneckDeviceIndex(devices);
	if (src_idx < 0) return moves;

	const DynDeviceStatus &src_dev = devices[(size_t)src_idx];
	const int current_heads = src_dev.current_compute.count();
	if (current_heads <= 1) return moves; // keep at least 1 head

	const double t_src_unit = src_dev.execution_time_ms / (double)current_heads;
	const int max_movable = current_heads - 1;

	int move_left = 0;
	int move_right = 0;

	// --- Move to Left neighbor ---
	if (src_idx > 0) {
		const DynDeviceStatus &left_dev = devices[(size_t)(src_idx - 1)];
		const int l_heads = left_dev.current_compute.count();
		const double t_left_unit = (l_heads > 0) ? (left_dev.execution_time_ms / (double)l_heads) : t_src_unit;
		if (src_dev.execution_time_ms > left_dev.execution_time_ms) {
			// Solve for k in: (Tsrc - k*t_src_unit) <= (Tleft + k*t_left_unit)
			const double denom = (t_src_unit + t_left_unit);
			if (denom > 0.0) {
				const double k = (src_dev.execution_time_ms - left_dev.execution_time_ms) / denom;
				move_left = (int)std::floor(k);
			}
		}
	}

	// --- Move to Right neighbor ---
	if (src_idx < (int)devices.size() - 1) {
		const DynDeviceStatus &right_dev = devices[(size_t)(src_idx + 1)];
		const int r_heads = right_dev.current_compute.count();
		const double t_right_unit = (r_heads > 0) ? (right_dev.execution_time_ms / (double)r_heads) : t_src_unit;
		if (src_dev.execution_time_ms > right_dev.execution_time_ms) {
			const double denom = (t_src_unit + t_right_unit);
			if (denom > 0.0) {
				const double k = (src_dev.execution_time_ms - right_dev.execution_time_ms) / denom;
				move_right = (int)std::floor(k);
			}
		}
	}

	if (move_left < 0) move_left = 0;
	if (move_right < 0) move_right = 0;

	// Clamp total move by max_movable
	if (move_left + move_right > max_movable) {
		const float total_req = (float)(move_left + move_right);
		const float ratio = (total_req > 0.0f) ? ((float)max_movable / total_req) : 0.0f;
		int new_move_left = (int)std::floor(move_left * ratio);
		int new_move_right = (int)std::floor(move_right * ratio);
		while (new_move_left + new_move_right > max_movable) {
			if (new_move_left >= new_move_right && new_move_left > 0) new_move_left--;
			else if (new_move_right > 0) new_move_right--;
			else break;
		}
		move_left = new_move_left;
		move_right = new_move_right;
	}

	// KV holding constraints (neighbor must already hold the moved heads)
	if (move_left > 0 && src_idx > 0) {
		const DynDeviceStatus &left_dev = devices[(size_t)(src_idx - 1)];
		const int current_L_end = left_dev.current_compute.end;
		int max_possible = left_dev.kv_cache_holding.end - current_L_end;
		if (max_possible < 0) max_possible = 0;
		move_left = std::min(move_left, max_possible);
	}

	if (move_right > 0 && src_idx < (int)devices.size() - 1) {
		const DynDeviceStatus &right_dev = devices[(size_t)(src_idx + 1)];
		const int current_R_start = right_dev.current_compute.start;
		int max_possible = current_R_start - right_dev.kv_cache_holding.start;
		if (max_possible < 0) max_possible = 0;
		move_right = std::min(move_right, max_possible);
	}

	if (move_left > 0 && src_idx > 0) {
		moves.push_back(DynRebalanceMove{
			src_dev.device_id,
			devices[(size_t)(src_idx - 1)].device_id,
			1,
			move_left,
			0,
		});
	}
	if (move_right > 0 && src_idx < (int)devices.size() - 1) {
		moves.push_back(DynRebalanceMove{
			src_dev.device_id,
			devices[(size_t)(src_idx + 1)].device_id,
			1,
			move_right,
			0,
		});
	}
	return moves;
}

// =====================================================================================
// Utilities
// =====================================================================================

static bool dynParseEnvBool(const char *name, bool defaultValue) {
	const char *v = std::getenv(name);
	if (v == nullptr || v[0] == '\0') return defaultValue;
	if (std::strcmp(v, "1") == 0) return true;
	if (std::strcmp(v, "0") == 0) return false;
	if (std::strcmp(v, "true") == 0 || std::strcmp(v, "TRUE") == 0) return true;
	if (std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0) return false;
	return defaultValue;
}

static int dynParseEnvInt(const char *name, int defaultValue) {
	const char *v = std::getenv(name);
	if (v == nullptr || v[0] == '\0') return defaultValue;
	char *end = nullptr;
	long x = std::strtol(v, &end, 10);
	if (end == v) return defaultValue;
	return (int)x;
}

static double dynParseEnvDouble(const char *name, double defaultValue) {
	const char *v = std::getenv(name);
	if (v == nullptr || v[0] == '\0') return defaultValue;
	char *end = nullptr;
	double x = std::strtod(v, &end);
	if (end == v) return defaultValue;
	return x;
}

static double dynNormalizeDropRatio(double r) {
	// Accept 0.3 or 30 (percent)
	if (r > 1.0) r = r / 100.0;
	if (r < 0.0) r = 0.0;
	if (r > 0.99) r = 0.99;
	return r;
}

#ifndef _WIN32
static bool dynReadLineFd(int fd, std::string &out) {
	out.clear();
	char ch;
	while (true) {
		ssize_t n = ::read(fd, &ch, 1);
		if (n == 0) return !out.empty();
		if (n < 0) return false;
		if (ch == '\n') return true;
		out.push_back(ch);
		if (out.size() > 1024 * 256) return false;
	}
}

static bool dynWriteAllFd(int fd, const void *data, size_t len) {
	const char *p = (const char *)data;
	size_t left = len;
	while (left > 0) {
		ssize_t w = ::write(fd, p, left);
		if (w < 0) return false;
		if (w == 0) return false;
		p += (size_t)w;
		left -= (size_t)w;
	}
	return true;
}

static json dynUdsRequest(const std::string &socketPath, const json &req, int timeoutMs) {
	int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd < 0) throw std::runtime_error("socket(AF_UNIX) failed");

	// Best-effort timeouts
	struct timeval tv;
	tv.tv_sec = timeoutMs / 1000;
	tv.tv_usec = (timeoutMs % 1000) * 1000;
	::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
	::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

	sockaddr_un addr;
	std::memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	if (socketPath.size() >= sizeof(addr.sun_path)) {
		::close(fd);
		throw std::runtime_error("socket path too long");
	}
	std::strncpy(addr.sun_path, socketPath.c_str(), sizeof(addr.sun_path) - 1);

	if (::connect(fd, (sockaddr *)&addr, sizeof(addr)) != 0) {
		::close(fd);
		throw std::runtime_error("connect(AF_UNIX) failed");
	}

	const std::string line = req.dump() + "\n";
	if (!dynWriteAllFd(fd, line.data(), line.size())) {
		::close(fd);
		throw std::runtime_error("write request failed");
	}

	std::string respLine;
	if (!dynReadLineFd(fd, respLine)) {
		::close(fd);
		throw std::runtime_error("read response failed");
	}
	::close(fd);
	return json::parse(respLine);
}
#endif

// =====================================================================================
// DynamicLayerController
// =====================================================================================

std::unique_ptr<DynamicLayerController> DynamicLayerController::start(const std::string &socketPath, RootLlmInference *inference) {
#ifdef _WIN32
	(void)socketPath;
	(void)inference;
	return nullptr;
#else
	const bool enabled = dynParseEnvBool("DLLAMA_DYNAMIC_LAYER_ENABLE", false);
	if (!enabled) return nullptr;
	if (socketPath.empty()) return nullptr;

	std::unique_ptr<DynamicLayerController> ctrl(new DynamicLayerController(socketPath, inference));
	DynamicLayerController *c = ctrl.get();
	ctrl->worker_ = std::thread([c]() { c->run(); });
	std::fprintf(stderr, "[dyn-layer] enabled, scheduler thread started (socket=%s)\n", socketPath.c_str());
	return ctrl;
#endif
}

DynamicLayerController::DynamicLayerController(const std::string &socketPath, RootLlmInference *inference)
	: socketPath_(socketPath), inference_(inference) {}

DynamicLayerController::~DynamicLayerController() {
	stop_.store(true);
	if (worker_.joinable()) {
		worker_.join();
	}
}

static const NnStageConfig *dynFindStage(const NnUnevenPartitionPlan *plan, uint32_t stageIndex) {
	if (plan == nullptr || plan->stages == nullptr) return nullptr;
	for (NnUint s = 0; s < plan->nStages; ++s) {
		const NnStageConfig *st = &plan->stages[s];
		if (st->stageIndex == stageIndex) return st;
	}
	return nullptr;
}

void DynamicLayerController::run() {
#ifdef _WIN32
	return;
#else
	const int pollMs = std::max(10, dynParseEnvInt("DLLAMA_DYN_POLL_MS", 200));
	const int timeoutMs = std::max(100, dynParseEnvInt("DLLAMA_DYN_UDS_TIMEOUT_MS", 2000));
	const double dropRatio = dynNormalizeDropRatio(dynParseEnvDouble("DLLAMA_DYN_DROP_RATIO", 0.30));

	const bool alwaysOptimize = dynParseEnvBool("DLLAMA_DYN_ALWAYS_OPTIMIZE", false);
	const bool forceOnce = dynParseEnvBool("DLLAMA_DYN_FORCE_TRIGGER", false);
	const int forcedLayer = dynParseEnvInt("DLLAMA_DYN_OPT_LAYER", 0);

	int stageIndexEnv = dynParseEnvInt("DLLAMA_DYN_STAGE_INDEX", 0);

	unsigned long long lastEpoch = 0ull;
	unsigned long long lastIssuedEpoch = 0ull;
	uint32_t seq = (uint32_t)std::max(1, dynParseEnvInt("DLLAMA_DYN_SEQ_START", 1));
	bool forcedUsed = false;

	while (!stop_.load()) {
		try {
			const NnUnevenPartitionPlan *plan = (inference_ != nullptr) ? inference_->getPartitionPlan() : nullptr;
			if (plan == nullptr) {
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			const uint32_t stageIndex = (uint32_t)stageIndexEnv;
			const NnStageConfig *stage = dynFindStage(plan, stageIndex);
			if (stage == nullptr || stage->nNodes == 0u || stage->nodeIndices == nullptr) {
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			uint32_t rootNodeIndex = (uint32_t)stage->rootNodeIndex;
			const int rootNodeEnv = dynParseEnvInt("DLLAMA_DYN_ROOT_NODE_INDEX", -1);
			if (rootNodeEnv >= 0) rootNodeIndex = (uint32_t)rootNodeEnv;

			json req;
			req["op"] = "layer_prof";
			req["all"] = true;
			req["stageIndex"] = stageIndex;
			req["rootNodeIndex"] = rootNodeIndex;

			const char *pathEnv = std::getenv("DLLAMA_DYN_LAYER_PROF_PATH");
			if (pathEnv != nullptr && pathEnv[0] != '\0') {
				req["path"] = std::string(pathEnv);
			}

			const json resp = dynUdsRequest(socketPath_, req, timeoutMs);
			if (!resp.value("ok", false)) {
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			if (!resp.contains("layer_prof")) {
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			const json &lp = resp.at("layer_prof");
			const unsigned long long epoch = lp.value("epoch", 0ull);
			if (epoch == 0ull) {
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}
			if (epoch == lastEpoch) {
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}
			lastEpoch = epoch;

			const json &hdr = lp.at("header");
			const uint32_t nLayers = hdr.value("nLayers", 0u);
			const uint32_t nStageNodes = hdr.value("nStageNodes", 0u);

			if (!lp.contains("layers") || !lp.at("layers").is_array()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			const json &layers = lp.at("layers");
			if (nLayers == 0u || (uint32_t)layers.size() != nLayers) {
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			// Save per-layer/per-device time (us) and layer time as max(device time).
			std::vector<std::map<uint32_t, uint32_t>> devTimeByLayer;
			std::vector<uint32_t> layerTimeUs;
			devTimeByLayer.resize(nLayers);
			layerTimeUs.resize(nLayers, 0u);

			for (uint32_t li = 0u; li < nLayers; ++li) {
				const json &row = layers.at(li);
				if (!row.is_array()) continue;
				uint32_t maxUs = 0u;
				for (uint32_t i = 0u; i < (uint32_t)row.size() && i < nStageNodes; ++i) {
					const json &cell = row.at(i);
					if (!cell.is_object()) continue;
					if (!cell.value("ok", false)) continue;
					const uint32_t nodeIndex = cell.value("nodeIndex", 0u);
					const uint32_t attnUs = cell.value("attnUs", 0u);
					const uint32_t ffnUs = cell.value("ffnUs", 0u);
					const uint32_t totalUs = attnUs + ffnUs;
					devTimeByLayer[li][nodeIndex] = totalUs;
					if (totalUs > maxUs) maxUs = totalUs;
				}
				layerTimeUs[li] = maxUs;
			}

			int candidateLayer = -1;
			if (alwaysOptimize) {
				if (forcedLayer >= 0 && forcedLayer < (int)nLayers) candidateLayer = forcedLayer;
			} else {
				for (uint32_t li = 1u; li < nLayers; ++li) {
					const uint32_t prev = layerTimeUs[li - 1u];
					const uint32_t cur = layerTimeUs[li];
					if (prev == 0u || cur == 0u) continue;
					const double thresh = (double)prev * (1.0 - dropRatio);
					if ((double)cur <= thresh) {
						candidateLayer = (int)li;
						break;
					}
				}
			}

			if (candidateLayer < 0 && forceOnce && !forcedUsed) {
				candidateLayer = std::min(1, (int)nLayers - 1);
			}

			if (candidateLayer < 0) {
				std::fprintf(stderr, "[dyn-layer] epoch=%llu no-trigger (dropRatio=%.2f)\n", epoch, dropRatio);
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			if (epoch == lastIssuedEpoch) {
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			// Build optimizer inputs from current partition plan + layer_prof times.
			std::vector<DynDeviceStatus> devices;
			devices.reserve(stage->nNodes);
			const std::map<uint32_t, uint32_t> &times = devTimeByLayer[(size_t)candidateLayer];

			for (NnUint r = 0u; r < stage->nNodes; ++r) {
				const uint32_t nodeIndex = stage->nodeIndices[r];
				const NnUint start0 = plan->kvHeadSplit.starts[nodeIndex];
				const NnUint len0 = plan->kvHeadSplit.lengths[nodeIndex];
				const NnUint start1 = plan->kvHeadComputeSplit.starts[nodeIndex];
				const NnUint len1 = plan->kvHeadComputeSplit.lengths[nodeIndex];

				if (len0 == 0u || len1 == 0u) continue;

				uint32_t tUs = 0u;
				std::map<uint32_t, uint32_t>::const_iterator it = times.find(nodeIndex);
				if (it != times.end()) tUs = it->second;
				if (tUs == 0u) continue;

				DynDeviceStatus st;
				st.device_id = (int)nodeIndex;
				st.execution_time_ms = (double)tUs / 1000.0;
				st.current_compute.start = (int)start0;
				st.current_compute.end = (int)(start0 + len0 - 1u);
				st.kv_cache_holding.start = (int)start1;
				st.kv_cache_holding.end = (int)(start1 + len1 - 1u);
				devices.push_back(st);
			}

			if (devices.size() < 2) {
				std::fprintf(stderr, "[dyn-layer] epoch=%llu layer=%d insufficient devices for optimize\n", epoch, candidateLayer);
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			const std::vector<DynRebalanceMove> moves = dynRebalanceHeadMoves(devices);
			if (moves.empty()) {
				std::fprintf(stderr, "[dyn-layer] epoch=%llu layer=%d optimizer produced no moves\n", epoch, candidateLayer);
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			json cmd;
			cmd["seq"] = seq;
			cmd["mode"] = "next_barrier";
			cmd["stageIndex"] = stageIndex;

			json jmoves = json::array();
			for (size_t i = 0; i < moves.size(); ++i) {
				const DynRebalanceMove &m = moves[i];
				if (m.headMove <= 0 && m.ffnMove <= 0) continue;
				jmoves.push_back(json{
					{"fromNodeIndex", (uint32_t)m.from_device_id},
					{"toNodeIndex", (uint32_t)m.to_device_id},
					{"cmdKind", (uint32_t)m.cmdKind},
					{"headMove", (uint32_t)std::max(0, m.headMove)},
					{"ffnMove", (uint32_t)std::max(0, m.ffnMove)},
				});
			}
			cmd["moves"] = jmoves;

			json setReq;
			setReq["op"] = "set_plan";
			setReq["cmd"] = cmd;

			const json setResp = dynUdsRequest(socketPath_, setReq, timeoutMs);
			if (!setResp.value("ok", false)) {
				std::fprintf(stderr, "[dyn-layer] set_plan failed: %s\n", setResp.dump().c_str());
				std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
				continue;
			}

			const uint64_t cacheSeq = setResp.value("cacheSeq", 0ull);
			std::fprintf(stderr, "[dyn-layer] epoch=%llu layer=%d issued set_plan seq=%u moves=%zu cacheSeq=%llu\n",
						 epoch, candidateLayer, seq, moves.size(), (unsigned long long)cacheSeq);

			// Requirement 3: check downlink result (best-effort: status roundtrip)
			json stReq;
			stReq["op"] = "status";
			const json stResp = dynUdsRequest(socketPath_, stReq, timeoutMs);
			if (stResp.value("ok", false)) {
				const bool barrier = stResp.value("enablePlanBarrier", false);
				const uint64_t cacheSeq2 = stResp.value("cacheSeq", 0ull);
				std::fprintf(stderr, "[dyn-layer] status ok enablePlanBarrier=%d cacheSeq=%llu\n", barrier ? 1 : 0, (unsigned long long)cacheSeq2);
			}

			lastIssuedEpoch = epoch;
			if (forceOnce) forcedUsed = true;
			seq++;

		} catch (const std::exception &e) {
			std::fprintf(stderr, "[dyn-layer] exception: %s\n", e.what());
		} catch (...) {
			std::fprintf(stderr, "[dyn-layer] unknown exception\n");
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
	}
#endif
}

