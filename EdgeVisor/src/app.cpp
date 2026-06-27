#include "app.hpp"
#include "plan-controller.hpp"
#include "dynamic/dynamic_layer.hpp"
#include "dynamic/kv_collector.hpp"
#include <cassert>
#include <cstring>
#include <sstream>
#include <numeric>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <utility>
#include <ctime>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <thread>
#if defined(DLLAMA_VULKAN)
#include <cstdlib>
    #include "nn/nn-vulkan.hpp"
#endif
#if defined(DLLAMA_CUDA)
    #include "nn/nn-cuda.hpp"
#endif

// 引入 LLM 头文件以获取 createPartitionPlan 等定义
#include "llm.hpp"

// ---------------------------------------------------------
// Root control-packet logging (default OFF)
// Enable with: -DDLLAMA_CONTROL_LOG=1
// ---------------------------------------------------------
#ifndef DLLAMA_CONTROL_LOG
#define DLLAMA_CONTROL_LOG 0
#endif

static inline void logRootControlSend(const LlmControlPacket& p) {
#if DLLAMA_CONTROL_LOG
    // Match worker-side format for easy diff/align.
    printf("📤 [Root] Send Control: Batch=%u, Pos=%u, Flags=0x%x\n", p.batchSize, p.position, p.flags);
#else
    (void)p;
#endif
}

static bool envFlagEnabledDefault(const char *name, bool fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "False") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
        return false;
    }
    return true;
}

static bool bubbleShadowKvEnabled() {
    return envFlagEnabledDefault("DLLAMA_BUBBLE_SHADOW_KV", false);
}

static bool bubbleShadowKvAsyncEnabled() {
    return bubbleShadowKvEnabled() && envFlagEnabledDefault("DLLAMA_BUBBLE_SHADOW_KV_ASYNC", true);
}

static bool bubbleShadowKvLogEnabled() {
    return envFlagEnabledDefault("DLLAMA_BUBBLE_SHADOW_KV_LOG", false);
}

static bool lastStageSamplingEnabled() {
    return envFlagEnabledDefault("DLLAMA_LAST_STAGE_SAMPLING", false);
}

static bool lastStageSamplingPlanSupported(const NnUnevenPartitionPlan *plan) {
    if (!lastStageSamplingEnabled()) return false;
    if (plan == nullptr || plan->stages == nullptr || plan->nStages < 2u) return false;
    const NnStageConfig &last = plan->stages[plan->nStages - 1u];
    return last.nNodes > 0u && last.rootNodeIndex < plan->nNodes;
}

static NnUint findPipeIndexByName(const NnNetConfig *netConfig, const char *name) {
    if (netConfig == nullptr || name == nullptr) return (NnUint)-1;
    for (NnUint i = 0u; i < netConfig->nPipes; ++i) {
        const char *pipeName = netConfig->pipes[i].name;
        if (pipeName != nullptr && std::strcmp(pipeName, name) == 0) return i;
    }
    return (NnUint)-1;
}

static NnBubbleShadowStats maybeRunBubbleShadowKv(NnExecutor *executor, const char *who, NnUint nodeIndex, NnUint position, NnUint batchSize) {
    NnBubbleShadowStats stats{};
    if (!bubbleShadowKvEnabled() || executor == nullptr) return stats;
    const bool asyncMode = executor->isBubbleShadowAsyncModeEnabled();
    stats = asyncMode ? executor->getLastBubbleShadowStats() : executor->runBubbleShadowRedundant(0u);
    if (bubbleShadowKvLogEnabled()) {
        std::printf(
            "🫧 [bubble-shadow-kv] who=%s node=%u pos=%u batch=%u mode=%s segments=%u attn=%u ffn=%u other=%u layers=%u ops=%u skipped_sync=%u budget_hit=%u completed=%u drain_us=%u elapsed_us=%llu\n",
            who == nullptr ? "unknown" : who,
            (unsigned)nodeIndex,
            (unsigned)position,
            (unsigned)batchSize,
            asyncMode ? "async" : "sync",
            (unsigned)stats.segmentsVisited,
            (unsigned)stats.attnSegments,
            (unsigned)stats.ffnSegments,
            (unsigned)stats.otherSegments,
            (unsigned)stats.uniqueLayers,
            (unsigned)stats.opStepsExecuted,
            (unsigned)stats.skippedSyncSteps,
            (unsigned)stats.budgetHit,
            (unsigned)stats.completed,
            (unsigned)stats.drainUs,
            (unsigned long long)stats.elapsedUs);
        std::fflush(stdout);
    }
    return stats;
}

static bool parseEnvInt(const char *name, int &out) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return false;
    char *end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return false;
    out = (int)x;
    return true;
}

static std::vector<NnUint> parseLayerListEnv(const char *name) {
    std::vector<NnUint> out;
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return out;
    std::stringstream ss{std::string(v)};
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        try {
            int x = std::stoi(token);
            if (x >= 0) out.push_back((NnUint)x);
        } catch (...) {
        }
    }
    return out;
}

static void appendUniqueLayer(std::vector<NnUint> &layers, NnUint layer) {
    if (std::find(layers.begin(), layers.end(), layer) == layers.end()) {
        layers.push_back(layer);
    }
}

static void setRuntimeRedundantEnv(NnUint boundaryLayers, bool activeSegEnabled, bool redundantSegEnabled, const char *primarySkipLayersStr) {
    std::string boundary = std::to_string((unsigned)boundaryLayers);
    std::string active = activeSegEnabled ? "1" : "0";
    std::string redundant = redundantSegEnabled ? "1" : "0";
    setenv("DLLAMA_RUNTIME_REDUNDANT_BOUNDARY_LAYERS", boundary.c_str(), 1);
    setenv("DLLAMA_RUNTIME_ACTIVE_SEG_ENABLED", active.c_str(), 1);
    setenv("DLLAMA_RUNTIME_REDUNDANT_SEG_ENABLED", redundant.c_str(), 1);
    if (primarySkipLayersStr != nullptr && primarySkipLayersStr[0] != '\0') {
        setenv("DLLAMA_RUNTIME_PRIMARY_SKIP_LAYERS", primarySkipLayersStr, 1);
    } else {
        unsetenv("DLLAMA_RUNTIME_PRIMARY_SKIP_LAYERS");
    }
}

static void maybeSeedPlanCommandFromLegacyEnv() {
    // Deprecated legacy hook: env hard-migrate → PlanCommand(EXACT).
    // External controller should be preferred.
    int pos = -1;
    int layer = -1;
    bool hasPos = parseEnvInt("DLLAMA_HARD_MIGRATE_POS", pos);
    bool hasLayer = parseEnvInt("DLLAMA_HARD_MIGRATE_LAYER", layer);
    if (!hasPos && !hasLayer) return;

    static bool warned = false;
    if (!warned) {
        std::fprintf(stderr,
            "[DEPRECATED] DLLAMA_HARD_MIGRATE_* env hooks are deprecated. "
            "Use PlanCommand via UDS controller instead. (env will be used as fallback)\n");
        warned = true;
    }

    if (!(hasPos && hasLayer) || pos < 0 || layer < 0) {
        std::fprintf(stderr,
            "[plan][env] ignored: DLLAMA_HARD_MIGRATE_POS and DLLAMA_HARD_MIGRATE_LAYER must both be set (>=0)\n");
        return;
    }

    int kind = 3;
    (void)parseEnvInt("DLLAMA_HARD_MIGRATE_KIND", kind);
    int headMove = 1;
    (void)parseEnvInt("DLLAMA_HARD_MIGRATE_HEAD_MOVE", headMove);
    int ffnMove = 256;
    (void)parseEnvInt("DLLAMA_HARD_MIGRATE_FFN_MOVE", ffnMove);

    PlanCommand cmd = makeEmptyPlanCommand();
    cmd.mode = PLAN_CMD_MODE_EXACT;
    cmd.stageIndex = 0u; // legacy behavior
    cmd.triggerPos = (uint32_t)pos;
    cmd.triggerLayer = (uint32_t)layer;
    cmd.fromNodeIndex = 0u;
    cmd.toNodeIndex = 1u;
    cmd.cmdKind = (kind < 1 ? PLAN_CMD_KIND_BOTH : (uint32_t)kind);
    cmd.nHeadsToMove = (headMove < 0 ? 0u : (uint32_t)headMove);
    cmd.nFfnToMove = (ffnMove < 0 ? 0u : (uint32_t)ffnMove);

    planCommandCache().store(cmd);
}

static void writeBootstrapPacket(NnNetwork *network, NnUint socketIndex, const AppCliArgs *args) {
    LlmBootstrapPacket p{};
    p.magic = LLM_BOOTSTRAP_MAGIC;
    p.version = LLM_BOOTSTRAP_VERSION;
    p.flags = 0u;
    p.benchmarkEnabled = args->benchmark ? 1u : 0u;
    p.enablePlanBarrier = args->enablePlanBarrier ? 1u : 0u;
    p.enableStageFullWeights = args->enableStageFullWeights ? 1u : 0u;
    p.enableKvRedundancyDuringMigration = args->enableKvRedundancyDuringMigration ? 1u : 0u;
    p.allowNoShadowHeadMigration = args->allowNoShadowHeadMigration ? 1u : 0u;
    p.enableKvAggregate = args->enableKvAggregate ? 1u : 0u;
    p.runtimeRedundantBoundaryLayers = args->runtimeRedundantBoundaryLayers;
    p.runtimeActiveSegEnabled = args->runtimeActiveSegEnabled ? 1u : 0u;
    p.runtimeRedundantSegEnabled = (args->runtimeRedundantSegEnabled && !bubbleShadowKvEnabled()) ? 1u : 0u;
    p.bubbleShadowKvEnabled = bubbleShadowKvEnabled() ? 1u : 0u;
    p.samplerTemperature = args->temperature;
    p.samplerTopP = args->topp;
    p.samplerSeed = args->seed;
    if (p.bubbleShadowKvEnabled != 0u) {
        p.flags |= LLM_BOOTSTRAP_ENABLE_BUBBLE_SHADOW_KV;
        if (!bubbleShadowKvAsyncEnabled()) {
            p.flags |= LLM_BOOTSTRAP_DISABLE_BUBBLE_SHADOW_KV_ASYNC;
        }
    }
    if (args->lastStageSampling) {
        p.flags |= LLM_BOOTSTRAP_LAST_STAGE_SAMPLING;
    }
    p.maxSeqLen = args->maxSeqLen;
    p.syncType = (NnUint)args->syncType;
    p.modelPathLen = 0u;
    p.ratiosLen = 0u;
    p.primarySkipLayersLen = 0u;
    p.kvRedundancyLen = 0u;

    if (args->modelPath != nullptr) {
        p.flags |= LLM_BOOTSTRAP_HAS_MODEL_PATH;
        p.modelPathLen = (NnUint)std::strlen(args->modelPath) + 1u;
    }
    if (args->ratiosStr != nullptr) {
        p.flags |= LLM_BOOTSTRAP_HAS_RATIOS;
        p.ratiosLen = (NnUint)std::strlen(args->ratiosStr) + 1u;
    }
    if (args->runtimePrimarySkipLayersStr != nullptr && args->runtimePrimarySkipLayersStr[0] != '\0') {
        p.flags |= LLM_BOOTSTRAP_HAS_PRIMARY_SKIP_LAYERS;
        p.primarySkipLayersLen = (NnUint)std::strlen(args->runtimePrimarySkipLayersStr) + 1u;
    }
    if (args->kvRedundancyStr != nullptr && args->kvRedundancyStr[0] != '\0') {
        p.flags |= LLM_BOOTSTRAP_HAS_KV_REDUNDANCY;
        p.kvRedundancyLen = (NnUint)std::strlen(args->kvRedundancyStr) + 1u;
    }
    if (args->enableKvAggregate) {
        p.flags |= LLM_BOOTSTRAP_ENABLE_KV_AGGREGATE;
    }

    network->write(socketIndex, &p, sizeof(p));
    if (p.modelPathLen > 0u) network->write(socketIndex, args->modelPath, p.modelPathLen);
    if (p.ratiosLen > 0u) network->write(socketIndex, args->ratiosStr, p.ratiosLen);
    if (p.primarySkipLayersLen > 0u) network->write(socketIndex, args->runtimePrimarySkipLayersStr, p.primarySkipLayersLen);
    if (p.kvRedundancyLen > 0u) network->write(socketIndex, args->kvRedundancyStr, p.kvRedundancyLen);
}

static LlmBootstrapPacket readBootstrapPacket(
    NnNetwork *network,
    std::string &modelPath,
    std::string &ratiosStr,
    std::string &primarySkipLayersStr,
    std::string &kvRedundancyStr) {
    LlmBootstrapPacket p;
    network->read(ROOT_SOCKET_INDEX, &p, sizeof(p));
    if (p.magic != LLM_BOOTSTRAP_MAGIC)
        throw std::runtime_error("Invalid bootstrap magic (root/worker binary mismatch)");
    if (p.version != LLM_BOOTSTRAP_VERSION)
        throw std::runtime_error("Unsupported bootstrap version (root/worker binary mismatch)");

    modelPath.clear();
    ratiosStr.clear();
    primarySkipLayersStr.clear();
    kvRedundancyStr.clear();
    if ((p.flags & LLM_BOOTSTRAP_HAS_MODEL_PATH) != 0u) {
        std::vector<char> buf(p.modelPathLen);
        network->read(ROOT_SOCKET_INDEX, buf.data(), p.modelPathLen);
        modelPath.assign(buf.data());
    }
    if ((p.flags & LLM_BOOTSTRAP_HAS_RATIOS) != 0u) {
        std::vector<char> buf(p.ratiosLen);
        network->read(ROOT_SOCKET_INDEX, buf.data(), p.ratiosLen);
        ratiosStr.assign(buf.data());
    }
    if ((p.flags & LLM_BOOTSTRAP_HAS_PRIMARY_SKIP_LAYERS) != 0u) {
        std::vector<char> buf(p.primarySkipLayersLen);
        network->read(ROOT_SOCKET_INDEX, buf.data(), p.primarySkipLayersLen);
        primarySkipLayersStr.assign(buf.data());
    }
    if ((p.flags & LLM_BOOTSTRAP_HAS_KV_REDUNDANCY) != 0u) {
        std::vector<char> buf(p.kvRedundancyLen);
        network->read(ROOT_SOCKET_INDEX, buf.data(), p.kvRedundancyLen);
        kvRedundancyStr.assign(buf.data());
    }
    return p;
}

static NnFloatType parseFloatType(char *val) {
    if (std::strcmp(val, "f32") == 0) return F_32;
    if (std::strcmp(val, "f16") == 0) return F_16;
    if (std::strcmp(val, "q40") == 0) return F_Q40;
    if (std::strcmp(val, "q80") == 0) return F_Q80;
    throw std::runtime_error("Invalid float type: " + std::string(val));
}

static ChatTemplateType parseChatTemplateType(char *val) {
    if (std::strcmp(val, "llama2") == 0) return TEMPLATE_LLAMA2;
    if (std::strcmp(val, "llama3") == 0) return TEMPLATE_LLAMA3;
    if (std::strcmp(val, "deepSeek3") == 0) return TEMPLATE_DEEP_SEEK3;
    throw std::runtime_error("Invalid chat template type: " + std::string(val));
}

static AppCliArgs::Backend parseBackendType(char *val) {
    if (std::strcmp(val, "cpu") == 0) return AppCliArgs::BACKEND_CPU;
    if (std::strcmp(val, "vulkan") == 0) return AppCliArgs::BACKEND_VULKAN;
    if (std::strcmp(val, "cuda") == 0) return AppCliArgs::BACKEND_CUDA;
    throw std::runtime_error("Invalid backend: " + std::string(val) + " (expected cpu, vulkan, or cuda)");
}

const char *AppCliArgs::backendToString(AppCliArgs::Backend backend) {
    switch (backend) {
        case BACKEND_AUTO: return "auto";
        case BACKEND_CPU: return "cpu";
        case BACKEND_VULKAN: return "vulkan";
        case BACKEND_CUDA: return "cuda";
        default: return "unknown";
    }
}

AppCliArgs AppCliArgs::parse(int argc, char* *argv, bool requireMode) {
    AppCliArgs args;
    args.info = true;
    args.help = false;
    args.backend = BACKEND_AUTO;
    args.backendStr = nullptr;
    args.mode = nullptr;
    args.nBatches = 32;
    args.nThreads = 1;
    args.modelPath = nullptr;
    args.tokenizerPath = nullptr;
    args.prompt = nullptr;
    args.interactive = false;
    args.syncType = F_32;
    args.nWorkers = 0;
    args.workerAllocCount = 0;
    args.workerHosts = nullptr;
    args.workerPorts = nullptr;
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 0;
    args.benchmark = false;
    args.lastStageSampling = lastStageSamplingEnabled();
    args.seed = (unsigned long long)time(nullptr);
    args.chatTemplateType = TEMPLATE_UNKNOWN;
    args.maxSeqLen = 0;
    args.netTurbo = true;
    args.gpuIndex = -1;
    args.gpuSegmentFrom = -1;
    args.gpuSegmentTo = -1;
    args.ratiosStr = nullptr;
    args.warmupEnabled = false;
    args.warmupSteps = 16u;
    args.warmupBudget = 8u;
    args.warmupCandidatesStr = nullptr;
    args.kvRedundancyStr = nullptr;
    args.enablePlanBarrier = false;
    args.enableStageFullWeights = false;
    args.enableKvRedundancyDuringMigration = true;
    args.allowNoShadowHeadMigration = false;
    args.enableKvAggregate = false;
    args.enablePpMigration = false;
    args.runtimeRedundantBoundaryLayers = 1u;
    args.runtimeActiveSegEnabled = true;
    args.runtimeRedundantSegEnabled = false;
    args.runtimePrimarySkipLayersStr = nullptr;
    args.edgevisorAblationConfigPath = nullptr;
    args.shadowKvModeStr = nullptr;
    args.pointerSwizzlingModeStr = nullptr;
    args.jitModeStr = nullptr;
    args.vgModeStr = nullptr;
    args.disableShardingControllerStr = nullptr;
    args.disablePipelineBalancerStr = nullptr;
    args.fallbackPolicyStr = nullptr;
    args.ablationLogPath = nullptr;
    args.experimentId = nullptr;

    {
        int envBoundaryLayers = -1;
        bool hasEnvBoundaryLayers = parseEnvInt("DLLAMA_RUNTIME_REDUNDANT_BOUNDARY_LAYERS", envBoundaryLayers);
        if (hasEnvBoundaryLayers) {
            if (envBoundaryLayers < 0) envBoundaryLayers = 0;
            args.runtimeRedundantBoundaryLayers = (NnUint)envBoundaryLayers;
        }
    }

    int i = 1;
    if (requireMode && argc > 1) {
        args.mode = argv[1];
        i++;
    }
    // First see if any of the args are asking for help/usage and fail fast
    for (int x = 0; x < argc; x++) {
        if ((std::strcmp(argv[x], "--usage") == 0) ||
            (std::strcmp(argv[x], "--help") == 0) ||
            (std::strcmp(argv[x], "-h") == 0)) {
            args.help = true;
            return args;
        }
    }
    // Parse arguments. Some options are flags (no value), others require a value.
    // NOTE: --workers consumes a variable number of args.
    for (; i < argc; ) {
        char *name = argv[i];

        // Flags (no value)
        if (std::strcmp(name, "--benchmark") == 0) {
            // Support both: "--benchmark" and "--benchmark 1|0".
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.benchmark = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.benchmark = true;
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--last-stage-sampling") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.lastStageSampling = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.lastStageSampling = true;
                i += 1;
            }
            setenv("DLLAMA_LAST_STAGE_SAMPLING", args.lastStageSampling ? "1" : "0", 1);
            continue;
        }

        if (std::strcmp(name, "--enable-plan-barrier") == 0) {
            // Support both: "--enable-plan-barrier" and "--enable-plan-barrier 1|0".
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.enablePlanBarrier = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.enablePlanBarrier = true;
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--enable-stage-full-weights") == 0) {
            // Support both: "--enable-stage-full-weights" and "--enable-stage-full-weights 1|0".
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.enableStageFullWeights = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.enableStageFullWeights = true;
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--enable-kv-redundancy-during-migration") == 0) {
            // Support both: "--enable-kv-redundancy-during-migration" and "--enable-kv-redundancy-during-migration 1|0".
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.enableKvRedundancyDuringMigration = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.enableKvRedundancyDuringMigration = true;
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--allow-no-shadow-head-migration") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.allowNoShadowHeadMigration = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.allowNoShadowHeadMigration = true;
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--enable-kv-aggregate") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.enableKvAggregate = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.enableKvAggregate = true;
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--enable-pp-migration") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.enablePpMigration = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.enablePpMigration = true;
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--warmup") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.warmupEnabled = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.warmupEnabled = true;
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--runtime-active-seg-enabled") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.runtimeActiveSegEnabled = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.runtimeActiveSegEnabled = true;
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--runtime-redundant-seg-enabled") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.runtimeRedundantSegEnabled = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.runtimeRedundantSegEnabled = true;
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--disable-sharding-controller") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.disableShardingControllerStr = argv[i + 1];
                i += 2;
            } else {
                args.disableShardingControllerStr = (char *)"1";
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--disable-pipeline-balancer") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.disablePipelineBalancerStr = argv[i + 1];
                i += 2;
            } else {
                args.disablePipelineBalancerStr = (char *)"1";
                i += 1;
            }
            continue;
        }

        if (std::strcmp(name, "--interactive") == 0) {
            // Support both: "--interactive" and "--interactive 1|0".
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.interactive = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.interactive = true;
                i += 1;
            }
            continue;
        }

        // Options with special arity
        if (std::strcmp(name, "--workers") == 0) {
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;
            if (count <= 0)
                throw std::runtime_error("--workers requires at least one worker in host:port format");

            args.nWorkers = count;
            args.workerAllocCount = count;
            args.workerHosts = new char*[count];
            args.workerPorts = new NnUint[count];

            for (int s = 0; s < count; s++) {
                char *v = argv[i + 1 + s];
                char *separator = std::strstr(v, ":");
                if (separator == NULL) {
                    throw std::runtime_error("Invalid worker address: " + std::string(v));
                }
                int hostLen = separator - v;
                args.workerHosts[s] = new char[hostLen + 1];
                std::memcpy(args.workerHosts[s], v, hostLen);
                args.workerHosts[s][hostLen] = '\0';
                args.workerPorts[s] = std::atoi(separator + 1);
            }

            i = j;
            continue;
        }

        // All remaining options require a value
        if (i + 1 >= argc)
            throw std::runtime_error(std::string("Missing value for option: ") + name);
        char *value = argv[i + 1];

        if (std::strcmp(name, "--model") == 0) {
            args.modelPath = value;
        } else if (std::strcmp(name, "--tokenizer") == 0) {
            args.tokenizerPath = value;
        } else if (std::strcmp(name, "--prompt") == 0) {
            args.prompt = value;
        } else if (std::strcmp(name, "--buffer-float-type") == 0) {
            args.syncType = parseFloatType(value);
        } else if (std::strcmp(name, "--backend") == 0) {
            args.backend = parseBackendType(value);
            args.backendStr = value;
        } else if (std::strcmp(name, "--ratios") == 0) {
            args.ratiosStr = value;
        } else if (std::strcmp(name, "--warmup-steps") == 0) {
            int x = std::atoi(value);
            if (x < 1) x = 1;
            args.warmupSteps = (NnUint)x;
        } else if (std::strcmp(name, "--warmup-budget") == 0) {
            int x = std::atoi(value);
            if (x < 1) x = 1;
            args.warmupBudget = (NnUint)x;
        } else if (std::strcmp(name, "--warmup-candidates") == 0) {
            args.warmupCandidatesStr = value;
        } else if (std::strcmp(name, "--kv-redundancy") == 0) {
            args.kvRedundancyStr = value;
        } else if (std::strcmp(name, "--port") == 0) {
            args.port = atoi(value);
        } else if (std::strcmp(name, "--nthreads") == 0) {
            args.nThreads = atoi(value);
        } else if (std::strcmp(name, "--steps") == 0) {
            args.steps = atoi(value);
        } else if (std::strcmp(name, "--temperature") == 0) {
            args.temperature = atof(value);
        } else if (std::strcmp(name, "--topp") == 0) {
            args.topp = atof(value);
        } else if (std::strcmp(name, "--seed") == 0) {
            args.seed = atoll(value);
        } else if (std::strcmp(name, "--chat-template") == 0) {
            args.chatTemplateType = parseChatTemplateType(value);
        } else if (std::strcmp(name, "--max-seq-len") == 0) {
            args.maxSeqLen = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--gpu-index") == 0) {
            args.gpuIndex = atoi(value);
        } else if (std::strcmp(name, "--gpu-segments") == 0) {
            char *separator = std::strstr(value, ":");
            if (separator == NULL)
                throw std::runtime_error("GPU segments expected in the format <from>:<to>");
            args.gpuSegmentFrom = atoi(value);
            args.gpuSegmentTo = atoi(separator + 1);
        } else if (std::strcmp(name, "--net-turbo") == 0) {
            args.netTurbo = atoi(value) == 1;
        } else if (std::strcmp(name, "--runtime-redundant-boundary-layers") == 0) {
            int x = std::atoi(value);
            if (x < 0) x = 0;
            args.runtimeRedundantBoundaryLayers = (NnUint)x;
        } else if (std::strcmp(name, "--runtime-primary-skip-layers") == 0) {
            args.runtimePrimarySkipLayersStr = value;
        } else if (std::strcmp(name, "--edgevisor-ablation-config") == 0) {
            args.edgevisorAblationConfigPath = value;
        } else if (std::strcmp(name, "--shadow-kv-mode") == 0) {
            args.shadowKvModeStr = value;
        } else if (std::strcmp(name, "--pointer-swizzling-mode") == 0) {
            args.pointerSwizzlingModeStr = value;
        } else if (std::strcmp(name, "--jit-mode") == 0) {
            args.jitModeStr = value;
        } else if (std::strcmp(name, "--vg-mode") == 0) {
            args.vgModeStr = value;
        } else if (std::strcmp(name, "--fallback-policy") == 0) {
            args.fallbackPolicyStr = value;
        } else if (std::strcmp(name, "--ablation-log-path") == 0) {
            args.ablationLogPath = value;
        } else if (std::strcmp(name, "--experiment-id") == 0) {
            args.experimentId = value;
        } else {
            throw std::runtime_error("Unknown option: " + std::string(name));
        }

        i += 2;
    }

    if (args.nThreads < 1)
        throw std::runtime_error("Number of threads must be at least 1");
    if (args.backend == BACKEND_CPU && args.gpuIndex >= 0) {
        throw std::runtime_error("--backend cpu conflicts with --gpu-index; remove --gpu-index or choose --backend vulkan/cuda");
    }
    if (args.backend == BACKEND_CPU && (args.gpuSegmentFrom >= 0 || args.gpuSegmentTo >= 0)) {
        throw std::runtime_error("--backend cpu conflicts with --gpu-segments; CPU mode cannot own GPU segments");
    }
    if (args.backend == BACKEND_AUTO) {
        args.backend = (args.gpuIndex >= 0) ? BACKEND_VULKAN : BACKEND_CPU;
    } else if ((args.backend == BACKEND_VULKAN || args.backend == BACKEND_CUDA) && args.gpuIndex < 0) {
        args.gpuIndex = 0;
    }
#if !defined(DLLAMA_VULKAN)
    if (args.backend == BACKEND_VULKAN) {
        throw std::runtime_error("--backend vulkan requested, but this build was not compiled with DLLAMA_VULKAN=1");
    }
#endif
#if !defined(DLLAMA_CUDA)
    if (args.backend == BACKEND_CUDA) {
        throw std::runtime_error("--backend cuda requested, but this build was not compiled with DLLAMA_CUDA=1");
    }
#endif
    if (args.enablePpMigration && !args.enableKvAggregate) {
        args.enableKvAggregate = true;
        std::printf("⚠️  [pp-migrate] --enable-pp-migration requires KV aggregate; auto enabling --enable-kv-aggregate\n");
        std::fflush(stdout);
    }
    if (args.enablePpMigration && !args.enableStageFullWeights) {
        args.enableStageFullWeights = true;
        std::printf("⚠️  [pp-migrate] --enable-pp-migration requires stage full weights; auto enabling --enable-stage-full-weights\n");
        std::fflush(stdout);
    }
    return args;
}

AppCliArgs::~AppCliArgs() {
    if (workerHosts != nullptr) {
        for (NnUint i = 0; i < workerAllocCount; i++)
            delete[] workerHosts[i];
        delete[] workerHosts;
    }
    if (workerPorts != nullptr)
        delete[] workerPorts;
}

static std::vector<float> parseRatios(const char *ratiosStr, NnUint nNodes) {
    if (ratiosStr == nullptr) {
        throw std::invalid_argument("Ratios 字符串不能为空");
    }
    std::vector<float> ratios;
    std::string s(ratiosStr);
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            ratios.push_back(std::stof(item));
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::string("无效的比例值: ") + item);
        }
    }
    if (ratios.size() != nNodes) {
        throw std::invalid_argument(
            std::string("Ratios 数量 (") + std::to_string(ratios.size()) + 
            std::string(") 必须等于节点总数 (nNodes = ") + std::to_string(nNodes) + ")"
        );
    }
    return ratios;
}

// 解析 KV 冗余范围字符串
// 格式：
//   - "2" - 所有节点都使用 2 个冗余 head
//   - "2,3,2,3" - 每个节点指定不同的冗余 head 数量
// 返回每个节点的冗余 head 数量
static std::vector<NnUint> parseKvRedundancy(const char *redundancyStr, NnUint nNodes) {
    if (redundancyStr == nullptr) {
        // 默认值为 2（与 NN_KV_REDUNDANCY_PAD_HEADS 保持一致）
        std::vector<NnUint> redundancies(nNodes, 2u);
        return redundancies;
    }

    std::vector<NnUint> redundancies;
    std::string s(redundancyStr);
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        try {
            NnUint val = (NnUint)std::stoul(item);
            redundancies.push_back(val);
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::string("无效的 KV 冗余值: ") + item);
        }
    }

    if (redundancies.size() == 1) {
        // 单个值，应用到所有节点
        std::vector<NnUint> result(nNodes, redundancies[0]);
        return result;
    }

    if (redundancies.size() != nNodes) {
        throw std::invalid_argument(
            std::string("KV 冗余值数量 (") + std::to_string(redundancies.size()) +
            std::string(") 必须等于节点总数 (nNodes = ") + std::to_string(nNodes) + ")"
        );
    }

    return redundancies;
}

static NnUint inferRuntimeRedundantBoundaryLayers(const char *redundancyStr, NnUint nNodes, NnUint fallback) {
    try {
        std::vector<NnUint> redundancies = parseKvRedundancy(redundancyStr, nNodes);
        NnUint maxRedundancy = fallback;
        for (NnUint value : redundancies) {
            maxRedundancy = std::max(maxRedundancy, value);
        }
        return maxRedundancy;
    } catch (...) {
        return fallback;
    }
}

// [修改] 解析多 Stage 格式: "1.0:10;0.5,0.5:14"
// 1. 分号 ';' 或 竖线 '|' 分隔不同的 Stage
// 2. 冒号 ':' 分隔 TP比例 和 层数 (可选，如果不填则自动分配)
// 3. 逗号 ',' 分隔同一 Stage 内的 TP 节点比例
// [修改] 解析多 Stage 格式，并支持按算力比例自动切分层数
static std::vector<NnStageDef> parseStageDefs(const char *ratiosStr, NnUint nNodes, NnUint nLayers) {
    printf("🔍 [DEBUG] parseStageDefs received: \"%s\"\n", ratiosStr);

    // ---------------------------------------------------------
    // Ratios string formats (both supported; auto-detected):
    //
    // (A) Legacy per-stage TP ratios (recommended to keep using):
    //   "tp0*tp1*tp2"  where each tp is node ratios (',' or ':' separated)
    //   Optional explicit layers:
    //     - Preferred: append "@<nLayers>" to a stage (works with ':' or ',')
    //     - Legacy:    append ":<nLayers>" ONLY when ratios use commas (e.g. "1,1:10")
    //   Examples:
    //     - 2 nodes, 2 stages: "1*1"
    //     - 4 nodes, 2 stages: "1,1*1,1" or "1:1*1:1"
    //     - explicit layers:   "1:1@10*1:1@18" or "1,1:10*1,1:18"
    //
    // (B) Two-level ratios (stage weights + per-stage TP ratios):
    //   "stageWeights*tpStage0*tpStage1*..."
    //   stageWeights is a list of weights (':' or ',' separated), one per stage.
    //   Each tpStageK is that stage's intra-TP node ratios (':' or ',' separated).
    //   Example (your case): stage weights 1:2; stage0 nodes 1:1; stage1 nodes 2:3
    //     - nNodes=4: "1:2*1:1*2:3"
    // ---------------------------------------------------------

    auto splitStages = [](const std::string& raw) -> std::vector<std::string> {
        std::string s2 = raw;
        for (char &c : s2) {
            if (c == ';' || c == '|') c = '*';
        }
        std::vector<std::string> parts;
        std::stringstream ss2(s2);
        std::string seg;
        while (std::getline(ss2, seg, '*')) {
            if (!seg.empty()) parts.push_back(seg);
        }
        return parts;
    };

    auto isAllDigits = [](const std::string& t) -> bool {
        if (t.empty()) return false;
        for (char c : t) {
            if (c < '0' || c > '9') return false;
        }
        return true;
    };

    // Parse a segment that may be:
    //  - ratios only: "1,1" or "1:1"
    //  - ratios + explicit layers (unambiguous): "1:1@10" or "1,1@10"
    //  - legacy ratios + explicit layers: "1,1:10" (ONLY when ratios use commas)
    //
    // NOTE: We intentionally DO NOT interpret a trailing ":<digits>" as layers when
    // ratios are separated by ':' (e.g. "1:2"), because that would be ambiguous.
    // Returns {ratios, nLayersExplicit(0 if none)}
    auto parseRatiosAndMaybeLayers = [&](const std::string& segment) -> std::pair<std::vector<float>, NnUint> {
        NnUint explicitLayers = 0;
        std::string ratioPart = segment;

        // Preferred, unambiguous layer syntax: "...@<int>"
        {
            const size_t atPos = segment.rfind('@');
            if (atPos != std::string::npos && atPos + 1 < segment.size()) {
                const std::string tail = segment.substr(atPos + 1);
                if (isAllDigits(tail)) {
                    try {
                        explicitLayers = (NnUint)std::stoul(tail);
                        ratioPart = segment.substr(0, atPos);
                    } catch (...) {
                        // ignore
                        explicitLayers = 0;
                        ratioPart = segment;
                    }
                }
            }
        }

        // Legacy layer syntax: "1,1:10" (ONLY when ratios use commas)
        if (explicitLayers == 0u) {
            const bool hasComma = (segment.find(',') != std::string::npos);
            if (hasComma) {
                const size_t lastColon = segment.rfind(':');
                if (lastColon != std::string::npos && lastColon + 1 < segment.size()) {
                    const std::string tail = segment.substr(lastColon + 1);
                    if (isAllDigits(tail)) {
                        try {
                            explicitLayers = (NnUint)std::stoul(tail);
                            ratioPart = segment.substr(0, lastColon);
                        } catch (...) {
                            explicitLayers = 0;
                            ratioPart = segment;
                        }
                    }
                }
            }
        }

        // Now parse ratios: allow both ',' and ':' as separators.
        // We normalize ',' to ':' and then split.
        std::string rp = ratioPart;
        for (char &c : rp) {
            if (c == ',') c = ':';
        }
        std::vector<float> ratios;
        std::stringstream ss4(rp);
        std::string r;
        while (std::getline(ss4, r, ':')) {
            if (r.empty()) continue;
            try {
                ratios.push_back(std::stof(r));
            } catch (...) {
                throw std::invalid_argument("Invalid ratio value: " + r);
            }
        }
        if (ratios.empty()) throw std::invalid_argument("Empty ratio list in segment: " + segment);
        return {ratios, explicitLayers};
    };

    auto sumNodeCounts = [](const std::vector<NnStageDef>& st) -> NnUint {
        NnUint n = 0;
        for (const auto& s : st) n += (NnUint)s.tpRatios.size();
        return n;
    };

    auto autoAssignLayers = [&](std::vector<NnStageDef>& stages, const std::vector<float>& stageWeights) {
        // Collect explicit layers
        NnUint totalExplicitLayers = 0;
        std::vector<int> autoLayerIndices;
        for (size_t i = 0; i < stages.size(); ++i) {
            if (stages[i].nLayers == 0) {
                autoLayerIndices.push_back((int)i);
            } else {
                totalExplicitLayers += stages[i].nLayers;
            }
        }

        if (totalExplicitLayers > nLayers) {
            throw std::invalid_argument("Explicit layers count exceeds total model layers");
        }
        NnUint remainingLayers = nLayers - totalExplicitLayers;

        if (autoLayerIndices.empty()) {
            if (remainingLayers != 0) {
                throw std::invalid_argument("Explicit layers sum does not match total model layers");
            }
            return;
        }

        // Compute weights for auto stages
        float totalWeight = 0.0f;
        std::vector<float> w;
        w.reserve(autoLayerIndices.size());
        for (int idx : autoLayerIndices) {
            float ww = (idx >= 0 && (size_t)idx < stageWeights.size()) ? stageWeights[(size_t)idx] : 0.0f;
            w.push_back(ww);
            totalWeight += ww;
        }

        if (totalWeight <= 1e-6f) {
            // Fallback: uniform
            NnUint base = remainingLayers / (NnUint)autoLayerIndices.size();
            NnUint rem = remainingLayers % (NnUint)autoLayerIndices.size();
            for (size_t i = 0; i < autoLayerIndices.size(); ++i) {
                stages[(size_t)autoLayerIndices[i]].nLayers = base + (i < rem ? 1u : 0u);
            }
            return;
        }

        // Proportional with rounding (last stage gets the remainder)
        NnUint allocatedSoFar = 0;
        for (size_t i = 0; i < autoLayerIndices.size(); ++i) {
            int stageIdx = autoLayerIndices[i];
            NnUint myLayers = 0;
            if (i + 1 == autoLayerIndices.size()) {
                myLayers = remainingLayers - allocatedSoFar;
            } else {
                float ratio = w[i] / totalWeight;
                myLayers = (NnUint)std::round(remainingLayers * ratio);
                if (allocatedSoFar + myLayers > remainingLayers) myLayers = remainingLayers - allocatedSoFar;
            }
            stages[(size_t)stageIdx].nLayers = myLayers;
            allocatedSoFar += myLayers;
            printf("⚖️  [Auto-Split] Stage %d (Weight %.2f): Assigned %u layers\n", stageIdx, w[i], myLayers);
        }
    };

    // ---------- Pass 0: tokenize stage segments ----------
    const std::vector<std::string> parts = splitStages(std::string(ratiosStr));
    if (parts.empty()) throw std::invalid_argument("Ratios string is empty");

    // ---------- Pass 1: try legacy parsing ----------
    {
        std::vector<NnStageDef> stages;
        stages.reserve(parts.size());
        for (const auto& seg : parts) {
            NnStageDef st;
            st.nLayers = 0;
            auto parsed = parseRatiosAndMaybeLayers(seg);
            st.tpRatios = std::move(parsed.first);
            st.nLayers = parsed.second;
            stages.push_back(std::move(st));
        }

        const NnUint totalNodesParsed = sumNodeCounts(stages);
        if (totalNodesParsed == nNodes) {
            // Legacy semantics: stage weight derived from sum(tpRatios)
            std::vector<float> stageWeights;
            stageWeights.reserve(stages.size());
            for (const auto& st : stages) {
                float w = 0.0f;
                for (float r : st.tpRatios) w += r;
                stageWeights.push_back(w);
            }
            autoAssignLayers(stages, stageWeights);
            return stages;
        }
    }

    // ---------- Pass 2: two-level parsing (stageWeights + per-stage tp) ----------
    {
        if (parts.size() < 2) {
            throw std::invalid_argument("Invalid ratios format: not enough segments");
        }

        // First segment = stage weights
        std::vector<float> stageWeights;
        {
            auto parsed = parseRatiosAndMaybeLayers(parts[0]);
            if (parsed.second != 0u) {
                throw std::invalid_argument("Stage-weights segment must not specify layers: " + parts[0]);
            }
            stageWeights = std::move(parsed.first);
        }

        const size_t nStages = stageWeights.size();
        if (nStages == 0) throw std::invalid_argument("Stage weights cannot be empty");
        if (parts.size() != 1 + nStages) {
            std::stringstream msg;
            msg << "Two-level ratios expects 1+" << nStages
                << " segments, but got " << parts.size() << ".\n"
                << "Format: stageWeights*tpStage0*tpStage1*...\n"
            	<< "Example: \"1:2*1:1*2:3\"\n"
                << "Optional explicit layers: tpStage0@10 (e.g. \"1:2*1:1@10*2:3@18\")";
            throw std::invalid_argument(msg.str());
        }

        std::vector<NnStageDef> stages;
        stages.reserve(nStages);
        for (size_t i = 0; i < nStages; ++i) {
            const std::string& seg = parts[1 + i];
            NnStageDef st;
            st.nLayers = 0;
            auto parsed = parseRatiosAndMaybeLayers(seg);
            st.tpRatios = std::move(parsed.first);
            st.nLayers = parsed.second;
            stages.push_back(std::move(st));
        }

        const NnUint totalNodesParsed = sumNodeCounts(stages);
        if (totalNodesParsed != nNodes) {
            std::stringstream msg;
            msg << "Ratios defined " << totalNodesParsed
                << " nodes, but expected " << nNodes << ".\n"
                << "Two-level format example (nNodes=4): \"1:2*1:1*2:3\"\n"
                << "(Stage weights 1:2; stage0 nodes 1:1; stage1 nodes 2:3)\n"
                << "Note: use '@<layers>' if you need explicit layer counts (e.g. \"1:2*1:1@10*2:3@18\").";
            throw std::invalid_argument(msg.str());
        }

        autoAssignLayers(stages, stageWeights);
        return stages;
    }
}

struct WarmupCandidateResult {
    std::string ratios;
    NnUint nWorkers = 0u;
    std::vector<NnUint> workerIndices;
    double scoreMs = std::numeric_limits<double>::infinity();
    bool ok = false;
    std::string error;
};

struct WarmupCandidate {
    NnUint nWorkers = 0u;
    std::vector<NnUint> workerIndices;
    std::string ratios;
};

struct WarmupSelection {
    NnUint nWorkers = 0u;
    std::vector<NnUint> workerIndices;
    std::string ratios;
    double scoreMs = std::numeric_limits<double>::infinity();
};

struct ArgsRestoreGuard {
    AppCliArgs *args;
    char *ratios;
    bool benchmark;
    NnUint nWorkers;
    ~ArgsRestoreGuard() {
        args->ratiosStr = ratios;
        args->benchmark = benchmark;
        args->nWorkers = nWorkers;
    }
};

struct InferenceFinishGuard {
    RootLlmInference *inference;
    bool active;
    ~InferenceFinishGuard() {
        if (active && inference != nullptr) inference->finish();
    }
    void finishNow() {
        if (!active || inference == nullptr) return;
        inference->finish();
        active = false;
    }
};

static std::vector<NnExecutorDevice> resolveDevices(
    AppCliArgs *args,
    NnNetConfig *netConfig,
    NnNodeConfig *nodeConfig,
    NnNetExecution *netExecution,
    const NnUnevenPartitionPlan *plan = nullptr);

static std::vector<std::string> splitWarmupCandidates(const char *raw) {
    std::vector<std::string> out;
    if (raw == nullptr || raw[0] == '\0') return out;
    std::string s(raw);
    for (char &c : s) {
        if (c == ';' || c == '\n' || c == '\t') c = ' ';
    }
    std::stringstream ss(s);
    std::string item;
    while (ss >> item) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static std::string joinUniformTp(NnUint n) {
    std::ostringstream oss;
    for (NnUint i = 0u; i < n; ++i) {
        if (i != 0u) oss << ":";
        oss << "1";
    }
    return oss.str();
}

static std::string makePpRatios(const std::vector<NnUint> &stageSizes, NnUint nLayers) {
    if (stageSizes.empty()) return "";
    std::ostringstream oss;
    const NnUint nStages = (NnUint)stageSizes.size();
    NnUint assigned = 0u;
    for (NnUint i = 0u; i < nStages; ++i) {
        if (i != 0u) oss << "*";
        NnUint layers = 0u;
        if (i + 1u == nStages) {
            layers = nLayers - assigned;
        } else {
            layers = nLayers / nStages;
            if (i < (nLayers % nStages)) layers += 1u;
            assigned += layers;
        }
        oss << joinUniformTp(stageSizes[i]) << "@" << layers;
    }
    return oss.str();
}

static std::string makePpRatiosWithLayerCounts(const std::vector<NnUint> &stageSizes, const std::vector<NnUint> &layerCounts) {
    if (stageSizes.empty() || stageSizes.size() != layerCounts.size()) return "";
    std::ostringstream oss;
    for (size_t i = 0; i < stageSizes.size(); ++i) {
        if (i != 0u) oss << "*";
        oss << joinUniformTp(stageSizes[i]) << "@" << layerCounts[i];
    }
    return oss.str();
}

static void appendTopologyCandidates(std::vector<std::string> &candidates, NnUint nNodes, NnUint nLayers) {
    std::set<std::string> seen(candidates.begin(), candidates.end());
    auto add = [&](const std::string &s) {
        if (s.empty()) return;
        if (seen.insert(s).second) candidates.push_back(s);
    };

    add(joinUniformTp(nNodes));

    std::vector<NnUint> purePp(nNodes, 1u);
    add(makePpRatios(purePp, nLayers));
    if (nNodes == 3u && nLayers >= 3u) {
        const NnUint base = nLayers / 3u;
        const NnUint rem = nLayers % 3u;
        std::vector<NnUint> counts{base, base, base};
        for (NnUint i = 0u; i < rem; ++i) counts[i] += 1u;
        add(makePpRatiosWithLayerCounts({1u, 1u, 1u}, {counts[1], counts[2], counts[0]}));
        add(makePpRatiosWithLayerCounts({1u, 1u, 1u}, {counts[2], counts[0], counts[1]}));
    }

    for (NnUint group = 2u; group <= std::min<NnUint>(nNodes, 4u); ++group) {
        if (nNodes % group == 0u) {
            std::vector<NnUint> sizes(nNodes / group, group);
            add(makePpRatios(sizes, nLayers));
        }
    }
    if (nNodes >= 4u) {
        add(makePpRatios({2u, nNodes - 2u}, nLayers));
        add(makePpRatios({nNodes - 2u, 2u}, nLayers));
    }
    if (nNodes >= 5u) {
        add(makePpRatios({1u, 2u, nNodes - 3u}, nLayers));
        add(makePpRatios({nNodes - 3u, 2u, 1u}, nLayers));
    }
}

static std::vector<WarmupCandidate> buildWarmupCandidates(NnUint maxWorkers, NnUint nLayers, const AppCliArgs *args) {
    std::vector<WarmupCandidate> candidates;
    std::set<std::string> seen;
    auto add = [&](const std::vector<NnUint> &workerIndices, const std::string &ratios) {
        std::ostringstream key;
        for (NnUint idx : workerIndices) key << idx << ",";
        key << "|" << ratios;
        if (seen.insert(key.str()).second) {
            WarmupCandidate c;
            c.nWorkers = (NnUint)workerIndices.size();
            c.workerIndices = workerIndices;
            c.ratios = ratios;
            candidates.push_back(c);
        }
    };
    auto prefixWorkers = [](NnUint nWorkers) {
        std::vector<NnUint> out;
        out.reserve(nWorkers);
        for (NnUint i = 0u; i < nWorkers; ++i) out.push_back(i);
        return out;
    };

    std::vector<std::string> overrideCandidates = splitWarmupCandidates(args->warmupCandidatesStr);
    if (!overrideCandidates.empty()) {
        for (const std::string &r : overrideCandidates) add(prefixWorkers(maxWorkers), r);
        return candidates;
    }

    // First pass: front-load representative configurations so the default
    // budget covers both small-model subsets and full-worker PP/TP variants.
    // nWorkers=0 means root-only single-device inference.
    add({}, std::string());
    if (maxWorkers > 0u) {
        std::vector<NnUint> fullSubset = prefixWorkers(maxWorkers);
        add(fullSubset, joinUniformTp(maxWorkers + 1u));
    }
    for (NnUint idx = 0u; idx < maxWorkers; ++idx) {
        add({idx}, joinUniformTp(2u));
    }
    if (maxWorkers > 0u) {
        const NnUint nNodes = maxWorkers + 1u;
        std::vector<NnUint> fullSubset = prefixWorkers(maxWorkers);
        std::vector<std::string> fullRatios;
        appendTopologyCandidates(fullRatios, nNodes, nLayers);
        for (const std::string &r : fullRatios) add(fullSubset, r);
    }
    for (NnUint workers = 2u; workers < maxWorkers; ++workers) {
        add(prefixWorkers(workers), joinUniformTp(workers + 1u));
    }

    // Second pass: add bounded PP/hybrid variants for smaller prefix subsets.
    for (NnUint workers = 1u; workers < maxWorkers; ++workers) {
        const NnUint nNodes = workers + 1u;
        std::vector<NnUint> subset = prefixWorkers(workers);
        std::vector<std::string> ratios;
        appendTopologyCandidates(ratios, nNodes, nLayers);
        for (const std::string &r : ratios) add(subset, r);
    }
    return candidates;
}

static int warmupRelistenDelayMs() {
    int delayMs = 1000;
    parseEnvInt("DLLAMA_WARMUP_RELISTEN_DELAY_MS", delayMs);
    if (delayMs < 0) delayMs = 0;
    return delayMs;
}

static void waitForWarmupWorkersToRelisten() {
    const int delayMs = warmupRelistenDelayMs();
    if (delayMs <= 0) return;
    // Workers loop back to serve() after receiving finish(). Give them a bounded
    // relisten window before the next root connects.
    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
}

static void buildCandidateWorkerArrays(
    AppCliArgs *args,
    const WarmupCandidate &candidate,
    std::vector<char*> &hosts,
    std::vector<NnUint> &ports) {
    hosts.clear();
    ports.clear();
    hosts.reserve(candidate.workerIndices.size());
    ports.reserve(candidate.workerIndices.size());
    for (NnUint idx : candidate.workerIndices) {
        if (idx >= args->workerAllocCount) {
            throw std::runtime_error("warmup worker subset index out of range");
        }
        hosts.push_back(args->workerHosts[idx]);
        ports.push_back(args->workerPorts[idx]);
    }
}

static void applyWarmupWorkerSubset(AppCliArgs *args, const WarmupSelection &selection) {
    if (selection.nWorkers == args->nWorkers && selection.workerIndices.empty()) return;
    if (selection.nWorkers == 0u) {
        args->nWorkers = 0u;
        return;
    }
    std::vector<bool> selected(args->workerAllocCount, false);
    std::vector<char*> newHosts;
    std::vector<NnUint> newPorts;
    newHosts.reserve(args->workerAllocCount);
    newPorts.reserve(args->workerAllocCount);
    for (NnUint idx : selection.workerIndices) {
        if (idx >= args->workerAllocCount) throw std::runtime_error("warmup selected worker index out of range");
        if (selected[idx]) continue;
        selected[idx] = true;
        newHosts.push_back(args->workerHosts[idx]);
        newPorts.push_back(args->workerPorts[idx]);
    }
    for (NnUint idx = 0u; idx < args->workerAllocCount; ++idx) {
        if (selected[idx]) continue;
        newHosts.push_back(args->workerHosts[idx]);
        newPorts.push_back(args->workerPorts[idx]);
    }
    for (NnUint idx = 0u; idx < args->workerAllocCount; ++idx) {
        args->workerHosts[idx] = newHosts[idx];
        args->workerPorts[idx] = newPorts[idx];
    }
    args->nWorkers = (NnUint)selection.workerIndices.size();
}

static std::string warmupWorkerIndicesString(const std::vector<NnUint> &indices) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i != 0u) oss << ",";
        oss << indices[i];
    }
    oss << "]";
    return oss.str();
}

static const char *warmupRatiosLabel(const std::string &ratios) {
    return ratios.empty() ? "(single)" : ratios.c_str();
}

static void accumulateWarmupPerf(
    const std::vector<LlmPerfPacket> &perf,
    std::vector<unsigned long long> &execUs,
    std::vector<unsigned long long> &syncUs,
    std::vector<unsigned long long> &bubbleUs,
    std::vector<unsigned long long> &tokens,
    std::vector<NnUint> &stageIndex,
    std::vector<bool> &hasStage) {
    for (const LlmPerfPacket &p : perf) {
        if (p.nodeIndex >= execUs.size()) continue;
        execUs[p.nodeIndex] += p.execUs;
        syncUs[p.nodeIndex] += p.syncUs;
        bubbleUs[p.nodeIndex] += p.bubbleUs;
        tokens[p.nodeIndex] += (unsigned long long)std::max<NnUint>(1u, p.batchSize);
        stageIndex[p.nodeIndex] = p.stageIndex;
        hasStage[p.nodeIndex] = true;
    }
}

static double warmupMaxStageNodeMs(
    const std::vector<unsigned long long> &execUs,
    const std::vector<unsigned long long> &syncUs,
    const std::vector<unsigned long long> &bubbleUs,
    const std::vector<unsigned long long> &tokens,
    const std::string &ratios) {
    double maxMs = 0.0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == 0ull) continue;
        const double ms = (double)(execUs[i] + syncUs[i] + bubbleUs[i]) / 1000.0 / (double)tokens[i];
        if (ms > maxMs) maxMs = ms;
    }
    if (maxMs <= 0.0) {
        throw std::runtime_error("warmup produced no stage/node profile packets for ratios=" + ratios);
    }
    return maxMs;
}

static WarmupCandidateResult probeWarmupCandidate(
    AppCliArgs *args,
    LlmHeader *header,
    Tokenizer *tokenizer,
    const WarmupCandidate &candidate) {
    WarmupCandidateResult result;
    result.ratios = candidate.ratios;
    result.nWorkers = candidate.nWorkers;
    result.workerIndices = candidate.workerIndices;
    try {
        const NnUint nWorkers = candidate.nWorkers;
        const NnUint nNodes = nWorkers + 1u;
        const bool distributed = nWorkers > 0u;
        const bool uneven = distributed && !candidate.ratios.empty();
        printf("🔥 [warmup] probing workers=%u workerIndices=%s nNodes=%u ratios=%s\n",
            (unsigned)nWorkers,
            warmupWorkerIndicesString(candidate.workerIndices).c_str(),
            (unsigned)nNodes,
            warmupRatiosLabel(candidate.ratios));
        std::fflush(stdout);

        std::unique_ptr<NnUnevenPartitionPlan> planPtr;
        if (uneven) {
            std::vector<NnStageDef> stageDefs = parseStageDefs(candidate.ratios.c_str(), nNodes, header->nLayers);
            const NnUint ffDim = (header->archType == QWEN3_MOE) ? header->moeHiddenDim : header->hiddenDim;
            std::vector<NnUint> kvRedundancyPerNode = parseKvRedundancy(args->kvRedundancyStr, nNodes);
            planPtr.reset(new NnUnevenPartitionPlan(
                createPartitionPlan(stageDefs, header->nHeads, header->nKvHeads, header->vocabSize, ffDim, header->dim, kvRedundancyPerNode)
            ));
        }

        LlmNet net = uneven
            ? buildLlmNetUneven(header, nNodes, args->nBatches, planPtr.get())
            : buildLlmNet(header, nNodes, args->nBatches);
        std::unique_ptr<LlmNet, void(*)(LlmNet *)> netPtr(&net, releaseLlmNet);
        NnNodeConfig *rootNodeConfig = &net.nodeConfigs[0];
        NnNetExecution execution(args->nThreads, &net.netConfig);
        std::vector<char*> candidateHosts;
        std::vector<NnUint> candidatePorts;
        if (distributed) buildCandidateWorkerArrays(args, candidate, candidateHosts, candidatePorts);
        std::unique_ptr<NnNetwork> networkPtr = distributed
            ? NnNetwork::connect(nWorkers, candidateHosts.data(), candidatePorts.data())
            : std::unique_ptr<NnNetwork>(nullptr);
        NnNetwork *network = networkPtr.get();

        char *savedRatios = args->ratiosStr;
        const bool savedBenchmark = args->benchmark;
        const NnUint savedWorkers = args->nWorkers;
        ArgsRestoreGuard argsGuard{args, savedRatios, savedBenchmark, savedWorkers};
        std::string probeRatios = candidate.ratios;
        args->nWorkers = nWorkers;
        args->ratiosStr = uneven ? const_cast<char*>(probeRatios.c_str()) : nullptr;
        args->benchmark = true;
        for (NnUint nodeIndex = 1u; nodeIndex < nNodes && network != nullptr; ++nodeIndex) {
            writeBootstrapPacket(network, nodeIndex - 1u, args);
        }
        args->ratiosStr = savedRatios;
        args->benchmark = savedBenchmark;
        args->nWorkers = savedWorkers;

        std::unique_ptr<NnNodeSynchronizer> synchronizer;
        if (distributed) {
            synchronizer.reset(new NnNetworkNodeSynchronizer(network, &execution, &net.netConfig, rootNodeConfig, planPtr.get(), false));
            NnRootConfigWriter configWriter(network);
            configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
        } else {
            synchronizer.reset(new NnFakeNodeSynchronizer());
        }
        std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution, planPtr.get());
        NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), true);

        if (uneven) {
            NnLocalWeightLoader localLoader(&executor, 0);
            loadLlmNetWeightUneven(args->modelPath, &net, &localLoader, planPtr.get(), 0);
        } else {
            NnRootWeightLoader weightLoader(&executor, network, nNodes);
            loadLlmNetWeight(args->modelPath, &net, &weightLoader);
        }

        RootLlmInference inference(&net, &execution, &executor, network, planPtr.get(), true, args->enablePpMigration);
        InferenceFinishGuard finishGuard{&inference, true};
        if (network != nullptr) {
            network->resetStats();
            if (args->netTurbo) network->setTurbo(true);
        }

        const char *prompt = (args->prompt != nullptr && args->prompt[0] != '\0')
            ? args->prompt
            : "Write a comma-separated list of the numbers from 1 to 20.";
        std::string effectivePrompt(prompt);
        std::vector<int> inputTokensVec(effectivePrompt.size() + 3);
        int *inputTokens = inputTokensVec.data();
        int nInputTokens = (int)inputTokensVec.size();
        tokenizer->encode(const_cast<char*>(effectivePrompt.c_str()), inputTokens, &nInputTokens, true, true);
        if (nInputTokens <= 1) throw std::runtime_error("warmup prompt produced too few tokens");
        NnUint steps = std::max<NnUint>((NnUint)nInputTokens + args->warmupSteps, 2u);
        steps = std::min<NnUint>(steps, header->seqLen);

        std::vector<unsigned long long> execUs(nNodes, 0ull);
        std::vector<unsigned long long> syncUs(nNodes, 0ull);
        std::vector<unsigned long long> bubbleUs(nNodes, 0ull);
        std::vector<unsigned long long> tokens(nNodes, 0ull);
        std::vector<NnUint> stageIndex(nNodes, 0u);
        std::vector<bool> hasStage(nNodes, false);

        NnUint pos = 0u;
        while ((long)(nInputTokens - 1) - (long)pos > 0) {
            const long remaining = (long)(nInputTokens - 1) - (long)pos;
            const NnUint batchSize = (NnUint)std::min<long>(remaining, (long)args->nBatches);
            inference.setBatchSize(batchSize);
            inference.setPosition(pos);
            for (NnUint i = 0u; i < batchSize; ++i) inference.setToken(i, (NnUint)inputTokens[pos + i]);
            inference.forward();
            accumulateWarmupPerf(inference.getLastPerf(), execUs, syncUs, bubbleUs, tokens, stageIndex, hasStage);
            pos += batchSize;
        }

        int token = inputTokens[nInputTokens - 1];
        inference.setBatchSize(1u);
        tokenizer->resetDecoder();
        Sampler warmupSampler(tokenizer->vocabSize, args->temperature, args->topp, args->seed);
        for (; pos < steps; ++pos) {
            inference.setPosition(pos);
            inference.setToken(0u, (NnUint)token);
            inference.forward();
            accumulateWarmupPerf(inference.getLastPerf(), execUs, syncUs, bubbleUs, tokens, stageIndex, hasStage);
            token = warmupSampler.sample(inference.logitsPipe);
        }

        result.scoreMs = warmupMaxStageNodeMs(execUs, syncUs, bubbleUs, tokens, warmupRatiosLabel(candidate.ratios));
        result.ok = true;
        printf("🔥 [warmup] workers=%u workerIndices=%s ratios=%s score_max_stage_node_ms_per_tok=%.3f\n",
            (unsigned)nWorkers,
            warmupWorkerIndicesString(candidate.workerIndices).c_str(),
            warmupRatiosLabel(candidate.ratios),
            result.scoreMs);
        for (NnUint node = 0u; node < nNodes; ++node) {
            if (tokens[node] == 0ull) continue;
            const double ms = (double)(execUs[node] + syncUs[node] + bubbleUs[node]) / 1000.0 / (double)tokens[node];
            printf("🔥 [warmup]   Stage %u Node %u per_tok_total=%.3f ms tok=%llu\n",
                (unsigned)(hasStage[node] ? stageIndex[node] : 0xFFFFFFFFu),
                (unsigned)node,
                ms,
                tokens[node]);
        }
        std::fflush(stdout);
        finishGuard.finishNow();
    } catch (const std::exception &e) {
        result.ok = false;
        result.error = e.what();
        printf("🔥 [warmup] workers=%u workerIndices=%s ratios=%s failed: %s\n",
            (unsigned)candidate.nWorkers,
            warmupWorkerIndicesString(candidate.workerIndices).c_str(),
            warmupRatiosLabel(candidate.ratios),
            e.what());
        std::fflush(stdout);
    }
    return result;
}

static WarmupSelection selectWarmupConfiguration(
    AppCliArgs *args,
    LlmHeader *header,
    Tokenizer *tokenizer,
    NnUint maxWorkers) {
    std::vector<WarmupCandidate> candidates = buildWarmupCandidates(maxWorkers, header->nLayers, args);
    if (candidates.empty()) throw std::runtime_error("--warmup has no candidate ratios");

    const NnUint budget = std::max<NnUint>(1u, args->warmupBudget);
    WarmupCandidateResult best;
    printf("🔥 [warmup] enabled maxWorkers=%u steps=%u budget=%u candidates=%zu scoring=max Stage/Node Profile per-token total\n",
        (unsigned)maxWorkers,
        (unsigned)args->warmupSteps,
        (unsigned)budget,
        candidates.size());
    for (NnUint i = 0u; i < (NnUint)candidates.size() && i < budget; ++i) {
        WarmupCandidateResult r = probeWarmupCandidate(args, header, tokenizer, candidates[i]);
        if (r.ok && (!best.ok || r.scoreMs < best.scoreMs)) {
            best = r;
        }
        if (r.nWorkers > 0u && i + 1u < (NnUint)candidates.size() && i + 1u < budget) {
            waitForWarmupWorkersToRelisten();
        }
    }
    if (!best.ok) {
        throw std::runtime_error("--warmup failed for all probed candidate ratios");
    }
    printf("🔥 [warmup] selected workers=%u workerIndices=%s nNodes=%u ratios=%s score_max_stage_node_ms_per_tok=%.3f\n",
        (unsigned)best.nWorkers,
        warmupWorkerIndicesString(best.workerIndices).c_str(),
        (unsigned)(best.nWorkers + 1u),
        warmupRatiosLabel(best.ratios),
        best.scoreMs);
    std::fflush(stdout);
    WarmupSelection selection;
    selection.nWorkers = best.nWorkers;
    selection.workerIndices = best.workerIndices;
    selection.ratios = best.ratios;
    selection.scoreMs = best.scoreMs;
    return selection;
}

void printPartitionPlanDebug(const NnUnevenPartitionPlan* plan) {
    printf("\n🔍 [DEBUG] Pipeline Partition Plan Verification:\n");
    printf("===================================================\n");
    printf("🌎 Global Stats: Total Nodes: %u, Total Stages: %u\n", plan->nNodes, plan->nStages);

    for (NnUint s = 0; s < plan->nStages; ++s) {
        const NnStageConfig& stage = plan->stages[s];
        printf("\n➡️  [Stage %u]\n", stage.stageIndex);
        printf("    ├─ Range:      Layers %u to %u (Count: %u)\n", 
               stage.startLayer, stage.endLayer - 1, stage.nLayers);
        printf("    ├─ Root Node:  %u\n", stage.rootNodeIndex);
        printf("    ├─ Member Nodes: [ ");
        for(NnUint i=0; i<stage.nNodes; ++i) printf("%u ", stage.nodeIndices[i]);
        printf("]\n");

        printf("    └─ 🔍 TP Split Isolation Check:\n");
        NnUint headSum = 0;
        NnUint kvSum = 0;
        NnUint dimSum = 0;

        for(NnUint i=0; i<stage.nNodes; ++i) {
            NnUint globalNodeIdx = stage.nodeIndices[i];
            
            NnUint hLen = plan->headSplit.lengths[globalNodeIdx];
            NnUint kLen = plan->kvHeadSplit.lengths[globalNodeIdx];
            NnUint dLen = plan->dimSplit.lengths[globalNodeIdx];
            
            headSum += hLen;
            kvSum += kLen;
            dimSum += dLen;

            printf("       • Node %u: Heads=%u, KV=%u, Dim=%u\n", 
                   globalNodeIdx, hLen, kLen, dLen);
        }
        printf("       ✅ Stage Sums: Heads=%u, KV=%u, Dim=%u\n", headSum, kvSum, dimSum);
    }
    printf("===================================================\n\n");
}

static std::vector<NnExecutorDevice> resolveDevices(AppCliArgs *args, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *plan) {
    std::vector<NnExecutorDevice> devices;

    if (args->backend == AppCliArgs::BACKEND_CPU) {
        devices.push_back(NnExecutorDevice(new NnCpuDevice(netConfig, nodeConfig, netExecution, plan), -1, -1));
        return devices;
    }

    if (args->backend == AppCliArgs::BACKEND_VULKAN) {
#if defined(DLLAMA_VULKAN)
        devices.push_back(NnExecutorDevice(
            new NnVulkanDevice(args->gpuIndex, netConfig, nodeConfig, netExecution, plan),
            args->gpuSegmentFrom,
            args->gpuSegmentTo
        ));
#else
        throw std::runtime_error("--backend vulkan requested, but this build was not compiled with DLLAMA_VULKAN=1");
#endif
    } else if (args->backend == AppCliArgs::BACKEND_CUDA) {
#if defined(DLLAMA_CUDA)
        devices.push_back(NnExecutorDevice(
            new NnCudaDevice(args->gpuIndex, netConfig, nodeConfig, netExecution, plan),
            args->gpuSegmentFrom,
            args->gpuSegmentTo
        ));
#else
        throw std::runtime_error("--backend cuda requested, but this build was not compiled with DLLAMA_CUDA=1");
#endif
    } else {
        throw std::runtime_error("Internal error: unresolved backend " + std::string(AppCliArgs::backendToString(args->backend)));
    }

    if (args->gpuSegmentFrom >= 0 && args->gpuSegmentTo >= 0) {
        devices.push_back(NnExecutorDevice(new NnCpuDevice(netConfig, nodeConfig, netExecution, plan), -1, -1));
    }
    return devices;
}

static NnUint getStageIndexForNode(const NnUnevenPartitionPlan* plan, NnUint nodeIndex) {
    if (plan == nullptr || plan->nStages == 0) return 0;
    for (NnUint s = 0; s < plan->nStages; ++s) {
        const NnStageConfig* st = &plan->stages[s];
        for (NnUint i = 0; i < st->nNodes; ++i) {
            if (st->nodeIndices[i] == nodeIndex) return st->stageIndex;
        }
    }
    return 0;
}

static const NnStageConfig* findStageForNodeLocal(const NnUnevenPartitionPlan* plan, NnUint nodeIndex) {
    if (plan == nullptr || plan->nStages == 0) return nullptr;
    for (NnUint s = 0; s < plan->nStages; ++s) {
        const NnStageConfig* st = &plan->stages[s];
        for (NnUint i = 0; i < st->nNodes; ++i) {
            if (st->nodeIndices[i] == nodeIndex) return st;
        }
    }
    return nullptr;
}

static const NnStageConfig* findStageByIndexLocal(const NnUnevenPartitionPlan* plan, NnUint stageIndex) {
    if (plan == nullptr || plan->nStages == 0) return nullptr;
    for (NnUint s = 0; s < plan->nStages; ++s) {
        if (plan->stages[s].stageIndex == stageIndex) return &plan->stages[s];
    }
    return nullptr;
}

static bool stageContainsNodeLocal(const NnStageConfig *stage, NnUint nodeIndex) {
    if (stage == nullptr) return false;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        if (stage->nodeIndices[i] == nodeIndex) return true;
    }
    return false;
}

static bool areNodesInSameStageLocal(const NnUnevenPartitionPlan *plan, NnUint nodeA, NnUint nodeB) {
    if (plan == nullptr) return nodeA == nodeB;
    const NnStageConfig *stageA = findStageForNodeLocal(plan, nodeA);
    if (stageA == nullptr) return nodeA == nodeB;
    return stageContainsNodeLocal(stageA, nodeB);
}

static std::vector<NnUint> stageNodeListLocal(const NnStageConfig *stage) {
    std::vector<NnUint> out;
    if (stage == nullptr) return out;
    out.reserve(stage->nNodes);
    for (NnUint i = 0u; i < stage->nNodes; ++i) {
        out.push_back(stage->nodeIndices[i]);
    }
    return out;
}

static bool stageRankLocal(const NnStageConfig *stage, NnUint node, NnUint *rank) {
    if (rank != nullptr) *rank = 0u;
    if (stage == nullptr || stage->nodeIndices == nullptr) return false;
    for (NnUint i = 0u; i < stage->nNodes; ++i) {
        if (stage->nodeIndices[i] == node) {
            if (rank != nullptr) *rank = i;
            return true;
        }
    }
    return false;
}

static bool getHeadMigrationTargetRangeLocal(
    const NnUnevenPartitionPlan *plan,
    const PlanCommand &cmd,
    bool planAfterApply,
    NnUint *stageIndex,
    NnUint *fromNode,
    NnUint *toNode,
    NnUint *rangeStart,
    NnUint *rangeLen) {
    if (stageIndex != nullptr) *stageIndex = 0xFFFFFFFFu;
    if (fromNode != nullptr) *fromNode = 0u;
    if (toNode != nullptr) *toNode = 0u;
    if (rangeStart != nullptr) *rangeStart = 0u;
    if (rangeLen != nullptr) *rangeLen = 0u;
    if (plan == nullptr || plan->nStages == 0u ||
        plan->kvHeadSplit.starts == nullptr || plan->kvHeadSplit.lengths == nullptr) {
        return false;
    }

    NnUint f = 0u;
    NnUint t = 0u;
    NnUint move = 0u;
    if (cmd.version == DLLAMA_PLAN_CMD_VERSION_V2 && cmd.nMoves > 0u) {
        const PlanMove &m = cmd.moves[0];
        if (m.cmdKind != PLAN_CMD_KIND_HEAD && m.cmdKind != PLAN_CMD_KIND_BOTH) return false;
        f = m.fromNodeIndex;
        t = m.toNodeIndex;
        move = m.headMove;
    } else {
        if (cmd.cmdKind != PLAN_CMD_KIND_HEAD && cmd.cmdKind != PLAN_CMD_KIND_BOTH) return false;
        f = cmd.fromNodeIndex;
        t = cmd.toNodeIndex;
        move = cmd.nHeadsToMove;
    }
    if (move == 0u || f >= plan->nNodes || t >= plan->nNodes || f == t) return false;

    const NnStageConfig *stage = findStageForNodeLocal(plan, f);
    if (stage == nullptr || !stageContainsNodeLocal(stage, t)) return false;
    NnUint rf = 0u;
    NnUint rt = 0u;
    if (!stageRankLocal(stage, f, &rf) || !stageRankLocal(stage, t, &rt)) return false;
    const NnUint dist = (rf >= rt) ? (rf - rt) : (rt - rf);
    if (dist != 1u) return false;

    const NnUint curStart = plan->kvHeadSplit.starts[f];
    const NnUint curLen = plan->kvHeadSplit.lengths[f];
    const NnUint preLen = planAfterApply ? (curLen + move) : curLen;
    if (preLen < move) return false;

    NnUint start = 0u;
    if (rt > rf) {
        start = curStart + preLen - move;
    } else {
        if (planAfterApply) {
            if (curStart < move) return false;
            start = curStart - move;
        } else {
            start = curStart;
        }
    }
    if (stageIndex != nullptr) *stageIndex = stage->stageIndex;
    if (fromNode != nullptr) *fromNode = f;
    if (toNode != nullptr) *toNode = t;
    if (rangeStart != nullptr) *rangeStart = start;
    if (rangeLen != nullptr) *rangeLen = move;
    return true;
}

RootLlmInference::RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network, const NnUnevenPartitionPlan* plan, bool profileEnabled, bool ppMigrationEnabled) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->kvAggKPipe = (net->kvAggKPipeIndex != (NnUint)-1) ? (float *)execution->pipes[net->kvAggKPipeIndex] : nullptr;
    this->kvAggVPipe = (net->kvAggVPipeIndex != (NnUint)-1) ? (float *)execution->pipes[net->kvAggVPipeIndex] : nullptr;
    this->execution = execution;
    this->executor = executor;
    this->network = network;
    this->plan = plan;
    this->runtimePlan = (net != nullptr) ? &net->runtimeStageLayerPlan : nullptr;
    this->profileEnabled = profileEnabled;
    this->controlPacket.flags = profileEnabled ? LLM_CTRL_PROFILE : 0u;
    this->controlPacket.planCmdSeq = 0u;
    this->lastPlanCmdSeqSent = 0u;
    this->lastBubbleShadowStats = {};
    this->asyncKvCollectLayer = -1;
    this->asyncKvCollectPos = -1;
    waitingKvAckReceivedCount = 0u;
    waitingKvAckReceivedLayers.clear();
    waitingKvAckNodes.clear();
    waitingKvAckNodeExpected.clear();
    waitingKvAckNodeReceived.clear();
    tokenHistory.clear();

    boundaryLayerForMigration = -1;
    migrationLayers.clear();
    migrationStageStartLayer = -1;
    migrationStageEndLayer = -1;
    migrationLayerCount = 1;
    migrationLayerListPinnedByEnv = false;
    this->ppMigrationEnabled = ppMigrationEnabled;
    migrationBatchSubmitted = false;
    migrationAckSeen = false;
    migrationAckPos = -1;
    migrationAckLayer = -1;
    lastPpPlanCacheSeqApplied = 0ull;
    lastHeadRecoveryPlanCacheSeqApplied = 0ull;
    headRecoveryRangeCacheSeq = 0ull;
    headRecoveryRangeCached = false;
    headRecoveryPlanAfterApply = false;
    headRecoveryStageIndex = 0xFFFFFFFFu;
    headRecoveryFromNode = 0u;
    headRecoveryToNode = 0u;
    headRecoveryRangeHeadStart = 0u;
    headRecoveryRangeHeadLen = 0u;
    migrationFromNodeIndex = 0u;
    nextStageRootNode = (NnUint)-1;
    kvAckSocketIndex = -1;
    nextKvExportRequestId = 1u;
    lastMigrationStateTransferBytes = 0u;
    lastMigrationExportedRows = 0u;

    if (plan != nullptr) {
        const NnStageConfig *myStage = findStageForNodeLocal(plan, 0u);
        if (myStage != nullptr && myStage->endLayer > myStage->startLayer) {
            migrationFromNodeIndex = myStage->rootNodeIndex;
            boundaryLayerForMigration = (int)(myStage->endLayer - 1u);
            migrationStageStartLayer = (int)myStage->startLayer;
            migrationStageEndLayer = (int)myStage->endLayer;
            const NnUint targetStageIndex = myStage->stageIndex + 1u;
            const NnStageConfig *targetStage = findStageByIndexLocal(plan, targetStageIndex);
            if (targetStage != nullptr) {
                nextStageRootNode = targetStage->rootNodeIndex;
                if (network != nullptr) {
                    kvAckSocketIndex = network->getSocketIndexForNode(nextStageRootNode, 0u);
                }
            } else {
                std::printf("⚠️  [kv-migrate] default migration route(next) has no target stage from stage=%u\n",
                    (unsigned)myStage->stageIndex);
                std::fflush(stdout);
            }
        }
    }

    migrationLayers = parseLayerListEnv("DLLAMA_MIGRATION_LAYER_LIST");
    migrationLayerListPinnedByEnv = !migrationLayers.empty();
    if (migrationLayerCount < 1) migrationLayerCount = 1;
    if (migrationLayers.empty() && boundaryLayerForMigration >= 0) {
        const int minLayer = (migrationStageStartLayer >= 0) ? migrationStageStartLayer : 0;
        for (int i = 0; i < migrationLayerCount; ++i) {
            const int layer = boundaryLayerForMigration - i;
            if (layer < minLayer) break;
            appendUniqueLayer(migrationLayers, (NnUint)layer);
        }
    }
    std::sort(migrationLayers.begin(), migrationLayers.end());

    if (!migrationLayers.empty() && this->ppMigrationEnabled) {
        std::ostringstream oss;
        for (size_t i = 0; i < migrationLayers.size(); ++i) {
            if (i > 0) oss << ",";
            oss << migrationLayers[i];
        }
        std::printf("🧭 [kv-migrate] root migration route=%u->%u layers=[%s] layerCount=%d pinnedByEnv=%s\n",
            (unsigned)migrationFromNodeIndex,
            (unsigned)nextStageRootNode,
            oss.str().c_str(),
            migrationLayerCount,
            migrationLayerListPinnedByEnv ? "yes" : "no");
        std::fflush(stdout);
    } else if (!this->ppMigrationEnabled) {
        std::printf("ℹ️  [kv-migrate] PP migration is disabled (enable with --enable-pp-migration)\n");
        std::fflush(stdout);
    }

    if (this->ppMigrationEnabled && asyncKvCollectLayer < 0 && boundaryLayerForMigration >= 0) {
        asyncKvCollectLayer = boundaryLayerForMigration;
    }

    int envPpPos = -1;
    if (this->ppMigrationEnabled && parseEnvInt("DLLAMA_PP_MIGRATION_POS", envPpPos) && envPpPos >= 0) {
        asyncKvCollectPos = envPpPos;
        std::printf("🧭 [kv-migrate] armed by env DLLAMA_PP_MIGRATION_POS=%d\n", asyncKvCollectPos);
        std::fflush(stdout);
    }
}

void RootLlmInference::setBatchSize(NnUint batchSize) {
    execution->setBatchSize(batchSize);
    controlPacket.batchSize = batchSize;
}

void RootLlmInference::setPosition(NnUint position) {
    assert(position >= 0);
    assert(position + execution->batchSize - 1 < header->seqLen);

    controlPacket.position = position;
    for (NnUint i = 0; i < execution->batchSize; i++)
        positionPipe[i] = (float)(position + i);
}

void RootLlmInference::setToken(NnUint batchIndex, NnUint token) {
    assert(batchIndex >= 0 && batchIndex < execution->batchSize);
    tokenPipe[batchIndex] = (float)token;
    const NnUint pos = controlPacket.position + batchIndex;
    if (header != nullptr && pos < header->seqLen) {
        if (tokenHistory.size() <= pos) {
            tokenHistory.resize((size_t)pos + 1u, -1);
        }
        tokenHistory[pos] = (int)token;
    }
}

bool RootLlmInference::tryReceiveLastStageSampledToken(NnUint &token, float *logit) {
    if (!lastStageSamplingPlanSupported(plan) || network == nullptr) return false;
    const NnStageConfig &last = plan->stages[plan->nStages - 1u];
    const NnUint sourceNode = last.rootNodeIndex;
    if (sourceNode == 0u || sourceNode >= plan->nNodes) return false;
    const int socketIndex = network->getSocketIndexForNode(sourceNode, 0u);
    if (socketIndex < 0) return false;

    LlmSampledTokenPacket packet{};
    network->read((NnUint)socketIndex, &packet, sizeof(packet));
    if (packet.magic != LLM_SAMPLED_TOKEN_MAGIC || packet.version != LLM_SAMPLED_TOKEN_VERSION) {
        std::printf("⚠️  [last-stage-sampling] unexpected packet magic=0x%08x version=%u fromNode=%u\n",
            (unsigned)packet.magic,
            (unsigned)packet.version,
            (unsigned)sourceNode);
        std::fflush(stdout);
        return false;
    }
    token = packet.token;
    if (logit != nullptr) *logit = packet.logit;
    return true;
}

void RootLlmInference::setRuntimeLayerGate(bool enablePrimarySegments, bool enableRedundantSegments) {
    if (executor == nullptr) return;
    executor->setRuntimeLayerGate(enablePrimarySegments, enableRedundantSegments);
}

void RootLlmInference::setPrimaryLayerEnabled(NnUint layerIndex, bool enabled) {
    if (executor == nullptr) return;
    executor->setPrimaryLayerEnabled(layerIndex, enabled);
}

void RootLlmInference::setShiftedPpStartLayerEnabled(NnUint layerIndex, bool enabled) {
    if (executor == nullptr) return;
    executor->setShiftedPpStartLayerEnabled(layerIndex, enabled);
}

void RootLlmInference::maybeEnableShiftedPpStartForSourceStage(
    const std::vector<NnUint> &switchLayers,
    NnUint sourceNodeIndex,
    NnUint targetNodeIndex,
    bool selfIsSource) {
    if (!selfIsSource || executor == nullptr || switchLayers.empty()) return;
    const NnStageConfig *fromStage = findStageForNodeLocal(plan, sourceNodeIndex);
    const NnStageConfig *toStage = findStageForNodeLocal(plan, targetNodeIndex);
    if (fromStage == nullptr || toStage == nullptr) return;
    if (toStage->stageIndex >= fromStage->stageIndex) return;

    // Root applies batched switches. After migrating multiple consecutive
    // start layers, the first retained layer becomes the new PP entry.
    const NnUint shiftedStartLayer = *std::max_element(switchLayers.begin(), switchLayers.end()) + 1u;
    if (shiftedStartLayer < fromStage->endLayer) {
        executor->setShiftedPpStartLayerEnabled(shiftedStartLayer, true);
    }
}

bool RootLlmInference::hasAsyncKvCollector() const {
    return (asyncKvCollectLayer >= 0) && (asyncKvCollectPos >= 0) &&
           (kvAggKPipe != nullptr) && (kvAggVPipe != nullptr) &&
           (execution != nullptr) && (header != nullptr);
}

bool RootLlmInference::tryPopAsyncKvRow(RootKvAggRowPacket &packet) {
    std::lock_guard<std::mutex> lock(asyncKvRowsMutex);
    if (asyncKvRows.empty()) return false;
    packet = std::move(asyncKvRows.front());
    asyncKvRows.pop_front();
    return true;
}

RootLlmInference::KvTransferSubmitStatus RootLlmInference::submitBoundaryKvTransferDetailed(
    NnUint layerIndex,
    NnUint position,
    const std::vector<float> &kRow,
    const std::vector<float> &vRow,
    NnUint rangeStart,
    NnUint rangeLen) {
    if (network == nullptr) return KV_TRANSFER_SUBMIT_NO_NETWORK;
    if (nextStageRootNode == (NnUint)-1) return KV_TRANSFER_SUBMIT_NO_TARGET_STAGE;
    if (!migrationLayers.empty()) {
        const bool inList = std::find(migrationLayers.begin(), migrationLayers.end(), layerIndex) != migrationLayers.end();
        if (!inList) return KV_TRANSFER_SUBMIT_LAYER_NOT_IN_LIST;
    }
    if (kRow.empty() || vRow.empty() || kRow.size() != vRow.size()) return KV_TRANSFER_SUBMIT_INVALID_ROW;
    if (rangeLen != 0u && rangeLen != kRow.size()) return KV_TRANSFER_SUBMIT_INVALID_ROW;

    std::lock_guard<std::mutex> lk(kvTransferMutex);
    if (waitingKvAck) return KV_TRANSFER_SUBMIT_WAITING_ACK;

    auto exists = [&](NnUint layer, NnUint pos, NnUint targetNode) {
        for (const auto &it : pendingKvTransfers) {
            if (it.header.layerIndex == layer &&
                it.header.position == pos &&
                it.header.targetNodeIndex == targetNode) return true;
        }
        return false;
    };

    std::vector<NnUint> targetNodes;
    if (plan != nullptr) {
        const NnStageConfig *targetStage = findStageForNodeLocal(plan, nextStageRootNode);
        if (targetStage != nullptr && targetStage->nNodes > 0u) {
            targetNodes.reserve(targetStage->nNodes);
            for (NnUint i = 0u; i < targetStage->nNodes; ++i) {
                targetNodes.push_back(targetStage->nodeIndices[i]);
            }
        }
    }
    if (targetNodes.empty()) {
        targetNodes.push_back(nextStageRootNode);
    }
    if (plan != nullptr &&
        nextStageRootNode < plan->nNodes &&
        stageContainsNodeLocal(findStageForNodeLocal(plan, nextStageRootNode), migrationFromNodeIndex)) {
        targetNodes.clear();
        targetNodes.push_back(nextStageRootNode);
    }

    bool queuedAny = false;
    bool localAppliedAny = false;
    for (NnUint targetNode : targetNodes) {
        if (targetNode == 0u) {
            if (executor != nullptr) {
                const bool localOk = executor->applyTransferredKvRow(layerIndex, position, kRow, vRow, rangeStart, rangeLen);
                if (localOk) localAppliedAny = true;
            }
            continue;
        }
        if (exists(layerIndex, position, targetNode)) {
            continue;
        }

        PendingKvTransferItem item;
        item.header.magic = LLM_KV_TRANSFER_MAGIC;
        item.header.version = LLM_KV_TRANSFER_VERSION;
        item.header.layerIndex = layerIndex;
        item.header.position = position;
        item.header.kvDim = (NnUint)kRow.size();
        item.header.fromNodeIndex = migrationFromNodeIndex;
        item.header.targetNodeIndex = targetNode;
        item.header.rangeStart = rangeStart;
        item.header.rangeLen = rangeLen;
        item.kRow = kRow;
        item.vRow = vRow;
        pendingKvTransfers.push_back(std::move(item));
        queuedAny = true;
    }

    if (localAppliedAny) return KV_TRANSFER_SUBMIT_OK;
    if (!queuedAny) return KV_TRANSFER_SUBMIT_DUP_LAYER_PENDING;
    return KV_TRANSFER_SUBMIT_OK;
}

bool RootLlmInference::submitBoundaryKvTransfer(NnUint layerIndex, NnUint position, const std::vector<float> &kRow, const std::vector<float> &vRow) {
    return submitBoundaryKvTransferDetailed(layerIndex, position, kRow, vRow) == KV_TRANSFER_SUBMIT_OK;
}

bool RootLlmInference::collectSourceStageKvTransfers(
    NnUint endPos,
    NnUint *exportedRows,
    NnUint *queuedRows,
    uint64_t *sourceTransferBytes) {
    if (exportedRows != nullptr) *exportedRows = 0u;
    if (queuedRows != nullptr) *queuedRows = 0u;
    if (sourceTransferBytes != nullptr) *sourceTransferBytes = 0u;
    if (header == nullptr || migrationLayers.empty()) return false;

    const NnUint kvDim = header->kvDim;
    if (kvDim == 0u) return false;

    const NnStageConfig *sourceStage = findStageForNodeLocal(plan, migrationFromNodeIndex);
    if (sourceStage == nullptr) return false;
    const std::vector<NnUint> sourceNodes = stageNodeListLocal(sourceStage);
    if (sourceNodes.empty()) return false;

    struct RowPair {
        std::vector<float> k;
        std::vector<float> v;
        NnUint sources = 0u;
    };
    std::map<std::pair<NnUint, NnUint>, RowPair> rows;
    uint64_t rxBytes = 0u;
    NnUint exported = 0u;

    auto mergeRow = [&](NnUint sourceNode, NnUint layer, NnUint pos, const std::vector<float> &kRow, const std::vector<float> &vRow) {
        if (kRow.size() != kvDim || vRow.size() != kvDim) return;
        RowPair &slot = rows[std::make_pair(layer, pos)];
        if (slot.k.empty()) {
            slot.k.assign(kvDim, 0.0f);
            slot.v.assign(kvDim, 0.0f);
        }
        for (NnUint i = 0u; i < kvDim; ++i) {
            if (kRow[i] != 0.0f) slot.k[i] = kRow[i];
            if (vRow[i] != 0.0f) slot.v[i] = vRow[i];
        }
        slot.sources += 1u;
        exported += 1u;
        rxBytes += (uint64_t)sizeof(LlmKvTransferHeader) + (uint64_t)kRow.size() * sizeof(float) + (uint64_t)vRow.size() * sizeof(float);
        std::printf("🧩 [kv-export-merge] sourceNode=%u layer=%u pos=%u kvDim=%u\n",
            (unsigned)sourceNode,
            (unsigned)layer,
            (unsigned)pos,
            (unsigned)kvDim);
        std::fflush(stdout);
    };

    bool requestedRemote = false;
    const NnUint requestId = nextKvExportRequestId++;
    LlmKvExportRequestHeader req{};
    req.magic = LLM_KV_EXPORT_REQUEST_MAGIC;
    req.version = LLM_KV_EXPORT_REQUEST_VERSION;
    req.requestId = requestId;
    req.fromNodeIndex = 0xFFFFFFFFu;
    req.targetStageRootNodeIndex = nextStageRootNode;
    req.endPosition = endPos;
    req.layerCount = (NnUint)migrationLayers.size();
    req.kvDim = kvDim;
    req.rangeStart = 0u;
    req.rangeLen = 0u;

    for (NnUint node : sourceNodes) {
        if (node == 0u) continue;
        const int socketIndex = (network != nullptr) ? network->getSocketIndexForNode(node, 0u) : -1;
        if (socketIndex < 0) {
            std::printf("⚠️  [kv-export-request] missing socket for source node=%u\n", (unsigned)node);
            std::fflush(stdout);
            continue;
        }
        LlmControlPacket ctrl = controlPacket;
        ctrl.flags = LLM_CTRL_HAS_KV_EXPORT_REQUEST | LLM_CTRL_CONTROL_ONLY;
        ctrl.batchSize = 1u;
        ctrl.planCmdSeq = 0u;
        logRootControlSend(ctrl);
        network->write((NnUint)socketIndex, &ctrl, sizeof(ctrl));
        network->write((NnUint)socketIndex, &req, sizeof(req));
        for (NnUint layer : migrationLayers) {
            network->write((NnUint)socketIndex, &layer, sizeof(layer));
        }
        requestedRemote = true;
        std::printf("🧩 [kv-export-request] requestId=%u sourceNode=%u layers=%u posRange=[0,%u]\n",
            (unsigned)requestId,
            (unsigned)node,
            (unsigned)migrationLayers.size(),
            (unsigned)endPos);
        std::fflush(stdout);
    }

    if (stageContainsNodeLocal(sourceStage, 0u) && executor != nullptr) {
        for (NnUint layer : migrationLayers) {
            for (NnUint pos = 0u; pos <= endPos; ++pos) {
                std::vector<float> kRow;
                std::vector<float> vRow;
                if (executor->exportLayerKvRow(layer, pos, kvDim, kRow, vRow)) {
                    mergeRow(0u, layer, pos, kRow, vRow);
                }
            }
        }
    }

    if (requestedRemote && network != nullptr) {
        for (NnUint node : sourceNodes) {
            if (node == 0u) continue;
            const int socketIndex = network->getSocketIndexForNode(node, 0u);
            if (socketIndex < 0) continue;

            LlmKvExportResponseHeader resp{};
            network->read((NnUint)socketIndex, &resp, sizeof(resp));
            if (resp.magic != LLM_KV_EXPORT_RESPONSE_MAGIC ||
                resp.version != LLM_KV_EXPORT_RESPONSE_VERSION ||
                resp.requestId != requestId ||
                resp.fromNodeIndex != node) {
                std::printf("⚠️  [kv-export-response] bad header from node=%u magic=0x%08x version=%u request=%u rows=%u\n",
                    (unsigned)node,
                    (unsigned)resp.magic,
                    (unsigned)resp.version,
                    (unsigned)resp.requestId,
                    (unsigned)resp.rowCount);
                std::fflush(stdout);
                continue;
            }
            rxBytes += sizeof(resp);
            for (NnUint i = 0u; i < resp.rowCount; ++i) {
                LlmKvTransferHeader hdr{};
                network->read((NnUint)socketIndex, &hdr, sizeof(hdr));
                if (hdr.magic != LLM_KV_TRANSFER_MAGIC ||
                    hdr.version != LLM_KV_TRANSFER_VERSION ||
                    hdr.kvDim != kvDim ||
                    hdr.rangeLen != 0u ||
                    hdr.fromNodeIndex != node) {
                    if (hdr.kvDim > 0u) {
                        std::vector<float> discard((size_t)hdr.kvDim * 2u);
                        network->read((NnUint)socketIndex, discard.data(), discard.size() * sizeof(float));
                    }
                    continue;
                }
                std::vector<float> kRow(kvDim);
                std::vector<float> vRow(kvDim);
                network->read((NnUint)socketIndex, kRow.data(), kRow.size() * sizeof(float));
                network->read((NnUint)socketIndex, vRow.data(), vRow.size() * sizeof(float));
                mergeRow(node, hdr.layerIndex, hdr.position, kRow, vRow);
            }
            std::printf("🧩 [kv-export-response] requestId=%u sourceNode=%u rowCount=%u exportedRows=%u\n",
                (unsigned)requestId,
                (unsigned)node,
                (unsigned)resp.rowCount,
                (unsigned)resp.exportedRows);
            std::fflush(stdout);
        }
    }

    NnUint queued = 0u;
    for (auto &entry : rows) {
        const NnUint layer = entry.first.first;
        const NnUint pos = entry.first.second;
        if (submitBoundaryKvTransfer(layer, pos, entry.second.k, entry.second.v)) {
            queued += 1u;
        }
    }

    if (exportedRows != nullptr) *exportedRows = exported;
    if (queuedRows != nullptr) *queuedRows = queued;
    if (sourceTransferBytes != nullptr) *sourceTransferBytes = rxBytes;
    return queued > 0u;
}

bool RootLlmInference::collectHeadKvTransfers(
    const PlanCommand &cmd,
    NnUint endPos,
    NnUint *exportedRows,
    NnUint *queuedRows,
    uint64_t *sourceTransferBytes) {
    if (exportedRows != nullptr) *exportedRows = 0u;
    if (queuedRows != nullptr) *queuedRows = 0u;
    if (sourceTransferBytes != nullptr) *sourceTransferBytes = 0u;
    if (header == nullptr || plan == nullptr || executor == nullptr) return false;
    if (header->headDim == 0u || header->kvDim == 0u) return false;

    NnUint stageIndex = 0u;
    NnUint fromNode = 0u;
    NnUint toNode = 0u;
    NnUint rangeHeadStart = 0u;
    NnUint rangeHeadLen = 0u;
    if (headRecoveryRangeCached) {
        stageIndex = headRecoveryStageIndex;
        fromNode = headRecoveryFromNode;
        toNode = headRecoveryToNode;
        rangeHeadStart = headRecoveryRangeHeadStart;
        rangeHeadLen = headRecoveryRangeHeadLen;
    } else if (!getHeadMigrationTargetRangeLocal(plan, cmd, false, &stageIndex, &fromNode, &toNode, &rangeHeadStart, &rangeHeadLen) &&
               !getHeadMigrationTargetRangeLocal(plan, cmd, true, &stageIndex, &fromNode, &toNode, &rangeHeadStart, &rangeHeadLen)) {
        std::printf("⚠️  [head-migrate] cannot derive transfer range for command seq=%u\n", (unsigned)cmd.seq);
        std::fflush(stdout);
        return false;
    }
    const NnStageConfig *stage = (stageIndex < plan->nStages) ? &plan->stages[stageIndex] : nullptr;
    if (stage == nullptr || stage->endLayer <= stage->startLayer) return false;

    const NnUint rangeStart = rangeHeadStart * header->headDim;
    const NnUint rangeLen = rangeHeadLen * header->headDim;
    if (rangeLen == 0u || rangeStart >= header->kvDim || rangeStart + rangeLen > header->kvDim) return false;

    migrationFromNodeIndex = fromNode;
    nextStageRootNode = toNode;

    std::vector<NnUint> layers;
    layers.reserve(stage->endLayer - stage->startLayer);
    for (NnUint layer = stage->startLayer; layer < stage->endLayer; ++layer) {
        layers.push_back(layer);
    }

    struct PartialRow {
        NnUint layer = 0u;
        NnUint pos = 0u;
        std::vector<float> k;
        std::vector<float> v;
    };
    std::vector<PartialRow> rows;
    rows.reserve((size_t)layers.size() * ((size_t)endPos + 1u));

    auto appendPartial = [&](NnUint layer, NnUint pos, const std::vector<float> &fullK, const std::vector<float> &fullV) {
        if (fullK.size() < rangeStart + rangeLen || fullV.size() < rangeStart + rangeLen) return;
        PartialRow row;
        row.layer = layer;
        row.pos = pos;
        row.k.assign(fullK.begin() + rangeStart, fullK.begin() + rangeStart + rangeLen);
        row.v.assign(fullV.begin() + rangeStart, fullV.begin() + rangeStart + rangeLen);
        rows.push_back(std::move(row));
    };

    uint64_t rxBytes = 0u;
    NnUint exported = 0u;

    if (fromNode == 0u) {
        for (NnUint layer : layers) {
            for (NnUint pos = 0u; pos <= endPos; ++pos) {
                std::vector<float> kRow;
                std::vector<float> vRow;
                if (executor->exportLayerKvRow(layer, pos, header->kvDim, kRow, vRow, rangeStart, rangeLen)) {
                    if (kRow.size() == rangeLen && vRow.size() == rangeLen) {
                        PartialRow row;
                        row.layer = layer;
                        row.pos = pos;
                        row.k = std::move(kRow);
                        row.v = std::move(vRow);
                        rows.push_back(std::move(row));
                    } else {
                        appendPartial(layer, pos, kRow, vRow);
                    }
                    exported += 1u;
                    rxBytes += (uint64_t)sizeof(LlmKvTransferHeader) + (uint64_t)rangeLen * sizeof(float) * 2ull;
                }
            }
        }
    } else if (network != nullptr) {
        const int socketIndex = network->getSocketIndexForNode(fromNode, 0u);
        if (socketIndex < 0) return false;
        const NnUint requestId = nextKvExportRequestId++;
        LlmKvExportRequestHeader req{};
        req.magic = LLM_KV_EXPORT_REQUEST_MAGIC;
        req.version = LLM_KV_EXPORT_REQUEST_VERSION;
        req.requestId = requestId;
        req.fromNodeIndex = fromNode;
        req.targetStageRootNodeIndex = toNode;
        req.endPosition = endPos;
        req.layerCount = (NnUint)layers.size();
        req.kvDim = header->kvDim;
        req.rangeStart = rangeStart;
        req.rangeLen = rangeLen;

        LlmControlPacket ctrl = controlPacket;
        ctrl.flags = LLM_CTRL_HAS_KV_EXPORT_REQUEST | LLM_CTRL_CONTROL_ONLY;
        ctrl.batchSize = 1u;
        ctrl.planCmdSeq = 0u;
        logRootControlSend(ctrl);
        network->write((NnUint)socketIndex, &ctrl, sizeof(ctrl));
        network->write((NnUint)socketIndex, &req, sizeof(req));
        for (NnUint layer : layers) {
            network->write((NnUint)socketIndex, &layer, sizeof(layer));
        }
        std::printf("🧩 [head-kv-export-request] requestId=%u sourceNode=%u targetNode=%u layers=%u posRange=[0,%u] range=[%u,%u)\n",
            (unsigned)requestId,
            (unsigned)fromNode,
            (unsigned)toNode,
            (unsigned)layers.size(),
            (unsigned)endPos,
            (unsigned)rangeStart,
            (unsigned)(rangeStart + rangeLen));
        std::fflush(stdout);

        LlmKvExportResponseHeader resp{};
        network->read((NnUint)socketIndex, &resp, sizeof(resp));
        if (resp.magic != LLM_KV_EXPORT_RESPONSE_MAGIC ||
            resp.version != LLM_KV_EXPORT_RESPONSE_VERSION ||
            resp.requestId != requestId ||
            resp.fromNodeIndex != fromNode) {
            std::printf("⚠️  [head-kv-export-response] bad header from node=%u magic=0x%08x version=%u request=%u rows=%u\n",
                (unsigned)fromNode,
                (unsigned)resp.magic,
                (unsigned)resp.version,
                (unsigned)resp.requestId,
                (unsigned)resp.rowCount);
            std::fflush(stdout);
            return false;
        }
        rxBytes += sizeof(resp);
        for (NnUint i = 0u; i < resp.rowCount; ++i) {
            LlmKvTransferHeader hdr{};
            network->read((NnUint)socketIndex, &hdr, sizeof(hdr));
            const bool okHeader =
                hdr.magic == LLM_KV_TRANSFER_MAGIC &&
                hdr.version == LLM_KV_TRANSFER_VERSION &&
                hdr.kvDim == rangeLen &&
                hdr.rangeStart == rangeStart &&
                hdr.rangeLen == rangeLen &&
                hdr.fromNodeIndex == fromNode;
            if (!okHeader) {
                if (hdr.kvDim > 0u) {
                    std::vector<float> discard((size_t)hdr.kvDim * 2u);
                    network->read((NnUint)socketIndex, discard.data(), discard.size() * sizeof(float));
                }
                continue;
            }
            PartialRow row;
            row.layer = hdr.layerIndex;
            row.pos = hdr.position;
            row.k.resize(rangeLen);
            row.v.resize(rangeLen);
            network->read((NnUint)socketIndex, row.k.data(), row.k.size() * sizeof(float));
            network->read((NnUint)socketIndex, row.v.data(), row.v.size() * sizeof(float));
            rows.push_back(std::move(row));
            exported += 1u;
            rxBytes += (uint64_t)sizeof(LlmKvTransferHeader) + (uint64_t)rangeLen * sizeof(float) * 2ull;
        }
    }

    NnUint queued = 0u;
    for (const PartialRow &row : rows) {
        if (submitBoundaryKvTransferDetailed(row.layer, row.pos, row.k, row.v, rangeStart, rangeLen) == KV_TRANSFER_SUBMIT_OK) {
            queued += 1u;
        }
    }

    if (exportedRows != nullptr) *exportedRows = exported;
    if (queuedRows != nullptr) *queuedRows = queued;
    if (sourceTransferBytes != nullptr) *sourceTransferBytes = rxBytes;
    std::printf("🧩 [head-kv-transfer-collect] stage=%u route=%u->%u heads=[%u,%u) dimRange=[%u,%u) exported=%u queued=%u posRange=[0,%u]\n",
        (unsigned)stageIndex,
        (unsigned)fromNode,
        (unsigned)toNode,
        (unsigned)rangeHeadStart,
        (unsigned)(rangeHeadStart + rangeHeadLen),
        (unsigned)rangeStart,
        (unsigned)(rangeStart + rangeLen),
        (unsigned)exported,
        (unsigned)queued,
        (unsigned)endPos);
    std::fflush(stdout);
    return queued > 0u;
}

bool RootLlmInference::flushPendingKvTransfersControlOnly(uint64_t *targetTransferBytes) {
    if (targetTransferBytes != nullptr) *targetTransferBytes = 0u;
    if (network == nullptr) return false;

    bool sentAny = false;
    uint64_t bytes = 0u;
    for (int guard = 0; guard < 10000; ++guard) {
        bool sendKvTransfer = false;
        std::vector<PendingKvTransferItem> kvTransfers;
        {
            std::lock_guard<std::mutex> lk(kvTransferMutex);
            if (!pendingKvTransfers.empty() && !waitingKvAck) {
                sendKvTransfer = true;
                kvTransfers = pendingKvTransfers;
                pendingKvTransfers.clear();
                waitingKvAck = true;
                waitingKvAckExpectedCount = (NnUint)kvTransfers.size();
                waitingKvAckLayers.clear();
                waitingKvAckReceivedCount = 0u;
                waitingKvAckReceivedLayers.clear();
                waitingKvAckNodes.clear();
                waitingKvAckNodeExpected.clear();
                waitingKvAckNodeReceived.clear();
                for (const auto &it : kvTransfers) {
                    appendUniqueLayer(waitingKvAckLayers, it.header.layerIndex);
                    const NnUint targetNode = it.header.targetNodeIndex;
                    if (targetNode == 0u) continue;
                    auto itNode = std::find(waitingKvAckNodes.begin(), waitingKvAckNodes.end(), targetNode);
                    if (itNode == waitingKvAckNodes.end()) {
                        waitingKvAckNodes.push_back(targetNode);
                        waitingKvAckNodeExpected.push_back(1u);
                        waitingKvAckNodeReceived.push_back(0u);
                    } else {
                        const size_t idx = (size_t)(itNode - waitingKvAckNodes.begin());
                        waitingKvAckNodeExpected[idx] += 1u;
                    }
                }
                std::sort(waitingKvAckLayers.begin(), waitingKvAckLayers.end());
            }
        }

        if (sendKvTransfer) {
            LlmControlPacket out = controlPacket;
            out.flags = LLM_CTRL_HAS_KV_TRANSFER | LLM_CTRL_CONTROL_ONLY;
            out.batchSize = 1u;
            out.planCmdSeq = 0u;
            logRootControlSend(out);
            network->writeAll(&out, sizeof(LlmControlPacket));

            LlmKvTransferBatchHeader bh{};
            bh.magic = LLM_KV_TRANSFER_BATCH_MAGIC;
            bh.version = LLM_KV_TRANSFER_BATCH_VERSION;
            bh.count = (NnUint)kvTransfers.size();
            bh.reserved = 0u;
            network->writeAll(&bh, sizeof(bh));
            bytes += (uint64_t)network->nSockets * ((uint64_t)sizeof(LlmControlPacket) + (uint64_t)sizeof(bh));
            for (const auto &it : kvTransfers) {
                network->writeAll(&it.header, sizeof(LlmKvTransferHeader));
                network->writeAll(it.kRow.data(), it.kRow.size() * sizeof(float));
                network->writeAll(it.vRow.data(), it.vRow.size() * sizeof(float));
                bytes += (uint64_t)network->nSockets *
                    ((uint64_t)sizeof(LlmKvTransferHeader) + (uint64_t)it.kRow.size() * sizeof(float) + (uint64_t)it.vRow.size() * sizeof(float));
            }
            sentAny = true;
        }

        if (!waitingKvAck) break;

        const NnStageConfig *targetStage = findStageForNodeLocal(plan, nextStageRootNode);
        NnUint validAckCountInc = 0u;
        std::vector<NnUint> ackedLayersInc;
        std::vector<NnUint> nodeAckInc(waitingKvAckNodes.size(), 0u);
        NnUint ackPos = 0u;
        bool ackPosSet = false;

        for (size_t ni = 0u; ni < waitingKvAckNodes.size(); ++ni) {
            if (ni >= waitingKvAckNodeExpected.size() || ni >= waitingKvAckNodeReceived.size()) continue;
            if (waitingKvAckNodeReceived[ni] >= waitingKvAckNodeExpected[ni]) continue;
            const NnUint ackNode = waitingKvAckNodes[ni];
            const int socketIndex = network->getSocketIndexForNode(ackNode, 0u);
            if (socketIndex < 0) continue;

            LlmKvAckBatchHeader abh{};
            network->read((NnUint)socketIndex, &abh, sizeof(abh));
            if (abh.magic != LLM_KV_ACK_BATCH_MAGIC || abh.version != LLM_KV_ACK_BATCH_VERSION) {
                std::printf("⚠️  [kv-migrate] unexpected packet on ack socket node=%u (magic=0x%08x ver=%u), skip\n",
                    (unsigned)ackNode,
                    (unsigned)abh.magic,
                    (unsigned)abh.version);
                std::fflush(stdout);
                continue;
            }

            for (NnUint i = 0u; i < abh.count; ++i) {
                LlmKvAckPacket ack{};
                network->read((NnUint)socketIndex, &ack, sizeof(ack));
                if (ack.magic != LLM_KV_ACK_MAGIC || ack.version != LLM_KV_ACK_VERSION) continue;
                if (ack.toNodeIndex != 0u) continue;
                if (ack.fromNodeIndex != ackNode) continue;
                if (targetStage != nullptr) {
                    if (!stageContainsNodeLocal(targetStage, ack.fromNodeIndex)) continue;
                } else {
                    if (ack.fromNodeIndex != nextStageRootNode) continue;
                }
                validAckCountInc += 1u;
                nodeAckInc[ni] += 1u;
                appendUniqueLayer(ackedLayersInc, ack.layerIndex);
                if (!ackPosSet) {
                    ackPos = ack.position;
                    ackPosSet = true;
                }
                std::printf("🔁 [kv-migrate] ack received layer=%u pos=%u fromNode=%u\n",
                    (unsigned)ack.layerIndex,
                    (unsigned)ack.position,
                    (unsigned)ack.fromNodeIndex);
            }
        }

        if (validAckCountInc > 0u) {
            std::sort(ackedLayersInc.begin(), ackedLayersInc.end());
            std::lock_guard<std::mutex> lk(kvTransferMutex);
            waitingKvAckReceivedCount += validAckCountInc;
            for (size_t ni = 0u; ni < nodeAckInc.size() && ni < waitingKvAckNodeReceived.size(); ++ni) {
                waitingKvAckNodeReceived[ni] += nodeAckInc[ni];
            }
            for (NnUint layer : ackedLayersInc) {
                appendUniqueLayer(waitingKvAckReceivedLayers, layer);
            }
            std::sort(waitingKvAckReceivedLayers.begin(), waitingKvAckReceivedLayers.end());

            const bool layersMatched = !waitingKvAckLayers.empty() && waitingKvAckReceivedLayers == waitingKvAckLayers;
            const bool rowsMatched = (waitingKvAckExpectedCount == 0u) || (waitingKvAckReceivedCount >= waitingKvAckExpectedCount);
            if (layersMatched && rowsMatched) {
                waitingKvAck = false;
                waitingKvAckExpectedCount = 0u;
                waitingKvAckReceivedCount = 0u;
                waitingKvAckNodes.clear();
                waitingKvAckNodeExpected.clear();
                waitingKvAckNodeReceived.clear();
                pendingLayerSwitchLayers = waitingKvAckReceivedLayers;
                waitingKvAckReceivedLayers.clear();
                migrationAckSeen = true;
                migrationAckPos = ackPosSet ? (int)ackPos : migrationAckPos;
                migrationAckLayer = !pendingLayerSwitchLayers.empty() ? (int)pendingLayerSwitchLayers.back() : migrationAckLayer;
                std::printf("🔁 [kv-migrate] ack batch complete layers=%u -> switch ownership\n",
                    (unsigned)pendingLayerSwitchLayers.size());
                std::fflush(stdout);
            }
        }
        if (!waitingKvAck) break;
    }

    if (targetTransferBytes != nullptr) *targetTransferBytes = bytes;
    return sentAny && !waitingKvAck;
}

bool RootLlmInference::sendPendingLayerSwitchControlOnly() {
    if (network == nullptr) return false;
    std::vector<NnUint> switchLayers;
    {
        std::lock_guard<std::mutex> lk(kvTransferMutex);
        if (pendingLayerSwitchLayers.empty()) return false;
        switchLayers = pendingLayerSwitchLayers;
        pendingLayerSwitchLayers.clear();
        if (executor != nullptr) {
            const NnUint selfNodeIndex = 0u;
            const bool selfIsSource = areNodesInSameStageLocal(plan, selfNodeIndex, migrationFromNodeIndex);
            const bool selfIsTarget = areNodesInSameStageLocal(plan, selfNodeIndex, nextStageRootNode);
            for (NnUint layer : switchLayers) {
                if (selfIsSource) executor->setPrimaryLayerEnabled(layer, false);
                if (selfIsTarget) executor->setRedundantLayerEnabled(layer, true);
            }
            maybeEnableShiftedPpStartForSourceStage(
                switchLayers,
                migrationFromNodeIndex,
                nextStageRootNode,
                selfIsSource);
        }
    }

    LlmControlPacket out = controlPacket;
    out.flags = LLM_CTRL_HAS_LAYER_SWITCH | LLM_CTRL_CONTROL_ONLY;
    out.batchSize = 1u;
    out.planCmdSeq = 0u;
    logRootControlSend(out);
    network->writeAll(&out, sizeof(LlmControlPacket));

    LlmLayerSwitchBatchHeader sbh{};
    sbh.magic = LLM_LAYER_SWITCH_BATCH_MAGIC;
    sbh.version = LLM_LAYER_SWITCH_BATCH_VERSION;
    sbh.count = (NnUint)switchLayers.size();
    sbh.reserved = 0u;
    network->writeAll(&sbh, sizeof(sbh));
    for (NnUint layer : switchLayers) {
        LlmLayerSwitchPacket switchPkt{};
        switchPkt.magic = LLM_LAYER_SWITCH_MAGIC;
        switchPkt.version = LLM_LAYER_SWITCH_VERSION;
        switchPkt.boundaryLayer = layer;
        switchPkt.fromNodeIndex = migrationFromNodeIndex;
        switchPkt.toNodeIndex = nextStageRootNode;
        switchPkt.reserved0 = 0u;
        switchPkt.reserved1 = 0u;
        switchPkt.reserved2 = 0u;
        network->writeAll(&switchPkt, sizeof(switchPkt));
    }
    return true;
}

void RootLlmInference::collectProfilePackets() {
    if (!profileEnabled) return;

    lastPerf.clear();
    lastPerf.reserve((network != nullptr ? network->nSockets : 0u) + 1u);

    LlmPerfPacket rootPacket{};
    rootPacket.position = controlPacket.position;
    rootPacket.batchSize = controlPacket.batchSize;
    rootPacket.nodeIndex = 0;
    rootPacket.stageIndex = getStageIndexForNode(plan, 0);
    rootPacket.execUs = executor != nullptr ? executor->getTotalTime(STEP_EXECUTE_OP) : 0u;
    rootPacket.syncUs = executor != nullptr ? executor->getTotalTime(STEP_SYNC_NODES) : 0u;
    rootPacket.bubbleUs = (NnUint)std::min<unsigned long long>(lastBubbleShadowStats.elapsedUs, (unsigned long long)UINT32_MAX);
    rootPacket.bubbleSegments = lastBubbleShadowStats.segmentsVisited;
    rootPacket.bubbleOps = lastBubbleShadowStats.opStepsExecuted;
    rootPacket.bubbleSkippedSyncs = lastBubbleShadowStats.skippedSyncSteps;
    lastPerf.push_back(rootPacket);

    if (network != nullptr && network->nSockets > 0) {
        const NnUint nWorkers = network->nSockets;
        const size_t base = lastPerf.size();
        lastPerf.resize(base + nWorkers);

        std::vector<NnSocketIo> ios(nWorkers);
        for (NnUint i = 0; i < nWorkers; ++i) {
            ios[i].socketIndex = i;
            ios[i].data = &lastPerf[base + i];
            ios[i].size = sizeof(LlmPerfPacket);
        }
        network->readMany(nWorkers, &ios[0]);
    }
}

bool RootLlmInference::replayHistoryForMigrationRecompute(NnUint endPos, double *recomputeMs, uint64_t *recomputeTokens) {
    if (recomputeMs != nullptr) *recomputeMs = 0.0;
    if (recomputeTokens != nullptr) *recomputeTokens = 0u;
    if (executor == nullptr || execution == nullptr || header == nullptr) return false;
    if (endPos >= header->seqLen) return false;
    if (tokenHistory.size() <= endPos) return false;
    for (NnUint pos = 0u; pos <= endPos; ++pos) {
        if (tokenHistory[pos] < 0) return false;
    }

    const NnUint savedBatchSize = controlPacket.batchSize;
    const NnUint savedPosition = controlPacket.position;
    std::vector<float> savedTokens(savedBatchSize);
    for (NnUint i = 0u; i < savedBatchSize; ++i) {
        savedTokens[i] = tokenPipe[i];
    }
    std::vector<float> savedLogits;
    if (logitsPipe != nullptr && header->vocabSize > 0u && savedBatchSize > 0u) {
        savedLogits.assign(
            logitsPipe,
            logitsPipe + ((size_t)savedBatchSize * (size_t)header->vocabSize));
    }

    auto t0 = std::chrono::steady_clock::now();
    const NnUint replayBatch = 1u;
    for (NnUint pos = 0u; pos <= endPos; ++pos) {
        execution->setBatchSize(replayBatch);
        controlPacket.batchSize = replayBatch;
        controlPacket.position = pos;
        positionPipe[0] = (float)pos;
        tokenPipe[0] = (float)tokenHistory[pos];

        if (network != nullptr) {
            LlmControlPacket out = controlPacket;
            out.flags = controlPacket.flags & LLM_CTRL_PROFILE;
            out.planCmdSeq = 0u;
            logRootControlSend(out);
            network->writeAll(&out, sizeof(LlmControlPacket));
        }
        executor->forward();
        collectProfilePackets();
    }
    auto t1 = std::chrono::steady_clock::now();

    execution->setBatchSize(savedBatchSize);
    controlPacket.batchSize = savedBatchSize;
    controlPacket.position = savedPosition;
    for (NnUint i = 0u; i < savedBatchSize; ++i) {
        positionPipe[i] = (float)(savedPosition + i);
        tokenPipe[i] = savedTokens[i];
    }
    if (!savedLogits.empty() && logitsPipe != nullptr) {
        std::memcpy(logitsPipe, savedLogits.data(), savedLogits.size() * sizeof(float));
    }

    if (recomputeMs != nullptr) {
        *recomputeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    if (recomputeTokens != nullptr) {
        *recomputeTokens = (uint64_t)(endPos + 1u) * (uint64_t)migrationLayers.size();
    }
    return true;
}

bool RootLlmInference::replayHistoryForHeadMigrationRecompute(
    const PlanCommand &cmd,
    NnUint endPos,
    double *recomputeMs,
    uint64_t *recomputeTokens) {
    if (recomputeMs != nullptr) *recomputeMs = 0.0;
    if (recomputeTokens != nullptr) *recomputeTokens = 0u;
    NnUint stageIndex = 0u;
    NnUint fromNode = 0u;
    NnUint toNode = 0u;
    NnUint rangeHeadStart = 0u;
    NnUint rangeHeadLen = 0u;
    if (headRecoveryRangeCached) {
        stageIndex = headRecoveryStageIndex;
        fromNode = headRecoveryFromNode;
        toNode = headRecoveryToNode;
        rangeHeadStart = headRecoveryRangeHeadStart;
        rangeHeadLen = headRecoveryRangeHeadLen;
    } else if (!getHeadMigrationTargetRangeLocal(plan, cmd, false, &stageIndex, &fromNode, &toNode, &rangeHeadStart, &rangeHeadLen) &&
               !getHeadMigrationTargetRangeLocal(plan, cmd, true, &stageIndex, &fromNode, &toNode, &rangeHeadStart, &rangeHeadLen)) {
        return false;
    }
    if (stageIndex >= plan->nStages) return false;
    const NnStageConfig *stage = &plan->stages[stageIndex];
    if (stage->endLayer <= stage->startLayer) return false;

    const bool ok = replayHistoryForMigrationRecompute(endPos, recomputeMs, recomputeTokens);
    if (ok && recomputeTokens != nullptr) {
        *recomputeTokens = (uint64_t)(endPos + 1u) *
            (uint64_t)(stage->endLayer - stage->startLayer) *
            (uint64_t)rangeHeadLen;
    }
    return ok;
}

void RootLlmInference::emitPpMigrationRecoverEvent(
    bool applySuccess,
    uint64_t stateTransferBytes,
    uint64_t recomputeTokensOrLayers,
    double statePrepareMs,
    double recoverMs,
    double stallMs,
    const std::string &fallbackReason) {
    const EdgeVisorAblationConfig &ablationCfg = getEdgeVisorAblationConfig();
    EdgeVisorAblationEvent ev;
    ev.eventId = "pp_migration_recover";
    ev.triggerPos = (asyncKvCollectPos >= 0) ? (NnUint)asyncKvCollectPos : 0xFFFFFFFFu;
    ev.triggerLayer = (asyncKvCollectLayer >= 0) ? (NnUint)asyncKvCollectLayer : 0xFFFFFFFFu;
    const NnStageConfig *fromStage = findStageForNodeLocal(plan, migrationFromNodeIndex);
    ev.affectedStage = (fromStage != nullptr) ? fromStage->stageIndex : 0xFFFFFFFFu;
    ev.fromNode = migrationFromNodeIndex;
    ev.toNode = nextStageRootNode;
    ev.selectedPolicy = std::string("shadow_") + toString(ablationCfg.shadowKvMode);
    ev.tStatePrepareMs = statePrepareMs;
    ev.tRecoverMs = recoverMs;
    ev.stallTimeMs = stallMs;
    ev.stateTransferBytes = stateTransferBytes;
    ev.recomputeTokensOrLayers = recomputeTokensOrLayers;
    ev.bindingUpdateCount = (uint64_t)migrationLayers.size();
    ev.physicalDeviceGroup = "pp_route";
    ev.logicalGroup = "pp_stage_boundary";
    ev.fallbackReason = fallbackReason;
    ev.applySuccess = applySuccess;
    edgevisorAblationLogEvent(ev);
}

bool RootLlmInference::recoverHeadMigrationNoShadow(const PlanCommand &cmd, NnUint triggerPos) {
    if (header == nullptr || plan == nullptr) return false;
    const EdgeVisorAblationConfig &ablationCfg = getEdgeVisorAblationConfig();
    if (ablationCfg.shadowKvMode == ShadowKvMode::ENABLED) return false;
    if (ablationCfg.shadowKvMode != ShadowKvMode::DISABLED_TRANSFER &&
        ablationCfg.shadowKvMode != ShadowKvMode::DISABLED_RECOMPUTE) return false;

    NnUint stageIndex = 0u;
    NnUint fromNode = 0u;
    NnUint toNode = 0u;
    NnUint rangeHeadStart = 0u;
    NnUint rangeHeadLen = 0u;
    if (headRecoveryRangeCached) {
        stageIndex = headRecoveryStageIndex;
        fromNode = headRecoveryFromNode;
        toNode = headRecoveryToNode;
        rangeHeadStart = headRecoveryRangeHeadStart;
        rangeHeadLen = headRecoveryRangeHeadLen;
    } else if (!getHeadMigrationTargetRangeLocal(plan, cmd, false, &stageIndex, &fromNode, &toNode, &rangeHeadStart, &rangeHeadLen) &&
               !getHeadMigrationTargetRangeLocal(plan, cmd, true, &stageIndex, &fromNode, &toNode, &rangeHeadStart, &rangeHeadLen)) {
        return false;
    }
    if (stageIndex >= plan->nStages) return false;
    const NnStageConfig *stage = &plan->stages[stageIndex];
    const NnUint endPos = std::min(triggerPos, (header->seqLen > 0u) ? (header->seqLen - 1u) : 0u);

    auto tStall0 = std::chrono::steady_clock::now();
    bool applyOk = false;
    uint64_t stateBytes = 0u;
    uint64_t recomputeUnits = 0u;
    double statePrepareMs = 0.0;
    double recoverMs = 0.0;
    std::string fallbackReason;

    if (ablationCfg.shadowKvMode == ShadowKvMode::DISABLED_TRANSFER) {
        auto t0 = std::chrono::steady_clock::now();
        NnUint exported = 0u;
        NnUint queued = 0u;
        uint64_t sourceBytes = 0u;
        const bool collected = collectHeadKvTransfers(cmd, endPos, &exported, &queued, &sourceBytes);
        auto t1 = std::chrono::steady_clock::now();
        uint64_t targetBytes = 0u;
        const bool localTarget = (toNode == 0u);
        const bool transferred = collected && (localTarget || flushPendingKvTransfersControlOnly(&targetBytes));
        auto t2 = std::chrono::steady_clock::now();
        statePrepareMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        recoverMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        stateBytes = sourceBytes + targetBytes;
        applyOk = transferred;
        fallbackReason = "shadow_kv_disabled_head_real_transfer";
        std::printf("🧩 [head-migrate] real transfer stage=%u route=%u->%u heads=[%u,%u) exported=%u queued=%u sourceBytes=%llu targetBytes=%llu status=%s\n",
            (unsigned)stageIndex,
            (unsigned)fromNode,
            (unsigned)toNode,
            (unsigned)rangeHeadStart,
            (unsigned)(rangeHeadStart + rangeHeadLen),
            (unsigned)exported,
            (unsigned)queued,
            (unsigned long long)sourceBytes,
            (unsigned long long)targetBytes,
            applyOk ? "ok" : "fail");
        std::fflush(stdout);
    } else if (ablationCfg.shadowKvMode == ShadowKvMode::DISABLED_RECOMPUTE) {
        auto t0 = std::chrono::steady_clock::now();
        double replayMs = 0.0;
        uint64_t replayUnits = 0u;
        const bool recomputed = replayHistoryForHeadMigrationRecompute(cmd, endPos, &replayMs, &replayUnits);
        auto t1 = std::chrono::steady_clock::now();
        statePrepareMs = replayMs;
        recoverMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        recomputeUnits = replayUnits;
        applyOk = recomputed;
        fallbackReason = recomputed ? "shadow_kv_disabled_head_real_recompute_replay" : "head_recompute_replay_failed_missing_token_history";
    }

    auto tStall1 = std::chrono::steady_clock::now();
    const double stallMs = std::chrono::duration<double, std::milli>(tStall1 - tStall0).count();

    EdgeVisorAblationEvent ev;
    ev.eventId = "head_migration_recover";
    ev.triggerPos = endPos;
    ev.triggerLayer = (stage != nullptr && stage->endLayer > stage->startLayer) ? (stage->endLayer - 1u) : 0xFFFFFFFFu;
    ev.affectedStage = stageIndex;
    ev.fromNode = fromNode;
    ev.toNode = toNode;
    ev.selectedPolicy = std::string("shadow_") + toString(ablationCfg.shadowKvMode);
    ev.tStatePrepareMs = statePrepareMs;
    ev.tRecoverMs = recoverMs;
    ev.stallTimeMs = stallMs;
    ev.stateTransferBytes = stateBytes;
    ev.recomputeTokensOrLayers = recomputeUnits;
    ev.bindingUpdateCount = rangeHeadLen;
    ev.physicalDeviceGroup = "intra_stage_head_route";
    ev.logicalGroup = "intra_stage_head_migration";
    ev.fallbackReason = fallbackReason;
    ev.applySuccess = applyOk;
    edgevisorAblationLogEvent(ev);

    std::printf("🧩 [head-migrate] recover mode=%s status=%s stallMs=%.3f stateBytes=%llu recomputeUnits=%llu stage=%u route=%u->%u heads=[%u,%u) posRange=[0,%u]\n",
        toString(ablationCfg.shadowKvMode),
        applyOk ? "ok" : "fail",
        stallMs,
        (unsigned long long)stateBytes,
        (unsigned long long)recomputeUnits,
        (unsigned)stageIndex,
        (unsigned)fromNode,
        (unsigned)toNode,
        (unsigned)rangeHeadStart,
        (unsigned)(rangeHeadStart + rangeHeadLen),
        (unsigned)endPos);
    std::fflush(stdout);
    return applyOk;
}

void RootLlmInference::forward() {
    lastBubbleShadowStats = {};
    bool sendKvTransfer = false;
    bool sendLayerSwitch = false;
    std::vector<PendingKvTransferItem> kvTransfers;
    std::vector<NnUint> switchLayers;

    {
        std::lock_guard<std::mutex> lk(kvTransferMutex);
        if (!pendingKvTransfers.empty() && !waitingKvAck) {
            sendKvTransfer = true;
            kvTransfers = pendingKvTransfers;
            pendingKvTransfers.clear();
            waitingKvAck = true;
            waitingKvAckExpectedCount = (NnUint)kvTransfers.size();
            waitingKvAckLayers.clear();
            waitingKvAckReceivedCount = 0u;
            waitingKvAckReceivedLayers.clear();
            waitingKvAckNodes.clear();
            waitingKvAckNodeExpected.clear();
            waitingKvAckNodeReceived.clear();
            for (const auto &it : kvTransfers) {
                appendUniqueLayer(waitingKvAckLayers, it.header.layerIndex);
                const NnUint targetNode = it.header.targetNodeIndex;
                if (targetNode == 0u) continue;
                auto itNode = std::find(waitingKvAckNodes.begin(), waitingKvAckNodes.end(), targetNode);
                if (itNode == waitingKvAckNodes.end()) {
                    waitingKvAckNodes.push_back(targetNode);
                    waitingKvAckNodeExpected.push_back(1u);
                    waitingKvAckNodeReceived.push_back(0u);
                } else {
                    const size_t idx = (size_t)(itNode - waitingKvAckNodes.begin());
                    waitingKvAckNodeExpected[idx] += 1u;
                }
            }
            std::sort(waitingKvAckLayers.begin(), waitingKvAckLayers.end());
        }
        if (!pendingLayerSwitchLayers.empty()) {
            sendLayerSwitch = true;
            switchLayers = pendingLayerSwitchLayers;
            pendingLayerSwitchLayers.clear();
            if (executor != nullptr) {
                const NnUint selfNodeIndex = 0u;
                const bool selfIsSource = areNodesInSameStageLocal(plan, selfNodeIndex, migrationFromNodeIndex);
                const bool selfIsTarget = areNodesInSameStageLocal(plan, selfNodeIndex, nextStageRootNode);
                for (NnUint layer : switchLayers) {
                    if (selfIsSource) {
                        executor->setPrimaryLayerEnabled(layer, false);
                    }
                    if (selfIsTarget) {
                        executor->setRedundantLayerEnabled(layer, true);
                    }
                }
                maybeEnableShiftedPpStartForSourceStage(
                    switchLayers,
                    migrationFromNodeIndex,
                    nextStageRootNode,
                    selfIsSource);
            }
        }
    }

    if (network != nullptr) {
        // Prefer external PlanCommand; env fallback is applied once at startup.
        const PlanCommandSnapshot snap = planCommandCache().load();
        const NnUint planCmdSeqLo = (NnUint)(snap.cacheSeq & 0xFFFFFFFFu);

        LlmControlPacket out = controlPacket;
        out.flags = controlPacket.flags;
        out.planCmdSeq = planCmdSeqLo;

        const bool planChanged = (planCmdSeqLo != lastPlanCmdSeqSent);
        if (planChanged) {
            out.flags |= LLM_CTRL_HAS_PLAN_CMD;
            lastPlanCmdSeqSent = planCmdSeqLo;
        } else {
            out.flags &= ~LLM_CTRL_HAS_PLAN_CMD;
        }
        if (sendKvTransfer) out.flags |= LLM_CTRL_HAS_KV_TRANSFER;
        if (sendLayerSwitch) out.flags |= LLM_CTRL_HAS_LAYER_SWITCH;

        logRootControlSend(out);
        network->writeAll(&out, sizeof(LlmControlPacket));
        if (planChanged) {
            network->writeAll(&snap.cmd, sizeof(PlanCommand));
        }
        if (sendKvTransfer) {
            LlmKvTransferBatchHeader bh{};
            bh.magic = LLM_KV_TRANSFER_BATCH_MAGIC;
            bh.version = LLM_KV_TRANSFER_BATCH_VERSION;
            bh.count = (NnUint)kvTransfers.size();
            bh.reserved = 0u;
            network->writeAll(&bh, sizeof(bh));
            for (const auto &it : kvTransfers) {
                network->writeAll(&it.header, sizeof(LlmKvTransferHeader));
                network->writeAll(it.kRow.data(), it.kRow.size() * sizeof(float));
                network->writeAll(it.vRow.data(), it.vRow.size() * sizeof(float));
            }
        }
        if (sendLayerSwitch) {
            LlmLayerSwitchBatchHeader sbh{};
            sbh.magic = LLM_LAYER_SWITCH_BATCH_MAGIC;
            sbh.version = LLM_LAYER_SWITCH_BATCH_VERSION;
            sbh.count = (NnUint)switchLayers.size();
            sbh.reserved = 0u;
            network->writeAll(&sbh, sizeof(sbh));
            for (NnUint layer : switchLayers) {
                LlmLayerSwitchPacket switchPkt{};
                switchPkt.magic = LLM_LAYER_SWITCH_MAGIC;
                switchPkt.version = LLM_LAYER_SWITCH_VERSION;
                switchPkt.boundaryLayer = layer;
                switchPkt.fromNodeIndex = migrationFromNodeIndex;
                switchPkt.toNodeIndex = nextStageRootNode;
                switchPkt.reserved0 = 0u;
                switchPkt.reserved1 = 0u;
                switchPkt.reserved2 = 0u;
                network->writeAll(&switchPkt, sizeof(switchPkt));
            }
        }
    }

    {
        const PlanCommandSnapshot preSnap = planCommandCache().load();
        const PlanCommand &pc = preSnap.cmd;
        const bool hasHeadPayload =
            (pc.cmdKind == PLAN_CMD_KIND_HEAD || pc.cmdKind == PLAN_CMD_KIND_BOTH) ||
            (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves > 0u);
        const bool shouldCache =
            getAllowNoShadowHeadMigration() &&
            isValidPlanCommandHeader(pc) &&
            pc.mode != PLAN_CMD_MODE_NONE &&
            hasHeadPayload &&
            preSnap.cacheSeq != lastHeadRecoveryPlanCacheSeqApplied &&
            preSnap.cacheSeq != headRecoveryRangeCacheSeq;
        if (shouldCache) {
            NnUint stageIndex = 0u;
            NnUint fromNode = 0u;
            NnUint toNode = 0u;
            NnUint rangeHeadStart = 0u;
            NnUint rangeHeadLen = 0u;
            if (getHeadMigrationTargetRangeLocal(plan, pc, false, &stageIndex, &fromNode, &toNode, &rangeHeadStart, &rangeHeadLen)) {
                headRecoveryRangeCacheSeq = preSnap.cacheSeq;
                headRecoveryRangeCached = true;
                headRecoveryPlanAfterApply = false;
                headRecoveryStageIndex = stageIndex;
                headRecoveryFromNode = fromNode;
                headRecoveryToNode = toNode;
                headRecoveryRangeHeadStart = rangeHeadStart;
                headRecoveryRangeHeadLen = rangeHeadLen;
                std::printf("🧩 [head-migrate] cached pre-apply range cacheSeq=%llu stage=%u route=%u->%u heads=[%u,%u)\n",
                    (unsigned long long)preSnap.cacheSeq,
                    (unsigned)stageIndex,
                    (unsigned)fromNode,
                    (unsigned)toNode,
                    (unsigned)rangeHeadStart,
                    (unsigned)(rangeHeadStart + rangeHeadLen));
                std::fflush(stdout);
            }
        }
    }
    executor->forward();
    lastBubbleShadowStats = maybeRunBubbleShadowKv(executor, "root", 0u, controlPacket.position, controlPacket.batchSize);

    if (network != nullptr && waitingKvAck) {
        const NnStageConfig *targetStage = findStageForNodeLocal(plan, nextStageRootNode);
        NnUint validAckCountInc = 0u;
        std::vector<NnUint> ackedLayersInc;
        std::vector<NnUint> nodeAckInc(waitingKvAckNodes.size(), 0u);
        NnUint ackPos = 0u;
        bool ackPosSet = false;

        for (size_t ni = 0u; ni < waitingKvAckNodes.size(); ++ni) {
            if (ni >= waitingKvAckNodeExpected.size() || ni >= waitingKvAckNodeReceived.size()) continue;
            if (waitingKvAckNodeReceived[ni] >= waitingKvAckNodeExpected[ni]) continue;
            const NnUint ackNode = waitingKvAckNodes[ni];
            const int socketIndex = network->getSocketIndexForNode(ackNode, 0u);
            if (socketIndex < 0) continue;

            LlmKvAckBatchHeader abh{};
            if (!network->tryReadWithMaxAttempts((NnUint)socketIndex, &abh, sizeof(abh), 1ul)) continue;
            if (abh.magic != LLM_KV_ACK_BATCH_MAGIC || abh.version != LLM_KV_ACK_BATCH_VERSION) {
                std::printf("⚠️  [kv-migrate] unexpected packet on ack socket node=%u (magic=0x%08x ver=%u), skip\n",
                    (unsigned)ackNode,
                    (unsigned)abh.magic,
                    (unsigned)abh.version);
                std::fflush(stdout);
                continue;
            }

            for (NnUint i = 0u; i < abh.count; ++i) {
                LlmKvAckPacket ack{};
                network->read((NnUint)socketIndex, &ack, sizeof(ack));
                if (ack.magic != LLM_KV_ACK_MAGIC || ack.version != LLM_KV_ACK_VERSION) continue;
                if (ack.toNodeIndex != 0u) continue;
                if (ack.fromNodeIndex != ackNode) continue;
                if (targetStage != nullptr) {
                    if (!stageContainsNodeLocal(targetStage, ack.fromNodeIndex)) continue;
                } else {
                    if (ack.fromNodeIndex != nextStageRootNode) continue;
                }
                validAckCountInc += 1u;
                nodeAckInc[ni] += 1u;
                appendUniqueLayer(ackedLayersInc, ack.layerIndex);
                if (!ackPosSet) {
                    ackPos = ack.position;
                    ackPosSet = true;
                }
                std::printf("🔁 [kv-migrate] ack received layer=%u pos=%u fromNode=%u\n",
                    (unsigned)ack.layerIndex,
                    (unsigned)ack.position,
                    (unsigned)ack.fromNodeIndex);
            }
        }

        if (validAckCountInc > 0u) {
            std::sort(ackedLayersInc.begin(), ackedLayersInc.end());
            std::lock_guard<std::mutex> lk(kvTransferMutex);
            waitingKvAckReceivedCount += validAckCountInc;
            for (size_t ni = 0u; ni < nodeAckInc.size() && ni < waitingKvAckNodeReceived.size(); ++ni) {
                waitingKvAckNodeReceived[ni] += nodeAckInc[ni];
            }
            for (NnUint layer : ackedLayersInc) {
                appendUniqueLayer(waitingKvAckReceivedLayers, layer);
            }
            std::sort(waitingKvAckReceivedLayers.begin(), waitingKvAckReceivedLayers.end());

            const bool layersMatched = !waitingKvAckLayers.empty() && waitingKvAckReceivedLayers == waitingKvAckLayers;
            const bool rowsMatched = (waitingKvAckExpectedCount == 0u) || (waitingKvAckReceivedCount >= waitingKvAckExpectedCount);
            if (layersMatched && rowsMatched) {
                waitingKvAck = false;
                waitingKvAckExpectedCount = 0u;
                waitingKvAckReceivedCount = 0u;
                waitingKvAckNodes.clear();
                waitingKvAckNodeExpected.clear();
                waitingKvAckNodeReceived.clear();
                pendingLayerSwitchLayers = waitingKvAckReceivedLayers;
                waitingKvAckReceivedLayers.clear();
                migrationAckSeen = true;
                migrationAckPos = ackPosSet ? (int)ackPos : migrationAckPos;
                migrationAckLayer = !pendingLayerSwitchLayers.empty() ? (int)pendingLayerSwitchLayers.back() : migrationAckLayer;
                std::printf("🔁 [kv-migrate] ack batch complete layers=%u -> switch ownership\n",
                    (unsigned)pendingLayerSwitchLayers.size());
                std::fflush(stdout);
            } else if (layersMatched && !rowsMatched) {
                std::printf("⚠️  [kv-migrate] ack rows incomplete got=%u expected=%u (waiting)\n",
                    (unsigned)waitingKvAckReceivedCount,
                    (unsigned)waitingKvAckExpectedCount);
                std::fflush(stdout);
            }
        }
    }

    if (ppMigrationEnabled) {
        const PlanCommandSnapshot snap = planCommandCache().load();
        const PlanCommand &pc = snap.cmd;
        const bool isNewPpCommand = (snap.cacheSeq != lastPpPlanCacheSeqApplied);
        const bool isPpMigrationCmd =
            (pc.cmdKind == 0u) &&
            ((pc.version == DLLAMA_PLAN_CMD_VERSION_V1) ||
             (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves == 0u));
        if (isNewPpCommand && isValidPlanCommandHeader(pc) && pc.mode != PLAN_CMD_MODE_NONE && isPpMigrationCmd) {
            if (waitingKvAck) {
                std::printf("⚠️  [kv-migrate] defer new pp command cacheSeq=%llu: waiting previous ack\n",
                    (unsigned long long)snap.cacheSeq);
                std::fflush(stdout);
            } else {
            const int cmdLayerCount = (pc.reserved0 > 0u) ? (int)pc.reserved0 : 1;
            migrationLayerCount = (cmdLayerCount < 1) ? 1 : cmdLayerCount;
            migrationLayerListPinnedByEnv = false;
            migrationBatchSubmitted = false;
            migrationExportRequested = false;
            lastMigrationStateTransferBytes = 0u;
            lastMigrationExportedRows = 0u;
            asyncKvCollectPos = -1;
            asyncKvCollectLayer = -1;

            NnUint routeFromNode = migrationFromNodeIndex;
            NnUint routeToNode = nextStageRootNode;
            if (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves > 0u) {
                routeFromNode = pc.moves[0].fromNodeIndex;
                routeToNode = pc.moves[0].toNodeIndex;
                if (pc.nMoves > 1u) {
                    std::printf("⚠️  [kv-migrate] PlanCommand has %u moves; PP route uses the first one (%u->%u)\n",
                        (unsigned)pc.nMoves,
                        (unsigned)routeFromNode,
                        (unsigned)routeToNode);
                    std::fflush(stdout);
                }
            } else {
                routeFromNode = pc.fromNodeIndex;
                routeToNode = pc.toNodeIndex;
            }

            const NnStageConfig *fromStage = findStageForNodeLocal(plan, routeFromNode);
            const NnStageConfig *toStage = findStageForNodeLocal(plan, routeToNode);
            const bool validRoute =
                (routeToNode != (NnUint)-1) &&
                (fromStage != nullptr) &&
                (toStage != nullptr) &&
                (fromStage->stageIndex != toStage->stageIndex) &&
                (fromStage->endLayer > fromStage->startLayer);

            const NnStageConfig *effectiveFromStage = nullptr;
            const NnStageConfig *effectiveToStage = nullptr;
            NnUint effectiveFromNode = migrationFromNodeIndex;
            NnUint effectiveToNode = nextStageRootNode;

            if (!validRoute) {
                std::printf("⚠️  [kv-migrate] ignore plan route from=%u to=%u (invalid stage route), fallback to current route %u->%u\n",
                    (unsigned)routeFromNode,
                    (unsigned)routeToNode,
                    (unsigned)migrationFromNodeIndex,
                    (unsigned)nextStageRootNode);
                std::fflush(stdout);
                effectiveFromStage = findStageForNodeLocal(plan, migrationFromNodeIndex);
                effectiveToStage = findStageForNodeLocal(plan, nextStageRootNode);
            } else {
                migrationFromNodeIndex = routeFromNode;
                nextStageRootNode = routeToNode;
                kvAckSocketIndex = (network != nullptr) ? network->getSocketIndexForNode(nextStageRootNode, 0u) : -1;
                effectiveFromNode = migrationFromNodeIndex;
                effectiveToNode = nextStageRootNode;
                effectiveFromStage = fromStage;
                effectiveToStage = toStage;
            }

            const bool canApplyRoute =
                (effectiveToNode != (NnUint)-1) &&
                (effectiveFromStage != nullptr) &&
                (effectiveToStage != nullptr) &&
                (effectiveFromStage->stageIndex != effectiveToStage->stageIndex) &&
                (effectiveFromStage->endLayer > effectiveFromStage->startLayer);

            if (canApplyRoute) {
                migrationStageStartLayer = (int)effectiveFromStage->startLayer;
                migrationStageEndLayer = (int)effectiveFromStage->endLayer;

                const bool migrateToPrev = effectiveToStage->stageIndex < effectiveFromStage->stageIndex;
                boundaryLayerForMigration = migrateToPrev
                    ? (int)effectiveFromStage->startLayer
                    : (int)(effectiveFromStage->endLayer - 1u);

                migrationLayers.clear();
                if (migrateToPrev) {
                    const int maxLayer = (int)effectiveFromStage->endLayer - 1;
                    for (int i = 0; i < migrationLayerCount; ++i) {
                        const int layer = boundaryLayerForMigration + i;
                        if (layer > maxLayer) break;
                        appendUniqueLayer(migrationLayers, (NnUint)layer);
                    }
                } else {
                    const int minLayer = (int)effectiveFromStage->startLayer;
                    for (int i = 0; i < migrationLayerCount; ++i) {
                        const int layer = boundaryLayerForMigration - i;
                        if (layer < minLayer) break;
                        appendUniqueLayer(migrationLayers, (NnUint)layer);
                    }
                }
                std::sort(migrationLayers.begin(), migrationLayers.end());
                std::ostringstream layersOss;
                for (size_t i = 0; i < migrationLayers.size(); ++i) {
                    if (i > 0) layersOss << ",";
                    layersOss << migrationLayers[i];
                }
                std::printf("🧭 [kv-migrate] apply pp command cacheSeq=%llu layerCount=%d route=%u->%u layers=[%s]\n",
                    (unsigned long long)snap.cacheSeq,
                    migrationLayerCount,
                    (unsigned)effectiveFromNode,
                    (unsigned)effectiveToNode,
                    layersOss.str().c_str());
                std::fflush(stdout);
            } else {
                std::printf("⚠️  [kv-migrate] no effective stage route, keep existing layers=[%zu]\n",
                    migrationLayers.size());
                std::fflush(stdout);
            }

            int triggerPos = -1;
            if (pc.mode == PLAN_CMD_MODE_EXACT && pc.triggerPos != 0xFFFFFFFFu) {
                triggerPos = (int)pc.triggerPos;
            } else {
                NnUint nextPos = controlPacket.position + ((controlPacket.batchSize > 0u) ? controlPacket.batchSize : 1u);
                if (header != nullptr && header->seqLen > 0u && nextPos >= header->seqLen) {
                    nextPos = header->seqLen - 1u;
                }
                triggerPos = (int)nextPos;
            }

            if (triggerPos >= 0 && !migrationLayers.empty() && nextStageRootNode != (NnUint)-1) {
                asyncKvCollectPos = triggerPos;
                asyncKvCollectLayer = !migrationLayers.empty() ? (int)migrationLayers.back() : -1;
                std::printf("🧭 [kv-migrate] auto collect armed layer=%d pos=%d mode=%u route=%u->%u layers=%zu\n",
                    asyncKvCollectLayer,
                    asyncKvCollectPos,
                    (unsigned)pc.mode,
                    (unsigned)migrationFromNodeIndex,
                    (unsigned)nextStageRootNode,
                    migrationLayers.size());
                std::fflush(stdout);
            }
            {
                const EdgeVisorAblationConfig &ablationCfg = getEdgeVisorAblationConfig();
                EdgeVisorAblationEvent ev;
                ev.eventId = "pp_migration_apply";
                ev.triggerPos = (triggerPos >= 0) ? (NnUint)triggerPos : 0xFFFFFFFFu;
                ev.triggerLayer = (boundaryLayerForMigration >= 0) ? (NnUint)boundaryLayerForMigration : 0xFFFFFFFFu;
                ev.affectedStage = (effectiveFromStage != nullptr) ? effectiveFromStage->stageIndex : 0xFFFFFFFFu;
                ev.fromNode = effectiveFromNode;
                ev.toNode = effectiveToNode;
                ev.selectedPolicy = std::string("vg_") + toString(ablationCfg.vgMode);
                ev.bindingUpdateCount = (uint64_t)migrationLayers.size();
                ev.vgMappingBefore = std::string("vg_") + toString(ablationCfg.vgMode);
                ev.vgMappingAfter = std::string("vg_") + toString(ablationCfg.vgMode);
                ev.physicalDeviceGroup = "pp_route";
                ev.logicalGroup = "pp_stage_boundary";
                ev.applySuccess = canApplyRoute && !migrationLayers.empty();
                if (!validRoute) {
                    ev.rejectedMoves = 1u;
                    ev.fallbackCount = 1u;
                    ev.fallbackReason = "invalid_pp_route_fallback_to_current_route";
                } else if (ablationCfg.vgMode != VgMode::ENABLED) {
                    ev.fallbackReason = std::string("vg_mode_") + toString(ablationCfg.vgMode);
                }
                edgevisorAblationLogEvent(ev);
            }
                lastPpPlanCacheSeqApplied = snap.cacheSeq;
            }
        }
    }

    {
        const PlanCommandSnapshot snap = planCommandCache().load();
        const PlanCommand &pc = snap.cmd;
        const bool isNewHeadCommand = (snap.cacheSeq != lastHeadRecoveryPlanCacheSeqApplied);
        const bool hasHeadPayload =
            (pc.cmdKind == PLAN_CMD_KIND_HEAD || pc.cmdKind == PLAN_CMD_KIND_BOTH) ||
            (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves > 0u);
        if (isNewHeadCommand &&
            isValidPlanCommandHeader(pc) &&
            pc.mode != PLAN_CMD_MODE_NONE &&
            hasHeadPayload &&
            getAllowNoShadowHeadMigration()) {
            NnUint stageIndex = 0u;
            NnUint fromNode = 0u;
            NnUint toNode = 0u;
            NnUint rangeHeadStart = 0u;
            NnUint rangeHeadLen = 0u;
            bool haveRange = false;
            if (headRecoveryRangeCached && headRecoveryRangeCacheSeq == snap.cacheSeq) {
                stageIndex = headRecoveryStageIndex;
                fromNode = headRecoveryFromNode;
                toNode = headRecoveryToNode;
                rangeHeadStart = headRecoveryRangeHeadStart;
                rangeHeadLen = headRecoveryRangeHeadLen;
                haveRange = true;
            } else {
                haveRange =
                    getHeadMigrationTargetRangeLocal(plan, pc, false, &stageIndex, &fromNode, &toNode, &rangeHeadStart, &rangeHeadLen) ||
                    getHeadMigrationTargetRangeLocal(plan, pc, true, &stageIndex, &fromNode, &toNode, &rangeHeadStart, &rangeHeadLen);
                if (haveRange) {
                    headRecoveryRangeCacheSeq = snap.cacheSeq;
                    headRecoveryRangeCached = true;
                    headRecoveryPlanAfterApply = false;
                    headRecoveryStageIndex = stageIndex;
                    headRecoveryFromNode = fromNode;
                    headRecoveryToNode = toNode;
                    headRecoveryRangeHeadStart = rangeHeadStart;
                    headRecoveryRangeHeadLen = rangeHeadLen;
                }
            }
            if (haveRange && !headRecoveryPlanAfterApply) {
                const bool exactReady =
                    pc.mode != PLAN_CMD_MODE_EXACT ||
                    (pc.triggerPos != 0xFFFFFFFFu && controlPacket.position >= pc.triggerPos);
                if (exactReady) {
                    NnUint recoverPos = controlPacket.position;
                    if (pc.mode == PLAN_CMD_MODE_EXACT && pc.triggerPos != 0xFFFFFFFFu) {
                        recoverPos = std::max(controlPacket.position, pc.triggerPos);
                    }
                    if (header != nullptr && header->seqLen > 0u && recoverPos >= header->seqLen) {
                        recoverPos = header->seqLen - 1u;
                    }
                    const bool recovered = recoverHeadMigrationNoShadow(pc, recoverPos);
                    if (recovered) {
                        lastHeadRecoveryPlanCacheSeqApplied = snap.cacheSeq;
                        planCommandCache().consumeIfCacheSeq(snap.cacheSeq);
                        headRecoveryRangeCached = false;
                    } else {
                        std::printf("⚠️  [head-migrate] recovery failed for cacheSeq=%llu; keep command pending for retry\n",
                            (unsigned long long)snap.cacheSeq);
                        std::fflush(stdout);
                    }
                }
            } else if (haveRange && headRecoveryPlanAfterApply) {
                planCommandCache().consumeIfCacheSeq(snap.cacheSeq);
                headRecoveryRangeCached = false;
                headRecoveryPlanAfterApply = false;
            }
        }
    }

    if (!migrationBatchSubmitted && asyncKvCollectPos >= 0 &&
        controlPacket.position == (NnUint)asyncKvCollectPos &&
        !migrationLayers.empty() && executor != nullptr && header != nullptr) {
        const NnUint endPos = std::min((NnUint)asyncKvCollectPos, (header->seqLen > 0u) ? (header->seqLen - 1u) : 0u);
        const EdgeVisorAblationConfig &ablationCfg = getEdgeVisorAblationConfig();
        auto tStall0 = std::chrono::steady_clock::now();
        bool applyOk = false;
        uint64_t stateBytes = 0u;
        uint64_t recomputeUnits = 0u;
        double statePrepareMs = 0.0;
        double recoverMs = 0.0;
        std::string fallbackReason;
        const NnStageConfig *fromStage = findStageForNodeLocal(plan, migrationFromNodeIndex);
        const NnStageConfig *toStage = findStageForNodeLocal(plan, nextStageRootNode);
        const bool migrateToLaterStage =
            fromStage != nullptr && toStage != nullptr &&
            toStage->stageIndex > fromStage->stageIndex;

        if (ablationCfg.shadowKvMode == ShadowKvMode::ENABLED) {
            auto t0 = std::chrono::steady_clock::now();
            if (migrateToLaterStage) {
                NnUint exported = 0u;
                NnUint queued = 0u;
                uint64_t sourceBytes = 0u;
                const bool collected = collectSourceStageKvTransfers(endPos, &exported, &queued, &sourceBytes);
                auto t1 = std::chrono::steady_clock::now();
                uint64_t targetBytes = 0u;
                const bool transferred = collected && flushPendingKvTransfersControlOnly(&targetBytes);
                const bool switched = transferred && sendPendingLayerSwitchControlOnly();
                auto t2 = std::chrono::steady_clock::now();
                statePrepareMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                recoverMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
                stateBytes = sourceBytes + targetBytes;
                lastMigrationStateTransferBytes = stateBytes;
                lastMigrationExportedRows = exported;
                applyOk = switched;
                fallbackReason = "shadow_kv_forward_real_transfer";
                std::printf("🧩 [kv-migrate] shadow forward transfer exported=%u queued=%u layers=%u posRange=[0,%u] sourceBytes=%llu targetBytes=%llu targetRoot=%u status=%s\n",
                    (unsigned)exported,
                    (unsigned)queued,
                    (unsigned)migrationLayers.size(),
                    (unsigned)endPos,
                    (unsigned long long)sourceBytes,
                    (unsigned long long)targetBytes,
                    (unsigned)nextStageRootNode,
                    applyOk ? "ok" : "fail");
            } else {
                {
                    std::lock_guard<std::mutex> lk(kvTransferMutex);
                    pendingLayerSwitchLayers = migrationLayers;
                }
                applyOk = sendPendingLayerSwitchControlOnly();
                auto t1 = std::chrono::steady_clock::now();
                recoverMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                fallbackReason = "shadow_kv_precomputed_redundant_state";
            }
            migrationBatchSubmitted = applyOk;
        } else if (ablationCfg.shadowKvMode == ShadowKvMode::DISABLED_TRANSFER) {
            auto t0 = std::chrono::steady_clock::now();
            NnUint exported = 0u;
            NnUint queued = 0u;
            uint64_t sourceBytes = 0u;
            const bool collected = collectSourceStageKvTransfers(endPos, &exported, &queued, &sourceBytes);
            auto t1 = std::chrono::steady_clock::now();
            uint64_t targetBytes = 0u;
            const bool transferred = collected && flushPendingKvTransfersControlOnly(&targetBytes);
            const bool switched = transferred && sendPendingLayerSwitchControlOnly();
            auto t2 = std::chrono::steady_clock::now();
            statePrepareMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            recoverMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
            stateBytes = sourceBytes + targetBytes;
            lastMigrationStateTransferBytes = stateBytes;
            lastMigrationExportedRows = exported;
            applyOk = switched;
            fallbackReason = "shadow_kv_disabled_real_transfer";
            migrationBatchSubmitted = applyOk;
            std::printf("🧩 [kv-migrate] real transfer prepared exported=%u queued=%u layers=%u posRange=[0,%u] sourceBytes=%llu targetBytes=%llu targetRoot=%u status=%s\n",
                (unsigned)exported,
                (unsigned)queued,
                (unsigned)migrationLayers.size(),
                (unsigned)endPos,
                (unsigned long long)sourceBytes,
                (unsigned long long)targetBytes,
                (unsigned)nextStageRootNode,
                applyOk ? "ok" : "fail");
        } else if (ablationCfg.shadowKvMode == ShadowKvMode::DISABLED_RECOMPUTE) {
            auto t0 = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lk(kvTransferMutex);
                pendingLayerSwitchLayers = migrationLayers;
            }
            const bool switched = sendPendingLayerSwitchControlOnly();
            auto tSwitch = std::chrono::steady_clock::now();
            double replayMs = 0.0;
            uint64_t replayUnits = 0u;
            const bool recomputed = switched && replayHistoryForMigrationRecompute(endPos, &replayMs, &replayUnits);
            auto t1 = std::chrono::steady_clock::now();
            recomputeUnits = replayUnits;
            statePrepareMs = replayMs;
            recoverMs = std::chrono::duration<double, std::milli>(tSwitch - t0).count();
            fallbackReason = recomputed ? "shadow_kv_disabled_real_recompute_replay" : "real_recompute_replay_failed_missing_token_history";
            applyOk = switched && recomputed;
            migrationBatchSubmitted = applyOk;
        }

        auto tStall1 = std::chrono::steady_clock::now();
        const double stallMs = std::chrono::duration<double, std::milli>(tStall1 - tStall0).count();
        emitPpMigrationRecoverEvent(applyOk, stateBytes, recomputeUnits, statePrepareMs, recoverMs, stallMs, fallbackReason);
        if (applyOk) {
            migrationAckSeen = true;
            migrationAckPos = (int)endPos;
            migrationAckLayer = !migrationLayers.empty() ? (int)migrationLayers.back() : migrationAckLayer;
        }
        std::printf("🧩 [kv-migrate] recover mode=%s status=%s stallMs=%.3f stateBytes=%llu recomputeUnits=%llu layers=%u posRange=[0,%u]\n",
            toString(ablationCfg.shadowKvMode),
            applyOk ? "ok" : "fail",
            stallMs,
            (unsigned long long)stateBytes,
            (unsigned long long)recomputeUnits,
            (unsigned)migrationLayers.size(),
            (unsigned)endPos);
        std::fflush(stdout);
    }

    if (hasAsyncKvCollector() && execution->batchSize > 0u) {
        const NnUint kvDim = header->kvDim;
        const NnUint seqLen = header->seqLen;
        const NnUint basePos = controlPacket.position;
        std::lock_guard<std::mutex> lock(asyncKvRowsMutex);
        for (NnUint b = 0u; b < execution->batchSize; ++b) {
            const NnUint pos = basePos + b;
            if (pos >= seqLen) continue;
            if (asyncKvCollectPos >= 0 && pos != (NnUint)asyncKvCollectPos) continue;

            RootKvAggRowPacket packet;
            packet.layerIndex = (NnUint)asyncKvCollectLayer;
            packet.position = pos;
            packet.kRow.resize(kvDim);
            packet.vRow.resize(kvDim);

            const float *srcK = kvAggKPipe + (NnSize)b * (NnSize)kvDim;
            const float *srcV = kvAggVPipe + (NnSize)b * (NnSize)kvDim;
            std::memcpy(packet.kRow.data(), srcK, (size_t)kvDim * sizeof(float));
            std::memcpy(packet.vRow.data(), srcV, (size_t)kvDim * sizeof(float));
            asyncKvRows.push_back(std::move(packet));
        }

        const size_t maxQueued = 64u;
        while (asyncKvRows.size() > maxQueued) {
            asyncKvRows.pop_front();
        }
    }

    collectProfilePackets();
}

void RootLlmInference::finish() {
    if (network != nullptr) {
        controlPacket.batchSize = 0;
        // Stop packet: position is not meaningful when batchSize==0.
        // Set to 0 to avoid confusing logs / downstream checks.
        controlPacket.position = 0;
        LlmControlPacket out = controlPacket;
        out.flags = controlPacket.flags & LLM_CTRL_PROFILE;
        out.flags &= ~LLM_CTRL_HAS_PLAN_CMD;
        out.planCmdSeq = 0u;
        logRootControlSend(out);
        network->writeAll(&out, sizeof(LlmControlPacket));
    }
}

WorkerLlmInference::WorkerLlmInference(NnNetExecution *execution, NnNetwork *network, NnUint localNodeIndex, NnUint logitsPipeIndex, Sampler *lastStageSampler) {
    this->isFinished = false;
    this->execution = execution;
    this->network = network;
    this->localNodeIndex = localNodeIndex;
    this->logitsPipeIndex = logitsPipeIndex;
    this->lastStageSampler = lastStageSampler;
    this->positionPipe = (float *)execution->pipes[0];
    this->logitsPipe = (logitsPipeIndex != (NnUint)-1) ? (float *)execution->pipes[logitsPipeIndex] : nullptr;
}

void WorkerLlmInference::maybeSendLastStageSampledToken(const NnUnevenPartitionPlan *plan) {
    if (!lastStageSamplingPlanSupported(plan)) return;
    if (network == nullptr || execution == nullptr || logitsPipe == nullptr || lastStageSampler == nullptr) return;
    if (controlPacket.batchSize != 1u) return;
    const NnStageConfig &last = plan->stages[plan->nStages - 1u];
    if (last.rootNodeIndex != localNodeIndex) return;

    NnUint vocabSize = 0u;
    if (plan->vocabSplit.lengths != nullptr) {
        for (NnUint node = 0u; node < plan->nNodes; ++node) vocabSize += plan->vocabSplit.lengths[node];
    }
    if (vocabSize == 0u) return;

    const int localToken = lastStageSampler->sample(logitsPipe);
    if (localToken < 0) return;
    const NnUint token = (NnUint)localToken;
    const float sampledLogit = ((NnUint)localToken < vocabSize) ? logitsPipe[localToken] : 0.0f;

    LlmSampledTokenPacket packet{};
    packet.magic = LLM_SAMPLED_TOKEN_MAGIC;
    packet.version = LLM_SAMPLED_TOKEN_VERSION;
    packet.nodeIndex = localNodeIndex;
    packet.position = controlPacket.position;
    packet.token = token;
    packet.logit = sampledLogit;
    network->write(ROOT_SOCKET_INDEX, &packet, sizeof(packet));
}

bool WorkerLlmInference::tryReadControlPacket() {
    const unsigned long maxAttempts = 10000;
    if (!network->tryReadWithMaxAttempts(ROOT_SOCKET_INDEX, &controlPacket, sizeof(LlmControlPacket), maxAttempts))
        return false;
    if (controlPacket.batchSize == 0) {
        // Stop packet: position is ignored by design.
        printf("📨 [Worker] Recv Control: Batch=0 (stop)\n");
        isFinished = true;
        return true;
    }

    // Optional PlanCommand packet piggybacked after control packet.
    if ((controlPacket.flags & LLM_CTRL_HAS_PLAN_CMD) != 0u) {
        PlanCommand cmd;
        network->read(ROOT_SOCKET_INDEX, &cmd, sizeof(cmd));
        if (cmd.magic == DLLAMA_PLAN_CMD_MAGIC && (cmd.version == DLLAMA_PLAN_CMD_VERSION_V1 || cmd.version == DLLAMA_PLAN_CMD_VERSION_V2)) {
            // Avoid redundant stores when multiple forwards share the same seq.
            if (controlPacket.planCmdSeq != lastPlanCmdSeqRecv) {
                planCommandCache().store(cmd);
                lastPlanCmdSeqRecv = controlPacket.planCmdSeq;
            }
        } else {
            printf("⚠️  [Worker] Bad PlanCommand packet (magic=0x%08x version=%u)\n", cmd.magic, cmd.version);
        }
    }

    if ((controlPacket.flags & LLM_CTRL_HAS_KV_TRANSFER) != 0u) {
        LlmKvTransferBatchHeader bh{};
        network->read(ROOT_SOCKET_INDEX, &bh, sizeof(bh));
        if (bh.magic == LLM_KV_TRANSFER_BATCH_MAGIC && bh.version == LLM_KV_TRANSFER_BATCH_VERSION) {
            for (NnUint i = 0u; i < bh.count; ++i) {
                LlmKvTransferHeader hdr{};
                network->read(ROOT_SOCKET_INDEX, &hdr, sizeof(hdr));
                std::vector<float> kRow;
                std::vector<float> vRow;
                if (hdr.magic == LLM_KV_TRANSFER_MAGIC && hdr.version == LLM_KV_TRANSFER_VERSION && hdr.kvDim > 0u) {
                    kRow.resize(hdr.kvDim);
                    vRow.resize(hdr.kvDim);
                    network->read(ROOT_SOCKET_INDEX, kRow.data(), kRow.size() * sizeof(float));
                    network->read(ROOT_SOCKET_INDEX, vRow.data(), vRow.size() * sizeof(float));
                    if (hdr.targetNodeIndex == localNodeIndex) {
                        PendingKvTransferItem item;
                        item.header = hdr;
                        item.kRow = std::move(kRow);
                        item.vRow = std::move(vRow);
                        pendingKvTransfers.push_back(std::move(item));

                        LlmKvAckPacket ack{};
                        ack.magic = LLM_KV_ACK_MAGIC;
                        ack.version = LLM_KV_ACK_VERSION;
                        ack.layerIndex = hdr.layerIndex;
                        ack.position = hdr.position;
                        ack.fromNodeIndex = localNodeIndex;
                        ack.toNodeIndex = 0u;
                        pendingKvAcks.push_back(ack);
                    }
                } else {
                    if (hdr.kvDim > 0u) {
                        std::vector<float> discard((size_t)hdr.kvDim * 2u);
                        network->read(ROOT_SOCKET_INDEX, discard.data(), discard.size() * sizeof(float));
                    }
                }
            }
        }
    }

    if ((controlPacket.flags & LLM_CTRL_HAS_LAYER_SWITCH) != 0u) {
        LlmLayerSwitchBatchHeader sbh{};
        network->read(ROOT_SOCKET_INDEX, &sbh, sizeof(sbh));
        if (sbh.magic == LLM_LAYER_SWITCH_BATCH_MAGIC && sbh.version == LLM_LAYER_SWITCH_BATCH_VERSION) {
            for (NnUint i = 0u; i < sbh.count; ++i) {
                LlmLayerSwitchPacket pkt{};
                network->read(ROOT_SOCKET_INDEX, &pkt, sizeof(pkt));
                if (pkt.magic == LLM_LAYER_SWITCH_MAGIC && pkt.version == LLM_LAYER_SWITCH_VERSION) {
                    pendingLayerSwitches.push_back(pkt);
                }
            }
        }
    }

    if ((controlPacket.flags & LLM_CTRL_HAS_KV_EXPORT_REQUEST) != 0u) {
        LlmKvExportRequestHeader req{};
        network->read(ROOT_SOCKET_INDEX, &req, sizeof(req));
        std::vector<NnUint> layers;
        if (req.magic == LLM_KV_EXPORT_REQUEST_MAGIC &&
            req.version == LLM_KV_EXPORT_REQUEST_VERSION &&
            req.layerCount > 0u) {
            layers.resize(req.layerCount);
            for (NnUint i = 0u; i < req.layerCount; ++i) {
                network->read(ROOT_SOCKET_INDEX, &layers[i], sizeof(NnUint));
            }
            if (req.fromNodeIndex == localNodeIndex || req.fromNodeIndex == 0xFFFFFFFFu) {
                pendingKvExportRequests.push_back(std::make_pair(req, std::move(layers)));
            }
        } else {
            for (NnUint i = 0u; i < req.layerCount; ++i) {
                NnUint discard = 0u;
                network->read(ROOT_SOCKET_INDEX, &discard, sizeof(discard));
            }
            std::printf("⚠️  [Worker] Bad KV export request (magic=0x%08x version=%u layers=%u)\n",
                (unsigned)req.magic,
                (unsigned)req.version,
                (unsigned)req.layerCount);
            std::fflush(stdout);
        }
    }

    printf("📨 [Worker] Recv Control: Batch=%u, Pos=%u\n", controlPacket.batchSize, controlPacket.position);
    for (NnUint i = 0; i < controlPacket.batchSize; i++)
        positionPipe[i] = (float)(controlPacket.position + i);
    execution->setBatchSize(controlPacket.batchSize);
    return true;
}

bool WorkerLlmInference::consumeLayerSwitch(LlmLayerSwitchPacket &packet) {
    if (pendingLayerSwitches.empty()) return false;
    packet = pendingLayerSwitches.front();
    pendingLayerSwitches.pop_front();
    return true;
}

bool WorkerLlmInference::consumePendingKvTransfer(LlmKvTransferHeader &hdr, std::vector<float> &kRow, std::vector<float> &vRow) {
    if (pendingKvTransfers.empty()) return false;
    PendingKvTransferItem item = std::move(pendingKvTransfers.front());
    pendingKvTransfers.pop_front();
    hdr = item.header;
    kRow = std::move(item.kRow);
    vRow = std::move(item.vRow);
    return true;
}

bool WorkerLlmInference::consumeKvExportRequest(LlmKvExportRequestHeader &hdr, std::vector<NnUint> &layers) {
    if (pendingKvExportRequests.empty()) return false;
    auto item = std::move(pendingKvExportRequests.front());
    pendingKvExportRequests.pop_front();
    hdr = item.first;
    layers = std::move(item.second);
    return true;
}

void WorkerLlmInference::flushKvExportResponse(
    const LlmKvExportRequestHeader &req,
    const std::vector<NnUint> &layers,
    NnUint kvDim,
    NnExecutor *executor) {
    if (network == nullptr || executor == nullptr || kvDim == 0u) return;
    const NnUint payloadStart = req.rangeLen != 0u ? req.rangeStart : 0u;
    const NnUint payloadLen = req.rangeLen != 0u ? req.rangeLen : kvDim;
    if (payloadStart >= kvDim || payloadStart + payloadLen > kvDim || payloadLen == 0u) return;

    struct ExportedRow {
        LlmKvTransferHeader header;
        std::vector<float> kRow;
        std::vector<float> vRow;
    };
    std::vector<ExportedRow> rows;
    const NnUint endPos = req.endPosition;
    for (NnUint layer : layers) {
        for (NnUint pos = 0u; pos <= endPos; ++pos) {
            std::vector<float> kRow;
            std::vector<float> vRow;
            if (!executor->exportLayerKvRow(layer, pos, kvDim, kRow, vRow, payloadStart, req.rangeLen != 0u ? payloadLen : 0u)) {
                continue;
            }
            ExportedRow row;
            row.header.magic = LLM_KV_TRANSFER_MAGIC;
            row.header.version = LLM_KV_TRANSFER_VERSION;
            row.header.layerIndex = layer;
            row.header.position = pos;
            row.header.kvDim = payloadLen;
            row.header.fromNodeIndex = localNodeIndex;
            row.header.targetNodeIndex = req.targetStageRootNodeIndex;
            row.header.rangeStart = req.rangeLen != 0u ? payloadStart : 0u;
            row.header.rangeLen = req.rangeLen != 0u ? payloadLen : 0u;
            if (req.rangeLen != 0u) {
                row.kRow = std::move(kRow);
                row.vRow = std::move(vRow);
            } else {
                row.kRow.assign(kRow.begin() + payloadStart, kRow.begin() + payloadStart + payloadLen);
                row.vRow.assign(vRow.begin() + payloadStart, vRow.begin() + payloadStart + payloadLen);
            }
            rows.push_back(std::move(row));
        }
    }

    LlmKvExportResponseHeader resp{};
    resp.magic = LLM_KV_EXPORT_RESPONSE_MAGIC;
    resp.version = LLM_KV_EXPORT_RESPONSE_VERSION;
    resp.requestId = req.requestId;
    resp.fromNodeIndex = localNodeIndex;
    resp.rowCount = (NnUint)rows.size();
    resp.kvDim = payloadLen;
    resp.exportedRows = (NnUint)rows.size();
    resp.reserved = 0u;
    network->write(ROOT_SOCKET_INDEX, &resp, sizeof(resp));
    for (const ExportedRow &row : rows) {
        network->write(ROOT_SOCKET_INDEX, &row.header, sizeof(row.header));
        network->write(ROOT_SOCKET_INDEX, row.kRow.data(), row.kRow.size() * sizeof(float));
        network->write(ROOT_SOCKET_INDEX, row.vRow.data(), row.vRow.size() * sizeof(float));
    }
    std::printf("🧩 [kv-export-response] node=%u requestId=%u rows=%u layers=%zu posRange=[0,%u] range=[%u,%u)\n",
        (unsigned)localNodeIndex,
        (unsigned)req.requestId,
        (unsigned)rows.size(),
        layers.size(),
        (unsigned)endPos,
        (unsigned)payloadStart,
        (unsigned)(payloadStart + payloadLen));
    std::fflush(stdout);
}

void WorkerLlmInference::flushPendingKvAck() {
    if (pendingKvAcks.empty() || network == nullptr) return;
    LlmKvAckBatchHeader abh{};
    abh.magic = LLM_KV_ACK_BATCH_MAGIC;
    abh.version = LLM_KV_ACK_BATCH_VERSION;
    abh.count = (NnUint)pendingKvAcks.size();
    abh.reserved = 0u;
    network->write(ROOT_SOCKET_INDEX, &abh, sizeof(abh));
    for (const auto &ack : pendingKvAcks) {
        network->write(ROOT_SOCKET_INDEX, &ack, sizeof(ack));
    }
    pendingKvAcks.clear();
}

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context)) {
    NnUint nNodes = args->nWorkers + 1;
    LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
    EdgeVisorAblationConfig ablationConfig = edgevisorAblationConfigFromSources(
        args->edgevisorAblationConfigPath,
        args->shadowKvModeStr,
        args->pointerSwizzlingModeStr,
        args->jitModeStr,
        args->vgModeStr,
        args->disableShardingControllerStr,
        args->disablePipelineBalancerStr,
        args->fallbackPolicyStr,
        args->ablationLogPath,
        args->experimentId);
    setEdgeVisorAblationConfig(ablationConfig);
    if (edgevisorAblationLogEnabled()) {
        edgevisorAblationLogSimpleEvent("runtime_config", "full_or_ablation");
    }

    if (nNodes > header.nKvHeads)
        // TODO: https://github.com/b4rtaz/distributed-llama/issues/70
        throw std::runtime_error("This version does not support more nodes than the number of KV heads in the model");

    Tokenizer tokenizer(args->tokenizerPath);
    if (args->info && tokenizer.vocabSize != header.vocabSize)
        printf("Tokenizer vocab size (%d) does not match the model vocab size (%d)\n", tokenizer.vocabSize, header.vocabSize);

    Sampler sampler(tokenizer.vocabSize, args->temperature, args->topp, args->seed);
    LlmNet net;
    std::unique_ptr<NnUnevenPartitionPlan> planPtr;
    std::vector<float> ratios;
    std::string warmupSelectedRatios;
    NnUint warmupSelectedWorkers = args->nWorkers;

    // IMPORTANT: plan barrier affects graph construction (insertion of OP_PLAN_BARRIER/OP_PLAN_APPLY)
    // so we must enable it before building the LLM net.
    setEnablePlanBarrier(args->enablePlanBarrier);
    // IMPORTANT: stage full weights affects weight loading and buffer allocation
    // so we must enable it before building the LLM net.
    setEnableStageFullWeights(args->enableStageFullWeights);
    // IMPORTANT: migration-time KV redundancy safety checks and compute range selection
    // depend on this flag during graph construction.
    setEnableKvRedundancyDuringMigration(args->enableKvRedundancyDuringMigration);
    setAllowNoShadowHeadMigration(args->allowNoShadowHeadMigration);
    // IMPORTANT: KV aggregate pipe build is decided during graph construction.
    setEnableKvAggregate(args->enableKvAggregate);
    // Runtime redundant plan/gate defaults (consumed by buildLlmNetUneven + NnExecutor).
    if (args->enablePpMigration || bubbleShadowKvEnabled()) {
        args->runtimeRedundantBoundaryLayers = inferRuntimeRedundantBoundaryLayers(
            args->kvRedundancyStr,
            nNodes,
            args->runtimeRedundantBoundaryLayers);
    }
    setRuntimeRedundantEnv(
        args->runtimeRedundantBoundaryLayers,
        args->runtimeActiveSegEnabled,
        args->runtimeRedundantSegEnabled && !bubbleShadowKvEnabled(),
        args->runtimePrimarySkipLayersStr);

    if (args->warmupEnabled) {
        if (args->ratiosStr != nullptr) {
            printf("🔥 [warmup] skipped: explicit --ratios=%s was provided\n", args->ratiosStr);
        } else {
            WarmupSelection selection = selectWarmupConfiguration(args, &header, &tokenizer, args->nWorkers);
            warmupSelectedWorkers = selection.nWorkers;
            warmupSelectedRatios = selection.ratios;
            applyWarmupWorkerSubset(args, selection);
            nNodes = args->nWorkers + 1u;
            args->ratiosStr = warmupSelectedRatios.empty() ? nullptr : const_cast<char*>(warmupSelectedRatios.c_str());
            if (warmupSelectedWorkers > 0u) waitForWarmupWorkersToRelisten();
        }
    }

    if(args->ratiosStr != nullptr){
        printf("nNodes=%d\n", nNodes);
        std::vector<NnStageDef> stageDefs = parseStageDefs(args->ratiosStr, nNodes, header.nLayers);
        NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;

        // Parse KV redundancy per node
        std::vector<NnUint> kvRedundancyPerNode = parseKvRedundancy(args->kvRedundancyStr, nNodes);
        if (args->info && args->kvRedundancyStr != nullptr) {
            printf("⚡ KV Redundancy: \"");
            for (size_t i = 0; i < kvRedundancyPerNode.size(); ++i) {
                if (i > 0) printf(",");
                printf("%u", kvRedundancyPerNode[i]);
            }
            printf("\"\n");
        }

        planPtr.reset(new NnUnevenPartitionPlan(
            createPartitionPlan(stageDefs, header.nHeads, header.nKvHeads, header.vocabSize, ffDim, header.dim, kvRedundancyPerNode)
        ));
        
        // 使用 Uneven Builder (传入 planPtr)
        net = buildLlmNetUneven(&header, nNodes, args->nBatches, planPtr.get());
        
        if (args->info) {
            printf("⚖️  Uneven partitioning strategy enabled: %s\n", args->ratiosStr);
            printPartitionPlanDebug(planPtr.get());
        }
    } else {
        printf("⚖️  Even partitioning strategy enabled: ");
        net = buildLlmNet(&header, nNodes, args->nBatches);
    }
    
    std::unique_ptr<LlmNet, void(*)(LlmNet *)> netPtr(&net, releaseLlmNet);

    NnNodeConfig *rootNodeConfig = &net.nodeConfigs[0];

    if (args->info) {
        tokenizer.printHeader();
        printLlmHeader(&header);
        printNodeRequiredMemory(&net.netConfig, rootNodeConfig);
    }

    NnNetExecution execution(args->nThreads, &net.netConfig);

    std::unique_ptr<NnNodeSynchronizer> synchronizer(nullptr);
    std::unique_ptr<NnNetwork> networkPtr(nullptr);
    NnNetwork *network = nullptr;

    const bool profileEnabled = args->benchmark;
    const bool layerProfileEnabled = profileEnabled && envFlagEnabledDefault("DLLAMA_LAYER_PROF_ENABLE", false);

    if (nNodes == 1) {
        synchronizer.reset(new NnFakeNodeSynchronizer());
    } else {
        networkPtr = NnNetwork::connect(args->nWorkers, args->workerHosts, args->workerPorts);
        network = networkPtr.get();

        // Bootstrap: send modelPath/ratios/maxSeqLen/syncType to workers so they don't need CLI args.
        for (NnUint nodeIndex = 1; nodeIndex < nNodes; ++nodeIndex) {
            const NnUint socketIndex = nodeIndex - 1;
            writeBootstrapPacket(network, socketIndex, args);
        }

        // 初始化 Synchronizer (传入 Plan)
        synchronizer.reset(new NnNetworkNodeSynchronizer(network, &execution, &net.netConfig, rootNodeConfig, planPtr.get(), layerProfileEnabled));

        NnRootConfigWriter configWriter(network);
        configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
    }

    std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution, planPtr.get());
    NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), profileEnabled);

    // Load weights
    if (args->ratiosStr != nullptr) {
        // [非均匀/PP 模式]：强制使用本地加载 (Local Loading)
        printf("🚀 Local Loading Mode (Root): Loading weights locally...\n");
        
        NnLocalWeightLoader localLoader(&executor, 0); 
        // 传入 0 作为 Root 的 nodeIndex
        loadLlmNetWeightUneven(args->modelPath, &net, &localLoader, planPtr.get(), 0);
        printf("✅ Root: Weights loaded locally.\n");

    } else {
        // [均匀模式]：保持原有行为 (网络分发)
        NnRootWeightLoader weightLoader(&executor, network, nNodes);
        loadLlmNetWeight(args->modelPath, &net, &weightLoader);
    }

    RootLlmInference inference(&net, &execution, &executor, network, planPtr.get(), profileEnabled, args->enablePpMigration);

    // Step 1: seed PlanCommand cache from deprecated env hooks (fallback).
    maybeSeedPlanCommandFromLegacyEnv();

    // enablePlanBarrier was already applied before net construction.

    // Step 4: external controller (UDS). Enabled by setting DLLAMA_PLAN_CTRL_SOCKET.
    std::unique_ptr<PlanUdsController> planCtrl;
    const char *planSock = std::getenv("DLLAMA_PLAN_CTRL_SOCKET");
    if (planSock != nullptr && planSock[0] != '\0') {
        planCtrl = PlanUdsController::start(std::string(planSock), &inference);
    }

    // Step 5: internal dynamic scheduler (queries UDS layer_prof and issues set_plan).
    std::unique_ptr<DynamicLayerController> dynCtrl;
    std::unique_ptr<RootKvCollector> kvCollector;
    kvCollector = RootKvCollector::start(&inference);
    if (planSock != nullptr && planSock[0] != '\0') {
        dynCtrl = DynamicLayerController::start(std::string(planSock), &inference);
    }

    if (network != nullptr) {
        network->resetStats();
        if (args->netTurbo) {
            network->setTurbo(true);
            printf("🚁 Network is in non-blocking mode\n");
        }
    }

    AppInferenceContext context;
    context.args = args;
    context.header = &header;
    context.inference = &inference;
    context.sampler = &sampler;
    context.tokenizer = &tokenizer;
    context.network = network;
    context.executor = &executor;
    context.nodeConfig = rootNodeConfig;

    handler(&context);

    inference.finish();
}

void runWorkerApp(AppCliArgs *args) {
    while (true) {
        std::unique_ptr<NnNetwork> networkPtr = NnNetwork::serve(args->port);
        NnNetwork *network = networkPtr.get();

        // Read bootstrap settings from root.
        std::string bootModelPath;
        std::string bootRatios;
        std::string bootPrimarySkipLayers;
        std::string bootKvRedundancy;
        LlmBootstrapPacket boot = readBootstrapPacket(network, bootModelPath, bootRatios, bootPrimarySkipLayers, bootKvRedundancy);

        const bool hasBootModel = !bootModelPath.empty();
        const bool hasBootRatios = !bootRatios.empty();
        const bool useLocalLoading = hasBootModel && hasBootRatios;
        NnUint samplerVocabSize = 0u;
        const NnUint bootMaxSeqLen = boot.maxSeqLen;
        const NnFloatType bootSyncType = (NnFloatType)boot.syncType;
        const bool bootBenchmarkEnabled = boot.benchmarkEnabled != 0u;
        const bool bootEnablePlanBarrier = boot.enablePlanBarrier != 0u;
        const bool bootEnableStageFullWeights = boot.enableStageFullWeights != 0u;
        const bool bootEnableKvRedundancyDuringMigration = boot.enableKvRedundancyDuringMigration != 0u;
        const bool bootAllowNoShadowHeadMigration = boot.allowNoShadowHeadMigration != 0u;
        const bool bootEnableKvAggregate = boot.enableKvAggregate != 0u;
        const NnUint bootRuntimeRedundantBoundaryLayers = boot.runtimeRedundantBoundaryLayers;
        const bool bootRuntimeActiveSegEnabled = boot.runtimeActiveSegEnabled != 0u;
        const bool bootRuntimeRedundantSegEnabled = boot.runtimeRedundantSegEnabled != 0u;
        const bool bootBubbleShadowKvEnabled = ((boot.flags & LLM_BOOTSTRAP_ENABLE_BUBBLE_SHADOW_KV) != 0u) || boot.bubbleShadowKvEnabled != 0u;
        const bool bootLastStageSamplingEnabled = (boot.flags & LLM_BOOTSTRAP_LAST_STAGE_SAMPLING) != 0u;
        if (bootBubbleShadowKvEnabled) {
            setenv("DLLAMA_BUBBLE_SHADOW_KV", "1", 1);
            if ((boot.flags & LLM_BOOTSTRAP_DISABLE_BUBBLE_SHADOW_KV_ASYNC) != 0u) {
                setenv("DLLAMA_BUBBLE_SHADOW_KV_ASYNC", "0", 1);
            } else {
                unsetenv("DLLAMA_BUBBLE_SHADOW_KV_ASYNC");
            }
        } else {
            unsetenv("DLLAMA_BUBBLE_SHADOW_KV");
            unsetenv("DLLAMA_BUBBLE_SHADOW_KV_ASYNC");
        }
        if (bootLastStageSamplingEnabled) {
            setenv("DLLAMA_LAST_STAGE_SAMPLING", "1", 1);
        } else {
            unsetenv("DLLAMA_LAST_STAGE_SAMPLING");
        }

        // Set enable plan barrier flag from bootstrap packet
        setEnablePlanBarrier(bootEnablePlanBarrier);
        // Set enable stage full weights flag from bootstrap packet
        setEnableStageFullWeights(bootEnableStageFullWeights);
        // Set migration-time KV redundancy behavior from bootstrap packet
        setEnableKvRedundancyDuringMigration(bootEnableKvRedundancyDuringMigration);
        setAllowNoShadowHeadMigration(bootAllowNoShadowHeadMigration);
        // Set KV aggregate graph-build flag from bootstrap packet.
        setEnableKvAggregate(bootEnableKvAggregate);
        // Apply runtime redundant plan/gate defaults from bootstrap packet.
        setRuntimeRedundantEnv(
            bootRuntimeRedundantBoundaryLayers,
            bootRuntimeActiveSegEnabled,
            bootRuntimeRedundantSegEnabled && !bubbleShadowKvEnabled(),
            bootPrimarySkipLayers.c_str());

        NnWorkerConfigReader configReader(network);
        NnNetConfig netConfig = configReader.readNet();
        NnNodeConfig nodeConfig = configReader.readNode();
        
        std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
        std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

        printNodeRequiredMemory(&netConfig, &nodeConfig);

        NnNetExecution execution(args->nThreads, &netConfig);

           // 1. Initialize Plan Pointer (Worker Side)
           std::unique_ptr<NnUnevenPartitionPlan> planPtr;
        
           if (useLocalLoading) {
               // Worker 需要重新加载 Header 和 Plan 以确定加载逻辑和切分
               LlmHeader header = loadLlmHeader((char*)bootModelPath.c_str(), bootMaxSeqLen, bootSyncType);
             samplerVocabSize = header.vocabSize;
             
             // [兼容性修复] 自动切换 Q80
             if (header.weightType == F_Q40 && header.syncType != F_Q80) {
                 header.syncType = F_Q80;
             }

               std::vector<NnStageDef> stageDefs = parseStageDefs(bootRatios.c_str(), netConfig.nNodes, header.nLayers);
             NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;

             std::vector<NnUint> kvRedundancyPerNode = parseKvRedundancy(
                 bootKvRedundancy.empty() ? nullptr : bootKvRedundancy.c_str(),
                 netConfig.nNodes);
             planPtr.reset(new NnUnevenPartitionPlan(
                 createPartitionPlan(stageDefs, header.nHeads, header.nKvHeads, header.vocabSize, ffDim, header.dim, kvRedundancyPerNode)
             ));
        }

        std::vector<NnExecutorDevice> devices = resolveDevices(args, &netConfig, &nodeConfig, &execution, planPtr.get());
        
        // Initialize Synchronizer with Plan
        const bool layerProfileEnabled = bootBenchmarkEnabled && envFlagEnabledDefault("DLLAMA_LAYER_PROF_ENABLE", false);
        NnNetworkNodeSynchronizer synchronizer(network, &execution, &netConfig, &nodeConfig, planPtr.get(), layerProfileEnabled);
        
        // Benchmark flag is provided by root to keep all nodes consistent.
        // Worker CLI --benchmark is no longer required.
        const bool profileEnabled = bootBenchmarkEnabled;
        NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, profileEnabled);

        if (useLocalLoading) {
            // [Local Loading Mode]
            printf("🚀 Worker %d: Local Loading Mode from %s\n", nodeConfig.nodeIndex, bootModelPath.c_str());
            
            // Reload header for temporary network construction
            LlmHeader header = loadLlmHeader((char*)bootModelPath.c_str(), bootMaxSeqLen, bootSyncType);
            
            // Build temporary Net for loading context
            // 这里我们需要构建一个临时的 LlmNet 结构，因为 loader 需要 net 指针
            // 关键：确保临时 Net 的 Plan 和 NodeConfig 绑定正确
            LlmNet tempNet = buildLlmNetUneven(&header, netConfig.nNodes, 1, planPtr.get());

            // Execute local loading
            NnLocalWeightLoader localLoader(&executor, nodeConfig.nodeIndex);
            
            // 使用新版 5 参数加载函数
            loadLlmNetWeightUneven((char*)bootModelPath.c_str(), &tempNet, &localLoader, planPtr.get(), nodeConfig.nodeIndex);

            releaseLlmNet(&tempNet);
            printf("✅ Worker %d: Weights loaded locally.\n", nodeConfig.nodeIndex);

        } else {
            // [Network Loading Mode] (Legacy)
            printf("📡 Worker %d: Waiting for weights from Root...\n", nodeConfig.nodeIndex);
            NnWorkerWeightReader weightReader(&executor, network);
            weightReader.read();
        }

        const NnUint logitsPipeIndex = findPipeIndexByName(&netConfig, "LG");
        std::unique_ptr<Sampler> lastStageSampler;
        if (bootLastStageSamplingEnabled && lastStageSamplingPlanSupported(planPtr.get())) {
            const NnStageConfig &last = planPtr->stages[planPtr->nStages - 1u];
            if (last.rootNodeIndex == nodeConfig.nodeIndex && planPtr->vocabSplit.lengths != nullptr) {
                samplerVocabSize = 0u;
                for (NnUint node = 0u; node < planPtr->nNodes; ++node) samplerVocabSize += planPtr->vocabSplit.lengths[node];
            } else {
                samplerVocabSize = 0u;
            }
        }
        if (bootLastStageSamplingEnabled && samplerVocabSize > 0u) {
            lastStageSampler.reset(new Sampler((int)samplerVocabSize, boot.samplerTemperature, boot.samplerTopP, boot.samplerSeed));
        }
        WorkerLlmInference inference(&execution, network, nodeConfig.nodeIndex, logitsPipeIndex, lastStageSampler.get());
        bool isFirstAttempt = true;
        bool isTurboEnabled = false;
        clock_t startTime;
        
        while (true) {
            try {
                if (isFirstAttempt)
                    startTime = clock();

                if (!inference.tryReadControlPacket()) {
                    if (isTurboEnabled && !isFirstAttempt && clock() - startTime > CLOCKS_PER_SEC) {
                        network->setTurbo(false);
                        isTurboEnabled = false;
                        printf("🚁 Network is in blocking mode\n");
                    }
                    isFirstAttempt = false;
                    continue;
                }
                if (inference.isFinished)
                    break;

                if (args->netTurbo && !isTurboEnabled) {
                    network->setTurbo(true);
                    isTurboEnabled = true;
                    printf("🚁 Network is in non-blocking mode\n");
                }

                LlmKvTransferHeader kvHdr{};
                std::vector<float> kvK;
                std::vector<float> kvV;
                while (inference.consumePendingKvTransfer(kvHdr, kvK, kvV)) {
                    const bool wOk = executor.applyTransferredKvRow(
                        kvHdr.layerIndex,
                        kvHdr.position,
                        kvK,
                        kvV,
                        kvHdr.rangeStart,
                        kvHdr.rangeLen);
                    std::printf("🧩 [kv-write-check] node=%u layer=%u pos=%u kvDim=%u range=[%u,%u) status=%s\n",
                        (unsigned)nodeConfig.nodeIndex,
                        (unsigned)kvHdr.layerIndex,
                        (unsigned)kvHdr.position,
                        (unsigned)kvHdr.kvDim,
                        (unsigned)kvHdr.rangeStart,
                        (unsigned)(kvHdr.rangeStart + kvHdr.rangeLen),
                        wOk ? "ok" : "fail");
                    std::fflush(stdout);
                }

                LlmLayerSwitchPacket switchPkt{};
                while (inference.consumeLayerSwitch(switchPkt)) {
                    bool localIsSourceStage = (switchPkt.fromNodeIndex == nodeConfig.nodeIndex);
                    bool localIsTargetStage = (switchPkt.toNodeIndex == nodeConfig.nodeIndex);

                    if (planPtr != nullptr) {
                        const NnStageConfig *fromStage = findStageForNodeLocal(planPtr.get(), switchPkt.fromNodeIndex);
                        const NnStageConfig *toStage = findStageForNodeLocal(planPtr.get(), switchPkt.toNodeIndex);

                        if (fromStage != nullptr) {
                            localIsSourceStage = false;
                            for (NnUint i = 0u; i < fromStage->nNodes; ++i) {
                                if (fromStage->nodeIndices[i] == nodeConfig.nodeIndex) {
                                    localIsSourceStage = true;
                                    break;
                                }
                            }
                        }

                        if (toStage != nullptr) {
                            localIsTargetStage = false;
                            for (NnUint i = 0u; i < toStage->nNodes; ++i) {
                                if (toStage->nodeIndices[i] == nodeConfig.nodeIndex) {
                                    localIsTargetStage = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (localIsSourceStage) {
                        executor.setPrimaryLayerEnabled(switchPkt.boundaryLayer, false);
                        std::printf("🔁 [worker-switch] node=%u sleep primary layer=%u (source-stage)\n",
                            (unsigned)nodeConfig.nodeIndex,
                            (unsigned)switchPkt.boundaryLayer);
                        std::fflush(stdout);
                    }
                    if (localIsTargetStage) {
                        executor.setRedundantLayerEnabled(switchPkt.boundaryLayer, true);
                        std::printf("🔁 [worker-switch] node=%u activate redundant layer=%u (target-stage)\n",
                            (unsigned)nodeConfig.nodeIndex,
                            (unsigned)switchPkt.boundaryLayer);
                        std::fflush(stdout);
                    }
                    if (planPtr != nullptr) {
                        const NnStageConfig *fromStage = findStageForNodeLocal(planPtr.get(), switchPkt.fromNodeIndex);
                        const NnStageConfig *toStage = findStageForNodeLocal(planPtr.get(), switchPkt.toNodeIndex);
                        if (localIsSourceStage && fromStage != nullptr && toStage != nullptr && toStage->stageIndex < fromStage->stageIndex) {
                            const NnUint shiftedStartLayer = switchPkt.boundaryLayer + 1u;
                            if (shiftedStartLayer < fromStage->endLayer) {
                                executor.setShiftedPpStartLayerEnabled(shiftedStartLayer, true);
                                std::printf("🔁 [worker-switch] node=%u shifted pp-start layer=%u (source-stage)\n",
                                    (unsigned)nodeConfig.nodeIndex,
                                    (unsigned)shiftedStartLayer);
                                std::fflush(stdout);
                            }
                        }
                    }
                }

                LlmKvExportRequestHeader exportReq{};
                std::vector<NnUint> exportLayers;
                while (inference.consumeKvExportRequest(exportReq, exportLayers)) {
                    inference.flushKvExportResponse(exportReq, exportLayers, exportReq.kvDim, &executor);
                }

                if ((inference.flags() & LLM_CTRL_CONTROL_ONLY) != 0u) {
                    inference.flushPendingKvAck();
                    if ((inference.flags() & LLM_CTRL_PROFILE) != 0u) {
                        LlmPerfPacket p{};
                        p.position = inference.position();
                        p.batchSize = inference.batchSize();
                        p.nodeIndex = nodeConfig.nodeIndex;
                        p.stageIndex = getStageIndexForNode(planPtr.get(), nodeConfig.nodeIndex);
                        p.execUs = 0u;
                        p.syncUs = 0u;
                        network->write(ROOT_SOCKET_INDEX, &p, sizeof(LlmPerfPacket));
                    }
                    isFirstAttempt = true;
                    continue;
                }

                executor.forward();
                NnBubbleShadowStats bubbleStats = maybeRunBubbleShadowKv(&executor, "worker", nodeConfig.nodeIndex, inference.position(), inference.batchSize());
                inference.flushPendingKvAck();

                // Send per-forward profile packet to root (optional)
                // IMPORTANT: root will block waiting for these packets when profiling is enabled.
                // So workers must reply whenever the control packet requests profiling,
                // even if the worker binary was started without --benchmark (times may be 0).
                if ((inference.flags() & LLM_CTRL_PROFILE) != 0u) {
                    LlmPerfPacket p{};
                    p.position = inference.position();
                    p.batchSize = inference.batchSize();
                    p.nodeIndex = nodeConfig.nodeIndex;
                    p.stageIndex = getStageIndexForNode(planPtr.get(), nodeConfig.nodeIndex);
                    p.execUs = executor.getTotalTime(STEP_EXECUTE_OP);
                    p.syncUs = executor.getTotalTime(STEP_SYNC_NODES);
                    p.bubbleUs = (NnUint)std::min<unsigned long long>(bubbleStats.elapsedUs, (unsigned long long)UINT32_MAX);
                    p.bubbleSegments = bubbleStats.segmentsVisited;
                    p.bubbleOps = bubbleStats.opStepsExecuted;
                    p.bubbleSkippedSyncs = bubbleStats.skippedSyncSteps;
                    network->write(ROOT_SOCKET_INDEX, &p, sizeof(LlmPerfPacket));
                }
                inference.maybeSendLastStageSampledToken(planPtr.get());
                isFirstAttempt = true;
            } catch (const NnTransferSocketException &e) {
                printf("🚨 Network error: %s\n", e.what());
                break;
            } catch (const NnExecutorException &e) {
                printf("🚨 Inference error: %s\n", e.what());
                break;
            }
        }
    }
}
