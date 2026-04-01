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
#if defined(DLLAMA_VULKAN)
#include <cstdlib>
    #include "nn/nn-vulkan.hpp"
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
    LlmBootstrapPacket p;
    p.magic = LLM_BOOTSTRAP_MAGIC;
    p.version = LLM_BOOTSTRAP_VERSION;
    p.flags = 0u;
    p.benchmarkEnabled = args->benchmark ? 1u : 0u;
    p.enablePlanBarrier = args->enablePlanBarrier ? 1u : 0u;
    p.enableStageFullWeights = args->enableStageFullWeights ? 1u : 0u;
    p.enableKvRedundancyDuringMigration = args->enableKvRedundancyDuringMigration ? 1u : 0u;
    p.enableKvAggregate = args->enableKvAggregate ? 1u : 0u;
    p.runtimeRedundantBoundaryLayers = args->runtimeRedundantBoundaryLayers;
    p.runtimeActiveSegEnabled = args->runtimeActiveSegEnabled ? 1u : 0u;
    p.runtimeRedundantSegEnabled = args->runtimeRedundantSegEnabled ? 1u : 0u;
    p.maxSeqLen = args->maxSeqLen;
    p.syncType = (NnUint)args->syncType;
    p.modelPathLen = 0u;
    p.ratiosLen = 0u;
    p.primarySkipLayersLen = 0u;

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
    if (args->enableKvAggregate) {
        p.flags |= LLM_BOOTSTRAP_ENABLE_KV_AGGREGATE;
    }

    network->write(socketIndex, &p, sizeof(p));
    if (p.modelPathLen > 0u) network->write(socketIndex, args->modelPath, p.modelPathLen);
    if (p.ratiosLen > 0u) network->write(socketIndex, args->ratiosStr, p.ratiosLen);
    if (p.primarySkipLayersLen > 0u) network->write(socketIndex, args->runtimePrimarySkipLayersStr, p.primarySkipLayersLen);
}

static LlmBootstrapPacket readBootstrapPacket(NnNetwork *network, std::string &modelPath, std::string &ratiosStr, std::string &primarySkipLayersStr) {
    LlmBootstrapPacket p;
    network->read(ROOT_SOCKET_INDEX, &p, sizeof(p));
    if (p.magic != LLM_BOOTSTRAP_MAGIC)
        throw std::runtime_error("Invalid bootstrap magic (root/worker binary mismatch)");
    if (p.version != LLM_BOOTSTRAP_VERSION)
        throw std::runtime_error("Unsupported bootstrap version (root/worker binary mismatch)");

    modelPath.clear();
    ratiosStr.clear();
    primarySkipLayersStr.clear();
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

AppCliArgs AppCliArgs::parse(int argc, char* *argv, bool requireMode) {
    AppCliArgs args;
    args.info = true;
    args.help = false;
    args.mode = nullptr;
    args.nBatches = 32;
    args.nThreads = 1;
    args.modelPath = nullptr;
    args.tokenizerPath = nullptr;
    args.prompt = nullptr;
    args.interactive = false;
    args.syncType = F_32;
    args.nWorkers = 0;
    args.workerHosts = nullptr;
    args.workerPorts = nullptr;
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 0;
    args.benchmark = false;
    args.seed = (unsigned long long)time(nullptr);
    args.chatTemplateType = TEMPLATE_UNKNOWN;
    args.maxSeqLen = 0;
    args.netTurbo = true;
    args.gpuIndex = -1;
    args.gpuSegmentFrom = -1;
    args.gpuSegmentTo = -1;
    args.ratiosStr = nullptr;
    args.kvRedundancyStr = nullptr;
    args.enablePlanBarrier = false;
    args.enableStageFullWeights = false;
    args.enableKvRedundancyDuringMigration = true;
    args.enableKvAggregate = false;
    args.enablePpMigration = false;
    args.runtimeRedundantBoundaryLayers = 1u;
    args.runtimeActiveSegEnabled = true;
    args.runtimeRedundantSegEnabled = false;
    args.runtimePrimarySkipLayersStr = nullptr;

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
        } else if (std::strcmp(name, "--ratios") == 0) {
            args.ratiosStr = value;
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
        } else {
            throw std::runtime_error("Unknown option: " + std::string(name));
        }

        i += 2;
    }

    if (args.nThreads < 1)
        throw std::runtime_error("Number of threads must be at least 1");
    if (args.enablePpMigration && !args.enableKvAggregate) {
        args.enableKvAggregate = true;
        std::printf("⚠️  [pp-migrate] --enable-pp-migration requires KV aggregate; auto enabling --enable-kv-aggregate\n");
        std::fflush(stdout);
    }
    return args;
}

AppCliArgs::~AppCliArgs() {
    if (workerHosts != nullptr) {
        for (NnUint i = 0; i < nWorkers; i++)
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

static std::vector<NnExecutorDevice> resolveDevices(AppCliArgs *args, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *plan = nullptr) {
    std::vector<NnExecutorDevice> devices;

    if (args->gpuIndex >= 0) {
#if defined(DLLAMA_VULKAN)
        devices.push_back(NnExecutorDevice(
            new NnVulkanDevice(args->gpuIndex, netConfig, nodeConfig, netExecution),
            args->gpuSegmentFrom,
            args->gpuSegmentTo
        ));
#else
        throw std::runtime_error("This build does not support GPU");
#endif
    }

    if (args->gpuIndex < 0 || (args->gpuSegmentFrom >= 0 && args->gpuSegmentTo >= 0)) {
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
    this->profileEnabled = profileEnabled;
    this->controlPacket.flags = profileEnabled ? LLM_CTRL_PROFILE : 0u;
    this->controlPacket.planCmdSeq = 0u;
    this->lastPlanCmdSeqSent = 0u;
    this->asyncKvCollectLayer = -1;
    this->asyncKvCollectPos = -1;
    waitingKvAckReceivedCount = 0u;
    waitingKvAckReceivedLayers.clear();
    waitingKvAckNodes.clear();
    waitingKvAckNodeExpected.clear();
    waitingKvAckNodeReceived.clear();

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
    migrationFromNodeIndex = 0u;
    nextStageRootNode = (NnUint)-1;
    kvAckSocketIndex = -1;

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
}

void RootLlmInference::setRuntimeLayerGate(bool enablePrimarySegments, bool enableRedundantSegments) {
    if (executor == nullptr) return;
    executor->setRuntimeLayerGate(enablePrimarySegments, enableRedundantSegments);
}

void RootLlmInference::setPrimaryLayerEnabled(NnUint layerIndex, bool enabled) {
    if (executor == nullptr) return;
    executor->setPrimaryLayerEnabled(layerIndex, enabled);
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
    const std::vector<float> &vRow) {
    if (network == nullptr) return KV_TRANSFER_SUBMIT_NO_NETWORK;
    if (nextStageRootNode == (NnUint)-1) return KV_TRANSFER_SUBMIT_NO_TARGET_STAGE;
    if (!migrationLayers.empty()) {
        const bool inList = std::find(migrationLayers.begin(), migrationLayers.end(), layerIndex) != migrationLayers.end();
        if (!inList) return KV_TRANSFER_SUBMIT_LAYER_NOT_IN_LIST;
    }
    if (kRow.empty() || vRow.empty() || kRow.size() != vRow.size()) return KV_TRANSFER_SUBMIT_INVALID_ROW;

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

    bool queuedAny = false;
    for (NnUint targetNode : targetNodes) {
        if (targetNode == 0u) {
            if (executor != nullptr) {
                const bool localOk = executor->applyTransferredKvRow(layerIndex, position, kRow, vRow);
                std::printf("🧩 [kv-write-local] layer=%u pos=%u targetNode=0 status=%s\n",
                    (unsigned)layerIndex,
                    (unsigned)position,
                    localOk ? "ok" : "fail");
                std::fflush(stdout);
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
        item.header.reserved = 0u;
        item.kRow = kRow;
        item.vRow = vRow;
        pendingKvTransfers.push_back(std::move(item));
        queuedAny = true;
    }

    if (!queuedAny) return KV_TRANSFER_SUBMIT_DUP_LAYER_PENDING;
    return KV_TRANSFER_SUBMIT_OK;
}

bool RootLlmInference::submitBoundaryKvTransfer(NnUint layerIndex, NnUint position, const std::vector<float> &kRow, const std::vector<float> &vRow) {
    return submitBoundaryKvTransferDetailed(layerIndex, position, kRow, vRow) == KV_TRANSFER_SUBMIT_OK;
}

void RootLlmInference::forward() {
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
            // Mark this command as published-to-workers for this process.
            setPlanCommandPublishedSeq(snap.cmd.seq);
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
    executor->forward();

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
                lastPpPlanCacheSeqApplied = snap.cacheSeq;
            }
        }
    }

    if (!migrationBatchSubmitted && asyncKvCollectPos >= 0 &&
        controlPacket.position == (NnUint)asyncKvCollectPos &&
        !migrationLayers.empty() && executor != nullptr && header != nullptr) {
        NnUint exported = 0u;
        NnUint queued = 0u;
        const NnUint endPos = std::min((NnUint)asyncKvCollectPos, (header->seqLen > 0u) ? (header->seqLen - 1u) : 0u);
        for (NnUint layer : migrationLayers) {
            for (NnUint pos = 0u; pos <= endPos; ++pos) {
                std::vector<float> kRow;
                std::vector<float> vRow;
                if (executor->exportLayerKvRow(layer, pos, header->kvDim, kRow, vRow)) {
                    exported += 1u;
                    if (submitBoundaryKvTransfer(layer, pos, kRow, vRow)) {
                        queued += 1u;
                    }
                } else {
                    std::printf("⚠️  [kv-migrate] export row failed layer=%u pos=%u\n",
                        (unsigned)layer,
                        (unsigned)pos);
                }
            }
        }
        migrationBatchSubmitted = (queued > 0u);
        std::printf("🧩 [kv-migrate] prepared history batch exported=%u queued=%u layers=%u posRange=[0,%u] targetRoot=%u\n",
            (unsigned)exported,
            (unsigned)queued,
            (unsigned)migrationLayers.size(),
            (unsigned)endPos,
            (unsigned)nextStageRootNode);
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

    if (!profileEnabled) return;

    // Collect per-node timings for this forward() call.
    lastPerf.clear();
    lastPerf.reserve((network != nullptr ? network->nSockets : 0u) + 1u);

    // Root node (node 0)
    {
        LlmPerfPacket p;
        p.position = controlPacket.position;
        p.batchSize = controlPacket.batchSize;
        p.nodeIndex = 0;
        p.stageIndex = getStageIndexForNode(plan, 0);
        p.execUs = executor->getTotalTime(STEP_EXECUTE_OP);
        p.syncUs = executor->getTotalTime(STEP_SYNC_NODES);
        lastPerf.push_back(p);
    }

    // Worker nodes
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

WorkerLlmInference::WorkerLlmInference(NnNetExecution *execution, NnNetwork *network, NnUint localNodeIndex) {
    this->isFinished = false;
    this->execution = execution;
    this->network = network;
    this->localNodeIndex = localNodeIndex;
    this->positionPipe = (float *)execution->pipes[0];
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
                // This process has received/published this command.
                setPlanCommandPublishedSeq(cmd.seq);
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

    // IMPORTANT: plan barrier affects graph construction (insertion of OP_PLAN_BARRIER/OP_PLAN_APPLY)
    // so we must enable it before building the LLM net.
    setEnablePlanBarrier(args->enablePlanBarrier);
    // IMPORTANT: stage full weights affects weight loading and buffer allocation
    // so we must enable it before building the LLM net.
    setEnableStageFullWeights(args->enableStageFullWeights);
    // IMPORTANT: migration-time KV redundancy safety checks and compute range selection
    // depend on this flag during graph construction.
    setEnableKvRedundancyDuringMigration(args->enableKvRedundancyDuringMigration);
    // IMPORTANT: KV aggregate pipe build is decided during graph construction.
    setEnableKvAggregate(args->enableKvAggregate);
    // Runtime redundant plan/gate defaults (consumed by buildLlmNetUneven + NnExecutor).
    setRuntimeRedundantEnv(
        args->runtimeRedundantBoundaryLayers,
        args->runtimeActiveSegEnabled,
        args->runtimeRedundantSegEnabled,
        args->runtimePrimarySkipLayersStr);

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
        synchronizer.reset(new NnNetworkNodeSynchronizer(network, &execution, &net.netConfig, rootNodeConfig, planPtr.get(), profileEnabled));

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
        LlmBootstrapPacket boot = readBootstrapPacket(network, bootModelPath, bootRatios, bootPrimarySkipLayers);

        const bool hasBootModel = !bootModelPath.empty();
        const bool hasBootRatios = !bootRatios.empty();
        const bool useLocalLoading = hasBootModel && hasBootRatios;
        const NnUint bootMaxSeqLen = boot.maxSeqLen;
        const NnFloatType bootSyncType = (NnFloatType)boot.syncType;
        const bool bootBenchmarkEnabled = boot.benchmarkEnabled != 0u;
        const bool bootEnablePlanBarrier = boot.enablePlanBarrier != 0u;
        const bool bootEnableStageFullWeights = boot.enableStageFullWeights != 0u;
        const bool bootEnableKvRedundancyDuringMigration = boot.enableKvRedundancyDuringMigration != 0u;
        const bool bootEnableKvAggregate = boot.enableKvAggregate != 0u;
        const NnUint bootRuntimeRedundantBoundaryLayers = boot.runtimeRedundantBoundaryLayers;
        const bool bootRuntimeActiveSegEnabled = boot.runtimeActiveSegEnabled != 0u;
        const bool bootRuntimeRedundantSegEnabled = boot.runtimeRedundantSegEnabled != 0u;

        // Set enable plan barrier flag from bootstrap packet
        setEnablePlanBarrier(bootEnablePlanBarrier);
        // Set enable stage full weights flag from bootstrap packet
        setEnableStageFullWeights(bootEnableStageFullWeights);
        // Set migration-time KV redundancy behavior from bootstrap packet
        setEnableKvRedundancyDuringMigration(bootEnableKvRedundancyDuringMigration);
        // Set KV aggregate graph-build flag from bootstrap packet.
        setEnableKvAggregate(bootEnableKvAggregate);
        // Apply runtime redundant plan/gate defaults from bootstrap packet.
        setRuntimeRedundantEnv(
            bootRuntimeRedundantBoundaryLayers,
            bootRuntimeActiveSegEnabled,
            bootRuntimeRedundantSegEnabled,
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
             
             // [兼容性修复] 自动切换 Q80
             if (header.weightType == F_Q40 && header.syncType != F_Q80) {
                 header.syncType = F_Q80;
             }

               std::vector<NnStageDef> stageDefs = parseStageDefs(bootRatios.c_str(), netConfig.nNodes, header.nLayers);
             NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;

             // Worker side: use default KV redundancy (empty vector = default to NN_KV_REDUNDANCY_PAD_HEADS)
             planPtr.reset(new NnUnevenPartitionPlan(
                 createPartitionPlan(stageDefs, header.nHeads, header.nKvHeads, header.vocabSize, ffDim, header.dim, {})
             ));
        }

        std::vector<NnExecutorDevice> devices = resolveDevices(args, &netConfig, &nodeConfig, &execution, planPtr.get());
        
        // Initialize Synchronizer with Plan
        NnNetworkNodeSynchronizer synchronizer(network, &execution, &netConfig, &nodeConfig, planPtr.get(), bootBenchmarkEnabled);
        
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

        WorkerLlmInference inference(&execution, network, nodeConfig.nodeIndex);
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
                    const bool wOk = executor.applyTransferredKvRow(kvHdr.layerIndex, kvHdr.position, kvK, kvV);
                    std::printf("🧩 [kv-write-check] node=%u layer=%u pos=%u kvDim=%u status=%s\n",
                        (unsigned)nodeConfig.nodeIndex,
                        (unsigned)kvHdr.layerIndex,
                        (unsigned)kvHdr.position,
                        (unsigned)kvHdr.kvDim,
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
                }

                executor.forward();
                inference.flushPendingKvAck();

                // Send per-forward profile packet to root (optional)
                // IMPORTANT: root will block waiting for these packets when profiling is enabled.
                // So workers must reply whenever the control packet requests profiling,
                // even if the worker binary was started without --benchmark (times may be 0).
                if ((inference.flags() & LLM_CTRL_PROFILE) != 0u) {
                    LlmPerfPacket p;
                    p.position = inference.position();
                    p.batchSize = inference.batchSize();
                    p.nodeIndex = nodeConfig.nodeIndex;
                    p.stageIndex = getStageIndexForNode(planPtr.get(), nodeConfig.nodeIndex);
                    p.execUs = executor.getTotalTime(STEP_EXECUTE_OP);
                    p.syncUs = executor.getTotalTime(STEP_SYNC_NODES);
                    network->write(ROOT_SOCKET_INDEX, &p, sizeof(LlmPerfPacket));
                }
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
