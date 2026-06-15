#include "ablation.hpp"
#include "nn/nn-executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

bool getEnableStageFullWeights() { return false; }
bool getEnableKvRedundancyDuringMigration() { return false; }
bool getAllowNoShadowHeadMigration() { return false; }

class BenchDeviceSegment : public NnDeviceSegment {
private:
    NnSegmentConfig *segmentConfig;
    const NnUnevenPartitionPlan *plan;
    std::vector<uint64_t> pointerViews;
    uint64_t refreshSalt;

public:
    explicit BenchDeviceSegment(NnSegmentConfig *segmentConfig)
        : segmentConfig(segmentConfig), plan(nullptr), pointerViews(segmentConfig->nOps, 0u), refreshSalt(0u) {}

    void loadWeight(NnUint, NnSize, NnSize, NnByte *) override {}

    void forward(NnUint, NnUint, NnUint, NnUint) override {}

    void setPartitionPlan(const NnUnevenPartitionPlan *newPlan) override {
        plan = newPlan;
    }

    void refreshPointers() override {
        const NnUint node = 0u;
        const NnUint headStart = (plan != nullptr && plan->headSplit.starts != nullptr) ? plan->headSplit.starts[node] : 0u;
        const NnUint kvStart = (plan != nullptr && plan->kvHeadSplit.starts != nullptr) ? plan->kvHeadSplit.starts[node] : 0u;
        const NnUint ffnStart = (plan != nullptr && plan->ffnSplit.starts != nullptr) ? plan->ffnSplit.starts[node] : 0u;
        for (NnUint i = 0; i < segmentConfig->nOps; ++i) {
            const NnOpConfig &op = segmentConfig->ops[i];
            pointerViews[i] =
                ((uint64_t)op.input.pointerIndex << 48) ^
                ((uint64_t)op.output.pointerIndex << 32) ^
                ((uint64_t)op.input.sliceTag << 24) ^
                ((uint64_t)op.output.sliceTag << 16) ^
                ((uint64_t)headStart << 8) ^
                ((uint64_t)kvStart << 4) ^
                (uint64_t)ffnStart ^
                refreshSalt;
        }
        refreshSalt += 0x9e3779b97f4a7c15ull;
    }
};

class BenchDevice : public NnDevice {
private:
    NnNodeConfig *nodeConfig;

public:
    explicit BenchDevice(NnNodeConfig *nodeConfig) : nodeConfig(nodeConfig) {}

    NnUint maxNThreads() override {
        return 1u;
    }

    NnDeviceSegment *createSegment(NnUint segmentIndex) override {
        return new BenchDeviceSegment(&nodeConfig->segments[segmentIndex]);
    }
};

struct BenchArgs {
    int enabledIterations = 1000;
    int operatorIterations = 300;
    int weightIterations = 30;
    int warmup = 10;
    int segments = 4;
    int opsPerSegment = 64;
    int repeats = 2;
    std::string outDir = "artifacts/pointer_swizzling_microbench";
};

struct ModeStats {
    std::string mode;
    int repeat = 0;
    int iterations = 0;
    double avg = 0.0;
    double median = 0.0;
    double p99 = 0.0;
    double total = 0.0;
    double min = 0.0;
    double max = 0.0;
    uint64_t materializedBytes = 0u;
    uint64_t bindingUpdates = 0u;
    std::string logPath;
};

static int parseIntArg(const char *s, const char *name) {
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || v < 0) {
        throw std::runtime_error(std::string("Invalid ") + name + ": " + s);
    }
    return (int)v;
}

static BenchArgs parseArgs(int argc, char **argv) {
    BenchArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string k(argv[i]);
        auto needValue = [&](const char *name) -> const char * {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++i];
        };
        if (k == "--enabled-iters") args.enabledIterations = parseIntArg(needValue("--enabled-iters"), "--enabled-iters");
        else if (k == "--operator-iters") args.operatorIterations = parseIntArg(needValue("--operator-iters"), "--operator-iters");
        else if (k == "--weight-iters") args.weightIterations = parseIntArg(needValue("--weight-iters"), "--weight-iters");
        else if (k == "--warmup") args.warmup = parseIntArg(needValue("--warmup"), "--warmup");
        else if (k == "--segments") args.segments = parseIntArg(needValue("--segments"), "--segments");
        else if (k == "--ops-per-segment") args.opsPerSegment = parseIntArg(needValue("--ops-per-segment"), "--ops-per-segment");
        else if (k == "--repeats") args.repeats = parseIntArg(needValue("--repeats"), "--repeats");
        else if (k == "--out-dir") args.outDir = needValue("--out-dir");
        else if (k == "--help" || k == "-h") {
            std::printf(
                "Usage: pointer-swizzling-microbench [--out-dir DIR] [--repeats N]\n"
                "       [--enabled-iters N] [--operator-iters N] [--weight-iters N]\n"
                "       [--warmup N] [--segments N] [--ops-per-segment N]\n");
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + k);
        }
    }
    if (args.segments < 1 || args.opsPerSegment < 1 || args.repeats < 1) {
        throw std::runtime_error("segments, ops-per-segment and repeats must be positive");
    }
    return args;
}

static char *dupName(const std::string &s) {
    char *out = new char[s.size() + 1u];
    std::copy(s.begin(), s.end(), out);
    out[s.size()] = '\0';
    return out;
}

static NnSize3D benchSize(NnFloatType type, NnUint z, NnUint y, NnUint x) {
    const NnSize lenXY = (NnSize)y * (NnSize)x;
    const NnSize len = (NnSize)z * lenXY;
    return NnSize3D{type, z, y, x, len, getBytes(type, len), getBytes(type, lenXY)};
}

static NnPointerConfig ptr(NnPointerSource source, NnUint index, NnPointerType type, NnSliceTag tag) {
    NnPointerConfig p;
    p.source = source;
    p.pointerIndex = index;
    p.type = type;
    p.sliceTag = tag;
    return p;
}

static void allocSplit(NnDimSplit &split, NnUint nNodes, NnUint length) {
    split.starts = new NnUint[nNodes];
    split.lengths = new NnUint[nNodes];
    for (NnUint i = 0; i < nNodes; ++i) {
        split.starts[i] = 0u;
        split.lengths[i] = length;
    }
}

static void initPlan(NnUnevenPartitionPlan &plan, NnUint nHeads, NnUint nKvHeads, NnUint ffnDim, NnUint dim) {
    plan.nNodes = 1u;
    plan.nStages = 1u;
    plan.stages = new NnStageConfig[1];
    plan.stages[0].stageIndex = 0u;
    plan.stages[0].startLayer = 0u;
    plan.stages[0].endLayer = 2u;
    plan.stages[0].nLayers = 2u;
    plan.stages[0].rootNodeIndex = 0u;
    plan.stages[0].nNodes = 1u;
    plan.stages[0].nodeIndices = new NnUint[1];
    plan.stages[0].nodeIndices[0] = 0u;
    allocSplit(plan.headSplit, 1u, nHeads);
    allocSplit(plan.kvHeadSplit, 1u, nKvHeads);
    allocSplit(plan.kvHeadComputeSplit, 1u, nKvHeads);
    allocSplit(plan.vocabSplit, 1u, 32000u);
    allocSplit(plan.ffnSplit, 1u, ffnDim);
    allocSplit(plan.dimSplit, 1u, dim);
}

static void makeConfigs(
    int nSegments,
    int opsPerSegment,
    NnNetConfig &netConfig,
    NnNodeConfig &nodeConfig,
    std::vector<NnPipeConfig> &pipes,
    std::vector<NnBufferConfig> &buffers,
    std::vector<NnSegmentConfig> &segments,
    std::vector<std::vector<NnOpConfig>> &ops,
    std::vector<std::string> &names) {
    const NnUint dim = 2560u;
    const NnUint ffn = 6912u;
    const NnUint nHeads = 32u;
    const NnUint nKvHeads = 8u;
    (void)nHeads;
    (void)nKvHeads;

    pipes.resize(1);
    pipes[0].name = dupName("bench_pipe");
    pipes[0].size = benchSize(F_32, 1u, 1u, dim);

    buffers.resize(3);
    buffers[0].name = dupName("bench_x");
    buffers[0].size = benchSize(F_32, 1u, 1u, dim);
    buffers[1].name = dupName("bench_y");
    buffers[1].size = benchSize(F_32, 1u, 1u, dim);
    buffers[2].name = dupName("bench_ffn");
    buffers[2].size = benchSize(F_32, 1u, 1u, ffn);

    segments.resize((size_t)nSegments);
    ops.resize((size_t)nSegments);
    names.reserve((size_t)nSegments * (size_t)opsPerSegment);

    for (int s = 0; s < nSegments; ++s) {
        ops[(size_t)s].resize((size_t)opsPerSegment);
        for (int i = 0; i < opsPerSegment; ++i) {
            const bool isFfn = (s % 2) == 1;
            const bool isQkv = (i % 4) != 3;
            std::string name = isFfn
                ? "block_matmul_w" + std::to_string((i % 3) + 1) + "_layer_" + std::to_string(s / 2)
                : (isQkv ? "block_matmul_q_layer_" : "block_multihead_att_layer_") + std::to_string(s / 2);
            names.push_back(name);
            NnOpConfig &op = ops[(size_t)s][(size_t)i];
            op.code = isFfn ? OP_MATMUL : (isQkv ? OP_MATMUL : OP_MULTIHEAD_ATT);
            op.name = const_cast<char *>(names.back().c_str());
            op.index = (NnUint)(s / 2);
            op.input = ptr(SRC_BUFFER, (NnUint)(i % buffers.size()), PNTR_BATCHED_SLICE, isFfn ? NN_SLICE_FFN : NN_SLICE_HEAD);
            op.output = ptr(SRC_BUFFER, (NnUint)((i + 1) % buffers.size()), PNTR_BATCHED_SLICE, isFfn ? NN_SLICE_DIM : NN_SLICE_KV_HEAD);
            op.weightSize = isFfn ? benchSize(F_16, ffn, dim, 1u) : benchSize(F_16, dim, dim / 4u, 1u);
            op.config = nullptr;
            op.configSize = 128u;
        }
        segments[(size_t)s].nOps = (NnUint)ops[(size_t)s].size();
        segments[(size_t)s].ops = ops[(size_t)s].data();
        segments[(size_t)s].nSyncs = 0u;
        segments[(size_t)s].syncs = nullptr;
    }

    netConfig.nBatches = 1u;
    netConfig.nNodes = 1u;
    netConfig.nPipes = (NnUint)pipes.size();
    netConfig.pipes = pipes.data();
    netConfig.nPreSyncs = 0u;
    netConfig.preSyncs = nullptr;

    nodeConfig.nodeIndex = 0u;
    nodeConfig.nBuffers = (NnUint)buffers.size();
    nodeConfig.buffers = buffers.data();
    nodeConfig.nSegments = (NnUint)segments.size();
    nodeConfig.segments = segments.data();
}

static std::vector<double> readBindTimes(const std::string &path, const std::string &mode) {
    std::ifstream in(path.c_str());
    if (!in.is_open()) throw std::runtime_error("Cannot open log: " + path);
    std::vector<double> out;
    std::string line;
    const std::string modeNeedle = "\"pointer_swizzling_mode\":\"" + mode + "\"";
    const std::string eventNeedle = "\"event_id\":\"binding_refresh\"";
    while (std::getline(in, line)) {
        if (line.find(eventNeedle) == std::string::npos || line.find(modeNeedle) == std::string::npos) continue;
        const std::string key = "\"t_bind_ms\":";
        size_t p = line.find(key);
        if (p == std::string::npos) continue;
        p += key.size();
        size_t e = p;
        while (e < line.size() && (std::isdigit(line[e]) || line[e] == '.' || line[e] == 'e' || line[e] == 'E' || line[e] == '+' || line[e] == '-')) ++e;
        out.push_back(std::strtod(line.substr(p, e - p).c_str(), nullptr));
    }
    return out;
}

static uint64_t readLastUintField(const std::string &path, const std::string &field) {
    std::ifstream in(path.c_str());
    uint64_t value = 0u;
    std::string line;
    const std::string key = "\"" + field + "\":";
    while (std::getline(in, line)) {
        size_t p = line.find(key);
        if (p == std::string::npos) continue;
        p += key.size();
        value = (uint64_t)std::strtoull(line.c_str() + p, nullptr, 10);
    }
    return value;
}

static ModeStats summarize(const std::string &mode, int repeat, int iterations, int warmup, const std::string &logPath) {
    std::vector<double> values = readBindTimes(logPath, mode);
    if ((int)values.size() < warmup + iterations) {
        throw std::runtime_error("Too few binding_refresh events for mode " + mode + ": " + std::to_string(values.size()));
    }
    values.erase(values.begin(), values.begin() + warmup);
    if ((int)values.size() > iterations) values.resize((size_t)iterations);

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    const size_t p99Index = std::min(sorted.size() - 1u, (size_t)std::ceil((double)sorted.size() * 0.99) - 1u);

    ModeStats st;
    st.mode = mode;
    st.repeat = repeat;
    st.iterations = iterations;
    st.total = total;
    st.avg = total / (double)values.size();
    st.median = sorted[sorted.size() / 2u];
    st.p99 = sorted[p99Index];
    st.min = sorted.front();
    st.max = sorted.back();
    st.materializedBytes = readLastUintField(logPath, "materialized_bytes");
    st.bindingUpdates = readLastUintField(logPath, "binding_update_count");
    st.logPath = logPath;
    return st;
}

static PointerSwizzlingMode modeEnum(const std::string &mode) {
    PointerSwizzlingMode out;
    if (!parsePointerSwizzlingMode(mode, out)) throw std::runtime_error("bad mode " + mode);
    return out;
}

static int iterationsForMode(const BenchArgs &args, const std::string &mode) {
    if (mode == "enabled") return args.enabledIterations;
    if (mode == "operator_rebuild") return args.operatorIterations;
    if (mode == "weight_rematerialize") return args.weightIterations;
    return 0;
}

static ModeStats runMode(const BenchArgs &args, const std::string &mode, int repeat) {
    std::vector<NnPipeConfig> pipes;
    std::vector<NnBufferConfig> buffers;
    std::vector<NnSegmentConfig> segments;
    std::vector<std::vector<NnOpConfig>> ops;
    std::vector<std::string> names;
    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;
    makeConfigs(args.segments, args.opsPerSegment, netConfig, nodeConfig, pipes, buffers, segments, ops, names);

    NnUnevenPartitionPlan planA;
    NnUnevenPartitionPlan planB;
    initPlan(planA, 32u, 8u, 6912u, 2560u);
    initPlan(planB, 32u, 8u, 6912u, 2560u);
    planB.headSplit.starts[0] = 1u;
    planB.kvHeadSplit.starts[0] = 1u;
    planB.ffnSplit.starts[0] = 64u;
    nodeConfig.partitionPlan = &planA;

    NnNetExecution execution(1u, &netConfig);
    std::vector<NnExecutorDevice> devices;
    devices.emplace_back(new BenchDevice(&nodeConfig), -1, -1);
    NnFakeNodeSynchronizer sync;
    NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &sync, false);

    const std::string logPath = args.outDir + "/pointer_" + mode + "_repeat" + std::to_string(repeat) + ".jsonl";
    std::remove(logPath.c_str());
    EdgeVisorAblationConfig cfg;
    cfg.pointerSwizzlingMode = modeEnum(mode);
    cfg.experimentId = "pointer_swizzling_microbench_r" + std::to_string(repeat);
    cfg.ablationLogPath = logPath;
    setEdgeVisorAblationConfig(cfg);

    const int iterations = iterationsForMode(args, mode);
    const int total = args.warmup + iterations;
    for (int i = 0; i < total; ++i) {
        executor.applyPartitionPlan((i % 2) == 0 ? &planB : &planA);
    }
    return summarize(mode, repeat, iterations, args.warmup, logPath);
}

static void writeCsv(const std::string &path, const std::vector<ModeStats> &stats) {
    std::ofstream out(path.c_str());
    out << "mode,repeat,iterations,avg_ms,median_ms,p99_ms,min_ms,max_ms,total_ms,binding_updates,materialized_bytes,log_path\n";
    for (const ModeStats &s : stats) {
        out << s.mode << ","
            << s.repeat << ","
            << s.iterations << ","
            << s.avg << ","
            << s.median << ","
            << s.p99 << ","
            << s.min << ","
            << s.max << ","
            << s.total << ","
            << s.bindingUpdates << ","
            << s.materializedBytes << ","
            << s.logPath << "\n";
    }
}

static void printModeLine(const ModeStats &s) {
    std::printf("%-22s repeat=%d avg=%9.4f ms median=%9.4f ms p99=%9.4f ms total=%10.2f ms\n",
                s.mode.c_str(), s.repeat, s.avg, s.median, s.p99, s.total);
}

int main(int argc, char **argv) {
    try {
        BenchArgs args = parseArgs(argc, argv);
        std::string mkdirCmd = "mkdir -p '" + args.outDir + "'";
        if (std::system(mkdirCmd.c_str()) != 0) {
            throw std::runtime_error("Failed to create output directory: " + args.outDir);
        }

        std::vector<ModeStats> stats;
        const std::vector<std::string> modes = {"enabled", "operator_rebuild", "weight_rematerialize"};
        std::printf("Pointer Swizzling Microbenchmark:\n");
        std::printf("  segments=%d ops_per_segment=%d warmup=%d repeats=%d\n",
                    args.segments, args.opsPerSegment, args.warmup, args.repeats);
        for (int r = 1; r <= args.repeats; ++r) {
            for (const std::string &mode : modes) {
                ModeStats st = runMode(args, mode, r);
                stats.push_back(st);
                printModeLine(st);
            }
        }

        const std::string csvPath = args.outDir + "/pointer_swizzling_microbench.csv";
        writeCsv(csvPath, stats);
        std::printf("CSV: %s\n", csvPath.c_str());

        double enabledAvg = 0.0;
        double opAvg = 0.0;
        double weightAvg = 0.0;
        int enabledN = 0, opN = 0, weightN = 0;
        for (const ModeStats &s : stats) {
            if (s.mode == "enabled") { enabledAvg += s.avg; ++enabledN; }
            else if (s.mode == "operator_rebuild") { opAvg += s.avg; ++opN; }
            else if (s.mode == "weight_rematerialize") { weightAvg += s.avg; ++weightN; }
        }
        enabledAvg /= std::max(1, enabledN);
        opAvg /= std::max(1, opN);
        weightAvg /= std::max(1, weightN);
        std::printf("Ratios: operator/enabled=%.1fx weight/enabled=%.1fx\n",
                    opAvg / std::max(enabledAvg, 1e-9), weightAvg / std::max(enabledAvg, 1e-9));
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
