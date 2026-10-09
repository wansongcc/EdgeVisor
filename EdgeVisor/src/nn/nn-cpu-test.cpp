#include "nn-core.hpp"
#include "nn-config-builder.hpp"
#include "nn-cpu.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>

#define DIM 32
#define N_BATCHES 2

void buildConfig(NnNetConfig *netConfig, NnNodeConfig *nodeConfig) {
    NnUint nNodes = 1;
    NnNetConfigBuilder netBuilder(nNodes, N_BATCHES);
    NnUint xPipeIndex = netBuilder.addPipe("X", size2D(F_32, N_BATCHES, DIM));

    NnNodeConfigBuilder nodeBuilder(0);
    NnUint invRmsBufferIndex = nodeBuilder.addBuffer("inv_rms", size2D(F_32, N_BATCHES, 1));
    NnSegmentConfigBuilder segmentBuilder;
    segmentBuilder.addSync(xPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);

    segmentBuilder.addOp(OP_INV_RMS, "inv_rms", 0,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
        size0(),
        NnInvRmsOpConfig{1e-5f, 1});

    segmentBuilder.addOp(OP_RMS_NORM, "rms_norm", 0,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        size1D(F_32, DIM),
        NnRmsNormOpConfig{invRmsBufferIndex, 1});

    nodeBuilder.addSegment(segmentBuilder.build());

    *netConfig = netBuilder.build();
    *nodeConfig = nodeBuilder.build();
}

void print2D(const char *name, NnUint x, NnUint y, float *w) {
    for (NnUint i = 0; i < y; i++) {
        printf("%s[%d] = ", name, i);
        for (NnUint j = 0; j < x; j++)
            printf("%f ", w[i * x + j]);
        printf("\n");
    }
}

void testShiftedPpStartRoundTrip() {
    NnNetConfigBuilder netBuilder(1u, 1u);
    const NnUint xPipeIndex = netBuilder.addPipe("X", size2D(F_32, 1u, 1u));
    NnNodeConfigBuilder nodeBuilder(0u);
    const NnUint bufferIndex = nodeBuilder.addBuffer("scratch", size2D(F_32, 1u, 1u));

    NnSegmentConfigBuilder primary;
    primary.addOp(OP_INV_RMS, "block_matmul_q", 8u,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_BUFFER, bufferIndex),
        size0(), NnInvRmsOpConfig{1e-5f, 1u});
    nodeBuilder.addSegment(primary.build());

    NnSegmentConfigBuilder shifted;
    shifted.addOp(OP_INV_RMS, "runtime_shifted_pp_start_block_matmul_q", 8u,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_BUFFER, bufferIndex),
        size0(), NnInvRmsOpConfig{1e-5f, 1u});
    nodeBuilder.addSegment(shifted.build());

    NnNetConfig netConfig = netBuilder.build();
    NnNodeConfig nodeConfig = nodeBuilder.build();
    {
        NnNetExecution execution(1u, &netConfig);
        std::vector<NnExecutorDevice> devices;
        devices.emplace_back(new NnCpuDevice(&netConfig, &nodeConfig, &execution), -1, -1);
        NnFakeNodeSynchronizer synchronizer;
        NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);

        assert(executor.isSegmentEnabled(0u));
        assert(!executor.isSegmentEnabled(1u));
        executor.setShiftedPpStartLayerEnabled(8u, true);
        assert(!executor.isSegmentEnabled(0u));
        assert(executor.isSegmentEnabled(1u));
        executor.setShiftedPpStartLayerEnabled(8u, false);
        assert(executor.isSegmentEnabled(0u));
        assert(!executor.isSegmentEnabled(1u));
    }
    releaseNetConfig(&netConfig);
    releaseNodeConfig(&nodeConfig);
}

// Small graph with observable shared scratch: primary compute and network sync
// must finish before shadow work. Adjacent shadow layers need redundant replay,
// and a second call in the same token must not replay from an advanced cache.
void testSynchronousShadowKv() {
    struct State {
        unsigned primary = 0u, syncs = 0u, replay = 0u;
        unsigned cache = 0u, first = 0u, second = 0u;
    } state;
    class Segment : public NnDeviceSegment {
        State &state;
        NnUint index;
    public:
        Segment(State &state, NnUint index) : state(state), index(index) {}
        void loadWeight(NnUint, NnSize, NnSize, NnByte *) override {}
        void forward(NnUint, NnUint nThreads, NnUint, NnUint) override {
            assert(nThreads == 1u);
            if (index == 0u) { state.cache = 10u + state.primary++; }
            else if (index == 1u) { state.first = state.cache; }
            else if (index == 2u) { state.cache += 20u; ++state.replay; }
            else if (index == 3u) { state.second = state.cache; }
        }
    };
    class Device : public NnDevice {
        State &state;
    public:
        Device(State &state) : state(state) {}
        NnUint maxNThreads() override { return 2u; }
        NnDeviceSegment *createSegment(NnUint index) override { return new Segment(state, index); }
    };
    class Synchronizer : public NnNodeSynchronizer {
        State &state;
    public:
        Synchronizer(State &state) : state(state) {}
        void sync(NnUint, NnUint, NnUint) override {
            assert(state.replay == state.syncs); // shadow has not touched scratch
            assert(state.cache == 9u + state.primary);
            ++state.syncs;
        }
    };

    NnNetConfigBuilder netBuilder(1u, 1u);
    NnUint pipe = netBuilder.addPipe("X", size2D(F_32, 1u, 1u));
    NnNodeConfigBuilder nodeBuilder(0u);
    const char *names[] = {"block_matmul_q", "runtime_shadow_kv_copy",
        "runtime_redundant_block_matmul_q", "runtime_shadow_kv_copy"};
    const NnUint layers[] = {0u, 1u, 1u, 2u};
    for (NnUint i = 0u; i < 4u; ++i) {
        NnSegmentConfigBuilder segment;
        segment.addOp(OP_INV_RMS, names[i], layers[i],
            pointerBatchConfig(SRC_PIPE, pipe), pointerBatchConfig(SRC_PIPE, pipe),
            size0(), NnInvRmsOpConfig{1e-5f, 1u});
        if (i == 0u) segment.addSync(pipe, SYNC_NODE_SLICES_EXCEPT_ROOT);
        nodeBuilder.addSegment(segment.build());
    }
    NnNetConfig net = netBuilder.build();
    NnNodeConfig node = nodeBuilder.build();
    setenv("DLLAMA_BUBBLE_SHADOW_KV", "1", 1);
    unsetenv("DLLAMA_RUNTIME_REDUNDANT_SEG_ENABLED");
    for (const char *legacyAsync : {"0", "1"}) {
        setenv("DLLAMA_BUBBLE_SHADOW_KV_ASYNC", legacyAsync, 1);
        state = State{};
        NnNetExecution execution(1u, &net);
        execution.setBatchSize(1u);
        std::vector<NnExecutorDevice> devices;
        devices.emplace_back(new Device(state), -1, -1);
        Synchronizer synchronizer(state);
        NnExecutor executor(&net, &node, &devices, &execution, &synchronizer, false);
        for (NnUint position = 0u; position < 3u; ++position) {
            execution.setPosition(position);
            executor.forward();
            assert(state.first == 10u + position && state.second == 30u + position);
            assert(state.primary == position + 1u && state.replay == position + 1u);
            assert(executor.shadowCovers(1u, 3u, position + 1u));
            assert(!executor.shadowCovers(1u, 3u, position + 2u));
            for (unsigned repeat = 0u; repeat < 2u; ++repeat) {
                NnBubbleShadowStats stats = executor.runBubbleShadowRedundant(0u);
                assert(stats.completed && stats.opStepsExecuted == 2u);
                assert(state.first == 10u + position && state.second == 30u + position);
                assert(state.replay == position + 1u);
            }
        }
    }
    {
        NnNetExecution execution(2u, &net);
        std::vector<NnExecutorDevice> devices;
        devices.emplace_back(new Device(state), -1, -1);
        Synchronizer synchronizer(state);
        bool rejected = false;
        try { NnExecutor executor(&net, &node, &devices, &execution, &synchronizer, false); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected); // embedding callers cannot silently skip shadow work
    }
    unsetenv("DLLAMA_BUBBLE_SHADOW_KV");
    unsetenv("DLLAMA_BUBBLE_SHADOW_KV_ASYNC");
    releaseNetConfig(&net);
    releaseNodeConfig(&node);
}

int main() {
    initQuants();
    testShiftedPpStartRoundTrip();
    testSynchronousShadowKv();

    NnUint nThreads = 2;
    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;
    buildConfig(&netConfig, &nodeConfig);

    NnNetExecution execution(nThreads, &netConfig);
    float *x = (float *)execution.pipes[0];
    for (NnUint b = 0; b < N_BATCHES; b++) {
        for (NnUint i = 0; i < DIM; i++)
            x[b * DIM + i] = i / (float)DIM + (float)b;
    }

    print2D("x", DIM, N_BATCHES, x);

    float rmsNormWeight[DIM];
    for (NnUint i = 0; i < DIM; i++)
        rmsNormWeight[i] = 0.5 + i / (float)DIM;

    NnCpuDevice *device = new NnCpuDevice(&netConfig, &nodeConfig, &execution);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(device, -1, -1));

    NnFakeNodeSynchronizer synchronizer;
    float *rms = (float *)device->buffers[0];
    NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);
    executor.loadWeight("rms_norm", 0u, 0u, sizeof(rmsNormWeight), (NnByte *)rmsNormWeight);

    execution.setBatchSize(2);
    executor.forward();

    print2D("rms", N_BATCHES, 1, rms);
    print2D("x", DIM, N_BATCHES, x);

    releaseNetConfig(&netConfig);
    releaseNodeConfig(&nodeConfig);
    return 0;
}
