#include "nn/nn-network.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-executor.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

static unsigned calls;
static bool covered;
static void testBoundedFrameProbe(bool turbo) {
    int pair[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    std::vector<NnSocket> sockets; sockets.emplace_back(pair[0]);
    std::vector<NnUint> peers{1u};
    NnNetwork network(&sockets, &peers); network.resetStats(); network.setTurbo(turbo);
    const int header = 12345; int received = 0;
    std::thread sender([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        assert(send(pair[1], &header, sizeof(header), 0) == sizeof(header));
    });
    const auto start = std::chrono::steady_clock::now();
    // Frame pumps must never block awaiting a future token, including sockets
    // installed after initial setTurbo() and explicitly blocking connections.
    assert(!network.tryPeekWithMaxAttempts(0u, &received, sizeof(received), 1ul));
    assert(!network.tryReadWithMaxAttempts(0u, &received, sizeof(received), 1ul));
    assert(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(150));
    sender.join();
    assert(network.tryPeekWithMaxAttempts(0u, &received, sizeof(received), 1ul));
    assert(received == header);
    network.read(0u, &received, sizeof(received));
    assert(received == header);
    assert(!network.tryPeekWithMaxAttempts(0u, &received, sizeof(received), 1ul));
    close(pair[1]);
}
static bool bypass(NnUnevenPartitionPlan *plan, NnUint self, NnUint dead, bool replay) {
    assert(self == 0u && dead == 1u && replay);
    ++calls;
    return covered && applyPpStageBypass(plan, 1u, 0u);
}

// Reproduce a successful buffered PP send followed by a dead middle hop,
// while the root is waiting on a different, healthy tail socket.
static void testBufferedSend(bool cacheReady, bool turbo) {
    int middle[2], tail[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, middle) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tail) == 0);
    std::vector<NnSocket> sockets;
    sockets.reserve(2);
    sockets.emplace_back(middle[0]); sockets.emplace_back(tail[0]);
    std::vector<NnUint> peers{1u, 2u};
    NnNetwork network(&sockets, &peers);
    network.resetStats(); network.setTurbo(turbo);
    NnUnevenPartitionPlan plan;
    plan.nNodes = plan.nStages = 3u;
    plan.stages = new NnStageConfig[3];
    plan.ppPrevStageIndex = new NnUint[3]{(NnUint)-1, 0u, 1u};
    plan.ppNextStageIndex = new NnUint[3]{1u, 2u, (NnUint)-1};
    for (NnUint i = 0; i < 3u; ++i) {
        auto &s = plan.stages[i]; s.stageIndex = s.rootNodeIndex = i;
        s.startLayer = i; s.endLayer = i+1; s.nLayers = s.nNodes = 1;
        s.nodeIndices = new NnUint[1]{i};
    }
    int activation = 17, result = 0;
    network.write(0u, &activation, sizeof(activation));
    // Drain the buffered send, then close: the send has already succeeded.
    int discarded; assert(recv(middle[1], &discarded, sizeof(discarded), MSG_WAITALL) == sizeof(discarded));
    assert(discarded == activation); close(middle[1]);
    calls = 0u; covered = cacheReady; setNnPpFailoverHook(bypass);
    std::thread worker;
    if (cacheReady) worker = std::thread([&]() {
        int received;
        assert(recv(tail[1], &received, sizeof(received), MSG_WAITALL) == sizeof(received));
        assert(received == activation);
        const int answer = 12345;
        // Partial reads must preserve their position across recovery checks.
        assert(send(tail[1], &answer, 1, 0) == 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        assert(send(tail[1], (const char *)&answer+1, sizeof(answer)-1, 0) == sizeof(answer)-1);
    });
    NnSocketIo io{1u, &result, sizeof(result)};
    bool offline = false;
    try {
        network.readMany(1u, &io, [&]() {
            network.recoverPpIfNextOffline(&plan, 0u, (NnByte *)&activation, sizeof(activation));
        });
    } catch (const NnPeerOfflineException &e) { offline = true; assert(e.peerNodeIndex == 1u); }
    if (cacheReady) { worker.join(); assert(!offline && result == 12345); }
    else assert(offline);
    assert(calls == 1u);
    setNnPpFailoverHook(nullptr); close(tail[1]);
}
class CountingSegment : public NnDeviceSegment {
    unsigned &count;
public:
    explicit CountingSegment(unsigned &count) : count(count) {}
    void loadWeight(NnUint, NnSize, NnSize, NnByte *) override {}
    void forward(NnUint, NnUint, NnUint, NnUint) override { ++count; }
};
class CountingDevice : public NnDevice {
    unsigned &count;
public:
    explicit CountingDevice(unsigned &count) : count(count) {}
    NnUint maxNThreads() override { return 1u; }
    NnDeviceSegment *createSegment(NnUint) override { return new CountingSegment(count); }
};
static void testShadowOnce(const char *async) {
    setenv("DLLAMA_BUBBLE_SHADOW_KV", "1", 1);
    setenv("DLLAMA_BUBBLE_SHADOW_KV_ASYNC", async, 1);
    NnNetConfigBuilder netBuilder(1u, 1u);
    const NnUint pipe = netBuilder.addPipe("X", size2D(F_32, 1u, 1u));
    NnNodeConfigBuilder nodeBuilder(0u);
    NnSegmentConfigBuilder segment;
    segment.addOp(OP_CAST, "runtime_shadow_kv_att_in_right", 1u,
        pointerBatchConfig(SRC_PIPE, pipe), pointerBatchConfig(SRC_PIPE, pipe),
        size0(), NnCastOpCodeConfig{});
    nodeBuilder.addSegment(segment.build());
    NnNetConfig config = netBuilder.build();
    NnNodeConfig node = nodeBuilder.build();
    unsigned count = 0u;
    {
        NnNetExecution execution(1u, &config); execution.setBatchSize(1u);
        std::vector<NnExecutorDevice> devices;
        devices.emplace_back(new CountingDevice(count), -1, -1);
        NnFakeNodeSynchronizer sync;
        NnExecutor executor(&config, &node, &devices, &execution, &sync, false);
        assert(executor.getTotalTime(STEP_EXECUTE_OP) == 0u);
        for (NnUint pos = 0u; pos < 3u; ++pos) {
            execution.setPosition(pos); executor.forward();
            assert(count == pos+1u);
            auto stats = executor.runBubbleShadowRedundant(0u);
            assert(stats.completed && count == pos+1u);
            executor.runBubbleShadowRedundant(0u);
            assert(count == pos+1u && executor.shadowCovers(1u, 2u, pos+1u));
        }
    }
    releaseNetConfig(&config); releaseNodeConfig(&node);
    unsetenv("DLLAMA_BUBBLE_SHADOW_KV"); unsetenv("DLLAMA_BUBBLE_SHADOW_KV_ASYNC");
}
int main() {
    setenv("DLLAMA_IO_TIMEOUT_MS", "2000", 1);
    testBoundedFrameProbe(false); testBoundedFrameProbe(true);
    testBufferedSend(true, false); testBufferedSend(true, true);
    testBufferedSend(false, false); testBufferedSend(false, true);
    testShadowOnce("0"); testShadowOnce("1");
    std::puts("PP buffered-send recovery, uncovered rejection and once-per-token shadow KV passed");
}
