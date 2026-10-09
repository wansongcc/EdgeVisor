#include "nn/nn-network.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-executor.hpp"
#include "app.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

static unsigned calls;
static bool covered;

// A worker uninvolved in KV installation may still have profile/sample
// frames queued before the raw precommit ACK. Keep both frames for decode.
// fault: 1=no ACK, 2=partial frame header, 3=partial payload, 4=invalid ACK.
static void testControlAckFrames(bool turbo, unsigned fault = 0u) {
    unsetenv("DLLAMA_IO_TIMEOUT_MS");
    setenv("DLLAMA_KV_ACK_TIMEOUT_MS", "180000", 1);
    int pair[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    std::vector<NnSocket> sockets; sockets.emplace_back(pair[0]);
    std::vector<NnUint> peers{1u}; NnNetwork network(&sockets, &peers);
    std::vector<NnSocket> workerSockets; workerSockets.emplace_back(pair[1]);
    std::vector<NnUint> workerPeers{0u}; NnNetwork worker(&workerSockets, &workerPeers);
    network.resetStats(); worker.resetStats();
    network.setTurbo(turbo);
    NnUnevenPartitionPlan plan;
    plan.nNodes = plan.nStages = 2u; plan.stages = new NnStageConfig[2];
    for (NnUint i = 0u; i < 2u; ++i) {
        auto &s = plan.stages[i]; s.stageIndex = s.rootNodeIndex = i;
        s.startLayer = i; s.endLayer = i + 1u; s.nLayers = s.nNodes = 1u;
        s.nodeIndices = new NnUint[1]{i};
    }
    NnNetConfigBuilder builder(2u, 1u); builder.addPipe("X", size2D(F_32, 1u, 1u));
    LlmHeader header{}; header.nLayers = 2u; header.seqLen = 32u;
    LlmNet net{}; net.header = &header; net.netConfig = builder.build();
    net.nodeConfigs = new NnNodeConfig[2]{};
    {
        NnNetExecution execution(1u, &net.netConfig);
        RootLlmInference inference(&net, &execution, nullptr, &network, &plan, true, false);
        inference.setBatchSize(1u); inference.setPosition(7u);
        LlmWorkerFrameHeader frame{LLM_WORKER_FRAME_MAGIC, LLM_WORKER_FRAME_VERSION,
            LLM_WORKER_FRAME_PROFILE, sizeof(LlmPerfPacket)};
        LlmPerfPacket perf{}; perf.position = 7u; perf.nodeIndex = 1u;
        if (fault != 1u && fault != 4u) {
            const size_t headerBytes = fault == 2u ? sizeof(frame) / 2u : sizeof(frame);
            worker.write(0u, &frame, headerBytes);
            if (fault != 2u) worker.write(0u, &perf, fault == 3u ? sizeof(perf) / 2u : sizeof(perf));
        }
        if (fault == 0u) {
            LlmSampledTokenPacket sample{};
            sample.magic = LLM_SAMPLED_TOKEN_MAGIC; sample.version = LLM_SAMPLED_TOKEN_VERSION;
            sample.position = 7u; sample.nodeIndex = 1u; sample.token = 123u;
            frame.kind = LLM_WORKER_FRAME_SAMPLED_TOKEN; frame.payloadBytes = sizeof(sample);
            worker.write(0u, &frame, sizeof(frame)); worker.write(0u, &sample, sizeof(sample));
            worker.writeAck(0u);
        } else if (fault == 4u) {
            const NnUint invalid = 0u; worker.write(0u, &invalid, sizeof(invalid));
        }
        const auto start = std::chrono::steady_clock::now();
        bool timeout = false, invalid = false;
        try { inference.waitWorkerControlAck(0u, 100); }
        catch (const NnPeerTimeoutException &e) { timeout = true; assert(e.peerNodeIndex == 1u); }
        catch (const std::runtime_error &) { invalid = true; }
        assert(timeout == (fault >= 1u && fault <= 3u));
        assert(invalid == (fault == 4u));
        assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
        if (fault == 0u) {
            NnUint token = 0u; assert(inference.tryReceiveLastStageSampledToken(token, nullptr));
            assert(token == 123u);
            std::vector<LlmPerfPacket> profiles;
            inference.collectDeferredProfile(LlmPerfPacket{}, profiles);
            assert(profiles.size() == 2u && profiles[1].nodeIndex == 1u && profiles[1].position == 7u);
        }
    }
    delete[] net.nodeConfigs; releaseNetConfig(&net.netConfig);
    unsetenv("DLLAMA_KV_ACK_TIMEOUT_MS"); setenv("DLLAMA_IO_TIMEOUT_MS", "2000", 1);
}

// Exercise the real root frame reader without model weights or a GPU.
// fault: 1=stalled payload, 2=no response, 3=bad identity, 4=bad header,
// 5=profile collector observes EOF after caching a complete sample,
// 6=closed partial header, 7=stalled partial header.
static void testTailFrameClose(bool turbo, bool complete, bool profileFirst, unsigned fault = 0u) {
    const bool stalled = fault == 1u || fault == 2u || fault == 7u;
    setenv("DLLAMA_KV_ACK_TIMEOUT_MS", "100", 1);
    if (stalled) unsetenv("DLLAMA_IO_TIMEOUT_MS");
    int pair[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    std::vector<NnSocket> sockets; sockets.emplace_back(pair[0]);
    std::vector<NnUint> peers{1u};
    NnNetwork network(&sockets, &peers); network.resetStats(); network.setTurbo(turbo);
    NnUnevenPartitionPlan plan;
    plan.nNodes = plan.nStages = 2u; plan.stages = new NnStageConfig[2];
    for (NnUint i = 0u; i < 2u; ++i) {
        auto &s = plan.stages[i]; s.stageIndex = s.rootNodeIndex = i;
        s.startLayer = i; s.endLayer = i + 1u; s.nLayers = s.nNodes = 1u;
        s.nodeIndices = new NnUint[1]{i};
    }
    NnNetConfigBuilder builder(2u, 1u);
    builder.addPipe("X", size2D(F_32, 1u, 1u));
    LlmHeader header{}; header.nLayers = 2u; header.seqLen = 32u;
    LlmNet net{}; net.header = &header; net.netConfig = builder.build();
    net.nodeConfigs = new NnNodeConfig[2]{};
    {
        NnNetExecution execution(1u, &net.netConfig);
        RootLlmInference inference(&net, &execution, nullptr, &network, &plan, true, false);
        inference.setBatchSize(1u); inference.setPosition(7u);
        LlmWorkerFrameHeader frame{};
        frame.magic = LLM_WORKER_FRAME_MAGIC; frame.version = LLM_WORKER_FRAME_VERSION;
        if (profileFirst) {
            frame.kind = LLM_WORKER_FRAME_PROFILE; frame.payloadBytes = sizeof(LlmPerfPacket);
            LlmPerfPacket perf{}; perf.position = 7u; perf.nodeIndex = 1u;
            assert(send(pair[1], &frame, sizeof(frame), 0) == sizeof(frame));
            assert(send(pair[1], &perf, sizeof(perf), 0) == sizeof(perf));
        }
        frame.kind = LLM_WORKER_FRAME_SAMPLED_TOKEN; frame.payloadBytes = sizeof(LlmSampledTokenPacket);
        if (fault == 4u) frame.magic = 0u;
        LlmSampledTokenPacket sample{};
        sample.magic = LLM_SAMPLED_TOKEN_MAGIC; sample.version = LLM_SAMPLED_TOKEN_VERSION;
        sample.position = 7u; sample.nodeIndex = 1u; sample.token = 123u;
        if (fault == 3u) sample.position = 6u;
        const size_t bytes = complete ? sizeof(sample) : sizeof(sample) / 2u;
        if (fault != 2u) {
            const size_t headerBytes = fault >= 6u ? sizeof(frame) / 2u : sizeof(frame);
            assert(send(pair[1], &frame, headerBytes, 0) == (ssize_t)headerBytes);
            if (fault < 6u) assert(send(pair[1], &sample, bytes, 0) == (ssize_t)bytes);
        }
        if (!stalled) close(pair[1]);
        if (complete && !stalled) assert(!network.peerLooksOffline(1u));
        if (fault == 5u) {
            std::vector<LlmPerfPacket> perf;
            inference.collectDeferredProfile(LlmPerfPacket{}, perf);
            assert(!network.isSocketActive(0u));
        }
        NnUint token = 0u; bool offline = false, timeout = false, invalid = false;
        const auto start = std::chrono::steady_clock::now();
        try {
            assert(inference.tryReceiveLastStageSampledToken(token, nullptr));
            assert(complete && token == 123u && (fault == 0u || fault == 5u));
        } catch (const NnPeerOfflineException &e) { offline = true; assert(e.peerNodeIndex == 1u); }
        catch (const NnPeerTimeoutException &e) { timeout = true; assert(e.peerNodeIndex == 1u); }
        catch (const std::runtime_error &) { invalid = true; }
        assert(offline == (!complete && !stalled));
        assert(timeout == stalled);
        assert(invalid == (fault == 3u || fault == 4u));
        assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
        if (stalled) { assert(network.isSocketActive(0u)); close(pair[1]); }
        if (complete && !stalled && !invalid) {
            // The result belongs to this token even if EOF follows it. The next
            // position must discover the dead tail, never reuse this token.
            inference.setPosition(8u); offline = false;
            try { inference.tryReceiveLastStageSampledToken(token, nullptr); }
            catch (const NnPeerOfflineException &e) { offline = true; assert(e.peerNodeIndex == 1u); }
            assert(offline);
        }
    }
    delete[] net.nodeConfigs; releaseNetConfig(&net.netConfig);
    unsetenv("DLLAMA_KV_ACK_TIMEOUT_MS");
    setenv("DLLAMA_IO_TIMEOUT_MS", "2000", 1);
}
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
    setenv("DLLAMA_LAST_STAGE_SAMPLING", "1", 1);
    for (unsigned fault = 0u; fault <= 4u; ++fault) {
        testControlAckFrames(false, fault); testControlAckFrames(true, fault);
    }
    testTailFrameClose(false, true, false); testTailFrameClose(true, true, true);
    testTailFrameClose(false, false, false); testTailFrameClose(true, false, true);
    testTailFrameClose(false, false, false, 1u); testTailFrameClose(true, true, false, 2u);
    testTailFrameClose(false, true, false, 3u); testTailFrameClose(true, true, false, 4u);
    testTailFrameClose(true, true, false, 5u);
    testTailFrameClose(false, false, false, 6u); testTailFrameClose(true, false, false, 7u);
    testBoundedFrameProbe(false); testBoundedFrameProbe(true);
    testBufferedSend(true, false); testBufferedSend(true, true);
    testBufferedSend(false, false); testBufferedSend(false, true);
    testShadowOnce("0"); testShadowOnce("1");
    std::puts("PP recovery, buffered tail frames, bounded timeouts, protocol rejection and shadow KV passed");
}
