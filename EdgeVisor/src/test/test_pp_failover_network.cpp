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
static unsigned replayCalls;
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
        // fault 3 carries a sample for another position: it is skipped, so the
        // wait ends on the tail's EOF rather than on an invalid frame.
        assert(offline == ((!complete && !stalled) || fault == 3u));
        assert(timeout == stalled);
        assert(invalid == (fault == 4u));
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
    assert(self == 0u && dead == 1u);
    ++calls;
    if (replay) replayCalls += 1u;
    return covered && applyPpStageBypass(plan, 1u, 0u);
}

// A dead middle hop can have already forwarded the token, or not. The stage
// after it answers the question: HAVE relinks only, NEED replays once.
static void testBufferedSend(bool alreadyDelivered, bool turbo) {
    int middle[2], tail[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, middle) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tail) == 0);
    std::vector<NnSocket> sockets; sockets.reserve(2u);
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
    network.setCurrentPosition(0u);
    network.setPpPlan(&plan);
    if (alreadyDelivered) {
        network.write(0u, &activation, sizeof(activation));
        int consumed; assert(recv(middle[1], &consumed, sizeof(consumed), MSG_WAITALL) == sizeof(consumed));
        assert(consumed == activation);
    } else {
        // Not delivered: keep the bytes the dead hop never got, so the NEED
        // answer can replay exactly them.
        network.noteLastPpActivation(&activation, sizeof(activation));
    }
    close(middle[1]);
    calls = 0u; replayCalls = 0u; covered = true; setNnPpFailoverHook(bypass);
    std::thread worker;
    if (alreadyDelivered) worker = std::thread([&]() {
        // This stage already has the activation and is computing; delaying
        // the result does not authorize another copy on its control socket.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const int answer = 12345;
        assert(send(tail[1], &answer, sizeof(answer), 0) == sizeof(answer));
    });
    else worker = std::thread([&]() {
        // One DELIVERY_NEED frame: header plus three NnUint payload words.
        LlmWorkerFrameHeader frame{};
        frame.magic = LLM_WORKER_FRAME_MAGIC; frame.version = LLM_WORKER_FRAME_VERSION;
        frame.kind = LLM_WORKER_FRAME_DELIVERY_NEED;
        frame.payloadBytes = 3u * sizeof(NnUint);
        NnUint payload[3] = {NN_PP_DELIVERY_NEED_MAGIC, 0u, 1u};
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        assert(send(tail[1], &frame, sizeof(frame), 0) == sizeof(frame));
        assert(send(tail[1], payload, sizeof(payload), 0) == sizeof(payload));
        // The replay is exactly one activation value.
        int replayed = 0;
        assert(recv(tail[1], &replayed, sizeof(replayed), MSG_WAITALL) == sizeof(replayed));
        assert(replayed == activation);
        // The stage then finishes the token it was given: one result value.
        const int answer = 12345;
        assert(send(tail[1], &answer, sizeof(answer), 0) == sizeof(answer));
    });
    NnSocketIo io{1u, &result, sizeof(result)};
    bool offline = false;
    const auto start = std::chrono::steady_clock::now();
    try {
        network.readMany(1u, &io, [&]() { network.checkPpNextWhileWaiting(&plan, 0u); });
    } catch (const NnPeerOfflineException &e) { offline = true; assert(e.peerNodeIndex == 1u); }
    if (worker.joinable()) worker.join();
    assert(!offline);
    assert(calls == 1u);
    // Either answer skips the dead stage for later tokens.
    assert(getPpNextStageIndex(&plan, 0u) == 2u);
    if (alreadyDelivered) {
        // The delayed result stays readable and nothing was replayed.
        assert(result == 12345);
        assert(replayCalls == 0u);
        // The wait itself returned without writing anything to the tail.
    } else {
        // The replay let the stage finish, and its result arrived once.
        assert(result == 12345);
        assert(replayCalls == 1u);
    }
    int duplicate;
    assert(recv(tail[1], &duplicate, sizeof(duplicate), MSG_DONTWAIT) == -1);
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(2));
    setNnPpFailoverHook(nullptr); close(tail[1]);
}

// A frame already read into the root's caches is the other HAVE signal: the
// token reached the stage after the dead hop even though its socket is quiet.
static void testBufferedSendHaveFromCache(bool turbo) {
    int middle[2], tail[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, middle) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tail) == 0);
    std::vector<NnSocket> sockets; sockets.reserve(2u);
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
    int activation = 17;
    network.setCurrentPosition(4u);
    network.setPpPlan(&plan);
    network.noteLastPpActivation(&activation, sizeof(activation));
    close(middle[1]);
    bool haveCalled = false;
    network.setPpDeliveryHaveHook(
        [](void *ctx, NnUint downstream, NnUint position) -> bool {
            bool *called = (bool *)ctx;
            *called = true;
            return downstream == 2u && position == 4u;
        }, &haveCalled);
    calls = 0u; replayCalls = 0u; covered = true; setNnPpFailoverHook(bypass);
    const auto start = std::chrono::steady_clock::now();
    // Must return normally: HAVE relinks the plan and writes nothing.
    network.checkPpNextWhileWaiting(&plan, 0u);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    assert(haveCalled && calls == 1u && replayCalls == 0u);
    assert(getPpNextStageIndex(&plan, 0u) == 2u);
    assert(elapsed < 400);
    int duplicate;
    assert(recv(tail[1], &duplicate, sizeof(duplicate), MSG_DONTWAIT) == -1);
    network.setPpDeliveryHaveHook(nullptr, nullptr);
    setNnPpFailoverHook(nullptr); close(tail[1]);
}

// Silence after the wait is HAVE: the next stage is computing this token,
// so the plan relinks and nothing is written onto its socket.
static void testBufferedSendNoAnswer(bool turbo) {
    int middle[2], tail[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, middle) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tail) == 0);
    std::vector<NnSocket> sockets; sockets.reserve(2u);
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
    int activation = 17;
    network.setCurrentPosition(0u);
    network.setPpPlan(&plan);
    network.noteLastPpActivation(&activation, sizeof(activation));
    close(middle[1]);
    calls = 0u; replayCalls = 0u; covered = true; setNnPpFailoverHook(bypass);
    bool offline = false;
    const auto start = std::chrono::steady_clock::now();
    try {
        network.checkPpNextWhileWaiting(&plan, 0u);
    } catch (const NnPeerOfflineException &e) { offline = true; assert(e.peerNodeIndex == 1u); }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    assert(!offline);
    assert(calls == 1u && replayCalls == 0u);
    assert(getPpNextStageIndex(&plan, 0u) == 2u);
    int duplicate;
    assert(recv(tail[1], &duplicate, sizeof(duplicate), MSG_DONTWAIT) == -1);
    assert(elapsed >= 400 && elapsed < 2500);
    setNnPpFailoverHook(nullptr); close(tail[1]);
}

// A queued frame from an earlier token is not proof that this one arrived.
// The probe waits out the window, then relinks without replaying. The stale
// frame stays on the socket for the reader that owns it.
static void testBufferedSendStaleFrame(bool turbo) {
    int middle[2], tail[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, middle) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tail) == 0);
    std::vector<NnSocket> sockets; sockets.reserve(2u);
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
    int activation = 17;
    network.setCurrentPosition(9u);    network.setPpPlan(&plan);
    network.noteLastPpActivation(&activation, sizeof(activation));
    network.setPpFramePositionHook(
        [](void *, NnUint, NnUint kind, const void *payload, NnSize bytes, NnUint *position) -> bool {
            if (kind != LLM_WORKER_FRAME_PROFILE || bytes != sizeof(LlmPerfPacket)) return false;
            LlmPerfPacket pkt{};
            std::memcpy(&pkt, payload, sizeof(pkt));
            *position = pkt.position;
            return true;
        }, nullptr);
    // A profile for the previous token is already on the stream.
    LlmWorkerFrameHeader frame{};
    frame.magic = LLM_WORKER_FRAME_MAGIC; frame.version = LLM_WORKER_FRAME_VERSION;
    frame.kind = LLM_WORKER_FRAME_PROFILE; frame.payloadBytes = sizeof(LlmPerfPacket);
    LlmPerfPacket stale{}; stale.position = 8u; stale.nodeIndex = 2u;
    assert(send(tail[1], &frame, sizeof(frame), 0) == sizeof(frame));
    assert(send(tail[1], &stale, sizeof(stale), 0) == sizeof(stale));
    close(middle[1]);
    calls = 0u; replayCalls = 0u; covered = true; setNnPpFailoverHook(bypass);
    // Call the probe directly: readMany would consume the queued frame as the
    // result and never reach the wait path.
    bool offline = false;
    const auto start = std::chrono::steady_clock::now();
    try {
        network.checkPpNextWhileWaiting(&plan, 0u);
    } catch (const NnPeerOfflineException &e) { offline = true; assert(e.peerNodeIndex == 1u); }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    assert(!offline && calls == 1u && replayCalls == 0u);
    assert(getPpNextStageIndex(&plan, 0u) == 2u);
    assert(elapsed >= 400 && elapsed < 2500);
    // The stale frame was never consumed; the reader that owns it still can.
    LlmWorkerFrameHeader kept{};
    assert(network.tryPeekWithMaxAttempts(1u, &kept, sizeof(kept), 1ul));
    assert(kept.kind == LLM_WORKER_FRAME_PROFILE && kept.payloadBytes == sizeof(LlmPerfPacket));
    int duplicate;
    assert(recv(tail[1], &duplicate, sizeof(duplicate), MSG_DONTWAIT) == -1);
    setNnPpFailoverHook(nullptr); close(tail[1]);
}

// When the peer is dead before any bytes for this token are sent, the normal
// PP synchronizer can take over and send exactly once with ready shadow KV.
static void testBeforePpSend(bool cacheReady, bool turbo, bool alreadyInactive = false) {
    int middle[2], tail[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, middle) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tail) == 0);
    std::vector<NnSocket> sockets; sockets.reserve(2u);
    sockets.emplace_back(middle[0]); sockets.emplace_back(tail[0]);
    std::vector<NnUint> peers{1u, 2u};
    NnNetwork network(&sockets, &peers); network.resetStats(); network.setTurbo(turbo);
    NnUnevenPartitionPlan plan;
    plan.nNodes = plan.nStages = 3u; plan.stages = new NnStageConfig[3];
    plan.ppPrevStageIndex = new NnUint[3]{(NnUint)-1, 0u, 1u};
    plan.ppNextStageIndex = new NnUint[3]{1u, 2u, (NnUint)-1};
    for (NnUint i = 0u; i < 3u; ++i) {
        auto &s = plan.stages[i]; s.stageIndex = s.rootNodeIndex = i;
        s.startLayer = i; s.endLayer = i+1u; s.nLayers = s.nNodes = 1u;
        s.nodeIndices = new NnUint[1]{i};
    }
    NnNetConfigBuilder builder(3u, 1u);
    NnUint pipe = builder.addPipe("X", size2D(F_32, 1u, 1u));
    NnNodeConfigBuilder nodeBuilder(0u);
    NnSegmentConfigBuilder segment; segment.addSync(pipe, SYNC_PP_SEND);
    nodeBuilder.addSegment(segment.build());
    NnNetConfig config = builder.build(); NnNodeConfig node = nodeBuilder.build();
    {
        NnNetExecution execution(1u, &config); execution.setBatchSize(1u);
        *(float *)execution.pipes[pipe] = 17.0f;
        NnNetworkNodeSynchronizer synchronizer(&network, &execution, &config, &node, &plan, false);
        calls = 0u; covered = cacheReady; setNnPpFailoverHook(bypass);
        close(middle[1]);
        if (alreadyInactive) network.deactivateNode(1u, 0u);
        bool offline = false;
        try { synchronizer.sync(0u, 1u, 0u); }
        catch (const NnPeerOfflineException &e) { offline = true; assert(e.peerNodeIndex == 1u); }
        assert(offline == !cacheReady && calls == 1u);
        float received;
        if (cacheReady) {
            assert(recv(tail[1], &received, sizeof(received), MSG_WAITALL) == sizeof(received));
            assert(received == 17.0f);
        }
        assert(recv(tail[1], &received, sizeof(received), MSG_DONTWAIT) == -1);
    }
    setNnPpFailoverHook(nullptr); close(tail[1]);
    releaseNetConfig(&config); releaseNodeConfig(&node);
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
// The root is waiting on the tail, so a dead middle stage is not its next hop.
// The wait must still relink that stage once, without replaying.
static unsigned sideCalls = 0u;
static bool sideSawReplay = false;
static bool sideBypass(NnUnevenPartitionPlan *plan, NnUint self, NnUint dead, bool replay) {
    assert(self == 0u && dead == 2u);
    sideCalls += 1u;
    sideSawReplay = sideSawReplay || replay;
    return applyPpStageBypass(plan, 2u, 1u);
}
static void testDeadHopBeyondNext(bool turbo) {
    int hop1[2], hop2[2], hop3[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, hop1) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, hop2) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, hop3) == 0);
    std::vector<NnSocket> sockets;
    sockets.reserve(3u);
    sockets.emplace_back(hop1[0]);
    sockets.emplace_back(hop2[0]);
    sockets.emplace_back(hop3[0]);
    std::vector<NnUint> peers{1u, 2u, 3u};
    NnNetwork network(&sockets, &peers);
    network.resetStats();
    network.setTurbo(turbo);
    NnUnevenPartitionPlan plan;
    plan.nNodes = plan.nStages = 4u;
    plan.stages = new NnStageConfig[4];
    plan.ppPrevStageIndex = new NnUint[4]{(NnUint)-1, 0u, 1u, 2u};
    plan.ppNextStageIndex = new NnUint[4]{1u, 2u, 3u, (NnUint)-1};
    for (NnUint i = 0u; i < 4u; ++i) {
        auto &s = plan.stages[i];
        s.stageIndex = s.rootNodeIndex = i;
        s.startLayer = i;
        s.endLayer = i + 1u;
        s.nLayers = s.nNodes = 1u;
        s.nodeIndices = new NnUint[1]{i};
    }
    network.setCurrentPosition(0u);
    network.setPpPlan(&plan);
    close(hop2[1]);
    sideCalls = 0u;
    sideSawReplay = false;
    setNnPpFailoverHook(sideBypass);
    network.checkPpNextWhileWaiting(&plan, 0u);
    assert(sideCalls == 1u && !sideSawReplay);
    assert(getPpNextStageIndex(&plan, 1u) == 3u);
    assert(getPpPrevStageIndex(&plan, 3u) == 1u);
    network.checkPpNextWhileWaiting(&plan, 0u);
    assert(sideCalls == 1u && !sideSawReplay);
    setNnPpFailoverHook(nullptr);
    close(hop1[1]);
    close(hop3[1]);
}
int main() {
    setenv("DLLAMA_IO_TIMEOUT_MS", "2000", 1);
    // The delivery probe sizes its answer window from TCP_USER_TIMEOUT
    // (userMs + 1s). The 20s production default would blow the ~2s bound the
    // buffered-send cases assert; keep a short default and let an explicit
    // external value win.
    setenv("DLLAMA_TCP_USER_TIMEOUT_MS", "1000", 0);
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
    testBufferedSendHaveFromCache(false); testBufferedSendHaveFromCache(true);
    testBufferedSendNoAnswer(false); testBufferedSendNoAnswer(true);
    testBufferedSendStaleFrame(false); testBufferedSendStaleFrame(true);
    testBeforePpSend(true, false); testBeforePpSend(true, true);
    testBeforePpSend(false, false); testBeforePpSend(false, true);
    testBeforePpSend(true, false, true); testBeforePpSend(false, true, true);
    testDeadHopBeyondNext(false); testDeadHopBeyondNext(true);
    testShadowOnce("0"); testShadowOnce("1");
    std::puts("PP recovery, buffered tail frames, bounded timeouts, protocol rejection and shadow KV passed");
}
