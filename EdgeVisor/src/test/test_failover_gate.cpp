#include "failover_gate.hpp"

#include <cstdio>
#include <cstdlib>

static int failures = 0;

static void expect(bool cond, const char *name) {
    if (cond) {
        std::printf("ok %s\n", name);
        return;
    }
    std::printf("FAIL %s\n", name);
    failures += 1;
}

int main() {
    // Offline, shadow filled through the current position: fast path may run.
    // Token identity is required only for that path.
    const unsigned filled[] = {0u, 12u, 12u, 4u};
    expect(shadowHistoryCovers(filled, 4u, 1u, 3u, 12u), "takeover when history is full");
    expect(!shadowHistoryCovers(filled, 4u, 1u, 3u, 13u), "stop when one layer is short");
    expect(!shadowHistoryCovers(filled, 4u, 3u, 4u, 12u), "stop when the only layer is short");
    expect(!shadowHistoryCovers(nullptr, 0u, 1u, 2u, 1u), "stop when shadow never ran");
    expect(shadowHistoryCovers(filled, 4u, 1u, 2u, 0u), "position 0 needs no history");
    expect(!shadowHistoryCovers(filled, 4u, 2u, 2u, 0u), "empty range is not a takeover");

    // Previous stage keeps the dead stage's prefix, next stage keeps its suffix.
    const ShadowTakeoverSplit meet = bidirectionalShadowTakeover(10u, 18u, 3u, 5u);
    expect(meet.complete && meet.frontBegin == 10u && meet.frontEnd == 13u
        && meet.backBegin == 13u && meet.backEnd == 18u,
        "prefix and suffix that meet cover the stage");
    const ShadowTakeoverSplit overlap = bidirectionalShadowTakeover(10u, 18u, 5u, 5u);
    expect(overlap.complete && overlap.frontEnd == 15u && overlap.backBegin == 15u,
        "overlap stays with the previous stage");
    const ShadowTakeoverSplit gap = bidirectionalShadowTakeover(10u, 18u, 3u, 3u);
    expect(!gap.complete, "a gap between prefix and suffix is not a takeover");
    const ShadowTakeoverSplit onlyFront = bidirectionalShadowTakeover(10u, 18u, 8u, 0u);
    expect(onlyFront.complete && onlyFront.backBegin == onlyFront.backEnd,
        "a full prefix needs nothing from the next stage");
    const ShadowTakeoverSplit onlyBack = bidirectionalShadowTakeover(10u, 18u, 0u, 8u);
    expect(onlyBack.complete && onlyBack.frontBegin == onlyBack.frontEnd
        && onlyBack.backBegin == 10u && onlyBack.backEnd == 18u,
        "a full suffix needs nothing from the previous stage");
    expect(!bidirectionalShadowTakeover(4u, 4u, 1u, 1u).complete, "an empty stage is not a takeover");

    // Unknown times keep the overlap with the previous stage.
    const ShadowTakeoverSplit timedUnknown = bidirectionalShadowTakeover(10u, 18u, 5u, 5u, 0u, 30u);
    expect(timedUnknown.complete && timedUnknown.frontEnd == 15u && timedUnknown.backBegin == 15u,
        "a missing execution time leaves the overlap with the previous stage");

    // B owns [5, 9). A covers 5..7 (prefix 3). C covers 6..8 (suffix 3).
    // Layer 5 can only be A's. Layer 8 can only be C's. Layers 6 and 7 follow speed.
    const ShadowTakeoverSplit fasterPrev = bidirectionalShadowTakeover(5u, 9u, 3u, 3u, 10u, 30u);
    expect(fasterPrev.complete && fasterPrev.frontBegin == 5u && fasterPrev.frontEnd == 8u
        && fasterPrev.backBegin == 8u && fasterPrev.backEnd == 9u,
        "the faster previous stage takes the overlap");
    const ShadowTakeoverSplit fasterNext = bidirectionalShadowTakeover(5u, 9u, 3u, 3u, 30u, 10u);
    expect(fasterNext.complete && fasterNext.frontEnd == 6u && fasterNext.backBegin == 6u
        && fasterNext.backEnd == 9u,
        "the faster next stage takes the overlap");
    const ShadowTakeoverSplit even = bidirectionalShadowTakeover(5u, 9u, 3u, 3u, 20u, 20u);
    expect(even.complete && even.frontEnd == 7u && even.backBegin == 7u && even.backEnd == 9u,
        "equal times split the overlap and the odd layer stays with the previous stage");
    const ShadowTakeoverSplit oddOverlap = bidirectionalShadowTakeover(0u, 5u, 4u, 4u, 10u, 10u);
    expect(oddOverlap.complete && oddOverlap.frontEnd == 3u && oddOverlap.backBegin == 3u
        && oddOverlap.backEnd == 5u,
        "an odd overlap gives the extra layer to the previous stage when times match");
    const ShadowTakeoverSplit stillGap = bidirectionalShadowTakeover(5u, 9u, 2u, 1u, 10u, 10u);
    expect(!stillGap.complete, "execution times do not close a coverage gap");

    expect(!offlineProbeDue(0u, 10u), "position 0 is not a probe");
    expect(!offlineProbeDue(9u, 10u), "a down device is not probed between strides");
    expect(offlineProbeDue(10u, 10u) && offlineProbeDue(20u, 10u), "a down device is probed every stride");

    const RejoinCatchupStep close = planRejoinCatchup(90u, 100u, 10u, 16u);
    expect(close.seal && close.sendEnd == 100u, "a gap inside the threshold is copied and inserted");
    const RejoinCatchupStep behind = planRejoinCatchup(0u, 100u, 10u, 16u);
    expect(!behind.seal && behind.sendEnd == 16u, "a long gap copies one chunk and keeps decoding");
    const RejoinCatchupStep almost = planRejoinCatchup(0u, 11u, 10u, 16u);
    expect(almost.seal && almost.sendEnd == 11u, "one token past the threshold cannot shrink, so it inserts");
    const RejoinCatchupStep shrink = planRejoinCatchup(0u, 30u, 10u, 16u);
    expect(!shrink.seal && shrink.sendEnd == 16u, "a chunk stops short of the insert threshold");
    const RejoinCatchupStep ready = planRejoinCatchup(40u, 40u, 10u, 16u);
    expect(ready.seal && ready.sendEnd == 40u, "a filled history inserts without another copy");

    // A rejoin that cannot connect must time out, not wait forever.
    expect(kJoinConnectTimeoutMs > 0 && kJoinConnectTimeoutMs <= 15000, "join connect timeout is bounded");
    expect(kJoinAcceptTimeoutMs > 0 && kJoinAcceptTimeoutMs <= 15000, "join accept timeout is bounded");
    expect(kRootReconnectWaitMs >= 60000, "root reload has time to reconnect");

    if (failures != 0) {
        std::printf("%d failover gate checks failed\n", failures);
        return 1;
    }
    std::printf("failover gate checks passed\n");
    return 0;
}
