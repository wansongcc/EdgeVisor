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
