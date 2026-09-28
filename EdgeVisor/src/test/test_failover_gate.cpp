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
    expect(!shadowHistoryCovers(filled, 4u, 1u, 3u, 13u), "restart when one layer is short");
    expect(!shadowHistoryCovers(filled, 4u, 3u, 4u, 12u), "restart when the only layer is short");
    expect(!shadowHistoryCovers(nullptr, 0u, 1u, 2u, 1u), "restart when shadow never ran");
    expect(shadowHistoryCovers(filled, 4u, 1u, 2u, 0u), "position 0 needs no history");
    expect(!shadowHistoryCovers(filled, 4u, 2u, 2u, 0u), "empty range is not a takeover");

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
