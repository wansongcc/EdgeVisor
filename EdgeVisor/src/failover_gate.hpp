#ifndef FAILOVER_GATE_HPP
#define FAILOVER_GATE_HPP

// Fast-path takeover is allowed only when every stolen layer already has
// shadow KV for positions [0, position). A missing row falls back to a
// session restart. `filledThrough[layer]` is the exclusive end position
// written by shadow (or by the chained replay of that layer).
inline bool shadowHistoryCovers(const unsigned *filledThrough, unsigned nLayers,
                                unsigned begin, unsigned end, unsigned position) {
    if (begin >= end) return false;
    for (unsigned layer = begin; layer < end; ++layer) {
        const unsigned filled = (filledThrough != nullptr && layer < nLayers)
            ? filledThrough[layer] : 0u;
        if (filled < position) return false;
    }
    return true;
}

// Join connect/accept must fail closed instead of blocking the decode loop.
constexpr int kJoinConnectTimeoutMs = 5000;
constexpr int kJoinAcceptTimeoutMs = 5000;
// After a session drops, the root reloads the model before it dials again.
// 15s is shorter than a 14B load, so the worker exits and the second restart fails.
constexpr int kRootReconnectWaitMs = 120000;

#endif
