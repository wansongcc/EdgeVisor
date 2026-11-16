#ifndef FAILOVER_GATE_HPP
#define FAILOVER_GATE_HPP

// Fast-path takeover is allowed only when every stolen layer already has
// shadow KV for positions [0, position). A missing row stops the inference
// task. `filledThrough[layer]` is the exclusive end position written by
// shadow (or by the chained replay of that layer).
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

// Offline takeover splits one stage across its two neighbors. The previous
// stage holds a prefix of the dead stage, the next stage holds a suffix, and
// the two runs must meet. An overlap stays with the previous stage; the next
// stage runs only the remainder, which is inside its own suffix. A gap, or a
// missing neighbor that leaves one side at zero, is not a takeover.
struct ShadowTakeoverSplit {
    bool complete = false;
    unsigned frontBegin = 0u;
    unsigned frontEnd = 0u;
    unsigned backBegin = 0u;
    unsigned backEnd = 0u;
};

inline ShadowTakeoverSplit bidirectionalShadowTakeover(
    unsigned start, unsigned end, unsigned prefixLayers, unsigned suffixLayers) {
    ShadowTakeoverSplit split;
    if (start >= end) return split;
    const unsigned width = end - start;
    if (prefixLayers > width) prefixLayers = width;
    if (suffixLayers > width) suffixLayers = width;
    if (prefixLayers + suffixLayers < width) return split;
    split.complete = true;
    split.frontBegin = start;
    split.frontEnd = start + prefixLayers;
    split.backBegin = split.frontEnd;
    split.backEnd = end;
    return split;
}

// Join connect/accept must fail closed instead of blocking the decode loop.
constexpr int kJoinConnectTimeoutMs = 5000;
constexpr int kJoinAcceptTimeoutMs = 5000;
// After a session drops, the root reloads the model before it dials again.
// 15s is shorter than a 14B load, so the worker exits and the second restart fails.
constexpr int kRootReconnectWaitMs = 120000;

#endif
