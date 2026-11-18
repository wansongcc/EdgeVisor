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
// the two runs must meet. A gap, or a missing neighbor that leaves one side
// at zero, is not a takeover. Layers only one neighbor covers stay with that
// neighbor. The overlap is split by execution time when both times are known:
// the faster device (smaller milliseconds) takes more of it. Unknown or zero
// times keep the overlap with the previous stage.
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

// prevExecMs / nextExecMs are the neighbors' last stage execution times.
// Zero means the time is unknown, so the overlap stays with the previous stage.
inline ShadowTakeoverSplit bidirectionalShadowTakeover(
    unsigned start, unsigned end, unsigned prefixLayers, unsigned suffixLayers,
    unsigned prevExecMs, unsigned nextExecMs) {
    if (prevExecMs == 0u || nextExecMs == 0u)
        return bidirectionalShadowTakeover(start, end, prefixLayers, suffixLayers);
    ShadowTakeoverSplit split;
    if (start >= end) return split;
    const unsigned width = end - start;
    if (prefixLayers > width) prefixLayers = width;
    if (suffixLayers > width) suffixLayers = width;
    if (prefixLayers + suffixLayers < width) return split;
    const unsigned exclusiveFront = width - suffixLayers;
    const unsigned overlap = prefixLayers + suffixLayers - width;
    const unsigned long long den =
        (unsigned long long)prevExecMs + (unsigned long long)nextExecMs;
    const unsigned long long num = (unsigned long long)overlap * (unsigned long long)nextExecMs;
    unsigned toPrev = (unsigned)(num / den);
    const unsigned long long twiceRem = 2ull * (num % den);
    // Exact halves go to the faster side, and to the previous stage when the
    // two times are equal, so the cut is stable across every process.
    if (toPrev < overlap && (twiceRem > den || (twiceRem == den && prevExecMs <= nextExecMs)))
        toPrev += 1u;
    split.complete = true;
    split.frontBegin = start;
    split.frontEnd = start + exclusiveFront + toPrev;
    split.backBegin = split.frontEnd;
    split.backEnd = end;
    return split;
}

// A down device is probed once per stride tokens, not on every token.
inline bool offlineProbeDue(unsigned position, unsigned stride) {
    return position > 0u && stride > 0u && (position % stride) == 0u;
}

// While the returning device is more than `threshold` tokens behind, copy at
// most `chunk` positions and keep decoding. The copy stops short of the
// threshold so the pipeline is not paused for the whole backlog. Once the gap
// is within the threshold (or one token past it, which cannot shrink without
// crossing it), copy the tail and insert.
struct RejoinCatchupStep {
    unsigned sendEnd = 0u;
    bool seal = false;
};

inline RejoinCatchupStep planRejoinCatchup(
    unsigned filled, unsigned position, unsigned threshold, unsigned chunk) {
    RejoinCatchupStep step;
    step.sendEnd = filled;
    if (position < filled) return step;
    if (position == filled) {
        step.seal = true;
        return step;
    }
    if (chunk < 2u) chunk = 2u;
    const unsigned gap = position - filled;
    if (gap <= threshold) {
        step.sendEnd = position;
        step.seal = true;
        return step;
    }
    unsigned send = chunk;
    if (send > gap - threshold) send = gap - threshold;
    // A one-position copy loses to the token the pipeline is about to run.
    if (send < 2u) {
        step.sendEnd = position;
        step.seal = true;
        return step;
    }
    step.sendEnd = filled + send;
    step.seal = false;
    return step;
}

constexpr unsigned kRejoinProbeStride = 10u;
constexpr unsigned kRejoinCatchupTokens = 10u;
constexpr unsigned kRejoinCatchupChunk = 16u;

// Join connect/accept must fail closed instead of blocking the decode loop.
constexpr int kJoinConnectTimeoutMs = 5000;
constexpr int kJoinAcceptTimeoutMs = 5000;
// After a session drops, the root reloads the model before it dials again.
// 15s is shorter than a 14B load, so the worker exits and the second restart fails.
constexpr int kRootReconnectWaitMs = 120000;

#endif
