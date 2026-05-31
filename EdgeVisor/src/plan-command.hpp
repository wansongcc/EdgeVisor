#ifndef PLAN_COMMAND_HPP
#define PLAN_COMMAND_HPP

#include <atomic>
#include <cstdint>
#include <cstring>

// PlanCommand: distributed online-migration command.
// - Produced by external controller (preferred) or legacy env fallback (deprecated).
// - Distributed root->worker via control-plane.
// - Consumed by OP_PLAN_BARRIER on stage-root, then broadcast via plan pipe.

static constexpr uint32_t DLLAMA_PLAN_CMD_MAGIC = 0x44434e50u; // 'PNCD' little-endian
static constexpr uint32_t DLLAMA_PLAN_CMD_VERSION_V1 = 1u;
static constexpr uint32_t DLLAMA_PLAN_CMD_VERSION_V2 = 2u;

// Fixed-capacity move list for PlanCommand v2.
// Runtime validation should still cap to <= 2 * stageNodes.
static constexpr uint32_t DLLAMA_PLAN_CMD_MAX_MOVES = 64u;

enum PlanCommandMode : uint32_t {
    PLAN_CMD_MODE_NONE = 0u,
    PLAN_CMD_MODE_EXACT = 1u,       // trigger only when (pos, layer) matches
    PLAN_CMD_MODE_NEXT_BARRIER = 2u // trigger on the next barrier in the stage
};

// cmdKind: keep compatibility with existing encoding in plan pipe.
// 1=headSplit, 2=ffnSplit, 3=both
enum PlanCommandKind : uint32_t {
    PLAN_CMD_KIND_HEAD = 1u,
    PLAN_CMD_KIND_FFN = 2u,
    PLAN_CMD_KIND_BOTH = 3u
};

#pragma pack(push, 1)
struct PlanMove {
    uint32_t fromNodeIndex;
    uint32_t toNodeIndex;

    // Units:
    // - headMove: heads (non-GQA) OR KV heads (GQA lockstep)
    // - ffnMove: FFN hidden units
    uint32_t headMove;
    uint32_t ffnMove;

    uint32_t cmdKind; // PlanCommandKind; optional hint
};

struct PlanCommand {
    uint32_t magic;   // DLLAMA_PLAN_CMD_MAGIC
    uint32_t version; // DLLAMA_PLAN_CMD_VERSION_V1 or _V2

    uint32_t seq;     // monotonic id assigned by issuer
    uint32_t mode;    // PlanCommandMode

    uint32_t stageIndex;   // UINT32_MAX = any stage
    uint32_t triggerPos;   // used only for EXACT
    uint32_t triggerLayer; // used only for EXACT

    uint32_t fromNodeIndex;
    uint32_t toNodeIndex;

    uint32_t cmdKind;      // PlanCommandKind
    uint32_t nHeadsToMove; // units: heads
    uint32_t nFfnToMove;   // units: FFN hidden units

    // v2 extension: move list.
    // If version==V2 and nMoves>0, prefer moves[] over the legacy single-edge fields above.
    uint32_t nMoves;
    PlanMove moves[DLLAMA_PLAN_CMD_MAX_MOVES];

    uint32_t reserved0;
};
#pragma pack(pop)

static inline PlanCommand makeEmptyPlanCommand() {
    PlanCommand c;
    std::memset(&c, 0, sizeof(c));
    c.magic = DLLAMA_PLAN_CMD_MAGIC;
    c.version = DLLAMA_PLAN_CMD_VERSION_V2;
    c.seq = 0u;
    c.mode = PLAN_CMD_MODE_NONE;
    c.stageIndex = 0xFFFFFFFFu;
    c.triggerPos = 0xFFFFFFFFu;
    c.triggerLayer = 0xFFFFFFFFu;
    c.fromNodeIndex = 0u;
    c.toNodeIndex = 0u;
    c.cmdKind = 0u;
    c.nHeadsToMove = 0u;
    c.nFfnToMove = 0u;
    c.nMoves = 0u;
    c.reserved0 = 0u;
    return c;
}

static inline bool isValidPlanCommandHeader(const PlanCommand &pc) {
    if (pc.magic != DLLAMA_PLAN_CMD_MAGIC) return false;
    if (pc.version != DLLAMA_PLAN_CMD_VERSION_V1 && pc.version != DLLAMA_PLAN_CMD_VERSION_V2) return false;
    return true;
}

static inline bool planCommandHasMoveList(const PlanCommand &pc) {
    return (pc.version == DLLAMA_PLAN_CMD_VERSION_V2) && (pc.nMoves != 0u);
}

struct PlanCommandSnapshot {
    uint64_t cacheSeq; // cache version for seqlock (not the cmd.seq)
    PlanCommand cmd;
};

class PlanCommandCache {
public:
    PlanCommandCache();

    PlanCommandSnapshot load() const;
    uint64_t store(const PlanCommand &cmd);

    // Clears current command (mode=NONE). Returns new cacheSeq.
    uint64_t clear();

    // Consume (clear) only if cacheSeq still matches.
    bool consumeIfCacheSeq(uint64_t cacheSeq);

private:
    mutable std::atomic<uint64_t> seq_; // seqlock version
    PlanCommand cmd_;
};

// Process-local singleton cache.
PlanCommandCache &planCommandCache();

#endif
