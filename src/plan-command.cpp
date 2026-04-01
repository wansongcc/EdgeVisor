#include "plan-command.hpp"

static PlanCommandCache g_planCmdCache;
static PlanApplySummaryCache g_planApplySummaryCache;
static std::atomic<uint32_t> g_planCmdPublishedSeq{0u};

PlanCommandCache::PlanCommandCache() {
    seq_.store(0u, std::memory_order_relaxed);
    cmd_ = makeEmptyPlanCommand();
}

PlanCommandSnapshot PlanCommandCache::load() const {
    PlanCommandSnapshot snap;
    while (true) {
        uint64_t s0 = seq_.load(std::memory_order_acquire);
        if (s0 & 1u) continue; // writer in progress
        PlanCommand c = cmd_;
        uint64_t s1 = seq_.load(std::memory_order_acquire);
        if (s0 == s1 && ((s1 & 1u) == 0u)) {
            snap.cacheSeq = s1;
            snap.cmd = c;
            return snap;
        }
    }
}

uint64_t PlanCommandCache::store(const PlanCommand &cmd) {
    // start write
    uint64_t s = seq_.fetch_add(1u, std::memory_order_acq_rel) + 1u; // make it odd
    (void)s;
    cmd_ = cmd;
    // end write
    return seq_.fetch_add(1u, std::memory_order_acq_rel) + 1u; // make it even
}

uint64_t PlanCommandCache::clear() {
    PlanCommand empty = makeEmptyPlanCommand();
    return store(empty);
}

bool PlanCommandCache::consumeIfCacheSeq(uint64_t cacheSeq) {
    // Try to clear only if nothing changed since we read.
    // This is best-effort; if a new command arrives concurrently, do nothing.
    uint64_t expected = cacheSeq;
    // Quick check: if current seq differs, can't consume.
    if (seq_.load(std::memory_order_acquire) != expected) return false;

    // Acquire write lock by making seq odd, but only if expected matches.
    if (!seq_.compare_exchange_strong(expected, expected + 1u, std::memory_order_acq_rel)) return false;
    cmd_ = makeEmptyPlanCommand();
    seq_.store(expected + 2u, std::memory_order_release);
    return true;
}

PlanCommandCache &planCommandCache() {
    return g_planCmdCache;
}

PlanApplySummaryCache::PlanApplySummaryCache() {
    seq_.store(0u, std::memory_order_relaxed);
    summary_ = makeEmptyPlanApplySummary();
}

PlanApplySummarySnapshot PlanApplySummaryCache::load() const {
    PlanApplySummarySnapshot snap;
    while (true) {
        uint64_t s0 = seq_.load(std::memory_order_acquire);
        if (s0 & 1u) continue;
        PlanApplySummary s = summary_;
        uint64_t s1 = seq_.load(std::memory_order_acquire);
        if (s0 == s1 && ((s1 & 1u) == 0u)) {
            snap.cacheSeq = s1;
            snap.summary = s;
            return snap;
        }
    }
}

uint64_t PlanApplySummaryCache::store(const PlanApplySummary &summary) {
    uint64_t s = seq_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    (void)s;
    summary_ = summary;
    return seq_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
}

uint64_t PlanApplySummaryCache::clear() {
    return store(makeEmptyPlanApplySummary());
}

PlanApplySummaryCache &planApplySummaryCache() {
    return g_planApplySummaryCache;
}

void setPlanCommandPublishedSeq(uint32_t seq) {
    g_planCmdPublishedSeq.store(seq, std::memory_order_release);
}

uint32_t getPlanCommandPublishedSeq() {
    return g_planCmdPublishedSeq.load(std::memory_order_acquire);
}
