#include "api-token-ledger.hpp"
#include <cassert>
int main() {
    ApiTokenLedger ledger;
    auto start = ledger.begin("a", {1, 2});
    assert(start.cachedTokens == 0 && start.lineage == 1);
    // Forward consumes the last prompt token; sampled 42 is pending, not KV.
    ledger.finish({1, 2});
    start = ledger.begin("a", {1, 2, 42, 3});
    assert(start.cachedTokens == 2 && start.lineage == 1 && start.resetReason.empty());
    ledger.finish({1, 2, 42});
    start = ledger.begin("a", {1, 2, 99, 3});
    assert(start.cachedTokens == 0 && start.lineage == 2 && start.resetReason == "token_prefix_changed");
    ledger.finish({1, 2});
    start = ledger.begin("b", {1, 2, 42});
    assert(start.cachedTokens == 0 && start.lineage == 3 && start.resetReason == "session_changed");
    ledger.finish({1, 2});
    start = ledger.begin("b", {1, 2}); // Repeating a prompt requires regeneration.
    assert(start.cachedTokens == 0 && start.lineage == 4);
    start = ledger.begin("", {7}); // One-token prompt has no prefill but is valid.
    assert(start.cachedTokens == 0 && start.resetReason == "session_not_provided");
    ledger.finish({7});
    start = ledger.begin("", {7, 42});
    assert(start.cachedTokens == 0);
    ledger.invalidate();
    start = ledger.begin("a", {7, 42});
    assert(start.cachedTokens == 0 && start.resetReason == "no_valid_history");
    return 0;
}
