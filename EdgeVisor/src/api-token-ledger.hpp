#ifndef API_TOKEN_LEDGER_HPP
#define API_TOKEN_LEDGER_HPP
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

// A sampled output is not KV until a subsequent forward consumes it. Reuse
// requires an explicit session and an exact prefix of the full canonical prompt.
class ApiTokenLedger {
public:
    struct Start {
        size_t cachedTokens;
        uint64_t lineage;
        std::string resetReason;
    };
    Start begin(const std::string &session, const std::vector<int> &prompt) {
        std::string reason;
        if (session.empty()) reason = "session_not_provided";
        else if (!valid_) reason = "no_valid_history";
        else if (session != session_) reason = "session_changed";
        else if (consumed_.size() >= prompt.size() ||
                 !std::equal(consumed_.begin(), consumed_.end(), prompt.begin())) reason = "token_prefix_changed";
        if (!reason.empty()) {
            invalidate();
            session_ = session;
            ++lineage_;
        }
        return {consumed_.size(), lineage_, reason};
    }
    void finish(const std::vector<int> &consumed) { consumed_ = consumed; valid_ = true; }
    void invalidate() { consumed_.clear(); valid_ = false; }
private:
    std::vector<int> consumed_;
    std::string session_;
    uint64_t lineage_ = 0;
    bool valid_ = false;
};
#endif
