#include "token_timing.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>

bool dllamaTokenTimingPrintEnabled() {
    const char *v = std::getenv("DLLAMA_TOKEN_TIMING_PRINT");
    if (v == nullptr || v[0] == '\0') return false;
    if (std::strcmp(v, "0") == 0 ||
        std::strcmp(v, "false") == 0 ||
        std::strcmp(v, "False") == 0 ||
        std::strcmp(v, "off") == 0 ||
        std::strcmp(v, "OFF") == 0) {
        return false;
    }
    return true;
}

std::string formatTokenE2eTimingLine(unsigned int pos, int token, double wallMs) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "[token-e2e] pos=" << pos
        << " token=" << token
        << " wall=" << wallMs << "ms";
    return oss.str();
}

std::string formatTokenNodeTimingLine(unsigned int pos, const DllamaTokenNodeTiming &timing) {
    const double execMs = (double)timing.execUs / 1000.0;
    const double syncMs = (double)timing.syncUs / 1000.0;
    const double bubbleMs = (double)timing.bubbleUs / 1000.0;
    const double totalMs = execMs + syncMs + bubbleMs;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "[token-prof] pos=" << pos
        << " node=" << timing.nodeIndex;
    if (timing.hasStage) oss << " stage=" << timing.stageIndex;
    else oss << " stage=unknown";
    oss << " total=" << totalMs << "ms"
        << " exec=" << execMs << "ms"
        << " sync=" << syncMs << "ms"
        << " bubble=" << bubbleMs << "ms";
    return oss.str();
}
