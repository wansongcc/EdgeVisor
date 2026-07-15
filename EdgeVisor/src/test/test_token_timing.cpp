#include "token_timing.hpp"

#include <cassert>
#include <cstdlib>
#include <string>

static void testEnvGate() {
    unsetenv("DLLAMA_TOKEN_TIMING_PRINT");
    assert(!dllamaTokenTimingPrintEnabled());

    setenv("DLLAMA_TOKEN_TIMING_PRINT", "0", 1);
    assert(!dllamaTokenTimingPrintEnabled());

    setenv("DLLAMA_TOKEN_TIMING_PRINT", "false", 1);
    assert(!dllamaTokenTimingPrintEnabled());

    setenv("DLLAMA_TOKEN_TIMING_PRINT", "1", 1);
    assert(dllamaTokenTimingPrintEnabled());
}

static void testFormatRootLine() {
    const std::string line = formatTokenE2eTimingLine(42u, 1234, 18.7654);
    assert(line == "[token-e2e] pos=42 token=1234 wall=18.77ms");
}

static void testFormatNodeLine() {
    DllamaTokenNodeTiming timing{};
    timing.nodeIndex = 2u;
    timing.stageIndex = 1u;
    timing.hasStage = true;
    timing.execUs = 10110ull;
    timing.syncUs = 2200ull;
    timing.bubbleUs = 30ull;

    const std::string line = formatTokenNodeTimingLine(42u, timing);
    assert(line == "[token-prof] pos=42 node=2 stage=1 total=12.34ms exec=10.11ms sync=2.20ms bubble=0.03ms");
}

int main() {
    testEnvGate();
    testFormatRootLine();
    testFormatNodeLine();
    return 0;
}
