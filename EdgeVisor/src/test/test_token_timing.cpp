#include "token_timing.hpp"

#include <cassert>
#include <cstdlib>
#include <string>
#include <fstream>
#include <map>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>

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

static void testRawTrace(const char *executable) {
    char path[] = "/tmp/edgevisor-trace-test-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0); close(fd); unlink(path);
    setenv("EDGEVISOR_TRACE_PATH", path, 1);
    setenv("EDGEVISOR_RUN_ID", "trace-test", 1);
    experimentTraceNewContext();
    {
        ExperimentTraceSpan outer("generation");
        ExperimentTraceSpan inner("token", {{"position", 7u}});
        experimentTracePoint("token_emitted", {{"token", 42u}});
    }
    std::thread actor([] {
        ExperimentTraceSpan span("background");
        experimentTracePoint("observed");
    });
    actor.join();
    requireExperimentTraceHealthy();
    std::ifstream input(path);
    std::string line;
    uint64_t seq = 0u;
    std::map<uint64_t, nlohmann::json> spans;
    std::map<std::string, uint64_t> clocks;
    while (std::getline(input, line)) {
        const auto event = nlohmann::json::parse(line);
        assert(event["seq"].get<uint64_t>() == seq++);
        assert(event["run_id"] == "trace-test" && event["runtime_context_id"] == 1u);
        const std::string id = event["actor_id"].get<std::string>();
        const uint64_t time = event["ts_ns"].get<uint64_t>();
        assert(time >= clocks[id]); clocks[id] = time;
        const uint64_t span = event["span_id"].get<uint64_t>();
        if (event["type"] == "begin") spans[span] = event;
        if (event["type"] == "end") {
            assert(spans.count(span));
            assert(event["attributes"]["duration_ns"].get<uint64_t>() == time - spans[span]["ts_ns"].get<uint64_t>());
            spans.erase(span);
        }
    }
    assert(seq == 9u && spans.empty() && clocks.size() == 2u);
    // A fresh process must refuse to append to or truncate existing evidence.
    const pid_t child = fork(); assert(child >= 0);
    if (child == 0) { execl(executable, executable, "--trace-collision", nullptr); _exit(99); }
    int status; assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 17);
    input.close(); unlink(path);
    unsetenv("EDGEVISOR_TRACE_PATH"); unsetenv("EDGEVISOR_RUN_ID");
}

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "--trace-collision") {
        try { experimentTracePoint("must-refuse"); } catch (const std::runtime_error &) { return 17; }
        return 99;
    }
    testEnvGate();
    testFormatRootLine();
    testFormatNodeLine();
    testRawTrace(argv[0]);
    return 0;
}
