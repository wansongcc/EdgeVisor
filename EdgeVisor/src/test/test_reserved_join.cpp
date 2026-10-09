// Model-backed acceptance entry point. Reuse the production CLI/decode loop,
// but let the test controller start a reserved worker after bootstrap and
// before the very first token. No timing sleeps are inserted in production.
#define main edgeVisorCliMain
#include "../dllama.cpp"
#undef main

#include <fstream>

static void awaitReservedWorker(AppInferenceContext *context) {
    if (planCommandCache().load().cmd.mode != PLAN_CMD_MODE_NONE)
        throw std::runtime_error("New context retained a migration from an old session");
    const char *ready = std::getenv("EDGEVISOR_TEST_READY_FILE");
    const char *go = std::getenv("EDGEVISOR_TEST_GO_FILE");
    if (ready == nullptr || go == nullptr)
        throw std::runtime_error("Reserved join test requires READY_FILE and GO_FILE");
    std::vector<int> tokens(std::strlen(context->args->prompt) + 3u);
    int nTokens = 0;
    context->tokenizer->encode(context->args->prompt, tokens.data(), &nTokens, true, true);
    if (nTokens != 1 || !promptLooksChatFormatted(context->args->prompt))
        throw std::runtime_error("Reserved join test requires one preformatted prompt token");
    {
        std::ofstream marker(ready);
        marker << "single-token context ready\n";
        if (!marker) throw std::runtime_error("Cannot write reserved join ready marker");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!std::ifstream(go).good()) {
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("Reserved join controller did not release decode");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::puts("[acceptance] stale command cleared; one prompt token; worker started before decode position 0");
    inference(context);
}

int main(int argc, char **argv) {
    initSockets();
    int result = EXIT_SUCCESS;
    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, true);
        PlanCommand stale = makeEmptyPlanCommand();
        stale.mode = PLAN_CMD_MODE_NEXT_BARRIER;
        stale.fromNodeIndex = 1u; stale.toNodeIndex = 0u;
        stale.triggerLayer = 14u; stale.reserved0 = 1u;
        planCommandCache().store(stale);
        runInferenceApp(&args, awaitReservedWorker);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Reserved join acceptance failed: %s\n", error.what());
        result = EXIT_FAILURE;
    }
    cleanupSockets();
    return result;
}
