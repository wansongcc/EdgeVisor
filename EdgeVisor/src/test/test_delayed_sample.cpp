// Linux-only linker interception for a model-backed worker acceptance test.
// Production forward/profiling stays unchanged; only one sampled-token payload
// is delayed, after its frame header has already reached the root.
#include "app.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <thread>

extern "C" ssize_t __real_send(int, const void *, size_t, int);
extern "C" ssize_t __wrap_send(int socket, const void *data, size_t bytes, int flags) {
    static std::atomic<bool> delayed{false};
    const char *delay = std::getenv("EDGEVISOR_TEST_SAMPLE_DELAY_MS");
    const char *position = std::getenv("EDGEVISOR_TEST_SAMPLE_DELAY_POS");
    const char *repeat = std::getenv("EDGEVISOR_TEST_SAMPLE_DELAY_REPEAT");
    if (delay != nullptr && position != nullptr && bytes == sizeof(LlmSampledTokenPacket)) {
        LlmSampledTokenPacket packet{};
        std::memcpy(&packet, data, sizeof(packet));
        if (packet.magic == LLM_SAMPLED_TOKEN_MAGIC && packet.version == LLM_SAMPLED_TOKEN_VERSION &&
                packet.position >= (NnUint)std::stoul(position) &&
                (!delayed.exchange(true) || (repeat != nullptr && repeat[0] == '1'))) {
            const unsigned ms = (unsigned)std::stoul(delay);
            std::printf("[acceptance] delay sampled payload pos=%u ms=%u\n", (unsigned)packet.position, ms);
            std::fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }
    }
    return __real_send(socket, data, bytes, flags);
}
