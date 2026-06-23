#ifndef NN_TEST_UTILS_HPP
#define NN_TEST_UTILS_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace nn_test {

struct ErrorStats {
    float maxAbs;
    float maxRelative;
    std::size_t maxAbsIndex;
    std::size_t maxRelativeIndex;
};

inline void fillRandom(float *output, std::size_t count, std::uint32_t seed) {
    std::uint32_t state = seed != 0u ? seed : 0x9e3779b9u;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float unit = (float)(state >> 8) / 16777215.0f;
        output[i] = unit * 2.0f - 1.0f;
    }
}

inline ErrorStats errorStats(
    const float *actual,
    const float *expected,
    std::size_t count
) {
    ErrorStats stats{0.0f, 0.0f, 0u, 0u};
    for (std::size_t i = 0; i < count; ++i) {
        const float absError = std::fabs(actual[i] - expected[i]);
        const float relativeError = absError / (std::fabs(expected[i]) + 1e-12f);
        if (absError > stats.maxAbs) {
            stats.maxAbs = absError;
            stats.maxAbsIndex = i;
        }
        if (relativeError > stats.maxRelative) {
            stats.maxRelative = relativeError;
            stats.maxRelativeIndex = i;
        }
    }
    return stats;
}

inline void requireClose(
    const char *name,
    const float *actual,
    const float *expected,
    std::size_t count,
    float maxAbsError
) {
    const ErrorStats stats = errorStats(actual, expected, count);
    if (stats.maxAbs <= maxAbsError) return;

    std::fprintf(stderr,
        "FAIL %s: max_abs=%g at [%zu] actual=%g expected=%g; "
        "max_relative=%g at [%zu]\n",
        name,
        stats.maxAbs,
        stats.maxAbsIndex,
        actual[stats.maxAbsIndex],
        expected[stats.maxAbsIndex],
        stats.maxRelative,
        stats.maxRelativeIndex);
    std::exit(1);
}

} // namespace nn_test

#endif
