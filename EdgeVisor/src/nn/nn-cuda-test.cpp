#include "nn-cuda.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static void usage(const char *prog) {
    std::fprintf(stderr, "Usage: %s [--gpu-index <index>]\n", prog);
}

int main(int argc, char **argv) {
    NnUint gpuIndex = 0u;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu-index") == 0 && i + 1 < argc) {
            gpuIndex = (NnUint)std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    try {
        const int count = nnCudaDeviceCount();
        std::printf("CUDA devices: %d\n", count);
        nnCudaPrintDeviceInfo(gpuIndex);
        NnNetConfig netConfig{};
        NnNodeConfig nodeConfig{};
        NnCudaDevice device(gpuIndex, &netConfig, &nodeConfig, nullptr, nullptr);
        std::printf("CUDA device lifecycle: ok\n");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "CUDA test failed: %s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
