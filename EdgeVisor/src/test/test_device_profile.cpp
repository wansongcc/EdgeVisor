#include "device_profile.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int g_failures = 0;

static void expect(bool ok, const char *message) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", message);
    g_failures += 1;
}

int main() {
    SpeedDeviceProfile known;
    expect(knownSpeedProfile("192.168.137.13", &known) && std::strcmp(known.name, "nx") == 0 && known.cap == 22u,
        "nx1 stays on the lab table");
    expect(knownSpeedProfile("192.168.137.15", &known) && known.cap == 22u, "nx2 stays on the lab table");
    expect(knownSpeedProfile("192.168.137.18", &known) && known.cap == 8u && std::strcmp(known.name, "nano") == 0,
        "nano cap stays 8");
    expect(knownSpeedProfile("192.168.137.31", &known) && known.cap == 11u, "laptop stays on the lab table");
    expect(!knownSpeedProfile("10.0.0.8", &known), "an unknown address is not in the lab table");

    ModelShape shape;
    shape.dim = 512;
    shape.hiddenDim = 512;
    shape.nHeads = 8;
    shape.nKvHeads = 8;
    shape.maxSeqLen = 0;
    shape.nLayers = 40;
    shape.weightType = 2;
    const unsigned long long params = 7ull * 512ull * 512ull;
    const double perLayer = (double)params * (18.0 / 32.0) + (2.0 * 2048.0 * 512.0 * (34.0 / 32.0));
    const unsigned long long freeBytes = (unsigned long long)((5.5 * perLayer) / 0.70);
    expect(estimateLayerCap(freeBytes, shape) == 4u, "cap keeps one layer of headroom");
    expect(estimateLayerCap(1ull, shape) == 1u, "a tiny memory budget still keeps one layer");

    std::vector<SpeedDeviceProfile> profiles;
    profiles.push_back(makeSpeedProfile(5.0, 20u, "fast"));
    profiles.push_back(makeSpeedProfile(10.0, 4u, "slow"));
    const std::string ratios = assignSpeedPack(profiles, 30u, nullptr);
    expect(ratios == "1@26*1@4", "faster device fills to its cap, then takes the leftover");

    std::vector<char> live(2, 1);
    live[1] = 0;
    const std::string solo = assignSpeedPack(profiles, 10u, &live);
    expect(solo == "1@10*1@0", "an offline device keeps its slot and receives no layers");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d device profile checks failed\n", g_failures);
        return 1;
    }
    std::printf("device profile checks passed\n");
    return 0;
}
