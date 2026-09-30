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

    expect(clampLayerCap(22u, freeBytes, shape) == 4u, "a known machine uses the smaller model cap");
    const unsigned long long roomy = (unsigned long long)((40.0 * perLayer) / 0.70);
    expect(clampLayerCap(8u, roomy, shape) == 8u, "a known machine keeps the table cap when memory is larger");

    std::vector<unsigned> ceilings;
    std::string relaxed;
    expect(relaxPackedRatios("1@22*1@14", 1u, &ceilings, &relaxed) && relaxed == "1@23*1@13",
        "a worker that cannot allocate gives one layer to the root");
    expect(relaxPackedRatios(relaxed, 1u, &ceilings, &relaxed) && relaxed == "1@24*1@12",
        "the same worker can only shrink further");
    std::vector<unsigned> rootCeilings;
    expect(relaxPackedRatios("1@22*1@14", 0u, &rootCeilings, &relaxed) && relaxed == "1@21*1@15",
        "a root allocation failure moves one layer to the other machine");
    expect(!relaxPackedRatios("1@2*1@1", 1u, &rootCeilings, &relaxed),
        "a one-layer slice is not shrunk");
    expect(startupAllocFailure("cudaMalloc(&devicePointer, bufferSize) failed: out of memory (2)"),
        "cuda alloc failure is retried");
    expect(!startupAllocFailure("The number of prompt tokens is greater than the number of steps"),
        "a prompt error is not an allocation failure");
    std::vector<unsigned> packed;
    expect(parsePackedCounts("1@22*1@14", &packed) && failedPackedNode("Socket closed", packed) == 1u,
        "a closed worker socket shrinks the worker with the most layers");

    std::vector<unsigned> primary;
    std::vector<unsigned> memoryCap;
    primary.push_back(22u);
    primary.push_back(6u);
    memoryCap.push_back(28u);
    memoryCap.push_back(28u);
    std::vector<BoundaryOverlap> overlaps = assignBoundaryOverlap(primary, memoryCap);
    expect(overlaps.size() == 1u && overlaps[0].leftHolds == 5u && overlaps[0].rightHolds == 21u,
        "spare memory fills both directions without emptying a stage");
    expect(formatBoundaryOverlap(overlaps) == "5,21", "overlap is encoded left,right");

    primary.clear();
    memoryCap.clear();
    primary.push_back(8u);
    primary.push_back(8u);
    memoryCap.push_back(8u);
    memoryCap.push_back(20u);
    overlaps = assignBoundaryOverlap(primary, memoryCap);
    expect(overlaps.size() == 1u && overlaps[0].leftHolds == 0u && overlaps[0].rightHolds == 7u,
        "a full small stage cannot receive copies, but its neighbor can");

    primary.clear();
    memoryCap.clear();
    primary.push_back(10u);
    primary.push_back(4u);
    primary.push_back(10u);
    memoryCap.push_back(14u);
    memoryCap.push_back(6u);
    memoryCap.push_back(14u);
    overlaps = assignBoundaryOverlap(primary, memoryCap);
    expect(overlaps.size() == 2u
        && overlaps[0].leftHolds == 3u && overlaps[0].rightHolds == 1u
        && overlaps[1].leftHolds == 1u && overlaps[1].rightHolds == 3u,
        "a middle stage splits its spare across both boundaries");

    ModelShape residentShape;
    residentShape.dim = 5120u;
    residentShape.hiddenDim = 17408u;
    residentShape.nHeads = 40u;
    residentShape.nKvHeads = 8u;
    residentShape.maxSeqLen = 40960u;
    residentShape.nLayers = 40u;
    residentShape.weightType = 2;
    const unsigned long long residentFree = 12ull << 30;
    const unsigned syncCap = estimateLayerCap(residentFree, residentShape);
    const unsigned residentCap = estimateResidentLayerCap(residentFree, residentShape);
    expect(residentCap < syncCap && residentCap >= 1u,
        "full F32 KV admits fewer resident layers than the sync-byte cap");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d device profile checks failed\n", g_failures);
        return 1;
    }
    std::printf("device profile checks passed\n");
    return 0;
}
