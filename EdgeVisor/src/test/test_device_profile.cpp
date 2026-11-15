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
    // The slow device caps at 4 layers, the fast one at 20; the remaining 6 do
    // not fit anywhere and overflow onto the slowest device, which the
    // allocation-failure path then shrinks. What matters is that the split is
    // proportional (2:1 by speed) and that caps are respected on the way in.
    expect(ratios == "1@20*1@10", "split is proportional to speed, then overflow lands on the slowest");

    // Pipelines are limited by their slowest stage, so four identical devices
    // must come out even. The previous first-fill rule gave 25/1/1/1 here,
    // which ran the whole chain at single-card speed.
    std::vector<SpeedDeviceProfile> four;
    for (int i = 0; i < 4; ++i) four.push_back(makeSpeedProfile(5.0, 22u, "t4"));
    expect(assignSpeedPack(four, 28u, nullptr) == "1@7*1@7*1@7*1@7",
        "identical devices split evenly");
    expect(assignSpeedPack(four, 36u, nullptr) == "1@9*1@9*1@9*1@9",
        "identical devices split evenly at another model size");

    // One device at half speed should take half the layers of its peers.
    std::vector<SpeedDeviceProfile> mixed;
    mixed.push_back(makeSpeedProfile(5.0, 22u, "a"));
    mixed.push_back(makeSpeedProfile(10.0, 22u, "b"));
    mixed.push_back(makeSpeedProfile(5.0, 22u, "c"));
    mixed.push_back(makeSpeedProfile(5.0, 22u, "d"));
    expect(assignSpeedPack(mixed, 28u, nullptr) == "1@8*1@4*1@8*1@8",
        "a half-speed device takes half the layers so stage times match");

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
    std::vector<unsigned> five;
    expect(parsePackedCounts("1@18*1@14*1@3*1@5*1@0", &five)
        && failedPackedNode("Socket offline peer=1", five) == 1u,
        "a named offline peer is the slice that shrinks");
    std::vector<unsigned> reservedCeilings;
    expect(relaxPackedRatios("1@18*1@14*1@3*1@5*1@0", 1u, &reservedCeilings, &relaxed)
        && relaxed == "1@18*1@13*1@4*1@5*1@0",
        "a reserved empty slot does not receive the freed layer");

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

    std::vector<unsigned> balancedPrimary(4u, 7u);
    std::vector<double> balancedMs(4u, 5.0);
    const std::vector<unsigned> balancedBubble = bubbleShadowLayerCap(balancedPrimary, balancedMs);
    expect(balancedBubble.size() == 4u
        && balancedBubble[0] == 21u && balancedBubble[1] == 21u
        && balancedBubble[2] == 21u && balancedBubble[3] == 21u,
        "a balanced pipeline leaves each device the other three stages of slack");

    std::vector<unsigned> skewedPrimary;
    std::vector<double> skewedMs;
    skewedPrimary.push_back(2u);
    skewedPrimary.push_back(20u);
    skewedMs.push_back(10.0);
    skewedMs.push_back(1.0);
    const std::vector<unsigned> skewedBubble = bubbleShadowLayerCap(skewedPrimary, skewedMs);
    expect(skewedBubble.size() == 2u && skewedBubble[0] == 2u && skewedBubble[1] == 20u,
        "bubble depth is the other stages' time divided by this device's layer time");

    std::vector<double> unknownMs;
    unknownMs.push_back(0.0);
    unknownMs.push_back(5.0);
    const std::vector<unsigned> unknownBubble = bubbleShadowLayerCap(skewedPrimary, unknownMs);
    expect(unknownBubble.size() == 2u
        && unknownBubble[0] == 0xffffffffu && unknownBubble[1] == 0xffffffffu,
        "a missed probe leaves the bubble uncapped");

    std::vector<unsigned> bottleneckPrimary;
    std::vector<double> bottleneckMs;
    bottleneckPrimary.push_back(10u);
    bottleneckPrimary.push_back(1u);
    bottleneckMs.push_back(10.0);
    bottleneckMs.push_back(1.0);
    const std::vector<unsigned> bottleneckBubble = bubbleShadowLayerCap(bottleneckPrimary, bottleneckMs);
    expect(bottleneckBubble.size() == 2u && bottleneckBubble[0] == 0u && bottleneckBubble[1] == 100u,
        "a stage that already fills the token has no bubble left for shadow");

    std::vector<unsigned> tightBubble;
    tightBubble.push_back(2u);
    tightBubble.push_back(100u);
    primary.clear();
    memoryCap.clear();
    primary.push_back(22u);
    primary.push_back(6u);
    memoryCap.push_back(28u);
    memoryCap.push_back(28u);
    overlaps = assignBoundaryOverlap(primary, memoryCap, &tightBubble);
    expect(overlaps.size() == 1u && overlaps[0].leftHolds == 2u && overlaps[0].rightHolds == 21u,
        "bubble depth cuts a side that memory would still have filled");
    overlaps = assignBoundaryOverlap(primary, memoryCap);
    expect(overlaps.size() == 1u && overlaps[0].leftHolds == 5u && overlaps[0].rightHolds == 21u,
        "without a bubble cap the memory spare is unchanged");

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

    // Four stages of 7 layers with a cap of 13: each device has 6 spare and a
    // single side can take all 6. Equal weights must reproduce the old
    // round-robin result exactly, otherwise every earlier run changed meaning.
    std::vector<unsigned> fourPrimary(4u, 7u);
    std::vector<unsigned> fourCaps(4u, 13u);
    std::vector<unsigned> equalWeights(4u, 3u);
    const std::vector<BoundaryOverlap> equalOverlaps =
        assignBoundaryOverlap(fourPrimary, fourCaps, nullptr, &equalWeights);
    expect(formatBoundaryOverlap(assignBoundaryOverlap(fourPrimary, fourCaps)) == "6,3;3,3;3,6",
        "equal weights reproduce the default split");
    expect(formatBoundaryOverlap(equalOverlaps) == "6,3;3,3;3,6",
        "an explicit mid weight vector matches a null one");
    if (equalOverlaps.size() == 3u) {
        expect(equalOverlaps[1].rightHolds == 3u && equalOverlaps[2].leftHolds == 3u,
            "stage 2 splits its spare evenly across both boundaries");
    } else {
        expect(false, "four stages produce three boundaries");
    }

    // The high neighbor pulls copies onto both of its peers; the low neighbor
    // pulls fewer. Weighting the holder instead would swap these numbers.
    std::vector<unsigned> highB(4u, 3u);
    highB[1] = 4u;
    highB[3] = 2u;
    const std::vector<BoundaryOverlap> highBOverlaps =
        assignBoundaryOverlap(fourPrimary, fourCaps, nullptr, &highB);
    expect(highBOverlaps.size() == 3u
        && highBOverlaps[1].rightHolds == 4u && highBOverlaps[2].leftHolds == 2u,
        "a high-priority left neighbor takes more of the middle stage's spare");

    std::vector<unsigned> highD(4u, 3u);
    highD[1] = 2u;
    highD[3] = 4u;
    const std::vector<BoundaryOverlap> highDOverlaps =
        assignBoundaryOverlap(fourPrimary, fourCaps, nullptr, &highD);
    expect(highDOverlaps.size() == 3u
        && highDOverlaps[1].rightHolds == 2u && highDOverlaps[2].leftHolds == 4u,
        "a high-priority right neighbor takes more of the middle stage's spare");

    unsigned xLeft = 0u;
    unsigned xRight = 0u;
    splitDeviceShadowBudget(5u, 10u, 10u, 4u, 2u, &xLeft, &xRight);
    expect(xLeft == 3u && xRight == 2u, "a 2:1 weight split gives the larger side the extra layer");
    splitDeviceShadowBudget(5u, 10u, 10u, 3u, 3u, &xLeft, &xRight);
    expect(xLeft == 3u && xRight == 2u, "an exact tie goes to the left side");
    splitDeviceShadowBudget(10u, 2u, 100u, 3u, 3u, &xLeft, &xRight);
    expect(xLeft == 2u && xRight == 8u, "a capped side hands the rest to its neighbor");
    splitDeviceShadowBudget(6u, 0u, 6u, 4u, 2u, &xLeft, &xRight);
    expect(xLeft == 0u && xRight == 6u, "an inactive side keeps nothing and the budget goes across");

    // The example file is the contract for the parser: comment lines are
    // skipped, missing stages stay mid, and reversed swaps the extremes.
    const std::string exampleText =
        "# stage priority reason\n"
        "# Read only when EDGEVISOR_SHADOW_ALLOCATION=priority or reversed.\n"
        "# Weights: low=2 mid=3 high=4. Missing stages stay mid.\n"
        "# reversed swaps low and high. equal ignores this file.\n"
        "0 low dedicated-and-plugged-in\n"
        "1 high mobile-unplugged\n"
        "2 mid shared-workstation\n"
        "3 low dedicated-and-plugged-in\n";
    std::vector<unsigned> parsedWeights;
    std::string parseError;
    expect(parseShadowPriorityText(exampleText, 4u, &parsedWeights, &parseError), "the example file parses");
    expect(parsedWeights.size() == 4u
        && parsedWeights[0] == 2u && parsedWeights[1] == 4u
        && parsedWeights[2] == 3u && parsedWeights[3] == 2u,
        "the example file reads as low, high, mid, low");
    reverseShadowPriorityWeights(&parsedWeights);
    expect(parsedWeights.size() == 4u
        && parsedWeights[0] == 4u && parsedWeights[1] == 2u
        && parsedWeights[2] == 3u && parsedWeights[3] == 4u,
        "reversed swaps low and high and leaves mid alone");
    std::vector<unsigned> outOfRange;
    expect(!parseShadowPriorityText("9 high\n", 4u, &outOfRange, &parseError),
        "a stage index past the pipeline is rejected");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d device profile checks failed\n", g_failures);
        return 1;
    }
    std::printf("device profile checks passed\n");
    return 0;
}
