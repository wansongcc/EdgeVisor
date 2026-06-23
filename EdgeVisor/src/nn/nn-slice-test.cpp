#include "nn-core.hpp"

#include <cstdio>
#include <functional>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception>
void requireThrows(const std::function<void()> &fn, const char *message) {
    try {
        fn();
    } catch (const Exception &) {
        return;
    }
    throw std::runtime_error(message);
}

void setSplit(NnDimSplit &split, const NnUint *starts, const NnUint *lengths, NnUint n) {
    split.starts = new NnUint[n];
    split.lengths = new NnUint[n];
    for (NnUint i = 0; i < n; ++i) {
        split.starts[i] = starts[i];
        split.lengths[i] = lengths[i];
    }
}

void setAllSplits(
    NnUnevenPartitionPlan &plan,
    const NnUint *starts,
    const NnUint *lengths
) {
    setSplit(plan.vocabSplit, starts, lengths, plan.nNodes);
    setSplit(plan.ffnSplit, starts, lengths, plan.nNodes);
    setSplit(plan.dimSplit, starts, lengths, plan.nNodes);
    setSplit(plan.headSplit, starts, lengths, plan.nNodes);
    setSplit(plan.kvHeadSplit, starts, lengths, plan.nNodes);
}

struct ConfigFixture {
    NnPipeConfig pipe;
    NnBufferConfig buffer;
    NnNetConfig net;
    NnNodeConfig node;

    ConfigFixture(NnFloatType type, NnUint z, NnUint batches, NnUint x, NnUint nNodes, NnUint nodeIndex) {
        pipe = {nullptr, size3D(type, z, batches, x)};
        buffer = {nullptr, size3D(type, z, batches, x)};
        net = {batches, nNodes, 1u, &pipe, 0u, nullptr};
        node.nodeIndex = nodeIndex;
        node.nBuffers = 1u;
        node.buffers = &buffer;
        node.nSegments = 0u;
        node.segments = nullptr;
        node.partitionPlan = nullptr;
    }
};

NnPointerLayout layout(
    ConfigFixture &fixture,
    NnPointerSource source,
    NnPointerType type,
    NnSliceTag tag,
    const NnUnevenPartitionPlan *plan = nullptr
) {
    const NnPointerConfig config{source, 0u, type, tag};
    return resolvePointerLayout(&fixture.net, &fixture.node, plan, &config);
}

void testOpCount() {
    require(OP_COUNT == OP_PLAN_APPLY + 1, "OP_COUNT must follow the final opcode");
    require(N_OP_CODES == OP_COUNT, "legacy opcode count alias must cover every opcode");
    for (NnUint code = 0u; code < OP_COUNT; ++code) {
        require(opCodeToString((NnOpCode)code) != nullptr, "opcode is missing a name");
    }
}

void testRawAndBatchPointers() {
    ConfigFixture f(F_32, 2u, 3u, 8u, 2u, 1u);
    const NnPointerLayout raw = layout(f, SRC_PIPE, PNTR_RAW, NN_SLICE_AUTO);
    require(raw.logicalSize.x == 48u && raw.logicalSize.y == 1u, "raw pointer shape");
    require(raw.byteOffset == 0u, "raw pointer offset");

    const NnPointerLayout batch = layout(f, SRC_BUFFER, PNTR_BATCH, NN_SLICE_AUTO);
    require(batch.logicalSize.x == 8u && batch.logicalSize.y == 3u && batch.logicalSize.z == 2u, "batch shape");
    require(batch.batchStrideBytes == 8u * sizeof(float), "batch stride");
    require(batch.zStrideBytes == 3u * 8u * sizeof(float), "Z stride");
}

void testUniformAndUnevenSlices() {
    {
        ConfigFixture f(F_32, 1u, 2u, 12u, 3u, 1u);
        const NnPointerLayout p = layout(f, SRC_BUFFER, PNTR_BATCHED_SLICE, NN_SLICE_AUTO);
        require(p.logicalOffset == 4u && p.logicalSize.x == 4u, "uniform slice");
        require(p.byteOffset == 4u * sizeof(float), "uniform byte offset");
    }

    const NnUint starts[] = {0u, 2u, 5u};
    const NnUint lengths[] = {2u, 3u, 5u};
    NnUnevenPartitionPlan plan;
    plan.nNodes = 3u;
    setAllSplits(plan, starts, lengths);
    ConfigFixture f(F_32, 1u, 2u, 10u, 3u, 1u);

    const NnSliceTag tags[] = {
        NN_SLICE_AUTO, NN_SLICE_VOCAB, NN_SLICE_FFN, NN_SLICE_DIM,
        NN_SLICE_HEAD, NN_SLICE_KV_HEAD
    };
    for (NnSliceTag tag : tags) {
        const NnPointerLayout p = layout(f, SRC_BUFFER, PNTR_BATCHED_SLICE, tag, &plan);
        require(p.logicalOffset == 2u && p.logicalSize.x == 3u, "tagged uneven slice");
    }

    ConfigFixture head(F_32, 1u, 2u, 80u, 3u, 1u);
    const NnPointerLayout hp = layout(
        head, SRC_BUFFER, PNTR_BATCHED_SLICE, NN_SLICE_HEAD, &plan);
    require(hp.logicalOffset == 16u && hp.logicalSize.x == 24u, "head multiplier slice");
}

void testStageLocalAndStackedSlices() {
    NnUnevenPartitionPlan plan;
    plan.nNodes = 4u;
    plan.nStages = 2u;
    plan.stages = new NnStageConfig[2];
    plan.stages[0].nNodes = 2u;
    plan.stages[0].nodeIndices = new NnUint[2]{0u, 1u};
    plan.stages[1].nNodes = 2u;
    plan.stages[1].nodeIndices = new NnUint[2]{2u, 3u};
    const NnUint starts[] = {0u, 4u, 0u, 3u};
    const NnUint lengths[] = {4u, 4u, 3u, 5u};
    setAllSplits(plan, starts, lengths);

    ConfigFixture local(F_32, 1u, 2u, 8u, 4u, 3u);
    const NnPointerLayout stage = layout(
        local, SRC_BUFFER, PNTR_BATCHED_SLICE, NN_SLICE_DIM, &plan);
    require(stage.logicalOffset == 3u && stage.logicalSize.x == 5u, "stage-local slice");

    ConfigFixture stacked(F_32, 1u, 2u, 32u, 4u, 3u);
    const NnPointerLayout explicitStack = layout(
        stacked, SRC_PIPE, PNTR_BATCHED_SLICE, NN_SLICE_STACKED_BY_NODE, &plan);
    require(explicitStack.logicalOffset == 24u && explicitStack.logicalSize.x == 8u, "stacked slice");

    ConfigFixture autoStack(F_32, 1u, 2u, 32u, 4u, 2u);
    const NnPointerLayout inferredStack = layout(
        autoStack, SRC_PIPE, PNTR_BATCHED_SLICE, NN_SLICE_AUTO, &plan);
    require(inferredStack.logicalOffset == 16u && inferredStack.logicalSize.x == 8u, "inferred stacked slice");
}

void testQuantizedAlignmentAndBounds() {
    {
        NnUnevenPartitionPlan plan;
        plan.nNodes = 2u;
        const NnUint starts[] = {0u, 64u};
        const NnUint lengths[] = {64u, 64u};
        setAllSplits(plan, starts, lengths);
        ConfigFixture f(F_Q80, 1u, 2u, 128u, 2u, 1u);
        const NnPointerLayout p = layout(
            f, SRC_BUFFER, PNTR_BATCHED_SLICE, NN_SLICE_FFN, &plan);
        require(p.logicalOffset == 64u && p.logicalSize.x == 64u, "Q80 aligned slice");
        require(p.byteOffset == 2u * sizeof(NnBlockQ80), "Q80 byte offset");
    }
    {
        NnUnevenPartitionPlan plan;
        plan.nNodes = 2u;
        const NnUint starts[] = {0u, 48u};
        const NnUint lengths[] = {48u, 80u};
        setAllSplits(plan, starts, lengths);
        ConfigFixture f(F_Q80, 1u, 2u, 128u, 2u, 1u);
        requireThrows<std::invalid_argument>([&]() {
            layout(f, SRC_BUFFER, PNTR_BATCHED_SLICE, NN_SLICE_FFN, &plan);
        }, "misaligned Q80 slice must be rejected");
    }
    {
        NnUnevenPartitionPlan plan;
        plan.nNodes = 2u;
        const NnUint starts[] = {0u, 8u};
        const NnUint lengths[] = {5u, 5u};
        setAllSplits(plan, starts, lengths);
        ConfigFixture f(F_32, 1u, 2u, 10u, 2u, 1u);
        requireThrows<std::out_of_range>([&]() {
            layout(f, SRC_BUFFER, PNTR_BATCHED_SLICE, NN_SLICE_DIM, &plan);
        }, "out-of-bounds partition slice must be rejected");
    }
}

void testTensorViews() {
    const NnTensorView packed{0u, 0u, 0u, 0u, 0u};
    const NnTensorViewLayout a = resolveTensorView(&packed, 3u, 8u, 24u);
    require(a.sizeY == 3u && a.sizeX == 8u, "packed view shape");
    require(a.strideY == 8u && a.strideX == 1u && a.spanElements == 24u, "packed view strides");

    const NnTensorView strided{2u, 2u, 3u, 8u, 2u};
    const NnTensorViewLayout b = resolveTensorView(&strided, 0u, 0u, 32u);
    require(b.offset == 2u && b.spanElements == 13u, "strided view span");

    const NnTensorView bad{7u, 2u, 3u, 8u, 1u};
    requireThrows<std::out_of_range>([&]() {
        resolveTensorView(&bad, 0u, 0u, 16u);
    }, "out-of-bounds tensor view must be rejected");
}

} // namespace

int main() {
    initQuants();
    testOpCount();
    testRawAndBatchPointers();
    testUniformAndUnevenSlices();
    testStageLocalAndStackedSlices();
    testQuantizedAlignmentAndBounds();
    testTensorViews();
    std::printf("All pointer/slice/view tests passed\n");
    return 0;
}
