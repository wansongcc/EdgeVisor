#include "nn-cuda.hpp"
#include "nn-quants.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void usage(const char *prog) {
    std::fprintf(stderr, "Usage: %s [--gpu-index <index>]\n", prog);
}

static std::vector<NnByte> pattern(NnSize nBytes, NnUint seed) {
    std::vector<NnByte> out(nBytes);
    for (NnSize i = 0; i < nBytes; ++i) {
        out[i] = (NnByte)((i * 37u + seed * 17u + (i >> 3)) & 0xffu);
    }
    return out;
}

static void expectBytes(const char *name, const std::vector<NnByte> &actual, const std::vector<NnByte> &expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(name) + " size mismatch");
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "%s mismatch at byte %zu: got=%u expected=%u", name, i, (unsigned)actual[i], (unsigned)expected[i]);
            throw std::runtime_error(msg);
        }
    }
}

static void expectF32Near(const char *name, const std::vector<float> &actual, const std::vector<float> &expected, float maxAbsError) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(name) + " size mismatch");
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    size_t maxIndex = 0u;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float denom = std::max(std::fabs(expected[i]), 1.0e-12f);
        const float rel = diff / denom;
        if (diff > maxAbs) {
            maxAbs = diff;
            maxRel = rel;
            maxIndex = i;
        }
    }
    if (maxAbs > maxAbsError) {
        char msg[512];
        std::snprintf(msg, sizeof(msg),
            "%s max_abs_error=%g exceeds %g at %zu: got=%g expected=%g max_rel_error=%g",
            name, maxAbs, maxAbsError, maxIndex, actual[maxIndex], expected[maxIndex], maxRel);
        throw std::runtime_error(msg);
    }
    std::printf("%s: ok max_abs_error=%g max_rel_error=%g\n", name, maxAbs, maxRel);
}

static void expectQ80DequantNear(
    const char *name,
    const std::vector<NnBlockQ80> &actual,
    const std::vector<NnBlockQ80> &expected,
    NnUint logicalElements,
    float maxAbsError) {
    std::vector<float> actualF32(logicalElements);
    std::vector<float> expectedF32(logicalElements);
    dequantizeQ80toF32(actual.data(), actualF32.data(), logicalElements, 1u, 0u);
    dequantizeQ80toF32(expected.data(), expectedF32.data(), logicalElements, 1u, 0u);
    expectF32Near(name, actualF32, expectedF32, maxAbsError);
}

static NnSize f32Index(const NnSize3D &size, NnUint z, NnUint y, NnUint x) {
    return ((NnSize)z * size.y + y) * size.x + x;
}

static NnSize q80Index(const NnSize3D &size, NnUint z, NnUint y, NnUint block) {
    return ((NnSize)z * size.y + y) * (size.x / Q80_BLOCK_SIZE) + block;
}

static std::vector<float> f32Pattern(NnSize n, float scale, float bias) {
    std::vector<float> out(n);
    for (NnSize i = 0; i < n; ++i) {
        const int v = (int)((i * 17u + 11u) % 97u) - 48;
        out[i] = (float)v * scale + bias;
    }
    return out;
}

static void runMemoryDataPathTest(NnUint gpuIndex) {
    NnPipeConfig pipes[5];
    pipes[0] = {(char *)"input_pipe", size2D(F_32, 4u, 8u)};
    pipes[1] = {(char *)"output_pipe", size2D(F_32, 4u, 8u)};
    pipes[2] = {(char *)"position_pipe", size1D(F_32, 4u)};
    pipes[3] = {(char *)"input_pipe_2", size2D(F_32, 4u, 8u)};
    pipes[4] = {(char *)"output_pipe_2", size2D(F_32, 4u, 8u)};

    NnPreSyncConfig preSyncs[1];
    preSyncs[0] = {2u};

    NnBufferConfig buffers[3];
    buffers[0] = {(char *)"f32_buffer", size2D(F_32, 4u, 8u)};
    buffers[1] = {(char *)"q40_buffer", size1D(F_Q40, 64u)};
    buffers[2] = {(char *)"q80_buffer", size1D(F_Q80, 64u)};

    NnCastOpCodeConfig castConfig{};

    NnOpConfig ops[3];
    std::memset(ops, 0, sizeof(ops));
    ops[0].code = OP_CAST;
    ops[0].name = (char *)"cuda_data_path_0";
    ops[0].index = 0u;
    ops[0].input = pointerBatchConfig(SRC_PIPE, 0u);
    ops[0].output = pointerBatchConfig(SRC_PIPE, 1u);
    ops[0].weightSize = size1D(F_32, 64u);
    ops[0].config = (NnByte *)&castConfig;
    ops[0].configSize = sizeof(castConfig);

    ops[1].code = OP_CAST;
    ops[1].name = (char *)"cuda_data_path_1_same_pipe";
    ops[1].index = 1u;
    ops[1].input = pointerBatchConfig(SRC_PIPE, 0u);
    ops[1].output = pointerBatchConfig(SRC_PIPE, 1u);
    ops[1].weightSize = size0();
    ops[1].config = (NnByte *)&castConfig;
    ops[1].configSize = sizeof(castConfig);

    ops[2].code = OP_CAST;
    ops[2].name = (char *)"cuda_data_path_2_second_pipe";
    ops[2].index = 2u;
    ops[2].input = pointerBatchConfig(SRC_PIPE, 3u);
    ops[2].output = pointerBatchConfig(SRC_PIPE, 4u);
    ops[2].weightSize = size0();
    ops[2].config = (NnByte *)&castConfig;
    ops[2].configSize = sizeof(castConfig);

    NnSegmentConfig segments[1];
    segments[0].nOps = 3u;
    segments[0].ops = ops;
    segments[0].nSyncs = 0u;
    segments[0].syncs = nullptr;

    NnNetConfig netConfig{};
    netConfig.nBatches = 4u;
    netConfig.nNodes = 1u;
    netConfig.nPipes = 5u;
    netConfig.pipes = pipes;
    netConfig.nPreSyncs = 1u;
    netConfig.preSyncs = preSyncs;

    NnNodeConfig nodeConfig{};
    nodeConfig.nodeIndex = 0u;
    nodeConfig.nBuffers = 3u;
    nodeConfig.buffers = buffers;
    nodeConfig.nSegments = 1u;
    nodeConfig.segments = segments;

    NnNetExecution execution(1u, &netConfig);
    execution.setBatchSize(2u);

    NnCudaDevice device(gpuIndex, &netConfig, &nodeConfig, &execution, nullptr);

    for (NnUint i = 0u; i < nodeConfig.nBuffers; ++i) {
        const NnSize nBytes = nodeConfig.buffers[i].size.nBytes;
        std::vector<NnByte> src = pattern(nBytes, 10u + i);
        std::vector<NnByte> got(nBytes, 0u);
        device.writeBuffer(i, src.data(), 0u, nBytes);
        device.readBuffer(i, got.data(), 0u, nBytes);
        expectBytes(nodeConfig.buffers[i].name, got, src);
    }
    std::printf("CUDA F32/Q40/Q80 buffer upload/download: ok\n");

    for (NnUint i = 0u; i < netConfig.nPipes; ++i) {
        const NnSize nBytes = netConfig.pipes[i].size.nBytes;
        std::vector<NnByte> src = pattern(nBytes, 20u + i);
        std::vector<NnByte> got(nBytes, 0u);
        device.writePipe(i, src.data(), 0u, nBytes);
        device.readPipe(i, got.data(), 0u, nBytes);
        expectBytes(netConfig.pipes[i].name, got, src);
    }
    std::printf("CUDA multiple pipe upload/download: ok\n");

    std::unique_ptr<NnDeviceSegment> baseSegment(device.createSegment(0u));
    NnCudaDeviceSegment *segment = dynamic_cast<NnCudaDeviceSegment *>(baseSegment.get());
    if (segment == nullptr) throw std::runtime_error("createSegment did not return NnCudaDeviceSegment");

    const NnSize weightBytes = ops[0].weightSize.nBytes;
    std::vector<NnByte> weight = pattern(weightBytes, 30u);
    segment->loadWeight(0u, 0u, 17u, weight.data());
    segment->loadWeight(0u, 17u, 111u, weight.data() + 17u);
    segment->loadWeight(0u, 128u, weightBytes - 128u, weight.data() + 128u);
    std::vector<NnByte> gotWeight(weightBytes, 0u);
    segment->readWeight(0u, 0u, weightBytes, gotWeight.data());
    expectBytes("chunked weight", gotWeight, weight);
    std::printf("CUDA chunked weight load/readback: ok\n");

    std::vector<NnByte> input = pattern(pipes[0].size.nBytes, 40u);
    std::vector<NnByte> input2 = pattern(pipes[3].size.nBytes, 43u);
    std::vector<NnByte> pos = pattern(pipes[2].size.nBytes, 41u);
    std::memcpy(execution.pipes[0], input.data(), input.size());
    std::memcpy(execution.pipes[3], input2.data(), input2.size());
    std::memcpy(execution.pipes[2], pos.data(), pos.size());

    std::vector<NnByte> deviceOutput = pattern(pipes[1].size.nBytes, 42u);
    std::vector<NnByte> deviceOutput2 = pattern(pipes[4].size.nBytes, 44u);
    device.writePipe(1u, deviceOutput.data(), 0u, deviceOutput.size());
    device.writePipe(4u, deviceOutput2.data(), 0u, deviceOutput2.size());
    std::memset(execution.pipes[1], 0, pipes[1].size.nBytes);
    std::memset(execution.pipes[4], 0, pipes[4].size.nBytes);

    segment->forward(0u, 1u, 0u, 2u);

    const NnSize activeInputBytes = device.data.resolvePipe(0u)->calcSliceSize(2u, netConfig.nBatches);
    std::vector<NnByte> gotInput(activeInputBytes, 0u);
    device.readPipe(0u, gotInput.data(), 0u, activeInputBytes);
    std::vector<NnByte> expectedInput(input.begin(), input.begin() + activeInputBytes);
    expectBytes("segment input upload", gotInput, expectedInput);

    const NnSize activeInput2Bytes = device.data.resolvePipe(3u)->calcSliceSize(2u, netConfig.nBatches);
    std::vector<NnByte> gotInput2(activeInput2Bytes, 0u);
    device.readPipe(3u, gotInput2.data(), 0u, activeInput2Bytes);
    std::vector<NnByte> expectedInput2(input2.begin(), input2.begin() + activeInput2Bytes);
    expectBytes("segment second input upload", gotInput2, expectedInput2);

    const NnSize activePosBytes = device.data.resolvePipe(2u)->calcSliceSize(2u, netConfig.nBatches);
    std::vector<NnByte> gotPos(activePosBytes, 0u);
    device.readPipe(2u, gotPos.data(), 0u, activePosBytes);
    std::vector<NnByte> expectedPos(pos.begin(), pos.begin() + activePosBytes);
    expectBytes("segment pre-sync upload", gotPos, expectedPos);

    const NnSize activeOutputBytes = device.data.resolvePipe(1u)->calcSliceSize(2u, netConfig.nBatches);
    std::vector<NnByte> hostOutput(activeOutputBytes, 0u);
    std::memcpy(hostOutput.data(), execution.pipes[1], activeOutputBytes);
    std::vector<NnByte> expectedOutput(input.begin(), input.begin() + activeInputBytes);
    expectBytes("segment output download", hostOutput, expectedOutput);

    const NnSize activeOutput2Bytes = device.data.resolvePipe(4u)->calcSliceSize(2u, netConfig.nBatches);
    std::vector<NnByte> hostOutput2(activeOutput2Bytes, 0u);
    std::memcpy(hostOutput2.data(), execution.pipes[4], activeOutput2Bytes);
    std::vector<NnByte> expectedOutput2(input2.begin(), input2.begin() + activeInput2Bytes);
    expectBytes("segment second output download", hostOutput2, expectedOutput2);
    std::printf("CUDA segment boundary multi-input/multi-output transfer: ok\n");

    std::vector<NnByte> beforeNoop(activeInputBytes, 0u);
    device.readPipe(0u, beforeNoop.data(), 0u, activeInputBytes);
    std::vector<NnByte> changedInput = pattern(pipes[0].size.nBytes, 77u);
    std::memcpy(execution.pipes[0], changedInput.data(), changedInput.size());
    segment->forward(1u, 1u, 0u, 2u);
    std::vector<NnByte> afterNoop(activeInputBytes, 0u);
    device.readPipe(0u, afterNoop.data(), 0u, activeInputBytes);
    expectBytes("opIndex nonzero no-op", afterNoop, beforeNoop);
    std::printf("CUDA opIndex nonzero no-op: ok\n");

    size_t freeBefore = 0u;
    size_t totalBefore = 0u;
    if (cudaSetDevice((int)gpuIndex) != cudaSuccess ||
        cudaMemGetInfo(&freeBefore, &totalBefore) != cudaSuccess) {
        throw std::runtime_error("cudaMemGetInfo before stress loop failed");
    }
    for (int i = 0; i < 1000; ++i) {
        segment->forward(0u, 1u, 0u, 2u);
    }
    size_t freeAfter = 0u;
    size_t totalAfter = 0u;
    if (cudaMemGetInfo(&freeAfter, &totalAfter) != cudaSuccess) {
        throw std::runtime_error("cudaMemGetInfo after stress loop failed");
    }
    if (totalBefore != totalAfter || freeAfter + (1u << 20) < freeBefore) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "CUDA forward loop memory grew unexpectedly: freeBefore=%zu freeAfter=%zu totalBefore=%zu totalAfter=%zu",
            freeBefore, freeAfter, totalBefore, totalAfter);
        throw std::runtime_error(msg);
    }
    std::printf("CUDA 1000-forward memory stability: ok freeBefore=%zu freeAfter=%zu\n", freeBefore, freeAfter);

    std::printf("CUDA memory/data-path tests: ok\n");
}

static void runPr4ElementwiseOpsTest(NnUint gpuIndex) {
    const NnUint nBatches = 4u;

    NnPipeConfig pipes[18];
    pipes[0] = {(char *)"cast_f32_in", size3D(F_32, 2u, nBatches, 13u)};
    pipes[1] = {(char *)"cast_f32_out", size3D(F_32, 2u, nBatches, 13u)};
    pipes[2] = {(char *)"silu_io", size3D(F_32, 2u, nBatches, 17u)};
    pipes[3] = {(char *)"gelu_io", size3D(F_32, 1u, nBatches, 17u)};
    pipes[4] = {(char *)"mul_in", size3D(F_32, 2u, nBatches, 17u)};
    pipes[5] = {(char *)"mul_out", size3D(F_32, 2u, nBatches, 17u)};
    pipes[6] = {(char *)"scale_in", size3D(F_32, 2u, nBatches, 17u)};
    pipes[7] = {(char *)"scale_out", size3D(F_32, 2u, nBatches, 17u)};
    pipes[8] = {(char *)"merge_add_in", size3D(F_32, 1u, nBatches, 15u)};
    pipes[9] = {(char *)"merge_add_out", size3D(F_32, 1u, nBatches, 5u)};
    pipes[10] = {(char *)"merge_sum_in", size3D(F_32, 3u, nBatches, 11u)};
    pipes[11] = {(char *)"merge_sum_out", size3D(F_32, 1u, nBatches, 11u)};
    pipes[12] = {(char *)"cast_q80_in", size3D(F_32, 2u, nBatches, 96u)};
    pipes[13] = {(char *)"cast_q80_out", size3D(F_Q80, 2u, nBatches, 96u)};
    pipes[14] = {(char *)"repeat_z_in", size3D(F_32, 1u, nBatches, 64u)};
    pipes[15] = {(char *)"repeat_z_out", size3D(F_Q80, 3u, nBatches, 64u)};
    pipes[16] = {(char *)"view2d_in", size3D(F_32, 1u, nBatches, 20u)};
    pipes[17] = {(char *)"view2d_out", size3D(F_32, 1u, nBatches, 20u)};

    NnBufferConfig buffers[2];
    buffers[0] = {(char *)"mul_multiplier", size3D(F_32, 2u, nBatches, 17u)};
    buffers[1] = {(char *)"scale_values", size1D(F_32, 2u * nBatches)};

    NnCastOpCodeConfig castF32Cfg{NnTensorView{2u, 0u, 9u, 0u, 1u}};
    NnSiluOpCodeConfig siluCfg{NnTensorView{3u, 0u, 11u, 0u, 1u}};
    NnGeluOpCodeConfig geluCfg{NnTensorView{1u, 0u, 13u, 0u, 1u}};
    NnMulOpCodeConfig mulCfg{0u, NnTensorView{2u, 0u, 13u, 0u, 1u}};
    NnScaleOpCodeConfig scaleCfg{1u, NnTensorView{1u, 0u, 15u, 0u, 1u}};
    NnCastOpCodeConfig castQ80Cfg{NnTensorView{32u, 0u, 64u, 0u, 1u}};
    NnRepeatZOpCodeConfig repeatCfg{};
    NnCastOpCodeConfig view2dCfg{NnTensorView{1u, 2u, 3u, 7u, 2u}};

    NnOpConfig ops[10];
    std::memset(ops, 0, sizeof(ops));
    ops[0].code = OP_CAST; ops[0].name = (char *)"cast_f32_f32_view"; ops[0].index = 0u;
    ops[0].input = pointerBatchConfig(SRC_PIPE, 0u); ops[0].output = pointerBatchConfig(SRC_PIPE, 1u);
    ops[0].config = (NnByte *)&castF32Cfg; ops[0].configSize = sizeof(castF32Cfg);
    ops[1].code = OP_SILU; ops[1].name = (char *)"silu_inplace"; ops[1].index = 1u;
    ops[1].input = pointerBatchConfig(SRC_PIPE, 2u); ops[1].output = pointerBatchConfig(SRC_PIPE, 2u);
    ops[1].config = (NnByte *)&siluCfg; ops[1].configSize = sizeof(siluCfg);
    ops[2].code = OP_GELU; ops[2].name = (char *)"gelu_inplace"; ops[2].index = 2u;
    ops[2].input = pointerBatchConfig(SRC_PIPE, 3u); ops[2].output = pointerBatchConfig(SRC_PIPE, 3u);
    ops[2].config = (NnByte *)&geluCfg; ops[2].configSize = sizeof(geluCfg);
    ops[3].code = OP_MUL; ops[3].name = (char *)"mul_view"; ops[3].index = 3u;
    ops[3].input = pointerBatchConfig(SRC_PIPE, 4u); ops[3].output = pointerBatchConfig(SRC_PIPE, 5u);
    ops[3].config = (NnByte *)&mulCfg; ops[3].configSize = sizeof(mulCfg);
    ops[4].code = OP_SCALE; ops[4].name = (char *)"scale_view"; ops[4].index = 4u;
    ops[4].input = pointerBatchConfig(SRC_PIPE, 6u); ops[4].output = pointerBatchConfig(SRC_PIPE, 7u);
    ops[4].config = (NnByte *)&scaleCfg; ops[4].configSize = sizeof(scaleCfg);
    ops[5].code = OP_MERGE_ADD; ops[5].name = (char *)"merge_add"; ops[5].index = 5u;
    ops[5].input = pointerBatchConfig(SRC_PIPE, 8u); ops[5].output = pointerBatchConfig(SRC_PIPE, 9u);
    ops[6].code = OP_MERGE_SUM; ops[6].name = (char *)"merge_sum"; ops[6].index = 6u;
    ops[6].input = pointerBatchConfig(SRC_PIPE, 10u); ops[6].output = pointerBatchConfig(SRC_PIPE, 11u);
    ops[7].code = OP_CAST; ops[7].name = (char *)"cast_f32_q80_view"; ops[7].index = 7u;
    ops[7].input = pointerBatchConfig(SRC_PIPE, 12u); ops[7].output = pointerBatchConfig(SRC_PIPE, 13u);
    ops[7].config = (NnByte *)&castQ80Cfg; ops[7].configSize = sizeof(castQ80Cfg);
    ops[8].code = OP_REPEAT_Z; ops[8].name = (char *)"repeat_z_f32_q80"; ops[8].index = 8u;
    ops[8].input = pointerBatchConfig(SRC_PIPE, 14u); ops[8].output = pointerBatchConfig(SRC_PIPE, 15u);
    ops[8].config = (NnByte *)&repeatCfg; ops[8].configSize = sizeof(repeatCfg);
    ops[9].code = OP_CAST; ops[9].name = (char *)"cast_f32_f32_2d_view"; ops[9].index = 9u;
    ops[9].input = pointerBatchConfig(SRC_PIPE, 16u); ops[9].output = pointerBatchConfig(SRC_PIPE, 17u);
    ops[9].config = (NnByte *)&view2dCfg; ops[9].configSize = sizeof(view2dCfg);

    NnSegmentConfig segments[1];
    segments[0].nOps = 10u;
    segments[0].ops = ops;
    segments[0].nSyncs = 0u;
    segments[0].syncs = nullptr;

    NnNetConfig netConfig{};
    netConfig.nBatches = nBatches;
    netConfig.nNodes = 1u;
    netConfig.nPipes = 18u;
    netConfig.pipes = pipes;
    netConfig.nPreSyncs = 0u;
    netConfig.preSyncs = nullptr;

    NnNodeConfig nodeConfig{};
    nodeConfig.nodeIndex = 0u;
    nodeConfig.nBuffers = 2u;
    nodeConfig.buffers = buffers;
    nodeConfig.nSegments = 1u;
    nodeConfig.segments = segments;

    NnNetExecution execution(1u, &netConfig);
    execution.setBatchSize(nBatches);
    NnCudaDevice device(gpuIndex, &netConfig, &nodeConfig, &execution, nullptr);

    std::vector<std::vector<float> > hostF32(18);
    for (NnUint i = 0u; i < 18u; ++i) {
        if (pipes[i].size.floatType == F_32) {
            hostF32[i] = f32Pattern(pipes[i].size.length, 0.03125f + (float)i * 0.001f, -0.25f + (float)i * 0.01f);
            std::memcpy(execution.pipes[i], hostF32[i].data(), pipes[i].size.nBytes);
        }
    }

    std::vector<NnBlockQ80> castQ80Initial(pipes[13].size.length / Q80_BLOCK_SIZE);
    std::vector<float> castQ80InitialF32(pipes[13].size.length, 0.0f);
    quantizeF32toQ80(castQ80InitialF32.data(), castQ80Initial.data(), (NnUint)castQ80InitialF32.size(), 1u, 0u);
    std::memcpy(execution.pipes[13], castQ80Initial.data(), pipes[13].size.nBytes);
    device.writePipe(13u, (const NnByte *)castQ80Initial.data(), 0u, pipes[13].size.nBytes);

    std::vector<NnBlockQ80> repeatInitial(pipes[15].size.length / Q80_BLOCK_SIZE);
    std::vector<float> repeatInitialF32(pipes[15].size.length, 0.0f);
    quantizeF32toQ80(repeatInitialF32.data(), repeatInitial.data(), (NnUint)repeatInitialF32.size(), 1u, 0u);
    std::memcpy(execution.pipes[15], repeatInitial.data(), pipes[15].size.nBytes);
    device.writePipe(15u, (const NnByte *)repeatInitial.data(), 0u, pipes[15].size.nBytes);

    const NnUint outputPipes[] = {1u, 5u, 7u, 9u, 11u, 17u};
    for (size_t i = 0; i < sizeof(outputPipes) / sizeof(outputPipes[0]); ++i) {
        const NnUint p = outputPipes[i];
        device.writePipe(p, (const NnByte *)hostF32[p].data(), 0u, pipes[p].size.nBytes);
    }

    std::vector<float> mulMultiplier = f32Pattern(buffers[0].size.length, 0.015f, 1.125f);
    std::vector<float> scaleValues = f32Pattern(buffers[1].size.length, 0.02f, 0.75f);
    device.writeBuffer(0u, (const NnByte *)mulMultiplier.data(), 0u, buffers[0].size.nBytes);
    device.writeBuffer(1u, (const NnByte *)scaleValues.data(), 0u, buffers[1].size.nBytes);

    std::vector<float> expectedCast = hostF32[1];
    for (NnUint z = 0u; z < pipes[0].size.z; ++z)
        for (NnUint y = 0u; y < nBatches; ++y)
            for (NnUint x = 2u; x < 11u; ++x)
                expectedCast[f32Index(pipes[1].size, z, y, x)] = hostF32[0][f32Index(pipes[0].size, z, y, x)];

    std::vector<float> expectedSilu = hostF32[2];
    for (NnUint z = 0u; z < pipes[2].size.z; ++z)
        for (NnUint y = 0u; y < nBatches; ++y)
            for (NnUint x = 3u; x < 14u; ++x) {
                const NnSize idx = f32Index(pipes[2].size, z, y, x);
                const float v = expectedSilu[idx];
                expectedSilu[idx] = v / (1.0f + std::exp(-v));
            }

    std::vector<float> expectedGelu = hostF32[3];
    for (NnUint y = 0u; y < nBatches; ++y)
        for (NnUint x = 1u; x < 14u; ++x) {
            const NnSize idx = f32Index(pipes[3].size, 0u, y, x);
            const float v = expectedGelu[idx];
            expectedGelu[idx] = 0.5f * v * (1.0f + std::tanh(0.79788456080286535587989211986876f * v * (1.0f + 0.044715f * v * v)));
        }

    std::vector<float> expectedMul = hostF32[5];
    for (NnUint z = 0u; z < pipes[4].size.z; ++z)
        for (NnUint y = 0u; y < nBatches; ++y)
            for (NnUint x = 2u; x < 15u; ++x) {
                const NnSize idx = f32Index(pipes[5].size, z, y, x);
                expectedMul[idx] = hostF32[4][f32Index(pipes[4].size, z, y, x)] * mulMultiplier[f32Index(buffers[0].size, z, y, x)];
            }

    std::vector<float> expectedScale = hostF32[7];
    for (NnUint z = 0u; z < pipes[6].size.z; ++z)
        for (NnUint y = 0u; y < nBatches; ++y)
            for (NnUint x = 1u; x < 16u; ++x) {
                const NnSize idx = f32Index(pipes[7].size, z, y, x);
                expectedScale[idx] = hostF32[6][f32Index(pipes[6].size, z, y, x)] * scaleValues[z * nBatches + y];
            }

    std::vector<float> expectedMergeAdd = hostF32[9];
    for (NnUint y = 0u; y < nBatches; ++y)
        for (NnUint x = 0u; x < pipes[9].size.x; ++x) {
            float v = expectedMergeAdd[f32Index(pipes[9].size, 0u, y, x)];
            for (NnUint s = 0u; s < 3u; ++s) v += hostF32[8][f32Index(pipes[8].size, 0u, y, s * pipes[9].size.x + x)];
            expectedMergeAdd[f32Index(pipes[9].size, 0u, y, x)] = v;
        }

    std::vector<float> expectedMergeSum = hostF32[11];
    for (NnUint y = 0u; y < nBatches; ++y)
        for (NnUint x = 0u; x < pipes[11].size.x; ++x) {
            float v = 0.0f;
            for (NnUint z = 0u; z < pipes[10].size.z; ++z) v += hostF32[10][f32Index(pipes[10].size, z, y, x)];
            expectedMergeSum[f32Index(pipes[11].size, 0u, y, x)] = v;
        }

    std::vector<NnBlockQ80> expectedCastQ80 = castQ80Initial;
    for (NnUint z = 0u; z < pipes[12].size.z; ++z)
        for (NnUint y = 0u; y < nBatches; ++y) {
            const NnSize src = f32Index(pipes[12].size, z, y, 32u);
            const NnSize dst = q80Index(pipes[13].size, z, y, 1u);
            quantizeF32toQ80(&hostF32[12][src], &expectedCastQ80[dst], 64u, 1u, 0u);
        }

    std::vector<NnBlockQ80> expectedRepeat(pipes[15].size.length / Q80_BLOCK_SIZE);
    for (NnUint y = 0u; y < nBatches; ++y) {
        NnBlockQ80 row[2];
        quantizeF32toQ80(&hostF32[14][f32Index(pipes[14].size, 0u, y, 0u)], row, 64u, 1u, 0u);
        for (NnUint z = 0u; z < pipes[15].size.z; ++z) {
            expectedRepeat[q80Index(pipes[15].size, z, y, 0u)] = row[0];
            expectedRepeat[q80Index(pipes[15].size, z, y, 1u)] = row[1];
        }
    }

    std::vector<float> expectedView2d = hostF32[17];
    for (NnUint y = 0u; y < nBatches; ++y)
        for (NnUint vy = 0u; vy < 2u; ++vy)
            for (NnUint vx = 0u; vx < 3u; ++vx) {
                const NnUint x = 1u + vy * 7u + vx * 2u;
                expectedView2d[f32Index(pipes[17].size, 0u, y, x)] = hostF32[16][f32Index(pipes[16].size, 0u, y, x)];
            }

    std::unique_ptr<NnDeviceSegment> segment(device.createSegment(0u));
    segment->forward(0u, 1u, 0u, nBatches);

    std::vector<float> got;
    got.resize(expectedCast.size()); std::memcpy(got.data(), execution.pipes[1], pipes[1].size.nBytes);
    expectF32Near("CUDA CAST F32->F32 oracle", got, expectedCast, 1.0e-5f);
    got.resize(expectedSilu.size()); std::memcpy(got.data(), execution.pipes[2], pipes[2].size.nBytes);
    expectF32Near("CUDA SILU oracle", got, expectedSilu, 2.0e-5f);
    got.resize(expectedGelu.size()); std::memcpy(got.data(), execution.pipes[3], pipes[3].size.nBytes);
    expectF32Near("CUDA GELU oracle", got, expectedGelu, 2.0e-5f);
    got.resize(expectedMul.size()); std::memcpy(got.data(), execution.pipes[5], pipes[5].size.nBytes);
    expectF32Near("CUDA MUL oracle", got, expectedMul, 1.0e-5f);
    got.resize(expectedScale.size()); std::memcpy(got.data(), execution.pipes[7], pipes[7].size.nBytes);
    expectF32Near("CUDA SCALE oracle", got, expectedScale, 1.0e-5f);
    got.resize(expectedMergeAdd.size()); std::memcpy(got.data(), execution.pipes[9], pipes[9].size.nBytes);
    expectF32Near("CUDA MERGE_ADD oracle", got, expectedMergeAdd, 1.0e-5f);
    got.resize(expectedMergeSum.size()); std::memcpy(got.data(), execution.pipes[11], pipes[11].size.nBytes);
    expectF32Near("CUDA MERGE_SUM oracle", got, expectedMergeSum, 1.0e-5f);

    std::vector<NnBlockQ80> gotCastQ80(expectedCastQ80.size());
    std::memcpy(gotCastQ80.data(), execution.pipes[13], pipes[13].size.nBytes);
    expectQ80DequantNear("CUDA CAST F32->Q80 oracle", gotCastQ80, expectedCastQ80, (NnUint)pipes[13].size.length, 1.0e-6f);

    std::vector<NnBlockQ80> gotRepeat(expectedRepeat.size());
    std::memcpy(gotRepeat.data(), execution.pipes[15], pipes[15].size.nBytes);
    expectQ80DequantNear("CUDA REPEAT_Z F32->Q80 oracle", gotRepeat, expectedRepeat, (NnUint)pipes[15].size.length, 1.0e-6f);

    got.resize(expectedView2d.size()); std::memcpy(got.data(), execution.pipes[17], pipes[17].size.nBytes);
    expectF32Near("CUDA NnTensorView 2D oracle", got, expectedView2d, 1.0e-5f);

    std::printf("CUDA PR4 elementwise/cast/aggregate tests: ok\n");
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
        initQuants();
        const int count = nnCudaDeviceCount();
        std::printf("CUDA devices: %d\n", count);
        nnCudaPrintDeviceInfo(gpuIndex);
        NnNetConfig netConfig{};
        NnNodeConfig nodeConfig{};
        NnCudaDevice device(gpuIndex, &netConfig, &nodeConfig, nullptr, nullptr);
        std::printf("CUDA device lifecycle: ok\n");
        runMemoryDataPathTest(gpuIndex);
        runPr4ElementwiseOpsTest(gpuIndex);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "CUDA test failed: %s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
