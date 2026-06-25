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

static void expectF32NearAbsRel(const char *name, const std::vector<float> &actual, const std::vector<float> &expected, float maxAbsError, float maxRelError) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(name) + " size mismatch");
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    size_t maxIndex = 0u;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float denom = std::max(std::fabs(expected[i]), 1.0e-6f);
        const float rel = diff / denom;
        if (diff > maxAbs || rel > maxRel) {
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            maxIndex = i;
        }
    }
    if (maxAbs > maxAbsError || maxRel > maxRelError) {
        char msg[512];
        std::snprintf(msg, sizeof(msg),
            "%s max_abs_error=%g max_relative_error=%g exceeds abs=%g rel=%g at %zu: got=%g expected=%g",
            name, maxAbs, maxRel, maxAbsError, maxRelError, maxIndex, actual[maxIndex], expected[maxIndex]);
        throw std::runtime_error(msg);
    }
    std::printf("%s: ok max_abs_error=%g max_relative_error=%g\n", name, maxAbs, maxRel);
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

static void applyMatmulExpected(
    std::vector<float> &output,
    const NnSize3D &outputSize,
    const std::vector<float> &input,
    const NnSize3D &inputSize,
    const std::vector<float> &weight,
    const NnSize3D &weightSize,
    const NnMatmulOpConfig &config,
    const std::vector<float> &activeExpertIndexes,
    NnUint activeExpertRowStride,
    NnUint batchSize) {
    const NnUint nActiveExpertsOr1 = config.nActiveExperts == 0u ? 1u : config.nActiveExperts;
    const NnUint aOffset = config.aView.offset;
    const NnUint cOffset = config.cView.offset;
    const NnUint aLen = config.aView.sizeX == 0u ? inputSize.x : config.aView.sizeX;
    const NnUint cLen = config.cView.sizeX == 0u ? outputSize.x : config.cView.sizeX;
    const NnUint kTotal = weightSize.y;
    const NnUint dTotal = weightSize.x;
    for (NnUint y = 0u; y < batchSize; ++y) {
        for (NnUint e = 0u; e < nActiveExpertsOr1; ++e) {
            const NnUint activeExpert = config.nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[(NnSize)y * activeExpertRowStride + e];
            for (NnUint i = 0u; i < cLen; ++i) {
                float sum = 0.0f;
                for (NnUint j = 0u; j < aLen; ++j) {
                    const float x = input[f32Index(inputSize, e, y, aOffset + j)];
                    NnUint wOut = i;
                    NnUint wIn = j;
                    if (config.view == 1u) {
                        wOut = config.outStart + i;
                    } else if (config.view == 2u) {
                        wOut = config.outStart + i;
                        wIn = config.inStart + j;
                    }
                    sum += weight[((NnSize)activeExpert * kTotal * dTotal) + (NnSize)wOut * kTotal + wIn] * x;
                }
                output[f32Index(outputSize, e, y, cOffset + i)] = sum;
            }
        }
    }
}

static float q80q40BlockDotHost(const NnBlockQ80 *xb, const NnBlockQ40 *wb) {
    int acc = 0;
    for (NnUint k = 0u; k < Q40_BLOCK_SIZE / 2u; ++k) {
        const int w0 = (int)(wb->qs[k] & 0x0fu) - 8;
        const int w1 = (int)(wb->qs[k] >> 4) - 8;
        const int x0 = (int)xb->qs[k];
        const int x1 = (int)xb->qs[k + Q80_BLOCK_SIZE / 2u];
        acc += w0 * x0 + w1 * x1;
    }
    return (float)acc * CONVERT_F16_TO_F32(xb->d) * CONVERT_F16_TO_F32(wb->d);
}

static void applyMatmulQ80Q40Expected(
    std::vector<float> &output,
    const NnSize3D &outputSize,
    const std::vector<NnBlockQ80> &input,
    const NnSize3D &inputSize,
    const std::vector<NnBlockQ40> &weight,
    const NnSize3D &weightSize,
    const NnMatmulOpConfig &config,
    const std::vector<float> &activeExpertIndexes,
    NnUint activeExpertRowStride,
    NnUint batchSize) {
    const NnUint nActiveExpertsOr1 = config.nActiveExperts == 0u ? 1u : config.nActiveExperts;
    const NnUint aOffset = config.aView.offset;
    const NnUint cOffset = config.cView.offset;
    const NnUint aLen = config.aView.sizeX == 0u ? inputSize.x : config.aView.sizeX;
    const NnUint cLen = config.cView.sizeX == 0u ? outputSize.x : config.cView.sizeX;
    const NnUint aOffsetBlocks = aOffset / Q80_BLOCK_SIZE;
    const NnUint aBlocks = aLen / Q80_BLOCK_SIZE;
    const NnUint inputBlocksPerRow = inputSize.x / Q80_BLOCK_SIZE;
    const NnUint weightBlocksPerRow = weightSize.y / Q40_BLOCK_SIZE;
    const NnUint inStartBlocks = config.inStart / Q40_BLOCK_SIZE;
    for (NnUint y = 0u; y < batchSize; ++y) {
        for (NnUint e = 0u; e < nActiveExpertsOr1; ++e) {
            const NnUint activeExpert = config.nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[(NnSize)y * activeExpertRowStride + e];
            const NnBlockQ80 *x = input.data() + ((NnSize)e * inputSize.y + y) * inputBlocksPerRow + aOffsetBlocks;
            for (NnUint i = 0u; i < cLen; ++i) {
                const NnUint row = config.view == 0u ? i : config.outStart + i;
                const NnUint firstBlock = config.view == 2u ? inStartBlocks : 0u;
                const NnBlockQ40 *w = weight.data() + ((NnSize)activeExpert * weightSize.x + row) * weightBlocksPerRow + firstBlock;
                float sum = 0.0f;
                for (NnUint b = 0u; b < aBlocks; ++b) {
                    sum += q80q40BlockDotHost(x + b, w + b);
                }
                output[f32Index(outputSize, e, y, cOffset + i)] = sum;
            }
        }
    }
}

static std::vector<float> qPattern(NnSize n, NnUint mode, float scale, float bias) {
    std::vector<float> out(n);
    for (NnSize i = 0u; i < n; ++i) {
        if (mode == 0u) {
            const int v = (int)((i * 29u + 7u) % 101u) - 50;
            out[i] = (float)v * scale + bias;
        } else if (mode == 1u) {
            out[i] = 0.0f;
        } else if (mode == 2u) {
            const float sign = (i % 2u) == 0u ? 1.0f : -1.0f;
            out[i] = sign * (2.0f + (float)(i % 17u) * scale);
        } else {
            out[i] = ((i % 2u) == 0u ? 1.0f : -1.0f) * (0.25f + (float)(i % 5u) * scale);
        }
    }
    return out;
}

static void applyRopeExpected(std::vector<float> &x, const NnSize3D &size, const std::vector<float> &positions, const NnRopeOpConfig &config, NnUint batchSize) {
    std::vector<float> cache(config.slice.cacheSize.length, 0.0f);
    fullfillRopeCache(&config, cache.data());
    const bool isQ = config.isQ == 1u;
    const NnUint offset = config.view.offset;
    const NnUint len = config.view.sizeX == 0u ? size.x : config.view.sizeX;

    for (NnUint y = 0u; y < batchSize; ++y) {
        const NnUint pos = (NnUint)positions[y];
        float *row = &x[f32Index(size, 0u, y, 0u)];
        if (config.type == ROPE_LLAMA || config.type == ROPE_LLAMA3_1) {
            const NnUint shift = isQ ? config.slice.qShift : 0u;
            const float *posCache = cache.data() + (NnSize)pos * config.slice.sliceDim + shift + offset;
            for (NnUint i = 0u; i < len; i += 2u) {
                const float fcr = posCache[i];
                const float fci = posCache[i + 1u];
                const float v0 = row[offset + i];
                const float v1 = row[offset + i + 1u];
                row[offset + i] = v0 * fcr - v1 * fci;
                row[offset + i + 1u] = v0 * fci + v1 * fcr;
            }
        } else if (config.type == ROPE_FALCON) {
            const NnUint headDim = config.slice.headDim;
            const float *posCache = cache.data() + (NnSize)pos * headDim;
            for (NnUint h = 0u; h < len / headDim; ++h) {
                const NnUint base = offset + h * headDim;
                for (NnUint j = 0u; j < headDim / 2u; ++j) {
                    const float fcr = posCache[j];
                    const float fci = posCache[j + headDim / 2u];
                    const float v0 = row[base + j];
                    const float v1 = row[base + j + headDim / 2u];
                    row[base + j] = v0 * fcr - v1 * fci;
                    row[base + j + headDim / 2u] = v0 * fci + v1 * fcr;
                }
            }
        }
    }
}

static void applySoftmaxExpected(std::vector<float> &x, const NnSize3D &size, NnUint batchSize, NnUint offset, NnUint len, NnUint stride) {
    for (NnUint y = 0u; y < batchSize; ++y) {
        float *row = &x[f32Index(size, 0u, y, 0u)];
        float maxv = -std::numeric_limits<float>::infinity();
        for (NnUint i = 0u; i < len; ++i) {
            maxv = std::max(maxv, row[offset + i * stride]);
        }
        float sum = 0.0f;
        for (NnUint i = 0u; i < len; ++i) {
            sum += std::exp(row[offset + i * stride] - maxv);
        }
        for (NnUint i = 0u; i < len; ++i) {
            const NnUint idx = offset + i * stride;
            row[idx] = std::exp(row[idx] - maxv) / sum;
        }
    }
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

static void runPr5EmbeddingNormRopeShiftSoftmaxTest(NnUint gpuIndex) {
    const NnUint nBatches = 4u;
    const NnUint activeBatches = 3u;
    const NnUint seqLen = 8u;
    const NnUint kvStride = 8u;
    const float sentinel = -77.0f;

    NnRopeSlice llamaSlice = sliceRope(ROPE_LLAMA, 16u, 8u, 1u, 1u, 16u, 8u, 10000.0f, 0u);
    NnRopeSlice falconSlice = sliceRope(ROPE_FALCON, 16u, 8u, 1u, 1u, 16u, 8u, 10000.0f, 0u);
    NnRopeSlice llama31Slice = sliceRope(ROPE_LLAMA3_1, 16u, 8u, 1u, 1u, 16u, 8u, 500000.0f, 0u);

    NnPipeConfig pipes[11];
    pipes[0] = {(char *)"token", size2D(F_32, nBatches, 1u)};
    pipes[1] = {(char *)"embedding_out", size2D(F_32, nBatches, 6u)};
    pipes[2] = {(char *)"rms_in", size2D(F_32, nBatches, 8u)};
    pipes[3] = {(char *)"rms_out", size2D(F_32, nBatches, 8u)};
    pipes[4] = {(char *)"rope_llama", size2D(F_32, nBatches, 16u)};
    pipes[5] = {(char *)"rope_falcon", size2D(F_32, nBatches, 16u)};
    pipes[6] = {(char *)"rope_llama31", size2D(F_32, nBatches, 16u)};
    pipes[7] = {(char *)"position", size2D(F_32, nBatches, 1u)};
    pipes[8] = {(char *)"shift_in", size2D(F_32, nBatches, 4u)};
    pipes[9] = {(char *)"shift_index", size2D(F_32, nBatches, 1u)};
    pipes[10] = {(char *)"softmax", size2D(F_32, nBatches, 9u)};

    NnBufferConfig buffers[5];
    buffers[0] = {(char *)"inv_rms", size2D(F_32, nBatches, 2u)};
    buffers[1] = {(char *)"rope_cache_llama", llamaSlice.cacheSize};
    buffers[2] = {(char *)"rope_cache_falcon", falconSlice.cacheSize};
    buffers[3] = {(char *)"rope_cache_llama31", llama31Slice.cacheSize};
    buffers[4] = {(char *)"kv_cache", size1D(F_32, seqLen * kvStride)};

    NnEmbeddingOpConfig embeddingCfg{};
    NnInvRmsOpConfig invRmsCfg{1.0e-5f, 2u, NnTensorView{}};
    NnRmsNormOpConfig rmsNormCfg{0u, 2u, NnTensorView{}};
    NnRopeOpConfig ropeLlamaCfg{ROPE_LLAMA, 1u, 7u, 1u, 1.0f, 1.0f, 4.0f, 8192u, llamaSlice, NnTensorView{}};
    NnRopeOpConfig ropeFalconCfg{ROPE_FALCON, 1u, 7u, 2u, 1.0f, 1.0f, 4.0f, 8192u, falconSlice, NnTensorView{8u, 0u, 8u, 0u, 1u}};
    NnRopeOpConfig ropeLlama31Cfg{ROPE_LLAMA3_1, 1u, 7u, 3u, 8.0f, 1.0f, 4.0f, 8192u, llama31Slice, NnTensorView{}};
    NnShiftOpCodeConfig shiftCfg{9u, 2u, kvStride, 2u};
    NnSoftmaxOpCodeConfig softmaxCfg{NnTensorView{1u, 0u, 7u, 0u, 1u}};

    NnOpConfig ops[8];
    std::memset(ops, 0, sizeof(ops));
    ops[0].code = OP_EMBEDDING; ops[0].name = (char *)"embedding"; ops[0].index = 0u;
    ops[0].input = pointerBatchConfig(SRC_PIPE, 0u); ops[0].output = pointerBatchConfig(SRC_PIPE, 1u);
    ops[0].weightSize = size2D(F_32, 5u, 6u); ops[0].config = (NnByte *)&embeddingCfg; ops[0].configSize = sizeof(embeddingCfg);
    ops[1].code = OP_INV_RMS; ops[1].name = (char *)"inv_rms"; ops[1].index = 1u;
    ops[1].input = pointerBatchConfig(SRC_PIPE, 2u); ops[1].output = pointerBatchConfig(SRC_BUFFER, 0u);
    ops[1].config = (NnByte *)&invRmsCfg; ops[1].configSize = sizeof(invRmsCfg);
    ops[2].code = OP_RMS_NORM; ops[2].name = (char *)"rms_norm"; ops[2].index = 2u;
    ops[2].input = pointerBatchConfig(SRC_PIPE, 2u); ops[2].output = pointerBatchConfig(SRC_PIPE, 3u);
    ops[2].weightSize = size1D(F_32, 4u); ops[2].config = (NnByte *)&rmsNormCfg; ops[2].configSize = sizeof(rmsNormCfg);
    ops[3].code = OP_ROPE; ops[3].name = (char *)"rope_llama"; ops[3].index = 3u;
    ops[3].input = pointerBatchConfig(SRC_PIPE, 4u); ops[3].output = pointerBatchConfig(SRC_PIPE, 4u);
    ops[3].config = (NnByte *)&ropeLlamaCfg; ops[3].configSize = sizeof(ropeLlamaCfg);
    ops[4].code = OP_ROPE; ops[4].name = (char *)"rope_falcon"; ops[4].index = 4u;
    ops[4].input = pointerBatchConfig(SRC_PIPE, 5u); ops[4].output = pointerBatchConfig(SRC_PIPE, 5u);
    ops[4].config = (NnByte *)&ropeFalconCfg; ops[4].configSize = sizeof(ropeFalconCfg);
    ops[5].code = OP_ROPE; ops[5].name = (char *)"rope_llama31"; ops[5].index = 5u;
    ops[5].input = pointerBatchConfig(SRC_PIPE, 6u); ops[5].output = pointerBatchConfig(SRC_PIPE, 6u);
    ops[5].config = (NnByte *)&ropeLlama31Cfg; ops[5].configSize = sizeof(ropeLlama31Cfg);
    ops[6].code = OP_SHIFT; ops[6].name = (char *)"shift_strided_kv"; ops[6].index = 6u;
    ops[6].input = pointerBatchConfig(SRC_PIPE, 8u); ops[6].output = pointerRawConfig(SRC_BUFFER, 4u);
    ops[6].config = (NnByte *)&shiftCfg; ops[6].configSize = sizeof(shiftCfg);
    ops[7].code = OP_SOFTMAX; ops[7].name = (char *)"softmax"; ops[7].index = 7u;
    ops[7].input = pointerBatchConfig(SRC_PIPE, 10u); ops[7].output = pointerBatchConfig(SRC_PIPE, 10u);
    ops[7].config = (NnByte *)&softmaxCfg; ops[7].configSize = sizeof(softmaxCfg);

    NnSegmentConfig segments[1];
    segments[0].nOps = 8u;
    segments[0].ops = ops;
    segments[0].nSyncs = 0u;
    segments[0].syncs = nullptr;

    NnNetConfig netConfig{};
    netConfig.nBatches = nBatches;
    netConfig.nNodes = 1u;
    netConfig.nPipes = 11u;
    netConfig.pipes = pipes;
    netConfig.nPreSyncs = 0u;
    netConfig.preSyncs = nullptr;

    NnNodeConfig nodeConfig{};
    nodeConfig.nodeIndex = 0u;
    nodeConfig.nBuffers = 5u;
    nodeConfig.buffers = buffers;
    nodeConfig.nSegments = 1u;
    nodeConfig.segments = segments;

    NnNetExecution execution(1u, &netConfig);
    execution.setBatchSize(activeBatches);
    NnCudaDevice device(gpuIndex, &netConfig, &nodeConfig, &execution, nullptr);
    std::unique_ptr<NnCudaDeviceSegment> segment((NnCudaDeviceSegment *)device.createSegment(0u));

    std::vector<float> embeddingWeight(ops[0].weightSize.length);
    for (NnUint token = 0u; token < 5u; ++token)
        for (NnUint x = 0u; x < 6u; ++x)
            embeddingWeight[token * 6u + x] = (float)(token * 100u + x);
    segment->loadWeight(0u, 0u, ops[0].weightSize.nBytes, (NnByte *)embeddingWeight.data());

    std::vector<float> rmsWeight = {0.5f, 0.75f, 1.0f, 1.25f};
    segment->loadWeight(2u, 0u, ops[2].weightSize.nBytes, (NnByte *)rmsWeight.data());

    std::vector<float> tokens(nBatches, 0.0f);
    tokens[0] = 0.0f; tokens[1] = 3.0f; tokens[2] = 4.0f;
    std::memcpy(execution.pipes[0], tokens.data(), pipes[0].size.nBytes);

    std::vector<float> rmsIn = f32Pattern(pipes[2].size.length, 0.043f, 0.3f);
    std::memcpy(execution.pipes[2], rmsIn.data(), pipes[2].size.nBytes);

    std::vector<float> ropeLlama = f32Pattern(pipes[4].size.length, 0.017f, -0.2f);
    std::vector<float> ropeFalcon = f32Pattern(pipes[5].size.length, 0.019f, 0.1f);
    std::vector<float> ropeLlama31 = f32Pattern(pipes[6].size.length, 0.021f, -0.15f);
    std::memcpy(execution.pipes[4], ropeLlama.data(), pipes[4].size.nBytes);
    std::memcpy(execution.pipes[5], ropeFalcon.data(), pipes[5].size.nBytes);
    std::memcpy(execution.pipes[6], ropeLlama31.data(), pipes[6].size.nBytes);

    std::vector<float> positions(nBatches, 0.0f);
    positions[0] = 0.0f; positions[1] = 1.0f; positions[2] = 2.0f;
    std::memcpy(execution.pipes[7], positions.data(), pipes[7].size.nBytes);

    std::vector<float> shiftIn = f32Pattern(pipes[8].size.length, 0.11f, 2.0f);
    std::vector<float> shiftIndex(nBatches, 0.0f);
    shiftIndex[0] = 0.0f; shiftIndex[1] = 1.0f; shiftIndex[2] = 2.0f;
    std::memcpy(execution.pipes[8], shiftIn.data(), pipes[8].size.nBytes);
    std::memcpy(execution.pipes[9], shiftIndex.data(), pipes[9].size.nBytes);

    std::vector<float> softmax = f32Pattern(pipes[10].size.length, 0.067f, -1.0f);
    std::memcpy(execution.pipes[10], softmax.data(), pipes[10].size.nBytes);

    std::vector<float> kvExpected(seqLen * kvStride, sentinel);
    device.writeBuffer(4u, (const NnByte *)kvExpected.data(), 0u, buffers[4].size.nBytes);

    std::vector<float> expectedEmbedding(pipes[1].size.length, 0.0f);
    for (NnUint y = 0u; y < activeBatches; ++y) {
        const NnUint token = (NnUint)tokens[y];
        for (NnUint x = 0u; x < 6u; ++x) {
            expectedEmbedding[f32Index(pipes[1].size, 0u, y, x)] = embeddingWeight[token * 6u + x];
        }
    }

    std::vector<float> expectedInvRms(buffers[0].size.length, 0.0f);
    std::vector<float> expectedRmsOut(pipes[3].size.length, 0.0f);
    for (NnUint y = 0u; y < activeBatches; ++y) {
        for (NnUint col = 0u; col < 2u; ++col) {
            float ss = 0.0f;
            for (NnUint x = 0u; x < 4u; ++x) {
                const float v = rmsIn[f32Index(pipes[2].size, 0u, y, col * 4u + x)];
                ss += v * v;
            }
            const float inv = 1.0f / std::sqrt(ss / 4.0f + invRmsCfg.epsilon);
            expectedInvRms[f32Index(buffers[0].size, 0u, y, col)] = inv;
            for (NnUint x = 0u; x < 4u; ++x) {
                const NnUint globalX = col * 4u + x;
                expectedRmsOut[f32Index(pipes[3].size, 0u, y, globalX)] =
                    rmsWeight[x] * inv * rmsIn[f32Index(pipes[2].size, 0u, y, globalX)];
            }
        }
    }

    std::vector<float> expectedRopeLlama = ropeLlama;
    std::vector<float> expectedRopeFalcon = ropeFalcon;
    std::vector<float> expectedRopeLlama31 = ropeLlama31;
    applyRopeExpected(expectedRopeLlama, pipes[4].size, positions, ropeLlamaCfg, activeBatches);
    applyRopeExpected(expectedRopeFalcon, pipes[5].size, positions, ropeFalconCfg, activeBatches);
    applyRopeExpected(expectedRopeLlama31, pipes[6].size, positions, ropeLlama31Cfg, activeBatches);

    for (NnUint y = 0u; y < activeBatches; ++y) {
        const NnUint row = (NnUint)shiftIndex[y];
        for (NnUint x = 0u; x < 4u; ++x) {
            kvExpected[row * kvStride + shiftCfg.dstColStart + x] = shiftIn[f32Index(pipes[8].size, 0u, y, x)];
        }
    }

    std::vector<float> expectedSoftmax = softmax;
    applySoftmaxExpected(expectedSoftmax, pipes[10].size, activeBatches, 1u, 7u, 1u);

    segment->forward(0u, 1u, 0u, activeBatches);

    std::vector<float> got;
    got.resize(expectedEmbedding.size()); std::memcpy(got.data(), execution.pipes[1], pipes[1].size.nBytes);
    expectF32Near("CUDA EMBEDDING oracle", got, expectedEmbedding, 1.0e-5f);
    got.assign(expectedInvRms.size(), 0.0f);
    device.readBuffer(0u, (NnByte *)got.data(), 0u, buffers[0].size.nBytes);
    expectF32Near("CUDA INV_RMS oracle", got, expectedInvRms, 1.0e-4f);
    got.resize(expectedRmsOut.size()); std::memcpy(got.data(), execution.pipes[3], pipes[3].size.nBytes);
    expectF32Near("CUDA RMS_NORM oracle", got, expectedRmsOut, 1.0e-4f);
    got.resize(expectedRopeLlama.size()); std::memcpy(got.data(), execution.pipes[4], pipes[4].size.nBytes);
    expectF32Near("CUDA ROPE Llama oracle", got, expectedRopeLlama, 1.0e-4f);
    got.resize(expectedRopeFalcon.size()); std::memcpy(got.data(), execution.pipes[5], pipes[5].size.nBytes);
    expectF32Near("CUDA ROPE Falcon oracle", got, expectedRopeFalcon, 1.0e-4f);
    got.resize(expectedRopeLlama31.size()); std::memcpy(got.data(), execution.pipes[6], pipes[6].size.nBytes);
    expectF32Near("CUDA ROPE Llama3.1 oracle", got, expectedRopeLlama31, 1.0e-4f);
    got.resize(expectedSoftmax.size()); std::memcpy(got.data(), execution.pipes[10], pipes[10].size.nBytes);
    expectF32Near("CUDA SOFTMAX oracle", got, expectedSoftmax, 1.0e-4f);

    std::vector<float> gotKv(kvExpected.size(), 0.0f);
    device.readBuffer(4u, (NnByte *)gotKv.data(), 0u, buffers[4].size.nBytes);
    expectF32Near("CUDA SHIFT prefill strided KV oracle", gotKv, kvExpected, 1.0e-5f);

    std::vector<float> kvBeforeDecode = kvExpected;
    std::vector<float> decodeShiftIn(pipes[8].size.length, 0.0f);
    decodeShiftIn[0] = 9.25f; decodeShiftIn[1] = -3.5f; decodeShiftIn[2] = 4.75f; decodeShiftIn[3] = 8.5f;
    std::vector<float> decodeShiftIndex(nBatches, 0.0f);
    decodeShiftIndex[0] = 3.0f;
    positions[0] = 3.0f;
    std::memcpy(execution.pipes[7], positions.data(), pipes[7].size.nBytes);
    std::memcpy(execution.pipes[8], decodeShiftIn.data(), pipes[8].size.nBytes);
    std::memcpy(execution.pipes[9], decodeShiftIndex.data(), pipes[9].size.nBytes);

    for (NnUint x = 0u; x < 4u; ++x) {
        kvExpected[3u * kvStride + shiftCfg.dstColStart + x] = decodeShiftIn[x];
    }
    segment->forward(0u, 1u, 0u, 1u);
    gotKv.assign(kvExpected.size(), 0.0f);
    device.readBuffer(4u, (NnByte *)gotKv.data(), 0u, buffers[4].size.nBytes);
    expectF32Near("CUDA SHIFT decode strided KV oracle", gotKv, kvExpected, 1.0e-5f);
    for (NnUint row = 0u; row < 3u; ++row) {
        for (NnUint col = 0u; col < kvStride; ++col) {
            if (gotKv[row * kvStride + col] != kvBeforeDecode[row * kvStride + col]) {
                throw std::runtime_error("CUDA SHIFT decode modified prefill KV row");
            }
        }
    }
    std::printf("CUDA SHIFT prefill+decode KV preservation: ok\n");
    std::printf("CUDA PR5 embedding/norm/rope/shift/softmax tests: ok\n");
}

static void runPr6F32MatmulTest(NnUint gpuIndex) {
    const NnUint nBatches = 3u;

    NnPipeConfig pipes[10];
    pipes[0] = {(char *)"mm_small_a", size2D(F_32, nBatches, 5u)};
    pipes[1] = {(char *)"mm_small_c", size2D(F_32, nBatches, 7u)};
    pipes[2] = {(char *)"mm_large_k_a", size2D(F_32, nBatches, 257u)};
    pipes[3] = {(char *)"mm_large_k_c", size2D(F_32, nBatches, 9u)};
    pipes[4] = {(char *)"mm_row_slice_a", size2D(F_32, nBatches, 6u)};
    pipes[5] = {(char *)"mm_row_slice_c", size2D(F_32, nBatches, 8u)};
    pipes[6] = {(char *)"mm_col_slice_a", size2D(F_32, nBatches, 9u)};
    pipes[7] = {(char *)"mm_col_slice_c", size2D(F_32, nBatches, 6u)};
    pipes[8] = {(char *)"mm_moe_a", size3D(F_32, 2u, nBatches, 4u)};
    pipes[9] = {(char *)"mm_moe_c", size3D(F_32, 2u, nBatches, 3u)};

    NnBufferConfig buffers[1];
    buffers[0] = {(char *)"active_expert_indexes", size2D(F_32, nBatches, 2u)};

    NnMatmulOpConfig cfgSmall{};
    cfgSmall.activeExpertIndexesBufferIndex = 0u;
    cfgSmall.view = 0u;
    cfgSmall.aView = NnTensorView{0u, 0u, 5u, 0u, 1u};
    cfgSmall.cView = NnTensorView{0u, 0u, 7u, 0u, 1u};

    NnMatmulOpConfig cfgLarge{};
    cfgLarge.activeExpertIndexesBufferIndex = 0u;
    cfgLarge.view = 0u;
    cfgLarge.aView = NnTensorView{0u, 0u, 257u, 0u, 1u};
    cfgLarge.cView = NnTensorView{0u, 0u, 9u, 0u, 1u};

    NnMatmulOpConfig cfgRow{};
    cfgRow.activeExpertIndexesBufferIndex = 0u;
    cfgRow.view = 1u;
    cfgRow.outStart = 4u;
    cfgRow.outResidentStart = 2u;
    cfgRow.aView = NnTensorView{0u, 0u, 6u, 0u, 1u};
    cfgRow.cView = NnTensorView{2u, 0u, 3u, 0u, 1u};

    NnMatmulOpConfig cfgCol{};
    cfgCol.activeExpertIndexesBufferIndex = 0u;
    cfgCol.view = 2u;
    cfgCol.inStart = 2u;
    cfgCol.outStart = 1u;
    cfgCol.inResidentStart = 1u;
    cfgCol.aView = NnTensorView{2u, 0u, 4u, 0u, 1u};
    cfgCol.cView = NnTensorView{1u, 0u, 4u, 0u, 1u};

    NnMatmulOpConfig cfgMoe{};
    cfgMoe.nExperts = 4u;
    cfgMoe.nActiveExperts = 2u;
    cfgMoe.activeExpertIndexesBufferIndex = 0u;
    cfgMoe.view = 0u;
    cfgMoe.aView = NnTensorView{0u, 0u, 4u, 0u, 1u};
    cfgMoe.cView = NnTensorView{0u, 0u, 3u, 0u, 1u};

    NnOpConfig ops[5];
    std::memset(ops, 0, sizeof(ops));
    ops[0].code = OP_MATMUL; ops[0].name = (char *)"matmul_small_non_square"; ops[0].index = 0u;
    ops[0].input = pointerBatchConfig(SRC_PIPE, 0u); ops[0].output = pointerBatchConfig(SRC_PIPE, 1u);
    ops[0].weightSize = size2D(F_32, 5u, 7u); ops[0].config = (NnByte *)&cfgSmall; ops[0].configSize = sizeof(cfgSmall);
    ops[1].code = OP_MATMUL; ops[1].name = (char *)"matmul_large_k"; ops[1].index = 1u;
    ops[1].input = pointerBatchConfig(SRC_PIPE, 2u); ops[1].output = pointerBatchConfig(SRC_PIPE, 3u);
    ops[1].weightSize = size2D(F_32, 257u, 9u); ops[1].config = (NnByte *)&cfgLarge; ops[1].configSize = sizeof(cfgLarge);
    ops[2].code = OP_MATMUL; ops[2].name = (char *)"matmul_row_slice_view1"; ops[2].index = 2u;
    ops[2].input = pointerBatchConfig(SRC_PIPE, 4u); ops[2].output = pointerBatchConfig(SRC_PIPE, 5u);
    ops[2].weightSize = size2D(F_32, 6u, 8u); ops[2].config = (NnByte *)&cfgRow; ops[2].configSize = sizeof(cfgRow);
    ops[3].code = OP_MATMUL; ops[3].name = (char *)"matmul_col_slice_view2"; ops[3].index = 3u;
    ops[3].input = pointerBatchConfig(SRC_PIPE, 6u); ops[3].output = pointerBatchConfig(SRC_PIPE, 7u);
    ops[3].weightSize = size2D(F_32, 9u, 6u); ops[3].config = (NnByte *)&cfgCol; ops[3].configSize = sizeof(cfgCol);
    ops[4].code = OP_MATMUL; ops[4].name = (char *)"matmul_moe_expert"; ops[4].index = 4u;
    ops[4].input = pointerBatchConfig(SRC_PIPE, 8u); ops[4].output = pointerBatchConfig(SRC_PIPE, 9u);
    ops[4].weightSize = size3D(F_32, 4u, 4u, 3u); ops[4].config = (NnByte *)&cfgMoe; ops[4].configSize = sizeof(cfgMoe);

    NnSegmentConfig segments[1];
    segments[0].nOps = 5u;
    segments[0].ops = ops;
    segments[0].nSyncs = 0u;
    segments[0].syncs = nullptr;

    NnNetConfig netConfig{};
    netConfig.nBatches = nBatches;
    netConfig.nNodes = 1u;
    netConfig.nPipes = 10u;
    netConfig.pipes = pipes;
    netConfig.nPreSyncs = 0u;
    netConfig.preSyncs = nullptr;

    NnNodeConfig nodeConfig{};
    nodeConfig.nodeIndex = 0u;
    nodeConfig.nBuffers = 1u;
    nodeConfig.buffers = buffers;
    nodeConfig.nSegments = 1u;
    nodeConfig.segments = segments;

    NnNetExecution execution(1u, &netConfig);
    execution.setBatchSize(nBatches);
    NnCudaDevice device(gpuIndex, &netConfig, &nodeConfig, &execution, nullptr);
    std::unique_ptr<NnCudaDeviceSegment> segment((NnCudaDeviceSegment *)device.createSegment(0u));

    std::vector<std::vector<float> > inputs(10);
    std::vector<std::vector<float> > expected(10);
    for (NnUint i = 0u; i < 10u; ++i) {
        inputs[i] = f32Pattern(pipes[i].size.length, 0.006f + 0.001f * i, 0.35f + 0.03f * i);
        expected[i] = inputs[i];
        std::memcpy(execution.pipes[i], inputs[i].data(), pipes[i].size.nBytes);
        device.writePipe(i, (const NnByte *)inputs[i].data(), 0u, pipes[i].size.nBytes);
    }

    std::vector<float> weightSmall = f32Pattern(ops[0].weightSize.length, 0.004f, 0.2f);
    std::vector<float> weightLarge = f32Pattern(ops[1].weightSize.length, 0.0015f, 0.15f);
    std::vector<float> weightRow = f32Pattern(ops[2].weightSize.length, 0.005f, 0.25f);
    std::vector<float> weightCol = f32Pattern(ops[3].weightSize.length, 0.0045f, 0.3f);
    std::vector<float> weightMoe = f32Pattern(ops[4].weightSize.length, 0.0035f, 0.4f);
    segment->loadWeight(0u, 0u, ops[0].weightSize.nBytes, (NnByte *)weightSmall.data());
    segment->loadWeight(1u, 0u, ops[1].weightSize.nBytes, (NnByte *)weightLarge.data());
    segment->loadWeight(2u, 0u, ops[2].weightSize.nBytes, (NnByte *)weightRow.data());
    segment->loadWeight(3u, 0u, ops[3].weightSize.nBytes, (NnByte *)weightCol.data());
    segment->loadWeight(4u, 0u, ops[4].weightSize.nBytes, (NnByte *)weightMoe.data());

    std::vector<float> expertIndexes(buffers[0].size.length, 0.0f);
    for (NnUint y = 0u; y < nBatches; ++y) {
        for (NnUint e = 0u; e < 2u; ++e) {
            expertIndexes[y * 2u + e] = (float)((y + e * 2u) % 4u);
        }
    }
    device.writeBuffer(0u, (const NnByte *)expertIndexes.data(), 0u, buffers[0].size.nBytes);

    const std::vector<float> noExperts;
    applyMatmulExpected(expected[1], pipes[1].size, inputs[0], pipes[0].size, weightSmall, ops[0].weightSize, cfgSmall, noExperts, 0u, nBatches);
    applyMatmulExpected(expected[3], pipes[3].size, inputs[2], pipes[2].size, weightLarge, ops[1].weightSize, cfgLarge, noExperts, 0u, nBatches);
    applyMatmulExpected(expected[5], pipes[5].size, inputs[4], pipes[4].size, weightRow, ops[2].weightSize, cfgRow, noExperts, 0u, nBatches);
    applyMatmulExpected(expected[7], pipes[7].size, inputs[6], pipes[6].size, weightCol, ops[3].weightSize, cfgCol, noExperts, 0u, nBatches);
    applyMatmulExpected(expected[9], pipes[9].size, inputs[8], pipes[8].size, weightMoe, ops[4].weightSize, cfgMoe, expertIndexes, buffers[0].size.x, nBatches);

    segment->forward(0u, 1u, 0u, nBatches);

    std::vector<float> got;
    got.resize(expected[1].size()); std::memcpy(got.data(), execution.pipes[1], pipes[1].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL F32 view0 small/non-square oracle", got, expected[1], 1.0e-4f, 1.0e-4f);
    got.resize(expected[3].size()); std::memcpy(got.data(), execution.pipes[3], pipes[3].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL F32 view0 large-K oracle", got, expected[3], 1.0e-4f, 1.0e-4f);
    got.resize(expected[5].size()); std::memcpy(got.data(), execution.pipes[5], pipes[5].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL F32 view1 row-slice oracle", got, expected[5], 1.0e-4f, 1.0e-4f);
    got.resize(expected[7].size()); std::memcpy(got.data(), execution.pipes[7], pipes[7].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL F32 view2 col-slice oracle", got, expected[7], 1.0e-4f, 1.0e-4f);
    got.resize(expected[9].size()); std::memcpy(got.data(), execution.pipes[9], pipes[9].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL F32 MoE expert-Z oracle", got, expected[9], 1.0e-4f, 1.0e-4f);

    std::printf("CUDA PR6 F32 matmul/cuBLAS tests: ok\n");
}

static void runPr7Q80Q40MatmulTest(NnUint gpuIndex) {
    const NnUint nBatches = 3u;

    NnPipeConfig pipes[12];
    pipes[0] = {(char *)"q_random_small_a", size2D(F_Q80, nBatches, 64u)};
    pipes[1] = {(char *)"q_random_small_c", size2D(F_32, nBatches, 5u)};
    pipes[2] = {(char *)"q_zero_a", size2D(F_Q80, nBatches, 32u)};
    pipes[3] = {(char *)"q_zero_c", size2D(F_32, nBatches, 4u)};
    pipes[4] = {(char *)"q_extreme_large_a", size2D(F_Q80, nBatches, 512u)};
    pipes[5] = {(char *)"q_extreme_large_c", size2D(F_32, nBatches, 7u)};
    pipes[6] = {(char *)"q_row_slice_a", size2D(F_Q80, nBatches, 64u)};
    pipes[7] = {(char *)"q_row_slice_c", size2D(F_32, nBatches, 8u)};
    pipes[8] = {(char *)"q_col_slice_a", size2D(F_Q80, nBatches, 128u)};
    pipes[9] = {(char *)"q_col_slice_c", size2D(F_32, nBatches, 7u)};
    pipes[10] = {(char *)"q_moe_a", size3D(F_Q80, 2u, nBatches, 64u)};
    pipes[11] = {(char *)"q_moe_c", size3D(F_32, 2u, nBatches, 3u)};

    NnBufferConfig buffers[1];
    buffers[0] = {(char *)"q_active_expert_indexes", size2D(F_32, nBatches, 2u)};

    NnMatmulOpConfig cfgRandom{};
    cfgRandom.activeExpertIndexesBufferIndex = 0u;
    cfgRandom.view = 0u;
    cfgRandom.aView = NnTensorView{0u, 0u, 64u, 0u, 1u};
    cfgRandom.cView = NnTensorView{0u, 0u, 5u, 0u, 1u};

    NnMatmulOpConfig cfgZero{};
    cfgZero.activeExpertIndexesBufferIndex = 0u;
    cfgZero.view = 0u;
    cfgZero.aView = NnTensorView{0u, 0u, 32u, 0u, 1u};
    cfgZero.cView = NnTensorView{0u, 0u, 4u, 0u, 1u};

    NnMatmulOpConfig cfgLarge{};
    cfgLarge.activeExpertIndexesBufferIndex = 0u;
    cfgLarge.view = 0u;
    cfgLarge.aView = NnTensorView{0u, 0u, 512u, 0u, 1u};
    cfgLarge.cView = NnTensorView{0u, 0u, 7u, 0u, 1u};

    NnMatmulOpConfig cfgRow{};
    cfgRow.activeExpertIndexesBufferIndex = 0u;
    cfgRow.view = 1u;
    cfgRow.outStart = 3u;
    cfgRow.outResidentStart = 2u;
    cfgRow.aView = NnTensorView{0u, 0u, 64u, 0u, 1u};
    cfgRow.cView = NnTensorView{2u, 0u, 4u, 0u, 1u};

    NnMatmulOpConfig cfgCol{};
    cfgCol.activeExpertIndexesBufferIndex = 0u;
    cfgCol.view = 2u;
    cfgCol.inStart = 32u;
    cfgCol.outStart = 2u;
    cfgCol.inResidentStart = 32u;
    cfgCol.aView = NnTensorView{32u, 0u, 64u, 0u, 1u};
    cfgCol.cView = NnTensorView{1u, 0u, 5u, 0u, 1u};

    NnMatmulOpConfig cfgMoe{};
    cfgMoe.nExperts = 4u;
    cfgMoe.nActiveExperts = 2u;
    cfgMoe.activeExpertIndexesBufferIndex = 0u;
    cfgMoe.view = 0u;
    cfgMoe.aView = NnTensorView{0u, 0u, 64u, 0u, 1u};
    cfgMoe.cView = NnTensorView{0u, 0u, 3u, 0u, 1u};

    NnOpConfig ops[6];
    std::memset(ops, 0, sizeof(ops));
    ops[0].code = OP_MATMUL; ops[0].name = (char *)"q80q40_random_small"; ops[0].index = 0u;
    ops[0].input = pointerBatchConfig(SRC_PIPE, 0u); ops[0].output = pointerBatchConfig(SRC_PIPE, 1u);
    ops[0].weightSize = size2D(F_Q40, 64u, 5u); ops[0].config = (NnByte *)&cfgRandom; ops[0].configSize = sizeof(cfgRandom);
    ops[1].code = OP_MATMUL; ops[1].name = (char *)"q80q40_zero"; ops[1].index = 1u;
    ops[1].input = pointerBatchConfig(SRC_PIPE, 2u); ops[1].output = pointerBatchConfig(SRC_PIPE, 3u);
    ops[1].weightSize = size2D(F_Q40, 32u, 4u); ops[1].config = (NnByte *)&cfgZero; ops[1].configSize = sizeof(cfgZero);
    ops[2].code = OP_MATMUL; ops[2].name = (char *)"q80q40_extreme_large"; ops[2].index = 2u;
    ops[2].input = pointerBatchConfig(SRC_PIPE, 4u); ops[2].output = pointerBatchConfig(SRC_PIPE, 5u);
    ops[2].weightSize = size2D(F_Q40, 512u, 7u); ops[2].config = (NnByte *)&cfgLarge; ops[2].configSize = sizeof(cfgLarge);
    ops[3].code = OP_MATMUL; ops[3].name = (char *)"q80q40_row_slice_view1"; ops[3].index = 3u;
    ops[3].input = pointerBatchConfig(SRC_PIPE, 6u); ops[3].output = pointerBatchConfig(SRC_PIPE, 7u);
    ops[3].weightSize = size2D(F_Q40, 64u, 10u); ops[3].config = (NnByte *)&cfgRow; ops[3].configSize = sizeof(cfgRow);
    ops[4].code = OP_MATMUL; ops[4].name = (char *)"q80q40_col_slice_view2"; ops[4].index = 4u;
    ops[4].input = pointerBatchConfig(SRC_PIPE, 8u); ops[4].output = pointerBatchConfig(SRC_PIPE, 9u);
    ops[4].weightSize = size2D(F_Q40, 128u, 9u); ops[4].config = (NnByte *)&cfgCol; ops[4].configSize = sizeof(cfgCol);
    ops[5].code = OP_MATMUL; ops[5].name = (char *)"q80q40_moe_expert"; ops[5].index = 5u;
    ops[5].input = pointerBatchConfig(SRC_PIPE, 10u); ops[5].output = pointerBatchConfig(SRC_PIPE, 11u);
    ops[5].weightSize = size3D(F_Q40, 4u, 64u, 3u); ops[5].config = (NnByte *)&cfgMoe; ops[5].configSize = sizeof(cfgMoe);

    NnSegmentConfig segments[1];
    segments[0].nOps = 6u;
    segments[0].ops = ops;
    segments[0].nSyncs = 0u;
    segments[0].syncs = nullptr;

    NnNetConfig netConfig{};
    netConfig.nBatches = nBatches;
    netConfig.nNodes = 1u;
    netConfig.nPipes = 12u;
    netConfig.pipes = pipes;
    netConfig.nPreSyncs = 0u;
    netConfig.preSyncs = nullptr;

    NnNodeConfig nodeConfig{};
    nodeConfig.nodeIndex = 0u;
    nodeConfig.nBuffers = 1u;
    nodeConfig.buffers = buffers;
    nodeConfig.nSegments = 1u;
    nodeConfig.segments = segments;

    NnNetExecution execution(1u, &netConfig);
    execution.setBatchSize(nBatches);
    NnCudaDevice device(gpuIndex, &netConfig, &nodeConfig, &execution, nullptr);
    std::unique_ptr<NnCudaDeviceSegment> segment((NnCudaDeviceSegment *)device.createSegment(0u));

    std::vector<std::vector<NnBlockQ80> > qInputs(12);
    std::vector<std::vector<float> > outputInitial(12);
    std::vector<std::vector<float> > expected(12);
    const NnUint inputModes[6] = {0u, 1u, 2u, 3u, 0u, 3u};
    const NnUint inputPipeForOp[6] = {0u, 2u, 4u, 6u, 8u, 10u};
    const NnUint outputPipeForOp[6] = {1u, 3u, 5u, 7u, 9u, 11u};
    for (NnUint opi = 0u; opi < 6u; ++opi) {
        const NnUint p = inputPipeForOp[opi];
        std::vector<float> src = qPattern(pipes[p].size.length, inputModes[opi], 0.031f + 0.003f * opi, 0.05f * opi);
        qInputs[p].resize(pipes[p].size.length / Q80_BLOCK_SIZE);
        quantizeF32toQ80(src.data(), qInputs[p].data(), (NnUint)src.size(), 1u, 0u);
        std::memcpy(execution.pipes[p], qInputs[p].data(), pipes[p].size.nBytes);
    }
    for (NnUint opi = 0u; opi < 6u; ++opi) {
        const NnUint p = outputPipeForOp[opi];
        outputInitial[p] = f32Pattern(pipes[p].size.length, 0.011f + 0.001f * opi, -0.4f + 0.02f * opi);
        expected[p] = outputInitial[p];
        std::memcpy(execution.pipes[p], outputInitial[p].data(), pipes[p].size.nBytes);
        device.writePipe(p, (const NnByte *)outputInitial[p].data(), 0u, pipes[p].size.nBytes);
    }

    std::vector<std::vector<NnBlockQ40> > qWeights(6);
    const NnUint weightModes[6] = {0u, 1u, 2u, 3u, 0u, 3u};
    for (NnUint opi = 0u; opi < 6u; ++opi) {
        std::vector<float> w = qPattern(ops[opi].weightSize.length, weightModes[opi], 0.027f + 0.002f * opi, -0.03f * opi);
        qWeights[opi].resize(ops[opi].weightSize.length / Q40_BLOCK_SIZE);
        quantizeF32toQ40(w.data(), qWeights[opi].data(), (NnUint)w.size(), 1u, 0u);
        segment->loadWeight(opi, 0u, ops[opi].weightSize.nBytes, (NnByte *)qWeights[opi].data());
    }

    std::vector<float> expertIndexes(buffers[0].size.length, 0.0f);
    for (NnUint y = 0u; y < nBatches; ++y) {
        for (NnUint e = 0u; e < 2u; ++e) {
            expertIndexes[y * 2u + e] = (float)((y * 2u + e) % 4u);
        }
    }
    device.writeBuffer(0u, (const NnByte *)expertIndexes.data(), 0u, buffers[0].size.nBytes);

    const std::vector<float> noExperts;
    applyMatmulQ80Q40Expected(expected[1], pipes[1].size, qInputs[0], pipes[0].size, qWeights[0], ops[0].weightSize, cfgRandom, noExperts, 0u, nBatches);
    applyMatmulQ80Q40Expected(expected[3], pipes[3].size, qInputs[2], pipes[2].size, qWeights[1], ops[1].weightSize, cfgZero, noExperts, 0u, nBatches);
    applyMatmulQ80Q40Expected(expected[5], pipes[5].size, qInputs[4], pipes[4].size, qWeights[2], ops[2].weightSize, cfgLarge, noExperts, 0u, nBatches);
    applyMatmulQ80Q40Expected(expected[7], pipes[7].size, qInputs[6], pipes[6].size, qWeights[3], ops[3].weightSize, cfgRow, noExperts, 0u, nBatches);
    applyMatmulQ80Q40Expected(expected[9], pipes[9].size, qInputs[8], pipes[8].size, qWeights[4], ops[4].weightSize, cfgCol, noExperts, 0u, nBatches);
    applyMatmulQ80Q40Expected(expected[11], pipes[11].size, qInputs[10], pipes[10].size, qWeights[5], ops[5].weightSize, cfgMoe, expertIndexes, buffers[0].size.x, nBatches);

    segment->forward(0u, 1u, 0u, nBatches);

    std::vector<float> got;
    got.resize(expected[1].size()); std::memcpy(got.data(), execution.pipes[1], pipes[1].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL Q80xQ40 random small-K oracle", got, expected[1], 2.0e-3f, 1.0e-2f);
    got.resize(expected[3].size()); std::memcpy(got.data(), execution.pipes[3], pipes[3].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL Q80xQ40 all-zero oracle", got, expected[3], 2.0e-3f, 1.0e-2f);
    got.resize(expected[5].size()); std::memcpy(got.data(), execution.pipes[5], pipes[5].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL Q80xQ40 extreme large-K oracle", got, expected[5], 2.0e-3f, 1.0e-2f);
    got.resize(expected[7].size()); std::memcpy(got.data(), execution.pipes[7], pipes[7].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL Q80xQ40 view1 row-slice oracle", got, expected[7], 2.0e-3f, 1.0e-2f);
    got.resize(expected[9].size()); std::memcpy(got.data(), execution.pipes[9], pipes[9].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL Q80xQ40 view2 non-uniform col-slice oracle", got, expected[9], 2.0e-3f, 1.0e-2f);
    got.resize(expected[11].size()); std::memcpy(got.data(), execution.pipes[11], pipes[11].size.nBytes);
    expectF32NearAbsRel("CUDA MATMUL Q80xQ40 MoE expert-Z oracle", got, expected[11], 2.0e-3f, 1.0e-2f);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess) {
        cudaEventRecord(start, (cudaStream_t)device.getStream());
        segment->forward(0u, 1u, 0u, nBatches);
        cudaEventRecord(stop, (cudaStream_t)device.getStream());
        cudaEventSynchronize(stop);
        float elapsedMs = 0.0f;
        cudaEventElapsedTime(&elapsedMs, start, stop);
        std::printf("CUDA Q80xQ40 repeated forward without weight reload: ok elapsed_ms=%g\n", elapsedMs);
    }
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);

    {
        NnPipeConfig badPipes[2];
        badPipes[0] = {(char *)"bad_q_in", size2D(F_Q80, 1u, 64u)};
        badPipes[1] = {(char *)"bad_out", size2D(F_32, 1u, 2u)};
        NnBufferConfig badBuffers[1];
        badBuffers[0] = {(char *)"bad_idx", size2D(F_32, 1u, 1u)};
        NnMatmulOpConfig badCfg{};
        badCfg.activeExpertIndexesBufferIndex = 0u;
        badCfg.view = 0u;
        badCfg.aView = NnTensorView{1u, 0u, 32u, 0u, 1u};
        badCfg.cView = NnTensorView{0u, 0u, 2u, 0u, 1u};
        NnOpConfig badOp{};
        badOp.code = OP_MATMUL;
        badOp.name = (char *)"bad_q80q40_unaligned";
        badOp.input = pointerBatchConfig(SRC_PIPE, 0u);
        badOp.output = pointerBatchConfig(SRC_PIPE, 1u);
        badOp.weightSize = size2D(F_Q40, 32u, 2u);
        badOp.config = (NnByte *)&badCfg;
        badOp.configSize = sizeof(badCfg);
        NnSegmentConfig badSegment{};
        badSegment.nOps = 1u;
        badSegment.ops = &badOp;
        NnNetConfig badNet{};
        badNet.nBatches = 1u;
        badNet.nNodes = 1u;
        badNet.nPipes = 2u;
        badNet.pipes = badPipes;
        NnNodeConfig badNode{};
        badNode.nodeIndex = 0u;
        badNode.nBuffers = 1u;
        badNode.buffers = badBuffers;
        badNode.nSegments = 1u;
        badNode.segments = &badSegment;
        NnNetExecution badExecution(1u, &badNet);
        NnCudaDevice badDevice(gpuIndex, &badNet, &badNode, &badExecution, nullptr);
        bool rejected = false;
        try {
            std::unique_ptr<NnDeviceSegment> badCreated(badDevice.createSegment(0u));
        } catch (const std::exception &) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("CUDA Q80xQ40 unaligned matmul was not rejected at segment creation");
        std::printf("CUDA Q80xQ40 block-alignment rejection at segment creation: ok\n");
    }

    std::printf("CUDA PR7 Q80xQ40 custom matmul tests: ok\n");
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
        runPr5EmbeddingNormRopeShiftSoftmaxTest(gpuIndex);
        runPr6F32MatmulTest(gpuIndex);
        runPr7Q80Q40MatmulTest(gpuIndex);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "CUDA test failed: %s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
