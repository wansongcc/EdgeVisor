#include "nn-cpu-ops.cpp"
#include "nn-test-utils.hpp"
#include <vector>

// framework

void printPassed(const char *name) {
    printf("✅ %24s passed\n", name);
    fflush(stdout);
}

void rand(float *o, const NnUint n, const NnUint seed) {
    nn_test::fillRandom(o, n, seed + 123456u);
}

void compare_F32(const char *name, const float *a, const float *b, const NnUint n, const float epsilon) {
    nn_test::requireClose(name, a, b, n, epsilon);
    printPassed(name);
}

// tests

void testSplitThreads() {
    // <0; 32> across 3 threads
    {
        SPLIT_THREADS(a0Start, a0End, 32, 3, 0); // thread 0
        assert(a0Start == 0);
        assert(a0End == 11);
    }
    {
        SPLIT_THREADS(a1Start, a1End, 32, 3, 1); // thread 1
        assert(a1Start == 11);
        assert(a1End == 22);
    }
    {
        SPLIT_THREADS(a2Start, a2End, 32, 3, 2); // thread 2
        assert(a2Start == 22);
        assert(a2End == 32);
    }

    // <0; 4> across 8 threads
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 0); // thread 0
        assert(b0Start == 0);
        assert(b0End == 1);
    }
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 3); // thread 3
        assert(b0Start == 3);
        assert(b0End == 4);
    }
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 4); // thread 4
        assert(b0Start == 4); 
        assert(b0End == 4);
    }
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 7); // thread 7
        assert(b0Start == 4);
        assert(b0End == 4);
    }

    printPassed("splitThreads");
}

void testConvertF32toF16() {
    float x[] = {0.0f, 0.25f, 0.3456f, 1.0f};
    for (NnUint i = 0; i < sizeof(x) / sizeof(float); i++) {
        NnFp16 f16 = CONVERT_F32_TO_F16(x[i]);
        float f32 = CONVERT_F16_TO_F32(f16);
        compare_F32("convertF32toF16", &x[i], &f32, 1, 0.0005);
    }
}

// quantization
void testQuantization(const NnUint m) {
    std::vector<float> a(m * Q40_BLOCK_SIZE);
    std::vector<float> aTemp(m * Q40_BLOCK_SIZE);
    std::vector<NnBlockQ40> aQ40(m);
    std::vector<NnBlockQ80> aQ80(m);

    rand(a.data(), m * Q40_BLOCK_SIZE, m);

    quantizeF32toQ40(a.data(), aQ40.data(), m * Q40_BLOCK_SIZE, 1, 0);
    dequantizeQ40toF32(aQ40.data(), aTemp.data(), m * Q40_BLOCK_SIZE, 1, 0);

    compare_F32("testQuantization_Q40", a.data(), aTemp.data(), m * Q40_BLOCK_SIZE, 0.13);

    quantizeF32toQ80(a.data(), aQ80.data(), m * Q80_BLOCK_SIZE, 1, 0);
    dequantizeQ80toF32(aQ80.data(), aTemp.data(), m * Q80_BLOCK_SIZE, 1, 0);

    compare_F32("testQuantization_Q80", a.data(), aTemp.data(), m * Q80_BLOCK_SIZE, 0.01);
}

// invRms
void testInvRms() {
    const float epsilon = 0.00001;

    std::vector<float> x(8);
    x[0] = 0.1f;
    x[1] = 0.3f;
    x[2] = 0.2f;
    x[3] = 0.4f;
    x[4] = 0.6f;
    x[5] = 0.5f;
    x[6] = 0.0f;
    x[7] = 0.8f;

    const float y0 = invRms_F32(x.data(), 8, epsilon);
    float ev0 = 1.0f / 0.4402f;
    compare_F32("rms_F32", &y0, &ev0, 1, 0.001f);
}

// rmsNorm
void testRmsNorm(const NnUint m) {
    std::vector<float> x(m);
    std::vector<NnBlockQ80> xQ80(m / Q80_BLOCK_SIZE);
    std::vector<float> w(m);
    std::vector<float> y(m);
    std::vector<float> yTemp(m);

    rand(x.data(), m, m);
    rand(w.data(), m, m * m);
    quantizeF32toQ80(x.data(), xQ80.data(), m, 1, 0);
    const float rms = invRms_F32(x.data(), m, 1e-5f);

    rmsNorm_F32(y.data(), x.data(), rms, w.data(), m, 1, 0);
    rmsNorm_Q80_F32_F32(yTemp.data(), xQ80.data(), rms, w.data(), m, 1, 0);

    compare_F32("rmsNorm_Q80_F32_F32", y.data(), yTemp.data(), m, 0.01);
}

// a *= b
void testMul(const NnUint m) {
    const NnUint n = Q80_BLOCK_SIZE * m;

    std::vector<float> a0(n);
    std::vector<float> b0(n);

    std::vector<float> aQ(n);
    std::vector<NnBlockQ80> b1(n / Q80_BLOCK_SIZE);

    rand(a0.data(), n, m);
    rand(aQ.data(), n, m);
    rand(b0.data(), n, m);
    quantizeF32toQ80(b0.data(), b1.data(), n, 1, 0);

    mul_F32(a0.data(), a0.data(), b0.data(), n, 1, 0);
    mul_Q80_F32(aQ.data(), aQ.data(), b1.data(), n, 1, 0);

    compare_F32("mul_Q80_F32", a0.data(), aQ.data(), n, 0.005);
}

// y += x
void testAdd(const NnUint m) {
    const NnUint n = Q80_BLOCK_SIZE * m;

    std::vector<float> y(n);
    std::vector<float> yTemp(n);
    std::vector<float> x(n);
    std::vector<NnBlockQ80> xQ80(n / Q80_BLOCK_SIZE);

    rand(y.data(), n, m);
    rand(yTemp.data(), n, m);
    rand(x.data(), n, m);
    quantizeF32toQ80(x.data(), xQ80.data(), n, 1, 0);

    add_F32(y.data(), x.data(), n, 1, 0);
    add_Q80_F32(yTemp.data(), xQ80.data(), n, 1, 0);

    compare_F32("add_Q80_F32", y.data(), yTemp.data(), n, 0.01);
}

void testMergeSum() {
    float inp[] = {
        /* [z0, y0] */ 0.1f, 0.2f,
        /* [z0, y1] */ 0.3f, 0.4f,
        /* [z1, y0] */ 0.5f, 0.6f,
        /* [z1, y1] */ 0.7f, 0.8f,
    };
    float out[4];

    float *i[4] = {
        &inp[0],
        &inp[2],
        &inp[4],
        &inp[6],
    };
    float *o[2] = {
        &out[0],
        &out[2]
    };

    mergeSum_F32(o, i, 2u, 2u, 2u, 2u, 1u, 0u);

    const float expectedOutput[4] = {
        0.6f,
        0.8f,
        1.0f,
        1.2f,
    };
    compare_F32("mergeSum_F32", out, expectedOutput, 4u, 0.00000001f);
}

void testSoftmax() {
    std::vector<float> y(8);
    for (NnUint i = 0; i < 8; i++)
        y[i] = i / 8.0f;

    softmax_F32(y.data(), 8);

    float expectedOutput[8] = {
        0.077399f,
        0.087780f,
        0.099500f,
        0.112761f,
        0.127778f,
        0.144793f,
        0.164072f,
        0.185917f
    };
    compare_F32("softmax_F32", y.data(), expectedOutput, 8, 0.001);
}

void testSilu() {
    std::vector<float> y(8);
    for (NnUint i = 0; i < 8; i++)
        y[i] = i / 8.0f;

    silu_F32(y.data(), 8, 1, 0);

    float expectedOutput[8] = {
        0.000000f,
        0.066401f,
        0.140544f,
        0.222250f,
        0.311233f,
        0.407116f,
        0.509461f,
        0.617802f
    };
    compare_F32("silu_F32", y.data(), expectedOutput, 8, 0.001);
}

void testGeluViewSlice() {
    const NnUint n = 16u;
    const NnUint offset = 4u;
    const NnUint len = 6u;

    std::vector<float> y(n);
    std::vector<float> expected(n);
    for (NnUint i = 0; i < n; i++) {
        // deterministic inputs, include negatives
        y[i] = ((float)i - 8.0f) / 8.0f;
        expected[i] = y[i];
    }

    // Apply GELU only to the view slice for expected output
    gelu_F32(expected.data() + offset, len, 1u, 0u);

    // Run geluForward with a view config
    NnGeluOpCodeConfig cfg{NnTensorView{offset, 0u, len, 0u, 1u}};

    NnByte *io0 = (NnByte *)y.data();
    NnByte *io[1] = { io0 };

    NnCpuOpContext ctx{};
    ctx.name = "test_gelu_view";
    ctx.nBatches = 1;
    ctx.opConfig = &cfg;
    ctx.input = io;
    ctx.output = io;
    ctx.inputSize = size2D(F_32, 1u, n);
    ctx.outputSize = size2D(F_32, 1u, n);
    ctx.weightSize = size0();

    geluForward_F32_F32_F32(1u, 0u, 1u, &ctx);

    compare_F32("geluForward_view_slice", y.data(), expected.data(), n, 1e-6f);
}

void testRopeViewSlice() {
    const NnUint dim0 = 8u;
    const NnUint offset = 2u;
    const NnUint len = 4u;
    const NnUint seqLen = 4u;
    const NnUint pos = 1u;

    std::vector<float> x(dim0);
    std::vector<float> full(dim0);
    std::vector<float> sliced(dim0);

    for (NnUint i = 0; i < dim0; i++) {
        x[i] = ((float)i - 4.0f) / 4.0f;
        full[i] = x[i];
        sliced[i] = x[i];
    }

    // Build rope cache
    std::vector<float> cache(seqLen * dim0);
    NnRopeSlice slice{};
    slice.qDim0 = dim0;
    slice.qDimStart = 0u;
    slice.qDimEnd = dim0;
    slice.qShift = 0u;
    slice.kvDim = dim0;
    slice.kvDim0 = dim0;
    slice.kvDimStart = 0u;
    slice.sliceDim = dim0;
    slice.seqLen = seqLen;
    slice.headDim = dim0;
    slice.nKvHeads = 1u;
    slice.ropeTheta = 1000000.0f;

    float positionsVal[1] = { (float)pos };
    NnByte *pipes[1] = { (NnByte *)positionsVal };
    NnPipeConfig pipeCfgs[1] = {};
    pipeCfgs[0].name = (char *)"pos";
    pipeCfgs[0].size = size2D(F_32, 1u, 1u);

    NnByte flag = 0;
    NnByte *flags = &flag;
    NnByte *bufs[1] = { (NnByte *)cache.data() };
    NnBufferConfig bufCfgs[1] = {};
    bufCfgs[0].name = (char *)"rope_cache";
    bufCfgs[0].size = size2D(F_32, seqLen, dim0);

    // full
    NnRopeOpConfig fullCfg{};
    fullCfg.type = ROPE_LLAMA;
    fullCfg.isQ = 1u;
    fullCfg.positionPipeIndex = 0u;
    fullCfg.ropeCacheBufferIndex = 0u;
    fullCfg.ropeScalingFactor = 1.0f;
    fullCfg.ropeScalingLowFreqFactor = 1.0f;
    fullCfg.ropeScalingHighFreqFactor = 1.0f;
    fullCfg.ropeScalingOrigMaxSeqLen = 0u;
    fullCfg.slice = slice;
    fullCfg.view = NnTensorView{0u, 0u, 0u, 0u, 1u};

    NnByte *fullIn[1] = { (NnByte *)full.data() };
    NnCpuOpContext fullCtx{};
    fullCtx.name = "test_rope_full";
    fullCtx.nBatches = 1;
    fullCtx.bufferFlags = flags;
    fullCtx.buffers = bufs;
    fullCtx.bufferConfigs = bufCfgs;
    fullCtx.pipes = pipes;
    fullCtx.pipeConfigs = pipeCfgs;
    fullCtx.opConfig = &fullCfg;
    fullCtx.input = fullIn;
    fullCtx.inputSize = size2D(F_32, 1u, dim0);
    fullCtx.outputSize = size2D(F_32, 1u, dim0);

    initRopeForward_F32(&fullCtx);
    ropeForward_F32_F32(1u, 0u, 1u, &fullCtx);

    // slice
    NnRopeOpConfig sliceCfg = fullCfg;
    sliceCfg.view = NnTensorView{offset, 0u, len, 0u, 1u};

    NnByte *sliceIn[1] = { (NnByte *)sliced.data() };
    NnCpuOpContext sliceCtx = fullCtx;
    sliceCtx.name = "test_rope_slice";
    sliceCtx.opConfig = &sliceCfg;
    sliceCtx.input = sliceIn;

    // Expected: only slice region matches full; outside stays original.
    std::vector<float> expected(dim0);
    for (NnUint i = 0; i < dim0; i++) {
        expected[i] = x[i];
    }
    for (NnUint i = offset; i < offset + len; i++) {
        expected[i] = full[i];
    }

    ropeForward_F32_F32(1u, 0u, 1u, &sliceCtx);
    compare_F32("ropeForward_view_slice", sliced.data(), expected.data(), dim0, 1e-6f);
}


void testCastViewSlice() {
    // Case 1: F32 -> F32 (castForward_ANY) only overwrites slice
    {
        const NnUint n = 16u;
        const NnUint offset = 5u;
        const NnUint len = 7u;

        std::vector<float> input(n);
        std::vector<float> output(n);
        std::vector<float> expected(n);
        for (NnUint i = 0; i < n; i++) {
            input[i] = ((float)i - 8.0f) / 8.0f;
            output[i] = -123.0f;
            expected[i] = -123.0f;
        }
        for (NnUint i = offset; i < offset + len; i++) {
            expected[i] = input[i];
        }

        NnCastOpCodeConfig cfg{NnTensorView{offset, 0u, len, 0u, 1u}};
        NnByte *inArr[1] = { (NnByte *)input.data() };
        NnByte *outArr[1] = { (NnByte *)output.data() };

        NnCpuOpContext ctx{};
        ctx.name = "test_cast_f32_f32_view";
        ctx.nBatches = 1;
        ctx.opConfig = &cfg;
        ctx.input = inArr;
        ctx.output = outArr;
        ctx.inputSize = size2D(F_32, 1u, n);
        ctx.outputSize = size2D(F_32, 1u, n);
        ctx.weightSize = size0();

        castForward_ANY(1u, 0u, 1u, &ctx);
        compare_F32("castForward_F32_F32_view_slice", output.data(), expected.data(), n, 0.0f);
    }

    // Case 2: F32 -> Q80 (castForward_F32_Q80) only overwrites slice (block-aligned)
    {
        const NnUint n = 64u;
        const NnUint offset = 32u;
        const NnUint len = 32u;

        std::vector<float> input(n);
        for (NnUint i = 0; i < n; i++)
            input[i] = ((float)i - 32.0f) / 16.0f;

        // Initialize output as quantized zeros so "unchanged" region is stable.
        std::vector<float> zeros(n, 0.0f);
        std::vector<NnBlockQ80> outQ(n / Q80_BLOCK_SIZE);
        quantizeF32toQ80(zeros.data(), outQ.data(), n, 1u, 0u);

        std::vector<NnBlockQ80> expectedQ = outQ;
        std::vector<NnBlockQ80> sliceQ(len / Q80_BLOCK_SIZE);
        quantizeF32toQ80(input.data() + offset, sliceQ.data(), len, 1u, 0u);
        expectedQ[offset / Q80_BLOCK_SIZE] = sliceQ[0];

        NnCastOpCodeConfig cfg{NnTensorView{offset, 0u, len, 0u, 1u}};
        NnByte *inArr[1] = { (NnByte *)input.data() };
        NnByte *outArr[1] = { (NnByte *)outQ.data() };

        NnCpuOpContext ctx{};
        ctx.name = "test_cast_f32_q80_view";
        ctx.nBatches = 1;
        ctx.opConfig = &cfg;
        ctx.input = inArr;
        ctx.output = outArr;
        ctx.inputSize = size2D(F_32, 1u, n);
        ctx.outputSize = size2D(F_Q80, 1u, n);
        ctx.weightSize = size0();

        castForward_F32_Q80(1u, 0u, 1u, &ctx);

        std::vector<float> outF32(n);
        std::vector<float> expectedF32(n);
        dequantizeQ80toF32(outQ.data(), outF32.data(), n, 1u, 0u);
        dequantizeQ80toF32(expectedQ.data(), expectedF32.data(), n, 1u, 0u);
        compare_F32("castForward_F32_Q80_view_slice", outF32.data(), expectedF32.data(), n, 1e-6f);
    }

    // Case 3: Q80 -> F32 (castForward_Q80_F32) only overwrites slice (block-aligned)
    {
        const NnUint n = 64u;
        const NnUint offset = 32u;
        const NnUint len = 32u;

        std::vector<float> inputF32(n);
        for (NnUint i = 0; i < n; i++)
            inputF32[i] = ((float)i - 32.0f) / 16.0f;

        std::vector<NnBlockQ80> inputQ(n / Q80_BLOCK_SIZE);
        quantizeF32toQ80(inputF32.data(), inputQ.data(), n, 1u, 0u);

        std::vector<float> fullDeq(n);
        dequantizeQ80toF32(inputQ.data(), fullDeq.data(), n, 1u, 0u);

        std::vector<float> output(n, -321.0f);
        std::vector<float> expected(n, -321.0f);
        for (NnUint i = offset; i < offset + len; i++)
            expected[i] = fullDeq[i];

        NnCastOpCodeConfig cfg{NnTensorView{offset, 0u, len, 0u, 1u}};
        NnByte *inArr[1] = { (NnByte *)inputQ.data() };
        NnByte *outArr[1] = { (NnByte *)output.data() };

        NnCpuOpContext ctx{};
        ctx.name = "test_cast_q80_f32_view";
        ctx.nBatches = 1;
        ctx.opConfig = &cfg;
        ctx.input = inArr;
        ctx.output = outArr;
        ctx.inputSize = size2D(F_Q80, 1u, n);
        ctx.outputSize = size2D(F_32, 1u, n);
        ctx.weightSize = size0();

        castForward_Q80_F32(1u, 0u, 1u, &ctx);
        compare_F32("castForward_Q80_F32_view_slice", output.data(), expected.data(), n, 1e-6f);
    }
}

// matmul
void testMatmul_F32_Q40_F32(const NnUint m = 2) {
    const NnUint n = Q80_BLOCK_SIZE * m;
    const NnUint d = Q80_BLOCK_SIZE * m;

    std::vector<float> x(n);
    std::vector<float> w(n * d);
    std::vector<float> o(d);
    std::vector<float> oTemp(d);
    std::vector<NnBlockQ80> xQ80(n / Q80_BLOCK_SIZE);
    std::vector<NnBlockQ40> wQ40((n * d) / Q40_BLOCK_SIZE);

    rand(x.data(), n, m);
    rand(w.data(), n * d, m);
    quantizeF32toQ40(w.data(), wQ40.data(), n * d, 1, 0);
    quantizeF32toQ80(x.data(), xQ80.data(), n, 1, 0);

    matmul_F32_F32_F32(o.data(), x.data(), w.data(), n, d, 1, 0);

    matmul_Q80_Q40_F32(oTemp.data(), xQ80.data(), wQ40.data(), n, d, 1, 0);
    compare_F32("matmul_Q80_Q40_F32", o.data(), oTemp.data(), d, 4.0f);
}

void testLlamafileSgemm() {
    const NnUint batchSize = 8;
    const NnUint n = 256;
    const NnUint d = 128;

    std::vector<float> x(n * batchSize);
    std::vector<NnBlockQ80> xQ((n * batchSize) / Q80_BLOCK_SIZE);
    std::vector<float> w(n * d);
    std::vector<NnBlockQ40> wQ((n * d) / Q40_BLOCK_SIZE);
    std::vector<float> o(d * batchSize);
    std::vector<float> oTemp(d * batchSize);

    rand(x.data(), n * batchSize, 12345);
    rand(w.data(), n * d, 23456);

    quantizeF32toQ80(x.data(), xQ.data(), n * batchSize, 1, 0);
    quantizeF32toQ40(w.data(), wQ.data(), n * d, 1, 0);

    // f32

    for (NnUint i = 0; i < batchSize; i++) {
        matmul_F32_F32_F32(o.data() + i * d, x.data() + i * n, w.data(), n, d, 1, 0);
    }

    assert(llamafile_sgemm(
        d, batchSize, n,
        w.data(), n,
        x.data(), n,
        oTemp.data(), d,
        0, 1, 0,
        F_32, F_32, F_32
    ));

    compare_F32("llamafileSgemm_F32", o.data(), oTemp.data(), d * batchSize, 0.01f);

#if __ARM_FEATURE_DOTPROD
    // q40ᵀ * q80

    assert(llamafile_sgemm(
        d, batchSize, n / Q80_BLOCK_SIZE,
        wQ.data(), n / Q80_BLOCK_SIZE,
        xQ.data(), n / Q80_BLOCK_SIZE,
        oTemp.data(), d,
        0, 1, 0,
        F_Q40, F_Q80, F_32
    ));

    compare_F32("llamafileSgemm_Q80_Q40", o.data(), oTemp.data(), d * batchSize, 1.5f);
#endif
}

void testMatmulOpViewsZeroOffset() {
    // This test ensures that when physical buffers == logical buffers and views have offset=0,
    // the matmul op produces the same output as the reference kernels.

    // Case 1: F32 input + F32 weight -> F32 output
    {
        const NnUint nBatches = 3u;
        const NnUint n = 32u;
        const NnUint d = 64u;

        std::vector<float> x(nBatches * n);
        std::vector<float> w(n * d);
        std::vector<float> out(nBatches * d, 0.0f);
        std::vector<float> expected(nBatches * d, 0.0f);
        rand(x.data(), nBatches * n, 1001u);
        rand(w.data(), n * d, 1002u);

        for (NnUint b = 0; b < nBatches; b++) {
            matmul_F32_F32_F32(
                expected.data() + b * d,
                x.data() + b * n,
                w.data(),
                n,
                d,
                1u,
                0u);
        }

        NnMatmulOpConfig cfg{};
        cfg.nExperts = 0u;
        cfg.nActiveExperts = 0u;
        cfg.activeExpertIndexesBufferIndex = 0u;
        cfg.view = 0u;
        cfg.inStart = 0u;
        cfg.outStart = 0u;
        cfg.aView = NnTensorView{0u, 0u, n, 0u, 1u};
        cfg.cView = NnTensorView{0u, 0u, d, 0u, 1u};

        float dummyIdx = 0.0f;
        NnByte *buffers[1] = { (NnByte *)&dummyIdx };
        NnBufferConfig bufferCfgs[1] = {};
        bufferCfgs[0].name = (char *)"dummy";
        bufferCfgs[0].size = size2D(F_32, 1u, 1u);

        std::vector<NnByte *> inPtrs(nBatches);
        std::vector<NnByte *> outPtrs(nBatches);
        for (NnUint b = 0; b < nBatches; b++) {
            inPtrs[b] = (NnByte *)(x.data() + b * n);
            outPtrs[b] = (NnByte *)(out.data() + b * d);
        }

        NnCpuOpContext ctx{};
        ctx.name = "test_matmul_op_view_zero_f32";
        ctx.nBatches = nBatches;
        ctx.opConfig = &cfg;
        ctx.input = inPtrs.data();
        ctx.output = outPtrs.data();
        ctx.inputSize = size2D(F_32, nBatches, n);
        ctx.outputSize = size2D(F_32, nBatches, d);
        ctx.weight = (NnByte *)w.data();
        ctx.weightSize = size2D(F_32, n, d);
        ctx.buffers = buffers;
        ctx.bufferConfigs = bufferCfgs;
        ctx.hasInputContinuousMemory = true;
        ctx.hasOutputContinuousMemory = true;

        initMatmulForward(&ctx);
        matmulForward_F32_F32_F32(1u, 0u, nBatches, &ctx);

        compare_F32("matmulOp_view_zeroOffset_F32", out.data(), expected.data(), nBatches * d, 1e-6f);
    }

    // Case 2: Q80 input + Q40 weight -> F32 output
    {
        const NnUint nBatches = 2u;
        const NnUint n = Q80_BLOCK_SIZE * 2u;
        const NnUint d = Q80_BLOCK_SIZE * 1u;

        std::vector<float> xF32(nBatches * n);
        std::vector<float> wF32(n * d);
        rand(xF32.data(), nBatches * n, 2001u);
        rand(wF32.data(), n * d, 2002u);

        std::vector<NnBlockQ80> xQ((nBatches * n) / Q80_BLOCK_SIZE);
        quantizeF32toQ80(xF32.data(), xQ.data(), nBatches * n, 1u, 0u);

        std::vector<NnBlockQ40> wQ((n * d) / Q40_BLOCK_SIZE);
        quantizeF32toQ40(wF32.data(), wQ.data(), n * d, 1u, 0u);

        std::vector<float> expected(nBatches * d, 0.0f);
        for (NnUint b = 0; b < nBatches; b++) {
            matmul_Q80_Q40_F32(
                expected.data() + b * d,
                xQ.data() + b * (n / Q80_BLOCK_SIZE),
                wQ.data(),
                n,
                d,
                1u,
                0u);
        }

        std::vector<float> out(nBatches * d, 0.0f);

        NnMatmulOpConfig cfg{};
        cfg.nExperts = 0u;
        cfg.nActiveExperts = 0u;
        cfg.activeExpertIndexesBufferIndex = 0u;
        cfg.view = 0u;
        cfg.inStart = 0u;
        cfg.outStart = 0u;
        cfg.aView = NnTensorView{0u, 0u, n, 0u, 1u};
        cfg.cView = NnTensorView{0u, 0u, d, 0u, 1u};

        float dummyIdx = 0.0f;
        NnByte *buffers[1] = { (NnByte *)&dummyIdx };
        NnBufferConfig bufferCfgs[1] = {};
        bufferCfgs[0].name = (char *)"dummy";
        bufferCfgs[0].size = size2D(F_32, 1u, 1u);

        std::vector<NnByte *> inPtrs(nBatches);
        std::vector<NnByte *> outPtrs(nBatches);
        for (NnUint b = 0; b < nBatches; b++) {
            inPtrs[b] = (NnByte *)(xQ.data() + b * (n / Q80_BLOCK_SIZE));
            outPtrs[b] = (NnByte *)(out.data() + b * d);
        }

        NnCpuOpContext ctx{};
        ctx.name = "test_matmul_op_view_zero_q80q40";
        ctx.nBatches = nBatches;
        ctx.opConfig = &cfg;
        ctx.input = inPtrs.data();
        ctx.output = outPtrs.data();
        ctx.inputSize = size2D(F_Q80, nBatches, n);
        ctx.outputSize = size2D(F_32, nBatches, d);
        ctx.weight = (NnByte *)wQ.data();
        ctx.weightSize = size2D(F_Q40, n, d);
        ctx.buffers = buffers;
        ctx.bufferConfigs = bufferCfgs;
        ctx.hasInputContinuousMemory = true;
        ctx.hasOutputContinuousMemory = true;

        initMatmulForward(&ctx);
        matmulForward_Q80_Q40_F32(1u, 0u, nBatches, &ctx);

        compare_F32("matmulOp_view_zeroOffset_Q80Q40", out.data(), expected.data(), nBatches * d, 5.0f);
    }
}

void testScale() {
    float i[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float o[4];
    scale_F32(i, o, 0.5f, 4u, 1u, 0u);
    float expectedOutput[] = {0.5f, 1.0f, 1.5f, 2.0f};
    compare_F32("scale_F32", o, expectedOutput, 4u, 0.00001f);
}

void testTopk() {
    float x[] = {1.0f, 4.0f, 2.0f, 3.0f};
    std::vector<NnUint> topk(2);
    topk_F32(x, topk.data(), 4u, 2u);
    assert(topk[0] == 1u);
    assert(topk[1] == 3u);
    printPassed("testTopk");
}

int main() {
    initQuants();

    printCpuInstructionSet();
    testSplitThreads();
    testConvertF32toF16();
    testQuantization(32);
    testQuantization(2);
    testQuantization(1);
    testInvRms();
    testRmsNorm(128);
    testMul(32);
    testMul(2);
    testMul(1);
    testAdd(32);
    testAdd(2);
    testAdd(1);
    testMergeSum();
    testSoftmax();
    testSilu();
    testGeluViewSlice();
    testRopeViewSlice();
    testCastViewSlice();
    testMatmul_F32_Q40_F32(32);
    testMatmul_F32_Q40_F32(2);
    testMatmul_F32_Q40_F32(1);
    testMatmulOpViewsZeroOffset();
    testLlamafileSgemm();
    testScale();
    testTopk();
    return 0;
}
