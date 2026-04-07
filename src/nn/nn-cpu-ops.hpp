#ifndef NN_CPU_OPS_H
#define NN_CPU_OPS_H

#include "nn-core.hpp"
#include <cstdint>

#define ASSERT_EQ(a, b) \
    if (a != b) { \
        printf("Assertion failed: %d != %d (%s:%d)\n", a, b, __FILE__, __LINE__); \
        exit(-1); \
    }

typedef struct {
    const char *name;
    NnOpCode opCode;
    NnUint opIndex;
    // Which node/device this op context belongs to.
    // Filled by NnCpuDevice::createSegment().
    NnUint nodeIndex;
    NnByte nBatches;
    NnByte *bufferFlags;
    NnByte **buffers;
    NnBufferConfig *bufferConfigs;
    NnByte **pipes;
    NnPipeConfig *pipeConfigs;
    NnUint nPipes;
    const NnUnevenPartitionPlan *partitionPlan;
    void *opConfig;

    NnByte **input;
    NnSize3D inputSize;
    bool hasInputContinuousMemory;

    NnByte **output;
    NnSize3D outputSize;
    bool hasOutputContinuousMemory;

    NnByte *weight;
    NnSize3D weightSize;

    // Debug: track loaded weight ranges (offsets in bytes within `weight`).
    NnSize weightLoadedMin;     // inclusive
    NnSize weightLoadedMax;     // exclusive
    NnSize weightLoadedBytes;   // sum of loadWeight(nBytes)
    NnUint weightLoadCalls;
    NnByte weightReadPrinted;
} NnCpuOpContext;

typedef void (*NnCpuOpForwardInit)(NnCpuOpContext *context);
typedef void (*NnCpuOpForward)(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context);

void printCpuInstructionSet();
NnCpuOpForwardInit getCpuOpForwardInit(NnOpCode code, NnOpQuantType quantType);
NnCpuOpForward getCpuOpForward(NnOpCode code, NnOpQuantType quantType);

void softmax_F32(float *x, const NnUint size);

#endif
