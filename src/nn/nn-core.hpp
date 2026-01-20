#ifndef NN_CORE_H
#define NN_CORE_H

#include <chrono>
#include <list>
#include <memory>
#include <cstdint>
#include <cassert>
#include <vector>
#include <cstring> // for std::memset
#include <utility> // for std::move
#include <atomic>
#include "nn-quants.hpp"
#include "nn-core.hpp"

// ======================================================================================
// Primitives
// ======================================================================================

typedef struct {
    NnFloatType floatType;
    NnUint z;
    NnUint y;
    NnUint x;
    NnSize length;
    NnSize nBytes;
    NnSize nBytesXY;
} NnSize3D;

typedef struct {
    NnUint* starts;    // start positions
    NnUint* lengths;   // lengths
} NnDimSplit;

// ======================================================================================
// Pipeline Parallelism Configs
// ======================================================================================

// [新增] 用于 createPartitionPlan 的输入参数，描述一个 Stage 的需求
struct NnStageDef {
    NnUint nLayers;              // 该 Stage 负责多少层
    std::vector<float> tpRatios; // 该 Stage 内部的 TP 比例 (例如 {1.0, 3.0})
};

// [新增] 描述一个 Stage 的具体配置 (生成后的结果)
struct NnStageConfig {
    NnUint stageIndex;      // Stage ID (0, 1, ...)
    NnUint startLayer;      // 起始层 (全局索引)
    NnUint endLayer;        // 结束层 (全局索引, 不包含)
    NnUint nLayers;         // 层数
    
    // 拓扑信息
    NnUint rootNodeIndex;   // 该 Stage 的 Root 节点全局 ID (用于 Stage 间通信)
    NnUint nNodes;          // 该 Stage 内的节点数量
    NnUint *nodeIndices;    // 该 Stage 包含的全局节点 ID 列表
    
    // 构造/析构
    NnStageConfig() : nodeIndices(nullptr) {}
    ~NnStageConfig() { if (nodeIndices) delete[] nodeIndices; }
    
    // 移动构造
    NnStageConfig(NnStageConfig&& other) noexcept 
        : stageIndex(other.stageIndex), startLayer(other.startLayer), 
          endLayer(other.endLayer), nLayers(other.nLayers),
          rootNodeIndex(other.rootNodeIndex), nNodes(other.nNodes), 
          nodeIndices(other.nodeIndices) {
        other.nodeIndices = nullptr;
    }
    NnStageConfig(const NnStageConfig&) = delete;
};

// ======================================================================================
// Uneven Partition Plan (Memory Safe)
// ======================================================================================

struct NnUnevenPartitionPlan {
    NnUint nNodes;

    NnUint nStages;
    NnStageConfig *stages; // Stage 数组

    NnDimSplit headSplit;
    NnDimSplit kvHeadSplit;
    NnDimSplit vocabSplit;
    NnDimSplit ffnSplit;
    NnDimSplit dimSplit;

    // 默认构造函数
    NnUnevenPartitionPlan() : nNodes(0) {
        std::memset(&headSplit, 0, sizeof(headSplit));
        std::memset(&kvHeadSplit, 0, sizeof(kvHeadSplit));
        std::memset(&vocabSplit, 0, sizeof(vocabSplit));
        std::memset(&ffnSplit, 0, sizeof(ffnSplit));
        std::memset(&dimSplit, 0, sizeof(dimSplit));
    }

    // 析构函数：释放内部数组
    ~NnUnevenPartitionPlan() {
        freeSplit(headSplit);
        freeSplit(kvHeadSplit);
        freeSplit(vocabSplit);
        freeSplit(ffnSplit);
        freeSplit(dimSplit);
    }

    // 移动构造函数：用于 std::move 和 unique_ptr
    NnUnevenPartitionPlan(NnUnevenPartitionPlan&& other) noexcept 
        : nNodes(other.nNodes),
          headSplit(other.headSplit),
          kvHeadSplit(other.kvHeadSplit),
          vocabSplit(other.vocabSplit),
          ffnSplit(other.ffnSplit),
          dimSplit(other.dimSplit) {
        
        // 剥夺源对象的所有权
        other.stages = nullptr; // 接管 stages
        zeroSplit(other.headSplit);
        zeroSplit(other.kvHeadSplit);
        zeroSplit(other.vocabSplit);
        zeroSplit(other.ffnSplit);
        zeroSplit(other.dimSplit);
        other.nNodes = 0;
    }

    // 禁用拷贝 (防止 Double Free)
    NnUnevenPartitionPlan(const NnUnevenPartitionPlan&) = delete;
    NnUnevenPartitionPlan& operator=(const NnUnevenPartitionPlan&) = delete;

private:
    void freeSplit(NnDimSplit& split) {
        if (split.starts) { delete[] split.starts; split.starts = nullptr; }
        if (split.lengths) { delete[] split.lengths; split.lengths = nullptr; }
    }
    void zeroSplit(NnDimSplit& split) {
        split.starts = nullptr;
        split.lengths = nullptr;
    }
};

// ======================================================================================
// Runtime Layer-Level Sharding (Dynamic)
// ======================================================================================

enum NnSplitKind : NnUint {
    SPLIT_HEAD = 0,
    SPLIT_KV_HEAD = 1,
    SPLIT_VOCAB = 2,
    SPLIT_FFN = 3,
    SPLIT_DIM = 4,
    N_SPLIT_KINDS = 5,
};

struct NnLayerSplits {
    NnDimSplit headSplit;
    NnDimSplit kvHeadSplit;
    NnDimSplit vocabSplit;
    NnDimSplit ffnSplit;
    NnDimSplit dimSplit;

    // Optional: per-unit owner mapping for KV migration.
    // headOwners size = nHeadUnits (heads); kvHeadOwners size = nKvHeadUnits (kv heads)
    NnUint nHeadUnits = 0;
    NnUint nKvHeadUnits = 0;
    NnUint *headOwners = nullptr;
    NnUint *kvHeadOwners = nullptr;

    NnLayerSplits() {
        std::memset(&headSplit, 0, sizeof(headSplit));
        std::memset(&kvHeadSplit, 0, sizeof(kvHeadSplit));
        std::memset(&vocabSplit, 0, sizeof(vocabSplit));
        std::memset(&ffnSplit, 0, sizeof(ffnSplit));
        std::memset(&dimSplit, 0, sizeof(dimSplit));
        nHeadUnits = 0;
        nKvHeadUnits = 0;
        headOwners = nullptr;
        kvHeadOwners = nullptr;
    }
};

struct NnLayerShardingTable {
    NnUint nLayers = 0;
    NnUint nNodes = 0;
    std::atomic_uint epoch{0u};
    NnLayerSplits *layers = nullptr; // [nLayers]

    NnLayerShardingTable() = default;

    NnLayerShardingTable(NnUint nLayers, NnUint nNodes) {
        init(nLayers, nNodes);
    }

    ~NnLayerShardingTable() {
        reset();
    }

    void init(NnUint nLayers_, NnUint nNodes_) {
        reset();
        nLayers = nLayers_;
        nNodes = nNodes_;
        epoch.store(0u);
        layers = new NnLayerSplits[nLayers];
    }

    void reset() {
        if (layers != nullptr) {
            for (NnUint i = 0; i < nLayers; ++i) {
                freeSplit(layers[i].headSplit);
                freeSplit(layers[i].kvHeadSplit);
                freeSplit(layers[i].vocabSplit);
                freeSplit(layers[i].ffnSplit);
                freeSplit(layers[i].dimSplit);
                freeOwners(layers[i]);
            }
            delete[] layers;
            layers = nullptr;
        }
        nLayers = 0;
        nNodes = 0;
        epoch.store(0u);
    }

    void setLayerSplit(NnUint layerIndex, NnSplitKind kind, const NnUint *starts, const NnUint *lengths) {
        if (layers == nullptr) return;
        assert(layerIndex < nLayers);
        assert(starts != nullptr);
        assert(lengths != nullptr);

        NnDimSplit *dst = nullptr;
        switch (kind) {
            case SPLIT_HEAD: dst = &layers[layerIndex].headSplit; break;
            case SPLIT_KV_HEAD: dst = &layers[layerIndex].kvHeadSplit; break;
            case SPLIT_VOCAB: dst = &layers[layerIndex].vocabSplit; break;
            case SPLIT_FFN: dst = &layers[layerIndex].ffnSplit; break;
            case SPLIT_DIM: dst = &layers[layerIndex].dimSplit; break;
            default: return;
        }

        ensureAllocated(*dst);
        std::memcpy(dst->starts, starts, nNodes * sizeof(NnUint));
        std::memcpy(dst->lengths, lengths, nNodes * sizeof(NnUint));
    }

private:
    void freeSplit(NnDimSplit &split) {
        if (split.starts) { delete[] split.starts; split.starts = nullptr; }
        if (split.lengths) { delete[] split.lengths; split.lengths = nullptr; }
    }

    void freeOwners(NnLayerSplits &splits) {
        if (splits.headOwners) {
            delete[] splits.headOwners;
            splits.headOwners = nullptr;
        }
        if (splits.kvHeadOwners) {
            delete[] splits.kvHeadOwners;
            splits.kvHeadOwners = nullptr;
        }
        splits.nHeadUnits = 0;
        splits.nKvHeadUnits = 0;
    }

    void ensureAllocated(NnDimSplit &split) {
        if (split.starts == nullptr) split.starts = new NnUint[nNodes];
        if (split.lengths == nullptr) split.lengths = new NnUint[nNodes];
    }
};

// Packed sharding update message (typically broadcast within a stage).
// Payload layout: for each split kind in order (HEAD, KV_HEAD, VOCAB, FFN, DIM)
//   starts[nNodes] then lengths[nNodes]
typedef struct {
    NnUint magic;      // 'DLSH'
    NnUint version;    // schema version
    NnUint epoch;      // monotonically increasing
    NnUint layerIndex; // global layer index
    NnUint nNodes;     // number of nodes in split arrays
    NnUint flags;      // reserved
    // Apply range: update the splits for layers in [applyStartLayer, applyEndLayerExclusive).
    // This enables "one change affects the rest of the stage" semantics.
    NnUint applyStartLayer;
    NnUint applyEndLayerExclusive;
} NnShardingUpdateHeader;

static constexpr NnUint NN_SHARDING_UPDATE_MAGIC = 0x48534c44u; // 'DLSH' little-endian
static constexpr NnUint NN_SHARDING_UPDATE_VERSION = 2u;

// NnShardingUpdateHeader::flags
static constexpr NnUint NN_SHARDING_UPDATE_FLAG_DEBUG_MATMUL = 1u << 0;

// ======================================================================================
// Slices (Original & Uneven)
// ======================================================================================

// --- Original Slices ---

typedef struct {
    NnUint kvDim0;
    NnSize3D keySize;
    NnSize3D valueSize;
} NnKvCacheSlice;

typedef struct {
    NnFloatType type;
    NnUint nNodes;
    NnUint d0;
    NnUint n;
    NnSize3D size;
    NnSize3D sliceSize;
} NnRowMatmulSlice;

typedef struct {
    NnFloatType type;
    NnUint nNodes;
    NnUint n;
    NnUint n0;
    NnUint d;
    NnSize3D size;
    NnSize3D sliceSize;
} NnColMatmulSlice;

typedef struct {
    NnUint qDim0;
    NnUint qDimStart;
    NnUint qDimEnd;
    NnUint qShift;
    NnUint kvDim;
    NnUint kvDim0;
    NnUint kvDimStart;
    NnUint sliceDim;
    NnUint seqLen;
    NnUint headDim;
    NnUint nKvHeads;
    float ropeTheta;
    NnSize3D cacheSize;
} NnRopeSlice;

typedef struct {
    NnUint nHeads;
    NnUint nHeads0;
    NnSize3D attSize;
} NnMultiHeadAttSlice;

// --- Uneven Slices ---

typedef struct {
    NnUint kvStart;   // 本节点 KV 起点
    NnUint kvLen;     // 本节点 KV 长度
    NnUint kvDim0;    // 兼容字段
    NnSize3D keySize;
    NnSize3D valueSize;
} NnKvCacheSliceUneven;

typedef struct {
    NnFloatType type;
    NnUint inStart;   // 输入维度起点 (Global Row Start)
    NnUint inLen;     // 输入维度长度 (Global Rows Count)
    NnUint d0;        // 局部输出列 (Local Output Cols)
    NnUint n;         // 全局输入维度 (Global Input Dim)
    NnSize3D size;
    NnSize3D sliceSize;
} NnRowMatmulSliceUneven; 

typedef struct {
    NnFloatType type;
    NnUint outStart;  // 输出维度起点 (Global Col Start)
    NnUint outLen;    // 输出维度长度 (Local Input Cols)
    NnUint n;         // 全局输入维度
    NnUint n0;        // 局部输入维度
    NnUint d;         // 全局输出维度
    NnSize3D size;        
    NnSize3D sliceSize;
} NnColMatmulSliceUneven;

typedef struct {
    NnUint qDim0;
    NnUint qDimStart;
    NnUint qDimLen;   // 本节点 Q 长度
    NnUint qShift;

    NnUint kvDim;
    NnUint kvDim0;
    NnUint kvDimStart;
    NnUint kvDimLen;  // 本节点 KV 长度

    NnUint sliceDim;
    NnUint seqLen;
    NnUint headDim;
    NnUint nKvHeads;
    float  ropeTheta;
    NnSize3D cacheSize;
} NnRopeSliceUneven;

typedef struct {
    NnUint headStart; // 本节点 Head 起点
    NnUint headLen;   // 本节点 Head 数量
    NnUint nHeads;
    NnUint nHeads0;
    NnSize3D attSize;
} NnMultiHeadAttSliceUneven;

// ======================================================================================
// Base Enums
// ======================================================================================

enum NnOpCode {
    OP_MERGE_ADD,
    OP_MERGE_SUM,
    OP_EMBEDDING,
    OP_INV_RMS,
    OP_RMS_NORM,
    OP_MATMUL,
    OP_ROPE,
    OP_MULTIHEAD_ATT,
    OP_GELU,
    OP_SILU,
    OP_MUL,
    OP_SCALE,
    OP_CAST,
    OP_REPEAT_Z,
    OP_SHIFT,
    OP_SOFTMAX,
    OP_MOE_GATE,
    OP_PP_RECV,
    OP_PP_SEND,
    OP_WRITE_U32,
    OP_UPDATE_SHARDING,
};

enum NnOpQuantType {
    // <input>_<weight>_<output>
    F32_F32_F32,
    F32_Q40_F32,
    F32_Q40_Q80,
    F32_F32_Q80,
    Q80_Q80_Q80,
    Q80_Q80_F32,
    Q80_Q40_F32,
    Q80_F32_F32,
};

#define N_OP_CODES (OP_UPDATE_SHARDING + 1)
#define N_OP_QUANTS (Q80_F32_F32 + 1)

enum NnPointerSource {
    SRC_PIPE,
    SRC_BUFFER,
};

enum NnPointerType {
    PNTR_RAW,
    PNTR_BATCH,
    PNTR_BATCHED_SLICE
};

enum NnSyncType {
    SYNC_WITH_ROOT, // whole pipe to all nodes
    SYNC_NODE_SLICES, // my slice of pipe to all nodes
    SYNC_NODE_SLICES_EXCEPT_ROOT, // only workers send slices to root, root does not send
    SYNC_KV_ALLGATHER, // stage-local all-gather for KV cache slices
    SYNC_PP_SEND,                     // PP: 当前 Stage 发送给 Next Stage
    SYNC_PP_RECV,                      // PP: 当前 Stage 从 Prev Stage 接收

    // Control-plane: stage root may broadcast a packed sharding update message.
    // If root decides "no update" for this layer, it broadcasts a header with magic=0.
    SYNC_STAGE_SHARDING_UPDATE,
};

enum NnRopeType {
    ROPE_LLAMA = 0,
    ROPE_FALCON = 1,
    ROPE_LLAMA3_1 = 2,
};

// ======================================================================================
// Base Configs
// ======================================================================================

typedef struct {
    char *name;
    NnSize3D size;
} NnPipeConfig;

typedef struct {
    char *name;
    NnSize3D size;
} NnBufferConfig;

typedef struct {
    NnPointerSource source;
    NnUint pointerIndex;
    NnPointerType type;
} NnPointerConfig;

typedef struct {
    NnOpCode code;
    char *name;
    NnUint index;
    NnPointerConfig input;
    NnPointerConfig output;
    NnSize3D weightSize;
    // Optional: allocate more bytes than weightSize.nBytes (e.g., stage-local weight replication).
    // When 0, runtime should treat it as weightSize.nBytes.
    NnSize weightAllocBytes;
    NnByte *config;
    NnUint configSize;
} NnOpConfig;

typedef struct {
    NnUint pipeIndex;
} NnPreSyncConfig;

typedef struct {
    NnUint pipeIndex;
    NnSyncType syncType;
} NnSyncConfig;

typedef struct  {
    NnUint nOps;
    NnOpConfig *ops;
    NnUint nSyncs;
    NnSyncConfig *syncs;
} NnSegmentConfig;

typedef struct {
    NnUint nBatches;
    NnUint nNodes;
    NnUint nPipes;
    NnPipeConfig *pipes;
    NnUint nPreSyncs;
    NnPreSyncConfig *preSyncs;
} NnNetConfig;

typedef struct {
    NnUint nodeIndex;
    NnUint nBuffers;
    NnBufferConfig *buffers;
    NnUint nSegments;
    NnSegmentConfig *segments;
    const struct NnUnevenPartitionPlan *partitionPlan = nullptr;
} NnNodeConfig;

// ======================================================================================
// Op Configs
// ======================================================================================

typedef struct {
    // empty
} NnEmbeddingOpConfig;

typedef struct {
    NnUint value;
} NnWriteU32OpConfig;

typedef struct {
    float epsilon;
    NnUint nColumns;
} NnInvRmsOpConfig;

typedef struct {
    NnUint invRmsBufferIndex;
    NnUint nColumns;
} NnRmsNormOpConfig;

typedef struct {
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint activeExpertIndexesBufferIndex;

    // Optional: stage-local weight replication.
    // When enabled, the weight buffer packs all stage-node slices for each expert.
    // Compute uses only the current node's slice at (expertStrideBytes * activeExpert + myOffsetBytes).
    NnUint replicateMode; // 0/1
    NnUint _pad0;
    NnSize replicateExpertStrideBytes;
    NnSize replicateMyOffsetBytes;

    // Runtime sharding (optional): when replicateMode!=0, compute can select
    // which shard to execute based on NnLayerShardingTable for this layer.
    // These fields are all zero by default and are safe to leave unset.
    NnUint layerIndex;          // global layer index for sharding lookup
    NnUint splitKind;           // NnSplitKind, or >=N_SPLIT_KINDS to disable
    NnUint splitAxis;           // 0=none, 1=OUT_ROWS, 2=IN_COLS
    NnUint splitUnit;           // multiplier from split-units -> elements (e.g. headDim)
    NnUint staticStartUnits;    // graph-time start (in split units)
    NnUint staticLenUnits;      // graph-time len (in split units)
    NnUint replicateGlobalInDim; // for IN_COLS mode: global input dim (elements) of full weight
} NnMatmulOpConfig;

typedef struct {
    NnRopeType type;
    NnUint isQ; // Cannot use `bool` here due to GPU memory alignment
    NnUint positionPipeIndex;
    NnUint ropeCacheBufferIndex;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactor;
    NnUint ropeScalingOrigMaxSeqLen;
    NnRopeSlice slice;

    // Optional: runtime sharding length override for rope (treat input as a sharded vector).
    // If splitKind is invalid or layerIndex==0, uses staticLenUnits (or full x when staticLenUnits==0).
    NnUint layerIndex;
    NnUint splitKind;
    NnUint splitUnit;
    NnUint staticLenUnits;
    NnUint staticStartUnits;
} NnRopeOpConfig;

typedef struct {
    // Identity for runtime sharding lookup/debug.
    NnUint nodeIndex;
    NnUint layerIndex;
    NnUint headStart; // static head start (graph-time)
    NnUint nHeads;
    NnUint nHeads0;
    NnUint nKvHeads;
    NnUint headDim;
    NnUint seqLen;
    NnUint qSliceD0;
    NnUint kvDim0;
    NnUint positionPipeIndex;
    NnUint queryBufferIndex;
    NnUint keyCacheBufferIndex;
    NnUint valueCacheBufferIndex;
    NnUint attBufferIndex;

    // Optional: graph-time planned head len (in heads) for fallback when runtime sharding is disabled.
    // If 0, defaults to nHeads0.
    NnUint staticHeadLenUnits;
    NnUint kvHeadStart;
} NnMultiHeadAttOpConfig;

typedef struct {
    // empty
} NnMergeAddOpCodeConfig;

typedef struct {
    // empty
} NnMergeSumOpCodeConfig;

typedef struct {
    // Optional: runtime sharding length override for elementwise ops.
    // If splitKind is invalid or layerIndex==0, uses staticLenUnits (or full x when staticLenUnits==0).
    NnUint layerIndex;
    NnUint splitKind;
    NnUint splitUnit;
    NnUint staticLenUnits;
} NnSiluOpCodeConfig;

typedef struct {
    NnUint multiplierBufferIndex;
    NnUint layerIndex;
    NnUint splitKind;
    NnUint splitUnit;
    NnUint staticLenUnits;
} NnMulOpCodeConfig;

typedef struct {
    NnUint scaleBufferIndex;
} NnScaleOpCodeConfig;

typedef struct {
    // Optional: runtime sharding length override for casts operating on sharded vectors.
    NnUint layerIndex;
    NnUint splitKind;
    NnUint splitUnit;
    NnUint staticLenUnits;
} NnCastOpCodeConfig;

typedef struct {
    // empty
} NnRepeatZOpCodeConfig;

typedef struct {
    NnUint indexPipeIndex;

    // Optional: runtime sharding length override for shift (treat input as a sharded vector).
    // If splitKind is invalid or layerIndex==0, uses staticLenUnits (or full x when staticLenUnits==0).
    NnUint layerIndex;
    NnUint splitKind;
    NnUint splitUnit;
    NnUint staticLenUnits;
} NnShiftOpCodeConfig;

typedef struct {
    // empty (reads packed update from input pipe; applies to runtime sharding table)
} NnUpdateShardingOpConfig;

typedef struct {
    // empty
} NnSoftmaxOpCodeConfig;

typedef struct {
    NnUint k;
    NnUint normTopk;
    NnUint indexesBufferIndex;
} NnMoeGateOpCodeConfig;

// ======================================================================================
// Functions Declarations
// ======================================================================================

const char *opCodeToString(NnOpCode code);
const char *opQuantTypeToString(NnOpQuantType type);

NnSize getBytes(NnFloatType floatType, NnSize n);
NnSize getBlockSize(NnFloatType floatType);
NnOpQuantType getOpQuantType(NnFloatType input, NnFloatType weight, NnFloatType output);
NnSize3D size0();
NnSize3D size1D(NnFloatType floatType, NnUint x);
NnSize3D size2D(NnFloatType floatType, NnUint y, NnUint x);
NnSize3D size3D(NnFloatType floatType, NnUint z, NnUint y, NnUint x);
NnPointerConfig pointerBatchConfig(NnPointerSource source, NnUint index);
NnPointerConfig pointerBatchedSliceConfig(NnPointerSource source, NnUint index);
NnPointerConfig pointerRawConfig(NnPointerSource source, NnUint index);
bool hasPointerContinuousMemory(NnPointerConfig *config);

void releaseNetConfig(NnNetConfig *netConfig);
void releaseNodeConfig(NnNodeConfig *nodeConfig);

void printNodeRequiredMemory(NnNetConfig *netConfig, NnNodeConfig *nodeConfig);

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
public:
    Timer();
    void reset();
    NnUint elapsedMiliseconds();
    NnUint elapsedMicroseconds();
};

// --- Legacy Slicers ---

NnKvCacheSlice sliceKvCache(NnUint kvDim, NnUint seqLen, NnUint nNodes);
NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);
NnColMatmulSlice sliceColMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);
NnRopeSlice sliceRope(NnRopeType type, NnUint qDim, NnUint kvDim, NnUint nKvHeads, NnUint nNodes, NnUint seqLen, NnUint headDim, float ropeTheta, NnUint nodeIndex);
NnMultiHeadAttSlice sliceMultiHeadAtt(NnUint nHeads, NnUint seqLen, NnUint nNodes, NnUint nBatches);

// --- Legacy Splitters ---

NnUint splitRowMatmulWeight(NnRowMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0);
NnUint splitColMatmulWeight(NnColMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0);

// --- Rope Util ---

void fullfillRopeCache(const NnRopeOpConfig *config, float *cache);

// ======================================================================================
// Uneven Partition Functions
// ======================================================================================

// 创建计划 (包含 GQA 对齐修复)
NnUnevenPartitionPlan createPartitionPlan(
    const std::vector<NnStageDef>& stageDefs,
    NnUint globalNHeads,
    NnUint globalNKvHeads,
    NnUint globalVocabSize,
    NnUint globalFfnDim,
    NnUint globalDim
);

// 释放计划 (旧接口，如果使用栈上对象+析构函数可忽略，但保留以防遗留调用)
void releasePartitionPlan(NnUnevenPartitionPlan* plan);

// Slicers

NnKvCacheSliceUneven sliceKvCacheUneven(NnUint seqLen, NnUint headDim,
    const NnUnevenPartitionPlan* plan, NnUint nodeIndex);

NnMultiHeadAttSliceUneven sliceMultiHeadAttUneven(NnUint nBatches, NnUint globalNHeads, NnUint globalSeqLen,
    const NnUnevenPartitionPlan* plan, NnUint nodeIndex);

NnRowMatmulSliceUneven sliceRowMatmulAttUneven(NnFloatType type, NnUint globalInDim, NnUint headDim,
    const NnDimSplit* headSplit, NnUint globalOutDim, NnUint nodeIndex);

NnColMatmulSliceUneven sliceColMatmulAttUneven(NnFloatType type, NnUint globalInDimQ, NnUint globalOutDim, NnUint headDim,
    const NnUnevenPartitionPlan* plan, NnUint nodeIndex);

NnRowMatmulSliceUneven sliceRowMatmulFfnUneven(NnFloatType type, NnUint globalInDim, NnUint globalFfnDim,
    const NnUnevenPartitionPlan* plan, NnUint nodeIndex);

NnColMatmulSliceUneven sliceColMatmulFfnUneven(NnFloatType type, NnUint globalFfnDim, NnUint globalOutDim,
    const NnUnevenPartitionPlan* plan, NnUint nodeIndex);

NnRopeSliceUneven sliceRopeUneven(NnRopeType type, NnUint seqLen, 
    NnUint globalKvDim, NnUint globalNKvHeads, NnUint headDim, float ropeTheta,
    const NnUnevenPartitionPlan* plan, NnUint nodeIndex);

NnRowMatmulSliceUneven sliceRowMatmulLogitsUneven(NnFloatType type, NnUint globalInDim, NnUint globalVocabSize,
    const NnUnevenPartitionPlan* plan, NnUint nodeIndex);

// Splitters

NnUint splitRowMatmulWeightUneven(NnRowMatmulSliceUneven *slice, NnByte *weight, NnByte *weight0);
NnUint splitColMatmulWeightUneven(NnColMatmulSliceUneven *slice, NnByte *weight, NnByte *weight0);

#endif // NN_CORE_H