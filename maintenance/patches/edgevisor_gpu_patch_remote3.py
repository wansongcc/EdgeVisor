from pathlib import Path

root = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor")
p = root / "src/nn/nn-vulkan.cpp"
s = p.read_text()


def replace_once(old: str, new: str) -> None:
    global s
    if new in s:
        return
    if old not in s:
        raise RuntimeError(f"pattern not found: {old[:120]!r}")
    s = s.replace(old, new, 1)


def insert_after_once(needle: str, text: str) -> None:
    global s
    if text.strip() in s:
        return
    if needle not in s:
        raise RuntimeError(f"needle not found: {needle[:120]!r}")
    s = s.replace(needle, needle + text, 1)


replace_once(
    '#include "nn-vulkan.hpp"\n',
    '#include "nn-vulkan.hpp"\n#include "plan-command.hpp"\n#include "llm.hpp"\n#include <algorithm>\n#include <cstdio>\n#include <cstring>\n#include <vector>\n',
)

insert_after_once(
    """static NnByte *findFirstOpConfig(NnNodeConfig *nodeConfig, NnOpCode opCode) {
    for (NnUint i = 0; i < nodeConfig->nSegments; i++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[i];
        for (NnUint j = 0; j < segmentConfig->nOps; j++) {
            if (segmentConfig->ops[j].code == opCode)
                return segmentConfig->ops[j].config;
        }
    }
    return nullptr;
}
""",
    r'''

static bool isVulkanControlOp(NnOpCode code) {
    return code == OP_PLAN_BARRIER || code == OP_PLAN_APPLY;
}

static NnUint getSplitTotalVulkan(const NnDimSplit *split, NnUint nNodes) {
    if (split == nullptr || split->lengths == nullptr) return 0u;
    NnUint total = 0u;
    for (NnUint i = 0; i < nNodes; ++i) total += split->lengths[i];
    return total;
}

static const NnStageConfig *findStageForNodeVulkan(const NnUnevenPartitionPlan *plan, NnUint nodeIndex) {
    if (plan == nullptr || plan->stages == nullptr) return nullptr;
    for (NnUint s = 0; s < plan->nStages; ++s) {
        const NnStageConfig *st = &plan->stages[s];
        for (NnUint i = 0; i < st->nNodes; ++i) {
            if (st->nodeIndices[i] == nodeIndex) return st;
        }
    }
    return nullptr;
}

static NnUint findStageRankVulkan(const NnStageConfig *stage, NnUint nodeIndex) {
    if (stage == nullptr) return nodeIndex;
    for (NnUint i = 0; i < stage->nNodes; ++i) {
        if (stage->nodeIndices[i] == nodeIndex) return i;
    }
    return nodeIndex;
}

static NnUint getSplitTotalForStageVulkan(const NnDimSplit *split, const NnStageConfig *stage) {
    if (split == nullptr || split->lengths == nullptr || stage == nullptr) return 0u;
    NnUint total = 0u;
    for (NnUint i = 0; i < stage->nNodes; ++i) total += split->lengths[stage->nodeIndices[i]];
    return total;
}

static bool resolveUnevenSliceVulkan(
    NnNetConfig *netConfig,
    NnNodeConfig *nodeConfig,
    NnPointerConfig *pointerConfig,
    const NnSize3D &sourceSize,
    NnUint *outOffset,
    NnUint *outLength
) {
    if (outOffset == nullptr || outLength == nullptr) return false;
    if (pointerConfig->type != PNTR_BATCHED_SLICE) return false;

    const NnUnevenPartitionPlan *plan = (nodeConfig != nullptr) ? nodeConfig->partitionPlan : nullptr;
    const NnUint nodeIdx = (nodeConfig != nullptr) ? nodeConfig->nodeIndex : 0u;
    bool stackedByNode = (pointerConfig->sliceTag == NN_SLICE_STACKED_BY_NODE);
    bool splitFound = false;
    NnUint myOffset = 0u;
    NnUint myLength = 0u;

    const NnStageConfig *myStage = (plan != nullptr && plan->nStages > 0u)
        ? findStageForNodeVulkan(plan, nodeIdx)
        : nullptr;

    if (plan != nullptr && netConfig != nullptr && netConfig->nNodes == plan->nNodes) {
        const NnUint totalDim = sourceSize.x;

        if (!stackedByNode && pointerConfig->source == SRC_PIPE && myStage != nullptr) {
            const NnUint dimTotal = getSplitTotalForStageVulkan(&plan->dimSplit, myStage);
            if (dimTotal > 0u && totalDim == dimTotal * netConfig->nNodes) {
                stackedByNode = true;
            }
        }

        auto tryApplySplit = [&](const NnDimSplit &split, bool allowMultiplier) -> bool {
            if (stackedByNode || split.starts == nullptr || split.lengths == nullptr) return false;
            const NnUint splitTotal = (myStage != nullptr)
                ? getSplitTotalForStageVulkan(&split, myStage)
                : getSplitTotalVulkan(&split, plan->nNodes);
            if (splitTotal == 0u || totalDim % splitTotal != 0u) return false;
            const NnUint multiplier = totalDim / splitTotal;
            if (!allowMultiplier && multiplier != 1u) return false;
            myOffset = split.starts[nodeIdx] * multiplier;
            myLength = split.lengths[nodeIdx] * multiplier;
            if (sourceSize.floatType == F_Q80) {
                if ((myOffset % Q80_BLOCK_SIZE) != 0u || (myLength % Q80_BLOCK_SIZE) != 0u) {
                    return false;
                }
            }
            return true;
        };

        if (pointerConfig->sliceTag != NN_SLICE_AUTO) {
            switch (pointerConfig->sliceTag) {
                case NN_SLICE_VOCAB:
                    splitFound = tryApplySplit(plan->vocabSplit, false);
                    break;
                case NN_SLICE_FFN:
                    splitFound = tryApplySplit(plan->ffnSplit, false);
                    break;
                case NN_SLICE_DIM:
                    splitFound = tryApplySplit(plan->dimSplit, false);
                    break;
                case NN_SLICE_HEAD:
                    splitFound = tryApplySplit(plan->headSplit, true);
                    break;
                case NN_SLICE_KV_HEAD:
                    splitFound = tryApplySplit(plan->kvHeadSplit, true);
                    break;
                case NN_SLICE_STACKED_BY_NODE:
                case NN_SLICE_AUTO:
                default:
                    break;
            }
        } else {
            splitFound = tryApplySplit(plan->vocabSplit, false);
            if (!splitFound) splitFound = tryApplySplit(plan->ffnSplit, false);
            if (!splitFound) splitFound = tryApplySplit(plan->dimSplit, false);
            if (!splitFound && sourceSize.floatType != F_Q80) splitFound = tryApplySplit(plan->headSplit, true);
            if (!splitFound && sourceSize.floatType != F_Q80) splitFound = tryApplySplit(plan->kvHeadSplit, true);
        }
    }

    if (!splitFound) {
        if (stackedByNode && netConfig != nullptr && netConfig->nNodes != 0u) {
            myLength = sourceSize.x / netConfig->nNodes;
            myOffset = myLength * nodeIdx;
        } else {
            const NnUint nSplitNodes = (myStage != nullptr) ? myStage->nNodes : (netConfig ? netConfig->nNodes : 1u);
            const NnUint rank = (myStage != nullptr) ? findStageRankVulkan(myStage, nodeIdx) : nodeIdx;
            myLength = (nSplitNodes != 0u) ? sourceSize.x / nSplitNodes : sourceSize.x;
            myOffset = myLength * rank;
        }
    }

    if (myOffset > sourceSize.x) {
        myOffset = 0u;
        myLength = 0u;
    }
    *outOffset = myOffset;
    *outLength = myLength;
    return true;
}

static bool handleVulkanPlanBarrier(
    NnVulkanDevice *device,
    NnNetExecution *netExecution,
    NnSegmentConfig *segmentConfig,
    NnUint opIndex,
    NnUint batchSize
) {
    if (device == nullptr || netExecution == nullptr || segmentConfig == nullptr) return true;
    NnOpConfig *op = &segmentConfig->ops[opIndex];
    if (op->input.source != SRC_PIPE || op->output.source != SRC_PIPE) return true;

    const float *posPipe = (const float *)netExecution->pipes[op->input.pointerIndex];
    float *planPipe = (float *)netExecution->pipes[op->output.pointerIndex];
    const NnUint layerIndex = op->index;
    const NnUint pos = (posPipe != nullptr && posPipe[0] >= 0.0f) ? (NnUint)posPipe[0] : 0u;

    const NnUnevenPartitionPlan *planConst = device->getPartitionPlan();
    const NnUint myNode = device->getNodeIndex();
    const NnStageConfig *myStage = (planConst != nullptr && planConst->nStages > 0u)
        ? findStageForNodeVulkan(planConst, myNode)
        : nullptr;

    const PlanCommandSnapshot snap = planCommandCache().load();
    const PlanCommand &pc = snap.cmd;
    const bool hasPlanPayload =
        (pc.cmdKind == PLAN_CMD_KIND_HEAD ||
         pc.cmdKind == PLAN_CMD_KIND_FFN ||
         pc.cmdKind == PLAN_CMD_KIND_BOTH ||
         (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves != 0u));
    const bool hasCmd =
        (pc.magic == DLLAMA_PLAN_CMD_MAGIC) &&
        (pc.version == DLLAMA_PLAN_CMD_VERSION_V1 || pc.version == DLLAMA_PLAN_CMD_VERSION_V2) &&
        (pc.mode == PLAN_CMD_MODE_EXACT || pc.mode == PLAN_CMD_MODE_NEXT_BARRIER) &&
        hasPlanPayload;

    const bool stageOk = (pc.stageIndex == 0xFFFFFFFFu) || (myStage != nullptr && myStage->stageIndex == pc.stageIndex);
    const bool isStageRoot = stageOk && (myStage != nullptr) && (myStage->rootNodeIndex == myNode);
    const unsigned int curEpoch = device->getPlanEpoch();
    unsigned int emitEpoch = curEpoch;
    unsigned int cmd = 0u;
    unsigned int headMove = 0u;
    unsigned int ffnMove = 0u;
    unsigned int fromNode = 0u;
    unsigned int toNode = 0u;

    bool trigger = false;
    if (hasCmd && isStageRoot) {
        const unsigned int lastEmitted = device->getLastPlanCmdSeqEmitted();
        if (!(pc.seq != 0u && pc.seq <= lastEmitted)) {
            if (pc.mode == PLAN_CMD_MODE_EXACT) {
                trigger = (layerIndex == pc.triggerLayer) && (pos == pc.triggerPos);
            } else {
                trigger = true;
            }
        }
    }

    if (trigger) {
        emitEpoch = curEpoch + 1u;
        if (pc.version == DLLAMA_PLAN_CMD_VERSION_V2 && pc.nMoves != 0u) {
            cmd = 4u;
            fromNode = pc.seq;
            printf("🧭 [plan][emit][gpu] node=%u stage=%u layer=%u pos=%u epoch=%u kind=cmdlist seq=%u moves=%u\n",
                (unsigned)myNode,
                (unsigned)(myStage != nullptr ? myStage->stageIndex : 0u),
                (unsigned)layerIndex,
                (unsigned)pos,
                (unsigned)emitEpoch,
                (unsigned)pc.seq,
                (unsigned)pc.nMoves);
        } else {
            cmd = pc.cmdKind;
            headMove = pc.nHeadsToMove;
            ffnMove = pc.nFfnToMove;
            fromNode = pc.fromNodeIndex;
            toNode = pc.toNodeIndex;
            printf("🧭 [plan][emit][gpu] node=%u stage=%u layer=%u pos=%u epoch=%u kind=%u headMove=%u ffnMove=%u from=%u to=%u\n",
                (unsigned)myNode,
                (unsigned)(myStage != nullptr ? myStage->stageIndex : 0u),
                (unsigned)layerIndex,
                (unsigned)pos,
                (unsigned)emitEpoch,
                (unsigned)cmd,
                (unsigned)headMove,
                (unsigned)ffnMove,
                (unsigned)fromNode,
                (unsigned)toNode);
        }
        device->setLastPlanCmdSeqEmitted(pc.seq);
        std::fflush(stdout);
    }

    for (NnUint b = 0; b < batchSize; ++b) {
        float *out = planPipe + b * 8u;
        out[0] = (float)emitEpoch;
        out[1] = (float)cmd;
        out[2] = (float)fromNode;
        out[3] = (float)toNode;
        out[4] = (float)headMove;
        out[5] = (float)layerIndex;
        out[6] = (float)pos;
        out[7] = (float)ffnMove;
    }
    return true;
}

static bool handleVulkanPlanApply(
    NnVulkanDevice *device,
    NnNetExecution *netExecution,
    NnSegmentConfig *segmentConfig,
    NnUint opIndex
) {
    if (device == nullptr || netExecution == nullptr || segmentConfig == nullptr) return true;
    NnOpConfig *op = &segmentConfig->ops[opIndex];
    if (op->input.source != SRC_PIPE) return true;

    const float *in0 = (const float *)netExecution->pipes[op->input.pointerIndex];
    if (in0 == nullptr) return true;
    const unsigned int curEpoch = device->getPlanEpoch();
    const unsigned int msgEpoch = (in0[0] >= 0.0f) ? (unsigned int)in0[0] : 0u;
    const unsigned int cmd = (in0[1] >= 0.0f) ? (unsigned int)in0[1] : 0u;
    const unsigned int fromNode = (in0[2] >= 0.0f) ? (unsigned int)in0[2] : 0u;
    const unsigned int toNode = (in0[3] >= 0.0f) ? (unsigned int)in0[3] : 0u;
    const unsigned int headMove = (in0[4] >= 0.0f) ? (unsigned int)in0[4] : 0u;
    const unsigned int layerIndex = (in0[5] >= 0.0f) ? (unsigned int)in0[5] : 0u;
    const unsigned int pos = (in0[6] >= 0.0f) ? (unsigned int)in0[6] : 0u;
    const unsigned int ffnMove = (in0[7] >= 0.0f) ? (unsigned int)in0[7] : 0u;

    if (!(cmd == 1u || cmd == 2u || cmd == 3u || cmd == 4u) || msgEpoch <= curEpoch) return true;

    auto *plan = const_cast<NnUnevenPartitionPlan *>(device->getPartitionPlan());
    if (plan == nullptr || plan->nStages == 0u) return true;
    const auto *cfg = (const NnPlanApplyOpCodeConfig *)op->config;
    const NnUint onlyStage = cfg != nullptr ? cfg->onlyStageIndex : 0u;
    if (onlyStage >= plan->nStages) return true;
    const NnUint myNode = device->getNodeIndex();
    const NnStageConfig *myStage = findStageForNodeVulkan(plan, myNode);
    if (myStage == nullptr || myStage->stageIndex != onlyStage) return true;
    const NnStageConfig *st = &plan->stages[onlyStage];

    auto inStage = [&](NnUint node) -> bool {
        for (NnUint i = 0; i < st->nNodes; ++i) if (st->nodeIndices[i] == node) return true;
        return false;
    };
    auto stageRank = [&](NnUint node) -> int {
        for (NnUint i = 0; i < st->nNodes; ++i) if (st->nodeIndices[i] == node) return (int)i;
        return -1;
    };
    auto adjacent = [&](NnUint a, NnUint b) -> bool {
        const int ra = stageRank(a), rb = stageRank(b);
        if (ra < 0 || rb < 0) return false;
        const int d = (ra >= rb) ? (ra - rb) : (rb - ra);
        return d == 1;
    };
    auto recomputeStarts = [&](NnDimSplit &split) {
        NnUint run = 0u;
        for (NnUint i = 0; i < st->nNodes; ++i) {
            const NnUint n = st->nodeIndices[i];
            split.starts[n] = run;
            run += split.lengths[n];
        }
    };

    NnUint stageQHeadsTotal = 0u, stageKvHeadsTotal = 0u;
    if (plan->kvHeadSplit.lengths != nullptr && plan->headSplit.lengths != nullptr) {
        for (NnUint i = 0; i < st->nNodes; ++i) {
            const NnUint n = st->nodeIndices[i];
            stageQHeadsTotal += plan->headSplit.lengths[n];
            stageKvHeadsTotal += plan->kvHeadSplit.lengths[n];
        }
    }
    NnUint gqaGroupSize = 1u;
    if (stageKvHeadsTotal != 0u && stageQHeadsTotal != 0u && (stageQHeadsTotal % stageKvHeadsTotal) == 0u) {
        gqaGroupSize = stageQHeadsTotal / stageKvHeadsTotal;
        if (gqaGroupSize == 0u) gqaGroupSize = 1u;
    }

    std::vector<int> deltaHeadOrKv(plan->nNodes, 0);
    std::vector<int> deltaFfn(plan->nNodes, 0);
    bool reject = false;

    auto addMove = [&](NnUint f, NnUint t, NnUint h, NnUint ffn) {
        if (!inStage(f) || !inStage(t) || f == t || !adjacent(f, t)) { reject = true; return; }
        if (h != 0u) {
            if (gqaGroupSize > 1u) {
                if (!getEnableKvRedundancyDuringMigration() || h > NN_KV_REDUNDANCY_PAD_HEADS) { reject = true; return; }
            }
            deltaHeadOrKv[f] -= (int)h;
            deltaHeadOrKv[t] += (int)h;
        }
        if (ffn != 0u) {
            deltaFfn[f] -= (int)ffn;
            deltaFfn[t] += (int)ffn;
        }
    };

    if (cmd == 4u) {
        const unsigned int wantSeq = fromNode;
        const PlanCommandSnapshot snap2 = planCommandCache().load();
        const PlanCommand &pc2 = snap2.cmd;
        if (!isValidPlanCommandHeader(pc2) || pc2.seq != wantSeq || !planCommandHasMoveList(pc2)) {
            printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u layer=%u pos=%u cmdlist missing/mismatch\n",
                (unsigned)myNode, (unsigned)onlyStage, (unsigned)msgEpoch, (unsigned)layerIndex, (unsigned)pos);
            std::fflush(stdout);
            return true;
        }
        const uint32_t maxMovesAllowed = std::min<uint32_t>(DLLAMA_PLAN_CMD_MAX_MOVES, (uint32_t)(2u * st->nNodes));
        if (pc2.nMoves > maxMovesAllowed) {
            printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u reject: nMoves=%u > maxAllowed=%u\n",
                (unsigned)myNode, (unsigned)onlyStage, (unsigned)msgEpoch, (unsigned)pc2.nMoves, (unsigned)maxMovesAllowed);
            std::fflush(stdout);
            return true;
        }
        for (uint32_t i = 0; i < pc2.nMoves; ++i) {
            addMove(pc2.moves[i].fromNodeIndex, pc2.moves[i].toNodeIndex, pc2.moves[i].headMove, pc2.moves[i].ffnMove);
            if (reject) break;
        }
    } else {
        addMove(fromNode, toNode, (cmd == 1u || cmd == 3u) ? headMove : 0u, (cmd == 2u || cmd == 3u) ? ffnMove : 0u);
    }

    if (reject) {
        printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u layer=%u pos=%u reject: bad move\n",
            (unsigned)myNode, (unsigned)onlyStage, (unsigned)msgEpoch, (unsigned)layerIndex, (unsigned)pos);
        std::fflush(stdout);
        return true;
    }

    bool changed = false;
    if (gqaGroupSize > 1u && plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
        for (NnUint i = 0; i < st->nNodes; ++i) {
            const NnUint n = st->nodeIndices[i];
            const int newLen = (int)plan->kvHeadSplit.lengths[n] + deltaHeadOrKv[n];
            if (newLen < 0) reject = true;
        }
        if (!reject) {
            for (NnUint i = 0; i < st->nNodes; ++i) {
                const NnUint n = st->nodeIndices[i];
                if (deltaHeadOrKv[n] != 0) {
                    plan->kvHeadSplit.lengths[n] = (NnUint)((int)plan->kvHeadSplit.lengths[n] + deltaHeadOrKv[n]);
                    changed = true;
                }
            }
            if (changed) {
                recomputeStarts(plan->kvHeadSplit);
                for (NnUint i = 0; i < st->nNodes; ++i) {
                    const NnUint n = st->nodeIndices[i];
                    plan->headSplit.starts[n] = plan->kvHeadSplit.starts[n] * gqaGroupSize;
                    plan->headSplit.lengths[n] = plan->kvHeadSplit.lengths[n] * gqaGroupSize;
                }
            }
        }
    } else if (plan->headSplit.starts && plan->headSplit.lengths) {
        for (NnUint i = 0; i < st->nNodes; ++i) {
            const NnUint n = st->nodeIndices[i];
            const int newLen = (int)plan->headSplit.lengths[n] + deltaHeadOrKv[n];
            if (newLen < 0) reject = true;
        }
        if (!reject) {
            for (NnUint i = 0; i < st->nNodes; ++i) {
                const NnUint n = st->nodeIndices[i];
                if (deltaHeadOrKv[n] != 0) {
                    plan->headSplit.lengths[n] = (NnUint)((int)plan->headSplit.lengths[n] + deltaHeadOrKv[n]);
                    changed = true;
                }
            }
            if (changed) recomputeStarts(plan->headSplit);
        }
    }

    bool ffnChanged = false;
    if (!reject && plan->ffnSplit.starts && plan->ffnSplit.lengths) {
        for (NnUint i = 0; i < st->nNodes; ++i) {
            const NnUint n = st->nodeIndices[i];
            const int newLen = (int)plan->ffnSplit.lengths[n] + deltaFfn[n];
            if (newLen < 0) reject = true;
        }
        if (!reject) {
            for (NnUint i = 0; i < st->nNodes; ++i) {
                const NnUint n = st->nodeIndices[i];
                if (deltaFfn[n] != 0) {
                    plan->ffnSplit.lengths[n] = (NnUint)((int)plan->ffnSplit.lengths[n] + deltaFfn[n]);
                    ffnChanged = true;
                }
            }
            if (ffnChanged) {
                recomputeStarts(plan->ffnSplit);
                changed = true;
            }
        }
    }

    if (reject) {
        printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u layer=%u pos=%u reject: split underflow\n",
            (unsigned)myNode, (unsigned)onlyStage, (unsigned)msgEpoch, (unsigned)layerIndex, (unsigned)pos);
        std::fflush(stdout);
        return true;
    }

    if (changed) {
        printf("🧭 [plan][apply][gpu] node=%u stage=%u epoch=%u layer=%u pos=%u cmd=%u kvHeads=%u qHeads=%u ffnDim=%u\n",
            (unsigned)myNode,
            (unsigned)onlyStage,
            (unsigned)msgEpoch,
            (unsigned)layerIndex,
            (unsigned)pos,
            (unsigned)cmd,
            (unsigned)(plan->kvHeadSplit.lengths ? plan->kvHeadSplit.lengths[myNode] : 0u),
            (unsigned)(plan->headSplit.lengths ? plan->headSplit.lengths[myNode] : 0u),
            (unsigned)(plan->ffnSplit.lengths ? plan->ffnSplit.lengths[myNode] : 0u));
        std::fflush(stdout);
        device->setPlanEpoch(msgEpoch);
    }
    return true;
}
''',
)

# Replace Vulkan batch offset/width with uneven-aware versions.
start = s.index("NnUint NnVulkanDeviceData::resolveBufferBatchOffset")
end = s.index("NnVulkanBuffer *NnVulkanDeviceData::resolvePipeByIndex", start)
new_block = r'''NnUint NnVulkanDeviceData::resolveBufferBatchOffset(NnPointerConfig *config, NnUint batchIndex, NnUint zIndex) {
    assert(batchIndex < netConfig->nBatches);
    if (config->type == PNTR_RAW)
        return 0;

    const NnSize3D bufferSize = resolveBufferSize(config);
    const NnSize blockSize = getBlockSize(bufferSize.floatType);
    assert(bufferSize.x % blockSize == 0);
    const NnUint sizeX = bufferSize.x / blockSize;
    const NnUint offsetZ = sizeX * netConfig->nBatches * zIndex;

    if (config->type == PNTR_BATCH)
        return offsetZ + sizeX * batchIndex;
    if (config->type == PNTR_BATCHED_SLICE) {
        NnUint logicalOffset = 0u;
        NnUint logicalLength = bufferSize.x;
        resolveUnevenSliceVulkan(netConfig, nodeConfig, config, bufferSize, &logicalOffset, &logicalLength);
        (void)logicalLength;
        assert((logicalOffset % blockSize) == 0u);
        return offsetZ + sizeX * batchIndex + (logicalOffset / blockSize);
    }
    throw std::runtime_error("Cannot determine buffer offset");
}

NnUint NnVulkanDeviceData::resolveBufferBatchWidth(NnPointerConfig *config) {
    const NnSize3D bufferSize = resolveBufferSize(config);
    const NnSize blockSize = getBlockSize(bufferSize.floatType);
    assert(bufferSize.x % blockSize == 0);
    const NnUint sizeX = bufferSize.x / blockSize;

    if (config->type == PNTR_RAW)
        return sizeX;
    if (config->type == PNTR_BATCH)
        return sizeX;
    if (config->type == PNTR_BATCHED_SLICE) {
        NnUint logicalOffset = 0u;
        NnUint logicalLength = bufferSize.x;
        resolveUnevenSliceVulkan(netConfig, nodeConfig, config, bufferSize, &logicalOffset, &logicalLength);
        (void)logicalOffset;
        assert((logicalLength % blockSize) == 0u);
        return logicalLength / blockSize;
    }
    throw std::runtime_error("Cannot determine buffer width");
}

'''
s = s[:start] + new_block + s[end:]

insert_after_once(
    """NnVulkanBuffer *NnVulkanDeviceData::resolveBufferByIndex(NnUint bufferIndex) {
    assert(bufferIndex < nodeConfig->nBuffers);
    return buffers[bufferIndex].get();
}
""",
    r'''

void NnVulkanDeviceData::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    if (nodeConfig != nullptr) {
        nodeConfig->partitionPlan = plan;
    }
}

const NnUnevenPartitionPlan *NnVulkanDeviceData::getPartitionPlan() const {
    return nodeConfig != nullptr ? nodeConfig->partitionPlan : nullptr;
}

NnUint NnVulkanDeviceData::getNodeIndex() const {
    return nodeConfig != nullptr ? nodeConfig->nodeIndex : 0u;
}
''',
)

replace_once(
    """NnVulkanDevice::NnVulkanDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) :
    context(gpuIndex),
    copier(&context),
    bufferFactory(&context, &copier),
    data(&bufferFactory, netConfig, nodeConfig),
    netConfig(netConfig),
    nodeConfig(nodeConfig),
    netExecution(netExecution)
{}""",
    """NnVulkanDevice::NnVulkanDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan) :
    context(gpuIndex),
    copier(&context),
    bufferFactory(&context, &copier),
    netConfig(netConfig),
    nodeConfig(nodeConfig),
    netExecution(netExecution),
    partitionPlan(partitionPlan),
    data(&bufferFactory, netConfig, nodeConfig)
{
    setPartitionPlan(partitionPlan);
}""",
)

insert_after_once(
    """NnVulkanDevice::~NnVulkanDevice() {}
""",
    r'''

void NnVulkanDevice::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    partitionPlan = plan;
    if (nodeConfig != nullptr) {
        nodeConfig->partitionPlan = plan;
    }
    data.setPartitionPlan(plan);
}
''',
)

replace_once(
    """NnDeviceSegment *NnVulkanDevice::createSegment(NnUint segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    return new NnVulkanDeviceSegment(&context, &copier, &bufferFactory, &data, netConfig, segmentIndex, segmentConfig, netExecution);
};""",
    """NnDeviceSegment *NnVulkanDevice::createSegment(NnUint segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    return new NnVulkanDeviceSegment(this, &context, &copier, &bufferFactory, &data, netConfig, segmentIndex, segmentConfig, netExecution);
};""",
)

replace_once(
    """        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];

        std::vector<NnVulkanBatchInfo> batchInfo = buildBatchInfo(opConfig, data, nBatches);""",
    """        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        if (isVulkanControlOp(opConfig->code)) {
            continue;
        }

        std::vector<NnVulkanBatchInfo> batchInfo = buildBatchInfo(opConfig, data, nBatches);""",
)

replace_once(
    """NnVulkanDeviceSegment::NnVulkanDeviceSegment(NnVulkanContext *context, NnVulkanStagingCopier *copier, NnVulkanBufferFactory *bufferFactory, NnVulkanDeviceData *data, NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution) :
    context(context),
    copier(copier),
    data(data),
    netConfig(netConfig),
    segmentIndex(segmentIndex),
    segmentConfig(segmentConfig),
    netExecution(netExecution),""",
    """NnVulkanDeviceSegment::NnVulkanDeviceSegment(NnVulkanDevice *ownerDevice, NnVulkanContext *context, NnVulkanStagingCopier *copier, NnVulkanBufferFactory *bufferFactory, NnVulkanDeviceData *data, NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution) :
    context(context),
    copier(copier),
    data(data),
    netConfig(netConfig),
    segmentIndex(segmentIndex),
    segmentConfig(segmentConfig),
    netExecution(netExecution),
    ownerDevice(ownerDevice),""",
)
replace_once(
    """    this->segmentData.reset(new NnVulkanDeviceSegmentData(bufferFactory, data, segmentConfig, netExecution->nBatches));
    this->lastBatchSize = 0;""",
    """    this->segmentData.reset(new NnVulkanDeviceSegmentData(bufferFactory, data, segmentConfig, netExecution->nBatches));
    this->lastBatchSize = 0;
    this->commandBufferDirty = true;""",
)
replace_once(
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnSize3D inputSize = data->resolveBufferSize(&opConfig->input);""",
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        if (isVulkanControlOp(opConfig->code)) {
            continue;
        }
        NnSize3D inputSize = data->resolveBufferSize(&opConfig->input);""",
)
replace_once(
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        std::vector<NnOpBufferAccess> &accesses = opBufferAccesses[opIndex];""",
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (isVulkanControlOp(segmentConfig->ops[opIndex].code)) {
            vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
                vk::DescriptorSetLayoutCreateFlags(),
                0,
                nullptr
            );
            descriptorSetLayouts[opIndex] = context->device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);
            continue;
        }
        std::vector<NnOpBufferAccess> &accesses = opBufferAccesses[opIndex];""",
)
replace_once(
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        std::vector<NnOpBufferAccess> &accesses = opBufferAccesses[opIndex];
        for (NnUint i = 0; i < accesses.size(); i++) {""",
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (isVulkanControlOp(segmentConfig->ops[opIndex].code)) {
            continue;
        }
        std::vector<NnOpBufferAccess> &accesses = opBufferAccesses[opIndex];
        for (NnUint i = 0; i < accesses.size(); i++) {""",
)
replace_once(
    """    if (nUniformBuffers > 0)
        descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, nUniformBuffers));

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(""",
    """    if (nUniformBuffers > 0)
        descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, nUniformBuffers));
    if (descriptorPoolSizes.empty())
        descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1));

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(""",
)
replace_once(
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &descriptorSetLayouts[opIndex]);""",
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (isVulkanControlOp(segmentConfig->ops[opIndex].code)) {
            continue;
        }
        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &descriptorSetLayouts[opIndex]);""",
)
replace_once(
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        context->device.destroyPipeline(pipelines[opIndex]);
        context->device.destroyPipelineLayout(pipelineLayouts[opIndex]);
    }""",
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (pipelines[opIndex]) context->device.destroyPipeline(pipelines[opIndex]);
        if (pipelineLayouts[opIndex]) context->device.destroyPipelineLayout(pipelineLayouts[opIndex]);
    }""",
)
replace_once(
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        context->device.destroyDescriptorSetLayout(descriptorSetLayouts[opIndex]);
        context->device.destroyShaderModule(shaderModules[opIndex]);
    }""",
    """    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        if (descriptorSetLayouts[opIndex]) context->device.destroyDescriptorSetLayout(descriptorSetLayouts[opIndex]);
        if (shaderModules[opIndex]) context->device.destroyShaderModule(shaderModules[opIndex]);
    }""",
)

insert_after_once(
    """void NnVulkanDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    assert(opIndex < segmentConfig->nOps);
    assert(offset + nBytes <= segmentConfig->ops[opIndex].weightSize.nBytes);
    NnVulkanBuffer *buffer = segmentData->resolveOpWeightVulkanBuffer(opIndex);
    buffer->write(weight, offset, nBytes);
}
""",
    r'''

void NnVulkanDeviceSegment::setPartitionPlan(const NnUnevenPartitionPlan *plan) {
    if (ownerDevice != nullptr) {
        ownerDevice->setPartitionPlan(plan);
    } else if (data != nullptr) {
        data->setPartitionPlan(plan);
    }
    refreshPointers();
}

void NnVulkanDeviceSegment::refreshPointers() {
    if (data == nullptr || segmentConfig == nullptr || segmentData == nullptr) return;
    const NnUnevenPartitionPlan *plan = (ownerDevice != nullptr) ? ownerDevice->getPartitionPlan() : data->getPartitionPlan();
    data->setPartitionPlan(plan);
    const NnUint nodeIndex = data->getNodeIndex();

    auto getSplitStart = [&](NnSliceTag tag, NnUint *outStart) -> bool {
        if (plan == nullptr || outStart == nullptr) return false;
        switch (tag) {
            case NN_SLICE_VOCAB:
                if (plan->vocabSplit.starts) { *outStart = plan->vocabSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_FFN:
                if (plan->ffnSplit.starts) { *outStart = plan->ffnSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_DIM:
                if (plan->dimSplit.starts) { *outStart = plan->dimSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_HEAD:
                if (plan->headSplit.starts) { *outStart = plan->headSplit.starts[nodeIndex]; return true; }
                return false;
            case NN_SLICE_KV_HEAD:
                if (plan->kvHeadSplit.starts) { *outStart = plan->kvHeadSplit.starts[nodeIndex]; return true; }
                return false;
            default:
                return false;
        }
    };

    auto findRmsNormColSize = [&](NnUint fromOpIndex, const NnSize3D &inputSizeNow) -> NnUint {
        for (NnUint j = fromOpIndex + 1u; j < segmentConfig->nOps && j < fromOpIndex + 8u; ++j) {
            NnOpConfig *c = &segmentConfig->ops[j];
            if (c->code != OP_RMS_NORM) continue;
            if (c->weightSize.x == 0u) continue;
            if (inputSizeNow.x % c->weightSize.x != 0u) continue;
            return c->weightSize.x;
        }
        return 0u;
    };

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; ++opIndex) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        if (isVulkanControlOp(opConfig->code)) continue;

        std::vector<NnVulkanBatchInfo> batchInfo = buildBatchInfo(opConfig, data, netExecution->nBatches);
        NnVulkanBuffer *batchInfoBuffer = segmentData->resolveOpBatchInfoVulkanBuffer(opIndex);
        batchInfoBuffer->write((NnByte *)batchInfo.data());

        NnSize3D inputSize = data->resolveBufferSize(&opConfig->input);
        NnSize3D outputSize = data->resolveBufferSize(&opConfig->output);
        if (opConfig->code == OP_CAST &&
            opConfig->output.type == PNTR_BATCHED_SLICE &&
            inputSize.x != outputSize.x) {
            outputSize = size3D(outputSize.floatType, outputSize.z, outputSize.y, inputSize.x);
        }

        if (plan != nullptr && opConfig->config != nullptr) {
            if (opConfig->code == OP_MULTIHEAD_ATT) {
                auto *cfg = (NnMultiHeadAttOpConfig *)opConfig->config;
                if (plan->headSplit.starts && plan->headSplit.lengths) {
                    cfg->nHeads0 = plan->headSplit.lengths[nodeIndex];
                    cfg->qStart = plan->headSplit.starts[nodeIndex] * cfg->headDim;
                    cfg->qSliceD0 = cfg->nHeads0 * cfg->headDim;
                    cfg->qStride = cfg->nHeads * cfg->headDim;
                }
                if (plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
                    const NnUint kvHeads0 = plan->kvHeadSplit.lengths[nodeIndex];
                    cfg->kvStart = plan->kvHeadSplit.starts[nodeIndex] * cfg->headDim;
                    cfg->kvDim0 = kvHeads0 * cfg->headDim;
                    cfg->kvStride = cfg->nKvHeads * cfg->headDim;
                }
            } else if (opConfig->code == OP_SHIFT) {
                auto *cfg = (NnShiftOpCodeConfig *)opConfig->config;
                if (cfg->dstRowStride != 0u && cfg->dstColStartUnit != 0u &&
                    plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
                    cfg->dstColStart = plan->kvHeadSplit.starts[nodeIndex] * cfg->dstColStartUnit;
                }
            } else if (opConfig->code == OP_MATMUL) {
                auto *cfg = (NnMatmulOpConfig *)opConfig->config;
                if (cfg->view == 1u && cfg->outSliceTag != NN_SLICE_AUTO && cfg->outStartUnit != 0u) {
                    NnUint st = 0u;
                    if (getSplitStart(cfg->outSliceTag, &st)) cfg->outStart = st * cfg->outStartUnit;
                }
                if (cfg->view == 2u && cfg->inSliceTag != NN_SLICE_AUTO && cfg->inStartUnit != 0u) {
                    NnUint st = 0u;
                    if (getSplitStart(cfg->inSliceTag, &st)) cfg->inStart = st * cfg->inStartUnit;
                }
                if (cfg->aView.sizeX != 0u) cfg->aView.sizeX = inputSize.x;
                if (cfg->cView.sizeX != 0u) cfg->cView.sizeX = outputSize.x;
            } else if (opConfig->code == OP_MUL) {
                auto *cfg = (NnMulOpCodeConfig *)opConfig->config;
                if (plan->ffnSplit.starts && plan->ffnSplit.lengths && cfg->view.sizeX != 0u) {
                    cfg->view.offset = plan->ffnSplit.starts[nodeIndex];
                    cfg->view.sizeX = plan->ffnSplit.lengths[nodeIndex];
                    cfg->view.strideX = 1u;
                }
            } else if (opConfig->code == OP_INV_RMS) {
                auto *cfg = (NnInvRmsOpConfig *)opConfig->config;
                const NnUint colSize = findRmsNormColSize(opIndex, inputSize);
                if (colSize != 0u && inputSize.x % colSize == 0u) {
                    const NnUint newCols = inputSize.x / colSize;
                    if (outputSize.x >= newCols && newCols != 0u) cfg->nColumns = newCols;
                }
            } else if (opConfig->code == OP_RMS_NORM) {
                auto *cfg = (NnRmsNormOpConfig *)opConfig->config;
                const NnUint colSize = opConfig->weightSize.x;
                if (colSize != 0u && inputSize.x % colSize == 0u) {
                    const NnUint newCols = inputSize.x / colSize;
                    if (newCols != 0u) cfg->nColumns = newCols;
                }
            } else if (opConfig->code == OP_ROPE) {
                auto *cfg = (NnRopeOpConfig *)opConfig->config;
                if (cfg->isQ == 1u) cfg->slice.qDim0 = inputSize.x;
                else cfg->slice.kvDim0 = inputSize.x;
            }
        }

        if (opConfig->configSize > 0) {
            NnVulkanBuffer *configBuffer = segmentData->resolveOpConfigVulkanBuffer(opIndex);
            configBuffer->write(opConfig->config);
        }
    }

    commandBufferDirty = true;
    lastBatchSize = 0u;
    if (ownerDevice != nullptr) {
        planEpochReady.store(ownerDevice->getPlanEpoch(), std::memory_order_release);
    }
}
''',
)

replace_once(
    """void NnVulkanDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize)  {
    assert(threadIndex == 0);

    if (opIndex != 0) {
        // TODO: this is a design problem, executor tries to forward all ops in a segment
        return;
    }

    // TODO: this should be called only after weights loading is done
    copier->tryRelease();""",
    """void NnVulkanDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize)  {
    assert(threadIndex == 0);

    if (opIndex != 0) {
        // TODO: this is a design problem, executor tries to forward all ops in a segment
        return;
    }

    if (ownerDevice != nullptr) {
        const unsigned int deviceEpoch = ownerDevice->getPlanEpoch();
        const unsigned int readyEpoch = planEpochReady.load(std::memory_order_acquire);
        if (readyEpoch != deviceEpoch) {
            refreshPointers();
            planEpochReady.store(deviceEpoch, std::memory_order_release);
        }
    }

    // TODO: this should be called only after weights loading is done
    copier->tryRelease();""",
)

insert_after_once(
    """        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (opConfig->input.source == SRC_PIPE) {
                NnByte *pipeData = netExecution->pipes[opConfig->input.pointerIndex];
                NnVulkanBuffer *buffer = data->pipes[opConfig->input.pointerIndex].get();
                buffer->write(pipeData, 0u, buffer->calcSliceSize(batchSize, netConfig->nBatches));
            }
        }
    }
""",
    r'''

    bool hasOnlyControlOps = true;
    for (NnUint i = 0; i < segmentConfig->nOps; ++i) {
        NnOpConfig *opConfig = &segmentConfig->ops[i];
        if (opConfig->code == OP_PLAN_BARRIER) {
            handleVulkanPlanBarrier(ownerDevice, netExecution, segmentConfig, i, batchSize);
        } else if (opConfig->code == OP_PLAN_APPLY) {
            handleVulkanPlanApply(ownerDevice, netExecution, segmentConfig, i);
        } else {
            hasOnlyControlOps = false;
        }
    }
    if (hasOnlyControlOps) {
        return;
    }
''',
)
replace_once(
    """    if (lastBatchSize != batchSize) {
        lastBatchSize = batchSize;""",
    """    if (lastBatchSize != batchSize || commandBufferDirty) {
        lastBatchSize = batchSize;
        commandBufferDirty = false;""",
)
replace_once(
    """        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            std::vector<vk::BufferMemoryBarrier> memoryBarriers;""",
    """        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            if (isVulkanControlOp(segmentConfig->ops[opIndex].code)) {
                continue;
            }
            std::vector<vk::BufferMemoryBarrier> memoryBarriers;""",
)

p.write_text(s)
print("vulkan dynamic patch applied")
