from pathlib import Path

repls = {
    "src/nn/nn-cpu.cpp": (
        """            if (opContext->opCode == OP_MULTIHEAD_ATT) {
                auto *cfg = (NnMultiHeadAttOpConfig *)opContext->opConfig;
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
""",
        """            if (opContext->opCode == OP_MULTIHEAD_ATT) {
                auto *cfg = (NnMultiHeadAttOpConfig *)opContext->opConfig;
                const bool fullAttBuffer =
                    opConfig->input.type == PNTR_BATCHED_SLICE ||
                    opConfig->output.type == PNTR_BATCHED_SLICE;
                if (plan->headSplit.starts && plan->headSplit.lengths) {
                    cfg->nHeads0 = plan->headSplit.lengths[nodeIndex];
                    cfg->qSliceD0 = cfg->nHeads0 * cfg->headDim;
                    if (fullAttBuffer) {
                        cfg->qStart = plan->headSplit.starts[nodeIndex] * cfg->headDim;
                        cfg->qStride = cfg->nHeads * cfg->headDim;
                    } else {
                        cfg->qStart = 0u;
                        cfg->qStride = cfg->qSliceD0;
                    }
                }
                if (plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
                    const NnUint kvHeads0 = plan->kvHeadSplit.lengths[nodeIndex];
                    cfg->kvDim0 = kvHeads0 * cfg->headDim;
                    if (fullAttBuffer) {
                        cfg->kvStart = plan->kvHeadSplit.starts[nodeIndex] * cfg->headDim;
                        cfg->kvStride = cfg->nKvHeads * cfg->headDim;
                    } else {
                        cfg->kvStart = 0u;
                        cfg->kvStride = cfg->kvDim0;
                    }
                }
""",
    ),
    "src/nn/nn-vulkan.cpp": (
        """            if (opConfig->code == OP_MULTIHEAD_ATT) {
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
""",
        """            if (opConfig->code == OP_MULTIHEAD_ATT) {
                auto *cfg = (NnMultiHeadAttOpConfig *)opConfig->config;
                const bool fullAttBuffer =
                    opConfig->input.type == PNTR_BATCHED_SLICE ||
                    opConfig->output.type == PNTR_BATCHED_SLICE;
                if (plan->headSplit.starts && plan->headSplit.lengths) {
                    cfg->nHeads0 = plan->headSplit.lengths[nodeIndex];
                    cfg->qSliceD0 = cfg->nHeads0 * cfg->headDim;
                    if (fullAttBuffer) {
                        cfg->qStart = plan->headSplit.starts[nodeIndex] * cfg->headDim;
                        cfg->qStride = cfg->nHeads * cfg->headDim;
                    } else {
                        cfg->qStart = 0u;
                        cfg->qStride = cfg->qSliceD0;
                    }
                }
                if (plan->kvHeadSplit.starts && plan->kvHeadSplit.lengths) {
                    const NnUint kvHeads0 = plan->kvHeadSplit.lengths[nodeIndex];
                    cfg->kvDim0 = kvHeads0 * cfg->headDim;
                    if (fullAttBuffer) {
                        cfg->kvStart = plan->kvHeadSplit.starts[nodeIndex] * cfg->headDim;
                        cfg->kvStride = cfg->nKvHeads * cfg->headDim;
                    } else {
                        cfg->kvStart = 0u;
                        cfg->kvStride = cfg->kvDim0;
                    }
                }
""",
    ),
}

base = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor")
for rel, (old, new) in repls.items():
    path = base / rel
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"old block not found in {rel}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched {rel}")
