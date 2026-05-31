from pathlib import Path

base = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor")
p = base / "src/llm.cpp"
s = p.read_text()

old = """    // 4. Start Segment (Embedding)
    NnSegmentConfigBuilder start;
    if (isFirstStage) {
        // [修改] First Stage 所有节点都负责 Embedding
        // 1. 先同步 Token (广播: Root -> Stage 0 Workers)
        // 注意：这里假设 SYNC_WITH_ROOT 能正确处理 Node 0 到 Stage 0 其他节点的广播
        start.addSync(n->tokenPipeIndex, SYNC_WITH_ROOT);

        // 2. 所有节点本地计算 Embedding (避免传输大的 Embedding 向量)
        start.addOp(OP_EMBEDDING, "embedding", 0, 
            pointerBatchConfig(SRC_PIPE, n->tokenPipeIndex),
            pointerBatchConfig(SRC_PIPE, n->xPipeIndex), 
            n->tokenEmbeddingSize, NnEmbeddingOpConfig{});
    }
    addSegmentLogged(start, "start", 0u);
"""

new = """    // 4. Start Segment (Embedding)
    NnSegmentConfigBuilder start;
    if (isFirstStage) {
        // Segment syncs run after all ops in this executor. Therefore the
        // stage root computes the embedding first, then broadcasts X to peers.
        const bool amStageRoot = (myStage == nullptr)
            ? (nodeIndex == 0u)
            : (nodeIndex == myStage->rootNodeIndex);
        if (amStageRoot) {
            start.addOp(OP_EMBEDDING, "embedding", 0,
                pointerBatchConfig(SRC_PIPE, n->tokenPipeIndex),
                pointerBatchConfig(SRC_PIPE, n->xPipeIndex),
                n->tokenEmbeddingSize, NnEmbeddingOpConfig{});
        }
        start.addSync(n->xPipeIndex, SYNC_WITH_ROOT);
    }
    addSegmentLogged(start, "start", 0u);
"""

if old not in s:
    raise SystemExit("start segment block not found")
s = s.replace(old, new, 1)

old2 = """    // --- 1. Embedding ---
    if (isFirstStage) {
        b += loader->loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);
    } else {
        b += net->tokenEmbeddingSize.nBytes; 
    }
"""

new2 = """    // --- 1. Embedding ---
    const bool isStageRoot = (myStage == nullptr)
        ? (nodeIndex == 0u)
        : (nodeIndex == myStage->rootNodeIndex);
    if (isFirstStage && isStageRoot) {
        b += loader->loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);
    } else {
        b += net->tokenEmbeddingSize.nBytes;
    }
"""

if old2 not in s:
    raise SystemExit("embedding load block not found")
s = s.replace(old2, new2, 1)

p.write_text(s)
print("patched src/llm.cpp")
