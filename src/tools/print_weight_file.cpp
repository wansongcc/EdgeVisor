#include "llm.hpp"
#include "nn/nn-core.hpp"
#include "nn/nn-quants.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>

// -----------------------------------------------------------------------------
// 辅助工具函数
// -----------------------------------------------------------------------------

static bool streq(const char *a, const char *b) {
    return a && b && std::strcmp(a, b) == 0;
}

static NnFloatType parseFloatType(const char *s) {
    if (s == nullptr) return F_32;
    if (streq(s, "F_32") || streq(s, "f32") || streq(s, "fp32")) return F_32;
    if (streq(s, "F_16") || streq(s, "f16") || streq(s, "fp16")) return F_16;
    if (streq(s, "F_Q40") || streq(s, "q40")) return F_Q40;
    if (streq(s, "F_Q80") || streq(s, "q80")) return F_Q80;
    return F_32;
}

// 核心计算函数：根据数据类型和元素个数计算字节偏移
// 用于计算 Column Parallel (Input Split) 的 Byte Offset
static NnSize calcTypeOffset(NnFloatType type, NnUint elements) {
    if (type == F_Q40) {
        // Q40 格式: 每 32 个元素组成一个 Block，占用 18 字节
        // 前提: elements 必须是 32 的倍数 (Hidden Dim 通常满足)
        return (NnSize)((uint64_t)elements / 32 * 18);
    } else if (type == F_Q80) {
        // Q80 格式: 每 32 个元素 34 字节
        return (NnSize)((uint64_t)elements / 32 * 34);
    } else if (type == F_16) {
        return (NnSize)elements * 2;
    } else if (type == F_32) {
        return (NnSize)elements * 4;
    }
    return 0; // 未知类型
}

static void printRegion(const char *name, NnUint layer, NnUint expert, NnSize offset, NnSize nBytes, const char *typeTag) {
    if (layer == 0xffffffffu && expert == 0xffffffffu) {
        std::printf("%10zu  %10zu  %-22s  type=%s\n", (size_t)offset, (size_t)nBytes, name, typeTag);
        return;
    }
    if (expert == 0xffffffffu) {
        std::printf("%10zu  %10zu  %-22s  layer=%u  type=%s\n", (size_t)offset, (size_t)nBytes, name, layer, typeTag);
        return;
    }
    std::printf("%10zu  %10zu  %-22s  layer=%u  expert=%u  type=%s\n", (size_t)offset, (size_t)nBytes, name, layer, expert, typeTag);
}

static bool seekFile(FILE *f, NnSize offset) {
    if (!f) return false;
#ifdef _WIN32
    return _fseeki64(f, (long long)offset, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t)offset, SEEK_SET) == 0;
#endif
}

// 打印文件指定偏移处的数据，模拟 [WEIGHT DATA] 日志格式
static void printWeightDataAtOffset0File(FILE *f, NnSize fileSize, NnSize regionOffset, NnSize regionBytes, NnUint dumpBytes) {
    if (dumpBytes == 0u)
        return;
    if (regionOffset >= fileSize) {
        std::printf("   [WEIGHT DATA] At Offset +0: <out-of-file>\n");
        return;
    }
    
    // 确保不越界
    const NnSize avail = (fileSize - regionOffset);
    const NnSize n = std::min<NnSize>((NnSize)dumpBytes, std::min<NnSize>(regionBytes, avail));
    
    if (n == 0u) {
        std::printf("   [WEIGHT DATA] At Offset +0: <empty>\n");
        return;
    }
    if (!seekFile(f, regionOffset)) {
        std::printf("   [WEIGHT DATA] At Offset +0: <seek-failed>\n");
        return;
    }
    
    std::vector<unsigned char> buf((size_t)n);
    const size_t got = std::fread(buf.data(), 1, (size_t)n, f);
    if (got == 0) {
        std::printf("   [WEIGHT DATA] At Offset +0: <read-failed>\n");
        return;
    }
    
    // 打印数据
    std::printf("   [WEIGHT DATA] At Offset +0:");
    for (size_t i = 0; i < got; ++i) {
        std::printf(" %02x", (unsigned)buf[i]);
    }
    std::printf("\n");
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
    "Usage: %s <model_path> [--max-seq <n>] [--sync-type <type>] [--limit-layers <n>] [--dump-bytes <n>]\n",
        argv0);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    const char *path = argv[1];
    NnUint maxSeq = 4096;
    NnUint limitLayers = 0; // 0 means all
    NnFloatType syncType = F_32;
    NnUint dumpBytes = 8;

    // 参数解析
    for (int i = 2; i < argc; ++i) {
        if (streq(argv[i], "--max-seq") && i + 1 < argc) {
            maxSeq = (NnUint)std::strtoul(argv[++i], nullptr, 10);
        } else if (streq(argv[i], "--limit-layers") && i + 1 < argc) {
            limitLayers = (NnUint)std::strtoul(argv[++i], nullptr, 10);
        } else if (streq(argv[i], "--dump-bytes") && i + 1 < argc) {
            dumpBytes = (NnUint)std::strtoul(argv[++i], nullptr, 10);
        } else if (streq(argv[i], "--sync-type") && i + 1 < argc) {
            syncType = parseFloatType(argv[++i]);
        }
    }

    try {
        LlmHeader h = loadLlmHeader(path, maxSeq, syncType);
        printLlmHeader(&h);

        std::unique_ptr<FILE, int(*)(FILE *)> fdPtr(std::fopen(path, "rb"), std::fclose);
        FILE *fd = fdPtr.get();
        if (fd == nullptr)
            throw std::runtime_error(std::string("Cannot open model file: ") + path);

        const NnUint ffDim = (h.archType == QWEN3_MOE) ? h.moeHiddenDim : h.hiddenDim;

        // --- 定义 Tensor 尺寸 (Logic View) ---
        const NnSize3D embedding = size2D(F_32, h.vocabSize, h.dim);
        const NnSize3D rmsNorm = size1D(F_32, h.dim);
        const NnSize3D qkRmsNorm = size1D(F_32, h.headDim);
        const NnSize3D moeGate = size2D(F_32, h.dim, h.nExperts);

        const NnSize3D w_q = size2D(h.weightType, h.dim, h.qDim);
        const NnSize3D w_k = size2D(h.weightType, h.dim, h.kvDim);
        const NnSize3D w_v = size2D(h.weightType, h.dim, h.kvDim);
        const NnSize3D w_wo = size2D(h.weightType, h.qDim, h.dim); // Input = qDim

        const NnSize3D w_w1 = size2D(h.weightType, h.dim, ffDim);
        const NnSize3D w_w2 = size2D(h.weightType, ffDim, h.dim); // Input = ffDim
        const NnSize3D w_w3 = size2D(h.weightType, h.dim, ffDim);

        const NnSize3D w_logits = size2D(h.weightType, h.dim, h.vocabSize);

        // --- 预计算 Node 1 的 Offset (Input Split) ---
        // Node 1 负责后半部分的 Input
        // Wo: Input is qDim. Offset = calc(qDim / 2)
        const NnSize offset_Wo_Node1 = calcTypeOffset(h.weightType, h.qDim / 2);
        
        // W2: Input is ffDim. Offset = calc(ffDim / 2)
        const NnSize offset_W2_Node1 = calcTypeOffset(h.weightType, ffDim / 2);

        std::printf("\n# Layout: offset(bytes)  length(bytes)  name\n");
        NnSize off = (NnSize)h.headerSize;

        // 1. Embedding
        printRegion("embedding", 0xffffffffu, 0xffffffffu, off, embedding.nBytes, floatTypeToString(F_32));
        off += embedding.nBytes;

        // 2. Layers
        const NnUint nLayersToPrint = (limitLayers > 0u) ? std::min(limitLayers, h.nLayers) : h.nLayers;
        for (NnUint layer = 0u; layer < nLayersToPrint; ++layer) {
            
            // --- Q (Row Parallel: Split Output) ---
            printRegion("block_matmul_q", layer, 0xffffffffu, off, w_q.nBytes, floatTypeToString(h.weightType));
            // Node 1 偏移 = 总大小的一半
            std::printf("   [NODE 1 OFFSET +%zu] ", (size_t)(w_q.nBytes / 2));
            printWeightDataAtOffset0File(fd, h.fileSize, off + (w_q.nBytes / 2), w_q.nBytes, dumpBytes);
            off += w_q.nBytes;

            // --- K (Row Parallel: Split Output) ---
            printRegion("block_matmul_k", layer, 0xffffffffu, off, w_k.nBytes, floatTypeToString(h.weightType));
            std::printf("   [NODE 1 OFFSET +%zu] ", (size_t)(w_k.nBytes / 2));
            printWeightDataAtOffset0File(fd, h.fileSize, off + (w_k.nBytes / 2), w_k.nBytes, dumpBytes);
            off += w_k.nBytes;

            // --- V (Row Parallel: Split Output) ---
            printRegion("block_matmul_v", layer, 0xffffffffu, off, w_v.nBytes, floatTypeToString(h.weightType));
            std::printf("   [NODE 1 OFFSET +%zu] ", (size_t)(w_v.nBytes / 2));
            printWeightDataAtOffset0File(fd, h.fileSize, off + (w_v.nBytes / 2), w_v.nBytes, dumpBytes);
            off += w_v.nBytes;

            // --- Wo (Column Parallel: Split Input) ---
            printRegion("block_matmul_wo", layer, 0xffffffffu, off, w_wo.nBytes, floatTypeToString(h.weightType));
            // Node 1 偏移 = Input 维度的后半截
            std::printf("   [NODE 1 OFFSET +%zu] ", (size_t)offset_Wo_Node1);
            printWeightDataAtOffset0File(fd, h.fileSize, off + offset_Wo_Node1, w_wo.nBytes, dumpBytes);
            off += w_wo.nBytes;

            if (h.nExperts > 0u) {
                // MOE 逻辑 (简化处理，通常 Norm 在 MOE 之前/之后)
                off += moeGate.nBytes;
                for (NnUint e = 0u; e < h.nExperts; ++e) {
                    off += w_w1.nBytes + w_w2.nBytes + w_w3.nBytes;
                }
            } else {
                // --- W1 (Row Parallel: Split Output) ---
                printRegion("block_matmul_w1", layer, 0xffffffffu, off, w_w1.nBytes, floatTypeToString(h.weightType));
                std::printf("   [NODE 1 OFFSET +%zu] ", (size_t)(w_w1.nBytes / 2));
                printWeightDataAtOffset0File(fd, h.fileSize, off + (w_w1.nBytes / 2), w_w1.nBytes, dumpBytes);
                off += w_w1.nBytes;

                // --- W2 (Column Parallel: Split Input) ---
                printRegion("block_matmul_w2", layer, 0xffffffffu, off, w_w2.nBytes, floatTypeToString(h.weightType));
                // Node 1 偏移 = Input 维度的后半截
                std::printf("   [NODE 1 OFFSET +%zu] ", (size_t)offset_W2_Node1);
                printWeightDataAtOffset0File(fd, h.fileSize, off + offset_W2_Node1, w_w2.nBytes, dumpBytes);
                off += w_w2.nBytes;

                // --- W3 (Row Parallel: Split Output) ---
                printRegion("block_matmul_w3", layer, 0xffffffffu, off, w_w3.nBytes, floatTypeToString(h.weightType));
                std::printf("   [NODE 1 OFFSET +%zu] ", (size_t)(w_w3.nBytes / 2));
                printWeightDataAtOffset0File(fd, h.fileSize, off + (w_w3.nBytes / 2), w_w3.nBytes, dumpBytes);
                off += w_w3.nBytes;
            }

            // Norm 层 (不打印 Node 1 Check，因为通常不切分)
            if (h.archType == QWEN3 || h.archType == QWEN3_MOE) {
                printRegion("block_norm_q", layer, 0xffffffffu, off, qkRmsNorm.nBytes, floatTypeToString(F_32));
                off += qkRmsNorm.nBytes;
                printRegion("block_norm_k", layer, 0xffffffffu, off, qkRmsNorm.nBytes, floatTypeToString(F_32));
                off += qkRmsNorm.nBytes;
            }
            printRegion("block_norm_0", layer, 0xffffffffu, off, rmsNorm.nBytes, floatTypeToString(F_32));
            off += rmsNorm.nBytes;
            printRegion("block_norm_1", layer, 0xffffffffu, off, rmsNorm.nBytes, floatTypeToString(F_32));
            off += rmsNorm.nBytes;
        }

        // 3. Final Layers
        // Fast-forward 如果限制了打印层数
        if (nLayersToPrint < h.nLayers) {
            const NnUint skipped = h.nLayers - nLayersToPrint;
            NnSize perLayer = 0;
            perLayer += w_q.nBytes + w_k.nBytes + w_v.nBytes + w_wo.nBytes;
            if (h.nExperts == 0) perLayer += w_w1.nBytes + w_w2.nBytes + w_w3.nBytes;
            // ... (简化计算，略去 MOE/Norm 细节) ...
            // 实际使用时如果 limitLayers 生效，最后的文件大小校验可能会 fail，这不影响 Debug
        }

        // ...
        
        std::printf("\n# Check Finished.\n");

    } catch (const std::exception &e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}