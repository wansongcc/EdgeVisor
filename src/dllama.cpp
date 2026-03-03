#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-cpu-ops.hpp"
#include "nn/nn-network-local.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-executor.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "app.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <iostream>
#include <deque>

static void computeLogitsStats(const float* logits, NnUint vocabSize,
                              bool &hasNaN, bool &hasInf,
                              float &minLogit, float &maxLogit,
                              int &maxIndex, NnUint &zeroCount) {
    hasNaN = false;
    hasInf = false;
    minLogit = 1e9f;
    maxLogit = -1e9f;
    maxIndex = -1;
    zeroCount = 0u;
    if (logits == nullptr || vocabSize == 0u) return;
    for (NnUint i = 0; i < vocabSize; ++i) {
        float v = logits[i];
        if (v == 0.0f) zeroCount++;
        if (std::isnan(v)) { hasNaN = true; continue; }
        if (std::isinf(v)) { hasInf = true; continue; }
        if (v > maxLogit) { maxLogit = v; maxIndex = (int)i; }
        if (v < minLogit) minLogit = v;
    }
}

#ifndef DLLAMA_DEBUG_TOPK_LOGITS
#define DLLAMA_DEBUG_TOPK_LOGITS 0
#endif

#if DLLAMA_DEBUG_TOPK_LOGITS
static int debugLogitsBatchSelect() {
    const char *v = std::getenv("DLLAMA_DEBUG_LOGITS_BATCH");
    if (v == nullptr || *v == '\0') return 0; // default: batch 0
    return std::atoi(v); // -1 => all batches
}

static void debugValidateLogits(const float* logits, NnUint vocabSize, bool &hasNaN, bool &hasInf, float &minLogit, float &maxLogit, int &maxIndex) {
    hasNaN = false;
    hasInf = false;
    maxLogit = -1e9f;
    minLogit = 1e9f;
    maxIndex = -1;
    for (NnUint i = 0; i < vocabSize; ++i) {
        float val = logits[i];
        if (std::isnan(val)) hasNaN = true;
        if (std::isinf(val)) hasInf = true;
        if (val > maxLogit) { maxLogit = val; maxIndex = (int)i; }
        if (val < minLogit) minLogit = val;
    }
}

static void printEscapedPiece(const char* s, size_t maxLen = 32) {
    if (s == nullptr) {
        printf("~");
        return;
    }
    for (size_t i = 0; s[i] != '\0' && i < maxLen; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\n') { printf("\\n"); continue; }
        if (c == '\r') { printf("\\r"); continue; }
        if (c == '\t') { printf("\\t"); continue; }
        if (c < 32 || c >= 127) {
            printf("\\x%02x", (unsigned)c);
            continue;
        }
        putchar((int)c);
    }
}

static void debugTopKLogits(const AppInferenceContext* context, const float* logits, NnUint vocabSize, int k, const char* tag) {
    if (context == nullptr || context->tokenizer == nullptr || context->tokenizer->vocab == nullptr) return;
    if (k <= 0) return;
    if ((NnUint)k > vocabSize) k = (int)vocabSize;

    struct Item { float v; int i; };
    std::vector<Item> top;
    top.reserve((size_t)k);

    for (NnUint i = 0; i < vocabSize; ++i) {
        float v = logits[i];
        if ((int)top.size() < k) {
            top.push_back(Item{v, (int)i});
            if ((int)top.size() == k) {
                std::sort(top.begin(), top.end(), [](const Item& a, const Item& b) { return a.v > b.v; });
            }
            continue;
        }
        if (v <= top.back().v) continue;
        int lo = 0;
        int hi = k;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (v > top[mid].v) hi = mid; else lo = mid + 1;
        }
        for (int j = k - 1; j > lo; --j) top[j] = top[j - 1];
        top[lo] = Item{v, (int)i};
    }

    printf("🧭 [TopK] %s k=%d\n", tag == nullptr ? "" : tag, k);
    for (int j = 0; j < (int)top.size(); ++j) {
        int id = top[j].i;
        const char* piece = (id >= 0 && (NnUint)id < context->tokenizer->vocabSize) ? context->tokenizer->vocab[id] : nullptr;
        printf("  #%d id=%d logit=%+.4f piece=\"", j, id, top[j].v);
        printEscapedPiece(piece);
        printf("\"\n");
    }
}

static void debugVocabCoverage(const float* logits, NnUint vocabSize, const char* tag) {
    if (logits == nullptr || vocabSize == 0) return;
    const int blocks = 16;
    NnUint blockSize = (vocabSize + blocks - 1) / blocks;
    NnUint zeroCount = 0;
    NnUint nearZeroCount = 0;
    for (NnUint i = 0; i < vocabSize; ++i) {
        float v = logits[i];
        if (v == 0.0f) zeroCount++;
        if (std::fabs(v) < 1e-6f) nearZeroCount++;
    }
    printf("🧱 [VocabCoverage] %s vocab=%u zero=%u (%.1f%%) | |v|<1e-6=%u (%.1f%%)\n",
        tag == nullptr ? "" : tag,
        vocabSize,
        zeroCount,
        vocabSize ? (100.0f * (float)zeroCount / (float)vocabSize) : 0.0f,
        nearZeroCount,
        vocabSize ? (100.0f * (float)nearZeroCount / (float)vocabSize) : 0.0f);

    for (int b = 0; b < blocks; ++b) {
        NnUint lo = (NnUint)b * blockSize;
        NnUint hi = std::min(vocabSize, lo + blockSize);
        if (lo >= hi) break;
        float mx = -1e30f;
        int mxIdx = -1;
        for (NnUint i = lo; i < hi; ++i) {
            float v = logits[i];
            if (v > mx) { mx = v; mxIdx = (int)i; }
        }
        printf("  block[%2d] [%6u..%6u) max=%+.4f idx=%d\n", b, lo, hi, mx, mxIdx);
    }
}
#endif

static void inferenceRunOnce(AppInferenceContext *context, const char* prompt, NnUint steps) {
    if (prompt == nullptr)
        throw std::runtime_error("Prompt is required");
    if (steps == 0)
        throw std::runtime_error("Number of steps is required");

    std::vector<int> inputTokensVec(std::strlen(prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    NnUint pos = 0;
    int nInputTokens;
    context->tokenizer->encode(const_cast<char*>(prompt), inputTokens, &nInputTokens, true, true);

#if DLLAMA_DEBUG_TOPK_LOGITS
    if (context->tokenizer->vocabSize != context->header->vocabSize) {
        printf("⚠️ Tokenizer vocabSize=%u != Model vocabSize=%u (强烈怀疑 tokenizer/model 不匹配)\n",
            context->tokenizer->vocabSize,
            context->header->vocabSize);
    }
    {
        int dumpN = std::min(nInputTokens, 32);
        printf("🧾 Prompt tokens n=%d: ", nInputTokens);
        for (int i = 0; i < dumpN; ++i) {
            int id = inputTokens[i];
            const char* piece = (id >= 0 && (NnUint)id < context->tokenizer->vocabSize) ? context->tokenizer->vocab[id] : nullptr;
            printf("%d(\"", id);
            printEscapedPiece(piece, 16);
            printf("\")%s", (i + 1 < dumpN) ? " " : "");
        }
        if (dumpN < nInputTokens) printf(" ...");
        printf("\n");
    }
#endif

    if (nInputTokens > context->header->seqLen)
        throw std::runtime_error("The number of prompt tokens is greater than the sequence length");
    if ((NnUint)nInputTokens > steps)
        throw std::runtime_error("The number of prompt tokens is greater than the number of steps");

    NnSize sentBytes = 0;
    NnSize recvBytes = 0;
    NnUint evalTotalTime = 0;
    NnUint predTotalTime = 0;

    // Per-stage/per-node profiling stats (enabled when executor benchmark is on)
    struct NodePerfAgg {
        unsigned long long execUs = 0;
        unsigned long long syncUs = 0;
        unsigned long long forwardCount = 0;
        unsigned long long tokenCount = 0;
        NnUint stageIndex = 0;
        bool hasStage = false;
    };
    struct TokenNodePerf {
        unsigned long long execUs = 0;
        unsigned long long syncUs = 0;
        bool hasValue = false;
    };
    struct TokenPerfSample {
        NnUint pos = 0u;
        std::vector<TokenNodePerf> nodePerf;
    };
    const NnUint nNodes = (context->args->nWorkers + 1);
    std::vector<NodePerfAgg> perfAgg;
    std::deque<TokenPerfSample> predPerfHistory;
    bool migrationPivotKnown = false;
    bool migrationWindowReported = false;
    NnUint migrationPivotPos = 0u;
    NnUint migrationPivotLayer = 0u;
    const NnUint migrationWindowTokens = 20u;
    if (context->args->benchmark) {
        perfAgg.resize(nNodes);
    }

    int token = inputTokens[pos];
    printf("%s\n", prompt);
    for (;;) {
        long remainingTokens = nInputTokens - 1 - (long)pos;
        if (remainingTokens <= 0)
            break;
        NnUint batchSize = remainingTokens < context->args->nBatches
            ? remainingTokens
            : context->args->nBatches;

        context->inference->setBatchSize(batchSize);
        context->inference->setPosition(pos);
        for (NnUint i = 0; i < batchSize; i++)
            context->inference->setToken(i, inputTokens[pos + i]);

        context->inference->forward();

        if (context->args->benchmark) {
            const std::vector<LlmPerfPacket>& perf = context->inference->getLastPerf();
            for (const LlmPerfPacket& p : perf) {
                if (p.nodeIndex >= perfAgg.size()) continue;
                NodePerfAgg& a = perfAgg[p.nodeIndex];
                a.execUs += p.execUs;
                a.syncUs += p.syncUs;
                a.forwardCount += 1;
                a.tokenCount += (unsigned long long)std::max<NnUint>(1u, p.batchSize);
                a.stageIndex = p.stageIndex;
                a.hasStage = true;
            }
        }
        float* logits = context->inference->logitsPipe;
        NnUint vocabSize = context->header->vocabSize;
        bool hasNaN = false;
        bool hasInf = false;
        float maxLogit = -1e9f;
        float minLogit = 1e9f;
        int maxIndex = -1;

        // Always compute basic logits stats (even when TOPK debug isn't compiled).
        // In eval stage logits pipe is [batch][vocab]. Use the latest row of this window.
        const NnUint statBatch = (batchSize > 0u) ? (batchSize - 1u) : 0u;
        const float* logitsRow = logits + (size_t)statBatch * (size_t)vocabSize;
        NnUint zeroCount = 0u;
        computeLogitsStats(logitsRow, vocabSize, hasNaN, hasInf, minLogit, maxLogit, maxIndex, zeroCount);

#if DLLAMA_DEBUG_TOPK_LOGITS
        debugValidateLogits(logits, vocabSize, hasNaN, hasInf, minLogit, maxLogit, maxIndex);


        // 只在 eval 阶段前几步打印，避免刷屏
        {
            const int sel = debugLogitsBatchSelect();
            if (sel < 0) {
                for (NnUint bi = 0; bi < batchSize; ++bi) {
                    char tag[96];
                    std::snprintf(tag, sizeof(tag), "eval pos=%u batchSize=%u batchIndex=%u", (unsigned)pos, (unsigned)batchSize, (unsigned)bi);
                    debugTopKLogits(context, logits + (size_t)bi * (size_t)vocabSize, vocabSize, 10, tag);
                    debugVocabCoverage(logits + (size_t)bi * (size_t)vocabSize, vocabSize, tag);
                }
            } else {
                const NnUint bi = (NnUint)std::max(sel, 0);
                if (bi < batchSize) {
                    char tag[96];
                    std::snprintf(tag, sizeof(tag), "eval pos=%u batchSize=%u batchIndex=%u", (unsigned)pos, (unsigned)batchSize, (unsigned)bi);
                    debugTopKLogits(context, logits + (size_t)bi * (size_t)vocabSize, vocabSize, 10, tag);
                    debugVocabCoverage(logits + (size_t)bi * (size_t)vocabSize, vocabSize, tag);
                } else {
                    char tag[96];
                    std::snprintf(tag, sizeof(tag), "eval pos=%u batchSize=%u batchIndex=%d(out-of-range)", (unsigned)pos, (unsigned)batchSize, sel);
                    debugTopKLogits(context, logits, vocabSize, 10, tag);
                    debugVocabCoverage(logits, vocabSize, tag);
                }
            }
        }
#endif


        pos += batchSize;
        // 注意：这里是在“评估 prompt（不含最后一个 token）”的循环中，不能用 pos+1，
        // 否则在 pos==nInputTokens-1 时会越界，导致后续生成从错误 token 开始。

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);

        NnUint evalTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf("🔷️ Eval%5u ms Sync%5u ms | Sent%6zu kB Recv%6zu kB | (%d tokens)\n",
            evalTime / 1000,
            syncTime / 1000,
            sentBytes / 1024,
            recvBytes / 1024,
            batchSize);
        const bool statsOk = (!hasNaN && !hasInf && maxIndex >= 0 && vocabSize > 0u);
        printf("🧪 [Root Logits] (eval batchIndex=%u) Valid: %s | Range: [%.2f, %.2f] | MaxIdx: %d | Zero: %u/%u | NetDelta: S=%zu R=%zu\n",
            (unsigned)statBatch,
            statsOk ? "✅ OK" : "❌ FAIL",
            minLogit, maxLogit, maxIndex,
            (unsigned)zeroCount, (unsigned)vocabSize,
            sentBytes, recvBytes);
        evalTotalTime += evalTime + syncTime;
    }

    // 生成阶段的起始 token 应该是 prompt 的最后一个 token（位置为 nInputTokens-1）
    token = inputTokens[nInputTokens - 1];

    fflush(stdout);

    context->inference->setBatchSize(1);
    context->tokenizer->resetDecoder();

    const NnUint maxPos = std::min(context->header->seqLen, steps);
    for (; pos < maxPos; pos++) {
        context->inference->setPosition(pos);
        context->inference->setToken(0, token);
        context->inference->forward();

        // In pred stage batchSize==1. Always compute logits stats for debugging.
        {
            bool hasNaN = false;
            bool hasInf = false;
            float maxLogit = -1e9f;
            float minLogit = 1e9f;
            int maxIndex = -1;
            NnUint zeroCount = 0u;
            computeLogitsStats(context->inference->logitsPipe, context->header->vocabSize,
                               hasNaN, hasInf, minLogit, maxLogit, maxIndex, zeroCount);
            const bool statsOk = (!hasNaN && !hasInf && maxIndex >= 0 && context->header->vocabSize > 0u);
            printf("🧪 [Root Logits] (pred) Valid: %s | Range: [%.2f, %.2f] | MaxIdx: %d | Zero: %u/%u\n",
                statsOk ? "✅ OK" : "❌ FAIL",
                minLogit, maxLogit, maxIndex,
                (unsigned)zeroCount, (unsigned)context->header->vocabSize);
        }

        if (context->args->benchmark) {
            const std::vector<LlmPerfPacket>& perf = context->inference->getLastPerf();
            for (const LlmPerfPacket& p : perf) {
                if (p.nodeIndex >= perfAgg.size()) continue;
                NodePerfAgg& a = perfAgg[p.nodeIndex];
                a.execUs += p.execUs;
                a.syncUs += p.syncUs;
                a.forwardCount += 1;
                a.tokenCount += (unsigned long long)std::max<NnUint>(1u, p.batchSize);
                a.stageIndex = p.stageIndex;
                a.hasStage = true;
            }

            TokenPerfSample sample;
            sample.pos = pos;
            sample.nodePerf.resize(nNodes);
            for (const LlmPerfPacket& p : perf) {
                if (p.nodeIndex >= sample.nodePerf.size()) continue;
                sample.nodePerf[p.nodeIndex].execUs = p.execUs;
                sample.nodePerf[p.nodeIndex].syncUs = p.syncUs;
                sample.nodePerf[p.nodeIndex].hasValue = true;
            }
            predPerfHistory.push_back(std::move(sample));
            while (predPerfHistory.size() > 512u) {
                predPerfHistory.pop_front();
            }

            if (!migrationPivotKnown && context->inference->hasMigrationAck()) {
                const int ackPos = context->inference->getMigrationAckPos();
                const int ackLayer = context->inference->getMigrationAckLayer();
                if (ackPos >= 0 && ackLayer >= 0) {
                    migrationPivotKnown = true;
                    migrationPivotPos = (NnUint)ackPos;
                    migrationPivotLayer = (NnUint)ackLayer;
                    std::printf("📍 [migrate-prof] anchor layer=%u pos=%u window=%u-before/%u-after\n",
                        (unsigned)migrationPivotLayer,
                        (unsigned)migrationPivotPos,
                        (unsigned)migrationWindowTokens,
                        (unsigned)migrationWindowTokens);
                    std::fflush(stdout);
                }
            }

            if (migrationPivotKnown && !migrationWindowReported) {
                struct WindowAgg {
                    unsigned long long execUs = 0;
                    unsigned long long syncUs = 0;
                    unsigned long long count = 0;
                };

                const NnUint beforeStart = (migrationPivotPos >= migrationWindowTokens)
                    ? (migrationPivotPos - migrationWindowTokens)
                    : 0u;
                const NnUint beforeEnd = migrationPivotPos;
                const NnUint afterStart = migrationPivotPos + 1u;
                const NnUint afterEnd = migrationPivotPos + 1u + migrationWindowTokens;

                std::vector<WindowAgg> beforeAgg(nNodes);
                std::vector<WindowAgg> afterAgg(nNodes);
                for (const TokenPerfSample &s : predPerfHistory) {
                    const bool inBefore = (s.pos >= beforeStart && s.pos < beforeEnd);
                    const bool inAfter = (s.pos >= afterStart && s.pos < afterEnd);
                    if (!inBefore && !inAfter) continue;
                    for (NnUint node = 0; node < nNodes; ++node) {
                        if (node >= s.nodePerf.size()) continue;
                        const TokenNodePerf &np = s.nodePerf[node];
                        if (!np.hasValue) continue;
                        WindowAgg &dst = inBefore ? beforeAgg[node] : afterAgg[node];
                        dst.execUs += np.execUs;
                        dst.syncUs += np.syncUs;
                        dst.count += 1ull;
                    }
                }

                bool ready = true;
                for (NnUint node = 0; node < nNodes; ++node) {
                    if (beforeAgg[node].count < migrationWindowTokens || afterAgg[node].count < migrationWindowTokens) {
                        ready = false;
                        break;
                    }
                }

                if (ready) {
                    std::printf("\n⏱️  [Migration 20-token Avg] anchor(layer=%u pos=%u) before=[%u,%u) after=[%u,%u)\n",
                        (unsigned)migrationPivotLayer,
                        (unsigned)migrationPivotPos,
                        (unsigned)beforeStart,
                        (unsigned)beforeEnd,
                        (unsigned)afterStart,
                        (unsigned)afterEnd);
                    for (NnUint node = 0; node < nNodes; ++node) {
                        const double bExecMs = (double)beforeAgg[node].execUs / 1000.0 / (double)beforeAgg[node].count;
                        const double bSyncMs = (double)beforeAgg[node].syncUs / 1000.0 / (double)beforeAgg[node].count;
                        const double aExecMs = (double)afterAgg[node].execUs / 1000.0 / (double)afterAgg[node].count;
                        const double aSyncMs = (double)afterAgg[node].syncUs / 1000.0 / (double)afterAgg[node].count;
                        const double bTotalMs = bExecMs + bSyncMs;
                        const double aTotalMs = aExecMs + aSyncMs;
                        std::printf("  • Node %u: before=%6.2f ms (exec=%6.2f sync=%6.2f) | after=%6.2f ms (exec=%6.2f sync=%6.2f) | Δ=%+6.2f ms\n",
                            (unsigned)node,
                            bTotalMs,
                            bExecMs,
                            bSyncMs,
                            aTotalMs,
                            aExecMs,
                            aSyncMs,
                            aTotalMs - bTotalMs);
                    }
                    std::printf("\n");
                    std::fflush(stdout);
                    migrationWindowReported = true;
                }
            }
        }

#if DLLAMA_DEBUG_TOPK_LOGITS
        // 预测阶段：前若干步打印 topK，快速判断 logits 是否“像正常语言模型”
        {
            char tag[64];
            std::snprintf(tag, sizeof(tag), "pred pos=%u", (unsigned)pos);
            debugTopKLogits(context, context->inference->logitsPipe, context->header->vocabSize, 10, tag);
            debugVocabCoverage(context->inference->logitsPipe, context->header->vocabSize, tag);

            bool hasNaN = false;
            bool hasInf = false;
            float maxLogit = -1e9f;
            float minLogit = 1e9f;
            int maxIndex = -1;
            debugValidateLogits(context->inference->logitsPipe, context->header->vocabSize, hasNaN, hasInf, minLogit, maxLogit, maxIndex);
            printf("🧪 [Root Logits] (pred) Valid: %s | Range: [%.2f, %.2f] | MaxIdx: %d\n",
                (hasNaN || hasInf) ? "❌ FAIL" : "✅ OK",
                minLogit, maxLogit, maxIndex);
        }
#endif
        token = context->sampler->sample(context->inference->logitsPipe);

        char *piece = context->tokenizer->decode(token);

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);

        NnUint predTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf("🔶 Pred%5u ms Sync%5u ms | pos=%u | Sent%6zu kB Recv%6zu kB | %s\n",
            predTime / 1000,
            syncTime / 1000,
            (unsigned)pos,
            sentBytes / 1024,
            recvBytes / 1024,
            piece == nullptr ? "~" : piece);
        fflush(stdout);
        predTotalTime += predTime + syncTime;
    }

    NnUint nEvalTokens = nInputTokens - 1;
    NnUint nPredTokens = pos - nEvalTokens;
    float evalTotalTimeMs = evalTotalTime / 1000.0;
    float predTotalTimeMs = predTotalTime / 1000.0;
    printf("\n");
    printf("Evaluation\n");
    printf("   nBatches: %d\n", context->args->nBatches);
    printf("    nTokens: %d\n", nEvalTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        (nEvalTokens * 1000) / evalTotalTimeMs,
        evalTotalTimeMs / ((float) nEvalTokens));
    printf("Prediction\n");
    printf("    nTokens: %d\n", nPredTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        (nPredTokens * 1000) / predTotalTimeMs,
        predTotalTimeMs / ((float) nPredTokens));

    if (context->args->benchmark && !perfAgg.empty()) {
        printf("\n");
        printf("⏱️  [Stage/Node Profile Summary]\n");
        for (NnUint node = 0; node < (NnUint)perfAgg.size(); ++node) {
            const NodePerfAgg& a = perfAgg[node];
            if (a.forwardCount == 0 || a.tokenCount == 0) continue;

            const double execPerFwdMs = (double)a.execUs / 1000.0 / (double)a.forwardCount;
            const double syncPerFwdMs = (double)a.syncUs / 1000.0 / (double)a.forwardCount;
            const double totalPerFwdMs = execPerFwdMs + syncPerFwdMs;

            const double execPerTokMs = (double)a.execUs / 1000.0 / (double)a.tokenCount;
            const double syncPerTokMs = (double)a.syncUs / 1000.0 / (double)a.tokenCount;
            const double totalPerTokMs = execPerTokMs + syncPerTokMs;

            printf("  • Stage %u Node %u: per-fwd total=%6.2f ms (exec=%6.2f sync=%6.2f) | per-tok total=%6.2f ms (exec=%6.2f sync=%6.2f) | fwd=%llu tok=%llu\n",
                a.hasStage ? a.stageIndex : 0u,
                node,
                totalPerFwdMs,
                execPerFwdMs,
                syncPerFwdMs,
                totalPerTokMs,
                execPerTokMs,
                syncPerTokMs,
                (unsigned long long)a.forwardCount,
                (unsigned long long)a.tokenCount);
        }
        printf("\n");
        printf("Hint: prompt eval uses batchSize>1, so per-token is usually the meaningful metric for rebalancing.\n");

        if (migrationPivotKnown && !migrationWindowReported) {
            const NnUint beforeStart = (migrationPivotPos >= migrationWindowTokens)
                ? (migrationPivotPos - migrationWindowTokens)
                : 0u;
            const NnUint beforeEnd = migrationPivotPos;
            const NnUint afterStart = migrationPivotPos + 1u;
            const NnUint afterEnd = migrationPivotPos + 1u + migrationWindowTokens;
            std::printf("[migrate-prof] insufficient tokens for full window: anchor(layer=%u pos=%u) need before=[%u,%u) after=[%u,%u)\n",
                (unsigned)migrationPivotLayer,
                (unsigned)migrationPivotPos,
                (unsigned)beforeStart,
                (unsigned)beforeEnd,
                (unsigned)afterStart,
                (unsigned)afterEnd);
        }
    }
}

static bool isInteractiveQuitLine(const std::string& s) {
    return s == ":q" || s == ":quit" || s == "q" || s == "quit" || s == "exit";
}

static bool parseStepsCommand(const std::string& line, NnUint& outSteps) {
    // Accept: ":s 128" or ":steps 128"
    const char* prefixes[] = {":s", ":steps"};
    for (const char* p : prefixes) {
        if (line.rfind(p, 0) != 0) continue;
        size_t i = std::strlen(p);
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
        if (i >= line.size()) return false;
        char* end = nullptr;
        unsigned long v = std::strtoul(line.c_str() + i, &end, 10);
        if (end == nullptr || end == (line.c_str() + i)) return false;
        if (v == 0ul) return false;
        outSteps = (NnUint)v;
        return true;
    }
    return false;
}

static void inference(AppInferenceContext *context) {
    if (!context->args->interactive) {
        inferenceRunOnce(context, context->args->prompt, context->args->steps);
        return;
    }

    // Keep stdio/iostreams in sync since we mix printf() and std::getline().
    std::ios::sync_with_stdio(true);

    std::string curPrompt = (context->args->prompt != nullptr) ? std::string(context->args->prompt) : std::string();
    NnUint curSteps = context->args->steps;

    for (;;) {
        // Ask for missing inputs (so --interactive can be used standalone).
        while (curPrompt.empty() || curSteps == 0u) {
            printf("🕹️  [interactive] 请输入 prompt（或 :q 退出；:s <steps> 设置 steps）\n> ");
            fflush(stdout);
            std::string line;
            if (!std::getline(std::cin, line)) return;
            if (isInteractiveQuitLine(line)) return;
            NnUint newSteps = curSteps;
            if (parseStepsCommand(line, newSteps)) {
                curSteps = newSteps;
                printf("🕹️  [interactive] steps=%u\n", curSteps);
                continue;
            }
            if (!line.empty()) curPrompt = line;
        }

        inferenceRunOnce(context, curPrompt.c_str(), curSteps);

        printf("\n🕹️  [interactive] 回车复用上次 prompt 继续；直接输入新 prompt；:s <steps> 修改 steps；:q 退出\n> ");
        fflush(stdout);

        std::string line;
        if (!std::getline(std::cin, line)) {
            // EOF (e.g. Ctrl-D) => exit
            break;
        }
        if (isInteractiveQuitLine(line)) break;

        NnUint newSteps = curSteps;
        if (parseStepsCommand(line, newSteps)) {
            curSteps = newSteps;
            printf("🕹️  [interactive] steps=%u\n", curSteps);
            continue;
        }

        if (!line.empty()) {
            curPrompt = line;
        }
    }
}

// =====================================================================================
// E2E Matmul-View Zero-Offset Check (Inference Mode)
//
// Enable via env var:
//   DLLAMA_E2E_MATMUL_VIEW0_CHECK=1
//
// What it does:
// - Run prompt eval stage twice in two independent runInferenceApp() invocations.
// - Pass A: clears all OP_MATMUL a/b/c views (legacy behavior)
// - Pass B: keeps current views (offset=0 plumbing path)
// - Compares logits after each eval forward() call.
// =====================================================================================

static bool envFlagEnabled(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "TRUE") == 0;
}

static void clearMatmulViews(NnNodeConfig* nodeConfig) {
    if (nodeConfig == nullptr || nodeConfig->segments == nullptr) return;
    for (NnUint s = 0; s < nodeConfig->nSegments; ++s) {
        NnSegmentConfig* seg = &nodeConfig->segments[s];
        for (NnUint i = 0; i < seg->nOps; ++i) {
            NnOpConfig* op = &seg->ops[i];
            if (op->code != OP_MATMUL) continue;
            if (op->config == nullptr) continue;
            if (op->configSize < sizeof(NnMatmulOpConfig)) continue;
            NnMatmulOpConfig* cfg = (NnMatmulOpConfig*)op->config;
            std::memset(&cfg->aView, 0, sizeof(cfg->aView));
            std::memset(&cfg->bView, 0, sizeof(cfg->bView));
            std::memset(&cfg->cView, 0, sizeof(cfg->cView));
        }
    }
}

struct EvalLogitsCapture {
    NnUint vocabSize = 0;
    std::vector<std::vector<float>> logitsPerForward;
};

static EvalLogitsCapture g_evalCapture;
static bool g_evalCaptureClearViews = false;
static bool g_evalCompareClearViews = false;
static bool g_evalCompareOk = true;

static void evalOnlyCapture(AppInferenceContext* context) {
    if (context == nullptr) throw std::runtime_error("Internal error: context is null");
    if (context->args->prompt == nullptr) throw std::runtime_error("Prompt is required");
    if (context->nodeConfig == nullptr) throw std::runtime_error("Internal error: nodeConfig is null");

    if (g_evalCaptureClearViews) {
        clearMatmulViews(context->nodeConfig);
    }

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int* inputTokens = inputTokensVec.data();
    int nInputTokens = 0;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);
    if (nInputTokens <= 1) throw std::runtime_error("Prompt produced too few tokens");
    if ((NnUint)nInputTokens > context->header->seqLen) throw std::runtime_error("Prompt too long for model seqLen");

    g_evalCapture = EvalLogitsCapture();
    g_evalCapture.vocabSize = context->header->vocabSize;

    {
        const int evalTokens = nInputTokens - 1;
        int expectedForwards = 0;
        if (evalTokens > 0) {
            const int nb = (int)std::max<NnUint>(1u, context->args->nBatches);
            expectedForwards = (evalTokens + nb - 1) / nb;
        }
        printf("🧪 [E2E View0][Capture] promptTokens=%d evalTokens=%d nBatches=%u expectedForwards=%d clearMatmulViews=%s\n",
            nInputTokens,
            evalTokens,
            context->args->nBatches,
            expectedForwards,
            g_evalCaptureClearViews ? "true" : "false");
    }

    NnUint pos = 0;
    size_t forwardIndex = 0;
    while ((int)pos < nInputTokens - 1) {
        long remainingTokens = nInputTokens - 1 - (long)pos;
        if (remainingTokens <= 0) break;
        NnUint batchSize = remainingTokens < context->args->nBatches ? (NnUint)remainingTokens : context->args->nBatches;

        printf("🧪 [E2E View0][Capture] forward=%zu pos=%u batch=%u\n", forwardIndex, pos, batchSize);

        context->inference->setBatchSize(batchSize);
        context->inference->setPosition(pos);
        for (NnUint i = 0; i < batchSize; ++i) {
            context->inference->setToken(i, (NnUint)inputTokens[pos + i]);
        }
        context->inference->forward();

        const NnUint vocabSize = context->header->vocabSize;
        std::vector<float> snapshot;
        snapshot.resize(vocabSize);
        std::memcpy(snapshot.data(), context->inference->logitsPipe, (size_t)vocabSize * sizeof(float));
        g_evalCapture.logitsPerForward.push_back(std::move(snapshot));

        pos += batchSize;
        forwardIndex++;
    }

    printf("🧪 [E2E View0][Capture] done forwards=%zu\n", g_evalCapture.logitsPerForward.size());
}

static void evalOnlyCompare(AppInferenceContext* context) {
    if (context == nullptr) throw std::runtime_error("Internal error: context is null");
    if (context->args->prompt == nullptr) throw std::runtime_error("Prompt is required");
    if (context->nodeConfig == nullptr) throw std::runtime_error("Internal error: nodeConfig is null");
    if (g_evalCapture.vocabSize == 0 || g_evalCapture.logitsPerForward.empty())
        throw std::runtime_error("Internal error: missing reference capture");

    if (g_evalCompareClearViews) {
        clearMatmulViews(context->nodeConfig);
    }

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int* inputTokens = inputTokensVec.data();
    int nInputTokens = 0;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);
    if (nInputTokens <= 1) throw std::runtime_error("Prompt produced too few tokens");
    if ((NnUint)nInputTokens > context->header->seqLen) throw std::runtime_error("Prompt too long for model seqLen");

    const NnUint vocabSize = context->header->vocabSize;
    if (vocabSize != g_evalCapture.vocabSize)
        throw std::runtime_error("Vocab size mismatch between runs");

    {
        const int evalTokens = nInputTokens - 1;
        int expectedForwards = 0;
        if (evalTokens > 0) {
            const int nb = (int)std::max<NnUint>(1u, context->args->nBatches);
            expectedForwards = (evalTokens + nb - 1) / nb;
        }
        printf("🧪 [E2E View0][Compare] promptTokens=%d evalTokens=%d nBatches=%u expectedForwards=%d capturedForwards=%zu clearMatmulViews=%s\n",
            nInputTokens,
            evalTokens,
            context->args->nBatches,
            expectedForwards,
            g_evalCapture.logitsPerForward.size(),
            g_evalCompareClearViews ? "true" : "false");
    }

    const float absEps = 1e-6f;
    const float relEps = 1e-6f;

    NnUint pos = 0;
    size_t forwardIndex = 0;
    while ((int)pos < nInputTokens - 1) {
        long remainingTokens = nInputTokens - 1 - (long)pos;
        if (remainingTokens <= 0) break;
        NnUint batchSize = remainingTokens < context->args->nBatches ? (NnUint)remainingTokens : context->args->nBatches;

        printf("🧪 [E2E View0][Compare] forward=%zu pos=%u batch=%u\n", forwardIndex, pos, batchSize);

        context->inference->setBatchSize(batchSize);
        context->inference->setPosition(pos);
        for (NnUint i = 0; i < batchSize; ++i) {
            context->inference->setToken(i, (NnUint)inputTokens[pos + i]);
        }
        context->inference->forward();

        if (forwardIndex >= g_evalCapture.logitsPerForward.size()) {
            g_evalCompareOk = false;
            printf("❌ [E2E View0] Forward count mismatch: got extra forward at index=%zu\n", forwardIndex);
            return;
        }

        const std::vector<float>& ref = g_evalCapture.logitsPerForward[forwardIndex];
        const float* got = context->inference->logitsPipe;

        float maxAbsDiff = 0.0f;
        int maxIdx = -1;
        for (NnUint i = 0; i < vocabSize; ++i) {
            float a = ref[i];
            float b = got[i];
            float diff = std::fabs(a - b);
            float tol = absEps + relEps * std::max(std::fabs(a), std::fabs(b));
            if (diff > maxAbsDiff) { maxAbsDiff = diff; maxIdx = (int)i; }
            if (diff > tol) {
                g_evalCompareOk = false;
                printf("❌ [E2E View0] Logits mismatch at forward=%zu pos=%u idx=%u ref=%+.9f got=%+.9f diff=%.9g tol=%.9g\n",
                    forwardIndex, pos, i, a, b, diff, tol);
                printf("❌ [E2E View0] MaxAbsDiff=%.9g at idx=%d (may be same as first mismatch)\n", maxAbsDiff, maxIdx);
                return;
            }
        }

        printf("✅ [E2E View0] forward=%zu pos=%u batch=%u maxAbsDiff=%.3g\n",
            forwardIndex, pos, batchSize, maxAbsDiff);

        pos += batchSize;
        forwardIndex++;
    }

    if (forwardIndex != g_evalCapture.logitsPerForward.size()) {
        g_evalCompareOk = false;
        printf("❌ [E2E View0] Forward count mismatch: expected=%zu got=%zu\n",
            g_evalCapture.logitsPerForward.size(), forwardIndex);
    }

    printf("🧪 [E2E View0][Compare] done forwards=%zu compareOk=%s\n", forwardIndex, g_evalCompareOk ? "true" : "false");
}

static NnUint readStdin(const char *guide, char *buffer, NnUint size) {
    std::fflush(stdin);
    std::printf("%s", guide);
    if (std::fgets(buffer, size, stdin) != NULL) {
        NnUint length = std::strlen(buffer);
        if (length > 0 && buffer[length - 1] == '\n') {
            buffer[length - 1] = '\0';
            length--;
        }
        return length;
    }
    return 0;
}

static void perplexity(AppInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);

    printf("Evaluating %d tokens...\n", nInputTokens);

    float totalLogProb = 0.0f;
    NnUint pos = 0;

    context->inference->setBatchSize(1);

    for (pos = 0; pos < nInputTokens - 1; pos++) {
        context->inference->setPosition(pos);
        context->inference->setToken(0, inputTokens[pos]);
        context->inference->forward();

        float *logits = context->inference->logitsPipe;
        softmax_F32(logits, context->header->vocabSize);

        int targetToken = inputTokens[pos + 1];
        float prob = logits[targetToken];

        totalLogProb += std::log(std::max(prob, 1e-30f));
        printf("%5d / %d, prob=%f\n", pos + 1, nInputTokens - 1, prob);
    }

    float avgLogProb = totalLogProb / (float)(nInputTokens - 1);
    float perplexity = expf(-avgLogProb);

    printf("\n");
    printf("Results\n");
    printf("   perplexity: %f (lower = better)\n", perplexity);
    printf("   avgLogProb: %f\n", avgLogProb);
    printf("   bitPerToken: %f\n", -avgLogProb / std::log(2.0));
}

static void chat(AppInferenceContext *context) {
    const NnUint seqLen = context->header->seqLen;
    char prompt[2048];

    TokenizerChatStops stops(context->tokenizer);
    ChatTemplateGenerator templateGenerator(context->args->chatTemplateType, context->tokenizer->chatTemplate, stops.stops[0]);
    EosDetector eosDetector(stops.nStops, context->tokenizer->eosTokenIds.data(), stops.stops, stops.maxStopLength, stops.maxStopLength);

    const NnUint sysPromptLength = readStdin("💻 System prompt (optional): ", prompt, sizeof(prompt));
    std::vector<ChatItem> deltaItems;
    if (sysPromptLength > 0)
        deltaItems.push_back(ChatItem{"system", prompt});

    NnUint pos = 0;
    NnUint userPromptLength;
    int token;
    int nInputTokens;
    do {
        do {
            userPromptLength = readStdin("\n👱 User\n> ", prompt, sizeof(prompt));
        } while (userPromptLength == 0);

        deltaItems.push_back(ChatItem{"user", prompt});

        GeneratedChat inputPrompt = templateGenerator.generate(deltaItems.size(), deltaItems.data(), true);
        std::unique_ptr<int[]> inputTokensPtr(new int[inputPrompt.length + 2]);
        int *inputTokens = inputTokensPtr.get();

        bool isStart = pos == 0;
        context->tokenizer->encode((char*)inputPrompt.content, inputTokens, &nInputTokens, isStart, true);

        NnUint userPromptEndPos = (NnUint)std::min<unsigned int>(seqLen, pos + nInputTokens - 1);
        NnUint i = 0;
        for (;;) {
            int remainingTokens = userPromptEndPos - pos;
            if (remainingTokens <= 0)
                break;
            NnUint batchSize = remainingTokens < context->args->nBatches
                ? remainingTokens
                : context->args->nBatches;

            context->inference->setBatchSize(batchSize);
            context->inference->setPosition(pos);
            for (NnUint j = 0; j < batchSize; j++)
                context->inference->setToken(j, inputTokens[i + j]);

            context->inference->forward();

            i += batchSize;
            pos += batchSize;
            // 这里同 inference()：prompt eval 只跑到最后一个 token 的前一位，
            // 循环结束后再用 i 指向的那个“最后 token”启动生成。
        }

        // 生成阶段起始 token：prompt 的最后一个 token
        if (i < (NnUint)nInputTokens) token = inputTokens[i];
        else token = inputTokens[nInputTokens - 1];

        context->inference->setBatchSize(1);
        context->tokenizer->resetDecoder();

        printf("\n🤖 Assistant\n");
        if (inputPrompt.publicPrompt != nullptr)
            printf("%s", inputPrompt.publicPrompt);

        while (pos < seqLen) {
            context->inference->setPosition(pos);
            context->inference->setToken(0, token);
            context->inference->forward();

            token = context->sampler->sample(context->inference->logitsPipe);

            char *piece = context->tokenizer->decode(token);
            EosDetectorType eosType = eosDetector.append(token, piece);
            if (eosType == NOT_EOS || eosType == EOS) {
                char *delta = eosDetector.getDelta();
                if (delta != nullptr) {
                    printf("%s", delta);
                    fflush(stdout);
                }
                eosDetector.reset();
            }
            pos++;
            if (eosType == EOS) break;
        }

        deltaItems.clear();
    } while (pos < seqLen);

    printf("(end of context)\n");
}

int main(int argc, char **argv) {
    initQuants();
    initSockets();

    int returnCode = EXIT_SUCCESS;
    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, true);
        if (std::strcmp(args.mode, "inference") == 0) {
            printf("nNodes=%d\n", args.nWorkers);
            if (envFlagEnabled("DLLAMA_E2E_MATMUL_VIEW0_CHECK")) {
                if (args.nWorkers != 0) {
                    throw std::runtime_error("DLLAMA_E2E_MATMUL_VIEW0_CHECK currently requires single-node run (--n-workers 0)");
                }

                printf("🧪 [E2E View0] Running prompt-eval twice: (A) clear matmul views, (B) keep matmul views\n");

                g_evalCaptureClearViews = true;
                runInferenceApp(&args, &evalOnlyCapture);

                g_evalCompareOk = true;
                g_evalCompareClearViews = false;
                runInferenceApp(&args, &evalOnlyCompare);

                if (!g_evalCompareOk) {
                    printf("❌ [E2E View0] FAILED\n");
                    returnCode = EXIT_FAILURE;
                } else {
                    printf("✅ [E2E View0] OK\n");
                }
            } else {
                runInferenceApp(&args, &inference);
            }
        } else if (std::strcmp(args.mode, "perplexity") == 0)
            runInferenceApp(&args, &perplexity);
        else if (std::strcmp(args.mode, "chat") == 0)
            runInferenceApp(&args, &chat);
        else if (std::strcmp(args.mode, "worker") == 0)
            runWorkerApp(&args);
        else
            throw std::runtime_error("Unsupported mode");
    } catch (const std::exception &e) {
        printf("🚨 Critical error: %s\n", e.what());
        returnCode = EXIT_FAILURE;
    }

    cleanupSockets();
    return returnCode;
}
