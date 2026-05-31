#include "dynamic/kv_collector.hpp"

#include "app.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

static int kvcParseEnvInt(const char *name, int defaultValue) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return defaultValue;
    char *end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return defaultValue;
    return (int)x;
}

std::unique_ptr<RootKvCollector> RootKvCollector::start(RootLlmInference *inference) {
#ifdef _WIN32
    (void)inference;
    return nullptr;
#else
    if (inference == nullptr) return nullptr;

    std::unique_ptr<RootKvCollector> ctrl(new RootKvCollector(inference));
    RootKvCollector *c = ctrl.get();
    ctrl->worker_ = std::thread([c]() { c->run(); });
    const int pos = inference->getAsyncKvCollectPos();
    const int layer = inference->getAsyncKvCollectLayer();
    if (pos >= 0 && layer >= 0) {
        std::fprintf(stderr, "[kv-collector] enabled, thread started (target pos=%d, layer=%d)\n", pos, layer);
    } else {
        std::fprintf(stderr, "[kv-collector] enabled, waiting for migration arming from runtime/UDS\n");
    }
    return ctrl;
#endif
}

RootKvCollector::RootKvCollector(RootLlmInference *inference)
    : inference_(inference) {}

RootKvCollector::~RootKvCollector() {
    stop_.store(true);
    if (worker_.joinable()) {
        worker_.join();
    }
}

void RootKvCollector::run() {
#ifdef _WIN32
    return;
#else
    int collectLayer = -1;
    int collectPos = -1;

    const int pollMs = std::max(1, kvcParseEnvInt("DLLAMA_ASYNC_KV_COLLECT_POLL_MS", 20));
    NnUint kvDim = 0u;
    NnUint seqLen = 0u;
    bool collectorReady = false;
    bool rowDumped = false;

    while (!stop_.load()) {
        try {
            if (inference_ == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
                continue;
            }

            if (collectPos < 0) {
                collectPos = inference_->getAsyncKvCollectPos();
            }
            if (collectLayer < 0) {
                collectLayer = inference_->getAsyncKvCollectLayer();
            }
            if (collectPos < 0 || collectLayer < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
                continue;
            }

            if (!collectorReady && inference_->hasAsyncKvCollector()) {
                kvDim = inference_->getKvDim();
                seqLen = inference_->getSeqLen();
                if (kvDim > 0u && seqLen > 0u && (NnUint)collectPos < seqLen) {
                    collectorReady = true;
                    rowDumped = false;
                    std::fprintf(stderr, "[kv-collector] ready layer=%d pos=%d seqLen=%u kvDim=%u\n",
                        collectLayer, collectPos, (unsigned)seqLen, (unsigned)kvDim);
                }
            }

            if (collectorReady && !rowDumped) {
                RootKvAggRowPacket packet;
                int drained = 0;
                while (inference_->tryPopAsyncKvRow(packet)) {
                    if (packet.layerIndex != (NnUint)collectLayer) continue;
                    if (packet.position != (NnUint)collectPos) continue;
                    if (packet.kRow.size() != kvDim || packet.vRow.size() != kvDim) continue;
                    drained += 1;

                    char kPath[256];
                    char vPath[256];
                    std::snprintf(kPath, sizeof(kPath), "/tmp/dllama_async_layer%u_pos%u_root_k.bin", (unsigned)collectLayer, (unsigned)collectPos);
                    std::snprintf(vPath, sizeof(vPath), "/tmp/dllama_async_layer%u_pos%u_root_v.bin", (unsigned)collectLayer, (unsigned)collectPos);

                    auto dumpOne = [&](const char *path, const std::vector<float> &row) {
                        std::unique_ptr<FILE, int(*)(FILE *)> f(std::fopen(path, "wb"), std::fclose);
                        if (!f) return false;
                        const uint32_t magic = 0x4b565250u; // 'PRVK'
                        const uint32_t version = 1u;
                        const uint32_t layer = (uint32_t)collectLayer;
                        const uint32_t pos = (uint32_t)collectPos;
                        const uint32_t seq = (uint32_t)seqLen;
                        const uint32_t dim = (uint32_t)kvDim;
                        std::fwrite(&magic, sizeof(magic), 1, f.get());
                        std::fwrite(&version, sizeof(version), 1, f.get());
                        std::fwrite(&layer, sizeof(layer), 1, f.get());
                        std::fwrite(&pos, sizeof(pos), 1, f.get());
                        std::fwrite(&seq, sizeof(seq), 1, f.get());
                        std::fwrite(&dim, sizeof(dim), 1, f.get());
                        std::fwrite(row.data(), sizeof(float), row.size(), f.get());
                        return true;
                    };

                    const bool kOk = dumpOne(kPath, packet.kRow);
                    const bool vOk = dumpOne(vPath, packet.vRow);
                    const RootLlmInference::KvTransferSubmitStatus submitStatus =
                        inference_->submitBoundaryKvTransferDetailed((NnUint)collectLayer, (NnUint)collectPos, packet.kRow, packet.vRow);
                    const bool enqueued = (submitStatus == RootLlmInference::KV_TRANSFER_SUBMIT_OK);
                    std::fprintf(stderr, "[kv-collector] layer=%d pos=%d row dumped k=%s(%s) v=%s(%s) drained=%d\n",
                        collectLayer,
                        collectPos,
                        kPath,
                        kOk ? "ok" : "fail",
                        vPath,
                        vOk ? "ok" : "fail",
                        drained);
                    if (!enqueued) {
                        const char *reason = "unknown";
                        switch (submitStatus) {
                            case RootLlmInference::KV_TRANSFER_SUBMIT_NO_NETWORK: reason = "no-network"; break;
                            case RootLlmInference::KV_TRANSFER_SUBMIT_NO_TARGET_STAGE: reason = "no-target-stage"; break;
                            case RootLlmInference::KV_TRANSFER_SUBMIT_LAYER_NOT_IN_LIST: reason = "layer-not-in-migration-list"; break;
                            case RootLlmInference::KV_TRANSFER_SUBMIT_INVALID_ROW: reason = "invalid-row"; break;
                            case RootLlmInference::KV_TRANSFER_SUBMIT_WAITING_ACK: reason = "waiting-ack"; break;
                            case RootLlmInference::KV_TRANSFER_SUBMIT_DUP_LAYER_PENDING: reason = "dup-layer-pending"; break;
                            case RootLlmInference::KV_TRANSFER_SUBMIT_OK: reason = "ok"; break;
                            default: reason = "unknown"; break;
                        }
                        std::fprintf(stderr,
                            "[kv-collector] warn: transfer not queued for layer=%d pos=%d reason=%s\n",
                            collectLayer,
                            collectPos,
                            reason);
                    }
                    rowDumped = true;
                    break;
                }
            }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "[kv-collector] exception: %s\n", e.what());
        } catch (...) {
            std::fprintf(stderr, "[kv-collector] unknown exception\n");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
    }
#endif
}
