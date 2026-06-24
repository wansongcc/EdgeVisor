#include "app.hpp"
#include "llm.hpp" // 包含 loadLlmNetWeightUneven, buildLlmNetUneven 等声明
#include "nn/nn-network.hpp"
#include "nn/nn-network-local.hpp" // [新增] 本地加载器
#include <cassert>
#include <cstring>
#include <sstream>
#include <numeric>
#include <cmath>
#include <vector>
#include <stdexcept>

#if defined(DLLAMA_VULKAN)
    #include "nn/nn-vulkan.hpp"
#endif
#if defined(DLLAMA_CUDA)
    #include "nn/nn-cuda.hpp"
#endif

// --- 辅助函数 ---

static NnFloatType parseFloatType(char *val) {
    if (std::strcmp(val, "f32") == 0) return F_32;
    if (std::strcmp(val, "f16") == 0) return F_16;
    if (std::strcmp(val, "q40") == 0) return F_Q40;
    if (std::strcmp(val, "q80") == 0) return F_Q80;
    throw std::runtime_error("Invalid float type: " + std::string(val));
}

static ChatTemplateType parseChatTemplateType(char *val) {
    if (std::strcmp(val, "llama2") == 0) return TEMPLATE_LLAMA2;
    if (std::strcmp(val, "llama3") == 0) return TEMPLATE_LLAMA3;
    if (std::strcmp(val, "deepSeek3") == 0) return TEMPLATE_DEEP_SEEK3;
    throw std::runtime_error("Invalid chat template type: " + std::string(val));
}

static AppCliArgs::Backend parseBackendType(char *val) {
    if (std::strcmp(val, "cpu") == 0) return AppCliArgs::BACKEND_CPU;
    if (std::strcmp(val, "vulkan") == 0) return AppCliArgs::BACKEND_VULKAN;
    if (std::strcmp(val, "cuda") == 0) return AppCliArgs::BACKEND_CUDA;
    throw std::runtime_error("Invalid backend: " + std::string(val));
}

const char *AppCliArgs::backendToString(AppCliArgs::Backend backend) {
    switch (backend) {
        case BACKEND_AUTO: return "auto";
        case BACKEND_CPU: return "cpu";
        case BACKEND_VULKAN: return "vulkan";
        case BACKEND_CUDA: return "cuda";
        default: return "unknown";
    }
}

// [新增] 解析逗号分隔的比例字符串
static std::vector<float> parseRatios(const char *ratiosStr, NnUint nNodes) {
    if (ratiosStr == nullptr) {
        throw std::invalid_argument("Ratios string cannot be empty");
    }
    std::vector<float> ratios;
    std::string s(ratiosStr);
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            try {
                ratios.push_back(std::stof(item));
            } catch (...) {
                throw std::invalid_argument(std::string("Invalid ratio value: ") + item);
            }
        }
    }
    if (ratios.size() != nNodes) {
        throw std::invalid_argument("Number of ratios (" + std::to_string(ratios.size()) + 
                                  ") must match number of nodes (" + std::to_string(nNodes) + ")");
    }
    return ratios;
}

// --- AppCliArgs 实现 ---

AppCliArgs AppCliArgs::parse(int argc, char* *argv, bool requireMode) {
    AppCliArgs args;
    args.info = true;
    args.help = false;
    args.backend = BACKEND_AUTO;
    args.backendStr = nullptr;
    args.mode = nullptr;
    args.nBatches = 32;
    args.nThreads = 1;
    args.modelPath = nullptr;
    args.tokenizerPath = nullptr;
    args.prompt = nullptr;
    args.syncType = F_32;
    args.nWorkers = 0;
    args.workerHosts = nullptr;
    args.workerPorts = nullptr;
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 0;
    args.seed = (unsigned long long)time(nullptr);
    args.chatTemplateType = TEMPLATE_UNKNOWN;
    args.maxSeqLen = 0;
    args.netTurbo = true;
    args.gpuIndex = -1;
    args.gpuSegmentFrom = -1;
    args.gpuSegmentTo = -1;
    args.ratiosStr = nullptr;

    int i = 1;
    if (requireMode && argc > 1) {
        args.mode = argv[1];
        i++;
    }
    for (int x = 0; x < argc; x++) {
        if ((std::strcmp(argv[x], "--usage") == 0) ||
            (std::strcmp(argv[x], "--help") == 0) ||
            (std::strcmp(argv[x], "-h") == 0)) {
            args.help = true;
            return args;
        }
    }
    for (; i + 1 < argc; i += 2) {
        char *name = argv[i];
        char *value = argv[i + 1];
        if (std::strcmp(name, "--model") == 0) {
            args.modelPath = value;
        } else if (std::strcmp(name, "--tokenizer") == 0) {
            args.tokenizerPath = value;
        } else if (std::strcmp(name, "--prompt") == 0) {
            args.prompt = value;
        } else if (std::strcmp(name, "--buffer-float-type") == 0) {
            args.syncType = parseFloatType(value);
        } else if (std::strcmp(name, "--backend") == 0) {
            args.backend = parseBackendType(value);
            args.backendStr = value;
        } else if (std::strcmp(name, "--workers") == 0) {
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;

            args.nWorkers = count;
            args.workerHosts = new char*[count];
            args.workerPorts = new NnUint[count];

            for (int s = 0; s < count; s++) {
                char *v = argv[i + 1 + s];
                char *separator = std::strstr(v, ":");
                if (separator == NULL) {
                    throw std::runtime_error("Invalid worker address: " + std::string(v));
                }
                int hostLen = separator - v;
                args.workerHosts[s] = new char[hostLen + 1];
                std::memcpy(args.workerHosts[s], v, hostLen);
                args.workerHosts[s][hostLen] = '\0';
                args.workerPorts[s] = std::atoi(separator + 1);
            }
            i += count - 1;
        } else if (std::strcmp(name, "--ratios") == 0) {
            // 修复之前的 bug：不使用 ++i，直接使用 value
            args.ratiosStr = value;
        } else if (std::strcmp(name, "--port") == 0) {
            args.port = atoi(value);
        } else if (std::strcmp(name, "--nthreads") == 0) {
            args.nThreads = atoi(value);
        } else if (std::strcmp(name, "--steps") == 0) {
            args.steps = atoi(value);
        } else if (std::strcmp(name, "--temperature") == 0) {
            args.temperature = atof(value);
        } else if (std::strcmp(name, "--topp") == 0) {
            args.topp = atof(value);
        } else if (std::strcmp(name, "--seed") == 0) {
            args.seed = atoll(value);
        } else if (std::strcmp(name, "--chat-template") == 0) {
            args.chatTemplateType = parseChatTemplateType(value);
        } else if (std::strcmp(name, "--max-seq-len") == 0) {
            args.maxSeqLen = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--gpu-index") == 0) {
            args.gpuIndex = atoi(value);
        } else if (std::strcmp(name, "--gpu-segments") == 0) {
            char *separator = std::strstr(value, ":");
            if (separator == NULL)
                throw std::runtime_error("GPU segments expected in the format <from>:<to>");
            args.gpuSegmentFrom = atoi(value);
            args.gpuSegmentTo = atoi(separator + 1);
        } else if (std::strcmp(name, "--net-turbo") == 0) {
            args.netTurbo = atoi(value) == 1;
        } else {
            throw std::runtime_error("Unknown option: " + std::string(name));
        }
    }

    if (args.nThreads < 1)
        throw std::runtime_error("Number of threads must be at least 1");
    if (args.backend == BACKEND_CPU && args.gpuIndex >= 0)
        throw std::runtime_error("--backend cpu conflicts with --gpu-index");
    if (args.backend == BACKEND_AUTO)
        args.backend = (args.gpuIndex >= 0) ? BACKEND_VULKAN : BACKEND_CPU;
    else if ((args.backend == BACKEND_VULKAN || args.backend == BACKEND_CUDA) && args.gpuIndex < 0)
        args.gpuIndex = 0;
#if !defined(DLLAMA_VULKAN)
    if (args.backend == BACKEND_VULKAN)
        throw std::runtime_error("--backend vulkan requested, but this build was not compiled with DLLAMA_VULKAN=1");
#endif
#if !defined(DLLAMA_CUDA)
    if (args.backend == BACKEND_CUDA)
        throw std::runtime_error("--backend cuda requested, but this build was not compiled with DLLAMA_CUDA=1");
#endif
    return args;
}

AppCliArgs::~AppCliArgs() {
    if (workerHosts != nullptr) {
        for (NnUint i = 0; i < nWorkers; i++)
            delete[] workerHosts[i];
        delete[] workerHosts;
    }
    if (workerPorts != nullptr)
        delete[] workerPorts;
}

// --- Device Resolution ---

static std::vector<NnExecutorDevice> resolveDevices(AppCliArgs *args, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *plan = nullptr) {
    std::vector<NnExecutorDevice> devices;

    if (args->backend == AppCliArgs::BACKEND_CPU) {
        devices.push_back(NnExecutorDevice(new NnCpuDevice(netConfig, nodeConfig, netExecution, plan), -1, -1));
        return devices;
    }

    if (args->backend == AppCliArgs::BACKEND_VULKAN) {
#if defined(DLLAMA_VULKAN)
        devices.push_back(NnExecutorDevice(
            new NnVulkanDevice(args->gpuIndex, netConfig, nodeConfig, netExecution),
            args->gpuSegmentFrom,
            args->gpuSegmentTo
        ));
#else
        throw std::runtime_error("--backend vulkan requested, but this build was not compiled with DLLAMA_VULKAN=1");
#endif
    } else if (args->backend == AppCliArgs::BACKEND_CUDA) {
#if defined(DLLAMA_CUDA)
        devices.push_back(NnExecutorDevice(
            new NnCudaDevice(args->gpuIndex, netConfig, nodeConfig, netExecution, plan),
            args->gpuSegmentFrom,
            args->gpuSegmentTo
        ));
#else
        throw std::runtime_error("--backend cuda requested, but this build was not compiled with DLLAMA_CUDA=1");
#endif
    } else {
        throw std::runtime_error("Internal error: unresolved backend " + std::string(AppCliArgs::backendToString(args->backend)));
    }

    if (args->gpuSegmentFrom >= 0 && args->gpuSegmentTo >= 0) {
        // 传入 plan 以支持非均匀切分时的稳定性检查和指针计算
        devices.push_back(NnExecutorDevice(new NnCpuDevice(netConfig, nodeConfig, netExecution, plan), -1, -1));
    }
    return devices;
}

// --- Inference Implementations (Root & Worker) ---

RootLlmInference::RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
    this->network = network; 
}

void RootLlmInference::setBatchSize(NnUint batchSize) {
    execution->setBatchSize(batchSize);
    controlPacket.batchSize = batchSize;
}

void RootLlmInference::setPosition(NnUint position) {
    assert(position >= 0);
    assert(position + execution->batchSize - 1 < header->seqLen);

    controlPacket.position = position;
    for (NnUint i = 0; i < execution->batchSize; i++)
        positionPipe[i] = (float)(position + i);
}

void RootLlmInference::setToken(NnUint batchIndex, NnUint token) {
    assert(batchIndex >= 0 && batchIndex < execution->batchSize);
    tokenPipe[batchIndex] = (float)token;
}

void RootLlmInference::forward() {
    if (network != nullptr) 
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
    executor->forward();
}

void RootLlmInference::finish() {
    if (network != nullptr) {
        controlPacket.batchSize = 0;
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
    }
}

WorkerLlmInference::WorkerLlmInference(NnNetExecution *execution, NnNetwork *network) {
    this->isFinished = false;
    this->execution = execution;
    this->network = network;
    this->positionPipe = (float *)execution->pipes[0];
}

bool WorkerLlmInference::tryReadControlPacket() {
    const unsigned long maxAttempts = 10000;
    if (!network->tryReadWithMaxAttempts(ROOT_SOCKET_INDEX, &controlPacket, sizeof(LlmControlPacket), maxAttempts))
        return false;
    if (controlPacket.batchSize == 0) {
        printf("🛑 Stop signal\n");
        isFinished = true;
        return true;
    }
    for (NnUint i = 0; i < controlPacket.batchSize; i++)
        positionPipe[i] = (float)(controlPacket.position + i);
    execution->setBatchSize(controlPacket.batchSize);
    return true;
}

// --- Main Application Entry Points ---

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context)) {
    NnUint nNodes = args->nWorkers + 1;

    LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
    if (nNodes > header.nKvHeads)
        // TODO: https://github.com/b4rtaz/distributed-llama/issues/70
        throw std::runtime_error("This version does not support more nodes than the number of KV heads in the model");
    if (header.weightType == F_Q40 && header.syncType != F_Q80)
        throw std::runtime_error("This version supports only Q40 weights with Q80 sync type");

    Tokenizer tokenizer(args->tokenizerPath);
    if (args->info && tokenizer.vocabSize != header.vocabSize)
        printf("Tokenizer vocab size (%d) does not match the model vocab size (%d)\n", tokenizer.vocabSize, header.vocabSize);

    Sampler sampler(tokenizer.vocabSize, args->temperature, args->topp, args->seed);

    // --- 1. Build Network ---
    LlmNet net;
    std::vector<float> ratios;
    // 用于保存 plan 指针，以便传递给 resolveDevices
    std::unique_ptr<NnUnevenPartitionPlan> planPtr; 

    if (args->ratiosStr != nullptr) {
        // [非均匀模式]
        printf("🚀 Mode: Uneven Partitioning (%s)\n", args->ratiosStr);
        ratios = parseRatios(args->ratiosStr, nNodes);
        net = buildLlmNetUneven(&header, nNodes, args->nBatches, ratios);
    } else {
        // [均匀模式]
        printf("🚀 Mode: Uniform Partitioning\n");
        net = buildLlmNet(&header, nNodes, args->nBatches);
    }

    std::unique_ptr<LlmNet, void(*)(LlmNet *)> netPtr(&net, releaseLlmNet);
    NnNodeConfig *rootNodeConfig = &net.nodeConfigs[0];

    if (args->info) {
        tokenizer.printHeader();
        printLlmHeader(&header);
        printNodeRequiredMemory(&net.netConfig, rootNodeConfig);
    }

    NnNetExecution execution(args->nThreads, &net.netConfig);
    std::unique_ptr<NnNodeSynchronizer> synchronizer(nullptr);
    std::unique_ptr<NnNetwork> networkPtr(nullptr);
    NnNetwork *network = nullptr;

    if (nNodes == 1) {
        synchronizer.reset(new NnFakeNodeSynchronizer());
    } else {
        networkPtr = NnNetwork::connect(args->nWorkers, args->workerHosts, args->workerPorts);
        network = networkPtr.get();
        synchronizer.reset(new NnNetworkNodeSynchronizer(network, &execution, &net.netConfig, rootNodeConfig));

        NnRootConfigWriter configWriter(network);
        configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
    }

    // --- 2. Initialize Executor ---
    // 如果是非均匀模式，我们需要创建 Plan 并传递给 resolveDevices
    if (args->ratiosStr != nullptr) {
        NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;
        planPtr.reset(new NnUnevenPartitionPlan(
            createPartitionPlan(nNodes, ratios, header.nHeads, header.nKvHeads, header.vocabSize, ffDim)
        ));
    }
    
    std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution, planPtr.get());
    NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), args->benchmark);

    // --- 3. Load Weights ---
    if (args->ratiosStr != nullptr) {
        // [非均匀 + 本地加载]
        printf("🚀 Local Loading Mode (Root): Loading weights locally...\n");
        
        // 创建本地加载器 (Node 0)
        NnLocalWeightLoader localLoader(&executor, 0); 
        
        // 加载
        loadLlmNetWeightUneven(args->modelPath, &net, &localLoader, planPtr.get());
        
    } else {
        // [均匀 + 网络分发] (Legacy)
        NnRootWeightLoader weightLoader(&executor, network, nNodes);
        loadLlmNetWeight(args->modelPath, &net, &weightLoader);
    }

    // --- 4. Inference ---
    RootLlmInference inference(&net, &execution, &executor, network);

    if (network != nullptr) {
        network->resetStats();
        if (args->netTurbo) {
            network->setTurbo(true);
            printf("🚁 Network is in non-blocking mode\n");
        }
    }

    AppInferenceContext context;
    context.args = args;
    context.header = &header;
    context.inference = &inference;
    context.sampler = &sampler;
    context.tokenizer = &tokenizer;
    context.network = network;
    context.executor = &executor;

    handler(&context);

    inference.finish();
    
    // Plan 由 unique_ptr 自动释放
}

void runWorkerApp(AppCliArgs *args) {
    while (true) {
        std::unique_ptr<NnNetwork> networkPtr = NnNetwork::serve(args->port);
        NnNetwork *network = networkPtr.get();

        NnWorkerConfigReader configReader(network);
        NnNetConfig netConfig = configReader.readNet();
        NnNodeConfig nodeConfig = configReader.readNode();
        std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
        std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

        printNodeRequiredMemory(&netConfig, &nodeConfig);

        NnNetExecution execution(args->nThreads, &netConfig);
        
        // 准备 Plan (如果是本地加载模式) 用于 resolveDevices
        std::unique_ptr<NnUnevenPartitionPlan> planPtr;
        if (args->ratiosStr != nullptr && args->modelPath != nullptr) {
             LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
             std::vector<float> ratios = parseRatios(args->ratiosStr, netConfig.nNodes);
             NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;
             
             planPtr.reset(new NnUnevenPartitionPlan(
                 createPartitionPlan(netConfig.nNodes, ratios, header.nHeads, header.nKvHeads, header.vocabSize, ffDim)
             ));
        }

        std::vector<NnExecutorDevice> devices = resolveDevices(args, &netConfig, &nodeConfig, &execution, planPtr.get());
        NnNetworkNodeSynchronizer synchronizer(network, &execution, &netConfig, &nodeConfig);
        NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);

        // --- Weight Loading Logic ---
        if (args->ratiosStr != nullptr && args->modelPath != nullptr) {
            // [本地加载]
            printf("🚀 Worker %d: Local Loading Mode from %s\n", nodeConfig.nodeIndex, args->modelPath);
            
            // 重新加载头信息用于构建临时 Net
            LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
            std::vector<float> ratios = parseRatios(args->ratiosStr, netConfig.nNodes);
            
            // 修正辅助维度
            if (header.headDim == 0 && header.nHeads > 0) header.headDim = header.dim / header.nHeads;
            header.qDim = header.nHeads * header.headDim;
            header.kvDim = header.nKvHeads * header.headDim;

            // 构建临时 Net
            LlmNet tempNet = buildLlmNetUneven(&header, netConfig.nNodes, 1, ratios);

            // 执行本地加载
            NnLocalWeightLoader localLoader(&executor, nodeConfig.nodeIndex);
            loadLlmNetWeightUneven(args->modelPath, &tempNet, &localLoader, planPtr.get());

            releaseLlmNet(&tempNet);
            printf("✅ Worker %d: Weights loaded locally.\n", nodeConfig.nodeIndex);

        } else {
            // [网络加载] (Legacy)
            printf("📡 Worker %d: Waiting for weights from Root...\n", nodeConfig.nodeIndex);
            NnWorkerWeightReader weightReader(&executor, network);
            weightReader.read();
        }

        WorkerLlmInference inference(&execution, network);
        bool isFirstAttempt = true;
        bool isTurboEnabled = false;
        clock_t startTime;
        while (true) {
            try {
                if (isFirstAttempt)
                    startTime = clock();

                if (!inference.tryReadControlPacket()) {
                    if (isTurboEnabled && !isFirstAttempt && clock() - startTime > CLOCKS_PER_SEC) {
                        network->setTurbo(false);
                        isTurboEnabled = false;
                        printf("🚁 Network is in blocking mode\n");
                    }
                    isFirstAttempt = false;
                    continue;
                }
                if (inference.isFinished)
                    break;

                if (args->netTurbo && !isTurboEnabled) {
                    network->setTurbo(true);
                    isTurboEnabled = true;
                    printf("🚁 Network is in non-blocking mode\n");
                }
                executor.forward();
                isFirstAttempt = true;
            } catch (const NnTransferSocketException &e) {
                printf("🚨 Network error: %s\n", e.what());
                break;
            } catch (const NnExecutorException &e) {
                printf("🚨 Inference error: %s\n", e.what());
                break;
            }
        }
        
        // Plan 由 unique_ptr 自动释放
    }
}
