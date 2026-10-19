#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <csignal>
#include <thread>
#include <chrono>
#include <cmath>
#include "api-token-ledger.hpp"
#include "token_timing.hpp"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

#include "tokenizer.hpp"
#include "app.hpp"
#include "json.hpp"
#include "api-types.hpp"
#include "nn/nn-network.hpp"

typedef unsigned int pos_t;

using json = nlohmann::json;

enum class HttpMethod {
    METHOD_GET = 0,
    METHOD_POST = 1,
    METHOD_PUT = 2,
    METHOD_DELETE = 3,
    METHOD_OPTIONS = 4,
    METHOD_UNKNOWN = 5
};

class HttpRequest {
public:
    static HttpRequest read(int serverSocket) {
        HttpRequest req(serverSocket);

        std::vector<char> httpRequest = req.readHttpRequest();
        // Parse the HTTP request
        std::string data = std::string(httpRequest.begin(), httpRequest.end());

        // Split request into lines
        std::istringstream iss(data);
        std::string line;
        std::getline(iss, line);

        // Parse request line
        std::istringstream lineStream(line);
        std::string methodStr, path;
        lineStream >> methodStr >> path;
        req.method = parseMethod(methodStr);
        req.path = path;

        // Parse headers
        while (std::getline(iss, line) && line != "\r") {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 2); // Skip ': ' after key
                // Trim whitespace and non-printable characters from header value
                value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
                    return std::isspace(c) || !std::isprint(c);
                }), value.end());
                req.headers[key] = value;
            }
        }

        // Parse body
        std::getline(iss, req.body, '\0');

        if (req.body.size() > 0) {
            // printf("body: %s\n", req.body.c_str());
            req.parsedJson = json::parse(req.body, nullptr, false);
        }
        return req;
    }

    static HttpMethod parseMethod(const std::string& method) {
        if (method == "GET") return HttpMethod::METHOD_GET;
        if (method == "POST") return HttpMethod::METHOD_POST;
        if (method == "PUT") return HttpMethod::METHOD_PUT;
        if (method == "DELETE") return HttpMethod::METHOD_DELETE;
        if (method == "OPTIONS") return HttpMethod::METHOD_OPTIONS;
        return HttpMethod::METHOD_UNKNOWN;
    }

private:
    int serverSocket;
    void writeRaw(const std::string &data) {
        if (writeFailed) return;
        try {
            writeSocket(serverSocket, data.data(), data.size());
            responseStarted = true;
        } catch (const std::exception &error) {
            // The client may disconnect at any point. A broken HTTP peer must
            // not be confused with a broken inference worker.
            writeFailed = true;
            std::fprintf(stderr, "API client write failed: %s\n", error.what());
        }
    }
public:
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    json parsedJson;
    HttpMethod method;
    bool responseStarted = false;
    bool streaming = false;
    bool writeFailed = false;

    HttpRequest(int serverSocket) {
        this->serverSocket = serverSocket;
    }

    std::vector<char> readHttpRequest() {
        std::string httpRequest;
        char buffer[1024 * 64];
        ssize_t bytesRead;

        // First, read all headers
        std::string headerData;
        size_t headerEnd;
        bool headerDone = false;
        std::string extraReadPastHeader;
        while (!headerDone) {
            bytesRead = recv(serverSocket, buffer, sizeof(buffer) - 1, 0);
            if (bytesRead <= 0) {
                throw std::runtime_error("Error while reading headers from socket");
            }
            buffer[bytesRead] = '\0';
            headerData.append(buffer);

            // Check for end of headers (http header says "\r\n\r\n")
            headerEnd = headerData.find("\r\n\r\n");
            if (headerEnd != std::string::npos) {
                headerDone = true;
                if (headerEnd < headerData.size()-4) {
                    // We read something past the header
                    extraReadPastHeader = headerData.substr(headerEnd+4);
                }
            }
        }

        httpRequest.append(headerData);

        // Next, find Content-Length header for body length
        std::istringstream headerStream(headerData);
        std::string line;
        ssize_t contentLength = 0;
        while (std::getline(headerStream, line) && line != "\r") {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 2); // Skip ': ' after key
                if (key == "Content-Length") {
                    try {
                      contentLength = std::stoi(value);  // stoi ignores any whitespace
                    } catch (const std::invalid_argument& e) {
                      throw std::runtime_error("Bad Content-Length header - not a number");
                    }
                    break;
                }
            }
        }

        // Now read the full content body
        if (contentLength > 0) {
            // If we read any extra past the header before, read that much less now
            // But first, sanity check to make sure Content-Length isn't lying and there is actually more
            if (extraReadPastHeader.size() > static_cast<size_t>(contentLength)) {
                throw std::runtime_error("Received more body data than Content-Length header said");
            }
            contentLength -= extraReadPastHeader.size();

            std::vector<char> body(contentLength);
            ssize_t totalRead = 0;
            while (totalRead < contentLength) {
                bytesRead = recv(serverSocket, body.data() + totalRead, contentLength - totalRead, 0);
                if (bytesRead <= 0) {
                    throw std::runtime_error("Error while reading body from socket");
                }
                totalRead += bytesRead;
            }
            if (body.size() > 0) {
              httpRequest.append(body.data(), contentLength);
            }
        }

        return std::vector<char>(httpRequest.begin(), httpRequest.end());
    }

    std::string getMethod() {
        if (method == HttpMethod::METHOD_GET) return "GET";
        if (method == HttpMethod::METHOD_POST) return "POST";
        if (method == HttpMethod::METHOD_PUT) return "PUT";
        if (method == HttpMethod::METHOD_DELETE) return "DELETE";
        if (method == HttpMethod::METHOD_OPTIONS) return "OPTIONS";
        return "UNKNOWN";
    }
 
    void writeCors() {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 204 No Content\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Access-Control-Allow-Methods: GET, POST, PUT, DELETE\r\n"
            << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            << "Connection: close\r\n"
            << "\r\n";
        writeRaw(buffer.str());
    }

    void writeNotFound() {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 404 Not Found\r\n"
            << "Connection: close\r\n"
            << "Content-Length: 9\r\n"
            << "\r\n"
            << "Not Found";
        writeRaw(buffer.str());
    }

    static const char *reasonPhrase(int status) {
        switch (status) {
            case 200: return "OK";
            case 204: return "No Content";
            case 400: return "Bad Request";
            case 404: return "Not Found";
            case 500: return "Internal Server Error";
            case 503: return "Service Unavailable";
            default: return "Response";
        }
    }

    void writeJson(std::string json, int status = 200) {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 " << status << " " << reasonPhrase(status) << "\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Content-Type: application/json; charset=utf-8\r\n"
            << "Connection: close\r\n"
            << "Content-Length: " << json.length() << "\r\n\r\n" << json;
        writeRaw(buffer.str());
    }

    void writeStreamStartChunk() {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 200 OK\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Content-Type: text/event-stream; charset=utf-8\r\n"
            << "Connection: close\r\n"
            << "Transfer-Encoding: chunked\r\n\r\n";
        writeRaw(buffer.str());
        streaming = !writeFailed;
    }

    void writeStreamChunk(const std::string data) {
        if (writeFailed || !streaming) return;
        std::ostringstream buffer;
        buffer << std::hex << data.size() << "\r\n" << data << "\r\n";
        writeRaw(buffer.str());
    }

    void writeStreamEndChunk() {
        if (writeFailed || !streaming) return;
        const char *endChunk = "0000\r\n\r\n";
        writeRaw(endChunk);
        streaming = false;
    }

    // Report a runtime failure without assuming that no bytes were sent yet.
    // A non-stream request can still use a proper HTTP status; once SSE
    // headers are on the wire the only safe option is an error event followed
    // by a terminated chunked body.
    void writeFailure(int status, const std::string &type, const std::string &code, const std::string &message) {
        if (writeFailed) return;
        json body = {{"error", {{"message", message}, {"type", type}, {"code", code}}}};
        if (!responseStarted) {
            writeJson(body.dump(), status);
            return;
        }
        if (streaming) {
            writeStreamChunk("data: " + body.dump() + "\r\n\r\n");
            writeStreamChunk("data: [DONE]");
            writeStreamEndChunk();
        }
    }
};

struct Route {
    std::string path;
    HttpMethod method;
    std::function<void(HttpRequest&)> handler;
};

class Router {
public:
    static void resolve(HttpRequest& request, std::vector<Route>& routes) {
        if (request.method == HttpMethod::METHOD_OPTIONS) {
            request.writeCors();
            return;
        }
        for (const auto& route : routes) {
            if (request.method == route.method && request.path == route.path) {
                route.handler(request);
                return;
            }
        }
        request.writeNotFound();
    }
};

void writeChatCompletionChunk(HttpRequest &request, const std::string &delta, const bool stop, const json *metadata = nullptr, const char *finishReason = "stop"){
    ChunkChoice choice;
    if (stop) {
        choice.finish_reason = finishReason;
    } else {
        choice.delta = ChatMessageDelta("assistant", delta);
    }
    ChatCompletionChunk chunk = ChatCompletionChunk(choice);

    std::ostringstream buffer;
    json payload = chunk;
    if (metadata != nullptr) payload["edgevisor"] = *metadata;
    buffer << "data: " << payload.dump() << "\r\n\r\n";
    request.writeStreamChunk(buffer.str());

    if (stop) {
        request.writeStreamChunk("data: [DONE]");
        request.writeStreamEndChunk();
    }
}

class ApiServer {
    RootLlmInference *inference;
    Tokenizer *tokenizer;
    Sampler *sampler;
    NnNetwork *network;
    AppCliArgs *args;
    LlmHeader *header;
    ChatTemplateGenerator *templateGenerator;
    ApiTokenLedger ledger;
    uint64_t generationId = 0;

    InferenceParams parseRequest(HttpRequest &request) {
        const json &body = request.parsedJson;
        if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
            throw std::invalid_argument("messages must be a nonempty array");
        for (const auto &message : body["messages"]) {
            if (!message.is_object() || !message.contains("role") || !message["role"].is_string() ||
                    !message.contains("content") || !message["content"].is_string())
                throw std::invalid_argument("Each message requires string role and content");
            const std::string role = message["role"].get<std::string>();
            const bool toolRoleSupported = args->chatTemplateType == TEMPLATE_LLAMA3 || args->chatTemplateType == TEMPLATE_CHATML ||
                (args->chatTemplateType == TEMPLATE_UNKNOWN && tokenizer->chatTemplate != nullptr &&
                    (std::strstr(tokenizer->chatTemplate, "<|start_header_id|>") != nullptr ||
                     std::strstr(tokenizer->chatTemplate, "<|im_start|>") != nullptr));
            if (role != "system" && role != "user" && role != "assistant" && !(role == "tool" && toolRoleSupported))
                throw std::invalid_argument("Unsupported message role for this chat template");
        }
        InferenceParams params;
        params.messages = parseChatMessages(request.parsedJson["messages"]);
        params.temperature = body.value("temperature", args->temperature);
        params.top_p = body.value("top_p", args->topp);
        if (body.contains("seed") && (!body["seed"].is_number_integer() ||
                (!body["seed"].is_number_unsigned() && body["seed"].get<int64_t>() < 0)))
            throw std::invalid_argument("seed must be a nonnegative integer");
        params.seed = body.value("seed", args->seed);
        params.stream = body.value("stream", false);
        if (body.contains("max_tokens")) {
            const auto &limit = body["max_tokens"];
            if (!limit.is_number_integer() ||
                    (!limit.is_number_unsigned() && limit.get<int64_t>() <= 0) ||
                    limit.get<uint64_t>() == 0 || limit.get<uint64_t>() > header->seqLen)
                throw std::invalid_argument("max_tokens must be an integer within context capacity");
        }
        params.max_tokens = body.value("max_tokens", -1);
        if (!std::isfinite(params.temperature) || params.temperature < 0 ||
                !std::isfinite(params.top_p) || params.top_p <= 0 || params.top_p > 1)
            throw std::invalid_argument("temperature must be nonnegative and top_p must be in (0,1]");
        if (body.contains("stop") && !body["stop"].is_null()) {
            if (body["stop"].is_string()) params.stop.push_back(body["stop"].get<std::string>());
            else params.stop = body["stop"].get<std::vector<std::string>>();
        }
        if (params.stop.size() > 16) throw std::invalid_argument("At most 16 stop strings are supported");
        for (const auto &stop : params.stop)
            if (stop.empty() || stop.size() > 4096) throw std::invalid_argument("Invalid stop string length");
        return params;
    }

    void completePrepared(HttpRequest &request, const InferenceParams &params,
            const std::string &sessionId, const std::vector<int> &promptTokens,
            const std::string &publicPrompt) {
        const auto start = ledger.begin(sessionId, promptTokens);
        const uint64_t generation = ++generationId;
        struct Guard {
            ApiTokenLedger &ledger;
            bool done;
            explicit Guard(ApiTokenLedger &value) : ledger(value), done(false) {}
            ~Guard() { if (!done) ledger.invalidate(); }
        } guard{ledger};
        ExperimentTraceSpan generationTrace("api_generation", {{"session_id", sessionId},
            {"lineage_id", start.lineage}, {"generation_id", generation},
            {"cached_input_tokens", start.cachedTokens}, {"reset_reason", start.resetReason}});
        sampler->setParams(params.temperature, params.top_p, params.seed);
        inference->configureGenerationSampling(params.temperature, params.top_p, params.seed,
                                               generation, !start.resetReason.empty());
        const NnUint promptEnd = (NnUint)promptTokens.size() - 1u;
        const NnUint maxPred = params.max_tokens > 0 ? promptEnd + (NnUint)params.max_tokens : header->seqLen;
        NnUint pos = (NnUint)start.cachedTokens;
        std::vector<int> consumed(promptTokens.begin(), promptTokens.begin() + promptEnd);
        TokenizerChatStops stops(tokenizer);
        std::vector<int> stopIds = tokenizer->eosTokenIds;
        std::vector<const char *> stopPieces(stops.stops, stops.stops + stops.nStops);
        int maxStopLength = stops.maxStopLength;
        for (const auto &stop : params.stop) {
            stopIds.push_back(-1); stopPieces.push_back(stop.c_str());
            maxStopLength = std::max(maxStopLength, (int)stop.size());
        }
        EosDetector detector(stopIds.size(), stopIds.data(), stopPieces.data(), maxStopLength, maxStopLength);
        std::string buffer = publicPrompt;
        if (params.stream) {
            request.writeStreamStartChunk();
            if (!publicPrompt.empty()) writeChatCompletionChunk(request, publicPrompt, false);
        }
        {
            ExperimentTraceSpan prefillTrace("prefill", {{"generation_id", generation},
                {"prompt_tokens", promptTokens.size()}, {"start_position", pos}});
            inference->setSkipLogits(true);
            while (pos < promptEnd) {
                const NnUint batch = std::min<NnUint>(args->nBatches, promptEnd - pos);
                inference->setBatchSize(batch); inference->setPosition(pos);
                for (NnUint index = 0; index < batch; ++index) inference->setToken(index, promptTokens[pos + index]);
                inference->forward(); pos += batch;
            }
            inference->setSkipLogits(false);
        }
        int token = promptTokens.back();
        bool stopped = false;
        EosDetectorType eosType = NOT_EOS;
        inference->setBatchSize(1); tokenizer->resetDecoder();
        {
            ExperimentTraceSpan decodeTrace("decode", {{"generation_id", generation}});
            for (; pos < maxPred; ++pos) {
                ExperimentTraceSpan tokenTrace("token", {{"generation_id", generation}, {"position", pos}});
                inference->setPosition(pos); inference->setToken(0, token);
                inference->forward();
                consumed.push_back(token);
                NnUint tailToken = 0u;
                token = args->lastStageSampling && network != nullptr &&
                        inference->tryReceiveLastStageSampledToken(tailToken, nullptr)
                    ? (int)tailToken : sampler->sample(inference->logitsPipe);
                eosType = detector.append(token, tokenizer->decode(token));
                if (eosType == NOT_EOS || eosType == EOS) {
                    const char *delta = detector.getDelta();
                    if (delta != nullptr) {
                        buffer += delta;
                        if (params.stream) writeChatCompletionChunk(request, delta, false);
                    }
                    detector.reset();
                }
                experimentTracePoint("token_emitted", {{"generation_id", generation}, {"position", pos}, {"token", token}});
                if (eosType == EOS) { stopped = true; ++pos; break; }
            }
            if (!stopped && eosType == MAYBE_EOS) {
                const char *pending = detector.getDelta();
                if (pending != nullptr) {
                    buffer += pending;
                    if (params.stream) writeChatCompletionChunk(request, pending, false);
                }
            }
        }
        ledger.finish(consumed);
        const json metadata = {{"runtime_context_key", experimentRuntimeContextKey()}, {"session_id", sessionId}, {"lineage_id", start.lineage},
            {"generation_id", generation}, {"temperature", params.temperature}, {"top_p", params.top_p},
            {"seed", params.seed}, {"cached_input_tokens", start.cachedTokens},
            {"start_position", start.cachedTokens}, {"new_prefill_tokens", promptEnd - start.cachedTokens},
            {"kv_frontier_count", consumed.size()}, {"pending_input_token", token},
            {"reset_reason", start.resetReason}, {"status", "completed"}};
        experimentTracePoint("api_generation_complete", metadata);
        generationTrace.end(); requireExperimentTraceHealthy();
        guard.done = true;
        if (params.stream) writeChatCompletionChunk(request, "", true, &metadata, stopped ? "stop" : "length");
        else {
            ChatMessage message("assistant", buffer);
            Choice choice(message); choice.finish_reason = stopped ? "stop" : "length";
            const int count = pos - promptEnd;
            ChatCompletion completion(choice, ChatUsage(promptTokens.size(), count, promptTokens.size() + count));
            json response = completion; response["edgevisor"] = metadata;
            request.writeJson(response.dump());
        }
    }
public:
    ApiServer(RootLlmInference *inference, Tokenizer *tokenizer, Sampler *sampler,
            NnNetwork *network, AppCliArgs *args, LlmHeader *header,
            EosDetector *, ChatTemplateGenerator *templateGenerator)
        : inference(inference), tokenizer(tokenizer), sampler(sampler), network(network),
          args(args), header(header), templateGenerator(templateGenerator) {}

    void complete(HttpRequest &request) {
        InferenceParams params;
        std::string session;
        std::vector<int> tokens;
        std::string publicPrompt;
        // Validate before sending a stream header or touching any live KV history.
        try {
            params = parseRequest(request);
            session = request.parsedJson.value("edgevisor_session_id", std::string());
            if (session.size() > 256) throw std::invalid_argument("edgevisor_session_id is too long");
            std::vector<ChatItem> items(params.messages.size());
            for (size_t index = 0; index < items.size(); ++index) {
                items[index].role = params.messages[index].role; items[index].message = params.messages[index].content;
            }
            GeneratedChat prompt = templateGenerator->generate(items.size(), items.data(), true);
            if (prompt.length > 32u * 1024u * 1024u) throw std::invalid_argument("Prompt is too large");
            tokens.resize(prompt.length + 3u);
            int count = 0; tokenizer->encode(const_cast<char *>(prompt.content), tokens.data(), &count, true, true);
            if (count <= 0 || (NnUint)count > header->seqLen)
                throw std::invalid_argument("Prompt exceeds max-seq-len or has no tokens");
            tokens.resize(count);
            const NnUint available = header->seqLen - (NnUint)count + 1u;
            if (params.max_tokens > 0 && (NnUint)params.max_tokens > available)
                throw std::invalid_argument("max_tokens exceeds remaining context capacity: " + std::to_string(available));
            if (prompt.publicPrompt != nullptr) publicPrompt = prompt.publicPrompt;
        } catch (const json::exception &error) {
            request.writeJson(json({{"error", {{"message", error.what()}, {"type", "invalid_request_error"}}}}).dump(), 400); return;
        } catch (const std::invalid_argument &error) {
            request.writeJson(json({{"error", {{"message", error.what()}, {"type", "invalid_request_error"}}}}).dump(), 400); return;
        }
        completePrepared(request, params, session, tokens, publicPrompt);
    }
};

void handleCompletionsRequest(HttpRequest& request, ApiServer *api) {
    api->complete(request);
}

void handleModelsRequest(HttpRequest& request, const char* modelPath) {
    std::string path(modelPath);
    size_t pos = path.find_last_of("/\\");
    std::string modelName = (pos == std::string::npos) ? path : path.substr(pos + 1);

    Model model(modelName);
    ModelList list(model);
    std::string response = ((json)list).dump();
    request.writeJson(response);
}

static void server(AppInferenceContext *context) {
    const char *apiBind = std::getenv("DLLAMA_API_BIND");
    if (apiBind == nullptr || apiBind[0] == '\0') apiBind = "127.0.0.1";
    NnSocket serverSocket(createServerSocket(context->args->port, apiBind));

    TokenizerChatStops stops(context->tokenizer);
    const char *templateEos = selectChatTemplateEos(context->tokenizer, &stops, context->args->chatTemplateType);
    ChatTemplateGenerator templateGenerator(context->args->chatTemplateType, context->tokenizer->chatTemplate, templateEos);
    EosDetector eosDetector(stops.nStops, context->tokenizer->eosTokenIds.data(), stops.stops, stops.maxStopLength, stops.maxStopLength);
    ApiServer api(context->inference, context->tokenizer, context->sampler, context->network, context->args, context->header, &eosDetector, &templateGenerator);

    printf("Server URL: http://%s:%d/v1/\n", apiBind, context->args->port);

    std::vector<Route> routes = {
        {
            "/v1/chat/completions",
            HttpMethod::METHOD_POST,
            std::bind(&handleCompletionsRequest, std::placeholders::_1, &api)
        },
        {
            "/v1/models",
            HttpMethod::METHOD_GET,
            std::bind(&handleModelsRequest, std::placeholders::_1, context->args->modelPath)
        }
    };

    while (true) {
        bool restart = false;
        std::string restartReason;
        try {
            NnSocket clientSocket(acceptSocket(serverSocket.fd));
            HttpRequest request = HttpRequest::read(clientSocket.fd);
            printf("🔷 %s %s\n", request.getMethod().c_str(), request.path.c_str());
            try {
                Router::resolve(request, routes);
            } catch (const NnPeerTimeoutException &e) {
                try { request.writeFailure(503, "runtime_unavailable", "peer_timeout", e.what()); } catch (...) {}
                restart = true; restartReason = std::string("peer timeout: ") + e.what();
            } catch (const NnPeerOfflineException &e) {
                try { request.writeFailure(503, "runtime_unavailable", "peer_offline", e.what()); } catch (...) {}
                restart = true; restartReason = std::string("peer offline: ") + e.what();
            } catch (const NnSessionRestartException &e) {
                try { request.writeFailure(503, "runtime_unavailable", "session_restart", e.what()); } catch (...) {}
                restart = true; restartReason = std::string("session restart: ") + e.what();
            } catch (const NnTransferSocketException &e) {
                try { request.writeFailure(503, "runtime_unavailable", "worker_socket", e.what()); } catch (...) {}
                restart = true; restartReason = std::string("worker socket error: ") + e.what();
            } catch (const NnExecutorException &e) {
                try { request.writeFailure(500, "server_error", "executor_error", e.what()); } catch (...) {}
                restart = true; restartReason = std::string("executor error: ") + e.what();
            } catch (const std::exception &e) {
                try { request.writeFailure(500, "server_error", "internal_error", e.what()); } catch (...) {}
                restart = true; restartReason = std::string("internal error: ") + e.what();
            } catch (...) {
                try { request.writeFailure(500, "server_error", "internal_error", "unknown generation failure"); } catch (...) {}
                restart = true; restartReason = "unknown generation failure";
            }
        } catch (const std::exception &e) {
            // Accept/read failures and client disconnects only invalidate this
            // HTTP connection; they must not restart the inference runtime.
            printf("API request I/O error: %s\n", e.what());
            std::fflush(stdout);
        }
        if (restart) throw ApiRuntimeRestartException(restartReason);
    }
}

#ifdef _WIN32
    #define EXECUTABLE_NAME "dllama-api.exe"
#else
    #define EXECUTABLE_NAME "dllama-api"
#endif

void usage() {
    fprintf(stderr, "Usage: %s {--model <path>} {--tokenizer <path>} [--port <p>]\n", EXECUTABLE_NAME);
    fprintf(stderr, "        [--buffer-float-type {f32|f16|q40|q80}]\n");
    fprintf(stderr, "        [--backend {cpu|vulkan|cuda}] [--gpu-index <index>]\n");
    fprintf(stderr, "        [--weights-float-type {f32|f16|q40|q80}]\n");
    fprintf(stderr, "        [--max-seq-len <max>]\n");
    fprintf(stderr, "        [--nthreads <n>]\n");
    fprintf(stderr, "        [--workers <ip:port> ...]\n");
    fprintf(stderr, "        [--temperature <temp>]\n");
    fprintf(stderr, "        [--topp <t>]\n");
    fprintf(stderr, "        [--seed <s>]\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  sudo nice -n -20 ./dllama-api --port 9990 --nthreads 4 \\\n");
    fprintf(stderr, "    --model dllama_model_llama3_2_3b_instruct_q40.m \\\n");
    fprintf(stderr, "    --tokenizer dllama_tokenizer_llama3_2_3b_instruct_q40.t \\\n");
    fprintf(stderr, "    --buffer-float-type q80 --max-seq-len 8192 \\\n");
    fprintf(stderr, "    --workers 10.0.0.2:9998 10.0.0.3:9998 10.0.0.4:9998\n");
    fflush(stderr);
}

int main(int argc, char *argv[]) {
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif

    AppCliArgs args = AppCliArgs::parse(argc, argv, false);
    if (args.help) {
        usage();
        return EXIT_SUCCESS;
    }

    initQuants();
    initSockets();

    while (true) {
        try {
            runInferenceApp(&args, server);
        } catch (const ApiRuntimeRestartException &e) {
            printf("🚨 API runtime restart: %s\n", e.what());
        } catch (const NnConnectionSocketException &e) {
            printf("🚨 Connection error: %s\n", e.what());
        } catch (const NnExecutorException &e) {
            printf("🚨 Inference error: %s\n", e.what());
        }
        std::fflush(stdout);

        printf("🔄 Retrying in 3 seconds...\n");
        std::this_thread::sleep_for(std::chrono::seconds(3));
        args.info = false;
    }

    cleanupSockets();
    return EXIT_SUCCESS;
}
