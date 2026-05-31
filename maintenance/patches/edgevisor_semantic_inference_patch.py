from pathlib import Path

root = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor")
p = root / "src/dllama.cpp"
s = p.read_text()

helper_marker = """static void inferenceRunOnce(AppInferenceContext *context, const char* prompt, NnUint steps) {
"""
helper = r'''static bool promptLooksChatFormatted(const char *prompt) {
    if (prompt == nullptr) return false;
    return std::strstr(prompt, "<|start_header_id|>") != nullptr ||
        std::strstr(prompt, "[INST]") != nullptr ||
        std::strstr(prompt, "<｜User｜>") != nullptr ||
        std::strstr(prompt, "<|im_start|>") != nullptr;
}

static std::string buildInferencePrompt(AppInferenceContext *context, const char *prompt, TokenizerChatStops *stops) {
    if (context == nullptr || context->tokenizer == nullptr || prompt == nullptr) {
        return prompt == nullptr ? std::string() : std::string(prompt);
    }
    if (promptLooksChatFormatted(prompt) || context->tokenizer->chatTemplate == nullptr || stops == nullptr || stops->nStops == 0) {
        return std::string(prompt);
    }

    ChatTemplateGenerator generator(context->args->chatTemplateType, context->tokenizer->chatTemplate, stops->stops[0]);
    ChatItem item{"user", prompt};
    GeneratedChat generated = generator.generate(1u, &item, true);
    return std::string(generated.content, generated.length);
}

'''
if helper not in s:
    if helper_marker not in s:
        raise SystemExit("inferenceRunOnce marker not found")
    s = s.replace(helper_marker, helper + helper_marker, 1)

old = """static void inferenceRunOnce(AppInferenceContext *context, const char* prompt, NnUint steps) {
    if (prompt == nullptr)
        throw std::runtime_error(\"Prompt is required\");
    if (steps == 0)
        throw std::runtime_error(\"Number of steps is required\");

    std::vector<int> inputTokensVec(std::strlen(prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    NnUint pos = 0;
    int nInputTokens;
    context->tokenizer->encode(const_cast<char*>(prompt), inputTokens, &nInputTokens, true, true);
"""
new = """static void inferenceRunOnce(AppInferenceContext *context, const char* prompt, NnUint steps) {
    if (prompt == nullptr)
        throw std::runtime_error(\"Prompt is required\");
    if (steps == 0)
        throw std::runtime_error(\"Number of steps is required\");

    TokenizerChatStops stops(context->tokenizer);
    EosDetector eosDetector(stops.nStops, context->tokenizer->eosTokenIds.data(), stops.stops, stops.maxStopLength, stops.maxStopLength);
    std::string effectivePrompt = buildInferencePrompt(context, prompt, &stops);

    std::vector<int> inputTokensVec(effectivePrompt.size() + 3);
    int *inputTokens = inputTokensVec.data();

    NnUint pos = 0;
    int nInputTokens;
    context->tokenizer->encode(const_cast<char*>(effectivePrompt.c_str()), inputTokens, &nInputTokens, true, true);
"""
if old not in s:
    raise SystemExit("inferenceRunOnce prologue pattern not found")
s = s.replace(old, new, 1)

s = s.replace('''    int token = inputTokens[pos];
    printf("%s\\n", prompt);
''', '''    int token = inputTokens[pos];
    printf("%s\\n", effectivePrompt.c_str());
''', 1)

old_gen = """        token = context->sampler->sample(context->inference->logitsPipe);

        char *piece = context->tokenizer->decode(token);

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);

        NnUint predTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf(\"🔶 Pred%5u ms Sync%5u ms | pos=%u | Sent%6zu kB Recv%6zu kB | %s\\n\",
            predTime / 1000,
            syncTime / 1000,
            (unsigned)pos,
            sentBytes / 1024,
            recvBytes / 1024,
            piece == nullptr ? \"~\" : piece);
        fflush(stdout);
        predTotalTime += predTime + syncTime;
"""
new_gen = """        token = context->sampler->sample(context->inference->logitsPipe);

        char *piece = context->tokenizer->decode(token);
        EosDetectorType eosType = eosDetector.append(token, piece);
        char *delta = nullptr;
        if (eosType == NOT_EOS || eosType == EOS) {
            delta = eosDetector.getDelta();
        }

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);

        NnUint predTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf(\"🔶 Pred%5u ms Sync%5u ms | pos=%u | Sent%6zu kB Recv%6zu kB | %s\\n\",
            predTime / 1000,
            syncTime / 1000,
            (unsigned)pos,
            sentBytes / 1024,
            recvBytes / 1024,
            delta == nullptr ? (eosType == EOS ? \"[EOS]\" : \"\") : delta);
        fflush(stdout);
        if (eosType == NOT_EOS || eosType == EOS) {
            eosDetector.reset();
        }
        predTotalTime += predTime + syncTime;
        if (eosType == EOS) {
            pos++;
            break;
        }
"""
if old_gen not in s:
    raise SystemExit("generation sample/decode pattern not found")
s = s.replace(old_gen, new_gen, 1)

p.write_text(s)
print("patched inference mode: auto chat template + EOS stop")
