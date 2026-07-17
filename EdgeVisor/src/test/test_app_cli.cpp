#include "app.hpp"

#include <assert.h>
#include <cstdio>
#include <stdlib.h>
#include <string.h>

static AppCliArgs parseArgs(int argc, char **argv) {
    return AppCliArgs::parse(argc, argv, true);
}

static void clearTpotEnvironment() {
    unsetenv("DLLAMA_TPOT_WINDOW_TOKENS");
    unsetenv("DLLAMA_TPOT_MIN_SAMPLES");
    unsetenv("DLLAMA_TPOT_COOLDOWN_TOKENS");
    unsetenv("DLLAMA_TPOT_ROLLBACK_WINDOW");
    unsetenv("DLLAMA_TPOT_MIN_PP_GAIN_MS");
    unsetenv("DLLAMA_TPOT_PP_RISK_MARGIN_MS");
    unsetenv("DLLAMA_TPOT_PP_MIGRATION_COST_MS");
    unsetenv("DLLAMA_TPOT_EXPECTED_REMAINING_TOKENS");
    unsetenv("DLLAMA_TPOT_PP_GAIN_RATIO");
    unsetenv("DLLAMA_LAYER_PROF_ENABLE");
    unsetenv("DLLAMA_SYNC_ENV_VARS");
}

static bool rejectsPpGainRatio(const char *text) {
    char arg0[] = "dllama";
    char arg1[] = "inference";
    char arg2[] = "--tpot-pp-gain-ratio";
    char value[32];
    std::snprintf(value, sizeof(value), "%s", text);
    char *argv[] = {arg0, arg1, arg2, value};
    try {
        parseArgs(4, argv);
    } catch (...) {
        return true;
    }
    return false;
}

int main() {
    {
        char arg0[] = "dllama";
        char arg1[] = "inference";
        char arg2[] = "--memory-limit-gib";
        char arg3[] = "4.5";
        char *argv[] = {arg0, arg1, arg2, arg3};
        AppCliArgs args = parseArgs(4, argv);
        assert(args.memoryLimitBytes == (NnSize) (4.5 * 1024 * 1024 * 1024));
    }

    {
        char arg0[] = "dllama";
        char arg1[] = "inference";
        char arg2[] = "--memory-limit-gib";
        char arg3[] = "0";
        char *argv[] = {arg0, arg1, arg2, arg3};
        bool rejected = false;
        try {
            parseArgs(4, argv);
        } catch (...) {
            rejected = true;
        }
        assert(rejected);
    }

    clearTpotEnvironment();
    {
        char arg0[] = "dllama";
        char arg1[] = "inference";
        char arg2[] = "--enable-dynamic-tpot";
        char arg3[] = "--dynamic-tpot-profile";
        char arg4[] = "aggressive";
        char arg5[] = "--tpot-window-tokens";
        char arg6[] = "9";
        char arg7[] = "--workers";
        char arg8[] = "127.0.0.1:9999";
        char arg9[] = "--tpot-pp-gain-ratio";
        char arg10[] = "0";
        char *argv[] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10};
        AppCliArgs args = parseArgs(11, argv);
        assert(args.lastStageSampling);
        assert(strcmp(getenv("DLLAMA_TPOT_WINDOW_TOKENS"), "9") == 0);
        assert(strcmp(getenv("DLLAMA_TPOT_MIN_SAMPLES"), "6") == 0);
        assert(strcmp(getenv("DLLAMA_TPOT_COOLDOWN_TOKENS"), "8") == 0);
        assert(strcmp(getenv("DLLAMA_TPOT_PP_GAIN_RATIO"), "0") == 0);
        assert(strcmp(getenv("DLLAMA_LAYER_PROF_ENABLE"), "1") == 0);
        assert(strstr(getenv("DLLAMA_SYNC_ENV_VARS"), "DLLAMA_LAYER_PROF_ENABLE") != nullptr);
    }

    assert(rejectsPpGainRatio("-0.01"));
    assert(rejectsPpGainRatio("1.01"));
    assert(rejectsPpGainRatio("nan"));
    assert(rejectsPpGainRatio("inf"));
    assert(rejectsPpGainRatio("not-a-number"));

    return 0;
}
