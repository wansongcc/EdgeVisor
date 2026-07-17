#include "app.hpp"
#include "dynamic/dynamic_tpot.hpp"
#include "dynamic/tpot_algorithm.hpp"

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

static bool rejectsEnvironmentPpGainRatio(const char *text) {
    setenv("DLLAMA_TPOT_PP_GAIN_RATIO", text, 1);
    try {
        (void)dllama::dynamic_tpot::loadSchedulerConfigFromEnvironment();
    } catch (...) {
        unsetenv("DLLAMA_TPOT_PP_GAIN_RATIO");
        return true;
    }
    unsetenv("DLLAMA_TPOT_PP_GAIN_RATIO");
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
        const dllama::dynamic_tpot::SchedulerConfig cfg =
            dllama::dynamic_tpot::loadSchedulerConfigFromEnvironment();
        assert(cfg.ppGainRatio == 0.03);
    }
    setenv("DLLAMA_TPOT_PP_GAIN_RATIO", "0.5", 1);
    {
        const dllama::dynamic_tpot::SchedulerConfig cfg =
            dllama::dynamic_tpot::loadSchedulerConfigFromEnvironment();
        assert(cfg.ppGainRatio == 0.5);
    }
    unsetenv("DLLAMA_TPOT_PP_GAIN_RATIO");

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
    assert(rejectsPpGainRatio("0.5junk"));

    assert(rejectsEnvironmentPpGainRatio("garbage"));
    assert(rejectsEnvironmentPpGainRatio("0.5junk"));
    assert(rejectsEnvironmentPpGainRatio("nan"));
    assert(rejectsEnvironmentPpGainRatio("inf"));
    assert(rejectsEnvironmentPpGainRatio("-0.01"));
    assert(rejectsEnvironmentPpGainRatio("1.01"));

    return 0;
}
