#include "app.hpp"
#include "dynamic/dynamic_tpot.hpp"
#include "dynamic/tpot_algorithm.hpp"
#include "json.hpp"
#include "plan-controller.hpp"

#include <assert.h>
#include <cstdio>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

using json = nlohmann::json;

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
    unsetenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE");
    unsetenv("DLLAMA_RUNTIME_REDUNDANT_BOUNDARY_LAYERS");
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

static bool rejectsMaxPpLayerMove(const char *text) {
    char arg0[] = "dllama";
    char arg1[] = "inference";
    char arg2[] = "--tpot-max-pp-layer-move";
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

static bool rejectsEnvironmentMaxPpLayerMove(const char *text) {
    setenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE", text, 1);
    try {
        (void)dllama::dynamic_tpot::loadSchedulerConfigFromEnvironment();
    } catch (...) {
        unsetenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE");
        return true;
    }
    unsetenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE");
    return false;
}

int main() {
    static_assert(sizeof(PlanCommand) == 1336u, "PlanCommand wire layout must remain unchanged");
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
        assert(cfg.maxPpLayerMove == 1u);
    }
    setenv("DLLAMA_TPOT_PP_GAIN_RATIO", "0.5", 1);
    {
        const dllama::dynamic_tpot::SchedulerConfig cfg =
            dllama::dynamic_tpot::loadSchedulerConfigFromEnvironment();
        assert(cfg.ppGainRatio == 0.5);
    }
    unsetenv("DLLAMA_TPOT_PP_GAIN_RATIO");

    setenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE", "4", 1);
    {
        const dllama::dynamic_tpot::SchedulerConfig cfg =
            dllama::dynamic_tpot::loadSchedulerConfigFromEnvironment();
        assert(cfg.maxPpLayerMove == 4u);
    }
    setenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE", "64", 1);
    {
        const dllama::dynamic_tpot::SchedulerConfig cfg =
            dllama::dynamic_tpot::loadSchedulerConfigFromEnvironment();
        assert(cfg.maxPpLayerMove == 64u);
    }
    unsetenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE");

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

    {
        char arg0[] = "dllama";
        char arg1[] = "inference";
        char arg2[] = "--enable-dynamic-tpot";
        char arg3[] = "--tpot-max-pp-layer-move";
        char arg4[] = "4";
        char *argv[] = {arg0, arg1, arg2, arg3, arg4};
        AppCliArgs args = parseArgs(5, argv);
        (void)args;
        assert(strcmp(getenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE"), "4") == 0);
        assert(args.runtimeRedundantBoundaryLayers == 4u);
    }

    clearTpotEnvironment();
    setenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE", "2", 1);
    {
        char arg0[] = "dllama";
        char arg1[] = "inference";
        char arg2[] = "--enable-dynamic-tpot";
        char arg3[] = "--tpot-max-pp-layer-move";
        char arg4[] = "64";
        char *argv[] = {arg0, arg1, arg2, arg3, arg4};
        AppCliArgs args = parseArgs(5, argv);
        assert(strcmp(getenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE"), "64") == 0);
        assert(args.runtimeRedundantBoundaryLayers == 64u);
    }

    clearTpotEnvironment();
    setenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE", "4", 1);
    {
        char arg0[] = "dllama";
        char arg1[] = "inference";
        char arg2[] = "--enable-dynamic-tpot";
        char *argv[] = {arg0, arg1, arg2};
        AppCliArgs args = parseArgs(3, argv);
        assert(args.runtimeRedundantBoundaryLayers == 4u);
    }

    clearTpotEnvironment();
    {
        char arg0[] = "dllama";
        char arg1[] = "inference";
        char arg2[] = "--enable-dynamic-tpot";
        char arg3[] = "--tpot-max-pp-layer-move";
        char arg4[] = "4";
        char arg5[] = "--runtime-redundant-boundary-layers";
        char arg6[] = "2";
        char *argv[] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6};
        bool rejected = false;
        try {
            (void)parseArgs(7, argv);
        } catch (...) {
            rejected = true;
        }
        assert(rejected);
    }

    clearTpotEnvironment();
    setenv("DLLAMA_RUNTIME_REDUNDANT_BOUNDARY_LAYERS", "4", 1);
    NnUnevenPartitionPlan rangePlan;
    rangePlan.nNodes = 2u;
    rangePlan.nStages = 2u;
    rangePlan.stages = new NnStageConfig[2];
    rangePlan.stages[0].stageIndex = 0u;
    rangePlan.stages[0].startLayer = 0u;
    rangePlan.stages[0].endLayer = 8u;
    rangePlan.stages[0].nLayers = 8u;
    rangePlan.stages[0].rootNodeIndex = 0u;
    rangePlan.stages[0].nNodes = 1u;
    rangePlan.stages[0].nodeIndices = new NnUint[1]{0u};
    rangePlan.stages[1].stageIndex = 1u;
    rangePlan.stages[1].startLayer = 8u;
    rangePlan.stages[1].endLayer = 16u;
    rangePlan.stages[1].nLayers = 8u;
    rangePlan.stages[1].rootNodeIndex = 1u;
    rangePlan.stages[1].nNodes = 1u;
    rangePlan.stages[1].nodeIndices = new NnUint[1]{1u};

    RuntimeStageLayerPlan runtimePlan = buildRuntimeStageLayerPlan(&rangePlan, 16u);
    for (NnUint layer = 4u; layer < 8u; ++layer) {
        assert(runtimePlan.getRole(0u, layer) == RUNTIME_LAYER_PRIMARY);
        assert(runtimePlan.getRole(1u, layer) == RUNTIME_LAYER_REDUNDANT);
    }
    for (NnUint layer = 8u; layer < 12u; ++layer) {
        assert(runtimePlan.getRole(0u, layer) == RUNTIME_LAYER_REDUNDANT);
        assert(runtimePlan.getRole(1u, layer) == RUNTIME_LAYER_PRIMARY);
    }

    dllama::dynamic_tpot::Candidate issued;
    issued.kind = dllama::dynamic_tpot::CandidateKind::PP_MOVE;
    issued.valid = true;
    issued.fromStageIndex = 0u;
    issued.toStageIndex = 1u;
    issued.fromNodeIndex = 0u;
    issued.toNodeIndex = 1u;
    issued.layerIndex = 4u;
    issued.layerCount = 4u;
    const json request = dllama::dynamic_tpot::makePpCommandRequest(17u, issued);
    assert(request.at("op").get<std::string>() == "set_pp_migration");
    assert(request.at("cmd").at("firstLayer").get<uint32_t>() == 4u);
    assert(request.at("cmd").at("layerCount").get<uint32_t>() == 4u);

    const PlanCommand forward = decodePpMigrationCommand(request.at("cmd"));
    assert(forward.mode == PLAN_CMD_MODE_NEXT_BARRIER);
    assert(forward.triggerLayer == 4u);
    assert(forward.reserved0 == 4u);
    std::vector<NnUint> forwardLayers;
    std::string rangeReason;
    assert(resolvePpMigrationLayers(forward, &rangePlan, &runtimePlan, forwardLayers, &rangeReason));
    assert((forwardLayers == std::vector<NnUint>{4u, 5u, 6u, 7u}));

    dllama::dynamic_tpot::Candidate rollback = dllama::dynamic_tpot::reversePpCandidate(issued);
    const PlanCommand reverse = decodePpMigrationCommand(
        dllama::dynamic_tpot::makePpCommandRequest(18u, rollback).at("cmd"));
    std::vector<NnUint> reverseLayers;
    assert(resolvePpMigrationLayers(reverse, &rangePlan, &runtimePlan, reverseLayers, &rangeReason));
    assert(reverseLayers == forwardLayers);

    runtimePlan.setRole(1u, 6u, RUNTIME_LAYER_DISABLED);
    std::vector<NnUint> unsafeLayers;
    assert(!resolvePpMigrationLayers(forward, &rangePlan, &runtimePlan, unsafeLayers, &rangeReason));
    assert(unsafeLayers.empty());
    assert(rangeReason.find("target") != std::string::npos);
    clearTpotEnvironment();

    {
        char arg0[] = "dllama";
        char arg1[] = "inference";
        char arg2[] = "--tpot-max-pp-layer-move";
        char *argv[] = {arg0, arg1, arg2};
        bool rejected = false;
        try {
            parseArgs(3, argv);
        } catch (...) {
            rejected = true;
        }
        assert(rejected);
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

    assert(rejectsMaxPpLayerMove("0"));
    assert(rejectsMaxPpLayerMove("65"));
    assert(rejectsMaxPpLayerMove("-1"));
    assert(rejectsMaxPpLayerMove("garbage"));
    assert(rejectsMaxPpLayerMove("4junk"));
    assert(rejectsMaxPpLayerMove(""));

    assert(rejectsEnvironmentMaxPpLayerMove("0"));
    assert(rejectsEnvironmentMaxPpLayerMove("65"));
    assert(rejectsEnvironmentMaxPpLayerMove("-1"));
    assert(rejectsEnvironmentMaxPpLayerMove("garbage"));
    assert(rejectsEnvironmentMaxPpLayerMove("4junk"));
    assert(rejectsEnvironmentMaxPpLayerMove(""));

    return 0;
}
