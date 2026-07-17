# Dynamic TPOT PP Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the default synthetic PP load penalty, expose the relative PP gain threshold through CLI/environment configuration, and log the strongest rejected PP candidate.

**Architecture:** Keep scoring in `tpot_algorithm`, CLI mapping in `app`, and log formatting in a pure header-only helper. PP scanning returns the best valid candidate when one exists; otherwise it returns the highest-gain structurally eligible rejected candidate with complete diagnostics.

**Tech Stack:** C++11, GNU Make, existing assertion-based C++ tests.

## Global Constraints

- Default `loadPenaltyBeta` is `0.0`; `DLLAMA_TPOT_LOAD_PENALTY_BETA` remains supported.
- `--tpot-pp-gain-ratio` maps to `DLLAMA_TPOT_PP_GAIN_RATIO`; default `0.03`, accepted CLI range `[0,1]`.
- Ratio `0` disables only the relative threshold component.
- TP scoring, migration execution, cooldown, verification, rollback, protocols, and model behavior remain unchanged.
- Existing scheduler fields remain compatible; new `pp_best_*` fields are appended.
- Never stage or overwrite unrelated dirty-worktree files.

---

### Task 1: Scoring defaults and rejected-candidate retention

**Files:**
- Modify: `EdgeVisor/src/dynamic/tpot_algorithm.hpp`
- Modify: `EdgeVisor/src/dynamic/tpot_algorithm.cpp`
- Test: `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`

**Interfaces:**
- Produces: `SchedulerConfig::ppGainRatio` (`double`, default `0.03`).
- Produces: `bestPpCandidate()` returning the best valid candidate or, when none is valid, the strongest eligible rejected candidate.

- [ ] **Step 1: Write failing default and threshold tests**

Replace the initial beta setup and add ratio assertions:

```cpp
tpot::SchedulerConfig cfg;
require(near(cfg.loadPenaltyBeta, 0.0), "load penalty defaults to zero");
require(near(cfg.ppGainRatio, 0.03), "PP gain ratio defaults to three percent");
cfg.minPpGainMs = 5.0;
cfg.minTpGainMs = 2.0;

tpot::StageSnapshot s = stage(0, 0, 0, 4, 10.0);
require(near(tpot::ppDeltaInMs(s, cfg), 10.0), "default delta_in has no load penalty");
cfg.loadPenaltyBeta = 0.08;
require(near(tpot::ppDeltaInMs(s, cfg), 14.0), "explicit beta preserves legacy cost");
cfg.loadPenaltyBeta = 0.0;

cfg.ppGainRatio = 0.03;
require(near(tpot::ppGainThresholdMs(200.0, cfg), 6.0), "ratio controls PP threshold");
cfg.ppGainRatio = 0.0;
require(near(tpot::ppGainThresholdMs(200.0, cfg), 5.0), "zero ratio leaves absolute threshold");
cfg.ppGainRatio = 0.10;
require(near(tpot::ppGainThresholdMs(200.0, cfg), 20.0), "custom ratio changes threshold");
cfg.ppGainRatio = 0.03;
```

- [ ] **Step 2: Write failing rejected-candidate tests**

```cpp
cfg.minPpGainMs = 100.0;
tpot::Candidate noPp = tpot::bestPpCandidate(pp, 110.0, cfg);
require(!noPp.valid, "PP candidate rejected below threshold");
require(noPp.reason == "gain below threshold", "rejected candidate reports reason");
require(noPp.fromStageIndex == 0u && noPp.toStageIndex == 1u, "rejected candidate retains direction");
require(noPp.layerIndex == 3u, "rejected candidate retains boundary layer");
require(noPp.gainMs > 0.0 && near(noPp.thresholdMs, 100.0), "rejected candidate retains scores");
cfg.minPpGainMs = 5.0;
```

Add a mixed valid/rejected scan:

```cpp
std::vector<tpot::StageSnapshot> mixed;
mixed.push_back(stage(0, 0, 0, 4, 30.0));
mixed.push_back(stage(1, 1, 4, 4, 5.0));
mixed.push_back(stage(2, 2, 8, 4, 3.0));
mixed[0].rightBoundaryLayerMs = 30.0;
tpot::Candidate mixedBest = tpot::bestPpCandidate(mixed, 152.0, cfg);
require(mixedBest.valid, "valid PP candidate wins over rejected alternatives");
require(mixedBest.fromStageIndex == 0u && mixedBest.toStageIndex == 1u, "valid PP route remains selected");
```

- [ ] **Step 3: Run RED**

```bash
make -C EdgeVisor dynamic-tpot-test
```

Expected: compilation fails because `ppGainRatio` is absent, or the zero-default assertion fails against `0.08`.

- [ ] **Step 4: Implement minimal scoring changes**

In `SchedulerConfig`:

```cpp
double loadPenaltyBeta = 0.0;
double ppGainRatio = 0.03;
```

Use the ratio in both threshold functions:

```cpp
return maxDouble(cfg.minPpGainMs, cfg.ppGainRatio * currentTpotMs);
```

and:

```cpp
return maxDouble(cfg.minPpGainMs, cfg.ppGainRatio * localMs);
```

Initialize and update the PP result as follows:

```cpp
Candidate best;
best.kind = CandidateKind::PP_MOVE;
best.thresholdMs = cfg.minPpGainMs;
best.reason = "no eligible pp candidate";
bool haveRejected = false;

// After Candidate c is fully populated:
if (c.valid) {
    considerBest(best, c);
} else if (!best.valid && (!haveRejected || c.gainMs > best.gainMs)) {
    best = c;
    haveRejected = true;
}
```

- [ ] **Step 5: Run GREEN and commit**

```bash
make -C EdgeVisor dynamic-tpot-test
EdgeVisor/dynamic-tpot-test
git add EdgeVisor/src/dynamic/tpot_algorithm.hpp EdgeVisor/src/dynamic/tpot_algorithm.cpp EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp
git commit -m "feat: make PP threshold model configurable"
```

Expected: output contains `PASS dynamic_tpot_algorithm`.

### Task 2: CLI mapping and validation

**Files:**
- Modify: `EdgeVisor/src/app.hpp`
- Modify: `EdgeVisor/src/app.cpp`
- Test: `EdgeVisor/src/test/test_app_cli.cpp`

**Interfaces:**
- Produces: `AppCliArgs::tpotPpGainRatioStr`.
- Produces: CLI option `--tpot-pp-gain-ratio VALUE` and environment mapping `DLLAMA_TPOT_PP_GAIN_RATIO`.

- [ ] **Step 1: Write failing CLI tests**

Clear the new environment variable in `clearTpotEnvironment()`:

```cpp
unsetenv("DLLAMA_TPOT_PP_GAIN_RATIO");
```

Add `--tpot-pp-gain-ratio 0` to the aggressive-profile argv and assert:

```cpp
assert(strcmp(getenv("DLLAMA_TPOT_PP_GAIN_RATIO"), "0") == 0);
```

Include `<cstdio>`, add this helper, and assert each invalid value is rejected:

```cpp
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

assert(rejectsPpGainRatio("-0.01"));
assert(rejectsPpGainRatio("1.01"));
assert(rejectsPpGainRatio("nan"));
assert(rejectsPpGainRatio("inf"));
assert(rejectsPpGainRatio("not-a-number"));
```

- [ ] **Step 2: Run RED**

```bash
make -C EdgeVisor app-cli-test
EdgeVisor/app-cli-test
```

Expected: `Unknown option: --tpot-pp-gain-ratio` or a failed invalid-value assertion.

- [ ] **Step 3: Implement CLI and scheduler configuration**

Add and initialize:

```cpp
char *tpotPpGainRatioStr;
```

Validate CLI values with this helper:

```cpp
static void validateUnitIntervalOption(const char *name, const char *value) {
    errno = 0;
    char *end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0' ||
        !std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0) {
        throw std::runtime_error(std::string(name) + " must be a finite number in [0,1]");
    }
}
```

The parse branch calls `validateUnitIntervalOption(name, value)` before assigning `args.tpotPpGainRatioStr = value`.

Parse and map:

```cpp
setTpotOverride("DLLAMA_TPOT_PP_GAIN_RATIO", args.tpotPpGainRatioStr);
```

Load it in `loadSchedulerConfig()`:

```cpp
cfg.ppGainRatio = parseEnvDouble("DLLAMA_TPOT_PP_GAIN_RATIO", cfg.ppGainRatio);
if (!std::isfinite(cfg.ppGainRatio) || cfg.ppGainRatio < 0.0 || cfg.ppGainRatio > 1.0) {
    throw std::runtime_error("DLLAMA_TPOT_PP_GAIN_RATIO must be a finite number in [0,1]");
}
```

- [ ] **Step 4: Run GREEN and commit**

```bash
make -C EdgeVisor app-cli-test dynamic-tpot-test
EdgeVisor/app-cli-test
EdgeVisor/dynamic-tpot-test
git add EdgeVisor/src/app.hpp EdgeVisor/src/app.cpp EdgeVisor/src/test/test_app_cli.cpp
git commit -m "feat: add PP gain ratio CLI option"
```

Expected: both executables exit `0`.

### Task 3: Candidate log formatting and scheduler integration

**Files:**
- Create: `EdgeVisor/src/dynamic/tpot_log.hpp`
- Modify: `EdgeVisor/src/dynamic/dynamic_tpot.cpp`
- Test: `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`

**Interfaces:**
- Produces: `formatPpCandidateLogFields(const Candidate&) -> std::string`.
- Produces: appended `pp_best_*` fields on every evaluated decision line.

- [ ] **Step 1: Write failing formatter tests**

Include `dynamic/tpot_log.hpp`. Format the rejected candidate from Task 1 and assert presence of:

```text
pp_best_valid=0
pp_best_gain_ms=
pp_best_threshold_ms=100
pp_best_reason=gain_below_threshold
pp_best_from_stage=0
pp_best_to_stage=1
pp_best_layer=3
```

Format a valid candidate and assert `pp_best_valid=1` and `pp_best_reason=none`.

- [ ] **Step 2: Run RED**

```bash
make -C EdgeVisor dynamic-tpot-test
```

Expected: compilation fails because `dynamic/tpot_log.hpp` is absent.

- [ ] **Step 3: Implement pure formatter**

Create this header-only C++11 helper:

```cpp
#pragma once

#include "dynamic/tpot_algorithm.hpp"

#include <cctype>
#include <sstream>
#include <string>

namespace dllama {
namespace dynamic_tpot {

inline std::string sanitizeCandidateLogValue(const std::string &value) {
    if (value.empty()) return "none";
    std::string out = value;
    for (size_t i = 0u; i < out.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(out[i]);
        if (std::isspace(c) || out[i] == '=') out[i] = '_';
    }
    return out;
}

inline std::string formatPpCandidateLogFields(const Candidate &candidate) {
    std::ostringstream oss;
oss << "pp_best_valid=" << (candidate.valid ? 1 : 0)
    << " pp_best_gain_ms=" << candidate.gainMs
    << " pp_best_threshold_ms=" << candidate.thresholdMs
    << " pp_best_reason=" << sanitizeCandidateLogValue(candidate.reason)
    << " pp_best_from_stage=" << candidate.fromStageIndex
    << " pp_best_to_stage=" << candidate.toStageIndex
    << " pp_best_layer=" << candidate.layerIndex;
    return oss.str();
}

} // namespace dynamic_tpot
} // namespace dllama
```

- [ ] **Step 4: Integrate into scheduler logging**

Include `tpot_log.hpp` and change the formatter signature to:

```cpp
static void logDecision(
    ControllerRuntime &rt,
    const WindowSummary &window,
    const tpot::Candidate &best,
    const tpot::Candidate &bestPp,
    bool issued,
    const char *extra,
    const PendingAction *comparison = nullptr,
    uint32_t verifyElapsedTokens = 0u)
```

Before appending `note`, append:

```cpp
oss << " " << tpot::formatPpCandidateLogFields(bestPp);
```

Update the three evaluated call paths:

```cpp
logDecision(rt, window, pendingForLog.candidate, bestPp, verifyIssued, note, &pendingForLog, elapsed);
logDecision(rt, window, best, bestPp, false, note);
logDecision(rt, window, best, bestPp, issued, note);
```

Leave startup, disabled, and exception-only lines unchanged.

- [ ] **Step 5: Run GREEN and commit**

```bash
make -C EdgeVisor dynamic-tpot-test app-cli-test
EdgeVisor/dynamic-tpot-test
EdgeVisor/app-cli-test
git add EdgeVisor/src/dynamic/tpot_log.hpp EdgeVisor/src/dynamic/dynamic_tpot.cpp EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp
git commit -m "feat: log rejected PP candidates"
```

Expected: both executables exit `0`.

### Task 4: Documentation and final verification

**Files:**
- Modify: `EdgeVisor/docs/README_ENV_VARS.md`

**Interfaces:**
- Documents: beta default `0`, ratio CLI/environment forms, and all appended candidate fields.

- [ ] **Step 1: Update documentation**

Change `DLLAMA_TPOT_LOAD_PENALTY_BETA` default to `0`. Add:

```markdown
| `DLLAMA_TPOT_PP_GAIN_RATIO` / `--tpot-pp-gain-ratio` | 浮点 `[0,1]` | `0.03` | PP 收益相对门槛系数；`0` 表示只使用绝对毫秒门槛 | 越小越容易触发 PP 迁移 |
```

Replace the `DLLAMA_TPOT_LOG` description with text that explicitly lists `pp_best_valid`, `pp_best_gain_ms`, `pp_best_threshold_ms`, `pp_best_reason`, `pp_best_from_stage`, `pp_best_to_stage`, and `pp_best_layer`.

- [ ] **Step 2: Run clean focused verification**

```bash
make -C EdgeVisor clean
make -C EdgeVisor dynamic-tpot-test app-cli-test
EdgeVisor/dynamic-tpot-test
EdgeVisor/app-cli-test
```

Expected: clean rebuild succeeds, `PASS dynamic_tpot_algorithm` appears, and no assertion fails.

- [ ] **Step 3: Build production CPU binary**

```bash
make -C EdgeVisor dllama
```

Expected: exit `0` and `EdgeVisor/dllama` exists. Jetson Vulkan validation is a separate deployment step.

- [ ] **Step 4: Verify scope and commit docs**

```bash
git diff --check
git status --short
git add EdgeVisor/docs/README_ENV_VARS.md
git commit -m "docs: describe PP threshold controls"
```

- [ ] **Step 5: Run post-commit verification**

```bash
make -C EdgeVisor dynamic-tpot-test app-cli-test dllama
EdgeVisor/dynamic-tpot-test
EdgeVisor/app-cli-test
git status --short --branch
```

Expected: all commands exit `0`; only pre-existing unrelated workspace changes remain uncommitted.
