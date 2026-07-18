# Multi-Layer PP Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the dynamic TPOT scheduler select and issue a contiguous PP migration of `1..K` layers while preserving one-layer behavior by default.

**Architecture:** Extend the scheduler candidate with a layer count, enumerate bounded counts in the pure TPOT algorithm, and propagate the selected count through prediction, rollback, logging, and the existing `set_pp_migration` command. Add one strictly validated root CLI/environment control whose default remains `1`; the existing Plan Controller and KV migration executor need no protocol change.

**Tech Stack:** C++11, nlohmann JSON already used by the controller, POSIX environment configuration, Make-based C++ tests.

## Global Constraints

- `SchedulerConfig::maxPpLayerMove` defaults to exactly `1u`.
- `--tpot-max-pp-layer-move` and `DLLAMA_TPOT_MAX_PP_LAYER_MOVE` accept only completely parsed decimal integers in `[1,64]`.
- A PP move may contain only contiguous layers at one boundary between adjacent stages.
- A PP move must leave at least one layer in its source stage.
- TP candidate scoring and TP head/FFN movement remain unchanged.
- For `k=1`, candidate gain, command behavior, rollback behavior, and logs remain compatible except for the appended `pp_best_layer_count=1` field.
- No new dependency and no wire-format version change.
- Preserve unrelated dirty files in the main checkout.

---

## File Structure

- `EdgeVisor/src/dynamic/tpot_algorithm.hpp`: add the candidate layer-count contract and pure helpers used by controller/tests.
- `EdgeVisor/src/dynamic/tpot_algorithm.cpp`: enumerate and score multi-layer candidates; update predicted state by the selected count.
- `EdgeVisor/src/dynamic/tpot_log.hpp`: append the stable layer-count diagnostic field.
- `EdgeVisor/src/dynamic/dynamic_tpot.cpp`: strictly load the maximum, preserve count during rollback, and send it in PP commands.
- `EdgeVisor/src/app.hpp`: store the new CLI override string.
- `EdgeVisor/src/app.cpp`: parse, validate, and map the CLI override to the scheduler environment.
- `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`: cover enumeration, scoring, state updates, command-count helper, rollback, and logging.
- `EdgeVisor/src/test/test_app_cli.cpp`: cover CLI/environment defaults, valid bounds, and malformed values.
- `EdgeVisor/docs/README_ENV_VARS.md`: document the new CLI mapping and corrected variable semantics.

### Task 1: Multi-Layer Candidate Model and Prediction

**Files:**
- Modify: `EdgeVisor/src/dynamic/tpot_algorithm.hpp`
- Modify: `EdgeVisor/src/dynamic/tpot_algorithm.cpp`
- Test: `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`

**Interfaces:**
- Produces: `Candidate::layerCount`, `reversePpCandidate(const Candidate &)`, `ppCommandLayerCount(const Candidate &)`, and multi-layer behavior in `bestPpCandidate()` / `applyPpMove()`.
- Consumes: existing `SchedulerConfig::maxPpLayerMove`, `StageSnapshot`, and `stageCostMs()`.

- [ ] **Step 1: Write failing algorithm tests**

Add assertions that express the new public behavior before production edits:

```cpp
require(cfg.maxPpLayerMove == 1u, "PP batch size defaults to one layer");

cfg.maxPpLayerMove = 4u;
cfg.loadPenaltyBeta = 0.5;
cfg.minPpGainMs = 0.0;
cfg.ppGainRatio = 0.0;
std::vector<tpot::StageSnapshot> multi;
multi.push_back(stage(0, 0, 0, 6, 20.0));
multi.push_back(stage(1, 1, 6, 2, 5.0));
multi[0].rightBoundaryLayerMs = 20.0;
const tpot::Candidate multiMove = tpot::bestPpCandidate(multi, 130.0, cfg);
require(multiMove.valid && multiMove.layerCount == 2u,
    "scheduler selects the interior optimum instead of the largest allowed batch");
require(multiMove.layerIndex == multi[0].endLayer - multiMove.layerCount,
    "forward candidate identifies the first layer in the contiguous range");

tpot::Candidate predicted = multiMove;
tpot::applyPpMove(multi, predicted);
require(multi[0].nLayers == 6u - predicted.layerCount &&
        multi[1].nLayers == 2u + predicted.layerCount,
    "predicted stage counts move the selected layer batch");

const tpot::Candidate reverse = tpot::reversePpCandidate(predicted);
require(reverse.layerCount == predicted.layerCount,
    "rollback preserves the selected PP layer count");
require(tpot::ppCommandLayerCount(predicted) == predicted.layerCount,
    "PP command count uses the selected candidate count");
```

This fixture has exact modeled gains of `7.5`, `10.0`, and `7.5` ms for `k=1`, `k=2`, and `k=3`, so it proves enumeration rather than merely accepting the maximum. Add these exact edge assertions:

```cpp
cfg.ppMigrationCostMs = 384.0; // 3 ms per layer at 128 remaining tokens
const tpot::Candidate costLimited = tpot::bestPpCandidate(originalMulti, 130.0, cfg);
require(costLimited.layerCount == 1u,
    "per-layer migration cost can make a smaller batch optimal");

std::vector<tpot::StageSnapshot> reverseStages;
reverseStages.push_back(stage(0, 0, 0, 2, 5.0));
reverseStages.push_back(stage(1, 1, 2, 6, 20.0));
reverseStages[1].leftBoundaryLayerMs = 20.0;
cfg.ppMigrationCostMs = 0.0;
const tpot::Candidate reverseMove = tpot::bestPpCandidate(reverseStages, 130.0, cfg);
require(reverseMove.fromStageIndex == 1u && reverseMove.toStageIndex == 0u &&
        reverseMove.layerIndex == reverseStages[1].startLayer,
    "reverse candidate begins at the source left boundary");

std::vector<tpot::StageSnapshot> twoLayerSource;
twoLayerSource.push_back(stage(0, 0, 0, 2, 20.0));
twoLayerSource.push_back(stage(1, 1, 2, 2, 5.0));
const tpot::Candidate leavesOne = tpot::bestPpCandidate(twoLayerSource, 50.0, cfg);
require(leavesOne.layerCount <= 1u,
    "configured K never empties a two-layer source stage");
```

Copy `multi` before applying the prediction so `originalMulti` retains its original stage counts. Restore `loadPenaltyBeta`, migration cost, and PP threshold settings before existing TP assertions; keep every existing TP assertion unchanged.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
make -C EdgeVisor dynamic-tpot-test && EdgeVisor/dynamic-tpot-test
```

Expected: compilation fails because `Candidate::layerCount`, `reversePpCandidate()`, and `ppCommandLayerCount()` do not exist.

- [ ] **Step 3: Add the candidate contract and helpers**

In `tpot_algorithm.hpp`, add:

```cpp
uint32_t layerCount = 1u;
```

to `Candidate`, plus:

```cpp
Candidate reversePpCandidate(const Candidate &candidate);
uint32_t ppCommandLayerCount(const Candidate &candidate);
```

Implement `reversePpCandidate()` by swapping source/target stage and node indexes while preserving `layerCount`; implement `ppCommandLayerCount()` as a defensive `max(1u, candidate.layerCount)` conversion for command emission.

- [ ] **Step 4: Implement bounded candidate enumeration**

Replace the single candidate per direction with:

```cpp
const uint32_t maxMove = std::min(cfg.maxPpLayerMove, sourceLayers - 1u);
for (uint32_t k = 1u; k <= maxMove; ++k) {
    const double firstOut = ppBoundaryDeltaOutMs(source, target, cfg);
    const double remainingOut = k > 1u
        ? stageCostMs(source, sourceLayers - 1u, cfg) -
          stageCostMs(source, sourceLayers - k, cfg)
        : 0.0;
    const double targetIn =
        stageCostMs(target, targetLayers + k, cfg) -
        stageCostMs(target, targetLayers, cfg);
    const double scaledRisk = (double)k *
        (cfg.ppRiskMarginMs + target.riskPenalty * maxDouble(1.0, target.avgLayerMs));
    const double gain = firstOut + remainingOut - targetIn -
        (double)k * migCost - scaledRisk;
    // Populate layerCount, contiguous range start, threshold, and route.
}
```

For a move to the next stage, set `layerIndex = source.endLayer - k`; for a move to the previous stage, set `layerIndex = source.startLayer`. Reuse the existing valid/rejected candidate selection rules across every `k`.

- [ ] **Step 5: Update predicted stage state by `k`**

Make `applyPpMove()` reject `k == 0` or `from->nLayers <= k`, then adjust counts and boundaries by `k`:

```cpp
from->nLayers -= k;
to->nLayers += k;
```

For a forward move decrement `from->endLayer` and `to->startLayer` by `k`; for a reverse move increment `from->startLayer` and `to->endLayer` by `k`.

- [ ] **Step 6: Run algorithm tests and verify GREEN**

Run:

```bash
make -C EdgeVisor dynamic-tpot-test && EdgeVisor/dynamic-tpot-test
```

Expected: `PASS dynamic_tpot_algorithm`.

- [ ] **Step 7: Commit Task 1**

```bash
git add EdgeVisor/src/dynamic/tpot_algorithm.hpp \
        EdgeVisor/src/dynamic/tpot_algorithm.cpp \
        EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp
git commit -m "feat: model multi-layer PP candidates"
```

### Task 2: Strict CLI and Environment Configuration

**Files:**
- Modify: `EdgeVisor/src/app.hpp`
- Modify: `EdgeVisor/src/app.cpp`
- Modify: `EdgeVisor/src/dynamic/dynamic_tpot.cpp`
- Test: `EdgeVisor/src/test/test_app_cli.cpp`

**Interfaces:**
- Consumes: `SchedulerConfig::maxPpLayerMove` from Task 1.
- Produces: `AppCliArgs::tpotMaxPpLayerMoveStr`, CLI `--tpot-max-pp-layer-move`, and strict `DLLAMA_TPOT_MAX_PP_LAYER_MOVE` loading.

- [ ] **Step 1: Write failing CLI/config tests**

Extend `clearTpotEnvironment()` to unset `DLLAMA_TPOT_MAX_PP_LAYER_MOVE`. Add helpers mirroring the PP ratio tests and assertions for:

```cpp
assert(loadSchedulerConfigFromEnvironment().maxPpLayerMove == 1u);
setenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE", "4", 1);
assert(loadSchedulerConfigFromEnvironment().maxPpLayerMove == 4u);
setenv("DLLAMA_TPOT_MAX_PP_LAYER_MOVE", "64", 1);
assert(loadSchedulerConfigFromEnvironment().maxPpLayerMove == 64u);
```

CLI tests must verify `--tpot-max-pp-layer-move 4` sets the environment to `4`. Add `rejectsMaxPpLayerMove(const char *)` and `rejectsEnvironmentMaxPpLayerMove(const char *)` helpers, then assert both helpers reject `0`, `65`, `-1`, `garbage`, `4junk`, and the empty string. Add a direct `parseArgs()` case with `--tpot-max-pp-layer-move` as the last argument and assert that the missing value throws.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
make -C EdgeVisor app-cli-test && EdgeVisor/app-cli-test
```

Expected: failure because the CLI option is unknown and environment values are not bounded/strictly parsed.

- [ ] **Step 3: Add and initialize the CLI field**

Add to `AppCliArgs`:

```cpp
char *tpotMaxPpLayerMoveStr;
```

Initialize it to `nullptr`, parse `--tpot-max-pp-layer-move`, and map it in `configureDynamicTpot()`:

```cpp
setTpotOverride("DLLAMA_TPOT_MAX_PP_LAYER_MOVE", args.tpotMaxPpLayerMoveStr);
```

- [ ] **Step 4: Add strict bounded integer validation**

Use `strtol` with `errno`, full end-pointer consumption, and explicit `[1,64]` checks for the CLI value. Add an equivalent strict environment helper in `dynamic_tpot.cpp`; replace the current `std::max(1, parseEnvInt(...))` load with a throwing bounded parse. Configuration exceptions remain caught in `DynamicTpotController::start()` before thread creation.

- [ ] **Step 5: Run CLI/config tests and verify GREEN**

Run:

```bash
make -C EdgeVisor app-cli-test && EdgeVisor/app-cli-test
```

Expected: exit `0`; only existing expected dynamic-TPOT auto-enable warnings may be printed.

- [ ] **Step 6: Commit Task 2**

```bash
git add EdgeVisor/src/app.hpp EdgeVisor/src/app.cpp \
        EdgeVisor/src/dynamic/dynamic_tpot.cpp \
        EdgeVisor/src/test/test_app_cli.cpp
git commit -m "feat: configure multi-layer PP migration"
```

### Task 3: Command, Rollback, and Stable Logging Propagation

**Files:**
- Modify: `EdgeVisor/src/dynamic/dynamic_tpot.cpp`
- Modify: `EdgeVisor/src/dynamic/tpot_log.hpp`
- Test: `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`

**Interfaces:**
- Consumes: `Candidate::layerCount`, `reversePpCandidate()`, and `ppCommandLayerCount()` from Task 1.
- Produces: runtime PP commands and rollback candidates that preserve the count, plus `pp_best_layer_count` diagnostics.

- [ ] **Step 1: Write failing propagation/log tests**

Before production edits, require:

```cpp
const std::string multiLog = tpot::formatPpCandidateLogFields(multiMove);
require(multiLog.find("pp_best_layer_count=" + std::to_string(multiMove.layerCount)) != std::string::npos,
    "PP candidate log includes selected layer count");
```

Keep the Task 1 helper tests that prove command and rollback counts are preserved.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
make -C EdgeVisor dynamic-tpot-test && EdgeVisor/dynamic-tpot-test
```

Expected: failure because `pp_best_layer_count` is absent.

- [ ] **Step 3: Propagate the count through the controller**

Replace the command hard-code with:

```cpp
cmd["layerCount"] = tpot::ppCommandLayerCount(c);
```

Replace the controller-local reverse implementation with `tpot::reversePpCandidate()` so rollback retains the count by contract. Do not alter pending-candidate or ACK serialization.

- [ ] **Step 4: Append the stable log field**

Append to `formatPpCandidateLogFields()`:

```cpp
<< " pp_best_layer_count=" << candidate.layerCount;
```

Do not rename or reorder existing fields.

- [ ] **Step 5: Run focused and combined tests**

Run:

```bash
make -C EdgeVisor dynamic-tpot-test app-cli-test
EdgeVisor/dynamic-tpot-test
EdgeVisor/app-cli-test
```

Expected: both test binaries exit `0`; algorithm output includes `PASS dynamic_tpot_algorithm`.

- [ ] **Step 6: Commit Task 3**

```bash
git add EdgeVisor/src/dynamic/dynamic_tpot.cpp \
        EdgeVisor/src/dynamic/tpot_log.hpp \
        EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp
git commit -m "feat: issue multi-layer PP migrations"
```

### Task 4: Documentation and Full Verification

**Files:**
- Modify: `EdgeVisor/docs/README_ENV_VARS.md`

**Interfaces:**
- Consumes: final CLI/environment behavior from Tasks 1-3.
- Produces: operator-facing usage and compatibility documentation.

- [ ] **Step 1: Document the control**

Document:

```text
--tpot-max-pp-layer-move K
DLLAMA_TPOT_MAX_PP_LAYER_MOVE=K
default=1, valid range=1..64
```

Explain that the scheduler evaluates all counts from one through the configured maximum, never empties the source stage, and that larger values increase migration state/KV cost. Include the experiment example `--tpot-max-pp-layer-move 4`.

- [ ] **Step 2: Run a clean full build and tests**

Run:

```bash
make -C EdgeVisor clean
make -C EdgeVisor dynamic-tpot-test app-cli-test dllama
EdgeVisor/dynamic-tpot-test
EdgeVisor/app-cli-test
test -x EdgeVisor/dllama
git diff --check
```

Expected: all commands exit `0`, both tests pass, `dllama` exists and is executable, and `git diff --check` has no output.

- [ ] **Step 3: Review compatibility evidence**

Confirm from tests and diff that default `K=1` reproduces the existing one-layer candidate, `set_pp_migration` still uses the existing protocol field, and TP functions are unchanged.

- [ ] **Step 4: Commit Task 4**

```bash
git add EdgeVisor/docs/README_ENV_VARS.md
git commit -m "docs: describe multi-layer PP migration"
```
