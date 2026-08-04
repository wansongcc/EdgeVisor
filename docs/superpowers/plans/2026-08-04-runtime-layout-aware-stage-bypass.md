# Runtime-layout-aware Stage Bypass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage Bypass derive its layer range from the committed runtime primary-owner layout after PP migration, while preserving the static redundancy provision map and existing ACK/verified-generation safety.

**Architecture:** Keep the startup `RuntimeStageLayerPlan` as an immutable provision map for normal PP/rollback admission and segment classification. Add a mutable Root-owned runtime snapshot that records current primary ownership and is changed only after a completed PP switch or fully verified bypass. Extract pure runtime-layout and ACK validation helpers so admission, atomic transitions, negative cases, and protocol assertions are directly testable.

**Tech Stack:** C++11, existing `RuntimeStageLayerPlan`/`NnUnevenPartitionPlan`, nlohmann JSON, GNU Make, existing app and dynamic TPOT unit-test binaries.

## Global Constraints

- Runtime layout is the only source for bypass required/applied layers; do not use static `NnStageConfig::startLayer/endLayer` for bypass coverage or switch construction.
- Only support a non-empty, strictly continuous, in-range primary layer interval for the ejected stage.
- Require no pending PP migration and require the prior bypass generation to be verified before accepting another bypass.
- Require unique primary ownership for every model layer and runtime redundant coverage on every required layer.
- Keep the existing fixed layer-switch wire ABI and ACK + `verifiedGeneration` + TopologyFence consistency requirements.
- Keep the affected ACK participants as previous stage, ejected stage, and target stage.
- On any rejection, leave layout, route, executor, and pending switch state unchanged and expose an explicit failure reason.
- Preserve unrelated dirty files and do not stage `.codegraph`, `EdgeVisor/scripts`, or existing plans.

---

## File Structure

- `EdgeVisor/src/app.hpp`: public pure-helper declarations, ACK validation interface, Root runtime snapshot fields/accessors.
- `EdgeVisor/src/app.cpp`: runtime layout validation/transition helpers, Root snapshot lifecycle, bypass admission, ACK validation, and worker ACK range reporting.
- `EdgeVisor/src/plan-controller.cpp`: serialize `runtimeRequiredLayers` and `layerCount` in `status.stageBypass`.
- `EdgeVisor/src/dynamic/dynamic_tpot.cpp`: require status runtime range to match applied range before committing the logical bypass layout.
- `EdgeVisor/src/test/test_app_cli.cpp`: runtime-layout, admission/generation, ACK, rejection-immutability, and status-facing helper regression tests.
- `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`: post-PP logical layout `[38,39)` → bypass `[38]` → target `[38,40)` and PP2 eligibility tests.
- `EdgeVisor/Makefile`: only if a new test binary is needed; prefer the existing `app-cli-test` and `dynamic-tpot-test` targets.

## Interfaces Produced Across Tasks

Task 1 produces these pure helpers for Root and tests:

```cpp
bool validateRuntimeStageLayerPlan(
    const RuntimeStageLayerPlan &layout,
    std::string *reason = nullptr);

bool currentOwnedLayers(
    const RuntimeStageLayerPlan &layout,
    NnUint stageIndex,
    std::vector<NnUint> &layers,
    std::string *reason = nullptr);

bool resolveStageBypassLayers(
    const NnUnevenPartitionPlan *plan,
    const RuntimeStageLayerPlan *runtimeLayout,
    NnUint ejectedStageIndex,
    NnUint targetStageIndex,
    std::vector<NnUint> &layers,
    std::string *reason = nullptr);

bool applyRuntimeLayerOwnershipMove(
    RuntimeStageLayerPlan &layout,
    NnUint sourceStageIndex,
    NnUint targetStageIndex,
    const std::vector<NnUint> &layers,
    bool requireRedundantTarget,
    std::string *reason = nullptr);
```

Task 2 produces a pure generation gate used by command admission:

```cpp
bool stageBypassGenerationReady(
    bool pendingPpMigration,
    unsigned long long appliedGeneration,
    unsigned long long verifiedGeneration,
    std::string *reason = nullptr);
```

Task 3 produces a pure ACK field validator used by `consumeStageBypassAckFrame()`:

```cpp
bool validateStageBypassAck(
    const LlmStageBypassAckPacket &ack,
    const std::vector<NnUint> &ackChain,
    NnUint expectedNode,
    NnUint expectedStage,
    NnUint expectedRoleFlags,
    unsigned long long expectedGeneration,
    NnUint expectedEjectedStage,
    NnUint expectedTargetStage,
    const std::vector<NnUint> &expectedLayers,
    const std::vector<NnUint> &expectedChain,
    std::string *reason = nullptr);
```

The implementation must clear output vectors on failed pure-helper calls and mutate
runtime layout only through a validated temporary copy.

### Task 1: Add runtime layout helpers and prove the PP1 → bypass range

**Files:**
- Modify: `EdgeVisor/src/app.hpp`
- Modify: `EdgeVisor/src/app.cpp`
- Test: `EdgeVisor/src/test/test_app_cli.cpp`

**Interfaces:** Produces the four runtime-layout helpers above. The helpers accept a
mutable runtime role matrix, but never consult `NnStageConfig::startLayer/endLayer` to
derive ejected required layers.

- [ ] **Step 1: Write the failing tests**

Add a four-stage fixture in `test_app_cli.cpp` with model layers 40 and static ranges
`[0,18)`, `[18,34)`, `[34,39)`, `[39,40)`. Set
`DLLAMA_RUNTIME_REDUNDANT_BOUNDARY_LAYERS=4`, build the initial runtime plan, and copy
it as the committed snapshot. Add assertions equivalent to:

```cpp
std::string reason;
assert(validateRuntimeStageLayerPlan(runtime, &reason));
assert(applyRuntimeLayerOwnershipMove(
    runtime, 2u, 1u, std::vector<NnUint>{34u, 35u, 36u, 37u}, false, &reason));

std::vector<NnUint> owned;
assert(currentOwnedLayers(runtime, 2u, owned, &reason));
assert((owned == std::vector<NnUint>{38u}));

std::vector<NnUint> required;
assert(resolveStageBypassLayers(&plan, &runtime, 2u, 3u, required, &reason));
assert((required == std::vector<NnUint>{38u}));
```

Add negative assertions for target role 38 changed to `RUNTIME_LAYER_DISABLED`, a
non-contiguous Stage2 owner set, empty Stage2 owner set, duplicate primary owners,
and an out-of-range role matrix. Every failure must provide a non-empty reason and
leave the input snapshot unchanged.

- [ ] **Step 2: Run the test to verify RED**

Run: `make -C EdgeVisor app-cli-test`

Expected: compile failure naming the missing runtime-layout helper declarations or
definitions. Do not modify production code before observing this failure.

- [ ] **Step 3: Implement the minimal helpers**

Implement the following behavior in `app.cpp`:

```cpp
// validateRuntimeStageLayerPlan:
// - reject nLayers==0, nStages==0, a short role vector, or an oversized role vector;
// - for every layer [0,nLayers), count RUNTIME_LAYER_PRIMARY across all stages;
// - require exactly one owner and report the layer number on failure.

// currentOwnedLayers:
// - scan exactly one stage row for RUNTIME_LAYER_PRIMARY;
// - reject no layers with "current ejected-stage layers are empty";
// - require layers[i] == layers[0] + i, otherwise report
//   "current ejected-stage layers are non-contiguous".

// resolveStageBypassLayers:
// - validate the runtime snapshot first;
// - find both stage indices and require target to be the current prev/next neighbor;
// - call currentOwnedLayers for ejected;
// - require target row == RUNTIME_LAYER_REDUNDANT for every returned layer;
// - return only the current owner interval.

// applyRuntimeLayerOwnershipMove:
// - copy layout to a candidate;
// - require a non-empty consecutive layer list and source PRIMARY for every layer;
// - if requireRedundantTarget, require target REDUNDANT; otherwise require target
//   is not already PRIMARY;
// - set source to DISABLED and target to PRIMARY in the candidate;
// - validate the candidate's unique full-layer ownership, then swap it into layout.
```

Use exact failure strings for the user-visible cases: `current ejected-stage layers are
empty`, `current ejected-stage layers are non-contiguous`, and
`target lacks redundant runtime layer 38` for the example fixture, with the actual
layer number substituted for other inputs. Malformed layouts must report a concrete
unique-owner/coverage reason.

- [ ] **Step 4: Run the tests to verify GREEN**

Run: `make -C EdgeVisor app-cli-test` then `EdgeVisor/app-cli-test`

Expected: the new PP1 snapshot returns `[38]`, static `[34,39)` is never returned by
`resolveStageBypassLayers`, and every negative case keeps the original snapshot.

- [ ] **Step 5: Commit the helper slice**

```bash
git add EdgeVisor/src/app.hpp EdgeVisor/src/app.cpp EdgeVisor/src/test/test_app_cli.cpp
git commit -m "test: define runtime stage bypass layer ownership"
```

### Task 2: Add Root runtime snapshot lifecycle and bypass admission

**Files:**
- Modify: `EdgeVisor/src/app.hpp`
- Modify: `EdgeVisor/src/app.cpp`
- Modify: `EdgeVisor/src/plan-controller.cpp`
- Test: `EdgeVisor/src/test/test_app_cli.cpp`

**Interfaces:** Consumes Task 1 helpers. Produces a Root-owned mutable snapshot, a
`getStageBypassRuntimeRequiredLayers()` accessor, and the generation gate helper.

- [ ] **Step 1: Write the failing generation/status tests**

Add assertions for the pure gate:

```cpp
std::string reason;
assert(stageBypassGenerationReady(false, 0ull, 0ull, &reason));
assert(!stageBypassGenerationReady(true, 0ull, 0ull, &reason));
assert(reason == "bypass deferred: pending PP generation is unverified");
assert(!stageBypassGenerationReady(false, 2ull, 1ull, &reason));
assert(reason == "bypass deferred: pending PP generation is unverified");
assert(stageBypassGenerationReady(false, 2ull, 2ull, &reason));
```

Add a status-facing fixture assertion through the existing `status` response builder
that the serialized bypass object contains `runtimeRequiredLayers` and `layerCount`
alongside `appliedLayers`, generations, and `failureReason`.

- [ ] **Step 2: Run the tests to verify RED**

Run: `make -C EdgeVisor app-cli-test`

Expected: compile failure for the missing generation helper/accessor or a failing
status assertion because the new field is absent.

- [ ] **Step 3: Add the Root snapshot and admission wiring**

Add these Root fields in `app.hpp`:

```cpp
RuntimeStageLayerPlan runtimeLayoutSnapshot;
std::vector<NnUint> stageBypassRuntimeRequiredLayers;
```

Initialize `runtimeLayoutSnapshot` as a copy of the startup `runtimePlan` in the Root
constructor. Keep `runtimePlan` immutable for static provision use.

Change `recordPpMigrationApplied()` to resolve source/target stage indices from the
completed migration route and apply `applyRuntimeLayerOwnershipMove(..., false, ...)`
to a temporary snapshot before swapping it. This is the only point at which a
completed PP switch updates the current primary-owner snapshot.

In the stage-bypass command branch:

1. Inspect pending KV transfers, waiting KV ACK, and pending layer-switch state under
   `kvTransferMutex`.
2. Call `stageBypassGenerationReady()` with the pending state and the last bypass
   applied/verified generations.
3. Call `resolveStageBypassLayers(plan, &runtimeLayoutSnapshot, ejected, target, ...)`.
4. Store exactly the returned vector in `stageBypassRuntimeRequiredLayers` and
   `migrationLayers`/`pendingLayerSwitchLayers`.
5. Clear `stageBypassFailureReason` only after admission succeeds. On any failure set
   the reason, log the concrete message, consume the command as rejected, and do not
   set any pending bypass or executor state.

Remove the static loop over `ejectedStage->startLayer .. endLayer` from bypass command
construction. The existing `sendPendingLayerSwitchControlOnly()` then sends the one
runtime-derived batch without reconstructing its range.

Add the accessor and serialize:

```cpp
bypass["runtimeRequiredLayers"] = json::array({ ... });
bypass["appliedLayers"] = json::array({ ... });
bypass["layerCount"] = inference_->getStageBypassAppliedLayers().size();
```

Keep `rootApplyGeneration`, `verifiedGeneration`, active chain, ACK node arrays, and
failure reason unchanged. Ensure a rejected command leaves the previous successful
status snapshot intact except for its explicit failure reason.

- [ ] **Step 4: Run the tests to verify GREEN**

Run: `make -C EdgeVisor app-cli-test` then `EdgeVisor/app-cli-test`

Expected: generation gating and status-field assertions pass; the Root code path no
longer constructs a bypass list from static stage endpoints.

- [ ] **Step 5: Commit the Root admission slice**

```bash
git add EdgeVisor/src/app.hpp EdgeVisor/src/app.cpp EdgeVisor/src/plan-controller.cpp EdgeVisor/src/test/test_app_cli.cpp
git commit -m "feat: admit stage bypass from runtime ownership"
```

### Task 3: Enforce exact range and generation on all bypass ACKs

**Files:**
- Modify: `EdgeVisor/src/app.hpp`
- Modify: `EdgeVisor/src/app.cpp`
- Test: `EdgeVisor/src/test/test_app_cli.cpp`

**Interfaces:** Produces and consumes `validateStageBypassAck()`; preserves the existing
`LlmStageBypassAckPacket` and fixed layer-switch ABI.

- [ ] **Step 1: Write the failing ACK tests**

Build a valid ACK for generation 1, ejected 2, target 3, stage 3, target role, chain
`[0,1,3]`, and layers `[38]`. Assert the validator accepts it and rejects one mutation
at a time:

```cpp
assert(validateStageBypassAck(validAck, chain, 16u, 3u, targetRole, 1ull, 2u, 3u,
    std::vector<NnUint>{38u}, chain, &reason));

LlmStageBypassAckPacket bad = validAck;
bad.bypassGeneration = 2u;
assert(!validateStageBypassAck(bad, chain, 16u, 3u, targetRole, 1ull, 2u, 3u,
    std::vector<NnUint>{38u}, chain, &reason) && !reason.empty());

bad = validAck;
bad.roleFlags = LLM_STAGE_BYPASS_ACK_EJECTED_EXITED;
assert(!validateStageBypassAck(bad, chain, 16u, 3u, targetRole, 1ull, 2u, 3u,
    std::vector<NnUint>{38u}, chain, &reason) && !reason.empty());

bad = validAck;
bad.startLayer = 34u;
bad.endLayer = 39u;
bad.layerCount = 5u;
assert(!validateStageBypassAck(bad, chain, 16u, 3u, targetRole, 1ull, 2u, 3u,
    std::vector<NnUint>{38u}, chain, &reason) && !reason.empty());

const std::vector<NnUint> wrongChain{0u, 1u, 2u, 3u};
assert(!validateStageBypassAck(validAck, wrongChain, 16u, 3u, targetRole, 1ull, 2u, 3u,
    std::vector<NnUint>{38u}, chain, &reason) && !reason.empty());
```

Use the actual C++ test style in `test_app_cli.cpp` rather than introducing a testing
framework or mock network.

- [ ] **Step 2: Run the tests to verify RED**

Run: `make -C EdgeVisor app-cli-test`

Expected: compile failure for the missing ACK validator.

- [ ] **Step 3: Implement and wire exact ACK validation**

Implement the validator to check magic/version, expected node/stage, generation,
ejected/target identity, exact role flags, exact `startLayer`, `endLayer`, and
`layerCount` derived from the continuous expected vector, plus exact active chain.

In `consumeStageBypassAckFrame()` call it for each participant after selecting the
participant expectation. Keep duplicate/unknown-node detection and set explicit
failure reasons for stale, duplicate, malformed, role, range, and chain errors.
Require the range/count on previous, ejected, and target ACKs to equal the applied
runtime interval; do not special-case the target only.

In the worker ACK builder, once a stage-bypass batch is seen and `bypassLayers` is
sorted, set `startLayer/endLayer/layerCount` to that batch for every participant.
Continue reporting participant-specific role flags and the post-apply active chain.

In `tryVerifyStageBypassAcks()`, after all expected ACKs exist, apply
`applyRuntimeLayerOwnershipMove(runtimeLayoutSnapshot, ejected, target, appliedLayers,
true, ...)` on a temporary snapshot. Only swap it and set
`stageBypassVerifiedGeneration = stageBypassPendingGeneration` when that transition
and the full-layout validation succeed. A failed snapshot commit must leave verified
generation unchanged and set `failureReason`.

- [ ] **Step 4: Run the tests to verify GREEN**

Run: `make -C EdgeVisor app-cli-test` then `EdgeVisor/app-cli-test`

Expected: valid ACKs pass, every wrong-field mutation remains unverified, and the
runtime snapshot changes only at the all-ACK verification point.

- [ ] **Step 5: Commit the ACK slice**

```bash
git add EdgeVisor/src/app.hpp EdgeVisor/src/app.cpp EdgeVisor/src/test/test_app_cli.cpp
git commit -m "fix: verify runtime stage bypass ranges on every ACK"
```

### Task 4: Require runtime range equality in the TPOT topology commit

**Files:**
- Modify: `EdgeVisor/src/dynamic/dynamic_tpot.cpp`
- Modify: `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`

**Interfaces:** Consumes `status.stageBypass.runtimeRequiredLayers` and
`status.stageBypass.appliedLayers`; preserves the existing logical
`commitStageBypassLayout()` API.

- [ ] **Step 1: Write the failing post-PP layout test**

Add a logical layout fixture representing the committed PP1 result:

```cpp
std::vector<tpot::StageSnapshot> layout;
layout.push_back(stage(0u, 0u, 0u, 18u, 10.0));
layout.push_back(stage(1u, 1u, 18u, 38u, 10.0));
layout.push_back(stage(2u, 2u, 38u, 39u, 10.0));
layout.push_back(stage(3u, 3u, 39u, 40u, 10.0));
std::vector<uint32_t> chain;
assert(tpot::commitStageBypassLayout(layout, 2u, 3u, {38u}, &chain));
assert(chain == std::vector<uint32_t>({0u, 1u, 3u}));
assert(layout[2].stageIndex == 3u && layout[2].startLayer == 38u && layout[2].endLayer == 40u);
```

Add a negative case proving `[34,35,36,37,38]` cannot commit against the post-PP
Stage2 `[38,39)` snapshot, and a case proving a non-contiguous applied list fails.

- [ ] **Step 2: Run the test to verify RED**

Run: `make -C EdgeVisor dynamic-tpot-test`

Expected: the new post-PP assertions fail because the test fixture/helper does not yet
cover the runtime interval contract.

- [ ] **Step 3: Tighten scheduler status admission**

In `commitAppliedStageBypass()` parse both arrays and require:

```cpp
runtimeRequiredLayers == appliedLayers
```

before calling `commitStageBypassLayout()`. Keep the existing generation and verified
checks, pending-action ordering, exact logical range check, and active-chain rebuild.
If the arrays differ or the required field is missing, append a concrete
`control_plane_failed`/decision reason and leave `committedPpLayout` unchanged. The
The successful path must still erase ejected-stage EWMA/capacity/risk state and release the
TopologyFence only after verified generation is observed.

- [ ] **Step 4: Run the tests to verify GREEN**

Run: `make -C EdgeVisor dynamic-tpot-test` then `EdgeVisor/dynamic-tpot-test`

Expected: the post-PP bypass commits `[38]` into Stage3 `[38,40)`, active chain is
`[0,1,3]`, the old `[34..37]` range is never reintroduced, and PP2 candidate logic
sees no Stage2.

- [ ] **Step 5: Commit the scheduler slice**

```bash
git add EdgeVisor/src/dynamic/dynamic_tpot.cpp EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp
git commit -m "fix: commit stage bypass from verified runtime range"
```

### Task 5: Run full targeted regression and requirement checks

**Files:**
- Test: `EdgeVisor/src/test/test_app_cli.cpp`
- Test: `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`
- No source changes are expected in this verification task; any required source fix
  returns to the owning task before this task is re-run.

**Interfaces:** Verifies the end-to-end contract through the existing build targets and
the pure helper tests.

- [ ] **Step 1: Run the focused red/green suite**

Run:

```bash
make -C EdgeVisor app-cli-test dynamic-tpot-test
EdgeVisor/app-cli-test
EdgeVisor/dynamic-tpot-test
```

Expected: both binaries exit zero, with runtime owner `[38]`, required/applied `[38]`,
active chain `[0,1,3]`, complete ACK validation, and explicit negative reasons.

- [ ] **Step 2: Build the production binary**

Run: `make -C EdgeVisor dllama`

Expected: exit zero with no C++ warnings promoted to errors and no layer-switch ABI
static assertion failures.

- [ ] **Step 3: Re-run the existing application CLI regression**

Run: `EdgeVisor/app-cli-test`

Expected: existing runtime redundancy, active-chain PP adjacency, command decoding,
and `15/15/5/5` bypass assertions remain green.

- [ ] **Step 4: Inspect the final diff and dirty-file boundary**

Run:

```bash
git status --short
git diff --check HEAD~4..HEAD
git diff --stat HEAD~4..HEAD
```

Confirm only the intended source/tests/spec/plan commits are present from this work;
do not alter or stage the pre-existing `.codegraph`, `EdgeVisor/scripts`, or other
unrelated files.

- [ ] **Step 5: Record the verification boundary**

Do not make further source changes in this task. If a command fails, return to the
task that owns the failing file, add its failing regression test first, and repeat that
task's red/green cycle before rerunning this complete suite. Do not claim completion
until the fresh focused suite and production build both report exit code zero.

## Plan Self-Review

- Spec coverage: runtime ownership source, PP1 update, continuous-range validation,
  redundancy admission, generation gate, no-mutation rejection, all three ACK roles,
  status fields, topology fence, PP2, and `15/15/5/5` regression are covered by Tasks 1–5.
- Placeholder scan: no TBD/TODO/“implement later” steps are used; all commands and
  expected outcomes are explicit.
- Type consistency: Task 1 helper signatures feed Task 2 Root admission and Task 3
  ACK/runtime commit; Task 4 consumes the status field emitted by Task 2.
- Scope: no new wire ABI, no non-contiguous bypass, no multi-stage bypass, and no
  unrelated refactor are included.
