# Continuous PP Rebalance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit Root-applied PP migrations into scheduler state, allowing later PP moves to use the latest layout.

**Architecture:** Root publishes a monotonic applied generation after KV/state recovery and the layer switch complete. The scheduler overlays fresh measurements onto a committed logical topology, commits only a matching applied migration, then releases its PP candidate guard.

**Tech Stack:** C++11, nlohmann JSON, POSIX UDS, Make.

## Global Constraints

- Plan-controller acceptance is not a migration application acknowledgement.
- A PP layout is committed only after matching generation, route, and layer range.
- Static runtime roles remain the redundant-segment provision map; logical snapshots record active ownership.
- A rollback remains locked until its reverse migration has applied.
- Existing Root coverage/KV checks remain authoritative for every candidate.

---

## File Structure

- `EdgeVisor/src/app.hpp`, `app.cpp`: Root applied-generation state and accessors.
- `EdgeVisor/src/plan-controller.cpp`: `status.ppMigration` serialization.
- `EdgeVisor/src/dynamic/tpot_algorithm.*`: atomic logical move, capacity rebase, and guard commit.
- `EdgeVisor/src/dynamic/dynamic_tpot.cpp`: committed topology overlay and pending-action state machine.
- `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`: pure layout/guard regression tests.
- `EdgeVisor/src/test/test_app_cli.cpp`: status-facing regression where available.

### Task 1: Publish Root Application State

**Files:** `EdgeVisor/src/app.hpp`, `EdgeVisor/src/app.cpp`, `EdgeVisor/src/plan-controller.cpp`, relevant existing app test.

**Produces:** `appliedGeneration`, applied route, and applied layer list in `status.ppMigration`.

- [ ] **Step 1: Write a failing status/accessor test**

Require the public status path to expose `ppMigration.appliedGeneration`, defaulting to zero before a migration.

- [ ] **Step 2: Run RED verification**

Run `make -C EdgeVisor app-cli-test && EdgeVisor/app-cli-test`; expect compile/assertion failure because the field is absent.

- [ ] **Step 3: Add Root state and accessors**

Add an unsigned monotonic generation and snapshots of the successfully applied source node, target node, and layer list. Add `recordPpMigrationApplied()` which increments the generation and copies current migration data.

- [ ] **Step 4: Record only completed PP work**

Call the recorder only after the normal PP recovery path reports `applyOk`, which already requires KV/state preparation and `sendPendingLayerSwitchControlOnly()`. Do not record command acceptance, rejection, failure, or stage bypass.

- [ ] **Step 5: Serialize the fields**

Append `appliedGeneration`, `appliedFromNodeIndex`, `appliedToNodeIndex`, and `appliedLayers` to the existing status JSON without removing current fields.

- [ ] **Step 6: Run GREEN verification and commit**

Run `make -C EdgeVisor app-cli-test && EdgeVisor/app-cli-test`, then commit only Task 1 files with `feat: report applied PP migration generation`.

### Task 2: Commit Logical Layout Safely

**Files:** `EdgeVisor/src/dynamic/tpot_algorithm.hpp`, `tpot_algorithm.cpp`, `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`.

**Produces:** `bool applyPpMove(...)`, `rebasePpSoftCapacity(...)`, and `PpLayoutGuard::markCommitted(...)`.

- [ ] **Step 1: Write failing algorithm tests**

Use two contiguous snapshots and a valid forward candidate. Assert that applying it returns true, changes both ranges/counts, and makes the next boundary candidate start at the updated boundary. Assert invalid/non-adjacent candidates return false without mutation. Assert issued guard blocks, committed guard unblocks, and a formerly full soft capacity follows the new count while an existing lower cap remains lower.

- [ ] **Step 2: Run RED verification**

Run `make -C EdgeVisor dynamic-tpot-test && EdgeVisor/dynamic-tpot-test`; expect missing API/behavior failure.

- [ ] **Step 3: Implement atomic validation and update**

Return false before mutation for invalid kind, unknown/nonadjacent stages, inconsistent layer range, zero count, or an empty-source move. Otherwise update both topology ranges/counts and return true.

- [ ] **Step 4: Implement guard commit and capacity rebase**

`markCommitted()` removes the issued key and clears only the pending-layout lock. Rebase absent/formerly-full capacity to the new layer count; preserve any previous conservative cap below its old count.

- [ ] **Step 5: Run GREEN verification and commit**

Run `make -C EdgeVisor dynamic-tpot-test && EdgeVisor/dynamic-tpot-test`, then commit only Task 2 files with `feat: commit dynamic PP layout state`.

### Task 3: Consume Applied State in the Scheduler

**Files:** `EdgeVisor/src/dynamic/dynamic_tpot.cpp`, `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`.

**Produces:** next PP candidate is scored from committed ranges; no PP guard release before matching Root application.

- [ ] **Step 1: Write failing applied-match tests**

Cover a stale generation and mismatched route/layers leaving topology and guard unchanged, and a newer exact match committing topology/unlocking the guard.

- [ ] **Step 2: Run RED verification**

Run `make -C EdgeVisor dynamic-tpot-test && EdgeVisor/dynamic-tpot-test`; expect failure before matching/commit integration exists.

- [ ] **Step 3: Persist and overlay topology**

Add committed layout, initialization flag, and last applied generation to controller runtime. Initialize from measured snapshots once; each window overlays committed `startLayer`, `endLayer`, `nLayers`, and rebased capacity before PP scoring.

- [ ] **Step 4: Match, commit, and unlock**

At PP issue capture current generation and expected route/layers. While pending, require a strictly newer exact applied status. Apply the candidate to committed layout, rebase both capacities, record generation, and call `markCommitted()`. Before that point retain the guard and issue no later PP candidate.

- [ ] **Step 5: Sequence rollback**

After an applied forward move degrades, issue its reverse as a new pending action with a fresh generation baseline. Commit the reverse only after its applied status, then apply risk penalty and unlock. If the forward move is not yet applied, wait rather than issuing a reverse.

- [ ] **Step 6: Run GREEN verification and commit**

Run `make -C EdgeVisor dynamic-tpot-test && EdgeVisor/dynamic-tpot-test`, then commit only Task 3 files with `feat: allow sequential applied PP migrations`.

### Task 4: Integration Verification

- [ ] **Step 1: Build both regression targets**

Run `make -C EdgeVisor app-cli-test dynamic-tpot-test`; expect exit 0.

- [ ] **Step 2: Execute both regression binaries**

Run `EdgeVisor/app-cli-test && EdgeVisor/dynamic-tpot-test`; expect exit 0 and `PASS dynamic_tpot_algorithm`.

- [ ] **Step 3: Inspect scope**

Run `git diff --check HEAD~3..HEAD && git status --short`; expect no whitespace errors and no unintended files.
