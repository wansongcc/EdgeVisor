# Continuous PP Rebalance Design

## Goal

Allow the dynamic TPOT scheduler to perform more than one successful PP layer
migration during one inference. Each later decision must be evaluated against
the PP layout that was actually applied by the Root runtime, rather than the
static partition plan.

## Success Boundary

Issuing `set_pp_migration` only confirms that the plan controller accepted a
command. It does not prove that KV preparation and the layer-switch control
packet completed. A PP move is committed only after Root reports a new
monotonic `appliedGeneration` following a successful KV/state preparation and
layer switch.

The status response exposes the generation together with the applied route and
layer range. This makes the scheduler reject stale or mismatched acknowledgements.

## Scheduler Layout State

The scheduler owns a mutable logical PP layout made of `StageSnapshot` ranges
and counts. Window collection still obtains timing, EWMA, risk, and node data
from the static stage identities; before scoring, those measurements are
overlaid onto the committed logical ranges and capacities.

When the pending PP action observes its matching applied generation:

1. apply the candidate to the committed layout;
2. rebase source and target soft capacities to their updated layer counts while
   preserving an existing conservative cap;
3. retain the logical owner of every moved layer in the committed ranges;
4. clear the pending PP layout guard, so the next candidate is derived from the
   new boundary.

The runtime's static `RuntimeStageLayerPlan` remains a provision map for
primary/redundant graph segments. It is not overwritten with active ownership:
doing so would invalidate the complementary segments needed for a rollback.
The new logical layout is the active ownership record; the static roles are
used to verify that each candidate remains backed by source/target segments.

## KV and Redundancy Admission

The Root increments `appliedGeneration` only after `sendPendingLayerSwitchControlOnly()` succeeds. For paths with state transfer this already follows KV transfer/ACK completion; for shadow/recompute paths it follows their respective recovery success. The controller records the generation returned in `status` when it issues a PP command and only commits the exact matching route, range, and count.

Before issuing a later migration, Root's existing `resolvePpMigrationLayers()`
remains authoritative for redundant coverage and layer/KV admission. The
scheduler also keeps one PP command pending at a time. Thus continuous moves
are supported only within the prebuilt runtime redundant-boundary span; an
unsupported next boundary is rejected safely instead of using stale layout.

## Rollback

A degraded window issues the reverse candidate and keeps the layout guard
locked. The original forward candidate is committed only if its applied
generation was observed. A rollback receives its own applied generation; after
that generation is observed the reverse move is committed, the risk/capacity
penalty is applied, and the guard is released. A rollback that never applies
cannot unlock the layout.

## Tests

Pure algorithm tests cover committing a PP move, recomputing the next boundary,
and releasing the guard only after commit. Controller/runtime tests cover
status serialization of the applied generation and ensure it increments only
after a successful layer switch. Existing PP range and CLI tests continue to
cover redundant segment validation and command shape.

## Out of Scope

- Creating new runtime graph segments or transferring weights beyond the
  configured redundant-boundary span.
- Concurrent PP migrations.
- Changing TP head/FFN reallocation rules.
