# Stage Bypass Verification Design

## Goal

Make a PP stage bypass an acknowledged, verified topology transition. A
dynamic TPOT scheduler may derive PP2 only after root and the affected workers
agree that the bypass has switched layer ownership and PP routing.

## Transition Identity and Telemetry

Root allocates a monotonically increasing bypass generation before it sends a
layer-switch control packet. Every stage-bypass layer-switch packet carries
that generation. Root records the same value as `rootApplyGeneration` only
after its local `applyPpStageBypass()` succeeds.

Workers ACK the generation after they have applied all packets for that
transition, changed local ownership, updated their local partition routing,
and applied the PP-sync role. The ACK contains the worker node, its active
stage, ejected and target stages, complete layer list, active stage chain, and
role observations needed for validation.

The root status response exposes a `stageBypass` object with the root apply
generation, command identity, full applied layer list, expected/received ACK
workers, ACK details, active stage chain, `verifiedGeneration`, and a terminal
failure reason when present. `appliedGeneration` remains a root-local apply
fact; only `verifiedGeneration` grants scheduler or experiment progression.

For a 2 -> 3 bypass, verification requires these ACKs:

- Stage 1 / node .15: next route is Stage 3.
- Stage 2 / node .18: PP synchronization is disabled and Stage 2 is absent
  from the active chain.
- Stage 3 / node .16: it owns the entire ejected layer range and its previous
  route is Stage 1.

Every ACK must carry the same generation and exact bypass identity. Duplicate,
stale, mismatched, or incomplete acknowledgements cannot advance verification.

## Topology Fence and Scheduler State

Seeing a newer root-applied bypass creates a topology fence that retains the
telemetry and freezes all new PP and TP candidate issuance. It does not discard
an existing pending PP migration.

If PP is pending, the controller first waits for its explicit terminal state:

1. applied and matching telemetry: commit the PP logical layout;
2. rejected or cancelled: clear the pending local action without committing it;
3. timeout: record a control-plane failure and stop the scheduler round.

Only then may the controller validate and commit the retained bypass telemetry.
The fence remains active until the matching bypass `verifiedGeneration` is
observed. Verification timeout is a control-plane failure; it must stop the
round rather than silently resume scheduling.

## Bypass Commit Validation

The controller changes `committedPpLayout` only when all conditions hold:

1. ejected and target stages both occur exactly once in the committed layout;
2. they are adjacent in `activeStageChain`, and target direction is either the
   immediate predecessor or immediate successor;
3. `appliedLayers` is the complete, strictly consecutive interval
   `[ejected.startLayer, ejected.endLayer)` in ascending order;
4. the joined target/ejected ranges are contiguous; the resulting
   `startLayer`, `endLayer`, and `nLayers == endLayer - startLayer` agree.

The merge derives layer count from the merged endpoints, then checks total
layout coverage before updating the layout. It never uses `to.nLayers +=
from.nLayers` as authoritative state. Invalid telemetry leaves all scheduler
state unchanged and remains observable as a rejected control-plane event.

On a valid commit, the controller removes the ejected stage from EWMA,
capacity, and risk maps; invalidates PP candidate guards; rebuilds the active
chain from the committed layout; and waits for worker verification before
scoring PP2. Snapshot and candidate construction exclude the ejected stage.

## Root and Worker Protocol

Introduce a dedicated bypass-ACK frame rather than overloading KV ACK packets.
The frame has a versioned header and includes bypass generation, bypass stage
identity, reporting node/stage, active-chain length and members, complete
ejected-layer interval, and role flags (`prevRerouted`, `ejectedExited`,
`targetOwned`, `ppSyncDisabled`). Root expects the participants selected from
the pre-bypass active chain and records one validated ACK per worker.

Workers accumulate multi-layer switch packets by generation. Once all layers
for the generation are applied, they update `applyPpStageBypass()` and emit the
ACK on the root socket. Root consumes frames without confusing them for perf or
KV ACK traffic and advances `verifiedGeneration` only after the participant
role checks and active-chain checks succeed.

## Experiment Tooling

The four-Jetson orchestrator waits for `verifiedGeneration` to reach the root
apply generation and for the status active chain to be `[0, 1, 3]`. It records
and validates the PP1 and PP2 layer lists before declaring either phase
verified. Any status/control timeout writes a failed event and exits nonzero.

The recorder emits independent PP1, bypass, and PP2 action windows, each with
baseline, onset, issue, apply, verify, recovery, and metrics, instead of only
adding event labels to token rows.

Network disturbance uses directional traffic-control filters for `.13 -> .18`
and `.18 -> .13`. The orchestrator snapshots only the handles/filters it adds
and removes/restores those on cleanup; it never replaces the interface root
qdisc.

## Tests

Integration coverage drives the actual controller path through PP1 applied,
accepted bypass, root/worker verification, committed layout `[0,1,3]`, stage 2
exclusion from snapshots/candidates, and PP2 `3 -> 1`. Negative cases cover
unknown stages, non-adjacent/direction-invalid bypasses, mismatched layer lists,
pending PP fencing, PP terminal failure, and bypass verification timeout.
