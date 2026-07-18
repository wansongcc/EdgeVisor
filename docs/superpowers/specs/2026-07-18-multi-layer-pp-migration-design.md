# Multi-Layer PP Migration Design

## Goal

Extend the dynamic TPOT scheduler so one PP migration may move multiple contiguous boundary layers when that produces a better predicted TPOT. Preserve the current single-layer behavior unless the operator explicitly raises the configured maximum.

## Compatibility

- `SchedulerConfig::maxPpLayerMove` remains `1` by default.
- Add root CLI option `--tpot-max-pp-layer-move <K>`.
- The CLI option maps to the existing `DLLAMA_TPOT_MAX_PP_LAYER_MOVE` environment variable.
- Valid values are integers in `[1,64]`.
- Existing commands that omit both controls continue to evaluate and issue only one-layer PP migrations.
- TP candidate scoring and TP movement are unchanged.

The existing `maxPpLayerMove` check currently behaves like a minimum source-stage layer guard. This implementation corrects its meaning: it becomes the maximum number of layers considered in one PP migration. The source-stage safety rule is expressed independently as `sourceLayers - k >= 1`.

## Candidate Model

Add `uint32_t layerCount = 1u` to `Candidate`.

For every adjacent stage pair, evaluate both migration directions. For each direction, enumerate:

```text
k = 1 .. min(maxPpLayerMove, sourceLayers - 1)
```

Only contiguous layers at the source-stage boundary are eligible. The boundary layer index remains:

- `source.endLayer - k` when moving to the next stage;
- `source.startLayer` when moving to the previous stage.

`layerIndex` identifies the first layer in the contiguous migrated range and `layerCount` identifies its length.

## Gain Calculation

For a candidate that moves `k` layers from source to target:

```text
source_out(k)
  = measured_boundary_layer_ms
  + stageCost(source, n - 1) - stageCost(source, n - k)

target_in(k)
  = stageCost(target, m + k) - stageCost(target, m)

gain(k)
  = source_out(k)
  - target_in(k)
  - k * migration_cost_per_token
  - k * pp_risk_margin_ms
  - k * target_risk_penalty * max(1, target.avg_layer_ms)
```

Here `n` and `m` are the current source and target layer counts. When no measured boundary time is available, the first source layer uses the existing modeled one-layer delta. For `k=1`, the formula is behaviorally identical to the current model.

The acceptance threshold remains:

```text
max(minPpGainMs, ppGainRatio * (source.stageTimeMs + target.stageTimeMs))
```

The threshold does not scale with `k`; the accumulated migration cost and risk already make larger candidates harder to accept.

The scheduler chooses the valid PP candidate with the largest `gainMs`. If every structurally eligible candidate is rejected, it retains the rejected candidate with the largest `gainMs` for diagnostics.

## Command and Runtime State

- `makePpCommand()` sends `candidate.layerCount` as `set_pp_migration.cmd.layerCount` instead of hard-coding `1`.
- The existing Plan Controller and Root migration path remain responsible for expanding the count into a contiguous boundary-layer list and performing batched KV transfer.
- `applyPpMove()` updates stage layer counts and boundaries by `candidate.layerCount`.
- A rollback reverses the route while preserving `layerCount`, so the same contiguous range is returned.
- The existing single-pending-candidate rule remains: no new PP migration is issued while the previous migration is awaiting acknowledgement or verification.

## Logging

Append the stable field below to every existing best-PP-candidate field group:

```text
pp_best_layer_count=<K>
```

The field is emitted for accepted and rejected candidates. The command issue/verification logs also include the selected layer count through the existing candidate and migration command context.

## Error Handling

- CLI values that are missing, non-integer, partially parsed, below `1`, or above `64` fail argument parsing with a clear message.
- `DLLAMA_TPOT_MAX_PP_LAYER_MOVE` receives the same strict validation before the scheduler thread starts.
- A candidate is not generated if moving `k` layers would leave the source stage empty.
- Existing runtime checks for full target weights, adjacent stages, KV acknowledgements, and command serialization remain authoritative.

## Tests

Algorithm tests must demonstrate:

- default configuration selects only `layerCount=1`;
- a configured maximum greater than one enumerates multiple counts and can select an interior optimum rather than always selecting the maximum;
- source stages retain at least one layer;
- forward and reverse candidate ranges use correct `layerIndex` and `layerCount`;
- migration cost and risk scale with `k`;
- `applyPpMove()` updates counts and boundaries by `k`;
- TP candidate behavior is unchanged.

CLI/configuration tests must demonstrate:

- default value `1`;
- valid CLI/environment values, including `4` and `64`;
- rejection of `0`, `65`, negative, non-numeric, and partially parsed values.

Command/log tests must demonstrate:

- `set_pp_migration.layerCount` equals the selected candidate count;
- reverse candidates preserve the count;
- `pp_best_layer_count` is present for valid and rejected candidates.

## Out of Scope

- Non-contiguous layer migration.
- Moving layers across non-adjacent stages in one command.
- Concurrent PP migration commands.
- A new GPU-memory estimator or admission controller.
- Changes to TP head/FFN movement.
- Automatically enabling multi-layer migration by changing the default above `1`.
