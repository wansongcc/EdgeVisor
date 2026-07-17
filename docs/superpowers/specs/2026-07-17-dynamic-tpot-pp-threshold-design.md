# Dynamic TPOT PP Threshold and Rejected-Candidate Logging Design

## Goal

Make PP migration decisions match the current heterogeneous Jetson experiments by disabling the default synthetic load penalty, making the relative PP gain threshold configurable, and preserving the best rejected PP candidate in scheduler logs.

## Scope

This change affects only the dynamic TPOT PP candidate model, its CLI/environment configuration, related documentation, and decision logging. TP candidate scoring, migration execution, cooldown, verification, and rollback behavior remain unchanged.

## Configuration

### Load penalty

- Change `SchedulerConfig::loadPenaltyBeta` default from `0.08` to `0.0`.
- Preserve `DLLAMA_TPOT_LOAD_PENALTY_BETA` as an environment override.
- A nonzero explicit override retains the existing stage-cost formula and behavior.

### Relative PP gain threshold

- Add `SchedulerConfig::ppGainRatio` with default `0.03`.
- Add CLI option `--tpot-pp-gain-ratio VALUE`.
- Map the CLI option to `DLLAMA_TPOT_PP_GAIN_RATIO` during dynamic TPOT configuration.
- Accept finite values in the inclusive range `[0, 1]`.
- Reject missing, nonnumeric, negative, infinite, NaN, or greater-than-one values with a clear CLI error.
- A value of `0` disables the relative component and leaves only `DLLAMA_TPOT_MIN_PP_GAIN_MS` as the threshold.

The local PP threshold becomes:

```text
max(minPpGainMs, ppGainRatio * (source.stageTimeMs + target.stageTimeMs))
```

Dynamic TPOT profiles continue to select their existing absolute PP threshold and window settings. They do not override `ppGainRatio`.

## PP Candidate Selection

`bestPpCandidate()` currently discards every candidate whose gain does not exceed its threshold. It will instead track two results while scanning adjacent stages in both directions:

1. The valid candidate with the greatest gain, if one exists.
2. Otherwise, the structurally eligible rejected candidate with the greatest gain.

A structurally eligible candidate has a source that can give up one layer and a target with full weights. The rejected result retains:

- source and target stage/node indices;
- boundary layer index;
- calculated gain;
- calculated threshold;
- `valid=false`;
- reason `gain below threshold`.

If no structurally eligible PP direction exists, the result remains invalid with reason `no eligible pp candidate`.

Valid-candidate behavior and migration selection remain unchanged.

## Scheduler Logging

Every normal `tpot_sched` decision line will append fields describing the independently calculated best PP candidate:

```text
pp_best_valid=0
pp_best_gain_ms=3.13
pp_best_threshold_ms=8.99
pp_best_reason=gain_below_threshold
pp_best_from_stage=0
pp_best_to_stage=1
pp_best_layer=16
```

These fields are emitted whether the PP candidate is valid or rejected. Spaces and other field-breaking whitespace in the reason are normalized to underscores. Existing `best=`, `gain_ms=`, and migration fields retain their current meaning and format for compatibility.

Startup, disabled, and exception-only scheduler lines that have no candidate evaluation are unchanged.

## Testing

Tests will cover:

- zero default load penalty and the resulting marginal layer cost;
- preservation of the old behavior when beta is explicitly set to `0.08`;
- PP thresholds at ratios `0`, `0.03`, and a custom value;
- retention of the strongest rejected candidate and all routing fields;
- preference for a valid candidate when both valid and rejected candidates exist;
- CLI-to-environment mapping for `--tpot-pp-gain-ratio`;
- CLI rejection of invalid ratio values;
- scheduler log formatting for rejected and valid PP candidates;
- the existing dynamic TPOT algorithm and CLI regression suites.

## Compatibility

- Existing commands that omit the new option use `0.03`, preserving the current relative threshold.
- Existing scripts that explicitly set `DLLAMA_TPOT_LOAD_PENALTY_BETA` retain their configured behavior.
- Existing log parsers can ignore the appended fields.
- No model, network protocol, plan command, or on-device worker change is required beyond deploying the rebuilt binary.
