# Task 2: CLI mapping and validation report

## Scope

Implemented `--tpot-pp-gain-ratio VALUE` and its mapping to
`DLLAMA_TPOT_PP_GAIN_RATIO`. The value is stored in
`AppCliArgs::tpotPpGainRatioStr`, validated as a finite number in `[0,1]`, and
loaded into `SchedulerConfig::ppGainRatio` with the same interval validation.
No logging changes were made.

The new setting is PP-specific: it maps only to `ppGainRatio`; no TP option,
TP environment variable, or TP scheduler configuration is modified.

## RED evidence

After adding the CLI tests and before implementation, ran:

```sh
make -C EdgeVisor app-cli-test && EdgeVisor/app-cli-test
```

Result: exited `134` after the test executable threw the expected missing
feature error:

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  Unknown option: --tpot-pp-gain-ratio
```

This demonstrated that the new test exercised the unimplemented CLI option.

## GREEN evidence

After implementing the option, mapping, and scheduler configuration loading,
ran:

```sh
make -C EdgeVisor app-cli-test dynamic-tpot-test && EdgeVisor/app-cli-test && EdgeVisor/dynamic-tpot-test
```

Result: exited `0`. `app-cli-test` passed, including mapping `0` to
`DLLAMA_TPOT_PP_GAIN_RATIO` and rejecting `-0.01`, `1.01`, `nan`, `inf`, and
`not-a-number`. `dynamic-tpot-test` reported:

```text
PASS dynamic_tpot_algorithm
```

## Files changed

- `EdgeVisor/src/app.hpp`
- `EdgeVisor/src/app.cpp`
- `EdgeVisor/src/test/test_app_cli.cpp`
- `EdgeVisor/src/dynamic/dynamic_tpot.cpp`
