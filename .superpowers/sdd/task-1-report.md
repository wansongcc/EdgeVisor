# Task 1 Report: Scoring defaults and rejected-candidate retention

## Implementation

- Changed `SchedulerConfig::loadPenaltyBeta` default from `0.08` to `0.0`.
- Added `SchedulerConfig::ppGainRatio` with default `0.03`.
- Replaced hard-coded `0.03` in PP and TP threshold calculations, including the local PP threshold, with `cfg.ppGainRatio`.
- Updated `bestPpCandidate()` to retain the strongest eligible rejected candidate when no valid candidate exists. Valid candidates continue to take precedence.
- Updated the dynamic TPOT algorithm test with default, configurable-ratio, rejected-candidate metadata, and mixed valid/rejected-candidate assertions. The existing soft-capacity assertion now explicitly opts into the legacy `loadPenaltyBeta = 0.08` behavior.

## Files

- `EdgeVisor/src/dynamic/tpot_algorithm.hpp`
- `EdgeVisor/src/dynamic/tpot_algorithm.cpp`
- `EdgeVisor/src/test/test_dynamic_tpot_algorithm.cpp`

## RED evidence

Command:

```bash
make -C EdgeVisor dynamic-tpot-test
```

Output:

```text
make: Entering directory '/home/cc/EdgeVisor/.worktrees/dynamic-tpot-pp-threshold/EdgeVisor'
g++ -std=c++11 -Werror -Wformat -Werror=format-security -Isrc -MMD -MP -DDLLAMA_CONTROL_LOG=0 -DDEBUG_OP_INPUT_OUTPUT=0 -DDLLAMA_DEBUG_ATTN=0 -march=native -mtune=native -O3 src/test/test_dynamic_tpot_algorithm.cpp tpot-algorithm.o -o dynamic-tpot-test -lpthread
src/test/test_dynamic_tpot_algorithm.cpp: In function ‘int main()’:
src/test/test_dynamic_tpot_algorithm.cpp:51:22: error: ‘struct dllama::dynamic_tpot::SchedulerConfig’ has no member named ‘ppGainRatio’
   51 |     require(near(cfg.ppGainRatio, 0.03), "PP gain ratio defaults to three percent");
      |                      ^~~~~~~~~~~
src/test/test_dynamic_tpot_algorithm.cpp:65:9: error: ‘struct dllama::dynamic_tpot::SchedulerConfig’ has no member named ‘ppGainRatio’
   65 |     cfg.ppGainRatio = 0.03;
      |         ^~~~~~~~~~~
src/test/test_dynamic_tpot_algorithm.cpp:67:9: error: ‘struct dllama::dynamic_tpot::SchedulerConfig’ has no member named ‘ppGainRatio’
   67 |     cfg.ppGainRatio = 0.0;
      |         ^~~~~~~~~~~
src/test/test_dynamic_tpot_algorithm.cpp:69:9: error: ‘struct dllama::dynamic_tpot::SchedulerConfig’ has no member named ‘ppGainRatio’
   69 |     cfg.ppGainRatio = 0.10;
      |         ^~~~~~~~~~~
src/test/test_dynamic_tpot_algorithm.cpp:71:9: error: ‘struct dllama::dynamic_tpot::SchedulerConfig’ has no member named ‘ppGainRatio’
   71 |     cfg.ppGainRatio = 0.03;
      |         ^~~~~~~~~~~
make: *** [Makefile:185: dynamic-tpot-test] Error 1
make: Leaving directory '/home/cc/EdgeVisor/.worktrees/dynamic-tpot-pp-threshold/EdgeVisor'
```

The failure was due to the missing required interface, before implementation changes.

## GREEN evidence

Command:

```bash
make -C EdgeVisor dynamic-tpot-test && EdgeVisor/dynamic-tpot-test
```

Output:

```text
make: Entering directory '/home/cc/EdgeVisor/.worktrees/dynamic-tpot-pp-threshold/EdgeVisor'
g++ -std=c++11 -Werror -Wformat -Werror=format-security -Isrc -MMD -MP -DDLLAMA_CONTROL_LOG=0 -DDEBUG_OP_INPUT_OUTPUT=0 -DDLLAMA_DEBUG_ATTN=0 -march=native -mtune=native -O3 -c src/dynamic/tpot_algorithm.cpp -o tpot-algorithm.o
g++ -std=c++11 -Werror -Wformat -Werror=format-security -Isrc -MMD -MP -DDLLAMA_CONTROL_LOG=0 -DDEBUG_OP_INPUT_OUTPUT=0 -DDLLAMA_DEBUG_ATTN=0 -march=native -mtune=native -O3 src/test/test_dynamic_tpot_algorithm.cpp tpot-algorithm.o -o dynamic-tpot-test -lpthread
make: Leaving directory '/home/cc/EdgeVisor/.worktrees/dynamic-tpot-pp-threshold/EdgeVisor'
PASS dynamic_tpot_algorithm
```

Post-commit verification reran the same command; Make reported the target up to date and the test again printed `PASS dynamic_tpot_algorithm`.

## Self-review

- Requirements coverage: all exact defaults, ratio assertions, rejected-candidate fields, and valid-over-rejected precedence cases are covered.
- Scope: the commit contains only the three owned source/test files.
- Safety: rejected candidates are only selected while no valid candidate has been found; once a valid candidate exists, later rejected candidates cannot replace it.
- Build quality: the target compiles with the repository's `-Werror` flags, and `git diff --check` passed.

## Commit

`f373ff4 feat: make PP threshold model configurable`

## Concerns

- `.superpowers/` remains pre-existing untracked worktree metadata and was not staged or modified by the implementation. The requested report is written under that directory.

## Review fix: preserve TP scoring

The earlier implementation statement that the configurable PP ratio was used in TP scoring is superseded by this review fix. TP scoring remains unchanged at `max(minTpGainMs, 0.03 * stage.stageTimeMs)`. The regression test sets `ppGainRatio` to `0.0` and verifies both the TP threshold (`12.0` for a `400.0 ms` stage) and TP candidate rejection remain governed by the fixed TP ratio.

### Fix RED evidence

Command:

```bash
make -C EdgeVisor dynamic-tpot-test && EdgeVisor/dynamic-tpot-test
```

Output:

```text
make: Entering directory '/home/cc/EdgeVisor/.worktrees/dynamic-tpot-pp-threshold/EdgeVisor'
g++ -std=c++11 -Werror -Wformat -Werror=format-security -Isrc -MMD -MP -DDLLAMA_CONTROL_LOG=0 -DDEBUG_OP_INPUT_OUTPUT=0 -DDLLAMA_DEBUG_ATTN=0 -march=native -mtune=native -O3 src/test/test_dynamic_tpot_algorithm.cpp tpot-algorithm.o -o dynamic-tpot-test -lpthread
make: Leaving directory '/home/cc/EdgeVisor/.worktrees/dynamic-tpot-pp-threshold/EdgeVisor'
FAIL: PP gain ratio does not change TP threshold
```

Exit status: `1`. The new regression assertion failed against commit `f373ff4` because `tpGainThresholdMs()` incorrectly used `cfg.ppGainRatio`.

### Fix GREEN evidence

Command:

```bash
make -C EdgeVisor dynamic-tpot-test && EdgeVisor/dynamic-tpot-test
```

Output:

```text
make: Entering directory '/home/cc/EdgeVisor/.worktrees/dynamic-tpot-pp-threshold/EdgeVisor'
g++ -std=c++11 -Werror -Wformat -Werror=format-security -Isrc -MMD -MP -DDLLAMA_CONTROL_LOG=0 -DDEBUG_OP_INPUT_OUTPUT=0 -DDLLAMA_DEBUG_ATTN=0 -march=native -mtune=native -O3 -c src/dynamic/tpot_algorithm.cpp -o tpot-algorithm.o
g++ -std=c++11 -Werror -Wformat -Werror=format-security -Isrc -MMD -MP -DDLLAMA_CONTROL_LOG=0 -DDEBUG_OP_INPUT_OUTPUT=0 -DDLLAMA_DEBUG_ATTN=0 -march=native -mtune=native -O3 src/test/test_dynamic_tpot_algorithm.cpp tpot-algorithm.o -o dynamic-tpot-test -lpthread
make: Leaving directory '/home/cc/EdgeVisor/.worktrees/dynamic-tpot-pp-threshold/EdgeVisor'
PASS dynamic_tpot_algorithm
```

Exit status: `0`.

### Fix self-review

- Production change is limited to restoring the fixed `0.03` TP ratio.
- Regression coverage proves `ppGainRatio = 0.0` changes neither `tpGainThresholdMs()` nor TP candidate validity.
- PP threshold configurability and rejected-candidate retention are unchanged.
- No concerns identified for this fix.
