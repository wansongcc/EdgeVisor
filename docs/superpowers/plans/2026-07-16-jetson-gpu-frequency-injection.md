# Jetson GPU Frequency Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe one-shot GPU frequency disturbance injector for Jetson Orin Nano and Orin NX.

**Architecture:** A standalone Bash CLI discovers the Orin GPU devfreq files beneath a configurable sysfs root, maps a percentage to a supported frequency, locks min/max for a bounded interval, and restores state from an EXIT trap. A Python semantic test drives the real Bash script against a temporary fake sysfs tree.

**Tech Stack:** Bash 4+, Linux devfreq/sysfs, util-linux `flock`, Python 3 standard library.

## Global Constraints

- Modify only GPU frequency controls; never invoke `nvpmodel`, `jetson_clocks`, CPU controls, or EMC controls.
- Support Jetson Orin Nano and Orin NX through `/devices/platform/17000000.gpu`.
- `--level` is 0 through 100 and maps linearly by frequency value to the nearest available frequency, choosing the lower frequency on a tie.
- `--duration` defaults to exactly 20 seconds and accepts positive decimals.
- Always restore original min/max and 3D scaling state after any trappable exit.
- Preserve unrelated uncommitted files.

---

### Task 1: Executable behavior and restoration

**Files:**
- Create: `tests/semantic/test_jetson_gpu_freq_injector.py`
- Create: `EdgeVisor/scripts/jetson_gpu_freq_inject.sh`

**Interfaces:**
- Consumes: Orin GPU devfreq files `available_frequencies`, `min_freq`, `max_freq`, and `cur_freq`; optional sibling `enable_3d_scaling`.
- Produces: CLI options `--level`, `--duration`, `--list`, `--sysfs-root`, and `--help`; test hooks `JETSON_GPU_TEST_EUID`, `JETSON_GPU_TEST_NO_SLEEP`, `JETSON_GPU_WRITE_LOG`, and `JETSON_GPU_LOCK_FILE`.

- [ ] **Step 1: Write the failing semantic test**

Create a temporary tree with these exact values:

```python
available = "306000000 408000000 612000000 816000000 1020000000\n"
minimum = "306000000\n"
maximum = "1020000000\n"
current = "612000000\n"
scaling = "1\n"
```

Use `subprocess.run` and `subprocess.Popen` to assert: help states the 20-second
default; list works with test EUID 1000; level 20 selects 408 MHz and restores
all state; level 100 selects 1020 MHz; a starting 816/816 MHz state writes
`min=306` before `max=306` and restores `max=816` before `min=816`; SIGTERM
restores state; invalid levels and durations fail; a missing frequency file
fails; EUID 1000 cannot inject; and an externally held `flock` causes failure.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 tests/semantic/test_jetson_gpu_freq_injector.py
```

Expected: FAIL with `missing script` because
`EdgeVisor/scripts/jetson_gpu_freq_inject.sh` does not exist.

- [ ] **Step 3: Implement the minimal Bash CLI**

Implement small functions with these contracts:

```bash
die()                         # print to stderr and exit nonzero
read_uint FILE                # read and validate one unsigned integer
discover_paths()              # initialize GPU_DIR and SCALING_FILE
load_frequencies()            # populate sorted FREQUENCIES within original max
select_frequency LEVEL        # print nearest frequency, lower on ties
write_value FILE VALUE        # write sysfs and append to optional write log
set_bounds MIN MAX            # safely transition without transient min > max
restore_state()               # restore bounds/scaling and report manual commands
cleanup()                     # EXIT trap preserving failure and restore status
seconds_to_ms SEC             # print positive duration as integer milliseconds
run_interval DURATION_MS      # print remaining milliseconds/cur_freq and sleep
```

Parse options without `eval`; reject unknown or duplicate positional input.
Check root after read-only argument handling. Acquire the lock with:

```bash
exec 9>"${JETSON_GPU_LOCK_FILE:-/tmp/jetson_gpu_freq_inject.lock}"
flock -n 9 || die "another GPU frequency injection is already running"
```

Set the EXIT, INT, and TERM traps before the first sysfs write. Mark restoration
active immediately after capturing the original values.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
python3 tests/semantic/test_jetson_gpu_freq_injector.py
bash -n EdgeVisor/scripts/jetson_gpu_freq_inject.sh
```

Expected: both commands exit 0 and the Python test prints a PASS summary.

- [ ] **Step 5: Commit the tested component**

```bash
git add tests/semantic/test_jetson_gpu_freq_injector.py EdgeVisor/scripts/jetson_gpu_freq_inject.sh
git commit -m "Add Jetson GPU frequency disturbance injector"
```

### Task 2: Usage documentation and regression verification

**Files:**
- Modify: `EdgeVisor/README.md`
- Test: `tests/semantic/test_jetson_gpu_freq_injector.py`

**Interfaces:**
- Consumes: the Task 1 CLI.
- Produces: copy-paste examples for list, default 20-second injection, and custom duration, plus the thermal/power-limit caveat.

- [ ] **Step 1: Extend the semantic test with a failing documentation check**

Read `EdgeVisor/README.md` and require these literal tokens:

```python
"jetson_gpu_freq_inject.sh"
"--level 20"
"--duration 8"
"thermal"
```

- [ ] **Step 2: Run the test and verify RED**

Run `python3 tests/semantic/test_jetson_gpu_freq_injector.py`.

Expected: FAIL because the README does not mention the new script.

- [ ] **Step 3: Add concise README instructions**

Document:

```bash
EdgeVisor/scripts/jetson_gpu_freq_inject.sh --list
sudo EdgeVisor/scripts/jetson_gpu_freq_inject.sh --level 20
sudo EdgeVisor/scripts/jetson_gpu_freq_inject.sh --level 20 --duration 8
```

State that the default duration is 20 seconds, only the GPU is changed, state is
restored on normal/signal exit, and thermal or power limits may still throttle.

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
python3 tests/semantic/test_jetson_gpu_freq_injector.py
python3 tests/semantic/test_gpu_compute_disturbance_script.py
bash -n EdgeVisor/scripts/jetson_gpu_freq_inject.sh
```

Expected: all commands exit 0 with no warnings or tracebacks.

- [ ] **Step 5: Commit documentation**

```bash
git add EdgeVisor/README.md tests/semantic/test_jetson_gpu_freq_injector.py
git commit -m "Document Jetson GPU frequency injection"
```
