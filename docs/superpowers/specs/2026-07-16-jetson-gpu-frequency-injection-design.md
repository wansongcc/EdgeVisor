# Jetson GPU Frequency Injection Design

## Scope

Add a standalone Bash utility for one-shot GPU compute-capacity disturbance on
Jetson Orin Nano and Orin NX. The utility locks only the GPU frequency for a
bounded interval and restores the prior state afterward. It does not change the
CPU, EMC, `nvpmodel`, or EdgeVisor processes.

## Interface

The script is `EdgeVisor/scripts/jetson_gpu_freq_inject.sh`.

```text
sudo EdgeVisor/scripts/jetson_gpu_freq_inject.sh --level 20
sudo EdgeVisor/scripts/jetson_gpu_freq_inject.sh --level 20 --duration 8
EdgeVisor/scripts/jetson_gpu_freq_inject.sh --list
```

- `--level 0..100` is required for injection. Zero selects the lowest usable
  frequency and 100 selects the highest frequency allowed by the current power
  mode.
- `--duration SEC` accepts positive integer or decimal seconds and defaults to
  20 seconds.
- `--list` prints available, current, minimum, and maximum frequencies without
  changing the device and does not require root.
- `--sysfs-root PATH` replaces `/sys` for testing or a mounted target sysfs.

## Frequency Selection

Read `available_frequencies` from the GPU devfreq directory, sort and deduplicate
it, and retain frequencies no greater than the pre-injection `max_freq`. Map the
level linearly over the numeric frequency range, then select the nearest real
frequency. A tie selects the lower frequency.

The default GPU path is:

```text
/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
```

## Injection and Restoration

Before mutation, save `min_freq`, `max_freq`, and, when present,
`enable_3d_scaling`. Acquire a nonblocking `flock` lock so only one injector can
own the GPU state. Set a safe pair of temporary bounds without ever creating a
state where minimum exceeds maximum, disable 3D scaling when the control exists,
and verify both bounds by reading them back.

Print the selected target and duration, then report remaining time and
`cur_freq` approximately once per second. Thermal, electrical, and current power
mode limits remain authoritative, so the output warns that hardware throttling
may still affect the observed clock.

An EXIT trap restores the original bounds and 3D scaling state on success,
SIGINT, SIGTERM, or an intermediate error. Restoration uses the same safe bound
ordering. A restoration failure makes the command fail and prints explicit
manual recovery commands.

## Validation

A Python semantic test uses a temporary fake sysfs tree and test-only environment
hooks. It covers help/defaults, list mode, level mapping, safe write order,
normal restoration, SIGTERM restoration, root enforcement, invalid input,
missing sysfs data, and lock contention. Verification also runs `bash -n` and
the existing semantic test suite.

