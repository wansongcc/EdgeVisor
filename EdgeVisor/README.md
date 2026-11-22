# EdgeVisor engine

Build and run instructions are in the repository root [README](../README.md).
Current runtime and migration guides are in [`docs/`](docs/); older engine guides are in [`docs/archive/`](docs/archive/).

## Jetson frequency-injection helpers

The maintained Jetson disturbance utilities are in `scripts/`. They temporarily apply a frequency level and restore the previous hardware state on exit.

Lock GPU, CPU, and EMC together:

```bash
sudo ./scripts/jetson_freq_inject.sh --level 50 --target gpu,cpu,emc --duration 20
```

Lock only the GPU:

```bash
sudo ./scripts/jetson_gpu_freq_inject.sh --level 20 --duration 8
```

Use `--list` to inspect available frequencies without changing them, and `--target` with the combined injector to select `gpu`, `cpu`, `emc`, or `all`. Run these tools only when no other experiment depends on the device's clock settings. After thermal or frequency experiments, confirm that the previous limits have been restored before collecting benchmark results.
