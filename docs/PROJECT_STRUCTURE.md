# Project Structure

This project keeps the original `EdgeVisor` engine layout intact for build compatibility and organizes project-level scripts and artifacts around it.

- `EdgeVisor/`: core C++ engine, Vulkan kernels, examples, and engine docs.
- `config/env.sh`: shared model, tokenizer, log, and Vulkan dependency configuration.
- `scripts/semantic/`: maintained semantic regression and distributed inference scripts.
- `scripts/gpu/`: maintained GPU PP and GPU regression scripts.
- `tests/semantic/`: reproducible six-case benchmark test runner and Markdown record generator.
- `docs/test_records/`: generated acceptance-style records for the six required tests.
- `artifacts/`: historical logs and experiment results.
- `maintenance/`: historical patch scripts and debug helpers.

Top-level `run_*.sh` files are compatibility wrappers. Update scripts under `scripts/`; wrappers should remain thin.
