# Document Status and Cleanup Candidates

This file tracks which documents should be treated as current references and which ones are historical records or cleanup candidates. Do not delete anything from this list automatically; deletion should be a separate user decision.

## Current Authoritative Docs

- `EdgeVisor/docs/README_ENV_VARS.md`
  - Canonical runtime environment variable and CLI switch reference.
  - Includes current automatic TPOT migration entry: `--enable-dynamic-tpot --plan-ctrl-socket <path>`.

- `EdgeVisor/docs/README_DYNAMIC_UDS.md`
  - Manual UDS JSON-line protocol reference for `status`, `perf`, `set_plan`, `set_pp_migration`, and `clear`.

- `EdgeVisor/docs/HOW_TO_ONLINE_MIGRATION.md`
  - Workflow guide for manual UDS migration and automatic TPOT online migration.

- `EdgeVisor/docs/UNEVEN_TP_PP_CONFIG.md`
  - `--ratios`, TP/PP startup, KV redundancy, and current migration startup examples.

- `EdgeVisor/docs/HOW_TO_RUN_GPU.md`
  - Baseline GPU/Vulkan startup. Now links to the current migration parameter docs.

- `EdgeVisor/docs/HOW_TO_RUN_LINUX_MACOS_WIN.md`
  - Baseline distributed startup. Now links to the current migration parameter docs.

- `EdgeVisor/docs/HOW_TO_RUN_RASPBERRYPI.md`
  - Baseline Raspberry Pi startup. Now links to the current migration parameter docs.

- `EdgeVisor/GPU_BASED_USAGE.md`
  - GPU/Vulkan usage, dynamic heads migration, and PP layer migration notes.

- `README.md`
  - Top-level project overview and current quick-start pointers.

- `docs/PROJECT_STRUCTURE.md`
  - Project layout and documentation entry points.

## Compatibility Or Secondary Docs

- `EdgeVisor/README_new.md`
  - Useful Chinese onboarding guide. It now references current migration parameters, but overlaps with `README.md`, `README_ENV_VARS.md`, and `HOW_TO_ONLINE_MIGRATION.md`.

- `EdgeVisor/README.md`
  - Mostly upstream Distributed Llama README retained for compatibility. It now has a status note and should not be treated as authoritative for EdgeVisor migration behavior.

- `agent_bench/README.md`
  - Current for agent-bench usage. Dynamic migration text has been updated, but backend implementation may need corresponding code changes if it still injects `DLLAMA_PLAN_CTRL_SOCKET` instead of CLI args.

- `EdgeVisor/docs/CUDA_SUPPORT.md`
  - Current CUDA support matrix. Dynamic migration entry has been updated at a high level.

- `EdgeVisor/docs/HOW_TO_CONVERT_HF_MODEL.md`
  - Model conversion only. It does not carry runtime migration parameters.

- `EdgeVisor/docs/HOW_TO_CROSS_COMPILE_ARM.md`
  - Build/deployment only. It does not carry runtime migration parameters.

## Historical Test Records: Keep Unless You Want To Archive

These are acceptance or experiment records. Their original commands and outputs should not be rewritten to current syntax, because that would corrupt the historical record. Each now has a status note pointing to current parameters.

- `docs/test_records/01_CPU_单机测试.md`
- `docs/test_records/02_GPU_单机测试.md`
- `docs/test_records/03_CPU_非均匀静态测试.md`
- `docs/test_records/04_GPU_非均匀静态测试.md`
- `docs/test_records/05_CPU_非均匀动态迁移测试.md`
- `docs/test_records/06_GPU_非均匀动态迁移测试.md`
- `docs/test_records/07_Bubble_Shadow_KV_3GPU_PP_对照测试.md`
- `EdgeVisor/docs/CUDA_PR10_12_MANUAL_TEST.md`

## Generated Or Run Metadata: Usually Not Documentation

These files are historical run metadata. They contain old command lines by design and should not be edited as docs.

- `EdgeVisor/logs/20260309_084108/run.meta.txt`
- `EdgeVisor/logs/20260309_085342/run.meta.txt`
- `EdgeVisor/logs/20260309_090635/run.meta.txt`

## Cleanup Candidates For User Decision

The following files are candidates for deletion or archival after you confirm no scripts or workflows still link to them.

- `EdgeVisor/README.md`
  - Reason: mostly upstream README; current EdgeVisor docs live elsewhere.
  - Safer action: archive/rename rather than delete if external links may point to it.

- `EdgeVisor/README_new.md`
  - Reason: overlaps with current canonical docs. Useful as Chinese onboarding, but redundant if `README.md` + `README_ENV_VARS.md` + `HOW_TO_ONLINE_MIGRATION.md` are maintained.

- `docs/test_records/*.md`
  - Reason: generated/historical acceptance records. Keep if you need audit trail; archive if the repo should only keep current docs.

- `EdgeVisor/docs/CUDA_PR10_12_MANUAL_TEST.md`
  - Reason: historical PR-specific manual test plan. Keep if you need CUDA milestone audit history; archive otherwise.

- `EdgeVisor/logs/**/run.meta.txt`
  - Reason: generated run artifacts, not docs. Archive or delete according to experiment retention policy.

- `maintenance/debug/gdbcmd.txt`
  - Reason: local debug helper, not product documentation. Keep only if still used by developers.
