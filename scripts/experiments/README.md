# Experiment evidence foundation

This directory separates code correctness gates from paper performance/quality
experiments. The current live gate is **four CUDA containers and a 28-layer Q40
model**, each stage owning seven layers. It does not represent mobile devices,
14B models, multi-tenant GPU isolation, thermal control, or statistical results.

## Build and validate a coding snapshot

Use the owned lab contract and configuration in `../lab/README.md`. No models or
images are downloaded. Build in a fresh remote directory; compile both binaries
and run all model-free C++ checks before deployment:

```bash
python3 scripts/lab/deploy_build.py \
  --config /path/to/lab_config.json \
  --output /path/to/new-build-evidence --image existing-cuda-image
python3 scripts/experiments/research_gate.py \
  --config /path/to/lab_config.json \
  --build-evidence /path/to/new-build-evidence \
  --output /path/to/new-gate-evidence --network
```

The gate verifies deployed source and binary against that accepted build,
records model/tokenizer hashes and GPU snapshots, and compares every
`(position, token)` with an uninstrumented static reference. Root and tail
round-trips must commit twice; automatic migration must actually commit. Raw
trace tokens, routes and KV row counts must agree with output and ownership
commits. TC probes measure the achieved rate instead of treating the requested
rate as the observation. Inference runs reset traffic control and stop only
run-owned processes. Run one lab fixture at a time.

Each directory is exclusive. `manifest.json` declares status, source changes,
configuration and hashes for all artifacts. Failures and incomplete traces are
retained. A normal return code alone cannot produce a successful gate. Do not
modify engine sources during a build or gate; keep each snapshot reproducible.
Verify copied evidence using `recording.verify_artifacts(directory)`.

## Engine raw trace

Set `EDGEVISOR_TRACE_PATH` to a fresh writable JSONL path and
`EDGEVISOR_RUN_ID` to the trial ID. Existing paths are refused. Tracing is opt-in
and synchronous: it incurs measurable overhead, so compare against untraced
runs before using traced measurements as performance results. No model prompts
or answers are recorded by the raw trace; token IDs are recorded for correctness.

Events carry schema version, trial, root process clock, runtime context, actor,
sequence, raw steady-clock timestamp, span/parent identity and attributes.
Loading, generation, prefill, decode, forward, token and migration phases have
raw endpoints. KV export, installation/ACK and binding verification are nested
intervals. Ownership/result points distinguish actual commits from intent.
The API server lifetime is not an infinite enclosing interval; request-scoped
API lineage and tool instrumentation belong to the following implementation
stage.

```bash
python3 scripts/experiments/timeline.py /path/to/root.trace.jsonl
python3 -m unittest discover -s scripts/experiments/tests -v
```

The validator rejects missing endpoints, reversed actor clocks, sequence gaps,
clock-domain mixing and children outside parent time. Phase totals are interval
unions; **do not sum a parent with its children**, or sum overlapping phases as
an episode total. Root forward includes compute, network wait and protocol work;
it is not an isolated compute duration. Worker profile durations remain a
separate diagnostic source and cannot be subtracted from root timestamps.

Client events use a separate client monotonic clock. Fault injection and
confirmation to first-token intervals require all corresponding markers on a
single clock with a transition ID; absent milestones stay `not_observed`, never
zero. Current CLI gates do not manufacture those fault milestones. GPU power
snapshots are observations, not integrated energy. This gate reports one
independent episode per case and does not infer tail percentiles or confidence
intervals from the many tokens in one episode.

Legacy textual timing and agent reports do not become valid paper evidence
merely by wrapping them in a new manifest. Continuous token lineage, autonomous
quality, isolated preparation budgets, transaction fencing, fair baselines and
paired independent episode analysis remain separate implementation/gate stages.

## Serial API and real tool gates

```bash
python3 scripts/experiments/api_session_gate.py \
  --config /path/to/lab_config.json --build-evidence /path/to/build \
  --output /path/to/new-api-evidence
python3 scripts/experiments/agent_tool_gate.py \
  --config /path/to/lab_config.json --build-evidence /path/to/build \
  --output /path/to/new-agent-evidence
```

API tests use real models with prefill batch=1 on single GPU, four-node root
sampling and four-node tail sampling. They check canonical cold vs warm
continuation, changed/sessionless histories, capacity/invalid input isolation,
request seed/temperature/top-p resets and streaming. Use an explicit
`edgevisor_session_id` to permit KV reuse. The response `edgevisor` metadata
reports runtime context, lineage, generation, consumed KV row count and pending
sampled input. Only one serial KV owner exists; switching sessions re-prefills.
A sampled output is never counted as already consumed KV. This is not a
concurrent multi-session KV implementation.

The autonomous runner needs only Python's standard library. It executes model
chosen tools, records their real start/end, parses complete single JSON objects
strictly and scores against references only after generation. It does not
supply target arguments or target answers to the model. Failed output remains
failed. Without an objective checker, quality stays unknown. The small live
calculator gate is a protocol/lineage diagnostic; it is not a task-suite quality
result. Keep prompt/protocol failures as separate retained trials.

`agent_bench.runtime_loop.run_loop_episode` now defaults to `autonomous`.
`execution_mode=scripted` explicitly selects the existing LangGraph controlled
workflow and still needs its optional dependencies. Its forced workflow result
is labeled `scripted_workflow_completed`; it cannot populate autonomous task
success. Two existing forced ablation episodes are labeled scripted. Existing
legacy latency summaries remain separate from the new raw interval evidence.
Tool latency measured on the client includes actual local tool execution; it
cannot be subtracted from root process timestamps or presented as device energy.
Preparation overlap/budgets and broad task-quality/paired-episode experiments
remain following stages.
