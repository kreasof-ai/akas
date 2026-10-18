# Optimize a deployment

Give AKAS a local checkpoint, the workload it must serve, and a time budget.
The optimizer detects the model and available backend, then chooses compiled
geometries, prefill chunking, request grouping, replay, kernel schedules and
applicable operation policies. Supported speculative heads are evaluated against
ordinary autoregressive generation. Weight loading and cache precision follow
the model's declared execution defaults; the optimizer does not choose additional
requantization. Source-file precision and execution precision can differ; see
[model-level constraints](benchmarking.md#prepare-an-unattended-campaign).

Install the `compiler` and `serving` extras, plus `webgpu` when using that backend.
Complete the [project setup](../README.md) for CUDA or
[WebGPU prepared submission](benchmarking.md#run-an-akas-measurement) before
searching.

```sh
akas optimize --model /path/to/model.gguf --budget 30m \
  --workload standard-v1 --out build/deployment
akas serve --deployment build/deployment/deployment.json
```

The output directory retains programs, candidate measurements and campaign state.
`deployment.json` points only to the best fully qualified candidate. Serving reads
the selected settings from this file; it does not compile, tune or ask for model
hyperparameters. Keep the output directory intact. Checkpoint bytes, runtime
source, runtime dependencies, hardware and compiled artifacts are checked when
loading a deployment.

## Describe requirements

`standard-v1` means active request counts 1/4/16/64/256, prompt lengths 32/2048,
128 total output tokens, greedy generation, empty caches and no early EOS. A
custom workload needs just its shapes:

```json
{"batches":[1,4],"prefix_tokens":[32,2048],"output_tokens":128}
```

Pass its path with `--workload`. This optimizer measures concurrent request bursts;
arrival-rate traces and long-context traffic require separate workload support.
Every requested shape must pass. An oversized workload is not silently reduced.

Optional requirements are `--backend cuda|webgpu`, `--device`, `--memory-gib`,
`--ttft-ms` and `--tpot-ms`. The latency limits apply to the worst measured request
over repeated loopback HTTP bursts. TTFT starts before submitting the burst;
TPOT averages the interval between the first and final output token for each
request. These are observed pilot limits, not a production percentile guarantee.

Built-in model families are discovered from checkpoint architecture metadata.
Use `--definition module:class` only for a custom model. Model developers declare
architecture identifiers and optimization capabilities beside their definition;
no central model-name switch is required.

## Understand selection

The first successful candidate establishes a model reference after independent
operation qualification. Subsequent candidates must preserve every greedy output
token and stay within 1% relative error at sampled diagnostic boundaries
(logits, or hidden features when logits are fused away). Repeated requests
exercise slot reuse. Models without diagnostics are checked through output
equivalence and operation qualification. This preserves
the baseline model; it does not establish agreement with an external framework.
Use independent model evidence when qualifying a new architecture or checkpoint.

Each candidate runs one warmup and three measured repetitions for native token
phases, the shared engine and the actual HTTP service. Native prefill and decode
rates remain separate from serving measurements. For speculative candidates those
native phase rates describe the AR target; HTTP throughput includes proposal,
verification and repair, and counts only committed outputs.

Selection maximizes geometric-mean HTTP speedup across the requested shapes.
No shape may regress by more than 5% against the reference, and all supplied
latency constraints must pass. A new winner needs more than 1% improvement over
the incumbent. The baseline remains available if no searched candidate improves
it. Loading and compilation are excluded from timed inference but included in
the optimization budget.

The compiler searches existing operation implementations and their declared
schedule axes. Operation-owned policies offer alternative physical algorithms;
this workflow does not invent new kernel source. Unsupported combinations and
rejected candidates retain their reasons in campaign state and benchmark history.
Operation qualification can reuse an exact source/device-qualified cache entry;
new candidates still run complete model and HTTP checks with fresh timings.
Program and serving alternatives receive a cheap qualification pass first.
Larger kernel-search budgets then refine the best measured configuration,
retaining its selected chunk, grouping, policy and speculative mode.
Programs retain a smaller prefill geometry for the workload's shortest prompt.
The executor chooses among qualified geometries so a larger throughput tile
does not force short prompts to execute a padded full tile.

## Stop and resume

The budget covers this invocation, including probing, imports, compilation,
qualification and measurement. Each worker also receives a bounded share, so one
stalled compilation cannot consume the whole campaign. A baseline that times
out while making progress receives one
automatic continuation before the workflow explores broader configurations.
Workers run in isolated process groups; timeout or interruption terminates them. Linux workers also exit
when their supervisor dies. Shutdown and evidence finalization can add a few
seconds beyond a deadline.

Run the same command and output directory to grant another budget. Completed
candidates are retained. Partial beam progress and compiled artifacts are reused;
the selected resumed schedule is qualified and timed again before promotion.
Compatible historical schedules are discovered automatically and requalified.
Changed model bytes, source, dependencies or workload require a new output
directory. A previous qualified deployment survives an interrupted or failed
attempt; publication replaces its pointer atomically after all checks pass.

`campaign.json` gives the current status, attempts, chosen settings, memory samples
and winner. `jobs/` contains raw measurements and compiler caches. Numerical
allocation estimates and sampled whole-device VRAM are recorded separately;
sampling can miss brief peaks. Memory ceilings reject oversized persistent state,
oversized measured allocations and sampled device usage. Runtime OOM remains a
recorded candidate failure. Independent optimizers targeting the same device are
serialized; avoid simultaneous unrelated GPU measurements.

The search may exhaust its declared alternatives before using the whole budget,
or stop at budget exhaustion with a usable incumbent. A run without a qualified
deployment exits unsuccessfully. A time allowance does not guarantee that a
particular model/backend is supported or that a throughput target will be reached.

All candidate records use [benchmark history](benchmarking.md), including actual
source snapshots, exact settings, timing samples and winning kernel geometries.
The [implementation guide](../src/akas/optimization/README.md) explains where to
extend the workflow.

## Run a model matrix

A deployment search covers one checkpoint and workload. The repository's matrix
runner coordinates the declared models, file formats, backends, cache precisions
and generation modes serially:

```sh
python -m tools.campaign --spec benchmarks/campaigns/standard-v1.json \
  --budget 4d --slice 30m --out build/standard-campaign
```

For the single-user LFM2 MoE workload, use its pinned QAD Q4_0 checkpoint and
both backends, allowing enough time per slice to repeat the longest prompts:

```sh
python -m tools.campaign --spec benchmarks/campaigns/single-user-long-context-v1.json \
  --budget 4d --slice 4h --out build/single-user-long-context
```

Size a slice from measured runtime: one candidate repeats every shape through
native phases, the scheduler and HTTP serving, as well as compiling and checking
numerics. It can take much longer than one native capacity probe. A larger slice
gives those checks room to finish within the campaign's total budget.

The specification contains workload requirements, not kernel hyperparameters.
`--mode ar|mtp` and `--kv-cache float16|bfloat16|fp8` apply the same requirements
to an individual `akas optimize` run. Without those restrictions it evaluates
supported modes using the model's default cache precision.

The matrix keeps every requested combination, including unsupported paths and
failures. It gives each configuration a first allowance before returning to
unfinished searches. Each allowance resumes that configuration's compiler and
serving search. The four-day limit is cumulative active time across invocations;
repeating the same command resumes the remaining allowance. Finite searches may
finish earlier. `campaign.json` retains per-configuration status and deployment
paths. An unsupported entry is not a measured zero-throughput result.

Each invocation also writes a durable [campaign report](../benchmarks/reports/README.md)
under `benchmarks/reports/`. It includes all requested shapes and configurations,
available baseline tables, failures, CSV exports and selected kernel schedules.
The report retains its own snapshot and measurement data for regeneration after
build directories are cleaned. See [reporting commands](benchmarking.md#find-a-previous-result).

Run the [baseline measurements](../tools/baselines/README.md) and memory/model
qualification before starting a long matrix. The supplied specification expresses
the requested coverage; it does not imply that every model supports every path.
Model-level constraints remain in the [benchmark guide](benchmarking.md).

Each matrix declares its per-configuration ceiling as `configuration_budget`
in its JSON, including continuations. The standard matrix uses one hour; the
single-user long-context matrix allows 48 hours for each of its two backends.
The overall four-day allowance is an upper limit, not a requested run duration;
this matrix can finish sooner. After a hard interruption, the coordinator
conservatively charges its last reserved allowance when exact elapsed time is
unavailable. Graceful interruption retains measured elapsed time.

Worker grants grow when completed checks show that the workload needs more time
than the initial share. Diagnostic sampling follows the final active token of a
consistent request across prefill chunk and grouping changes. Every output ID is
still checked for every request.
