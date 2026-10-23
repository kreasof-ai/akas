# Benchmarking

Use the same workload and measurement rules for AKAS and every baseline.
Keep results in [benchmark history](../benchmarks/README.md); use this guide to
run, compare and build on them.

## Choose a workload

[standard-v1](../src/akas/benchmarking/workloads/standard-v1.json) measures ten shapes:

| Active requests | Prompt tokens per request | Total output tokens per request |
| --- | --- | --- |
| 1, 4, 16, 64, 256 | 32 or 2048 | 128 |

For single-user inference, use
[single-user-long-context-v1](../src/akas/benchmarking/workloads/single-user-long-context-v1.json):
B1 with prompts of 32, 512, 2,048, 8,192, 16,384, 32,768, 65,536 and 130,944
tokens, each generating 128 tokens. The largest case fills a 131,072-token total
context window. The [matching campaign](../benchmarks/campaigns/single-user-long-context-v1.json)
selects the pinned LFM2.5-24B-A2B QAD Q4_0 GGUF on CUDA and WebGPU; its external
baseline is llama.cpp on CUDA and Vulkan using identical checkpoint bytes.
Workload IDs and JSON paths are accepted by the optimizer and baseline tools.

A prompt starts with an empty cache. Disable prefix sharing, early EOS and CPU
offload. Use greedy sampling with repetition penalty 1 and presence/frequency
penalties 0; set these explicitly rather than inheriting server defaults.
Give each request a distinct prefix when a runtime
cannot disable reuse. Long-context and ragged-serving workloads should receive
their own versioned definitions rather than changing these shapes.

Synthetic inputs use uniform model-vocabulary IDs with seed
`17 + B × 4099 + prompt_tokens` across native execution, serving and external
baselines. Native rows record the effective seed and SHA-256 fingerprints of
the prompt and cached prefix. Regression comparisons reject different recorded
inputs. Pass an explicit native `seed` to replay a historical measurement.

Use real text when evaluating speculative acceptance. Random IDs can produce
repetitive continuations and misleading draft speedups. The
[text workload tool](../tools/text_workload.py) pins conversation, code, math and
long-document sources and creates separate calibration and held-out prompts:

```sh
python -m tools.text_workload --checkpoint /path/to/lfm2-moe.gguf \
  --cache build/datasets --output build/text-workload.json
```

Pass each manifest's token IDs to `benchmarking.run(..., prompts=[token_ids])`
with a matching `Case`. Use identical IDs and output limits for AR, speculation
and baselines. Retain the manifest as a run artifact, including its source
revisions, example IDs, tokenizer identity and document-cropping recipe. Select
runtime settings on the calibration split before measuring held-out examples.
Report acceptance, emitted tokens per round, complete decode time and output
agreement; acceptance alone does not measure a speedup. Cropped documents and a
fixed 128-token output budget are runtime probes rather than dataset accuracy
scores. Keep the original synthetic workload available for shape comparisons.

`B` means independent active requests. A runtime's token batch budget, prefill
chunk size and HTTP concurrency are different quantities; record them separately.
Provision cache space for prompt + output + speculative guard, rounded to the
backend's block size. Check actual allocations and device binding limits before
attempting a large batch.

For a hybrid model, logical KV bytes are
`2 × attention_layers × B × context × kv_heads × head_dimension × bytes_per_value`.
Count only attention layers; short-convolution layers retain fixed-size recurrent
state. Add that state, packed weights, cache metadata, scratch and backend
allocations separately. Use physical buffer storage width when planning VRAM.
Native FP16 caches occupy two bytes per value on CUDA and WebGPU; the latter
requires `shader-f16` support from the adapter.

## Run an AKAS measurement

Compile a model through [the shared compiler](../src/akas/compilation/README.md).
Then select a small shape before expanding to the full workload:

```sh
akas bench --definition akas.models.lfm2.model:LFM2 \
  --checkpoint /path/to/model.gguf --bundles build/bundles \
  --provider cuda --target sm_89 --kv-dtype float16 \
  --batches 1 --prompts 32 --generated 128 --prefill-chunk 32 \
  --graphs --warmup 2 --repeats 5 --out build/report.json \
  --change "Measure the packed projection schedule"
```

Omit `--batches` and `--prompts` to select all ten shapes. Match the compiled
program's request/context capacities, cache precision and geometries. CUDA and
WebGPU use separate target-qualified programs. Unsupported model/backend
combinations must remain visible as unsupported results.

Prefill includes the head and first predicted token. Decode measures the remaining
127 committed tokens. Reports separate input tok/s, decode tok/s, complete-output
tok/s, TTFT and time per subsequent token. Timings synchronize at both boundaries,
include required control uploads, and exclude compile, load, reset and warmup.
The protocol identifier prevents comparison with older reports that counted
additional decode steps after the first output.

Compare performance at the same KV-cache precision before attributing a gain to
the compiler. Keep F16 and FP8 results separately labeled. Changing the cache
can change logits, expert routing and greedy output tokens even when both paths
pass their implementation checks.

For a cache-precision comparison, retain identical prompts, checkpoint bytes,
tokenizer and sampling settings. Compare the generated text and token IDs, then
measure logits while supplying the same F16-generated continuation to both
paths. This separates numerical drift from differences that compound after the
first changed token. Record the prompt count, context lengths, generation limit,
EOS behavior and number of shared-history steps. A small numerical probe does
not establish corpus accuracy or task quality.

Paired diagnostic runs use the ordinary checksummed run archive. Campaigns can
attach their candidate run IDs through `precision_comparison_runs`; each candidate
links to its F16 reference through its parent run. The generated overview includes
one fixed comparison table, and portable report data includes both generated
outputs. Report export rejects failed probes, changed artifacts and mismatched
checkpoint or prompt identities.

Campaigns attach real-text runtime probes through `text_workload_runs`. Export
checks each sealed run against its checksummed `benchmark/text-workload.json`
artifact, including the tokenizer identity, assigned split, prompt IDs, complete
output counts and speculative accounting. The generated overview has a fixed
per-example table for each split and confidence threshold, covering AR and draft
prefill/decode, acceptance, tokens per round and output agreement. The portable
report retains both outputs and dataset provenance. These diagnostic rows remain
separate from the synthetic shape matrix.

Native reports also retain process peak host RSS, including loading and any
in-process compilation. Packed expert validation decodes small coefficient tiles;
it does not expand the entire model on the CPU. GGUF uploads release mapped
pages after transfer, and WebGPU submits writes in bounded staging batches.

For WebGPU performance, build Tensor with `TENSOR_BUILD_WEBGPU_NATIVE=1`
and enable `--graphs`. CUDA uses captured replay; WebGPU uses prepared command
encoding and submission. The record identifies Python versus native prepared
paths. Keep this execution choice fixed when comparing kernel candidates.

The pinned Tensor dependency supports the optional encoder build below. A C
compiler is required. Force a fresh build so an existing Python-only wheel is
not reused; keep the revision aligned with `uv.lock` when upgrading Tensor.

```sh
TENSOR_BUILD_WEBGPU_NATIVE=1 uv pip install --python .venv/bin/python \
  --reinstall --no-deps --no-cache \
  'tensor-workspace @ git+https://github.com/kreasof-ai/tensor@541d11ac9d93de92915082d09b68d7503f1c9741'
```

Use `--check module:callable` to attach independent model correctness evidence.
Kernel search qualification is recorded separately; a timed model is not
necessarily accuracy-qualified. Run one GPU measurement process at a time.

## Find a previous result

All published performance results use the generated archive described below.
Individual measurements use `akas.benchmark-run.v1` records; campaign reports use
`akas.campaign-report.v1`. Record a focused kernel experiment as an individual run
with its change description and parent run. Use a campaign for workload-wide
tables and baseline comparisons. Do not add dated result pages, handwritten
performance tables or research logs to the authoring guides.

The [campaign report index](../benchmarks/reports/README.md) provides complete
tables for a matrix: all requested configurations and shapes, available baseline
prefill/decode and burst rates, qualification failures, latency, settings and
selected kernel schedules. Reports live in `benchmarks/reports/`; compiled
programs and temporary build logs remain in `build/`.

The matrix runner generates these reports when an invocation finishes or is
interrupted gracefully. Export an existing campaign with:

```sh
python -m tools.campaign_report --campaign build/standard-campaign/campaign.json
```

Each report keeps its campaign snapshot and portable `report.json`. Its Markdown,
CSV tables and selected schedules can be regenerated after cleaning build files:

```sh
python -m tools.campaign_report --render benchmarks/reports/REPORT_ID/report.json
```

Baseline selection uses matching full-workload records completed before the
campaign began. Failed profiles retain their measured values and qualification
status. Missing measurements remain empty. Timing protocol labels distinguish
native engine invocations, HTTP bursts and native kernel phases; the tables do
not manufacture speedups across these boundaries.

Every campaign uses `reports/<start-UTC>-<campaign-identity>/` with this same bundle:

| File | Contents |
| --- | --- |
| `README.md` | Date, measured commit/source tree, hardware/software, settings, coverage and outcomes |
| `COMPARISON.md` | Prefill, decode and complete-output tables for every requested shape |
| `AKAS.md`, `BASELINES.md` | Every configuration, phase protocol, latency, qualification and evidence link |
| `CANDIDATES.md` | Every search attempt, its budget, outcome and failure reason |
| `measurements.csv` | All requested AKAS and baseline configuration/shape rows |
| `candidate-measurements.csv`, `candidates.csv` | Measured search candidates and every attempt, including failures |
| `SCHEDULES.json` | Selected implementations, parameters, geometries and qualification evidence |
| `report.json`, `campaign.json` | Portable versioned report data and exact campaign snapshot |

Generate these files with `tools.campaign_report`; edit the underlying records
only through the measurement tools. Keep sealed historical evidence unchanged.
Use the shared renderer for presentation changes and regenerate existing tables.
An incompatible data change requires a new schema version and explicit support
for older reports. New shapes belong in a versioned workload definition, which
the report expands without removing missing or unsupported configurations.

Before publishing, run the same read-only check used by CI:

```sh
python -m tools.campaign_report --check
```

The check rejects unknown schemas, missing configurations or shapes, missing
metrics, added report files, handwritten changes to generated tables and a stale
report index. It regenerates the archive in temporary storage and compares every
file; it does not compile models or run benchmarks. Keep benchmark documentation
under `benchmarks/` limited to its guide entry point, generated indexes and these
report bundles. Sealed files under `runs/` remain measurement evidence.

`akas bench` automatically writes a durable run under `benchmarks/runs/` as well as
`--out`. Use `--history /path/to/history` to select another archive or
`--no-history` for disposable local checks. Browse [the generated results
index](../benchmarks/RESULTS.md), or query:

```sh
akas history list --hardware L40S
akas history list --model qwen --runtime akas --status passed --purpose benchmark
akas history index
```

A record contains:

| Field | What it establishes |
| --- | --- |
| UTC start/end, run ID, status | When the measurement ran and whether it completed |
| Git commit, source hashes, source snapshot | The actual measured code, including uncommitted files |
| Runtime/dependency versions and compiler evidence | The installed software and compilation identity |
| Hardware, provider and target | Which device/backend the result applies to |
| Workload definition/hash and exact shapes | Prompt, output, batch, cache depth and raggedness |
| Metadata and program configuration | Checkpoint digest, precision, capacity, chunk size, graphs, policy, seed and timing repetitions |
| Results and correctness evidence | Raw samples, summaries, failures and independent checks |
| Search phase, parameters, selected implementation and schedule | The qualified winning geometry for each operation |
| Change description and parent run | Which kernel, algorithm, schedule or configuration was measured |
| Artifact paths and SHA256 digests | Where retained evidence lives and whether it changed |

The commit alone is insufficient for a dirty checkout. Every run retains a
compressed snapshot and per-file hashes of measurement-relevant source. Source
changes during measurement invalidate the run. A wheel without Git metadata is
identified by its captured package source and dependencies; its commit remains
unknown. Do not invent a revision.

Program evidence includes qualified phase schedules and compressed generated
kernel IR/shader metadata, identifying the compiled kernels that actually ran.
`run.json` is the authoritative record. `index.json` and `RESULTS.md` can be
rebuilt. Large compiler traces, binaries and model files belong in `build/` or
content-addressed external storage. A useful small log or benchmark harness can
be retained beside its run with a checksum. Keep dates and results out of the
authoring guides.

## Improve a previous schedule

Export a qualified phase from a passed run:

```sh
akas history seed RUN_ID --phase decode --batch 64 --chunk 1 \
  --provider cuda --target sm_89 --out build/incumbent.json
```

Pass that profile to `akas bench --compile --schedule-profile build/incumbent.json`
and name the previous record with `--parent-run RUN_ID`. The compiler requalifies
and retimes the incumbent before comparing new candidates. Previous timings are
never evidence that a changed kernel remains correct or fast. Provider, target,
phase and geometry must match; modified evidence or incomplete runs cannot
supply an incumbent.

An exported profile supplies an incumbent. The compiler separately saves partial
beam progress and compiled artifacts under the program directory. Keep that
directory to resume exploration; selected resumed schedules qualify and time
again. Use [the optimization workflow](optimization.md) for isolated workers,
wall-clock deadlines, complete-workload selection and a serving deployment.
A time budget cannot guarantee that a search will beat every alternative.

## Compare runtimes fairly

Pin every runtime and checkpoint revision. Share identical GGUF bytes with
llama.cpp. Record weight, activation, accumulation, KV and recurrent precision
separately: BF16 input files can still execute FP16 arithmetic. Quantized FP8 KV
and integer Q8_0 KV are different experiments.

First run [short baseline probes](../tools/baselines/README.md) for the selected models and generation modes. Verify exact output
counts, finite results, cache behavior and greedy AR/MTP equivalence. Eager-mode
preflight establishes execution, not competitive throughput. Final baselines
need their optimized graph/attention settings, stable warmups and repeated
measurements. Record their tuning effort too.

Use fixed native batches for model/kernel throughput. Use a separate serving
workload for TTFT/TPOT, queueing, latency percentiles and goodput under an arrival
rate or concurrency limit. Count committed speculative tokens, including draft,
verification and repair costs. Never compare proposed-token rates with AR
output-token rates. Record observed acceptance rates and effective GPU batches.

`SpeculativeSession` measures the whole block round through the same coordinator
used by serving, including draft execution, readbacks and cache repair. It clips
the final round to the requested output count. `native_phase_mode` identifies the
measured mode; older records marked `ar-target` retain AR target phase timings
even when their separate serving samples enable speculation. Native speculation
counters belong to native samples and are excluded from serving counters.

Keep comparisons within matching model, shape, precision and protocol. Report
AR-to-AR and MTP-to-MTP, then the best mode each runtime offers. Preserve losing,
OOM, unsupported and interrupted rows alongside wins. Hardware changes require
new measurements; L40S results are not L4 measurements. MFU and roofline analysis
explain bottlenecks; throughput and latency under equal quality constraints are
the performance results.

## Prepare an unattended campaign

Check model-level support before expanding a matrix. An individual codec or
kernel can support a backend while the complete model graph does not.

| Requested AKAS path | Model-level constraint |
| --- | --- |
| LFM2.5 dense text, safetensors or GGUF | CUDA and WebGPU composition; validate each format/cache/shape combination |
| LFM2 MoE text, GGUF | Packed expert execution on CUDA and WebGPU; safetensors MoE loading is not implemented |
| LFM2.5 with BF16 source weights | Floating matrix execution uses FP16 with FP32 intermediates; this is not a strict BF16 execution baseline |
| Qwen3.5 dense text, safetensors or GGUF | CUDA composition with BF16 KV; WebGPU and FP8 KV require implementation |
| Qwen3.5 dense MTP | Native checkpoint branch exists on CUDA; GGUF MTP is unavailable |

Before a long command, establish usable devices, qualify each requested path,
verify peak VRAM/RSS at the largest requested shape, run all baseline probes, and estimate runtime from a
small representative pilot. Freeze code and workload inputs. Apply explicit
candidate, family and campaign deadlines, preserve qualified incumbents, and
record unfinished work as budget exhaustion. Spend additional tuning time on
measured whole-model bottlenecks rather than allocating an hour to every matrix
cell. Keep the campaign's limits in its configuration and results, not in
another dated planning document.
