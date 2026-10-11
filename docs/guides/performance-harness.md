# Daily performance measurements

`akas bench` measures a native session's prompt processing and token generation.
The harness is model-independent; LFM2's portable batched reference is the first
complete example. Like [llama-benchy](https://github.com/eugr/llama-benchy), it
reports prompt/decode throughput at different context depths. Native timing avoids
network effects and explicitly distinguishes aggregate throughput from throughput
per request.

```bash
akas bench \
  --definition akas.models.lfm2.model:LFM2Reference \
  --checkpoint /path/to/LFM2-350M \
  --bundles build/lfm-reference \
  --compile --provider cuda --kv-dtype fp8 --graphs \
  --batches 1 4 16 64 256 --prompts 32 2048 --depths 0 \
  --generated 32 --warmup 1 --repeats 5 \
  --out build/lfm-baseline.json
```

A safetensors directory or a GGUF file works with the same command. Change
`--provider webgpu` to compile/run the same model composition for WebGPU.
`--target` defaults to the selected device's architecture. Omit `--graphs` for
ordinary submission. WebGPU uses its prepared dispatch plan when graphs are enabled.

Start a fast iteration with `--batches 1 16 --prompts 32 --generated 16`. Add
`--depths 0 2048 8192` to check decode when the cache is already populated, and
`--mixed` to stagger prompt lengths across slots. The bank is compiled once per
batch, sized for the largest case, and reused at every live sequence length.
Running without `--compile` uses an existing bank and needs no compiler packages.

Keep the baseline, change kernels/schedules, and repeat the identical command with:

```bash
--baseline build/lfm-baseline.json --max-regression 0.10 --out build/lfm-candidate.json
```

Throughput loss beyond the requested fraction exits with status 2. Failed build,
load, allocation, or runtime cases are recorded as errors and exit with status 1.
Reports are written after each case so an interrupted run retains completed results.
Comparison requires matching checkpoint content, definition, device, provider,
cache dtype, target, graphs, seed, protocol, and workload cases. Compiler/load times
are excluded; reset and context preparation are excluded from pp/tg but context
preparation has its own duration. Every timed phase synchronizes before and after.

Reports include every repetition, mean/stdev/median/p95/min/max, pp and aggregate
tg tokens/s, tg tokens/s/request, TTFT, mean time per decode step, allocations,
seed, device details, revision, and timing rules. TTFT means prefill through the
first greedy sample; it has no network component. Decode measures the following
N token steps; the first sample is accounted for in prefill. TTFT after a populated
context excludes the context preparation time. Prompts use reproducible random
valid token IDs; this is a performance workload rather than a language-quality test.

Use `--check module:callable` to require a numerical preflight per case. The
callback receives `(session, case)` and returns JSON evidence with `correct=True`.
For small LFM fixtures:

```bash
--check examples.lfm2.validation:check
```

This runs independent NumPy equations before timing and records the errors.
Without a callback the report states that correctness was not checked. Qualification
and performance measurements serve separate purposes; fast incorrect kernels must
not be promoted into tuning profiles.

The Python API is `akas.benchmarking.Case`, `matrix`, and `run`. A session needs
`batch`, `capacity`, `config.vocab`, `allocated_bytes`, `device.synchronize()`,
`reset()`, `prefill(token_lists, read=False)`, and `decode(count, read=False)`.
Another model can supply that protocol without changing the runner or reports.
The measured operation schedules can be selected through the existing
`akas.compilation.tuning` API and passed to reference compilation as `schedules`.

The reference runtime serializes prefill in time, and its projections use F32
reductions. CUDA attention loops over live length. Portable WebGPU attention loops
over compiled capacity with a causal mask to keep workgroup barriers uniform.
Consequently these measurements describe the readable baseline; optimized packed
LFM bundles remain available separately. Memory/binding limits still apply, and
software WebGPU adapters are identified as CPU adapters in each report.

Use `--schedule-profile path/to/profile.json --compile` to rebuild with a
qualified Tensor schedule profile before measuring. The profile's hash is
recorded; provider and target matching are enforced by shared compilation.
