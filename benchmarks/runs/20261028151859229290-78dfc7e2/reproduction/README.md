# LFM3 QAD Q4_0 on Radeon RX 6700 XT

The October 28, 2026 showcase uses completed wall decode, FP16 KV, one request,
131,072 context capacity and 128 output tokens (127 timed decode steps).
Prefill rates are one completed-wall observation per prefix, and can include
first-use phase setup; they are not warmed medians. Decode has two warmups and
five samples from complete restored real-prefill
KV/recurrent state and its initial greedy prediction. Load, prefill, restoration
and validation are excluded. Prompts use identical fixed synthetic IDs, never
fabricated KV. No end-to-end or prefill parity claim is made.

## Retrieve the settings

Fetch this run with `python -m tools.benchmark_archive fetch runs 20261028151859229290-78dfc7e2`.
Use the verified directory printed by that command as RUN_DIRECTORY. The recipe,
selected 8K schedule template and policy are under `reproduction/`. The compiler
transforms capacity to 131,072 and inserts the explicit cooperative schedules.

Install Python 3.12 and the locked dependencies with compiler and WebGPU extras.
Download the official `LiquidAI/LFM3-1.7B-GGUF` revision
`b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf`, file
`LFM3-1.7B-QAD-Q4_0.gguf`, as `build/checkpoints/lfm3-1.7b/q4_0.gguf`.
Its SHA256 must be `43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452`.
The baseline uses official llama.cpp b11946 Vulkan Windows binaries, revision
`8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`, under
`build/baselines/llama-b11946-vulkan/`. The C API checks `llama.dll` SHA256
`5032cde7cafd0787e3287ec05c67a6c510f635e85b99e4ff077847da9027a9d9`.
Dependency versions, native-library hashes and driver details are in `environment.json`.

## Compile and qualify

Replace RUN_DIRECTORY with the fetched directory. Run from the source checkout.

```powershell
python -m tools.lfm3_long_decode compile --context 131072 --program build/lfm3-vulkan/long-32 --splits 32 --schedule-profile RUN_DIRECTORY/reproduction/schedules.json --policy RUN_DIRECTORY/reproduction/policy.json --out build/lfm3-vulkan/short-compile.json
python -m tools.lfm3_long_decode compile --context 131072 --program build/lfm3-vulkan/long-final --splits 512 --cooperative-decode --score-tile 256 --score-unroll --value-tile 64 --schedule-profile RUN_DIRECTORY/reproduction/schedules.json --policy RUN_DIRECTORY/reproduction/policy.json --out build/lfm3-vulkan/long-compile.json
python -m tools.lfm3_qualify --provider webgpu --cooperative-decode --cooperative-score-tile 256 --cooperative-score-unroll --cooperative-value-tile 64 --out build/lfm3-vulkan/qualification.json
```

## Measure and export

Use the qualification run ID printed above as QUALIFICATION_RUN. Omitting
`--prefixes` measures 32, 512, 2,048, 8,192, 16,384, 32,768, 65,536 and 130,944.

```powershell
python -m tools.lfm3_long_decode llama --warmup 2 --repeats 5 --out build/lfm3-vulkan/llama.json
python -m tools.lfm3_long_decode native --program build/lfm3-vulkan/long-final --short-program build/lfm3-vulkan/long-32 --short-program-max-prefix 8192 --warmup 2 --repeats 5 --qualification-run QUALIFICATION_RUN --state-cache-dir build/lfm3-vulkan/real-prefill-states --out build/lfm3-vulkan/native.json
python -m tools.lfm3_showcase --runs NATIVE_RUN LLAMA_RUN --assets-run 20261028151859229290-78dfc7e2
python -m tools.lfm3_vulkan_demo --program build/lfm3-vulkan/long-final --out build/lfm3-vulkan/text-demo.json
```

Replace NATIVE_RUN and LLAMA_RUN with the printed run IDs. The showcase exporter
also accepts multiple non-overlapping runs per runtime, as in the retained sweep.
After fetching its report, regenerate the figure with:

```sh
python -m tools.lfm3_vulkan_plot REPORT_DIRECTORY/report.json build/throughput.svg
```

The maximum-prefix state may be reused with `--input-state-cache` and
`--state-producer-run` instead of `--state-cache-dir`. Import checks every state
array hash, checkpoint/prompt/program identity, seed, dimensions and precision.
Only decode scheduling can change. The retained maximum-prefix producer is
`20261028130402880342-2c41aef6`; its prefill took about 34 minutes. Snapshot arrays,
model weights and executables are disposable build files, not archive payloads.

## Precision and interpretation

Original quantized weights stay packed, including the official tied Q6_K head;
KV is FP16 with explicit rounding. Decode activations, scores, softmax and value
sums are FP32. Mixed prefill uses Q16 or short FP16 accumulations in selected
projections, as declared in the program and validated independently.
Long decode shares KV reads across GQA heads, stages 256-key score tiles in
64-channel pieces within 32 KiB shared storage and hierarchically merges 512
value partitions. Prefixes through 8,192 select the 32-partition program.

All eight decode medians exceed the pinned baseline. Full output trajectories
match at three of eight synthetic prefixes; this is speed parity, not universal
text identity. The natural-language demo matches 88/88 tokens between engines.
Independent qualification passed eight checks, including six GPU model cases
with existing error bounds and exact greedy-token agreement. The repository-wide
test suite was not established as passing on Windows (older campaign tests
require Unix fcntl). Focused checks and qualified source evidence are retained.
