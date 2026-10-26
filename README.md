<p align="center">
  <img src="logo/logo-black.svg#gh-light-mode-only" alt="Tensor" width="420">
  <img src="logo/logo-white.svg#gh-dark-mode-only" alt="Tensor" width="420">
</p>

<h1 align="center">Compiler-first model inference.</h1>

<p align="center">
  Compose models in pure Python, fuse quantization codecs at compile time. <br> 
  Serve AOT-compiled .tbin programs without PyTorch or TileLang at runtime.
</p>

---

AKAS is an inference engine built on [Tensor](https://github.com/kreasof-ai/tensor).
It owns model definitions, checkpoint loading, request scheduling, persistent state,
generation and serving. Tensor provides compilation and device execution.

## Benchmark showcase

**LFM2.5-24B-A2B · Q4_0 weights · NVIDIA L4 (23,034 MiB) · CUDA**

Selected results from the [October 23, 2026 comparison](https://github.com/kreasof-ai/akas/blob/0b702210b8c4d722b1c1f420e00272f418658136/benchmarks/reports/20261023T155400Z-cdcde64a14b5/COMPARISON.md),
with one request generating 128 tokens. All rates are tokens per second; FP16 and
FP8 below refer to the KV cache dtype.

![Prefill and decode throughput across all measured prefix lengths for autoregressive and Dspark speculative decoding, comparing AKAS FP16 KV, AKAS FP8 KV and llama.cpp FP16 KV on NVIDIA L4.](https://raw.githubusercontent.com/kreasof-ai/akas/0b702210b8c4d722b1c1f420e00272f418658136/benchmarks/reports/20261023T155400Z-cdcde64a14b5/throughput.svg)

<details>
<summary>Selected measurements (tokens per second)</summary>

| Mode | Phase | Prefix tokens | AKAS FP16 KV | AKAS FP8 KV | llama.cpp FP16 KV |
| --- | --- | ---: | ---: | ---: | ---: |
| Autoregressive | Prefill | 2,048 | 2,631.2 | 2,648.7 | 3,279.2 |
| Autoregressive | Prefill | 16,384 | 4,259.0 | 4,220.0 | 3,112.6 |
| Autoregressive | Prefill | 130,944 | 2,771.7 | 2,755.7 | 1,593.4 |
| Autoregressive | Decode | 2,048 | 128.9 | 129.0 | 129.9 |
| Autoregressive | Decode | 16,384 | 112.7 | 118.6 | 111.0 |
| Autoregressive | Decode | 130,944 | 55.7 | 75.1 | 51.3 |
| Dspark speculative | Prefill | 2,048 | 3,430.9 | 3,409.4 | 2,544.8 |
| Dspark speculative | Decode | 2,048 | 256.1 | 191.0 | 171.2 |

</details>

These are reported phase rates from a diagnostic context sweep with one retained
sample per shape. AKAS uses `native-token-step-v2`; llama.cpp uses
`llama-server-eval-real-text-v1`. Their timing boundaries differ, so these values
do not establish equivalent kernel timings or end-to-end speedups. Complete-burst
throughput was not measured. Speculative performance varies with context and draft
acceptance; the [full comparison](https://github.com/kreasof-ai/akas/blob/0b702210b8c4d722b1c1f420e00272f418658136/benchmarks/reports/20261023T155400Z-cdcde64a14b5/COMPARISON.md)
includes shorter and longer contexts, including speculative slowdowns. See the
[campaign report](https://github.com/kreasof-ai/akas/blob/0b702210b8c4d722b1c1f420e00272f418658136/benchmarks/reports/20261023T155400Z-cdcde64a14b5/README.md)
for hardware, retained settings, qualification and output agreement.

Regenerate the plot with `uv run python tools/plot_comparison.py`.

**LFM2.5-2.6B · Q4_0 weights · AMD Radeon RX 6700 XT (12 GB) · Vulkan**

Results from the [October 25, 2026 comparison](https://github.com/kreasof-ai/akas/blob/1313601fac2716250811b46f727493a64df4b82d/benchmarks/reports/20261024T191042Z-60f28059ed28/COMPARISON.md),
with one request generating 128 tokens. All rates are tokens per second;
both runtimes use FP16 KV.

![Prefill and decode throughput across prefixes 32 through 130,944, comparing AKAS and llama.cpp with FP16 KV on AMD Radeon RX 6700 XT.](https://raw.githubusercontent.com/kreasof-ai/akas/1313601fac2716250811b46f727493a64df4b82d/benchmarks/reports/20261024T191042Z-60f28059ed28/throughput.svg)

<details>
<summary>Measurements (tokens per second)</summary>

| Mode | Phase | Prefix tokens | AKAS FP16 KV | llama.cpp FP16 KV |
| --- | --- | ---: | ---: | ---: |
| Autoregressive | Prefill | 32 | 458.0 | 921.0 |
| Autoregressive | Prefill | 512 | 974.8 | 2,596.5 |
| Autoregressive | Prefill | 2,048 | 953.1 | 2,470.2 |
| Autoregressive | Prefill | 8,192 | 835.6 | 1,997.9 |
| Autoregressive | Prefill | 16,384 | 712.8 | 1,556.3 |
| Autoregressive | Prefill | 32,768 | 556.0 | 1,068.5 |
| Autoregressive | Prefill | 65,536 | 386.2 | 655.5 |
| Autoregressive | Prefill | 130,944 | 239.5 | 367.6 |
| Autoregressive | Decode | 32 | 169.8 | 165.4 |
| Autoregressive | Decode | 512 | 168.5 | 161.5 |
| Autoregressive | Decode | 2,048 | 163.3 | 160.8 |
| Autoregressive | Decode | 8,192 | 152.9 | 151.5 |
| Autoregressive | Decode | 16,384 | 139.8 | 144.0 |
| Autoregressive | Decode | 32,768 | 122.2 | 127.9 |
| Autoregressive | Decode | 65,536 | 97.1 | 105.2 |
| Autoregressive | Decode | 130,944 | 68.2 | 78.2 |

</details>

These are reported phase rates with one retained sample per prefix. AKAS preloads
its phases; llama.cpp has one warmup per prefix. Timing boundaries differ, and
complete-burst throughput was not measured for AKAS. See the
[tuning report](https://github.com/kreasof-ai/akas/blob/1313601fac2716250811b46f727493a64df4b82d/benchmarks/reports/20261024T191042Z-60f28059ed28/TUNING.md)
for mixed-precision prefill, FP32 decode arithmetic, qualification and output
agreement, and the [replay instructions](https://github.com/kreasof-ai/akas/blob/2a5fff2394e5822373e18fdb9f072c9d204169e8/benchmarks/runs/20261025123404575934-98dca4dc/REPRODUCE.md)
for retained programs and plotting commands.

Regenerate the plot after retrieving the archived report:

```sh
python -m tools.benchmark_archive fetch reports 20261024T191042Z-60f28059ed28
uv run python -m tools.vulkan_plot REPORT_DIRECTORY/report.json build/throughput.png
```

Replace `REPORT_DIRECTORY` with the verified directory printed by the fetch command.

## Spark X2.6 on NVIDIA L4

![Spark X2.6 Q4_K_M throughput, observed sample ranges and 2× llama.cpp prefill target.](https://raw.githubusercontent.com/kreasof-ai/akas/566e756f90540e01b03628558e33444c94c18a40/benchmarks/runs/20261026103015172978-b083ceb8/figures/throughput.svg)

[Spark model support](src/akas/models/spark/README.md) accepts BF16 and FP8 safetensors and packed GGUF. BF16/FP8 imports execute with FP16 matrices. This full comparison uses only the official Q4_K_M GGUF and FP16 KV. The optimized configuration enables the optional FP16 prefill weight cache, shared rotary tables and optimized attention, with a 1024-token maximum prefill chunk.

Decode medians exceed llama.cpp at 8/8 prefixes; all retained decode samples exceed all corresponding llama.cpp samples at 8/8. The requested 2× llama.cpp prefill target is met at 0/8 prefixes. These observations describe this measured configuration.

| Prefix tokens | AKAS prefill tok/s | llama.cpp prefill tok/s | AKAS decode tok/s | llama.cpp decode tok/s | Estimated AKAS prefill MFU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1,075.4 | 844.1 | 71.2 | 63.7 | 6.74% |
| 512 | 4,606.0 | 4,179.9 | 69.8 | 65.6 | 29.33% |
| 2,048 | 4,734.2 | 4,542.6 | 68.5 | 50.6 | 30.92% |
| 8,192 | 4,394.3 | 4,359.9 | 64.9 | 41.3 | 30.42% |
| 16,384 | 3,990.7 | 4,038.9 | 60.3 | 55.4 | 29.63% |
| 32,768 | 3,339.7 | 3,136.7 | 52.0 | 46.8 | 28.13% |
| 65,536 | 2,500.2 | 2,138.9 | 41.7 | 37.2 | 26.06% |
| 130,944 | 1,667.2 | 1,337.8 | 30.2 | 27.5 | 24.02% |

[Full report, raw samples, output comparisons and selected schedules](https://github.com/kreasof-ai/akas/blob/9a0dcb83131433e0955515894c973c3897076e61/benchmarks/reports/20261026T100758Z-6d809fbde297/README.md). [Target assessment, MFU method and GPU telemetry](https://github.com/kreasof-ai/akas/blob/566e756f90540e01b03628558e33444c94c18a40/benchmarks/runs/20261026103015172978-b083ceb8/analysis/README.md).

One request produces 128 total output tokens. Values are medians of three retained repetitions after one warmup per prefix, using identical checkpoint bytes and synthetic prompt IDs. AKAS uses synchronized native phases; llama.cpp uses server eval phases. Loading is excluded and phase timing boundaries differ. MFU estimates useful model math relative to L4’s 121 TFLOP/s dense FP16 peak; it is not a hardware instruction counter.

[The original Spark comparison](https://github.com/kreasof-ai/akas/blob/ac2323d0ea0527aae1bfaded42ee80947770aefd/benchmarks/reports/20261025T235635Z-d80177ee179d/README.md) remains archived. Payloads are sealed on the artifact branches listed in [the archive catalog](benchmarks/archive.json).

## Getting started

Use Python 3.12. `uv.lock` pins Tensor to a published Git revision.

```sh
uv sync --locked --extra compiler --extra serving
uv run --no-sync python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
```

The compiler is needed to build programs. Precompiled programs run without compiler
packages. Optional extras are `webgpu`, `serving`, `reference`, and `cloud`.

Optimize a serving deployment by giving the checkpoint, workload and time budget.
AKAS detects the model/backend and chooses inference settings:

```sh
akas optimize --model /path/to/model.gguf --workload standard-v1 \
  --budget 30m --out build/deployment
akas serve --deployment build/deployment/deployment.json
```

Install the `serving` extra for this workflow. See the
[optimization guide](docs/optimization.md) for resource constraints, measurements
and resuming a search. To control compilation directly, use the model definition
and shared compiler. Kernel beam search runs by default:

```sh
akas compile --definition akas.models.lfm2.model:LFM2 \
  --checkpoint /path/to/model.gguf --out build/model \
  --config '{"geometries":[[1,1],[1,128]],"max_requests":1,"max_context":4096,"provider":"cuda","target":"sm_89"}'
akas generate --definition akas.models.lfm2.model:LFM2 \
  --model /path/to/model.gguf --bundle build/model --prompt Hello --max-tokens 32
```

Supported checkpoint formats and operation implementations depend on the model
and provider. See the [model declarations](src/akas/models/README.md).

Use `akas bench` for repeated prompt/decode measurements across batch sizes and
context lengths. `akas bench --help` lists compilation, timing, baseline comparison
and scaling regression options. `akas serve --factory module:callable` runs the
shared HTTP server with a caller-supplied service factory.

The package layout is documented beside the implementation:

- [Models](src/akas/models/README.md): architecture, checkpoint mapping and composition.
- [Quantization](src/akas/quantization/README.md): format-owned storage, codecs and kernels.
- [Operations](src/akas/ops/README.md): shared TileLang kernels, backend decisions and a beginner authoring guide.
- [Compilation](src/akas/compilation/README.md): graph composition, optimization, program building and default tuning.
- [Optimization](src/akas/optimization/README.md): autonomous compilation, serving measurement and deployment selection.
- [Benchmarking](src/akas/benchmarking/README.md): measurements, reproducible run history and schedule reuse.
- [Execution](src/akas/execution/README.md): shared request state and phase execution.

[The complete bigram model](examples/models/bigram/) demonstrates the uniform
model declaration and request lifecycle. [The residual graph](examples/graphs/residual/)
demonstrates standalone numerical composition using existing operations. `tests/support` contains synthetic checkpoints and
independent numerical oracles used by core tests.

```sh
uv run --no-sync python -m pytest
uv build
```

Read the [guides](docs/README.md) to extend the engine.
[Benchmark history](benchmarks/README.md) records measurements by source, hardware,
workload and configuration, with qualified schedules available for later searches.
