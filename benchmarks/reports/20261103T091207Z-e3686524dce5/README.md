# Campaign benchmark report

UTC: 2026-11-03T09:12:07.139029Z → 2026-11-03T09:13:17.072688Z. Status: `complete`. Active time: 0.02 hours; per-configuration ceiling: 0.50 hours.

Source: `None`; measured source tree: `d8238d0ae1dcd7c266f2407d0eed9ad2de67c6c37ccc21231f69d4890a8e0603`.

Qualified AKAS deployments: **1/1**. Qualified baseline profiles: **1/2**.

- [Side-by-side tables for every requested shape](COMPARISON.md)
- [All AKAS configurations and per-shape measurements](AKAS.md)
- [All baseline profiles and per-shape measurements](BASELINES.md)
- [Every candidate attempt](CANDIDATES.md)
- [Full measurement CSV](measurements.csv) · [Candidate measurement CSV](candidate-measurements.csv) · [Candidate outcomes CSV](candidates.csv)
- [Selected kernel implementations and schedules](SCHEDULES.json) · [Portable report data](report.json) · [Campaign snapshot](campaign.json)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

## Hardware and measurement provenance

| Hardware | Driver | Baseline/runtime | Version or revision | Source | Evidence |
| --- | --- | --- | --- | --- | --- |
| NVIDIA H200 |  | akas / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.1.0 | `None` | [20261103091212903514-d753d574](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103091212903514-d753d574/run.json) |
| NVIDIA H200 | 610.57.04 | vllm / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.31.0 | `None` | [20261102190120772675-8245f839](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190120772675-8245f839/run.json) |
| NVIDIA H200 | 610.57.04 | sglang / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.5.21 | `None` | [20261102190540048720-a1e7c623](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190540048720-a1e7c623/run.json) |

Captured host: 20 logical CPUs; Linux-4.19.0-gvisor-x86_64-with-glibc2.39.

```text
index, uuid, name, driver_version, memory.total [MiB], power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
0, GPU-056ec9c3-616f-ea39-ad79-3b58bec5a89b, NVIDIA H200, 610.57.04, 143771 MiB, 700.00 W, 1980 MHz, 3201 MHz
```

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar | float16 | 2048 | 16 | True | 1 | 1.000x | 51.1 | [20261103091212903514-d753d574](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103091212903514-d753d574/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-24b-a2b-fp8 | fp8 | cuda | float16 | ar | [qualified; complete](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | 1.2 | [20261103091212903514-d753d574](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103091212903514-d753d574/run.json) |  |

## Candidate outcomes

1 attempts: 1 passed.

| Rejection or deadline | Attempts |
| --- | ---: |
