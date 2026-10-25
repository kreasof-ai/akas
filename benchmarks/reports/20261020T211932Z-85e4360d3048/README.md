# Campaign benchmark report

UTC: 2026-10-20T21:19:32.758011Z → 2026-10-20T22:00:30.225794Z. Status: `complete`. Active time: 0.68 hours; per-configuration ceiling: not set.

Source: `98fedcc76c834d813c86c92bf28e94497c79dfd6`; measured source tree: `0650fee2356a380d3d898fa993e285312317535307d5874e65c0833bd4b7e6df`.

Qualified AKAS deployments: **1/1**. Qualified baseline profiles: **2/2**.

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
| NVIDIA L4 |  | akas / lfm2.5-24b-a2b / q4_0 / ar | 0.1.0 | `98fedcc76c834d813c86c92bf28e94497c79dfd6` | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-24b-a2b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `7bd2befec004b622cc74bc11f6ef5d1558426d0c` | [20261018202355478915-6adad97d](https://github.com/akbar2habibullah/akas/blob/d7bc136e8e5ac9b29639cd8cc1e60c77d71be2e4/benchmarks/runs/20261018202355478915-6adad97d/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-24b-a2b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `7bd2befec004b622cc74bc11f6ef5d1558426d0c` | [20261018203703691867-28072a0a](https://github.com/akbar2habibullah/akas/blob/c6a8dc66246bde32619bc1f587caf528c2eeb942/benchmarks/runs/20261018203703691867-28072a0a/run.json) |

Captured host: 4 logical CPUs; Linux-6.12.103-129.197.amzn2023.x86_64-x86_64-with-glibc2.39.

```text
index, uuid, name, driver_version, memory.total [MiB], power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
0, GPU-b03cd627-998c-2476-6169-f62b483e2c27, NVIDIA L4, 595.91.07, 23034 MiB, 72.00 W, 2040 MHz, 6251 MHz
```

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lfm2.5-24b-a2b-q4_0-cuda-fp8-ar | fp8 | 4096 | 1 | True | 1 | 1.000x | — | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-24b-a2b | q4_0 | cuda | fp8 | ar | [qualified; complete](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-fp8-ar) | 41.0 | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |  |

## Candidate outcomes

1 attempts: 1 passed.

| Rejection or deadline | Attempts |
| --- | ---: |
