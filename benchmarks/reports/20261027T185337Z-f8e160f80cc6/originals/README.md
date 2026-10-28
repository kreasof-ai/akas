# Campaign benchmark report

UTC: 2026-10-27T18:53:37.181202Z → 2026-10-27T19:04:41.388897Z. Status: `completed`. Active time: 0.15 hours; per-configuration ceiling: not set.

Source: `ceb52a49d83440274c773737ce6841b95de3d841`; measured source tree: `24b72e222f9318543d0d2600c137ce24f86e85de73698c5c85f30bf5b545f103`.

Qualified AKAS deployments: **1/1**. Qualified baseline profiles: **1/1**.

- [Side-by-side tables for every requested shape](COMPARISON.md)
- [All AKAS configurations and per-shape measurements](AKAS.md)
- [All baseline profiles and per-shape measurements](BASELINES.md)
- [Every candidate attempt](CANDIDATES.md)
- [Full measurement CSV](measurements.csv) · [Candidate measurement CSV](candidate-measurements.csv) · [Candidate outcomes CSV](candidates.csv)
- [Selected kernel implementations and schedules](SCHEDULES.json) · [Portable report data](report.json) · [Campaign snapshot](campaign.json)

Both-phase median criterion: **failed**. [Per-prefix acceptance results](PARITY.json).

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

## Hardware and measurement provenance

| Hardware | Driver | Baseline/runtime | Version or revision | Source | Evidence |
| --- | --- | --- | --- | --- | --- |
| NVIDIA L4 |  | akas / lfm3-1.7b / q4_0 / ar | 0.1.0 | `ceb52a49d83440274c773737ce6841b95de3d841` | [20261027185337181202-06e9f76f](../../runs/20261027185337181202-06e9f76f/run.json) |
| NVIDIA L4 |  | llama / lfm3-1.7b / q4_0 / ar | version: 0.7.0-dev (build 1, commit 19ff817) built with GNU 13.3.0 for Linux x86_64 | `ceb52a49d83440274c773737ce6841b95de3d841` | [20261027185932613950-5409e768](../../runs/20261027185932613950-5409e768/run.json) |

Captured host: 4 logical CPUs; Linux-6.12.110-135.202.amzn2023.x86_64-x86_64-with-glibc2.39.

```text
index, uuid, name, driver_version, memory.total [MiB], power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
0, GPU-ea5fd1c5-06d9-4852-3cad-193c77543d92, NVIDIA L4, 595.91.07, 23034 MiB, 72.00 W, 2040 MHz, 6251 MHz
```

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lfm3-1.7b-q4_0-cuda-float16-ar | float16 | 512 | 1 | True | 1 | — | — | [20261027185337181202-06e9f76f](../../runs/20261027185337181202-06e9f76f/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm3-1.7b | q4_0 | cuda | float16 | ar | [qualified; passed](AKAS.md#lfm3-1.7b-q4_0-cuda-float16-ar) | 5.0 | [20261027185337181202-06e9f76f](../../runs/20261027185337181202-06e9f76f/run.json) |  |

## Candidate outcomes

0 attempts: .

| Rejection or deadline | Attempts |
| --- | ---: |
