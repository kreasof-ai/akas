# Campaign benchmark report

UTC: 2026-10-19T14:16:44.589723Z → 2026-10-20T09:51:36.249948Z. Status: `interrupted`. Active time: 19.58 hours; per-configuration ceiling: 48.00 hours.

Source: `d1e28a0409b1b9334d47a2f902155279393c47f6`; measured source tree: `feb845bedb0196ff1ffe042b8fdaf829a9193cbba8af3e4422944335f3c88b9e`.

Qualified AKAS deployments: **1/2**. Qualified baseline profiles: **2/2**.

- [Side-by-side tables for every requested shape](COMPARISON.md)
- [All AKAS configurations and per-shape measurements](AKAS.md)
- [All baseline profiles and per-shape measurements](BASELINES.md)
- [Every candidate attempt](CANDIDATES.md)
- [Full measurement CSV](measurements.csv) · [Candidate measurement CSV](candidate-measurements.csv) · [Candidate outcomes CSV](candidates.csv)
- [Selected kernel implementations and schedules](SCHEDULES.json) · [Portable report data](report.json) · [Campaign snapshot](campaign.json)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; the measured LFM floating path executes FP16. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

## Hardware and measurement provenance

| Hardware | Driver | Baseline/runtime | Version or revision | Source | Evidence |
| --- | --- | --- | --- | --- | --- |
| NVIDIA L4 |  | akas / lfm2.5-24b-a2b / q4_0 / ar | 0.1.0 | `fb5dd2582a704d7704339361f2ea3e45d4333f1a` | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-24b-a2b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `7bd2befec004b622cc74bc11f6ef5d1558426d0c` | [20261018202355478915-6adad97d](../../runs/20261018202355478915-6adad97d/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-24b-a2b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `7bd2befec004b622cc74bc11f6ef5d1558426d0c` | [20261018203703691867-28072a0a](../../runs/20261018203703691867-28072a0a/run.json) |

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
| lfm2.5-24b-a2b-q4_0-cuda-default-ar | float16 | 512 | 1 | True | 1 | 1.281x | 15.8 | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-24b-a2b | q4_0 | cuda | default | ar | [qualified; complete](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-default-ar) | 579.5 | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |  |
| lfm2.5-24b-a2b | q4_0 | webgpu | default | ar | [interrupted](AKAS.md#lfm2.5-24b-a2b-q4_0-webgpu-default-ar) | 595.3 | — | TimeoutError: |

## Candidate outcomes

12 attempts: 5 failed, 6 passed, 1 running.

| Rejection or deadline | Attempts |
| --- | ---: |
| ValueError: model logits differ from qualified baseline: b1-p512 | 3 |
| ValueError: greedy output differs from baseline for B1/P32768 | 1 |
| TimeoutError: | 1 |
|  | 1 |
