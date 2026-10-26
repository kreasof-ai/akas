# Campaign benchmark report

UTC: 2026-10-26T10:18:11.969664Z → 2026-10-26T11:33:41.508605Z. Status: `completed`. Active time: 0.46 hours; per-configuration ceiling: not set.

Source: `1d66b718581b893b57f72552b4d17b3f50e3879c`; measured source tree: `51a055c4c2c20d2ba8d4d3f6ef4bb824731fc4f470f48ea7a8260e10375da28b`.

Qualified AKAS deployments: **1/1**. Qualified baseline profiles: **1/1**.

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
| NVIDIA L4 |  | akas / spark-x2.6-4b / q4_k_m / ar | 0.1.0 | `1d66b718581b893b57f72552b4d17b3f50e3879c` | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |
| NVIDIA L4 |  | llama / spark-x2.6-4b / q4_k_m / ar | version: 0.7.0-dev (build 1, commit 620d222) built with GNU 13.3.0 for Linux x86_64 | `4833136a2ccc74c50f1dd1d999dfbcc00c78ca28` | [20261026101811969664-715fa9e0](https://github.com/kreasof-ai/akas/blob/566e756f90540e01b03628558e33444c94c18a40/benchmarks/runs/20261026101811969664-715fa9e0/run.json) |

Captured host: 4 logical CPUs; Linux-6.12.110-135.202.amzn2023.x86_64-x86_64-with-glibc2.39.

```text
index, uuid, name, driver_version, memory.total [MiB], power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
0, GPU-099ae6bf-5418-1167-069c-1891feaeb255, NVIDIA L4, 595.91.07, 23034 MiB, 72.00 W, 2040 MHz, 6251 MHz
```

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| spark-x2.6-4b-q4_k_m-cuda-float16-ar | float16 | 1024 | 1 | True | 1 | — | — | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| spark-x2.6-4b | q4_k_m | cuda | float16 | ar | [qualified; passed](AKAS.md#spark-x2.6-4b-q4_k_m-cuda-float16-ar) | 16.5 | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |  |

## Candidate outcomes

0 attempts: .

| Rejection or deadline | Attempts |
| --- | ---: |
