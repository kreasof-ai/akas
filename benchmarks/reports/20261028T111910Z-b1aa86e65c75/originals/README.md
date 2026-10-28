# Campaign benchmark report

UTC: 2026-10-28T11:19:10.988203Z → 2026-10-28T11:21:37.402594Z. Status: `completed`. Active time: 0.04 hours; per-configuration ceiling: not set.

Source: `c8e7ac8dc1c771c4a9918a76d8d022676e4d3bc1`; measured source tree: `9897ec7b1e3d3c41062089af4c53aa9fa97c8a013cf925e1ad2e90b9700d2aa6`.

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
| AMD Radeon RX 6700 XT |  | akas / lfm3-1.7b / q4_0 / ar | 0.1.0 | `c8e7ac8dc1c771c4a9918a76d8d022676e4d3bc1` | [20261028112011535043-63e08789](../../runs/20261028112011535043-63e08789/run.json) |
| AMD Radeon RX 6700 XT |  | llama / lfm3-1.7b / q4_0 / ar | version: 0.7.0-dev (build 11946, commit 8ca4060b7) built with Clang 20.1.8 for Windows x86_64 | `c8e7ac8dc1c771c4a9918a76d8d022676e4d3bc1` | [20261028111910988203-333faeb6](../../runs/20261028111910988203-333faeb6/run.json) |

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lfm3-1.7b-q4_0-webgpu-float16-ar | float16 | 512 | 1 | True | 1 | — | — | [20261028112011535043-63e08789](../../runs/20261028112011535043-63e08789/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm3-1.7b | q4_0 | webgpu | float16 | ar | [qualified; passed](AKAS.md#lfm3-1.7b-q4_0-webgpu-float16-ar) | 1.2 | [20261028112011535043-63e08789](../../runs/20261028112011535043-63e08789/run.json) |  |

## Candidate outcomes

0 attempts: .

| Rejection or deadline | Attempts |
| --- | ---: |
