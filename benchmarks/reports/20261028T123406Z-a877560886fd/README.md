# Campaign benchmark report

UTC: 2026-10-28T12:34:06.615870Z → 2026-10-28T14:40:57.451881Z. Status: `completed`. Active time: 0.83 hours; per-configuration ceiling: not set.

Source: `c8e7ac8dc1c771c4a9918a76d8d022676e4d3bc1`; measured source tree: `6b553a96ffcab91106b65ae3bb1aaa58c1b533de3d578bac79178290f0af0fda`.

Qualified AKAS deployments: **1/1**. Qualified baseline profiles: **1/1**.

- [Side-by-side tables for every requested shape](COMPARISON.md)
- [All AKAS configurations and per-shape measurements](AKAS.md)
- [All baseline profiles and per-shape measurements](BASELINES.md)
- [Every candidate attempt](CANDIDATES.md)
- [Full measurement CSV](measurements.csv) · [Candidate measurement CSV](candidate-measurements.csv) · [Candidate outcomes CSV](candidates.csv)
- [Selected kernel implementations and schedules](SCHEDULES.json) · [Portable report data](report.json) · [Campaign snapshot](campaign.json)

Decode median criterion: **passed**. [Per-prefix acceptance results](PARITY.json).

Decode rates are medians of five samples after two warmups. One request generates 128 outputs; 127 decode steps are timed. Both timers cover completed wall decode including greedy sampling and control dispatch; model load, real prefill, snapshot restore and validation are excluded. AKAS uses GPU token feedback; llama.cpp b11946 Vulkan C API uses its stock greedy sampler. Both restore complete real-prefill recurrent/KV state and the initial prediction. Prompts use fixed synthetic token IDs, never fabricated KV. Prefill observations are single samples in the portable data, not repeated benchmarks, and do not gate decode parity. A dash means no measured median. The maximum-prefix AKAS state is imported from sealed real-prefill run 20261028130402880342-2c41aef6 with checked hashes and unchanged prefill arithmetic/storage. Short prefixes through 8,192 select the 32-partition program; longer ones select the shared-KV 512-partition program. Both reserve a 131,072-token context. Generated sequences differ at some prefixes; performance parity does not establish identical text or end-to-end request latency.

## Hardware and measurement provenance

| Hardware | Driver | Baseline/runtime | Version or revision | Source | Evidence |
| --- | --- | --- | --- | --- | --- |
| AMD Radeon RX 6700 XT |  | akas / lfm3-1.7b / q4_0 / ar | 0.1.0 | `c8e7ac8dc1c771c4a9918a76d8d022676e4d3bc1` | [20261028142101939494-891196a6](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028142101939494-891196a6/run.json) |
| AMD Radeon RX 6700 XT |  | llama / lfm3-1.7b / q4_0 / ar | 8ca4060b7e30e76baaa5670ee1ec1b760b7f8552 | `c8e7ac8dc1c771c4a9918a76d8d022676e4d3bc1` | [20261028123406615870-990f1f0a](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028123406615870-990f1f0a/run.json) |

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lfm3-1.7b-q4_0-webgpu-float16-ar | float16 | 512 | 1 | True | 1 | — | — | [20261028142101939494-891196a6](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028142101939494-891196a6/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm3-1.7b | q4_0 | webgpu | float16 | ar | [qualified; passed](AKAS.md#lfm3-1.7b-q4_0-webgpu-float16-ar) | 9.0 | [20261028142101939494-891196a6](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028142101939494-891196a6/run.json) |  |

## Candidate outcomes

0 attempts: .

| Rejection or deadline | Attempts |
| --- | ---: |
