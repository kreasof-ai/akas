# Campaign benchmark report

UTC: 2026-11-03T05:22:43.038756Z → 2026-11-03T05:24:15.963103Z. Status: `existing measurement verified`. Active time: 0.00 hours; per-configuration ceiling: not set.

Source: `None`; measured source tree: `bc9cb1ffe6e2e3b6b80ac99b4e633eb3fb289103aa61a3c4d1cba4ef7b2c375f`.

Qualified AKAS measurements: **1/1**. Qualified baseline profiles: **1/2**.

- [Side-by-side tables for every requested shape](COMPARISON.md)
- [All AKAS configurations and per-shape measurements](AKAS.md)
- [All baseline profiles and per-shape measurements](BASELINES.md)
- [Every candidate attempt](CANDIDATES.md)
- [Full measurement CSV](measurements.csv) · [Candidate measurement CSV](candidate-measurements.csv) · [Candidate outcomes CSV](candidates.csv)
- [Selected kernel implementations and schedules](SCHEDULES.json) · [Portable report data](report.json) · [Campaign snapshot](campaign.json)

Retained measurement timings with later correctness verification of the same program and checkpoint. No deployment search or tuning was performed during verification. Phase protocols, timing samples and frozen baseline records are unchanged. AKAS uses the native engine token stream; the frozen vendor runs use HTTP/SSE. All throughput values are medians over three repetitions; decode counts the 127 outputs after the first token. Missing values remain —. † retains an incomplete or failed qualification.

## Hardware and measurement provenance

| Hardware | Driver | Baseline/runtime | Version or revision | Source | Evidence |
| --- | --- | --- | --- | --- | --- |
| NVIDIA H200 |  | akas / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.1.0 | `None` | [20261103052243038756-378e9a45](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103052243038756-378e9a45/run.json) |
| NVIDIA H200 | 610.57.04 | vllm / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.31.0 | `None` | [20261102190120772675-8245f839](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190120772675-8245f839/run.json) |
| NVIDIA H200 | 610.57.04 | sglang / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.5.21 | `None` | [20261102190540048720-a1e7c623](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190540048720-a1e7c623/run.json) |

Captured host: 20 logical CPUs; Linux-4.19.0-gvisor-x86_64-with-glibc2.39.

```text
index, uuid, name, driver_version, memory.total [MiB], power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
0, GPU-77dc2193-7817-61e2-a9fc-18369acd106d, NVIDIA H200, 610.57.04, 143771 MiB, 700.00 W, 1980 MHz, 3201 MHz
```

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar | float16 | 2048 | 16 | True | 0 | — | — | [20261103052243038756-378e9a45](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103052243038756-378e9a45/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-24b-a2b-fp8 | fp8 | cuda | float16 | ar | [qualified; verified measurement](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | 0.0 | [20261103052243038756-378e9a45](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103052243038756-378e9a45/run.json) |  |

## Candidate outcomes

1 attempts: 1 passed.

| Rejection or deadline | Attempts |
| --- | ---: |
