# Campaign benchmark report

UTC: 2026-10-23T10:16:15.391936Z → 2026-10-23T10:57:26.923521Z. Status: `complete`. Active time: 0.69 hours; per-configuration ceiling: not set.

Source: `b56abb6da21d77590a098b139a25c7a1072c29cb`; measured source tree: `4fb38b7d46e7375f904ab5b5985c8953dd2838f238a4e1e572e68a001b738211`.

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
| NVIDIA L4 |  | akas / lfm2.5-24b-a2b / q4_0 / ar | 0.1.0 | `b56abb6da21d77590a098b139a25c7a1072c29cb` | [20261023101615391936-42d4989e](https://github.com/akbar2habibullah/akas/blob/2b50f4ee396c7acd907a168dbacdcf1742c0dd54/benchmarks/runs/20261023101615391936-42d4989e/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-24b-a2b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `1102164555deb2f734eedaf31ac07502b62fc479` | [20261023080534992856-19529564](https://github.com/akbar2habibullah/akas/blob/2b50f4ee396c7acd907a168dbacdcf1742c0dd54/benchmarks/runs/20261023080534992856-19529564/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-24b-a2b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `1102164555deb2f734eedaf31ac07502b62fc479` | [20261023084715832350-d710881c](https://github.com/akbar2habibullah/akas/blob/2b50f4ee396c7acd907a168dbacdcf1742c0dd54/benchmarks/runs/20261023084715832350-d710881c/run.json) |

Captured host: 4 logical CPUs; Linux-6.12.110-135.202.amzn2023.x86_64-x86_64-with-glibc2.39.

```text
index, uuid, name, driver_version, memory.total [MiB], power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
0, GPU-931f7610-9dd1-d8c2-b5a9-f7f0c04c3f46, NVIDIA L4, 595.91.07, 23034 MiB, 72.00 W, 2040 MHz, 6251 MHz
```

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lfm2.5-24b-a2b-q4_0-cuda-float16-ar | float16 | 4096 | 1 | True | 1 | 1.000x | — | [20261023101615391936-42d4989e](https://github.com/akbar2habibullah/akas/blob/2b50f4ee396c7acd907a168dbacdcf1742c0dd54/benchmarks/runs/20261023101615391936-42d4989e/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-24b-a2b | q4_0 | cuda | float16 | ar | [qualified; complete](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-ar) | 41.3 | [20261023101615391936-42d4989e](https://github.com/akbar2habibullah/akas/blob/2b50f4ee396c7acd907a168dbacdcf1742c0dd54/benchmarks/runs/20261023101615391936-42d4989e/run.json) |  |

## Candidate outcomes

1 attempts: 1 passed.

| Rejection or deadline | Attempts |
| --- | ---: |

## Cache precision and generated outputs

Reference: `float16`; candidate: `fp8`. 0/10 greedy outputs match exactly.
Across 320 shared-history positions, top-1 agreement is **97.19%** and mean KL divergence is **0.002866**.

Six natural chat prompts and four exact-length prose-padded retrieval prompts; greedy output agreement and F16-common-history distribution drift. A controlled numerical probe, not a corpus accuracy evaluation.

Common prefix counts matching tokens before the first changed token. Shared-history top-1 and KL compare both caches after supplying the same F16-generated continuation. Later free-generation differences can compound after an earlier changed token; they are not an accuracy score.

Reference evidence: [20261023081755421996-a2391492](https://github.com/akbar2habibullah/akas/blob/2b50f4ee396c7acd907a168dbacdcf1742c0dd54/benchmarks/runs/20261023081755421996-a2391492/run.json); candidate evidence: [20261023082107625146-9f634fc3](https://github.com/akbar2habibullah/akas/blob/2b50f4ee396c7acd907a168dbacdcf1742c0dd54/benchmarks/runs/20261023082107625146-9f634fc3/run.json). Both generated texts and token IDs are retained in report.json; prompt IDs and per-step metrics remain in the checksummed run artifacts.

| Prompt | Prefix | F16 outputs | Candidate outputs | Exact | Common prefix | Shared-history steps | Top-1 agreement | Mean KL |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| garden | 37 | 128 | 128 | no | 0 | 32 | 96.9% | 0.000779 |
| bridges | 37 | 128 | 128 | no | 49 | 32 | 100.0% | 0.000869 |
| catalogue | 41 | 128 | 128 | no | 52 | 32 | 100.0% | 0.000386 |
| arithmetic | 39 | 128 | 128 | no | 31 | 32 | 96.9% | 0.000787 |
| sorting | 31 | 128 | 128 | no | 0 | 32 | 96.9% | 0.001445 |
| summary | 56 | 128 | 128 | no | 0 | 32 | 96.9% | 0.003285 |
| retrieval-8192-early | 8192 | 128 | 63 | no | 26 | 32 | 96.9% | 0.007032 |
| retrieval-8192-middle | 8192 | 128 | 128 | no | 12 | 32 | 93.8% | 0.002325 |
| retrieval-130944-early | 130944 | 128 | 128 | no | 74 | 32 | 100.0% | 0.007907 |
| retrieval-130944-middle | 130944 | 128 | 128 | no | 12 | 32 | 93.8% | 0.003842 |
