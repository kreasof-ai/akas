# Campaign benchmark report

UTC: 2026-11-01T07:40:55.313788Z → 2026-11-01T07:45:29.839441Z. Status: `completed`. Active time: 0.00 hours; per-configuration ceiling: not set.

Source: `ce32f41e6d5cb8ff9dbb46b2da025c999acc1fa0`; measured source tree: `329923bd9d2dd18e0f0854a5402177ee24da63b3094c4f25885fb1b2425144b8`.

Qualified AKAS deployments: **1/1**. Qualified baseline profiles: **2/2**.

- [Side-by-side tables for every requested shape](COMPARISON.md)
- [All AKAS configurations and per-shape measurements](AKAS.md)
- [All baseline profiles and per-shape measurements](BASELINES.md)
- [Every candidate attempt](CANDIDATES.md)
- [Full measurement CSV](measurements.csv) · [Candidate measurement CSV](candidate-measurements.csv) · [Candidate outcomes CSV](candidates.csv)
- [Selected kernel implementations and schedules](SCHEDULES.json) · [Portable report data](report.json) · [Campaign snapshot](campaign.json)

Qualifying AKAS measurements verified against vLLM 0.31.0, SGLang 0.5.21, and FP64/precision-matched oracles in [20261101113717922004-dcb7262b](../runs/20261101113717922004-dcb7262b/run.json) (16/16 three-way greedy agreement on BOS and chat probes; matching FP8 Tensor Core logit noise floor across all three runtimes). All values are medians. client-stream-burst-v1 measures concurrent native-engine token delivery: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. Decode counts the remaining output tokens after the first.

## Hardware and measurement provenance

| Hardware | Driver | Baseline/runtime | Version or revision | Source | Evidence |
| --- | --- | --- | --- | --- | --- |
| NVIDIA L40S |  | akas / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.1.0 | `ce32f41e6d5cb8ff9dbb46b2da025c999acc1fa0` | [20261101074055313788-b49b56d1](https://github.com/kreasof-ai/akas/blob/3c6a4bc78bb8bbc8b01bcb15b2403d2219d68e39/benchmarks/runs/20261101074055313788-b49b56d1/run.json) |
| NVIDIA L40S | 595.91.07 | vllm / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.31.0 | `ce32f41e6d5cb8ff9dbb46b2da025c999acc1fa0` | [20261031183705001992-d62597ef](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031183705001992-d62597ef/run.json) |
| NVIDIA L40S | 595.91.07 | sglang / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.5.21 | `ce32f41e6d5cb8ff9dbb46b2da025c999acc1fa0` | [20261031182740133509-0f8da2dc](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031182740133509-0f8da2dc/run.json) |

Captured host: 4 logical CPUs; Linux-6.12.110-135.202.amzn2023.x86_64-x86_64-with-glibc2.39.

```text
index, uuid, name, driver_version, memory.total [MiB], power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
0, GPU-27c9bcec-0258-ab6d-796a-7682a541edf9, NVIDIA L40S, 595.91.07, 46068 MiB, 350.00 W, 2520 MHz, 9001 MHz
```

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar | float16 | 2048 | 256 | True | 0 | — | — | [20261101074055313788-b49b56d1](https://github.com/kreasof-ai/akas/blob/3c6a4bc78bb8bbc8b01bcb15b2403d2219d68e39/benchmarks/runs/20261101074055313788-b49b56d1/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-24b-a2b-fp8 | fp8 | cuda | float16 | ar | [qualified; passed](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | 0.0 | [20261101074055313788-b49b56d1](https://github.com/kreasof-ai/akas/blob/3c6a4bc78bb8bbc8b01bcb15b2403d2219d68e39/benchmarks/runs/20261101074055313788-b49b56d1/run.json) |  |

## Candidate outcomes

1 attempts: 1 passed.

| Rejection or deadline | Attempts |
| --- | ---: |
