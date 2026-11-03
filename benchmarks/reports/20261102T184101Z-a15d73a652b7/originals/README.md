# Campaign benchmark report

UTC: 2026-11-02T18:41:01.326680Z → 2026-11-02T18:42:31.631165Z. Status: `retained measurements`. Active time: 0.00 hours; per-configuration ceiling: not set.

Source: `None`; measured source tree: `e1c4b8c8cc37beb17f6c5af15b101a90a4d90b35a64c7b078e3670198d048969`.

Qualified AKAS deployments: **0/1**. Qualified baseline profiles: **2/2**.

- [Side-by-side tables for every requested shape](COMPARISON.md)
- [All AKAS configurations and per-shape measurements](AKAS.md)
- [All baseline profiles and per-shape measurements](BASELINES.md)
- [Every candidate attempt](CANDIDATES.md)
- [Full measurement CSV](measurements.csv) · [Candidate measurement CSV](candidate-measurements.csv) · [Candidate outcomes CSV](candidates.csv)
- [Selected kernel implementations and schedules](SCHEDULES.json) · [Portable report data](report.json) · [Campaign snapshot](campaign.json)

Retained measurements; importing evidence does not qualify an AKAS deployment. All values are medians. Phase protocol labels describe the measured boundaries. client-stream-burst-v1 measures concurrent native-engine token delivery: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. Decode counts the remaining output tokens after the first. Unqualified values retain †. Original run records and source snapshots are unchanged.

## Hardware and measurement provenance

| Hardware | Driver | Baseline/runtime | Version or revision | Source | Evidence |
| --- | --- | --- | --- | --- | --- |
| NVIDIA H200 | 610.57.04 | vllm / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.31.0 | `None` | [20261102182946501179-83424263](../../runs/20261102182946501179-83424263/run.json) |
| NVIDIA H200 | 610.57.04 | sglang / lfm2.5-24b-a2b-fp8 / fp8 / ar | 0.5.21 | `None` | [20261102183523780472-768136b6](../../runs/20261102183523780472-768136b6/run.json) |

Captured host: 20 logical CPUs; Linux-4.19.0-gvisor-x86_64-with-glibc2.39.

```text
index, uuid, name, driver_version, memory.total [MiB], power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
0, GPU-7b065812-57d8-5360-36c0-c2959832b688, NVIDIA H200, 610.57.04, 143771 MiB, 700.00 W, 1980 MHz, 3201 MHz
```

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-24b-a2b-fp8 | fp8 | cuda | float16 | ar | [unqualified](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | 0.0 | — | Retained measurements; no qualified deployment imported |

## Candidate outcomes

1 attempts: 1 passed.

| Rejection or deadline | Attempts |
| --- | ---: |
