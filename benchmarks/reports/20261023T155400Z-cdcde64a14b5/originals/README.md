# Campaign benchmark report

UTC: 2026-10-23T15:54:00.228961Z → 2026-10-23T16:35:30.784271Z. Status: `completed`. Active time: 0.69 hours; per-configuration ceiling: not set.

Source: `c135ff99c6f01d40f4222350f93f3dc92804acae`; measured source tree: `0ed746892b5bffe84394f15feab15770a50d3217089e748ef896886cd5d76e43`.

Qualified AKAS deployments: **4/4**. Qualified baseline profiles: **2/2**.

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
| NVIDIA L4 |  | akas / lfm2.5-24b-a2b / q4_0 / ar | 0.1.0 | `c135ff99c6f01d40f4222350f93f3dc92804acae` | [20261023155400228961-efa6319e](../../runs/20261023155400228961-efa6319e/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-24b-a2b / q4_0 / dspark | 0.1.0 | `c135ff99c6f01d40f4222350f93f3dc92804acae` | [20261023155400228961-efa6319e](../../runs/20261023155400228961-efa6319e/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-24b-a2b / q4_0 / ar | 0.1.0 | `c135ff99c6f01d40f4222350f93f3dc92804acae` | [20261023161702005532-82eb67d4](../../runs/20261023161702005532-82eb67d4/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-24b-a2b / q4_0 / dspark | 0.1.0 | `c135ff99c6f01d40f4222350f93f3dc92804acae` | [20261023161702005532-82eb67d4](../../runs/20261023161702005532-82eb67d4/run.json) |
| NVIDIA L4 |  | llama / lfm2.5-24b-a2b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `620d2225cc436ab0b001e3200e590e3b286ef9ab` | [20261023162435892403-5cca5399](../../runs/20261023162435892403-5cca5399/run.json) |
| NVIDIA L4 |  | llama / lfm2.5-24b-a2b / q4_0 / dspark | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `620d2225cc436ab0b001e3200e590e3b286ef9ab` | [20261023162947335687-fbeb0b96](../../runs/20261023162947335687-fbeb0b96/run.json) |

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
| lfm2.5-24b-a2b-q4_0-cuda-float16-ar | float16 | 4096 | 1 | True | 0 | — | — | [20261023155400228961-efa6319e](../../runs/20261023155400228961-efa6319e/run.json) |
| lfm2.5-24b-a2b-q4_0-cuda-float16-dspark | float16 | 4096 | 1 | True | 0 | — | — | [20261023155400228961-efa6319e](../../runs/20261023155400228961-efa6319e/run.json) |
| lfm2.5-24b-a2b-q4_0-cuda-fp8-ar | fp8 | 4096 | 1 | True | 0 | — | — | [20261023161702005532-82eb67d4](../../runs/20261023161702005532-82eb67d4/run.json) |
| lfm2.5-24b-a2b-q4_0-cuda-fp8-dspark | fp8 | 4096 | 1 | True | 0 | — | — | [20261023161702005532-82eb67d4](../../runs/20261023161702005532-82eb67d4/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-24b-a2b | q4_0 | cuda | float16 | ar | [qualified; passed](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-ar) | 0.0 | [20261023155400228961-efa6319e](../../runs/20261023155400228961-efa6319e/run.json) |  |
| lfm2.5-24b-a2b | q4_0 | cuda | float16 | dspark | [qualified; passed](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-dspark) | 0.0 | [20261023155400228961-efa6319e](../../runs/20261023155400228961-efa6319e/run.json) |  |
| lfm2.5-24b-a2b | q4_0 | cuda | fp8 | ar | [qualified; passed](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-fp8-ar) | 0.0 | [20261023161702005532-82eb67d4](../../runs/20261023161702005532-82eb67d4/run.json) |  |
| lfm2.5-24b-a2b | q4_0 | cuda | fp8 | dspark | [qualified; passed](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-fp8-dspark) | 0.0 | [20261023161702005532-82eb67d4](../../runs/20261023161702005532-82eb67d4/run.json) |  |

## Candidate outcomes

0 attempts: .

| Rejection or deadline | Attempts |
| --- | ---: |

## Real-text speculative decoding

Matched real prompts distinguish draft acceptance from synthetic shape tests. Calibration and held-out examples are separate. Long cases with one sample are diagnostic measurements. All rates are tokens/second for one request generating 128 tokens with EOS disabled.

AR uses device token feedback. Speculative decode includes proposal, verification, commit, repair and coordinator readbacks. Warmup and repetition counts are recorded per run; single-sample sweeps are diagnostic. Dataset revisions, example IDs, prompt hashes, crop recipes, full methodology and both outputs remain in report.json and the checksummed workload artifacts.

### heldout; confidence 0.5; Concatenated C4 quick context sweep

Draft weights: `q8_0`. Target/draft KV: `float16` / `float16`. Evidence: [20261023155400228961-efa6319e](../../runs/20261023155400228961-efa6319e/run.json); AR reference: [20261023155400228961-efa6319e](../../runs/20261023155400228961-efa6319e/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| c4-concat:32 | text-continuation | 32 | 581.16 | 500.43 | 131.80 | 64.94 | 0.493x | 62.5% | 1.46 | no | 35 |
| c4-concat:512 | text-continuation | 512 | 1090.25 | 2095.44 | 131.16 | 202.88 | 1.547x | 96.3% | 5.52 | yes | 128 |
| c4-concat:2048 | text-continuation | 2048 | 2631.19 | 3430.85 | 128.87 | 256.11 | 1.987x | 95.5% | 5.77 | yes | 128 |
| c4-concat:8192 | text-continuation | 8192 | 3407.76 | 4150.38 | 121.43 | 123.50 | 1.017x | 75.0% | 2.95 | no | 41 |
| c4-concat:16384 | text-continuation | 16384 | 4258.96 | 4092.03 | 112.74 | 95.47 | 0.847x | 81.2% | 2.19 | no | 37 |
| c4-concat:32768 | text-continuation | 32768 | 3956.57 | 3804.86 | 98.76 | 54.95 | 0.556x | 61.7% | 1.30 | yes | 128 |
| c4-concat:65536 | text-continuation | 65536 | 3465.21 | 3330.21 | 78.29 | 29.68 | 0.379x | 12.4% | 1.25 | no | 111 |
| c4-concat:130944 | text-continuation | 130944 | 2771.67 | 2715.70 | 55.74 | 16.83 | 0.302x | 3.6% | 1.22 | no | 58 |

### heldout; confidence 0.5; Concatenated C4 quick context sweep

Draft weights: `q8_0`. Target/draft KV: `fp8` / `float16`. Evidence: [20261023161702005532-82eb67d4](../../runs/20261023161702005532-82eb67d4/run.json); AR reference: [20261023161702005532-82eb67d4](../../runs/20261023161702005532-82eb67d4/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| c4-concat:32 | text-continuation | 32 | 588.17 | 510.06 | 131.22 | 66.55 | 0.507x | 58.8% | 1.46 | no | 4 |
| c4-concat:512 | text-continuation | 512 | 1106.19 | 2110.07 | 130.73 | 196.43 | 1.503x | 96.2% | 5.08 | no | 27 |
| c4-concat:2048 | text-continuation | 2048 | 2648.70 | 3409.43 | 128.98 | 190.99 | 1.481x | 80.2% | 4.88 | no | 65 |
| c4-concat:8192 | text-continuation | 8192 | 3406.76 | 4132.80 | 123.17 | 101.71 | 0.826x | 61.7% | 2.65 | no | 43 |
| c4-concat:16384 | text-continuation | 16384 | 4220.05 | 4052.23 | 118.62 | 108.70 | 0.916x | 81.7% | 3.34 | no | 34 |
| c4-concat:32768 | text-continuation | 32768 | 3934.41 | 3770.36 | 109.76 | 53.23 | 0.485x | 59.5% | 1.25 | no | 18 |
| c4-concat:65536 | text-continuation | 65536 | 3445.74 | 3309.36 | 95.04 | 29.41 | 0.309x | 25.6% | 1.74 | no | 33 |
| c4-concat:130944 | text-continuation | 130944 | 2755.72 | 2680.55 | 75.10 | 11.02 | 0.147x | 3.3% | 1.17 | no | 18 |

## Upstream speculative acceptance

These runs use the same pinned real-text token IDs and target checkpoint as the native tests. Acceptance is accepted draft tokens divided by proposed draft tokens, pooled over measured samples. It excludes the target bonus token and depends on confidence filtering. Prefill and decode rates use llama.cpp's reported server timing boundaries; cross-runtime speedups are not calculated. Different target arithmetic can change generated text and subsequent acceptance. Full outputs, per-sample counters, commands, checkpoint hashes and upstream source revision are retained in the linked evidence.

### llama; draft q8_0; confidence 0.5

Target/draft KV: `float16` / `float16`. Output tokens: 128; warmups: 0; repetitions: 1. Upstream revision: `620d2225cc436ab0b001e3200e590e3b286ef9ab`. Evidence: [20261023162947335687-fbeb0b96](../../runs/20261023162947335687-fbeb0b96/run.json).

| Example | Split | Category | Prefix | Accepted / proposed | Acceptance | Prefill tok/s | Decode tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| c4-concat:32 | heldout | text-continuation | 32 | 41 / 61 | 67.2% | 650.18 | 80.34 |
| c4-concat:512 | heldout | text-continuation | 512 | 58 / 100 | 58.0% | 1704.99 | 85.84 |
| c4-concat:2048 | heldout | text-continuation | 2048 | 103 / 111 | 92.8% | 2544.77 | 171.17 |
| c4-concat:8192 | heldout | text-continuation | 8192 | 81 / 101 | 80.2% | 2858.90 | 131.78 |
| c4-concat:16384 | heldout | text-continuation | 16384 | 90 / 103 | 87.4% | 2912.49 | 141.08 |
| c4-concat:32768 | heldout | text-continuation | 32768 | 30 / 52 | 57.7% | 2714.94 | 64.62 |
| c4-concat:65536 | heldout | text-continuation | 65536 | 27 / 222 | 12.2% | 2315.34 | 41.40 |
| c4-concat:130944 | heldout | text-continuation | 130944 | 33 / 577 | 5.7% | 1508.07 | 28.14 |
