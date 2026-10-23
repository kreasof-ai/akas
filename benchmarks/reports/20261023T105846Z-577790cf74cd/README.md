# Campaign benchmark report

UTC: 2026-10-23T10:58:46.475312Z → 2026-10-23T11:23:53.422329Z. Status: `complete`. Active time: 0.00 hours; per-configuration ceiling: not set.

Source: `1b2efafa15c4a028e6cdc844ee5bd6a34909b192`; measured source tree: `212db47ee21450c913acf84d3ce25e8d9d233b99f4ff8a4bb9d34f6efac1db08`.

Qualified AKAS deployments: **1/1**. Qualified baseline profiles: **0/2**.

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
| NVIDIA L4 |  | akas / lfm2.5-24b-a2b / q4_0 / dspark | 0.1.0 | `1b2efafa15c4a028e6cdc844ee5bd6a34909b192` | [20261023105846475312-580635c0](../../runs/20261023105846475312-580635c0/run.json) |
| {} |  | llama / lfm2.5-24b-a2b / q4_0 / ar |  | `unknown` | — |
| {} |  | llama / lfm2.5-24b-a2b / q4_0 / ar |  | `unknown` | — |

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
| lfm2.5-24b-a2b-q4_0-cuda-float16-dspark | float16 | 4096 | 1 | True | 1 | 1.000x | — | [20261023105846475312-580635c0](../../runs/20261023105846475312-580635c0/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-24b-a2b | q4_0 | cuda | float16 | dspark | [qualified; complete](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-dspark) | 0.0 | [20261023105846475312-580635c0](../../runs/20261023105846475312-580635c0/run.json) |  |

## Candidate outcomes

1 attempts: 1 passed.

| Rejection or deadline | Attempts |
| --- | ---: |

## Real-text speculative decoding

Matched real prompts distinguish draft acceptance from synthetic shape tests. Calibration and held-out examples are separate. Long cases with one sample are diagnostic measurements. All rates are tokens/second for one request generating 128 tokens with EOS disabled.

AR uses device token feedback. Speculative decode includes proposal, verification, commit, repair and coordinator readbacks. Warmup and repetition counts are recorded per run; single-sample sweeps are diagnostic. Dataset revisions, example IDs, prompt hashes, crop recipes, full methodology and both outputs remain in report.json and the checksummed workload artifacts.

### calibration; confidence 0

Target/draft KV: `float16` / `float16`. Evidence: [20261023115449810427-2d566c0f](../../runs/20261023115449810427-2d566c0f/run.json); AR reference: [20261023115449810427-2d566c0f](../../runs/20261023115449810427-2d566c0f/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mtbench:82:57 | writing | 57 | 435.22 | 445.17 | 131.61 | 99.53 | 0.756x | 40.7% | 4.54 | yes | 128 |
| mtbench:104:29 | reasoning | 29 | 530.86 | 551.74 | 131.44 | 88.53 | 0.674x | 35.6% | 4.10 | no | 107 |
| mtbench:121:36 | coding | 36 | 299.25 | 305.17 | 131.39 | 65.81 | 0.501x | 25.2% | 3.10 | no | 52 |
| gsm8k-train:0:48 | math | 48 | 384.14 | 385.09 | 131.19 | 156.04 | 1.189x | 75.3% | 7.47 | yes | 128 |
| gsm8k-train:1:43 | math | 43 | 352.74 | 352.60 | 131.00 | 96.44 | 0.736x | 40.7% | 4.54 | yes | 128 |
| longbench-v2:66f94f9ebb02136c067c4fde:8192 | long-document | 8192 | 3422.23 | 3994.39 | 120.86 | 81.63 | 0.675x | 36.0% | 3.85 | yes | 128 |
| longbench-v2:66f94f9ebb02136c067c4fde:130944 | long-document | 130944 | 2820.62 | 2682.64 | 56.02 | 14.80 | 0.264x | 3.4% | 1.30 | yes | 128 |

### calibration; confidence 0.5

Target/draft KV: `float16` / `float16`. Evidence: [20261023120252406846-154985ec](../../runs/20261023120252406846-154985ec/run.json); AR reference: [20261023115449810427-2d566c0f](../../runs/20261023115449810427-2d566c0f/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mtbench:82:57 | writing | 57 | 435.22 | 438.57 | 131.61 | 107.07 | 0.814x | 61.4% | 4.23 | yes | 128 |
| mtbench:104:29 | reasoning | 29 | 530.86 | 544.69 | 131.44 | 93.74 | 0.713x | 50.5% | 3.85 | no | 107 |
| mtbench:121:36 | coding | 36 | 299.25 | 300.46 | 131.39 | 79.30 | 0.604x | 53.5% | 2.89 | no | 52 |
| gsm8k-train:0:48 | math | 48 | 384.14 | 383.84 | 131.19 | 156.18 | 1.190x | 78.0% | 7.47 | yes | 128 |
| gsm8k-train:1:43 | math | 43 | 352.74 | 347.82 | 131.00 | 108.06 | 0.825x | 58.6% | 4.54 | yes | 128 |
| longbench-v2:66f94f9ebb02136c067c4fde:8192 | long-document | 8192 | 3422.23 | 3525.64 | 120.86 | 94.04 | 0.778x | 58.4% | 3.85 | yes | 128 |
| longbench-v2:66f94f9ebb02136c067c4fde:130944 | long-document | 130944 | 2820.62 | 2713.81 | 56.02 | 15.78 | 0.282x | 5.0% | 1.30 | yes | 128 |

### calibration; confidence 0.8

Target/draft KV: `float16` / `float16`. Evidence: [20261023120428755115-73bb510f](../../runs/20261023120428755115-73bb510f/run.json); AR reference: [20261023115449810427-2d566c0f](../../runs/20261023115449810427-2d566c0f/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mtbench:82:57 | writing | 57 | 435.22 | 433.46 | 131.61 | 75.34 | 0.572x | 89.7% | 2.23 | yes | 128 |
| mtbench:104:29 | reasoning | 29 | 530.86 | 539.34 | 131.44 | 94.38 | 0.718x | 96.6% | 3.02 | no | 107 |
| mtbench:121:36 | coding | 36 | 299.25 | 291.56 | 131.39 | 69.71 | 0.531x | 97.0% | 2.02 | no | 52 |
| gsm8k-train:0:48 | math | 48 | 384.14 | 369.69 | 131.19 | 137.98 | 1.052x | 91.2% | 5.52 | yes | 128 |
| gsm8k-train:1:43 | math | 43 | 352.74 | 345.36 | 131.00 | 88.93 | 0.679x | 92.1% | 2.82 | yes | 128 |
| longbench-v2:66f94f9ebb02136c067c4fde:8192 | long-document | 8192 | 3422.23 | 4074.71 | 120.86 | 83.97 | 0.695x | 100.0% | 2.76 | yes | 128 |
| longbench-v2:66f94f9ebb02136c067c4fde:130944 | long-document | 130944 | 2820.62 | 2693.21 | 56.02 | 17.12 | 0.306x | 26.7% | 1.10 | yes | 128 |

### heldout; confidence 0.5

Target/draft KV: `float16` / `float16`. Evidence: [20261023120702655988-e46a7e46](../../runs/20261023120702655988-e46a7e46/run.json); AR reference: [20261023120702655988-e46a7e46](../../runs/20261023120702655988-e46a7e46/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mtbench:81:33 | writing | 33 | 277.94 | 287.65 | 131.10 | 79.42 | 0.606x | 64.7% | 2.54 | yes | 128 |
| mtbench:101:48 | reasoning | 48 | 379.61 | 377.76 | 130.79 | 105.70 | 0.808x | 67.1% | 4.10 | yes | 128 |
| mtbench:122:24 | coding | 24 | 432.07 | 444.35 | 130.29 | 90.59 | 0.695x | 63.6% | 3.34 | yes | 128 |
| gsm8k-test:0:73 | math | 73 | 537.56 | 539.01 | 130.09 | 124.28 | 0.955x | 69.4% | 5.08 | yes | 128 |
| gsm8k-test:1:38 | math | 38 | 318.04 | 308.83 | 129.92 | 137.11 | 1.055x | 70.9% | 6.35 | yes | 128 |
| longbench-v2:66fcffd9bb02136c067c94c5:8192 | long-document | 8192 | 3214.22 | 3999.18 | 120.02 | 130.87 | 1.090x | 77.0% | 5.52 | yes | 128 |
| longbench-v2:66fcffd9bb02136c067c94c5:130944 | long-document | 130944 | 2784.57 | 2682.38 | 55.81 | 14.86 | 0.266x | 3.0% | 1.26 | no | 18 |

### calibration; confidence 0.5; Smaller verification windows and tuned projections

Target/draft KV: `float16` / `float16`. Evidence: [20261023124514712767-7ea28db7](../../runs/20261023124514712767-7ea28db7/run.json); AR reference: [20261023124514712767-7ea28db7](../../runs/20261023124514712767-7ea28db7/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mtbench:82:57 | writing | 57 | 436.83 | 440.92 | 131.68 | 119.39 | 0.907x | 61.4% | 4.23 | yes | 128 |
| mtbench:104:29 | reasoning | 29 | 529.61 | 538.35 | 131.09 | 103.76 | 0.792x | 50.5% | 3.85 | no | 107 |
| mtbench:121:36 | coding | 36 | 300.54 | 299.21 | 131.24 | 89.79 | 0.684x | 53.5% | 2.89 | no | 52 |
| gsm8k-train:0:48 | math | 48 | 381.78 | 382.28 | 131.08 | 169.66 | 1.294x | 78.0% | 7.47 | yes | 128 |
| gsm8k-train:1:43 | math | 43 | 350.87 | 350.15 | 130.93 | 119.23 | 0.911x | 58.6% | 4.54 | yes | 128 |

### heldout; confidence 0.5; Smaller verification windows and tuned projections

Target/draft KV: `float16` / `float16`. Evidence: [20261023124621285769-4582c3e0](../../runs/20261023124621285769-4582c3e0/run.json); AR reference: [20261023124621285769-4582c3e0](../../runs/20261023124621285769-4582c3e0/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mtbench:81:33 | writing | 33 | 274.56 | 280.53 | 130.57 | 93.50 | 0.716x | 64.7% | 2.54 | yes | 128 |
| mtbench:101:48 | reasoning | 48 | 376.93 | 381.24 | 130.64 | 117.19 | 0.897x | 67.1% | 4.10 | yes | 128 |
| mtbench:122:24 | coding | 24 | 432.76 | 434.40 | 130.29 | 101.72 | 0.781x | 63.6% | 3.34 | yes | 128 |
| gsm8k-test:0:73 | math | 73 | 537.70 | 531.05 | 130.16 | 138.05 | 1.061x | 69.4% | 5.08 | yes | 128 |
| gsm8k-test:1:38 | math | 38 | 314.80 | 312.39 | 129.94 | 148.92 | 1.146x | 70.9% | 6.35 | yes | 128 |

### calibration; confidence 0.5; Q8_0 draft with smaller verification windows and tuned projections

Draft weights: `q8_0`. Target/draft KV: `float16` / `float16`. Evidence: [20261023132226213745-5931c276](../../runs/20261023132226213745-5931c276/run.json); AR reference: [20261023132226213745-5931c276](../../runs/20261023132226213745-5931c276/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mtbench:82:57 | writing | 57 | 434.13 | 444.76 | 131.77 | 124.00 | 0.941x | 61.4% | 4.23 | yes | 128 |
| mtbench:104:29 | reasoning | 29 | 533.10 | 539.75 | 131.14 | 109.89 | 0.838x | 51.6% | 3.97 | no | 107 |
| mtbench:121:36 | coding | 36 | 299.06 | 298.51 | 131.31 | 93.28 | 0.710x | 53.9% | 2.89 | no | 52 |
| gsm8k-train:0:48 | math | 48 | 384.71 | 383.59 | 131.10 | 176.66 | 1.347x | 80.3% | 7.47 | yes | 128 |
| gsm8k-train:1:43 | math | 43 | 351.83 | 347.06 | 131.01 | 123.26 | 0.941x | 58.6% | 4.54 | yes | 128 |

### heldout; confidence 0.5; Q8_0 draft with smaller verification windows and tuned projections

Draft weights: `q8_0`. Target/draft KV: `float16` / `float16`. Evidence: [20261023132331975932-f38d7467](../../runs/20261023132331975932-f38d7467/run.json); AR reference: [20261023132331975932-f38d7467](../../runs/20261023132331975932-f38d7467/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mtbench:81:33 | writing | 33 | 279.16 | 280.97 | 130.94 | 98.50 | 0.752x | 64.7% | 2.54 | yes | 128 |
| mtbench:101:48 | reasoning | 48 | 380.88 | 380.02 | 130.70 | 123.53 | 0.945x | 67.8% | 4.23 | yes | 128 |
| mtbench:122:24 | coding | 24 | 432.55 | 437.13 | 130.28 | 105.93 | 0.813x | 63.6% | 3.34 | yes | 128 |
| gsm8k-test:0:73 | math | 73 | 541.70 | 527.84 | 130.17 | 137.88 | 1.059x | 66.4% | 4.88 | yes | 128 |
| gsm8k-test:1:38 | math | 38 | 317.29 | 310.41 | 129.93 | 152.61 | 1.175x | 70.9% | 6.35 | yes | 128 |

### calibration; confidence 0.5; Codec-owned compressed expert staging, Q8_0 draft

Draft weights: `q8_0`. Target/draft KV: `float16` / `float16`. Evidence: [20261023151432143989-67b63b5e](../../runs/20261023151432143989-67b63b5e/run.json); AR reference: [20261023151432143989-67b63b5e](../../runs/20261023151432143989-67b63b5e/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mtbench:82:57 | writing | 57 | 433.93 | 446.62 | 131.66 | 143.49 | 1.090x | 61.4% | 4.23 | yes | 128 |
| mtbench:104:29 | reasoning | 29 | 533.13 | 531.15 | 131.10 | 126.73 | 0.967x | 51.6% | 3.97 | no | 107 |
| mtbench:121:36 | coding | 36 | 294.94 | 300.13 | 131.25 | 107.30 | 0.818x | 53.9% | 2.89 | no | 52 |
| gsm8k-train:0:48 | math | 48 | 384.93 | 382.16 | 131.03 | 206.83 | 1.578x | 80.3% | 7.47 | yes | 128 |
| gsm8k-train:1:43 | math | 43 | 353.64 | 348.18 | 130.94 | 142.85 | 1.091x | 58.6% | 4.54 | yes | 128 |

### heldout; confidence 0.5; Codec-owned compressed expert staging, Q8_0 draft

Draft weights: `q8_0`. Target/draft KV: `float16` / `float16`. Evidence: [20261023151533293524-174061e4](../../runs/20261023151533293524-174061e4/run.json); AR reference: [20261023151533293524-174061e4](../../runs/20261023151533293524-174061e4/run.json).

| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mtbench:81:33 | writing | 33 | 281.74 | 278.64 | 130.95 | 110.96 | 0.847x | 64.7% | 2.54 | yes | 128 |
| mtbench:101:48 | reasoning | 48 | 381.01 | 386.44 | 130.79 | 142.57 | 1.090x | 67.8% | 4.23 | yes | 128 |
| mtbench:122:24 | coding | 24 | 435.47 | 436.07 | 130.35 | 121.60 | 0.933x | 63.6% | 3.34 | yes | 128 |
| gsm8k-test:0:73 | math | 73 | 538.28 | 532.07 | 130.32 | 159.76 | 1.226x | 66.4% | 4.88 | yes | 128 |
| gsm8k-test:1:38 | math | 38 | 317.63 | 308.36 | 130.17 | 178.48 | 1.371x | 70.9% | 6.35 | yes | 128 |

## Upstream speculative acceptance

These runs use the same pinned real-text token IDs and target checkpoint as the native tests. Acceptance is accepted draft tokens divided by proposed draft tokens, pooled over measured samples. It excludes the target bonus token and depends on confidence filtering. Prefill and decode rates use llama.cpp's reported server timing boundaries; cross-runtime speedups are not calculated. Different target arithmetic can change generated text and subsequent acceptance. Full outputs, per-sample counters, commands, checkpoint hashes and upstream source revision are retained in the linked evidence.

### llama.cpp; draft float16; confidence 0.5

Target/draft KV: `float16` / `float16`. Output tokens: 128; warmups: 1; repetitions: 2. Upstream revision: `620d2225cc436ab0b001e3200e590e3b286ef9ab`. Evidence: [20261023153020366805-120f2195](../../runs/20261023153020366805-120f2195/run.json).

| Example | Split | Category | Prefix | Accepted / proposed | Acceptance | Prefill tok/s | Decode tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| mtbench:81:33 | heldout | writing | 33 | 158 / 232 | 68.1% | 655.46 | 103.06 |
| mtbench:82:57 | calibration | writing | 57 | 202 / 300 | 67.3% | 909.07 | 151.24 |
| mtbench:101:48 | heldout | reasoning | 48 | 188 / 292 | 64.4% | 845.72 | 127.41 |
| mtbench:104:29 | calibration | reasoning | 29 | 192 / 382 | 50.3% | 659.43 | 119.48 |
| mtbench:121:36 | calibration | coding | 36 | 174 / 288 | 60.4% | 658.40 | 111.75 |
| mtbench:122:24 | heldout | coding | 24 | 192 / 274 | 70.1% | 544.63 | 132.61 |
| gsm8k-train:0:48 | calibration | math | 48 | 218 / 300 | 72.7% | 848.51 | 180.81 |
| gsm8k-train:1:43 | calibration | math | 43 | 200 / 328 | 61.0% | 761.36 | 139.33 |
| gsm8k-test:0:73 | heldout | math | 73 | 206 / 306 | 67.3% | 1005.51 | 157.38 |
| gsm8k-test:1:38 | heldout | math | 38 | 212 / 308 | 68.8% | 713.71 | 162.12 |

### llama.cpp; draft float16; confidence 0.0

Target/draft KV: `float16` / `float16`. Output tokens: 128; warmups: 1; repetitions: 2. Upstream revision: `620d2225cc436ab0b001e3200e590e3b286ef9ab`. Evidence: [20261023153519414727-f222e5b7](../../runs/20261023153519414727-f222e5b7/run.json).

| Example | Split | Category | Prefix | Accepted / proposed | Acceptance | Prefill tok/s | Decode tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| mtbench:81:33 | heldout | writing | 33 | 172 / 690 | 24.9% | 657.64 | 89.47 |
| mtbench:82:57 | calibration | writing | 57 | 196 / 514 | 38.1% | 892.51 | 118.55 |
| mtbench:101:48 | heldout | reasoning | 48 | 196 / 480 | 40.8% | 857.74 | 119.55 |
| mtbench:104:29 | calibration | reasoning | 29 | 192 / 540 | 35.6% | 668.32 | 111.90 |
| mtbench:121:36 | calibration | coding | 36 | 188 / 530 | 35.5% | 684.65 | 107.22 |
| mtbench:122:24 | heldout | coding | 24 | 200 / 480 | 41.7% | 558.37 | 122.67 |
| gsm8k-train:0:48 | calibration | math | 48 | 218 / 310 | 70.3% | 846.85 | 184.16 |
| gsm8k-train:1:43 | calibration | math | 43 | 200 / 468 | 42.7% | 781.70 | 125.78 |
| gsm8k-test:0:73 | heldout | math | 73 | 204 / 434 | 47.0% | 966.99 | 136.51 |
| gsm8k-test:1:38 | heldout | math | 38 | 214 / 356 | 60.1% | 722.16 | 165.56 |

### llama.cpp; draft q8_0; confidence 0.5

Target/draft KV: `float16` / `float16`. Output tokens: 128; warmups: 1; repetitions: 2. Upstream revision: `620d2225cc436ab0b001e3200e590e3b286ef9ab`. Evidence: [20261023154310543892-4e4811ca](../../runs/20261023154310543892-4e4811ca/run.json).

| Example | Split | Category | Prefix | Accepted / proposed | Acceptance | Prefill tok/s | Decode tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| mtbench:81:33 | heldout | writing | 33 | 156 / 230 | 67.8% | 660.77 | 123.29 |
| mtbench:82:57 | calibration | writing | 57 | 202 / 300 | 67.3% | 906.71 | 170.11 |
| mtbench:101:48 | heldout | reasoning | 48 | 192 / 286 | 67.1% | 856.71 | 151.12 |
| mtbench:104:29 | calibration | reasoning | 29 | 196 / 354 | 55.4% | 672.24 | 142.03 |
| mtbench:121:36 | calibration | coding | 36 | 174 / 294 | 59.2% | 677.85 | 126.31 |
| mtbench:122:24 | heldout | coding | 24 | 192 / 272 | 70.6% | 566.71 | 153.64 |
| gsm8k-train:0:48 | calibration | math | 48 | 218 / 296 | 73.6% | 856.41 | 198.29 |
| gsm8k-train:1:43 | calibration | math | 43 | 194 / 350 | 55.4% | 790.46 | 142.19 |
| gsm8k-test:0:73 | heldout | math | 73 | 206 / 308 | 66.9% | 1002.20 | 172.37 |
| gsm8k-test:1:38 | heldout | math | 38 | 214 / 302 | 70.9% | 730.46 | 183.21 |
