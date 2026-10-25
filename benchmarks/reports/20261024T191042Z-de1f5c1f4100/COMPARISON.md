# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-2.6b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / native-token-step-v2](AKAS.md#lfm2.5-2.6b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 323.2 | 921.0 |
| 1 | 512 | 816.3 | 2,596.5 |
| 1 | 2048 | 792.8 | 2,470.2 |
| 1 | 8192 | 693.9 | 1,997.9 |
| 1 | 16384 | 594.3 | 1,556.3 |
| 1 | 32768 | 465.3 | 1,068.5 |
| 1 | 65536 | 287.9 | 655.5 |
| 1 | 130944 | 168.5 | 367.6 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / native-token-step-v2](AKAS.md#lfm2.5-2.6b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 166.1 | 165.4 |
| 1 | 512 | 164.8 | 161.5 |
| 1 | 2048 | 158.9 | 160.8 |
| 1 | 8192 | 143.4 | 151.5 |
| 1 | 16384 | 124.5 | 144.0 |
| 1 | 32768 | 98.1 | 127.9 |
| 1 | 65536 | 68.7 | 105.2 |
| 1 | 130944 | 43.1 | 78.2 |

### Complete-burst output tok/s

| B | Prefix | [akas webgpu float16 / HTTP](AKAS.md#lfm2.5-2.6b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / HTTP](BASELINES.md#baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | — | 159.2 |
| 1 | 512 | — | 129.9 |
| 1 | 2048 | — | 78.9 |
| 1 | 8192 | — | 25.9 |
| 1 | 16384 | — | 11.2 |
| 1 | 32768 | — | 4.0 |
| 1 | 65536 | — | 1.3 |
| 1 | 130944 | — | 0.4 |

## Native prefill tuning pilots

Reference: [20261024181805114628-8438437d](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024181805114628-8438437d/run.json); selected pilot: [20261024215128041404-5569d5c7](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215128041404-5569d5c7/run.json). Chunks: 512 → 512. Each uses 1 warmup and 3 retained repetitions. These calibration prefixes use identical weights, inputs, cache precision and native timing protocol. The speedup below compares native prefill within AKAS; it does not compare runtimes.

| Prefix | Before prefill | After prefill | Prefill speedup | Before decode | After decode | Exact output IDs |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 333.7 | 333.4 | 0.999x | 134.7 | 166.9 | yes |
| 512 | 756.0 | 820.8 | 1.086x | 131.5 | 165.0 | yes |
| 2048 | 735.9 | 794.0 | 1.079x | 126.6 | 158.6 | yes |

## Tuning evidence

[20261024215308077571-967e0923](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215308077571-967e0923/run.json): Artifact collection for offline kernel searches, exact rounding/layout tests, archived Tensor schedule replay, synthetic decode screening, and intermediate real-text pilots. Kernel hot-buffer/reset timings are diagnostic, not additive model throughput. Rejected numerical shortcuts and slower layouts/policies are retained. The selected full program is independently qualified again and measured on retained C4 token IDs.

## Previous AKAS sweep

Previous: [20261024182401411794-847a6f68](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024182401411794-847a6f68/run.json); selected: [20261024215325218226-fd38f777](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215325218226-fd38f777/run.json). Weights, input IDs, cache precision, shape, repetition counts and native timing protocol match. These ratios compare AKAS with its own previous sweep, not with another runtime.

| Prefix | Before prefill | After prefill | Prefill ratio | Before decode | After decode | Decode ratio | Exact output IDs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 332.1 | 323.2 | 0.973x | 135.6 | 166.1 | 1.225x | yes |
| 512 | 752.0 | 816.3 | 1.086x | 132.8 | 164.8 | 1.241x | yes |
| 2048 | 737.0 | 792.8 | 1.076x | 126.5 | 158.9 | 1.256x | yes |
| 8192 | 648.7 | 693.9 | 1.070x | 106.0 | 143.4 | 1.353x | yes |
| 16384 | 566.9 | 594.3 | 1.048x | 87.8 | 124.5 | 1.417x | yes |
| 32768 | 445.1 | 465.3 | 1.046x | 65.1 | 98.1 | 1.508x | yes |
| 65536 | 276.7 | 287.9 | 1.041x | 42.4 | 68.7 | 1.619x | yes |
| 130944 | 154.2 | 168.5 | 1.093x | 25.1 | 43.1 | 1.720x | yes |

## Greedy output agreement

The untimed AKAS greedy pass is compared with the first retained llama.cpp response. Both use the same input IDs, checkpoint and cache precision. Different arithmetic can change the trajectory; agreement is not a corpus accuracy score.

| Prefix | Output tokens | Exact IDs | Common prefix tokens |
| ---: | ---: | --- | ---: |
| 32 | 128 | no | 16 |
| 512 | 128 | no | 6 |
| 2048 | 128 | yes | 128 |
| 8192 | 128 | yes | 128 |
| 16384 | 128 | no | 52 |
| 32768 | 128 | no | 31 |
| 65536 | 128 | no | 9 |
| 130944 | 128 | no | 0 |
