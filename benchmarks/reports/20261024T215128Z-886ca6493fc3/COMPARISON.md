# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-2.6b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / native-token-step-v2](AKAS.md#lfm2.5-2.6b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 333.4 | 906.0 |
| 1 | 512 | 820.8 | 2,599.1 |
| 1 | 2048 | 794.0 | 2,464.1 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / native-token-step-v2](AKAS.md#lfm2.5-2.6b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 166.9 | 165.2 |
| 1 | 512 | 165.0 | 163.0 |
| 1 | 2048 | 158.6 | 160.6 |

### Complete-burst output tok/s

| B | Prefix | [akas webgpu float16 / HTTP](AKAS.md#lfm2.5-2.6b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / HTTP](BASELINES.md#baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | — | 158.7 |
| 1 | 512 | — | 129.3 |
| 1 | 2048 | — | 78.8 |

## Native prefill tuning pilots

Reference: [20261024180036587579-b265e1b0](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261024180036587579-b265e1b0/run.json); selected pilot: [20261024181805114628-8438437d](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024181805114628-8438437d/run.json). Chunks: 32 → 512. Each uses 1 warmup and 3 retained repetitions. These calibration prefixes use identical weights, inputs, cache precision and native timing protocol. The speedup below compares native prefill within AKAS; it does not compare runtimes.

| Prefix | Before prefill | After prefill | Prefill speedup | Before decode | After decode | Exact output IDs |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 208.6 | 333.7 | 1.600x | 140.6 | 134.7 | yes |
| 512 | 208.8 | 756.0 | 3.621x | 138.6 | 131.5 | yes |
| 2048 | 202.6 | 735.9 | 3.631x | 131.8 | 126.6 | yes |

## Tuning evidence

[20261024215308077571-967e0923](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215308077571-967e0923/run.json): Artifact collection for offline kernel searches, exact rounding/layout tests, archived Tensor schedule replay, synthetic decode screening, and intermediate real-text pilots. Kernel hot-buffer/reset timings are diagnostic, not additive model throughput. Rejected numerical shortcuts and slower layouts/policies are retained. The selected full program is independently qualified again and measured on retained C4 token IDs.

## Previous AKAS sweep

Previous: [20261024181805114628-8438437d](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024181805114628-8438437d/run.json); selected: [20261024215128041404-5569d5c7](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215128041404-5569d5c7/run.json). Weights, input IDs, cache precision, shape, repetition counts and native timing protocol match. These ratios compare AKAS with its own previous sweep, not with another runtime.

| Prefix | Before prefill | After prefill | Prefill ratio | Before decode | After decode | Decode ratio | Exact output IDs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 333.7 | 333.4 | 0.999x | 134.7 | 166.9 | 1.239x | yes |
| 512 | 756.0 | 820.8 | 1.086x | 131.5 | 165.0 | 1.255x | yes |
| 2048 | 735.9 | 794.0 | 1.079x | 126.6 | 158.6 | 1.253x | yes |

## Greedy output agreement

The untimed AKAS greedy pass is compared with the first retained llama.cpp response. Both use the same input IDs, checkpoint and cache precision. Different arithmetic can change the trajectory; agreement is not a corpus accuracy score.

| Prefix | Output tokens | Exact IDs | Common prefix tokens |
| ---: | ---: | --- | ---: |
| 32 | 128 | no | 16 |
| 512 | 128 | no | 6 |
| 2048 | 128 | yes | 128 |
