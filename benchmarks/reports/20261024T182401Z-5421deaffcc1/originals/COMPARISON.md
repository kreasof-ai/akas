# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-2.6b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / native-token-step-v2](AKAS.md#lfm2.5-2.6b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 332.1 | 921.0 |
| 1 | 512 | 752.0 | 2,596.5 |
| 1 | 2048 | 737.0 | 2,470.2 |
| 1 | 8192 | 648.7 | 1,997.9 |
| 1 | 16384 | 566.9 | 1,556.3 |
| 1 | 32768 | 445.1 | 1,068.5 |
| 1 | 65536 | 276.7 | 655.5 |
| 1 | 130944 | 154.2 | 367.6 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / native-token-step-v2](AKAS.md#lfm2.5-2.6b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 135.6 | 165.4 |
| 1 | 512 | 132.8 | 161.5 |
| 1 | 2048 | 126.5 | 160.8 |
| 1 | 8192 | 106.0 | 151.5 |
| 1 | 16384 | 87.8 | 144.0 |
| 1 | 32768 | 65.1 | 127.9 |
| 1 | 65536 | 42.4 | 105.2 |
| 1 | 130944 | 25.1 | 78.2 |

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

Reference: [20261024180036587579-b265e1b0](../../runs/20261024180036587579-b265e1b0/run.json); selected pilot: [20261024181805114628-8438437d](../../runs/20261024181805114628-8438437d/run.json). Chunks: 32 → 512. Each uses 1 warmup and 3 retained repetitions. These calibration prefixes use identical weights, inputs, cache precision and native timing protocol. The speedup below compares native prefill within AKAS; it does not compare runtimes.

| Prefix | Before prefill | After prefill | Prefill speedup | Before decode | After decode | Exact output IDs |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 208.6 | 333.7 | 1.600x | 140.6 | 134.7 | yes |
| 512 | 208.8 | 756.0 | 3.621x | 138.6 | 131.5 | yes |
| 2048 | 202.6 | 735.9 | 3.631x | 131.8 | 126.6 | yes |

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
