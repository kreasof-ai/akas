# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-2.6b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / native-token-step-v2](AKAS.md#lfm2.5-2.6b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 458.0 | 921.0 |
| 1 | 512 | 974.8 | 2,596.5 |
| 1 | 2048 | 953.1 | 2,470.2 |
| 1 | 8192 | 835.6 | 1,997.9 |
| 1 | 16384 | 712.8 | 1,556.3 |
| 1 | 32768 | 556.0 | 1,068.5 |
| 1 | 65536 | 386.2 | 655.5 |
| 1 | 130944 | 239.5 | 367.6 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / native-token-step-v2](AKAS.md#lfm2.5-2.6b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 169.8 | 165.4 |
| 1 | 512 | 168.5 | 161.5 |
| 1 | 2048 | 163.3 | 160.8 |
| 1 | 8192 | 152.9 | 151.5 |
| 1 | 16384 | 139.8 | 144.0 |
| 1 | 32768 | 122.2 | 127.9 |
| 1 | 65536 | 97.1 | 105.2 |
| 1 | 130944 | 68.2 | 78.2 |

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

## Tuning evidence

[20261025121909558322-6c37463a](https://github.com/akbar2habibullah/akas/blob/2a5fff2394e5822373e18fdb9f072c9d204169e8/benchmarks/runs/20261025121909558322-6c37463a/run.json): Collection of qualified and rejected kernel trials, synthetic Gaussian-cache decoding screens, isolated hot-buffer timings, changed-arithmetic regression checks, and retained real-text pilots. This collection creates no new inference samples. Partial/interrupted searches and failures are included and are not claims of passing candidates. The selected program is separately independently qualified and benchmarked on retained C4 token IDs.

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
