# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-24b-a2b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 570.2 | 764.8 | 431.6 |
| 1 | 512 | 2,837.7 | 3,572.6 | 2,713.6 |
| 1 | 2048 | 4,219.3 | 3,510.0 | 2,695.9 |
| 1 | 8192 | 4,344.2 | 3,455.1 | 2,571.0 |
| 1 | 16384 | 4,188.3 | 3,323.3 | 2,378.3 |
| 1 | 32768 | 3,863.9 | 3,080.1 | 2,095.0 |
| 1 | 65536 | 3,414.8 | 2,591.0 | 1,678.5 |
| 1 | 130944 | 2,807.3 | 1,623.0 | 1,243.2 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 130.5 | 135.0 | 116.9 |
| 1 | 512 | 130.2 | 133.6 | 116.3 |
| 1 | 2048 | 128.1 | 131.2 | 114.6 |
| 1 | 8192 | 120.7 | 121.7 | 108.4 |
| 1 | 16384 | 112.0 | 111.1 | 100.7 |
| 1 | 32768 | 97.9 | 95.3 | 88.2 |
| 1 | 65536 | 77.9 | 74.2 | 71.0 |
| 1 | 130944 | 56.1 | 51.2 | 51.7 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / HTTP](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 119.7 | 130.1 | 110.3 |
| 1 | 512 | 107.1 | 116.8 | 99.8 |
| 1 | 2048 | 83.8 | 82.4 | 68.4 |
| 1 | 8192 | 42.2 | 37.4 | 29.3 |
| 1 | 16384 | 24.4 | 21.0 | 15.7 |
| 1 | 32768 | 12.8 | 10.7 | 7.5 |
| 1 | 65536 | 6.1 | 4.7 | 3.1 |
| 1 | 130944 | 2.6 | 1.5 | 1.2 |
