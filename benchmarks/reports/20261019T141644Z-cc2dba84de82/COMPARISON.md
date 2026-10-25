# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-24b-a2b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-default-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-webgpu-default-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 428.7 | 113.2 † | 763.0 | 425.4 |
| 1 | 512 | 2,010.2 | 293.6 † | 3,619.0 | 2,706.4 |
| 1 | 2048 | 2,039.8 | 289.6 † | 3,530.5 | 2,702.6 |
| 1 | 8192 | 1,988.9 | 266.1 † | 3,490.7 | 2,578.1 |
| 1 | 16384 | 1,898.1 | 243.2 † | 3,339.3 | 2,393.8 |
| 1 | 32768 | 1,784.2 | 211.2 † | 3,076.8 | 2,118.5 |
| 1 | 65536 | 1,606.0 | 166.7 † | 2,576.6 | 1,712.5 |
| 1 | 130944 | 1,364.3 | 115.2 † | 1,624.9 | 1,251.9 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-default-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-webgpu-default-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 88.0 | 41.8 † | 133.2 | 115.6 |
| 1 | 512 | 87.4 | 41.2 † | 131.9 | 114.0 |
| 1 | 2048 | 85.2 | 39.7 † | 129.3 | 112.9 |
| 1 | 8192 | 77.2 | 35.3 † | 120.2 | 106.4 |
| 1 | 16384 | 67.9 | 31.2 † | 109.5 | 98.0 |
| 1 | 32768 | 55.3 | 25.5 † | 93.8 | 86.5 |
| 1 | 65536 | 41.1 | 18.9 † | 72.9 | 69.8 |
| 1 | 130944 | 27.2 | 12.2 † | 50.6 | 51.0 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-default-ar) | [akas webgpu default / HTTP](AKAS.md#lfm2.5-24b-a2b-q4_0-webgpu-default-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 82.5 | 36.5 † | 128.4 | 108.9 |
| 1 | 512 | 73.1 | 25.5 † | 115.7 | 98.1 |
| 1 | 2048 | 49.7 | 12.0 † | 81.9 | 67.9 |
| 1 | 8192 | 21.5 | 3.7 † | 37.6 | 29.2 |
| 1 | 16384 | 11.9 | 1.8 † | 21.1 | 15.7 |
| 1 | 32768 | 6.1 | 0.8 † | 10.6 | 7.5 |
| 1 | 65536 | 2.9 | 0.3 † | 4.7 | 3.2 |
| 1 | 130944 | 1.2 | 0.1 † | 1.5 | 1.2 |
