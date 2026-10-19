# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; the measured LFM floating path executes FP16. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-24b-a2b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-default-ar) | [akas webgpu default / unmeasured](AKAS.md#lfm2.5-24b-a2b-q4_0-webgpu-default-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | 763.0 | 425.4 |
| 1 | 512 | — | — | 3,619.0 | 2,706.4 |
| 1 | 2048 | — | — | 3,530.5 | 2,702.6 |
| 1 | 8192 | — | — | 3,490.7 | 2,578.1 |
| 1 | 16384 | — | — | 3,339.3 | 2,393.8 |
| 1 | 32768 | — | — | 3,076.8 | 2,118.5 |
| 1 | 65536 | — | — | 2,576.6 | 1,712.5 |
| 1 | 130944 | — | — | 1,624.9 | 1,251.9 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-default-ar) | [akas webgpu default / unmeasured](AKAS.md#lfm2.5-24b-a2b-q4_0-webgpu-default-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | 133.2 | 115.6 |
| 1 | 512 | — | — | 131.9 | 114.0 |
| 1 | 2048 | — | — | 129.3 | 112.9 |
| 1 | 8192 | — | — | 120.2 | 106.4 |
| 1 | 16384 | — | — | 109.5 | 98.0 |
| 1 | 32768 | — | — | 93.8 | 86.5 |
| 1 | 65536 | — | — | 72.9 | 69.8 |
| 1 | 130944 | — | — | 50.6 | 51.0 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-default-ar) | [akas webgpu default / HTTP](AKAS.md#lfm2.5-24b-a2b-q4_0-webgpu-default-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | 128.4 | 108.9 |
| 1 | 512 | — | — | 115.7 | 98.1 |
| 1 | 2048 | — | — | 81.9 | 67.9 |
| 1 | 8192 | — | — | 37.6 | 29.2 |
| 1 | 16384 | — | — | 21.1 | 15.7 |
| 1 | 32768 | — | — | 10.6 | 7.5 |
| 1 | 65536 | — | — | 4.7 | 3.2 |
| 1 | 130944 | — | — | 1.5 | 1.2 |
