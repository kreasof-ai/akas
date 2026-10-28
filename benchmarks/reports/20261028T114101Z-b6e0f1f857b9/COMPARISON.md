# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm3-1.7b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / native-token-step-v2](AKAS.md#lfm3-1.7b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / llama-server-eval-v1](BASELINES.md#baseline-lfm3-1.7b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 903.7 | 1,200.4 |
| 1 | 512 | 1,399.9 | 3,680.8 |
| 1 | 2048 | 1,319.8 | 2,432.6 |
| 1 | 8064 | 1,012.1 | 1,003.3 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / native-token-step-v2](AKAS.md#lfm3-1.7b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / llama-server-eval-v1](BASELINES.md#baseline-lfm3-1.7b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 255.5 | 221.6 |
| 1 | 512 | 251.6 | 213.2 |
| 1 | 2048 | 231.3 | 209.9 |
| 1 | 8064 | 196.4 | 196.2 |

### Complete-burst output tok/s

| B | Prefix | [akas webgpu float16 / HTTP](AKAS.md#lfm3-1.7b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / HTTP](BASELINES.md#baseline-lfm3-1.7b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | — | 210.8 |
| 1 | 512 | — | 173.5 |
| 1 | 2048 | — | 88.3 |
| 1 | 8064 | — | 14.7 |

## Greedy output agreement

The untimed AKAS greedy pass is compared with the first retained llama.cpp response. Both use the same input IDs, checkpoint and cache precision. Different arithmetic can change the trajectory; agreement is not a corpus accuracy score.

| Prefix | Output tokens | Exact IDs | Common prefix tokens |
| ---: | ---: | --- | ---: |
| 32 | 128 | yes | 128 |
| 512 | 128 | no | 0 |
| 2048 | 128 | yes | 128 |
| 8064 | 128 | yes | 128 |
