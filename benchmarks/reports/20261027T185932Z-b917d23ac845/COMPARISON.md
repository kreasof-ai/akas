# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm3-1.7b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#lfm3-1.7b-q4_0-cuda-float16-ar) | [llama cuda float16 / llama-server-eval-v1](BASELINES.md#baseline-lfm3-1.7b-llama-cuda-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 3,315.6 | 3,206.4 |
| 1 | 512 | 13,932.3 | 13,559.7 |
| 1 | 2048 | 14,363.0 | 12,957.2 |
| 1 | 8192 | 13,293.5 | 12,381.4 |
| 1 | 16384 | 12,041.7 | 11,565.9 |
| 1 | 32768 | 10,176.7 | 9,988.2 |
| 1 | 65536 | 7,655.9 | 6,517.1 |
| 1 | 130944 | 5,081.0 | 3,942.3 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#lfm3-1.7b-q4_0-cuda-float16-ar) | [llama cuda float16 / llama-server-eval-v1](BASELINES.md#baseline-lfm3-1.7b-llama-cuda-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 204.6 | 190.1 |
| 1 | 512 | 203.7 | 186.1 |
| 1 | 2048 | 200.9 | 183.5 |
| 1 | 8192 | 190.0 | 172.8 |
| 1 | 16384 | 177.4 | 160.3 |
| 1 | 32768 | 150.9 | 140.2 |
| 1 | 65536 | 119.5 | 110.0 |
| 1 | 130944 | 85.4 | 79.3 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / HTTP](AKAS.md#lfm3-1.7b-q4_0-cuda-float16-ar) | [llama cuda float16 / HTTP](BASELINES.md#baseline-lfm3-1.7b-llama-cuda-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | — | 188.3 |
| 1 | 512 | — | 177.4 |
| 1 | 2048 | — | 150.1 |
| 1 | 8192 | — | 91.3 |
| 1 | 16384 | — | 57.7 |
| 1 | 32768 | — | 30.4 |
| 1 | 65536 | — | 11.4 |
| 1 | 130944 | — | 3.7 |

## Greedy output agreement

The untimed AKAS greedy pass is compared with the first retained llama.cpp response. Both use the same input IDs, checkpoint and cache precision. Different arithmetic can change the trajectory; agreement is not a corpus accuracy score.

| Prefix | Output tokens | Exact IDs | Common prefix tokens |
| ---: | ---: | --- | ---: |
| 32 | 128 | yes | 128 |
| 512 | 128 | no | 0 |
| 2048 | 128 | no | 3 |
| 8192 | 128 | yes | 128 |
| 16384 | 128 | no | 2 |
| 32768 | 128 | no | 6 |
| 65536 | 128 | no | 0 |
| 130944 | 128 | no | 1 |
