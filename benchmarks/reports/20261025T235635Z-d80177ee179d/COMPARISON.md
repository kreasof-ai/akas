# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## spark-x2.6-4b / q4_k_m / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#spark-x2.6-4b-q4_k_m-cuda-float16-ar) | [llama cuda float16 / llama-server-eval-v1](BASELINES.md#baseline-spark-x2.6-4b-llama-cuda-q4_k_m-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 677.6 | 843.6 |
| 1 | 512 | 1,545.8 | 4,235.9 |
| 1 | 2048 | 1,523.5 | 4,521.5 |
| 1 | 8192 | 1,448.1 | 4,356.3 |
| 1 | 16384 | 1,350.6 | 4,035.8 |
| 1 | 32768 | 1,190.7 | 3,139.8 |
| 1 | 65536 | 964.8 | 2,139.3 |
| 1 | 130944 | 714.0 | 1,337.0 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#spark-x2.6-4b-q4_k_m-cuda-float16-ar) | [llama cuda float16 / llama-server-eval-v1](BASELINES.md#baseline-spark-x2.6-4b-llama-cuda-q4_k_m-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 65.3 | 63.8 |
| 1 | 512 | 63.3 | 65.6 |
| 1 | 2048 | 61.6 | 50.7 |
| 1 | 8192 | 57.4 | 41.4 |
| 1 | 16384 | 51.2 | 55.4 |
| 1 | 32768 | 43.1 | 46.9 |
| 1 | 65536 | 32.9 | 37.4 |
| 1 | 130944 | 22.7 | 27.6 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / HTTP](AKAS.md#spark-x2.6-4b-q4_k_m-cuda-float16-ar) | [llama cuda float16 / HTTP](BASELINES.md#baseline-spark-x2.6-4b-llama-cuda-q4_k_m-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | — | 63.0 |
| 1 | 512 | — | 62.3 |
| 1 | 2048 | — | 43.3 |
| 1 | 8192 | — | 25.8 |
| 1 | 16384 | — | 20.1 |
| 1 | 32768 | — | 9.7 |
| 1 | 65536 | — | 3.8 |
| 1 | 130944 | — | 1.2 |

## Greedy output agreement

The untimed AKAS greedy pass is compared with the first retained llama.cpp response. Both use the same input IDs, checkpoint and cache precision. Different arithmetic can change the trajectory; agreement is not a corpus accuracy score.

| Prefix | Output tokens | Exact IDs | Common prefix tokens |
| ---: | ---: | --- | ---: |
| 32 | 128 | yes | 128 |
| 512 | 128 | yes | 128 |
| 2048 | 128 | no | 7 |
| 8192 | 128 | no | 35 |
| 16384 | 128 | yes | 128 |
| 32768 | 128 | yes | 128 |
| 65536 | 128 | no | 0 |
| 130944 | 128 | no | 0 |
