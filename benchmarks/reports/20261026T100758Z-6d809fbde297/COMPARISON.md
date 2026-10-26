# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## spark-x2.6-4b / q4_k_m / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#spark-x2.6-4b-q4_k_m-cuda-float16-ar) | [llama cuda float16 / llama-server-eval-v1](BASELINES.md#baseline-spark-x2.6-4b-llama-cuda-q4_k_m-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 1,075.4 | 844.1 |
| 1 | 512 | 4,606.0 | 4,179.9 |
| 1 | 2048 | 4,734.2 | 4,542.6 |
| 1 | 8192 | 4,394.3 | 4,359.9 |
| 1 | 16384 | 3,990.7 | 4,038.9 |
| 1 | 32768 | 3,339.7 | 3,136.7 |
| 1 | 65536 | 2,500.2 | 2,138.9 |
| 1 | 130944 | 1,667.2 | 1,337.8 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#spark-x2.6-4b-q4_k_m-cuda-float16-ar) | [llama cuda float16 / llama-server-eval-v1](BASELINES.md#baseline-spark-x2.6-4b-llama-cuda-q4_k_m-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 71.2 | 63.7 |
| 1 | 512 | 69.8 | 65.6 |
| 1 | 2048 | 68.5 | 50.6 |
| 1 | 8192 | 64.9 | 41.3 |
| 1 | 16384 | 60.3 | 55.4 |
| 1 | 32768 | 52.0 | 46.8 |
| 1 | 65536 | 41.7 | 37.2 |
| 1 | 130944 | 30.2 | 27.5 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / HTTP](AKAS.md#spark-x2.6-4b-q4_k_m-cuda-float16-ar) | [llama cuda float16 / HTTP](BASELINES.md#baseline-spark-x2.6-4b-llama-cuda-q4_k_m-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | — | 62.9 |
| 1 | 512 | — | 62.1 |
| 1 | 2048 | — | 43.2 |
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
