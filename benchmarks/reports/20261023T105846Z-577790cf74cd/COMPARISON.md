# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-24b-a2b / q4_0 / dspark

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-dspark) |
| ---: | ---: | ---: |
| 1 | 32 | 602.1 |
| 1 | 8192 | 4,257.3 |
| 1 | 130944 | 2,721.9 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-dspark) |
| ---: | ---: | ---: |
| 1 | 32 | 165.1 |
| 1 | 8192 | 176.9 |
| 1 | 130944 | 12.8 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / HTTP](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-dspark) |
| ---: | ---: | ---: |
| 1 | 32 | 155.2 |
| 1 | 8192 | 48.5 |
| 1 | 130944 | 2.2 |
