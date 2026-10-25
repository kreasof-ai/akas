# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-24b-a2b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-fp8-ar) | [llama cuda float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 581.2 | 588.2 | 683.1 |
| 1 | 512 | 1,090.3 | 1,106.2 | 2,198.6 |
| 1 | 2048 | 2,631.2 | 2,648.7 | 3,279.2 |
| 1 | 8192 | 3,407.8 | 3,406.8 | 3,208.3 |
| 1 | 16384 | 4,259.0 | 4,220.0 | 3,112.6 |
| 1 | 32768 | 3,956.6 | 3,934.4 | 2,897.3 |
| 1 | 65536 | 3,465.2 | 3,445.7 | 2,495.2 |
| 1 | 130944 | 2,771.7 | 2,755.7 | 1,593.4 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-fp8-ar) | [llama cuda float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 131.8 | 131.2 | 130.0 |
| 1 | 512 | 131.2 | 130.7 | 132.6 |
| 1 | 2048 | 128.9 | 129.0 | 129.9 |
| 1 | 8192 | 121.4 | 123.2 | 121.1 |
| 1 | 16384 | 112.7 | 118.6 | 111.0 |
| 1 | 32768 | 98.8 | 109.8 | 95.1 |
| 1 | 65536 | 78.3 | 95.0 | 74.2 |
| 1 | 130944 | 55.7 | 75.1 | 51.3 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / HTTP](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-ar) | [akas cuda fp8 / HTTP](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-fp8-ar) | [llama cuda float16 / HTTP](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — |
| 1 | 512 | — | — | — |
| 1 | 2048 | — | — | — |
| 1 | 8192 | — | — | — |
| 1 | 16384 | — | — | — |
| 1 | 32768 | — | — | — |
| 1 | 65536 | — | — | — |
| 1 | 130944 | — | — | — |

## lfm2.5-24b-a2b / q4_0 / dspark

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-dspark) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-fp8-dspark) | [llama cuda float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-dspark) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 500.4 | 510.1 | 650.2 |
| 1 | 512 | 2,095.4 | 2,110.1 | 1,705.0 |
| 1 | 2048 | 3,430.9 | 3,409.4 | 2,544.8 |
| 1 | 8192 | 4,150.4 | 4,132.8 | 2,858.9 |
| 1 | 16384 | 4,092.0 | 4,052.2 | 2,912.5 |
| 1 | 32768 | 3,804.9 | 3,770.4 | 2,714.9 |
| 1 | 65536 | 3,330.2 | 3,309.4 | 2,315.3 |
| 1 | 130944 | 2,715.7 | 2,680.5 | 1,508.1 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-dspark) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-fp8-dspark) | [llama cuda float16 / llama-server-eval-real-text-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-dspark) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 64.9 | 66.6 | 80.3 |
| 1 | 512 | 202.9 | 196.4 | 85.8 |
| 1 | 2048 | 256.1 | 191.0 | 171.2 |
| 1 | 8192 | 123.5 | 101.7 | 131.8 |
| 1 | 16384 | 95.5 | 108.7 | 141.1 |
| 1 | 32768 | 54.9 | 53.2 | 64.6 |
| 1 | 65536 | 29.7 | 29.4 | 41.4 |
| 1 | 130944 | 16.8 | 11.0 | 28.1 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / HTTP](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-float16-dspark) | [akas cuda fp8 / HTTP](AKAS.md#lfm2.5-24b-a2b-q4_0-cuda-fp8-dspark) | [llama cuda float16 / HTTP](BASELINES.md#baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-dspark) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — |
| 1 | 512 | — | — | — |
| 1 | 2048 | — | — | — |
| 1 | 8192 | — | — | — |
| 1 | 16384 | — | — | — |
| 1 | 32768 | — | — | — |
| 1 | 65536 | — | — | — |
| 1 | 130944 | — | — | — |
