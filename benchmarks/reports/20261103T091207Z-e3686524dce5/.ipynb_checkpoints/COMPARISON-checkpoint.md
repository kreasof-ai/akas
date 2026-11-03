# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings. † marks an incomplete correctness qualification or a failed correctness check; the reason is stated below. Timing samples are retained. Missing values are not replaced by zero.

## lfm2.5-24b-a2b-fp8 / fp8 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2 / reported transport](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [llama cuda default / unmeasured / reported transport](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-llama-cuda-fp8-ar) | [llama vulkan default / unmeasured / reported transport](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-llama-vulkan-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 64,458.8 | — | — |
| 4 | 2048 | 87,152.9 | — | — |
| 16 | 2048 | 92,201.5 | — | — |
| 64 | 2048 | 90,179.5 | — | — |
| 256 | 2048 | 89,432.5 | — | — |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2 / reported transport](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [llama cuda default / unmeasured / reported transport](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-llama-cuda-fp8-ar) | [llama vulkan default / unmeasured / reported transport](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-llama-vulkan-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 319.3 | — | — |
| 4 | 2048 | 932.5 | — | — |
| 16 | 2048 | 2,207.9 | — | — |
| 64 | 2048 | 6,343.1 | — | — |
| 256 | 2048 | 14,799.1 | — | — |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / reported transport](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [llama cuda default / reported transport](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-llama-cuda-fp8-ar) | [llama vulkan default / reported transport](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-llama-vulkan-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 298.0 | — | — |
| 4 | 2048 | 801.5 | — | — |
| 16 | 2048 | 1,605.4 | — | — |
| 64 | 2048 | 2,995.4 | — | — |
| 256 | 2048 | 4,065.9 | — | — |
