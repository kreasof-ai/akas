# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings. † marks an incomplete correctness qualification or a failed correctness check; the reason is stated below. Timing samples are retained. Missing values are not replaced by zero.

## lfm2.5-24b-a2b-fp8 / fp8 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2 / reported transport](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 64,458.8 | 25,618.1 † | 34,099.5 |
| 4 | 2048 | 87,152.9 | 51,464.2 † | 57,952.2 |
| 16 | 2048 | 92,201.5 | 78,728.2 † | 92,650.8 |
| 64 | 2048 | 90,179.5 | 87,455.9 † | 98,686.5 |
| 256 | 2048 | 89,432.5 | 86,126.2 † | 100,233.6 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-token-step-v2 / reported transport](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 319.3 | 333.8 † | 250.8 |
| 4 | 2048 | 932.5 | 998.5 † | 755.5 |
| 16 | 2048 | 2,207.9 | 2,107.2 † | 1,705.6 |
| 64 | 2048 | 6,343.1 | 3,699.2 † | 3,253.0 |
| 256 | 2048 | 14,799.1 | 5,071.3 † | 4,479.9 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / reported transport](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 298.0 | 277.0 † | 223.3 |
| 4 | 2048 | 801.5 | 769.2 † | 675.5 |
| 16 | 2048 | 1,605.4 | 1,759.5 † | 1,606.2 |
| 64 | 2048 | 2,995.4 | 3,206.4 † | 3,054.7 |
| 256 | 2048 | 4,065.9 | 4,301.0 † | 4,219.8 |

## Utilization estimates and targets

MFU is the retained mixed-arithmetic useful-FLOP estimate; MBU is expected model weight plus 16-bit KV traffic divided by H200 4.8 TB/s. These are resource-model estimates, not hardware counters or guaranteed remaining speedups. Dense FP8 peak is 1,979 TFLOP/s and 16-bit peak is 989.5 TFLOP/s. The same reference model normalizes all three runtimes; native engine and HTTP delivery overhead differ.

| C | vLLM MFU / MBU | SGLang MFU / MBU | AKAS MFU / MBU |
| ---: | ---: | ---: | ---: |
| 1 | 7.6% / 22.8% | 10.1% / 17.1% | 19.0% / 21.8% |
| 4 | 15.2% / 37.4% | 17.1% / 28.3% | 25.7% / 35.0% |
| 16 | 23.2% / 47.4% | 27.4% / 38.4% | 27.2% / 49.7% |
| 64 | 25.8% / 32.7% | 29.1% / 28.8% | 26.6% / 56.1% |
| 256 | 25.4% / 14.8% | 29.6% / 13.1% | 26.4% / 43.1% |

Prefill target 160,000 tok/s, first milestone 130,000; observed maximum 92,202 tok/s. Decode target is model MBU 50–70%; per-shape estimates are shown below.
