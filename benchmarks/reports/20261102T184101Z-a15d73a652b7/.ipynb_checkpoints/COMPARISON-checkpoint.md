# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

Retained measurements; importing evidence does not qualify an AKAS deployment. All values are medians. Phase protocol labels describe the measured boundaries. client-stream-burst-v1 measures concurrent native-engine token delivery: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. Decode counts the remaining output tokens after the first. Unqualified values retain †. Original run records and source snapshots are unchanged.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-24b-a2b-fp8 / fp8 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / client-stream-burst-v1](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / client-stream-burst-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / client-stream-burst-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 57,842.3 † | 27,790.0 | 37,178.2 |
| 4 | 2048 | 76,155.3 † | 25,222.3 | 39,472.7 |
| 16 | 2048 | 76,405.3 † | 25,663.6 | 39,649.1 |
| 64 | 2048 | 75,947.8 † | 25,963.6 | 39,721.6 |
| 256 | 2048 | 75,185.6 † | 24,477.2 | 39,188.9 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / client-stream-burst-v1](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / client-stream-burst-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / client-stream-burst-v1](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 300.6 † | 344.0 | 343.7 |
| 4 | 2048 | 838.6 † | 773.2 | 893.1 |
| 16 | 2048 | 2,102.8 † | 1,161.5 | 1,385.4 |
| 64 | 2048 | 5,930.2 † | 1,508.8 | 1,917.4 |
| 256 | 2048 | 11,867.2 † | 3,198.1 | 2,197.6 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / HTTP](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / native call](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / native call](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | — | 288.8 | 301.1 |
| 4 | 2048 | — | 623.2 | 760.4 |
| 16 | 2048 | — | 1,028.9 | 1,300.5 |
| 64 | 2048 | — | 1,358.6 | 1,879.7 |
| 256 | 2048 | — | 1,452.2 | 2,182.7 |
