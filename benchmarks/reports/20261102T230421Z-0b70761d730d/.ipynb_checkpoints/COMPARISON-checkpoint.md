# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

Retained measurements; importing evidence does not qualify an AKAS deployment. All values are medians. Phase protocol labels describe the measured boundaries. client-stream-burst-v1 uses concurrent token-arrival timestamps: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. native-gpu-prefill-ar127-v1 uses CUDA events around native prefill and autoregressive decode. Column labels distinguish native engine delivery, CUDA events and HTTP/SSE transports. Decode counts the remaining output tokens after the first. Unqualified values retain †. Original run records and source snapshots are unchanged.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings. † marks an incomplete correctness qualification or a failed correctness check; the reason is stated below. Timing samples are retained. Missing values are not replaced by zero.

Qualification notes:

- AKAS: [all 24 repaired attention/rotary kernels](../../runs/20261102225957833178-121e1c22/run.json) passed independent operation references with unchanged tolerances. Full-model numerical qualification and remaining search coverage are incomplete; these are current-attempt timings. All five shapes passed token-count and repeatability checks.
- vLLM: the frozen run retains one C256 request whose greedy output changed between repetitions. All requests completed 128 tokens; the frozen timing samples are unchanged.

## lfm2.5-24b-a2b-fp8 / fp8 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / client-stream-burst-v1 / native engine token stream](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 56,820.3 † | 25,618.1 † | 34,099.5 |
| 4 | 2048 | 78,350.1 † | 51,464.2 † | 57,952.2 |
| 16 | 2048 | 83,856.9 † | 78,728.2 † | 92,650.8 |
| 64 | 2048 | 81,538.1 † | 87,455.9 † | 98,686.5 |
| 256 | 2048 | 80,929.3 † | 86,126.2 † | 100,233.6 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / client-stream-burst-v1 / native engine token stream](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 293.8 † | 333.8 † | 250.8 |
| 4 | 2048 | 820.9 † | 998.5 † | 755.5 |
| 16 | 2048 | 1,941.5 † | 2,107.2 † | 1,705.6 |
| 64 | 2048 | 5,638.1 † | 3,699.2 † | 3,253.0 |
| 256 | 2048 | 13,327.7 † | 5,071.3 † | 4,479.9 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / native engine token stream](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | — | 277.0 † | 223.3 |
| 4 | 2048 | — | 769.2 † | 675.5 |
| 16 | 2048 | — | 1,759.5 † | 1,606.2 |
| 64 | 2048 | — | 3,206.4 † | 3,054.7 |
| 256 | 2048 | — | 4,301.0 † | 4,219.8 |

## Utilization estimates and targets

MFU is the retained mixed-arithmetic useful-FLOP estimate; MBU is expected model weight plus 16-bit KV traffic divided by H200 4.8 TB/s. These are resource-model estimates, not hardware counters or guaranteed remaining speedups. Dense FP8 peak is 1,979 TFLOP/s and 16-bit peak is 989.5 TFLOP/s. The same reference model normalizes all three runtimes; native engine and HTTP delivery overhead differ.

| C | vLLM MFU / MBU | SGLang MFU / MBU | AKAS MFU / MBU |
| ---: | ---: | ---: | ---: |
| 1 | 7.6% / 22.8% | 10.1% / 17.1% | 16.8% / 20.1% |
| 4 | 15.2% / 37.4% | 17.1% / 28.3% | 23.1% / 30.8% |
| 16 | 23.2% / 47.4% | 27.4% / 38.4% | 24.8% / 43.7% |
| 64 | 25.8% / 32.7% | 29.1% / 28.8% | 24.1% / 49.8% |
| 256 | 25.4% / 14.8% | 29.6% / 13.1% | 23.9% / 38.9% |

Prefill target 160,000 tok/s, first milestone 130,000; observed maximum 83,857 tok/s. Decode target is model MBU 50–70%; per-shape estimates are shown below.

## Tuning evidence

[20261102225957833178-121e1c22](../../runs/20261102225957833178-121e1c22/run.json): Built-in compiler search selected the retained winners; request-state copy addressing and rotary phase precision were repaired and independently qualified before timing. The complete execution log is retained beside this report.
