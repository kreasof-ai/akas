# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

Retained measurements; importing evidence does not qualify an AKAS deployment. All values are medians. Phase protocol labels describe the measured boundaries. client-stream-burst-v1 uses concurrent token-arrival timestamps: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. native-gpu-prefill-ar127-v1 uses CUDA events around native prefill and autoregressive decode. Column labels distinguish native engine delivery, CUDA events and HTTP/SSE transports. Decode counts the remaining output tokens after the first. Unqualified values retain †. Original run records and source snapshots are unchanged.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings. † marks an incomplete correctness qualification or a failed correctness check; the reason is stated below. Timing samples are retained. Missing values are not replaced by zero.

Qualification notes:

- AKAS: the retained H200 run passed its recorded checks. The [L40S follow-up numerical qualification](https://github.com/kreasof-ai/akas/blob/8ce8c3da31b2c74f0cd757be3bf152b2e25bc239/benchmarks/runs/20261101113717922004-dcb7262b/run.json) passed against vLLM, SGLang and FP64 references, with 8/8 first-token agreement for both BOS and chat prompt sets. † means this changed H200 AOT program has not yet repeated that qualification; it does not erase the L40S result.
- vLLM: one request at C256 changed its greedy output between repetitions, causing the run-level repeatability check to fail. All requests completed 128 tokens. The frozen timing samples remain in the tables; the run-level marker also appears on the other shapes.

## lfm2.5-24b-a2b-fp8 / fp8 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-gpu-prefill-ar127-v1 / CUDA events](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | — | 25,618.1 † | 34,099.5 |
| 4 | 2048 | — | 51,464.2 † | 57,952.2 |
| 16 | 2048 | — | 78,728.2 † | 92,650.8 |
| 64 | 2048 | — | 87,455.9 † | 98,686.5 |
| 256 | 2048 | 82,035.4 † | 86,126.2 † | 100,233.6 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / native-gpu-prefill-ar127-v1 / CUDA events](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | — | 333.8 † | 250.8 |
| 4 | 2048 | — | 998.5 † | 755.5 |
| 16 | 2048 | — | 2,107.2 † | 1,705.6 |
| 64 | 2048 | — | 3,699.2 † | 3,253.0 |
| 256 | 2048 | 13,610.0 † | 5,071.3 † | 4,479.9 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda float16 / CUDA events](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | — | 277.0 † | 223.3 |
| 4 | 2048 | — | 769.2 † | 675.5 |
| 16 | 2048 | — | 1,759.5 † | 1,606.2 |
| 64 | 2048 | — | 3,206.4 † | 3,054.7 |
| 256 | 2048 | — | 4,301.0 † | 4,219.8 |
