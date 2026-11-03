# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

Retained measurements; importing evidence does not qualify an AKAS deployment. All values are medians. Phase protocol labels describe the measured boundaries. client-stream-burst-v1 uses concurrent token-arrival timestamps: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. native-gpu-prefill-ar127-v1 uses CUDA events around native prefill and autoregressive decode. Column labels distinguish native engine delivery, CUDA events and HTTP/SSE transports. Decode counts the remaining output tokens after the first. Unqualified values retain †. Original run records and source snapshots are unchanged.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings. † marks an incomplete correctness qualification or a failed correctness check; the reason is stated below. Timing samples are retained. Missing values are not replaced by zero.

Qualification notes:

- AKAS: [the H200 numerical check](../../runs/20261102214625069675-7698a25e/run.json) matched all 8 BOS and 8 chat first tokens against the retained independent references. An additional per-prompt normalized-logit regression gate failed, so H200 remains unqualified. The original L40S qualification remains passed. Built-in kernel qualification/search is now running; timing values below are retained unchanged.
- vLLM: one request at C256 changed its greedy output between repetitions, causing the run-level repeatability check to fail. All requests completed 128 tokens. The frozen timing samples remain in the tables; the run-level marker also appears on the other shapes.

## lfm2.5-24b-a2b-fp8 / fp8 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / client-stream-burst-v1 / native engine token stream](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 57,719.4 † | 25,618.1 † | 34,099.5 |
| 4 | 2048 | 80,733.9 † | 51,464.2 † | 57,952.2 |
| 16 | 2048 | 84,352.1 † | 78,728.2 † | 92,650.8 |
| 64 | 2048 | 82,476.7 † | 87,455.9 † | 98,686.5 |
| 256 | 2048 | 82,193.5 † | 86,126.2 † | 100,233.6 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda float16 / client-stream-burst-v1 / native engine token stream](AKAS.md#lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar) | [vllm cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar) | [sglang cuda default / client-stream-burst-v1 / HTTP/SSE](BASELINES.md#baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 301.2 † | 333.8 † | 250.8 |
| 4 | 2048 | 836.8 † | 998.5 † | 755.5 |
| 16 | 2048 | 2,025.2 † | 2,107.2 † | 1,705.6 |
| 64 | 2048 | 5,875.3 † | 3,699.2 † | 3,253.0 |
| 256 | 2048 | 12,290.5 † | 5,071.3 † | 4,479.9 |

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
| 1 | 7.6% / 22.8% | 10.1% / 17.1% | 17.0% / 20.6% |
| 4 | 15.2% / 37.4% | 17.1% / 28.3% | 23.8% / 31.4% |
| 16 | 23.2% / 47.4% | 27.4% / 38.4% | 24.9% / 45.6% |
| 64 | 25.8% / 32.7% | 29.1% / 28.8% | 24.4% / 51.9% |
| 256 | 25.4% / 14.8% | 29.6% / 13.1% | 24.3% / 35.8% |

Targets: prefill 160,000 tok/s, first milestone 130,000; decoding model MBU 50–70%. Current AKAS prefill peaks at about 84,400 tok/s, so both prefill milestones remain unmet. C64 reaches the requested MBU range; C1, C4, C16 and C256 remain below 50%.

## Tuning evidence

[20261102212730368879-95d732f7](../../runs/20261102212730368879-95d732f7/run.json): Full Modal worker.log for the five-concurrency AKAS run 20261102212133108734-4a2dd18e; raw kernel timing experiments and candidate declarations are retained separately. Those operator candidates are not represented as full-model throughput gains in the table.
