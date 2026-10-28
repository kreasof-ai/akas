# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

Decode rates are medians of five samples after two warmups. One request generates 128 outputs; 127 decode steps are timed. Both timers cover completed wall decode including greedy sampling and control dispatch; model load, real prefill, snapshot restore and validation are excluded. AKAS uses GPU token feedback; llama.cpp b11946 Vulkan C API uses its stock greedy sampler. Both restore complete real-prefill recurrent/KV state and the initial prediction. Prompts use fixed synthetic token IDs, never fabricated KV. Prefill observations are single samples in the portable data, not repeated benchmarks, and do not gate decode parity. A dash means no measured median. The maximum-prefix AKAS state is imported from sealed real-prefill run 20261028130402880342-2c41aef6 with checked hashes and unchanged prefill arithmetic/storage. Short prefixes through 8,192 select the 32-partition program; longer ones select the shared-KV 512-partition program. Both reserve a 131,072-token context. Generated sequences differ at some prefixes; performance parity does not establish identical text or end-to-end request latency.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm3-1.7b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / real-prefill-state-restored-wall-decode-v1](AKAS.md#lfm3-1.7b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / real-prefill-state-restored-wall-decode-v1](BASELINES.md#baseline-lfm3-1.7b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | — | — |
| 1 | 512 | — | — |
| 1 | 2048 | — | — |
| 1 | 8192 | — | — |
| 1 | 16384 | — | — |
| 1 | 32768 | — | — |
| 1 | 65536 | — | — |
| 1 | 130944 | — | — |

### Decode tok/s — reported phase timing

| B | Prefix | [akas webgpu float16 / real-prefill-state-restored-wall-decode-v1](AKAS.md#lfm3-1.7b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / real-prefill-state-restored-wall-decode-v1](BASELINES.md#baseline-lfm3-1.7b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 247.5 | 208.2 |
| 1 | 512 | 244.1 | 202.1 |
| 1 | 2048 | 225.4 | 187.5 |
| 1 | 8192 | 186.6 | 180.1 |
| 1 | 16384 | 179.6 | 164.4 |
| 1 | 32768 | 153.4 | 140.9 |
| 1 | 65536 | 117.3 | 109.6 |
| 1 | 130944 | 79.5 | 74.6 |

### Complete-burst output tok/s

| B | Prefix | [akas webgpu float16 / HTTP](AKAS.md#lfm3-1.7b-q4_0-webgpu-float16-ar) | [llama vulkan float16 / HTTP](BASELINES.md#baseline-lfm3-1.7b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: |
| 1 | 32 | — | — |
| 1 | 512 | — | — |
| 1 | 2048 | — | — |
| 1 | 8192 | — | — |
| 1 | 16384 | — | — |
| 1 | 32768 | — | — |
| 1 | 65536 | — | — |
| 1 | 130944 | — | — |

## Greedy output agreement

The untimed AKAS greedy pass is compared with the first retained llama.cpp response. Both use the same input IDs, checkpoint and cache precision. Different arithmetic can change the trajectory; agreement is not a corpus accuracy score.

| Prefix | Output tokens | Exact IDs | Common prefix tokens |
| ---: | ---: | --- | ---: |
| 32 | 128 | yes | 128 |
| 512 | 128 | no | 0 |
| 2048 | 128 | yes | 128 |
| 8192 | 128 | no | 7 |
| 16384 | 128 | yes | 128 |
| 32768 | 128 | no | 0 |
| 65536 | 128 | no | 1 |
| 130944 | 128 | no | 1 |
