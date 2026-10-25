# RX 6700 XT context-wide Vulkan tuning

The requested targets are **not met at every prefix**. The full sweep clears half of reported llama.cpp prefill at 3/8 points and reported decode parity at 4/8 points. These are point estimates, with one retained sample per full-sweep prefix.

[Canonical phase comparison](COMPARISON.md) · [Retained target data](GOALS.json) · [Plot](throughput.svg)

The same standard LiquidAI LFM2.5-2.6B Q4_0 GGUF, retained C4 token IDs, FP16 KV, B1, 131,072 context capacity, and 128 greedy outputs are used. AKAS runs WebGPU over Vulkan; llama.cpp b11856 uses direct Vulkan. The full AKAS run preloads all phases and has zero per-prefix warmups; the retained pinned llama.cpp full run has one. Fresh one-warmup/three-sample native controls on October 25 agree closely with the retained baseline. Timing boundaries differ, so ratios below evaluate the requested reported-phase targets and do not establish end-to-end speedups.

## Prefill

| Prefix | Prior AKAS | Selected AKAS | llama.cpp | Fraction of llama | At least half |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 32 | 323.2 | 458.0 | 921.0 | 49.7% | no |
| 512 | 816.3 | 974.8 | 2596.5 | 37.5% | no |
| 2,048 | 792.8 | 953.1 | 2470.2 | 38.6% | no |
| 8,192 | 693.9 | 835.6 | 1997.9 | 41.8% | no |
| 16,384 | 594.3 | 712.8 | 1556.3 | 45.8% | no |
| 32,768 | 465.3 | 556.0 | 1068.5 | 52.0% | yes |
| 65,536 | 287.9 | 386.2 | 655.5 | 58.9% | yes |
| 130,944 | 168.5 | 239.5 | 367.6 | 65.1% | yes |

## Decode

| Prefix | Prior AKAS | Selected AKAS | llama.cpp | Fraction of llama | At least parity |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 32 | 166.1 | 169.8 | 165.4 | 102.7% | yes |
| 512 | 164.8 | 168.5 | 161.5 | 104.3% | yes |
| 2,048 | 158.9 | 163.3 | 160.8 | 101.6% | yes |
| 8,192 | 143.4 | 152.9 | 151.5 | 100.9% | yes |
| 16,384 | 124.5 | 139.8 | 144.0 | 97.1% | no |
| 32,768 | 98.1 | 122.2 | 127.9 | 95.6% | no |
| 65,536 | 68.7 | 97.1 | 105.2 | 92.4% | no |
| 130,944 | 43.1 | 68.2 | 78.2 | 87.2% | no |

## Selected changes and qualification

Prefill uses typed eight-FMA FP16 even/odd partials with FP32 totals for FFNs and triple-width input projections, and two signed-Q8 activation components for other Q4_0 projections. Exact RNE activation staging is timed once before reuse; transient exact F16 weight staging remains timed. This is explicitly different prefill arithmetic from the prior FP32-accumulating program. The Q4_0 checkpoint and external FP16 KV are unchanged. Decode remains FP32 arithmetic over packed shared KV. The selected decoder uses 64-token tiles, 128 physical partitions, overlapping K/V loads, and a subgroup-64 softmax exchange with a smaller-subgroup fallback.

Every selected operation was independently qualified. The final model regression gate matched all 512 teacher-forced predictions across prefixes 32, 512, 2,048 and 8,192, with sampled logit relative RMS below 0.27% (limit 1%) and cosine above 0.9999. This is a bounded regression check against the qualified prior AKAS program, not a corpus accuracy evaluation. The exact-half conversion gate includes ties, signed zero and subnormals. The short-half operation gate combines an analytic rounding bound and a 0.5% RMS limit; the ordinary FP32 gate is unchanged.

The final full-sweep greedy output trajectory matches the prior program at 8/8 prefixes. Cross-runtime output agreement is retained separately in the canonical report. The recurrent-state rewind verification reproduced all 384 prior output IDs in its three-prefix control. It avoids a second prefill and operates outside the phase timers.

Rejected two-pass attention, wider tiles, streamed half loops, direct FP32 dot prototypes, all-half prefill and integer tiling trials are retained in the tuning evidence. Isolated hot-buffer timings and synthetic Gaussian-cache decode screens were used to screen candidates; they are not full-model throughput claims. Some wider packed-query searches were interrupted in compiler layout inference.

## Task Manager engines

The verified RX 6700 XT inventory exposes family 0 with graphics plus compute and family 1 with compute without graphics. The exact installed wgpu-hal 27.0.2 source selects family 0. The pinned llama.cpp source prefers a non-graphics compute family. This matches the observed 3D versus Compute engine labels. Engines can share the same underlying cores ([Microsoft explanation](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)).

The same llama.cpp binary was measured with `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` and with its default compute queue, each with one warmup and three retained samples. This tests the queue effect in llama.cpp; AKAS itself was not switched to a compute-only queue.

| Prefix | Graphics PP | Compute PP | Graphics decode | Compute decode |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 916.6 | 918.2 | 165.5 | 165.6 |
| 512 | 2589.7 | 2608.1 | 162.5 | 163.1 |
| 2,048 | 2423.3 | 2465.9 | 159.8 | 160.8 |

The short-context queue effect is small compared with the remaining prefill gap. The inventory, exact crate archive/checksum, source capture, raw queue runs, trial profiles, qualification logs, and model regression captures are retained in tuning evidence `20261025121909558322-6c37463a`.

Full AKAS: `20261025121928791585-816cf540`. Pinned full llama.cpp: `20261024191042883833-974e6fff`. Prior AKAS: `20261024215325218226-fd38f777`.
