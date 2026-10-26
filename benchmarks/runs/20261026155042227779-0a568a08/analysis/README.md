# Spark X2.6: compact packed-weight prefill

Both runtimes were measured in this optimization session on the same NVIDIA L4 with the pinned official Q4_K_M checkpoint and identical synthetic input IDs. One request, 128 total output tokens, empty cache per request, FP16 KV, one warmup and three retained repetitions. Full 131072-token context reservation. Loading, compilation and phase preparation are excluded. Native synchronized AKAS phases and llama-server eval phases have different timing boundaries.

Observed peak GPU memory (nvidia-smi total used): **AKAS 7.71 GiB (7,900 MiB); llama.cpp 7.73 GiB (7,918 MiB)**. The earlier packed-weight AKAS run peaked at 9.50 GiB. GPU weight storage including compact rotary tables: 2.671 GiB. Persistent state: 4.553 GiB, almost entirely FP16 KV.

Both runtimes keep the input embedding on the CPU. AKAS maps 640 MiB of pinned host memory; all transformer weights and the output head stay on the GPU. Quantized weights retain checkpoint bytes in aligned storage. There is no persistent expanded FP16 weight matrix. Temporary on-chip tiles supply tensor-core operands.

AKAS uses a 2048-token maximum prefill chunk; llama.cpp uses 512. Both have the same full-attention capacity and KV precision. AKAS retains one sliding window per layer and reuses a temporary current-chunk K/V workspace across layers, committing only the live suffix after attention. Shared state allocations avoid per-layer allocation padding. Cross-phase scratch and identical kernel modules are shared. Compact rotary tables replace full-context tables. Fused prefill combines gate/up/exact GELU, residual addition/normalization, and QKV split/rotary/cache writes. Activation staging uses the following MMA operation’s FP16 precision; reductions and residuals stay FP32.

Prefill medians meet or exceed llama.cpp at 8/8 prefixes. Decode medians exceed it at 8/8; all retained decode samples exceed corresponding baseline samples at 8/8. These are measurements of this configuration and workload, not a guarantee for arbitrary inputs or hardware.

| Prefix | AKAS prefill tok/s | llama.cpp prefill tok/s | AKAS decode tok/s | llama.cpp decode tok/s | Estimated prefill MFU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1,085.6 | 839.9 | 71.16 | 63.80 | 6.80% |
| 512 | 4,238.4 | 4,195.7 | 69.80 | 65.73 | 26.99% |
| 2,048 | 4,632.6 | 4,536.1 | 68.47 | 50.75 | 30.25% |
| 8,192 | 4,406.6 | 4,351.3 | 64.86 | 41.04 | 30.50% |
| 16,384 | 4,061.0 | 4,041.3 | 60.35 | 55.40 | 30.15% |
| 32,768 | 3,507.1 | 3,135.5 | 51.90 | 46.93 | 29.55% |
| 65,536 | 2,724.4 | 2,139.1 | 41.60 | 37.29 | 28.39% |
| 130,944 | 1,879.4 | 1,336.2 | 30.07 | 27.60 | 27.08% |

Useful prefill MFU counts dense projections, one final-row vocabulary projection and causal QK/PV work against [NVIDIA L4’s 121 TFLOP/s dense FP16 peak](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf). It excludes dequantization, padding, elementwise operations, embedding lookups and cache/control traffic. It counts all model layers even where intermediate chunks prune final-layer outputs. It is not measured hardware instruction utilization.

An incomplete development run (`20261026145843633286-516a8e7e`) was rejected after a pruning bug was discovered. Its sealed record is retained for audit and must not be used for comparisons. The current run uses mapped-output identities preserved by name and a four-layer oracle covering middle full attention.

## Qualification

Independent operation equations cover packed Q4_K, Q6_K, Q8_0 and BF16 storage, odd matrix tails, FP16 activation staging, exact GELU, mixed-precision residual RMS, high-position compact RoPE, fused QKV writes, inactive rows, wrap and NaN padding. Complete BF16, native channel-FP8 and four-layer mixed-GGUF models are compared with independent Torch equations across chunk boundaries, physical slot reorder and reuse. GPU uploads are checked byte for byte. Allocation tests cover shared scratch, shared module integrity, typed arena slices, mapped embeddings and ownership after another phase closes. The corrected 4B checkpoint also produces bitwise-identical 131072 logits at an 8192-token prompt with intermediate state prefill enabled or disabled. Two legacy mapping tests reference removed entry points and fail identically on the pre-fix checkout; their logs are retained as exclusions. These tests do not establish full pretrained-model task accuracy. Cross-runtime greedy trajectories are retained separately in the canonical report. GPU measurements are serialized under stock clock/power management; the retained telemetry includes temperature, clocks and power. The fresh llama.cpp run was collected earlier in this session with the same pinned binary, checkpoint and protocol, and reused after the native compact-cache refinement.

## Reproduce

```sh
python -m tools.spark_benchmark compile --optimized --chunk 2048 --program build/spark/compact --out build/spark/compile.json
python -m tools.spark_benchmark akas --chunk 2048 --program build/spark/compact --out build/spark/akas.json
python -m tools.spark_benchmark llama --out build/spark/llama.json
python -m tools.spark_benchmark report --runs AKAS_RUN LLAMA_RUN --out build/spark/report
python -m tools.spark_mfu REPORT_DIRECTORY/report.json --checkpoint MODEL.gguf --out build/spark/mfu.json
```
